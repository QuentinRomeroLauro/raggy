"""
Retriever module for handling document retrieval
"""
import os
import uuid
import signal
import logging
import requests
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .extended_chroma import ExtendedChroma
from ..interfaces.helpers.counter import read_counter, increment_counter, write_counter
from ..utils.logger import DebugLogger
from multiprocessing import Process, Queue
from docling.document_converter import DocumentConverter
import pickle
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


logger = logging.getLogger(__name__)

# Define the relevance score function at the module level
def _retriever_relevance_score_fn(distance: float) -> float:
    """
    Calculates a relevance score from a distance measure, ensuring it's between 0 and 1.
    Suitable for cosine distance where lower values are better.
    """
    # Ensure the score is between 0 and 1
    similarity = 1.0 - min(max(0.0, distance), 1.0)
    return similarity


class Retriever:
    def __init__(self, docStore: str = '', client: bool = True):
        """Initialize the retriever.
        
        Args:
            docStore: Path to the directory containing documents
            client: Whether to send data to the debug interface
        """
        self.vectorStores: Dict[str, ExtendedChroma] = {}
        self.storesToChunks: Dict[str, List[Document]] = {}
        self.raptorDB = None
        self.docStore = Path(docStore)
        self.client = client
        self.logger = DebugLogger("retriever")
        self.queue = Queue()
        self.processes = {}
        
        # Load vector stores if they exist
        self.loadVectorDBs()
        try:
            self.loadRaptorCollapsedTreeRetriever()
        except ValueError:
            self.logger.warning("Raptor DB not found, continuing without it")
    
    def invoke(
        self,
        query: str,
        k: int = 5,
        chunkSize: int = 400,
        chunkOverlap: int = 0,
        retrievalMode: str = "vanilla",
        searchBy: str = "semantic similarity",
        send: bool = True,
        duplicate: bool = False,
        id: Optional[str] = None,
        dupDocsAndScores: Optional[List[Tuple[Document, float]]] = None,
        dupDocsAndScoresRaptor: Optional[List[Tuple[Document, float]]] = None,
        dupDocsAndScoresVanilla: Optional[List[Tuple[Document, float]]] = None,
    ) -> List[Tuple[Document, float]]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: The query string
            k: Number of documents to retrieve
            chunkSize: Size of document chunks
            chunkOverlap: Overlap between chunks
            retrievalMode: Either 'vanilla' or 'raptor'
            searchBy: One of 'semantic similarity', 'max marginal relevance', or 'tfidf'
            send: Whether to send data to the debug interface
            duplicate: Whether this is a duplicate request
            id: Request ID for duplicate requests
            dupDocsAndScores: Pre-computed docs and scores for duplicate requests
            dupDocsAndScoresRaptor: Pre-computed raptor docs and scores
            dupDocsAndScoresVanilla: Pre-computed vanilla docs and scores
        
        Returns:
            List of (document, score) tuples
        """
        self.logger.info(f"INVOKE: query={query}, k={k}, chunkSize={chunkSize}, mode={retrievalMode}, searchBy={searchBy}")

        # Convert parameters to correct types
        query = str(query)
        k = int(k)
        chunkSize = int(chunkSize)
        chunkOverlap = int(chunkOverlap)
        
        # Set loading state
        if (send and self.client) and not duplicate:
            requests.post('http://localhost:5001/set_loading', json={'loading': True})
        
        # Generate results if this isn't a duplicate call
        if not duplicate:
            collection_name = self.getCollectionName(chunkSize, chunkOverlap, searchBy)
            
            # Check if collection exists and create it if it doesn't
            if collection_name not in self.vectorStores:
                self.logger.info(f"Collection {collection_name} not found, attempting to create it...")
                try:
                    self.createVectorDB(
                        chunkSize=chunkSize,
                        chunkOverlap=chunkOverlap,
                        searchBy=searchBy
                    )
                except Exception as e:
                    self.logger.error(f"Failed to create collection {collection_name}: {e}")
                    raise ValueError(f"Collection {collection_name} does not exist and could not be created")
            
            vs = self.vectorStores[collection_name]
            docs_and_scores_vanilla = []
            docs_and_scores_raptor = []
            
            # Get results based on search method
            if searchBy == "semantic similarity":
                docs_and_scores_vanilla = vs.similarity_search_with_score(query, k=k+40)
                if self.raptorDB:
                    docs_and_scores_raptor = self.raptorDB.similarity_search_with_score(query, k=k+40)
            
            elif searchBy == "max marginal relevance":
                docs_and_scores_vanilla = vs.max_marginal_relevance_search_with_score(query=query, k=k+40, fetch_k=40)
                if self.raptorDB:
                    docs_and_scores_raptor = self.raptorDB.max_marginal_relevance_search_with_score(query=query, k=k+40, fetch_k=40)
            
            elif searchBy == "tfidf":
                # Get the paths from the store's metadata
                vectorizer_path = vs._collection.metadata.get("vectorizer_path")
                matrix_path = vs._collection.metadata.get("matrix_path")
                
                if vectorizer_path is None or matrix_path is None:
                    raise ValueError(f"TF-IDF vectorizer or matrix paths not found in collection {collection_name}")
                
                # Load the vectorizer and matrix
                with open(vectorizer_path, 'rb') as f:
                    vectorizer = pickle.load(f)
                matrix = load_npz(matrix_path)
                
                # Transform the query using the vectorizer
                query_vector = vectorizer.transform([query])
                
                # Calculate cosine similarity between query and documents
                similarities = cosine_similarity(query_vector, matrix).flatten()
                
                # Get the top k documents
                top_k_indices = similarities.argsort()[-(k+40):][::-1]
                docs_and_scores_vanilla = []
                
                # Get all documents from the collection
                collection_docs = vs._collection.get()['documents']
                
                for idx in top_k_indices:
                    doc = Document(page_content=collection_docs[idx])
                    score = float(similarities[idx])
                    docs_and_scores_vanilla.append((doc, score))
                
                if self.raptorDB:
                    docs_and_scores_raptor = self.raptorDB.tfidf(query=query, k=k+40)
            
            else:
                raise ValueError("Invalid searchBy, must be 'semantic similarity', 'max marginal relevance', or 'tfidf'")
        else:
            docs_and_scores_vanilla = dupDocsAndScoresVanilla
            docs_and_scores_raptor = dupDocsAndScoresRaptor
        
        # Generate ID if not provided
        if not id:
            id = str(uuid.uuid4())
        
        # Send data to debug interface
        if self.client and send:
            self.sendRetrievalData(
                query=query,
                docs_and_scores_vanilla=docs_and_scores_vanilla,
                docs_and_scores_raptor=docs_and_scores_raptor,
                searchBy=searchBy,
                chunkSize=chunkSize,
                chunkOverlap=chunkOverlap,
                k=k,
                id=id,
                retrievalMode=retrievalMode,
            )
        
        # Handle process forking for user interaction
        if send and not duplicate:
            pid = os.fork()
            if pid == 0:  # Child process
                child_pid = os.getpid()
                try:
                    # Register process
                    requests.post(f"http://localhost:5001/register_process/{id}/{child_pid}")
                    
                    # Wait for signal
                    sigset = {signal.SIGUSR1}
                    signal.pthread_sigmask(signal.SIG_BLOCK, sigset)

                    # sleep the process, until the server signals it to wake up
                    while signal.sigwait(sigset) != signal.SIGUSR1:
                        pass # wait

                    # Get updated data
                    print("id-from retriever", id)

                    data = requests.get(f"http://localhost:5001/get_updated_interface_data/{id}").json()
                    print("updated data", data)

                    selected_chunks = data.get('selectedChunks', [])
                    # Convert selected chunks back to Document objects
                    docs_and_scores_vanilla = [(Document(page_content=chunk['text']), chunk['score']) 
                                for chunk in selected_chunks]
                    
                    docs_and_scores_raptor = data.get('raptorChunks', [])
                    docs_and_scores = docs_and_scores_vanilla + docs_and_scores_raptor
                    
                    # Clean up process registration
                    requests.post(f"http://localhost:5001/delete_process/{id}")
                    
                    # Fork again to continue pipeline
                    nextPipelinePID = os.fork()
                    if nextPipelinePID == 0:
                        # get the actual pid of the process
                        pid = os.getpid()

                        # Define the set of signals to wait for
                        sigset = {signal.SIGUSR1}
                        # Block the signals so they can be waited for
                        signal.pthread_sigmask(signal.SIG_BLOCK, sigset)

                        # wait 
                        while signal.sigwait(sigset) != signal.SIGUSR1:
                            pass # wait

                        # regsiter the replacement process id with the server
                        requests.post(f"http://localhost:5001/register_process/{id}/{pid}")

                        # call the invoke function again to resest pipeline state
                        self.invoke(query=query, k=k, chunkSize=chunkSize, chunkOverlap=chunkOverlap, searchBy=searchBy, send=False, duplicate=True, id=id, dupDocsAndScores=docs_and_scores, dupDocsAndScoresRaptor=docs_and_scores_raptor, dupDocsAndScoresVanilla=docs_and_scores_vanilla)

                    
                except Exception as e:
                    self.logger.error(f"Error in child process: {e}")
                    os._exit(1)
        
        # Select final results based on retrieval mode
        docs_and_scores = (docs_and_scores_raptor if retrievalMode == "raptor" and docs_and_scores_raptor 
                          else docs_and_scores_vanilla)
        
        # Set loading state
        if (send and self.client) and not duplicate:
            requests.post('http://localhost:5001/set_loading', json={'loading': False})
        
        return docs_and_scores[:k] if self.client else docs_and_scores
    
    def loadVectorDBs(self):
        """Load all vector DBs from the store directory."""
        if not self.docStore.exists():
            return
        
        for file in os.listdir(self.docStore):
            if file.startswith("chroma"):
                collection_name = file
                try:
                    # Parse collection name to get parameters
                    parts = collection_name.split("_")
                    if len(parts) >= 4:  # chroma_{chunkSize}_{chunkOverlap}_{search_type}
                        chunkSize = int(parts[1])
                        chunkOverlap = int(parts[2])
                        search_type = parts[3]
                        
                        # Convert search_type back to the original format for internal use
                        searchBy = search_type.replace("_", " ").title()
                        
                        collection_dir = self.docStore / collection_name
                        
                        # For both TF-IDF and semantic search, we need to check if the collection directory exists
                        if not collection_dir.exists():
                            self.logger.warning(f"Collection directory {collection_dir} does not exist")
                            continue
                            
                        try:
                            if search_type == "tfidf":
                                # For TF-IDF, we need to check for the vectorizer and matrix files
                                vectorizer_path = collection_dir / "vectorizer.pkl"
                                matrix_path = collection_dir / "matrix.npz"
                                
                                if not vectorizer_path.exists() or not matrix_path.exists():
                                    self.logger.warning(f"Missing TF-IDF files for {collection_name}")
                                    continue
                                
                                # Create the database with the file paths in metadata
                                db = ExtendedChroma(
                                    persist_directory=str(collection_dir),
                                    collection_name=collection_name,
                                    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large", chunk_size=chunkSize),
                                    relevance_score_fn=_retriever_relevance_score_fn,
                                    collection_metadata={
                                        "search_type": "tfidf",
                                        "vectorizer_path": str(vectorizer_path),
                                        "matrix_path": str(matrix_path)
                                    }
                                )
                            else:
                                # For semantic search and MMR, we just need the collection directory
                                db = ExtendedChroma(
                                    persist_directory=str(collection_dir),
                                    collection_name=collection_name,
                                    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large", chunk_size=chunkSize),
                                    relevance_score_fn=_retriever_relevance_score_fn,
                                    collection_metadata={"hnsw:space": "cosine", "mmr": (search_type == "max_marginal_relevance")}
                                )
                            
                            # Add the database to our stores
                            self.vectorStores[collection_name] = db
                            self.logger.info(f"Loaded vector store {collection_name} with search type {searchBy}")
                            
                        except Exception as e:
                            self.logger.error(f"Failed to initialize vector store {collection_name}: {e}")
                            continue
                            
                except Exception as e:
                    self.logger.error(f"Failed to load vector store {collection_name}: {e}")
                    continue
    
    def loadRaptorCollapsedTreeRetriever(self):
        """Load the RAPTOR retriever."""
        if not self.docStore.exists():
            return
        
        for file in os.listdir(self.docStore):
            if file.startswith("chroma_raptor_summaries"):
                db = ExtendedChroma(
                    persist_directory=str(self.docStore / "chroma_raptor_summaries"),
                    embedding_function=OpenAIEmbeddings(),
                    relevance_score_fn=_retriever_relevance_score_fn,
                    collection_metadata={"hnsw:space": "cosine"}
                )
                self.raptorDB = db
                return
        raise ValueError(f"Raptor DB chroma_raptor_summaries not found in {self.docStore}")
    
    def configExists(self, chunkSize: int, chunkOverlap: int, searchBy: str = "semantic similarity") -> bool:
        """Check if a vector DB with the given configuration exists."""
        collection_name = self.getCollectionName(chunkSize, chunkOverlap, searchBy)
        
        # First check if the collection is in our loaded stores
        if collection_name in self.vectorStores:
            self.logger.info(f"Collection {collection_name} exists in vectorStores")
            return True
        
        # Then check if the collection directory exists
        collection_dir = self.docStore / collection_name
        if not collection_dir.exists():
            self.logger.info(f"Collection directory {collection_name} does not exist")
            return False
        
        # For TF-IDF, check for required files
        if searchBy.lower() == "tfidf":
            vectorizer_path = collection_dir / "vectorizer.pkl"
            matrix_path = collection_dir / "matrix.npz"
            if not vectorizer_path.exists() or not matrix_path.exists():
                self.logger.info(f"Missing TF-IDF files for {collection_name}")
                return False
        
        self.logger.info(f"Collection {collection_name} exists on disk")
        return True
    
    def createVectorDB(
        self,
        embeddings: Optional[OpenAIEmbeddings] = None,
        docs: Optional[List[Document]] = None,
        chunkSize: int = 1000,
        chunkOverlap: int = 0,
        searchBy: str = "semantic similarity"
    ) -> str:
        """Create a vector DB with the given configuration."""
        collection_name = self.getCollectionName(chunkSize, chunkOverlap, searchBy)
        if self.configExists(chunkSize, chunkOverlap, searchBy):
            self.logger.warning(f"Collection {collection_name} already exists")
            return collection_name
        
        if not docs and self.docStore:
            docs = self.getDocs(chunkSize, chunkOverlap)
        
        texts = self.splitDocuments(docs, chunkSize, chunkOverlap)
        
        store_dir = self.docStore
        store_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the collection directory
        collection_dir = store_dir / collection_name
        collection_dir.mkdir(parents=True, exist_ok=True)
        
        # For TF-IDF, we'll pre-compute the TF-IDF matrix
        if searchBy == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            from scipy.sparse import save_npz
            import pickle
            
            # Extract text content
            text_contents = [doc.page_content for doc in texts]
            
            # Create and fit TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_df=0.95,  # Ignore terms that appear in more than 95% of documents
                min_df=2,     # Ignore terms that appear in less than 2 documents
                stop_words='english',
                ngram_range=(1, 2)  # Use both single words and 2-word phrases
            )
            
            # Fit and transform the documents
            tfidf_matrix = vectorizer.fit_transform(text_contents)
            
            # Save the vectorizer and matrix for later use
            vectorizer_path = collection_dir / "vectorizer.pkl"
            matrix_path = collection_dir / "matrix.npz"
            
            try:
                # Save vectorizer
                with open(vectorizer_path, 'wb') as f:
                    pickle.dump(vectorizer, f)
                
                # Save matrix
                save_npz(matrix_path, tfidf_matrix)
                
                # Create the database with the pre-computed TF-IDF information
                db = ExtendedChroma(
                    persist_directory=str(collection_dir),
                    collection_name=collection_name,
                    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large", chunk_size=chunkSize),
                    relevance_score_fn=_retriever_relevance_score_fn,
                    collection_metadata={
                        "search_type": "tfidf",
                        "vectorizer_path": str(vectorizer_path),
                        "matrix_path": str(matrix_path)
                    }
                )
                
                # Add documents to the store
                db.add_documents(texts)
                self.vectorStores[collection_name] = db
                self.logger.info(f"Successfully created TF-IDF vector store {collection_name}")
                return collection_name
                
            except Exception as e:
                self.logger.error(f"Failed to create TF-IDF vector store: {e}")
                # Clean up any partially created files
                if vectorizer_path.exists():
                    vectorizer_path.unlink()
                if matrix_path.exists():
                    matrix_path.unlink()
                raise
        else:
            # Handle other search types as before
            if not embeddings:
                embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-large",
                    chunk_size=chunkSize
                )
            
            try:
                if searchBy == "semantic similarity":
                    db = ExtendedChroma(
                        persist_directory=str(collection_dir),
                        collection_name=collection_name,
                        embedding_function=embeddings,
                        relevance_score_fn=_retriever_relevance_score_fn,
                        collection_metadata={"hnsw:space": "cosine"}
                    )
                elif searchBy == "max marginal relevance":
                    db = ExtendedChroma(
                        persist_directory=str(collection_dir),
                        collection_name=collection_name,
                        embedding_function=embeddings,
                        relevance_score_fn=_retriever_relevance_score_fn,
                        collection_metadata={"hnsw:space": "cosine", "mmr": True}
                    )
                
                # Add documents to the store
                db.add_documents(texts)
                self.vectorStores[collection_name] = db
                self.logger.info(f"Successfully created vector store {collection_name} with search type {searchBy}")
                return collection_name
                
            except Exception as e:
                self.logger.error(f"Failed to create vector store {collection_name}: {e}")
                raise
    
    def splitDocuments(
        self,
        docs: List[Document],
        chunkSize: int,
        chunkOverlap: int,
        lengthFunction=len,
        isSeperatorRegex: bool = False
    ) -> List[Document]:
        """Split documents into chunks."""
        textSplitter = RecursiveCharacterTextSplitter(
            chunk_size=chunkSize,
            chunk_overlap=chunkOverlap,
            length_function=lengthFunction,
            is_separator_regex=isSeperatorRegex
        )
        
        contents = docs
        if docs and isinstance(docs[0], Document):
            contents = [doc.page_content for doc in docs]
        
        texts = textSplitter.create_documents(contents)
        collection_name = self.getCollectionName(chunkSize, chunkOverlap)
        self.storesToChunks[collection_name] = texts
        return texts
    
    def getCollectionName(self, chunkSize: int, chunkOverlap: int, searchBy: str = "semantic similarity") -> str:
        """Get the collection name for the given configuration."""
        # Convert searchBy to a valid format (replace spaces with underscores and make lowercase)
        search_type = searchBy.lower().replace(" ", "_")
        # Ensure the name follows ChromaDB's naming convention
        collection_name = f"chroma_{chunkSize}_{chunkOverlap}_{search_type}"
        self.logger.info(f"Generated collection name: {collection_name}")
        return collection_name
    
    def getDocs(self, chunkSize: int, chunkOverlap: int) -> List[Document]:
        """Get documents from the document store."""
        all_chunks = []
        for filename in os.listdir(self.docStore):
            if filename.endswith(".pdf"):
                pdf_filename = self.docStore / filename
                if not pdf_filename.exists():
                    self.logger.warning(f"File {pdf_filename} not found")
                    continue
                
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(str(pdf_filename))
                docs = loader.load()
                all_chunks.extend(docs)
        
        return all_chunks
    
    def sendRetrievalData(
        self,
        query: str,
        docs_and_scores_vanilla: List[Tuple[Document, float]],
        docs_and_scores_raptor: List[Tuple[Document, float]],
        searchBy: str,
        chunkSize: int,
        chunkOverlap: int,
        k: int,
        id: str,
        retrievalMode: str
    ):
        """Send retrieval data to the debug interface."""
        if retrievalMode == "raptor":
            vanillaChunks, __ = self.getChunksAndSelectedChunks(docs_and_scores_vanilla, 0)
            raptorChunks, selectedChunks = self.getChunksAndSelectedChunks(docs_and_scores_raptor, k)
        elif retrievalMode == "vanilla":
            vanillaChunks, selectedChunks = self.getChunksAndSelectedChunks(docs_and_scores_vanilla, k)
            raptorChunks, __ = self.getChunksAndSelectedChunks(docs_and_scores_raptor, 0)
        else:
            raise ValueError("Invalid retrieval mode, must be 'raptor' or 'vanilla'")
        
        data = {
            'type': 'Retrieval',
            'query': query,
            'vanillaChunks': vanillaChunks,
            'raptorChunks': raptorChunks,
            'selectedChunks': selectedChunks,
            'retrievalMode': retrievalMode,
            'searchBy': searchBy,
            'chunkSize': chunkSize,
            'chunkOverlap': chunkOverlap,
            'k': k,
            'order': increment_counter(),
            'id': id,
        }
        requests.post('http://localhost:5001/send_retrieval_data', json=data)
    
    def getChunksAndSelectedChunks(
        self,
        docs_and_scores: List[Tuple[Document, float]],
        k: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """Get chunks and selected chunks for the debug interface."""
        self.logger.info(f"Getting chunks and selected chunks, k={k}")
        
        chunks = []
        selectedChunks = []
        for index, (doc, score) in enumerate(docs_and_scores):
            try:
                normalized_score = float(score)
                if normalized_score > 1.0 or normalized_score < 0.0:
                    normalized_score = max(0.0, min(1.0, normalized_score))
                    self.logger.info(f"Normalized score from {score} to {normalized_score}")
            except (ValueError, TypeError):
                self.logger.warning(f"Could not convert score {score} to float, using 0.5")
                normalized_score = 0.5
            
            chunk = {
                'text': doc.page_content,
                'score': normalized_score,
                'id': index,
            }
            if not self.containsChunk(chunk, chunks):
                chunks.append(chunk)
            if index < k:
                selectedChunks.append(chunk)
        
        if chunks:
            chunk_scores = [c['score'] for c in chunks[:5]]
            self.logger.info(f"Chunk scores after processing: {chunk_scores}")
            
            for i, score in enumerate(chunk_scores):
                if score > 1.0:
                    self.logger.warning(f"Chunk {i} has score > 1.0: {score}")
        
        return chunks, selectedChunks
    
    def containsChunk(self, chunk: Dict, chunks: List[Dict]) -> bool:
        """Check if a chunk is already in the list."""
        for c in chunks:
            if c['text'] == chunk['text'] and c['score'] == chunk['score'] and c['id'] == chunk['id']:
                continue
        return False

    def createIndexingPipeline(
        self,
        chunk_sizes: List[int] = [400, 1500, 2000],
        chunk_overlaps: List[int] = [0, 100, 200],
        search_types: List[str] = ["semantic similarity", "max marginal relevance", "tfidf"]
    ) -> None:
        """
        Create vector stores for all combinations of chunk sizes, overlaps, and search types.
        Uses Docling for document processing.
        
        Args:
            chunk_sizes: List of chunk sizes to create indexes for
            chunk_overlaps: List of chunk overlaps to create indexes for
            search_types: List of search types to create indexes for
        """
        self.logger.info("Starting indexing pipeline...")
        self.logger.info(f"Chunk sizes: {chunk_sizes}")
        self.logger.info(f"Chunk overlaps: {chunk_overlaps}")
        self.logger.info(f"Search types: {search_types}")
        
        # Initialize Docling converter
        
        converter = DocumentConverter()
        
        # Get all PDF files from the document store
        pdf_files = [f for f in self.docStore.glob("*.pdf")]
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {self.docStore}")
            return
        
        # Process all documents first
        processed_docs = []
        for pdf_file in pdf_files:
            try:
                self.logger.info(f"Processing {pdf_file}")
                # Convert PDF to markdown using Docling
                result = converter.convert(str(pdf_file))
                markdown_content = result.document.export_to_markdown()
                
                # Create a Document object with the markdown content
                doc = Document(
                    page_content=markdown_content,
                    metadata={"source": str(pdf_file)}
                )
                processed_docs.append(doc)
                self.logger.info(f"Successfully processed {pdf_file}")
            except Exception as e:
                self.logger.error(f"Failed to process {pdf_file}: {e}")
        
        if not processed_docs:
            self.logger.error("No documents were successfully processed")
            return
        
        # Create indexes for each configuration
        for chunk_size in chunk_sizes:
            for chunk_overlap in chunk_overlaps:
                for search_type in search_types:
                    if not self.configExists(chunk_size, chunk_overlap, search_type):
                        self.logger.info(f"Creating vector store with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, search_type={search_type}...")
                        try:
                            # Split the processed documents into chunks
                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap
                            )
                            
                            # Create chunks from the processed documents
                            chunks = text_splitter.split_documents(processed_docs)
                            
                            # Create the vector store with the processed and chunked documents
                            self.createVectorDB(
                                docs=chunks,
                                chunkSize=chunk_size,
                                chunkOverlap=chunk_overlap,
                                searchBy=search_type
                            )
                        except Exception as e:
                            self.logger.error(f"Failed to create vector store with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, search_type={search_type}: {e}")
                    else:
                        self.logger.info(f"Vector store already exists for chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, search_type={search_type}")
        
        self.logger.info("Indexing pipeline completed.")

