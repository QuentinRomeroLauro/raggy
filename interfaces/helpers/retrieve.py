"""
This file defines the Retriever class which is used to retrieve documents from the vectorstore.
It also contains the logic to check if a vector DB with the given configuration exists, and create vector DBs.
"""
import requests
import uuid
from interfaces.helpers.counter import read_counter, increment_counter, write_counter

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

import logging
import os
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from interfaces.helpers.local_loader import get_document_text
from interfaces.helpers.remote_loader import download_file
from dotenv import load_dotenv
from time import sleep
import numpy as np
import signal

from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers import TFIDFRetriever

# ================================================
"""
Code adapted from langchain_community.vectorstores.chroma to support MMR search with scores

If this makes absolutely no sense, see the original code here: https://api.python.langchain.com/en/latest/_modules/langchain_chroma/vectorstores.html#Chroma.max_marginal_relevance_search
"""

DEFAULT_K = 4  # Number of Documents to return.

def _results_to_docs(results):
    return [doc for doc, _ in _results_to_docs_and_scores(results)]


def _results_to_docs_and_scores(results):
    return [
        # TODO: Chroma can do batch querying,
        # we shouldn't hard code to the 1st result
        (Document(page_content=result[0], metadata=result[1] or {}), result[2])
        for result in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]

def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: list,
    lambda_mult: float = 0.5,
    k: int = 4,
):
    """Calculate maximal marginal relevance.

    Args:
        query_embedding: Query embedding.
        embedding_list: List of embeddings to select from.
        lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
        k: Number of Documents to return. Defaults to 4.

    Returns:
        List of indices of embeddings selected by maximal marginal relevance.
    """
    if min(k, len(embedding_list)) <= 0:
        return []
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = cosine_similarity(embedding_list, selected)
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue
            redundant_score = max(similarity_to_selected[i])
            equation_score = (
                lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            )
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        idxs.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
    return idxs


def cosine_similarity(X, Y):
    """Row-wise cosine similarity between two equal-width matrices.

    Raises:
        ValueError: If the number of columns in X and Y are not the same.
    """
    if len(X) == 0 or len(Y) == 0:
        return np.array([])

    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            "Number of columns in X and Y must be the same. X has shape"
            f"{X.shape} "
            f"and Y has shape {Y.shape}."
        )

    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)
    # Ignore divide by zero errors run time warnings as those are handled below.
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
    similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
    return similarity


def BM25(query, k, chroma_db):
    docs = chroma_db._Chroma__get_or_create_collection("")

    retriever = BM25Retriever.from_documents()




# End of adapted langchain code
# ===========================================================================


class ExtendedTFIDFRetriever(TFIDFRetriever):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def invoke(self, query: str, k: int):
        self.k = k
        return self._get_relevant_documents(query)

    def _get_relevant_documents(
        self, query , *, run_manager= None
    ):
        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self.vectorizer.transform([query])
        results = cosine_similarity(self.tfidf_array, query_vec).reshape((-1,))
        top_indices = results.argsort()[-self.k :][::-1]
        return_docs = [(self.docs[i], results[i]) for i in top_indices]
        return return_docs



class ExtendedChroma(Chroma):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def max_marginal_relevance_search_with_score(self,
        query: str,
        k: int = DEFAULT_K,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter = None,
        where_document = None,
        **kwargs,
    ):
        
        if self._embedding_function is None:
            raise ValueError(
                "For MMR search, you must specify an embedding function on" "creation."
            )

        embedding = self._embedding_function.embed_query(query)
        
        results = self._Chroma__query_collection(
            query_embeddings=embedding,
            n_results=fetch_k,
            where=filter,
            where_document=where_document,
            include=["metadatas", "documents", "distances", "embeddings"],
            **kwargs,
        )
        mmr_selected = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            results["embeddings"][0],
            k=k,
            lambda_mult=lambda_mult,
        )

        candidates = _results_to_docs_and_scores(results)

        selected_results = [r for i, r in enumerate(candidates) if i in mmr_selected]
        return selected_results
    
    def tfidf(self, query: str, k: int = DEFAULT_K, **kwargs):
        """
        Search the collection for the query and return the top k results with scores.
        """

        db_get = self.get()
        db_ids = db_get["ids"]
        if not db_ids:
            raise ValueError("No documents in the collection")

        doc_list = []
        for x in range(len(db_ids)):
            doc = db_get["documents"][x]
            doc_list.append(doc)
        
        retriever = ExtendedTFIDFRetriever.from_texts(doc_list)
        docs_and_scores = retriever.invoke(query, k)
        return docs_and_scores

    


# Only one retriever is needed, just query the vector DB with the given configurations (!)
class Retriever:
    def __init__(self, docStore='', client=True):
        self.vectorStores = {}
        self.storesToChunks = {}
        # Load the vector DBs from the store folder
        self.raptorDB = None
        self.loadVectorDBs()
        self.loadRaptorCollapsedTreeRetriever()
        self.docStore = docStore
        self.client = client


    # query mode can be 'vanilla' or 'raptor'
    def invoke(self, query, k, chunkSize, chunkOverlap, retrievalMode="vanilla", searchBy="semantic similarity", send=True, duplicate=False, id=None, dupDocsAndScores=None, dupDocsAndScoresRaptor=None, dupDocsAndScoresVanilla=None):
        query = str(query)
        k = int(k)
        chunkSize = int(chunkSize)
        chunkOverlap = int(chunkOverlap)
        # tell the interface if we are loading or not
        if (send and self.client) and not duplicate:
            requests.post('http://localhost:5001/set_loading', json={'loading': True})


        # query the vector DB
        collection_name = self.getCollectionName(chunkSize, chunkOverlap)

        # If the collection doesn't exist, warn with instructions to create it
        if collection_name not in self.vectorStores:
            logging.warning(f"Collection {collection_name} does not exist. Please use a pre-indexed size.")
            # Throw an error if the user tries to create a vector db without a doc store
            raise ValueError("Please provide a document store to create a vector DB")
            # self.createVectorDB(chunkSize=chunkSize, chunkOverlap=chunkOverlap)
            # collection_name = self.getCollectionName(chunkSize, chunkOverlap)

            

        vs = self.vectorStores[collection_name]
        docs_and_scores = dupDocsAndScores
        docs_and_scores_vanilla = dupDocsAndScoresVanilla
        docs_and_scores_raptor = dupDocsAndScoresRaptor

        if not duplicate:
            if searchBy == "semantic similarity":
                docs_and_scores_raptor = self.raptorDB.similarity_search_with_score(query, k=k+40)
                docs_and_scores_vanilla = vs.similarity_search_with_score(query, k=k+40)
            elif searchBy == "max marginal relevance":
                docs_and_scores_vanilla = vs.max_marginal_relevance_search_with_score(query=query, k=k+40, fetch_k=40)
                docs_and_scores_raptor = self.raptorDB.max_marginal_relevance_search_with_score(query=query, k=k+40, fetch_k=40)
            elif searchBy == "tfidf":
                docs_and_scores_vanilla = vs.tfidf(query=query, k=k+40)
                docs_and_scores_raptor = self.raptorDB.tfidf(query=query, k=k+40)
            else:
                raise ValueError("Invalid searchBy, must be 'semantic similarity', 'max marginal relevance', or 'tfidf', but was " + searchBy)                

                # docs_and_scores_vanilla = vs.full_text_search_with_score(query=query, k=k+100)
                # docs_and_scores_raptor = self.raptorDB.full_text_search_with_score(query=query, k=k+100)

        if not id:
            id = str(uuid.uuid4())

        # send the results and parameters to the Flask app
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

        if retrievalMode == "raptor":
            docs_and_scores = docs_and_scores_raptor
        elif retrievalMode == "vanilla":
            docs_and_scores = docs_and_scores_vanilla
        else:
            raise ValueError("Invalid retrieval mode, must be 'raptor' or 'vanilla'")

        # Fork and stop the forked process to wait for a finish running pipeline signal
        pid = os.fork()
        if pid == 0 and (self.client and send) or duplicate:
            # get the actual pid of the process
            pid = os.getpid()

            # register the process with the server
            response = requests.post(f"http://localhost:5001/register_process/{id}/{pid}")

            # Pause the process, until the server signals it to wake up
            # ...
            # Define the set of signals to wait for
            sigset = {signal.SIGUSR1}

            # Block the signals so they can be waited for
            signal.pthread_sigmask(signal.SIG_BLOCK, sigset)

            # sleep the process, until the server signals it to wake up
            while signal.sigwait(sigset) != signal.SIGUSR1:
                pass # wait

            # process has woken up and is ready to continue, get the new data from the server
            data = requests.get(f"http://localhost:5001/get_data/{id}").json()


            selectedChunks = data['selectedChunks'] # looks like ['chunk text 1', 'chunk text 2', ...]
            print("selectedChunks", selectedChunks)
            vanillaChunks = data['vanillaChunks'] # looks like [{'text': 'text', 'score': 0.5, 'id': 0}, ...]
            raptorChunks = data['raptorChunks'] # looks like [{'text': 'text', 'score': 0.5, 'id': 0}, ...]
            retrievalMode = data['retrievalMode'] # either 'raptor' or 'vanilla'

            # format docs and scores from selectedChunks and chunks depending on the retrieval mode
            docs_and_scores_raptor = [(Document(page_content=chunk['text']), chunk['score']) for chunk in raptorChunks if chunk in selectedChunks]
            docs_and_scores_vanilla = [(Document(page_content=chunk['text']), chunk['score']) for chunk in vanillaChunks if chunk in selectedChunks]

            print("docs_and_scores_raptor", docs_and_scores_raptor) 
            print("docs_and_scores_vanilla", docs_and_scores_vanilla)

            # join docs_and_score_raptor and docs_and_scores_vanilla
            docs_and_scores = docs_and_scores_raptor + docs_and_scores_vanilla
            """
            We want to make sure that we can handle n number of finish running requests as needed
            this will fork another process to do that, and not send anything to the interface, and associate
            it with the same id having this after the signal wait ensures that the process is not forked until
            the server is ready so we don't have infinite recursion and infinite process running
            """
            # tell the server to delete the old id:pid mapping
            requests.post(f"http://localhost:5001/delete_process/{id}")

            nextPipelinePID = os.fork()
            if nextPipelinePID == 0:
                # get the actual pid of the process
                pid = os.getpid()

                # Define the set of signals to wait for
                sigset = {signal.SIGUSR1}
                # Block the signals so they can be waited for
                signal.pthread_sigmask(signal.SIG_BLOCK, sigset)

                # regsiter the replacement process id with the server
                response = requests.post(f"http://localhost:5001/register_process/{id}/{pid}")
                self.invoke(query=query, k=k, chunkSize=chunkSize, chunkOverlap=chunkOverlap, searchBy=searchBy, send=False, duplicate=True, id=id, dupDocsAndScores=docs_and_scores, dupDocsAndScoresRaptor=docs_and_scores_raptor, dupDocsAndScoresVanilla=docs_and_scores_vanilla)


        # tell the interface if we are loading or not
        if (send and self.client) and not duplicate:
            requests.post('http://localhost:5001/set_loading', json={'loading': False})

        if self.client:
            return docs_and_scores[0:k]
        return docs_and_scores


    def loadVectorDBs(self):        
        """"
        Note the naming pattern for different collections. This allows us to load the DBs reliably.
        chroma_{chunk_size}_{chunk_overlap}
        e.g. collection_name = "chroma_1000_0"
        """
        # Load all vector DBs in the ./task/store folder
        for file in os.listdir("./task/store"):
            if file.startswith("chroma"):
                collection_name = file
                # parse the file name for the chunk size and chunk overlap
                try:
                    chunkSize = int(collection_name.split("_")[1])
                    chunkOverlap = int(collection_name.split("_")[2])
                    db = ExtendedChroma(
                        persist_directory=os.path.join("task/store/", collection_name),
                        collection_name=collection_name,
                        embedding_function=OpenAIEmbeddings(model="text-embedding-3-large", chunk_size=chunkSize)
                    )
                    self.vectorStores[collection_name] = db
                except Exception as e:
                    continue

      
    def loadRaptorCollapsedTreeRetriever(self):
        for file in os.listdir("./task/store"):
            if file.startswith("chroma_raptor_summaries"):
                collection_name = file
                db = ExtendedChroma(
                    persist_directory=os.path.join("task/store/", "chroma_raptor_summaries"),
                    embedding_function=OpenAIEmbeddings()
                )
                self.raptorDB = db
                return
        raise ValueError("Raptor DB chroma_raptor_summaries not found in ./task/store")



    def configExists(self, chunkSize, chunkOverlap):
        # Returns True if a vector DB with the given configuration exists in the folder
        collection_name = self.getCollectionName(chunkSize, chunkOverlap)
        return collection_name in self.vectorStores


    def createVectorDB(self, embeddings=OpenAIEmbeddings(model="text-embedding-3-large"), docs=None, chunkSize=1000, chunkOverlap=0):
        # Create a vector DB with the given configuration

        # Check if a vector DB with the given configuration exists
        collection_name = self.getCollectionName(chunkSize, chunkOverlap)
        if self.configExists(chunkSize, chunkOverlap):
            logging.warning(f"Collection {collection_name} already exists. Load it with the loadVectorDBs method")
            return collection_name

        if not docs:
            print("No docs passed in, using default docs from " + str(self.docStore))
            docs = self.getDocs(chunkSize, chunkOverlap)

        # Select embeddings
        if not embeddings:
            open_ai_key=os.environ["OPENAI_API_KEY"]
            embeddings = OpenAIEmbeddings(
                            openai_api_key=open_ai_key, 
                            model="text-embedding-3-large",
                            chunk_size=chunkSize
                            )

        texts = self.splitDocuments(docs, chunkSize, chunkOverlap)
        
        db = Chroma(
            persist_directory=os.path.join("task/store/", collection_name), 
            collection_name=collection_name, 
            embedding_function=embeddings
        )
        db.add_documents(texts)
        self.vectorStores[collection_name] = db
        return collection_name


    def splitDocuments(self, docs, chunkSize, chunkOverlap, lengthFunction=len, isSeperatorRegex=False):
        # Split documents into chunks
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
        nChunks = len(texts)

        # Save the chunks to the storesToChunks dictionary
        collection_name = self.getCollectionName(chunkSize, chunkOverlap)
        self.storesToChunks[collection_name] = texts

        return texts

        
    def getCollectionName(self, chunkSize, chunkOverlap):
        # Returns the collection name for the given configuration of chunkSize and chunkOverlap
        return f"chroma_{chunkSize}_{chunkOverlap}"


    def getDocs(self, chunkSize, chunkOverlap):
        all_chunks = []
        for filename in os.listdir(self.docStore):
            if filename.endswith(".pdf"):
                pdf_filename = os.path.join(self.docStore, filename)
                if not os.path.exists(pdf_filename):
                    print(f"file {pdf_filename} not found. Moving onto next.")
                else:
                    local_pdf_path = pdf_filename

                with open(local_pdf_path, "rb") as pdf_file:
                    docs = get_document_text(pdf_file, title=pdf_filename)
                all_chunks.extend(docs)

        return all_chunks


        # # TODO: update this so that it is based off the original doc store, and can take a folder as a well as a document
        # if self.docStore and not os.path.exists(self.docStore):
        #     math_analysis_of_logic_by_boole = "https://www.gutenberg.org/files/36884/36884-pdf.pdf"
        #     local_pdf_path = download_file(math_analysis_of_logic_by_boole, pdf_filename)
        # else:
        #     local_pdf_path = self.docStore

        # with open(local_pdf_path, "rb") as pdf_file:
        #     docs = get_document_text(pdf_file, title="Analysis of Logic")

        # return docs

    def sendRetrievalData(self, query, docs_and_scores_vanilla, retrievalMode, docs_and_scores_raptor, searchBy, chunkSize, chunkOverlap, k, id):
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
        # POST the data to the server
        response = requests.post('http://localhost:5001/send_retrieval_data', json=data)


    def getChunksAndSelectedChunks(self, docs_and_scores, k):
        chunks = []
        selectedChunks = [] # the set of indicies that are selected
        for index, (doc, score) in enumerate(docs_and_scores):
            chunk = {
                'text': doc.page_content,
                'score': score,
                'id': index,
            }
            if not self.containsChunk(chunk, chunks):
                chunks.append(chunk)
            if index < k:
                selectedChunks.append(chunk)

        return chunks, selectedChunks

    def containsChunk(self, chunk, chunks):
        for c in chunks:
            if c['text'] == chunk['text'] and c['score'] == chunk['score'] and c['id'] == chunk['id']:
                return True
        return False