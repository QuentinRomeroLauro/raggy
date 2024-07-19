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
import signal
# Naming pattern for different collections, that contains information about the parameters used to create the collection
# chunk_size, chunk_overlap
# collection_name = "chroma_1000_0"
# then, in our retrievr code, we can use this information to load the correct collection, or create a new one if it doesn't exist
# we should also have some logic just to look for collections that have the same parameters, and if they exist, we should load them instead of creating a new one

# we should also have different search functions that would work on all of the collections, so that the user can easily chansge them 


# Only one retriever is needed, just query the vector DB with the given configurations (!)
class Retriever:
    def __init__(self, docStore='', client=True):
        self.vectorStores = {}
        self.storesToChunks = {}
        # Load the vector DBs from the store folder
        self.loadVectorDBs()
        self.docStore = docStore
        self.client = client


    def invoke(self, query, k, chunkSize, chunkOverlap, searchBy="similarity", send=True, duplicate=False, id=None, dupDocsAndScores=None):
        # tell the interface if we are loading or not
        if (send and self.client) and not duplicate:
            requests.post('http://localhost:5001/set_loading', json={'loading': True})


        # query the vector DB
        collection_name = self.getCollectionName(chunkSize, chunkOverlap)

        # If the collection doesn't exist, warn with instructions to create it
        if collection_name not in self.vectorStores:
            logging.warning(f"Collection {collection_name} does not exist. Creating it with default docs.")
            self.createVectorDB(chunkSize=chunkSize, chunkOverlap=chunkOverlap)
            collection_name = self.getCollectionName(chunkSize, chunkOverlap)
            

        vs = self.vectorStores[collection_name]
        docs_and_scores = dupDocsAndScores

        if not duplicate:
            if searchBy == "similarity" or True:
                # retriever = vs.as_retriever(search_type="similarity")
                # docs_and_scores = retriever.invoke(query, k=k)
                docs_and_scores = vs.similarity_search_with_score(query, k=k+100)
            elif searchBy == "mmr":
                retriever = vs.as_retriever(search_type="mmr")
                docs_and_scores = retriever.invoke(query, k=k+100)
            elif searchBy == "bm25":
                pass

        if not id:
            id = str(uuid.uuid4())

        # send the results and parameters to the Flask app
        if self.client and send:
            self.sendRetrievalData(
                query=query,
                docs_and_scores=docs_and_scores,
                searchBy=searchBy,
                chunkSize=chunkSize,
                chunkOverlap=chunkOverlap,
                k=k,
                id=id,
            )

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
            selectedChunks = data['selectedChunks'] # looks like [0, 1, 2, 3]
            chunks = data['chunks'] # looks like [{'text': 'text', 'score': 0.5, 'id': 0}, ...]

            # format docs and scores from selectedChunks and chunks
            docs_and_scores = [(Document(page_content=chunk['text']), chunk['score']) for chunk in chunks if chunk['id'] in selectedChunks]

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
                self.invoke(query=query, k=k, chunkSize=chunkSize, chunkOverlap=chunkOverlap, searchBy=searchBy, send=False, duplicate=True, id=id, dupDocsAndScores=docs_and_scores)


        # tell the interface if we are loading or not
        if (send and self.client) and not duplicate:
            requests.post('http://localhost:5001/set_loading', json={'loading': False})

        return docs_and_scores[0:k]


    def loadVectorDBs(self):
        # Load all vector DBs in the ./task/store folder
        for file in os.listdir("./task/store"):
            if file.startswith("chroma"):
                collection_name = file
                # parse the file name for the chunk size and chunk overlap
                chunkSize = int(collection_name.split("_")[1])
                chunkOverlap = int(collection_name.split("_")[2])
                print("found db in ./task/store", file)
                db = Chroma(
                    persist_directory=os.path.join("task/store/", collection_name),
                    collection_name=collection_name,
                    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large", chunk_size=chunkSize)
                )
                self.vectorStores[collection_name] = db


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

        print(f"Split into {nChunks} chunks")
        return texts

        
    def getCollectionName(self, chunkSize, chunkOverlap):
        # Returns the collection name for the given configuration of chunkSize and chunkOverlap
        return f"chroma_{chunkSize}_{chunkOverlap}"


    def getDocs(self, chunkSize, chunkOverlap):
        # TODO: update this so that it is based off the original doc store, and can take a folder as a well as a document
        if self.docStore and not os.path.exists(self.docStore):
            math_analysis_of_logic_by_boole = "https://www.gutenberg.org/files/36884/36884-pdf.pdf"
            local_pdf_path = download_file(math_analysis_of_logic_by_boole, pdf_filename)
        else:
            local_pdf_path = self.docStore

        with open(local_pdf_path, "rb") as pdf_file:
            docs = get_document_text(pdf_file, title="Analysis of Logic")

        return docs

    def sendRetrievalData(self, query, docs_and_scores, searchBy, chunkSize, chunkOverlap, k, id):
        chunks, selectedChunks = self.getChunksAndSelectedChunks(docs_and_scores, k)

        data = {
            'type': 'Retrieval',
            'query': query,
            'chunks': chunks,
            'selectedChunks': selectedChunks,
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
            chunks.append(chunk)
            if index < k:
                selectedChunks.append(index)

        return chunks, selectedChunks