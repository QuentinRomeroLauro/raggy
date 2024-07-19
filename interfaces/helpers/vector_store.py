"""
This code was adapted from https://github.com/streamlit/example-app-langchain-rag/blob/main/
which is licensed under the Apache License 2.0. 
"""
import logging
import os
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from .local_loader import get_document_text
from .remote_loader import download_file
from .splitter import split_documents
from dotenv import load_dotenv
from time import sleep


# This happens all at once, not ideal for large datasets.
def create_vector_db(texts, embeddings=None, collection_name="chroma"):
    if not texts:
        logging.warning("Empty texts passed in to create vector database")
    # Select embeddings
    if not embeddings:
        # To use HuggingFace embeddings instead:
        # from langchain_community.embeddings import HuggingFaceEmbeddings
        # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        openai_api_key = os.environ["OPENAI_API_KEY"]
        # Don't use old embeddings.
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-large")

    # Create a vectorstore from documents
    # this will be a chroma collection with a default name.
    db = Chroma(collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=os.path.join("store/", collection_name))
    db.add_documents(texts)

    return db


def find_similar(vs, query, k=5):
    docs_and_scores = vs.similarity_search_with_score(query, k=k)
    return docs_and_scores


def main():
    pass

if __name__ == "__main__":
    main()