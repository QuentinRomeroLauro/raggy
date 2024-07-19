"""
This code is used to load the documents into the vectorstore for the first time and create a vector database.
"""


import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from interfaces.helpers.local_loader import get_document_text
from interfaces.helpers.remote_loader import download_file
from interfaces.helpers.splitter import split_documents
from interfaces.helpers.vector_store import create_vector_db

load_dotenv()

pdf_filename = "./documents/QuentinRomeroLauro-SWE-Resume-24.pdf"

if not os.path.exists(pdf_filename):
    math_analysis_of_logic_by_boole = "https://www.gutenberg.org/files/36884/36884-pdf.pdf"
    local_pdf_path = download_file(math_analysis_of_logic_by_boole, pdf_filename)
else:
    local_pdf_path = pdf_filename

print(f"PDF path is {local_pdf_path}")

with open(local_pdf_path, "rb") as pdf_file:
    docs = get_document_text(pdf_file, title="Analysis of Logic")

texts = split_documents(docs, chunk_size=1000, chunk_overlap=0, length_function=len, is_separator_regex=False)
vs = create_vector_db(texts, embeddings=OpenAIEmbeddings(), collection_name="chroma")

