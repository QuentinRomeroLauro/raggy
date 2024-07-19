import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces import llm
from interfaces import create_vector_db
from interfaces import Retriever
from interfaces import Query
from interfaces import Answer

from dotenv import load_dotenv
from interfaces import get_document_text
from interfaces import download_file

from interfaces import find_similar
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import json
load_dotenv()

# Run the pipeline in the main function
def main():
    pdf_filename = "./task/documents/QuentinRomeroLauro-SWE-Resume-24.pdf"
    if not os.path.exists(pdf_filename):
        math_analysis_of_logic_by_boole = "https://www.gutenberg.org/files/36884/36884-pdf.pdf"
        local_pdf_path = download_file(math_analysis_of_logic_by_boole, pdf_filename)
    else:
        local_pdf_path = pdf_filename

    with open(local_pdf_path, "rb") as pdf_file:
        docs = get_document_text(pdf_file, title="Analysis of Logic")

    # Initialize Retriever
    retriever = Retriever(docStore="./task/documents/QuentinRomeroLauro-SWE-Resume-24.pdf")
    # Create a vector DB managed by the retriever with the given configurations
    retriever.createVectorDB(docs=docs, chunkSize=1000, chunkOverlap=10)


    QUERY_DECOMP="""
    You are an expert query decomposer. You are given a query from the user and you have to do one of the following:
    - If the query is simple, and would not benefit from decomposing, output the query in the array as the only element.
    - If the query is complex, and would benefit from decomposing, output a list of simpler subqueries that would help answer the query. Make sure queries are full sentences.

    Here is the query you have to decompose:
    {query}

    Give your output as a json parseable python list of strings.
    """
    
    query = Query(query="What are the different ways I can contact the applicant")

    res = llm(prompt=QUERY_DECOMP.format(query=query), max_tokens=100, temperature=0.7)
    queries = json.loads(res)

    ASK_OR_RETRIEVE="""
    You are an expert in determining whether a question should be answered by asking the user or retrieving the information from a database. You are given a query and a list of subqueries. You have to determine whether the query should be answered by asking the user or retrieving the information from a database.
    Here is the question:
    {query}
    Give your output as a string with the value "ask" or "retrieve".
    """

    answered_questions = ""

    retreived_context = ""

    for q in queries:
        res = llm(prompt=ASK_OR_RETRIEVE.format(query=q), max_tokens=100, temperature=0.7)
        if res == "retrieve":
            # Query the vector DB
            docs_and_scores = retriever.invoke(
                query=q,
                k=4,
                chunkSize=100,
                chunkOverlap=10,
            )

            # add all the relevant information to the context string
            for doc, score in docs_and_scores:
                retreived_context += doc.page_content + "\n"


        else:
            ans = llm(prompt=q, max_tokens=100, temperature=0.7)
            answered_questions += f"The query {q} has the answer: {ans}\n"

    final_prompt = f"""You are an expert at answering questions based on answers to sub questions and retrieved context.
    Here is the query you have to answer: {query}
    Here are the subqueries and their answers: {answered_questions}
    Here is the retrieved context: {retreived_context}
    Give a concise answer to the user's query:{query}"""

    final_ans = llm(prompt=final_prompt, max_tokens=100, temperature=0.7)
    
    Answer(final_ans)



# DO NOT REMOVE
from interfaces import run_interface
if __name__ == "__main__":
    # run with the interface
    run_interface()

    # run the pipeline
    main()