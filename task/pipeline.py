import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces import llm
from interfaces import Retriever
from interfaces import Query
from interfaces import Answer
from dotenv import load_dotenv
import json
load_dotenv()
# ------ Don't worry about anything above this line ------


"""
We've provided a basic RAG pipeline that you can use to get started.

Iterate on the pipeline to improve the quality of the answers.

Consult the ground-truth answers in `task/answers` to see what the hospital's 
ideal answer is. Get as close to them as you can.
"""
def main():
    # Initialize the retriever with the document store
    retriever = Retriever(docStore="./task/documents/policy_pdfs")

    # Use the Query class to wrap a query for the interface
    query = Query(query="What is the dress code at Ruby Memorial Hospital for doctors?")

    # use the retriever to get the relevant documents and scores
    docs_and_scores = retriever.invoke(
                query=str(query),
                k=10,
                chunkSize=1000,
                chunkOverlap=10,
            )

    # add all the relevant information to the context string
    retreived_context = ""
    for doc, score in docs_and_scores:
        retreived_context += doc.page_content + "\n"

    # Here's an example of how to create a prompt
    RAG_query="""
    Given these relavant passages from hospital policy documents, answer the following question. 
    Make sure to include all the relavant information provided in the context. 
    Question: {query}
    Policy documents: {docs}
    """

    # make a call to the language model by using `llm()`
    response = llm(
        prompt=RAG_query.format(query=query, docs=retreived_context), 
        max_tokens=400, 
        temperature=0
    )

    # Use Answer class to wrap the answer for the interface
    Answer(response)



# DO NOT REMOVE
from interfaces import run_interface
if __name__ == "__main__":
    # run the pipeline
    main()