"""
Copy of the original pipeline.py to copy and paste
for the start of other pipelines.
"""
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

Consult the ground-truth answers in `TASK.md` to see what the hospital system's
ideal answer is.
"""
def main():
    # use to quickly switch between different queries
    query_list = [
        "What is the over-award policy?",
        "Who is on the investigational drug services team?",
        "What languages can we translate in the hospital?",
        "Can we translate Spanish?",
        "What is the safety policy for job shadowing at Berkeley Medical Center?",
        "How many beds are in Berkeley Medical Center?",
        "How many beds are in Jefferson Medical Center?",
        "What are the different admissions policies?",
    ]

    # Initialize the retriever with the document store
    retriever = Retriever(docStore="./task/documents/policy_pdfs")

    # Use the Query class to wrap a query for the interface
    query = Query(query=query_list[2])

    rewriter_prompt = """GPT, rewrite this question in order to improve the retrieval on a semantic index and tf-idf index. 

    Original Question: {query}

    Return only the rewritten question."""

    # query_rewritten = llm(prompt=rewriter_prompt.format(query=str(query)), max_tokens=30, temperature=0.7)

    # use the retriever to get the relevant documents and scores
    docs_and_scores = retriever.invoke(
            query=str(query), 
            k=5,
            chunkSize=200,
            chunkOverlap=0,
            searchBy="semantic similarity"
        )

    # add all the relevant information to the context string
    retreived_context = ""
    for doc, score in docs_and_scores:
        retreived_context += doc.page_content + "\n"

    # Here's an example of how to create a prompt
    RAG_query="""
    Given these relevant passages from an external context, answer the following question. 
    Question: {query}
    Context: {docs}
    """

    # make a call to the language model by using `llm()`
    response = llm(
        prompt=RAG_query.format(query=query, docs=retreived_context), 
        max_tokens=4000, 
        temperature=0.7
    )

    response = llm(
        prompt=RAG_query.format(query=query, docs=retreived_context), 
        max_tokens=4000, 
        temperature=0.7
    )

    response = llm(
        prompt=RAG_query.format(query=query, docs=retreived_context), 
        max_tokens=4000, 
        temperature=0.7
    )

    # Use Answer class to wrap the answer for the interface
    Answer(response)

if __name__ == "__main__":
    main()