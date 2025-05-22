"""
Example pipeline demonstrating the Raggy package for RAG (Retrieval-Augmented Generation).
This example uses hospital policy documents to answer questions about hospital procedures and policies.
"""
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

doc_dir = Path(__file__).parent.parent / "hospital-rag" / "documents" / "policy_pdfs_medium" 

# Import raggy
from raggy import llm, Retriever, Query, Answer

def main():
    """Run the example pipeline."""

    # Create a query
    query_text = "languages we translate"
    
    # Process query
    query = Query(query_text)

    REWRITE=f"Re-write this query for semantic similarity retrieval: {query_text} Only output the question."

    query = llm(prompt=REWRITE, max_tokens=4000, temperature=0.7)

    retriever = Retriever(
        docStore=str(doc_dir)
    )
    
    # Retrieve relevant documents
    docs_and_scores = retriever.invoke(
        query=str(query),
        k=1,
        chunkSize=2000,
        chunkOverlap=0,
        searchBy="semantic similarity"
    )
    

    context_string = ""

    for index, (doc, score) in enumerate(docs_and_scores):
        context_string += f"Document {index+1}:\n{doc.page_content}\n\n"

    QUERY="""
    Question: {query}
    Context: {docs}
    """

    # Generate response
    response = llm(prompt=QUERY.format(query=query, docs=context_string), max_tokens=4000, temperature=0.7)

    # Create and send answer
    Answer(response)

if __name__ == "__main__":
    main() 