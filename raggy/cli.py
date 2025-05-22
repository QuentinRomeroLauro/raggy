"""
Command line interface for Raggy
"""
import argparse
from .core.query import Query
from .core.retriever import Retriever
from .core.llm import llm

def main():
    parser = argparse.ArgumentParser(description="Raggy - A RAG pipeline framework")
    parser.add_argument("query", help="The query to process")
    parser.add_argument("--debug", action="store_true", help="Start the debug interface")
    parser.add_argument("--docstore", default="./documents", help="Path to the document store")
    
    args = parser.parse_args()
    
    # Initialize components
    query = Query(args.query, debug=args.debug)
    retriever = Retriever(docStore=args.docstore)
    
    # Process the query
    docs_and_scores = retriever.invoke(query=query, k=5, chunkSize=400, chunkOverlap=0)
    
    # Generate response
    RAG_query = """
    Given these relevant passages from an external context, answer the following question. 
    Question: {query}
    Context: {docs}
    """
    
    response = llm(
        prompt=RAG_query.format(query=query, docs=docs_and_scores),
        max_tokens=4000,
        temperature=0.7
    )
    
    print(response)

if __name__ == "__main__":
    main() 