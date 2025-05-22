"""
Helper functions used to evaluate based on traces
"""

from langchain.evaluation import EmbeddingDistance
from langchain.evaluation import load_evaluator
from pathlib import Path
import json
import os

def get_traces_dir():
    """Get the absolute path to the traces directory"""
    return Path(__file__).parent.parent / "traces"

def get_traces_file():
    """Get the absolute path to the traces file"""
    return get_traces_dir() / "trace.jsonl"

def create_trace(input, output, type):
    """
    Create a trace for the given input, output, and step type
    input and output are strings
    type is a string: 'Retrieval' | 'Answer' | 'LLM' | 'Query' 
    """
    traces_file = get_traces_file()
    traces_dir = get_traces_dir()
    
    # Ensure the "traces" folder exists
    traces_dir.mkdir(parents=True, exist_ok=True)
    
    # Get current length if file exists
    length = 0
    if traces_file.exists():
        length = len(traces_file.read_text().splitlines())

    data = {
        'type': type,
        'input': input,
        'output': output,
        'id': length,
    }

    # Write the data to the trace.jsonl file
    with open(traces_file, 'a') as file:
        file.write(json.dumps(data) + '\n')
    
    return data

def get_traces():
    """
    Get all traces from the trace.jsonl file
    """
    traces_file = get_traces_file()
    try:
        if traces_file.exists():
            return traces_file.read_text().splitlines()
        return []
    except Exception as e:
        print(f"Error reading traces: {e}")
        return []

def get_traces_for_query(query):
    """
    Get all traces for a given query
    """
    traces = get_traces()
    query_traces = []
    for trace in traces:
        trace = json.loads(trace)
        if trace['input'] == query:
            query_traces.append(trace)
    return query_traces

def evaluate_traces_embedding_distance(query, answer):
    """
    Evaluate the answer based on traces, returns the min embedding distance
    based on the traces.
    """
    evaluator = load_evaluator("embedding_distance")

    traces = get_traces_for_query(query)
    distances = []
    for trace in traces:
        res = evaluator.evaluate_strings(reference=trace['output'], prediction=answer)
        distances.append(res['score'])

    return 1 - min(distances)