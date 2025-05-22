import json
import os

def create_trace(input, output, type):
    """
    Create a trace for the given input, output, and step type
    input and output are strings
    type is a string: 'Retrieval' | 'Answer' | 'LLM' | 'Query' 
    """
    length = len(open('traces/trace.jsonl', 'r').readlines())

    data = {
        'type': type,
        'input': input,
        'output': output,
        'id': length,
    }

    # Ensure the "traces" folder exists
    os.makedirs('traces', exist_ok=True)

    # Write the data to the trace.jsonl file inside the "traces" folder
    with open('traces/trace.jsonl', 'a') as file:
        file.write(json.dumps(data) + '\n')
    
    return data

def get_traces():
    """
    Get all traces from the trace.jsonl file
    """
    try:
        with open('traces/trace.jsonl', 'r') as file:
            traces = file.readlines()
    except: 
        # if the folder or file doesn't exist, return an empty list
        return []
    return traces

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