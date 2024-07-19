import json

def create_trace(input, output, type):
    """
    Create a trace for the given input, output, and step type
    input and output are strings
    type is a string: 'Retrieval' | 'Answer' | 'LLM' | 'Query' 
    """
    data = {
        'type': type,
        'input': input,
        'output': output,
    }

    # write the data to the .jsonl trace file
    with open('trace.jsonl', 'a') as file:
        file.write(json.dumps(data) + '\n')