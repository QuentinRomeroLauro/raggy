import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, Namespace, emit
from flask_cors import CORS
import requests
import uuid
import os
from interfaces.helpers.retrieve import Retriever
from interfaces.helpers.counter import read_counter, increment_counter, write_counter
from interfaces.helpers.trace import create_trace
from interfaces.helpers.llm import llm
import signal


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", logger=True) # add localhost:3000 to the list of allowed origins

# Initialize the counter
counter = read_counter()

# Initialize the waiting processes
waiting_processes = {}

# Initialize the data buffer to store changed data between awakening processes
data_buffer = {}

@app.route('/')
def index():
    return "WebSocket Server Running"

# TODO: have a init function that clears python processes and waiting processes for brand new pipeline runs
# TODO: clear waiting processes that were after the one we just ran


# LLM Data takes this structure:
# {
#      type: 'LLM',
#      promptText: 'This is a prompt',
#      responseText: 'This is a response',
#      temperature: 0.7,
#      maxTokens: 100,
# }
# This function is called from the llm.py file so that we can send any LLM requests to the user interface



def getChunksAndSelectedChunks(docs_and_scores, k):
    chunks = []
    selectedChunks = [] # the set of indicies that are selected
    for index, (doc, score) in enumerate(docs_and_scores):
        chunk = {
            'text': str(doc.page_content),
            'score': str(score),
            'id': str(index),
        }
        chunks.append(chunk)
        if index < int(k):
            selectedChunks.append(chunk)

    return chunks, selectedChunks


def runWhatIfLLM(prompt, max_tokens=100, temperature=0.7):
    res = llm(prompt=prompt, max_tokens=max_tokens, temperature=temperature, send=False)

    data = {
        'type': 'LLM',
        'promptText': prompt,
        'responseText': res,
        'temperature': temperature,
        'maxTokens': max_tokens,
        'order': increment_counter(), # maybe this shouldn't increment because it is a what if scenario
        'id': str(uuid.uuid4()),
    }

    return data


def runWhatIfRetrieval(query, searchBy, chunkSize, chunkOverlap, k, retrievalMode):
    retriever = Retriever(docStore="./task/documents/policy_pdfs", client=False)
    docs_and_scores_vanilla = retriever.invoke(
        query=str(query),
        searchBy=str(searchBy),
        chunkSize=int(chunkSize),
        chunkOverlap=int(chunkOverlap),
        k=int(k),
        send=False,
        retrievalMode='vanilla',
    )

    docs_and_scores_raptor = retriever.invoke(
        query=str(query),
        searchBy=str(searchBy),
        chunkSize=int(chunkSize),
        chunkOverlap=int(chunkOverlap),
        k=int(k),
        send=False,
        retrievalMode='raptor',
    )

    if retrievalMode == "raptor":
        vanillaChunks, __ = getChunksAndSelectedChunks(docs_and_scores_vanilla, 0)
        raptorChunks, selectedChunks = getChunksAndSelectedChunks(docs_and_scores_raptor, k)
    elif retrievalMode == "vanilla":
        vanillaChunks, selectedChunks = getChunksAndSelectedChunks(docs_and_scores_vanilla, k)
        raptorChunks, __ = getChunksAndSelectedChunks(docs_and_scores_raptor, 0)
    else:
        raise ValueError("Invalid retrieval mode, must be 'raptor' or 'vanilla'")

    id = str(uuid.uuid4())

    data = {
        'type': 'Retrieval',
        'query': query,
        'vanillaChunks': vanillaChunks,
        'raptorChunks': raptorChunks,
        'selectedChunks': selectedChunks,
        'searchBy': searchBy,
        'chunkSize': chunkSize,
        'chunkOverlap': chunkOverlap,
        'retrievalMode': retrievalMode,
        'k': k,
        'order': increment_counter(), # maybe this shouldn't increment because it is a what if scenario
        'id': id,
    }

    return data


@app.route('/send_retrieval_data', methods=['POST'])
def send_retrieval_data():
    data = request.json
    socketio.emit('retrieval', data)
    return f"Retrieval Data {str(data)} sent"

@app.route('/send_answer', methods=['POST'])
def send_answer():
    data = request.json
    socketio.emit('answer', data)
    return f"Answer {str(data)} sent"


@app.route('/send_query_data', methods=['POST'])
def send_query_data():
    kill_all_waiting_processes()
    data = request.json
    socketio.emit('query', data)
    return f"Query Data {str(data)} sent"


@app.route('/send_llm_data', methods=['POST'])
def send_llm_data():
    data = request.json  # This will parse the JSON sent to this route
    socketio.emit('llm', data)  # Emit the received data
    return f"LLM Data {str(data)} sent"

@app.route('/get_whatIf_data', methods=['POST'])
def get_whatIf_data():
    data = request.json
    print("data", data)
    data = runWhatIfRetrieval(query=data['query'], 
        searchBy=data['searchBy'], 
        chunkSize=data['chunkSize'], 
        chunkOverlap=data['chunkOverlap'], 
        k=data['k'], 
        retrievalMode=data['retrievalMode']
    )
    return jsonify(data), 200


@app.route('/finish_running_pipeline', methods=['POST'])
def finish_running_pipeline():
    print("Running finish running pipeline sequence")
    data = request.json
    id = data.get('id')
    print("waiting processes", waiting_processes)
    # Load the associated stopped process pid with the id
    # some fix:? while pid not in waiting_processes: pass, and then get the id? 
    pid = waiting_processes[id]

    # Parse if the data is a retrieval or LLM or query
    if data['type'] == 'Retrieval':
        # Change the return of the invoke on the associated clone process 
        # to return the selected chunks
        data_buffer[id] = data

        # Run the pipeline / program to finish
        print("Running Retrieval Pipeline")
    elif data['type'] == 'LLM':
        # Change the return of the invoke on the associated clone process
        # to return the responseText
        data_buffer[id] = data
    elif data ['type'] == 'Query':
        data_buffer[id] = data

        # Run the pipeline / program to finish 
        print("Running LLM Pipeline")

    # Run the pipeline
    awake_process(id)
    
    return jsonify(data), 200
    

@app.route('/get_whatIf_llm', methods=['POST'])
def get_whatIf_llm():
    data = request.json
    print("data", data)
    data = runWhatIfLLM(data['prompt'], data['max_tokens'], data['temperature'])
    return jsonify(data), 200


# registers a process
@app.route('/register_process/<unique_id>/<pid>/', methods=['POST'])
def register_process(unique_id, pid):
    print("regsitering process", unique_id, pid)
    waiting_processes[unique_id] = int(pid)
    return f"Process {pid} registered with id {unique_id}"

# delete a process from the waiting_processes
@app.route('/delete_process/<unique_id>/', methods=['POST'])
def delete_process(unique_id):
    print("deleting process", unique_id)
    del waiting_processes[unique_id]
    return f"Process with id {unique_id} deleted"


@app.route('/get_data/<unique_id>/', methods=['GET'])
def get_data(unique_id):
    return jsonify(data_buffer[unique_id])


@app.route('/show_waiting_processes', methods=['GET'])
def show_waiting_processes():
    return jsonify(waiting_processes)


@app.route('/get_traces', methods=['GET'])
def get_traces():
    try:
        with open('traces/trace.jsonl', 'r') as file:
            traces = file.readlines()
    except: 
        # if the folder or file doesn't exist, return an empty list
        return jsonify([])
    return jsonify(traces)

@app.route('/set_loading', methods=['POST'])
def set_loading():
    data = request.json
    socketio.emit('loading', data)
    return f"Loading {str(data)} sent"

from interfaces.helpers.evaluate import evaluate_traces_embedding_distance

@app.route('/get_evaluation_embedding_distance', methods=['POST'])
def get_evaluation_embedding_distance():
    data = request.json
    query = data['query']
    answer = data['answer']
    score = evaluate_traces_embedding_distance(query, answer)
    return jsonify(score)
    

@app.route('/save_trace', methods=['POST'])
def add_trace():
    data = request.json
    if data['type'] == 'Retrieval':
        selectedChunks = data['selectedChunks']
        chunks = data['chunks']
        output = [chunks[int(i)]['text'] for i in selectedChunks]
        data = create_trace(data['query'], output, data['type'])
    elif data['type'] == 'LLM':
        data = create_trace(data['promptText'], data['responseText'], data['type'])
    elif data['type'] == 'Answer':
        data = create_trace(data['query'], data['answer'], data['type'])
    socketio.emit('traces', data)
    return f"Trace for {str(data)} added"


def awake_process(unique_id):
    print(f"waiting_processes {waiting_processes}")
    pid = waiting_processes[unique_id]
    os.kill(pid, signal.SIGUSR1)
    return f"Process {pid} with id {unique_id} has been awakened"


def kill_all_waiting_processes():
    for pid in waiting_processes.values():
        os.kill(pid, signal.SIGTERM)
    return "All waiting processes have been killed"


if __name__ == '__main__':
    # Using port 5001 b/c Apple Airplay runs on port 5000 which can cause conflicts
    socketio.run(app, debug=True, port=5001) 
