"""
Server module for handling the debug interface backend
"""
import eventlet
import eventlet.wsgi
eventlet.monkey_patch()  # (Optional but recommended for compatibility)

import os
import sys
import json
import subprocess
import uuid
import signal
import requests
import psutil
import socket

from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from pathlib import Path
import logging
from raggy.core.llm import llm
from raggy.core.retriever import Retriever
from raggy.interfaces.helpers.counter import increment_counter
from raggy.interfaces.helpers.evaluate import evaluate_traces_embedding_distance, create_trace



# Add the parent directory to the Python path to allow sibling imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Assuming your Raggy core components are importable like this
# Adjust imports based on your actual project structure if necessary

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the path to the static frontend files
# Assumes server.py is in raggy/interfaces/ and the frontend build is in raggy/interfaces/front-end/dist/
STATIC_FOLDER = os.path.join(os.path.dirname(__file__), 'front-end', 'dist')

# Initialize Flask app and SocketIO
# Configure Flask to serve static files from the STATIC_FOLDER
app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path='')
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Store process mappings
process_mappings = {}
data_store = {}


# Global dictionaries to track waiting processes & data buffer
waiting_processes = {}
data_buffer = {}

def maybe_build_frontend():
    """
    Check if the front-end build already exists. 
    If not, run `npm run build` in the front-end directory.
    """
    dist_index = os.path.join(STATIC_FOLDER, "index.html")
    if not os.path.exists(dist_index):
        logger.info("No front-end build found. Building now (npm run build)...")
        front_end_dir = os.path.join(os.path.dirname(__file__), 'front-end')
        try:
            subprocess.check_call(["npm", "install"], cwd=front_end_dir)
            subprocess.check_call(["npm", "run", "build"], cwd=front_end_dir)
            logger.info("Front-end build completed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to build front-end. Error: {e}")

@socketio.on('connect')
def handle_connect():
    """Handles new client connections."""
    logger.info(f"Client connected: {request.sid}")
    emit('status', {'message': 'Connected to Raggy debug server.'})
    # Optionally send initial state if available

@socketio.on('disconnect')
def handle_disconnect(sid):
    """Handles client disconnections."""
    logger.info(f"Client disconnected: {sid}")
    return None

@socketio.on('request_initial_data')
def handle_initial_data_request():
    """Sends initial data or configuration to the frontend upon request."""
    logger.info(f"Client {request.sid} requested initial data.")
    # Example: Send configuration or current status
    # Replace with actual data relevant to your application
    emit('initial_data', {'config': 'example_config', 'status': 'ready'})

@app.route('/set_loading', methods=['POST'])
def set_loading():
    """Set loading state"""
    data = request.json
    socketio.emit('loading', data)
    return jsonify({'status': 'success'})

@app.route('/send_query_data', methods=['POST'])
def send_query_data():
    """Send query data to frontend"""
    data = request.json
    socketio.emit('query_data', data)
    data_store[data['id']] = data
    return jsonify({'status': 'success'})

@app.route('/send_llm_data', methods=['POST'])
def send_llm_data():
    """Send LLM data to frontend"""
    data = request.json
    socketio.emit('llm_data', data)
    data_store[data['id']] = data
    return jsonify({'status': 'success'})

@app.route('/send_retrieval_data', methods=['POST'])
def send_retrieval_data():
    """Send retrieval data to frontend"""
    data = request.json
    socketio.emit('retrieval_data', data)
    data_store[data['id']] = data
    return jsonify({'status': 'success'})

@app.route('/send_answer', methods=['POST'])
def send_answer():
    """Send answer data to frontend"""
    data = request.json
    socketio.emit('answer_data', data)
    data_store[data['id']] = data
    return jsonify({'status': 'success'})

@app.route('/register_process/<id>/<pid>', methods=['POST'])
def register_process(id, pid):
    """Register a process for a given ID"""
    try:
        # Store process info
        process_mappings[id] = int(pid)
        return jsonify({"status": "ok"})
    except Exception as e:
        logger.error(f"Failed to register process: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/delete_process/<id>', methods=['POST'])
def delete_process(id):
    """Delete a process mapping"""
    if id in process_mappings:
        del process_mappings[id]
    return jsonify({'status': 'success'})

@app.route('/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint to verify the debugging server is running.
    Returns a 200 OK status when the server is operational.
    """
    return jsonify({"status": "ok", "service": "debug_interface"}), 200

@app.route('/get_data/<id>', methods=['GET'])
def get_data(id):
    """Get data for a given ID"""
    return jsonify(data_store.get(id, {}))

@app.route('/get_updated_interface_data/<id>', methods=['GET'])
def get_updated_interface_data(id):
    """Get updated interface data"""
    from datetime import datetime
    print("Current time:", datetime.now())
    return jsonify(data_buffer.get(id, {}))

@app.route('/get_traces', methods=['GET'])
def get_traces():
    """Get all traces for debugging"""
    # Use an absolute path based on the location of the server.py file
    traces_dir = Path(__file__).parent / "traces"
    
    # Create the directory if it doesn't exist
    if not traces_dir.exists():
        traces_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created traces directory at {traces_dir}")
    
    traces_file = traces_dir / "trace.jsonl"
    
    # Check if the file exists
    if not traces_file.exists():
        logger.info(f"Traces file not found at {traces_file}")
        return jsonify([])
    
    try:
        with open(traces_file, 'r') as file:
            traces = file.readlines()
            logger.info(f"Read {len(traces)} traces from {traces_file}")
            return jsonify(traces)
    except Exception as e:
        logger.error(f"Error reading traces file: {e}")
        return jsonify([])

@app.route('/create_test_trace', methods=['GET'])
def create_test_trace():
    """Create a test trace for debugging"""
    traces_dir = Path(__file__).parent / "traces"
    traces_file = traces_dir / "trace.jsonl"
    
    # Create the directory if it doesn't exist
    if not traces_dir.exists():
        traces_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple test trace
    test_trace = {
        'input': 'What is the answer to the question?',
        'output': 'The answer to the question is 42.',
    }
    
    try:
        with open(traces_file, 'w') as file:
            file.write(json.dumps(test_trace) + '\n')
        return jsonify({"status": "success", "message": "Test trace created"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_spa(path):
    """Serves the single-page application's entry point (index.html)."""
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        # If the path exists as a static file, serve it directly
        return send_from_directory(app.static_folder, path)
    else:
        # Otherwise, serve the index.html file (for SPA routing)
        index_path = os.path.join(app.static_folder, 'index.html')
        if not os.path.exists(index_path):
            logger.error(f"index.html not found in static folder: {app.static_folder}")
            return jsonify({"error": "Frontend not built or index.html missing."}), 404
        return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({"status": "running"})

@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f"Server error: {error}")
    return jsonify({"error": str(error)}), 500

@socketio.on_error()
def error_handler(e):
    logger.error(f"SocketIO error: {e}")

def start_dev_server(frontend_dir: str):
    """
    Spawn the React dev server on port 3000 using `npm start`.
    Logs are forwarded to the current console (stdout/stderr).
    Returns the Popen object so we can kill it later.
    """
    logger.info("Starting React dev server (npm start) on port 3000...")
    logger.info(f"Frontend directory: {frontend_dir}")
    process = subprocess.Popen(
        ["npm", "start"],
        cwd=frontend_dir,
        stdout=sys.stdout,  # Redirect dev server logs to this console
        stderr=sys.stderr
    )
    logger.info(f"Dev server started with PID {process.pid}")
    return process

def start_flask_server(host="127.0.0.1", port=5001):
    """
    Blocking call that runs the Flask-SocketIO server on the given host/port.
    """
    logger.info(f"Starting Flask-SocketIO server on http://{host}:{port}")
    
    socketio.run(app, host=host, port=port, debug=True)
    # socketio.run(app, host=host, port=port, debug=False)

#############################################
# NEW: Support for "what if" analyses below #
#############################################

def runWhatIfRetrieval(query, searchBy, chunkSize, chunkOverlap, k, retrievalMode):
    """
    Run a retrieval invocation without streaming, 
    returning the data structure directly (a 'what if' scenario).
    """
    # Get the correct path relative to server.py
    server_dir = Path(__file__).parent.parent.parent  # Go up to raggy root
    doc_path = server_dir / "examples" / "hospital-rag" / "documents" / "policy_pdfs_medium"
    print(f"doc_path: {doc_path}")
    retriever = Retriever(docStore=str(doc_path), client=False)

    # "vanilla" retrieval
    docs_and_scores_vanilla = retriever.invoke(
        query=str(query),
        searchBy=str(searchBy),
        chunkSize=int(chunkSize),
        chunkOverlap=int(chunkOverlap),
        k=int(k),
        send=False,
        retrievalMode='vanilla'
    )

    # "raptor" retrieval
    docs_and_scores_raptor = retriever.invoke(
        query=str(query),
        searchBy=str(searchBy),
        chunkSize=int(chunkSize),
        chunkOverlap=int(chunkOverlap),
        k=int(k),
        send=False,
        retrievalMode='raptor'
    )

    # Build chunk arrays
    def getChunksAndSelectedChunks(docs_and_scores, kint: int):
        chunks = []
        selectedChunks = []
        for index, (doc, score) in enumerate(docs_and_scores):
            chunk = {
                'text': str(doc.page_content),
                'score': float(score),
                'id': index,
            }
            chunks.append(chunk)
            if int(index) < int(kint):
                selectedChunks.append(chunk)
        return (chunks, selectedChunks)

    if retrievalMode == "raptor":
        vanillaChunks, _ = getChunksAndSelectedChunks(docs_and_scores_vanilla, 0)
        raptorChunks, selectedChunks = getChunksAndSelectedChunks(docs_and_scores_raptor, k)
    else:
        vanillaChunks, selectedChunks = getChunksAndSelectedChunks(docs_and_scores_vanilla, k)
        raptorChunks, _ = getChunksAndSelectedChunks(docs_and_scores_raptor, 0)

    data = {
        'type': 'Retrieval',
        'query': query,
        'vanillaChunks': vanillaChunks,
        'raptorChunks': raptorChunks,
        'selectedChunks': selectedChunks,
        'searchBy': searchBy,
        'chunkSize': int(chunkSize),
        'chunkOverlap': int(chunkOverlap),
        'retrievalMode': retrievalMode,
        'k': int(k),
        'order': increment_counter(),
        'id': str(uuid.uuid4()),
    }
    return data

def runWhatIfLLM(prompt, max_tokens=100, temperature=0.7):
    """
    Run an LLM call (OpenAI) in a non-streaming scenario,
    returning the data structure with prompt & response.
    """
    res = llm(prompt=prompt, max_tokens=max_tokens, temperature=temperature, send=False)
    data = {
        'type': 'LLM',
        'promptText': prompt,
        'responseText': res,
        'temperature': temperature,
        'maxTokens': max_tokens,
        'order': increment_counter(),
        'id': str(uuid.uuid4())
    }
    return data


@app.route('/get_evaluation_embedding_distance', methods=['POST'])
def get_evaluation_embedding_distance():
    """
    Evaluate the semantic similarity between the current answer and stored traces
    """
    try:
        data = request.json
        query = data['query']
        answer = data['answer']
        
        if not query or not answer:
            return jsonify({
                "error": "Both query and answer are required"
            }), 400
            
        score = evaluate_traces_embedding_distance(query, answer)
        return jsonify(score)
    except Exception as e:
        logger.error(f"Error evaluating similarity: {str(e)}")
        return jsonify({
            "error": f"Failed to evaluate similarity: {str(e)}"
        }), 500


@app.route('/get_whatIf_data', methods=['POST'])
def get_whatIf_data():
    """
    Endpoint for a 'what if' retrieval scenario (no streaming).
    """
    print("Headers:", request.headers)
    print("Data:", request.data)
    data = request.json
    logger.info("data", data)
    data = runWhatIfRetrieval(query=data['query'], 
        searchBy=data['searchBy'], 
        chunkSize=data['chunkSize'], 
        chunkOverlap=data['chunkOverlap'], 
        k=data['k'], 
        retrievalMode=data['retrievalMode']
    )
    return jsonify(data), 200

@app.route('/get_whatIf_llm', methods=['POST'])
def get_whatIf_llm():
    """
    Endpoint for a 'what if' LLM call (no streaming).
    """
    form_data = request.json
    result = runWhatIfLLM(
        prompt=form_data['prompt'],
        max_tokens=form_data.get('max_tokens', 100),
        temperature=form_data.get('temperature', 0.7)
    )
    return jsonify(result), 200

@app.route('/finish_running_pipeline', methods=['POST'])
def finish_running_pipeline():
    """
    Un-freeze a pipeline run and resume execution, sending SIGUSR1 to the stored child process.
    """
    data = request.json
    unique_id = data.get('id')
    print("unique_id", unique_id)
    # Store updated data in data_buffer
    data_buffer[unique_id] = data
    from datetime import datetime
    print("Current time:", datetime.now())
    import time
    time.sleep(1)
    
    print("data_buffer", data_buffer[unique_id])
    # Unblock the child that was waiting

    for id in process_mappings:
        if id != unique_id:
            print("Killing process", id)
            pid = process_mappings[id]
            os.kill(pid, signal.SIGTERM)
    if unique_id in process_mappings:
        pid = process_mappings[unique_id]
        os.kill(pid, signal.SIGUSR1)
    else:
        logger.warning(f"No waiting process found for ID={unique_id}")

    return jsonify({"status": "resumed", "id": unique_id}), 200

@app.route('/show_waiting_processes', methods=['GET'])
def show_waiting_processes():
    """
    Debug route to list all waiting processes.
    """
    return jsonify(process_mappings)

def kill_all_waiting_processes():
    """
    Utility function to kill all waiting child processes (SIGTERM).
    """
    for unique_id, pid in process_mappings.items():
        logger.info(f"Killing process {pid} for ID={unique_id}")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    process_mappings.clear()
    return "All waiting processes terminated" 

def kill_port_5001():
    """Kill any processes using port 5001"""
    try:
        # Create a test socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', 5001))
        sock.close()
        
        if result == 0:  # Port is in use
            logger.info("Port 5001 is in use. Attempting to kill processes...")
            
            # Find and kill processes using port 5001
            for proc in psutil.process_iter(['pid', 'name', 'connections']):
                try:
                    connections = proc.connections()
                    for conn in connections:
                        if conn.laddr.port == 5001:
                            logger.info(f"Killing process {proc.pid} ({proc.name()})")
                            proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
            
            logger.info("Killed all processes using port 5001")
        else:
            logger.info("Port 5001 is free")
            
    except Exception as e:
        logger.error(f"Error checking/killing port 5001: {e}")

@app.route('/save_trace', methods=['POST'])
def save_trace():
    """
    Endpoint to save a trace from the frontend
    """
    try:
        data = request.json
        logger.info(f"Received trace data: {data}")
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Handle both formats
        if 'query' in data and 'answer' in data:
            # Frontend format
            input_data = data['query']
            output_data = data['answer']
        elif 'input' in data and 'output' in data:
            # Backend format
            input_data = data['input']
            output_data = data['output']
        else:
            return jsonify({
                "error": "Missing required fields. Need either (query, answer) or (input, output)",
                "received_data": data
            }), 400
            
        trace = create_trace(
            input=input_data,
            output=output_data,
            type=data['type']
        )
        return jsonify({"status": "success", "trace": trace}), 200
    except Exception as e:
        logger.error(f"Error saving trace: {str(e)}")
        logger.error(f"Request data: {request.data}")
        return jsonify({"error": str(e), "request_data": str(request.data)}), 500

def main():
    # Kill any existing processes on port 5001
    kill_port_5001()
    
    # 1) Figure out path to your React front-end
    #    Adjust if your path is different.
    frontend_dir = os.path.join(os.path.dirname(__file__), "front-end")
    print(f"Frontend directory: {frontend_dir}")

    # 2) Start the React dev server in a subprocess
    dev_server_proc = start_dev_server(frontend_dir)

    try:
        # 3) Run the Flask-SocketIO server (blocks until you Ctrl+C or it exits)
        start_flask_server(host="127.0.0.1", port=5001)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    finally:
        # 4) Stop the React dev server
        logger.info("Terminating dev server...")
        dev_server_proc.terminate()  # or dev_server_proc.kill(), if needed
        dev_server_proc.wait()
        logger.info("Dev server process ended. Exiting now.")


if __name__ == "__main__":
    main()