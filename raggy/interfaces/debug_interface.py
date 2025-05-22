"""
Debug interface for monitoring and debugging the RAG pipeline
"""
import logging
import multiprocessing
import os
import subprocess


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_debug_interface():
    """
    Start BOTH the Flask debug interface server (port 5001)
    AND the React dev server (port 3000) automatically.
    """
    try:
        # 1) Import the start_server function
        from .server import start_server

        # 2) Start the Flask backend in a new process (port 5001)
        server_process = multiprocessing.Process(target=start_server)
        server_process.daemon = True
        server_process.start()
        logger.info(f"Debug interface (Flask-SocketIO) started (PID: {server_process.pid}) on port 5001.")

        # 3) Also spawn create-react-app dev server on port 3000
        front_end_dir = os.path.join(os.path.dirname(__file__), 'front-end')
        logger.info("Starting React dev server (npm start) on port 3000...")
        dev_server_process = subprocess.Popen(
            ["npm", "start"],
            cwd=front_end_dir,
            shell=True
        )
        logger.info(f"React dev server process started (PID: {dev_server_process.pid}) on port 3000.")

        # You may optionally store dev_server_process somewhere if you need to kill it later
        
        return True

    except ImportError:
        logger.error("Failed to import 'start_server' from .server. Make sure server.py exists and is correctly structured.")
        return False
    except Exception as e:
        logger.error(f"Failed to start debug interface process or dev server: {e}")
        return False

if __name__ == "__main__":
    if not start_debug_interface():
        logger.error("Debug interface failed to start.") 