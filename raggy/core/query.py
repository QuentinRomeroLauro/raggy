import requests
import os
import uuid
import signal
import json
from ..interfaces.helpers.counter import increment_counter

"""
Our query type is going to need to support the following:
- run pipeline with query

This will require the same implementation and functionality as "finish running pipeline"
in the streamToInterface.py file
but...
that means we have to code carefully to ensure that we don't have infinite recursion
"""

class Query:
    def __init__(self, query):
        self.query = ""
        qid = str(uuid.uuid4())
        self.executeQuery(send=True, duplicate=False, query=query, qid=qid)

    def sendQueryData(self, qid):
        """
        Send the query data to the interface
        """
        data = {
            'type': 'Query',
            'query': self.query,
            'order': increment_counter(),
            'id': qid,
        }

        response = requests.post('http://localhost:5001/send_query_data', json=data)

    def __str__(self):
        return self.query

    def executeQuery(self, send=True, duplicate=False, query="", qid=None):
        """
        Render the query, send it to the interface, and continue running the pipeline

        Provide an interface so that the user can change the query from the pipeline
        """
        self.query = query            
        # tempQuery = query
        # set loading
        if send and not duplicate:
            requests.post('http://localhost:5001/set_loading', json={'loading': True})

        if send:
            self.sendQueryData(qid=qid)

        pid = os.fork()
        if pid == 0 and send or duplicate:
            # get the actual pid of the process
            pid = os.getpid()
            
            # regsiter the process with the server
            requests.post(f"http://localhost:5001/register_process/{qid}/{pid}")

            # Pause the process, until the server signals it to wake up
            # ...
            # Define the set of signals to wait for
            sigset = {signal.SIGUSR1}

            # Block the signals so they can be waited for
            signal.pthread_sigmask(signal.SIG_BLOCK, sigset)

            # Wait for the signal to be received
            while signal.sigwait(sigset) != signal.SIGUSR1:
                pass

            data = requests.get(f"http://localhost:5001/get_updated_interface_data/{qid}").json()
            self.query = data.get('query')

            # tell the server to delete the old id:pid mapping
            requests.post(f"http://localhost:5001/delete_process/{qid}")

            nextPipelinePID = os.fork()
            if nextPipelinePID == 0:
                # get the actual pid of the process
                pid = os.getpid()

                # Define the set of signals to wait for
                sigset = {signal.SIGUSR1}
                # block the signals so they can be waited for
                signal.pthread_sigmask(signal.SIG_BLOCK, sigset)

                data = requests.get(f"http://localhost:5001/get_updated_interface_data/{qid}").json()
                self.query = data.get('query')

                requests.post(f"http://localhost:5001/register_process/{qid}/{pid}")
                self.executeQuery(send=False, duplicate=True, query=query, qid=qid)

        if send and not duplicate:
            requests.post('http://localhost:5001/set_loading', json={'loading': False})

        return 