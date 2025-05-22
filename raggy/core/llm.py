# The LLM interface is used to call the OpenAI API and stream relevant information to the interface.
import openai
from dotenv import load_dotenv
import os
import requests
from ..interfaces.helpers.counter import increment_counter
import uuid
import time
import logging
import signal

# Load environment variables from .env file
load_dotenv()

# Access API key
api_key = os.getenv('OPENAI_API_KEY')


def sendLLMData(promptText, responseText, temperature, maxTokens, id):
    
    data = {
        'type': 'LLM',
        'promptText': promptText,
        'responseText': responseText,
        'temperature': temperature,
        'maxTokens': maxTokens,
        'order': increment_counter(),
        'id': id,
    }
    # POST the data to the server
    requests.post('http://localhost:5001/send_llm_data', json=data)

def llm(prompt="Tell the user they have to enter a prompt", max_tokens=100, temperature=0.7, send=True, model="gpt-4o"):

    def llm_internal(prompt="Internal call", max_tokens=max_tokens, temperature=0.7, send=False, duplicate=False, id=None, dupCompletion=""):

        # Tell the server if we are loading a new ui element or not
        if send and not duplicate:
            requests.post('http://localhost:5001/set_loading', json={'loading': True})

            
        completion = dupCompletion
        if not duplicate:
            client = openai.Client(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )

            completion = response.choices[0].message.content

        if not id:
            id = str(uuid.uuid4())

        if send:
            # Send relevant information to the interface
            sendLLMData(prompt, response.choices[0].message.content, temperature, max_tokens, id)

        pid = os.fork()
        if pid == 0 and send or duplicate:
            # get the actual pid of the process
            pid = os.getpid()
            
            # regsiter the process with the server
            response = requests.post(f"http://localhost:5001/register_process/{id}/{pid}")

            # Pause the process, until the server signals it to wake up
            # ...
            # Define the set of signals to wait for
            sigset = {signal.SIGUSR1}

            # Block the signals so they can be waited for
            signal.pthread_sigmask(signal.SIG_BLOCK, sigset)

            # Wait for the signal to be received
            while signal.sigwait(sigset) != signal.SIGUSR1:
                pass # wait
            
            # process has woken up and is ready to continue, get the new data from the server
            completion = requests.get(f"http://localhost:5001/get_updated_interface_data/{id}").json().get("responseText")

            """
            we want to make sure we have another fork to handle n number of finish running requests as needed
            this will fork another process to do that, and not send anything to the interface, and associate 
            it with the same id having this after the signal wait ensures that the process is not forked until
            the server is ready so we don't have infinite recursion and infinite process running
            """
            # tell the server to delete the old id:pid mapping
            requests.post(f"http://localhost:5001/delete_process/{id}")

            nextPipelinePID = os.fork()
            if nextPipelinePID == 0:
                # get the actual pid of the process
                pid = os.getpid()
                # Define the set of signals to wait for
                sigset = {signal.SIGUSR1}
                # Block the signals so they can be waited for
                signal.pthread_sigmask(signal.SIG_BLOCK, sigset)

                # regsiter the replacement process id with the server
                response = requests.post(f"http://localhost:5001/register_process/{id}/{pid}")
                print(f"Next pipeline process pid: {pid} has been registered under {id}")
                llm_internal(prompt=prompt, max_tokens=max_tokens, temperature=temperature, send=False, duplicate=True, id=id, dupCompletion=completion)

            
        if send and not duplicate:
            requests.post('http://localhost:5001/set_loading', json={'loading': False})

        return completion

    return llm_internal(prompt=prompt, max_tokens=max_tokens, temperature=temperature, send=send, duplicate=False, id=None, dupCompletion="")