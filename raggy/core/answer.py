"""
Answer module for handling answer presentation and debugging
"""
import requests
import uuid
from ..interfaces.helpers.counter import read_counter, increment_counter, write_counter

class Answer:
    def __init__(self, answer):
        self.answer = answer
        self.count = increment_counter()
        self.id = str(uuid.uuid4())
        self.sendAnswer()

    def __str__(self):
        return self.answer

    def sendAnswer(self):
        # set loading to true
        requests.post('http://localhost:5001/set_loading', json={'loading': True})

        data = {
            'type': 'Answer',
            'answer': str(self),
            'order': self.count,
            'id': self.id
        }
        # POST the data to the server
        response = requests.post('http://localhost:5001/send_answer', json=data)
    
        # set loading to false
        requests.post('http://localhost:5001/set_loading', json={'loading': False}) 