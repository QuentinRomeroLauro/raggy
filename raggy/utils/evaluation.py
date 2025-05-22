"""
Helper functions used to evaluate based on traces
"""

from langchain.evaluation import EmbeddingDistance
from langchain.evaluation import load_evaluator
from interfaces.helpers.trace import get_traces_for_query



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