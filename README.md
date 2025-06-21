# RAGGY
Raggy is an interactive debugging tool for retrieval-augmented generation (RAG) pipelines. Please see our [paper](https://arxiv.org/abs/2504.13587) for more details.


RAG Without the LAG: Interactive Debugging for Retrieval-Augmented Generation Pipelines by Quentin Romero Lauro, Shreya Shankar, Sepanta Zeighami, and Aditya Parameswaran.

Raggy automatically indexes your documents and allows you to interactively debug your RAG pipeline.
![Raggy Logo](/interfaceandcode-edit.jpg)

## Installation

```bash
git clone https://github.com/QuentinRomeroLauro/raggy.git
cd raggy
pip install -e .
```

## Usage

To generate indicies you want to experiment with, run the following code:
```python
# Initialize retriever
retriever = Retriever(docStore="./documents")

# Create indexing pipeline
# This will create 3^3 = 27 different indexes
retriever.createIndexingPipeline(
        chunk_sizes=[400, 1500, 2000],
        chunk_overlaps=[0, 100, 200],
        search_types=["semantic similarity", "max marginal relevance", "tfidf"]
    )
```
Once this one time pre-processing step is complete, you can use the following code to run the pipeline and pipeline with the debug interface.

### Run the debugging server
```bash
# In one terminal
cd raggy/interfaces
python server.py
```

### Run the front-end
```bash
# In another terminal
cd raggy/interfaces/front-end
npm run dev
```

After the debugging server and front-end are running, you can run the pipeline with the debug interface.

### Run the pipeline.
```python
from raggy import llm, Retriever, Query, Answer

# Initialize components
retriever = Retriever(docStore="./documents")
query = Query("What languages can we translate in the hospital?", debug=True)

# Get relevant documents
docs_and_scores = retriever.invoke(query=query, k=5, chunkSize=400, chunkOverlap=0)

# Generate response
RAG_query = """
Given these relevant passages from an external context, answer the following question. 
Question: {query}
Context: {docs}
"""

response = llm(
    prompt=RAG_query.format(query=query, docs=docs_and_scores),
    max_tokens=4000,
    temperature=0.7
)

# Get the answer
answer = Answer(response)
```

The package is organized as follows:

- `raggy/core/`: Core components (llm, retriever, query, answer)
- `raggy/interfaces/`: Debug interface components
- `raggy/utils/`: Utility functions

The debug interface consists of:
- Backend server (`interfaces/server.py`)
- Frontend application (`interfaces/front-end/`)

## Examples

We have an example in the `examples/` directory.
- `hospital-rag/`: A simple example demonstrating how to use the Raggy package for RAG-based question answering using hospital policy documents.

See the [examples/README.md](examples/README.md) for more details.

## Paper
[Our paper](https://arxiv.org/abs/2504.13587) and accompanying user study are now available on arXiv!
```bibtex
@misc{romerolauroShankar2025raglaginteractivedebugging,
      title={RAG Without the Lag: Interactive Debugging for Retrieval-Augmented Generation Pipelines}, 
      author={Quentin Romero Lauro and Shreya Shankar and Sepanta Zeighami and Aditya Parameswaran},
      year={2025},
      eprint={2504.13587},
      archivePrefix={arXiv},
      primaryClass={cs.HC},
      url={https://arxiv.org/abs/2504.13587}, 
}
```