# Raggy Examples

This directory contains example implementations of RAG pipelines using the Raggy package.

## Hospital RAG Example

A simple example demonstrating how to use the Raggy package for RAG-based question answering.

### Directory Structure

```
hospital-rag/
├── documents/         # Sample documents
└── scripts/          # Python scripts
    └── pipeline.py   # Main pipeline implementation
```

### Running the Example

1. Install the package in development mode:
```bash
pip install -e .
```

2. Run the example pipeline:
```bash
cd examples/hospital-rag
python scripts/pipeline.py
```

The pipeline will:
- Load and process documents from the `documents` directory
- Start the debug interface
- Process the example query
- Generate and display the response

### Code Overview

The example demonstrates:
- How to initialize a retriever with documents
- How to create and process queries
- How to use the debug interface
- How to generate responses using the LLM
- How to handle and display answers