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

1. First, make sure you have Python 3.10 installed:
```bash
# On macOS with Homebrew:
brew install python@3.10

# Create and activate virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Verify Python version (should show Python 3.10.x)
python --version
```

2. Install the package and required dependencies in the correct order:
```bash
pip install -e .
pip install --upgrade pip
pip install numpy==1.24.3
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn
pip install langchain_openai langchain_community
pip install flask-cors
pip install eventlet
pip install psutil
pip install docling==2.3.0
```

3. Run the example pipeline:
```bash
cd examples/hospital-rag
python pipeline.py
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