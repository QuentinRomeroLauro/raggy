# Task Coding Cheat Sheet
# Quick Links
- [run the pipeline](#run-the-pipeline)
- [retriever 🔍](#retriever-🔍)
- [llm 🧠](#llm-🧠)
- [query ❓](#query-❓)
- [answer ✅](#answer-✅)
- [Common Prompt Templates and Patterns](#common-prompt-templates-and-patterns)
  - [Formatting](#formatting)
  - [Parse-Json-Output](#Parse-JSON-Output)
  - [Task Decomposition](#task-decomposition)
  - [Re-rank](#re-rank)
  - [Route](#route)
  - [Re-retrieve](#re-retrieve)

# Run the pipeline
```
python task/pipeline.py
```

# retriever 🔍
- `query`: string
- `k`: integer
- `chunkSize`: 100 | 200 | 400 | 800 | 1000 | 1500 | 2000
- `chunkOverlap`: 0 | 10 | 25 | 50 | 100 |200 | 400
- `retrievalMode`: "vanilla" | "raptor"
    - whether to retrieve summary chunks or regular chunks
- `@returns`: [(chunk text, score)] e.g. ("chunk_text", .63)
```
retriever.invoke(
            query="Your query here",
            k=10,
            chunkSize=1000,
            chunkOverlap=10,
            retrievalMode="vanilla",
        )
```
![retriever image](/images/retriever_open.png)

# llm 🧠
- `prompt`: string
- `max_tokens:` integer
- `temperature`: integer [0 < x < 1 ]
```
llm(
    prompt="this is the prompt",
    max_tokens=200,
    temperature=0.20
)
```
![llm open](/images/llm_open.png)

# query ❓
```
Query("This is the query string")
```
![query image](/images/query_open.png)

# answer ✅
```
Answer("This is the answer string")
```
![answer image](/images/answer_open.png)


# Prompt Templates
These prompts are for building blocks for common RAG pipeline formats. Adjust them as you need to.

## Formating
```python
FORMAT="""
You are an expert at answering questions given relevant context. Given a question and context, answer the question according to the context.
Question: {question}
Context: {context}
"""
question = "Example question"
context = "Example context"
prompt = FORMAT.format(question=question, context=context)
```

## Parse JSON Output
```python
FORMAT="""
Answer this question: {question}

Follow this JSON output:
{{"answer": []}}
"""
answer = llm(
    prompt=FORMAT.format(question=question), 
    max_tokens=200, 
    temperature=0.2
)
json_decomp = json.loads(decomp)
answer = json_decomp["answer"]
```

## Task Decomposition
```python
QUERY_DECOMP="""
You are an expert at converting user questions into sub questions. \
You have access to a database of documents available on WVU Medicine's Hospital System's website. \

Perform query decomposition. Given a user question, break it down into distinct sub questions that \
you need to answer in order to answer the original question. Give at most three questions \

If there are acronyms or words you are not familiar with, do not try to rephrase them.

Follow this JSON output:
{{"sub_questions": []}}

Question: {inputQuery}
"""
decomp = llm(
    prompt=QUERY_DECOMP.format(inputQuery=inputQuery), 
    max_tokens=200, 
    temperature=0.2
)
json_decomp = json.loads(decomp)
questions = json_decomp["sub_questions"] # this will be a python array of questions []
```

## Re-rank
```python
RERANK="""
You are an expert at ranking relevant context. Given a query and list of relevant content, return the context in order of relevance to the query.
Question: {query}
Context: {context}
"""
docs_and_scores = [("context_1", 0.12), ("context_2", .34), ("context_3", 0.56)]
context = ""
for doc, score in docs_and_scores:
    context += doc
rerank = llm(prompt=RERANK.format(question=question, context=context), max_tokens=100, temperature=0.1)
```

## Route
You may use an LLM call to decide which path the pipeline should take.
```python
ROUTE="""
You are an expert query router in a RAG pipeline. Given a user question decide if it is better answered by summaries or raw document text information about a document. If the question is better answered by a summary output 'summary'. If the question is better answered by raw document text, output: 'raw'

Question: {question}
"""
res = llm(prompt=ROUTE.format(question=question), max_tokens=30, temperature=0.1)
if res == 'raw':
    # retrieve vanilla chunks
else if res == 'summary'
    # retrieve using summary chunks
```

## Re-retrieve
You may use an LLM to decide whether to re-retrieve context.
```python
RERETRIEVE="""
You are an expert at deciding if context is sufficient enough to answer a question thoroughly. Given a question and context output 'good' if the context is sufficient and output 'bad' if the context is not sufficient.

Question: {question}
Context: {context}
"""
start = choice = llm(prompt=RERETRIEVE.format(question=question, context=context), max_tokens=30, temperature=0.1)
while choice != 'good':
    # change the question
    # invoke the retriever
    # update context
    choice = llm(prompt=RERETRIEVE.format(question=question, context=context), max_tokens=30, temperature=0.1)
# ... continue the pipeline
```
