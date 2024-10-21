# raggy 🐶 onboarding
raggy 🐶 is a visual interactive debugging tool for RAG pipelines.

## How raggy works
raggy breaks down your RAG pipeline into four components:
- Query ❓
- LLM 🧠
- Retriever 🔍
- Answer ✅

Each time one of these components is called within your pipeline, the parameters and outputs will be streamed to raggy 🐶 in order. The tool allows you to modify intermediate values and test out different parameters (e.g. chunk size, prompts) for retrieval and llm steps.

## run step 
``run step`` is available on LLM 🧠 and retriever 🔍 components:

### In the retriever 🔍 component, you can modify your parameters and inspect possible outputs.
![retriever open](/images/retriever_open.png)

### In the llm 🧠 component you can tweak prompts and remove inserted context, to see how that affects downstream that affects the response.
![llm open](/images/llm_open.png)
## run all
``run all``, available on llm 🧠, retriever 🔍, and query❓ components.

In raggy 🐶 you can modify the outputs of retriever or llm components by selecting desired chunks or by typing your desired llm output in the response field.

By selecting run all, you will continue running the pipeline from the step you are on and onward, but continue with the modified outputs of the retriever or llm component you selected. Affected downstream components will be highlighted in red.

![downstream](/images/downstream.png)

## save traces
### To create an evaluation for your pipeline, you can start by clicking `save answer as trace` on the answer ✅ component.

### To run the trace, click `run selected trace` at the top right of the screen. Traces will only populate until you have saved them.
![alt text](/images/trace.png)