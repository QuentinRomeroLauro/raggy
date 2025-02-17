# Split documents into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


def split_documents(docs, chunk_size=1000, chunk_overlap=0, length_function=len, is_separator_regex=False):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=is_separator_regex
        )

    contents = docs
    if docs and isinstance(docs[0], Document):
        contents = [doc.page_content for doc in docs]

    texts = text_splitter.create_documents(contents)
    n_chunks = len(texts)
    print(f"Split into {n_chunks} chunks")
    return texts