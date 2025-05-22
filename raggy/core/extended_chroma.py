"""
Extended Chroma class with additional functionality for retrieval
"""
import numpy as np
from typing import List, Optional, Tuple
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import TFIDFRetriever

def cosine_similarity(X, Y):
    """Row-wise cosine similarity between two equal-width matrices."""
    if len(X) == 0 or len(Y) == 0:
        return np.array([])

    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            "Number of columns in X and Y must be the same. X has shape"
            f"{X.shape} "
            f"and Y has shape {Y.shape}."
        )

    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
    similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
    return similarity

def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: list,
    lambda_mult: float = 0.5,
    k: int = 4,
):
    """Calculate maximal marginal relevance."""
    if min(k, len(embedding_list)) <= 0:
        return []
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = cosine_similarity(embedding_list, selected)
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue
            redundant_score = max(similarity_to_selected[i])
            equation_score = (
                lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            )
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        idxs.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
    return idxs

class ExtendedTFIDFRetriever(TFIDFRetriever):
    """Extended TFIDF retriever with additional functionality."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def invoke(self, query: str, k: int):
        self.k = k
        return self._get_relevant_documents(query)

    def _get_relevant_documents(self, query, *, run_manager=None):
        from sklearn.metrics.pairwise import cosine_similarity
        query_vec = self.vectorizer.transform([query])
        results = cosine_similarity(self.tfidf_array, query_vec).reshape((-1,))
        top_indices = results.argsort()[-self.k:][::-1]
        return_docs = [(self.docs[i], results[i]) for i in top_indices]
        return return_docs

class ExtendedChroma(Chroma):
    """Extended Chroma class with additional retrieval methods."""
    
    def __init__(self, *args, relevance_score_fn=None, **kwargs):
        self.relevance_score_fn = relevance_score_fn or (lambda x: 1.0 - x)
        super().__init__(*args, **kwargs)
    
    def max_marginal_relevance_search_with_score(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter = None,
        where_document = None,
        **kwargs,
    ) -> List[Tuple[Document, float]]:
        """Search with maximal marginal relevance."""
        if self._embedding_function is None:
            raise ValueError("For MMR search, you must specify an embedding function on creation.")

        embedding = self._embedding_function.embed_query(query)
        
        results = self._collection.query(
            query_embeddings=embedding,
            n_results=fetch_k,
            where=filter,
            where_document=where_document,
            include=["metadatas", "documents", "distances", "embeddings"],
            **kwargs,
        )
        mmr_selected = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            results["embeddings"][0],
            k=k,
            lambda_mult=lambda_mult,
        )

        candidates = self._results_to_docs_and_scores(results)
        selected_results = [r for i, r in enumerate(candidates) if i in mmr_selected]
        return selected_results
    
    def _results_to_docs_and_scores(self, results):
        """Convert Chroma results to documents and scores with proper score transformation."""
        return [
            (Document(page_content=result[0], metadata=result[1] or {}), 
             self.relevance_score_fn(result[2]))
            for result in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]
    
    def similarity_search_with_score(self, *args, **kwargs):
        """Override to use our custom _results_to_docs_and_scores method."""
        results = super().similarity_search_with_score(*args, **kwargs)
        return [(doc, self.relevance_score_fn(score)) for doc, score in results]
    
    def tfidf(self, query: str, k: int = 4, **kwargs):
        """Search using TFIDF."""
        db_get = self.get()
        db_ids = db_get["ids"]
        if not db_ids:
            raise ValueError("No documents in the collection")

        doc_list = []
        for x in range(len(db_ids)):
            doc = db_get["documents"][x]
            doc_list.append(doc)
        
        retriever = ExtendedTFIDFRetriever.from_texts(doc_list)
        docs_and_scores = retriever.invoke(query, k)
        return docs_and_scores 