import faiss
from bm25 import BM25
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

class Retriever:
    def __init__(self, embeddings, documents):
        self.embeddings = embeddings
        self.texts = [doc["text"] for doc in documents]
        self.text_to_idx = {t: i for i, t in enumerate(self.texts)}
        self.metadatas = [{
            "passage_id": doc["passage_id"],
            "chunk_index": doc["chunk_index"]} for doc in documents
            ]
        
        embedding_dim = len(embeddings.embed_query("hello world"))
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.vector_store = FAISS(
            embedding_function=embeddings,
            index=self.index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        self.vector_store.add_texts(self.texts, metadatas=self.metadatas)

        self.bm25 = BM25(self.texts)

    def retrieve(self, query, k=5):
        query_vector = np.array([self.embeddings.embed_query(query)]).astype("float32")
        # Embed the query

        distances, indices = self.vector_store.index.search(query_vector, k)
        # Search FAISS directly

        results = []
        for i, idx in enumerate(indices[0]):
            doc_id = self.vector_store.index_to_docstore_id[idx]
            doc = self.vector_store.docstore.search(doc_id)
            results.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": distances[0][i],
                "idx": idx,
            })
        return results
    
    def hybrid_retrieve(self, query, k=20, alpha=0.9):
        # Step 1: Vector search
        vector_candidates = self.retrieve(query, k=k)
        
        # Step 2: BM25 search — score all docs, take top k
        bm25_scores = [(i, self.bm25.score(query, i)) for i in range(len(self.texts))]
        bm25_scores.sort(key=lambda x: x[1], reverse=True)
        bm25_top_k = bm25_scores[:k]
        
        # Step 3: Build RRF scores using text_to_idx for alignment
        rrf_scores = {}  # idx -> score
        
        # Vector RRF
        for rank, candidate in enumerate(vector_candidates, 1):
            idx = self.text_to_idx.get(candidate["text"])
            if idx is not None:
                rrf_scores[idx] = rrf_scores.get(idx, 0) + alpha * (1 / (60 + rank))
        
        # BM25 RRF
        for rank, (idx, score) in enumerate(bm25_top_k, 1):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + (1 - alpha) * (1 / (60 + rank))
        
        # Step 4: Sort by combined RRF score
        combined = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Step 5: Return top candidates
        results = []
        for idx, score in combined[:k]:
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadatas[idx],
                "rrf_score": score
            })
        
        return results
