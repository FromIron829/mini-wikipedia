from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def retrieve_and_rerank(retriever, query, k=10, top_n=3, use_hybrid=False):
    if use_hybrid:
        candidates = retriever.hybrid_retrieve(query, k=k)
    else:
        candidates = retriever.retrieve(query, k=k)

    pairs = []
    for candidate in candidates:
        pairs.append([query, candidate["text"]])
    
    scores = reranker.predict(pairs)

    for i, candidate in enumerate(candidates):
        candidate["rerank_score"] = float(scores[i])
    
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    return candidates[:top_n]