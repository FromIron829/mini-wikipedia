from reranker import retrieve_and_rerank
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-5.2")

def build_prompt(query, results):
    context = ""
    for i, r in enumerate(results):
        context += f"[Passage {i+1}]\n{r['text']}\n\n"
    
    system_message = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer or the context does not contain relevant "
        "information, just say that you don't know. Keep the answer concise."
    )

    prompt = f"{system_message}\n\nContext:\n{context}Question: {query}"
    return prompt

def rag_query(retriever, query, use_hybrid=False):
    results = retrieve_and_rerank(retriever, query, use_hybrid=use_hybrid)

    prompt = build_prompt(query, results)
    response = model.invoke(prompt)
    
    return {
        "answer": response.content,
        "sources": [r["text"] for r in results]
    }