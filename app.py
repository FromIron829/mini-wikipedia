from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from datasets import load_dataset
from chunker import chunk_text
from langchain_openai import OpenAIEmbeddings
from retriever import Retriever
from pipeline import rag_query

app = FastAPI()

# Load the dataset:
corpus = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
qa = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")

passages = corpus["passages"]
filtered = [
    {"text": p["passage"], "id": p["id"]}
    for p in passages if len(p["passage"]) >= 10
]

# Chunking
documents = []
for item in filtered:
    chunks = chunk_text(item["text"], chunk_size=500, chunk_overlap=50)
    for i, chunk in enumerate(chunks):
        documents.append(
            {
                "text": chunk,
                "passage_id": item["id"],
                "chunk_index": i
            }
        )

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
retriever = Retriever(embeddings, documents)

class Question(BaseModel):
    question: str
    use_hybrid: bool = False

@app.get('/')
def home():
    with open('templates/index.html') as f:
        return HTMLResponse(f.read())

@app.post("/ask")
def ask(q: Question):
    return rag_query(retriever, q.question, use_hybrid=q.use_hybrid)