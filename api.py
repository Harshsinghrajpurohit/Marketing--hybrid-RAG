# api.py — FastAPI service exposing the hybrid RAG engine (rag_core.py).

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import rag_core

app = FastAPI(title="Marketing Hybrid RAG API")

# Allow the Streamlit UI (localhost:8501) to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    result = rag_core.answer(req.question)
    return QueryResponse(**result)
