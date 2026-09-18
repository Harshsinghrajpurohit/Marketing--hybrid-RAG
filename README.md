# marketing-hybrid-rag

RAG over Apple's quarterly financial statements and the business sections of four marketing 10-Ks (Salesforce, HubSpot, Adobe, Twilio). 393 chunks across six documents.

Dense search and BM25 run over the same chunks, RRF merges the two rankings, a cross-encoder reranks the candidates, and a local Ollama model writes the answer with its sources. Everything runs on my machine apart from the Pinecone index.

## Stack

- Pinecone serverless + `nomic-embed-text` for dense retrieval
- `rank-bm25` for sparse retrieval
- RRF (k=60) for fusion, pool of 20 candidates before reranking
- FlashRank `ms-marco-MiniLM-L-12-v2` as the reranker
- Ollama `llama3.2:3b` for generation
- FastAPI + Streamlit for the service and the UI

## Files

```
rag_core.py         the pipeline: load -> chunk -> retrieve -> fuse -> rerank -> generate
api.py              FastAPI service (POST /query, GET /health)
ui.py               Streamlit front end, calls the API over HTTP
evaluate.py         precision@3 and a faithfulness check
prepare_martech.py  one-off script that built the martech corpus out of the 10-Ks
hybrid_rag.ipynb    the notebook I built this in, section by section
data/               the documents
```

## Running it

```bash
pip install -r requirements.txt
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Add your Pinecone key to `.env` (see `.env.example`), then:

```bash
python rag_core.py            # indexes 393 chunks, ~15 min on CPU, only needed once
python evaluate.py            # prints the scores below
uvicorn api:app --port 8000   # then POST /query, or open /docs
streamlit run ui.py           # UI on :8501, needs the API running
```

## Results

Precision@3 over 6 hand-labelled questions:

```
Dense   0.28
BM25    0.28
RRF     0.28
Rerank  0.28
```

Faithfulness: 3/3 answers quoted only figures that appear in the retrieved chunks.

The same test set scored 0.11 / 0.00 / 0.11 / 0.00 before I prepended each page's header to every chunk on that page. The Apple PDFs are nothing but tables, so the date a row belonged to ended up in a different chunk from the numbers, and retrieval kept handing the model title pages instead.

## Sources

Apple's quarterly statements come from Apple's investor relations site. The 10-Ks are public SEC filings: Salesforce (CIK 0001108524), HubSpot (0001404655), Adobe (0000796343), Twilio (0001447669).
