# rag_core.py — hybrid RAG engine (Apple statements + MarTech 10-K narratives).
# v4: fixes the nested TEXT_FILES loop and the missing `state` in build_index().

import os
from dotenv import load_dotenv
load_dotenv()


DATA_FILES = [
    "data/apple_fy24_q1.pdf",
    "data/apple_fy23_q4.pdf",
    "data/apple_fy23_q3.pdf",
]
TEXT_FILES = [
    "data/martech/salesforce.txt",
    "data/martech/hubspot.txt",
    "data/martech/adobe.txt",
    "data/martech/twilio.txt",
]
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "apple-hybrid-rag")
NAMESPACE = "app"                     # all rag_core vectors live here
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

POOL = 20          # candidate pool width before reranking — single source of truth

_state = None


def get_state():
    global _state
    if _state is None:
        _state = _build_state()
    return _state


def _build_state():
    # Heavy imports stay HERE (lazy) so `import rag_core` / API startup is fast (~2s).
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from rank_bm25 import BM25Okapi
    from langchain_ollama import OllamaEmbeddings, ChatOllama
    from pinecone import Pinecone
    from flashrank import Ranker
    from langchain_core.documents import Document

    # 1) PDF pages (with the $ glyph fix) — FIRST, so Apple chunk ids stay 0..60
    docs = []
    for path in DATA_FILES:
        for page in PyPDFLoader(path).load():
            page.page_content = page.page_content.replace("/dollarsign", "$")
            docs.append(page)

    # 2) MarTech narratives — already-clean text, one "page" per file.
    #    Same indent level as the PDF loop above (NOT nested inside it).
    for path in TEXT_FILES:
        with open(path, encoding="utf-8", errors="replace") as f:
            docs.append(Document(page_content=f.read(),
                                 metadata={"source": path, "page": 0, "kind": "text"}))

    # 3) Chunk everything uniformly
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # 4) Header propagation — ONLY for PDF pages (text files have no page headers)
    first = {}
    for cid, c in enumerate(chunks):
        if c.metadata.get("kind") == "text":
            continue
        key = (c.metadata["source"], c.metadata["page"])
        if key in first and cid != first[key]:
            c.page_content = chunks[first[key]].page_content + "\n" + c.page_content
        else:
            first[key] = cid

    # 5) Dense + sparse arms + reranker + LLM
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
    index = Pinecone(api_key=os.getenv("PINECONE_API_KEY")).Index(PINECONE_INDEX)
    # Naive whitespace tokenizer kept deliberately: a regex tokenizer promoted
    # corpus-wide tokens ("2023", "30", "apple") into query terms and measurably
    # hurt Precision@3 (0.28 -> 0.17). See Memory.md for the A/B.
    bm25 = BM25Okapi([c.page_content.lower().split() for c in chunks])

    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    llm = ChatOllama(model=LLM_MODEL, temperature=0.2, base_url=OLLAMA_URL)

    return {"chunks": chunks, "embeddings": embeddings, "index": index,
            "bm25": bm25, "ranker": ranker, "llm": llm}


def build_index(batch=16):
    """Embed every chunk and upsert into NAMESPACE. Run once after corpus changes."""
    state = get_state()
    print(f"Embedding {len(state['chunks'])} chunks...")

    # Serverless namespaces are lazy: they only exist after a first upsert.
    stats = state["index"].describe_index_stats()
    if NAMESPACE in stats.namespaces:
        state["index"].delete(delete_all=True, namespace=NAMESPACE)

    vecs, done = [], 0
    for cid, c in enumerate(state["chunks"]):
        vec = state["embeddings"].embed_documents([c.page_content])[0]
        vecs.append((f"chunk-{cid}", vec, c.metadata))
        if len(vecs) >= batch:
            state["index"].upsert(vectors=vecs, namespace=NAMESPACE)
            done += len(vecs)
            print(f"  upserted {done}/{len(state['chunks'])}")
            vecs = []
    if vecs:
        state["index"].upsert(vectors=vecs, namespace=NAMESPACE)
    print(f"Indexed {len(state['chunks'])} chunks into namespace '{NAMESPACE}'")


def _dense(state, query, k=3):
    ms = state["index"].query(vector=state["embeddings"].embed_query(query),
                              top_k=k, namespace=NAMESPACE, include_metadata=True).matches
    return [(int(m.id.split("-")[1]), m.score) for m in ms]


def _bm25(state, query, k=3):
    scores = state["bm25"].get_scores(query.lower().split())
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(i, scores[i]) for i in ranked]


def _rrf(rankings, k=60):
    fused = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            fused[cid] = fused.get(cid, 0) + 1 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def _merge(state, query, k=3):
    return _rrf([[i for i, _ in _dense(state, query, k)],
                 [i for i, _ in _bm25(state, query, k)]])[:k]


def _rerank(state, query, candidates, top_n=3):
    from flashrank import RerankRequest          # local import: needed in THIS scope
    passages = [{"text": state["chunks"][i].page_content, "meta": i} for i in candidates]
    results = state["ranker"].rerank(RerankRequest(query=query, passages=passages))
    return [(int(r["meta"]), float(r["score"]), j + 1) for j, r in enumerate(results[:top_n])]


def answer(question, top_n=3):
    state = get_state()
    ranked = _rerank(state, question, [i for i, _ in _merge(state, question, POOL)], top_n)


    blocks, sources = [], []
    for cid, _, _ in ranked:
        blocks.append(f"[{len(sources) + 1}] {state['chunks'][cid].page_content}")
        sources.append(f"{state['chunks'][cid].metadata['source']} — page {state['chunks'][cid].metadata['page']}")

    prompt = f"""You are a marketing & business analyst assistant. Answer ONLY from the context below.
Do not reproduce the provided context text.

CONTEXT:
{chr(10).join(blocks)}

QUESTION: {question}

Rules:
- If the context does not contain the answer, say "I could not find sufficient information".
- Cite each fact as [N] matching its context block.

ANSWER:"""

    response = state["llm"].invoke(prompt)
    return {"question": question, "answer": response.content, "sources": sources}


if __name__ == "__main__":
    build_index()                       # FIRST run: ~15-25 min, then reuse
    for q in ["What is Salesforce's go-to-market strategy for its AI platform?",
              "What is HubSpot's core product and who are its customers?"]:
        result = answer(q)
        print("Q:", q)
        print("ANSWER:", result["answer"])
        print("SOURCES:", result["sources"])
        print()
