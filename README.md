# Marketing & Enterprise Hybrid RAG

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Pinecone](https://img.shields.io/badge/vector_db-Pinecone_Serverless-purple.svg)](https://www.pinecone.io/)
[![Ollama](https://img.shields.io/badge/local_llm-Ollama_(Llama_3.2)-black.svg)](https://ollama.com/)

I built this to answer questions across a mixed document set: Apple's quarterly financial statements (tables, exact figures) and the business sections of four marketing-software 10-Ks — Salesforce, HubSpot, Adobe and Twilio. 393 chunks, six documents, one local LLM.

Retrieval is hybrid. Vector search and BM25 run side by side over the same chunks, RRF fuses their rankings, and a cross-encoder reranks the candidates before anything reaches the model. Answers come back with their sources, and the pipeline is exposed as a FastAPI service with a small Streamlit front end on top.

This started as a learning project, so the notebook walks through every stage step by step — including the experiments that failed. Those turned out to be more useful than the ones that worked, and they're written up further down.

---

## 🏗️ How it works

```mermaid
flowchart TD
    D[Apple financial PDFs + 4 MarTech 10-K narratives] --> L[Load & clean<br/>PyPDFLoader / text loader]
    L --> C[Recursive Character Text Splitter<br/>Chunk Size: 400 | Overlap: 100<br/>+ page-header propagation]
    
    C --> V[(Pinecone Cloud<br/>Dense Embeddings)]
    C --> B[(BM25 Sparse Index<br/>Exact Lexical Frequencies)]
    
    Q[🔍 User Query] --> V
    Q --> B
    
    V -- Top 20 Semantic Chunks --> RRF[Reciprocal Rank Fusion<br/>Score = 1 / (60 + rank)]
    B -- Top 20 Keyword Chunks --> RRF
    
    RRF --> CE[FlashRank Cross-Encoder<br/>Joint Query-Doc Scoring]
    CE --> T3[Top-3 High-Signal Chunks]
    
    T3 --> P[Grounded Prompt Template<br/>Zero-Hallucination Guardrails]
    P --> O[Local Ollama LLM<br/>llama3.2:3b]
    O --> A[Grounded Answer + Page Citations]
    
    A --> E1[Precision@3 Evaluation]
    A --> E2[Faithfulness Check]
    A --> SVC[FastAPI POST /query] --> UI[Streamlit UI]
```

---

## Why hybrid retrieval

Pure vector search is good at meaning and bad at specifics. An embedding squashes a whole chunk into one vector, so `$119,575 million` and `$117,154 million` end up looking almost identical, and a question about a specific quarter tends to pull back whatever *sounds* like that quarter rather than the one actually printed in the table.

BM25 has the opposite profile. It counts exact term matches, so figures, product names and line-item labels survive intact — but it has no idea that "services revenue" and "Services net sales" are the same thing.

Running both and fusing the results covers each one's blind spot:

* **Dense (Pinecone + `nomic-embed-text`):** semantic similarity, paraphrases, loosely-worded questions.
* **BM25 (`rank-bm25`):** exact figures, line-item names, fiscal periods.
* **RRF:** merges the two by rank position (k=60), which sidesteps having to normalise a cosine score bounded at 1.0 against an unbounded BM25 score.
* **Cross-encoder (FlashRank):** scores each `(query, passage)` pair jointly and cuts the candidates down to the three passages the LLM actually sees.

---

## Repo layout

```text
marketing-hybrid-rag/
│
├── data/
│   ├── apple_fy24_q1.pdf, apple_fy23_q4.pdf, apple_fy23_q3.pdf   # Apple financial statements
│   └── martech/                    # clean narrative cut out of SEC 10-K filings
│       └── salesforce.txt, hubspot.txt, adobe.txt, twilio.txt
│
├── hybrid_rag.ipynb                # the notebook: every stage, one cell at a time
├── rag_core.py                     # engine: load → chunk → retrieve → fuse → rerank → generate
├── api.py                          # FastAPI service (POST /query, GET /health)
├── ui.py                           # Streamlit UI, talks to the API over HTTP
├── evaluate.py                     # Precision@3 + faithfulness, run against rag_core
├── prepare_martech.py              # one-off: trims raw 10-K text down to the useful narrative
├── requirements.txt
├── .env.example
└── README.md
```

---

## What it's built with

| Piece | Choice |
| :--- | :--- |
| Dense retrieval | Pinecone Serverless + `nomic-embed-text` via Ollama |
| Sparse retrieval | `rank-bm25`, built over the same chunks |
| Fusion | Reciprocal Rank Fusion, k=60, candidate pool of 20 |
| Reranking | FlashRank cross-encoder, `ms-marco-MiniLM-L-12-v2` |
| Generation | Ollama `llama3.2:3b`, temperature 0.2 |
| Documents | `pypdf` / `langchain-community` + `RecursiveCharacterTextSplitter` |
| Serving | FastAPI + Uvicorn, Streamlit front end |
| Evaluation | `evaluate.py` — Precision@3 and a numeric-support faithfulness check |

---

## Running it

**1. Install**
```bash
git clone https://github.com/Harshsinghrajpurohit/marketing-hybrid-rag.git
cd marketing-hybrid-rag

python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

**2. Pull the models** (Ollama needs to be running)
```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

**3. Create a `.env` file**
```env
PINECONE_API_KEY=your_key_here
PINECONE_INDEX_NAME=apple-hybrid-rag
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
EMBEDDING_MODEL=nomic-embed-text:latest
```

**4. Build the index** — first run only. It embeds 393 chunks into the `app` namespace, which takes roughly 15–20 minutes on CPU.
```bash
python rag_core.py
```

**5. Use it**
```bash
python evaluate.py                   # Precision@3 + faithfulness scoreboard
uvicorn api:app --port 8000          # then POST to /query, or open /docs
streamlit run ui.py                  # chat UI on localhost:8501 (needs the API running)
```

If you'd rather read through the pipeline than run it, open the notebook:
```bash
jupyter notebook hybrid_rag.ipynb
```

### Rebuilding the corpus

Everything the pipeline reads is committed under `data/`, so this section isn't needed to run the project — it's only for regenerating the MarTech half from scratch.

The four raw 10-K filings aren't committed (60 MB of SEC HTML/XML the pipeline never reads). They follow a predictable EDGAR URL:

```text
https://www.sec.gov/Archives/edgar/data/<CIK>/<accession-no-dashes>/<accession>.txt
```

Download each one, save it as `data/<vendor>_10k.txt`, then:

```bash
python prepare_martech.py    # trims each filing to Item 1 (Business) + Item 7 (MD&A) -> data/martech/*.txt
python rag_core.py           # re-index (~15-20 min)
```

| Company | CIK | 10-K accession |
| :--- | :--- | :--- |
| Salesforce | 0001108524 | 0001108524-26-000060 |
| HubSpot | 0001404655 | 0001193125-26-046646 |
| Adobe | 0000796343 | 0000796343-26-000003 |
| Twilio | 0001447669 | 0001447669-26-000021 |

Apple's quarterly financial statements come from Apple's investor relations site. All sources are public records — no licensing restrictions on redistribution.

---

## Results

All numbers below come from `python evaluate.py` on this machine against a 6-question hand-labelled test set (ground truth lives in that file). Precision@3 is `|retrieved ∩ relevant| / 3`; the faithfulness check verifies that every figure in a generated answer appears in the context it was given.

**Current pipeline — 6-brand corpus, 393 chunks:**

```
Dense    | 0.67  0.33  0.33  0.00  0.33  0.33  | AVG 0.28
BM25     | 0.67  0.33  0.33  0.00  0.00  0.33  | AVG 0.28
RRF      | 0.67  0.33  0.33  0.00  0.00  0.33  | AVG 0.28
Rerank   | 0.33  0.33  0.33  0.00  0.33  0.33  | AVG 0.28

Faithfulness: 3/3 verified
```

**The fix that actually moved the needle — page-header propagation** (measured earlier, Apple-only, 61 chunks):

| Method | Before | After |
| :--- | :--- | :--- |
| Dense | 0.11 | **0.28** |
| BM25 | 0.00 | **0.22** |
| RRF | 0.11 | **0.28** |
| Rerank | 0.00 | **0.28** |

The Apple PDFs are nine pages of dense financial tables. Every chunk on a page had been split away from that page's header, so the only place the phrase *"December 30, 2023"* existed was in the first chunk of the page — which meant retrieval kept handing the model title pages instead of numbers. Prepending each page's header to every chunk on that page lifted all four methods, with the reranker going from worst to best. Nothing was re-embedded to test it: the vectors live in a separate Pinecone namespace, so the baseline stayed intact for comparison.

### Reading these numbers honestly

0.28 Precision@3 looks weak, and on this corpus, with this test set, it is. Two things are worth pointing out:

* **Hybrid did not beat dense here.** Fusing two retrievers and reranking costs latency, and on a 393-chunk corpus containing only finance tables and 10-K narrative it bought nothing measurable. The gain came from the chunking fix, not from the fusion. I'd expect the fusion to earn its keep on a larger, more heterogeneous corpus — but I haven't tested that, so I'm not claiming it.
* **Three of the six questions score 0.00 across every method**, including one (Apple's net income) that I never managed to explain. That's a real gap in the evaluation, not a rounding artefact.

---

## Things that went wrong

These are the parts I'd actually talk about in an interview, because they're where the real learning was.

**1. Wrong chunk, right document — twice diagnosed, twice the wrong hypothesis.** A question about Salesforce's go-to-market approach retrieved three `salesforce.txt` chunks and the model refused to answer. I assumed the candidate pool was too narrow, widened it from 5 to 20, then to 50 — the chunk containing the phrase only entered at rank #45. So I tested a second theory (that RRF's `1/(k+rank)` sum lets mediocre chunks found by *both* arms outrank a chunk that one arm ranks first — arithmetically true) and that wasn't it either, because neither arm had the chunk in its top 10. The actual answer: the question asked for a *synthesis* ("go-to-market strategy for its AI platform") that no single passage in the filing states. Rephrased in the document's own words, the same unmodified pipeline retrieved it at rank 1 and answered correctly. Same lesson as an earlier one — I asked about "Q1 2024" of a document that only ever says "Three Months Ended December 30, 2023" — so I've now stopped blaming the retriever when the question is at fault.

**2. A tokenizer "fix" that made things measurably worse.** BM25's top-10 for a Salesforce question was mostly Adobe and HubSpot chunks, and I put that down to naive `lower().split()` tokenization leaving possessives and punctuation attached (`salesforce's`, `platform?`). I swapped in a regex tokenizer. Precision@3 dropped from 0.28 to 0.17 — and preserving hyphens changed nothing, meaning my theory about `go-to-market` was untestable on this question set. Two things I'd got backwards: `rank_bm25` silently ignores query terms that aren't in the corpus vocabulary, so those tokens were inert rather than harmful; and normalising them (`2023?` → `2023`, `apple's` → `apple`) turned them into terms present in nearly *every* Apple chunk, which diluted the rare terms that actually discriminate. Reverted, with a comment in the code explaining why the naive version stays.

**3. A metric bug that reported a hallucination that wasn't there.** The faithfulness check flagged `6,057`, `223`, `823` as unsupported. They were all real — the harness rebuilt the context using a pool of 5 while `answer()` was using 20, so it compared the model's answer against text the model never saw. Fixed by making the pool width a single constant both files read.

**4. A hallucination that was actually arithmetic.** Asked how Apple's services business *performed*, the model returned a gross margin of `$16,837` derived from `23,117 − 6,280`. Both operands are in the source; the result isn't, and it appears nowhere in any of the seven source files. The maths checks out, but the figure is unverifiable, so the checker flags it — which I think is the correct behaviour, because LLM arithmetic that happens to be right this time isn't something you can rely on. Worth knowing that this behaviour is inconsistent: a later run answered the same question using only figures copied verbatim.

**5. Infrastructure, not code.** Ollama's CUDA worker died mid-session with `STATUS_STACK_BUFFER_OVERRUN` / "shared object initialization failed". The pipeline was fine. Worth recording only because the first thing I did was check the index — 393 vectors, intact — rather than assuming the crash had corrupted state. It hadn't.

---

## Known limitations

* **Six test questions.** Everything above is directional evidence, not a benchmark. The 0.28 → 0.17 tokenizer result rests on two questions.
* **One question scores 0.00 everywhere** (Apple net income, label `{3}`) and I never explained it.
* **Narrative citations are imprecise.** Each MarTech 10-K is loaded as one document, so every source line reads `salesforce.txt — page 0`. The page number is meaningless there.
* **The faithfulness check is a proxy.** It catches unsupported numbers and nothing else — a wrong claim made of words passes straight through.
* **No query rewriting.** Compound questions fail when no single passage joins the two halves of the question.
* **The reranker is off-domain.** `ms-marco-MiniLM-L-12-v2` was trained on web passages, not financial tables, and its scores cluster tightly (0.9971 vs 0.9965 has been typical).

---

## If I picked this up again

Resolve the unexplained miss first, since it's the only result I can't account for. After that: query rewriting for compound questions, verifying derived figures programmatically instead of just flagging them, and putting real chunk offsets into citations for the narrative sources.

---

## License

Released under the MIT License.
