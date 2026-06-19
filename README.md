# Cartly RAG Eval Harness — Week 16 Mini Project

> Evaluable RAG pipeline over synthetic e-commerce policy documents.  
> Stack: ChromaDB · BAAI/bge-small-en-v1.5 · Groq (Llama/GPT-OSS) · RAGAS · Langfuse · FastAPI

---

## Problem

Cartly is an e-commerce platform with 15 customer-facing policy documents (refunds, shipping, returns, etc.). Customers frequently ask questions like "How long does a refund take?" or "Can I cancel after shipping?". This project builds a **thin, evaluable RAG slice** that answers such queries grounded in policy text, then measures quality rigorously.

---

## Architecture

```
Policy Documents (.txt)
        ↓
   Document Loader
        ↓
RecursiveCharacterTextSplitter (400 chars, 50 overlap)
        ↓
BAAI/bge-small-en-v1.5 Embedder
        ↓
  ChromaDB (cosine similarity)
        ↓
  Retriever (Top-K chunks)
        ↓
  Groq (Llama/GPT-OSS)
        ↓
  Grounded Answer + Sources
        ↓
   RAGAS Evaluation
        ↓
  Langfuse Observability
```

---

## Project Structure

```
Week16-Evaluable-Core/
├── app/
│   ├── ingest.py          # Load → Chunk → Embed → ChromaDB
│   ├── rag_pipeline.py    # Retriever + Groq answer generation
│   └── main.py            # FastAPI backend with Langfuse tracing
├── data/
│   ├── policies/          # 15 synthetic Cartly policy .txt files
│   └── chroma_db/         # Persisted ChromaDB (generated)
├── eval/
│   ├── golden_set.csv     # 40 Q&A pairs with ground truth + source
│   ├── evaluate.py        # RAGAS evaluation script
│   └── results/           # Scorecard JSONs (generated)
├── experiments/
│   └── run_experiments.py # Runs 3 experiments (top-k, chunk size, prompt)
├── observability/         # Langfuse screenshots
├── screenshots/           # README screenshots
├── .env                   # API keys (not committed)
├── requirements.txt
├── README.md
└── EXEC_MEMO.md
```

---

## Setup

### 1. Prerequisites

- Python 3.11.9
- Windows (VS Code)
- Groq Cloud account (free tier) — https://console.groq.com/keys
- Langfuse account (free cloud tier)

### 2. Clone and create virtual environment

```bash
git clone <your-repo-url>
cd Week16-Evaluable-Core

python -m venv venv
venv\Scripts\activate        # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you don't have a GPU, install CPU-only torch first to avoid a multi-GB download:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 4. Configure environment

Copy `.env` and fill in your keys:

```env
GROQ_API_KEY=your_groq_api_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

## Running the Pipeline

### Step 1 — Ingest documents into ChromaDB

```bash
python app/ingest.py
```

Loads 15 policy documents → chunks → embeds → stores in `data/chroma_db/`.

### Step 2 — Start the FastAPI server

```bash
uvicorn app.main:app --reload --port 8000
```

Test with:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How long does a refund take?", "top_k": 5}'
```

### Step 3 — Run RAGAS evaluation

```bash
python eval/evaluate.py
```

Optional arguments:
```bash
python eval/evaluate.py --top_k 3 --chunk_size 300 --prompt_version v2 --reingest
```

### Step 4 — Run all experiments

```bash
python experiments/run_experiments.py
```

---

## Dataset

**Golden evaluation set:** `eval/golden_set.csv`  
40 questions across 4 difficulty levels:

| Category | Count | Example |
|---|---|---|
| Easy (direct) | 20 | "How long does a refund take?" |
| Ambiguous | 8 | "What happens to my payment if delivery fails?" |
| Multi-document | 7 | (requires reading 2+ policies) |
| Out-of-corpus | 5 | (Cartly doesn't cover this — expect honest "I don't know") |

Each row: `question`, `ground_truth`, `expected_source`

---

## Embedding Model

**BAAI/bge-small-en-v1.5** (local, free)
- 384-dimensional embeddings
- Cosine similarity index in ChromaDB
- Query-time prefix: `"Represent this sentence for searching relevant passages: "`

---

## Evaluation Results

### Baseline (top_k=5, chunk=400, prompt=v1)

| Metric | Score | Target | Status |
|---|---|---|---|
| Faithfulness | 0.92 | 0.90 | ✅ PASS |
| Answer Relevancy | 0.87 | 0.80 | ✅ PASS |
| Context Precision | 0.81 | 0.70 | ✅ PASS |
| Context Recall | 0.89 | 0.80 | ✅ PASS |



---

## Experiments

| Experiment | Variants Tested | Key Finding |
|---|---|---|
| Top-K | 3 vs **5** | top_k=5 improves context recall |
| Chunk Size | 300 vs **400** | 400-char chunks balance precision and recall |
| Prompt | v1 vs **v2** | Structured prompt (v2) improves faithfulness |


---

## Observability (Langfuse)

Every query logs:
- User query + generated answer
- Retrieved chunks and sources
- Latency (ms)
- Token estimates
- Retrieval and generation spans



---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/query` | Submit a customer query |
| GET | `/policies` | List available policy documents |

**POST /query body:**
```json
{
  "query": "How long does a refund take?",
  "top_k": 5,
  "prompt_version": "v1"
}
```

---

## Future Work

1. Add re-ranking (cross-encoder) between retrieval and generation.
2. Implement query expansion for ambiguous questions.
3. Add HyDE (Hypothetical Document Embeddings) for harder queries.
4. Move to streaming responses for lower perceived latency.
5. Build a Streamlit UI for interactive demos.
6. Add automated eval CI/CD to track metric regression.

---

## Tech Stack

| Component | Technology |
|---|---|
| Embedding | BAAI/bge-small-en-v1.5 (local) |
| Vector DB | ChromaDB |
| LLM | Groq — openai/gpt-oss-120b |
| Evaluation | RAGAS |
| Observability | Langfuse |
| Backend | FastAPI |
| Python | 3.11.9 |
