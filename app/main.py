"""
Part 6: main.py — FastAPI backend with Langfuse observability integration.

Run:
    uvicorn app.main:app --reload --port 8000
"""

import os
import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langfuse import Langfuse

from app.rag_pipeline import CartlyRAGPipeline, GROQ_MODEL

load_dotenv()

# ── Langfuse client ──────────────────────────────────────────────────────────
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)

# ── App state ────────────────────────────────────────────────────────────────
pipeline: CartlyRAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("Loading RAG pipeline...")
    pipeline = CartlyRAGPipeline(top_k=5)
    print("RAG pipeline ready.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="Cartly RAG Eval Harness",
    description="Evaluable RAG pipeline over Cartly synthetic policy documents.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    prompt_version: str = "v1"


class ChunkInfo(BaseModel):
    source: str
    chunk_index: int
    score: float
    text: str


class QueryResponse(BaseModel):
    query_id: str
    query: str
    answer: str
    sources: list[str]
    chunks: list[ChunkInfo]
    latency_ms: float
    top_k: int
    prompt_version: str


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "pipeline_loaded": pipeline is not None}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")

    # Dynamically set top_k and prompt_version if different from default
    pipeline.top_k = req.top_k
    pipeline.prompt_version = req.prompt_version

    query_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()

    result = pipeline.answer(req.query)

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    # ── Langfuse trace ────────────────────────────────────────────────────
    try:
        trace = langfuse.trace(
            id=query_id,
            name="cartly-rag-query",
            input={"query": req.query},
            output={"answer": result["answer"]},
            metadata={
                "top_k": req.top_k,
                "prompt_version": req.prompt_version,
                "sources": result["sources"],
                "latency_ms": latency_ms,
            },
        )

        # Retrieval span
        trace.span(
            name="retrieval",
            input={"query": req.query},
            output={
                "num_chunks": len(result["chunks"]),
                "sources": result["sources"],
            },
            metadata={"top_k": req.top_k},
        )

        # Generation span
        trace.span(
            name="generation",
            input={"context_length": len(result["context"]), "prompt_version": req.prompt_version},
            output={"answer_length": len(result["answer"])},
            metadata={"model": GROQ_MODEL, "latency_ms": latency_ms},
        )

        langfuse.flush()
    except Exception as e:
        print(f"[Langfuse] Logging failed (non-critical): {e}")

    return QueryResponse(
        query_id=query_id,
        query=req.query,
        answer=result["answer"],
        sources=result["sources"],
        chunks=[
            ChunkInfo(
                source=c["source"],
                chunk_index=c["chunk_index"],
                score=c["score"],
                text=c["text"][:300] + "..." if len(c["text"]) > 300 else c["text"],
            )
            for c in result["chunks"]
        ],
        latency_ms=latency_ms,
        top_k=req.top_k,
        prompt_version=req.prompt_version,
    )


@app.get("/policies")
def list_policies():
    """List all available policy documents in the corpus."""
    from pathlib import Path
    policies_dir = Path(os.getenv("POLICIES_DIR", "./data/policies"))
    files = [f.name for f in sorted(policies_dir.glob("*.txt"))]
    return {"count": len(files), "policies": files}
