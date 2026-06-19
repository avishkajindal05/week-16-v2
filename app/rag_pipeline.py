"""
Part 2 + 3: rag_pipeline.py
Retriever (ChromaDB + BGE) + Groq Llama/GPT-OSS answer generation with citations.
"""

import os
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "cartly_policies")
MODEL_NAME = "BAAI/bge-small-en-v1.5"

GROQ_MODEL = "openai/gpt-oss-120b"
# NOTE: llama-3.3-70b-versatile and llama-3.1-8b-instant were deprecated by Groq
# on 2026-06-17. openai/gpt-oss-120b is Groq's current recommended replacement.
# See https://console.groq.com/docs/deprecations for the latest list.
# Other valid free-tier options: "qwen/qwen3.6-27b", "llama-3.1-8b-instant"-> deprecated, use "openai/gpt-oss-20b" instead.

# BGE query instruction
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ── Prompt template ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Cartly's official customer support assistant.
Answer the customer's question using ONLY the provided policy context.
Be concise, accurate, and helpful.
At the end of your answer, always add a "Sources:" line listing the policy document(s) you used.
If the context does not contain enough information, say so clearly — do NOT guess or fabricate."""

USER_PROMPT_TEMPLATE = """CONTEXT:
{context}

CUSTOMER QUESTION:
{question}

Answer:"""


class CartlyRAGPipeline:
    def __init__(self, top_k: int = 5, prompt_version: str = "v1"):
        """
        Args:
            top_k: Number of chunks to retrieve.
            prompt_version: 'v1' (default) or 'v2' (more structured output).
        """
        # Configure Groq (OpenAI-compatible endpoint)
        self.client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )

        # Load embedding model
        self.embedder = SentenceTransformer(MODEL_NAME)

        # Connect to ChromaDB
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        self.collection = client.get_collection(COLLECTION_NAME)

        self.top_k = top_k
        self.prompt_version = prompt_version

    def retrieve(self, query: str) -> list[dict]:
        """Embed query and retrieve top-k chunks from ChromaDB."""
        query_with_prefix = BGE_QUERY_PREFIX + query
        query_embedding = self.embedder.encode(
            query_with_prefix, normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append(
                {
                    "text": doc,
                    "source": meta["source"],
                    "chunk_index": meta["chunk_index"],
                    "score": round(1 - dist, 4),  # cosine similarity
                }
            )
        return chunks

    def build_context(self, chunks: list[dict]) -> str:
        """Format retrieved chunks into a context string."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(f"[Chunk {i} | Source: {chunk['source']}]\n{chunk['text']}")
        return "\n\n---\n\n".join(parts)

    def _get_prompt(self, context: str, question: str) -> str:
        if self.prompt_version == "v2":
            return f"""{SYSTEM_PROMPT}

Format your answer as:
1. Direct answer (1–2 sentences)
2. Supporting detail (if needed)
3. Sources: [list document names]

CONTEXT:
{context}

CUSTOMER QUESTION:
{question}

Answer:"""
        # Default v1
        return USER_PROMPT_TEMPLATE.format(context=context, question=question)

    def answer(self, query: str) -> dict:
        """
        Full RAG pipeline: retrieve → build context → generate answer.

        Returns:
            dict with keys: query, answer, sources, chunks, context
        """
        chunks = self.retrieve(query)
        context = self.build_context(chunks)
        prompt = self._get_prompt(context, query)

        full_prompt = (
            f"{SYSTEM_PROMPT}\n\n{prompt}"
            if self.prompt_version == "v1"
            else prompt
        )

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )

        answer_text = response.choices[0].message.content.strip()

        sources = list({c["source"] for c in chunks})

        return {
            "query": query,
            "answer": answer_text,
            "sources": sources,
            "chunks": chunks,
            "context": context,
            "prompt_version": self.prompt_version,
            "top_k": self.top_k,
        }


if __name__ == "__main__":
    # Quick smoke test
    pipeline = CartlyRAGPipeline(top_k=5)
    result = pipeline.answer("How long does a refund take?")
    print("\n=== ANSWER ===")
    print(result["answer"])
    print("\n=== SOURCES ===")
    print(result["sources"])
    print("\n=== TOP CHUNKS (scores) ===")
    for c in result["chunks"]:
        print(f"  [{c['score']:.4f}] {c['source']} chunk_{c['chunk_index']}")
