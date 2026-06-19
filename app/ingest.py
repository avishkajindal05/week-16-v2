"""
Part 1 + 2: ingest.py
Load synthetic policy docs → chunk → embed (BAAI/bge-small-en-v1.5) → store in ChromaDB.

Run once:
    python app/ingest.py
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from rich.console import Console
from rich.progress import track

load_dotenv()

console = Console()

POLICIES_DIR = Path(os.getenv("POLICIES_DIR", "./data/policies"))
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "cartly_policies")

# ── Embedding model ─────────────────────────────────────────────────────────
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_documents(policies_dir: Path) -> list[dict]:
    """Load all .txt policy files from the policies directory."""
    docs = []
    for filepath in sorted(policies_dir.glob("*.txt")):
        text = filepath.read_text(encoding="utf-8")
        docs.append({"source": filepath.name, "text": text})
    console.print(f"[green]Loaded {len(docs)} policy documents.[/green]")
    return docs


def chunk_documents(
    docs: list[dict],
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> list[dict]:
    """
    Chunk documents using RecursiveCharacterTextSplitter.
    Each chunk retains its source filename for citation.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in docs:
        split_texts = splitter.split_text(doc["text"])
        for i, text in enumerate(split_texts):
            chunks.append(
                {
                    "id": f"{doc['source']}__chunk_{i}",
                    "text": text.strip(),
                    "source": doc["source"],
                    "chunk_index": i,
                }
            )

    console.print(f"[green]Created {len(chunks)} chunks.[/green]")
    return chunks


def build_chromadb(
    chunks: list[dict],
    model_name: str = MODEL_NAME,
    persist_dir: Path = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """Embed chunks and persist to ChromaDB."""

    console.print(f"[cyan]Loading embedding model: {model_name}[/cyan]")
    model = SentenceTransformer(model_name)

    # Build ChromaDB client
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    # Drop and recreate for clean ingest
    try:
        client.delete_collection(collection_name)
        console.print(f"[yellow]Dropped existing collection: {collection_name}[/yellow]")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    console.print("[cyan]Embedding and inserting chunks...[/cyan]")
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks]

    # Embed in batches
    batch_size = 64
    all_embeddings = []
    for i in track(range(0, len(texts), batch_size), description="Embedding"):
        batch = texts[i : i + batch_size]
        # BGE models benefit from the query instruction only at query time
        embeddings = model.encode(batch, normalize_embeddings=True).tolist()
        all_embeddings.extend(embeddings)

    collection.add(
        documents=texts,
        embeddings=all_embeddings,
        ids=ids,
        metadatas=metadatas,
    )

    console.print(
        f"[bold green]✓ Ingested {len(chunks)} chunks into ChromaDB "
        f"collection '{collection_name}'.[/bold green]"
    )
    return collection


def main(chunk_size: int = 400, chunk_overlap: int = 50):
    docs = load_documents(POLICIES_DIR)
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    build_chromadb(chunks)
    console.print("[bold green]Ingestion complete![/bold green]")


if __name__ == "__main__":
    main()
