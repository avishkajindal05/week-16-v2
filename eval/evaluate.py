"""
Part 5: evaluate.py — Run RAGAS evaluation over the golden dataset.

Usage:
    python eval/evaluate.py
    python eval/evaluate.py --top_k 3 --chunk_size 300   # experiment variant

Outputs:
    eval/results/ragas_scores_<variant>.json
    eval/results/scorecard_<variant>.txt
"""

from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings



import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from datasets import Dataset

# RAGAS metrics
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
# import google.generativeai as genai
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag_pipeline import CartlyRAGPipeline
from app.ingest import main as run_ingest

console = Console()
RESULTS_DIR = Path("eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_pipeline_on_golden_set(
    golden_path: Path,
    top_k: int = 5,
    prompt_version: str = "v1",
) -> pd.DataFrame:
    """Run the RAG pipeline over every row in the golden set and collect results."""
    df = pd.read_csv(golden_path)
    console.print(f"[cyan]Running pipeline on {len(df)} golden questions (top_k={top_k}, prompt={prompt_version})[/cyan]")

    pipeline = CartlyRAGPipeline(top_k=top_k, prompt_version=prompt_version)

    answers = []
    contexts = []

    for _, row in df.iterrows():
        result = pipeline.answer(row["question"])
        answers.append(result["answer"])
        contexts.append([c["text"] for c in result["chunks"]])

    df["answer"] = answers
    df["contexts"] = contexts
    return df


def build_ragas_dataset(df: pd.DataFrame) -> Dataset:
    """Convert the dataframe into a RAGAS-compatible HuggingFace Dataset."""
    return Dataset.from_dict(
        {
            "question": df["question"].tolist(),
            "answer": df["answer"].tolist(),
            "contexts": df["contexts"].tolist(),
            "ground_truth": df["ground_truth"].tolist(),
        }
    )



from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings


def compute_ragas_scores(dataset: Dataset):
    """Compute RAGAS metrics using Groq LLM + HuggingFace embeddings."""

    # Judge LLM (Groq)
    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="openai/gpt-oss-120b",  # llama-3.3-70b-versatile was deprecated by Groq on 2026-06-17
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            temperature=0,
        )
    )

    # Embedding model (avoid OpenAIEmbeddings)
    embedding_model = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=judge_llm,
        embeddings=embedding_model,
    )

    return result






def print_scorecard(scores: dict, variant_name: str):
    """Print a rich-formatted scorecard."""
    table = Table(title=f"RAGAS Scorecard — {variant_name}", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Score", justify="right", style="green")
    table.add_column("Target", justify="right", style="yellow")
    table.add_column("Status", justify="center")

    targets = {
        "faithfulness": 0.90,
        "answer_relevancy": 0.80,
        "context_precision": 0.70,
        "context_recall": 0.80,
    }

    for metric, target in targets.items():
        score = scores.get(metric, 0.0)
        status = "✅ PASS" if score >= target else "❌ MISS"
        table.add_row(metric.replace("_", " ").title(), f"{score:.4f}", f"{target:.2f}", status)

    console.print(table)


def save_results(scores: dict, variant_name: str):
    """Save scores as JSON and a plain text scorecard."""
    output = {
        "variant": variant_name,
        "timestamp": datetime.now().isoformat(),
        "scores": {k: round(float(v), 4) for k, v in scores.items()},
    }

    json_path = RESULTS_DIR / f"ragas_scores_{variant_name}.json"
    json_path.write_text(json.dumps(output, indent=2))

    txt_lines = [f"RAGAS Scorecard — {variant_name}", "=" * 40]
    for k, v in output["scores"].items():
        txt_lines.append(f"{k:<30} {v:.4f}")
    txt_path = RESULTS_DIR / f"scorecard_{variant_name}.txt"
    txt_path.write_text("\n".join(txt_lines))

    console.print(f"[green]Saved results to {json_path} and {txt_path}[/green]")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--chunk_size", type=int, default=400)
    parser.add_argument("--chunk_overlap", type=int, default=50)
    parser.add_argument("--prompt_version", type=str, default="v1")
    parser.add_argument("--reingest", action="store_true", help="Re-run ingestion before evaluation")
    args = parser.parse_args()

    variant_name = f"topk{args.top_k}_chunk{args.chunk_size}_prompt{args.prompt_version}"
    console.print(f"\n[bold cyan]Starting RAGAS evaluation: {variant_name}[/bold cyan]\n")

    if args.reingest:
        console.print("[yellow]Re-ingesting documents...[/yellow]")
        run_ingest(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    golden_path = Path("eval/golden_set.csv")
    df = run_pipeline_on_golden_set(golden_path, top_k=args.top_k, prompt_version=args.prompt_version)

    dataset = build_ragas_dataset(df)

    console.print("[cyan]Computing RAGAS scores...[/cyan]")
    result = compute_ragas_scores(dataset)

    scores = {
        "faithfulness": result["faithfulness"],
        "answer_relevancy": result["answer_relevancy"],
        "context_precision": result["context_precision"],
        "context_recall": result["context_recall"],
    }

    print_scorecard(scores, variant_name)
    save_results(scores, variant_name)


if __name__ == "__main__":
    main()
