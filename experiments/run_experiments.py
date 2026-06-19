"""
Part 7: run_experiments.py — Run 3 experiments comparing RAGAS scores.

Experiments:
  Exp 1: Top-K 3 vs 5
  Exp 2: Chunk size 300 vs 400
  Exp 3: Prompt v1 vs Prompt v2

Usage:
    python experiments/run_experiments.py
"""

import json
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()
RESULTS_DIR = Path("eval/results")


def run_variant(top_k: int, chunk_size: int, prompt_version: str, reingest: bool = False) -> dict:
    """Run evaluate.py for a given configuration and return scores."""
    args = [
        sys.executable, "eval/evaluate.py",
        f"--top_k={top_k}",
        f"--chunk_size={chunk_size}",
        f"--prompt_version={prompt_version}",
    ]
    if reingest:
        args.append("--reingest")

    console.print(f"\n[bold cyan]Running variant: top_k={top_k}, chunk={chunk_size}, prompt={prompt_version}[/bold cyan]")
    subprocess.run(args, check=True)

    variant_name = f"topk{top_k}_chunk{chunk_size}_prompt{prompt_version}"
    result_path = RESULTS_DIR / f"ragas_scores_{variant_name}.json"
    with open(result_path) as f:
        return json.load(f)


def compare_experiments(results: list[dict]):
    """Print a comparison table of all experiment variants."""
    table = Table(title="Experiment Comparison", show_header=True, header_style="bold magenta")
    table.add_column("Variant", style="cyan", width=35)
    table.add_column("Faithfulness", justify="right")
    table.add_column("Ans. Relevancy", justify="right")
    table.add_column("Ctx. Precision", justify="right")
    table.add_column("Ctx. Recall", justify="right")

    for r in results:
        s = r["scores"]
        table.add_row(
            r["variant"],
            f"{s.get('faithfulness', 0):.4f}",
            f"{s.get('answer_relevancy', 0):.4f}",
            f"{s.get('context_precision', 0):.4f}",
            f"{s.get('context_recall', 0):.4f}",
        )

    console.print(table)

    # Identify best variant by average score
    def avg_score(r):
        s = r["scores"]
        return sum(s.values()) / len(s)

    best = max(results, key=avg_score)
    console.print(f"\n[bold green]Best variant: {best['variant']} (avg score: {avg_score(best):.4f})[/bold green]")

    # Save comparison
    comparison_path = RESULTS_DIR / "experiment_comparison.json"
    comparison_path.write_text(json.dumps(results, indent=2))
    console.print(f"[green]Comparison saved to {comparison_path}[/green]")


def main():
    all_results = []

    # Experiment 1: Top-K comparison (chunk 400, prompt v1)
    console.print("\n[bold yellow]═══ EXPERIMENT 1: Top-K 3 vs 5 ═══[/bold yellow]")
    all_results.append(run_variant(top_k=3, chunk_size=400, prompt_version="v1", reingest=False))
    all_results.append(run_variant(top_k=5, chunk_size=400, prompt_version="v1", reingest=False))

    # Experiment 2: Chunk size comparison (top_k 5, prompt v1)
    console.print("\n[bold yellow]═══ EXPERIMENT 2: Chunk 300 vs 400 ═══[/bold yellow]")
    all_results.append(run_variant(top_k=5, chunk_size=300, prompt_version="v1", reingest=True))
    all_results.append(run_variant(top_k=5, chunk_size=400, prompt_version="v1", reingest=True))

    # Experiment 3: Prompt variant comparison (top_k 5, chunk 400)
    console.print("\n[bold yellow]═══ EXPERIMENT 3: Prompt v1 vs v2 ═══[/bold yellow]")
    all_results.append(run_variant(top_k=5, chunk_size=400, prompt_version="v1", reingest=False))
    all_results.append(run_variant(top_k=5, chunk_size=400, prompt_version="v2", reingest=False))

    console.print("\n[bold magenta]═══ FINAL COMPARISON ═══[/bold magenta]")
    compare_experiments(all_results)


if __name__ == "__main__":
    main()
