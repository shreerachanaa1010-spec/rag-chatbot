"""Run the checked-in offline retrieval benchmark.

Usage: python scripts/04_evaluate.py
"""
from __future__ import annotations

import json

from rich.console import Console

from hr_rag.evaluation import EvaluationExample, evaluate_retrieval

console = Console()


def main() -> None:
    examples = [
        EvaluationExample(
            question=item["question"],
            region=item.get("region"),
            expected_doc_ids=tuple(item["expected_doc_ids"]),
            expected_versions=tuple(item.get("expected_versions", [])),
        )
        for item in json.loads(open("data/eval/questions.json", encoding="utf-8").read())
    ]
    metrics = evaluate_retrieval(examples)
    console.print_json(json.dumps(metrics))
    if metrics["hit_at_k"] < 0.75:
        raise SystemExit("Retrieval quality gate failed: hit_at_k < 0.75")


if __name__ == "__main__":
    main()