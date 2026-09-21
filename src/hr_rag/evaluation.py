"""Offline retrieval evaluation for a small, versioned HR benchmark."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable



@dataclass(frozen=True)
class EvaluationExample:
    question: str
    region: str | None
    expected_doc_ids: tuple[str, ...]
    expected_versions: tuple[str, ...] = ()


def evaluate_retrieval(
    examples: list[EvaluationExample],
    retriever: Callable[..., list] | None = None,
    top_k: int = 5,
) -> dict[str, float | int]:
    """Return hit@k, reciprocal rank, region precision, and version recall."""
    if not examples:
        return {"examples": 0, "hit_at_k": 0.0, "mrr": 0.0, "region_precision": 0.0, "version_recall": 0.0}

    if retriever is None:
        from hr_rag.retrieval import retrieve

        retriever = retrieve

    hits = reciprocal_rank = region_precision = version_recall = 0.0
    for example in examples:
        results = retriever(example.question, region=example.region, top_k=top_k)
        expected_ids = set(example.expected_doc_ids)
        matching_positions = [index for index, item in enumerate(results, start=1) if item.chunk.doc_id in expected_ids]
        hits += bool(matching_positions)
        reciprocal_rank += 1 / matching_positions[0] if matching_positions else 0
        if results and example.region:
            region_precision += sum(item.chunk.region in {example.region, "Global"} for item in results) / len(results)
        elif not example.region:
            region_precision += 1
        expected_versions = set(example.expected_versions)
        if expected_versions:
            returned_versions = {item.chunk.version for item in results}
            version_recall += bool(expected_versions & returned_versions)
        else:
            version_recall += 1

    count = len(examples)
    return {
        "examples": count,
        "hit_at_k": hits / count,
        "mrr": reciprocal_rank / count,
        "region_precision": region_precision / count,
        "version_recall": version_recall / count,
    }


def citation_accuracy(citations: list[str], retrieved: list) -> float:
    """Measure the fraction of generated citations grounded in retrieved chunks."""
    if not citations:
        return 0.0
    valid_sources = {f"{item.chunk.doc_id} {item.chunk.version}" for item in retrieved}
    return sum(any(source in citation for source in valid_sources) for citation in citations) / len(citations)


def groundedness(answer: str, retrieved: list) -> float:
    """Approximate answer grounding using content-word overlap with evidence."""
    answer_words = {word.lower() for word in answer.split() if len(word) > 3}
    evidence_words = {
        word.lower()
        for item in retrieved
        for word in item.chunk.text.split()
        if len(word) > 3
    }
    return len(answer_words & evidence_words) / len(answer_words) if answer_words else 0.0


def evaluate_answers(answered_examples: list[tuple[dict, list]]) -> dict[str, float | int]:
    """Evaluate generated decisions without making additional model calls."""
    if not answered_examples:
        return {"examples": 0, "groundedness": 0.0, "citation_accuracy": 0.0}
    grounded_scores = [groundedness(decision.get("answer", ""), retrieved) for decision, retrieved in answered_examples]
    citation_scores = [citation_accuracy(decision.get("cited_sections", []), retrieved) for decision, retrieved in answered_examples]
    return {
        "examples": len(answered_examples),
        "groundedness": sum(grounded_scores) / len(grounded_scores),
        "citation_accuracy": sum(citation_scores) / len(citation_scores),
    }