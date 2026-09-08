"""
Step 7: Generation.

Builds a grounded prompt from retrieved chunks and calls GitHub Models
(via its OpenAI-compatible endpoint, so we can use the standard `openai`
SDK), forcing JSON-mode output so the response can be parsed directly into
our DecisionOutput schema (see schemas.py) without brittle text parsing.
"""
from __future__ import annotations

import json

from openai import OpenAI

from hr_rag.config import GITHUB_MODELS_BASE_URL, GITHUB_MODELS_MODEL, GITHUB_MODELS_TOKEN
from hr_rag.schemas import RetrievedChunk

_client = OpenAI(api_key=GITHUB_MODELS_TOKEN, base_url=GITHUB_MODELS_BASE_URL)

SYSTEM_PROMPT = """You are an HR policy assistant answering employee questions using ONLY the \
policy excerpts provided in the user message. Never use outside knowledge.

Rules:
1. If the excerpts contain multiple versions of the same policy (same doc_id, \
different version/effective_date), do NOT pick one arbitrarily. Set \
"conflict_flag" to true, set "next_action" to "escalate_to_hr_review", and \
explain the discrepancy in "answer" instead of giving a single definitive number.
2. Every claim in "answer" must be traceable to a specific excerpt. List those \
excerpts in "cited_sections" as "<doc_id> <version> - <short section/quote>".
3. If the excerpts don't contain enough information to answer, say so plainly, \
set confidence to "low", and set next_action to "escalate_to_hr_review".
4. Respond with ONLY a single JSON object, no markdown fences, no extra text, \
matching exactly this shape:
{
  "answer": "string",
  "cited_sections": ["string", "..."],
  "confidence": "high | medium | low",
  "conflict_flag": true | false,
  "next_action": "send_to_employee | escalate_to_hr_review",
  "draft_email": "a ready-to-send, professional email to the employee that \
answers their question with inline citations; if conflict_flag is true, this \
should instead briefly explain the case is being escalated to HR review"
}
"""


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for rc in chunks:
        c = rc.chunk
        blocks.append(
            f"[Excerpt: {c.doc_id} {c.version} | region={c.region} | "
            f"effective={c.effective_date} | source={c.source_file}]\n{c.text}"
        )
    return "\n\n---\n\n".join(blocks)


def generate_decision(question: str, retrieved: list[RetrievedChunk]) -> dict:
    """Call the LLM and return the parsed raw JSON dict (pre-validation)."""
    context = _format_context(retrieved)
    user_prompt = f"Employee question: {question}\n\nPolicy excerpts:\n{context}"

    response = _client.chat.completions.create(
        model=GITHUB_MODELS_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    return json.loads(response.choices[0].message.content)
