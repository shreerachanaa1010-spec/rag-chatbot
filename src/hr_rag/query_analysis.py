"""Rule-based query classification and conversational query rewriting."""
from __future__ import annotations

import re
from typing import Literal, cast

from hr_rag.schemas import SearchCriteria

_AREA_TERMS = {
    "parental_leave": ("parental", "maternity", "paternity", "caregiver"),
    "pto": ("pto", "paid time off", "vacation", "time off"),
    "compensation": ("salary", "bonus", "compensation", "raise", "pay increase"),
    "performance": ("performance review", "annual review", "pip", "goals"),
    "health_benefits": ("health insurance", "medical", "benefits", "deductible"),
    "remote_work": ("remote", "hybrid", "work from home"),
    "conduct": ("conduct", "harassment", "retaliation", "investigation"),
    "career_development": ("career", "learning", "mentoring", "development"),
}
_INTENT_TERMS = {
    "eligibility": ("eligible", "eligibility", "qualify", "who can"),
    "amount": ("how much", "how many", "weeks", "days", "percentage", "salary"),
    "deadline": ("when", "deadline", "within", "before", "notice"),
    "process": ("how do i apply", "application", "submit", "request", "process"),
    "exception": ("exception", "unless", "special case", "excluded"),
    "definition": ("what is", "define", "meaning"),
    "comparison": ("difference", "compare", "versus", "vs "),
}
_REGIONS = {"us": "US", "u.s.": "US", "united states": "US", "eu": "EU", "europe": "EU", "european union": "EU", "global": "Global"}


def _find_term(question: str, terms: dict[str, tuple[str, ...]]) -> str | None:
    lowered = question.lower()
    return next((key for key, values in terms.items() if any(value in lowered for value in values)), None)


def _find_region(question: str) -> str | None:
    lowered = question.lower()
    return next((code for value, code in _REGIONS.items() if value in lowered), None)


def analyze_query(question: str, region: str | None = None, history: list[dict] | None = None) -> SearchCriteria:
    """Extract search filters and resolve short follow-ups against recent history."""
    context = ""
    if history and len(re.findall(r"[A-Za-z]+", question)) < 6:
        previous = history[0].get("question", "")
        context = f" Context from previous question: {previous}."
    expanded = f"{question.strip()}{context}"
    intent_value = _find_term(expanded, _INTENT_TERMS) or "general"
    return SearchCriteria(
        policy_area=_find_term(expanded, _AREA_TERMS) or "general",
        region=region or _find_region(expanded),
        employee_type="contractor" if "contractor" in expanded.lower() else ("full_time" if "full-time" in expanded.lower() or "full time" in expanded.lower() else None),
        intent=cast(Literal["eligibility", "amount", "deadline", "process", "exception", "definition", "comparison", "general"], intent_value),
        rewritten_query=expanded,
    )