"""Small, dependency-free observability primitives for the prototype."""
from __future__ import annotations

import logging
import time
from collections import Counter
from contextlib import contextmanager
from typing import Iterator

from hr_rag.config import LOG_LEVEL

logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO))
logger = logging.getLogger("hr_rag")
_metrics = Counter()


def increment(name: str, value: int = 1) -> None:
    _metrics[name] += value


def metrics_snapshot() -> dict[str, int]:
    return dict(_metrics)


@contextmanager
def observe_operation(name: str, **fields: object) -> Iterator[None]:
    started = time.perf_counter()
    increment(f"{name}.started")
    try:
        yield
    except Exception:
        increment(f"{name}.failed")
        raise
    finally:
        increment(f"{name}.completed")
        logger.info("operation=%s duration_ms=%.1f %s", name, (time.perf_counter() - started) * 1000, " ".join(f"{key}={value}" for key, value in fields.items()))