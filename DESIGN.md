# Governed HR Policy RAG: Design Notes

## Positioning

> An end-to-end prototype of a governed HR-policy RAG system, with region-aware retrieval, policy-version conflict detection, deterministic review gates, structured outputs, and human escalation.

This is intentionally presented as a prototype, not a production-ready enterprise platform. It demonstrates practical RAG engineering and makes the reliability, security, and operating gaps visible.

## Decisions and tradeoffs

### Retrieval and policy currency

- FAISS plus BM25 and a CrossEncoder keeps local development transparent and inexpensive for a small policy corpus.
- Region filtering admits the requested region and `Global`, avoiding accidental US/EU substitution.
- Retrieved older versions remain in the audit context, while the newest effective date is selected for generation.
- Multiple versions or insufficient evidence fail closed into HR review. This sacrifices some automation for lower compliance risk.
- A production deployment should move the index and document metadata to a managed, access-controlled store and add an explicit policy approval state.

### Deterministic governance

- The LangGraph review gate is code, not a prompt instruction. The model cannot override missing evidence or unresolved version conflicts.
- Pydantic validates the structured decision contract before it reaches a caller.
- Cases sent to HR are persisted in SQLite with status transitions and reviewer notes. SQLite is suitable for a single-instance demonstration; production needs PostgreSQL or a case-management integration.

### Evaluation

- `data/eval/questions.json` is a small, versioned benchmark with expected documents, regions, and policy versions.
- `scripts/04_evaluate.py` reports hit@K, MRR, region precision, and version recall. It is deterministic and can run without an LLM call once models are available.
- The next evaluation layer should add labeled citation correctness, groundedness, refusal quality, and regression thresholds in CI. Human review is still needed for nuanced policy language.

### Observability and security

- Request counters, pipeline timings, and failures are exposed through `/api/metrics` and standard Python logs.
- An optional `HR_API_KEY` protects API routes for local deployment. This is a demonstration control, not a substitute for SSO, RBAC, secret rotation, audit export, or network policy.
- CORS is configured from `ALLOWED_ORIGINS`; request models enforce size bounds. Logs deliberately record operation metadata rather than employee question contents.

### Ingestion and deployment

- Ingestion validates duplicate document versions, rejects empty input, and writes processed JSON atomically so a failed run does not leave a half-written artifact.
- The checked-in Dockerfile runs the already-built frontend and API; Compose persists the review database in a volume.
- A production ingestion job should be scheduled, idempotent by source hash, signed or approved before indexing, observable, and able to roll back an index version.

## Explicit non-goals

The prototype does not claim enterprise identity, legal advice, PII classification, multi-tenant isolation, high availability, disaster recovery, or fully automated policy approval. Those are design topics to address before handling real employee data.

## Demonstration flow

```text
ingest -> validate -> chunk -> index -> evaluate retrieval -> serve API/UI
                                              |
                               conflict/missing evidence -> SQLite HR case
```

Suggested demo commands:

```powershell
python scripts/01_ingest.py
python scripts/02_build_index.py
python scripts/04_evaluate.py
docker compose up --build
```