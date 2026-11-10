# Milestones

The project was built in five increments; each one is a working commit series.
This page keeps the engineering narrative that does not fit in the README.

| Milestone | What it added |
|---|---|
| W1 | audit ingestion, DuckDB metrics, Streamlit dashboard, baseline invariants |
| W2 | first RAG loop: semantic cards/clusters, BGE-M3 + Chroma, cited Q&A |
| W3 | hybrid retrieval (BM25 + dense RRF + rerank) on Qdrant, guarded SQL router, three security analyses |
| W4 | 20-question golden set, retrieval ablation, latency/cost accounting |
| W5 | full dataset refresh, data hygiene, retrieval re-tuning, FastAPI/SSE backend |

## W1 — data foundation

`ingest.py` streams JSONL, validates against the Pydantic schema, repairs
double-encoded `args`, quarantines rows that fail the core contract with their
line number, and writes everything into DuckDB. The dashboard exposes KPI,
trend, category and tool tables.

A frozen snapshot of the author's corpus is used as a **regression baseline**:
because the audit log is append-only, volume metrics are compared with
monotonic semantics (`actual >= baseline` is healthy; a smaller number means
data loss), while structural invariants such as `fail-closed = 5` must still
match exactly. Counts are always recomputed — the baseline is never used to
fabricate results. Supplying a baseline is optional (`--baseline baseline.json`).

## W2 — first RAG loop

Evidence cards with ±5 surrounding events, session clusters, report chunks;
BGE-M3 embeddings on Apple Metal/MPS with a real-encode probe and CPU fallback,
batch 64 on GPU, HuggingFace offline mode when the model is cached; Chroma as
the W2 store; citation verification and the abstention contract.

## W3 — production hardening

Qdrant as the primary store, hybrid retrieval, guarded Text2SQL behind three
gates, and the security analyses that a generic RAG demo does not have:

* **risk-family attribution** — deny events merged into families
  (egress / out-of-sandbox write / process / fail-closed / unknown-unrollbackable
  / data exfil) with counts and representative events. The headline finding on
  the author's corpus: `unknown ≈ 75%` of denies, i.e. a conservative policy
  plus an observability blind spot.
* **bypass-sequence detection** — deny(A) followed by a same-intent call(B)
  within a window where both sides show network intent (`curl → web_fetch`,
  `web_fetch → python urllib`, `-A Mozilla` UA spoof, a read-only proxy
  gateway). 19 high-confidence signals found.
* **false-positive proxies + feedback loop** — the same substantive action
  (intent signature over network/package/permission/convert verbs plus target)
  denied then allowed within N minutes is a policy-tuning hint. 👍/👎 feedback
  is written to a **separate** `feedback.duckdb`; audit data is never mutated.
  This is a proxy, not a true FP rate — human confirmation is the point.

## W4 — offline evaluation

20-question golden set, ablation, latency/cost accounting, and two technical
articles ([01](../articles/01-siem-rag-the-right-way.md),
[02](../articles/02-why-not-let-llm-compute-numbers-sql-router.md)).

## W5 — full dataset refresh & tuning (2026-10-11)

Audit volume grew from 22k (frozen 09-19) to **122,245 events** (09-06 →
10-11). Corpus rebuilt to **7,689 docs** (event_card 1,844 / event_cluster 118 /
report_chunk 5,727); retrieval re-tuned end to end and the backend split into
FastAPI + SSE so the dashboard no longer reloads models on refresh.

* **Data hygiene** — 203 test records (`curl -s https://evil.example/x`,
  RFC 2606) were polluting the curl anchor and generating 100+ test event cards.
  Filtered out → `audit-clean.jsonl` (122,245 rows; deny 1,683; block rate
  1.38%; `fail-closed = 5` invariant kept; 0 quarantined).
* **Golden set re-grounded** — STATS numbers moved to the clean corpus; evidence
  gold extended with verified new-period events; one wrongly-added gold (a node
  test-script egress denial, not a curl block) removed after content verification.
* **Retrieval** — see [retrieval.md](retrieval.md#tuning-log-2026-10-11-refresh).
