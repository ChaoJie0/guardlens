# Changelog

All notable changes are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] — 2026-11-10

First public release. Consolidates the W4 evaluation work and the W5 dataset
refresh, and prepares the repository for outside use.

### Added

- FastAPI backend (`guardlens.server`) that loads BGE-M3, the reranker and the
  Qdrant client once, streaming answers over SSE; the Streamlit dashboard now
  talks to it instead of reloading models on every refresh.
- Retrieval tuning for the 7,689-document corpus: wider channels, three-way
  type balancing with a reserved cluster quota, per-channel top-2 boosts,
  expanded structured anchors, rerank safety rules, and rerank against
  question + intent expansions.
- `eval/diagnose_recall.py` and `eval/trace_recall.py` — per-question and
  per-stage recall diagnostics.
- Synthetic 42-event sample log + generator (`examples/make_sample_data.py`)
  so a fresh clone can run the pipeline without own data.
- `GUARDLENS_DB` / `GUARDLENS_CORPUS` / `GUARDLENS_API_BASE` path overrides.
- Open-source scaffolding: CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, CITATION,
  issue/PR templates, editorconfig, CI format check.

### Changed

- **LLM configuration is now generic**: `GUARDLENS_LLM_*` or the standard
  `OPENAI_API_KEY` / `OPENAI_BASE_URL` variables. The previous build read
  credentials from a local tool-specific file; that coupling is gone.
- **Baseline diff is opt-in**: `--baseline baseline.json` instead of a hard-coded
  snapshot that made every other corpus look "unhealthy" (and exit non-zero).
- Deny-reason buckets key on the generic `BLOCKED (<category>):` header shape.
- Project metadata: version, description, keywords, classifiers, PEP 621
  license fields; default index set to PyPI (mirror can be overridden locally).
- `uv.lock` URLs point at canonical PyPI files.

### Fixed

- Lint/format violations in `app.py` and the two recall diagnostics.
- Dashboard/API paths no longer hard-code an author-specific database location.

## [0.3.1] — 2026-09-19

### Changed

- BGE-M3 runs on Apple Metal/MPS with a real-encode probe and CPU fallback;
  ~3× faster indexing. HuggingFace offline mode when the model is cached.

## [0.3.0] — 2026-09-19

### Added

- Hybrid retrieval: dependency-free BM25 + BGE-M3 dense in Qdrant, per-channel
  type balancing, RRF fusion (k=60), `bge-reranker-v2-m3` rerank.
- Query router (STATS / EVIDENCE / MIXED) and guarded Text2SQL
  (SELECT-only, read-only copy, dry-run + row cap) — see ADR 0001/0002.
- Security analytics: risk-family attribution, bypass-sequence detection,
  false-positive proxies with a separate feedback database.
- W2-vs-W3 retrieval comparison example.

## [0.2.0] — 2026-09-19

### Added

- First RAG loop: evidence cards (±5 context), session clusters, report chunks.
- Local BGE-M3 embeddings into Chroma, dense retrieval with per-type quotas and
  structured fallback.
- Cited Q&A with citation verification, abstention contract, token streaming.

## [0.1.0] — 2026-09-19

### Added

- Streaming `audit.jsonl` ingestion with schema validation, `args` repair and
  quarantine, into DuckDB.
- Metric computation and dashboard (KPI, trend, category, tool tables).
- Optional regression baseline with monotonic and invariant semantics.
