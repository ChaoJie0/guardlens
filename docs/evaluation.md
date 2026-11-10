# Evaluation

Everything is produced by the harness — numbers are never hand-edited into
prose. The golden set, the runner and the reports live in `eval/`.

```bash
uv run python -m eval.run_eval                 # E2E (LLM calls) + ablation
uv run python -m eval.run_eval --ablation-only # retrieval grid only, no LLM
```

`eval_report.md` / `results.json` are written by the run; `eval_report.md` is
the human-readable artifact and is committed, `results.json` is a machine
artifact and is git-ignored.

## Golden set

[`eval/golden20.yaml`](../eval/golden20.yaml) — 20 questions across the three
answer paths:

* **STATS** — counts/rates that must come from DuckDB;
* **EVIDENCE** — "why was this blocked", "what happened next" questions with
  gold evidence ids;
* **MIXED** — needs both numbers and evidence;
* **2 deliberately unanswerable** questions (a reverse shell, a cryptominer
  that never appears in the log) that probe hallucination and abstention.

Each case carries gold evidence ids and the points an answer must contain, so
scoring is mechanical rather than vibes.

## Metrics

| Metric | Definition |
|---|---|
| Recall@5 | share of gold evidence ids present in the top-5 retrieved set |
| Citation accuracy | every cited id exists in the corpus **and** is in that question's retrieved set |
| Abstention accuracy | unanswerable questions must answer "not found in the logs" |
| Number accuracy | deterministic numbers must match DuckDB |
| Latency | p50 / p95 end-to-end |
| Token cost | prompt + completion tokens per question |

## Results

W4 run over all 20 questions (2026-09-19 corpus, 5,469 docs):

| Metric | Result |
|---|---|
| Recall@5 (all 20) | **79.6%** |
| Recall@5 (14 evidence/mixed) | **70.9%** |
| Citation accuracy | **100%** |
| Abstention accuracy | **100%** |
| Deterministic-number accuracy | **100%** |
| p50 / p95 latency | ≈ 2.0 s / 3.0 s |

Retrieval ablation on the same corpus:

| Retrieval mode | K=3 | K=5 | K=8 |
|---|---|---|---|
| dense only | 11% | 14% | 14% |
| BM25 + dense RRF, no rerank | 37% | 52% | 52% |
| hybrid + bge-reranker | 44% | **66%** | 66% |

After the 2026-10-11 refresh (7,689 docs) recall dropped to dense **10.8%** →
hybrid **29.5%** → +rerank **44.1%** on strict gold ids: the corpus grew 40%
and the golden set was re-grounded, while the top-5 is often filled with
equally valid equivalent evidence. See [retrieval.md](retrieval.md) for the
interpretation and the tuning log.

## Honest caveats

* One corpus, one author, one agent — the golden set is small (20 questions)
  and was written by the same person who tuned the retriever. Treat the
  absolute numbers as a regression harness, not a benchmark.
* E2E scoring needs an LLM key; `--ablation-only` needs none and is what CI can
  run cheaply.
* Recall is computed on gold ids, which understates answers that cite
  equivalent real evidence (documented in the W5 section of the README).
