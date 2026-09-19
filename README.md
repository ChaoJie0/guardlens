# GuardLens 🛡️

**RAG-native security analytics for autonomous Agent audit logs.**

GuardLens turns the JSONL audit trail produced by [agent-gate](https://github.com/ChaoJie0/agent-gate) — a zero-trust permission gate for AI coding agents — into a queryable, measurable dataset.

> W1: **data foundation + metrics layer** — streaming JSONL ingestion, pydantic validation, shell-command feature extraction, DuckDB persistence, baseline diffing, and a static Streamlit dashboard.
> W2 (current): **first RAG loop** — event semantic cards, local BGE-M3 embeddings in Chroma, and evidence-grounded streaming Q&A with mandatory citations.
> Planned W3–W4: hybrid retrieval (BM25 + embeddings + rerank), Qdrant migration, evasion-pattern detection, and eval-driven tuning.

## Why this shape

Security numbers must be exact, but security *triage* is semantic. GuardLens keeps them separate:

1. **Structured layer** — pandas/DuckDB computes counts and trends (no LLM near numbers).
2. **Semantic layer (W2+)** — hybrid RAG over deny events, command sequences, and turn reports.
3. **Grounded generation (W3+)** — the LLM writes judgements only from cited evidence; it never invents metrics.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) (Python 3.12+).

```bash
# 1. install deps (creates .venv automatically)
uv sync

# 2. ingest the real audit log into DuckDB (full reload, bad rows -> quarantine)
uv run python -m guardlens.ingest \
  --input "/path/to/audit.jsonl" \
  --db data/derived/guardlens.duckdb

# 3. open the dashboard (W1 metrics + W2 Q&A tabs)
uv run streamlit run app.py
```

## W2 — evidence-grounded RAG Q&A

```bash
# 1. build semantic cards/clusters + report chunks, embed with local BGE-M3,
#    and persist into Chroma (first run downloads the ~2GB model; on Apple
#    Silicon it auto-uses the Metal/MPS GPU: ~7 min for 5.5k docs at seq384,
#    vs ~22 min on CPU). HF mirror is used by default.
uv run python -m guardlens.build_index \
  --db data/derived/guardlens.duckdb \
  --reports "/path/to/turn-reports"

# 2. ask an evidence question headlessly (prints streaming answer + citations)
uv run python examples/eval_qa.py
```

Then open the **💬 问答（W2 RAG）** tab in Streamlit.

**Pipeline.** `corpus.py` builds three document types from DuckDB + the
plugin's `turn-*.md` reports:

* `event_card` — every deny / fail-closed / high-risk-allow event with its
  **±5 surrounding events**, so an interception is understood in context
  (e.g. `curl` denied → `web_fetch` to the same URL two seconds later);
* `event_cluster` — sessions split by a 10-minute idle gap that contain
  at least one deny/fail-closed event;
* `report_chunk` — turn reports split by H2 section.

`vectorstore.py` embeds them locally with **BGE-M3** (1024-d; embedding input
capped at 384 tokens for speed while full text is still stored for the LLM)
into Chroma (cosine). It auto-selects **Apple Metal/MPS** when available
(`device="mps"` with a real-encode probe, falling back to CPU; override with
the Embedder `device=` arg), uses batch 64 on GPU, and—if the model is cached—
runs HuggingFace in offline mode to skip the slow huggingface.co HEAD retry.
`retriever.py` does dense
top-k with two W2-only pragmatic fixes (W3 adds BM25 + rerank): a per-type
quota so concrete event cards aren't drowned by near-duplicate report chunks,
and a **structured fallback** that anchors rare-but-real events (the 5
fail-closed rows, `chmod`/`sudo` attempts) via metadata/keyword filters.

**LLM.** No key is hard-coded. `llm_config.py` resolves an OpenAI-compatible
endpoint from, in order: `GUARDLENS_LLM_*` env, `DEEPSEEK_API_KEY` (legacy-harness
credentials → api.deepseek.com), or `VOLC_ARK_API_KEY` (Volcengine Ark coding
endpoint). The Q&A prompt enforces: every claim cites a retrieved evidence id,
numbers come from injected DuckDB `<stats>` (never invented), and insufficient
evidence → “日志中未发现”. `qa.py` post-verifies every citation; SSE-style
token streaming is exposed via `QAEngine.stream`.

The ingest command prints core metrics and a **baseline diff** against a
frozen 2026-09-19 snapshot of the author's corpus (21,754 events / 20,629
allow / 1,125 deny / 5 fail-closed). Because the audit log is append-only
and the autonomous agent keeps running, volume metrics are compared with
**monotonic semantics** (`actual >= baseline` → healthy `OK+`; a smaller
number means data loss), while the structural `fail-closed = 5` invariant
must still match exactly. Counts are always recomputed — the baseline is
never used to fabricate results, and an unhealthy diff exits non-zero.

Latest local run (live corpus): **21,977 events / 20,839 allow / 1,138 deny
(5.18%) / 5 fail-closed / 0 quarantined**, ingested in ~6s.

**Dirty-data handling**: ~15 upstream rows wrote `args` as a double-encoded
or truncated JSON string. The loader first tries strict `json.loads`, then a
lax recovery (best-effort key extraction, raw payload preserved under
`args._raw_args`); only rows failing the core schema are quarantined with
their line number.

### Build a public, redacted sample

```bash
uv run python -m guardlens.ingest --redact \
  --input  "/path/to/audit.jsonl" \
  --db     data/derived/guardlens-public.duckdb
```

`--redact` rewrites `/Users/<name>` → `/Users/user` and scrubs emails, IPv4
addresses, and token-ish assignments in `reason`/`args`/`targetPaths`.

## Development

```bash
uv run pytest        # unit tests
uv run ruff check .  # lint
uv run ruff format . # format
```

## Data source & privacy

- **Source**: `audit.jsonl` written by agent-gate, one JSON object per line:
  `ts, tool, category, decision, reason, targetPaths, args, snapshotId?`.
- Raw logs and derived DuckDB/parquet files are **git-ignored**; they never
  leave the machine via this repo. Only a redacted sample may be published,
  and the `--redact` path exists for that purpose.
- All command-line parsing is static (`shlex`, no execution).

## License

MIT — see [LICENSE](LICENSE).
