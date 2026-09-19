# GuardLens 🛡️

**RAG-native security analytics for autonomous Agent audit logs.**

GuardLens turns the JSONL audit trail produced by [agent-gate](https://github.com/ChaoJie0/agent-gate) — a zero-trust permission gate for AI coding agents — into a queryable, measurable dataset.

> W1 milestone (current): **data foundation + metrics layer** — streaming JSONL ingestion, pydantic validation, shell-command feature extraction, DuckDB persistence, baseline diffing, and a static Streamlit dashboard.
> Planned W2–W4: hybrid retrieval (BM25 + embeddings), LLM evidence-grounded Q&A, evasion-pattern detection, and eval-driven tuning.

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

# 3. open the dashboard
uv run streamlit run app.py
```

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
