# Contributing to GuardLens

Thanks for taking a look. This project is a single-author research/engineering
build, so the fastest way to get a change merged is to keep it small, tested
and aligned with the design rules below.

## Setup

```bash
uv sync                 # Python 3.12 venv + all dependencies
uv run pytest           # unit tests, no network / no model downloads
uv run ruff check .     # lint
uv run ruff format .    # format (formatting is part of CI)
```

Notes:

* The bundled sample (`data/sample/audit-sample.jsonl`) is synthetic and is the
  only audit data that may be committed. Real logs, DuckDB files and vector
  stores are git-ignored — please keep it that way.
* `uv.lock` is regenerated against your local index. If you run `uv lock`
  behind a mirror, do not commit the mirror-pinned file without saying so in
  your PR; CI resolves from the default PyPI index.
* Full retrieval (`build_index`, `migrate_qdrant`) needs Docker and downloads
  ~2 GB of models — most contributions do not need it.

## Design rules (please do not break these)

1. **Numbers come from SQL, never from the model.** Any change touching
   `router.py`, `sql_tool.py` or `orchestrator.py` must keep the
   "deterministic numbers passed verbatim" property.
2. **Every claim is cited, and citations are verified.** A cited id must exist
   in the corpus *and* be in that question's retrieved set.
3. **Abstain rather than guess.** Missing evidence yields the "not found in the
   logs" answer.
4. **Read-only SQL only.** `SELECT`/`WITH`, a copied read-only database, and a
   row cap. No DDL/DML, no pragmas.
5. **No new outbound calls.** Retrieval, embedding, analytics and evaluation
   stay local; the only network dependency is the LLM endpoint the user
   configures.
6. **No secrets, no personal paths, no real log lines** in code, tests, docs or
   fixtures. Use `example.com` / `/Users/user` / RFC 2606 names.

## Tests

* Add a unit test for every behaviour change; tests must not require network
  access, model downloads, Docker or an API key.
* Retrieval changes should come with a note on what the ablation shows
  (`uv run python -m eval.run_eval --ablation-only`); golden-set edits need a
  reason in the PR description.

## Commits and pull requests

* Conventional-commit style subjects (`feat:`, `fix:`, `docs:`, `test:`,
  `refactor:`, `perf:`), imperative mood, ≤72 characters.
* One logical change per PR. CI must be green (`ruff check`, `ruff format
  --check`, `pytest`).
* Describe **what changed and why**, plus the trade-off you accepted. Numbers
  in the description must be reproducible with a command.

## Reporting bugs

Include: what you ran, what you expected, what happened, and the smallest
snippet of (redacted) input that reproduces it. For anything security-related,
use [SECURITY.md](SECURITY.md) instead of a public issue.

## Good first issues

* Improve `_REASON_BUCKETS` / risk-family mapping for a different gate's reason
  vocabulary.
* Add golden-set questions for answer paths that are currently thin.
* Streamline the first-run experience (model download progress, clearer errors
  when Qdrant is not running).
