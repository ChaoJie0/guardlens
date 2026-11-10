"""Dashboard smoke test: the app must render on the bundled sample corpus.

The Streamlit app executes all tabs on every run, so a hard failure in one tab
used to take the whole dashboard down. This test ingests the synthetic sample
into a temporary DuckDB and asserts the app still renders (metrics + tables)
with no LLM key and no backend running.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample" / "audit-sample.jsonl"


@pytest.fixture()
def sample_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from guardlens.ingest import run_ingestion

    db_path = tmp_path / "sample.duckdb"
    run_ingestion(SAMPLE, db_path)
    # AppTest runs the real script, so make sure it cannot pick up a stray key.
    monkeypatch.delenv("GUARDLENS_LLM_API_KEY", raising=False)
    monkeypatch.delenv("GUARDLENS_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GUARDLENS_DB", str(db_path))
    return db_path


def test_dashboard_renders_without_models_or_api(sample_db: Path) -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    app.run()

    assert not app.exception, app.exception
    # KPI metrics and the dashboard tables still render.
    assert len(app.metric) > 0
    assert len(app.dataframe) > 0
