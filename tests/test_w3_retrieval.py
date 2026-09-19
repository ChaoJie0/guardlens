"""W3 retrieval/router/SQL tests without BGE-M3, Qdrant server or a reranker.

BM25 and RRF run on a tiny in-memory corpus; HybridRetriever gets an injected
fake dense function and a fake reranker so the fusion/anchor logic is covered
deterministically.
"""

from __future__ import annotations

import json

import pytest

from guardlens.bm25 import BM25Index, build_index, tokenize
from guardlens.hybrid import HybridRetriever, rrf_fuse
from guardlens.retriever import Hit
from guardlens.router import EVIDENCE, MIXED, STATS, classify
from guardlens.sql_tool import UnsafeSQLError, run_readonly, validate_select

# --------------------------------------------------------------------------
# BM25 tokenizer / index
# --------------------------------------------------------------------------


def test_tokenizer_keeps_shell_terms_and_cjk_bigrams():
    toks = tokenize("curl -A Mozilla https://x.io 网络拦截 chmod +x run.sh")
    assert "curl" in toks
    assert "chmod" in toks
    assert "run.sh" in toks
    assert "网络" in toks and "拦截" in toks  # CJK bigrams


def _write_corpus(tmp_path, docs):
    p = tmp_path / "corpus.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    return p


def _doc(eid, text, doc_type="event_card", **kw):
    return {
        "evidence_id": eid,
        "doc_type": doc_type,
        "text": text,
        "ts": kw.get("ts"),
        "tool": kw.get("tool"),
        "category": kw.get("category"),
        "decision": kw.get("decision"),
        "metadata": {},
    }


def test_bm25_anchor_command_ranks_above_context_mention(tmp_path):
    # card A: chmod is THE anchor command; card B: chmod only in context,
    # card B is shorter so naive BM25 would prefer it.
    a = _doc(
        "EV-A",
        "[证据 A]\n时间: t\n工具: bash  类别: unknown  决策: deny\n"
        "命令: chmod +x /long/path/to/run.sh && /long/path/to/run.sh arg1 arg2 arg3\n"
        "理由: blocked\n前文事件:\n  - some prior context padding line one\n",
        tool="bash",
        category="unknown",
        decision="deny",
    )
    b = _doc(
        "EV-B",
        "[证据 B]\n时间: t\n工具: send_message  类别: unknown  决策: deny\n"
        "命令: send_message x\n理由: unknown tool\n前文事件:\n"
        "  - tool=bash | cmd=chmod 777 /etc/hosts padding padding\n",
        tool="send_message",
        category="unknown",
        decision="deny",
    )
    p = _write_corpus(tmp_path, [a, b])
    idx: BM25Index = build_index(p)
    ids = [eid for eid, _ in idx.search("chmod", k=5)]
    assert ids[0] == "EV-A"  # anchor command wins over a context mention


# --------------------------------------------------------------------------
# RRF
# --------------------------------------------------------------------------


def test_rrf_fuses_and_ranks_reciprocal():
    dense = ["a", "b", "c"]
    sparse = ["c", "d", "a"]
    fused = rrf_fuse([dense, sparse])
    ids = [i for i, _ in fused]
    # items in BOTH lists ('a','c') outrank single-list 'b'/'d'; under these
    # symmetric ranks a and c tie exactly, so assert the tier rather than order.
    assert set(ids[:2]) == {"a", "c"}
    assert set(ids) == {"a", "b", "c", "d"}
    assert len(ids) <= 20


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question,kind",
    [
        ("整体拦截率是多少？", STATS),
        ("越界写一共几起？", STATS),
        ("curl 被拦是为什么？", EVIDENCE),
        ("是否使用过 chmod 提权？", EVIDENCE),
        ("fail-closed 有几起，分别是什么场景？", MIXED),
    ],
)
def test_router_classification(question, kind):
    assert classify(question).kind == kind


# --------------------------------------------------------------------------
# SQL guardrails
# --------------------------------------------------------------------------


def test_sql_allows_select_and_blocks_writes():
    assert validate_select("WITH t AS (SELECT 1 x) SELECT * FROM t")
    with pytest.raises(UnsafeSQLError):
        validate_select("DELETE FROM events")
    with pytest.raises(UnsafeSQLError):
        validate_select("SELECT 1; DROP TABLE events")
    with pytest.raises(UnsafeSQLError):
        validate_select("UPDATE events SET decision='allow'")
    with pytest.raises(UnsafeSQLError):
        validate_select("ATTACH '/tmp/x.duckdb'")


def test_run_readonly_executes_on_real_db(tmp_path):
    import duckdb

    db = tmp_path / "e.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE events (decision VARCHAR, category VARCHAR)")
    con.executemany(
        "INSERT INTO events VALUES (?, ?)",
        [("deny", "network"), ("deny", "file_write"), ("allow", "read")],
    )
    con.close()
    res = run_readonly("SELECT decision, COUNT(*) c FROM events GROUP BY 1 ORDER BY c DESC", db)
    assert res.columns == ["decision", "c"]
    assert ("deny", 2) in res.rows


# --------------------------------------------------------------------------
# Hybrid retrieval with injected dense + identity fake reranker
# --------------------------------------------------------------------------


class _IdentityReranker:
    device = "cpu"

    def rerank(self, query, hits, top_k=5):
        return hits[:top_k]


def _hit(doc, score=0.1):
    return Hit(
        evidence_id=doc["evidence_id"],
        score=score,
        doc_type=doc["doc_type"],
        text=doc["text"],
        ts=doc.get("ts"),
        tool=doc.get("tool"),
        category=doc.get("category"),
        decision=doc.get("decision"),
        metadata={},
    )


def test_hybrid_anchors_failclosed_and_real_chmod(tmp_path):
    docs = [
        _doc(
            "EV-FC1",
            "[证据]\n时间: t\n工具: write  类别: fail-closed  决策: deny\n"
            "命令: write f\n理由: no rollback net\n前文事件:\n  - ctx\n",
            tool="write",
            category="fail-closed",
            decision="deny",
        ),
        _doc(
            "EV-CH",
            "[证据]\n时间: t\n工具: bash  类别: unknown  决策: deny\n"
            "命令: chmod +x deploy.sh && ./deploy.sh with extra padding tokens here\n"
            "理由: blocked\n前文事件:\n  - ctx padding padding\n",
            tool="bash",
            category="unknown",
            decision="deny",
        ),
        _doc(
            "EV-NET",
            "[证据]\n时间: t\n工具: bash  类别: network  决策: deny\n"
            "命令: curl https://example.com\n理由: blocked\n前文事件:\n  - c\n",
            tool="bash",
            category="network",
            decision="deny",
        ),
        _doc(
            "RP-1",
            "## 汇总\n网络与权限拦截的周报，泛泛而谈，不含具体命令。" * 3,
            doc_type="report_chunk",
            category="turn_report",
        ),
    ]
    p = _write_corpus(tmp_path, docs)
    by_id = {d["evidence_id"]: d for d in docs}

    def fake_dense(question, n, where=None):
        # honour payload filter like Qdrant would (fail-closed anchor path)
        if where and where.get("category") == "fail-closed":
            return [_hit(by_id["EV-FC1"], 0.95)]
        # otherwise dense weakly prefers reports (mimics W2 failure mode)
        return [_hit(by_id["RP-1"], 0.9), _hit(by_id["EV-NET"], 0.5)]

    r = HybridRetriever(
        p,
        embedder=object(),
        dense_fn=fake_dense,
        reranker=_IdentityReranker(),
        client=False,
    )
    fc = [h.evidence_id for h in r.retrieve("fail-closed 发生在什么场景", k=5)]
    assert "EV-FC1" in fc  # exact payload anchor beats dense report bias

    ch = [h.evidence_id for h in r.retrieve("是否用过 chmod 提权", k=5)]
    assert "EV-CH" in ch  # primary-command anchor surfaces the real chmod card


def test_hybrid_sparse_finds_exact_command_dense_misses(tmp_path):
    docs = [
        _doc(
            "EV-NODE",
            "[证据]\n时间: t\n工具: bash  类别: network  决策: deny\n"
            "命令: node -e \"fetch('https://registry.npmjs.org/prisma')\"\n"
            "理由: static-script-egress\n前文事件:\n  - padding line\n",
            tool="bash",
            category="network",
            decision="deny",
        ),
        _doc(
            "RP-9",
            "通用的开发周报内容，谈论构建与测试，但没有具体的 fetch 命令。" * 4,
            doc_type="report_chunk",
            category="turn_report",
        ),
    ]
    p = _write_corpus(tmp_path, docs)
    by_id = {d["evidence_id"]: d for d in docs}

    def fake_dense(question, n, where=None):
        return [_hit(by_id["RP-9"], 0.9)]  # dense completely misses the event

    r = HybridRetriever(
        p, embedder=object(), dense_fn=fake_dense, reranker=_IdentityReranker(), client=False
    )
    ids = [h.evidence_id for h in r.retrieve("node -e fetch npm registry 拦截", k=5)]
    assert "EV-NODE" in ids  # sparse exact-term path recovers it
