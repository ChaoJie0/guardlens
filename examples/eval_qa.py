"""W2 acceptance: run the 5 evidence questions end-to-end and dump results."""

from __future__ import annotations

import duckdb

from guardlens.qa import make_engine

con = duckdb.connect("data/derived/guardlens.duckdb", read_only=True)
total, allow_n, deny_n, fc = con.execute(
    "SELECT COUNT(*), SUM(decision='allow'), SUM(decision='deny'), "
    "SUM(category='fail-closed') FROM events"
).fetchone()
cat = con.execute(
    "SELECT category, COUNT(*) FROM events WHERE decision='deny' GROUP BY 1 ORDER BY 2 DESC"
).fetchall()
con.close()
STATS = (
    f"事件总数={total}；allow={allow_n}；deny={deny_n}；"
    f"拦截率={deny_n / total:.4f}；fail-closed={fc}\n"
    "拦截按类别: " + ", ".join(f"{c}={n}" for c, n in cat)
)

QUESTIONS = [
    "curl 被拦是为什么？请引用具体事件说明拦截理由。",
    "有没有绕过网络限制的尝试（比如 curl 被拦后换别的工具）？",
    "fail-closed 发生在什么场景？涉及哪些工具？",
    "写文件被拦截主要是什么原因？",
    "是否使用过 chmod 或 sudo 这类提权操作？结果如何？",
]

eng = make_engine("data/derived/chroma", k=6)
for i, q in enumerate(QUESTIONS, 1):
    print("=" * 78)
    print(f"Q{i}: {q}")
    chunks = []
    hits = []
    for ev in eng.stream(q, stats=STATS):
        if ev["type"] == "hit":
            hits.append(ev["hit"])
        elif ev["type"] == "token":
            chunks.append(ev["delta"])
        elif ev["type"] == "final":
            ans = ev["answer"]
    print(f"[retrieved] {[h.evidence_id for h in hits]}")
    print("-" * 78)
    print("".join(chunks).strip())
    print("-" * 78)
    print(
        f"[check] valid={ans.valid_citations} invalid={ans.invalid_citations} "
        f"abstained={ans.abstained}"
    )
