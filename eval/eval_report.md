# GuardLens W4 离线评测报告

> 本报告全部在本地生成：语料为已脱敏的审计事件，数字来自真实评测运行，未手工编造；不含真实用户名、主机路径、内网域名/IP 或邮箱。

- 黄金集：20 题（STATS / EVIDENCE / MIXED，含 2 道无答案拒答题）
- 生成模型：`deepseek-flash`（OpenAI 兼容端点，凭证取自环境变量，报告不含密钥）
- 检索：BGE-M3 稠密 + BM25 稀疏 RRF + bge-reranker-v2-m3（Qdrant）

## 1. 端到端总览（混合检索 + rerank，K=5）

> 口径：本节 Recall@5 对全部 20 题取均值；STATS 题无 gold 文档，记为 1（确定性取数）。纯检索层、仅证据/混合题的 Recall@5 见第 2 节对照实验。

| 指标 | 结果 |
|---|---|
| 题目数 | 20 |
| **Recall@5（20 题均值，STATS 记满分）** | **79.6%** |
| Recall@5（仅证据/混合题，14 题） | 70.9% |
| **引用正确率**（真实存在且在召回集内） | **100.0%** |
| **拒答正确率**（该拒答则拒答、否则不拒答） | **100.0%** |
| 数值正确率（STATS/MIXED 数字原样命中） | 100.0% |
| 延迟 p50 / p95 | 2.01s / 2.95s |
| 延迟均值 | 1.92s |
| token（prompt / completion 总量） | 84673 / 5794 |
| 每问 token（prompt / completion 均值） | 4233.6 / 289.7 |

## 2. 检索对照实验（Recall@5，仅检索层、无 LLM）

对比 W2 纯稠密向量 vs BM25 混合 vs 混合+rerank，以及 K=3/5/8、rerank 开关。
Recall@5 = 应命中证据进入前 5 的比例（只对有 gold 的证据/混合题统计）。

| 检索模式 | K | rerank | 题数 | Recall@5 |
|---|---|---|---|---|
| dense | 3 | — | 12 | 11.0% |
| dense | 5 | — | 12 | 14.3% |
| dense | 8 | — | 12 | 14.3% |
| hybrid | 3 | — | 12 | 37.0% |
| hybrid | 5 | — | 12 | 52.2% |
| hybrid | 8 | — | 12 | 52.2% |
| hybrid_rerank | 3 | ✅ | 12 | 44.1% |
| hybrid_rerank | 5 | ✅ | 12 | 66.0% |
| hybrid_rerank | 8 | ✅ | 12 | 66.0% |

**结论：W2 纯稠密 Recall@5=14.3% → 混合+rerank 最优 66.0%，净提升 51.7 个百分点。**

## 3. 逐题结果（脱敏，仅证据 id / 指标）

| 题号 | 类型 | 路由 | Recall@5 | 引用合法 | 拒答正确 | 数值正确 | 延迟s |
|---|---|---|---|---|---|---|---|
| q01 | STATS | STATS | 100% | ✅ | ✅ | ✅ | 1.18 |
| q02 | STATS | STATS | 100% | ✅ | ✅ | ✅ | 0.9 |
| q03 | STATS | STATS | 100% | ✅ | ✅ | ✅ | 0.9 |
| q04 | STATS | STATS | 100% | ✅ | ✅ | ✅ | 0.86 |
| q05 | STATS | STATS | 100% | ✅ | ✅ | — | 1.73 |
| q06 | STATS | STATS | 100% | ✅ | ✅ | ✅ | 1.08 |
| q07 | EVIDENCE | EVIDENCE | 80% | ✅ | ✅ | — | 2.03 |
| q08 | EVIDENCE | EVIDENCE | 50% | ✅ | ✅ | — | 2.52 |
| q09 | EVIDENCE | EVIDENCE | 100% | ✅ | ✅ | — | 2.21 |
| q10 | EVIDENCE | EVIDENCE | 75% | ✅ | ✅ | — | 2.27 |
| q11 | EVIDENCE | EVIDENCE | 40% | ✅ | ✅ | — | 1.67 |
| q12 | EVIDENCE | EVIDENCE | 40% | ✅ | ✅ | — | 3.12 |
| q13 | EVIDENCE | EVIDENCE | 75% | ✅ | ✅ | — | 2.0 |
| q14 | EVIDENCE | EVIDENCE | 50% | ✅ | ✅ | — | 1.82 |
| q15 | EVIDENCE | EVIDENCE | 100% | ✅ | ✅ | — | 2.54 |
| q16 | EVIDENCE | EVIDENCE | 57% | ✅ | ✅ | — | 2.55 |
| q17 | MIXED | MIXED | 25% | ✅ | ✅ | ✅ | 2.3 |
| q18 | MIXED | MIXED | 100% | ✅ | ✅ | ✅ | 2.95 |
| q19 | EVIDENCE | EVIDENCE | 100% | ✅ | ✅ | — | 2.01 |
| q20 | EVIDENCE | EVIDENCE | 100% | ✅ | ✅ | — | 1.76 |

## 4. 结论与说明

- **能确定的交给确定工具**：STATS/MIXED 的数字由只读 DuckDB 给出并原样进入 prompt，LLM 只组织语言，因此数值稳定、可复现。
- **混合检索 + rerank 的价值集中在精确命令词与绕行序列**：纯稠密对 curl/chmod/同 URL 换工具等词项不敏感，BM25 精确命中 + RRF + cross-encoder 重排把这些题的召回拉起来。
- **反幻觉**：每条引用必须真实存在且在本题召回集内；两道无答案题（反弹 shell、挖矿）应答“日志中未发现”，以拒答正确率量化。
- 延迟主要由 LLM 生成主导；本地 embedding/rerank 在 Apple Silicon MPS 上运行。
