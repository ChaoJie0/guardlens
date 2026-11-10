# GuardLens W4 离线评测报告

> 本报告全部在本地生成：语料为已脱敏的审计事件，数字来自真实评测运行，未手工编造；不含真实用户名、主机路径、内网域名/IP 或邮箱。

- 黄金集：20 题（STATS / EVIDENCE / MIXED，含 2 道无答案拒答题）
- 生成模型：`local-configured-LLM`（OpenAI 兼容端点，凭证取自环境变量，报告不含密钥）
- 检索：BGE-M3 稠密 + BM25 稀疏 RRF + bge-reranker-v2-m3（Qdrant）

## 2. 检索对照实验（Recall@5，仅检索层、无 LLM）

对比 W2 纯稠密向量 vs BM25 混合 vs 混合+rerank，以及 K=3/5/8、rerank 开关。
Recall@5 = 应命中证据进入前 5 的比例（只对有 gold 的证据/混合题统计）。

| 检索模式 | K | rerank | 题数 | Recall@5 |
|---|---|---|---|---|
| dense | 3 | — | 12 | 8.0% |
| dense | 5 | — | 12 | 10.8% |
| dense | 8 | — | 12 | 10.8% |
| hybrid | 3 | — | 12 | 22.9% |
| hybrid | 5 | — | 12 | 29.5% |
| hybrid | 8 | — | 12 | 29.5% |
| hybrid_rerank | 3 | ✅ | 12 | 30.7% |
| hybrid_rerank | 5 | ✅ | 12 | 44.1% |
| hybrid_rerank | 8 | ✅ | 12 | 43.4% |

**结论：W2 纯稠密 Recall@5=10.8% → 混合+rerank 最优 44.1%，净提升 33.3 个百分点。**

## 4. 结论与说明

- **能确定的交给确定工具**：STATS/MIXED 的数字由只读 DuckDB 给出并原样进入 prompt，LLM 只组织语言，因此数值稳定、可复现。
- **混合检索 + rerank 的价值集中在精确命令词与绕行序列**：纯稠密对 curl/chmod/同 URL 换工具等词项不敏感，BM25 精确命中 + RRF + cross-encoder 重排把这些题的召回拉起来。
- **反幻觉**：每条引用必须真实存在且在本题召回集内；两道无答案题（反弹 shell、挖矿）应答“日志中未发现”，以拒答正确率量化。
- 延迟主要由 LLM 生成主导；本地 embedding/rerank 在 Apple Silicon MPS 上运行。
