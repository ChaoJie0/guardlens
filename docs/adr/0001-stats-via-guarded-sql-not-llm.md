# ADR 0001 — 数值类问题走受保护的 SQL 工具，而不是让 LLM 生成/估算数字

- 状态：accepted（W3, v0.3-prod）
- 日期：2026-09-19
- 决策者：GuardLens

## 背景

审计分析台要回答两类截然不同的问题：

1. **STATS（确定取数）**：拦截率多少、越界写几起、fail-closed 几次。答案是
   DuckDB 里一个确定的聚合值。
2. **EVIDENCE（证据研判）**：curl 为什么被拦、有没有绕过尝试。需要语义检索 +
   对证据的归纳，LLM 在这里有价值。

W2 只有证据管线，所有问题都进向量检索 + LLM。数值类问题因此依赖模型“记得”或
估算，存在编造数字的风险——这在安全运营场景不可接受。

## 决策

加一个确定性 Query Router（`router.py`，规则分类，不额外调用 LLM），把问题分为
`STATS / EVIDENCE / MIXED`：

- STATS → DuckDB 只读取数（`sql_tool.py`），结果**原样**进 prompt，LLM 只负责
  把数字组织成一句话；
- EVIDENCE → BM25+稠密 RRF + bge-reranker（`hybrid.py`）；
- MIXED → 先取数再检索，两路结果都给 LLM。

自由形态的文本→SQL 即使接入，也必须经过**三道闸**：

1. **解析闸**：只允许单条 `SELECT/WITH`；拒绝多语句、`INSERT/UPDATE/DELETE/DDL/
   ATTACH/PRAGMA/COPY` 等一切非只读关键字；
2. **沙箱闸**：把 DuckDB 文件复制到临时目录，以 `read_only=True` 打开——即使
   漏网一条写语句，DuckDB 自身也拒绝写入，且碰不到生产库；
3. **dry-run + 行限**：先以 `LIMIT 0` 建临时视图验证计划可执行，再以
   `LIMIT max_rows+1` 有界执行并标记截断。

审计域常见统计（拦截率、各类别拦截数、fail-closed 数）由 `known_stats()`
**无 LLM** 直接给出，自由 SQL 仅作为受控扩展。

## 权衡

- **收益**：数字可复现、零幻觉、可审计；SQL 失败不影响证据管线；只读副本让
  “模型生成 SQL”这一攻击面被数据库权限兜底。
- **代价**：确定性 `known_stats` 需要为常见问法维护少量规则；自由 SQL 的表达力
  被关键字白名单限制（有意为之）。
- **不选的方案**：① 直接让 LLM 输出数字——不可复现、会编造；② 给 LLM 连可写
  库——违反最小权限；③ 上一套重型 text2SQL 框架——2.2 万行、5 个工具的规模不
  需要，薄编排更可讲清。

> 一句话原则：**能用确定工具回答的，绝不让模型自由生成。**
