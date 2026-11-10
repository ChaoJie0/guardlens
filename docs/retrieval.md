# Retrieval

GuardLens retrieves evidence with three stages: two independent first-stage
retrievers, RRF fusion with per-channel type balancing, then a cross-encoder
rerank.

## Why hybrid

Security questions mix two vocabularies:

* **exact terms** — `curl`, `chmod`, `/Users/x/.ssh/id_rsa`, a URL, a host name.
  Dense vectors blur these; exact matching must not.
* **intent** — "绕过" (evasion), "越界" (out of sandbox), "为什么会拦".
  These need semantics; BM25 alone cannot.

So both channels run, and fusion happens on ranked lists rather than scores.

## Components

* **Dense** — BGE-M3 (1024-d) in Qdrant; payload carries
  `ts / tool / category / decision / doc_type / evidence_id` for server-side
  pre-filtering.
* **Sparse** — a dependency-free BM25 (`bm25.py`) whose tokenizer keeps shell
  commands, flags, paths and URLs as exact terms, and indexes Chinese as
  overlapping bigrams. Anchor-body terms are boosted.
* **Fusion** — per-channel **type balancing** before **RRF (k=60)**: report
  chunks outnumber event cards ~3.4×, so without balancing RRF fills the pool
  with near-duplicate report text.
* **Rerank** — `bge-reranker-v2-m3` cross-encodes the top-20 pool (≈10 pairs/s
  on Apple MPS vs ≈4 on CPU) and produces the final top-5.

## Type balancing and anchors

* Three-way balance reserves a quota for `event_card` and `event_cluster`
  before "others" (report chunks).
* Intent expansions bridge abstract phrasings to concrete command cards
  (e.g. 绕过 → `curl ⇄ web_fetch same URL`), and the reranker scores the
  question *plus* expansions.
* Structured anchors (exact metadata/keyword filters for `curl`, `web_fetch`,
  `urllib`, `Mozilla` UA, `npm`, `node`, `chmod`, `sudo`) guarantee that
  rare-but-real events reach the pool.
* Rerank safety rules stop low-scoring keyword anchors and the top-RRF
  fallback from being pinned above genuinely relevant hits.

Rationale and trade-offs: [ADR 0002](adr/0002-hybrid-retrieval-type-balance-and-rerank.md).

## Ablation

Recall@5 on the golden set, retrieval only (no LLM calls):

| Retrieval mode | K=3 | K=5 | K=8 |
|---|---|---|---|
| dense vectors only | 11% | 14% | 14% |
| BM25 + dense RRF, no rerank | 37% | 52% | 52% |
| hybrid + bge-reranker | 44% | **66%** | 66% |

Measured on the 5,469-doc corpus (2026-09-19 snapshot). K=5 is the sweet spot:
K=3 misses, K=8 adds nothing — "retrieve wide, rerank strict, answer from five".

After the full dataset refresh (7,689 docs, 2026-10-11) the same grid measured
dense **10.8%** → hybrid **29.5%** → hybrid+rerank **44.1%**. Strict gold-id
recall understates the system there: for multi-gold questions the top-5 is
usually filled with *equivalent* real evidence (other cards of the same deny
class, same-session clusters, the same event in report form), which
LLM-grounded answers cite just as correctly.

Reproduce:

```bash
uv run python -m eval.run_eval --ablation-only
uv run python -m eval.diagnose_recall   # where each gold id lands
uv run python -m eval.trace_recall      # per-stage D/B/R/5 markers
```

## Tuning log (2026-10-11 refresh)

* wider channels (`DENSE_N`/`SPARSE_N` = 60);
* three-way type balance with a reserved cluster quota;
* per-channel top-2 boosts (deduped, capped at 12) so a single-channel strong
  match survives RRF dilution;
* expanded structured anchors (curl / web_fetch / urllib / Mozilla / npm / node);
* rerank safety: low-scoring keyword anchors and the top-RRF fallback can no
  longer outrank genuinely relevant hits;
* rerank against the question **plus** intent expansions (abstract phrasings
  under-score cluster text, while the concrete expansion lifts the same
  clusters to the top);
* per-channel best-cluster boosting, plus a guaranteed cluster slot for
  sequence-intent questions, so the LLM sees session-level evidence.
