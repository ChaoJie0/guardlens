"""Tiny, dependency-free BM25 over the corpus for exact-term sparse retrieval.

Why not the ``rank_bm25`` package: audit text mixes Chinese prose with shell
commands, tool names, paths and URLs. We need a tokenizer that keeps commands
and flags as precise terms while still indexing CJK by overlapping bigrams.
This module implements BM25 (Okapi, k1/b) over that tokenizer.

The sparse index is built once from corpus.jsonl and held in memory (a few MB
for ~5.5k docs). W3 uses it fused with dense results via RRF; Qdrant native
sparse vectors are a W4 refinement.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_ASCII_TOKEN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-/+:=@?&#%]{1,}")
_PATHISH = re.compile(r"[A-Za-z0-9_][\w.\-/+~]*(?:/[\w.\-]+)+")
_CJK = re.compile(r"[一-鿿]")


def tokenize(text: str) -> list[str]:
    """Tokens suited to audit search.

    * shell words / flags / versions / URLs kept as single tokens
      (``curl``, ``chmod``, ``--max-time``, ``web_fetch``, ``example.com``);
    * file paths kept whole (``/tmp/x.sh``) plus their basename;
    * CJK runs -> overlapping bigrams (``网络拦截`` -> ``网络``, ``络拦``,
      ``拦截``) so short Chinese queries match without an external segmenter.
    """
    text = text.lower()
    tokens: list[str] = []

    for m in _PATHISH.finditer(text):
        tokens.append(m.group(0))
        base = m.group(0).rstrip("/").split("/")[-1]
        if base and len(base) > 1:
            tokens.append(base)

    for m in _ASCII_TOKEN.finditer(text):
        tok = m.group(0)
        tokens.append(tok)
        # snake/kebab tool names also indexed by their parts, e.g. web_fetch
        parts = re.split(r"[_.\-/:=]+", tok)
        tokens.extend(p for p in parts if p)

    for run in re.findall(r"[一-鿿]+", text):
        tokens.append(run)
        tokens.extend(run[i : i + 2] for i in range(max(0, len(run) - 1)))

    return tokens


# Audit cards lead with the anchor event (time/tool/command/reason); the
# before/after context starts at this marker. Terms in the anchor body are
# weighted above the same term appearing only in surrounding context.
_ANCHOR_SPLIT = re.compile(r"\n(前文事件|后文事件|相关回合|上下文)")
ANCHOR_WEIGHT = 2.5


def _anchor_text(text: str) -> str:
    m = _ANCHOR_SPLIT.search(text)
    return text[: m.start()] if m else text


@dataclass(slots=True)
class BM25Index:
    docs: list[dict]
    doc_tokens: list[list[str]]
    doc_freq: dict[str, int]
    term_freq: list[Counter]
    anchor_freq: list[Counter]
    doc_len: list[int]
    avg_len: float
    k1: float = 1.5
    b: float = 0.75

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        q_terms = tokenize(query)
        n = len(self.docs)
        scores: dict[str, float] = {}
        for term in set(q_terms):
            df = self.doc_freq.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for doc_id, tf in self._postings(term):
                dl = self.doc_len[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_len)
                score = idf * (tf * (self.k1 + 1)) / denom
                # boost when the query term is on the anchor event itself
                if self.anchor_freq[doc_id].get(term):
                    score *= ANCHOR_WEIGHT
                scores[self.docs[doc_id]["evidence_id"]] = (
                    scores.get(self.docs[doc_id]["evidence_id"], 0.0) + score
                )
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:k]

    def _postings(self, term: str):
        # Small corpus: linear scan is fast and keeps the index trivial.
        for doc_id, tf in enumerate(self.term_freq):
            c = tf.get(term)
            if c:
                yield doc_id, c


def build_index(corpus_path: str | Path) -> BM25Index:
    docs: list[dict] = []
    with Path(corpus_path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    doc_tokens = [tokenize(d["text"]) for d in docs]
    anchor_freq = [Counter(tokenize(_anchor_text(d["text"]))) for d in docs]
    doc_freq: dict[str, int] = {}
    term_freq: list[Counter] = []
    for toks in doc_tokens:
        c = Counter(toks)
        term_freq.append(c)
        for term in c:
            doc_freq[term] = doc_freq.get(term, 0) + 1
    doc_len = [len(t) for t in doc_tokens]
    avg_len = sum(doc_len) / len(doc_len) if doc_len else 1.0
    return BM25Index(
        docs=docs,
        doc_tokens=doc_tokens,
        doc_freq=doc_freq,
        term_freq=term_freq,
        anchor_freq=anchor_freq,
        doc_len=doc_len,
        avg_len=avg_len,
    )
