"""bge-reranker-v2-m3 cross-encoder reranking (W3), MPS-aware.

Runs on Apple Metal/MPS with a CPU fallback. Measured on M5: ~10 pairs/s on
MPS vs ~4 pairs/s on CPU; reranking the fused top-20 is ~2-3s.
"""

from __future__ import annotations

import torch  # noqa: E402
from sentence_transformers import CrossEncoder  # noqa: E402

from .retriever import Hit  # noqa: E402
from .vectorstore import _ensure_model_cached  # noqa: E402

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_MAX_LEN = 512


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Reranker:
    def __init__(
        self,
        model_name: str = RERANK_MODEL,
        *,
        device: str | None = None,
        max_length: int = RERANK_MAX_LEN,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.requested = device
        self.device = "cpu"
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = self._load()
        return self._model

    def _load(self) -> CrossEncoder:
        _ensure_model_cached(self.model_name)
        target = self.requested or _pick_device()
        model = CrossEncoder(self.model_name, device="cpu", max_length=self.max_length)
        if target in ("mps", "cuda"):
            try:
                model.model.to(target)
                model.predict([("probe", "probe")], batch_size=1)
                if target == "mps":
                    torch.mps.synchronize()
                self.device = target
            except Exception as exc:  # noqa: BLE001
                print(f"[reranker] {target} probe failed ({exc}); using CPU")
                model.model.to("cpu")
                self.device = "cpu"
        else:
            self.device = "cpu"
        print(f"[reranker] {self.model_name} on {self.device} (max_len={self.max_length})")
        return model

    def rerank(
        self,
        query: str,
        hits: list[Hit],
        *,
        top_k: int = 5,
        batch_size: int = 32,
    ) -> list[Hit]:
        if not hits:
            return []
        pairs = [(query, h.text) for h in hits]
        scores = self.model.predict(pairs, batch_size=batch_size)
        if self.device == "mps":
            torch.mps.synchronize()
        scored = sorted(zip(hits, scores, strict=True), key=lambda x: float(x[1]), reverse=True)
        out: list[Hit] = []
        for hit, score in scored[:top_k]:
            hit.score = round(float(score), 4)  # cross-encoder relevance logit
            out.append(hit)
        return out
