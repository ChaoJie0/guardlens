"""Device selection logic (no model download: only torch device probing)."""

from __future__ import annotations

import torch

from guardlens.vectorstore import pick_device


def test_pick_device_returns_known_device():
    dev = pick_device("mps")
    assert dev in ("mps", "cpu")
    # If torch reports MPS available and the probe tensor succeeds, we must
    # actually pick mps on this machine; otherwise CPU fallback is correct.
    if torch.backends.mps.is_available():
        try:
            _ = torch.tensor([1.0], device="mps") * 2
            assert dev == "mps"
        except RuntimeError:
            assert dev == "cpu"


def test_explicit_cpu_is_honored():
    assert pick_device("cpu") == "cpu"
