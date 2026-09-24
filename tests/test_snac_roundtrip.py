"""Phase-4 SNAC compile tests: interleave parity with the reference + optional real codec."""
from __future__ import annotations

import importlib.util
import os
import random
from pathlib import Path

import pytest

from indic_speak_ft.data.compile import codes_to_tokens, num_frames
from indic_speak_ft.tokens import CODEBOOK_SIZE

BASE = 128266
REPO = Path(__file__).resolve().parents[1]


def _codes(n, seed=0):
    rng = random.Random(seed)
    return [
        [rng.randrange(CODEBOOK_SIZE) for _ in range(n)],
        [rng.randrange(CODEBOOK_SIZE) for _ in range(2 * n)],
        [rng.randrange(CODEBOOK_SIZE) for _ in range(4 * n)],
    ]


def _ref_ids_to_codes():
    spec = importlib.util.spec_from_file_location("ref", REPO / "reference" / "inference.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.ids_to_codes


def test_codes_to_tokens_matches_reference_interleave():
    codes = _codes(12, seed=1)
    tokens = codes_to_tokens(codes, BASE, deduplicate=False)
    assert num_frames(tokens) == 12
    c0, c1, c2 = (t[0].tolist() for t in _ref_ids_to_codes()(tokens, BASE, "cpu"))
    assert (c0, c1, c2) == tuple(codes)  # round-trips exactly through the reference decoder


def test_dedup_removes_consecutive_same_c0():
    # c0 = [5, 5, 9] -> the 2nd frame (dup c0) is dropped
    codes = [[5, 5, 9], [0] * 6, [0] * 12]
    assert num_frames(codes_to_tokens(codes, BASE, deduplicate=True)) == 2
    assert num_frames(codes_to_tokens(codes, BASE, deduplicate=False)) == 3


def test_bad_stream_lengths_rejected():
    with pytest.raises(ValueError, match="expected c1=2N"):
        codes_to_tokens([[1, 2], [0, 0, 0], [0] * 8], BASE)


@pytest.mark.skipif(
    not os.path.isdir(os.environ.get("SNAC_DIR", "")),
    reason="set SNAC_DIR to the snac_24khz model dir to run the real codec round-trip")
def test_real_snac_encode_decode():
    import numpy as np
    import torch

    from indic_speak_ft.data.compile import snac_encode_tokens
    from indic_speak_ft.model.loading import load_snac

    snac = load_snac(os.environ["SNAC_DIR"], device="cpu")
    wav = (0.1 * np.random.randn(24000)).astype(np.float32)  # 1 s
    tokens = snac_encode_tokens(snac, wav, BASE, device="cpu", deduplicate=False)
    assert num_frames(tokens) > 0
    codes = _ref_ids_to_codes()(tokens, BASE, "cpu")
    with torch.inference_mode():
        audio = snac.decoder(snac.quantizer.from_codes(codes))
    assert torch.isfinite(audio).all()
