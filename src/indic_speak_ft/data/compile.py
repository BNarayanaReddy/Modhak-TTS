"""SNAC compile step: 24 kHz waveform → flattened 7-token audio ids (Decision 8 — re-encode
raw audio through the correct SNAC pipeline, never reuse another model's pre-encoded tokens).

Sits between ``loader`` (waveforms) and ``collator`` (audio ids). The interleave is produced
via ``tokens.build_frame`` so it is the exact inverse of ``reference.ids_to_codes`` (guarded
by test_tokens/test_collator); consecutive-duplicate frames (same c0) are removed at encode
time, matching ``token_contract.md`` §3 and bodhan_genai's codec.
"""
from __future__ import annotations

from typing import Any

from indic_speak_ft.tokens import CODEBOOK_SIZE, NUM_CODEBOOKS, build_frame


def codes_to_tokens(codes: list[list[int]], base: int, *, deduplicate: bool = True) -> list[int]:
    """Interleave SNAC's 3 codebook streams ``[c0(N), c1(2N), c2(4N)]`` into flat frame ids.
    Pure (no torch/SNAC) — testable against build_frame/ids_to_codes."""
    c0, c1, c2 = codes
    n = len(c0)
    if len(c1) != 2 * n or len(c2) != 4 * n:
        raise ValueError(f"expected c1=2N,c2=4N for N={n}; got {len(c1)},{len(c2)}")

    frames: list[list[int]] = []
    prev_c0: int | None = None
    for i in range(n):
        if deduplicate and c0[i] == prev_c0:  # drop consecutive duplicate frames (same c0)
            continue
        prev_c0 = c0[i]
        frames.append(build_frame(
            c0[i], c1[2 * i], c1[2 * i + 1], c2[4 * i], c2[4 * i + 1], c2[4 * i + 2], c2[4 * i + 3],
            base=base))
    return [tid for frame in frames for tid in frame]


def snac_encode_tokens(
    snac_model: Any, wav, base: int, *, device: str = "cpu", deduplicate: bool = True,
) -> list[int]:
    """Encode a 24 kHz mono waveform to flat audio ids via the SNAC model + interleave."""
    import numpy as np
    import torch

    with torch.inference_mode():
        at = torch.as_tensor(np.asarray(wav, dtype=np.float32), device=device).unsqueeze(0).unsqueeze(0)
        codes = snac_model.encode(at)
    c0, c1, c2 = (c[0].cpu().tolist() for c in codes)
    return codes_to_tokens([c0, c1, c2], base, deduplicate=deduplicate)


def num_frames(tokens: list[int]) -> int:
    """Whole SNAC frames in a flat token list."""
    return len(tokens) // NUM_CODEBOOKS


def approx_seconds(tokens: list[int], tokens_per_second: float = 82.0) -> float:
    """Rough duration from token count (≈82 tok/s; token_contract.md §3 rule of thumb).
    Not exact because dedup removes frames, so this is a monitoring estimate only."""
    return len(tokens) / tokens_per_second


assert NUM_CODEBOOKS == 7 and CODEBOOK_SIZE == 4096  # guard: layout constants unchanged
