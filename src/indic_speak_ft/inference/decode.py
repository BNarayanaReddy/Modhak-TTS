"""SNAC-token → waveform decoding, matching reference/inference.py exactly.

``snac_codes_from_tokens`` is our own mirror of the reference ``ids_to_codes`` (de-interleave
+ strip per-position offsets → the 3 codebook streams); ``tests/test_reference_parity.py``
asserts it equals the reference on identical inputs. ``decode_to_wav`` then runs
``quantizer.from_codes`` → Vocos (primary) or the SNAC decoder (``stock=True``, for A/B
attribution — Decision 5), with the reference's clamp + float32 output.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from indic_speak_ft.tokens import CODEBOOK_SIZE, NUM_CODEBOOKS, is_audio_token


def take_audio_prefix(new_token_ids: list[int], base: int) -> list[int]:
    """Audio tokens from the start of the generation up to the first non-audio token (the
    reference stops decoding there — that's where ``<|end_of_speech|>`` lands)."""
    out: list[int] = []
    for t in new_token_ids:
        if not is_audio_token(base, t):
            break
        out.append(t)
    return out


def snac_codes_from_tokens(audio_ids: list[int], base: int, device: str = "cpu") -> list[Any]:
    """Flat audio ids → ``[c0, c1, c2]`` SNAC code tensors (mirror of reference.ids_to_codes).
    Drops any trailing partial frame and any frame with an out-of-range code."""
    import torch

    n = len(audio_ids) // NUM_CODEBOOKS
    if n == 0:
        raise ValueError("no complete SNAC frame in audio_ids")
    arr = np.array(audio_ids[: n * NUM_CODEBOOKS], np.int32).reshape(n, NUM_CODEBOOKS)
    raw = arr - (base + np.arange(NUM_CODEBOOKS, dtype=np.int32) * CODEBOOK_SIZE)
    valid = raw[np.all((raw >= 0) & (raw < CODEBOOK_SIZE), axis=1)]
    if valid.shape[0] == 0:
        raise ValueError("no in-range SNAC frame")

    c1 = np.empty(valid.shape[0] * 2, np.int32)
    c1[0::2], c1[1::2] = valid[:, 1], valid[:, 4]
    c2 = np.empty(valid.shape[0] * 4, np.int32)
    c2[0::4], c2[1::4], c2[2::4], c2[3::4] = valid[:, 2], valid[:, 3], valid[:, 5], valid[:, 6]
    return [torch.from_numpy(x.copy()).long().unsqueeze(0).to(device) for x in (valid[:, 0], c1, c2)]


def decode_to_wav(
    audio_ids: list[int], base: int, snac_model: Any, vocos: Any, *,
    stock: bool = False, device: str = "cpu",
) -> np.ndarray:
    """Decode flat audio ids to a float32 mono 24 kHz waveform. ``stock=True`` uses SNAC's own
    decoder instead of Vocos (A/B attribution). Matches the reference clamp/float32 output."""
    import torch

    codes = snac_codes_from_tokens(audio_ids, base, device=device)
    z_q = snac_model.quantizer.from_codes(codes)
    with torch.no_grad():
        wav = snac_model.decoder(z_q) if stock else vocos(z_q.float())
    return wav[0, 0].clamp(-1, 1).float().cpu().numpy()
