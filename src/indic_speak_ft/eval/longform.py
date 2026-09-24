"""Longform (30–60 s) metrics: n-gram repetition rate, speaker-similarity trajectory across
the utterance, and the count of max_new_tokens-hit warnings (GOAL Phase 6)."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from indic_speak_ft.eval.metrics import cosine_similarity, ngram_repetition_rate


def similarity_trajectory(wav: np.ndarray, ref_embedding: np.ndarray,
                          embed_fn: Callable[[np.ndarray, int], np.ndarray], *,
                          sr: int = 24000, window_s: float = 5.0) -> list[float]:
    """Cosine similarity of each ~window_s slice of a long utterance to a reference embedding —
    surfaces drift that only appears late in a long synthesis."""
    w = int(sr * window_s)
    traj: list[float] = []
    for start in range(0, max(1, len(wav) - w + 1), w):
        seg = wav[start:start + w]
        if seg.size >= sr:  # need >= 1 s to embed
            traj.append(cosine_similarity(embed_fn(seg, sr), ref_embedding))
    return traj


def longform_metrics(
    audio_ids: list[int], *, hit_max_new_tokens: bool,
    wav: np.ndarray | None = None, ref_embedding: np.ndarray | None = None,
    embed_fn: Callable[[np.ndarray, int], np.ndarray] | None = None, sr: int = 24000,
) -> dict:
    m: dict = {
        "n_frames": len(audio_ids) // 7,
        "repetition_rate_3gram": ngram_repetition_rate(audio_ids, 3),
        "repetition_rate_5gram": ngram_repetition_rate(audio_ids, 5),
        "hit_max_new_tokens": bool(hit_max_new_tokens),
    }
    if wav is not None and ref_embedding is not None and embed_fn is not None:
        traj = similarity_trajectory(wav, ref_embedding, embed_fn, sr=sr)
        m["similarity_trajectory"] = traj
        m["similarity_drop"] = float(traj[0] - traj[-1]) if len(traj) >= 2 else 0.0
    return m
