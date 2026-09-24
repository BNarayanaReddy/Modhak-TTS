"""Pure metric + statistics helpers for the eval panel — no model dependencies, fully testable.

WER reports insertions/deletions/substitutions separately (GOAL Phase 6); CIs are
utterance-level bootstrap (GOAL: 95% bootstrap CIs over utterances). Prosody stats use
librosa but operate on raw waveforms, so they are covered by cheap synthetic-signal tests.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _jiwer_measures(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """WER + ins/del/sub/hits, robust across jiwer versions (>=3 dropped compute_measures
    for process_words; older releases only have compute_measures)."""
    import jiwer

    if hasattr(jiwer, "process_words"):
        o = jiwer.process_words(references, hypotheses)
        return {"wer": float(o.wer), "insertions": int(o.insertions), "deletions": int(o.deletions),
                "substitutions": int(o.substitutions), "hits": int(o.hits)}
    m = jiwer.compute_measures(references, hypotheses)
    return {"wer": float(m["wer"]), "insertions": int(m["insertions"]), "deletions": int(m["deletions"]),
            "substitutions": int(m["substitutions"]), "hits": int(m["hits"])}


def wer_measures(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """Corpus WER with insertion/deletion/substitution counts (via jiwer)."""
    m = _jiwer_measures(references, hypotheses)
    m["ref_words"] = m["hits"] + m["substitutions"] + m["deletions"]
    return m


def per_utterance_wer(references: list[str], hypotheses: list[str]) -> list[float]:
    """WER per (ref, hyp) pair — the inputs to a bootstrap CI over utterances."""
    return [_jiwer_measures([r], [h])["wer"] if r.strip() else 0.0
            for r, h in zip(references, hypotheses)]


def ngram_repetition_rate(tokens: list[int], n: int = 3) -> float:
    """Fraction of n-grams that are non-unique — a longform runaway/repetition signal.
    0.0 = all distinct, →1.0 = highly repetitive."""
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def bootstrap_ci(
    values: list[float], *, statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 1000, alpha: float = 0.05, seed: int = 0,
) -> tuple[float, float, float]:
    """(point, lo, hi) for the given statistic via utterance-level resampling. Empty → NaNs."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.array([statistic(rng.choice(arr, size=arr.size, replace=True)) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(statistic(arr)), float(lo), float(hi)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def f0_stats(wav: np.ndarray, sr: int = 24000) -> dict[str, float | None]:
    """Mean/std of voiced F0 (librosa.pyin). None if no voiced frames."""
    import librosa

    f0, _, _ = librosa.pyin(wav.astype(np.float32), sr=sr,
                            fmin=float(librosa.note_to_hz("C2")),
                            fmax=float(librosa.note_to_hz("C7")))
    v = f0[np.isfinite(f0)]
    if v.size == 0:
        return {"mean_f0": None, "f0_std": None}
    return {"mean_f0": float(np.mean(v)), "f0_std": float(np.std(v))}


def energy_pause_stats(wav: np.ndarray, sr: int = 24000, *, frame_ms: float = 25.0,
                       hop_ms: float = 10.0, pause_db: float = -35.0) -> dict[str, float]:
    """Energy std (dynamics) + pause ratio (fraction of frames below ``pause_db`` re: peak)."""
    win, hop = max(1, int(sr * frame_ms / 1000)), max(1, int(sr * hop_ms / 1000))
    x = wav.astype(np.float64)
    frames = np.array([np.sqrt(np.mean(x[i:i + win] ** 2) + 1e-12)
                       for i in range(0, max(1, len(x) - win), hop)] or [0.0])
    peak = frames.max() + 1e-12
    db = 20 * np.log10(frames / peak + 1e-12)
    return {"energy_std": float(np.std(frames)), "pause_ratio": float(np.mean(db < pause_db))}
