"""Phase-6 metric tests (pure, hermetic)."""
from __future__ import annotations

import numpy as np

from indic_speak_ft.eval.metrics import (
    bootstrap_ci,
    cosine_similarity,
    energy_pause_stats,
    f0_stats,
    ngram_repetition_rate,
    per_utterance_wer,
    wer_measures,
)


def test_wer_counts_ids():
    # ref: 4 words; hyp: one substitution ("cat"->"dog"), one deletion ("mat" dropped)
    m = wer_measures(["the cat on mat"], ["the dog on"])
    assert m["substitutions"] == 1 and m["deletions"] == 1 and m["insertions"] == 0
    assert m["ref_words"] == 4
    assert abs(m["wer"] - 0.5) < 1e-9


def test_per_utterance_wer_length():
    w = per_utterance_wer(["a b c", "d e"], ["a b c", "d x"])
    assert len(w) == 2 and w[0] == 0.0 and w[1] > 0.0


def test_ngram_repetition_rate():
    assert ngram_repetition_rate([1, 2, 3, 4, 5], n=2) == 0.0            # all distinct
    assert ngram_repetition_rate([7, 7, 7, 7], n=2) > 0.5               # highly repetitive
    assert ngram_repetition_rate([1], n=3) == 0.0                       # too short


def test_bootstrap_ci_orders_and_contains_point():
    vals = [0.1, 0.2, 0.15, 0.25, 0.3, 0.05, 0.2, 0.18]
    point, lo, hi = bootstrap_ci(vals, seed=1)
    assert lo <= point <= hi
    assert abs(point - float(np.mean(vals))) < 1e-9
    assert bootstrap_ci([])[0] != bootstrap_ci([])[0]  # NaN for empty


def test_cosine_similarity():
    assert abs(cosine_similarity(np.array([1, 0, 0]), np.array([1, 0, 0])) - 1.0) < 1e-9
    assert abs(cosine_similarity(np.array([1, 0]), np.array([0, 1]))) < 1e-9


def test_f0_stats_on_tone():
    sr = 24000
    t = np.arange(sr) / sr
    tone = 0.5 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)  # 220 Hz
    s = f0_stats(tone, sr)
    assert s["mean_f0"] is not None and 180 < s["mean_f0"] < 260   # near 220 Hz


def test_energy_pause_stats_silence_vs_signal():
    sr = 24000
    sig = np.concatenate([np.zeros(sr // 2, np.float32),
                          0.5 * np.sin(2 * np.pi * 200 * np.arange(sr // 2) / sr).astype(np.float32)])
    s = energy_pause_stats(sig, sr)
    assert 0.3 < s["pause_ratio"] < 0.7   # ~half is silence
    assert s["energy_std"] > 0
