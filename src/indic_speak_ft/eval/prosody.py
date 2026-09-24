"""Prosody metrics: F0 mean/std, energy std, pause ratio (GOAL Phase 6), compared to paired
ground truth where available."""
from __future__ import annotations

import numpy as np

from indic_speak_ft.eval.metrics import energy_pause_stats, f0_stats


def prosody_vector(wav: np.ndarray, sr: int = 24000) -> dict[str, float | None]:
    return {**f0_stats(wav, sr), **energy_pause_stats(wav, sr)}


def compare_to_reference(generated: np.ndarray, reference: np.ndarray, sr: int = 24000) -> dict:
    """Prosody of generated + reference + signed deltas (None-safe for unvoiced F0)."""
    g, r = prosody_vector(generated, sr), prosody_vector(reference, sr)

    def _delta(k: str) -> float | None:
        gv, rv = g.get(k), r.get(k)
        return (gv - rv) if (gv is not None and rv is not None) else None

    deltas = {k: _delta(k) for k in ("mean_f0", "f0_std", "energy_std", "pause_ratio")}
    return {"generated": g, "reference": r, "delta": deltas}
