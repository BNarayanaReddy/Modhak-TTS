"""A/B decode attribution (Decision 5): decode the same audio ids with Vocos and with the
stock SNAC decoder and compare, so a regression can be attributed to the LM vs the vocoder."""
from __future__ import annotations

from typing import Any

import numpy as np

from indic_speak_ft.inference.decode import decode_to_wav


def _mel_corr(a: np.ndarray, b: np.ndarray, sr: int = 24000) -> float:
    import librosa

    def mel(y: np.ndarray) -> np.ndarray:
        return librosa.power_to_db(
            librosa.feature.melspectrogram(y=y.astype(np.float32), sr=sr, n_mels=80, hop_length=256) + 1e-9)

    ma, mb = mel(a), mel(b)
    n = min(ma.shape[1], mb.shape[1])
    return float(np.corrcoef(ma[:, :n].ravel(), mb[:, :n].ravel())[0, 1])


def ab_decode(audio_ids: list[int], base: int, snac_model: Any, vocos: Any, sr: int = 24000) -> dict:
    """Decode with both paths; report durations + mel correlation between them."""
    v = decode_to_wav(audio_ids, base, snac_model, vocos, stock=False)
    s = decode_to_wav(audio_ids, base, snac_model, vocos, stock=True)
    return {
        "vocos_seconds": len(v) / sr,
        "stock_seconds": len(s) / sr,
        "vocos_vs_stock_mel_corr": _mel_corr(v, s, sr),
    }
