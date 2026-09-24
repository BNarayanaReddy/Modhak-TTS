"""Speaker-similarity eval — calibrated with a ceiling and a floor (GOAL Phase 6).

Raw cosine similarity is uninterpretable alone, so every model score is reported alongside a
CEILING (two genuine clips of the same target voice, e.g. two Anagha) and a FLOOR (two
different voices, Anagha vs Chinmay). A model score near the ceiling = voice preserved; drifting
toward the floor = the D15 drift we watch for, tracked symmetrically on both target voices.

Embedder default: speechbrain ECAPA. Pluggable via any ``embed(wav, sr) -> vector`` callable.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from indic_speak_ft.eval.metrics import cosine_similarity

EmbedFn = Callable[[np.ndarray, int], np.ndarray]


class EcapaEmbedder:
    """Lazy speechbrain/spkrec-ecapa-voxceleb embedder."""

    def __init__(self, source: str = "speechbrain/spkrec-ecapa-voxceleb", device: str = "cpu"):
        self.source = source
        self.device = device
        self._model: Any = None

    def _ensure(self) -> None:
        if self._model is None:
            from speechbrain.inference.speaker import EncoderClassifier

            dev = "cuda:0" if self.device == "cuda" else self.device  # speechbrain wants an indexed device
            self._model = EncoderClassifier.from_hparams(source=self.source, run_opts={"device": dev})

    def __call__(self, wav: np.ndarray, sr: int) -> np.ndarray:
        self._ensure()
        import torch

        if sr != 16000:
            import librosa

            wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
        emb = self._model.encode_batch(torch.tensor(wav).unsqueeze(0))
        return emb.squeeze().detach().cpu().numpy()


def mean_pairwise_similarity(embed_fn: EmbedFn, group_a: list[np.ndarray],
                             group_b: list[np.ndarray], sr: int = 24000) -> float:
    """Mean cosine similarity across all cross pairs of two clip groups."""
    ea = [embed_fn(w, sr) for w in group_a]
    eb = [embed_fn(w, sr) for w in group_b]
    sims = [cosine_similarity(a, b) for a in ea for b in eb]
    return float(np.mean(sims)) if sims else float("nan")


def similarity_with_calibration(
    embed_fn: EmbedFn,
    generated: list[np.ndarray],
    same_voice_refs: list[np.ndarray],
    ceiling_pair: tuple[list[np.ndarray], list[np.ndarray]],
    floor_pair: tuple[list[np.ndarray], list[np.ndarray]],
    sr: int = 24000,
) -> dict[str, float]:
    """Model similarity (generated vs same-voice refs) with ceiling + floor anchors."""
    return {
        "model": mean_pairwise_similarity(embed_fn, generated, same_voice_refs, sr),
        "ceiling": mean_pairwise_similarity(embed_fn, *ceiling_pair, sr=sr),
        "floor": mean_pairwise_similarity(embed_fn, *floor_pair, sr=sr),
    }
