"""Per-codebook-position weighted cross-entropy on the audio span (Decision 4 + D14).

Loss falls only on tokens the collator labeled (audio frames + ``<|end_of_speech|>``; the
prompt is −100). Each audio token is weighted by its FRAME POSITION (0..6): codebook 0 gets
1.0, codebook 1 (positions 1,4) 0.7, codebook 2 (positions 2,3,5,6) 0.4 — codebook 0 carries
perceptual weight out of proportion to its bitrate. The stop token gets ``stop_weight`` (1.0
by default) so the model still learns to stop (D14). Weights are read from the label token id
itself (position = (label−base)//4096), so no extra collator output is needed.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from indic_speak_ft.tokens import CODEBOOK_SIZE, NUM_CODEBOOKS

IGNORE_INDEX = -100
# Default per-frame-position weights (Decision 4): [c0, c1, c2, c2, c1, c2, c2].
DEFAULT_POSITION_WEIGHTS: tuple[float, ...] = (1.0, 0.7, 0.4, 0.4, 0.7, 0.4, 0.4)


def per_token_weights(
    labels: torch.Tensor, base: int, position_weights: torch.Tensor, *,
    stop_token_id: int, stop_weight: float,
) -> torch.Tensor:
    """Weight for every label position: per-frame-position for audio tokens, ``stop_weight``
    for the stop token, 0 for ignored (−100), 1.0 otherwise."""
    lo, hi = base, base + NUM_CODEBOOKS * CODEBOOK_SIZE
    is_audio = (labels >= lo) & (labels < hi)
    pos = ((labels - lo) // CODEBOOK_SIZE).clamp(0, NUM_CODEBOOKS - 1)
    w = torch.ones_like(labels, dtype=position_weights.dtype)
    w = torch.where(is_audio, position_weights[pos], w)
    w = torch.where(labels == stop_token_id, torch.as_tensor(stop_weight, dtype=w.dtype, device=w.device), w)
    w = torch.where(labels == IGNORE_INDEX, torch.zeros_like(w), w)
    return w


def weighted_audio_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    base: int,
    stop_token_id: int,
    position_weights: tuple[float, ...] | torch.Tensor = DEFAULT_POSITION_WEIGHTS,
    stop_weight: float = 1.0,
    enable_weighting: bool = True,
) -> torch.Tensor:
    """Causal-LM weighted CE. ``logits`` [B,T,V], ``labels`` [B,T] (−100 on the prompt).
    Shifts internally (predict token t from t−1). ``enable_weighting=False`` → uniform."""
    if isinstance(position_weights, torch.Tensor):
        wvec = position_weights.to(dtype=logits.dtype, device=logits.device)
    else:
        wvec = torch.tensor(position_weights, dtype=logits.dtype, device=logits.device)
    if wvec.numel() != NUM_CODEBOOKS:
        raise ValueError(f"position_weights must have {NUM_CODEBOOKS} entries, got {wvec.numel()}")

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    v = shift_logits.size(-1)
    ce = F.cross_entropy(
        shift_logits.view(-1, v), shift_labels.view(-1),
        ignore_index=IGNORE_INDEX, reduction="none")  # 0 where ignored

    if not enable_weighting:
        n = (shift_labels.view(-1) != IGNORE_INDEX).sum().clamp(min=1)
        return ce.sum() / n

    w = per_token_weights(shift_labels.view(-1), base, wvec,
                          stop_token_id=stop_token_id, stop_weight=stop_weight)
    return (ce * w).sum() / w.sum().clamp(min=1e-8)
