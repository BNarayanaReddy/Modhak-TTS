"""Phase-4 loss tests: per-codebook-position weighting on the audio span."""
from __future__ import annotations

import torch

from indic_speak_ft.data.collator import IGNORE_INDEX
from indic_speak_ft.tokens import build_frame
from indic_speak_ft.train.loss import (
    DEFAULT_POSITION_WEIGHTS,
    per_token_weights,
    weighted_audio_ce,
)

BASE = 128266
STOP = 128258  # <|end_of_speech|>


def test_per_token_weights_by_frame_position():
    frame = build_frame(1, 2, 3, 4, 5, 6, 7, base=BASE)  # 7 audio ids at positions 0..6
    labels = torch.tensor([IGNORE_INDEX, IGNORE_INDEX, *frame, STOP])
    w = per_token_weights(labels, BASE, torch.tensor(DEFAULT_POSITION_WEIGHTS),
                          stop_token_id=STOP, stop_weight=1.0)
    assert w[:2].tolist() == [0.0, 0.0]                       # prompt masked
    assert torch.allclose(w[2:9], torch.tensor(DEFAULT_POSITION_WEIGHTS))  # per position
    assert w[-1].item() == 1.0                                # stop token


def test_weighted_differs_from_uniform():
    torch.manual_seed(0)
    frame = build_frame(*(i for i in range(7)), base=BASE)
    labels = torch.tensor([[IGNORE_INDEX, *frame, STOP]])
    logits = torch.randn(1, labels.size(1), 200000)
    w = weighted_audio_ce(logits, labels, base=BASE, stop_token_id=STOP, enable_weighting=True)
    u = weighted_audio_ce(logits, labels, base=BASE, stop_token_id=STOP, enable_weighting=False)
    assert torch.isfinite(w) and torch.isfinite(u)
    assert not torch.allclose(w, u)  # weighting changes the loss


def test_prompt_only_labels_give_finite_zeroish_loss():
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]])
    logits = torch.randn(1, 3, 1000)
    loss = weighted_audio_ce(logits, labels, base=BASE, stop_token_id=STOP)
    assert torch.isfinite(loss)  # no supervised tokens -> guarded denominator, no NaN


def test_matches_manual_weighted_mean():
    # tiny hand-checkable case: 1 audio token (pos 0, weight 1.0) + stop (weight 0.5)
    a0 = build_frame(3, 0, 0, 0, 0, 0, 0, base=BASE)[0]  # position-0 id
    labels = torch.tensor([[IGNORE_INDEX, a0, STOP]])
    logits = torch.randn(1, 3, BASE + 7 * 4096 + 8)
    import torch.nn.functional as F
    sl, st = logits[:, :-1, :].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1)
    ce = F.cross_entropy(sl, st, ignore_index=IGNORE_INDEX, reduction="none")
    w = torch.tensor([1.0, 0.5])  # pos-0 weight, stop_weight
    manual = (ce * w).sum() / w.sum()
    got = weighted_audio_ce(logits, labels, base=BASE, stop_token_id=STOP, stop_weight=0.5)
    assert torch.allclose(got, manual, atol=1e-5)
