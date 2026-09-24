"""Retention WER on retention_hindi + retention_english — the single most important
post-fine-tune metric (Decision 3). Also provides the fast retention-eval callable the
RegressionStopCallback runs every 500 steps during training.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from indic_speak_ft.eval.asr import ASRBackend
from indic_speak_ft.eval.metrics import wer_measures

# synthesize_fn(text, speaker) -> (wav, audio_ids, hit_max_new_tokens)
SynthesizeFn = Callable[[str, str], tuple[Any, list[int], bool]]


def bucket_wer(items: list[dict], synthesize_fn: SynthesizeFn, asr: ASRBackend, *, sr: int = 24000) -> dict:
    refs = [it["text_normalized"] for it in items]
    hyps = [asr.transcribe(synthesize_fn(it["text"], it["speaker"])[0], sr) for it in items]
    return {**wer_measures(refs, hyps), "hypotheses": hyps, "references": refs}


def make_retention_eval_fn(
    manifests: dict, synthesize_fn: SynthesizeFn, asr: ASRBackend, *,
    buckets: tuple[str, ...] = ("retention_hindi", "retention_english"),
    max_items: int | None = 12,
) -> Callable[[], dict[str, float]]:
    """A ``() -> {bucket: wer}`` closure for RegressionStopCallback. ``max_items`` keeps the
    in-training eval fast."""
    def _eval() -> dict[str, float]:
        out: dict[str, float] = {}
        for b in buckets:
            items = manifests[b]["items"]
            items = items[:max_items] if max_items else items
            if items:
                out[b] = bucket_wer(items, synthesize_fn, asr)["wer"]
        return out

    return _eval
