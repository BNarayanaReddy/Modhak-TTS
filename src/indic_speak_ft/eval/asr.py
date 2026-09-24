"""Marathi ASR for WER — Bodhan's ``indic-transcribe-core`` (confirmed choice; custom
"indic-canary" architecture, gated, loaded with ``trust_remote_code``).

Thin, lazy wrapper: the model loads on first use so importing the eval package stays cheap and
model-free. ``transcribe`` returns text; the WER math lives in ``eval.metrics``. Hypotheses are
persisted for audit (GOAL Phase 6). The ASR is pluggable — anything with a ``transcribe(wav, sr)``
method works — so the panel and tests can inject a fake.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

DEFAULT_ASR_MODEL = "bodhan-ai/indic-transcribe-core"


class ASRBackend(Protocol):
    def transcribe(self, wav: np.ndarray, sr: int, language: str | None = None) -> str: ...


class IndicTranscribeASR:
    """Lazy wrapper for Bodhan indic-transcribe-core. The model ships its own code inside the
    snapshot: a callable ``IndicTranscribe`` loaded from a LOCAL dir (its SentencePiece tokenizer
    is path-based, so a repo id is rejected) that transcribes a FILE PATH. We snapshot the repo,
    put its dir on ``sys.path``, and transcribe each waveform by writing it to a temp wav.
    Verified against the model card's `IndicTranscribe.from_pretrained(dir)(path, lang=...)` API."""

    def __init__(self, model_id: str = DEFAULT_ASR_MODEL, *, language: str = "mr", device: str = "cpu"):
        self.model_id = model_id
        self.language = language
        self.device = device
        self._asr: Any = None

    def _ensure_loaded(self) -> None:
        if self._asr is not None:
            return
        import sys

        from huggingface_hub import snapshot_download

        model_dir = (self.model_id if Path(self.model_id).is_dir()
                     else snapshot_download(self.model_id, ignore_patterns=["nemo/*"]))  # skip 2.5 GB .nemo
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)  # the model code (indic_transcribe.py) ships in the snapshot
        from indic_transcribe import IndicTranscribe  # type: ignore[import-not-found]

        self._asr = IndicTranscribe.from_pretrained(model_dir, device=self.device)

    def transcribe(self, wav: np.ndarray, sr: int, language: str | None = None) -> str:
        """Transcribe a mono waveform in ``language`` (falls back to the instance default). Each
        eval bucket carries its own language_id, so retention_hindi/english are transcribed in
        hi/en, not mr — otherwise a Marathi ASR on English audio inflates the retention WER."""
        self._ensure_loaded()
        import os
        import tempfile

        import soundfile as sf

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(path, np.asarray(wav, dtype=np.float32), sr)
            out = self._asr(path, lang=language or self.language)
            return out if isinstance(out, str) else str(out)
        finally:
            os.unlink(path)


def transcribe_all(asr: ASRBackend, wavs: list[np.ndarray], sr: int = 24000) -> list[str]:
    return [asr.transcribe(w, sr) for w in wavs]


def persist_hypotheses(out_dir: str, bucket: str, refs: list[str], hyps: list[str],
                       ids: list[str]) -> None:
    """Write raw ref/hyp pairs for audit (eval_outputs/<bucket>_hyps.json)."""
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / f"{bucket}_hyps.json").write_text(json.dumps(
        [{"id": i, "reference": r, "hypothesis": h} for i, r, h in zip(ids, refs, hyps)],
        ensure_ascii=False, indent=2))
