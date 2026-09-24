"""Fetch the real reference audio for eval clips (speaker-similarity anchor).

The eval buckets hold clips out of training by a path-based ``clip_key`` and record the source
shard; here we resolve each item back to its 24 kHz waveform so the panel can compare a synthesis
against the genuine voice. Reads are shard-cached: the first request for a shard streams it once
(stopping as soon as every needed clip in it is found — the held-out clips sit in the first rows),
keyed by the same ``eval_clip_key`` the manifest used, so audio is downloaded at most once per shard.
No audio is stored in the repo; it is fetched on demand at eval time.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from indic_speak_ft.data.loader import decode_audio, eval_clip_key, iter_parquet_rows, resample_to

# source_id → the HF directory its parquet shards live in (mirrors the loader/eval-bucket paths).
_SHARD_DIR: dict[str, str] = {
    "SPRINGLab/IndicTTS_Marathi": "datasets/SPRINGLab/IndicTTS_Marathi/data",
    "SPRINGLab/IndicTTS-Hindi": "datasets/SPRINGLab/IndicTTS-Hindi/data",
    "SPRINGLab/IndicTTS-English": "datasets/SPRINGLab/IndicTTS-English/data",
    "ai4bharat/Rasa": "datasets/ai4bharat/Rasa/Marathi",
}


def hf_shard_path(source_id: str, shard: str) -> str:
    """Full HfFileSystem path for a source's shard basename (e.g. test-00006-of-00007.parquet)."""
    return f"{_SHARD_DIR[source_id]}/{shard}"


class ReferenceAudioProvider:
    """Resolve eval items → their real 24 kHz reference waveform, caching each shard's audio on
    first use. Pass as ``reference_audio_fn`` to ``run_panel``. Items with no ``clip_key`` (e.g.
    the synthesis-only longform bucket) or an unknown source return ``None``."""

    def __init__(self, manifests: dict[str, dict], *, target_sr: int = 24000, filesystem: Any = None):
        self.sr = target_sr
        self._fs = filesystem
        # (source_id, shard) → set of clip_keys we need from that shard
        self._need: dict[tuple[str, str], set[str]] = {}
        for m in manifests.values():
            for it in m.get("items", []):
                key, shard, src = it.get("clip_key"), it.get("shard"), it.get("source_id")
                if key and shard and src in _SHARD_DIR:
                    self._need.setdefault((src, shard), set()).add(key)
        self._cache: dict[str, np.ndarray] = {}
        self._loaded: set[tuple[str, str]] = set()

    def _load_shard(self, source_id: str, shard: str) -> None:
        fs = self._fs
        if fs is None:
            from huggingface_hub import HfFileSystem

            fs = HfFileSystem()
        need = self._need.get((source_id, shard), set())
        found = 0
        for row in iter_parquet_rows(hf_shard_path(source_id, shard), columns=["audio"], filesystem=fs):
            key = eval_clip_key(source_id, row)
            if key in need and key not in self._cache:
                wav, sr = decode_audio(row["audio"]["bytes"])
                self._cache[key] = resample_to(wav, sr, self.sr)
                found += 1
                if found >= len(need):  # every held-out clip in this shard located → stop early
                    break
        self._loaded.add((source_id, shard))

    def __call__(self, item: dict) -> np.ndarray | None:
        key = item.get("clip_key")
        src, shard = item.get("source_id"), item.get("shard")
        if not key or src not in _SHARD_DIR or not shard:
            return None
        if (src, shard) not in self._loaded:
            self._load_shard(src, shard)
        return self._cache.get(key)
