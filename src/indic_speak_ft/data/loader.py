"""Load clips from the source parquet corpora, decode+resample audio, measure quality,
assign the target voice (D15), normalize text, and apply the DATA.md gates.

Streams parquet row-groups (local path or ``hf://`` via ``HfFileSystem``) so the 6–78 GB
shards never need to be fully materialized — the loader is used for local validation and
on the server compile step alike. Audio-heavy work (decode/resample/quality) is done per
row; the SNAC encode itself lives in the compile step downstream (it reuses the same
waveforms). Gate thresholds come from configs/data.yaml, never hardcoded here.
"""
from __future__ import annotations

import hashlib
import io
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from indic_speak_ft.data.schema import (
    AudioMeta,
    ClipMetadata,
    CodeSwitchSpan,
    DedupMeta,
    Provenance,
    SpeakerMeta,
    StyleMeta,
    TextFeatures,
    Transcript,
)
from indic_speak_ft.data.text_norm import normalize_text


# ------------------------------------------------------------------ quality metrics
@dataclass(frozen=True)
class ClipQuality:
    duration_s: float
    snr_db: float
    clipping_ratio: float
    bandwidth_hz: float          # 99%-energy spectral rolloff on the ORIGINAL sr
    is_effectively_wideband: bool  # bandwidth covers a real speech band (dead-man's-switch input)


def clip_quality(wav: np.ndarray, sr: int, *, wideband_floor_hz: float = 6000.0) -> ClipQuality:
    """Cheap, dependency-light clip QC. SNR is a percentile-of-frame-energy estimate (no VAD);
    bandwidth is the 99%-energy rolloff on the native-rate signal, so an upsampled-narrowband
    file (e.g. 8 kHz content in a 48 kHz container) is caught by ``is_effectively_wideband``.
    """
    x = wav.astype(np.float64)
    if x.size == 0:
        return ClipQuality(0.0, -np.inf, 1.0, 0.0, False)
    clipping_ratio = float(np.mean(np.abs(x) >= 0.999))

    # short-time energy (25 ms / 10 ms); SNR ≈ high-percentile vs noise-floor percentile
    win, hop = max(1, int(0.025 * sr)), max(1, int(0.010 * sr))
    frames = [x[i:i + win] for i in range(0, max(1, len(x) - win), hop)] or [x]
    e = np.array([float(np.mean(f * f)) + 1e-12 for f in frames])
    noise, signal = np.percentile(e, 10), np.percentile(e, 95)
    snr_db = float(10.0 * np.log10(signal / noise)) if noise > 0 else float("inf")

    # 99%-energy rolloff on the magnitude spectrum (native sr)
    mag = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    csum = np.cumsum(mag)
    bandwidth_hz = float(freqs[int(np.searchsorted(csum, 0.99 * csum[-1]))]) if csum[-1] > 0 else 0.0

    return ClipQuality(
        duration_s=len(x) / sr,
        snr_db=snr_db,
        clipping_ratio=clipping_ratio,
        bandwidth_hz=bandwidth_hz,
        is_effectively_wideband=bandwidth_hz >= wideband_floor_hz,
    )


# ------------------------------------------------------------------ speaker assignment (D15)
def assign_speaker(source_hf_path: str, row: dict[str, Any], speaker_assignment: dict) -> str:
    """Map a row to a library voice per configs/data.yaml `speaker_assignment` (D15)."""
    cfg = speaker_assignment.get(source_hf_path)
    if cfg is None:
        raise KeyError(f"no speaker_assignment for source {source_hf_path!r}")
    how = cfg["by"]
    if how == "fixed":
        return cfg["voice"]
    if how == "gender_int":
        return cfg["map"][int(row["gender"])]
    if how == "gender_str":
        return cfg["map"][str(row["gender"]).strip().capitalize()]
    raise ValueError(f"unknown speaker_assignment.by {how!r}")


# ------------------------------------------------------------------ audio decode + resample
def decode_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode a struct<bytes> audio cell to mono float32 + sample rate."""
    import soundfile as sf

    wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav, int(sr)


def resample_to(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return wav
    import librosa

    return librosa.resample(wav, orig_sr=sr, target_sr=target_sr)


# ------------------------------------------------------------------ gates
@dataclass
class GateResult:
    kept: bool
    hard_reject: str | None = None
    soft_flags: list[str] = field(default_factory=list)


def apply_gates(quality: ClipQuality, meta: ClipMetadata, gates: dict) -> GateResult:
    """HARD rejects drop the clip; SOFT filters flag it (kept, but recorded)."""
    hard, soft = gates["hard"], gates["soft"]
    lo, hi = hard["duration_s"]
    if hard.get("require_known_license", True) and meta.provenance.license.lower() in ("", "unknown"):
        return GateResult(False, "license_unknown")
    if not (lo <= quality.duration_s <= hi):
        return GateResult(False, f"duration_out_of_[{lo},{hi}]")
    if hard.get("require_nonempty_normalized", True) and not meta.transcript.text_normalized.strip():
        return GateResult(False, "empty_normalized_text")
    # speaker∈library is already enforced by SpeakerMeta construction (raises before here)

    flags: list[str] = []
    if quality.snr_db < soft["snr_floor_db"]:
        flags.append(f"snr<{soft['snr_floor_db']}")
    if quality.clipping_ratio > soft["clipping_ratio_max"]:
        flags.append(f"clip>{soft['clipping_ratio_max']}")
    if not quality.is_effectively_wideband:
        flags.append("narrowband")
    return GateResult(True, None, flags)


# ------------------------------------------------------------------ row -> clip
def _sha16(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def eval_clip_key(source_hf_path: str, row: dict[str, Any]) -> str:
    """Stable per-clip holdout key from the audio PATH (cheap — no bytes). The eval-bucket
    build and this loader compute it identically, so held-out eval clips are excluded from
    training (reject_eval_bucket_audio_hash gate) without a content hash."""
    path = (row.get("audio") or {}).get("path") or str(row.get("_idx", ""))
    return _sha16(f"{source_hf_path}::{path}".encode())


def row_to_clip(
    row: dict[str, Any],
    *,
    source_hf_path: str,
    license: str,
    language_id: str,
    speaker_assignment: dict,
    target_sr: int,
    gates: dict,
    number_backend=None,
    held_out_keys: set[str] | None = None,
) -> tuple[ClipMetadata | None, np.ndarray | None, GateResult]:
    """Decode one parquet row into (metadata, 24 kHz waveform, gate result). On a hard
    reject returns (None, None, GateResult(kept=False, ...))."""
    if held_out_keys and eval_clip_key(source_hf_path, row) in held_out_keys:
        return None, None, GateResult(False, "eval_holdout")  # reject before decode (cheap)
    wav_native, sr = decode_audio(row["audio"]["bytes"])
    quality = clip_quality(wav_native, sr)
    wav = resample_to(wav_native, sr, target_sr)

    speaker = assign_speaker(source_hf_path, row, speaker_assignment)
    norm = normalize_text(row["text"], language_id=language_id, number_backend=number_backend)

    meta = ClipMetadata(
        provenance=Provenance(source_id=f"{source_hf_path}#{row.get('filename', row.get('_idx',''))}",
                              license=license),
        audio=AudioMeta(sample_rate=target_sr, duration_s=quality.duration_s,
                        snr_estimate=quality.snr_db, clipping_ratio=quality.clipping_ratio),
        transcript=Transcript(text_raw=row["text"], text_normalized=norm.text_normalized,
                              normalizer_version=norm.normalizer_version),
        text=TextFeatures(language_id=language_id, script=norm.script, num_digits=norm.num_digits,
                          num_english_words=norm.num_english_words,
                          code_switch_spans=[CodeSwitchSpan(**s) for s in norm.code_switch_spans],
                          domain="general"),
        speaker=SpeakerMeta(name=speaker),
        style=StyleMeta(),
        dedup=DedupMeta(audio_hash=_sha16(row["audio"]["bytes"]),
                        transcript_shingle_hash=_sha16(norm.text_normalized.encode("utf-8"))),
    )
    gate = apply_gates(quality, meta, gates)
    if not gate.kept:
        return None, None, gate
    return meta, wav, gate


# ------------------------------------------------------------------ streaming parquet source
def iter_parquet_rows(
    file_path: str, *, columns: list[str] | None = None, max_rows: int | None = None,
    filesystem=None,
) -> Iterator[dict[str, Any]]:
    """Yield row dicts from a parquet file, streaming by row-group (no full-file load).
    ``file_path`` may be local or an ``HfFileSystem`` path with ``filesystem=HfFileSystem()``."""
    import time

    import pyarrow.parquet as pq

    opener = filesystem.open if filesystem is not None else open
    with opener(file_path, "rb") as f:
        pf = pq.ParquetFile(f)
        seen = 0
        for rg in range(pf.metadata.num_row_groups):
            for attempt in range(4):  # remote reads can break mid-fetch; retry the row group
                try:
                    batch = pf.read_row_group(rg, columns=columns).to_pylist()
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(1.5 * (attempt + 1))
            for i, r in enumerate(batch):
                r.setdefault("_idx", seen)
                yield r
                seen += 1
                if max_rows is not None and seen >= max_rows:
                    return
