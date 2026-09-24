"""Phase-2 loader validation (run against the real corpora, streamed remotely).

Two outputs the reviewer asked for before locking mixture/eval buckets:
  A) end-to-end loader test on a handful of rows per source (decode→resample→quality→
     speaker(D15)→normalize→gate), printed as compact records.
  B) SPRINGLab/IndicTTS_Marathi per-speaker (per-gender) quality stats on a sample.

Reads configs/data.yaml. Streams parquet by row-group via HfFileSystem — no full download.

Run:  Modhak-TTS/.venv/bin/python scripts/validate_loader.py
"""
from __future__ import annotations

import statistics as st
from pathlib import Path

import numpy as np
import yaml
from huggingface_hub import HfFileSystem

from indic_speak_ft.data.loader import clip_quality, decode_audio, iter_parquet_rows, row_to_clip

REPO = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((REPO / "configs" / "data.yaml").read_text())

# licenses (to confirm for DATA.md/D10); non-empty so the license gate passes in validation
LICENSE = {
    "SPRINGLab/IndicTTS_Marathi": "CC-BY-4.0", "ai4bharat/Rasa": "CC-BY-4.0",
    "SPRINGLab/IndicTTS-Hindi": "CC-BY-4.0", "SPRINGLab/IndicTTS-English": "CC-BY-4.0",
}
SHARD = {
    "SPRINGLab/IndicTTS_Marathi": "datasets/SPRINGLab/IndicTTS_Marathi/data/train-00000-of-00004.parquet",
    "ai4bharat/Rasa":             "datasets/ai4bharat/Rasa/Marathi/test-00000-of-00007.parquet",
    "SPRINGLab/IndicTTS-Hindi":   "datasets/SPRINGLab/IndicTTS-Hindi/data/train-00000-of-00010.parquet",
    "SPRINGLab/IndicTTS-English": "datasets/SPRINGLab/IndicTTS-English/data/train-00000-of-00028.parquet",
}
COLS = {  # non-audio columns to also project (audio is always read)
    "SPRINGLab/IndicTTS_Marathi": ["audio", "text", "gender"],
    "ai4bharat/Rasa":             ["audio", "text", "gender", "style", "duration", "filename"],
    "SPRINGLab/IndicTTS-Hindi":   ["audio", "text", "gender"],
    "SPRINGLab/IndicTTS-English": ["audio", "text"],
}
fs = HfFileSystem()


def source_cfg(hf_path: str) -> dict:
    src = next(s for s in CFG["sources"].values() if s["hf_path"] == hf_path)
    return {"source_hf_path": hf_path, "license": LICENSE[hf_path], "language_id": src["language_id"],
            "speaker_assignment": CFG["speaker_assignment"], "target_sr": CFG["audio"]["target_sample_rate"],
            "gates": CFG["gates"]}


def part_a_end_to_end(n_per_source: int = 4) -> None:
    print("=" * 78, "\nA) END-TO-END LOADER TEST (a few rows per source)\n" + "=" * 78)
    for hf_path, shard in SHARD.items():
        print(f"\n### {hf_path}")
        kept = dropped = 0
        for row in iter_parquet_rows(shard, columns=COLS[hf_path], max_rows=n_per_source, filesystem=fs):
            meta, _wav, gate = row_to_clip(row, **source_cfg(hf_path))
            if gate.kept:
                kept += 1
                print(f"   KEEP  spk={meta.speaker.name:<8} {meta.audio.duration_s:5.2f}s "
                      f"snr={meta.audio.snr_estimate:5.1f}dB clip={meta.audio.clipping_ratio:.3f} "
                      f"lang={meta.text.language_id} cs={meta.text.num_english_words} "
                      f"flags={gate.soft_flags} txt={meta.transcript.text_normalized[:34]!r}")
            else:
                dropped += 1
                print(f"   DROP  [{gate.hard_reject}]")
        print(f"   -> kept {kept}, dropped {dropped}")


def part_b_springlab_speaker_stats(target_clips_per_gender: int = 35) -> None:
    hf_path = "SPRINGLab/IndicTTS_Marathi"
    print("\n" + "=" * 78, f"\nB) {hf_path} PER-SPEAKER (gender) QUALITY STATS\n" + "=" * 78)
    print("   NOTE: gender-sharded — shard0 all female (→Anagha), shard3 male-heavy (→Chinmay).")
    smap = CFG["speaker_assignment"][hf_path]["map"]

    # females from shard 0, males from shard 3 (both start with their gender)
    plan = [
        (0, "datasets/SPRINGLab/IndicTTS_Marathi/data/train-00000-of-00004.parquet"),
        (1, "datasets/SPRINGLab/IndicTTS_Marathi/data/train-00003-of-00004.parquet"),
    ]
    per: dict[str, dict[str, list]] = {}
    for want_gender, shard in plan:
        got = 0
        for row in iter_parquet_rows(shard, columns=["audio", "gender"], filesystem=fs):
            g = int(row["gender"])
            if g != want_gender:
                continue
            wav, sr = decode_audio(row["audio"]["bytes"])
            q = clip_quality(wav, sr)
            voice = smap.get(g, f"g{g}")
            d = per.setdefault(voice, {"dur": [], "snr": [], "bw": [], "clip": [], "nb": 0, "sr": sr})
            d["dur"].append(q.duration_s); d["snr"].append(q.snr_db); d["bw"].append(q.bandwidth_hz)
            d["clip"].append(q.clipping_ratio); d["nb"] += int(not q.is_effectively_wideband)
            got += 1
            if got >= target_clips_per_gender:
                break

    def pct(v, p):
        return float(np.percentile(v, p)) if v else float("nan")
    for voice, d in sorted(per.items()):
        nb = d["dur"] and d["nb"] / len(d["dur"])
        print(f"\n   {voice}  (n={len(d['dur'])}, native_sr={d['sr']})")
        print(f"      duration_s  mean={st.mean(d['dur']):.2f}  p10={pct(d['dur'],10):.2f}  "
              f"p90={pct(d['dur'],90):.2f}  min={min(d['dur']):.2f}  max={max(d['dur']):.2f}")
        print(f"      snr_db      mean={st.mean(d['snr']):.1f}  p10={pct(d['snr'],10):.1f}  "
              f"p90={pct(d['snr'],90):.1f}")
        print(f"      bandwidth   mean={st.mean(d['bw']):.0f}Hz  p10={pct(d['bw'],10):.0f}Hz  "
              f"min={min(d['bw']):.0f}Hz  narrowband_frac={nb:.2f}")
        print(f"      clipping    mean={st.mean(d['clip']):.4f}  max={max(d['clip']):.4f}")


if __name__ == "__main__":
    part_a_end_to_end()
    part_b_springlab_speaker_stats()
    print("\nVALIDATE_LOADER_OK")
