"""Materialize + LOCK the 8 eval-bucket manifests into configs/eval_manifests/.

Reads only light columns (audio.path, text, gender, style, duration) — never the audio
bytes — so building the lock is cheap. Each selected clip is held out from training via a
path-based ``clip_key`` (loader.eval_clip_key), written to held_out_clip_keys.json.

Sources (held out from the training pools):
  - Rasa MR test split  → stem / adversarial / code-switch / rasa-retention (Anagha/Chinmay)
  - SPRINGLab MR        → general_unseen_domain (Anagha/Chinmay by gender)
  - SPRINGLab Hindi     → retention_hindi (voice forced to Amit — the model-card API example)
  - SPRINGLab English   → retention_english (Amit)

Run:  Modhak-TTS/.venv/bin/python scripts/build_eval_buckets.py
"""
from __future__ import annotations

from pathlib import Path

import yaml
from huggingface_hub import HfFileSystem

from indic_speak_ft.data.buckets import Candidate, materialize_manifests
from indic_speak_ft.data.loader import assign_speaker, eval_clip_key, iter_parquet_rows
from indic_speak_ft.data.text_norm import normalize_text

REPO = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((REPO / "configs" / "data.yaml").read_text())
OUT = REPO / "configs" / "eval_manifests"
fs = HfFileSystem()

# (source_hf_path, language_id, shard hf path, light columns, force_speaker) per read
READS = [
    ("ai4bharat/Rasa", "mr", "datasets/ai4bharat/Rasa/Marathi/test-00000-of-00007.parquet",
     ["audio.path", "text", "gender", "style", "duration"], None),
    ("ai4bharat/Rasa", "mr", "datasets/ai4bharat/Rasa/Marathi/test-00001-of-00007.parquet",
     ["audio.path", "text", "gender", "style", "duration"], None),
    ("SPRINGLab/IndicTTS_Marathi", "mr", "datasets/SPRINGLab/IndicTTS_Marathi/data/train-00000-of-00004.parquet",
     ["audio.path", "text", "gender"], None),   # female → Anagha
    ("SPRINGLab/IndicTTS_Marathi", "mr", "datasets/SPRINGLab/IndicTTS_Marathi/data/train-00003-of-00004.parquet",
     ["audio.path", "text", "gender"], None),   # male → Chinmay
    ("SPRINGLab/IndicTTS-Hindi", "hi", "datasets/SPRINGLab/IndicTTS-Hindi/data/train-00000-of-00010.parquet",
     ["audio.path", "text"], "Amit"),           # retention_hindi uses Amit (GOAL/Decision 3)
    ("SPRINGLab/IndicTTS-English", "en", "datasets/SPRINGLab/IndicTTS-English/data/train-00000-of-00028.parquet",
     ["audio.path", "text"], "Amit"),
]
MAX_ROWS_PER_SHARD = 700


def build_candidates() -> list[Candidate]:
    smap = CFG["speaker_assignment"]
    cands: list[Candidate] = []
    for source, lang, shard, cols, force_speaker in READS:
        shard_base = shard.split("/")[-1]
        n = 0
        for row in iter_parquet_rows(shard, columns=cols, max_rows=MAX_ROWS_PER_SHARD, filesystem=fs):
            text = (row.get("text") or "").strip()
            if not text:
                continue
            speaker = force_speaker or assign_speaker(source, row, smap)
            norm = normalize_text(text, language_id=lang)
            if not norm.text_normalized.strip():
                continue
            dur = float(row["duration"]) if row.get("duration") else 0.0
            cands.append(Candidate(
                source_hf_path=source, language_id=lang, speaker=speaker, text=text,
                text_normalized=norm.text_normalized, num_digits=norm.num_digits,
                num_english_words=norm.num_english_words, duration_s=dur,
                clip_key=eval_clip_key(source, row), rasa_style=row.get("style"),
                shard=shard_base, row_index=int(row.get("_idx", n))))
            n += 1
        print(f"  read {n:4d} candidates from {source} :: {shard_base}")
    return cands


def main() -> None:
    print("Building eval-bucket candidates (light columns only, no audio bytes)…")
    cands = build_candidates()
    print(f"total candidates: {len(cands)}")
    summary = materialize_manifests(cands, OUT, seed=1234)
    print("\nLOCKED manifests ->", OUT)
    for bucket, count in summary["buckets"].items():
        print(f"   {bucket:32s} {count}")
    print(f"   held-out clip keys: {summary['held_out_clip_keys']}")
    print("BUILD_EVAL_BUCKETS_OK")


if __name__ == "__main__":
    main()
