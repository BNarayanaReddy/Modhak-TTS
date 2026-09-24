"""Re-lock amendment (DECISIONS.md D17 CONFLICT): add an Anagha Rasa anchor to `retention_rasa`.

The original locked buckets drew Rasa only from test0/1, which are both Male → the
`retention_rasa_marathi` bucket was Chinmay-only, so Anagha's cross-lingual anchor retention
(D15 tightening #2) could not be measured against the true Rasa anchor. This script surgically
adds ~30 *general* (non-stem/adversarial/code-switch) Anagha clips from Rasa **test6** — a
Female/Anagha shard that the full training run never touched (training used test2-5), so there is
zero train/eval leakage — and updates ONLY `retention_rasa_marathi.json`, `held_out_clip_keys.json`,
and `_summary.json`. The other seven locked buckets are left byte-identical. Idempotent: re-running
after the Anagha items exist is a no-op.

Run:  Modhak-TTS/.venv/bin/python scripts/relock_add_anagha_anchor.py
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import yaml
from huggingface_hub import HfFileSystem

from indic_speak_ft.data.buckets import Candidate, EvalItem, assign_bucket, default_buckets
from indic_speak_ft.data.loader import assign_speaker, eval_clip_key
from indic_speak_ft.data.text_norm import normalize_text

REPO = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((REPO / "configs" / "data.yaml").read_text())
OUT = REPO / "configs" / "eval_manifests"
BUCKET = "retention_rasa_marathi"
# Anagha (Female) Rasa shard NOT used by the training run (FULL_PLAN uses test2-5) → no leakage.
SHARD = "datasets/ai4bharat/Rasa/Marathi/test-00006-of-00007.parquet"
COLS = ["audio.path", "text", "gender", "style", "duration"]
ADD_COUNT = 30
SEED = 1234


def build_anagha_candidates() -> list[Candidate]:
    fs = HfFileSystem()
    smap = CFG["speaker_assignment"]
    buckets = default_buckets({})
    from indic_speak_ft.data.loader import iter_parquet_rows

    cands: list[Candidate] = []
    for row in iter_parquet_rows(SHARD, columns=COLS, max_rows=500, filesystem=fs):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        speaker = assign_speaker("ai4bharat/Rasa", row, smap)
        if speaker != "Anagha":  # this shard is Female, but guard anyway
            continue
        norm = normalize_text(text, language_id="mr")
        if not norm.text_normalized.strip():
            continue
        c = Candidate(
            source_hf_path="ai4bharat/Rasa", language_id="mr", speaker=speaker, text=text,
            text_normalized=norm.text_normalized, num_digits=norm.num_digits,
            num_english_words=norm.num_english_words,
            duration_s=float(row["duration"]) if row.get("duration") else 0.0,
            clip_key=eval_clip_key("ai4bharat/Rasa", row), rasa_style=row.get("style"),
            shard=SHARD.split("/")[-1], row_index=int(row.get("_idx", 0)))
        # Keep only clips that route to retention (general MR) — same semantics as the Chinmay half.
        if assign_bucket(c, buckets) == BUCKET:
            cands.append(c)
    return cands


def main() -> None:
    manifest_path = OUT / f"{BUCKET}.json"
    manifest = json.loads(manifest_path.read_text())
    existing_speakers = {it["speaker"] for it in manifest["items"]}
    if "Anagha" in existing_speakers:
        print(f"Anagha anchor already present in {BUCKET} ({manifest['count']} items) — no-op.")
        return

    cands = build_anagha_candidates()
    print(f"eligible Anagha retention candidates from {SHARD.split('/')[-1]}: {len(cands)}")
    random.Random(SEED).shuffle(cands)
    chosen = cands[:ADD_COUNT]
    if not chosen:
        raise RuntimeError("no eligible Anagha Rasa candidates found — cannot re-lock")

    start = manifest["count"]
    new_items = [asdict(EvalItem(
        id=f"{BUCKET}_{start + i:03d}", bucket=BUCKET, language_id="mr", speaker=c.speaker,
        text=c.text, text_normalized=c.text_normalized, source_id="ai4bharat/Rasa",
        clip_key=c.clip_key, has_reference_audio=True, shard=c.shard, row_index=c.row_index,
        rasa_style=c.rasa_style)) for i, c in enumerate(chosen)]

    manifest["items"] += new_items
    manifest["count"] = len(manifest["items"])
    manifest["description"] += " [re-locked: +Anagha anchor from test6, D17]"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    # held-out keys: union the new Anagha clip keys (so future runs exclude them too)
    hk_path = OUT / "held_out_clip_keys.json"
    keys = set(json.loads(hk_path.read_text())) | {c.clip_key for c in chosen}
    hk_path.write_text(json.dumps(sorted(keys), ensure_ascii=False, indent=2))

    # summary
    sp = OUT / "_summary.json"
    summary = json.loads(sp.read_text())
    summary["buckets"][BUCKET] = manifest["count"]
    summary["held_out_clip_keys"] = len(keys)
    summary["relock_d17"] = {"added_anagha": len(new_items), "shard": SHARD.split("/")[-1]}
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    chinmay = sum(1 for it in manifest["items"] if it["speaker"] == "Chinmay")
    anagha = sum(1 for it in manifest["items"] if it["speaker"] == "Anagha")
    print(f"{BUCKET}: {manifest['count']} items now (Chinmay {chinmay} / Anagha {anagha})")
    print(f"held_out_clip_keys: {len(keys)}")
    print("RELOCK_ADD_ANAGHA_OK")


if __name__ == "__main__":
    main()
