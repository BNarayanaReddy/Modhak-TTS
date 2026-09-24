"""Run the metric vector over the locked eval buckets → a results dict (+ persisted
hypotheses), for any checkpoint. WER buckets get insertions/deletions/substitutions and a 95%
bootstrap CI over utterances; the longform bucket gets repetition + max-hit rates; voice
buckets additionally get calibrated speaker similarity when a reference-audio retriever and an
embedder are supplied. Orchestration is backend-pluggable, so it is testable with fakes.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from indic_speak_ft.eval.asr import ASRBackend, persist_hypotheses
from indic_speak_ft.eval.longform import longform_metrics
from indic_speak_ft.eval.metrics import bootstrap_ci, per_utterance_wer, wer_measures
from indic_speak_ft.eval.prosody import prosody_vector
from indic_speak_ft.eval.speaker import mean_pairwise_similarity

SynthesizeFn = Callable[[str, str], tuple[Any, list[int], bool]]
ReferenceAudioFn = Callable[[dict], np.ndarray | None]
EmbedFn = Callable[[np.ndarray, int], np.ndarray]

VOICE_BUCKETS = {"marathi_stem_seen_voice", "retention_rasa_marathi"}


def load_manifests(manifests_dir: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in sorted(Path(manifests_dir).glob("*.json")):
        if p.name.startswith("_") or p.name.startswith("held_out"):
            continue
        payload = json.loads(p.read_text())
        if "items" in payload:
            out[payload["bucket"]] = payload
    return out


def run_panel(
    manifests: dict[str, dict],
    synthesize_fn: SynthesizeFn,
    asr: ASRBackend,
    *,
    out_dir: str,
    embed_fn: EmbedFn | None = None,
    reference_audio_fn: ReferenceAudioFn | None = None,
    sr: int = 24000,
    seed: int = 0,
) -> dict:
    results: dict[str, Any] = {"buckets": {}}
    for bucket, payload in manifests.items():
        items = payload["items"]
        if not items:
            results["buckets"][bucket] = {"n": 0, "note": "empty bucket"}
            continue

        syn = [synthesize_fn(it["text"], it["speaker"]) for it in items]
        wavs = [s[0] for s in syn]
        hit_rate = float(np.mean([1.0 if s[2] else 0.0 for s in syn]))

        if bucket == "marathi_longform":
            per = [longform_metrics(s[1], hit_max_new_tokens=s[2]) for s in syn]
            results["buckets"][bucket] = {
                "n": len(items),
                "repetition_rate_3gram": float(np.mean([m["repetition_rate_3gram"] for m in per])),
                "repetition_rate_5gram": float(np.mean([m["repetition_rate_5gram"] for m in per])),
                "hit_max_new_tokens_rate": hit_rate,
                "mean_frames": float(np.mean([m["n_frames"] for m in per])),
            }
            continue

        refs = [it["text_normalized"] for it in items]
        hyps = [asr.transcribe(w, sr, language=it.get("language_id")) for w, it in zip(wavs, items)]
        persist_hypotheses(f"{out_dir}/eval_outputs", bucket, refs, hyps, [it["id"] for it in items])
        point, lo, hi = bootstrap_ci(per_utterance_wer(refs, hyps), seed=seed)
        entry = {"n": len(items), "wer": point, "wer_ci95": [lo, hi],
                 "hit_max_new_tokens_rate": hit_rate, **wer_measures(refs, hyps)}

        if bucket in VOICE_BUCKETS and embed_fn is not None and reference_audio_fn is not None:
            refs_audio = [reference_audio_fn(it) for it in items]
            triples = [(w, r, it["speaker"]) for w, r, it in zip(wavs, refs_audio, items) if r is not None]
            if triples:
                entry["speaker_similarity"] = mean_pairwise_similarity(
                    embed_fn, [t[0] for t in triples], [t[1] for t in triples], sr=sr)
                # Per-voice breakdown (D15 tightening #2): monitor BOTH target voices separately,
                # not the average — e.g. retention_rasa carries Anagha + Chinmay anchor references.
                by_voice = {
                    voice: mean_pairwise_similarity(
                        embed_fn, [t[0] for t in vt], [t[1] for t in vt], sr=sr)
                    for voice in sorted({t[2] for t in triples})
                    if (vt := [t for t in triples if t[2] == voice])
                }
                if len(by_voice) > 1:
                    entry["speaker_similarity_by_voice"] = by_voice
                entry["prosody_generated_mean"] = {
                    k: float(np.mean([v for v in (prosody_vector(w, sr).get(k) for w in wavs) if v is not None]))
                    for k in ("mean_f0", "f0_std", "energy_std", "pause_ratio")}
        results["buckets"][bucket] = entry

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "panel.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    return results
