"""Phase-2 eval-bucket tests: priority routing, disjointness, and hash-holdout."""
from __future__ import annotations

import json

from indic_speak_ft.data.buckets import (
    Candidate,
    assign_bucket,
    default_buckets,
    materialize_manifests,
)


def _c(hash_, **over) -> Candidate:
    base = {"source_hf_path": "ai4bharat/Rasa", "language_id": "mr", "speaker": "Chinmay",
            "text": "…", "text_normalized": "…", "num_digits": 0, "num_english_words": 0,
            "duration_s": 5.0, "clip_key": hash_, "rasa_style": None}
    base.update(over)
    return Candidate(**base)


BUCKETS = default_buckets({})


def test_priority_routing():
    # digits win over a STEM style (adversarial has priority)
    assert assign_bucket(_c("h1", num_digits=2, rasa_style="WIKI"), BUCKETS) == "marathi_adversarial"
    # proper-noun style → adversarial
    assert assign_bucket(_c("h2", rasa_style="PROPER NOUN"), BUCKETS) == "marathi_adversarial"
    # english words (no digits) → code_switch
    assert assign_bucket(_c("h3", num_english_words=3), BUCKETS) == "marathi_code_switch"
    # WIKI (no digits/english) → stem
    assert assign_bucket(_c("h4", rasa_style="WIKI"), BUCKETS) == "marathi_stem_seen_voice"
    # plain Rasa MR → rasa retention
    assert assign_bucket(_c("h5", rasa_style="UMANG"), BUCKETS) == "retention_rasa_marathi"
    # SPRINGLab plain MR → general
    assert assign_bucket(_c("h6", source_hf_path="SPRINGLab/IndicTTS_Marathi"), BUCKETS) \
        == "marathi_general_unseen_domain"
    # languages route to retention
    assert assign_bucket(_c("h7", language_id="hi", source_hf_path="SPRINGLab/IndicTTS-Hindi",
                            speaker="Amit"), BUCKETS) == "retention_hindi"
    assert assign_bucket(_c("h8", language_id="en", source_hf_path="SPRINGLab/IndicTTS-English",
                            speaker="Amit"), BUCKETS) == "retention_english"


def test_materialize_disjoint_and_holdout(tmp_path):
    cands = []
    # enough candidates to fill several buckets
    for i in range(20):
        cands.append(_c(f"adv{i}", num_digits=1))
        cands.append(_c(f"cs{i}", num_english_words=1))
        cands.append(_c(f"stem{i}", rasa_style="WIKI"))
        cands.append(_c(f"gen{i}", source_hf_path="SPRINGLab/IndicTTS_Marathi", speaker="Anagha"))
        cands.append(_c(f"hi{i}", language_id="hi", source_hf_path="SPRINGLab/IndicTTS-Hindi", speaker="Amit"))

    summary = materialize_manifests(cands, tmp_path, seed=1)

    # every referenced hash appears in exactly one bucket manifest
    seen: dict[str, str] = {}
    for f in tmp_path.glob("*.json"):
        if f.name in ("held_out_clip_keys.json", "_summary.json"):
            continue
        payload = json.loads(f.read_text())
        for it in payload["items"]:
            if it["clip_key"] is not None:
                assert it["clip_key"] not in seen, "clip appears in two buckets"
                seen[it["clip_key"]] = payload["bucket"]

    held = set(json.loads((tmp_path / "held_out_clip_keys.json").read_text()))
    assert held == set(seen)                      # held-out set == all used ref keys
    assert summary["held_out_clip_keys"] == len(held)
    assert summary["buckets"]["marathi_adversarial"] > 0


def test_longform_is_synthesis_only(tmp_path):
    # many MR sentences for one speaker so longform can reach length
    cands = [_c(f"g{i}", source_hf_path="SPRINGLab/IndicTTS_Marathi", speaker="Anagha",
                text_normalized="मराठी वाक्य क्रमांक " * 6) for i in range(60)]
    materialize_manifests(cands, tmp_path, seed=2)
    lf = json.loads((tmp_path / "marathi_longform.json").read_text())
    assert lf["needs_reference_audio"] is False
    if lf["items"]:
        it = lf["items"][0]
        assert it["clip_key"] is None and it["has_reference_audio"] is False
        assert it["target_seconds"] and it["target_seconds"] > 20
