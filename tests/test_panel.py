"""Phase-6 panel-orchestration test with fake backends (hermetic — no models)."""
from __future__ import annotations

import json

import numpy as np

from indic_speak_ft.eval import report
from indic_speak_ft.eval.panel import run_panel
from indic_speak_ft.tokens import build_frame


def _manifests():
    def wer_bucket(name, texts):
        return {"bucket": name, "items": [
            {"id": f"{name}_{i}", "text": t, "text_normalized": t, "speaker": "Anagha"}
            for i, t in enumerate(texts)]}
    return {
        "retention_hindi": wer_bucket("retention_hindi", ["नमस्ते दुनिया", "आज मौसम अच्छा है"]),
        "marathi_adversarial": wer_bucket("marathi_adversarial", ["पाणी हे रेणू", "दोन अणू"]),
        "marathi_longform": {"bucket": "marathi_longform", "items": [
            {"id": "lf_0", "text": "लांब वाक्य " * 20, "text_normalized": "लांब वाक्य", "speaker": "Chinmay"}]},
    }


def _fake_synth(text, speaker):
    # 3 frames of audio ids (valid band), silence wav, no max-hit
    ids = [t for i in range(3) for t in build_frame(i, i, i, i, i, i, i, base=128266)]
    return np.zeros(24000, np.float32), ids, False


def test_run_panel_produces_wer_and_longform(tmp_path):
    manifests = _manifests()

    # a fake ASR that returns a fixed hypothesis regardless (imperfect) → nonzero WER
    class FixedASR:
        def transcribe(self, wav, sr, language=None):
            return "काहीतरी वेगळे"

    results = run_panel(manifests, _fake_synth, FixedASR(), out_dir=str(tmp_path), seed=0)
    b = results["buckets"]
    assert set(b) == {"retention_hindi", "marathi_adversarial", "marathi_longform"}
    # WER buckets carry ins/del/sub + a CI
    assert "wer" in b["retention_hindi"] and "wer_ci95" in b["retention_hindi"]
    assert {"insertions", "deletions", "substitutions"} <= set(b["retention_hindi"])
    # longform carries repetition + hit-rate
    assert "repetition_rate_3gram" in b["marathi_longform"]
    assert (tmp_path / "panel.json").is_file()
    assert (tmp_path / "eval_outputs" / "retention_hindi_hyps.json").is_file()


def test_report_html_written(tmp_path):
    results = run_panel(_manifests(), _fake_synth,
                        type("A", (), {"transcribe": lambda self, w, s, language=None: "x"})(),
                        out_dir=str(tmp_path), seed=1)
    out = report.write_html(results, str(tmp_path / "panel.html"), title="T", subtitle="S")
    html = (tmp_path / "panel.html").read_text()
    assert out.endswith("panel.html")
    assert "<table" in html and "WER buckets" in html and "Longform" in html
    json.loads((tmp_path / "panel.json").read_text())  # valid JSON
