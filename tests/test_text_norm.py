"""Phase-2 text-normalization tests — Marathi cases (GOAL Phase 2)."""
from __future__ import annotations

import unicodedata

from indic_speak_ft.data.text_norm import NORMALIZER_VERSION, normalize_text


def test_nfc_applied():
    raw = "café"  # café, decomposed (e + combining acute) inside a code-switch word
    out = normalize_text(" हे " + raw + " आहे").text_normalized
    assert "café" in unicodedata.normalize("NFC", out)
    assert out == unicodedata.normalize("NFC", out)  # idempotent / already NFC


def test_danda_preserved():
    out = normalize_text("नमस्कार। कसे आहात॥").text_normalized
    assert "।" in out and "॥" in out


def test_no_lowercasing():
    r = normalize_text("मी Python शिकतो.")
    assert "Python" in r.text_normalized  # not 'python'


def test_code_switch_spans_offsets():
    r = normalize_text("मला Machine Learning आवडते.")
    words = [s["text"] for s in r.code_switch_spans]
    assert words == ["Machine", "Learning"]
    assert r.num_english_words == 2
    # offsets index back into the normalized text exactly
    for s in r.code_switch_spans:
        assert r.text_normalized[s["start"]:s["end"]] == s["text"]
    assert r.script == "Deva"


def test_digit_counting_ascii_and_devanagari():
    assert normalize_text("मी 10 वाजता येईन.").num_digits == 2          # ASCII
    assert normalize_text("मी १० वाजता येईन.").num_digits == 2          # Devanagari
    assert normalize_text("H2O हे पाणी आहे.").num_digits == 1


def test_number_backend_expands_and_records_todo_otherwise():
    # no backend -> digits kept, TODO recorded
    r = normalize_text("मी 2 सफरचंद खाल्ली.")
    assert "2" in r.text_normalized and any("number" in t for t in r.todos)
    # with a backend -> digits expanded, no leftover
    r2 = normalize_text("मी 2 सफरचंद खाल्ली.", number_backend=lambda d, lang: "दोन")
    assert "2" not in r2.text_normalized and "दोन" in r2.text_normalized


def test_raw_never_mutated_and_version_stamped():
    raw = "  हे   एक   वाक्य आहे।  "
    r = normalize_text(raw)
    assert raw == "  हे   एक   वाक्य आहे।  "          # caller's string untouched
    assert r.text_normalized == "हे एक वाक्य आहे।"    # collapsed + stripped, danda kept
    assert r.normalizer_version == NORMALIZER_VERSION


def test_latin_only_text_is_latn():
    assert normalize_text("hello world").script == "Latn"
