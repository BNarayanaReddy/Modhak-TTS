"""Phase-2 schema tests: hard-reject validators and auto-fill behavior."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from indic_speak_ft.data.schema import (
    AudioMeta,
    ClipMetadata,
    DedupMeta,
    Provenance,
    SpeakerMeta,
    TextFeatures,
    Transcript,
)


def _clip(**over):
    base = {
        "provenance": Provenance(source_id="SPRINGLab/IndicTTS_Marathi#1", license="CC-BY-4.0"),
        "audio": AudioMeta(sample_rate=24000, duration_s=3.2, snr_estimate=28.0, clipping_ratio=0.0),
        "transcript": Transcript(text_raw="हे एक वाक्य आहे.", text_normalized="हे एक वाक्य आहे.",
                                 normalizer_version="mr-norm-v1"),
        "text": TextFeatures(language_id="mr", script="Deva", domain="general"),
        "speaker": SpeakerMeta(name="Anagha"),
        "dedup": DedupMeta(audio_hash="a" * 16, transcript_shingle_hash="b" * 16),
    }
    base.update(over)
    return ClipMetadata(**base)


def test_valid_clip_and_gender_autofill():
    clip = _clip()
    assert clip.speaker.name == "Anagha"
    assert clip.speaker.gender == "F"  # auto-filled from the roster


def test_unknown_speaker_is_hard_reject():
    with pytest.raises(ValidationError, match="not in the 44-voice library"):
        SpeakerMeta(name="Nobody")


def test_gender_mismatch_rejected():
    with pytest.raises(ValidationError, match="disagrees with library roster"):
        SpeakerMeta(name="Anagha", gender="M")


def test_ranges_enforced():
    with pytest.raises(ValidationError):
        AudioMeta(sample_rate=24000, duration_s=-1.0)
    with pytest.raises(ValidationError):
        AudioMeta(sample_rate=24000, duration_s=1.0, clipping_ratio=1.5)


def test_english_must_be_latn():
    with pytest.raises(ValidationError, match="script 'Latn'"):
        _clip(text=TextFeatures(language_id="en", script="Deva"))
