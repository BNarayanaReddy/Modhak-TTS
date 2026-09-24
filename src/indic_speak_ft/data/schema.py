"""Pydantic per-clip metadata: the typed contract every training/eval clip is validated
against before it enters the pipeline (Decision-1 closed-voice constraints enforced here).

Field groups mirror docs/DATA.md §2. Structural invariants that are cheap and always
true (speaker ∈ library, ranges, enum values) are validated here; policy thresholds that
live in configs/data.yaml (SNR floor, duration window, clipping cap) are applied by the
loader's gates, not baked into the schema.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from indic_speak_ft.voices import VOICE_ROSTER, is_known_voice

LanguageId = Literal["mr", "hi", "en"]
Script = Literal["Deva", "Latn"]
Domain = Literal["stem", "general"]
Gender = Literal["F", "M"]


class Provenance(BaseModel):
    """Where the clip came from and under what license (license 'unknown' is a hard reject)."""

    source_id: str = Field(..., description="dataset id + row key, e.g. 'SPRINGLab/IndicTTS_Marathi#12345'")
    license: str = Field(..., description="SPDX-ish license string; 'unknown' fails the license gate")


class AudioMeta(BaseModel):
    sample_rate: int = Field(..., gt=0)
    duration_s: float = Field(..., gt=0.0)
    snr_estimate: float | None = Field(None, description="dB; None if not measured (soft gate)")
    clipping_ratio: float = Field(0.0, ge=0.0, le=1.0, description="fraction of samples at full-scale")


class Transcript(BaseModel):
    text_raw: str = Field(..., min_length=1, description="never overwritten")
    text_normalized: str = Field(..., description="output of text_norm; may be empty → hard reject at gate")
    normalizer_version: str = Field(..., description="text_norm ruleset version stamped at compile time")


class CodeSwitchSpan(BaseModel):
    """A Latin-script (loanword/English) span inside otherwise-Devanagari text."""

    start: int = Field(..., ge=0, description="char offset into text_normalized")
    end: int = Field(..., gt=0)
    text: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _end_after_start(self) -> CodeSwitchSpan:
        if self.end <= self.start:
            raise ValueError(f"code-switch span end ({self.end}) must be > start ({self.start})")
        return self


class TextFeatures(BaseModel):
    language_id: LanguageId
    script: Script
    num_digits: int = Field(0, ge=0)
    num_english_words: int = Field(0, ge=0)
    code_switch_spans: list[CodeSwitchSpan] = Field(default_factory=list)
    domain: Domain = "general"


class SpeakerMeta(BaseModel):
    """Speaker must be one of the 44 library voices (closed set — Decision 1)."""

    name: str
    gender: Gender | None = None  # auto-filled from the roster if omitted

    @model_validator(mode="after")
    def _known_and_consistent(self) -> SpeakerMeta:
        if not is_known_voice(self.name):
            raise ValueError(
                f"speaker {self.name!r} is not in the 44-voice library (closed set); "
                f"hard reject. No new voices are added in this project."
            )
        roster_gender = VOICE_ROSTER[self.name][1]
        if self.gender is None:
            object.__setattr__(self, "gender", roster_gender)
        elif self.gender != roster_gender:
            raise ValueError(
                f"speaker {self.name!r} gender {self.gender!r} disagrees with library "
                f"roster {roster_gender!r}"
            )
        return self


class StyleMeta(BaseModel):
    """Prosody stats — METADATA ONLY. Decision 6: style is scoped out; we never condition
    generation on these (no style prompt is built from f0)."""

    mean_f0: float | None = Field(None, ge=0.0)
    f0_std: float | None = Field(None, ge=0.0)


class DedupMeta(BaseModel):
    audio_hash: str = Field(..., description="hash of the decoded PCM; collides → drop / eval-leak check")
    transcript_shingle_hash: str = Field(..., description="minhash/shingle of text_normalized")


class ClipMetadata(BaseModel):
    """One clip's full metadata record (docs/DATA.md §2)."""

    provenance: Provenance
    audio: AudioMeta
    transcript: Transcript
    text: TextFeatures
    speaker: SpeakerMeta
    style: StyleMeta = Field(default_factory=StyleMeta)
    dedup: DedupMeta

    @model_validator(mode="after")
    def _script_language_consistency(self) -> ClipMetadata:
        # Devanagari clips (mr/hi) should carry a Deva script tag; en is Latn.
        if self.text.language_id == "en" and self.text.script != "Latn":
            raise ValueError("English clip must have script 'Latn'")
        return self
