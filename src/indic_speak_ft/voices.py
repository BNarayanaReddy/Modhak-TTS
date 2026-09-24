"""The closed 44-voice library (name → gender, native language).

Source of truth is ``reference/voices.md``; the roster is duplicated here as a frozen
mapping for fast, dependency-free lookup (the schema's speaker gate needs it per clip).
``tests/test_voices.py`` re-parses ``voices.md`` and asserts the two never drift.

Per Decision 1/GOAL the speaker vocabulary is CLOSED — no new voices — so a name not in
this roster is a hard reject at data time.
"""
from __future__ import annotations

from typing import Literal

Gender = Literal["F", "M"]

# name -> (native_language_code, gender). 22 languages × {female, male} = 44 voices.
VOICE_ROSTER: dict[str, tuple[str, Gender]] = {
    # Assamese
    "Prastuti": ("as", "F"), "Ankur": ("as", "M"),
    # Bengali
    "Ishita": ("bn", "F"), "Sourav": ("bn", "M"),
    # Bodo
    "Gwrbw": ("brx", "F"), "Sansuma": ("brx", "M"),
    # Dogri
    "Preeti": ("doi", "F"), "Sham": ("doi", "M"),
    # Gujarati
    "Dhara": ("gu", "F"), "Parth": ("gu", "M"),
    # Hindi
    "Kavya": ("hi", "F"), "Amit": ("hi", "M"),
    # Kannada
    "Deepika": ("kn", "F"), "Adarsh": ("kn", "M"),
    # Kashmiri
    "Zoon": ("ks", "F"), "Ishfaq": ("ks", "M"),
    # Konkani
    "Anjali": ("kok", "F"), "Sandeep": ("kok", "M"),
    # Maithili
    "Vaidehi": ("mai", "F"), "Madhukar": ("mai", "M"),
    # Malayalam
    "Lakshmi": ("ml", "F"), "Kiran": ("ml", "M"),
    # Manipuri
    "Thoibi": ("mni", "F"), "Chaoba": ("mni", "M"),
    # Marathi (the project's target voices)
    "Anagha": ("mr", "F"), "Chinmay": ("mr", "M"),
    # Nepali
    "Srijana": ("ne", "F"), "Sagar": ("ne", "M"),
    # Odia
    "Itishree": ("or", "F"), "Akash": ("or", "M"),
    # Punjabi
    "Kaur": ("pa", "F"), "Manpreet": ("pa", "M"),
    # Sanskrit
    "Bharati": ("sa", "F"), "Aryaman": ("sa", "M"),
    # Santali
    "Phulmani": ("sat", "F"), "Sibu": ("sat", "M"),
    # Sindhi
    "Moomal": ("sd", "F"), "Rano": ("sd", "M"),
    # Tamil
    "Anitha": ("ta", "F"), "Arun": ("ta", "M"),
    # Telugu
    "Sravani": ("te", "F"), "Vamsi": ("te", "M"),
    # Urdu
    "Saba": ("ur", "F"), "Zaid": ("ur", "M"),
}

VOICE_NAMES: frozenset[str] = frozenset(VOICE_ROSTER)

# The project's target voices (Decision 1): Marathi natives.
MARATHI_VOICES: tuple[str, ...] = tuple(n for n, (lang, _) in VOICE_ROSTER.items() if lang == "mr")


def is_known_voice(name: str) -> bool:
    """True iff ``name`` is one of the 44 library voices (exact match, case-sensitive)."""
    return name in VOICE_NAMES


def voice_gender(name: str) -> Gender:
    """Gender label for a library voice; raises ``KeyError`` for an unknown name."""
    return VOICE_ROSTER[name][1]


def voice_native_language(name: str) -> str:
    """Native-language code (e.g. ``"mr"``) for a library voice."""
    return VOICE_ROSTER[name][0]
