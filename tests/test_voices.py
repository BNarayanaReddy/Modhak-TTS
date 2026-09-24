"""Guard: the hardcoded VOICE_ROSTER must match reference/voices.md (no drift)."""
from __future__ import annotations

import re
from pathlib import Path

from indic_speak_ft.voices import MARATHI_VOICES, VOICE_NAMES, is_known_voice

REPO = Path(__file__).resolve().parents[1]


def _parse_voices_md() -> set[str]:
    """Extract the 44 names from the 'Recommended voices by language' table."""
    text = (REPO / "reference" / "voices.md").read_text(encoding="utf-8")
    names: set[str] = set()
    row = re.compile(r"^\|\s*[A-Za-z].*\([a-z]{2,3}\)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|")
    for line in text.splitlines():
        m = row.match(line)
        if m:
            names.add(m.group(1).strip())
            names.add(m.group(2).strip())
    return names


def test_roster_matches_reference_voices_md():
    parsed = _parse_voices_md()
    assert parsed == set(VOICE_NAMES), (
        f"roster drift vs voices.md: only-in-md={parsed - set(VOICE_NAMES)}, "
        f"only-in-code={set(VOICE_NAMES) - parsed}"
    )
    assert len(VOICE_NAMES) == 44


def test_marathi_target_voices():
    assert set(MARATHI_VOICES) == {"Anagha", "Chinmay"}


def test_known_voice_lookup():
    assert is_known_voice("Anagha") and is_known_voice("Amit")
    assert not is_known_voice("Nonexistent")
    assert not is_known_voice("anagha")  # case-sensitive
