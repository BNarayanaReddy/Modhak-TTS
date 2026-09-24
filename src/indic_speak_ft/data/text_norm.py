"""Marathi-aware text normalization: NFC + punctuation-preserving cleanup that never
lowercases, marks Latin-script (loanword/English) code-switch spans, and counts digits —
emitting ``text_normalized`` alongside the untouched ``text_raw`` (GOAL Phase 2 rules).

Numeric expansion (digits → words) is delegated to a pluggable backend; the Bodhan
``indic_normalizer`` Marathi engine is the intended one (it also handles STEM LaTeX /
chemistry). Until wired, digits are preserved and a TODO is recorded, per GOAL
("numeric expansion … where available; log a TODO if not"). Bump ``NORMALIZER_VERSION``
on any rule change — it is stamped into every clip's metadata.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field

NORMALIZER_VERSION = "mr-norm-v1"

# Devanagari block, incl. danda । (U+0964) and double danda ॥ (U+0965), which we PRESERVE.
_DEVANAGARI = r"ऀ-ॿ"
_DEVA_DIGITS = "०१२३४५६७८९"  # ०-९
_DEVA_RE = re.compile(f"[{_DEVANAGARI}]")
# A Latin "word" run (loanword / English), kept verbatim and marked as code-switch.
_LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’.\-]*")
_DIGIT_RE = re.compile(f"[0-9{_DEVA_DIGITS}]")
_WS_RUN_RE = re.compile(r"[^\S\n]+")  # horizontal whitespace runs (keep newlines)

# A number-expansion backend maps a digit string (+ language) to spoken words.
NumberBackend = Callable[[str, str], str]


@dataclass
class NormResult:
    text_normalized: str
    script: str                      # "Deva" if any Devanagari present, else "Latn"
    num_digits: int
    num_english_words: int
    code_switch_spans: list[dict]    # {start, end, text} offsets into text_normalized
    normalizer_version: str = NORMALIZER_VERSION
    todos: list[str] = field(default_factory=list)


def _clean(text: str) -> str:
    """NFC, strip control chars, collapse horizontal whitespace — no lowercasing, punctuation kept."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch == "\n")
    text = _WS_RUN_RE.sub(" ", text)
    return text.strip()


def normalize_text(
    text_raw: str,
    *,
    language_id: str = "mr",
    number_backend: NumberBackend | None = None,
) -> NormResult:
    """Normalize one transcript. ``text_raw`` is never mutated; the returned
    ``text_normalized`` is derived. Code-switch spans/offsets are computed against the
    FINAL normalized string. Pass ``number_backend`` to expand digits to words."""
    text = _clean(text_raw)

    todos: list[str] = []
    if number_backend is not None:
        # Expand each maximal digit run in place; offsets recomputed afterwards.
        def _expand(m: re.Match) -> str:
            try:
                return number_backend(m.group(0), language_id)
            except Exception:  # noqa: BLE001 — a backend bug must not drop the clip
                todos.append(f"number expansion failed for {m.group(0)!r}")
                return m.group(0)

        text = re.sub(r"[0-9]+", _expand, text)
    elif _DIGIT_RE.search(text):
        todos.append("digits present but no number_backend supplied — left verbatim "
                     "(wire indic_normalizer Marathi engine)")

    code_switch_spans = [
        {"start": m.start(), "end": m.end(), "text": m.group(0)}
        for m in _LATIN_WORD_RE.finditer(text)
    ]
    num_english_words = len(code_switch_spans)
    num_digits = len(_DIGIT_RE.findall(text))
    script = "Deva" if _DEVA_RE.search(text) else "Latn"

    return NormResult(
        text_normalized=text,
        script=script,
        num_digits=num_digits,
        num_english_words=num_english_words,
        code_switch_spans=code_switch_spans,
        todos=todos,
    )
