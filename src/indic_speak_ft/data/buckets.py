"""Locked eval-bucket definitions + manifest materialization (GOAL Phase 2).

Eight buckets test distinct failure surfaces. Each eval clip is HELD OUT from training:
materialization records every selected clip's ``audio_hash`` so the loader's
``reject_eval_bucket_audio_hash`` gate drops it from the training stream — no eval leak.
Buckets are disjoint (a clip lands in exactly one, by priority routing) and the whole set
is seeded/reproducible, so the committed manifests are a fixed reference.

This module is pure routing + manifest writing (testable without data); the streaming reads
that produce candidates live in ``scripts/build_eval_buckets.py``.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Rasa `style` tags we treat as STEM-adjacent (encyclopedic/technical) vs adversarial content.
STEM_STYLES = frozenset({"WIKI", "INDIC"})
ADVERSARIAL_STYLES = frozenset({"NAMES", "PROPER NOUN", "PROPER_NOUN"})


@dataclass(frozen=True)
class Candidate:
    """A normalized eval candidate (one dataset row + its computed text features)."""

    source_hf_path: str
    language_id: str            # mr / hi / en
    speaker: str                # library voice (already mapped by D15)
    text: str
    text_normalized: str
    num_digits: int
    num_english_words: int
    duration_s: float
    clip_key: str                  # stable holdout key = sha16(source::audio.path); cheap, no bytes
    rasa_style: str | None = None
    shard: str | None = None       # source parquet basename (retrieval locator)
    row_index: int | None = None   # row index within that shard


@dataclass(frozen=True)
class EvalItem:
    id: str
    bucket: str
    language_id: str
    speaker: str
    text: str
    text_normalized: str
    source_id: str
    clip_key: str | None          # holdout key; None for synthesis-only items (longform)
    has_reference_audio: bool
    shard: str | None = None
    row_index: int | None = None
    rasa_style: str | None = None
    target_seconds: float | None = None  # for longform synthesis targets


@dataclass(frozen=True)
class BucketSpec:
    name: str
    description: str
    is_retention: bool
    needs_reference_audio: bool
    eligible: Callable[[Candidate], bool]
    target_count: int


# --- routing predicates -----------------------------------------------------------
def _is_mr(c: Candidate) -> bool: return c.language_id == "mr"
def _has_digits(c: Candidate) -> bool: return c.num_digits > 0
def _has_code_switch(c: Candidate) -> bool: return c.num_english_words > 0
def _rasa(c: Candidate) -> bool: return c.source_hf_path == "ai4bharat/Rasa"
def _springlab_mr(c: Candidate) -> bool: return c.source_hf_path == "SPRINGLab/IndicTTS_Marathi"
def _style_in(c: Candidate, s: frozenset[str]) -> bool:
    st = c.rasa_style
    return st is not None and st.strip().upper() in s


def default_buckets(sizes: dict[str, int]) -> list[BucketSpec]:
    """The 8 locked buckets. ``sizes`` overrides per-bucket target counts."""
    def n(name: str, default: int) -> int:
        return int(sizes.get(name, default))
    return [
        BucketSpec("marathi_adversarial",
                   "MR numbers/dates/proper-nouns/STEM terms — the documented weak surface",
                   False, True, lambda c: _is_mr(c) and (_has_digits(c) or _style_in(c, ADVERSARIAL_STYLES)),
                   n("marathi_adversarial", 40)),
        BucketSpec("marathi_code_switch",
                   "MR with English loanwords / code-switch spans",
                   False, True, lambda c: _is_mr(c) and _has_code_switch(c) and not _has_digits(c),
                   n("marathi_code_switch", 30)),
        BucketSpec("marathi_stem_seen_voice",
                   "Anagha/Chinmay reading STEM-adjacent (Rasa WIKI/INDIC) content",
                   False, True, lambda c: _rasa(c) and _is_mr(c) and _style_in(c, STEM_STYLES),
                   n("marathi_stem_seen_voice", 40)),
        BucketSpec("retention_rasa_marathi",
                   "Rasa MR (Anagha/Chinmay) general — cross-lingual voice/anchor retention",
                   True, True, lambda c: _rasa(c) and _is_mr(c),
                   n("retention_rasa_marathi", 30)),
        BucketSpec("marathi_general_unseen_domain",
                   "SPRINGLab general MR (non-STEM) — quality check on the training domain",
                   False, True, _springlab_mr,
                   n("marathi_general_unseen_domain", 40)),
        BucketSpec("retention_hindi",
                   "Hindi (Amit) — highest forgetting risk (linguistically closest)",
                   True, True, lambda c: c.language_id == "hi",
                   n("retention_hindi", 40)),
        BucketSpec("retention_english",
                   "English (Amit) — general Llama backbone health",
                   True, True, lambda c: c.language_id == "en",
                   n("retention_english", 30)),
    ]


# priority order for disjoint routing (first eligible bucket wins)
_PRIORITY = [
    "marathi_adversarial", "marathi_code_switch", "marathi_stem_seen_voice",
    "retention_rasa_marathi", "marathi_general_unseen_domain",
    "retention_hindi", "retention_english",
]


def assign_bucket(cand: Candidate, buckets: list[BucketSpec]) -> str | None:
    """Route a candidate to exactly one bucket by priority, or None if it fits none."""
    by_name = {b.name: b for b in buckets}
    for name in _PRIORITY:
        b = by_name.get(name)
        if b is not None and b.eligible(cand):
            return name
    return None


def materialize_manifests(
    candidates: list[Candidate],
    out_dir: Path,
    *,
    sizes: dict[str, int] | None = None,
    seed: int = 1234,
    longform_target_seconds: float = 45.0,
    longform_count: int = 12,
    chars_per_second: float = 12.0,
) -> dict[str, Any]:
    """Fill buckets from ``candidates`` (disjoint, seeded), synthesize the longform bucket
    from leftover MR texts, and write one JSON manifest per bucket + ``held_out_hashes.json``.
    Returns a summary dict (counts per bucket, total held-out hashes)."""
    import random

    rng = random.Random(seed)
    buckets = default_buckets(sizes or {})
    pools: dict[str, list[Candidate]] = {b.name: [] for b in buckets}
    for c in candidates:
        name = assign_bucket(c, buckets)
        if name is not None:
            pools[name].append(c)

    out_dir.mkdir(parents=True, exist_ok=True)
    held_out: set[str] = set()
    summary: dict[str, Any] = {"seed": seed, "buckets": {}}
    used_mr_texts: list[Candidate] = []

    for b in buckets:
        pool = pools[b.name]
        rng.shuffle(pool)
        chosen = pool[: b.target_count]
        items = []
        for i, c in enumerate(chosen):
            held_out.add(c.clip_key)
            items.append(EvalItem(
                id=f"{b.name}_{i:03d}", bucket=b.name, language_id=c.language_id, speaker=c.speaker,
                text=c.text, text_normalized=c.text_normalized, source_id=c.source_hf_path,
                clip_key=c.clip_key, has_reference_audio=True, shard=c.shard,
                row_index=c.row_index, rasa_style=c.rasa_style))
            if c.language_id == "mr":
                used_mr_texts.append(c)
        _write_manifest(out_dir, b, items)
        summary["buckets"][b.name] = len(items)

    # longform (synthesis-only): concatenate leftover MR held-out texts to ~target seconds
    longform_items = _build_longform(used_mr_texts, rng, longform_target_seconds,
                                     longform_count, chars_per_second)
    _write_manifest(out_dir, _LONGFORM_SPEC, longform_items)
    summary["buckets"]["marathi_longform"] = len(longform_items)

    (out_dir / "held_out_clip_keys.json").write_text(
        json.dumps(sorted(held_out), ensure_ascii=False, indent=2))
    summary["held_out_clip_keys"] = len(held_out)
    (out_dir / "_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


_LONGFORM_SPEC = BucketSpec(
    "marathi_longform", "30–60 s Marathi syntheses — repetition/drift over length",
    False, False, _is_mr, 12)


def _build_longform(mr: list[Candidate], rng, target_s: float, count: int, cps: float) -> list[EvalItem]:
    """Concatenate held-out MR sentences (grouped by speaker) into ~target_s prompts."""
    target_chars = target_s * cps
    by_speaker: dict[str, list[Candidate]] = {}
    for c in mr:
        by_speaker.setdefault(c.speaker, []).append(c)
    items: list[EvalItem] = []
    for speaker, cs in by_speaker.items():
        rng.shuffle(cs)
        i = 0
        while i < len(cs) and len(items) < count:
            chunk, chars = [], 0.0
            while i < len(cs) and chars < target_chars:
                chunk.append(cs[i].text_normalized); chars += len(cs[i].text_normalized); i += 1
            if chars >= target_chars * 0.6:  # only keep if it reaches a real long-form length
                text = " ".join(chunk)
                items.append(EvalItem(
                    id=f"marathi_longform_{len(items):03d}", bucket="marathi_longform",
                    language_id="mr", speaker=speaker, text=text, text_normalized=text,
                    source_id="synthesized_from_heldout_mr", clip_key=None,
                    has_reference_audio=False, target_seconds=round(chars / cps, 1)))
    return items


def _write_manifest(out_dir: Path, spec: BucketSpec, items: list[EvalItem]) -> None:
    payload = {
        "bucket": spec.name,
        "description": spec.description,
        "is_retention": spec.is_retention,
        "needs_reference_audio": spec.needs_reference_audio,
        "count": len(items),
        "items": [asdict(it) for it in items],
    }
    (out_dir / f"{spec.name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
