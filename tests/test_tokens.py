"""Phase-1 token-contract tests: frame round-trip, interleave parity with the
reference, the SNAC band math, and the corrected contiguity assertion.

Hermetic and CPU-only (no model, no offload drive): a fake tokenizer supplies the
resolved IDs; the interleave parity check imports only ``reference/inference.py``
(numpy/torch, no weights). An optional check runs against the real ``tokenizer.json``
if ``INDIC_SPEAK_TOKENIZER_JSON`` points at one, and is skipped otherwise.
"""
from __future__ import annotations

import importlib.util
import os
import random
from pathlib import Path

import pytest

from indic_speak_ft.tokens import (
    CODEBOOK_SIZE,
    NUM_CODEBOOKS,
    TOTAL_AUDIO_TOKENS,
    TokenContract,
    build_frame,
    is_audio_token,
    load_tokens,
    parse_frame,
    snac_id_to_token,
    token_to_snac,
)

BASE = 128266  # <|snac_0|>, resolved in recon; used only as a fixed value for math tests
REPO = Path(__file__).resolve().parents[1]


# --- fakes ------------------------------------------------------------------------
_RESOLVED = {
    "<|begin_of_text|>": 128000, "<|eot_id|>": 128009,
    "<|start_of_human|>": 128259, "<|end_of_human|>": 128260,
    "<|start_of_ai|>": 128261, "<|end_of_ai|>": 128262,
    "<|start_of_speech|>": 128257, "<|end_of_speech|>": 128258,
    "<|speaker>": 156938, "<speaker|>": 156939, "<|style>": 156940, "<style|>": 156941,
}


class FakeTokenizer:
    """Correct tokenizer: per-code SNAC naming ``<|snac_N|>`` == base + N."""

    unk_token_id = 0

    def convert_tokens_to_ids(self, tok: str):
        if tok in _RESOLVED:
            return _RESOLVED[tok]
        if tok.startswith("<|snac_") and tok.endswith("|>"):
            return BASE + int(tok[len("<|snac_"):-2])
        return self.unk_token_id


class PerCodebookTokenizer(FakeTokenizer):
    """Wrong (hypothetical) tokenizer matching GOAL step-4's assumption:
    ``<|snac_N|>`` == base + N*4096. assert_snac_contract must reject it."""

    def convert_tokens_to_ids(self, tok: str):
        if tok.startswith("<|snac_") and tok.endswith("|>"):
            return BASE + int(tok[len("<|snac_"):-2]) * CODEBOOK_SIZE
        return super().convert_tokens_to_ids(tok)


def _reference_ids_to_codes():
    spec = importlib.util.spec_from_file_location("ref_inf", REPO / "reference" / "inference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ids_to_codes


def _random_frame(rng: random.Random) -> tuple[int, ...]:
    """7 raw codes in SNAC-source order (c0_i, c1_2i, c1_2i1, c2_4i, c2_4i1, c2_4i2, c2_4i3)."""
    return tuple(rng.randrange(CODEBOOK_SIZE) for _ in range(NUM_CODEBOOKS))


# --- frame math -------------------------------------------------------------------
def test_frame_roundtrip():
    rng = random.Random(0)
    for _ in range(1000):
        codes = _random_frame(rng)
        ids = build_frame(*codes, base=BASE)
        assert len(ids) == NUM_CODEBOOKS
        assert parse_frame(BASE, ids) == codes


def test_frame_length_multiple_of_7():
    rng = random.Random(1)
    n = 13
    flat = [tid for _ in range(n) for tid in build_frame(*_random_frame(rng), base=BASE)]
    assert len(flat) % NUM_CODEBOOKS == 0
    assert len(flat) // NUM_CODEBOOKS == n


def test_codebook0_lands_at_index_7i():
    rng = random.Random(2)
    n = 20
    flat = [tid for _ in range(n) for tid in build_frame(*_random_frame(rng), base=BASE)]
    for i in range(n):
        tid = flat[NUM_CODEBOOKS * i]
        pos, _ = token_to_snac(BASE, tid)
        assert pos == 0, f"index {NUM_CODEBOOKS*i} is frame position {pos}, expected 0"
        assert BASE <= tid < BASE + CODEBOOK_SIZE  # codebook-0 band


def test_snac_id_token_inverse():
    rng = random.Random(3)
    for _ in range(2000):
        pos = rng.randrange(NUM_CODEBOOKS)
        code = rng.randrange(CODEBOOK_SIZE)
        tid = snac_id_to_token(BASE, pos, code)
        assert is_audio_token(BASE, tid)
        assert token_to_snac(BASE, tid) == (pos, code)


def test_is_audio_token_boundaries():
    assert not is_audio_token(BASE, BASE - 1)
    assert is_audio_token(BASE, BASE)
    assert is_audio_token(BASE, BASE + TOTAL_AUDIO_TOKENS - 1)
    assert not is_audio_token(BASE, BASE + TOTAL_AUDIO_TOKENS)
    assert token_to_snac(BASE, BASE - 1) is None


def test_out_of_range_codes_raise():
    with pytest.raises(ValueError):
        snac_id_to_token(BASE, NUM_CODEBOOKS, 0)
    with pytest.raises(ValueError):
        snac_id_to_token(BASE, 0, CODEBOOK_SIZE)


# --- interleave parity with the reference -----------------------------------------
def test_build_frame_matches_reference_interleave():
    """build_frame → flat ids → reference.ids_to_codes must recover the exact codes,
    proving byte-for-byte agreement with reference/inference.py's interleave."""
    ids_to_codes = _reference_ids_to_codes()
    rng = random.Random(4)
    n = 16
    frames = [_random_frame(rng) for _ in range(n)]
    flat = [tid for f in frames for tid in build_frame(*f, base=BASE)]

    c0, c1, c2 = (t[0].tolist() for t in ids_to_codes(flat, BASE, "cpu"))
    assert len(c0) == n and len(c1) == 2 * n and len(c2) == 4 * n
    for i, (c0_i, c1_2i, c1_2i1, c2_4i, c2_4i1, c2_4i2, c2_4i3) in enumerate(frames):
        assert c0[i] == c0_i
        assert c1[2 * i] == c1_2i and c1[2 * i + 1] == c1_2i1
        assert c2[4 * i] == c2_4i and c2[4 * i + 1] == c2_4i1
        assert c2[4 * i + 2] == c2_4i2 and c2[4 * i + 3] == c2_4i3


# --- contract resolution ----------------------------------------------------------
def test_load_tokens_resolves_and_validates():
    c = load_tokens(FakeTokenizer())
    assert isinstance(c, TokenContract)
    assert c.snac_base == BASE
    assert c.start_of_speech == 128257 and c.end_of_speech == 128258
    assert c.audio_hi == BASE + TOTAL_AUDIO_TOKENS == 156938
    assert c.speaker_open == c.audio_hi  # band top sits just below <|speaker>
    assert c.frame_position_bases == tuple(BASE + p * CODEBOOK_SIZE for p in range(NUM_CODEBOOKS))


def test_per_codebook_naming_is_rejected():
    """The corrected assertion must reject GOAL step-4's per-codebook assumption."""
    with pytest.raises(AssertionError, match="not contiguous per-code"):
        load_tokens(PerCodebookTokenizer())


def test_missing_special_token_raises():
    class Missing(FakeTokenizer):
        def convert_tokens_to_ids(self, tok: str):
            if tok == "<|start_of_speech|>":
                return self.unk_token_id  # simulate absence -> unk
            return super().convert_tokens_to_ids(tok)

    with pytest.raises(ValueError, match="start_of_speech"):
        load_tokens(Missing())


# --- optional: validate against the real tokenizer.json if available --------------
def test_real_tokenizer_contract_if_available():
    path = os.environ.get("INDIC_SPEAK_TOKENIZER_JSON")
    if not path or not Path(path).is_file():
        pytest.skip("set INDIC_SPEAK_TOKENIZER_JSON to the model's tokenizer.json to run")
    from tokenizers import Tokenizer

    raw = Tokenizer.from_file(path)

    class Shim:
        unk_token_id = None

        def convert_tokens_to_ids(self, tok: str):
            return raw.token_to_id(tok)

    c = load_tokens(Shim())  # resolves + asserts the SNAC contract on the real vocab
    assert c.snac_base == 128266
    assert c.audio_hi == 156938
