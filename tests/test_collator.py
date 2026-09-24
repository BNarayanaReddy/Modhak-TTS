"""Phase-2 collator tests: prompt parity with the reference, loss-mask coverage, and the
codebook-0 alignment of the audio target."""
from __future__ import annotations

import importlib.util
import os
import random
from pathlib import Path

import pytest

from indic_speak_ft.data.collator import (
    IGNORE_INDEX,
    assemble_sequence,
    build_prompt_ids,
    build_training_sequence,
)
from indic_speak_ft.tokens import NUM_CODEBOOKS, build_frame, load_tokens, token_to_snac

REPO = Path(__file__).resolve().parents[1]

_SPECIALS = {
    "<|begin_of_text|>": 128000, "<|eot_id|>": 128009,
    "<|start_of_human|>": 128259, "<|end_of_human|>": 128260,
    "<|start_of_ai|>": 128261, "<|end_of_ai|>": 128262,
    "<|start_of_speech|>": 128257, "<|end_of_speech|>": 128258,
    "<|speaker>": 156938, "<speaker|>": 156939, "<|style>": 156940, "<style|>": 156941,
}


class FakeSpecialTokenizer:
    """Resolves special tokens + <|snac_N|>; text-encodes to fixed dummy ids (enough for
    the mask/alignment tests, which don't need real BPE)."""

    unk_token_id = 0

    def convert_tokens_to_ids(self, tok: str):
        if tok in _SPECIALS:
            return _SPECIALS[tok]
        if tok.startswith("<|snac_") and tok.endswith("|>"):
            return 128266 + int(tok[len("<|snac_"):-2])
        return self.unk_token_id

    def encode(self, text: str, add_special_tokens: bool = False):
        return [1000 + (ord(c) % 100) for c in text]  # deterministic dummy text ids


def _contract():
    return load_tokens(FakeSpecialTokenizer())


def _audio(n_frames: int, base: int, seed: int = 0) -> list[int]:
    rng = random.Random(seed)
    out: list[int] = []
    for _ in range(n_frames):
        codes = [rng.randrange(4096) for _ in range(NUM_CODEBOOKS)]
        out += build_frame(*codes, base=base)
    return out


# --- loss mask + alignment (synthetic, hermetic) ----------------------------------
def test_loss_mask_covers_exactly_audio_and_stop():
    c = _contract()
    prompt = list(range(50))  # arbitrary "prompt" ids
    audio = _audio(12, c.snac_base)
    seq = assemble_sequence(prompt, audio, c, include_stop_in_loss=True)

    assert seq.input_ids == [*prompt, *audio, c.end_of_speech]
    assert seq.prompt_len == len(prompt)
    # masked exactly on the prompt; supervised exactly on audio + stop
    assert seq.labels[:len(prompt)] == [IGNORE_INDEX] * len(prompt)
    assert seq.labels[len(prompt):] == [*audio, c.end_of_speech]
    supervised = [i for i, lbl in enumerate(seq.labels) if lbl != IGNORE_INDEX]
    assert supervised == list(range(len(prompt), len(seq.input_ids)))


def test_stop_token_masking_flag():
    c = _contract()
    seq = assemble_sequence(list(range(10)), _audio(3, c.snac_base), c, include_stop_in_loss=False)
    assert seq.input_ids[-1] == c.end_of_speech
    assert seq.labels[-1] == IGNORE_INDEX          # stop present as input, excluded from loss
    assert seq.labels[-2] != IGNORE_INDEX          # last audio token still supervised


def test_every_7i_in_audio_span_is_codebook0():
    c = _contract()
    prompt = list(range(23))
    n = 15
    seq = assemble_sequence(prompt, _audio(n, c.snac_base, seed=3), c)
    for i in range(n):
        tid = seq.input_ids[seq.prompt_len + NUM_CODEBOOKS * i]
        assert token_to_snac(c.snac_base, tid)[0] == 0


def test_malformed_audio_rejected():
    c = _contract()
    with pytest.raises(ValueError, match="multiple of 7"):
        assemble_sequence([1, 2], [c.snac_base] * 6, c)         # not a whole frame
    with pytest.raises(ValueError, match="outside the SNAC audio band"):
        assemble_sequence([1, 2], [c.snac_base - 1] * 7, c)     # out of band


# --- prompt parity with the reference (needs the real tokenizer) ------------------
def test_prompt_matches_reference_byte_for_byte():
    path = os.environ.get("INDIC_SPEAK_TOKENIZER_JSON")
    if not path or not Path(path).is_file():
        pytest.skip("set INDIC_SPEAK_TOKENIZER_JSON to run the reference-parity prompt check")
    from tokenizers import Tokenizer

    raw = Tokenizer.from_file(path)

    class Shim:
        unk_token_id = None
        bos_token_id = raw.token_to_id("<|begin_of_text|>")

        def convert_tokens_to_ids(self, tok: str):
            return raw.token_to_id(tok)

        def encode(self, text: str, add_special_tokens: bool = False):
            return raw.encode(text, add_special_tokens=add_special_tokens).ids

    spec = importlib.util.spec_from_file_location("ref_inf", REPO / "reference" / "inference.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)

    shim = Shim()
    contract = load_tokens(shim)
    for text, speaker in [
        ("पाण्याचा रेणू दोन हायड्रोजन अणूंचा बनतो.", "Anagha"),
        ("नमस्कार, आज आपण विज्ञान शिकू.", "Chinmay"),
    ]:
        ours = build_prompt_ids(shim, contract, text, speaker, style="")
        theirs = ref.build_prompt(shim, text, speaker, "")
        assert ours == theirs, f"prompt diverged from reference for {speaker}"
        # and the full training sequence keeps the prompt as its masked prefix
        seq = build_training_sequence(shim, contract, text, speaker, _audio(4, contract.snac_base))
        assert seq.input_ids[:seq.prompt_len] == theirs
