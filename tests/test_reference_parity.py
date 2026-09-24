"""Phase-5 reference-parity tests: our decode/prompt match reference/inference.py exactly.

The component parities (de-interleave + prompt) are hermetic. The full end-to-end parity
(our generate.py vs reference on the real LM) is gated on MODEL_DIR + transformers>=5 (it
loads the 6.6 GB model and the v5 tokenizer), so it runs on the server, skips on the laptop.
"""
from __future__ import annotations

import importlib.util
import os
import random
from pathlib import Path

import pytest

from indic_speak_ft.data.compile import codes_to_tokens
from indic_speak_ft.inference.decode import snac_codes_from_tokens, take_audio_prefix
from indic_speak_ft.tokens import CODEBOOK_SIZE

BASE = 128266
REPO = Path(__file__).resolve().parents[1]


def _reference():
    spec = importlib.util.spec_from_file_location("ref", REPO / "reference" / "inference.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_snac_codes_from_tokens_matches_reference():
    rng = random.Random(0)
    codes = [[rng.randrange(CODEBOOK_SIZE) for _ in range(n)] for n in (10, 20, 40)]
    tokens = codes_to_tokens(codes, BASE, deduplicate=False)
    ours = [t[0].tolist() for t in snac_codes_from_tokens(tokens, BASE, "cpu")]
    ref = [t[0].tolist() for t in _reference().ids_to_codes(tokens, BASE, "cpu")]
    assert ours == ref == codes


def test_take_audio_prefix_stops_at_first_non_audio():
    audio = codes_to_tokens([[1, 2], [0, 0, 0, 0], [0] * 8], BASE, deduplicate=False)
    stream = [*audio, 128258, 999]  # audio then <|end_of_speech|> then junk
    assert take_audio_prefix(stream, BASE) == audio


def test_prompt_matches_reference_byte_for_byte():
    path = os.environ.get("INDIC_SPEAK_TOKENIZER_JSON")
    if not path or not Path(path).is_file():
        pytest.skip("set INDIC_SPEAK_TOKENIZER_JSON to run the prompt-parity check")
    from tokenizers import Tokenizer

    from indic_speak_ft.data.collator import build_prompt_ids
    from indic_speak_ft.tokens import load_tokens

    raw = Tokenizer.from_file(path)

    class Shim:
        unk_token_id = None
        bos_token_id = raw.token_to_id("<|begin_of_text|>")

        def convert_tokens_to_ids(self, t: str):
            return raw.token_to_id(t)

        def encode(self, t: str, add_special_tokens: bool = False):
            return raw.encode(t, add_special_tokens=add_special_tokens).ids

    shim = Shim()
    contract = load_tokens(shim)
    ref = _reference()
    for text, spk in [("पाण्याचा रेणू दोन अणूंचा बनतो.", "Anagha"), ("नमस्कार.", "Chinmay")]:
        assert build_prompt_ids(shim, contract, text, spk, style="") == ref.build_prompt(shim, text, spk, "")


@pytest.mark.skipif(not os.path.isdir(os.environ.get("MODEL_DIR", "")),
                    reason="set MODEL_DIR (needs the 6.6 GB LM + transformers>=5) for e2e parity")
def test_end_to_end_generate_matches_reference():
    import transformers

    if int(transformers.__version__.split(".")[0]) < 5:
        pytest.skip("end-to-end parity needs transformers>=5 to load the reference tokenizer/model")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from indic_speak_ft.inference.generate import generate_audio_ids
    from indic_speak_ft.tokens import load_tokens

    model_dir = os.environ["MODEL_DIR"]
    ref = _reference()
    tok = AutoTokenizer.from_pretrained(model_dir)
    lm = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16,
                                              attn_implementation="sdpa").eval()
    contract = load_tokens(tok)
    text, spk, seed = "नमस्कार, आज आपण विज्ञान शिकू.", "Anagha", 1234

    torch.manual_seed(seed)
    ref_prompt = ref.build_prompt(tok, text, spk, "")
    with torch.no_grad():
        ref_gen = lm.generate(
            input_ids=torch.tensor([ref_prompt]), attention_mask=torch.ones(1, len(ref_prompt), dtype=torch.long),
            max_new_tokens=300, eos_token_id=[contract.end_of_speech, tok.eos_token_id],
            pad_token_id=tok.eos_token_id, do_sample=True, temperature=0.6, top_p=0.9, top_k=50)
    ref_audio = ref.ids_to_codes(ref_gen[0].tolist()[len(ref_prompt):], contract.snac_base, "cpu")

    ours = generate_audio_ids(lm, tok, contract, text, spk, max_new_tokens=300, seed=seed)
    our_codes = snac_codes_from_tokens(ours, contract.snac_base, "cpu")
    assert [t[0].tolist() for t in our_codes] == [t[0].tolist() for t in ref_audio]
