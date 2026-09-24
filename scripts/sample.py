"""Synthesize sample wavs from a base model (optionally with a LoRA adapter).

Ties generate.py + decode.py: build prompt → generate audio ids → SNAC/Vocos decode → wav.
Runs inference on this machine (the laptop GPU handles generation; training is the RAM-heavy
part that goes to the server). Used to produce artifacts/samples/{base_model,finetuned}/.

  python scripts/sample.py --text "…" --speaker Anagha                 # base model
  python scripts/sample.py --text "…" --speaker Anagha --adapter <dir> # fine-tuned
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from omegaconf import OmegaConf

from indic_speak_ft.inference.decode import decode_to_wav
from indic_speak_ft.inference.generate import attach_adapter, generate_audio_ids
from indic_speak_ft.model.loading import load_base_lm, load_snac, load_vocos
from indic_speak_ft.tokens import load_tokens

REPO = Path(__file__).resolve().parents[1]


def tokenizer_shim(model_dir: str):
    from tokenizers import Tokenizer

    raw = Tokenizer.from_file(f"{model_dir}/tokenizer.json")

    class Shim:
        bos_token_id = raw.token_to_id("<|begin_of_text|>")
        eos_token_id = raw.token_to_id("<|end_of_text|>")
        unk_token_id = None

        def convert_tokens_to_ids(self, s: str):
            return raw.token_to_id(s)

        def encode(self, s: str, add_special_tokens: bool = False):
            return raw.encode(s, add_special_tokens=add_special_tokens).ids

    return Shim()


def main() -> None:
    base = OmegaConf.load(REPO / "configs" / "base.yaml")
    OmegaConf.resolve(base)
    gen = OmegaConf.load(REPO / "configs" / "eval.yaml").generation

    ap = argparse.ArgumentParser()
    ap.add_argument("--text", action="append", required=True)
    ap.add_argument("--speaker", default="Anagha")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (fine-tuned run)")
    ap.add_argument("--out-dir", default=str(REPO / "artifacts" / "samples" / "adhoc"))
    ap.add_argument("--stock", action="store_true", help="also write the stock-SNAC A/B decode")
    ap.add_argument("--seed", type=int, default=int(gen.seed))
    ap.add_argument("--max-new-tokens", type=int, default=int(gen.max_new_tokens))
    ap.add_argument("--repetition-penalty", type=float, default=float(gen.repetition_penalty))
    a = ap.parse_args()

    import soundfile as sf

    model_dir, snac_dir = str(base.model_dir), str(base.snac_dir)
    tok = tokenizer_shim(model_dir)
    contract = load_tokens(tok)

    full_gpu = bool(base.get("full_gpu", False))
    model = load_base_lm(
        model_dir, dtype="bfloat16", attn_implementation="sdpa",
        **({"device_map": {"": 0}} if full_gpu else
           {"device_map": "auto", "max_memory": {0: str(base.gpu_mem), "cpu": "6GiB"},
            "offload_folder": f"{base.offload}/offload_tmp"}))
    if a.adapter:
        model = attach_adapter(model, a.adapter)
        print(f"attached adapter: {a.adapter}")
    snac = load_snac(snac_dir, device="cpu")
    vocos = load_vocos(model_dir, device="cpu")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(a.text):
        audio_ids = generate_audio_ids(
            model, tok, contract, text, a.speaker, temperature=float(gen.temperature),
            top_p=float(gen.top_p), top_k=int(gen.top_k), repetition_penalty=a.repetition_penalty,
            max_new_tokens=a.max_new_tokens, seed=a.seed)
        wav = decode_to_wav(audio_ids, contract.snac_base, snac, vocos, stock=False, device="cpu")
        slug = f"{i:02d}_" + (re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:32] or "utt")
        sf.write(out / f"{slug}.wav", wav, 24000)
        print(f"[{slug}] {len(audio_ids)//7} frames -> {out / f'{slug}.wav'}")
        if a.stock:
            wav_s = decode_to_wav(audio_ids, contract.snac_base, snac, vocos, stock=True, device="cpu")
            sf.write(out / f"{slug}_stock.wav", wav_s, 24000)


if __name__ == "__main__":
    main()
