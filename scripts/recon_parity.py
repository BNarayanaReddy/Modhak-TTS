"""Phase-0 step-6: reference baseline + same-seed determinism check.

Loads bodhan-ai/indic-speak (LlamaForCausalLM) and generates a short Marathi
utterance with speaker=Anagha, then decodes via the reference
ids_to_codes -> quantizer.from_codes -> Vocos path (validated in the round-trip).

Runs on the installed transformers 4.46.3 by driving the raw tokenizer.json + the
reference build_prompt directly (the model's v5 TokenizersBackend tokenizer cannot
be loaded by transformers 4.x). Weights split across GPU + CPU for the 6 GB card.

This is NOT the Phase-5 parity test (our generate.py vs reference) — generate.py
does not exist yet. It captures the reference baseline and proves the LM runs here.

Run:  Modhak-TTS/.venv/bin/python scripts/recon_parity.py
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
OFFLOAD_DEFAULT = "/media/narayana/Windows-SSD/ai-models/bodhan_offload"
TEXT_DEFAULT = "पाणी हे हायड्रोजन आणि ऑक्सिजन यांचे संयुग आहे."  # Marathi STEM sentence


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offload", default=OFFLOAD_DEFAULT)
    ap.add_argument("--text", default=TEXT_DEFAULT)
    ap.add_argument("--speaker", default="Anagha")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-new-tokens", type=int, default=700)
    ap.add_argument("--runs", type=int, default=2, choices=[1, 2],
                    help="2 = also do a same-seed determinism check")
    ap.add_argument("--out", default="10_baseline_anagha_marathi.wav")
    ap.add_argument("--gpu-mem", default="5GiB", help="GPU cap for accelerate device_map")
    args = ap.parse_args()

    model_dir = f"{args.offload}/models/indic-speak"
    snac_dir = f"{args.offload}/models/snac_24khz"

    ref = _load_module("ref_inference", REPO / "reference" / "inference.py")

    from tokenizers import Tokenizer

    raw = Tokenizer.from_file(f"{model_dir}/tokenizer.json")

    class TokShim:
        """Minimal adapter so reference.build_prompt runs on the raw tokenizer."""

        bos_token_id = raw.token_to_id("<|begin_of_text|>")
        eos_token_id = raw.token_to_id("<|end_of_text|>")

        def convert_tokens_to_ids(self, s):
            return raw.token_to_id(s)

        def encode(self, s, add_special_tokens=False):
            return raw.encode(s, add_special_tokens=add_special_tokens).ids

    tok = TokShim()
    base = raw.token_to_id("<|snac_0|>")
    eos_speech = raw.token_to_id("<|end_of_speech|>")

    import transformers
    from transformers import LlamaForCausalLM

    # transformers>=5 renamed the `torch_dtype` from_pretrained kwarg to `dtype`.
    dtype_kw = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    t0 = time.time()
    lm = LlamaForCausalLM.from_pretrained(
        model_dir,
        attn_implementation="sdpa",
        device_map="auto",
        max_memory={0: args.gpu_mem, "cpu": "6GiB"},
        offload_folder=f"{args.offload}/offload_tmp",
        low_cpu_mem_usage=True,
        **{dtype_kw: torch.bfloat16},
    ).eval()
    dev0 = next(lm.parameters()).device
    print(f"LM loaded in {time.time() - t0:.0f}s | device buckets: "
          f"{sorted({str(v) for v in lm.hf_device_map.values()})}")

    def generate(seed: int):
        prompt = ref.build_prompt(tok, args.text, args.speaker, "")
        ids = torch.tensor([prompt], device=dev0)
        torch.manual_seed(seed)
        t = time.time()
        with torch.no_grad():
            gen = lm.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=args.max_new_tokens,
                eos_token_id=[eos_speech, tok.eos_token_id],
                pad_token_id=tok.eos_token_id,
                do_sample=True, temperature=0.6, top_p=0.9, top_k=50)
        new_ids = gen[0].tolist()[len(prompt):]
        return prompt, new_ids, time.time() - t

    _, n1, dt1 = generate(args.seed)
    print(f"run1: {len(n1)} new tokens in {dt1:.0f}s ({len(n1)/max(dt1,1e-9):.1f} tok/s), "
          f"end_of_speech emitted={eos_speech in n1}")
    if args.runs == 2:
        _, n2, _ = generate(args.seed)
        print(f"run2: determinism (same seed) identical = {n1 == n2}")

    audio_ids = [t for t in n1 if base <= t < base + 7 * 4096]
    n_frames = len(audio_ids) // 7
    # coarse repetition signal: fraction of c0 (position-0) frames that are unique
    c0 = audio_ids[0::7]
    uniq_c0 = len(set(c0)) / max(len(c0), 1)
    print(f"audio tokens: {len(audio_ids)} ({n_frames} frames, "
          f"~{len(audio_ids)/82:.1f}s at 82 tok/s); unique-c0 frac={uniq_c0:.2f}")

    # decode on CPU (GPU is full of the LM); validated path
    from snac import SNAC

    snac = SNAC.from_config(f"{snac_dir}/config.json")
    snac.load_state_dict(torch.load(f"{snac_dir}/pytorch_model.bin", map_location="cpu", weights_only=True))
    snac = snac.eval()
    codes = ref.ids_to_codes(audio_ids, base, "cpu")
    z_q = snac.quantizer.from_codes(codes)
    sys.path.insert(0, model_dir)
    from vocos.load import load_vocos

    vocos = load_vocos(f"{model_dir}/vocos/best.pt", device="cpu")
    import soundfile as sf

    with torch.no_grad():
        wav = vocos(z_q.float())[0, 0].clamp(-1, 1).float().cpu().numpy()
    out = REPO / "artifacts" / "recon" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, wav, 24000)
    print(f"baseline audio: {len(wav)/24000:.2f}s -> {out}")
    print("PARITY_BASELINE_OK")


if __name__ == "__main__":
    main()
