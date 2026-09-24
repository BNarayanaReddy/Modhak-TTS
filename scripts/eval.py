"""Eval entry: run the metric vector on a checkpoint (base model, or +LoRA adapter) over the
locked eval buckets → panel.json + an HTML report. Loads the LM + SNAC + Vocos + Bodhan ASR;
runs on the server (or the laptop GPU for smaller passes). The orchestration itself is covered
by tests/test_panel.py with fake backends.

  python scripts/eval.py --label baseline                      # base model
  python scripts/eval.py --label finetuned --adapter <dir>     # fine-tuned checkpoint
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from indic_speak_ft.eval import report
from indic_speak_ft.eval.asr import IndicTranscribeASR
from indic_speak_ft.eval.panel import load_manifests, run_panel
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
    ev = OmegaConf.load(REPO / "configs" / "eval.yaml")

    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (omit for the base model)")
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--with-speaker", action="store_true", help="also compute speaker similarity")
    ap.add_argument("--limit", type=int, default=None, help="cap items per bucket (smoke eval)")
    ap.add_argument("--max-new-tokens", type=int, default=int(ev.generation.max_new_tokens))
    a = ap.parse_args()
    out_dir = a.out_dir or str(REPO / "artifacts" / "eval_reports" / a.label)

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
    snac = load_snac(snac_dir, device="cpu")
    vocos = load_vocos(model_dir, device="cpu")
    g = ev.generation

    def synthesize(text: str, speaker: str):
        audio_ids, meta = generate_audio_ids(
            model, tok, contract, text, speaker, temperature=float(g.temperature),
            top_p=float(g.top_p), top_k=int(g.top_k), repetition_penalty=float(g.repetition_penalty),
            max_new_tokens=a.max_new_tokens, seed=int(g.seed), return_meta=True)
        wav = (decode_to_wav(audio_ids, contract.snac_base, snac, vocos, stock=False, device="cpu")
               if audio_ids else np.zeros(1, np.float32))
        return wav, audio_ids, meta["hit_max_new_tokens"]

    device = "cuda" if full_gpu else "cpu"
    asr = IndicTranscribeASR(str(ev.eval.asr_model), language=str(ev.eval.asr_language), device=device)
    manifests = load_manifests(str(REPO / "configs" / "eval_manifests"))
    if a.limit:  # smoke: cap items per bucket
        for m in manifests.values():
            m["items"] = m["items"][: a.limit]

    embed_fn = ref_fn = None
    if a.with_speaker:  # speaker similarity needs BOTH an embedder and the real reference audio
        from indic_speak_ft.eval.reference import ReferenceAudioProvider
        from indic_speak_ft.eval.speaker import EcapaEmbedder

        embed_fn = EcapaEmbedder(device=device)
        ref_fn = ReferenceAudioProvider(manifests)

    results = run_panel(manifests, synthesize, asr, out_dir=out_dir, embed_fn=embed_fn,
                        reference_audio_fn=ref_fn, seed=int(g.seed))
    report.write_html(results, f"{out_dir}/panel.html",
                      title=f"Indic-Speak MR FT — {a.label}",
                      subtitle=f"adapter={a.adapter or 'base model'}")
    print("eval done ->", out_dir)


if __name__ == "__main__":
    main()
