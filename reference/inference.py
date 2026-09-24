"""Orpheus TTS: text -> SNAC codes (LM) -> z_q -> 24 kHz waveform (Vocos).

As a library:

    from inference import TTS
    tts = TTS("bodhan-ai/indic-speak-preview")      # or a local dir
    wav = tts("नमस्ते, आज हम विज्ञान पढ़ेंगे।", speaker="Amit")   # float32 numpy @ 24 kHz
    tts.save("hello.wav", wav)

The model loads once; call `tts(...)` as often as you like.

As a CLI:

    python inference.py --text "..." --speaker Amit

Needs: transformers>=5, torch, snac, soundfile.
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import torch

SR = 24_000
NUM_CODEBOOKS = 7
CODEBOOK_SIZE = 4096
SNAC_REPO = "hubertsiuzdak/snac_24khz"


def build_prompt(tok, text: str, speaker: str = "", style: str = "") -> list[int]:
    """<|start_of_human|><|begin_of_text|>[<|speaker>..<speaker|>][<|style>..<style|>]
    text<|eot_id|><|end_of_human|><|start_of_ai|><|start_of_speech|>"""
    tid = lambda s: tok.convert_tokens_to_ids(s)
    nl = tok.encode("\n", add_special_tokens=False)
    enc = lambda s: tok.encode(s, add_special_tokens=False)

    blocks = []
    if speaker.strip():
        blocks.append([tid("<|speaker>")] + enc(speaker.strip()) + [tid("<speaker|>")])
    if style.strip():
        blocks.append([tid("<|style>")] + enc(style.strip()) + [tid("<style|>")])

    meta: list[int] = []
    for i, b in enumerate(blocks):
        if i:
            meta += nl
        meta += b
    if blocks:
        meta += nl

    body = [tok.bos_token_id] + meta + enc(text) + [tid("<|eot_id|>")]
    return ([tid("<|start_of_human|>")] + body + [tid("<|end_of_human|>")]
            + [tid("<|start_of_ai|>"), tid("<|start_of_speech|>")])


def ids_to_codes(ids: list[int], base: int, device) -> list[torch.Tensor]:
    """Flat LM token ids -> SNAC's 3 hierarchical codebooks (rates 1:2:4).

    Frame i is [c0[i], c1[2i], c2[4i], c2[4i+1], c1[2i+1], c2[4i+2], c2[4i+3]],
    each offset by base + position*4096. Stops at the first non-audio token.
    """
    hi = base + NUM_CODEBOOKS * CODEBOOK_SIZE
    audio = []
    for t in ids:
        if not (base <= t < hi):
            break
        audio.append(t)

    n = len(audio) // NUM_CODEBOOKS
    if n == 0:
        raise ValueError("model emitted no complete SNAC frame")

    a = np.array(audio[: n * NUM_CODEBOOKS], np.int32).reshape(n, NUM_CODEBOOKS)
    a -= base + np.arange(NUM_CODEBOOKS, dtype=np.int32) * CODEBOOK_SIZE
    a = a[np.all((a >= 0) & (a < CODEBOOK_SIZE), axis=1)]
    if a.shape[0] == 0:
        raise ValueError("no in-range SNAC frame")

    c1 = np.empty(a.shape[0] * 2, np.int32)
    c1[0::2], c1[1::2] = a[:, 1], a[:, 4]
    c2 = np.empty(a.shape[0] * 4, np.int32)
    c2[0::4], c2[1::4], c2[2::4], c2[3::4] = a[:, 2], a[:, 3], a[:, 5], a[:, 6]

    return [torch.from_numpy(x.copy()).long().unsqueeze(0).to(device)
            for x in (a[:, 0], c1, c2)]


class TTS:
    """Loads the LM, SNAC quantizer and Vocos decoder once; synthesizes on call."""

    sample_rate = SR

    def __init__(self, model: str = ".", vocos: str | None = None,
                 device: str | None = None):
        from snac import SNAC
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if vocos is None:
            local = Path(model) / "vocos" / "best.pt"
            if local.is_file():
                vocos = str(local)
            else:  # model is a hub id -> pull the decoder from the same repo
                from huggingface_hub import hf_hub_download
                vocos = hf_hub_download(repo_id=model, filename="vocos/best.pt")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.lm = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch.bfloat16, attn_implementation="sdpa").to(self.device).eval()
        self.snac = SNAC.from_pretrained(SNAC_REPO).to(self.device).eval()
        self.vocos = _load_vocos(vocos, self.device)

        self._base = self.tok.convert_tokens_to_ids("<|snac_0|>")
        self._eos = self.tok.convert_tokens_to_ids("<|end_of_speech|>")

    def __call__(self, text: str, speaker: str = "", style: str = "", *,
                 temperature: float = 0.6, top_p: float = 0.9, top_k: int = 50,
                 max_new_tokens: int = 2520, seed: int | None = None,
                 stock: bool = False) -> np.ndarray:
        """Synthesize one utterance. Returns float32 mono @ 24 kHz.

        stock=True decodes with SNAC's own decoder instead of Vocos (for A/B).
        """
        if seed is not None:
            torch.manual_seed(seed)

        prompt = build_prompt(self.tok, text, speaker, style)
        ids = torch.tensor([prompt], device=self.device)

        with torch.no_grad():
            gen = self.lm.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=max_new_tokens,
                eos_token_id=[self._eos, self.tok.eos_token_id],
                pad_token_id=self.tok.eos_token_id,
                do_sample=temperature > 0, temperature=temperature,
                top_p=top_p, top_k=top_k)

            new_ids = gen[0].tolist()[len(prompt):]
            if len(new_ids) >= max_new_tokens:
                warnings.warn(
                    f"generation hit max_new_tokens ({max_new_tokens}) without emitting "
                    "<|end_of_speech|> — audio is likely truncated or runaway babble",
                    RuntimeWarning, stacklevel=2)
            codes = ids_to_codes(new_ids, self._base, self.device)
            z_q = self.snac.quantizer.from_codes(codes)
            dec = self.snac.decoder if stock else self.vocos
            wav = dec(z_q.float() if not stock else z_q)

        return wav[0, 0].clamp(-1, 1).float().cpu().numpy()

    @staticmethod
    def save(path: str, wav: np.ndarray) -> str:
        import soundfile as sf
        sf.write(path, wav, SR)
        return path


def _load_vocos(ckpt_path: str, device):
    """Delegate to vocos/load.py so the loading logic lives in exactly one place."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vocos.load import load_vocos

    return load_vocos(ckpt_path, device=device)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text", action="append", required=True)
    p.add_argument("--speaker", default="Amit")
    p.add_argument("--style", default="")
    p.add_argument("--model", default=".")
    p.add_argument("--vocos", default=None)
    p.add_argument("--out-dir", default="out")
    p.add_argument("--max-new-tokens", type=int, default=2520)  # ~82 tok/s -> ~30 s
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--stock", action="store_true", help="also write stock-SNAC decode")
    a = p.parse_args()

    tts = TTS(a.model, a.vocos)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for i, text in enumerate(a.text):
        kw = dict(speaker=a.speaker, style=a.style, temperature=a.temperature,
                  top_p=a.top_p, top_k=a.top_k, max_new_tokens=a.max_new_tokens,
                  seed=a.seed)
        wav = tts(text, **kw)
        slug = f"{i:02d}_" + (re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:40] or "utt")
        tts.save(str(out / f"{slug}.wav"), wav)
        print(f"[{slug}] {len(wav)/SR:.2f}s -> {out/f'{slug}.wav'}")
        if a.stock:
            tts.save(str(out / f"{slug}_stock.wav"), tts(text, stock=True, **kw))


if __name__ == "__main__":
    main()
