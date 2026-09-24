"""Generation wrapper: the reference ``build_prompt`` + ``model.generate`` + audio-extraction
semantics, plus LoRA-adapter attach/merge. No token math is reimplemented — the prompt comes
from ``build_prompt_ids`` (tested byte-for-byte vs the reference) and the audio prefix from
``decode.take_audio_prefix``.

Faithful defaults: ``repetition_penalty=1.0`` (the reference script applies none), so
``generate_audio_ids`` matches ``reference/inference.py`` exactly for the parity test. The eval
config overrides to the production ``1.2`` (Decision 5) — that lives in configs/eval.yaml, not
baked in here.
"""
from __future__ import annotations

import warnings
from typing import Any

from indic_speak_ft.data.collator import build_prompt_ids
from indic_speak_ft.inference.decode import take_audio_prefix
from indic_speak_ft.tokens import TokenContract


def attach_adapter(base_model: Any, adapter_path: str, *, merge: bool = False) -> Any:
    """Attach a LoRA adapter to a loaded base model; ``merge=True`` folds it into the weights."""
    from peft import PeftModel

    model = PeftModel.from_pretrained(base_model, adapter_path)
    if merge:
        model = model.merge_and_unload()
    return model.eval()


def generate_audio_ids(
    model: Any,
    tokenizer: Any,
    contract: TokenContract,
    text: str,
    speaker: str,
    *,
    style: str = "",
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    max_new_tokens: int = 2520,
    seed: int | None = None,
    device: str | None = None,
    return_meta: bool = False,
) -> list[int] | tuple[list[int], dict[str, Any]]:
    """Generate and return the flat SNAC audio ids (start → first non-audio token). Mirrors the
    reference ``TTS.__call__`` generate call; warns on a max_new_tokens hit without a stop."""
    import torch

    prompt = build_prompt_ids(tokenizer, contract, text, speaker, style=style)
    dev = device or next(model.parameters()).device
    ids = torch.tensor([prompt], device=dev)
    if seed is not None:
        torch.manual_seed(seed)

    gen_kwargs: dict[str, Any] = {
        "input_ids": ids, "attention_mask": torch.ones_like(ids), "max_new_tokens": max_new_tokens,
        "eos_token_id": [contract.end_of_speech, tokenizer.eos_token_id],
        "pad_token_id": tokenizer.eos_token_id, "do_sample": temperature > 0,
        "temperature": temperature, "top_p": top_p, "top_k": top_k}
    if repetition_penalty != 1.0:  # keep the parity call byte-identical to the reference
        gen_kwargs["repetition_penalty"] = repetition_penalty

    with torch.no_grad():
        out = model.generate(**gen_kwargs)
    new_ids = out[0].tolist()[len(prompt):]
    hit_max = len(new_ids) >= max_new_tokens
    if hit_max:
        warnings.warn(
            f"generation hit max_new_tokens ({max_new_tokens}) without <|end_of_speech|> — "
            "audio is likely truncated or runaway babble", RuntimeWarning, stacklevel=2)
    audio_ids = take_audio_prefix(new_ids, contract.snac_base)
    return (audio_ids, {"hit_max_new_tokens": hit_max}) if return_meta else audio_ids
