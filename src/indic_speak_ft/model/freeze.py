"""Freeze policy + verifier (Decision 2).

After the LoRA adapters are attached, ONLY the adapter params (``lora_*``) are trainable;
everything else is frozen — embed_tokens, lm_head, down_proj, all norms, and every base
weight. ``tie_word_embeddings`` makes lm_head share storage with embed_tokens, so freezing
the embedding freezes the head (RECON §2). The verifier counts trainable params and asserts
they equal the LoRA count implied by the config — a mismatch means a target was missed or an
extra tensor is training. Called at train start and in ``tests/test_freeze_policy.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

# Base tensors that MUST stay frozen (Decision 2). Matched as substrings of param names.
FROZEN_SUBSTRINGS = ("embed_tokens", "lm_head", "down_proj", "norm")


def apply_freeze_policy(model) -> None:
    """Make exactly the LoRA adapter params trainable; freeze everything else."""
    for name, p in model.named_parameters():
        p.requires_grad_("lora_" in name)


def count_trainable_params(model) -> tuple[int, int]:
    """(trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def expected_trainable_params(
    *,
    num_layers: int,
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    intermediate_size: int,
    mlp_layer_range: tuple[int, int],
    r: int,
    head_dim: int | None = None,
    attention_targets: tuple[str, ...] = ("q_proj", "v_proj"),
    mlp_targets: tuple[str, ...] = ("up_proj", "gate_proj"),
) -> int:
    """Total LoRA params implied by Decision 2's config. LoRA adds ``r*(in+out)`` per
    targeted Linear (A: r×in, B: out×r). q/v on all layers; up/gate on the MLP range only."""
    hd = head_dim or (hidden_size // num_attention_heads)
    out_dim = {
        "q_proj": num_attention_heads * hd,
        "k_proj": num_key_value_heads * hd,
        "v_proj": num_key_value_heads * hd,
        "o_proj": hidden_size,
        "up_proj": intermediate_size,
        "gate_proj": intermediate_size,
        "down_proj": hidden_size,
    }
    in_dim = {"up_proj": hidden_size, "gate_proj": hidden_size, "down_proj": intermediate_size}

    def cost(proj: str) -> int:
        return r * (in_dim.get(proj, hidden_size) + out_dim[proj])

    lo, hi = mlp_layer_range
    n_mlp_layers = hi - lo + 1
    attn = num_layers * sum(cost(p) for p in attention_targets)
    mlp = n_mlp_layers * sum(cost(p) for p in mlp_targets)
    return attn + mlp


@dataclass(frozen=True)
class FreezeReport:
    trainable: int
    total: int
    expected_trainable: int
    trainable_pct: float


def verify_freeze(model, *, expected_trainable: int) -> FreezeReport:
    """Assert the freeze policy holds: (1) every trainable param is a LoRA param; (2) the
    frozen tensors (embed/lm_head/down_proj/norms) are frozen; (3) the trainable count equals
    ``expected_trainable``. Raises ``AssertionError`` on any violation."""
    non_lora_trainable = [n for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    if non_lora_trainable:
        raise AssertionError(f"non-LoRA params are trainable: {non_lora_trainable[:5]}")

    for n, p in model.named_parameters():
        if "lora_" in n:
            continue
        if any(s in n for s in FROZEN_SUBSTRINGS) and p.requires_grad:
            raise AssertionError(f"{n} must be frozen but requires_grad=True")

    trainable, total = count_trainable_params(model)
    if trainable != expected_trainable:
        raise AssertionError(
            f"trainable params {trainable:,} != expected {expected_trainable:,} "
            f"(a LoRA target was missed or an extra tensor is training)"
        )
    return FreezeReport(trainable, total, expected_trainable, 100.0 * trainable / max(total, 1))
