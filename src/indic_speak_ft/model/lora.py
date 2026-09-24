"""LoRA target selection + PEFT config, per Decision 2.

Decision 2 asks for DIFFERENT layer coverage per projection: q_proj/v_proj on ALL layers,
but up_proj/gate_proj only on the MIDDLE layers (8..24). A single PEFT ``LoraConfig`` with
``layers_to_transform`` can't express that (it applies one layer set to every target module),
so we build an explicit ``target_modules`` list: bare ``q_proj``/``v_proj`` (PEFT suffix-
matches them on every layer) plus fully-qualified ``model.layers.<i>.mlp.{up,gate}_proj``
names for i in the MLP range only.

Rationale (Decision 2): Q/V adapt attention routing over the mixed text+speaker+audio
context (cheap, all layers); MLP up/gate on middle layers reshape content→acoustic mapping
without touching early acoustic-feature layers or the late layers that feed the (frozen)
LM head and carry codebook-0 intelligibility priors. Values live in configs/lora.yaml.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from peft import LoraConfig

ATTENTION_TARGETS = ("q_proj", "v_proj")
MLP_TARGETS = ("up_proj", "gate_proj")


def lora_target_modules(
    mlp_layer_range: tuple[int, int],
    *,
    attention_targets: tuple[str, ...] = ATTENTION_TARGETS,
    mlp_targets: tuple[str, ...] = MLP_TARGETS,
    layer_module_prefix: str = "model.layers",
) -> list[str]:
    """Explicit PEFT ``target_modules``: attention projections on all layers (suffix match)
    + MLP projections only on layers in ``[lo, hi]`` inclusive (fully-qualified names)."""
    lo, hi = mlp_layer_range
    if lo > hi:
        raise ValueError(f"mlp_layer_range lo={lo} > hi={hi}")
    targets = list(attention_targets)
    for i in range(lo, hi + 1):
        for proj in mlp_targets:
            targets.append(f"{layer_module_prefix}.{i}.mlp.{proj}")
    return targets


def build_lora_config(cfg: dict[str, Any]) -> LoraConfig:
    """Build a PEFT ``LoraConfig`` from a configs/lora.yaml-shaped dict."""
    from peft import LoraConfig

    return LoraConfig(
        r=int(cfg["r"]),
        lora_alpha=int(cfg["alpha"]),
        lora_dropout=float(cfg["dropout"]),
        target_modules=lora_target_modules(
            tuple(cfg["mlp_layer_range"]),
            attention_targets=tuple(cfg.get("attention_targets", ATTENTION_TARGETS)),
            mlp_targets=tuple(cfg.get("mlp_targets", MLP_TARGETS)),
        ),
        bias="none",
        task_type="CAUSAL_LM",
    )
