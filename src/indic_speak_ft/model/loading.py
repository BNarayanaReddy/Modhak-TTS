"""Load the base LM (bf16 + sdpa, matching reference/inference.py) and the eval-time codec
(SNAC quantizer + fine-tuned Vocos), and report parameter counts.

Loading is thin (the heavy lifting is transformers/snac/the repo's vocos loader); the value
here is a single place that pins the dtype/attention/device-map policy and computes the
param-count summary the training log prints. Works on transformers 4.x and 5.x (the
`torch_dtype`→`dtype` from_pretrained kwarg rename).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def load_base_lm(
    model_dir: str,
    *,
    dtype: str = "bfloat16",
    attn_implementation: str = "sdpa",
    device_map: Any | None = None,
    max_memory: dict | None = None,
    offload_folder: str | None = None,
    low_cpu_mem_usage: bool = True,
):
    """Load ``LlamaForCausalLM`` with the reference's dtype/attention policy."""
    import torch
    import transformers
    from transformers import LlamaForCausalLM

    dt = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    dtype_kw = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    kwargs: dict[str, Any] = {"attn_implementation": attn_implementation,
                              "low_cpu_mem_usage": low_cpu_mem_usage}
    if device_map is not None:
        kwargs["device_map"] = device_map
    if max_memory is not None:
        kwargs["max_memory"] = max_memory
    if offload_folder is not None:
        kwargs["offload_folder"] = offload_folder
    return LlamaForCausalLM.from_pretrained(model_dir, **kwargs, **{dtype_kw: dt}).eval()


def load_snac(snac_path_or_repo: str, device: str = "cpu"):
    """Load the SNAC 24 kHz codec (quantizer + decoder) from a local dir or a hub id."""
    import torch
    from snac import SNAC

    if os.path.isdir(snac_path_or_repo):
        model = SNAC.from_config(os.path.join(snac_path_or_repo, "config.json"))
        state = torch.load(os.path.join(snac_path_or_repo, "pytorch_model.bin"),
                           map_location=device, weights_only=True)
        model.load_state_dict(state)
    else:
        model = SNAC.from_pretrained(snac_path_or_repo)
    return model.eval().to(device)


def load_vocos(model_dir: str, device: str = "cpu"):
    """Load the fine-tuned Vocos decoder via the model repo's own ``vocos/load.py``."""
    sys.path.insert(0, str(model_dir))
    from vocos.load import load_vocos as _load

    return _load(str(Path(model_dir) / "vocos" / "best.pt"), device=device)


@dataclass(frozen=True)
class ParamCounts:
    total: int
    trainable: int
    frozen: int
    trainable_pct: float


def parameter_counts(model) -> ParamCounts:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return ParamCounts(total, trainable, total - trainable, 100.0 * trainable / max(total, 1))


def format_param_counts(model) -> str:
    c = parameter_counts(model)
    return (f"params: total={c.total/1e9:.3f}B trainable={c.trainable/1e6:.2f}M "
            f"({c.trainable_pct:.3f}%) frozen={c.frozen/1e9:.3f}B")
