"""HF Trainer subclass: swaps in the per-codebook weighted loss and, when the data is
pre-ordered by the mixture sampler, uses a sequential sampler so every batch keeps its
fixed source composition (Decision 3). LoRA-adapter checkpointing is the Trainer default
(``save_model`` on a PEFT model writes adapter_model.safetensors + adapter_config.json).
"""
from __future__ import annotations

from typing import Any

from transformers import Trainer

from indic_speak_ft.train.loss import DEFAULT_POSITION_WEIGHTS, weighted_audio_ce


class WeightedLoRATrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        snac_base: int,
        stop_token_id: int,
        position_weights: tuple[float, ...] = DEFAULT_POSITION_WEIGHTS,
        stop_weight: float = 1.0,
        enable_weighting: bool = True,
        preserve_mixture_order: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._snac_base = snac_base
        self._stop_id = stop_token_id
        self._pos_w = position_weights
        self._stop_w = stop_weight
        self._enable_w = enable_weighting
        self._preserve_order = preserve_mixture_order

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = weighted_audio_ce(
            outputs.logits, labels, base=self._snac_base, stop_token_id=self._stop_id,
            position_weights=self._pos_w, stop_weight=self._stop_w,
            enable_weighting=self._enable_w)
        return (loss, outputs) if return_outputs else loss

    def _get_train_sampler(self, *args: Any, **kwargs: Any) -> Any | None:
        if self._preserve_mixture_order():
            from torch.utils.data import SequentialSampler

            return SequentialSampler(self.train_dataset)
        return super()._get_train_sampler(*args, **kwargs)

    def _preserve_mixture_order(self) -> bool:
        return self._preserve_order and self.train_dataset is not None
