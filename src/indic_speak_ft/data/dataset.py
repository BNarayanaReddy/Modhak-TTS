"""Torch dataset + padding collate, and the compile driver that turns streamed corpus rows
into training examples (loader → SNAC compile → collator).

Right-pads a batch to its longest member (pad on input_ids, −100 on labels, 0 on attention)
— standard causal-LM SFT. The compile driver is what the smoke/full runs call to build the
example list; it applies the gates and skips rejects.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import torch
from torch.utils.data import Dataset

from indic_speak_ft.data.collator import build_training_sequence
from indic_speak_ft.data.compile import snac_encode_tokens
from indic_speak_ft.data.loader import row_to_clip
from indic_speak_ft.tokens import TokenContract


class PrecompiledTTSDataset(Dataset):
    """Wraps a list of ``{"input_ids", "labels"}`` examples (already compiled + collated)."""

    def __init__(self, examples: list[dict[str, list[int]]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        e = self.examples[i]
        return {
            "input_ids": torch.tensor(e["input_ids"], dtype=torch.long),
            "labels": torch.tensor(e["labels"], dtype=torch.long),
        }


def pad_collate(batch: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Right-pad a batch to its longest sequence."""
    maxlen = max(b["input_ids"].size(0) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = b["input_ids"].size(0)
        pad = maxlen - n
        input_ids.append(torch.cat([b["input_ids"], torch.full((pad,), pad_token_id, dtype=torch.long)]))
        labels.append(torch.cat([b["labels"], torch.full((pad,), -100, dtype=torch.long)]))
        attn.append(torch.cat([torch.ones(n, dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
    return {"input_ids": torch.stack(input_ids), "labels": torch.stack(labels),
            "attention_mask": torch.stack(attn)}


def compile_rows_to_examples(
    rows: Iterable[dict[str, Any]],
    *,
    source_hf_path: str,
    license: str,
    language_id: str,
    speaker_assignment: dict,
    target_sr: int,
    gates: dict,
    tokenizer,
    contract: TokenContract,
    snac_model,
    device: str = "cpu",
    include_stop_in_loss: bool = True,
    held_out_keys: set[str] | None = None,
    limit: int | None = None,
) -> Iterator[dict[str, list[int]]]:
    """loader → SNAC compile → collator, yielding ``{"input_ids","labels"}`` for kept rows."""
    made = 0
    for row in rows:
        meta, wav, gate = row_to_clip(
            row, source_hf_path=source_hf_path, license=license, language_id=language_id,
            speaker_assignment=speaker_assignment, target_sr=target_sr, gates=gates,
            held_out_keys=held_out_keys)
        if not gate.kept or meta is None or wav is None:
            continue
        audio_ids = snac_encode_tokens(snac_model, wav, contract.snac_base, device=device)
        if len(audio_ids) < 7:
            continue
        seq = build_training_sequence(
            tokenizer, contract, meta.transcript.text_normalized, meta.speaker.name, audio_ids,
            include_stop_in_loss=include_stop_in_loss)
        yield {"input_ids": seq.input_ids, "labels": seq.labels}
        made += 1
        if limit is not None and made >= limit:
            return
