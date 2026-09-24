"""Training-sequence builder: assemble ``[prompt | audio | <|end_of_speech|>]`` with a
label mask that trains only the audio span + the stop token (Decision 4, refined per the
2026-09-24 call to include ``<|end_of_speech|>`` so the model learns to STOP — directly
targeting the base model's Marathi over-generation).

The prompt mirrors ``reference/inference.py``'s ``build_prompt`` exactly (speaker always
injected, style always empty — Decision 6); ``tests/test_collator.py`` asserts byte-for-
byte equality against the reference on real inputs. The boundary matches inference.py:
the prompt ends at ``<|start_of_speech|>`` (the model's first generated token is audio),
and no ``<|end_of_ai|>`` is appended (the reference never emits it — resolves RECON
conflict D).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from indic_speak_ft.tokens import NUM_CODEBOOKS, TokenContract, is_audio_token

IGNORE_INDEX = -100


class TextTokenizer(Protocol):
    """Just the text-encoding surface the prompt needs (special-token IDs come from the
    resolved ``TokenContract``, not the tokenizer, so nothing is hardcoded)."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


def build_prompt_ids(
    tokenizer: TextTokenizer,
    contract: TokenContract,
    text: str,
    speaker: str,
    style: str = "",
) -> list[int]:
    """Mirror of ``reference.inference.build_prompt`` using resolved contract IDs.

    Returns ids up to and INCLUDING ``<|start_of_speech|>`` (the inference boundary).
    Kept structurally identical to the reference so the byte-for-byte test can guard it.
    """
    nl = tokenizer.encode("\n", add_special_tokens=False)

    blocks: list[list[int]] = []
    if speaker.strip():
        blocks.append([contract.speaker_open, *tokenizer.encode(speaker.strip()), contract.speaker_close])
    if style.strip():
        blocks.append([contract.style_open, *tokenizer.encode(style.strip()), contract.style_close])

    meta: list[int] = []
    for i, block in enumerate(blocks):
        if i:
            meta += nl
        meta += block
    if blocks:
        meta += nl

    body = [contract.begin_of_text, *meta, *tokenizer.encode(text), contract.eot_id]
    return [
        contract.start_of_human, *body, contract.end_of_human,
        contract.start_of_ai, contract.start_of_speech,
    ]


@dataclass(frozen=True)
class TrainingSequence:
    input_ids: list[int]
    labels: list[int]
    prompt_len: int  # number of leading (masked) prompt tokens, incl. <|start_of_speech|>

    @property
    def length(self) -> int:
        return len(self.input_ids)


def assemble_sequence(
    prompt_ids: list[int],
    audio_token_ids: list[int],
    contract: TokenContract,
    *,
    include_stop_in_loss: bool = True,
) -> TrainingSequence:
    """Compose the full training sequence + label mask from a prompt and the flattened
    SNAC audio ids. Loss falls on the audio tokens (always) and ``<|end_of_speech|>``
    (when ``include_stop_in_loss``); the prompt is masked with ``IGNORE_INDEX``.

    Validates that ``audio_token_ids`` are a whole number of in-band 7-token frames — a
    malformed audio target would otherwise train the model on garbage silently.
    """
    n = len(audio_token_ids)
    if n == 0 or n % NUM_CODEBOOKS != 0:
        raise ValueError(f"audio_token_ids length {n} is not a positive multiple of {NUM_CODEBOOKS}")
    if not all(is_audio_token(contract.snac_base, t) for t in audio_token_ids):
        raise ValueError("audio_token_ids contains a token outside the SNAC audio band")

    stop = contract.end_of_speech
    input_ids = [*prompt_ids, *audio_token_ids, stop]
    labels = [
        *([IGNORE_INDEX] * len(prompt_ids)),
        *audio_token_ids,
        stop if include_stop_in_loss else IGNORE_INDEX,
    ]
    return TrainingSequence(input_ids=input_ids, labels=labels, prompt_len=len(prompt_ids))


def build_training_sequence(
    tokenizer: TextTokenizer,
    contract: TokenContract,
    text: str,
    speaker: str,
    audio_token_ids: list[int],
    *,
    include_stop_in_loss: bool = True,
) -> TrainingSequence:
    """Convenience: prompt (speaker injected, style empty — Decision 6) + audio target."""
    prompt_ids = build_prompt_ids(tokenizer, contract, text, speaker, style="")
    return assemble_sequence(
        prompt_ids, audio_token_ids, contract, include_stop_in_loss=include_stop_in_loss,
    )
