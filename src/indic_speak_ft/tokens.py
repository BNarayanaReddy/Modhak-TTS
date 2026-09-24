"""Runtime token-contract helpers: resolve/validate the indic-speak token IDs from a
live tokenizer and provide the SNAC frame ↔ token-id math, so nothing is hardcoded.

This mirrors ``reference/inference.py`` and ``bodhan_genai.tts.codec.snac`` exactly:
the SNAC audio band is a single base token ``<|snac_0|>`` plus arithmetic per-frame-
position offsets ``base + position*4096`` (position 0..6). Callers pass a resolved
``TokenContract`` around (Decision: no module-level global token state) — the only
module constants here are *structural* SNAC layout facts, not model-specific IDs.

See docs/RECON.md conflict A: the tokenizer names audio tokens PER-CODE
(``<|snac_N|>`` == base + N, N=0..28671), NOT per-codebook — so GOAL.md's step-4
assertion is replaced by :func:`assert_snac_contract` below.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Protocol

# --- structural SNAC layout (token_contract.md §3; bodhan_genai codec/snac.py) ------
NUM_CODEBOOKS: int = 7                       # flattened tokens per SNAC frame
CODEBOOK_SIZE: int = 4096                    # raw codes per position, 0..4095
TOTAL_AUDIO_TOKENS: int = NUM_CODEBOOKS * CODEBOOK_SIZE  # 28,672

# The fixed 7-position interleave of frame i, naming the SNAC source code at each
# frame position (from reference ids_to_codes / token_contract.md §3):
#   frame i -> [c0[i], c1[2i], c2[4i], c2[4i+1], c1[2i+1], c2[4i+2], c2[4i+3]]
INTERLEAVE_PATTERN: tuple[str, ...] = (
    "c0[i]", "c1[2i]", "c2[4i]", "c2[4i+1]", "c1[2i+1]", "c2[4i+2]", "c2[4i+3]",
)

SNAC_BASE_TOKEN: str = "<|snac_0|>"

# Required special tokens (resolved from the tokenizer, never hardcoded).
_STRUCTURAL: tuple[tuple[str, str], ...] = (
    ("begin_of_text", "<|begin_of_text|>"),
    ("eot_id", "<|eot_id|>"),
    ("start_of_human", "<|start_of_human|>"),
    ("end_of_human", "<|end_of_human|>"),
    ("start_of_ai", "<|start_of_ai|>"),
    ("end_of_ai", "<|end_of_ai|>"),
    ("start_of_speech", "<|start_of_speech|>"),
    ("end_of_speech", "<|end_of_speech|>"),
)
_CONDITIONING: tuple[tuple[str, str], ...] = (
    ("speaker_open", "<|speaker>"),
    ("speaker_close", "<speaker|>"),
    ("style_open", "<|style>"),
    ("style_close", "<style|>"),
)


class TokenizerLike(Protocol):
    """The minimal surface we need — satisfied by a transformers tokenizer, the raw
    ``tokenizers.Tokenizer`` (via a thin shim), or a test fake."""

    def convert_tokens_to_ids(self, token: str) -> int | None: ...


def _require_id(tokenizer: TokenizerLike, name: str) -> int:
    """Resolve a required special token, raising loudly (never silently mapping to
    ``unk``) — the failure mode ``token_contract.md`` §6 warns costs a whole run."""
    tid = tokenizer.convert_tokens_to_ids(name)
    unk = getattr(tokenizer, "unk_token_id", None)
    if tid is None or (unk is not None and tid == unk):
        raise ValueError(
            f"Required special token {name!r} not found in tokenizer "
            f"(resolved to {tid!r}). Wrong or un-extended tokenizer — refusing to "
            f"continue (a silent unk mapping would train/generate on garbage)."
        )
    return int(tid)


@dataclass(frozen=True)
class TokenContract:
    """Resolved token IDs for one tokenizer. Immutable; passed to collator/generator."""

    begin_of_text: int
    eot_id: int
    start_of_human: int
    end_of_human: int
    start_of_ai: int
    end_of_ai: int
    start_of_speech: int
    end_of_speech: int
    speaker_open: int
    speaker_close: int
    style_open: int
    style_close: int
    snac_base: int

    @property
    def audio_lo(self) -> int:
        """Inclusive low bound of the SNAC audio band."""
        return self.snac_base

    @property
    def audio_hi(self) -> int:
        """Exclusive high bound of the SNAC audio band (== base + 28672)."""
        return self.snac_base + TOTAL_AUDIO_TOKENS

    @property
    def frame_position_bases(self) -> tuple[int, ...]:
        """The 7 per-frame-position offset bases: base + p*4096 for p in 0..6."""
        return tuple(self.snac_base + p * CODEBOOK_SIZE for p in range(NUM_CODEBOOKS))

    def as_dict(self) -> dict[str, int]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def load_tokens(tokenizer: TokenizerLike, *, check_snac: bool = True) -> TokenContract:
    """Resolve the full token contract from a live tokenizer and (by default) assert
    the SNAC band's integrity. Parameterizable by design — no global state."""
    resolved = {name: _require_id(tokenizer, tok) for name, tok in (*_STRUCTURAL, *_CONDITIONING)}
    snac_base = _require_id(tokenizer, SNAC_BASE_TOKEN)
    contract = TokenContract(snac_base=snac_base, **resolved)
    if check_snac:
        assert_snac_contract(tokenizer, contract)
    return contract


def assert_snac_contract(tokenizer: TokenizerLike, contract: TokenContract) -> None:
    """Verify the real SNAC contract (replaces GOAL.md step-4's incorrect assertion).

    The audio tokens are named per-code — ``<|snac_N|>`` == base + N for N in
    0..28671 — so we check: (1) the base token is ``<|snac_0|>``; (2) per-code
    contiguity ``<|snac_N|>`` == base + N across the band; (3) the band's top token
    ``<|snac_28671|>`` == base + 28671 and sits exactly below ``<|speaker>``; (4) the
    7 per-frame-position bases base + p*4096 all fall inside the band.
    """
    base = contract.snac_base
    if base != tokenizer.convert_tokens_to_ids(SNAC_BASE_TOKEN):
        raise AssertionError("snac_base does not match <|snac_0|>")

    # (2) per-code contiguity, sampled across the band (endpoints + codebook seams)
    seams = {0, 1, 2, CODEBOOK_SIZE - 1, CODEBOOK_SIZE, 2 * CODEBOOK_SIZE, TOTAL_AUDIO_TOKENS - 1}
    for n in sorted(seams):
        tid = tokenizer.convert_tokens_to_ids(f"<|snac_{n}|>")
        if tid != base + n:
            raise AssertionError(
                f"SNAC band not contiguous per-code: <|snac_{n}|> resolved to {tid!r}, "
                f"expected base+{n} = {base + n}. (GOAL step-4 assumed per-codebook "
                f"naming base+{n}*4096 — that scheme does not exist; see RECON conflict A.)"
            )

    # (3) band top just below <|speaker>
    top = tokenizer.convert_tokens_to_ids(f"<|snac_{TOTAL_AUDIO_TOKENS - 1}|>")
    if top != contract.audio_hi - 1:
        raise AssertionError(f"SNAC band top {top} != base+28671 {contract.audio_hi - 1}")
    if contract.speaker_open != contract.audio_hi:
        raise AssertionError(
            f"<|speaker> ({contract.speaker_open}) must sit immediately above the audio "
            f"band top ({contract.audio_hi - 1}); got a gap."
        )

    # (4) the 7 frame-position bases are all in-band
    for p, fb in enumerate(contract.frame_position_bases):
        if not (contract.audio_lo <= fb < contract.audio_hi):
            raise AssertionError(f"frame-position base {p} ({fb}) out of audio band")


# --- SNAC frame ↔ token-id math (pure arithmetic; the collator/generator reuse it) --
def is_audio_token(base: int, token_id: int) -> bool:
    """True iff ``token_id`` lies in the SNAC audio band [base, base+28672)."""
    return base <= token_id < base + TOTAL_AUDIO_TOKENS


def snac_id_to_token(base: int, codebook_pos: int, code_id: int) -> int:
    """Flat token id for a raw SNAC ``code_id`` at frame position ``codebook_pos``.

    ``codebook_pos`` is the 0..6 position within the 7-token frame (NOT the SNAC
    codebook index — positions 1&4 are both codebook c1, 2/3/5/6 are c2).
    """
    if not 0 <= codebook_pos < NUM_CODEBOOKS:
        raise ValueError(f"codebook_pos {codebook_pos} out of range 0..{NUM_CODEBOOKS - 1}")
    if not 0 <= code_id < CODEBOOK_SIZE:
        raise ValueError(f"code_id {code_id} out of range 0..{CODEBOOK_SIZE - 1}")
    return base + codebook_pos * CODEBOOK_SIZE + code_id


def token_to_snac(base: int, token_id: int) -> tuple[int, int] | None:
    """Inverse of :func:`snac_id_to_token`: ``(codebook_pos, code_id)`` or ``None`` if
    ``token_id`` is not an audio token."""
    if not is_audio_token(base, token_id):
        return None
    off = token_id - base
    return off // CODEBOOK_SIZE, off % CODEBOOK_SIZE


def build_frame(
    c0_i: int, c1_2i: int, c1_2i1: int, c2_4i: int, c2_4i1: int, c2_4i2: int, c2_4i3: int, *,
    base: int,
) -> list[int]:
    """Inverse of ``reference.ids_to_codes`` for one frame: the 7 raw SNAC codes (named
    by their SNAC source c0/c1/c2 indices) → 7 flat token ids in frame-position order
    ``[c0[i], c1[2i], c2[4i], c2[4i+1], c1[2i+1], c2[4i+2], c2[4i+3]]``.

    Used by the training collator to reproduce the interleave byte-for-byte. Arguments
    are in SNAC-source order (as GOAL.md Phase 1 specifies); the returned ids are in
    frame-position order.
    """
    # frame position -> raw code (see INTERLEAVE_PATTERN)
    codes_by_position = (c0_i, c1_2i, c2_4i, c2_4i1, c1_2i1, c2_4i2, c2_4i3)
    return [snac_id_to_token(base, p, code) for p, code in enumerate(codes_by_position)]


def parse_frame(base: int, ids: list[int]) -> tuple[int, int, int, int, int, int, int]:
    """Inverse of :func:`build_frame`: 7 frame-position ids → the 7 raw codes in SNAC-
    source order ``(c0_i, c1_2i, c1_2i1, c2_4i, c2_4i1, c2_4i2, c2_4i3)``."""
    if len(ids) != NUM_CODEBOOKS:
        raise ValueError(f"expected {NUM_CODEBOOKS} ids, got {len(ids)}")
    codes = []
    for p, tid in enumerate(ids):
        parsed = token_to_snac(base, tid)
        if parsed is None or parsed[0] != p:
            raise ValueError(f"id {tid} at frame position {p} is not a valid position-{p} audio token")
        codes.append(parsed[1])
    c0_i, c1_2i, c2_4i, c2_4i1, c1_2i1, c2_4i2, c2_4i3 = codes
    return c0_i, c1_2i, c1_2i1, c2_4i, c2_4i1, c2_4i2, c2_4i3
