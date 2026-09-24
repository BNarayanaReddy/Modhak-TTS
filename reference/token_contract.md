# Token contract — `llama-3-audio-tokenizer`

The authoritative description of what every token ID **means** in this project's
compiled data, checkpoints and eval paths. Generated from the tokenizer itself,
not transcribed by hand.

    tokenizer:   checkpoints/llama-3-audio-tokenizer
    vocab size:  156,960
    fingerprint: b645cf6612315393eea21d29e94304bdbcbfee131a25de5eecf6bae6ee9f39c5
    base model:  Llama-3.2-3B (128,000 BPE + 256 Llama specials)

**A token ID is meaningless without this contract.** Compiled parquet stores raw
integers; nothing in the data says which tokenizer minted them. Compiling with one
tokenizer and training with another is silent — training runs, loss descends, and
the model learns the wrong symbol for every audio frame. That is not hypothetical:
it is exactly how the gemma3 SNAC base mismatch produced undecodable checkpoints,
caught only at eval. Hence the fingerprint, stamped at compile time and asserted at
training startup and in the eval decode path (`scripts/token_contract.py`).

---

## 1. ID map

| range | count | what it is |
|---|---:|---|
| `0 – 127,999` | 128,000 | Llama-3 BPE text vocabulary |
| `128,000 – 128,255` | 256 | stock Llama-3 specials (`<\|begin_of_text\|>`, `<\|eot_id\|>`, `<\|reserved_special_token_N\|>` …) |
| `128,256 – 128,265` | 10 | **project control tokens** (§2) |
| `128,266 – 156,937` | 28,672 | **SNAC audio codes** — 7 codebooks × 4,096 (§3) |
| `156,938 – 156,959` | 22 | **conditioning + paralinguistic tokens** (§4) |

28,960 added tokens in total (28,672 SNAC + 288 non-SNAC).

Note the layout is *discontinuous*: the control block sits immediately below the
audio band and the conditioning block immediately above it. Anything that assumes
"all added tokens are contiguous above the base" is wrong.

---

## 2. Control tokens — `128,256 – 128,265`

| id | token | role |
|---:|---|---|
| 128256 | `<\|reserved_0\|>` | unused |
| **128257** | `<\|start_of_speech\|>` | opens the audio span; the model emits this itself as its first generated token |
| **128258** | `<\|end_of_speech\|>` | closes the audio span |
| **128259** | `<\|start_of_human\|>` | opens the prompt turn |
| **128260** | `<\|end_of_human\|>` | closes the prompt turn |
| **128261** | `<\|start_of_ai\|>` | opens the model turn |
| **128262** | `<\|end_of_ai\|>` | closes the model turn |
| 128263 | `<\|pad\|>` | padding |
| 128264 | `<\|reserved_8\|>` | unused |
| 128265 | `<\|reserved_9\|>` | unused |

---

## 3. Audio band — `128,266 – 156,937`

    audio_token_base_id = 128266
    codebooks           = 7
    codebook size       = 4096
    total audio tokens  = 28672        (base .. base + 7*4096 - 1 = 156937)

SNAC 24 kHz emits three codebooks at a 1:2:4 temporal ratio:

    c0: [seq_len]     12 Hz   coarsest
    c1: [2*seq_len]   23 Hz
    c2: [4*seq_len]   47 Hz   finest

These are flattened to **7 tokens per frame**, in this fixed interleave:

    frame i -> [ c0[i], c1[2i], c2[4i], c2[4i+1], c1[2i+1], c2[4i+2], c2[4i+3] ]

Each raw code (0–4095) is offset by its **position in the frame**, not by which
codebook it came from:

| frame position | source | offset | id range |
|---:|---|---|---|
| 0 | `c0[i]` | base + 0×4096 | 128,266 – 132,361 |
| 1 | `c1[2i]` | base + 1×4096 | 132,362 – 136,457 |
| 2 | `c2[4i]` | base + 2×4096 | 136,458 – 140,553 |
| 3 | `c2[4i+1]` | base + 3×4096 | 140,554 – 144,649 |
| 4 | `c1[2i+1]` | base + 4×4096 | 144,650 – 148,745 |
| 5 | `c2[4i+2]` | base + 5×4096 | 148,746 – 152,841 |
| 6 | `c2[4i+3]` | base + 6×4096 | 152,842 – 156,937 |

So positions 1 and 4 are both `c1`, and 2/3/5/6 are all `c2` — the offset encodes
*where in the frame* a code sits, which is what makes the stream decodable without
a separate structure signal.

**Consecutive duplicate frames are removed** at encode time (frames sharing the
same `c0`), so token count is not exactly proportional to duration.

Derived constants (`scripts/snac_tokenizer.py`):

    samples per c0 frame  = 512      (encoder_rates [2,4,8,8] x vq_stride 4)
    streaming window      = 4 frames = 28 tokens, middle frame kept
    long-audio window     = 512 frames (~43.7 s), 4-frame context each side

Rule of thumb used throughout the project: **82 tokens ≈ 1 second of audio.**

---

## 4. Conditioning + paralinguistics — `156,938 – 156,959`

| id | token | role |
|---:|---|---|
| 156938 / 156939 | `<\|speaker>` / `<speaker\|>` | wrap a speaker name |
| 156940 / 156941 | `<\|style>` / `<style\|>` | wrap a style label (rasa) or accent string (globe) |
| 156942 / 156943 | `<\|env>` / `<env\|>` | wrap an environment label; open class, label stays free text |

Non-verbals — a **closed set of 16**, one token each (no wrapper):

| id | token | | id | token |
|---:|---|---|---:|---|
| 156944 | `<\|nv_breath\|>` | | 156952 | `<\|nv_hum\|>` |
| 156945 | `<\|nv_stammer\|>` | | 156953 | `<\|nv_gasp\|>` |
| 156946 | `<\|nv_throat\|>` | | 156954 | `<\|nv_wheeze\|>` |
| 156947 | `<\|nv_laugh\|>` | | 156955 | `<\|nv_sneeze\|>` |
| 156948 | `<\|nv_swallow\|>` | | 156956 | `<\|nv_snort\|>` |
| 156949 | `<\|nv_sniff\|>` | | 156957 | `<\|nv_yawn\|>` |
| 156950 | `<\|nv_sigh\|>` | | 156958 | `<\|nv_groan\|>` |
| 156951 | `<\|nv_cough\|>` | | 156959 | `<\|nv_burp\|>` |

The asymmetry is deliberate: non-verbals are a closed vocabulary and get dedicated
tokens; environment labels are an open class, so only the wrapper is a token and
the label inside stays free text — a new environment class needs no tokenizer
change. Source-form mapping (`<breath>`, `[bird_squawk]`, …) is in
`scripts/paralinguistics.py`.

---

## 5. Sequence layouts

Built by `scripts/chat_templates.py`. `prompt_end` is defined as everything up to
and **including** `<|start_of_ai|>` — the prompt therefore stops *before*
`<|start_of_speech|>`, which the model emits itself.

**TTS with conditioning** (rasa, globe, bhili):

    <|start_of_human|><|begin_of_text|>
      <|speaker>NAME<speaker|>\n
      <|style>LABEL<style|>\n
      TEXT
    <|eot_id|><|end_of_human|><|start_of_ai|>
      <|start_of_speech|> ...audio... <|end_of_speech|>
    <|end_of_ai|>

Metadata order is always **speaker → style → accent**, each block followed by a
newline. An empty value emits nothing at all (no empty wrapper).

**Multi-turn conversation** (`gemini_vc_conversational`) — no metadata prefix;
speaker labels are inline turn markers inside the text:

    <|start_of_human|><|begin_of_text|>
      <|speaker>A<speaker|>\nturn one\n\n<|speaker>B<speaker|>\nturn two ...
    <|eot_id|><|end_of_human|><|start_of_ai|><|start_of_speech|> ...

Turn separator is a **blank line** (`\n\n`) before each subsequent
`<|speaker>` marker. `normalize_text` preserves newline runs (the old
collapse-to-one-space behavior was removed when the `\n\n` issue was fixed),
so the separator reaches the model verbatim; verified present in 100% of
compiled multi-turn rows in both `gemini_vc` and `gemini_src_conv`. The `\n`
after `<speaker|>` also survives. Inference prompts must match this form.

---

## 6. Fingerprint

`scripts/token_contract.py::compute_fingerprint` is a sha256 over exactly the
things that change what an ID means:

- the full added-token map (content → id), sorted
- vocab size
- the SNAC base id and layout constants (codebooks × codebook size)

It deliberately **excludes** `tokenizer_config.json` niceties — padding side, chat
template, `model_max_length` — because those do not change a token's meaning and
including them would fire on cosmetic edits.

    current fingerprint: b645cf6612315393eea21d29e94304bdbcbfee131a25de5eecf6bae6ee9f39c5

Why a content hash and not a range check: the audio-band range check in
`snac_tokenizer.decode_audio` catches the loud failure (IDs outside the band). It
cannot catch the quiet one — sibling tokenizers `llama-3-audio-tokenizer`
(156,938), `-tok_trimmed` (156,942) and `-style` (156,952) all share base 128,266,
so every range check passes while `<|style>` means something different in the data
than in the model.

Mismatch is a hard error, never a warning. A silent wrong-tokenizer run costs a
full training cycle.

---

*Generated from `checkpoints/llama-3-audio-tokenizer` with `scripts/token_contract.py`;
layout constants from `scripts/snac_tokenizer.py`, sequence templates from
`scripts/chat_templates.py`.*
