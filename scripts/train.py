"""LoRA SFT training entry point. ``--smoke`` runs 100 steps on 20 compiled samples end-to-end
(load → LoRA + freeze/verify → SNAC compile → weighted loss → train → save adapter) to prove
the pipeline before the full server run. Configs composed from configs/{base,train,data,lora}.yaml.

  python scripts/train.py --smoke      # laptop, 6 GB GPU (GPU+CPU offload)
  python scripts/train.py              # full run (server)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from omegaconf import OmegaConf

from indic_speak_ft.data.dataset import PrecompiledTTSDataset, compile_rows_to_examples, pad_collate
from indic_speak_ft.data.loader import iter_parquet_rows
from indic_speak_ft.tokens import load_tokens

REPO = Path(__file__).resolve().parents[1]
LICENSES = {"SPRINGLab/IndicTTS_Marathi": "CC-BY-4.0", "ai4bharat/Rasa": "CC-BY-4.0",
            "SPRINGLab/IndicTTS-Hindi": "CC-BY-4.0", "SPRINGLab/IndicTTS-English": "CC-BY-4.0"}
# (source, lang, shard, columns, n) — smoke pulls both Marathi target voices.
SMOKE_PLAN = [
    ("SPRINGLab/IndicTTS_Marathi", "mr",
     "datasets/SPRINGLab/IndicTTS_Marathi/data/train-00000-of-00004.parquet", ["audio", "text", "gender"], 12),
    ("ai4bharat/Rasa", "mr",
     "datasets/ai4bharat/Rasa/Marathi/test-00000-of-00007.parquet", ["audio", "text", "gender"], 8),
]

# Full-run source plan: (mixture_pool, source, lang, shard, columns). Each entry compiles unique
# clips INTO its pool; the pool is then drawn per configs/data.yaml `mixture` weights with a fixed
# per-batch composition (replay.py). Per-source shards are GENDER-HOMOGENEOUS (verified against the
# real corpora), so each pool deliberately lists the shards it needs to hit both target voices.
# The D15 speaker map (configs/data.yaml) turns each row's `gender` into the library voice.
# Per-pool clip COUNTS live in configs/train.yaml `full.clips` (coding standard: scale in config).
AUDIO_COLS = ["audio", "text", "gender"]
FULL_PLAN = [
    # springlab_marathi (0.56) — domain signal, BOTH target voices (MR shard 0 = gender0/Anagha,
    # shard 3 = gender1/Chinmay); mirrors the eval-bucket build.
    ("springlab_marathi", "SPRINGLab/IndicTTS_Marathi", "mr",
     "datasets/SPRINGLab/IndicTTS_Marathi/data/train-00000-of-00004.parquet", AUDIO_COLS),  # Anagha (F)
    ("springlab_marathi", "SPRINGLab/IndicTTS_Marathi", "mr",
     "datasets/SPRINGLab/IndicTTS_Marathi/data/train-00003-of-00004.parquet", AUDIO_COLS),  # Chinmay (M)
    # rasa_marathi (0.24) — the anchor, BALANCED across both voices so drift is monitored
    # symmetrically (D15 tightening). Rasa test shards are gender-homogeneous: test0-3 Male/Chinmay,
    # test4-6 Female/Anagha. Eval used only test0/test1 (both Chinmay), so training takes test2/3
    # (Chinmay) + test4/5 (Anagha) — disjoint from the eval shards, 50/50 across the two voices.
    ("rasa_marathi", "ai4bharat/Rasa", "mr",
     "datasets/ai4bharat/Rasa/Marathi/test-00002-of-00007.parquet", AUDIO_COLS),  # Chinmay (M)
    ("rasa_marathi", "ai4bharat/Rasa", "mr",
     "datasets/ai4bharat/Rasa/Marathi/test-00003-of-00007.parquet", AUDIO_COLS),  # Chinmay (M)
    ("rasa_marathi", "ai4bharat/Rasa", "mr",
     "datasets/ai4bharat/Rasa/Marathi/test-00004-of-00007.parquet", AUDIO_COLS),  # Anagha (F)
    ("rasa_marathi", "ai4bharat/Rasa", "mr",
     "datasets/ai4bharat/Rasa/Marathi/test-00005-of-00007.parquet", AUDIO_COLS),  # Anagha (F)
    # hindi_replay (0.15) — anti-forgetting for the MEASURED capability: retention_hindi is voiced by
    # Amit, so replay uses Amit too. Hindi shards: tr0-3 gender0/Kavya, tr4-9 gender1/Amit → use tr4/5.
    ("hindi_replay", "SPRINGLab/IndicTTS-Hindi", "hi",
     "datasets/SPRINGLab/IndicTTS-Hindi/data/train-00004-of-00010.parquet", AUDIO_COLS),  # Amit (M)
    ("hindi_replay", "SPRINGLab/IndicTTS-Hindi", "hi",
     "datasets/SPRINGLab/IndicTTS-Hindi/data/train-00005-of-00010.parquet", AUDIO_COLS),  # Amit (M)
    # english_replay (0.05) — backbone health; English has no gender column → fixed Amit (data.yaml).
    ("english_replay", "SPRINGLab/IndicTTS-English", "en",
     "datasets/SPRINGLab/IndicTTS-English/data/train-00000-of-00028.parquet", ["audio", "text"]),
]


def load_config():
    parts = {}
    for n in ("base", "train", "data", "lora"):
        part = OmegaConf.load(REPO / "configs" / f"{n}.yaml")
        OmegaConf.resolve(part)  # resolve within-file interpolations (base.yaml ${offload}) before nesting
        parts[n] = part
    return OmegaConf.create(parts)


def tokenizer_shim(model_dir: str):
    from tokenizers import Tokenizer

    raw = Tokenizer.from_file(f"{model_dir}/tokenizer.json")

    class Shim:
        bos_token_id = raw.token_to_id("<|begin_of_text|>")
        eos_token_id = raw.token_to_id("<|end_of_text|>")
        unk_token_id = None

        def convert_tokens_to_ids(self, s: str):
            return raw.token_to_id(s)

        def encode(self, s: str, add_special_tokens: bool = False):
            return raw.encode(s, add_special_tokens=add_special_tokens).ids

    return Shim()


def _compile_n(cfg, tok, contract, snac, held_out, *, source, lang, shard, cols, n,
               include_stop, device, max_seq_len, fs, smap, gates, tsr):
    """Stream+compile up to ``n`` clips (<= max_seq_len tokens) from one shard, stopping the
    SNAC-encode loop as soon as ``n`` short examples are collected (no wasted encodes)."""
    rows = iter_parquet_rows(shard, columns=cols, max_rows=n * 8, filesystem=fs)
    got: list[dict] = []
    for ex in compile_rows_to_examples(
            rows, source_hf_path=source, license=LICENSES[source], language_id=lang,
            speaker_assignment=smap, target_sr=tsr, gates=gates, tokenizer=tok, contract=contract,
            snac_model=snac, device=device, include_stop_in_loss=include_stop, held_out_keys=held_out):
        if len(ex["input_ids"]) <= max_seq_len:  # bound activation memory
            got.append(ex)
            if len(got) >= n:
                break
    return got


def build_examples(cfg, tok, contract, snac, held_out, *, plan, include_stop,
                   device="cpu", max_seq_len=1200):
    fs, smap, gates, tsr = _compile_env(cfg)
    examples: list[dict] = []
    for source, lang, shard, cols, n in plan:
        got = _compile_n(cfg, tok, contract, snac, held_out, source=source, lang=lang, shard=shard,
                         cols=cols, n=n, include_stop=include_stop, device=device,
                         max_seq_len=max_seq_len, fs=fs, smap=smap, gates=gates, tsr=tsr)
        examples += got
        print(f"  compiled {len(got):2d} examples from {source} (<= {max_seq_len} tok)")
    return examples


def build_full_examples(cfg, tok, contract, snac, held_out, *, include_stop, device):
    """Compile the full per-pool corpora and return (pools, ordered) where ``ordered`` is the
    mixture-sampled training sequence: every effective-batch window matches configs/data.yaml
    `mixture` (Decision 3/7). Per-pool clip counts + max_seq_len come from configs/train.yaml.
    """
    from indic_speak_ft.train.replay import build_mixture_ordered_examples

    fs, smap, gates, tsr = _compile_env(cfg)
    full = cfg.train.full
    clips = OmegaConf.to_container(full.clips, resolve=True)
    max_seq_len = int(full.max_seq_len)

    entries_by_pool: dict[str, list[tuple]] = {}
    for pool, source, lang, shard, cols in FULL_PLAN:
        entries_by_pool.setdefault(pool, []).append((source, lang, shard, cols))

    pools: dict[str, list[dict]] = {}
    for pool, entries in entries_by_pool.items():
        target = int(clips[pool])
        per_entry = math.ceil(target / len(entries))
        pool_examples: list[dict] = []
        for source, lang, shard, cols in entries:
            need = min(per_entry, target - len(pool_examples))
            if need <= 0:
                break
            got = _compile_n(cfg, tok, contract, snac, held_out, source=source, lang=lang,
                             shard=shard, cols=cols, n=need, include_stop=include_stop,
                             device=device, max_seq_len=max_seq_len, fs=fs, smap=smap,
                             gates=gates, tsr=tsr)
            pool_examples += got
            print(f"  [{pool:16s}] +{len(got):4d} from {source} :: {shard.split('/')[-1]}")
        if not pool_examples:
            raise RuntimeError(f"mixture pool {pool!r} compiled 0 examples")
        pools[pool] = pool_examples
        print(f"  pool {pool:16s} total {len(pool_examples)}")

    # Order for the mixture: guarantee the composition over a WINDOW large enough to seat every
    # source (a 4-example effective batch rounds English's 5% to 0). The trainer's sequential
    # sampler then walks this order in effective-batch groups, so each window spans a whole number
    # of optimizer steps and every source recurs on a fixed cadence.
    weights = OmegaConf.to_container(cfg.data.mixture, resolve=True)
    eff_batch = int(cfg.train.per_device_train_batch_size) * int(cfg.train.gradient_accumulation_steps)
    window = int(full.get("mixture_window", eff_batch))
    draws_needed = eff_batch * int(cfg.train.max_steps)
    ordered = build_mixture_ordered_examples(
        pools, weights, batch_size=window, num_batches=math.ceil(draws_needed / window),
        seed=int(cfg.train.seed))
    return pools, ordered[:draws_needed]


def _compile_env(cfg):
    """Shared compile inputs: HF filesystem + resolved speaker map / gates / target sample rate."""
    from huggingface_hub import HfFileSystem

    return (HfFileSystem(),
            OmegaConf.to_container(cfg.data.speaker_assignment, resolve=True),
            OmegaConf.to_container(cfg.data.gates, resolve=True),
            int(cfg.data.audio.target_sample_rate))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    tcfg = dict(cfg.train)
    if args.smoke:
        tcfg.update(OmegaConf.to_container(cfg.train.smoke, resolve=True))

    from indic_speak_ft.train.determinism import set_seed

    set_seed(int(cfg.train.seed), deterministic=bool(cfg.train.deterministic))

    from peft import get_peft_model

    from indic_speak_ft.model.freeze import (
        apply_freeze_policy,
        expected_trainable_params,
        verify_freeze,
    )
    from indic_speak_ft.model.loading import format_param_counts, load_base_lm, load_snac
    from indic_speak_ft.model.lora import build_lora_config

    model_dir, snac_dir = str(cfg.base.model_dir), str(cfg.base.snac_dir)
    tok = tokenizer_shim(model_dir)
    contract = load_tokens(tok)

    if bool(cfg.base.get("full_gpu", False)):
        print("loading base LM (bf16, sdpa) fully on GPU (server)…")
        lm = load_base_lm(model_dir, dtype="bfloat16", attn_implementation="sdpa", device_map={"": 0})
    else:
        print("loading base LM (bf16, sdpa, GPU+CPU split — laptop)…")
        lm = load_base_lm(model_dir, dtype="bfloat16", attn_implementation="sdpa", device_map="auto",
                          max_memory={0: str(cfg.base.gpu_mem), "cpu": "6GiB"},
                          offload_folder=f"{cfg.base.offload}/offload_tmp")
    model = get_peft_model(lm, build_lora_config(OmegaConf.to_container(cfg.lora, resolve=True)))
    apply_freeze_policy(model)
    c = lm.config
    exp = expected_trainable_params(
        num_layers=c.num_hidden_layers, hidden_size=c.hidden_size,
        num_attention_heads=c.num_attention_heads, num_key_value_heads=c.num_key_value_heads,
        intermediate_size=c.intermediate_size, mlp_layer_range=tuple(cfg.lora.mlp_layer_range),
        r=int(cfg.lora.r), head_dim=getattr(c, "head_dim", None))
    report = verify_freeze(model, expected_trainable=exp)
    print("freeze verified:", format_param_counts(model), "| trainable", f"{report.trainable:,}")

    model.config.use_cache = False
    if bool(cfg.train.gradient_checkpointing):
        model.enable_input_require_grads()

    compile_device = "cuda" if bool(cfg.base.get("full_gpu", False)) else "cpu"
    print(f"loading SNAC codec on {compile_device}…")
    snac = load_snac(snac_dir, device=compile_device)

    held_out = set(json.loads((REPO / "configs/eval_manifests/held_out_clip_keys.json").read_text()))
    include_stop = bool(cfg.train.loss.include_stop_in_loss)
    pools: dict[str, list] | None = None
    if args.smoke:
        print("compiling SMOKE training examples…")
        examples = build_examples(cfg, tok, contract, snac, held_out, plan=SMOKE_PLAN,
                                  include_stop=include_stop, device=compile_device)
        preserve_order = False
    else:
        print("compiling FULL training examples (mixture over all sources)…")
        pools, examples = build_full_examples(cfg, tok, contract, snac, held_out,
                                              include_stop=include_stop, device=compile_device)
        preserve_order = True
    if not examples:
        raise RuntimeError("no training examples were compiled")
    print(f"total examples (ordered draws): {len(examples)}")

    from transformers import TrainingArguments

    from indic_speak_ft.train.callbacks import ConfigSnapshotCallback
    from indic_speak_ft.train.trainer import WeightedLoRATrainer

    out = str(REPO / tcfg["output_dir"])
    pad_id = tok.convert_tokens_to_ids("<|pad|>")
    targs = TrainingArguments(
        output_dir=out, max_steps=int(tcfg["max_steps"]),
        per_device_train_batch_size=int(tcfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(tcfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg.train.learning_rate), warmup_steps=int(cfg.train.warmup_steps),
        logging_steps=int(cfg.train.logging_steps), save_steps=int(tcfg["save_steps"]),
        save_total_limit=1, bf16=bool(cfg.train.bf16), report_to=[], remove_unused_columns=False,
        gradient_checkpointing=bool(cfg.train.gradient_checkpointing), dataloader_num_workers=0)
    trainer = WeightedLoRATrainer(
        model=model, args=targs, train_dataset=PrecompiledTTSDataset(examples),
        data_collator=lambda b: pad_collate(b, pad_id),
        snac_base=contract.snac_base, stop_token_id=contract.end_of_speech,
        position_weights=tuple(cfg.train.loss.position_weights),
        stop_weight=float(cfg.train.loss.stop_weight),
        enable_weighting=bool(cfg.train.loss.enable_position_weighting),
        preserve_mixture_order=preserve_order,
        callbacks=[ConfigSnapshotCallback(OmegaConf.to_container(cfg, resolve=True), out_dir=out)])

    print(f"training {tcfg['max_steps']} steps…")
    t0 = time.time()
    result = trainer.train()
    dt = time.time() - t0
    trainer.save_model(out)
    pool_line = ("pools=" + " ".join(f"{k}:{len(v)}" for k, v in pools.items()) + "\n") if pools else ""
    unique = sum(len(v) for v in pools.values()) if pools else len(examples)
    # HF Trainer logs a custom compute_loss summed over the accumulation window, so the raw
    # training_loss is ~grad_accum× the per-token weighted-CE. Record both so runs at different
    # grad_accum (smoke=1, full=4) are comparable; the per-token figure is the honest one.
    grad_accum = int(tcfg["gradient_accumulation_steps"])
    Path(out, "training_log.txt").write_text(
        f"steps={int(tcfg['max_steps'])} draws={len(examples)} unique_clips={unique} "
        f"final_loss={result.training_loss:.4f} "
        f"final_loss_per_token={result.training_loss / grad_accum:.4f} seconds={dt:.0f}\n"
        f"trainable_params={report.trainable}\n" + pool_line)
    print(f"done in {dt:.0f}s | final_loss={result.training_loss:.4f} | adapter -> {out}")
    print("SMOKE_TRAIN_OK" if args.smoke else "TRAIN_OK")


if __name__ == "__main__":
    main()
