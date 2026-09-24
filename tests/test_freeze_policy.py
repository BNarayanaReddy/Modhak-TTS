"""Phase-3 freeze-policy tests on a TINY Llama (hermetic — no 6.6 GB model).

Builds a 28-layer tiny Llama (same structural shape as the real model: GQA, tied
embeddings) so the layer-range routing (up/gate on 8..24 only) and the trainable-param
count are exercised for real, in seconds, on CPU.
"""
from __future__ import annotations

import pytest

from indic_speak_ft.model.freeze import (
    apply_freeze_policy,
    count_trainable_params,
    expected_trainable_params,
    verify_freeze,
)
from indic_speak_ft.model.lora import build_lora_config, lora_target_modules

pytest.importorskip("peft")
pytest.importorskip("transformers")

TINY = {
    "vocab_size": 256, "hidden_size": 64, "intermediate_size": 128,
    "num_hidden_layers": 28, "num_attention_heads": 4, "num_key_value_heads": 2,
    "tie_word_embeddings": True, "max_position_embeddings": 64,
}
LORA = {"r": 8, "alpha": 16, "dropout": 0.0,
        "attention_targets": ["q_proj", "v_proj"], "mlp_targets": ["up_proj", "gate_proj"],
        "mlp_layer_range": [8, 24]}


def _peft_model():
    from peft import get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    base = LlamaForCausalLM(LlamaConfig(**TINY))
    return get_peft_model(base, build_lora_config(LORA))


def _expected() -> int:
    return expected_trainable_params(
        num_layers=28, hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=128, mlp_layer_range=(8, 24), r=8)


def test_only_lora_trainable_and_count_matches_expected():
    model = _peft_model()
    apply_freeze_policy(model)
    report = verify_freeze(model, expected_trainable=_expected())
    assert report.trainable == _expected()
    assert report.trainable == count_trainable_params(model)[0]
    assert 0.0 < report.trainable_pct < 100.0  # LoRA is a small fraction of the model


def test_frozen_tensors_stay_frozen():
    model = _peft_model()
    apply_freeze_policy(model)
    for n, p in model.named_parameters():
        if "lora_" in n:
            continue
        if any(s in n for s in ("embed_tokens", "lm_head", "down_proj", "norm")):
            assert not p.requires_grad, f"{n} should be frozen"


def test_mlp_adapters_only_in_range_qv_on_all_layers():
    model = _peft_model()
    lora_names = [n for n, _ in model.named_parameters() if "lora_" in n]
    # up/gate present at 8 and 24, absent at 7 and 25
    assert any("layers.8.mlp.up_proj" in n for n in lora_names)
    assert any("layers.24.mlp.gate_proj" in n for n in lora_names)
    assert not any("layers.7.mlp.up_proj" in n for n in lora_names)
    assert not any("layers.25.mlp.up_proj" in n for n in lora_names)
    # q/v on the extreme layers (all layers)
    assert any("layers.0.self_attn.q_proj" in n for n in lora_names)
    assert any("layers.27.self_attn.v_proj" in n for n in lora_names)


def test_verify_freeze_rejects_an_unfrozen_base_tensor():
    model = _peft_model()
    apply_freeze_policy(model)
    # unfreeze a base tensor -> verifier must catch it
    for n, p in model.named_parameters():
        if "embed_tokens" in n and "lora_" not in n:
            p.requires_grad_(True)
            break
    with pytest.raises(AssertionError):
        verify_freeze(model, expected_trainable=_expected())


def test_target_modules_list_shape():
    t = lora_target_modules((8, 24))
    assert "q_proj" in t and "v_proj" in t
    assert "model.layers.8.mlp.up_proj" in t and "model.layers.24.mlp.gate_proj" in t
    assert "model.layers.7.mlp.up_proj" not in t
    assert sum(1 for x in t if x.startswith("model.layers")) == 17 * 2  # 17 layers × {up,gate}
