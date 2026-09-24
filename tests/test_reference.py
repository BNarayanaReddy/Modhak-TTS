"""Hermetic tests for the reference-audio provider: path mapping + shard-cached routing."""
from __future__ import annotations

import numpy as np

from indic_speak_ft.eval.reference import ReferenceAudioProvider, hf_shard_path


def test_hf_shard_path() -> None:
    assert (hf_shard_path("ai4bharat/Rasa", "test-00006-of-00007.parquet")
            == "datasets/ai4bharat/Rasa/Marathi/test-00006-of-00007.parquet")
    assert hf_shard_path("SPRINGLab/IndicTTS_Marathi", "train-00000-of-00004.parquet").endswith(
        "/data/train-00000-of-00004.parquet")


def _manifests() -> dict[str, dict]:
    return {
        "retention_rasa_marathi": {"items": [
            {"clip_key": "k1", "shard": "test-00000-of-00007.parquet",
             "source_id": "ai4bharat/Rasa", "speaker": "Chinmay"},
            {"clip_key": "k2", "shard": "test-00006-of-00007.parquet",
             "source_id": "ai4bharat/Rasa", "speaker": "Anagha"},
        ]},
        "marathi_longform": {"items": [
            {"clip_key": None, "source_id": "synthesized_from_heldout_mr"}]},  # synthesis-only
    }


def test_need_indexing_routing_and_cache(monkeypatch) -> None:
    prov = ReferenceAudioProvider(_manifests())
    # only the two real-audio shards are indexed; the longform (no clip_key) is ignored
    assert set(prov._need) == {
        ("ai4bharat/Rasa", "test-00000-of-00007.parquet"),
        ("ai4bharat/Rasa", "test-00006-of-00007.parquet"),
    }

    loaded: list[tuple[str, str]] = []

    def fake_load(src: str, shard: str) -> None:
        loaded.append((src, shard))
        prov._cache["k2" if shard.endswith("06-of-00007.parquet") else "k1"] = np.ones(8, np.float32)
        prov._loaded.add((src, shard))

    monkeypatch.setattr(prov, "_load_shard", fake_load)

    assert prov({"clip_key": None, "source_id": "x"}) is None                 # no key → None
    assert prov({"clip_key": "z", "shard": "s", "source_id": "unknown/ds"}) is None  # unknown source
    w = prov({"clip_key": "k2", "shard": "test-00006-of-00007.parquet", "source_id": "ai4bharat/Rasa"})
    assert w is not None and w.shape == (8,)
    prov({"clip_key": "k2", "shard": "test-00006-of-00007.parquet", "source_id": "ai4bharat/Rasa"})
    assert loaded.count(("ai4bharat/Rasa", "test-00006-of-00007.parquet")) == 1  # shard loaded once
