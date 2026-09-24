"""Phase-2 mixture-sampler tests: per-batch composition is exact and deterministic."""
from __future__ import annotations

from collections import Counter

import pytest

from indic_speak_ft.data.mixture import MixtureSampler, batch_composition

WEIGHTS = {"springlab_marathi": 0.56, "rasa_marathi": 0.24, "hindi_replay": 0.15, "english_replay": 0.05}


def test_composition_sums_and_matches():
    comp = batch_composition(100, WEIGHTS)
    assert comp == {"springlab_marathi": 56, "rasa_marathi": 24, "hindi_replay": 15, "english_replay": 5}
    assert sum(comp.values()) == 100


def test_composition_largest_remainder_sums_exactly():
    for bs in (7, 16, 32, 33, 64, 128):
        comp = batch_composition(bs, WEIGHTS)
        assert sum(comp.values()) == bs
        assert all(v >= 0 for v in comp.values())


def test_replay_present_every_batch():
    # Decision 3 guarantee: non-Marathi replay never rounds to zero at a real batch size
    comp = batch_composition(32, WEIGHTS)
    assert comp["hindi_replay"] + comp["english_replay"] >= 1


def test_sampler_window_matches_composition_and_is_deterministic():
    sizes = {"springlab_marathi": 500, "rasa_marathi": 300, "hindi_replay": 200, "english_replay": 100}
    bs = 25
    s1 = MixtureSampler(sizes, WEIGHTS, batch_size=bs, seed=7)
    comp = s1.composition
    it = iter(s1)
    for _ in range(10):  # 10 consecutive batches
        window = [next(it) for _ in range(bs)]
        by_source = Counter(src for src, _ in window)
        assert dict(by_source) == comp
        assert all(0 <= idx < sizes[src] for src, idx in window)  # indices in range

    # same seed → identical stream; different seed → different stream
    def take(sampler, n):
        it = iter(sampler)
        return [next(it) for _ in range(n)]

    s_a = MixtureSampler(sizes, WEIGHTS, batch_size=bs, seed=7)
    s_b = MixtureSampler(sizes, WEIGHTS, batch_size=bs, seed=7)
    s_c = MixtureSampler(sizes, WEIGHTS, batch_size=bs, seed=99)
    assert take(s_a, 3 * bs) == take(s_b, 3 * bs)
    assert take(s_a, 3 * bs) != take(s_c, 3 * bs)


def test_sampler_covers_all_indices_within_an_epoch():
    sizes = {"a": 40, "b": 10}
    weights = {"a": 0.8, "b": 0.2}
    s = MixtureSampler(sizes, weights, batch_size=10, seed=1)
    it = iter(s)
    seen_a = set()
    for _ in range(sizes["a"]):  # one epoch's worth of 'a' picks (8 per batch of 10)
        for src, idx in [next(it) for _ in range(10)]:
            if src == "a":
                seen_a.add(idx)
    assert seen_a == set(range(sizes["a"]))  # every 'a' index appears (no starvation)


def test_invalid_weights_rejected():
    with pytest.raises(ValueError):
        MixtureSampler({"a": 5}, {"a": 1.0, "b": 1.0}, batch_size=4)   # unknown source
    with pytest.raises(ValueError):
        MixtureSampler({"a": 0}, {"a": 1.0}, batch_size=4)             # weighted but empty
