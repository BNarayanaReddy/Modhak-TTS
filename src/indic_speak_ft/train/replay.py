"""Replay mixing: order the compiled training examples so every batch has the fixed source
composition (Decision 3: 20% non-Marathi = 15% Hindi + 5% English; Decision 7: Rasa 30% of
the Marathi batch). Thin wrapper over ``mixture.MixtureSampler`` — the sampler owns the
per-batch-composition guarantee; here we just materialize an example ordering from it.

Pair with ``WeightedLoRATrainer(preserve_mixture_order=True)`` so the trainer uses a
sequential sampler and the batch composition survives to the GPU.
"""
from __future__ import annotations

from indic_speak_ft.data.mixture import MixtureSampler


def build_mixture_ordered_examples(
    pools: dict[str, list[dict]],
    weights: dict[str, float],
    *,
    batch_size: int,
    num_batches: int,
    seed: int = 0,
) -> list[dict]:
    """Draw ``batch_size * num_batches`` examples via the mixture sampler and return them in
    order, so consecutive ``batch_size`` windows match the configured composition."""
    sizes = {name: len(pool) for name, pool in pools.items()}
    sampler = MixtureSampler(sizes, weights, batch_size=batch_size, seed=seed)
    it = iter(sampler)
    ordered: list[dict] = []
    for _ in range(batch_size * num_batches):
        source, idx = next(it)
        ordered.append(pools[source][idx])
    return ordered
