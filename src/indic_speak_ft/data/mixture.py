"""Deterministic per-batch mixture sampling: every training batch holds a FIXED source
composition matching the configured weights (Decision 3 — "20% of every batch is
non-Marathi" — is a per-batch guarantee, not a long-run average), so replay never
disappears from a batch by chance.

Pure index logic (source name + local index), decoupled from how clips are stored: a
dataset maps ``(source, index)`` → clip. Seeded and reproducible (determinism.py wires the
seed). No global state.
"""
from __future__ import annotations

import math
import random
from collections.abc import Iterator


def batch_composition(batch_size: int, weights: dict[str, float]) -> dict[str, int]:
    """Split ``batch_size`` across sources by ``weights`` using largest-remainder rounding,
    so the counts sum to exactly ``batch_size`` (no drift from naive rounding)."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    exact = {k: batch_size * w / total for k, w in weights.items()}
    floor = {k: math.floor(v) for k, v in exact.items()}
    remainder = batch_size - sum(floor.values())
    # hand the leftover seats to the largest fractional parts
    order = sorted(exact, key=lambda k: exact[k] - floor[k], reverse=True)
    for k in order[:remainder]:
        floor[k] += 1
    return floor


class MixtureSampler:
    """Yields an infinite stream of ``(source, local_index)`` picks whose every
    ``batch_size``-length window matches :func:`batch_composition`. Each source is drawn
    without replacement within an epoch (reshuffled per epoch from the seed), so coverage
    is uniform; the per-batch order is shuffled so sources aren't blocked together.
    """

    def __init__(
        self,
        source_sizes: dict[str, int],
        weights: dict[str, float],
        *,
        batch_size: int,
        seed: int = 0,
    ) -> None:
        if set(weights) - set(source_sizes):
            raise ValueError(f"weights reference unknown sources: {set(weights) - set(source_sizes)}")
        for name, w in weights.items():
            if w > 0 and source_sizes.get(name, 0) <= 0:
                raise ValueError(f"source {name!r} has weight {w} but no items")
        self.source_sizes = source_sizes
        self.weights = weights
        self.batch_size = batch_size
        self.seed = seed
        self.composition = batch_composition(batch_size, weights)

    def _epoch_order(self, name: str, epoch: int) -> list[int]:
        rng = random.Random(f"{self.seed}:{name}:{epoch}")  # str seed: stable + hashable
        idx = list(range(self.source_sizes[name]))
        rng.shuffle(idx)
        return idx

    def __iter__(self) -> Iterator[tuple[str, int]]:
        cursors = {name: (0, 0, self._epoch_order(name, 0)) for name in self.composition}
        batch_rng = random.Random(f"{self.seed}:batch")
        while True:
            batch: list[tuple[str, int]] = []
            for name, count in self.composition.items():
                pos, epoch, order = cursors[name]
                for _ in range(count):
                    if pos >= len(order):  # exhausted the epoch — reshuffle
                        epoch += 1
                        order = self._epoch_order(name, epoch)
                        pos = 0
                    batch.append((name, order[pos]))
                    pos += 1
                cursors[name] = (pos, epoch, order)
            batch_rng.shuffle(batch)
            yield from batch
