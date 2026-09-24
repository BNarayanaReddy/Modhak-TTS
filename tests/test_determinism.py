"""Phase-4 determinism tests: seeding reproduces streams; RNG capture/restore round-trips."""
from __future__ import annotations

import random

import torch

from indic_speak_ft.train.determinism import capture_rng_state, restore_rng_state, set_seed


def test_set_seed_reproducible():
    set_seed(123, deterministic=False)
    a = (random.random(), torch.rand(3).tolist())
    set_seed(123, deterministic=False)
    b = (random.random(), torch.rand(3).tolist())
    assert a == b


def test_capture_restore_rng_round_trips():
    set_seed(7, deterministic=False)
    _ = torch.rand(5)  # advance the stream
    state = capture_rng_state()
    after_capture = torch.rand(4).tolist()
    restore_rng_state(state)
    assert torch.rand(4).tolist() == after_capture  # same continuation after restore
