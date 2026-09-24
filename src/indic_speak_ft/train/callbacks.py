"""Training callbacks: the regression-stop retention monitor (Decision 3) and a config/RNG
snapshot on save.

RegressionStopCallback runs a caller-supplied retention-eval function every N steps (the WER
eval on the retention buckets lives in the eval harness, Phase 6, and is injected here so this
module has no heavy deps). If retention WER regresses by more than the relative threshold vs.
the pre-training baseline, it logs a warning and drops a ``regression_detected`` marker in the
run dir — it never silently continues. With no eval fn (e.g. the smoke run) it is inert.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

# retention_eval_fn() -> {bucket_name: wer}
RetentionEvalFn = Callable[[], dict[str, float]]


class RegressionStopCallback(TrainerCallback):
    def __init__(
        self,
        retention_eval_fn: RetentionEvalFn | None,
        *,
        baseline: dict[str, float] | None = None,
        every_steps: int = 500,
        max_relative_regression: float = 0.15,
        out_dir: str = ".",
    ) -> None:
        self.eval_fn = retention_eval_fn
        self.baseline = baseline
        self.every = every_steps
        self.threshold = max_relative_regression
        self.out_dir = Path(out_dir)
        self.history: list[dict[str, Any]] = []

    def on_step_end(self, args, state, control, **kwargs):
        if self.eval_fn is None or state.global_step == 0 or state.global_step % self.every != 0:
            return control
        current = self.eval_fn()
        if self.baseline is None:  # first eval sets the baseline
            self.baseline = dict(current)
        regressions = {
            b: (current[b] - self.baseline[b]) / max(self.baseline[b], 1e-6)
            for b in current if b in self.baseline
        }
        step_dir = self.out_dir / "retention_evals"
        step_dir.mkdir(parents=True, exist_ok=True)
        record = {"step": state.global_step, "wer": current, "relative_regression": regressions}
        (step_dir / f"step_{state.global_step:06d}.json").write_text(json.dumps(record, indent=2))
        self.history.append(record)

        worst = max(regressions.values(), default=0.0)
        if worst > self.threshold:
            print(f"[RegressionStop] retention WER regressed {worst:.1%} > {self.threshold:.0%} "
                  f"at step {state.global_step}: {regressions}")
            (self.out_dir / "regression_detected.json").write_text(json.dumps(record, indent=2))
        return control


class ConfigSnapshotCallback(TrainerCallback):
    """On each save, write the exact run config + RNG state next to the checkpoint."""

    def __init__(self, config_snapshot: dict[str, Any], out_dir: str = ".") -> None:
        self.config = config_snapshot
        self.out_dir = Path(out_dir)

    def on_save(self, args, state, control, **kwargs):
        import yaml

        from indic_speak_ft.train.determinism import capture_rng_state

        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(self.config, sort_keys=False))
        rng = capture_rng_state()
        # store only the cheaply-serializable presence flags; full tensors live in the HF ckpt
        (self.out_dir / "rng_present.json").write_text(json.dumps(
            {"python": rng.python is not None, "numpy": rng.numpy is not None,
             "torch": rng.torch is not None, "cuda": rng.cuda is not None}))
        return control
