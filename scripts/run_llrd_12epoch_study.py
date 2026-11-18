#!/usr/bin/env python3
"""Run matched 12-epoch LLRD and full-stack-plus-LLRD comparisons."""
from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

from regbench.config import Config, RunSpec
from regbench.data import load_splits
from regbench.trainer import run

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "llrd_12epoch"
STACK_STAGES = ROOT / "outputs" / "stack_ranked" / "stages.json"
SEEDS = (42, 123, 456, 789, 1011)
DECAYS = (0.8, 0.9, 0.95)
EPOCHS = 12


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append_progress(row: dict) -> None:
    with (OUT / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_selected_stack() -> tuple[dict[str, float], float]:
    if not STACK_STAGES.is_file():
        raise FileNotFoundError(f"Run the ranked stack study first: {STACK_STAGES}")
    stages = json.loads(STACK_STAGES.read_text())["stages"]
    final_stage = next((stage for stage in stages if stage["stage"] == 5), None)
    if final_stage is None:
        raise ValueError("The selected stack does not contain its five-method stage.")
    reg = dict(final_stage["reg"])
    expected = {"mixup", "rdrop", "neftune", "augmentation", "label_smoothing"}
    if set(reg) != expected:
        raise ValueError(f"Unexpected selected stack regularizers: {sorted(reg)}")
    return reg, float(final_stage["lr"])


def main() -> None:
    os.chdir(ROOT)
    if not torch.cuda.is_available():
        raise SystemExit("Run this study inside a Slurm GPU allocation.")

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = replace(Config.load(ROOT / "configs/default.yaml"), epochs=EPOCHS,
                  save_final_model=False)
    stack_reg, lr = load_selected_stack()
    base_splits = load_splits(cfg)
    stack_splits = load_splits(cfg, stack_reg["augmentation"])
    if len(base_splits["train"]) != 1742 or len(base_splits["dev"]) != 374:
        raise ValueError("Unexpected RTE split sizes; refusing to run the comparison.")

    environment = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": torch.cuda.get_device_name(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "epochs": EPOCHS,
        "seeds": SEEDS,
        "llrd_decays": DECAYS,
        "learning_rate": lr,
        "selected_stack": stack_reg,
        "model_saving": False,
        "resume_checkpoints": False,
    }
    write_json(OUT / "environment.json", environment)
    write_json(OUT / "status.json", {"state": "running", "completed_runs": 0})
    started = time.monotonic()
    results = []

    try:
        for family, base_reg, splits in (
            ("llrd", {}, base_splits),
            ("stack_llrd", stack_reg, stack_splits),
        ):
            for decay in DECAYS:
                reg = {**base_reg, "llrd": decay}
                for seed in SEEDS:
                    run_dir = OUT / "runs" / f"{family}_decay{decay:g}" / f"seed{seed}"
                    metrics = run(
                        cfg, RunSpec(reg=reg, lr=lr, seed=seed), run_dir,
                        splits=splits, save_resume=False,
                    )
                    if len(metrics["history"]) != EPOCHS:
                        raise ValueError(f"Incomplete epoch history: {family}, {decay}, {seed}")
                    row = {
                        "family": family,
                        "decay": decay,
                        "seed": seed,
                        "lr": lr,
                        "reg": reg,
                        "epochs": EPOCHS,
                        "train_seconds": metrics["train_seconds"],
                        "dev": metrics["dev"],
                    }
                    results.append(row)
                    append_progress({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "completed_runs": len(results),
                        **row,
                    })
                    write_json(OUT / "status.json", {
                        "state": "running",
                        "completed_runs": len(results),
                        "current": {"family": family, "decay": decay, "seed": seed},
                    })

        write_json(OUT / "summary.json", {
            "protocol": environment,
            "runs": results,
            "controller_wall_seconds": time.monotonic() - started,
        })
        write_json(OUT / "status.json", {
            "state": "completed",
            "completed_runs": len(results),
            "controller_wall_seconds": time.monotonic() - started,
        })
    except BaseException as exc:
        write_json(OUT / "status.json", {
            "state": "failed",
            "completed_runs": len(results),
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise


if __name__ == "__main__":
    main()