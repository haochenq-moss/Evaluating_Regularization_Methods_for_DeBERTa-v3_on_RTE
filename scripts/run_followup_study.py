#!/usr/bin/env python3
"""Run the matched, development-only ablation and longer-duration study."""
from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import replace
from pathlib import Path

import datasets
import torch
import transformers
from transformers import AutoTokenizer

from regbench.config import Config, RunSpec
from regbench.data import load_splits
from regbench.trainer import run

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "followup_study"
SEEDS = (42, 123, 456, 789, 1011)
LR = 3e-5
METHODS = {
    "baseline": {},
    "mixup": {"mixup": 0.4},
    "rdrop": {"rdrop": 0.5},
    "neftune": {"neftune": 15.0},
    "mixup_rdrop": {"mixup": 0.4, "rdrop": 0.5},
    "stack": {"mixup": 0.4, "rdrop": 0.5, "neftune": 15.0},
}
CONDITIONS = tuple((method, 8) for method in METHODS) + (("stack", 10), ("stack", 12))


def best_epoch(history):
    """Best accuracy, then lower loss, then the earliest epoch."""
    if not history:
        raise ValueError("Cannot select a best epoch from an empty history")
    return max(history, key=lambda row: (row["dev_accuracy"], -row["dev_loss"], -row["epoch"]))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    os.chdir(ROOT)
    if not torch.cuda.is_available():
        raise SystemExit("A Slurm GPU allocation with CUDA is required.")
    base = Config.load(ROOT / "configs" / "default.yaml")
    splits = load_splits(base)
    if len(splits["train"]) != 1742 or len(splits["dev"]) != 374:
        raise ValueError("Unexpected internal train/development sizes.")
    pad_id = AutoTokenizer.from_pretrained(base.model_name).pad_token_id
    metadata = {
        "device": torch.cuda.get_device_name(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "selection_rule": "highest development accuracy, then lowest development loss, then earliest epoch",
        "evaluation": "internal development only; no new holdout evaluation",
        "checkpoints": "none; an interrupted run must restart",
    }
    write_json(OUT / "environment.json", metadata)
    started = time.monotonic()
    for method, epochs in CONDITIONS:
        cfg = replace(base, epochs=epochs, save_final_model=False)
        for seed in SEEDS:
            spec = RunSpec(reg=dict(METHODS[method]), lr=LR, seed=seed)
            directory = OUT / "runs" / f"{method}_{epochs}" / f"seed{seed}"
            print(f"[followup] {method} epochs={epochs} seed={seed}", flush=True)
            metrics = run(cfg, spec, directory, splits=splits, pad_id=pad_id,
                          record_train_accuracy=True, save_resume=False)
            if (metrics["reg"] != spec.reg or metrics["lr"] != LR
                    or metrics["seed"] != seed or metrics["epochs"] != epochs
                    or len(metrics["history"]) != epochs or metrics["dev"]["n"] != 374):
                raise ValueError(f"Unexpected metrics in {directory}")
            best = best_epoch(metrics["history"])
            print(f"[followup] best epoch={best['epoch']} dev={best['dev_accuracy']:.6f}", flush=True)
    metadata["controller_wall_seconds"] = time.monotonic() - started
    write_json(OUT / "environment.json", metadata)
    print("[followup] All 40 runs complete; no holdout was evaluated.", flush=True)


if __name__ == "__main__":
    main()
