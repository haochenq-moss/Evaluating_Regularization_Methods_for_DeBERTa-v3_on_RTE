#!/usr/bin/env python3
"""Compare R-Drop with EMA and SWA using matched existing RTE controls."""
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
OUT = ROOT / "outputs" / "regularization_extensions"
CONTROL = ROOT / "outputs" / "overfitting_study" / "runs" / "full"
SEEDS = (42, 123, 456, 789, 1011)
METHODS = ("ema", "swa", "neftune_ema")
EPOCHS = 12
LR = 3e-5
RDROP = 0.1
NEFTUNE = 15.0
METHOD_REG = {
    "ema": {"rdrop": RDROP},
    "swa": {"rdrop": RDROP},
    "neftune_ema": {"rdrop": RDROP, "neftune": NEFTUNE},
}


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


def load_control(method: str, seed: int) -> dict:
    path = CONTROL / method / f"seed{seed}" / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Matched control run is missing: {path}")
    metrics = json.loads(path.read_text())
    if metrics.get("seed") != seed or metrics.get("epochs") != EPOCHS:
        raise ValueError(f"Control run has the wrong seed or epoch count: {path}")
    return metrics


def paired_summary(results: dict) -> dict:
    comparison = {}
    for method, runs in results.items():
        baseline = {seed: load_control("baseline", seed)["dev"]["accuracy"] for seed in SEEDS}
        rdrop = {seed: load_control("rdrop", seed)["dev"]["accuracy"] for seed in SEEDS}
        averaged = {seed: run_result["dev"]["accuracy"]
                    for seed, run_result in zip(SEEDS, runs, strict=True)}
        comparison[method] = {
            "mean_dev_accuracy": sum(averaged.values()) / len(SEEDS),
            "mean_paired_gain_vs_baseline": sum(
                averaged[seed] - baseline[seed] for seed in SEEDS
            ) / len(SEEDS),
            "mean_paired_gain_vs_rdrop": sum(
                averaged[seed] - rdrop[seed] for seed in SEEDS
            ) / len(SEEDS),
            "per_seed": {
                str(seed): {
                    "baseline": baseline[seed], "rdrop": rdrop[seed],
                    method: averaged[seed],
                }
                for seed in SEEDS
            },
        }
    return comparison


def main() -> None:
    os.chdir(ROOT)
    if not torch.cuda.is_available():
        raise SystemExit("Run this study inside a Slurm GPU allocation.")
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = Config.load(ROOT / "configs/default.yaml")
    cfg = replace(cfg, epochs=EPOCHS, save_final_model=False)
    splits = load_splits(cfg)
    if len(splits["train"]) != 1742 or len(splits["dev"]) != 374:
        raise ValueError("Unexpected RTE split sizes; refusing to run the comparison.")

    started = time.monotonic()
    environment = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": torch.cuda.get_device_name(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "epochs": EPOCHS,
        "seeds": SEEDS,
        "learning_rate": LR,
        "rdrop": RDROP,
        "neftune": NEFTUNE,
        "model_saving": False,
        "resume_checkpoints": False,
    }
    write_json(OUT / "environment.json", environment)
    results = {}

    for method in METHODS:
        results[method] = []
        for seed in SEEDS:
            run_dir = OUT / "runs" / f"rdrop_{method}" / f"seed{seed}"
            identity = {
                "method": method, "seed": seed, "lr": LR,
                "regularizers": METHOD_REG[method], "epochs": EPOCHS,
                "weight_average": "ema" if method == "neftune_ema" else method,
                "ema_decay": 0.99 if method in ("ema", "neftune_ema") else None,
            }
            write_json(run_dir / "study_config.json", identity)
            write_json(OUT / "status.json", {
                "state": "running", "method": method, "seed": seed,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })

            def log_epoch(row: dict, method=method, seed=seed) -> None:
                append_progress({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "method": method,
                    "seed": seed,
                    **row,
                })

            metrics = run(
                cfg, RunSpec(METHOD_REG[method], LR, seed), run_dir,
                splits=splits, record_train_accuracy=True, record_train_clean_loss=True,
                save_resume=False,
                weight_average="ema" if method == "neftune_ema" else method,
                ema_decay=0.99,
                epoch_callback=log_epoch,
            )
            if len(metrics["history"]) != EPOCHS:
                raise ValueError(f"Incomplete epoch history for {method}, seed {seed}")
            if any((run_dir / name).exists() for name in ("model.pt", "resume.pt")):
                raise RuntimeError(f"Model/checkpoint unexpectedly saved: {run_dir}")
            results[method].append(metrics)
            write_json(OUT / "status.json", {
                "state": "running", "method": method, "seed": seed,
                "completed_runs": sum(len(items) for items in results.values()),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })

    summary = {
        "protocol": environment,
        "comparison": paired_summary(results),
        "controller_wall_seconds": time.monotonic() - started,
    }
    write_json(OUT / "summary.json", summary)
    write_json(OUT / "status.json", {
        "state": "completed", "completed_runs": len(METHODS) * len(SEEDS),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })


if __name__ == "__main__":
    main()