#!/usr/bin/env python3
"""Evaluate frozen baseline and Optimized Combination settings on internal holdout."""
from __future__ import annotations

import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from datasets import load_from_disk

from regbench.config import Config, RunSpec
from regbench.data import load_splits, prepare
from regbench.trainer import run

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "heldout_evaluation"
REPORT = ROOT / "outputs" / "overfitting_study" / "report"
SEEDS = (123, 456, 789, 1011, 201765793, 1801564980, 1323889826)
EPOCHS = 12
BASE_LR = 3e-5


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_report_table(result: dict) -> None:
    baseline = result["baseline"]
    optimized = result["optimized_combination"]
    rows = []
    for seed in SEEDS:
        rows.append(
            f"{seed} & {100 * baseline[str(seed)]:.2f} & "
            f"{100 * optimized[str(seed)]:.2f} & "
            f"{100 * (optimized[str(seed)] - baseline[str(seed)]):+.2f} \\\\"
        )

    baseline_values = [100 * value for value in baseline.values()]
    optimized_values = [100 * value for value in optimized.values()]
    gains = [100 * (optimized[str(seed)] - baseline[str(seed)]) for seed in SEEDS]
    mean_gain = statistics.mean(gains)
    table = [
        r"\begin{table*}[tp]",
        r"\centering\small\setlength{\tabcolsep}{4pt}",
        r"\caption{Internal holdout accuracy for seven matched seeds after 12 epochs. The 374-example split was examined in prior studies; results are exploratory, not an independent test.}",
        r"\label{tab:heldout-accuracy}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Seed & Baseline accuracy (\%) & Optimized Combination (\%) & Gain (pp) \\",
        r"\midrule",
        *rows,
        r"\midrule",
        f"Mean $\\pm$ SD & {statistics.mean(baseline_values):.2f} $\\pm$ "
        f"{statistics.stdev(baseline_values):.2f} & "
        f"{statistics.mean(optimized_values):.2f} $\\pm$ "
        f"{statistics.stdev(optimized_values):.2f} & {mean_gain:+.2f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    (REPORT / "heldout_results.tex").write_text("\n".join(table) + "\n", encoding="utf-8")


def append_progress(row: dict) -> None:
    with (OUT / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    os.chdir(ROOT)
    if not torch.cuda.is_available():
        raise SystemExit("Run this evaluation inside a Slurm GPU allocation.")

    selection_path = ROOT / "outputs" / "regularization_selection_curves" / "selection.json"
    selection = json.loads(selection_path.read_text())
    optimized_reg = dict(selection["optimized_parameters"])
    cfg = Config.load(ROOT / "configs/default.yaml", {"epochs": EPOCHS, "lr": BASE_LR,
                                                        "save_final_model": False})
    prepared = load_from_disk(str(prepare(cfg)))
    holdout = prepared["internal_holdout"]
    if len(holdout) != 374:
        raise ValueError(f"Expected 374 internal holdout rows, found {len(holdout)}")

    OUT.mkdir(parents=True, exist_ok=True)
    environment = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": torch.cuda.get_device_name(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "epochs": EPOCHS,
        "seeds": SEEDS,
        "learning_rate": BASE_LR,
        "holdout": "internal_holdout",
        "holdout_size": len(holdout),
        "prior_exposure": True,
        "optimized_parameters": optimized_reg,
    }
    write_json(OUT / "environment.json", environment)
    write_json(OUT / "status.json", {"state": "running", "completed_runs": 0})
    started = time.monotonic()
    splits_by_method = {}
    baseline_splits = load_splits(cfg)
    baseline_splits["test"] = holdout
    splits_by_method["baseline"] = baseline_splits
    optimized_splits = load_splits(cfg, optimized_reg.get("augmentation"))
    optimized_splits["test"] = holdout
    splits_by_method["optimized_combination"] = optimized_splits

    results = {"baseline": {}, "optimized_combination": {}}
    completed = 0
    try:
        for label, reg in (("baseline", {}), ("optimized_combination", optimized_reg)):
            for seed in SEEDS:
                run_dir = OUT / "runs" / label / f"seed{seed}"
                metrics = run(
                    cfg, RunSpec(reg=reg, lr=BASE_LR, seed=seed), run_dir,
                    final=True, splits=splits_by_method[label], save_resume=False,
                )
                if len(metrics.get("history", [])) != EPOCHS or "test" not in metrics:
                    raise ValueError(f"Incomplete held-out evaluation for {label}, seed {seed}")
                results[label][str(seed)] = metrics["test"]["accuracy"]
                completed += 1
                append_progress({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "completed_runs": completed,
                    "method": label,
                    "seed": seed,
                    "test_accuracy": metrics["test"]["accuracy"],
                    "test_examples": metrics["test"]["n"],
                })
                write_json(OUT / "status.json", {
                    "state": "running",
                    "completed_runs": completed,
                    "current": {"method": label, "seed": seed},
                })

        gains = [results["optimized_combination"][str(seed)]
                 - results["baseline"][str(seed)] for seed in SEEDS]
        results["mean_paired_gain_pp"] = 100 * statistics.mean(gains)
        results["seeds_improved"] = sum(gain > 0 for gain in gains)
        results["protocol"] = environment
        results["controller_wall_seconds"] = time.monotonic() - started
        write_json(OUT / "summary.json", results)
        write_report_table(results)
        write_json(OUT / "status.json", {
            "state": "completed",
            "completed_runs": completed,
            "controller_wall_seconds": time.monotonic() - started,
        })
    except BaseException as exc:
        write_json(OUT / "status.json", {
            "state": "failed",
            "completed_runs": completed,
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise


if __name__ == "__main__":
    main()