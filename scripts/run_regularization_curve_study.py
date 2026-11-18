#!/usr/bin/env python3
"""Run and plot matched 12-epoch full-fine-tuning regularization curves."""
from __future__ import annotations

import json
import argparse
import os
import platform
import statistics
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer

from regbench.config import Config, RunSpec
from regbench.data import load_splits
from regbench.trainer import run

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "regularization_selection_curves"
REPORT = ROOT / "outputs" / "overfitting_study" / "report"
OVERFIT_RUNS = ROOT / "outputs" / "overfitting_study" / "runs" / "full"
PARTIAL_CURVE_RUNS = ROOT / "outputs" / "regularization_curve_study" / "runs"
SEEDS = (123, 456, 789, 1011, 201765793, 1801564980, 1323889826)
MINIMUM_SEED_WINS = len(SEEDS) // 2 + 1
EPOCHS = 12
BASE_LR = 3e-5
METHODS = {
    "augmentation": [({"augmentation": value}, f"ratio {value:g}")
                     for value in (0.25, 0.50, 1.00)],
    "label_smoothing": [({"label_smoothing": value}, f"epsilon {value:g}")
                         for value in (0.05, 0.10, 0.15)],
    "mixup": [({"mixup": value}, f"alpha {value:g}") for value in (0.1, 0.2, 0.4)],
    "neftune": [({"neftune": value}, f"alpha {value:g}") for value in (5.0, 10.0, 15.0)],
    "rdrop": [({"rdrop": value}, f"coefficient {value:g}") for value in (0.1, 0.5, 1.0)],
    "llrd": [({"llrd": decay}, f"decay {decay:g}") for decay in (0.8, 0.9, 0.95)],
}
METHOD_LABELS = {
    "augmentation": "Filtered augmentation",
    "label_smoothing": "Label smoothing",
    "mixup": "Mixup",
    "neftune": "NEFTune",
    "rdrop": "R-Drop",
    "llrd": "LLRD",
    "optimized_combination": "Optimized Combination",
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


def candidate_tag(method: str, reg: dict[str, float]) -> str:
    value = next(iter(reg.values()))
    if method == "augmentation" and value == 0.5:
        return "augmentation"
    return f"{method}_{value:g}"


def candidate_run_dir(method: str, reg: dict[str, float], seed: int) -> Path:
    run_root = PARTIAL_CURVE_RUNS if candidate_tag(method, reg) == "augmentation" else OUT / "runs"
    return run_root / candidate_tag(method, reg) / f"seed{seed}"


def is_reusable_rdrop(reg: dict[str, float], seed: int) -> bool:
    path = OVERFIT_RUNS / "rdrop" / f"seed{seed}" / "metrics.json"
    return reg == {"rdrop": 0.1} and path.is_file()


def candidate_summary(method: str, reg: dict[str, float], label: str,
                      runs: dict[int, dict]) -> dict:
    values = next(iter(reg.values()))
    final_rows = [runs[seed]["history"][-1] for seed in SEEDS]
    return {
        "method": method,
        "setting": label,
        "reg": reg,
        "value": values,
        "mean_dev_accuracy": float(np.mean([row["dev_accuracy"] for row in final_rows])),
        "mean_dev_loss": float(np.mean([row["dev_loss"] for row in final_rows])),
    }


def selection_key(result: dict) -> tuple[float, float, float]:
    return result["mean_dev_accuracy"], -result["mean_dev_loss"], -result["value"]


def inclusion_decision(candidate_runs: dict[int, dict], baseline_runs: dict[int, dict]) -> dict:
    paired_gains = {
        seed: candidate_runs[seed]["history"][-1]["dev_accuracy"]
        - baseline_runs[seed]["history"][-1]["dev_accuracy"]
        for seed in SEEDS
    }
    mean_gain_pp = 100 * float(np.mean(list(paired_gains.values())))
    seeds_better = sum(gain > 0 for gain in paired_gains.values())
    return {
        "mean_paired_gain_pp": mean_gain_pp,
        "seeds_better_than_baseline": seeds_better,
        "included": mean_gain_pp > 0 and seeds_better >= MINIMUM_SEED_WINS,
    }


def load_existing(method: str, seed: int, expected_reg: dict[str, float]) -> dict:
    path = OVERFIT_RUNS / method / f"seed{seed}" / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Matched 12-epoch run is missing: {path}")
    metrics = json.loads(path.read_text())
    if (metrics.get("seed") != seed or metrics.get("epochs") != EPOCHS
            or metrics.get("lr") != BASE_LR or metrics.get("reg") != expected_reg
            or len(metrics.get("history", [])) != EPOCHS):
        raise ValueError(f"Unexpected matched run: {path}")
    validate_history(metrics, path)
    return metrics


def load_baseline(seed: int) -> dict:
    existing_path = OVERFIT_RUNS / "baseline" / f"seed{seed}" / "metrics.json"
    if existing_path.is_file():
        return load_existing("baseline", seed, {})
    path = OUT / "runs" / "baseline" / f"seed{seed}" / "metrics.json"
    metrics = json.loads(path.read_text())
    if metrics.get("seed") != seed or metrics.get("epochs") != EPOCHS or metrics.get("reg") != {}:
        raise ValueError(f"Unexpected baseline run: {path}")
    validate_history(metrics, path)
    return metrics


def validate_history(metrics: dict, path: Path) -> None:
    for epoch, row in enumerate(metrics["history"], start=1):
        if row.get("epoch") != epoch:
            raise ValueError(f"Unexpected epoch sequence: {path}")
        for field in ("train_accuracy", "dev_accuracy", "train_clean_loss", "dev_loss"):
            if field not in row or not np.isfinite(row[field]):
                raise ValueError(f"Missing or non-finite {field}: {path}")


def plot_curves(method: str, variants: list[tuple[str, dict]], baseline: dict[int, dict]) -> None:
    columns = len(variants)
    fig, axes = plt.subplots(2, columns, figsize=(6.1 * columns, 6.2),
                             sharex=True, squeeze=False)
    epochs = np.arange(1, EPOCHS + 1)
    method_label = METHOD_LABELS[method]
    colors = {"Baseline": "#356A9A", method_label: "#C54E52"}
    metric_fields = (("clean cross-entropy loss", "train_clean_loss", "dev_loss"),
                     ("Accuracy (%)", "train_accuracy", "dev_accuracy"))

    for column, (variant_label, method_runs) in enumerate(variants):
        series = (("Baseline", baseline), (method_label, method_runs))
        for row_index, (axis_label, train_field, dev_field) in enumerate(metric_fields):
            axis = axes[row_index, column]
            for name, run_group in series:
                for split, field, style, marker in (
                    ("training", train_field, "-", "o"),
                    ("development", dev_field, "--", "s"),
                ):
                    values = np.asarray([
                        [item["history"][epoch][field] for epoch in range(EPOCHS)]
                        for item in run_group.values()
                    ], dtype=float)
                    if "accuracy" in field:
                        values *= 100
                    mean = values.mean(axis=0)
                    deviation = values.std(axis=0, ddof=1)
                    color = colors[name]
                    axis.plot(epochs, mean, linestyle=style, marker=marker, markersize=3,
                              color=color, label=f"{name} {split}")
                    axis.fill_between(epochs, mean - deviation, mean + deviation,
                                      color=color, alpha=0.10)
            axis.set_ylabel(axis_label)
            axis.set_xlabel("Epoch")
            axis.set_xticks(epochs)
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7, ncol=2)
            if columns > 1:
                axis.set_title(variant_label if row_index == 0 else "")
        if columns == 1:
            axes[0, column].set_title(f"{method_label}: {variant_label}")

    fig.suptitle(f"{method_label}: candidate settings, training and development over 12 epochs")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    REPORT.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT / f"curves_{method}.pdf", bbox_inches="tight")
    fig.savefig(REPORT / f"curves_{method}.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def load_candidate(method: str, reg: dict[str, float], seed: int) -> dict:
    if is_reusable_rdrop(reg, seed):
        return load_existing("rdrop", seed, reg)
    path = candidate_run_dir(method, reg, seed) / "metrics.json"
    metrics = json.loads(path.read_text())
    if metrics.get("seed") != seed or metrics.get("epochs") != EPOCHS or metrics.get("reg") != reg:
        raise ValueError(f"Unexpected candidate run: {path}")
    validate_history(metrics, path)
    return metrics


def write_evaluation_matrix(baseline: dict, selected_runs: dict,
                            combination_runs: dict, selected_parameters: dict) -> None:
    def summary(run_group: dict[int, dict], metric: str, scale: float = 1.0) -> str:
        values = [run_group[seed]["history"][-1][metric] * scale for seed in SEEDS]
        return f"${statistics.mean(values):.2f} \\pm {statistics.stdev(values):.2f}$"

    rows = []
    rows.append(
        "Baseline & "
        + " & ".join((summary(baseline, "train_accuracy", 100),
                      summary(baseline, "train_clean_loss"),
                      summary(baseline, "dev_accuracy", 100),
                      summary(baseline, "dev_loss")))
        + r" \\"
    )
    display_names = {"label_smoothing": "epsilon"}
    for method, run_group in selected_runs.items():
        parameter = next(iter(METHODS[method][0][0]))
        label = f"{METHOD_LABELS[method]} ({display_names.get(parameter, parameter)}={selected_parameters[parameter]:g})"
        rows.append(
            label + " & "
            + " & ".join((summary(run_group, "train_accuracy", 100),
                          summary(run_group, "train_clean_loss"),
                          summary(run_group, "dev_accuracy", 100),
                          summary(run_group, "dev_loss")))
            + r" \\"
        )
    rows.append(
        "Optimized Combination & "
        + " & ".join((summary(combination_runs, "train_accuracy", 100),
                      summary(combination_runs, "train_clean_loss"),
                      summary(combination_runs, "dev_accuracy", 100),
                      summary(combination_runs, "dev_loss")))
        + r" \\"
    )

    table = [
        r"\begin{table*}[tp]",
        r"\centering\small\setlength{\tabcolsep}{4pt}",
        r"\caption{Final-epoch metrics after 12 epochs, mean $\pm$ sample SD over seven seeds. Training loss is clean evaluation-mode loss. Augmentation training metrics use the augmented training split.}",
        r"\label{tab:evaluation-matrix}",
        r"\begin{tabular}{p{0.27\textwidth}rrrr}",
        r"\toprule",
        r"Method & Train accuracy (\%) & Train loss & Validation accuracy (\%) & Validation loss \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "evaluation_matrix.tex").write_text("\n".join(table) + "\n", encoding="utf-8")


def write_report_tables() -> None:
    selection = json.loads((OUT / "selection.json").read_text())
    selected_parameters = selection["selected_parameters"]
    inclusion = selection["inclusion_decisions"]
    optimized_parameters = selection["optimized_parameters"]
    baseline = {seed: load_baseline(seed) for seed in SEEDS}
    selected_runs = {}
    rows = []

    baseline_values = {
        seed: baseline[seed]["history"][-1]["dev_accuracy"] for seed in SEEDS
    }
    baseline_accuracy = float(np.mean(list(baseline_values.values())))
    baseline_loss = float(np.mean([
        metrics["history"][-1]["dev_loss"] for metrics in baseline.values()
    ]))
    rows.append(
        f"Baseline & -- & -- & {100 * baseline_accuracy:.2f} & -- & "
        f"{baseline_loss:.3f} & -- \\\\"
    )

    parameter_names = {
        "augmentation": "ratio",
        "label_smoothing": "epsilon",
        "mixup": "alpha",
        "rdrop": "coefficient",
        "neftune": "alpha",
        "llrd": "decay",
    }
    for method, configurations in METHODS.items():
        parameter = next(iter(configurations[0][0]))
        selected_reg = {parameter: selected_parameters[parameter]}
        selected_group = {
            seed: load_candidate(method, selected_reg, seed) for seed in SEEDS
        }
        selected_runs[method] = selected_group
        final_rows = [metrics["history"][-1] for metrics in selected_group.values()]
        gains = [
            row["dev_accuracy"] - baseline_values[seed]
            for seed, row in zip(SEEDS, final_rows, strict=True)
        ]
        candidates = ", ".join(f"{reg[parameter]:g}" for reg, _ in configurations)
        decision = inclusion[method]
        rows.append(
            f"{METHOD_LABELS[method]} & {candidates} & "
            f"{parameter_names[method]}={selected_parameters[parameter]:g} & "
            f"{100 * np.mean([item['dev_accuracy'] for item in final_rows]):.2f} & "
            f"{100 * np.mean(gains):+.2f} & "
            f"{np.mean([item['dev_loss'] for item in final_rows]):.3f} & "
            f"{decision['seeds_better_than_baseline']}/{len(SEEDS)}; "
            f"{'include' if decision['included'] else 'exclude'} \\\\"
        )

    combination_runs = {}
    for seed in SEEDS:
        path = OUT / "runs" / "optimized_regularization_combination" / f"seed{seed}" / "metrics.json"
        metrics = json.loads(path.read_text())
        validate_history(metrics, path)
        combination_runs[seed] = metrics
    combination_final = [metrics["history"][-1] for metrics in combination_runs.values()]
    combination_gains = [
        row["dev_accuracy"] - baseline_values[seed]
        for seed, row in zip(SEEDS, combination_final, strict=True)
    ]
    included_names = ", ".join(METHOD_LABELS[name] for name in optimized_parameters) or "none"
    display_names = {"label_smoothing": "smoothing"}
    combination_setting = ", ".join(
        f"{display_names.get(key, METHOD_LABELS[key])}={value:g}"
        for key, value in optimized_parameters.items()
    ) or "baseline only"
    rows.append(
        f"Optimized Combination & {included_names} & {combination_setting} & "
        f"{100 * np.mean([item['dev_accuracy'] for item in combination_final]):.2f} & "
        f"{100 * np.mean(combination_gains):+.2f} & "
        f"{np.mean([item['dev_loss'] for item in combination_final]):.3f} & "
        f"{sum(value > 0 for value in combination_gains)}/{len(SEEDS)} \\\\"
    )

    table = [
        r"\begin{table*}[tp]",
        r"\centering\scriptsize\setlength{\tabcolsep}{3pt}",
        r"\caption{Candidate settings after 12 epochs and " + str(len(SEEDS)) + r" seeds. Include a method only if its mean paired gain is positive and a majority of seeds beat baseline.}",
        r"\label{tab:selected-hyperparameters}",
        r"\begin{tabular}{p{0.16\textwidth}p{0.13\textwidth}p{0.20\textwidth}rrrp{0.13\textwidth}}",
        r"\toprule",
        r"Method & Candidates & Selected & Dev acc. (\%) & Gain (pp) & Dev loss & Decision \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        r"\begin{table*}[tp]",
        r"\centering\scriptsize\setlength{\tabcolsep}{3pt}",
        r"\caption{Final development accuracy (\%) for all " + str(len(SEEDS)) + r" seeds after 12 epochs, ranked by Optimized Combination gain. Every seed remains in the analysis.}",
        r"\label{tab:selected-seeds}",
        r"\begin{tabular}{rrrrrrrrrrr}",
        r"\toprule",
        r"Rank & Seed & Baseline & Aug. & Smooth. & Mixup & R-Drop & NEFTune & LLRD & Optimized & Gain (pp) \\",
        r"\midrule",
    ]
    ranked_seeds = sorted(
        SEEDS,
        key=lambda seed: combination_runs[seed]["history"][-1]["dev_accuracy"]
        - baseline[seed]["history"][-1]["dev_accuracy"],
        reverse=True,
    )
    for rank, seed in enumerate(ranked_seeds, start=1):
        cells = [str(rank), str(seed), f"{100 * baseline_values[seed]:.2f}"]
        for method in ("augmentation", "label_smoothing", "mixup", "rdrop", "neftune", "llrd"):
            cells.append(f"{100 * selected_runs[method][seed]['history'][-1]['dev_accuracy']:.2f}")
        optimized_accuracy = combination_runs[seed]["history"][-1]["dev_accuracy"]
        baseline_accuracy = baseline[seed]["history"][-1]["dev_accuracy"]
        cells.extend((
            f"{100 * optimized_accuracy:.2f}",
            f"{100 * (optimized_accuracy - baseline_accuracy):+.2f}",
        ))
        table.append(" & ".join(cells) + r" \\")
    table.extend((
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ))
    (REPORT / "selection_results.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
    write_evaluation_matrix(baseline, selected_runs, combination_runs, selected_parameters)


def plot_saved_curves() -> None:
    baseline = {seed: load_baseline(seed) for seed in SEEDS}
    selection = json.loads((OUT / "selection.json").read_text())
    selected_parameters = selection["selected_parameters"]
    for method, configurations in METHODS.items():
        variants = []
        for reg, label in configurations:
            method_runs = {seed: load_candidate(method, reg, seed) for seed in SEEDS}
            key, value = next(iter(reg.items()))
            selected_label = f"SELECTED: {label}" if selected_parameters[key] == value else label
            variants.append((selected_label, method_runs))
        plot_curves(method, variants, baseline)

    optimized_runs = {}
    for seed in SEEDS:
        path = OUT / "runs" / "optimized_regularization_combination" / f"seed{seed}" / "metrics.json"
        metrics = json.loads(path.read_text())
        validate_history(metrics, path)
        optimized_runs[seed] = metrics
    setting = ", ".join(
        f"{name}={value:g}" for name, value in selection["optimized_parameters"].items()
    ) or "baseline only"
    plot_curves("optimized_combination", [(setting, optimized_runs)], baseline)
    write_report_tables()


def main() -> None:
    os.chdir(ROOT)
    if not torch.cuda.is_available():
        raise SystemExit("Run this study inside a Slurm GPU allocation.")

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = replace(Config.load(ROOT / "configs/default.yaml"), epochs=EPOCHS,
                  lr=BASE_LR, save_final_model=False)
    base_splits = load_splits(cfg)
    if len(base_splits["train"]) != 1742 or len(base_splits["dev"]) != 374:
        raise ValueError("Unexpected RTE split sizes; refusing to run the curve study.")
    pad_id = AutoTokenizer.from_pretrained(cfg.model_name).pad_token_id
    environment = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": torch.cuda.get_device_name(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "epochs": EPOCHS,
        "seeds": SEEDS,
        "base_learning_rate": BASE_LR,
        "candidate_settings": METHODS,
        "baseline_source": "reuse existing seed metrics; train only new seeds",
        "reused_augmentation_ratio": 0.5,
        "reused_rdrop_coefficient": 0.1,
        "selection_rule": "highest mean final development accuracy; tie lower mean development loss; then lower parameter value",
        "combination_inclusion_rule": f"include only when mean paired development-accuracy gain is positive and at least {MINIMUM_SEED_WINS} of {len(SEEDS)} seeds beat baseline",
    }
    write_json(OUT / "environment.json", environment)
    write_json(OUT / "status.json", {"state": "running", "completed_runs": 0})
    started = time.monotonic()
    completed = 0
    baseline = {}
    for seed in SEEDS:
        existing_baseline = OVERFIT_RUNS / "baseline" / f"seed{seed}" / "metrics.json"
        if existing_baseline.is_file():
            baseline[seed] = load_existing("baseline", seed, {})
            continue
        run_dir = OUT / "runs" / "baseline" / f"seed{seed}"
        metrics = run(
            cfg, RunSpec(reg={}, lr=BASE_LR, seed=seed), run_dir,
            splits=base_splits, pad_id=pad_id, record_train_accuracy=True,
            record_train_clean_loss=True, save_resume=False,
        )
        validate_history(metrics, run_dir)
        baseline[seed] = metrics
        completed += 1
        append_progress({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "completed_runs": completed,
            "method": "baseline",
            "setting": "default",
            "seed": seed,
            "train_seconds": metrics.get("train_seconds", 0.0),
            "dev_accuracy": metrics["history"][-1]["dev_accuracy"],
        })
        write_json(OUT / "status.json", {
            "state": "running", "completed_runs": completed,
            "current": {"method": "baseline", "seed": seed},
        })
    candidate_results = {}
    selected_parameters = {}
    optimized_parameters = {}
    inclusion_decisions = {}
    selected_run_groups = {}
    baseline_values = {
        seed: baseline[seed]["history"][-1]["dev_accuracy"] for seed in SEEDS
    }

    try:
        for method, configurations in METHODS.items():
            variants = []
            summaries = []
            for reg, label in configurations:
                splits = load_splits(cfg, reg["augmentation"]) if "augmentation" in reg else base_splits
                method_runs = {}
                for seed in SEEDS:
                    if is_reusable_rdrop(reg, seed):
                        metrics = load_existing("rdrop", seed, reg)
                    else:
                        run_dir = candidate_run_dir(method, reg, seed)
                        metrics = run(
                            cfg, RunSpec(reg=reg, lr=BASE_LR, seed=seed), run_dir,
                            splits=splits, pad_id=pad_id, record_train_accuracy=True,
                            record_train_clean_loss=True, save_resume=False,
                        )
                        if metrics.get("epochs") != EPOCHS or len(metrics.get("history", [])) != EPOCHS:
                            raise ValueError(f"Incomplete epoch history: {method}, {label}, seed {seed}")
                        validate_history(metrics, run_dir)
                    method_runs[seed] = metrics
                    completed += 1
                    append_progress({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "completed_runs": completed,
                        "method": method,
                        "setting": label,
                        "seed": seed,
                        "train_seconds": metrics.get("train_seconds", 0.0),
                        "dev_accuracy": metrics["history"][-1]["dev_accuracy"],
                    })
                    write_json(OUT / "status.json", {
                        "state": "running", "completed_runs": completed,
                        "current": {"method": method, "setting": label, "seed": seed},
                    })
                summary = candidate_summary(method, reg, label, method_runs)
                summaries.append(summary)
                variants.append((label, method_runs))
            selected = max(summaries, key=selection_key)
            selected_parameters.update(selected["reg"])
            selected_group = next(
                method_runs
                for (candidate_reg, _), (_, method_runs) in zip(
                    configurations, variants, strict=True
                )
                if candidate_reg == selected["reg"]
            )
            selected_run_groups[method] = selected_group
            decision = inclusion_decision(selected_group, baseline)
            selected.update(decision)
            selected["included_in_optimized_combination"] = decision["included"]
            inclusion_decisions[method] = decision
            if decision["included"]:
                optimized_parameters.update(selected["reg"])
            candidate_results[method] = summaries
            write_json(OUT / "selection.json", {
                "rule": environment["selection_rule"],
                "combination_inclusion_rule": environment["combination_inclusion_rule"],
                "selected_parameters": selected_parameters,
                "optimized_parameters": optimized_parameters,
                "inclusion_decisions": inclusion_decisions,
                "candidates": candidate_results,
            })
            variants = [
                (f"SELECTED: {label}" if reg == selected["reg"] else label, method_runs)
                for (reg, label), (_, method_runs) in zip(configurations, variants, strict=True)
            ]
            plot_curves(method, variants, baseline)

        write_json(OUT / "selection.json", {
            "rule": environment["selection_rule"],
            "combination_inclusion_rule": environment["combination_inclusion_rule"],
            "selected_parameters": selected_parameters,
            "optimized_parameters": optimized_parameters,
            "included_methods": list(optimized_parameters),
            "excluded_methods": [method for method in METHODS if method not in optimized_parameters],
            "inclusion_decisions": inclusion_decisions,
            "candidates": candidate_results,
        })

        stack_reg = dict(optimized_parameters)
        stack_splits = load_splits(cfg, stack_reg["augmentation"]) if "augmentation" in stack_reg else base_splits
        optimized_runs = {}
        for seed in SEEDS:
            run_dir = OUT / "runs" / "optimized_regularization_combination" / f"seed{seed}"
            metrics = run(
                cfg, RunSpec(reg=stack_reg, lr=BASE_LR, seed=seed), run_dir,
                splits=stack_splits, pad_id=pad_id, record_train_accuracy=True,
                record_train_clean_loss=True, save_resume=False,
            )
            if len(metrics.get("history", [])) != EPOCHS:
                raise ValueError(f"Incomplete full-stack history for seed {seed}")
            validate_history(metrics, run_dir)
            optimized_runs[seed] = metrics
            completed += 1
            write_json(OUT / "status.json", {
                "state": "running", "completed_runs": completed,
                "current": {"method": "optimized_regularization_combination", "seed": seed},
            })
        plot_curves("optimized_combination", [("baseline-screened selected settings", optimized_runs)], baseline)

        write_json(OUT / "summary.json", {
            "protocol": environment,
            "completed_runs": completed,
            "candidate_results": candidate_results,
            "selected_parameters": selected_parameters,
            "optimized_parameters": optimized_parameters,
            "inclusion_decisions": inclusion_decisions,
            "optimized_combination_dev_accuracy": {
                str(seed): metrics["dev"]["accuracy"] for seed, metrics in optimized_runs.items()
            },
            "figure_files": [*(f"curves_{method}.pdf" for method in METHODS),
                             "curves_optimized_combination.pdf"],
            "controller_wall_seconds": time.monotonic() - started,
        })
        write_report_tables()
        write_json(OUT / "status.json", {
            "state": "completed", "completed_runs": completed,
            "controller_wall_seconds": time.monotonic() - started,
        })
    except BaseException as exc:
        write_json(OUT / "status.json", {
            "state": "failed", "completed_runs": completed,
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots-only", action="store_true",
                        help="Regenerate figures from completed metric histories without training.")
    arguments = parser.parse_args()
    if arguments.plots_only:
        plot_saved_curves()
    else:
        main()