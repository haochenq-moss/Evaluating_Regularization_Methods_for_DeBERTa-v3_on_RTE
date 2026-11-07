#!/usr/bin/env python3
"""Extend the existing report with completed, development-only follow-up results."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import report_epoch_study as original
import run_followup_study as study

ROOT = study.ROOT
OUT = study.OUT
LABELS = {
    "baseline": "Baseline",
    "mixup": "Mixup",
    "rdrop": "R-Drop",
    "neftune": "NEFTune",
    "mixup_rdrop": "Mixup + R-Drop",
    "stack": "Full stack",
}


def load_runs(root=OUT):
    results = {}
    for method, epochs in study.CONDITIONS:
        key = (method, epochs)
        results[key] = {}
        for seed in study.SEEDS:
            path = root / "runs" / f"{method}_{epochs}" / f"seed{seed}" / "metrics.json"
            run = json.loads(path.read_text(encoding="utf-8"))
            history = run["history"]
            if (run["seed"] != seed or run["epochs"] != epochs or run["lr"] != study.LR
                    or run["reg"] != study.METHODS[method] or len(history) != epochs
                    or run["dev"]["n"] != 374):
                raise ValueError(f"Inconsistent run: {path}")
            for index, row in enumerate(history, 1):
                if row["epoch"] != index or row["dev_n"] != 374:
                    raise ValueError(f"Invalid history in {path}")
                for field in ("dev_accuracy", "dev_loss", "train_accuracy", "train_loss"):
                    value = row[field]
                    if not math.isfinite(value) or value < 0:
                        raise ValueError(f"Invalid {field} in {path}")
                    if "accuracy" in field and value > 1:
                        raise ValueError(f"Invalid {field} in {path}")
            if abs(run["dev"]["accuracy"] - history[-1]["dev_accuracy"]) > 1e-12:
                raise ValueError(f"Final development accuracy disagrees with history: {path}")
            results[key][seed] = run
    return results


def summary_rows(runs):
    base = runs[("baseline", 8)]
    rows = []
    for method in study.METHODS:
        group = runs[(method, 8)]
        scores = [100 * group[s]["history"][-1]["dev_accuracy"] for s in study.SEEDS]
        mean, sd = statistics.mean(scores), statistics.stdev(scores)
        if method == "baseline":
            interval = "--"
        else:
            diffs = [100 * (group[s]["history"][-1]["dev_accuracy"]
                            - base[s]["history"][-1]["dev_accuracy"]) for s in study.SEEDS]
            avg = statistics.mean(diffs)
            half = original.T_CRITICAL * statistics.stdev(diffs) / math.sqrt(len(diffs))
            interval = f"{avg:+.2f} [{avg-half:+.2f}, {avg+half:+.2f}]"
        rows.append(f"{LABELS[method]} & {mean:.2f} $\\pm$ {sd:.2f} & {interval} \\\\")
    return "\n".join(rows)


def epoch_rows(runs):
    rows = []
    for method, epochs in study.CONDITIONS:
        group = runs[(method, epochs)]
        best = [study.best_epoch(group[s]["history"]) for s in study.SEEDS]
        final = [100 * group[s]["history"][-1]["dev_accuracy"] for s in study.SEEDS]
        best_acc = [100 * row["dev_accuracy"] for row in best]
        selected = ", ".join(str(row["epoch"]) for row in best)
        rows.append(f"{LABELS[method]} & {epochs} & {statistics.mean(final):.2f} & "
                    f"{statistics.mean(best_acc):.2f} & {selected} \\\\")
    return "\n".join(rows)


def create_figure(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for method, color in zip(study.METHODS, plt.cm.tab10.colors):
        group = runs[(method, 8)]
        values = np.array([[row["dev_accuracy"] * 100 for row in group[s]["history"]]
                           for s in study.SEEDS])
        axes[0].plot(range(1, 9), values.mean(axis=0), marker="o",
                     label=LABELS[method], color=color, markersize=3)
    for epochs in (8, 10, 12):
        group = runs[("stack", epochs)]
        values = np.array([[row["dev_accuracy"] * 100 for row in group[s]["history"]]
                           for s in study.SEEDS])
        axes[1].plot(range(1, epochs + 1), values.mean(axis=0),
                     marker="o", markersize=3, label=f"{epochs}-epoch schedule")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Development accuracy (%)")
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    axes[0].set_title("Matched 8-epoch ablation")
    axes[1].set_title("Stack training duration")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"followup_curves.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def disagreements(root=ROOT):
    folder = root / "outputs/epoch_study/runs"
    records = {}
    for label, directory in (("baseline", "baseline"), ("rdrop", "rdrop"), ("stack", "stack8")):
        with np.load(folder / directory / "seed789/predictions_holdout.npz") as data:
            labels = data["labels"].copy()
            predictions = data["logits"].argmax(axis=-1)
            records[label] = (labels, predictions)
    labels, base = records["baseline"]
    lines = []
    for key in ("rdrop", "stack"):
        other_labels, other = records[key]
        if not np.array_equal(labels, other_labels):
            raise ValueError("Saved holdout label sequences differ across methods.")
        only_base = int(np.sum((base == labels) & (other != labels)))
        only_other = int(np.sum((base != labels) & (other == labels)))
        lines.append(f"{LABELS[key]} & {int(np.sum(base != other))} & "
                     f"{only_base} & {only_other} \\\\")
    return "\n".join(lines)


def followup_section(runs, environment, error_rows):
    total_seconds = sum(run["train_seconds"] for group in runs.values() for run in group.values())
    def final_mean(method, epochs):
        return statistics.mean(
            100 * runs[(method, epochs)][seed]["history"][-1]["dev_accuracy"]
            for seed in study.SEEDS
        )
    stack_diffs = [
        100 * (runs[("stack", 8)][seed]["history"][-1]["dev_accuracy"]
               - runs[("baseline", 8)][seed]["history"][-1]["dev_accuracy"])
        for seed in study.SEEDS
    ]
    stack_gain = statistics.mean(stack_diffs)
    stack_half_width = original.T_CRITICAL * statistics.stdev(stack_diffs) / math.sqrt(len(stack_diffs))
    summary = (
        f"At eight epochs, baseline, R-Drop and the full stack achieved "
        f"{final_mean('baseline', 8):.2f}\\%, {final_mean('rdrop', 8):.2f}\\% "
        f"and {final_mean('stack', 8):.2f}\\% mean development accuracy. "
        f"The stack's paired gain over baseline was {stack_gain:+.2f} percentage points "
        f"(95\\% seed interval [{stack_gain-stack_half_width:+.2f}, "
        f"{stack_gain+stack_half_width:+.2f}]). "
        f"The full stack's final-epoch means for 8, 10 and 12 epochs were "
        f"{final_mean('stack', 8):.2f}\\%, {final_mean('stack', 10):.2f}\\% "
        f"and {final_mean('stack', 12):.2f}\\%, respectively; "
        "these descriptive comparisons do not establish that longer training improves generalization."
    )
    return r"""
\section{Matched Ablation and Longer Training}
This follow-up is exploratory and uses only the previously used internal development set ($n=374$); no fresh test set was available and no new holdout evaluation was performed. Six pipelines were trained for eight epochs using identical batch size, learning rate $3\times10^{-5}$, schedule and seeds. Each was trained from the pretrained model. Table~\ref{tab:ablation} compares final-epoch accuracies with the matched baseline. The confidence intervals describe seed variation conditional on this development set; repeated tuning and multiple comparisons limit interpretation.
FOLLOWUP_SUMMARY
\begin{table}[tp]
\centering\scriptsize\setlength{\tabcolsep}{3pt}
\caption{Eight-epoch development ablation: accuracy (\%, mean $\pm$ SD) and paired gain over baseline (percentage points, 95\% seed interval).}
\label{tab:ablation}
\begin{tabular}{lrr}\toprule
Pipeline & Accuracy & Gain [95\% CI] \\ \midrule
ABLATION_ROWS
\bottomrule\end{tabular}
\end{table}
The full stack was also trained with separate 10- and 12-epoch schedules. For each run, the best observed validation epoch maximizes accuracy, breaking ties by lower validation loss and then the earlier epoch. Table~\ref{tab:best-epochs} compares this optimistic per-seed selection with final epochs. Because only metrics were recorded, no checkpoint weights were saved; these best-epoch results cannot be independently evaluated or deployed without reproducing the runs. Choosing among durations on the same development set adds selection bias. Figure~\ref{fig:followup} shows the mean trajectories, not independent test curves.
\begin{table}[tp]
\centering\scriptsize\setlength{\tabcolsep}{2pt}
\caption{Development accuracies at final and per-seed best epochs; selected epochs follow seed order 42, 123, 456, 789, 1011.}
\label{tab:best-epochs}
\begin{tabular}{lrrrl}\toprule
Pipeline & Max & Final & Best & Selected epochs \\ \midrule
EPOCH_ROWS
\bottomrule\end{tabular}
\end{table}
\begin{figure*}[tp]
\centering\includegraphics[width=\textwidth,height=0.36\textheight,keepaspectratio]{followup_curves.pdf}
\caption{Mean development accuracy over five matched seeds. The three duration curves use separately scheduled stack runs.}
\label{fig:followup}
\end{figure*}
\subsection{Prediction Disagreements}
The prior holdout evaluation produced equal seed-789 accuracy for all three models, but their errors differed. Table~\ref{tab:disagree} compares the original saved predictions against the baseline. This descriptive examination uses the already inspected holdout, not a fresh evaluation.
\begin{table}[tp]
\centering\small\setlength{\tabcolsep}{3pt}
\caption{Seed 789: disagreements and cases correctly predicted only by one model.}
\label{tab:disagree}
\begin{tabular}{lrrr}\toprule
Model & Disagree & Baseline only & Model only \\ \midrule
ERROR_ROWS
\bottomrule\end{tabular}
\end{table}
\subsection{Compute and Reproducibility}
Follow-up environment: GPU @DEVICE@; Python @PYTHON@; PyTorch @TORCH@ (CUDA @CUDA@); Transformers @TRANSFORMERS@; Datasets @DATASETS@. Across the 40 runs, recorded training-loop time totaled TRAIN_HOURS GPU-hours; the final controller invocation took WALL_HOURS hours, excluding Slurm queueing and report compilation and any earlier interrupted invocations. Training-loop time excludes evaluation and orchestration. Runs use the original grouped split seed 13 and the five matched training seeds.
""" .replace("FOLLOWUP_SUMMARY", summary).replace("ABLATION_ROWS", summary_rows(runs)).replace("EPOCH_ROWS", epoch_rows(runs)).replace(
        "ERROR_ROWS", error_rows).replace("@DEVICE@", original.latex_escape(environment["device"])).replace(
        "@PYTHON@", original.latex_escape(environment["python"])).replace(
        "@TORCH@", original.latex_escape(environment["torch"])).replace(
        "@CUDA@", original.latex_escape(environment["cuda"] or "not recorded")).replace(
        "@TRANSFORMERS@", original.latex_escape(environment["transformers"])).replace(
        "@DATASETS@", original.latex_escape(environment["datasets"])).replace(
        "TRAIN_HOURS", f"{total_seconds / 3600:.2f}").replace(
        "WALL_HOURS", f"{environment['controller_wall_seconds'] / 3600:.2f}")


def main():
    runs = load_runs(OUT)
    environment = json.loads((OUT / "environment.json").read_text(encoding="utf-8"))
    old = ROOT / "outputs/epoch_study"
    selection = json.loads((old / "selection.json").read_text(encoding="utf-8"))
    final = json.loads((old / "final_metrics.json").read_text(encoding="utf-8"))
    report_dir = OUT / "report"
    original.generate(selection, final, report_dir, old / "runs")
    create_figure(runs, report_dir)
    tex_path = report_dir / "report.tex"
    tex = tex_path.read_text(encoding="utf-8")
    marker = r"\section{Interpretation and Limitations}"
    if tex.count(marker) != 1:
        raise ValueError("Cannot locate unique interpretation section in original report.")
    tex = tex.replace(marker, followup_section(runs, environment, disagreements()) + "\n" + marker)
    tex_path.write_text(tex, encoding="utf-8")
    for file in ("aaai2026.sty", "aaai2026.bst"):
        shutil.copy2(old / "report" / file, report_dir / file)
    print(f"Follow-up report source written to {tex_path}", flush=True)


if __name__ == "__main__":
    main()
