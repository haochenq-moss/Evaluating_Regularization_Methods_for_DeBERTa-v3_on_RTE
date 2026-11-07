#!/usr/bin/env python3
"""Extend the combined report with development-only overfitting diagnostics."""
from __future__ import annotations

import json
import math
import shutil
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import report_epoch_study as original
import run_overfitting_study as study


def load_run(path, seed):
    item = json.loads((path / "metrics.json").read_text())
    if item["seed"] != seed or item["epochs"] != study.EPOCHS or len(item["history"]) != study.EPOCHS:
        raise ValueError(f"Invalid run: {path}")
    for i, row in enumerate(item["history"], 1):
        if row["epoch"] != i or row["dev_n"] != 374:
            raise ValueError(f"Invalid epoch or development size: {path}")
        for field in ("train_accuracy", "dev_accuracy", "train_clean_loss", "dev_loss"):
            value = row[field]
            if not math.isfinite(value) or value < 0 or ("accuracy" in field and value > 1):
                raise ValueError(f"Invalid {field}: {path}")
    return item


def collect(root):
    if json.loads((root / "protocol.json").read_text()) != json.loads(json.dumps(study.PROTOCOL)):
        raise ValueError("Unexpected study protocol")
    selection = json.loads((root / "selection.json").read_text())
    runs = {}
    for tuning in ("full", "lora"):
        for method in ("baseline", "rdrop"):
            group = {}
            for seed in study.EVAL_SEEDS:
                path = root / "runs" / tuning / method / f"seed{seed}"
                item = load_run(path, seed)
                expected_reg = {} if method == "baseline" else {"rdrop": selection[f"{tuning}_strength"]}
                if item["reg"] != expected_reg or item["lr"] != study.LRS[tuning]:
                    raise ValueError(f"Unexpected settings: {path}")
                identity = json.loads((path / "study_config.json").read_text())
                if tuning == "lora" and (identity["rank"] != selection["rank"]
                                          or identity["dropout"] != selection["dropout"]
                                          or identity["alpha"] != 2 * selection["rank"]):
                    raise ValueError(f"Unexpected LoRA adapter: {path}")
                group[seed] = item
            runs[(tuning, method)] = group
    screen = {}
    for rank in study.RANKS:
        for dropout in study.DROPOUTS:
            key = f"rank{rank}_drop{dropout:g}"
            screen[key] = [load_run(root / "screen" / key / f"seed{s}", s)
                           for s in study.SCREEN_SEEDS]
    for tuning in ("full", "lora"):
        for strength in study.STRENGTHS:
            key = f"{tuning}_rdrop{strength:g}"
            screen[key] = [load_run(root / "screen" / key / f"seed{s}", s)
                           for s in study.SCREEN_SEEDS]
            for seed in study.SCREEN_SEEDS:
                path = root / "screen" / key / f"seed{seed}"
                identity = json.loads((path / "study_config.json").read_text())
                if identity["reg"] != {"rdrop": strength}:
                    raise ValueError(f"Unexpected screening settings: {path}")
                if tuning == "lora" and (identity["rank"] != selection["rank"]
                                          or identity["dropout"] != selection["dropout"]):
                    raise ValueError(f"Unexpected screening adapter: {path}")
    rank_scores = {(rank, dropout): screen[f"rank{rank}_drop{dropout:g}"]
                   for rank in study.RANKS for dropout in study.DROPOUTS}
    if study.rank_screen(rank_scores) != (selection["rank"], selection["dropout"]):
        raise ValueError("Frozen adapter selection differs from screening results")
    for tuning in ("full", "lora"):
        scores = {(strength,): screen[f"{tuning}_rdrop{strength:g}"]
                  for strength in study.STRENGTHS}
        if study.rank_screen(scores)[0] != selection[f"{tuning}_strength"]:
            raise ValueError(f"Frozen {tuning} R-Drop selection differs from screen")
    return selection, runs, screen


def comparison_rows(runs):
    rows = []
    for tuning in ("full", "lora"):
        for method in ("baseline", "rdrop"):
            history = [runs[(tuning, method)][s]["history"] for s in study.EVAL_SEEDS]
            final = [h[-1] for h in history]
            best = [study.best_epoch(h) for h in history]
            train = statistics.mean(x["train_accuracy"] * 100 for x in final)
            dev = statistics.mean(x["dev_accuracy"] * 100 for x in final)
            gap = statistics.mean((x["train_accuracy"] - x["dev_accuracy"]) * 100 for x in final)
            train_loss = statistics.mean(x["train_clean_loss"] for x in final)
            loss = statistics.mean(x["dev_loss"] for x in final)
            best_dev = statistics.mean(x["dev_accuracy"] * 100 for x in best)
            epochs = ",".join(str(x["epoch"]) for x in best)
            label = "Full" if tuning == "full" else "LoRA"
            rows.append(f"{label} & {method.title()} & {train:.2f} & {dev:.2f} & "
                        f"{gap:.2f} & {train_loss:.3f} & {loss:.3f} & {best_dev:.2f} & {epochs} \\\\")
    return "\n".join(rows)


def screen_rows(screen):
    rows = []
    for key, group in screen.items():
        acc = statistics.mean(item["dev"]["accuracy"] * 100 for item in group)
        loss = statistics.mean(item["dev"]["loss"] for item in group)
        label = original.latex_escape(key)
        rows.append(f"{label} & {acc:.2f} & {loss:.3f} \\\\")
    return "\n".join(rows)


def gain_rows(runs):
    rows = []
    for tuning in ("full", "lora"):
        diffs = [100 * (runs[(tuning, "rdrop")][s]["history"][-1]["dev_accuracy"]
                        - runs[(tuning, "baseline")][s]["history"][-1]["dev_accuracy"])
                 for s in study.EVAL_SEEDS]
        mean = statistics.mean(diffs)
        half = original.T_CRITICAL * statistics.stdev(diffs) / math.sqrt(len(diffs))
        rows.append(f"{'Full' if tuning == 'full' else 'LoRA'} & "
                    f"{mean:+.2f} & [{mean-half:+.2f}, {mean+half:+.2f}] \\\\")
    return "\n".join(rows)


def create_figures(runs, screen, out):
    colors = {"baseline": "#0072B2", "rdrop": "#D55E00"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for col, tuning in enumerate(("full", "lora")):
        for method in ("baseline", "rdrop"):
            histories = [runs[(tuning, method)][s]["history"] for s in study.EVAL_SEEDS]
            for row, metric in enumerate(("accuracy", "loss")):
                for split, style in (("train", "--"), ("dev", "-")):
                    field = (f"{split}_accuracy" if metric == "accuracy"
                             else ("train_clean_loss" if split == "train" else "dev_loss"))
                    vals = np.array([[entry[field] for entry in h] for h in histories])
                    mean = vals.mean(axis=0) * (100 if metric == "accuracy" else 1)
                    sd = vals.std(axis=0, ddof=1) * (100 if metric == "accuracy" else 1)
                    axes[row, col].plot(range(1, study.EPOCHS + 1), mean, style,
                                        color=colors[method], label=f"{method.title()} {split}")
                    axes[row, col].fill_between(range(1, study.EPOCHS + 1), mean-sd, mean+sd,
                                                color=colors[method], alpha=.06)
        axes[0, col].set_title("Full fine-tuning" if tuning == "full" else "Standard LoRA")
        for row in (0, 1):
            axes[row, col].grid(alpha=.25)
            axes[row, col].legend(fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Clean accuracy (%)")
    axes[1, 0].set_ylabel("Clean cross-entropy")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"overfitting_curves.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharex=True)
    for col, tuning in enumerate(("full", "lora")):
        for method in ("baseline", "rdrop"):
            histories = [runs[(tuning, method)][s]["history"] for s in study.EVAL_SEEDS]
            gaps = np.array([[100 * (x["train_accuracy"] - x["dev_accuracy"]) for x in h]
                             for h in histories])
            mean, sd = gaps.mean(axis=0), gaps.std(axis=0, ddof=1)
            axes[col].plot(range(1, study.EPOCHS + 1), mean, color=colors[method], label=method.title())
            axes[col].fill_between(range(1, study.EPOCHS + 1), mean-sd, mean+sd,
                                   color=colors[method], alpha=.15)
        axes[col].set_title("Full fine-tuning" if tuning == "full" else "Standard LoRA")
        axes[col].set_xlabel("Epoch")
        axes[col].grid(alpha=.25)
        axes[col].legend()
    axes[0].set_ylabel("Train–development gap (percentage points)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"overfitting_gaps.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, keys, labels in (
        (axes[0], [f"rank{r}_drop{d:g}" for r in study.RANKS for d in study.DROPOUTS],
         [f"r={r}\nd={d:g}" for r in study.RANKS for d in study.DROPOUTS]),
        (axes[1], [f"{t}_rdrop{s:g}" for t in ("full", "lora") for s in study.STRENGTHS],
         [f"{t}\n{s:g}" for t in ("full", "lora") for s in study.STRENGTHS]),
    ):
        values = [[100 * item["dev"]["accuracy"] for item in screen[key]] for key in keys]
        ax.bar(range(len(keys)), [statistics.mean(v) for v in values], color="#0072B2", alpha=.8)
        ax.errorbar(range(len(keys)), [statistics.mean(v) for v in values],
                    yerr=[statistics.stdev(v) for v in values], fmt="none", ecolor="black", capsize=3)
        ax.set_xticks(range(len(labels)), labels, fontsize=8)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=.25)
    axes[0].set_title("LoRA adapter baseline screen")
    axes[1].set_title("R-Drop strength screen")
    axes[0].set_ylabel("Final development accuracy (%)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"overfitting_screen.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def paired_rows(runs):
    rows = []
    for seed in study.EVAL_SEEDS:
        cells = [str(seed)]
        for tuning in ("full", "lora"):
            base = runs[(tuning, "baseline")][seed]["history"][-1]
            reg = runs[(tuning, "rdrop")][seed]["history"][-1]
            cells.extend((f"{100*base['dev_accuracy']:.2f}", f"{100*reg['dev_accuracy']:.2f}",
                          f"{100*(reg['dev_accuracy']-base['dev_accuracy']):+.2f}"))
        rows.append(" & ".join(cells) + r" \\")
    return "\n".join(rows)


def render_section(selection, runs, screen, environment):
    screen_train = sum(item["train_seconds"] for group in screen.values() for item in group)
    eval_train = sum(item["train_seconds"] for group in runs.values() for item in group.values())
    return r"""
\section{Development-Only Overfitting Study}
We first screened baseline LoRA ranks 4, 8 and 16 and adapter dropout 0.0 and 0.1, with alpha twice the rank. We then screened R-Drop coefficients 0.1, 0.5 and 1.0 separately for full fine-tuning and LoRA, using the selected LoRA adapter. Every screen used seeds 11, 21 and 31, 12 epochs, and final development accuracy; ties were broken by lower mean development loss and then smaller parameter values. The selected LoRA configuration was rank RANK and dropout DROPOUT; selected R-Drop coefficients were FULL for full fine-tuning and LORA for LoRA. This sequential search is not an exhaustive joint search.

The five-seed comparisons use new runs with seeds 42, 123, 456, 789 and 1011, the selected settings, 12 epochs, and fixed learning rates $3\times10^{-5}$ (full) and $2\times10^{-4}$ (LoRA). These seeds differ from the screening seeds, but both stages reuse the same development set. No holdout was evaluated in this study. Model and resume weights were not saved; interruption restarts the current run.
\begin{table}[tp]
\centering\scriptsize\setlength{\tabcolsep}{3pt}
\caption{Development screening: final accuracy and loss, mean over three seeds. Rank/drop entries are LoRA baselines; LoRA R-Drop entries use the selected adapter.}
\begin{tabular}{lrr}\toprule
Candidate & Accuracy (\%) & Loss \\ \midrule
SCREEN_ROWS
\bottomrule\end{tabular}
\end{table}
\begin{table*}[tp]
\centering\small\setlength{\tabcolsep}{3pt}
\caption{Paired final development accuracy by seed (\%); changes are R-Drop minus baseline in percentage points.}
\begin{tabular}{rrrrrrr}\toprule
Seed & Full baseline & Full R-Drop & Full change & LoRA baseline & LoRA R-Drop & LoRA change \\ \midrule
PAIRED_ROWS
\bottomrule\end{tabular}
\end{table*}
\begin{table*}[tp]
\centering\small\setlength{\tabcolsep}{3pt}
\caption{Five-seed clean final-epoch metrics and per-run best-development accuracy. Best epochs are ordered 42, 123, 456, 789, 1011; they are selected on development and cannot be tested without retraining.}
\begin{tabular}{llrrrrrrl}\toprule
Tuning & Method & Train \% & Dev \% & Gap pp & Train loss & Dev loss & Best dev \% & Best epochs \\ \midrule
COMPARE_ROWS
\bottomrule\end{tabular}
\end{table*}
\begin{figure*}[tp]
\centering\includegraphics[width=\textwidth,height=0.34\textheight,keepaspectratio]{overfitting_screen.pdf}
\caption{Final development accuracy in the sequential screens. Bars show three-seed means; error bars show one sample standard deviation.}
\end{figure*}
\begin{table}[tp]
\centering\small
\caption{Final-epoch paired development accuracy change for R-Drop versus baseline (percentage points); 95\% interval describes seed variation conditional on this development set.}
\begin{tabular}{lrr}\toprule
Tuning & Mean & 95\% CI \\ \midrule
GAIN_ROWS
\bottomrule\end{tabular}
\end{table}
The best-development epoch is a retrospective checkpoint-selection comparison on the same scheduled trajectory, not a separate patience-based early-stopping run. No selected weights were saved or evaluated independently. A smaller gap alone may indicate underfitting; interpret it alongside development accuracy and clean development loss.
\begin{figure*}[tp]
\centering\includegraphics[width=\textwidth,height=0.67\textheight,keepaspectratio]{overfitting_curves.pdf}
\caption{Clean evaluation-mode train and development accuracy and cross-entropy, averaged across five seeds; shading is one sample standard deviation. Loss excludes regularization terms and is comparable across methods.}
\end{figure*}
\begin{figure*}[tp]
\centering\includegraphics[width=\textwidth,height=0.34\textheight,keepaspectratio]{overfitting_gaps.pdf}
\caption{Mean clean train--development accuracy gaps; shading denotes one sample standard deviation over five seeds.}
\end{figure*}
Recorded controller wall time across the latest completed phase invocations was WALL hours, excluding queueing and report compilation. The cumulative training-loop time was SCREEN_TIME hours for screening and EVAL_TIME hours for the matched comparison; it excludes evaluation and setup. If resumed, controller wall time excludes earlier attempts. The repeated development-set use and differing full/LoRA learning rates preclude independent generalization or causal claims. A fresh, previously unused labeled evaluation set remains necessary for a final test.
""".replace("RANK", str(selection["rank"])).replace("DROPOUT", f"{selection['dropout']:g}").replace(
        "FULL", f"{selection['full_strength']:g}").replace("LORA", f"{selection['lora_strength']:g}").replace(
        "SCREEN_ROWS", screen_rows(screen)).replace("COMPARE_ROWS", comparison_rows(runs)).replace(
        "GAIN_ROWS", gain_rows(runs)).replace("PAIRED_ROWS", paired_rows(runs)).replace(
        "WALL", f"{environment['controller_wall_seconds']/3600:.2f}").replace(
        "SCREEN_TIME", f"{screen_train/3600:.2f}").replace("EVAL_TIME", f"{eval_train/3600:.2f}")


def main():
    selection, runs, screen = collect(study.OUT)
    all_env = study.OUT / "environment_all.json"
    env_paths = ([all_env] if all_env.exists() else
                 [study.OUT / f"environment_{phase}.json" for phase in ("adapter", "strength", "compare")])
    environment = {"controller_wall_seconds": sum(
        json.loads(path.read_text())["controller_wall_seconds"] for path in env_paths)}
    source = study.ROOT / "outputs/lora_study/report"
    out = study.OUT / "report"
    out.mkdir(parents=True, exist_ok=True)
    for file in ("aaai2026.sty", "aaai2026.bst", "learning_curves.pdf", "paired_accuracy.pdf",
                 "followup_curves.pdf", "lora_comparison.pdf"):
        shutil.copy2(source / file, out / file)
    create_figures(runs, screen, out)
    tex = (source / "report.tex").read_text()
    marker = r"\section{Interpretation and Limitations}"
    if tex.count(marker) != 1:
        raise ValueError("Missing or repeated interpretation section.")
    tex = tex.replace(marker, render_section(selection, runs, screen, environment) + "\n" + marker)
    (out / "report.tex").write_text(tex, encoding="utf-8")
    print(f"Report written to {out}", flush=True)


if __name__ == "__main__":
    main()
