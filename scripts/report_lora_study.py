#!/usr/bin/env python3
"""Extend the prior report with frozen-selection LoRA results."""
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
import report_followup_study as previous
import run_lora_study as study


def mean_ci(values):
    mean = statistics.mean(values)
    half = original.T_CRITICAL * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - half, mean + half


def combined_abstract(runs, full):
    def average(group, method):
        return statistics.mean(100 * group[(method, 8)][s]["dev"]["accuracy"]
                               for s in study.SEEDS)
    return (
        r"\begin{abstract}" "\n"
        "We compare full fine-tuning and standard LoRA for DeBERTa-v3-base on RTE, "
        "including matched regularization ablations, learning-rate selection, training duration "
        "and computational cost. Across five matched seeds, eight-epoch development accuracies "
        f"for the full fine-tuning baseline and R-Drop were {average(full, 'baseline'):.2f}\\% "
        f"and {average(full, 'rdrop'):.2f}\\%; the corresponding LoRA results were "
        f"{average(runs, 'baseline'):.2f}\\% and {average(runs, 'rdrop'):.2f}\\%. "
        "The holdout results are exploratory because the holdout was previously inspected. "
        "Learning rates differ between the matched fine-tuning comparisons and confidence "
        "intervals describe seed variation conditional on the fixed data.\n"
        r"\end{abstract}"
    )


def study_overview(lr):
    return r"""
\subsection{Study Roadmap}
Table~\ref{tab:roadmap} separates the historical selection and holdout study from the later matched ablations. The historical holdout was evaluated after its original selection was frozen, but has since been examined and reused; subsequent holdout comparisons are exploratory, not fresh tests. In all stages the training seeds are matched where multiple seeds are used.
\begin{table*}[tp]
\centering\small\setlength{\tabcolsep}{4pt}
\caption{Study stages and evaluation data. Dev denotes internal development; holdout denotes the internal 374-example holdout.}
\label{tab:roadmap}
\begin{tabular}{lll}\toprule
Stage & Settings & Evaluation \\ \midrule
Original search & Full tuning; seed 42; LR and regularizer search & Dev selection \\
Original epoch study & Full tuning; five seeds; stack: 5/6/8 epochs & Dev selection; historical holdout \\
Matched full tuning & Six methods; five seeds; LR $3{\times}10^{-5}$; 8 epochs & Dev only \\
Longer full stack & Five seeds; LR $3{\times}10^{-5}$; 10/12 epochs & Dev only \\
LoRA screening & Baseline; seed 42; 8 epochs; 3 LR candidates & Dev selection \\
LoRA ablation & Six methods; five seeds; LR SELECTED; 8 epochs & Dev and reused holdout \\
Longer LoRA stack & Five seeds; LR SELECTED; 10/12 epochs & Dev and reused holdout \\
\bottomrule\end{tabular}
\end{table*}
""".replace("SELECTED", f"{lr:g}")


def combined_conclusion(runs, full, selection):
    def avg(group, method, epochs, split):
        return statistics.mean(100 * group[(method, epochs)][s][split]["accuracy"]
                               for s in study.SEEDS)

    def interaction(method):
        return mean_ci([
            100 * ((runs[(method, 8)][s]["dev"]["accuracy"]
                    - runs[("baseline", 8)][s]["dev"]["accuracy"])
                   - (full[(method, 8)][s]["dev"]["accuracy"]
                      - full[("baseline", 8)][s]["dev"]["accuracy"]))
            for s in study.SEEDS
        ])

    rdrop, low, high = interaction("rdrop")
    stack, stack_low, stack_high = interaction("stack")
    duration = [avg(runs, "stack", e, "dev") for e in (8, 10, 12)]
    return (
        r"\section{Interpretation and Limitations}" "\n"
        "Under matched eight-epoch settings within each fine-tuning approach, "
        f"the full-tuning baseline and R-Drop averaged {avg(full, 'baseline', 8, 'dev'):.2f}\\% "
        f"and {avg(full, 'rdrop', 8, 'dev'):.2f}\\% development accuracy, respectively; "
        f"LoRA averaged {avg(runs, 'baseline', 8, 'dev'):.2f}\\% "
        f"and {avg(runs, 'rdrop', 8, 'dev'):.2f}\\%. "
        f"The difference in R-Drop gains (LoRA minus full tuning) was {rdrop:+.2f} percentage points "
        f"(95\\% seed interval [{low:+.2f}, {high:+.2f}]); it does not establish a larger gain under LoRA. "
        f"For the full stack the difference in gains was {stack:+.2f} points "
        f"([{stack_low:+.2f}, {stack_high:+.2f}]). "
        "These are conditional comparisons on the same development set, not causal estimates.\n\n"
        f"The LoRA stack's final development means for 8, 10 and 12 epochs were "
        f"{duration[0]:.2f}\\%, {duration[1]:.2f}\\% and {duration[2]:.2f}\\%, respectively. "
        "The schedules differ and the same development data informed selection; these results motivate "
        "further testing rather than proving that additional epochs improve generalization. "
        "The 12-epoch stack's holdout result is exploratory, not evidence that it is the best model "
        "on unseen data.\n\n"
        f"Full fine-tuning used a learning rate of 3e-5 in the matched ablation while LoRA used "
        f"{selection['selected_lr']:g}; the original historical comparison also differed in duration "
        "and learning rate. The LoRA learning-rate screen used only seed 42 and did not tune regularizer "
        "strengths. "
        + ("The selected learning rate was the largest candidate tested. "
           if selection["selected_lr"] == max(study.LRS) else "")
        + "Repeated evaluation on development "
        "and holdout data and multiple comparisons can make observed gains optimistic. "
        "The seed intervals condition on the fixed data and do not quantify uncertainty across new "
        "test samples. A stronger follow-up would prespecify comparable learning-rate and duration "
        "settings, selection rules and matched seeds, then evaluate once on an independent labeled "
        "set. No new independent evaluation was performed here.\n"
    )


def load_runs(root, lr):
    runs = {}
    for method, epochs in study.CONDITIONS:
        runs[(method, epochs)] = {}
        for seed in study.SEEDS:
            path = root / "runs" / f"{method}_{epochs}" / f"seed{seed}"
            item = json.loads((path / "metrics.json").read_text())
            if (item["seed"] != seed or item["lr"] != lr or item["epochs"] != epochs
                    or item["reg"] != study.METHODS[method] or item["test"]["n"] != 374
                    or len(item["history"]) != epochs):
                raise ValueError(f"Invalid LoRA run: {path}")
            with np.load(path / "predictions_test.npz") as data:
                logits, labels = data["logits"], data["labels"]
                if (len(labels) != 374 or not np.isfinite(logits).all()
                        or abs(float((logits.argmax(-1) == labels).mean())
                               - item["test"]["accuracy"]) > 1e-12):
                    raise ValueError(f"Saved predictions disagree with test metrics: {path}")
            runs[(method, epochs)][seed] = item
    return runs


def report_section(runs, full, selection, environment, screening):
    lr_rows = []
    for lr in study.LRS:
        d = selection["screen"][str(lr)]["dev"]
        lr_rows.append(f"{lr:g} & {100*d['accuracy']:.2f} & {d['loss']:.4f} \\\\")
    rows = []
    paired = []
    baseline = runs[("baseline", 8)]
    for method in study.METHODS:
        group = runs[(method, 8)]
        dev = [100 * group[s]["dev"]["accuracy"] for s in study.SEEDS]
        test = [100 * group[s]["test"]["accuracy"] for s in study.SEEDS]
        if method == "baseline":
            interval = "--"
        else:
            diffs = [100 * (group[s]["test"]["accuracy"] - baseline[s]["test"]["accuracy"])
                     for s in study.SEEDS]
            avg, low, high = mean_ci(diffs)
            interval = f"{avg:+.2f} [{low:+.2f}, {high:+.2f}]"
        rows.append(f"{previous.LABELS[method]} & {statistics.mean(dev):.2f} & "
                    f"{statistics.mean(test):.2f} $\\pm$ {statistics.stdev(test):.2f} & {interval} \\\\")
        if method != "baseline":
            delta = [
                100 * ((group[s]["dev"]["accuracy"] - baseline[s]["dev"]["accuracy"])
                       - (full[(method, 8)][s]["dev"]["accuracy"]
                          - full[("baseline", 8)][s]["dev"]["accuracy"]))
                for s in study.SEEDS
            ]
            avg, low, high = mean_ci(delta)
            paired.append(f"{previous.LABELS[method]} & {avg:+.2f} & [{low:+.2f}, {high:+.2f}] \\\\")
    duration_rows = []
    best_epoch_rows = []
    for epochs in (8, 10, 12):
        group = runs[("stack", epochs)]
        dev = [100 * group[s]["dev"]["accuracy"] for s in study.SEEDS]
        test = [100 * group[s]["test"]["accuracy"] for s in study.SEEDS]
        duration_rows.append(f"{epochs} & {statistics.mean(dev):.2f} $\\pm$ {statistics.stdev(dev):.2f} & "
                             f"{statistics.mean(test):.2f} $\\pm$ {statistics.stdev(test):.2f} \\\\")
        best = [max(group[s]["history"], key=lambda row: (
            row["dev_accuracy"], -row["dev_loss"], -row["epoch"])) for s in study.SEEDS]
        best_epoch_rows.append(
            f"{epochs} & {statistics.mean(100 * r['dev_accuracy'] for r in best):.2f} & "
            + ", ".join(str(r["epoch"]) for r in best) + r" \\"
        )
    seed_rows = [
        f"{seed} & " + " & ".join(
            f"{100 * runs[(method, 8)][seed]['test']['accuracy']:.2f}"
            for method in study.METHODS
        ) + r" \\"
        for seed in study.SEEDS
    ]
    timing_rows = []
    for method, epochs in study.CONDITIONS:
        seconds = [runs[(method, epochs)][s]["train_seconds"] for s in study.SEEDS]
        if any(not math.isfinite(value) or value < 0 for value in seconds):
            raise ValueError(f"Invalid training time for {method}, {epochs} epochs")
        timing_rows.append(f"{previous.LABELS[method]} & {epochs} & "
                           f"{statistics.mean(seconds):.1f} & {sum(seconds):.1f} \\\\")
    screen_seconds = sum(item["train_seconds"] for item in screening.values())
    study_seconds = sum(run["train_seconds"] for group in runs.values() for run in group.values())
    if not math.isfinite(screen_seconds) or screen_seconds < 0:
        raise ValueError("Invalid screening training time")
    return r"""
\section{Standard LoRA Follow-up}
The same DeBERTa-v3-base model and splits are used with standard LoRA \cite{lora}: rank 8, alpha 16, dropout 0.1, applied to query and value projections; the classifier and pooler remain trainable. This changes fine-tuning, not the underlying pretrained model. The learning rate was screened on the baseline at eight epochs using seed 42 and development accuracy, ties by lower loss and then lower learning rate. The chosen learning rate SELECTED_LR was fixed for every subsequent pipeline. This limited search tests learning rate only; the strengths of the regularizers and LoRA parameters were not retuned. The original full-fine-tuning search used seed 42 and candidates $1,2,3\times10^{-5}$.
\begin{table}[tp]
\centering\small\caption{LoRA learning-rate screen: seed 42 final development accuracy and loss.}
\begin{tabular}{lrr}\toprule
LR & Dev accuracy (\%) & Dev loss \\ \midrule
LR_ROWS
\bottomrule\end{tabular}
\end{table}
The 40 matched LoRA runs comprise six methods at eight epochs plus the full stack at 10 and 12 epochs, each across five seeds. The final epoch was evaluated immediately on the same previously inspected 374-example internal holdout; this is exploratory reuse, not an independent test. No model checkpoints were retained, so best-development-epoch weights cannot be tested. The screening seed and development data were reused and selection bias is possible.
\begin{table*}[tp]
\centering\scriptsize\setlength{\tabcolsep}{2pt}
\caption{Eight-epoch LoRA: final development and holdout accuracy (\%); paired holdout gain over LoRA baseline is in percentage points with a 95\% seed interval.}
\begin{tabular}{lrrr}\toprule
Method & Dev mean & Holdout mean $\pm$ SD & Gain [95\% CI] \\ \midrule
RESULT_ROWS
\bottomrule\end{tabular}
\end{table*}
\begin{table*}[tp]
\centering\small\setlength{\tabcolsep}{4pt}
\caption{Per-seed final-epoch LoRA accuracy (\%) on the reused internal holdout.}
\begin{tabular}{lrrrrrr}\toprule
Seed & Baseline & Mixup & R-Drop & NEFTune & Mixup + R-Drop & Full stack \\ \midrule
SEED_ROWS
\bottomrule\end{tabular}
\end{table*}
\begin{table}[tp]
\centering\small\setlength{\tabcolsep}{3pt}
\caption{Full LoRA stack: final-epoch accuracy (\%, mean $\pm$ SD). Schedules differ by duration.}
\begin{tabular}{rrr}\toprule
Epochs & Development & Holdout \\ \midrule
DURATION_ROWS
\bottomrule\end{tabular}
\end{table}
\begin{table}[tp]
\centering\small\setlength{\tabcolsep}{3pt}
\caption{Stack best observed development epochs (seed order 42, 123, 456, 789, 1011). These are selected on development data and are not independent test results.}
\begin{tabular}{rrl}\toprule
Schedule & Best mean (\%) & Selected epochs \\ \midrule
BEST_EPOCH_ROWS
\bottomrule\end{tabular}
\end{table}
To assess whether regularization helps more under LoRA, Table~\ref{tab:lora-interaction} reports paired differences in development gains: (LoRA method minus LoRA baseline) minus (full fine-tuning method minus full baseline), for matched seeds. The full runs used $3\times10^{-5}$, while LoRA used SELECTED_LR; if these differ, learning rate and fine-tuning method are confounded. These conditional seed intervals do not capture test-sample uncertainty, and repeated use of development data makes results exploratory.
\begin{table}[tp]
\centering\small\setlength{\tabcolsep}{3pt}
\caption{Difference in development gains: LoRA minus full fine-tuning, percentage points, 95\% seed intervals.}
\label{tab:lora-interaction}
\begin{tabular}{lrr}\toprule
Method & Mean & 95\% CI \\ \midrule
PAIRED_ROWS
\bottomrule\end{tabular}
\end{table}
\begin{figure*}[tp]
\centering\includegraphics[width=\textwidth,height=0.36\textheight,keepaspectratio]{lora_comparison.pdf}
\caption{Eight-epoch validation trajectories (five-seed means) and exploratory final holdout accuracies.}
\end{figure*}
LoRA environment: @DEVICE@; Python @PYTHON@; PyTorch @TORCH@; PEFT @PEFT@; Transformers @TRANSFORMERS@. Controller wall time for the final invocation was @WALL@ hours, excluding Slurm queue and any previous interrupted invocations. Test predictions and metrics are saved; model weights are not.
The three screening runs used @SCREEN_SECONDS@ seconds of recorded training-loop time; the 40 matched runs used @STUDY_SECONDS@ seconds. Table~\ref{tab:lora-time} breaks down the latter. Training-loop time excludes model loading, validation, holdout inference, report generation, and other overhead; it is not the same as end-to-end wall time.
\begin{table}[tp]
\centering\small\setlength{\tabcolsep}{3pt}
\caption{LoRA training-loop time by condition, five seeds; seconds.}
\label{tab:lora-time}
\begin{tabular}{lrrr}\toprule
Method & Epochs & Mean/run & Total \\ \midrule
TIMING_ROWS
\bottomrule\end{tabular}
\end{table}
""".replace("SELECTED_LR", f"{selection['selected_lr']:g}").replace(
        "LR_ROWS", "\n".join(lr_rows)).replace("RESULT_ROWS", "\n".join(rows)).replace(
        "SEED_ROWS", "\n".join(seed_rows)).replace("DURATION_ROWS", "\n".join(duration_rows)).replace(
        "BEST_EPOCH_ROWS", "\n".join(best_epoch_rows)).replace(
        "PAIRED_ROWS", "\n".join(paired)).replace(
        "@DEVICE@", original.latex_escape(environment["device"])).replace(
        "@PYTHON@", original.latex_escape(environment["python"])).replace(
        "@TORCH@", original.latex_escape(environment["torch"])).replace(
        "@PEFT@", original.latex_escape(environment["peft"])).replace(
        "@TRANSFORMERS@", original.latex_escape(environment["transformers"])).replace(
        "@WALL@", f"{environment['controller_wall_seconds']/3600:.2f}").replace(
        "@SCREEN_SECONDS@", f"{screen_seconds:.1f}").replace(
        "@STUDY_SECONDS@", f"{study_seconds:.1f}").replace("TIMING_ROWS", "\n".join(timing_rows))


def main():
    selection = json.loads((study.OUT / "selection.json").read_text())
    environment = json.loads((study.OUT / "environment.json").read_text())
    runs = load_runs(study.OUT, selection["selected_lr"])
    screening = {
        lr: json.loads((study.OUT / "screen" / f"lr{lr:g}" / "metrics.json").read_text())
        for lr in study.LRS
    }
    full = previous.load_runs()
    out = study.OUT / "report"
    out.mkdir(parents=True, exist_ok=True)
    source = study.ROOT / "outputs/followup_study/report"
    for file in ("aaai2026.sty", "aaai2026.bst", "learning_curves.pdf",
                 "paired_accuracy.pdf", "followup_curves.pdf"):
        shutil.copy2(source / file, out / file)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for method in study.METHODS:
        histories = [[r["dev_accuracy"] * 100 for r in runs[(method, 8)][s]["history"]]
                     for s in study.SEEDS]
        axes[0].plot(range(1, 9), np.mean(histories, axis=0), label=previous.LABELS[method])
    axes[0].legend(fontsize=7)
    axes[0].set(xlabel="Epoch", ylabel="Development accuracy (%)")
    names = [previous.LABELS[m] for m in study.METHODS]
    means = [statistics.mean(100*runs[(m, 8)][s]["test"]["accuracy"] for s in study.SEEDS)
             for m in study.METHODS]
    axes[1].barh(names, means)
    axes[1].set(xlabel="Internal holdout accuracy (%)", xlim=(0, 100))
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(out / f"lora_comparison.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    tex = (source / "report.tex").read_text()
    title = r"\title{DeBERTa-v3 RTE: An Epoch Study of Regularized Pipelines}"
    if tex.count(title) != 1 or tex.count(r"\begin{abstract}") != 1 or tex.count(r"\end{abstract}") != 1:
        raise ValueError("Original report has no unique title and abstract.")
    tex = tex.replace(title, r"\title{Regularization for DeBERTa-v3 on RTE: Full Fine-Tuning and LoRA}")
    start = tex.index(r"\begin{abstract}")
    end = tex.index(r"\end{abstract}", start) + len(r"\end{abstract}")
    tex = tex[:start] + combined_abstract(runs, full) + tex[end:]
    methods = r"\section{Regularization Methods}"
    if tex.count(methods) != 1:
        raise ValueError("Original report has no unique methods section.")
    tex = tex.replace(methods, study_overview(selection["selected_lr"]) + "\n" + methods)
    marker = r"\section{Interpretation and Limitations}"
    bibliography = r"\begin{thebibliography}{9}"
    if tex.count(marker) != 1 or tex.count(bibliography) != 1:
        raise ValueError("Original report has no unique interpretation section or bibliography.")
    start = tex.index(marker)
    end = tex.index(bibliography, start)
    tex = (tex[:start] + report_section(runs, full, selection, environment, screening)
           + "\n" + combined_conclusion(runs, full, selection) + "\n" + tex[end:])
    tex = tex.replace(bibliography, bibliography + "\n" +
                      r"\bibitem[Hu et~al.(2021)]{lora} Hu et al. 2021. LoRA: Low-Rank Adaptation of Large Language Models. \url{https://arxiv.org/abs/2106.09685}.")
    (out / "report.tex").write_text(tex)
    print(f"Report written to {out}", flush=True)


if __name__ == "__main__":
    main()
