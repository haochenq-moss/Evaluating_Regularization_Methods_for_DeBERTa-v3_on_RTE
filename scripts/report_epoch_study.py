#!/usr/bin/env python3
"""Generate the epoch-study report from completed JSON outputs; never trains models."""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 456, 789, 1011)
T_CRITICAL = 2.776445105
METHODS = (("baseline", "Baseline"), ("rdrop", "R-Drop"), ("stack", "Selected stack"))
COLORS = {"baseline": "#0072B2", "rdrop": "#D55E00", "stack": "#009E73"}


def fail(message):
    raise ValueError(message)


def number(value, where, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        fail(f"{where} must be a finite number")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        fail(f"{where} is outside the permitted range")
    return float(value)


def validate(selection, final):
    """Reject incomplete matched runs before producing any output."""
    for name, obj in (("selection", selection), ("final_metrics", final)):
        if not isinstance(obj, dict):
            fail(f"{name} must be an object")
        seeds = obj.get("seeds")
        if not isinstance(seeds, list) or len(seeds) != len(SEEDS) or any(
            type(seed) is not int for seed in seeds
        ) or set(seeds) != set(SEEDS):
            fail(f"{name}.seeds must contain exactly {list(SEEDS)} once each")
        if type(obj.get("stack_epoch")) is not int or obj["stack_epoch"] not in (5, 6, 8):
            fail(f"{name}.stack_epoch must be one of 5, 6, 8")
    if selection["stack_epoch"] != final["stack_epoch"]:
        fail("selection and final_metrics stack_epoch disagree")

    def runs(container, key, required, label):
        value = container.get(key)
        if not isinstance(value, dict) or set(value) != set(required):
            fail(f"{label}.{key} must contain exactly {', '.join(required)}")
        for method in required:
            method_runs = value[method]
            if not isinstance(method_runs, dict) or set(method_runs) != {str(s) for s in SEEDS}:
                fail(f"{label}.{key}.{method} must contain all and only the five seeds")
        return value

    validation = runs(selection, "validation", ("baseline", "rdrop", "stack5", "stack6", "stack8"), "selection")
    holdout = runs(final, "runs", ("baseline", "rdrop", "stack"), "final_metrics")
    for method, method_runs in validation.items():
        expected_epochs = None
        for seed in SEEDS:
            where = f"selection.validation.{method}.{seed}"
            run = method_runs[str(seed)]
            if not isinstance(run, dict):
                fail(f"{where} must be an object")
            if "seed" in run and (type(run["seed"]) is not int or run["seed"] != seed):
                fail(f"{where}.seed disagrees with its key")
            if "lr" in run:
                number(run["lr"], f"{where}.lr", 0)
            history = run.get("history")
            if not isinstance(history, list) or not history:
                fail(f"{where}.history must be nonempty")
            epochs = []
            for entry in history:
                if not isinstance(entry, dict) or type(entry.get("epoch")) is not int:
                    fail(f"{where}.history contains an invalid epoch")
                epochs.append(entry["epoch"])
                for field in ("train_loss", "dev_loss"):
                    number(entry.get(field), f"{where}.{field}", 0)
                number(entry.get("dev_accuracy"), f"{where}.dev_accuracy", 0, 1)
                if "train_accuracy" in entry and entry["train_accuracy"] is not None:
                    number(entry["train_accuracy"], f"{where}.train_accuracy", 0, 1)
            if epochs != list(range(1, len(epochs) + 1)):
                fail(f"{where}.history epochs must be contiguous starting at 1")
            if expected_epochs is None:
                expected_epochs = epochs
            elif epochs != expected_epochs:
                fail(f"{where}.history epochs differ across seeds")
            if method.startswith("stack") and len(epochs) != int(method[5:]):
                fail(f"{where}.history length does not match {method}")
    for method, method_runs in holdout.items():
        for seed in SEEDS:
            where = f"final_metrics.runs.{method}.{seed}"
            metrics = method_runs[str(seed)]
            if not isinstance(metrics, dict):
                fail(f"{where} must be an object")
            if type(metrics.get("n")) is not int or metrics["n"] != 374:
                fail(f"{where}.n must be 374")
            for field in ("accuracy", "macro_f1", "ece"):
                number(metrics.get(field), f"{where}.{field}", 0, 1)
            number(metrics.get("loss"), f"{where}.loss", 0)
    return validation, holdout


def selected(validation, stack_epoch):
    return {key: validation[f"stack{stack_epoch}" if key == "stack" else key] for key, _ in METHODS}


def load_final_runs(runs_dir, validation, stack_epoch):
    """Load final-model trajectories without changing the frozen selection."""
    selection_runs = selected(validation, stack_epoch)
    result = {}
    for method, _ in METHODS:
        directory = f"stack{stack_epoch}" if method == "stack" else method
        expected_epochs = stack_epoch if method == "stack" else 5
        result[method] = {}
        for seed in SEEDS:
            path = runs_dir / directory / f"seed{seed}" / "metrics.json"
            try:
                run = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                fail(f"{path}: invalid JSON: {exc}")
            where = str(path)
            if not isinstance(run, dict):
                fail(f"{where} must contain an object")
            if type(run.get("seed")) is not int or run["seed"] != seed:
                fail(f"{where}: seed must be {seed}")
            if "epochs" in run and (type(run["epochs"]) is not int or run["epochs"] != expected_epochs):
                fail(f"{where}: epochs must be {expected_epochs}")
            history = run.get("history")
            if not isinstance(history, list) or len(history) != expected_epochs:
                fail(f"{where}: history must contain {expected_epochs} epochs")
            for index, entry in enumerate(history, 1):
                if not isinstance(entry, dict) or type(entry.get("epoch")) is not int or entry["epoch"] != index:
                    fail(f"{where}: history epochs must be contiguous starting at 1")
                for field in ("train_loss", "dev_loss"):
                    number(entry.get(field), f"{where}: epoch {index} {field}", 0)
                for field in ("train_accuracy", "dev_accuracy"):
                    number(entry.get(field), f"{where}: epoch {index} {field}", 0, 1)
            original = selection_runs[method][str(seed)]
            for field in ("lr", "reg"):
                if field in original and field in run:
                    if original[field] != run[field]:
                        fail(f"{where}: {field} differs from frozen selection for seed {seed}")
            result[method][str(seed)] = run
    return result


def replay_diagnostics(validation, final_runs, stack_epoch):
    originals = selected(validation, stack_epoch)
    result = {}
    for method, _ in METHODS:
        comparisons = {}
        for seed in SEEDS:
            original = originals[method][str(seed)]
            replay = final_runs[method][str(seed)]
            comparisons[seed] = 100 * abs(
                replay["history"][-1]["dev_accuracy"] - original["history"][-1]["dev_accuracy"]
            )
        checked = sum(
            field in originals[method][str(seed)] and field in final_runs[method][str(seed)]
            for seed in SEEDS for field in ("lr", "reg")
        )
        result[method] = (comparisons, checked)
    return result


def mean_sd(values):
    return statistics.mean(values), statistics.stdev(values)


def gain_stats(holdout, method):
    gains = [100 * (holdout[method][str(s)]["accuracy"] -
                    holdout["baseline"][str(s)]["accuracy"]) for s in SEEDS]
    mean, sd = mean_sd(gains)
    half_width = T_CRITICAL * sd / math.sqrt(len(SEEDS))
    return gains, mean, mean - half_width, mean + half_width


def learning_figure(runs):
    fig, axes = plt.subplots(3, 2, figsize=(7.1, 6.0), constrained_layout=True)
    fig.suptitle("Final evaluation run trajectories", fontsize=11)
    for row, (method, label) in enumerate(METHODS):
        histories = [runs[method][str(s)]["history"] for s in SEEDS]
        epochs = [entry["epoch"] for entry in histories[0]]
        for column, fields in enumerate((("train_loss", "dev_loss"), ("train_accuracy", "dev_accuracy"))):
            ax = axes[row, column]
            for field, style, text in ((fields[0], "--", "Train"), (fields[1], "-", "Internal dev")):
                means, sds = [], []
                for index in range(len(epochs)):
                    vals = [history[index].get(field) for history in histories]
                    if any(value is None for value in vals):
                        means.append(float("nan"))
                        sds.append(float("nan"))
                    else:
                        mean, sd = mean_sd(vals)
                        scale = 100 if column else 1
                        means.append(mean * scale)
                        sds.append(sd * scale)
                ax.plot(epochs, means, linestyle=style, marker="o", markersize=3,
                        color=COLORS[method], label=text)
                ax.fill_between(epochs, [m - s for m, s in zip(means, sds)],
                                [m + s for m, s in zip(means, sds)],
                                color=COLORS[method], alpha=.15)
                if field == "train_accuracy" and any(math.isnan(x) for x in means):
                    ax.text(.03, .04, "Train accuracy incomplete;\nno estimate at missing epochs",
                            transform=ax.transAxes, fontsize=7, va="bottom")
            ax.set_title(f"{label} — {'loss' if not column else 'accuracy'}", fontsize=9)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss" if not column else "Accuracy (%)")
            ax.set_xticks(epochs)
            ax.grid(alpha=.2)
            ax.legend(frameon=False, fontsize=7, loc="best")
    return fig


def paired_figure(holdout):
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.3), constrained_layout=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    for column, (method, label) in enumerate(METHODS[1:]):
        ax = axes[0, column]
        base = [100 * holdout["baseline"][str(s)]["accuracy"] for s in SEEDS]
        other = [100 * holdout[method][str(s)]["accuracy"] for s in SEEDS]
        for seed, a, b in zip(SEEDS, base, other):
            ax.plot([0, 1], [a, b], marker="o", markersize=4, alpha=.8, label=str(seed))
        ax.set_xticks([0, 1], ["Baseline", label])
        ax.set_ylabel("Internal holdout accuracy (%)")
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=.2)
        _, mean, low, high = gain_stats(holdout, method)
        gain_ax = axes[1, column]
        gain_ax.axvline(0, color="0.5", linestyle=":", linewidth=1)
        gain_ax.errorbar(mean, 0, xerr=[[mean - low], [high - mean]], fmt="o",
                         color=COLORS[method], capsize=3)
        gain_ax.set_yticks([])
        gain_ax.set_xlabel("Paired gain (percentage points)")
        gain_ax.set_title(f"{mean:+.2f} pp; 95% CI [{low:+.2f}, {high:+.2f}]", fontsize=8)
        gain_ax.grid(axis="x", alpha=.2)
    handles = [Line2D([0], [0], color=f"C{i}", marker="o", label=str(seed))
               for i, seed in enumerate(SEEDS)]
    axes[0, 0].legend(handles=handles, title="Seed", ncol=1, frameon=False, fontsize=7,
                      title_fontsize=7, loc="best")
    return fig


def latex_escape(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(char, char) for char in str(value))


def lr_description(method_runs):
    values = [method_runs[str(seed)].get("lr") for seed in SEEDS]
    if all(value is None for value in values):
        return "not recorded"
    if any(value is None for value in values):
        return "incomplete (not compared)"
    if len(set(values)) == 1:
        return f"{values[0]:.2g}"
    return "varies: " + ", ".join(f"{seed}={value:.2g}" for seed, value in zip(SEEDS, values))


def render_report(selection, validation, holdout, final_runs):
    chosen = selected(validation, selection["stack_epoch"])
    lr_rows = "\n".join(f"{latex_escape(label)} & 1, 2, 3 & {latex_escape(lr_description(chosen[key]))} \\\\"
                        for key, label in METHODS)
    rows = []
    for key, label in METHODS:
        cells = []
        for metric in ("accuracy", "macro_f1", "ece", "loss"):
            mean, sd = mean_sd([holdout[key][str(s)][metric] for s in SEEDS])
            if metric != "loss":
                mean, sd = 100 * mean, 100 * sd
            cells.append(f"{mean:.2f} $\\pm$ {sd:.2f}")
        rows.append(f"{latex_escape(label)} & " + " & ".join(cells) + r" \\")
    gain_rows = []
    interpretations = []
    for key, label in METHODS[1:]:
        _, mean, low, high = gain_stats(holdout, key)
        gain_rows.append(f"{latex_escape(label)} & {mean:+.2f} & [{low:+.2f}, {high:+.2f}] \\\\")
        crosses_zero = low <= 0 <= high
        interpretations.append(
            f"{latex_escape(label)} has a mean paired gain of {mean:+.2f} percentage points "
            f"({ 'the interval includes zero' if crosses_zero else 'the interval excludes zero' })."
        )
    development_rows = []
    for key, label in (("baseline", "Baseline"), ("rdrop", "R-Drop"),
                       ("stack5", "Stack, 5 epochs"), ("stack6", "Stack, 6 epochs"),
                       ("stack8", "Stack, 8 epochs")):
        histories = [validation[key][str(seed)]["history"] for seed in SEEDS]
        mean, sd = mean_sd([100 * history[-1]["dev_accuracy"] for history in histories])
        development_rows.append(
            f"{label} & {len(histories[0])} & {mean:.2f} $\\pm$ {sd:.2f} \\\\"
        )
    seed_rows = []
    for seed in SEEDS:
        base = 100 * holdout["baseline"][str(seed)]["accuracy"]
        rdrop = 100 * holdout["rdrop"][str(seed)]["accuracy"]
        stack = 100 * holdout["stack"][str(seed)]["accuracy"]
        seed_rows.append(f"{seed} & {base:.2f} & {rdrop:.2f} & {stack:.2f} \\\\")
    diagnostics = replay_diagnostics(validation, final_runs, selection["stack_epoch"])
    maxima = [max(diagnostics[method][0].values()) for method, _ in METHODS]
    drift_maxima = "; ".join(
        f"{latex_escape(label)}: {value:.3f}"
        for (_, label), value in zip(METHODS, maxima)
    )
    config_checks = "; ".join(
        f"{latex_escape(label)}: {diagnostics[method][1]}/10"
        for method, label in METHODS
    )
    stack_epoch = selection["stack_epoch"]
    baseline_mean = 100 * mean_sd([holdout["baseline"][str(s)]["accuracy"] for s in SEEDS])[0]
    stack_mean = 100 * mean_sd([holdout["stack"][str(s)]["accuracy"] for s in SEEDS])[0]
    _, stack_gain, stack_low, stack_high = gain_stats(holdout, "stack")
    abstract_results = (
        f"Baseline and stack achieved {baseline_mean:.2f}\\% and {stack_mean:.2f}\\% "
        f"mean accuracy, respectively; the stack's paired gain was {stack_gain:+.2f} percentage points "
        f"(95\\% seed interval [{stack_low:+.2f}, {stack_high:+.2f}])."
    )
    return r"""% Official AAAI-26 author kit: https://aaai.org/authorkit26-1/
% Place aaai2026.sty and aaai2026.bst beside this file. Run pdflatex twice.
% The article fallback used when aaai2026.sty is absent is NOT AAAI-compliant.
\IfFileExists{aaai2026.sty}{%
\documentclass[letterpaper]{article}
\usepackage[submission]{aaai2026}
\usepackage{times}
\usepackage{helvet}
\usepackage{courier}
\usepackage[hyphens]{url}
\urlstyle{rm}
\def\UrlFont{\rm}
\usepackage{natbib}
\usepackage{caption}
\frenchspacing
\setlength{\pdfpagewidth}{8.5in}
\setlength{\pdfpageheight}{11in}
\pdfinfo{/TemplateVersion (2026.1)}
\setcounter{secnumdepth}{2}
}{%
\documentclass[letterpaper,twocolumn]{article}
\usepackage[margin=0.75in]{geometry}
}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\title{DeBERTa-v3 RTE: An Epoch Study of Regularized Pipelines}
\author{Anonymous Submission}
\begin{document}
\maketitle
\begin{abstract}
We compare a baseline, R-Drop and a selected regularization stack on an internal RTE holdout across five matched training seeds. ABSTRACT_RESULTS The interval is conditional on the fixed holdout and includes zero when its endpoints straddle zero. Learning rates and training durations differ, limiting causal interpretation.
\end{abstract}
\section{Introduction and Related Work}
Fine-tuning large pretrained models on small datasets can produce overfitting and seed-dependent outcomes. We study whether regularization is associated with more accurate and better-calibrated predictions on RTE, a natural language inference task in GLUE \cite{glue}, using DeBERTaV3 \cite{deberta}. We compare a baseline with R-Drop \cite{rdrop} and a combined pipeline using mixup \cite{mixup}, R-Drop and NEFTune \cite{neftune}. We also document the earlier searches over label smoothing \cite{smoothing} and FLAN-T5 augmentation \cite{flan}. Our primary questions are how the pipelines compare across matched seeds and how training duration affects development accuracy. These comparisons do not by themselves establish the isolated effect of each regularizer.
\section{Experimental Setup}
All runs fine-tune DeBERTa-v3-base on RTE with AdamW (batch size 32, weight decay 0.01, warmup ratio 0.06). The data comprise 1,742 internal training, 374 internal development and 374 internal holdout examples. The pipelines are: baseline (learning rate $2\times10^{-5}$); R-Drop with coefficient 0.5 (learning rate $3\times10^{-5}$); and the selected stack with mixup 0.4, R-Drop 0.5 and NEFTune 15 (learning rate $3\times10^{-5}$). The stack epoch is the epoch with the highest mean development accuracy across seeds, with ties broken by lower mean development loss. Training accuracy, where available, is clean eval-mode inference over the full training set; it is distinct from the stochastic training loss, which is computed during training with dropout and method-specific perturbations.

\section{Regularization Methods}
The earlier benchmark considered the following five methods separately and in cumulative stacks. The present five-seed holdout comparison includes only the baseline, R-Drop, and the mixup--R-Drop--NEFTune stack; it does not separately retest augmentation or label smoothing.

\subsection{Data Augmentation}
FLAN-T5 generates a paraphrased or minimally edited hypothesis for an internal training premise, targeting either the original or the opposite entailment label. A selected baseline retains candidates only when it predicts the target label with confidence at least 0.85. Retained pairs are appended to training only. The augmentation ratio specifies the requested synthetic count as a fraction of original training size; the available filtered pool can cap the achieved ratio. Synthetic labels and model filtering can still introduce errors or bias.

\subsection{Label Smoothing}
Label smoothing replaces a one-hot target $y$ with $(1-\epsilon)y+\epsilon/K$, where $K=2$ classes. This reduces the target probability assigned to the annotated class, discouraging extreme confidence. The earlier individual winner used $\epsilon=0.10$; label smoothing is not used in the three pipelines of the present study.

\subsection{Embedding Mixup}
Mixup forms interpolated embeddings and soft labels for pairs of examples using a batch permutation and a coefficient sampled from $\operatorname{Beta}(\alpha,\alpha)$. The corresponding attention masks are combined by union. The selected stack uses $\alpha=0.4$. This is an embedding-space intervention, not the creation of new text examples.

\subsection{R-Drop}
R-Drop performs two stochastic forward passes with independent dropout masks. The objective averages the two cross-entropy terms and adds the symmetric KL divergence between their output distributions, scaled by a coefficient. Both the individual R-Drop pipeline and the selected stack use coefficient 0.5. When combined with mixup and NEFTune, both passes share the same mixed, perturbed embeddings.

\subsection{NEFTune}
NEFTune adds training-only uniform noise to token embeddings. In this implementation, noise magnitude is scaled by $\alpha/\sqrt{Ld}$, where $L$ is the number of unmasked tokens and $d$ is the embedding dimension; padding positions are not perturbed. The selected stack uses $\alpha=15$. Evaluation uses unperturbed inputs.

\section{Hyperparameter Selection}
The original single-seed search used seed 42 and internal development accuracy, breaking ties by lower development loss. Baseline learning rate and epoch count were jointly searched; all individual methods then used the baseline-selected five epochs. Each individual method was searched over the three learning rates and its three strengths shown in Table~\ref{tab:grid}. Methods were ranked by their single-seed development results. In the cumulative stack, the top method's chosen strength was fixed, and each newly added method's strength and the learning rate were tuned while earlier strengths remained fixed. Thus the original search was not a joint search over all combinations.

\begin{table}[tp]
\centering\small\setlength{\tabcolsep}{3pt}
\caption{Original search candidates and individual winners, using internal development data and seed 42. Augmentation ratios are requested fractions; availability may cap them.}
\label{tab:grid}
\begin{tabular}{llll}\toprule
Method & Candidates & Pick & LR \\ \midrule
Baseline & 3, 4, 5 epochs & 5 epochs & 2e-5 \\
Augmentation & 0.25, 0.50, 1.00 & 0.25 & 3e-5 \\
Label smoothing & 0.05, 0.10, 0.15 & 0.10 & 3e-5 \\
Mixup & 0.1, 0.2, 0.4 & 0.4 & 3e-5 \\
R-Drop & 0.5, 1.0, 4.0 & 0.5 & 3e-5 \\
NEFTune & 5, 10, 15 & 15 & 3e-5 \\
\bottomrule\end{tabular}
\end{table}
The original learning-rate candidates were $\{1,2,3\}\times10^{-5}$. The ranked order was mixup, R-Drop, NEFTune, augmentation, then label smoothing. The selected three-method stack used mixup 0.4, R-Drop 0.5, NEFTune 15, and learning rate $3\times10^{-5}$.

The subsequent five-seed epoch study held these selected strengths and learning rates fixed. It compared stack durations of 5, 6, and 8 epochs using mean development accuracy across seeds, with lower mean development loss breaking ties; baseline and R-Drop remained at five epochs. The comparison in Table~\ref{tab:dev} selected STACK_EPOCH epochs for the stack. These are final-epoch evaluations of separately scheduled runs, not checkpoints from a single longer run. The same development set informed both stages, and the original single-seed tuning and selected learning rates can confound conclusions about individual regularizers.

\section{Design and Selection}
The selected stack was trained for STACK_EPOCH epochs, chosen among 5, 6 and 8 epochs using internal development data. The same development set was used for selection and evaluation during selection, so development performance is optimistic and is not an independent estimate. The official RTE validation split was previously examined; it is not the final test in this study. The final test here is the internal holdout of 374 rows. Each method uses matched seeds 42, 123, 456, 789 and 1011.

Learning rates from the selection-run metrics are shown below; differences can confound method comparisons. Missing values are not inferred.
\begin{table}[t]
\centering\small\setlength{\tabcolsep}{3pt}
\caption{Original LR candidates ($\times 10^{-5}$) and selected rates; held fixed in the five-seed study.}
\begin{tabular}{lcc}\toprule
Pipeline & Candidates ($\times 10^{-5}$) & Selected \\ \midrule
LR_ROWS
\bottomrule\end{tabular}
\end{table}
Table~\ref{tab:dev} reports the final-epoch development accuracy for each candidate, not an independent test and not necessarily the complete selection rule.
\begin{table}[t]
\centering\small
\caption{Final-epoch internal development accuracy, mean $\pm$ SD (\%).}
\label{tab:dev}
\begin{tabular}{lrr}\toprule
Pipeline & Epoch & Accuracy \\ \midrule
DEV_ROWS
\bottomrule\end{tabular}
\end{table}

\section{Learning Dynamics}
Figure~\ref{fig:curves} shows final evaluation run trajectories, not the historical selection histories: the historical five-epoch seeds were retrained after the choice was frozen, and the curves use the replay trajectories. Each point is the mean of all five final runs and shading is one sample standard deviation. Historical selection runs lacked training accuracy; no values are imputed. Training loss is the method-specific training objective and is not directly comparable across regularizers; development loss uses the reported evaluation loss.
\begin{figure*}[tp]
\centering\includegraphics[width=\textwidth,height=0.68\textheight,keepaspectratio]{learning_curves.pdf}
\caption{Final evaluation run training and internal development curves; shading denotes $\pm$ one sample standard deviation across five seeds. Historical five-epoch runs are represented by their post-freeze replay trajectories, not their selection histories.}
\label{fig:curves}
\end{figure*}

The maximum absolute final-epoch development accuracy differences between frozen selection records and corresponding final runs were DRIFT_MAXIMA percentage points, respectively. These differences indicate reproduction drift, not a change to the frozen choice. The report checks learning rate and regularizer settings against the corresponding selection record when both values are present: CONFIG_CHECKS comparisons available out of ten per pipeline. A missing setting cannot be checked and is not inferred.

\section{Internal Holdout Results}
Table~\ref{tab:seeds} lists the final test accuracy for each seed on the internal holdout of 374 examples. Percentages are rounded to two decimal places. The official RTE validation split was previously examined and is not the test set used here.
\begin{table}[t]
\centering\small\setlength{\tabcolsep}{3pt}
\caption{Per-seed internal holdout test accuracy (\%); $n=374$ for every run.}
\label{tab:seeds}
\begin{tabular}{lrrr}\toprule
Seed & Baseline & R-Drop & Stack \\ \midrule
SEED_ROWS
\bottomrule\end{tabular}
\end{table}

Table~\ref{tab:metrics} reports mean $\pm$ sample standard deviation. Accuracy, macro-F1 and ECE are percentages; loss is unscaled. Table~\ref{tab:gains} and Figure~\ref{fig:paired} give paired accuracy differences in percentage points relative to the baseline. The 95\% interval is $\bar d \pm 2.776445105\,s_d/\sqrt{5}$, a two-sided paired $t$ interval with four degrees of freedom. It describes variation across training seeds conditional on the same 374-row holdout, not uncertainty from sampling new test examples.
\begin{table}[!htbp]
\centering\small
\caption{Paired holdout accuracy gain over baseline (percentage points).}
\label{tab:gains}
\begin{tabular}{lrr}\toprule
Pipeline & Mean & 95\% CI \\ \midrule
GAIN_ROWS
\bottomrule\end{tabular}
\end{table}
\begin{table*}[tp]
\centering\small\setlength{\tabcolsep}{4pt}
\caption{Internal holdout metrics, mean $\pm$ SD over five seeds.}
\label{tab:metrics}
\begin{tabular}{lrrrr}\toprule
Pipeline & Accuracy (\%) & Macro-F1 (\%) & ECE (\%) & Loss \\ \midrule
METRIC_ROWS
\bottomrule\end{tabular}

\end{table*}
Table~\ref{tab:seeds} lists the seed-by-seed accuracies underlying Figure~\ref{fig:paired}.
\begin{figure*}[tp]
\centering\includegraphics[width=\textwidth,height=0.55\textheight,keepaspectratio]{paired_accuracy.pdf}
\caption{Matched-seed internal holdout accuracy and paired gain intervals. Lines connect the same seed; the interval describes seed variation conditional on this holdout.}
\label{fig:paired}
\end{figure*}

\section{Interpretation and Limitations}
INTERPRETATION
These comparisons are observational, not causal: learning rates and method-specific objectives differ, and the same development data informed selection. The official validation set has also been previously examined, and the internal holdout is a single fixed set rather than independent holdouts per seed. The intervals quantify seed variation only, not a population-level test-set sampling interval.
\begin{thebibliography}{9}
\bibitem[He et~al.(2021)]{deberta} He et al. 2021. DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. \url{https://arxiv.org/abs/2111.09543}.
\bibitem[Wang et~al.(2018)]{glue} Wang et al. 2018. GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding. \url{https://arxiv.org/abs/1804.07461}.
\bibitem[Zhang et~al.(2018)]{mixup} Zhang et al. 2018. mixup: Beyond Empirical Risk Minimization. \url{https://arxiv.org/abs/1710.09412}.
\bibitem[Liang et~al.(2021)]{rdrop} Liang et al. 2021. R-Drop: Regularized Dropout for Neural Networks. \url{https://arxiv.org/abs/2106.14448}.
\bibitem[Jain et~al.(2023)]{neftune} Jain et al. 2023. NEFTune: Noisy Embeddings Improve Instruction Finetuning. \url{https://arxiv.org/abs/2310.05914}.
\bibitem[M\"uller et~al.(2019)]{smoothing} M\"uller et al. 2019. When Does Label Smoothing Help? \url{https://arxiv.org/abs/1906.02629}.
\bibitem[Chung et~al.(2022)]{flan} Chung et al. 2022. Scaling Instruction-Finetuned Language Models. \url{https://arxiv.org/abs/2210.11416}.
\end{thebibliography}
\end{document}
""".replace("ABSTRACT_RESULTS", abstract_results).replace("CONFIG_CHECKS", config_checks).replace("DRIFT_MAXIMA", drift_maxima).replace("SEED_ROWS", "\n".join(seed_rows)).replace("STACK_EPOCH", str(stack_epoch)).replace("LR_ROWS", lr_rows).replace(
        "DEV_ROWS", "\n".join(development_rows)).replace("METRIC_ROWS", "\n".join(rows)).replace(
        "GAIN_ROWS", "\n".join(gain_rows)).replace("INTERPRETATION", " ".join(interpretations))


def generate(selection, final, out, runs_dir):
    validation, holdout = validate(selection, final)
    final_runs = load_final_runs(runs_dir, validation, selection["stack_epoch"])
    report = render_report(selection, validation, holdout, final_runs)
    curves = learning_figure(final_runs)
    paired = paired_figure(holdout)
    out.mkdir(parents=True, exist_ok=True)
    for name, fig in (("learning_curves", curves), ("paired_accuracy", paired)):
        for extension in ("pdf", "png"):
            fig.savefig(out / f"{name}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    (out / "report.tex").write_text(report, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=ROOT / "outputs/epoch_study/selection.json")
    parser.add_argument("--final-metrics", type=Path, default=ROOT / "outputs/epoch_study/final_metrics.json")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "outputs/epoch_study/runs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/epoch_study/report")
    args = parser.parse_args(argv)
    try:
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        final = json.loads(args.final_metrics.read_text(encoding="utf-8"))
        generate(selection, final, args.output_dir, args.runs_dir)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"report generation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Report written to {args.output_dir}")
    print("Compile report.tex twice from that directory with pdflatex. Copy aaai2026.sty and")
    print("aaai2026.bst from the official kit https://aaai.org/authorkit26-1/ into it.")
    print("Without aaai2026.sty, the two-column article fallback is not AAAI-compliant.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
