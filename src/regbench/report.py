"""Aggregate final runs into tables and figures."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import reliability_bins, softmax

SPLITS = ("test",)


def _load(out: Path):
    runs = {}
    for mp in sorted((out / "final").glob("*/seed*/metrics.json")):
        runs.setdefault(mp.parent.parent.name, []).append((mp.parent, json.loads(mp.read_text())))
    return runs


def _ms(vals):
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)


def report(out: Path, ece_bins: int = 15):
    out = Path(out)
    runs = _load(out)
    if not runs:
        raise SystemExit("no final runs found; run `regbench final` first")
    rows = []
    for name, rs in runs.items():
        r0 = rs[0][1]
        row = {"config": name, "reg": json.dumps(r0["reg"]), "lr": r0["lr"], "seeds": len(rs)}
        for sp in SPLITS:
            for met in ("accuracy", "macro_f1", "ece"):
                m, s = _ms([r[sp][met] for _, r in rs])
                row[f"{sp}_{met}"], row[f"{sp}_{met}_std"] = m, s
        row["train_minutes"] = float(np.mean([r["train_seconds"] for _, r in rs])) / 60
        row["train_pflops_est"] = float(np.mean([r["train_flops_est"] for _, r in rs])) / 1e15
        rows.append(row)
    rep = out / "reports"
    rep.mkdir(exist_ok=True)
    with open(rep / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    for row in rows:
        row["validation_accuracy"] = float(np.mean([r["dev"]["accuracy"] for _, r in runs[row["config"]]]))
        row["validation_loss"] = float(np.mean([r["dev"]["loss"] for _, r in runs[row["config"]]]))

    pm = lambda r, k: f"{100 * r[k]:.2f}±{100 * r[k + '_std']:.2f}"
    lines = ["| Config | Hyper-params | LR | Val acc | Val loss | Test acc | Test F1 | Test ECE | min/run | PFLOPs (est.) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['config']} | `{r['reg']}` | {r['lr']:g} | " +
                     f"{100*r['validation_accuracy']:.2f} | {r['validation_loss']:.4f} | " +
                     " | ".join(pm(r, f"test_{m}") for m in ("accuracy", "macro_f1", "ece")) +
                     f" | {r['train_minutes']:.1f} | {r['train_pflops_est']:.2f} |")
    best = max(rows, key=lambda r: (r["validation_accuracy"], -r["validation_loss"]))
    base = next((r for r in rows if r["config"] == "baseline"), None)
    summary = ["# Results (mean±std over seeds; percentages except validation loss)\n", *lines, ""]
    if base:
        summary.append(f"Best config by validation accuracy (ties by validation loss): **{best['config']}**. "
                       f"Test accuracy difference vs baseline: "
                       f"{100*(best['test_accuracy']-base['test_accuracy']):+.2f} percentage points.")
    (rep / "results.md").write_text("\n".join(summary) + "\n")

    names = list(runs)
    fig, axes = plt.subplots(2, len(names), figsize=(3.2 * len(names), 6.4), squeeze=False)
    reliability_splits = ("dev", "test")
    for j, name in enumerate(names):
        for i, sp in enumerate(reliability_splits):
            lg, y = [], []
            for d, _ in runs[name]:
                z = np.load(d / f"predictions_{sp}.npz")
                lg.append(z["logits"]); y.append(z["labels"])
            probs, y = softmax(np.concatenate(lg).astype(np.float64)), np.concatenate(y)
            bc, ba, cnt = reliability_bins(probs, y, ece_bins)
            ax, nz = axes[i, j], cnt > 0
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.bar(bc[nz], ba[nz], width=1 / ece_bins, alpha=0.7, edgecolor="k")
            ax.set_title(f"{name[:26]}\n{sp}", fontsize=7)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            if j == 0:
                ax.set_ylabel("accuracy")
            if i == 1:
                ax.set_xlabel("confidence")
    fig.tight_layout(); fig.savefig(rep / "reliability_diagrams.png", dpi=150); plt.close(fig)

    individual_rows = [r for r in rows if not r["config"].startswith("stack")]
    stack_rows = [r for r in rows if r["config"].startswith("stack")]
    fig, ax = plt.subplots(figsize=(11, 5))
    for label, selected, marker in (
        ("Baseline / individual", individual_rows, "o"),
        ("Cumulative stacks", stack_rows, "s"),
    ):
        ax.errorbar(
            [r["config"] for r in selected],
            [100 * r["test_accuracy"] for r in selected],
            yerr=[100 * r["test_accuracy_std"] for r in selected],
            marker=marker, capsize=3, label=label,
        )
    ax.set_ylabel("Official RTE validation accuracy (%)")
    ax.tick_params(axis="x", labelrotation=55)
    ax.legend()
    fig.tight_layout()
    fig.savefig(rep / "individual_vs_stack_accuracy.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(rows))
    axes[0].bar(x, [100 * r["test_ece"] for r in rows], yerr=[100 * r["test_ece_std"] for r in rows],
                width=0.65, capsize=2)
    axes[0].set_ylabel("ECE (%)")
    axes[0].set_title("Official RTE validation calibration")
    axes[1].bar(x, [100 * r["test_accuracy"] for r in rows],
                yerr=[100 * r["test_accuracy_std"] for r in rows], width=0.65, capsize=2)
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Official RTE validation accuracy")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([r["config"][:20] for r in rows], rotation=60, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(rep / "ece_and_accuracy.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ("accuracy", "macro_f1", "ece")):
        ax.bar(
            x, [100 * row[f"test_{metric}"] for row in rows], width=0.65,
            yerr=[100 * row[f"test_{metric}_std"] for row in rows], capsize=2,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([row["config"][:18] for row in rows],
                           rotation=60, ha="right", fontsize=7)
        ax.set_title(f"Official validation {metric} (%)")
    fig.tight_layout()
    fig.savefig(rep / "test_metrics.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, run_items in runs.items():
        histories = [metrics.get("history", []) for _, metrics in run_items]
        if not any(histories):
            continue
        max_epoch = max(len(history) for history in histories)
        train_means, dev_means, epochs = [], [], []
        for i in range(max_epoch):
            available = [history[i] for history in histories if i < len(history)]
            epochs.append(i + 1)
            train_means.append(float(np.mean([item["train_loss"] for item in available])))
            dev_means.append(float(np.mean([item.get("dev_loss", np.nan) for item in available])))
        ax.plot(epochs, train_means, linestyle="--", alpha=0.75, label=f"{name} train")
        ax.plot(epochs, dev_means, label=f"{name} validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy / NLL")
    ax.set_title("Training and internal-validation loss")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(rep / "loss_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    has_train_accuracy = False
    for name, run_items in runs.items():
        histories = [metrics.get("history", []) for _, metrics in run_items]
        if not any(histories):
            continue
        max_epoch = max(len(history) for history in histories)
        epochs, train_means, dev_means = [], [], []
        for i in range(max_epoch):
            available = [history[i] for history in histories if i < len(history)]
            epochs.append(i + 1)
            dev_values = [item["dev_accuracy"] for item in available if "dev_accuracy" in item]
            train_values = [item["train_accuracy"] for item in available if "train_accuracy" in item]
            dev_means.append(float(np.mean(dev_values)) if dev_values else np.nan)
            train_means.append(float(np.mean(train_values)) if train_values else np.nan)
            has_train_accuracy = has_train_accuracy or bool(train_values)
        ax.plot(epochs, dev_means, label=f"{name} validation")
        if not np.isnan(train_means).all():
            ax.plot(epochs, train_means, linestyle="--", alpha=0.75, label=f"{name} train")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Internal-validation accuracy by epoch (train accuracy unavailable)")
    ax.yaxis.set_major_formatter(lambda value, _: f"{100 * value:.0f}%")
    if not has_train_accuracy:
        ax.text(
            0.5, 0.02, "Training accuracy was not recorded in the saved runs.",
            transform=ax.transAxes, ha="center", va="bottom",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray"},
        )
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(rep / "accuracy_curves.png", dpi=150)
    plt.close(fig)

    config_names = list(runs)
    fig, grid = plt.subplots(
        len(config_names), 2,
        figsize=(14, max(3.0 * len(config_names), 8)),
        squeeze=False,
    )
    for row_index, name in enumerate(config_names):
        run_items = runs[name]
        histories = [metrics.get("history", []) for _, metrics in run_items]
        max_epoch = max((len(history) for history in histories), default=0)
        epochs = np.arange(1, max_epoch + 1)
        train_loss, dev_loss, dev_accuracy = [], [], []
        for i in range(max_epoch):
            available = [history[i] for history in histories if i < len(history)]
            train_loss.append([item["train_loss"] for item in available if "train_loss" in item])
            dev_loss.append([item["dev_loss"] for item in available if "dev_loss" in item])
            dev_accuracy.append([item["dev_accuracy"] for item in available
                                 if "dev_accuracy" in item])

        def plot_seed_mean(ax, values, label, color, marker, linestyle="-"):
            means = np.array([np.mean(v) if v else np.nan for v in values], dtype=float)
            stds = np.array([np.std(v, ddof=1) if len(v) > 1 else 0.0 for v in values])
            ax.plot(epochs, means, color=color, marker=marker, linestyle=linestyle,
                    label=label, markersize=3)
            ax.fill_between(epochs, means - stds, means + stds, color=color, alpha=0.12)

        loss_ax, accuracy_ax = grid[row_index]
        plot_seed_mean(loss_ax, train_loss, "Training", "tab:blue", "o")
        plot_seed_mean(loss_ax, dev_loss, "Validation", "tab:red", "s", "--")
        accuracy_ax.plot(
            epochs,
            [np.mean(v) if v else np.nan for v in dev_accuracy],
            color="tab:red", marker="s", linestyle="--", markersize=3,
            label="Validation",
        )
        accuracy_ax.fill_between(
            epochs,
            [np.mean(v) - np.std(v, ddof=1) if len(v) > 1 else np.mean(v)
             for v in dev_accuracy],
            [np.mean(v) + np.std(v, ddof=1) if len(v) > 1 else np.mean(v)
             for v in dev_accuracy],
            color="tab:red", alpha=0.12,
        )

        run_metrics = run_items[0][1]
        reg_label = ", ".join(f"{key}={value:g}" for key, value in run_metrics["reg"].items())
        selection = f"{name} | {reg_label or 'no regularizer'} | LR={run_metrics['lr']:g}"
        loss_ax.set_title(f"{selection}\nLoss", fontsize=8)
        accuracy_ax.set_title("Accuracy (training accuracy not recorded)", fontsize=8)
        loss_ax.set_ylabel("Loss")
        accuracy_ax.set_ylabel("Accuracy (%)")
        accuracy_ax.yaxis.set_major_formatter(lambda value, _: f"{100 * value:.0f}%")
        for ax in (loss_ax, accuracy_ax):
            ax.set_xlabel("Epoch")
            ax.set_xticks(epochs)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7)
        accuracy_ax.set_ylim(0.45, 0.9)
    fig.suptitle(
        "Epoch curves for all selected hyperparameter configurations "
        "(mean ± standard deviation across seeds)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(rep / "epoch_curves_all_configs.png", dpi=160)
    plt.close(fig)
    return rows
