import importlib.util
import io
import json
import shutil
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "report_epoch_study.py"
spec = importlib.util.spec_from_file_location("report_epoch_study", SCRIPT)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def synthetic():
    selection = {"seeds": list(report.SEEDS), "stack_epoch": 6, "validation": {}}
    for method, epochs in (("baseline", 5), ("rdrop", 5), ("stack5", 5), ("stack6", 6), ("stack8", 8)):
        runs = {}
        for index, seed in enumerate(report.SEEDS):
            history = [
                {"epoch": epoch, "train_loss": .7 / epoch, "dev_loss": .5 / epoch,
                 "dev_accuracy": .7 + index * .01, **({"train_accuracy": .8} if method == "rdrop" else {})}
                for epoch in range(1, epochs + 1)
            ]
            runs[str(seed)] = {"seed": seed, "lr": 2e-5 if method == "baseline" else 3e-5,
                               "history": history}
        selection["validation"][method] = runs
    final = {"seeds": list(report.SEEDS), "stack_epoch": 6, "runs": {}}
    for method, offset in (("baseline", 0), ("rdrop", .02), ("stack", .01)):
        final["runs"][method] = {
            str(seed): {"accuracy": .7 + i * .01 + offset * (i + 1), "macro_f1": .7,
                        "ece": .1, "loss": .4, "n": 374}
            for i, seed in enumerate(report.SEEDS)
        }
    return selection, final


@pytest.fixture
def runs_dir():
    path = SCRIPT.parents[1] / "tests" / f"epoch_report_fixture_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def write_final_runs(path, selection):
    for key in ("baseline", "rdrop", f"stack{selection['stack_epoch']}"):
        for seed in report.SEEDS:
            original = selection["validation"][key][str(seed)]
            run = json.loads(json.dumps(original))
            run["reg"] = {"rdrop": .5} if key == "rdrop" else {}
            for entry in run["history"]:
                entry["train_accuracy"] = .9
                entry["dev_accuracy"] += .03
            run["epochs"] = len(run["history"])
            file = path / key / f"seed{seed}" / "metrics.json"
            file.parent.mkdir(parents=True)
            file.write_text(json.dumps(run))


def test_synthetic_generation_and_intervals(runs_dir):
    selection, final = synthetic()
    write_final_runs(runs_dir, selection)
    frozen = json.dumps(selection, sort_keys=True)
    _, mean, low, high = report.gain_stats(final["runs"], "rdrop")
    assert mean == pytest.approx(6)
    assert low == pytest.approx(6 - report.T_CRITICAL * (10 ** .5) / (5 ** .5))
    assert high == pytest.approx(6 + report.T_CRITICAL * (10 ** .5) / (5 ** .5))
    rendered = {}

    def write_text(path, text, **kwargs):
        rendered[path.name] = text
        return len(text)

    from matplotlib.figure import Figure
    original_savefig = Figure.savefig

    def real_savefig_to_memory(figure, path, **kwargs):
        image = io.BytesIO()
        original_savefig(figure, image, format=path.suffix[1:], **kwargs)
        rendered[path.name] = image.getvalue()

    with patch.object(Path, "mkdir"), patch.object(Path, "write_text", write_text), \
            patch.object(Figure, "savefig", real_savefig_to_memory):
        report.generate(selection, final, Path("synthetic-report"), runs_dir)
    assert json.dumps(selection, sort_keys=True) == frozen
    assert set(rendered) == {"learning_curves.pdf", "learning_curves.png",
                             "paired_accuracy.pdf", "paired_accuracy.png", "report.tex"}
    assert rendered["learning_curves.pdf"].startswith(b"%PDF")
    assert rendered["paired_accuracy.png"].startswith(b"\x89PNG")
    tex = rendered["report.tex"]
    assert r"\IfFileExists{aaai2026.sty}" in tex and r"\usepackage[submission]{aaai2026}" in tex
    assert "aaai26.sty" not in tex and "authorkit26/" not in tex
    assert "https://aaai.org/authorkit26-1/" in tex
    assert r"\documentclass[letterpaper,twocolumn]{article}" in tex
    assert "conditional on the same 374-row holdout" in tex
    assert "not causal" in tex and "previously examined" in tex
    assert "2e-05" in tex and "3e-05" in tex
    assert r"Pipeline & Candidates ($\times 10^{-5}$) & Selected" in tex
    assert r"Baseline & 1, 2, 3 & 2e-05 \\" in tex
    assert r"R-Drop & 1, 2, 3 & 3e-05 \\" in tex
    assert r"Selected stack & 1, 2, 3 & 3e-05 \\" in tex
    assert "+6.00" in tex
    for text in ("1,742 internal training", "374 internal development", "374 internal holdout",
                 "batch size 32", "weight decay 0.01", "warmup ratio 0.06", "mixup 0.4",
                 "NEFTune 15", "lower mean development loss", "eval-mode inference"):
        assert text in tex
    for seed, baseline, rdrop, stack in (
        (42, 70, 72, 71), (123, 71, 75, 73), (456, 72, 78, 75),
        (789, 73, 81, 77), (1011, 74, 84, 79),
    ):
        assert f"{seed} & {baseline:.2f} & {rdrop:.2f} & {stack:.2f}" in tex
    assert r"Per-seed internal holdout test accuracy (\%)" in tex
    assert "tab:seeds" in tex
    assert "post-freeze replay trajectories" in tex
    assert "Baseline: 3.000; R-Drop: 3.000; Selected stack: 3.000" in tex
    assert r"height=0.68\textheight,keepaspectratio" in tex
    assert r"height=0.55\textheight,keepaspectratio" in tex
    assert "Baseline: 5/10" in tex
    for text in (r"\section{Regularization Methods}", r"\subsection{Data Augmentation}",
                 r"\subsection{Label Smoothing}", r"\subsection{Embedding Mixup}",
                 r"\subsection{R-Drop}", r"\subsection{NEFTune}",
                 r"\section{Hyperparameter Selection}", r"\label{tab:grid}",
                 "original single-seed search used seed 42",
                 "compared stack durations of 5, 6, and 8 epochs",
                 "The comparison in Table~\\ref{tab:dev} selected 6 epochs"):
        assert text in tex


@pytest.mark.parametrize("corrupt", [
    lambda s, f: s["seeds"].pop(),
    lambda s, f: f["runs"]["stack"].pop("42"),
    lambda s, f: f.update(stack_epoch=8),
    lambda s, f: f["runs"]["rdrop"]["42"].update(n=277),
    lambda s, f: s["validation"]["stack6"]["42"]["history"].pop(),
    lambda s, f: s["validation"]["baseline"]["42"]["history"][0].update(dev_accuracy=float("nan")),
])
def test_reject_incomplete_or_inconsistent(corrupt):
    selection, final = synthetic()
    corrupt(selection, final)
    with pytest.raises(ValueError):
        report.validate(selection, final)


def test_missing_training_accuracy_never_estimated():
    selection, final = synthetic()
    validation, _ = report.validate(selection, final)
    figure = report.learning_figure(report.selected(validation, 6))
    assert all(value != value for value in figure.axes[1].lines[0].get_ydata())
    assert "incomplete" in figure.axes[1].texts[0].get_text()
    report.plt.close(figure)
    assert report.latex_escape(r"a_b & c%") == r"a\_b \& c\%"


def test_final_runs_drive_all_curves_and_selection_stays_frozen(runs_dir):
    selection, final = synthetic()
    write_final_runs(runs_dir, selection)
    validation, holdout = report.validate(selection, final)
    runs = report.load_final_runs(runs_dir, validation, 6)
    figure = report.learning_figure(runs)
    for row in range(3):
        assert figure.axes[row * 2 + 1].lines[0].get_ydata()[0] == pytest.approx(90)
        assert figure.axes[row * 2 + 1].lines[1].get_ydata()[0] == pytest.approx(75)
        assert figure.axes[row * 2].lines[0].get_ydata()[0] == pytest.approx(.7)
        assert figure.axes[row * 2].lines[1].get_ydata()[0] == pytest.approx(.5)
    report.plt.close(figure)
    tex = report.render_report(selection, validation, holdout, runs)
    assert "Baseline & 5 & 72.00" in tex
    assert "42 & 70.00 & 72.00 & 71.00" in tex
    assert "Baseline: 3.000; R-Drop: 3.000; Selected stack: 3.000" in tex


@pytest.mark.parametrize("epoch", [5, 6, 8])
def test_cli_reads_selected_stack_and_preserves_input_files(runs_dir, epoch):
    selection, final = synthetic()
    selection["stack_epoch"] = final["stack_epoch"] = epoch
    write_final_runs(runs_dir, selection)
    selection_path = runs_dir / "selection.json"
    final_path = runs_dir / "final_metrics.json"
    selection_path.write_text(json.dumps(selection))
    final_path.write_text(json.dumps(final))
    original = selection_path.read_bytes()
    out = runs_dir / "report"
    assert report.main(["--selection", str(selection_path), "--final-metrics", str(final_path),
                        "--runs-dir", str(runs_dir), "--output-dir", str(out)]) == 0
    assert selection_path.read_bytes() == original
    assert (out / "learning_curves.pdf").read_bytes().startswith(b"%PDF")
    assert (out / "paired_accuracy.png").read_bytes().startswith(b"\x89PNG")
    assert f"trained for {epoch} epochs" in (out / "report.tex").read_text()


def test_drift_reports_maximum_and_zero(runs_dir):
    selection, final = synthetic()
    write_final_runs(runs_dir, selection)
    path = runs_dir / "baseline" / "seed42" / "metrics.json"
    run = json.loads(path.read_text())
    run["history"][-1]["dev_accuracy"] = selection["validation"]["baseline"]["42"]["history"][-1]["dev_accuracy"]
    path.write_text(json.dumps(run))
    validation, holdout = report.validate(selection, final)
    runs = report.load_final_runs(runs_dir, validation, 6)
    tex = report.render_report(selection, validation, holdout, runs)
    assert "Baseline: 3.000; R-Drop: 3.000; Selected stack: 3.000" in tex
    assert r"\label{tab:drift}" not in tex


@pytest.mark.parametrize("corrupt", ["missing", "seed", "epochs", "history", "train_accuracy", "lr", "reg"])
def test_reject_bad_final_runs(runs_dir, corrupt):
    selection, final = synthetic()
    selection["validation"]["baseline"]["42"]["reg"] = {}
    write_final_runs(runs_dir, selection)
    path = runs_dir / "baseline" / "seed42" / "metrics.json"
    run = json.loads(path.read_text())
    if corrupt == "missing":
        path.unlink()
    else:
        if corrupt == "seed":
            run["seed"] = 123
        elif corrupt == "epochs":
            run["epochs"] = 6
        elif corrupt == "history":
            run["history"].pop()
        elif corrupt == "train_accuracy":
            run["history"][0].pop("train_accuracy")
        elif corrupt == "lr":
            run["lr"] = .5
        elif corrupt == "reg":
            run["reg"] = {"rdrop": .5}
        path.write_text(json.dumps(run))
    with pytest.raises((ValueError, OSError)):
        report.generate(selection, final, runs_dir / "report", runs_dir)
    assert not (runs_dir / "report").exists()
