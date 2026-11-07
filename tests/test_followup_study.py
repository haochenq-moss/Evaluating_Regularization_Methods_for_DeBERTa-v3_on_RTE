import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import report_followup_study as report
import run_followup_study as study


def test_study_design_and_best_epoch():
    assert len(study.CONDITIONS) * len(study.SEEDS) == 40
    assert set(study.METHODS) == {"baseline", "mixup", "rdrop", "neftune", "mixup_rdrop", "stack"}
    assert study.best_epoch([
        {"epoch": 1, "dev_accuracy": .8, "dev_loss": .5},
        {"epoch": 2, "dev_accuracy": .8, "dev_loss": .4},
        {"epoch": 3, "dev_accuracy": .8, "dev_loss": .4},
    ])["epoch"] == 2
    with pytest.raises(ValueError):
        study.best_epoch([])


def test_report_uses_all_matched_runs_and_rejects_bad_inputs(tmp_path):
    for method, epochs in study.CONDITIONS:
        for i, seed in enumerate(study.SEEDS):
            history = [
                {"epoch": epoch, "train_loss": .3, "train_accuracy": .9,
                 "dev_loss": .5 - epoch * .001, "dev_accuracy": .7 + i * .01,
                 "dev_n": 374}
                for epoch in range(1, epochs + 1)
            ]
            run = {"seed": seed, "epochs": epochs, "lr": study.LR, "reg": study.METHODS[method],
                   "history": history, "dev": {"accuracy": history[-1]["dev_accuracy"], "n": 374},
                   "train_seconds": 10}
            directory = tmp_path / "runs" / f"{method}_{epochs}" / f"seed{seed}"
            directory.mkdir(parents=True)
            (directory / "metrics.json").write_text(json.dumps(run))
    runs = report.load_runs(tmp_path)
    assert len(runs) == 8
    assert "Baseline & 72.00" in report.summary_rows(runs)
    assert "+0.00 [+0.00, +0.00]" in report.summary_rows(runs)
    assert "Full stack & 12 & 72.00 & 72.00 & 12, 12, 12, 12, 12" in report.epoch_rows(runs)
    report.create_figure(runs, tmp_path)
    assert (tmp_path / "followup_curves.pdf").read_bytes().startswith(b"%PDF")
    path = tmp_path / "runs" / "baseline_8" / "seed42" / "metrics.json"
    bad = json.loads(path.read_text())
    bad["history"][0]["dev_accuracy"] = float("nan")
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="Invalid dev_accuracy"):
        report.load_runs(tmp_path)


def test_disagreement_counts_use_predictions(tmp_path):
    for method, pred in (("baseline", [0, 0, 1, 1]),
                         ("rdrop", [1, 0, 0, 1]),
                         ("stack8", [0, 1, 1, 0])):
        directory = tmp_path / "outputs/epoch_study/runs" / method / "seed789"
        directory.mkdir(parents=True)
        logits = np.eye(2)[pred]
        np.savez(directory / "predictions_holdout.npz", labels=[0, 1, 1, 0], logits=logits)
    rows = report.disagreements(tmp_path)
    assert r"R-Drop & 2 & 2 & 0 \\" in rows
    assert r"Full stack & 2 & 0 & 2 \\" in rows
