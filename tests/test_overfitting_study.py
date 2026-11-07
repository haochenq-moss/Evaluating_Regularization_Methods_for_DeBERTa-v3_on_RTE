import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from transformers import DebertaV2Config, DebertaV2ForSequenceClassification

from regbench.config import Config, RunSpec
from regbench.trainer import run

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import report_overfitting_study as report
import run_overfitting_study as study


def test_clean_loss_is_opt_in_hash_changes_and_no_weights(tmp_path):
    data = Dataset.from_dict({"input_ids": [[1, 2], [3, 4], [5, 6], [7, 8]],
                              "attention_mask": [[1, 1]] * 4, "labels": [0, 1, 0, 1]})
    def builder(_):
        config = DebertaV2Config(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                 num_attention_heads=4, vocab_size=32, num_labels=2)
        return DebertaV2ForSequenceClassification(config)
    cfg = Config(epochs=1, batch_size=2, eval_batch_size=2, num_workers=0)
    spec = RunSpec({}, 1e-4, 42)
    args = dict(splits={"train": data, "dev": data}, pad_id=0, model_builder=builder,
                record_train_accuracy=True, save_resume=False)
    first = run(cfg, spec, tmp_path, **args)
    assert "train_clean_loss" not in first["history"][0]
    second = run(cfg, spec, tmp_path, record_train_clean_loss=True, **args)
    assert first["config_hash"] != second["config_hash"]
    assert second["history"][0]["train_clean_loss"] >= 0
    assert run(cfg, spec, tmp_path, record_train_clean_loss=True, **args) == second
    assert not list(tmp_path.glob("*.pt*"))
    assert not list(tmp_path.glob("predictions*"))


def test_selection_and_best_epoch():
    def values(acc, loss):
        return [{"dev": {"accuracy": acc, "loss": loss}}] * 3
    assert study.rank_screen({(8, .1): values(.8, .3), (4, 0.): values(.8, .3)}) == (4, 0.)
    with pytest.raises(ValueError, match="Incomplete"):
        study.rank_screen({(8, .1): values(.8, .3)[:2]})
    rows = [{"epoch": 1, "dev_accuracy": .8, "dev_loss": .5},
            {"epoch": 2, "dev_accuracy": .8, "dev_loss": .4},
            {"epoch": 3, "dev_accuracy": .8, "dev_loss": .4}]
    assert study.best_epoch(rows)["epoch"] == 2


@pytest.mark.parametrize("rank,dropout", [(4, 0.), (8, .1), (16, .1)])
def test_lora_configuration(monkeypatch, rank, dropout):
    config = DebertaV2Config(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                             num_attention_heads=4, vocab_size=128, num_labels=2)
    monkeypatch.setattr(study, "build_model", lambda _: DebertaV2ForSequenceClassification(config))
    model = study.build_lora(rank, dropout)(Config())
    adapter = model.peft_config["default"]
    assert (adapter.r, adapter.lora_alpha, adapter.lora_dropout) == (rank, 2 * rank, dropout)
    assert not model.get_input_embeddings().weight.requires_grad


def fake_run(seed, reg, lr, accuracy):
    history = [{"epoch": i, "train_accuracy": .9, "dev_accuracy": accuracy,
                "train_clean_loss": .25, "dev_loss": .5, "dev_n": 374}
               for i in range(1, study.EPOCHS + 1)]
    return {"seed": seed, "reg": reg, "lr": lr, "epochs": study.EPOCHS,
            "history": history, "dev": {"accuracy": accuracy, "loss": .5, "n": 374},
            "train_seconds": 2.0}


def test_full_phased_protocol_and_synthetic_report(tmp_path, monkeypatch):
    out = tmp_path / "study"
    monkeypatch.setattr(study, "OUT", out)
    monkeypatch.setattr(study.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(study.torch.cuda, "get_device_name", lambda: "synthetic")
    monkeypatch.setattr(study.Config, "load", lambda _: Config())
    monkeypatch.setattr(study, "load_splits", lambda _: {"train": range(1742), "dev": range(374)})
    monkeypatch.setattr(study.AutoTokenizer, "from_pretrained",
                        lambda _: type("Tokenizer", (), {"pad_token_id": 0})())
    calls = []
    def fake_trainer(cfg, spec, path, **kwargs):
        assert not cfg.save_final_model and not kwargs["save_resume"]
        assert kwargs["record_train_accuracy"] and kwargs["record_train_clean_loss"]
        assert set(kwargs["splits"]) == {"train", "dev"}
        calls.append((path, spec))
        value = fake_run(spec.seed, spec.reg, spec.lr, .8 + .01 * bool(spec.reg))
        study.write_json(path / "metrics.json", value)
        return value
    monkeypatch.setattr(study, "run", fake_trainer)
    study.main("adapter")
    assert len(calls) == 18 and not (out / "selection.json").exists()
    study.main("strength")
    assert len(calls) == 36 and (out / "selection.json").exists()
    study.main("compare")
    assert len(calls) == 56
    selected = json.loads((out / "selection.json").read_text())
    assert (selected["rank"], selected["dropout"], selected["full_strength"],
            selected["lora_strength"]) == (4, 0., .1, .1)
    for seed in study.SCREEN_SEEDS:
        identity = json.loads((out / "screen" / "lora_rdrop0.1" / f"seed{seed}"
                               / "study_config.json").read_text())
        assert (identity["rank"], identity["dropout"]) == (4, 0.)
    selection, runs, screen = report.collect(out)
    assert len(screen) == 12
    section = report.render_section(selection, runs, screen, {"controller_wall_seconds": 1800})
    assert "not an exhaustive joint search" in section
    assert "42 & 80.00 & 81.00 & +1.00" in section
    assert "no holdout" in section.lower()
    report.main()
    assert (out / "report" / "overfitting_curves.pdf").stat().st_size
    assert (out / "report" / "overfitting_gaps.png").stat().st_size
    tectonic = ROOT / "outputs/tools/tectonic"
    if tectonic.is_file():
        subprocess.run([str(tectonic), "--keep-logs", "report.tex"], cwd=out / "report",
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert (out / "report" / "report.pdf").stat().st_size
    identity_path = out / "runs" / "lora" / "baseline" / "seed42" / "study_config.json"
    identity = json.loads(identity_path.read_text())
    identity["rank"] = 16
    study.write_json(identity_path, identity)
    with pytest.raises(ValueError, match="adapter"):
        report.collect(out)
