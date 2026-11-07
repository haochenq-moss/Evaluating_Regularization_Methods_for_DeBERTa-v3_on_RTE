import json

import numpy as np
import pytest
import torch
import torch.nn as nn
from datasets import ClassLabel, Dataset, Features, Value

from regbench import search
from regbench.config import Config, RunSpec
from regbench.data import _grouped_split_indices
from regbench.data_aug import append_filtered_augmentation
from regbench.metrics import compute_metrics, ece
from regbench.report import report
from regbench.trainer import _dataloader_num_workers, run


def fake_runner(cfg, spec, run_dir, final=False):
    # Deterministic fake score to verify search ordering and parameter retention.
    base = {
        "rdrop": .025, "mixup": .020, "neftune": .015,
        "label_smoothing": .010, "augmentation": .005,
    }
    acc = .80 + sum(base[k] for k in spec.reg) + (0.001 if spec.lr == 2e-5 else 0) \
        + 0.0001 * sum(spec.reg.values()) * 0
    m = {"accuracy": acc, "macro_f1": acc, "ece": .05, "loss": 1 - acc, "n": 10}
    r = {"reg": spec.reg, "lr": spec.lr, "seed": spec.seed, "dev": m,
         "train_seconds": 1., "train_flops_est": 1e12, "epochs": cfg.epochs,
         "history": [{"epoch": epoch + 1, "train_loss": 0.5, "dev_loss": 0.4}
                     for epoch in range(cfg.epochs)]}
    if final:
        r.update(test=m)
    if cfg.save_final_model:
        from pathlib import Path
        import torch

        Path(run_dir).mkdir(parents=True, exist_ok=True)
        torch.save({}, Path(run_dir) / "model.pt")
    return r


def test_search_and_stack(tmp_path):
    cfg = Config()
    search.baseline(cfg, tmp_path, runner=fake_runner, log=lambda *_: None)
    s = search.individual(cfg, tmp_path, runner=fake_runner, log=lambda *_: None)
    assert s["ranking"] == ["rdrop", "mixup", "neftune", "label_smoothing", "augmentation"]
    assert sum(len(s["methods"][method]["runs"]) for method in s["ranking"]) == 45
    assert len(s["baseline_epoch_search"]["runs"]) == 9
    res = search.stack(cfg, tmp_path, runner=fake_runner, log=lambda *_: None)
    regs = [set(st["reg"]) for st in res["stages"]]
    assert regs[0] == {"rdrop", "mixup"} and regs[-1] == set(s["ranking"])
    # earlier params remain fixed
    assert res["stages"][1]["reg"]["rdrop"] == res["stages"][0]["reg"]["rdrop"]
    alt_order = ["neftune", "rdrop", "mixup", "label_smoothing", "augmentation"]
    alt = search.stack(cfg, tmp_path, alt_order, runner=fake_runner,
                       log=lambda *_: None)
    assert alt["tag"] != "ranked"
    final_data_sizes = []

    def final_runner(final_cfg, spec, run_dir, final=False):
        final_data_sizes.append((final_cfg.train_size, final_cfg.dev_eval_size, final_cfg.test_size))
        return fake_runner(final_cfg, spec, run_dir, final)

    budget_cfg = Config(train_size=436, dev_eval_size=2, test_size=1, final_seeds=[42, 123, 456])
    out = search.final(budget_cfg, tmp_path, runner=final_runner, log=lambda *_: None)
    assert len(out) == 10 and all(len(v) == 3 for v in out.values())
    assert len(final_data_sizes) == 30
    assert set(final_data_sizes) == {(None, None, None)}


def test_metrics():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 500)
    logits = np.eye(3)[y] * 20
    m = compute_metrics(logits, y)
    assert m["accuracy"] == 1 and m["macro_f1"] == 1 and m["ece"] < 1e-6
    # overconfident & always wrong -> ECE ~ 1
    wrong = np.eye(3)[(y + 1) % 3] * 20
    assert ece(np.exp(wrong) / np.exp(wrong).sum(-1, keepdims=True), y) > 0.99


def test_dataloader_worker_override(monkeypatch):
    cfg = Config(num_workers=2)
    monkeypatch.delenv("REGBENCH_DATALOADER_WORKERS", raising=False)
    assert _dataloader_num_workers(cfg) == 2
    monkeypatch.setenv("REGBENCH_DATALOADER_WORKERS", "0")
    assert _dataloader_num_workers(cfg) == 0
    monkeypatch.setenv("REGBENCH_DATALOADER_WORKERS", "-1")
    with pytest.raises(ValueError, match="non-negative"):
        _dataloader_num_workers(cfg)


def test_grouped_stratified_70_15_15_split_keeps_duplicate_pairs_together():
    rows = {
        "sentence1": [f"premise-{i}" for i in range(60)],
        "sentence2": [f"hypothesis-{i}" for i in range(60)],
        "label": [i % 2 for i in range(60)],
    }
    # Include repeated text pairs, including one pair with conflicting labels.
    rows["sentence1"] += ["premise-0", "premise-1", "premise-2"]
    rows["sentence2"] += ["hypothesis-0", "hypothesis-1", "hypothesis-2"]
    rows["label"] += [0, 0, 1]
    ds = Dataset.from_dict(rows, features=Features({
        "sentence1": Value("string"),
        "sentence2": Value("string"),
        "label": ClassLabel(names=["entailment", "not_entailment"]),
    }))
    indices = _grouped_split_indices(ds, Config())
    pair_partition = {}
    pair_counts = {}
    for name, rows_in_split in indices.items():
        pairs = set()
        for i in rows_in_split:
            pair = (ds[i]["sentence1"], ds[i]["sentence2"])
            assert pair_partition.setdefault(pair, name) == name
            pairs.add(pair)
        pair_counts[name] = len(pairs)
    assert sorted(pair_counts.values()) == [9, 9, 42]
    assert sum(map(len, indices.values())) == len(ds)


def test_augmentation_appends_only_requested_train_rows_and_caps_at_available(tmp_path):
    train = Dataset.from_dict({
        "input_ids": [[1, 2], [3, 4], [5, 6], [7, 8]],
        "attention_mask": [[1, 1]] * 4,
        "labels": [0, 1, 0, 1],
    })
    filtered_path = tmp_path / "filtered.jsonl"
    filtered_path.write_text("\n".join(json.dumps({
        "sentence1": f"premise-{i}",
        "sentence2": f"hypothesis-{i}",
        "labels": i % 2,
    }) for i in range(3)))

    class Tokenizer:
        def __call__(self, premises, hypotheses, truncation, max_length):
            return {
                "input_ids": [[10 + i, 20 + i] for i, _ in enumerate(premises)],
                "attention_mask": [[1, 1] for _ in premises],
            }

    combined, info = append_filtered_augmentation(Config(), train, 1.0, filtered_path, Tokenizer())
    assert len(combined) == 7
    assert combined[:4]["labels"] == train["labels"]
    assert info == {
        "requested_ratio": 1.0,
        "requested_synthetic_count": 4,
        "available_filtered_count": 3,
        "used_synthetic_count": 3,
        "achieved_ratio": 0.75,
    }


def test_report_uses_validation_for_selection_and_test_for_final_metrics(tmp_path):
    logits = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]] * 4)
    labels = np.tile(np.array([0, 1, 2]), 4)
    metrics = compute_metrics(logits, labels)
    for seed in (0, 1):
        run = tmp_path / "final" / "baseline" / f"seed{seed}"
        run.mkdir(parents=True)
        (run / "metrics.json").write_text(json.dumps({
            "reg": {}, "lr": 2e-5, "dev": metrics, "test": metrics,
            "train_seconds": 60.0, "train_flops_est": 1e12,
        }))
        np.savez_compressed(run / "predictions_dev.npz", logits=logits, labels=labels)
        np.savez_compressed(run / "predictions_test.npz", logits=logits, labels=labels)

    rows = report(tmp_path)
    assert rows[0]["test_accuracy"] == 1.0
    assert rows[0]["validation_accuracy"] == 1.0
    assert (tmp_path / "reports" / "test_metrics.png").exists()
    assert (tmp_path / "reports" / "reliability_diagrams.png").exists()
    assert (tmp_path / "reports" / "loss_curves.png").exists()
    assert (tmp_path / "reports" / "accuracy_curves.png").exists()
    assert (tmp_path / "reports" / "epoch_curves_all_configs.png").exists()
    assert (tmp_path / "reports" / "individual_vs_stack_accuracy.png").exists()


@pytest.mark.parametrize("save_resume", [True, False])
def test_training_promotes_half_precision_checkpoint_to_fp32(tmp_path, monkeypatch, save_resume):
    class HalfPrecisionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(32, 8)
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(8, 2)
            self.half()

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None):
            if inputs_embeds is None:
                inputs_embeds = self.embedding(input_ids)
            mask = attention_mask[..., None].to(inputs_embeds.dtype)
            pooled = (inputs_embeds * mask).sum(1) / mask.sum(1).clamp(min=1)
            return type("Output", (), {"logits": self.classifier(self.dropout(pooled))})()

    data = Dataset.from_dict({
        "input_ids": [[1, 2, 0], [3, 4, 5], [6, 7, 0], [8, 9, 10]],
        "attention_mask": [[1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1]],
        "labels": [0, 1, 0, 1],
    })
    model = HalfPrecisionModel()
    if not save_resume:
        def unexpected_save(*args, **kwargs):
            pytest.fail("No model or resume checkpoint should be written")
        monkeypatch.setattr(torch, "save", unexpected_save)
    result = run(
        Config(epochs=1, batch_size=2, eval_batch_size=2, num_workers=0, precision="fp32"),
        RunSpec(reg={}, lr=2e-5, seed=7),
        tmp_path / "half-checkpoint",
        splits={"train": data, "dev": data},
        model_builder=lambda _: model,
        pad_id=0,
        record_train_accuracy=True,
        save_resume=save_resume,
    )
    assert next(model.parameters()).dtype == torch.float32
    assert np.isfinite(result["dev"]["loss"])
    assert 0.0 <= result["history"][0]["train_accuracy"] <= 1.0
    assert not list((tmp_path / "half-checkpoint").glob("*.pt*"))
