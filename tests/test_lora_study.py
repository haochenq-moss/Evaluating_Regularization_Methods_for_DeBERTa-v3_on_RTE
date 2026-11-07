import sys
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import DebertaV2Config, DebertaV2ForSequenceClassification

from datasets import Dataset
from regbench.config import Config, RunSpec
from regbench.regularizers import training_loss
from regbench.trainer import run

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_lora_study as study
import report_lora_study as report


def test_learning_rate_screen_rule():
    results = {
        3e-5: {"dev": {"accuracy": .8, "loss": .4}},
        1e-4: {"dev": {"accuracy": .8, "loss": .3}},
        2e-4: {"dev": {"accuracy": .8, "loss": .3}},
    }
    assert study.choose_lr(results) == 1e-4
    with pytest.raises(ValueError):
        study.choose_lr({3e-5: results[3e-5]})


@pytest.mark.parametrize("reg", list(study.METHODS.values()))
def test_lora_regularizers_backpropagate(reg):
    config = DebertaV2Config(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                             num_attention_heads=4, vocab_size=128, num_labels=2)
    model = get_peft_model(DebertaV2ForSequenceClassification(config),
                           LoraConfig(task_type=TaskType.SEQ_CLS, **study.LORA))
    model.train()
    batch = {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
             "attention_mask": torch.ones(2, 3, dtype=torch.long),
             "labels": torch.tensor([0, 1])}
    loss, _, _ = training_loss(model, batch, reg)
    loss.backward()
    assert torch.isfinite(loss)
    assert not model.get_input_embeddings().weight.requires_grad
    for component in ("query_proj.lora_B", "value_proj.lora_B",
                      "classifier.modules_to_save", "pooler.modules_to_save"):
        assert any(p.grad is not None and torch.isfinite(p.grad).all()
                   for name, p in model.named_parameters()
                   if component in name and p.requires_grad), component


def test_report_includes_paired_interaction_and_seed_values():
    lora = {}
    full = {}
    for method, epochs in study.CONDITIONS:
        lora[(method, epochs)] = {}
        full[(method, epochs)] = {}
        for i, seed in enumerate(study.SEEDS):
            lora[(method, epochs)][seed] = {
                "dev": {"accuracy": .7 + i*.01}, "test": {"accuracy": .71 + i*.01},
                "train_seconds": 10.0,
                "history": [{"epoch": e, "dev_accuracy": .7 + i*.01, "dev_loss": .5/e}
                            for e in range(1, epochs + 1)],
            }
            full[(method, epochs)][seed] = {"dev": {"accuracy": .7 + i*.01}}
    selection = {"selected_lr": 1e-4, "screen": {
        str(lr): {"dev": {"accuracy": .7, "loss": .4}} for lr in study.LRS}}
    environment = {"device": "A40", "python": "3.12", "torch": "2.11",
                   "peft": "0.21", "transformers": "5.22", "controller_wall_seconds": 3600}
    screening = {lr: {"train_seconds": 4.0} for lr in study.LRS}
    section = report.report_section(lora, full, selection, environment, screening)
    assert "42 & 71.00 & 71.00 & 71.00 & 71.00 & 71.00 & 71.00" in section
    assert "R-Drop & +0.00 & [+0.00, +0.00]" in section
    assert "PEFT 0.21" in section and "not an independent test" in section
    assert "screening runs used 12.0 seconds" in section
    assert "40 matched runs used 400.0 seconds" in section
    assert r"Full stack & 12 & 10.0 & 50.0 \\" in section
    abstract = report.combined_abstract(lora, full)
    assert "full fine-tuning and standard LoRA" in abstract
    assert "previously inspected" in abstract and "Learning rates differ" in abstract
    roadmap = report.study_overview(selection["selected_lr"])
    assert "reused holdout" in roadmap and "0.0001" in roadmap
    conclusion = report.combined_conclusion(lora, full, selection)
    assert "The difference in R-Drop gains (LoRA minus full tuning) was +0.00" in conclusion
    assert "No new independent evaluation was performed here" in conclusion
    assert "original historical comparison also differed in duration" in conclusion


def test_end_to_end_metrics_and_predictions_without_checkpoints(tmp_path):
    data = Dataset.from_dict({"input_ids": [[1, 2], [3, 4], [5, 6], [7, 8]],
                              "attention_mask": [[1, 1]] * 4, "labels": [0, 1, 0, 1]})
    def builder(_):
        config = DebertaV2Config(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                 num_attention_heads=4, vocab_size=128, num_labels=2)
        return get_peft_model(DebertaV2ForSequenceClassification(config),
                              LoraConfig(task_type=TaskType.SEQ_CLS, **study.LORA))
    result = run(Config(epochs=1, batch_size=2, eval_batch_size=2, num_workers=0),
                 RunSpec(study.METHODS["stack"], 1e-4, 42), tmp_path,
                 final=True, splits={"train": data, "dev": data, "test": data},
                 model_builder=builder, pad_id=0, record_train_accuracy=True, save_resume=False)
    assert result["test"]["n"] == 4 and len(result["history"]) == 1
    assert (tmp_path / "predictions_test.npz").is_file()
    assert not list(tmp_path.glob("*.pt*"))
