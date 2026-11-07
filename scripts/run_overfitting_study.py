#!/usr/bin/env python3
"""Screen regularization and LoRA capacity, then run matched development comparisons."""
from __future__ import annotations

import json
import argparse
import os
import platform
import time
from dataclasses import replace
from pathlib import Path

import datasets
import peft
import torch
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer

from regbench.config import Config, RunSpec
from regbench.data import load_splits
from regbench.trainer import build_model, run
from run_followup_study import write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "overfitting_study"
SCREEN_SEEDS = (11, 21, 31)
EVAL_SEEDS = (42, 123, 456, 789, 1011)
STRENGTHS = (0.1, 0.5, 1.0)
RANKS = (4, 8, 16)
DROPOUTS = (0.0, 0.1)
EPOCHS = 12
LRS = {"full": 3e-5, "lora": 2e-4}
PROTOCOL = {
    "screen_seeds": SCREEN_SEEDS, "evaluation_seeds": EVAL_SEEDS,
    "strengths": STRENGTHS, "ranks": RANKS, "dropouts": DROPOUTS,
    "epochs": EPOCHS, "learning_rates": LRS,
    "alpha": "2 * rank", "lora_targets": ["query_proj", "value_proj"],
    "screen_rule": "highest mean final development accuracy; tie lower mean development loss, then lower values",
    "best_epoch": "highest development accuracy; tie lower development loss, then earlier epoch",
    "evaluation": "development only; no holdout", "weights": "none saved",
    "lora_strength_screen": "at the separately selected baseline rank and dropout",
}


def build_lora(rank, dropout):
    def builder(cfg):
        config = LoraConfig(task_type=TaskType.SEQ_CLS, r=rank, lora_alpha=2 * rank,
                            lora_dropout=dropout, target_modules=["query_proj", "value_proj"],
                            modules_to_save=["classifier", "pooler"])
        model = get_peft_model(build_model(cfg), config)
        names = [name for name, p in model.named_parameters() if p.requires_grad]
        for token in ("query_proj.lora_", "value_proj.lora_", "classifier.modules_to_save",
                      "pooler.modules_to_save"):
            if not any(token in name for name in names):
                raise ValueError(f"Missing trainable LoRA parameter: {token}")
        if model.get_input_embeddings().weight.requires_grad:
            raise ValueError("Base embeddings must be frozen.")
        return model
    return builder


def run_one(base, splits, pad_id, tuning, reg, rank, dropout, seed, path):
    identity = {"tuning": tuning, "reg": reg, "seed": seed, "lr": LRS[tuning],
                "epochs": EPOCHS, "rank": rank if tuning == "lora" else None,
                "dropout": dropout if tuning == "lora" else None,
                "alpha": 2 * rank if tuning == "lora" else None}
    path.mkdir(parents=True, exist_ok=True)
    identity_path = path / "study_config.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError(f"Run identity mismatch: {path}")
    elif (path / "metrics.json").exists():
        raise ValueError(f"Metrics without study identity: {path}")
    else:
        write_json(identity_path, identity)
    cfg = replace(base, epochs=EPOCHS, save_final_model=False)
    builder = build_lora(rank, dropout) if tuning == "lora" else None
    metrics = run(cfg, RunSpec(reg, LRS[tuning], seed), path, splits=splits, pad_id=pad_id,
                  model_builder=builder, record_train_accuracy=True, save_resume=False,
                  record_train_clean_loss=True)
    if (metrics["seed"] != seed or metrics["reg"] != reg or metrics["lr"] != LRS[tuning]
            or metrics["epochs"] != EPOCHS or len(metrics["history"]) != EPOCHS
            or any("train_clean_loss" not in row for row in metrics["history"])):
        raise ValueError(f"Unexpected metrics: {path}")
    return metrics


def rank_screen(scores):
    if not scores:
        raise ValueError("Empty screen.")
    def key(candidate):
        values = scores[candidate]
        if len(values) != len(SCREEN_SEEDS):
            raise ValueError(f"Incomplete screen for {candidate}")
        mean_acc = sum(v["dev"]["accuracy"] for v in values) / len(values)
        mean_loss = sum(v["dev"]["loss"] for v in values) / len(values)
        return mean_acc, -mean_loss, tuple(-v for v in candidate)
    return max(scores, key=key)


def best_epoch(history):
    if not history:
        raise ValueError("Empty history")
    return max(history, key=lambda x: (x["dev_accuracy"], -x["dev_loss"], -x["epoch"]))


def main(phase="all"):
    os.chdir(ROOT)
    if not torch.cuda.is_available():
        raise SystemExit("Run inside a Slurm GPU allocation.")
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / "protocol.json"
    protocol = json.loads(json.dumps(PROTOCOL))
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Stored protocol differs; refusing to reuse runs.")
    write_json(protocol_path, protocol)
    base = Config.load(ROOT / "configs/default.yaml")
    splits = load_splits(base)
    if len(splits["train"]) != 1742 or len(splits["dev"]) != 374:
        raise ValueError("Unexpected split sizes.")
    pad_id = AutoTokenizer.from_pretrained(base.model_name).pad_token_id
    started = time.monotonic()
    environment = {
        "job_id": os.environ.get("SLURM_JOB_ID"), "gpu": torch.cuda.get_device_name(),
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "datasets": datasets.__version__,
        "peft": peft.__version__,
    }
    environment["phase"] = phase
    environment_path = OUT / f"environment_{phase}.json"
    write_json(environment_path, environment)
    selection_path = OUT / "selection.json"
    rank_path = OUT / "rank_selection.json"
    if phase in ("all", "adapter"):
        rank_scores = {}
        for rank in RANKS:
            for dropout in DROPOUTS:
                values = []
                for seed in SCREEN_SEEDS:
                    print(f"[screen] lora rank={rank} dropout={dropout} seed={seed}", flush=True)
                    values.append(run_one(base, splits, pad_id, "lora", {}, rank, dropout, seed,
                                          OUT / "screen" / f"rank{rank}_drop{dropout:g}" / f"seed{seed}"))
                rank_scores[(rank, dropout)] = values
        rank, dropout = rank_screen(rank_scores)
        rank_selection = {"rank": rank, "dropout": dropout}
        if rank_path.exists() and json.loads(rank_path.read_text()) != rank_selection:
            raise ValueError("Frozen adapter selection differs from the screen.")
        write_json(rank_path, rank_selection)
    if phase in ("all", "strength"):
        rank_selection = json.loads(rank_path.read_text())
        rank, dropout = rank_selection["rank"], rank_selection["dropout"]
        if rank not in RANKS or dropout not in DROPOUTS:
            raise ValueError("Invalid frozen adapter selection.")
        selected = {"rank": rank, "dropout": dropout}
        for tuning in ("full", "lora"):
            scores = {}
            for strength in STRENGTHS:
                values = []
                for seed in SCREEN_SEEDS:
                    print(f"[screen] {tuning} rdrop={strength} seed={seed}", flush=True)
                    values.append(run_one(base, splits, pad_id, tuning, {"rdrop": strength}, rank, dropout,
                                          seed, OUT / "screen" / f"{tuning}_rdrop{strength:g}" / f"seed{seed}"))
                scores[(strength,)] = values
            selected[f"{tuning}_strength"] = rank_screen(scores)[0]
        selected["rule"] = PROTOCOL["screen_rule"]
        if selection_path.exists() and json.loads(selection_path.read_text()) != selected:
            raise ValueError("Frozen selection differs from screening results.")
        write_json(selection_path, selected)
    if phase in ("all", "compare"):
        selected = json.loads(selection_path.read_text())
        if (selected["full_strength"] not in STRENGTHS or selected["lora_strength"] not in STRENGTHS
                or selected["rank"] not in RANKS or selected["dropout"] not in DROPOUTS
                or json.loads(rank_path.read_text()) !=
                {"rank": selected["rank"], "dropout": selected["dropout"]}):
            raise ValueError("Invalid frozen selection.")
        print(f"[selection] {selected}", flush=True)
        for tuning in ("full", "lora"):
            for method in ("baseline", "rdrop"):
                reg = {} if method == "baseline" else {"rdrop": selected[f"{tuning}_strength"]}
                for seed in EVAL_SEEDS:
                    print(f"[evaluation] {tuning} {method} seed={seed}", flush=True)
                    run_one(base, splits, pad_id, tuning, reg, selected["rank"], selected["dropout"],
                            seed, OUT / "runs" / tuning / method / f"seed{seed}")
    environment["controller_wall_seconds"] = time.monotonic() - started
    write_json(environment_path, environment)
    print(f"[study] Phase {phase} completed; no holdout evaluated.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("all", "adapter", "strength", "compare"), default="all")
    main(parser.parse_args().phase)
