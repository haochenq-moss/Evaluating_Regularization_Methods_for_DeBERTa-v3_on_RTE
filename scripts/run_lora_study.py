#!/usr/bin/env python3
"""Screen LoRA learning rates on development data, then run a matched ablation."""
from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import replace
from pathlib import Path

import datasets
import peft
import torch
import transformers
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer

from regbench.config import Config, RunSpec
from regbench.data import cache_path, load_splits
from regbench.trainer import build_model, run
from run_followup_study import CONDITIONS, METHODS, SEEDS, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "lora_study"
LRS = (3e-5, 1e-4, 2e-4)
LORA = {"r": 8, "lora_alpha": 16, "lora_dropout": 0.1,
        "target_modules": ["query_proj", "value_proj"],
        "modules_to_save": ["classifier", "pooler"]}


def build_lora(cfg):
    model = get_peft_model(build_model(cfg), LoraConfig(task_type=TaskType.SEQ_CLS, **LORA))
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    if not any("query_proj.lora_" in name for name in trainable):
        raise ValueError("No trainable query LoRA weights were found.")
    if not any("value_proj.lora_" in name for name in trainable):
        raise ValueError("No trainable value LoRA weights were found.")
    if not all(any(f".{name}.modules_to_save." in n for n in trainable)
               for name in ("classifier", "pooler")):
        raise ValueError("Classification head and pooler must remain trainable.")
    if model.get_input_embeddings().weight.requires_grad:
        raise ValueError("Base embeddings must be frozen.")
    print(f"[lora] trainable parameters={sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
          f"of {sum(p.numel() for p in model.parameters()):,}", flush=True)
    return model


def choose_lr(results):
    if set(results) != set(LRS):
        raise ValueError("Learning-rate screening is incomplete.")
    return max(LRS, key=lambda lr: (results[lr]["dev"]["accuracy"],
                                     -results[lr]["dev"]["loss"], -lr))


def main():
    os.chdir(ROOT)
    if not torch.cuda.is_available():
        raise SystemExit("A Slurm GPU allocation with CUDA is required.")
    base = Config.load(ROOT / "configs/default.yaml")
    splits = load_splits(base)
    if len(splits["train"]) != 1742 or len(splits["dev"]) != 374:
        raise ValueError("Unexpected internal train/development sizes.")
    pad_id = AutoTokenizer.from_pretrained(base.model_name).pad_token_id
    meta = {"device": torch.cuda.get_device_name(), "python": platform.python_version(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformers": transformers.__version__, "datasets": datasets.__version__,
            "peft": peft.__version__, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "lora": LORA, "seeds": SEEDS, "lr_candidates": LRS,
            "selection": "seed 42 baseline 8-epoch development accuracy; tie lower loss then lower LR",
            "test": "previously inspected internal holdout; exploratory reuse",
            "weights": "none saved; best-epoch test unavailable"}
    write_json(OUT / "environment.json", meta)
    started = time.monotonic()
    selection_file = OUT / "selection.json"
    if selection_file.exists():
        selection = json.loads(selection_file.read_text())
        if selection["lora"] != LORA or selection["lr_candidates"] != list(LRS):
            raise ValueError("Existing frozen selection disagrees with LoRA protocol.")
        lr = selection["selected_lr"]
        if lr not in LRS:
            raise ValueError("Invalid frozen learning rate.")
    else:
        screen = {}
        cfg = replace(base, epochs=8, save_final_model=False)
        for lr in LRS:
            print(f"[lora] screening lr={lr}", flush=True)
            screen[lr] = run(cfg, RunSpec({}, lr, 42), OUT / "screen" / f"lr{lr:g}",
                             splits=splits, pad_id=pad_id, model_builder=build_lora,
                             record_train_accuracy=True, save_resume=False)
        lr = choose_lr(screen)
        write_json(selection_file, {
            "lora": LORA, "lr_candidates": LRS, "selected_lr": lr,
            "criterion": meta["selection"],
            "screen": {str(rate): {"dev": item["dev"], "history": item["history"]}
                       for rate, item in screen.items()},
        })
    print(f"[lora] frozen selected learning rate={lr}", flush=True)
    holdout = load_from_disk(str(cache_path(base)))["internal_holdout"]
    if len(holdout) != 374:
        raise ValueError("Unexpected internal holdout size.")
    final_splits = {**splits, "test": holdout}
    for method, epochs in CONDITIONS:
        cfg = replace(base, epochs=epochs, save_final_model=False)
        for seed in SEEDS:
            print(f"[lora] {method} epochs={epochs} seed={seed}", flush=True)
            result = run(cfg, RunSpec(dict(METHODS[method]), lr, seed),
                         OUT / "runs" / f"{method}_{epochs}" / f"seed{seed}",
                         final=True, splits=final_splits, pad_id=pad_id,
                         model_builder=build_lora, record_train_accuracy=True, save_resume=False)
            if (result["seed"] != seed or result["reg"] != METHODS[method]
                    or result["lr"] != lr or result["epochs"] != epochs
                    or len(result["history"]) != epochs or result["test"]["n"] != 374):
                raise ValueError(f"Invalid output for {method} epoch={epochs} seed={seed}")
    meta["controller_wall_seconds"] = time.monotonic() - started
    write_json(OUT / "environment.json", meta)
    print("[lora] All 40 final-epoch runs completed.", flush=True)


if __name__ == "__main__":
    main()
