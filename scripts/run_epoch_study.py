"""Run a validation-first, paired-seed RTE epoch study and evaluate a frozen holdout."""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer

from regbench.config import Config, RunSpec
from regbench.data import cache_path, load_splits, make_collator
from regbench.metrics import compute_metrics
from regbench.trainer import _device_and_amp, build_model, predict, run

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "epoch_study"
SEEDS = (42, 123, 456, 789, 1011)
OLD_SEEDS = (42, 123, 456)
SETTINGS = {
    "baseline": ({}, 2e-5, 5),
    "rdrop": ({"rdrop": 0.5}, 3e-5, 5),
    "stack5": ({"mixup": 0.4, "rdrop": 0.5, "neftune": 15.0}, 3e-5, 5),
    "stack6": ({"mixup": 0.4, "rdrop": 0.5, "neftune": 15.0}, 3e-5, 6),
    "stack8": ({"mixup": 0.4, "rdrop": 0.5, "neftune": 15.0}, 3e-5, 8),
}
OLD_NAMES = {
    "baseline": "baseline",
    "rdrop": "rdrop",
    "stack5": "stack3_ranked_mixup-rdrop-neftune",
}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def spec_for(name: str, seed: int) -> RunSpec:
    reg, lr, _ = SETTINGS[name]
    return RunSpec(reg=dict(reg), lr=lr, seed=seed)


def cfg_for(base: Config, name: str) -> Config:
    return replace(base, epochs=SETTINGS[name][2], save_final_model=True)


def old_validation(name: str, seed: int) -> dict:
    path = ROOT / "outputs" / "final" / OLD_NAMES[name] / f"seed{seed}"
    config = json.loads((path / "config.json").read_text())
    metrics = json.loads((path / "metrics.json").read_text())
    reg, lr, epochs = SETTINGS[name]
    if (config["spec"] != {"reg": reg, "lr": lr, "seed": seed}
            or metrics["epochs"] != epochs or metrics["dev"]["n"] != 374
            or config["cfg"]["train_size"] is not None
            or config["cfg"]["dev_eval_size"] is not None):
        raise ValueError(f"Existing run does not match the study protocol: {path}")
    return {key: metrics[key] for key in ("reg", "lr", "seed", "epochs", "history", "dev")}


def new_run(base: Config, name: str, seed: int, splits: dict, pad_id: int) -> dict:
    print(f"[study] training {name} seed={seed}", flush=True)
    return run(cfg_for(base, name), spec_for(name, seed),
               OUT / "runs" / name / f"seed{seed}", splits=splits, pad_id=pad_id,
               record_train_accuracy=True)


def freeze_selection(validation: dict) -> dict:
    expected_seeds = {str(seed) for seed in SEEDS}
    for name in SETTINGS:
        if set(validation[name]) != expected_seeds:
            raise ValueError(f"Incomplete or unexpected seeds for {name}")
    def key(name: str) -> tuple[float, float]:
        values = validation[name].values()
        scores = [item["dev"]["accuracy"] for item in values]
        losses = [item["dev"]["loss"] for item in validation[name].values()]
        return float(np.mean(scores)), -float(np.mean(losses))
    best = max(("stack5", "stack6", "stack8"), key=key)
    return {"seeds": list(SEEDS), "stack_epoch": SETTINGS[best][2],
            "selection_rule": "highest mean internal validation accuracy; ties by lower mean loss",
            "validation": validation}


def main() -> None:
    os.chdir(ROOT)
    if not torch.cuda.is_available():
        raise SystemExit("Run this study inside a Slurm GPU allocation; CUDA is unavailable.")
    base = Config.load(ROOT / "configs" / "default.yaml")
    splits = load_splits(base)
    if len(splits["train"]) != 1742 or len(splits["dev"]) != 374:
        raise ValueError("Unexpected internal train/validation sizes; refusing to run.")
    pad_id = AutoTokenizer.from_pretrained(base.model_name).pad_token_id
    selection_path = OUT / "selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        if selection["seeds"] != list(SEEDS):
            raise ValueError("Frozen study seeds do not match this protocol.")
        print("[study] using frozen validation selection", flush=True)
    else:
        validation = {name: {} for name in SETTINGS}
        # Longer stack runs are intentionally first.
        for name in ("stack6", "stack8", "baseline", "rdrop", "stack5"):
            for seed in SEEDS:
                if name in OLD_NAMES and seed in OLD_SEEDS:
                    metrics = old_validation(name, seed)
                else:
                    metrics = new_run(base, name, seed, splits, pad_id)
                validation[name][str(seed)] = metrics
        selection = freeze_selection(validation)
        write_json(selection_path, selection)
        print(f"[study] selection frozen: stack epochs={selection['stack_epoch']}", flush=True)

    chosen_stack = f"stack{selection['stack_epoch']}"
    if chosen_stack not in ("stack5", "stack6", "stack8"):
        raise ValueError("Invalid frozen stack epoch count.")

    # Recreate selected older models before opening the reserved holdout.
    for name in ("baseline", "rdrop", chosen_stack):
        for seed in SEEDS:
            if not (OUT / "runs" / name / f"seed{seed}" / "model.pt").is_file():
                new_run(base, name, seed, splits, pad_id)

    final_path = OUT / "final_metrics.json"
    if final_path.exists():
        print("[study] final metrics already exist; refusing to evaluate holdout again", flush=True)
        return
    holdout = load_from_disk(str(cache_path(base)))["internal_holdout"]
    if len(holdout) != 374:
        raise ValueError("Unexpected reserved holdout size; refusing to evaluate.")
    device, amp = _device_and_amp(base)
    collate = make_collator(pad_id)
    results = {"seeds": list(SEEDS), "stack_epoch": selection["stack_epoch"],
               "runs": {"baseline": {}, "rdrop": {}, "stack": {}}}
    for output_name, name in (("baseline", "baseline"), ("rdrop", "rdrop"),
                              ("stack", chosen_stack)):
        for seed in SEEDS:
            print(f"[holdout] {output_name} seed={seed}", flush=True)
            model = build_model(base).to(device=device, dtype=torch.float32)
            path = OUT / "runs" / name / f"seed{seed}" / "model.pt"
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            logits, labels, _ = predict(model, holdout, collate, base, device, amp)
            results["runs"][output_name][str(seed)] = compute_metrics(logits, labels, base.ece_bins)
            np.savez_compressed(OUT / "runs" / name / f"seed{seed}" / "predictions_holdout.npz",
                                logits=logits, labels=labels, preds=logits.argmax(-1))
            del model
    write_json(final_path, results)
    print(f"[study] final holdout results saved to {final_path}", flush=True)


if __name__ == "__main__":
    main()
