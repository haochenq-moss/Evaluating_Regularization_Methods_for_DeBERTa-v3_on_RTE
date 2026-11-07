"""Baseline tuning, individual regularizer search, cumulative stacks and final retraining."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .config import REGULARIZERS, Config, RunSpec
from . import trainer


def select_key(metrics: dict):
    return metrics["dev"]["accuracy"], -metrics["dev"]["loss"]


def _slim(result: dict) -> dict:
    keys = ("reg", "lr", "seed", "dev", "train_seconds", "train_flops_est", "epochs",
            "augmentation_info")
    return {key: result[key] for key in keys if key in result}


def _grid_runs(cfg: Config, out: Path, candidates: list[tuple[dict, float]], runner, log):
    results = []
    for reg, lr in candidates:
        spec = RunSpec(reg=reg, lr=lr, seed=cfg.search_seed)
        log(f"[run] {spec.key()}")
        results.append(runner(cfg, spec, out / spec.key()))
    return max(results, key=select_key), results


def baseline(cfg: Config, out: Path, runner=trainer.run, log=print) -> dict:
    """Tune LR and epochs on internal validation, then export the selected teacher weights."""
    out = Path(out)
    individual_dir = out / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for epochs in cfg.grid.baseline_epochs:
        candidate_cfg = replace(cfg, epochs=epochs)
        for lr in cfg.grid.lrs:
            spec = RunSpec(reg={}, lr=lr, seed=cfg.search_seed)
            run_dir = individual_dir / "baseline" / f"epochs{epochs}" / spec.key()
            log(f"[baseline] epochs={epochs} lr={lr:g}")
            result = runner(candidate_cfg, spec, run_dir)
            candidates.append({**result, "epochs": epochs})
    best = max(candidates, key=select_key)
    summary = {
        "baseline_epoch_search": {
            "epoch_candidates": cfg.grid.baseline_epochs,
            "runs": [_slim(result) for result in candidates],
        },
        "methods": {"baseline": {"best": _slim(best), "runs": [_slim(r) for r in candidates]}},
        "ranking": [],
    }
    (individual_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    teacher_cfg = replace(cfg, epochs=best["epochs"], save_final_model=True)
    teacher_spec = RunSpec(reg={}, lr=best["lr"], seed=cfg.search_seed)
    teacher_dir = out / "baseline_for_augmentation"
    log("[baseline] fitting selected configuration for augmentation filtering")
    runner(teacher_cfg, teacher_spec, teacher_dir)
    weights = teacher_dir / "model.pt"
    if not weights.is_file():
        raise RuntimeError(f"baseline teacher weights were not saved: {weights}")
    return summary


def individual(cfg: Config, out: Path, runner=trainer.run, log=print) -> dict:
    """Tune each regularizer independently, using the baseline-selected epoch count."""
    out = Path(out)
    individual_dir = out / "individual"
    summary_path = individual_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Run `regbench baseline` before `regbench individual`.")
    summary = json.loads(summary_path.read_text())
    epochs = summary["methods"]["baseline"]["best"]["epochs"]
    method_cfg = replace(cfg, epochs=epochs)
    for method in REGULARIZERS:
        candidates = [
            ({method: parameter}, lr)
            for lr in cfg.grid.lrs
            for parameter in getattr(cfg.grid, method)
        ]
        best, runs = _grid_runs(method_cfg, individual_dir / method, candidates, runner, log)
        summary["methods"][method] = {
            "best": {**_slim(best), "epochs": epochs},
            "runs": [{**_slim(run), "epochs": epochs} for run in runs],
        }
    summary["ranking"] = sorted(
        REGULARIZERS,
        key=lambda method: select_key(summary["methods"][method]["best"]),
        reverse=True,
    )
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def stack_tag(out: Path, order: list[str] | None) -> str:
    ranked = json.loads((Path(out) / "individual" / "summary.json").read_text())["ranking"]
    return "ranked" if not order or order == ranked else "-".join(order)


def stack(cfg: Config, out: Path, order: list[str] | None = None, runner=trainer.run, log=print) -> dict:
    """Add the second-ranked method first; tune only each newly added method and LR."""
    out = Path(out)
    summary = json.loads((out / "individual" / "summary.json").read_text())
    order = order or summary["ranking"]
    if len(order) != len(REGULARIZERS) or set(order) != set(REGULARIZERS):
        raise ValueError(f"stack order must contain each method exactly once: {REGULARIZERS}")
    tag = stack_tag(out, order)
    stack_dir = out / f"stack_{tag}"
    epochs = summary["methods"]["baseline"]["best"]["epochs"]
    method_cfg = replace(cfg, epochs=epochs)
    fixed = dict(summary["methods"][order[0]]["best"]["reg"])
    stages = []

    for count, method in enumerate(order[1:], start=2):
        candidates = [
            ({**fixed, method: parameter}, lr)
            for lr in cfg.grid.lrs
            for parameter in getattr(cfg.grid, method)
        ]
        best, _ = _grid_runs(
            method_cfg, stack_dir / f"stage{count}_{method}", candidates, runner, log
        )
        fixed = dict(best["reg"])
        stage = {
            "stage": count,
            "added": method,
            "reg": fixed,
            "lr": best["lr"],
            "epochs": epochs,
            "dev": best["dev"],
            "train_seconds": best["train_seconds"],
            "train_flops_est": best["train_flops_est"],
        }
        stages.append(stage)
        stage_dir = stack_dir / f"stage{count}_{method}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "stage.json").write_text(json.dumps(stage, indent=2))
    result = {"order": order, "tag": tag, "stages": stages}
    stack_dir.mkdir(parents=True, exist_ok=True)
    (stack_dir / "stages.json").write_text(json.dumps(result, indent=2))
    return result


def final_configs(out: Path, tag: str = "ranked") -> dict[str, tuple[dict, float]]:
    """Selected baseline, individual methods, and stacks of two through all five methods."""
    out = Path(out)
    summary = json.loads((out / "individual" / "summary.json").read_text())
    configs = {
        "baseline": (summary["methods"]["baseline"]["best"]["reg"],
                     summary["methods"]["baseline"]["best"]["lr"])
    }
    for method in REGULARIZERS:
        best = summary["methods"][method]["best"]
        configs[method] = (best["reg"], best["lr"])
    stages = json.loads((out / f"stack_{tag}" / "stages.json").read_text())["stages"]
    for stage in stages:
        names = "+".join(method for method in REGULARIZERS if method in stage["reg"])
        configs[f"stack{stage['stage']}[{tag}]:{names}"] = (stage["reg"], stage["lr"])
    return configs


def final(cfg: Config, out: Path, tag: str = "ranked", runner=trainer.run, log=print) -> dict:
    out = Path(out)
    final_dir = out / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((out / "individual" / "summary.json").read_text())
    epochs = summary["methods"]["baseline"]["best"]["epochs"]
    final_cfg = replace(
        cfg, epochs=epochs, train_size=None, dev_eval_size=None, test_size=None
    )
    results = {}
    for name, (reg, lr) in final_configs(out, tag).items():
        safe = name.replace(":", "_").replace("[", "_").replace("]", "").replace("+", "-")
        for seed in cfg.final_seeds:
            spec = RunSpec(reg=reg, lr=lr, seed=seed)
            log(f"[final] {name} seed={seed}")
            results.setdefault(name, []).append(
                runner(final_cfg, spec, final_dir / safe / f"seed{seed}", final=True)
            )
    (final_dir / "summary.json").write_text(json.dumps(results, indent=2))
    return results
