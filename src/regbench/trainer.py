"""Training / evaluation of one run (one regulariser configuration, one lr, one seed)."""
from __future__ import annotations

import json
import hashlib
import os
import pickle
import random
import re
import time
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .config import Config, RunSpec, config_hash
from .data import load_splits, make_collator
from .metrics import compute_metrics
from .regularizers import training_loss

NUM_LABELS = 2


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _device_and_amp(cfg: Config):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16 = cfg.precision == "bf16" or (cfg.precision == "auto" and dev.type == "cuda"
                                       and torch.cuda.is_bf16_supported())
    return dev, bf16 and dev.type == "cuda"


def _to(batch, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in batch.items()}


def _dataloader_num_workers(cfg: Config) -> int:
    override = os.environ.get("REGBENCH_DATALOADER_WORKERS")
    try:
        workers = cfg.num_workers if override is None else int(override)
    except ValueError as exc:
        raise ValueError("REGBENCH_DATALOADER_WORKERS must be a non-negative integer") from exc
    if workers < 0:
        raise ValueError("DataLoader worker count must be non-negative")
    return workers


@torch.no_grad()
def predict(model, ds, collate, cfg: Config, dev, amp: bool):
    model.eval()
    dl = DataLoader(ds, batch_size=cfg.eval_batch_size, collate_fn=collate)
    logits, labels, tokens = [], [], 0
    for b in dl:
        b = _to(b, dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
        if not torch.isfinite(out).all():
            raise FloatingPointError("Model produced non-finite logits during evaluation.")
        logits.append(out.float().cpu())
        labels.append(b["labels"].cpu())
        tokens += int(b["attention_mask"].sum())
    return torch.cat(logits).numpy(), torch.cat(labels).numpy(), tokens


def non_embedding_params(model) -> int:
    emb = sum(p.numel() for p in model.get_input_embeddings().parameters())
    return sum(p.numel() for p in model.parameters()) - emb


def build_model(cfg: Config):
    return AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=NUM_LABELS)


def _optimizer_parameter_groups(model, lr: float, weight_decay: float, reg: dict[str, float]):
    decay = float(reg.get("llrd", 1.0))
    if not 0.0 < decay <= 1.0:
        raise ValueError("llrd decay must be greater than 0 and at most 1")

    parameters = list(model.named_parameters())
    layer_indices = [
        int(match.group(1))
        for name, _ in parameters
        if (match := re.search(r"(?:^|\.)encoder\.layer\.(\d+)\.", name))
    ]
    layer_count = max(layer_indices, default=-1) + 1
    no_decay = ("bias", "LayerNorm.weight", "layernorm", "LayerNorm.bias")
    grouped = {}
    for name, parameter in parameters:
        layer_match = re.search(r"(?:^|\.)encoder\.layer\.(\d+)\.", name)
        if layer_match:
            lr_scale = decay ** (layer_count - 1 - int(layer_match.group(1)))
        elif re.search(r"(?:^|\.)embeddings\.", name):
            lr_scale = decay ** layer_count
        else:
            lr_scale = 1.0
        decay_value = 0.0 if any(key in name for key in no_decay) else weight_decay
        key = (lr * lr_scale, decay_value)
        grouped.setdefault(key, []).append(parameter)
    return [
        {"params": params, "lr": group_lr, "weight_decay": group_decay}
        for (group_lr, group_decay), params in grouped.items()
    ]


def run(cfg: Config, spec: RunSpec, run_dir: Path, final: bool = False, splits=None,
        model_builder=None, pad_id: int | None = None, record_train_accuracy: bool = False,
    save_resume: bool = True, record_train_clean_loss: bool = False,
    weight_average: str | None = None, ema_decay: float = 0.99,
    epoch_callback: Callable[[dict], None] | None = None) -> dict:
    """Train + evaluate. Idempotent: finished runs are returned from metrics.json, interrupted
    runs resume from the last epoch checkpoint when save_resume is enabled."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if weight_average not in (None, "ema", "swa"):
        raise ValueError("weight_average must be None, 'ema', or 'swa'")
    if not 0.0 < ema_decay < 1.0:
        raise ValueError("ema_decay must be between 0 and 1")
    metrics_path, resume_path = run_dir / "metrics.json", run_dir / "resume.pt"
    chash = config_hash(cfg, spec)
    if weight_average is not None:
        chash = hashlib.sha1(
            f"{chash}:{weight_average}:{ema_decay if weight_average == 'ema' else ''}".encode()
        ).hexdigest()[:12]
    if record_train_accuracy:
        chash = hashlib.sha1(f"{chash}:record_train_accuracy".encode()).hexdigest()[:12]
    if record_train_clean_loss:
        chash = hashlib.sha1(f"{chash}:record_train_clean_loss".encode()).hexdigest()[:12]
    if "augmentation" in spec.reg:
        augmentation_path = Path(cfg.augmentation_file)
        if not augmentation_path.is_file():
            raise FileNotFoundError(f"filtered augmentation file not found: {augmentation_path}")
        file_digest = hashlib.sha256(augmentation_path.read_bytes()).hexdigest()
        chash = hashlib.sha1(f"{chash}:{file_digest}".encode()).hexdigest()[:12]
    if metrics_path.exists():
        done = json.loads(metrics_path.read_text())
        if done.get("config_hash") == chash and (
            not cfg.save_final_model or (run_dir / "model.pt").is_file()
        ) and (not final or (
            "test" in done and (run_dir / "predictions_dev.npz").is_file()
            and (run_dir / "predictions_test.npz").is_file()
        )):
            return done
    (run_dir / "config.json").write_text(json.dumps(
        {"hash": chash, "spec": {"reg": spec.reg, "lr": spec.lr, "seed": spec.seed},
         "cfg": cfg.to_dict(), "effective_num_workers": _dataloader_num_workers(cfg)}, indent=2))

    dev, amp = _device_and_amp(cfg)
    num_workers = _dataloader_num_workers(cfg)
    print(f"[training] effective DataLoader workers={num_workers}", flush=True)
    augmentation_ratio = spec.reg.get("augmentation")
    splits = splits or load_splits(cfg, augmentation_ratio, include_test=final)
    if pad_id is None:
        pad_id = AutoTokenizer.from_pretrained(cfg.model_name).pad_token_id
    collate = make_collator(pad_id)

    set_seed(spec.seed)
    # Keep FP32 master parameters for AdamW; this checkpoint loads in FP16.
    # Autocast still accelerates forward/backward without half-precision optimizer state.
    model = (model_builder or build_model)(cfg).to(device=dev, dtype=torch.float32)
    n_nonemb = non_embedding_params(model)
    groups = _optimizer_parameter_groups(model, spec.lr, cfg.weight_decay, spec.reg)
    opt = torch.optim.AdamW(groups, lr=spec.lr)
    average_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters() if parameter.requires_grad
    } if weight_average == "ema" else {}
    swa_count = 0
    steps_per_epoch = len(splits["train"]) // cfg.batch_size
    total = steps_per_epoch * cfg.epochs
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * total), total)

    state = {"epoch": 0, "train_seconds": 0.0, "train_flops": 0.0, "history": []}
    if save_resume and resume_path.exists():
        try:
            ck = torch.load(resume_path, map_location="cpu", weights_only=False)
        except (EOFError, OSError, RuntimeError, pickle.UnpicklingError, zipfile.BadZipFile) as exc:
            print(f"[checkpoint] ignoring unreadable resume file {resume_path}: {exc}", flush=True)
            resume_path.unlink()
        else:
            if ck["config_hash"] == chash:
                model.load_state_dict(ck["model"])
                opt.load_state_dict(ck["opt"])
                sched.load_state_dict(ck["sched"])
                state = ck["state"]

    for epoch in range(state["epoch"], cfg.epochs):
        # per-epoch seeding makes data order / dropout / mixup / noise resume-deterministic
        seed_e = spec.seed * 1000 + epoch
        torch.manual_seed(seed_e)
        g = torch.Generator().manual_seed(seed_e)
        dl = DataLoader(splits["train"], batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                        collate_fn=collate, generator=g, num_workers=num_workers)
        model.train()
        t0, losses, tokens = time.time(), [], 0
        for step, b in enumerate(dl, start=1):
            b = _to(b, dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                loss, logits, passes = training_loss(model, b, spec.reg, NUM_LABELS)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch + 1}, step {step}; "
                    "check precision/model outputs."
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite gradient norm at epoch {epoch + 1}, step {step}.")
            opt.step()
            sched.step()
            if weight_average == "ema":
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        average_parameters[name].mul_(ema_decay).add_(
                            parameter.detach(), alpha=1.0 - ema_decay
                        )
            losses.append(loss.item())
            tokens += int(b["attention_mask"].sum()) * passes
        if dev.type == "cuda":
            torch.cuda.synchronize()
        state["train_seconds"] += time.time() - t0
        state["train_flops"] += 6.0 * n_nonemb * tokens  # fwd+bwd ~ 6*N*tokens (estimate)
        if weight_average == "swa":
            swa_count += 1
            if not average_parameters:
                average_parameters = {
                    name: parameter.detach().clone()
                    for name, parameter in model.named_parameters() if parameter.requires_grad
                }
            else:
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        average_parameters[name].add_(parameter.detach())
        original_parameters = {}
        if weight_average is not None and average_parameters:
            original_parameters = {
                name: parameter.detach().clone()
                for name, parameter in model.named_parameters() if parameter.requires_grad
            }
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        averaged = average_parameters[name]
                        if weight_average == "swa":
                            averaged = averaged / swa_count
                        parameter.copy_(averaged)
        try:
            lg, y, _ = predict(model, splits["dev"], collate, cfg, dev, amp)
            m = compute_metrics(lg, y, cfg.ece_bins)
        finally:
            if original_parameters:
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        if parameter.requires_grad:
                            parameter.copy_(original_parameters[name])
        entry = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                 "epoch_train_seconds": time.time() - t0,
                 **{f"dev_{k}": v for k, v in m.items()}}
        if weight_average is not None:
            entry["weight_average"] = weight_average
        if record_train_accuracy or record_train_clean_loss:
            train_logits, train_labels, _ = predict(model, splits["train"], collate, cfg, dev, amp)
            clean_train = compute_metrics(train_logits, train_labels, cfg.ece_bins)
            if record_train_accuracy:
                entry["train_accuracy"] = clean_train["accuracy"]
            if record_train_clean_loss:
                entry["train_clean_loss"] = clean_train["loss"]
        state["history"].append(entry)
        state["epoch"] = epoch + 1
        if epoch_callback is not None:
            epoch_callback(dict(entry))
        if save_resume:
            temporary_resume = resume_path.with_suffix(resume_path.suffix + ".tmp")
            try:
                torch.save(
                    {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                     "state": state, "config_hash": chash},
                    temporary_resume,
                )
                temporary_resume.replace(resume_path)
            finally:
                temporary_resume.unlink(missing_ok=True)

    result = {"config_hash": chash, "reg": spec.reg, "lr": spec.lr, "seed": spec.seed,
              "epochs": cfg.epochs,
              "effective_num_workers": num_workers,
              "history": state["history"],
              "train_seconds": state["train_seconds"],
              "train_flops_est": state["train_flops"]}
    if "augmentation_info" in splits:
        result["augmentation_info"] = splits["augmentation_info"]
    original_parameters = {}
    if weight_average is not None and average_parameters:
        original_parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    averaged = average_parameters[name]
                    if weight_average == "swa":
                        averaged = averaged / swa_count
                    parameter.copy_(averaged)
    lg, y, dev_tokens = predict(model, splits["dev"], collate, cfg, dev, amp)
    result["dev"] = compute_metrics(lg, y, cfg.ece_bins)
    if original_parameters:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(original_parameters[name])
    result["infer_flops_per_dev_est"] = 2.0 * n_nonemb * dev_tokens
    if final:
        for name in ("dev", "test"):
            lg, y, _ = predict(model, splits[name], collate, cfg, dev, amp)
            result[name] = compute_metrics(lg, y, cfg.ece_bins)
            np.savez_compressed(run_dir / f"predictions_{name}.npz", logits=lg, labels=y,
                                preds=lg.argmax(-1))
    if cfg.save_final_model:
        torch.save(model.state_dict(), run_dir / "model.pt")
    metrics_path.write_text(json.dumps(result, indent=2))
    if save_resume:
        resume_path.unlink(missing_ok=True)
    return result
