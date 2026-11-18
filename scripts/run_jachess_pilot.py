#!/usr/bin/env python3
"""Profile a reduced JacHess estimator pass on one unlabeled RTE batch."""
from __future__ import annotations

import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

from regbench.config import Config
from regbench.data import load_splits, make_collator
from regbench.trainer import build_model
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "regularization_extensions" / "jachess_pilot"
PILOT_EXAMPLES = 4
PROJECTIONS_PER_LAYER = 1
HESSIAN_SAMPLES_PER_LAYER = 1


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append_progress(row: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def prepare_embeddings(model, examples: list[dict], pad_id: int,
                       device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    batch = make_collator(pad_id)(examples)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    embeddings = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
    return embeddings, attention_mask


def estimate_layer_norms(model, embeddings: torch.Tensor, attention_mask: torch.Tensor,
                         projections: int, hessian_samples: int) -> list[dict]:
    output = model(
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    rows = []
    for layer_number, hidden in enumerate(output.hidden_states[1:], start=1):
        started = time.monotonic()
        representation = hidden[:, 0, :]
        jacobian_estimates = []
        for _ in range(projections):
            probe = torch.randn_like(representation)
            jacobian_vector = torch.autograd.grad(
                (representation * probe).sum(), embeddings,
                retain_graph=True, create_graph=True,
            )[0]
            jacobian_estimates.append(jacobian_vector.float().square().sum())

        hessian_estimates = []
        for _ in range(hessian_samples):
            feature = int(torch.randint(representation.size(-1), ()).item())
            first_derivative = torch.autograd.grad(
                representation[:, feature].sum(), embeddings,
                retain_graph=True, create_graph=True,
            )[0]
            probe = torch.randn_like(first_derivative)
            hessian_vector = torch.autograd.grad(
                first_derivative, embeddings, grad_outputs=probe,
                retain_graph=True, create_graph=False,
            )[0]
            hessian_estimates.append(hessian_vector.float().square().sum())

        row = {
            "layer": layer_number,
            "jacobian_frobenius_sq_estimate": torch.stack(jacobian_estimates).mean().item(),
            "hessian_frobenius_sq_estimate": torch.stack(hessian_estimates).mean().item(),
            "elapsed_seconds": time.monotonic() - started,
        }
        rows.append(row)
        append_progress({"timestamp": datetime.now(timezone.utc).isoformat(), **row})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise SystemExit("Run this pilot inside a Slurm GPU allocation.")
    started = time.monotonic()
    status_path = OUT / "status.json"
    write_json(status_path, {"state": "running", "started_at": datetime.now(timezone.utc).isoformat()})
    try:
        cfg = Config.load(ROOT / "configs/default.yaml")
        splits = load_splits(cfg)
        if len(splits["train"]) != 1742:
            raise ValueError("Unexpected RTE train split size.")
        model = build_model(cfg).to(device="cuda", dtype=torch.float32)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        examples = [splits["train"][index] for index in range(PILOT_EXAMPLES)]
        embeddings, attention_mask = prepare_embeddings(
            model, examples, tokenizer.pad_token_id, torch.device("cuda")
        )
        torch.cuda.reset_peak_memory_stats()
        layer_stats = estimate_layer_norms(
            model, embeddings, attention_mask,
            PROJECTIONS_PER_LAYER, HESSIAN_SAMPLES_PER_LAYER,
        )
        torch.cuda.synchronize()
        summary = {
            "purpose": "JacHess estimator feasibility/timing pilot; not a trained or scored model",
            "method_reference": "Jukić and Šnajder, TACL 2025, doi:10.1162/tacl_a_00739",
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "gpu": torch.cuda.get_device_name(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "examples": PILOT_EXAMPLES,
            "projections_per_layer": PROJECTIONS_PER_LAYER,
            "hessian_samples_per_layer": HESSIAN_SAMPLES_PER_LAYER,
            "layer_stats": layer_stats,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            "wall_seconds": time.monotonic() - started,
            "model_saving": False,
        }
        write_json(OUT / "summary.json", summary)
        write_json(status_path, {
            "state": "completed", "updated_at": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": summary["wall_seconds"],
        })
    except Exception as error:
        write_json(status_path, {
            "state": "failed", "updated_at": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(error).__name__}: {error}",
        })
        raise


if __name__ == "__main__":
    main()