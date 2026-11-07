"""Run one small, real-model GPU timing pilot and save its metrics."""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import torch

from regbench.config import Config, RunSpec
from regbench.trainer import run


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; this pilot must run inside a GPU Slurm allocation.")
    if torch.cuda.device_count() != 1:
        raise SystemExit(f"Expected one Slurm-visible GPU, found {torch.cuda.device_count()}.")

    root = Path(os.environ["SLURM_SUBMIT_DIR"])
    cfg = Config.load("configs/default.yaml", {
        "train_size": 512,
        "dev_eval_size": 256,
        "test_size": 256,
        "epochs": 1,
        "batch_size": 16,
        "eval_batch_size": 64,
        "num_workers": 0,
    })
    device = torch.cuda.get_device_name(0)
    print(json.dumps({
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": device,
        "train_examples": cfg.train_size,
        "validation_examples": cfg.dev_eval_size,
        "smoke_test_examples": cfg.test_size,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
    }, indent=2), flush=True)

    started = time.perf_counter()
    result = run(
        cfg,
        RunSpec(reg={}, lr=2e-5, seed=cfg.search_seed),
        root / "outputs" / "pilot" / "gpu-baseline-fp32master-bf16",
        final=True,
    )
    elapsed = time.perf_counter() - started
    result["pilot"] = {
        "host": platform.node(),
        "gpu": device,
        "elapsed_seconds_including_model_load_eval": elapsed,
        "training_examples": cfg.train_size,
        "training_examples_per_second": cfg.train_size / result["train_seconds"],
        "training_seconds": result["train_seconds"],
        "full_train_rows": 274889,
        "full_train_epochs": 3,
        "full_train_training_only_estimate_hours": (
            3 * 274889 / (cfg.train_size / result["train_seconds"]) / 3600
        ),
    }
    path = root / "outputs" / "pilot" / "gpu-baseline-fp32master-bf16" / "metrics.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps({"pilot": result["pilot"], "validation": result["dev"],
                      "test_smoke_only": result["test"], "metrics_file": str(path)}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
