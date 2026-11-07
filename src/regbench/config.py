"""Experiment configuration: one dataclass, loadable from YAML, serialisable per run."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Canonical order is only used for naming; ranking decides the stacking order.
REGULARIZERS = ("augmentation", "label_smoothing", "mixup", "rdrop", "neftune")


@dataclass
class Grid:
    lrs: list[float] = field(default_factory=lambda: [1e-5, 2e-5, 3e-5])
    baseline_epochs: list[int] = field(default_factory=lambda: [3, 4, 5])
    augmentation: list[float] = field(default_factory=lambda: [0.25, 0.50, 1.00])
    label_smoothing: list[float] = field(default_factory=lambda: [0.05, 0.10, 0.15])
    mixup: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.4])
    rdrop: list[float] = field(default_factory=lambda: [0.5, 1.0, 4.0])
    neftune: list[float] = field(default_factory=lambda: [5.0, 10.0, 15.0])


@dataclass
class Config:
    # model / data
    model_name: str = "microsoft/deberta-v3-base"
    dataset_name: str = "nyu-mll/glue"
    dataset_config: str = "rte"
    premise_column: str = "sentence1"
    hypothesis_column: str = "sentence2"
    max_len: int = 128
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    split_seed: int = 13
    augmentation_seed: int = 2718
    augmentation_generator: str = "google/flan-t5-large"
    augmentation_confidence: float = 0.85
    augmentation_batch_size: int = 8
    augmentation_max_new_tokens: int = 96
    augmentation_file: str = "outputs/augmentation/filtered.jsonl"
    train_size: int | None = None  # subsample the remaining train split (budget runs)
    dev_eval_size: int | None = None  # optionally truncate validation for speed
    # optimisation (identical for every method)
    lr: float = 2e-5
    weight_decay: float = 0.01
    batch_size: int = 32
    eval_batch_size: int = 128
    epochs: int = 3
    warmup_ratio: float = 0.06
    grad_clip: float = 1.0
    precision: str = "auto"      # auto | fp32 | bf16
    num_workers: int = 2
    # seeds
    search_seed: int = 42
    final_seeds: list[int] = field(default_factory=lambda: [42, 123, 456])
    # search grid
    grid: Grid = field(default_factory=Grid)
    # final-eval options
    save_final_model: bool = False
    test_size: int | None = None  # truncate test (smoke tests only)
    ece_bins: int = 10

    def __post_init__(self):
        fractions = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if any(value <= 0 or value >= 1 for value in fractions):
            raise ValueError("train/validation/test fractions must each be between 0 and 1")
        if abs(sum(fractions) - 1.0) > 1e-8:
            raise ValueError("train/validation/test fractions must sum to 1")

    @staticmethod
    def load(path: str | Path | None, overrides: dict | None = None) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) if path else {}
        raw = {**(raw or {}), **(overrides or {})}
        grid = Grid(**raw.pop("grid", {}))
        return Config(grid=grid, **raw)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class RunSpec:
    """A single training run: shared config + regularizer settings + lr + seed."""

    reg: dict[str, float]   # e.g. {"mixup": 0.2, "rdrop": 1.0}; empty == baseline
    lr: float
    seed: int

    def key(self) -> str:
        parts = [f"{k}{v:g}" for k, v in sorted(self.reg.items())] or ["baseline"]
        return "_".join(parts) + f"_lr{self.lr:g}_s{self.seed}"


def config_hash(cfg: Config, spec: RunSpec) -> str:
    blob = json.dumps({"cfg": cfg.to_dict(), "spec": dataclasses.asdict(spec)}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]
