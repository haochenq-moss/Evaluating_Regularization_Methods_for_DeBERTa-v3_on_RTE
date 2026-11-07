"""Grouped, stratified RTE partitions plus the official validation evaluation set."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer

from .config import Config


def _tokenize(cfg: Config, ds, tok):
    def fn(ex):
        return tok(ex[cfg.premise_column], ex[cfg.hypothesis_column],
                   truncation=True, max_length=cfg.max_len)

    ds = ds.map(fn, batched=True, desc="tokenize")
    keep = ["input_ids", "attention_mask", "label",
            cfg.premise_column, cfg.hypothesis_column]
    return ds.remove_columns([c for c in ds.column_names if c not in keep]).rename_column("label", "labels")


def cache_path(cfg: Config) -> Path:
    root = Path(os.environ.get("REGBENCH_DATA", "data_cache"))
    ratios = (round(cfg.train_fraction * 100), round(cfg.validation_fraction * 100),
              round(cfg.test_fraction * 100))
    dataset_tag = f"{cfg.dataset_name.replace('/', '_')}_{cfg.dataset_config}"
    tag = (f"v2_{dataset_tag}_{cfg.model_name.replace('/', '_')}_L{cfg.max_len}_"
           f"split{ratios[0]}-{ratios[1]}-{ratios[2]}_seed{cfg.split_seed}")
    return root / tag


def _grouped_split_indices(dataset, cfg: Config) -> dict[str, list[int]]:
    """Split by unique premise/hypothesis group so duplicate pairs cannot cross partitions."""
    group_by_pair: dict[tuple[str, str], int] = {}
    first_rows: list[int] = []
    row_groups: list[int] = []
    for row, (premise, hypothesis) in enumerate(
        zip(dataset[cfg.premise_column], dataset[cfg.hypothesis_column])
    ):
        pair = (premise, hypothesis)
        group_id = group_by_pair.get(pair)
        if group_id is None:
            group_id = len(first_rows)
            group_by_pair[pair] = group_id
            first_rows.append(row)
        row_groups.append(group_id)

    groups = dataset.select(first_rows).select_columns(["label"])
    groups = groups.add_column("group_id", list(range(len(groups))))
    holdout_fraction = cfg.validation_fraction + cfg.test_fraction
    first = groups.train_test_split(
        test_size=holdout_fraction, seed=cfg.split_seed, stratify_by_column="label"
    )
    validation_test = first["test"].train_test_split(
        test_size=cfg.test_fraction / holdout_fraction,
        seed=cfg.split_seed + 1,
        stratify_by_column="label",
    )
    group_ids = {
        "train": set(first["train"]["group_id"]),
        "validation": set(validation_test["train"]["group_id"]),
        "test": set(validation_test["test"]["group_id"]),
    }
    if any(group_ids[a] & group_ids[b] for a, b in
           (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise RuntimeError("group leakage detected between RTE partitions")
    result = {name: [] for name in group_ids}
    for row, group_id in enumerate(row_groups):
        for name, ids in group_ids.items():
            if group_id in ids:
                result[name].append(row)
                break
    if sum(map(len, result.values())) != len(dataset):
        raise RuntimeError("some RTE rows were not assigned to a partition")
    return result


def prepare(cfg: Config, force: bool = False) -> Path:
    """Cache 70:15:15 partitions of labeled train and official validation as final test."""
    path = cache_path(cfg)
    if path.exists() and not force:
        return path
    if path.exists():
        shutil.rmtree(path)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    path.parent.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    raw = load_dataset(cfg.dataset_name, cfg.dataset_config)
    source = raw["train"]
    row_indices = _grouped_split_indices(source, cfg)
    internal = DatasetDict({
        "train": _tokenize(cfg, source.select(row_indices["train"]), tok),
        "validation": _tokenize(cfg, source.select(row_indices["validation"]), tok),
        "internal_holdout": _tokenize(cfg, source.select(row_indices["test"]), tok),
        "test": _tokenize(cfg, raw["validation"], tok),
    })
    internal.save_to_disk(str(tmp))
    sizes = {name: len(indices) for name, indices in row_indices.items()}
    total = sum(sizes.values())
    manifest = {
        "source": cfg.dataset_name,
        "dataset_config": cfg.dataset_config,
        "source_split": "train",
        "final_test_source_split": "validation",
        "seed": cfg.split_seed,
        "requested_fractions": {
            "train": cfg.train_fraction,
            "validation": cfg.validation_fraction,
            "test": cfg.test_fraction,
        },
        "sizes": sizes,
        "official_validation_size": len(raw["validation"]),
        "official_validation_used_as_final_test": True,
        "actual_percentages": {name: 100 * size / total for name, size in sizes.items()},
        "unique_text_pairs": len(set(zip(
            source[cfg.premise_column], source[cfg.hypothesis_column]
        ))),
        "duplicate_rows_kept_with_their_text_pair_group": len(source) - len(
            set(zip(source[cfg.premise_column], source[cfg.hypothesis_column]))
        ),
        "stratified_by": source.features["label"].names,
        "source_row_indices": row_indices,
    }
    (tmp / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    tmp.rename(path)
    return path


def load_splits(cfg: Config, augmentation_ratio: float | None = None,
                include_test: bool = False) -> dict:
    ds = load_from_disk(str(prepare(cfg)))
    limits = {"train": cfg.train_size, "dev": cfg.dev_eval_size, "test": cfg.test_size}
    aliases = {"train": "train", "dev": "validation"}
    if include_test:
        aliases["test"] = "test"
    splits = {}
    for name, source in aliases.items():
        limit = limits[name]
        splits[name] = (ds[source].select(range(min(limit, len(ds[source]))))
                        if limit else ds[source])
    if augmentation_ratio is not None:
        from .data_aug import append_filtered_augmentation

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        splits["train"], augmentation_info = append_filtered_augmentation(
            cfg, splits["train"], augmentation_ratio, Path(cfg.augmentation_file), tokenizer
        )
        splits["augmentation_info"] = augmentation_info
    return splits


def make_collator(pad_id: int):
    def collate(features):
        n = max(len(f["input_ids"]) for f in features)
        ids = torch.full((len(features), n), pad_id, dtype=torch.long)
        mask = torch.zeros((len(features), n), dtype=torch.long)
        for i, f in enumerate(features):
            L = len(f["input_ids"])
            ids[i, :L] = torch.tensor(f["input_ids"])
            mask[i, :L] = 1
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    return collate
