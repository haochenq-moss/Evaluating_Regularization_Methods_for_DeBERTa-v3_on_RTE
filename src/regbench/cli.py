from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from . import data_aug, search
from .config import REGULARIZERS, Config
from .report import report


def _parse_overrides(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = json.loads(v)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="regbench")
    p.add_argument("stage", choices=[
        "prepare", "baseline", "augment", "individual", "stack", "final", "report", "all"
    ])
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out", default="outputs")
    p.add_argument("--order", help="comma-separated stacking order for an alternative-order analysis, "
                                   f"e.g. augmentation,rdrop,mixup,neftune,label_smoothing (from {REGULARIZERS})")
    p.add_argument("--set", nargs="*", metavar="KEY=JSON", help='override config, e.g. epochs=1 train_size=2000')
    a = p.parse_args(argv)

    cfg = Config.load(a.config, _parse_overrides(a.set))
    out = Path(a.out)
    cfg = replace(cfg, augmentation_file=str(out / "augmentation" / "filtered.jsonl"))
    out.mkdir(parents=True, exist_ok=True)
    if a.stage == "prepare":
        from .data import prepare
        print("tokenized data cached at", prepare(cfg))
        return
    order = a.order.split(",") if a.order else None
    if order and sorted(order) != sorted(REGULARIZERS):
        p.error("--order must be a permutation of " + ",".join(REGULARIZERS))
    if a.stage == "baseline":
        search.baseline(cfg, out)
        return
    if a.stage == "augment":
        weights = out / "baseline_for_augmentation" / "model.pt"
        data_aug.generate_and_filter(cfg, out / "augmentation", weights)
        return
    if a.stage == "individual":
        search.individual(cfg, out)
        return
    if a.stage in ("stack", "all"):
        if a.stage == "all":
            search.baseline(cfg, out)
            data_aug.generate_and_filter(
                cfg, out / "augmentation", out / "baseline_for_augmentation" / "model.pt"
            )
            search.individual(cfg, out)
        search.stack(cfg, out, order)
    tag = search.stack_tag(out, order) if a.stage in ("stack", "final", "all") else "ranked"
    if a.stage in ("final", "all"):
        search.final(cfg, out, tag)
    if a.stage in ("report", "all"):
        report(out, cfg.ece_bins)
        print(f"reports written to {out / 'reports'}")


if __name__ == "__main__":
    main()
