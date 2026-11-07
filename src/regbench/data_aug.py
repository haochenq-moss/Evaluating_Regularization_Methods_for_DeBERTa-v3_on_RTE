"""Reproducible FLAN-T5 hypothesis generation and baseline-confidence filtering."""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from datasets import Dataset, load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer, T5ForConditionalGeneration

from .config import Config
from .data import cache_path
from .trainer import NUM_LABELS, _device_and_amp

TASK_PROMPT = """Task Definition:
Given a premise and a hypothesis, decide whether the premise entails the hypothesis.
The only labels are entailment and not_entailment.

Positive example (preserve the label):
Premise: A person is riding a bicycle on the street.
Hypothesis: Someone is cycling outdoors.
Label: entailment
Instruction: Write a short paraphrase of the hypothesis that remains entailed by the premise.
Hypothesis:
Someone is riding a bike outside.

Negative example (flip the label with a minimal edit):
Premise: A person is riding a bicycle on the street.
Hypothesis: The person is driving a car.
Label: not_entailment
Instruction: Change as little as possible in the hypothesis so the premise entails it.
Hypothesis:
The person is riding a bicycle.

Now create one hypothesis for this example.
Premise: {premise}
Original hypothesis: {hypothesis}
Original label: {label}
Instruction: {instruction}
Return only the new hypothesis text."""


def _prompt(row: dict, preserve: bool) -> str:
    if preserve:
        instruction = "Paraphrase while preserving the original entailment label."
    else:
        instruction = (
            "Make a minimal edit so the correct label flips to the other label. "
            "Keep the new hypothesis plausible and grammatical."
        )
    label = "entailment" if row["labels"] == 0 else "not_entailment"
    return TASK_PROMPT.format(
        premise=row["sentence1"], hypothesis=row["sentence2"],
        label=label, instruction=instruction,
    )


def _clean_generated(text: str, original: str) -> str:
    text = " ".join(text.strip().split()).strip('"\'` ')
    for prefix in ("Hypothesis:", "New hypothesis:", "Output:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    return text if text and text != original else ""


def generate_candidates(cfg: Config, out: Path, log=print) -> Path:
    """Generate one deterministic preserve-or-flip hypothesis for each internal train row."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_from_disk(str(cache_path(cfg)))["train"]
    tok = AutoTokenizer.from_pretrained(cfg.augmentation_generator)
    generator = T5ForConditionalGeneration.from_pretrained(cfg.augmentation_generator)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device).eval()
    prompts, metadata = [], []
    rng = random.Random(cfg.augmentation_seed)
    for i, row in enumerate(ds):
        preserve = bool(rng.getrandbits(1))
        prompts.append(_prompt(row, preserve))
        target = int(row["labels"]) if preserve else 1 - int(row["labels"])
        metadata.append({
            "source_index": i,
            "sentence1": row["sentence1"],
            "original_sentence2": row["sentence2"],
            "preserve": preserve,
            "target_label": target,
        })

    generated = []
    for start in range(0, len(prompts), cfg.augmentation_batch_size):
        encoded = tok(
            prompts[start:start + cfg.augmentation_batch_size],
            return_tensors="pt", padding=True, truncation=True, max_length=1024,
        ).to(device)
        with torch.inference_mode():
            outputs = generator.generate(
                **encoded, do_sample=False, num_beams=1,
                max_new_tokens=cfg.augmentation_max_new_tokens,
            )
        generated.extend(tok.batch_decode(outputs, skip_special_tokens=True))
        if (start // cfg.augmentation_batch_size) % 25 == 0:
            log(f"[augmentation] generated {min(start + len(outputs), len(prompts))}/{len(prompts)}")

    candidates = []
    for meta, text in zip(metadata, generated):
        hypothesis = _clean_generated(text, meta["original_sentence2"])
        if not hypothesis:
            continue
        candidates.append({
            "sentence1": meta["sentence1"],
            "sentence2": hypothesis,
            "labels": meta["target_label"],
            "source_index": meta["source_index"],
            "preserve": meta["preserve"],
        })
    path = out / "candidates.jsonl"
    with path.open("w") as f:
        for row in candidates:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "generation.json").write_text(json.dumps({
        "generator": cfg.augmentation_generator,
        "seed": cfg.augmentation_seed,
        "source_training_rows": len(ds),
        "generated_candidates": len(candidates),
        "generation_failures": len(ds) - len(candidates),
    }, indent=2))
    return path


@torch.inference_mode()
def filter_candidates(cfg: Config, candidate_path: Path, baseline_weights: Path, out: Path,
                      log=print) -> Path:
    """Keep only synthetic training pairs the tuned baseline predicts with >= threshold confidence."""
    out = Path(out)
    candidate_path, baseline_weights = Path(candidate_path), Path(baseline_weights)
    candidates = [json.loads(line) for line in candidate_path.read_text().splitlines() if line.strip()]
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=NUM_LABELS)
    state = torch.load(baseline_weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    device, amp = _device_and_amp(cfg)
    model.to(device=device, dtype=torch.float32).eval()
    kept, scored = [], []
    for start in range(0, len(candidates), cfg.eval_batch_size):
        batch = candidates[start:start + cfg.eval_batch_size]
        encoded = tok(
            [r["sentence1"] for r in batch], [r["sentence2"] for r in batch],
            return_tensors="pt", padding=True, truncation=True, max_length=cfg.max_len,
        ).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(**encoded).logits.float()
        probs = torch.softmax(logits, dim=-1)
        for row, p in zip(batch, probs.cpu().tolist()):
            confidence = float(p[row["labels"]])
            record = {**row, "baseline_confidence": confidence}
            scored.append(record)
            if int(max(range(len(p)), key=p.__getitem__)) == row["labels"] and confidence >= cfg.augmentation_confidence:
                kept.append(record)
        if start % (cfg.eval_batch_size * 20) == 0:
            log(f"[augmentation filter] scored {min(start + len(batch), len(candidates))}/{len(candidates)}")
    out.mkdir(parents=True, exist_ok=True)
    filtered_path = out / "filtered.jsonl"
    with filtered_path.open("w") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "filter_metrics.json").write_text(json.dumps({
        "confidence_threshold": cfg.augmentation_confidence,
        "candidate_count": len(candidates),
        "retained_count": len(kept),
        "retention_fraction": len(kept) / len(candidates) if candidates else 0.0,
    }, indent=2))
    return filtered_path


def generate_and_filter(cfg: Config, out: Path, baseline_weights: Path, log=print) -> Path:
    out = Path(out)
    candidates = generate_candidates(cfg, out, log)
    return filter_candidates(cfg, candidates, baseline_weights, out, log)


def append_filtered_augmentation(cfg: Config, train: Dataset, ratio: float,
                                 filtered_path: Path, tokenizer) -> tuple[Dataset, dict]:
    """Append a deterministic ratio of retained synthetic pairs to originals, never eval data."""
    if not 0 < ratio <= 1:
        raise ValueError(f"augmentation ratio must be in (0, 1], got {ratio}")
    rows = [json.loads(line) for line in Path(filtered_path).read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(
            "The confidence filter retained no synthetic examples; augmentation trials cannot run."
        )
    desired = round(len(train) * ratio)
    rng = random.Random(cfg.augmentation_seed + round(ratio * 10_000))
    selected = rng.sample(rows, min(desired, len(rows)))
    raw = Dataset.from_list([
        {cfg.premise_column: r["sentence1"], cfg.hypothesis_column: r["sentence2"],
         "label": int(r["labels"])}
        for r in selected
    ])
    synthetic = raw.map(
        lambda ex: tokenizer(ex[cfg.premise_column], ex[cfg.hypothesis_column],
                             truncation=True, max_length=cfg.max_len),
        batched=True,
        remove_columns=[cfg.premise_column, cfg.hypothesis_column],
        desc="tokenize filtered augmentation",
    ).rename_column("label", "labels")
    synthetic = synthetic.remove_columns([c for c in synthetic.column_names
                                          if c not in ("input_ids", "attention_mask", "labels")])
    combined = Dataset.from_dict({
        "input_ids": list(train["input_ids"]) + list(synthetic["input_ids"]),
        "attention_mask": list(train["attention_mask"]) + list(synthetic["attention_mask"]),
        "labels": list(train["labels"]) + list(synthetic["labels"]),
    })
    return combined, {
        "requested_ratio": ratio,
        "requested_synthetic_count": desired,
        "available_filtered_count": len(rows),
        "used_synthetic_count": len(selected),
        "achieved_ratio": len(selected) / len(train),
    }
