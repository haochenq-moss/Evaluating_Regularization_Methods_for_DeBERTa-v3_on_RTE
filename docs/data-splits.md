# Data splits, augmentation boundary and evaluation

## Source and partition policy

- Dataset: `nyu-mll/glue`, `rte` configuration, source `train` split (2,490 labeled rows).
- Task fields: `sentence1` (premise), `sentence2` (hypothesis); class IDs are entailment `0`, not-entailment `1`.
- Requested internal partition: 70% training, 15% validation, 15% holdout.
- Label balance: two-stage holdout splitting is stratified by the RTE label.
- Leakage prevention: identical `(sentence1, sentence2)` pairs are grouped together, including conflicting-label duplicates; no source rows are dropped.
- Seed: `split_seed` in the YAML configuration (default `13`).
- The internal 15% holdout is reserved and unused. The official GLUE RTE `validation` split (277 labeled rows) is the final evaluation set, used only after all choices and retraining are complete.

The internal split first holds out the combined validation/holdout fraction, then divides that holdout according to the requested relative sizes. Grouping may slightly change realized percentages; the generated `split_manifest.json` is authoritative.

The prepared internal split (seed 13) has:

| Partition | Rows | Actual share |
|---|---:|---:|
| Train | 1,742 | 69.9598% |
| Validation | 374 | 15.0201% |
| Test | 374 | 15.0201% |

The final row is named `internal_holdout` in the cache (not `test`) to avoid confusing it with the official final evaluation split. All 2,490 source rows are assigned once; the two duplicate rows remain with their text-pair group.

## Reproduce and inspect

```bash
regbench prepare --config configs/default.yaml
python -m json.tool data_cache/*/split_manifest.json
```

The manifest records source row membership, exact sizes, realized percentages, seed and requested ratios, plus the official validation size and its final-test role. Data is tokenized with the configured model tokenizer, sentence1 first and sentence2 second, and truncated to `max_len`. Original text is retained for augmentation and audit.

## Evaluation boundary

Only internal train rows are augmented. Internal validation is used for tuning, ranking and epoch monitoring. The official validation data is not loaded by training or search; it is loaded for final selected retraining/evaluation only. The internal holdout is not used. Final predictions are saved separately from internal validation predictions.

## Synthetic data

FLAN-T5 generation is seeded and limited to internal training rows. Each row produces either a label-preserving hypothesis paraphrase or a minimal-edit counterfactual with the opposite binary label. The selected baseline model filters candidates by predicted target label and confidence threshold `0.85`. Each augmentation ratio requests that fraction of the original training size; selected synthetic rows are deterministic and appended only to train. If filtering yields fewer rows than requested, the actual synthetic count and achieved ratio are recorded in run metrics.
