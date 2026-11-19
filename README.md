# DeBERTa-v3 Regularization on RTE

This repository contains a 12-epoch, seven-seed study of regularization for full fine-tuning of DeBERTa-v3-base on GLUE RTE. It compares a baseline with augmentation, label smoothing, Mixup, NEFTune, R-Drop and layer-wise learning-rate decay (LLRD), then evaluates an Optimized Combination of selected methods.

## Study protocol

- The 2,490 labeled RTE training examples are grouped by identical premise-hypothesis pair and stratified into 1,742 training, 374 development and 374 internal holdout examples. The split seed is 13. Inputs are tokenized with the DeBERTa-v3-base tokenizer and truncated to 128 tokens.
- All reported runs use 12 epochs, learning rate `3e-5`, batch size 32, weight decay `0.01`, and seeds `123`, `456`, `789`, `1011`, `201765793`, `1801564980` and `1323889826`.
- Three values per method are compared by mean final development accuracy; ties are broken by lower mean development loss, then the smaller parameter value. A method enters the Optimized Combination only if its mean paired development gain is positive and it improves over baseline in at least four of seven seeds.
- The selected combination uses label smoothing `0.1`, Mixup `0.2`, R-Drop `0.1` and LLRD decay `0.8`. Augmentation and NEFTune did not meet the inclusion rule.
- The internal holdout was examined in previous project analyses. Its comparison is exploratory, not an independent test; the official RTE validation split is not used in this comparison.

On development, the Optimized Combination averaged 86.40% accuracy versus 84.19% for baseline. On the exploratory internal holdout, it averaged 84.64% versus 80.33%, a mean paired gain of 4.32 percentage points; all seven seeds improved.

## Repository layout

```text
configs/       Training and data configuration
docs/          Data protocol
src/regbench/  Data preparation, model training and regularization
scripts/       Study, held-out evaluation and Slurm scripts
tests/         Training and regularization tests
outputs/       Final study results, augmentation pool and report
```

The report is `outputs/overfitting_study/report/combined_report.pdf`. The saved candidate runs and selection are in `outputs/regularization_selection_curves/`; held-out results are in `outputs/heldout_evaluation/`. Some matched baseline, R-Drop and augmentation records are reused from their original run directories. These final artifacts and the filtered augmentation pool are available to add to version control; caches, logs and checkpoints remain ignored.

## Setup and execution

Use a Python environment with a compatible CUDA-enabled PyTorch installation. The study was run with Python 3.12 and PyTorch 2.11 with CUDA 12.8 on an NVIDIA A40.

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
mkdir -p outputs/slurm
sbatch scripts/regularization_curves.slurm
```

After the candidate study finishes, submit `scripts/heldout_evaluation.slurm` to evaluate the frozen selection. Slurm settings are cluster-specific; adjust them before submission. Set `REGBENCH_PYTHON` if the intended Python interpreter is not the project `.venv/bin/python` or available as `python3` on `PATH`. The held-out job also requires Tectonic on `PATH`, or its path in `REGBENCH_TECTONIC`.

To regenerate curves and tables from complete saved runs without retraining:

```bash
PYTHONPATH=src python scripts/run_regularization_curve_study.py --plots-only
```

The report requires its generated tables, figures, and LaTeX style files in the report directory. Compile `combined_report.tex` with a suitable LaTeX installation, for example Tectonic. See [the data protocol](docs/data-splits.md) for split and reproducibility details.