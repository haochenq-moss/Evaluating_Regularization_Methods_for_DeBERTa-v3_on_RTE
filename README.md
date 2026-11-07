# DeBERTa-v3 GLUE RTE augmentation and regularization benchmark

Reproducible PyTorch/Hugging Face experiments comparing a standard AdamW baseline with training-only FLAN-T5 hypothesis augmentation, label smoothing, embedding mixup, R-Drop and NEFTune, individually and in cumulative stacks.

## Repository layout

```text
configs/       Full, budget and smoke experiment configurations
data_cache/    Generated tokenized data and split manifest (git-ignored)
docs/          Data protocol and contributor workflow
outputs/       Checkpoints, metrics, predictions and plots (git-ignored)
src/regbench/  Data, model training, regularizers, search, metrics and reports
tests/         Regularizer correctness and experiment workflow tests
```

## Data protocol

Source: [`nyu-mll/glue`](https://huggingface.co/datasets/nyu-mll/glue), `rte` configuration. `sentence1` is the premise, `sentence2` the hypothesis, and labels are `entailment` (0) / `not_entailment` (1).
The 2,490 labeled source training examples are stratified and grouped by exact text pair into **70% internal training, 15% internal validation and 15% internal holdout**. Duplicate pairs never cross partitions; no source rows are dropped.
The internal holdout and official 277-example RTE validation split have both been examined in earlier studies; neither is an untouched test set for subsequent work.
The cache manifest records the requested and realized split sizes, seed and source row indices.

The split seed is `13`; final model seeds are `[42, 123, 456]`. Hyperparameter and stacking selection uses only the internal validation split. The official validation set is evaluated only for final selected configurations after retraining.

## Development-only overfitting study

The separate overfitting study selects LoRA rank/dropout on three screening
seeds, selects R-Drop strength using the chosen adapter, then compares baseline
and R-Drop on five different matched seeds. It reuses the same development set
throughout; no new holdout evaluation is performed. It saves clean train and
development accuracy/loss each epoch, but no model or optimizer weights.
Completed runs are reused after an interruption; an interrupted run restarts.

The Slurm phases accept `adapter`, `strength`, or `compare` in that order. Submit
each only after its predecessor succeeds, using `afterok` dependencies. The
strength phase can instead submit the comparison automatically after success
when `REGBENCH_QUEUE_COMPARE=1`, reducing the number of queued jobs under a
two-job submission limit. Its log and `compare_job.txt` record the new ID.
The final phase generates `outputs/overfitting_study/report/report.pdf` with the
historical combined report, screening tables, paired comparisons, and learning
curve, generalization-gap and screening figures. The best development epoch is
a retrospective selection on the same trajectory, not an independent test.
To resume, submit the failed phase again and reconnect subsequent dependencies;
do not run two instances of the same study simultaneously.

## Setup

Use your own Python virtual environment with a PyTorch CUDA build compatible with
your driver. The previously verified environment used Python 3.12 and PyTorch
2.11 with CUDA 12.8. From the project root, activate your environment and install
this project without replacing its existing CUDA-enabled PyTorch:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Slurm scripts use `REGBENCH_PYTHON` if set, then `.venv/bin/python` in the
project, and otherwise `python3` from the submitted environment's `PATH`.
Set `REGBENCH_PYTHON` to your own executable when the environment is elsewhere.
Hugging Face uses its normal cache location unless you set `HF_HOME`.
The supplied Slurm partition and resource requests are cluster-specific; adjust
the `#SBATCH` settings for your cluster before submitting.

## Prepare and run

```bash
regbench prepare --config configs/default.yaml
regbench baseline --config configs/default.yaml --out outputs
regbench augment --config configs/default.yaml --out outputs
regbench individual --config configs/default.yaml --out outputs
regbench stack --config configs/default.yaml --out outputs
regbench final --config configs/default.yaml --out outputs
regbench report --config configs/default.yaml --out outputs
```

Or submit the seven GPU stages through the controller. It waits for each job to succeed before submitting the next, keeping within per-user job-submission limits; a failed stage stops the pipeline. The controller remains active until completion:

```bash
bash scripts/submit_pipeline.sh
squeue -u "$USER"
sacct -u "$USER" --format=JobID,JobName,State,Elapsed,ExitCode
```

`regbench all --config configs/default.yaml --out outputs` runs the stages in one process for local use.
Tokenized splits are cached once; set `REGBENCH_DATA=/shared/path` to place the cache outside the repository.
Configurations: `default.yaml` (full requested search), `smoke.yaml` (tiny training smoke check). Overrides use `--set`, for example `--set epochs=1 train_size=64`.
Epoch checkpoints support resume; completed runs are skipped only when their config and augmentation-pool hashes match.

### Slurm smoke test and timing pilot

The pilot requests one A40, trains the real model for one epoch on 512 examples, evaluates small validation/test samples, and writes measured throughput to `outputs/pilot/gpu-baseline-fp32master-bf16/metrics.json`.

```bash
mkdir -p outputs/slurm
sbatch scripts/gpu_pilot.slurm
squeue -u "$USER"
tail -f outputs/slurm/regbench-pilot-<job-id>.out
```

The pilot script uses the cluster CUDA environment above and shared Hugging Face/data caches. It is not a full experiment run.
Slurm training defaults to `REGBENCH_DATALOADER_WORKERS=0` to avoid multiprocessing shared-memory bus errors on compute nodes. Override it only if the node's shared-memory limit has been verified.

## Experiment protocol

- Baseline: AdamW, lr `2e-5`, weight decay `0.01`, DeBERTa default dropout.
- Model weights are promoted to FP32 before AdamW creates optimizer state; CUDA autocast uses BF16 when supported.
- Baseline search tunes the three learning rates and epochs `{3, 4, 5}`; the selected epoch count is fixed for all later training.
- Augmentation uses `google/flan-t5-large` with a deterministic 2-shot NLI instruction prompt. It generates one label-preserving paraphrase or minimal-edit label-flipping hypothesis per internal training row. A selected baseline checkpoint filters candidates at confidence `>= 0.85`. Ratios `{0.25, 0.50, 1.00}` are fractions of original training size; if fewer filtered pairs exist, the achieved ratio is recorded and capped by the available pool.
- Individual method grids combine learning rates `{1e-5, 2e-5, 3e-5}` with three method parameters. Ties use lower internal validation loss.
- The five methods are ranked by internal validation score. Stacks start with the top two methods, then add one method at a time through the five-method stack, tuning only the new method parameter and learning rate.
- Final retraining includes the baseline, five individual winners, and four stacks (2–5 methods), each with seeds `[42, 123, 456]`.
- Metrics are accuracy, macro-F1 and 10-bin ECE; configs, per-epoch train/validation losses, predictions and metrics are saved. Plots include individual-vs-stack accuracy, calibration, reliability and loss curves. FLOPs are estimates (`6 × non-embedding parameters × processed tokens`); wall time is measured.

The requested modules are `regbench.data_aug`, `regbench.models_regularization`, `regbench.train_eval`, `regbench.run_experiments` and `regbench.plot_results`.

## Independent epoch-study report

The separate five-seed epoch study uses the previously unexamined internal holdout as its test set; the official RTE validation split has already been examined. After its producer writes `outputs/epoch_study/selection.json` and `outputs/epoch_study/final_metrics.json`, generate the report without training:

```bash
python scripts/report_epoch_study.py
cd outputs/epoch_study/report
pdflatex report.tex
pdflatex report.tex
```

The script requires Matplotlib and overwrites only its own `report.tex`, `learning_curves.{pdf,png}`, and `paired_accuracy.{pdf,png}` in that directory. Input and output paths can be overridden with `--selection`, `--final-metrics`, `--runs-dir`, and `--output-dir`. The default runs directory is `outputs/epoch_study/runs`; the generator requires all 15 selected final-run `metrics.json` files with complete epoch histories and train accuracy. Curves use the final evaluation trajectories, including post-freeze replays of historical five-epoch runs; the frozen selection file alone supplies the development selection table. The report shows per-seed and maximum absolute final-epoch dev accuracy differences between selection and final runs, and checks regularizers and learning rates wherever both records provide them. It rejects missing or mismatched seeds, inconsistent epochs and holdouts other than 374 rows. The paired intervals describe seed variation on the same holdout, not test-example sampling uncertainty.

Compilation requires LaTeX with `graphicx`, `booktabs`, `amsmath`, `times`, `helvet`, `courier`, `url`, `natbib` and `caption`. Obtain the official [AAAI-26 author kit](https://aaai.org/authorkit26-1/) separately and copy `aaai2026.sty` and `aaai2026.bst` into `outputs/epoch_study/report/` (or the TeX search path). The report uses the kit's anonymous-submission preamble (`\usepackage[submission]{aaai2026}`); change options only per the kit's camera-ready instructions. The script does not install or download the kit and emits no bibliography. If `aaai2026.sty` is absent, the document falls back to a two-column article requiring `geometry`; **this fallback is not AAAI-compliant**.

See [the split and reproducibility details](docs/data-splits.md), [GPU pilot results](docs/gpu-pilot.md), [contributor workflow](docs/contributing.md), and generated `outputs/reports/results.md`.

### Matched ablation and longer-duration follow-up

The development-only follow-up runs six pipelines (baseline, Mixup, R-Drop,
NEFTune, Mixup + R-Drop, and the full stack) at a matched learning rate of
`3e-5` and eight epochs, plus the full stack at 10 and 12 epochs. Each uses
seeds 42, 123, 456, 789, and 1011. The original internal development set
selects the highest-accuracy epoch, with lower loss and then the earlier epoch
breaking ties. These are exploratory development results, not a new independent
test; no additional holdout evaluation is performed.

The follow-up saves metrics, not model weights or resume checkpoints. If a job
is interrupted, completed runs are reused, but the interrupted run starts again.
Best-epoch metrics do not provide a deployable or independently testable model.
Submit from the project root:

```bash
mkdir -p outputs/slurm
sbatch scripts/followup_study.slurm
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode
```

The batch job requires Tectonic on `PATH`, at `outputs/tools/tectonic`, or via
`REGBENCH_TECTONIC`. It validates the compiler before training and writes the
results and compiled report under `outputs/followup_study/`. Logs are in
`outputs/slurm/rte-followup-JOB_ID.out` and `.err`. Slurm continues after a
local terminal disconnects; no email notification is configured by default.

### Standard LoRA comparison

Install the optional LoRA dependencies with `uv pip install --python .venv/bin/python -e '.[lora]'`.
The LoRA study uses rank 8, alpha 16, dropout 0.1, query/value projections
and trainable classifier and pooler. It screens learning rates 3e-5, 1e-4 and
2e-4 on seed 42 baseline development data, freezes the winner, then repeats
the 40-run matched design. The previously examined internal holdout is
evaluated at each run's final epoch, and predictions and metrics are saved;
it is an exploratory reused test, not an independent test. No model or resume
checkpoints are saved. Regularization strengths and LoRA parameters are fixed,
not re-tuned. Results and the extended report are in `outputs/lora_study/`.

Submit with `sbatch scripts/lora_study.slurm` from the project root and check
`sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode`. Logs are
`outputs/slurm/rte-lora-JOB_ID.out` and `.err`. Completed runs are reused on
resubmission; an interrupted run must start over. The report compares matched
development-set gains against the earlier full-fine-tuning run and warns when
selected learning rates differ.

## Five-seed epoch study

The separate `scripts/epoch_study.slurm` study compares the selected baseline, R-Drop,
and Mixup + R-Drop + NEFTune pipelines using matched seeds 42, 123, 456, 789, and
1011. It compares 5, 6, and 8 epochs for the stack using internal validation,
freezes the choice in `outputs/epoch_study/selection.json`, and only then evaluates
the selected configurations on the reserved internal holdout. The previously
inspected official RTE validation split is not used as this study's final test.

```bash
sbatch scripts/epoch_study.slurm
squeue -u "$USER"
bash scripts/monitor_epoch_study.sh JOB_ID
```

The monitor checks Slurm accounting every 60 seconds by default, reports failure
states, and checks that the results and report were produced on success. Its
optional second argument changes the polling interval. The study job itself
generates the LaTeX report and figures after the final evaluation.

New runs record training accuracy by evaluating the complete training set in
evaluation mode after each epoch. Existing historical runs did not record training
accuracy. The study keeps the selected learning rates for each pipeline, so its
comparisons do not isolate the effect of regularization from learning rate.
