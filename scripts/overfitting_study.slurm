#!/usr/bin/env bash
#SBATCH --job-name=rte-overfit
#SBATCH --partition=MGPU-TC2
#SBATCH --gres=gpu:nvidia:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source scripts/slurm_env.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export REGBENCH_DATALOADER_WORKERS=0
tectonic="${REGBENCH_TECTONIC:-$PWD/outputs/tools/tectonic}"
test -x "$tectonic"
"$python_executable" -c 'import peft, transformers; print("PEFT", peft.__version__, "Transformers", transformers.__version__)'
phase="${1:?Pass adapter, strength, or compare}"
case "$phase" in adapter|strength|compare) ;; *) printf 'Invalid phase: %s\n' "$phase" >&2; exit 2;; esac
mkdir -p outputs/overfitting_study
if [[ "$phase" == strength && -n "${REGBENCH_ADAPTER_JOB:-}" ]]; then
    bash scripts/record_slurm_timing.sh "$REGBENCH_ADAPTER_JOB" outputs/overfitting_study/adapter_slurm_timing.txt 5
fi
if [[ "$phase" == compare && -n "${REGBENCH_STRENGTH_JOB:-}" ]]; then
    bash scripts/record_slurm_timing.sh "$REGBENCH_STRENGTH_JOB" outputs/overfitting_study/strength_slurm_timing.txt 5
    timing_job="$(sbatch --parsable --dependency="afterany:${SLURM_JOB_ID}" scripts/overfitting_timing.slurm "$SLURM_JOB_ID")"
    printf '%s\n' "$timing_job" > outputs/overfitting_study/timing_job.txt
    printf '[study] Submitted final accounting job %s\n' "$timing_job"
fi
"$python_executable" scripts/run_overfitting_study.py --phase "$phase"
if [[ "$phase" == strength && "${REGBENCH_QUEUE_COMPARE:-0}" == 1 ]]; then
    compare_job="$(sbatch --parsable --dependency="afterok:${SLURM_JOB_ID}" \
        --export="ALL,REGBENCH_STRENGTH_JOB=${SLURM_JOB_ID},REGBENCH_QUEUE_COMPARE=0" \
        scripts/overfitting_study.slurm compare)"
    printf '%s\n' "$compare_job" > outputs/overfitting_study/compare_job.txt
    printf '[study] Submitted dependent comparison job %s\n' "$compare_job"
fi
if [[ "$phase" == compare ]]; then
    "$python_executable" scripts/report_overfitting_study.py
    cd outputs/overfitting_study/report
    "$tectonic" --keep-logs report.tex
    test -s report.pdf
    printf '[study] Report complete: %s/report.pdf\n' "$PWD"
fi
