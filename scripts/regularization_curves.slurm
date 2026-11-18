#!/usr/bin/env bash
#SBATCH --job-name=rte-reg-curves
#SBATCH --partition=MGPU-TC2
#SBATCH --gres=gpu:nvidia:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=05:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source scripts/slurm_env.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export REGBENCH_DATALOADER_WORKERS=0

mkdir -p outputs/regularization_curve_study outputs/slurm
printf '[regularization-curves] job=%s GPU allocation starts\n' "$SLURM_JOB_ID"
nvidia-smi
"$python_executable" scripts/run_regularization_curve_study.py
printf '[regularization-curves] job=%s completed\n' "$SLURM_JOB_ID"