#!/usr/bin/env bash
#SBATCH --job-name=rte-overfit-timing
#SBATCH --partition=MGPU-TC2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
job="${1:?Pass comparison job ID}"
bash scripts/record_slurm_timing.sh "$job" outputs/overfitting_study/compare_slurm_timing.txt 5
