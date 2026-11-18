#!/usr/bin/env bash
#SBATCH --job-name=rte-curves-report
#SBATCH --partition=MGPU-TC2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source scripts/slurm_env.sh
export MPLBACKEND=Agg

"$python_executable" scripts/run_regularization_curve_study.py --plots-only
cd outputs/overfitting_study/report
../../tools/tectonic --keep-logs combined_report.tex