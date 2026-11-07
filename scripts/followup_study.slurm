#!/usr/bin/env bash
#SBATCH --job-name=rte-followup
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

if [[ -n "${REGBENCH_TECTONIC:-}" ]]; then
    tectonic="$REGBENCH_TECTONIC"
elif [[ -x "$PWD/outputs/tools/tectonic" ]]; then
    tectonic="$PWD/outputs/tools/tectonic"
else
    tectonic="tectonic"
fi
if ! command -v "$tectonic" >/dev/null 2>&1; then
    printf 'Tectonic is required to compile the report; set REGBENCH_TECTONIC.\n' >&2
    exit 1
fi
"$tectonic" --version
"$python_executable" scripts/run_followup_study.py
"$python_executable" scripts/report_followup_study.py
cd outputs/followup_study/report
"$tectonic" --keep-logs report.tex
test -s report.pdf
printf '[followup] Complete: %s/report.pdf\n' "$PWD"
