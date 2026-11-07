#!/usr/bin/env bash
#SBATCH --job-name=rte-lora
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
"$python_executable" scripts/run_lora_study.py
"$python_executable" scripts/report_lora_study.py
cd outputs/lora_study/report
"$tectonic" --keep-logs report.tex
test -s report.pdf
printf '[lora] Completed: %s/report.pdf\n' "$PWD"
