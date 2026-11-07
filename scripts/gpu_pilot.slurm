#!/usr/bin/env bash
#SBATCH --job-name=regbench-pilot
#SBATCH --partition=MGPU-TC2
#SBATCH --gres=gpu:nvidia:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:?Submit this job from the project root}"

cd "$project_root"
source scripts/slurm_env.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

printf 'job=%s host=%s python=%s HF_HOME=%s\n' \
    "$SLURM_JOB_ID" "$(hostname)" "$python_executable" "${HF_HOME:-HuggingFace-default}"
nvidia-smi

"$python_executable" - <<'PY'
import torch
import transformers
import datasets

print("torch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("transformers:", transformers.__version__, "datasets:", datasets.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable on the allocated Slurm node")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"Expected one visible GPU, got {torch.cuda.device_count()}")
print("GPU:", torch.cuda.get_device_name(0))
PY

"$python_executable" scripts/run_gpu_pilot.py
