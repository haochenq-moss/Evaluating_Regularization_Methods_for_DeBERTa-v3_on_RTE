#!/usr/bin/env bash
#SBATCH --job-name=regbench
#SBATCH --partition=MGPU-TC2
#SBATCH --gres=gpu:nvidia:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:?Submit from the project root}"
stage="${REGBENCH_STAGE:?Set REGBENCH_STAGE when submitting this job}"
config="${REGBENCH_CONFIG:-configs/default.yaml}"

cd "$project_root"
source scripts/slurm_env.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export REGBENCH_DATALOADER_WORKERS="${REGBENCH_DATALOADER_WORKERS:-0}"
export HF_HUB_OFFLINE="${REGBENCH_HUB_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${REGBENCH_DATASETS_OFFLINE:-0}"

printf 'stage=%s job=%s host=%s python=%s\n' \
    "$stage" "$SLURM_JOB_ID" "$(hostname)" "$python_executable"
nvidia-smi

case "$stage" in
    prepare|baseline|augment|individual|stack|final|report)
        if [[ "$stage" == "report" ]]; then
            "$python_executable" -m regbench.cli report --config "$config" --out outputs
        elif [[ "$stage" == "prepare" ]]; then
            "$python_executable" -m regbench.cli prepare --config "$config" --out outputs
        else
            "$python_executable" -m regbench.cli "$stage" --config "$config" --out outputs
        fi
        ;;
    *)
        echo "unknown pipeline stage: $stage" >&2
        exit 2
        ;;
esac
