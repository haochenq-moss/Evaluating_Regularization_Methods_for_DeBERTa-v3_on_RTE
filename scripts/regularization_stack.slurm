#!/usr/bin/env bash
#SBATCH --job-name=rte-reg-stack
#SBATCH --partition=MGPU-TC2
#SBATCH --gres=gpu:nvidia:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source scripts/slurm_env.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export REGBENCH_DATALOADER_WORKERS=0

out=outputs/regularization_extensions
mkdir -p "$out" outputs/slurm
status_log="$out/slurm_status.log"

monitor_status() {
    while true; do
        state="$(squeue -h -j "$SLURM_JOB_ID" -o '%T' 2>/dev/null | head -n 1)"
        printf '%s job=%s state=%s\n' "$(date -Is)" "$SLURM_JOB_ID" "${state:-not-in-queue}" >> "$status_log"
        [[ -n "$state" ]] || return 0
        sleep 60
    done
}

monitor_status &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true' EXIT

printf '[stack] job=%s GPU allocation starts\n' "$SLURM_JOB_ID"
nvidia-smi
"$python_executable" scripts/run_ema_swa_study.py
printf '%s job=%s stack=completed\n' "$(date -Is)" "$SLURM_JOB_ID" >> "$status_log"