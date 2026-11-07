#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
mkdir -p outputs/slurm

log_file="outputs/slurm/pipeline-controller.log"
exec > >(tee -a "$log_file") 2>&1

wait_for_success() {
    local job_id="$1"
    local state=""
    while true; do
        state="$(sacct -n -P -j "$job_id" --format=JobIDRaw,State |
            awk -F'|' -v id="$job_id" '$1 == id {print $2; exit}' | xargs)"
        case "$state" in
            COMPLETED)
                printf 'Job %s completed successfully.\n' "$job_id"
                return 0
                ;;
            FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|PREEMPTED*|BOOT_FAIL*|DEADLINE*|REVOKED*|SPECIAL_EXIT*)
                printf 'Job %s ended with state %s; stopping pipeline.\n' "$job_id" "$state" >&2
                return 1
                ;;
            "")
                printf 'Waiting for accounting status for job %s.\n' "$job_id"
                ;;
            *)
                printf 'Job %s state: %s\n' "$job_id" "$state"
                ;;
        esac
        sleep 60
    done
}

if [[ "${1:-}" == "--resume-after" && -n "${2:-}" && "${3:-}" == "--start-at" && -n "${4:-}" && $# -eq 4 ]]; then
    printf 'Resuming after existing job %s, starting at %s.\n' "$2" "$4"
    wait_for_success "$2"
    all_stages=(prepare baseline augment individual stack final report)
    start_index=-1
    for index in "${!all_stages[@]}"; do
        if [[ "${all_stages[$index]}" == "$4" ]]; then
            start_index="$index"
            break
        fi
    done
    if (( start_index < 0 )); then
        echo "unknown restart stage: $4" >&2
        exit 2
    fi
    stages=("${all_stages[@]:$start_index}")
elif [[ "${1:-}" == "--resume-after" && -n "${2:-}" && $# -eq 2 ]]; then
    printf 'Resuming after existing job %s.\n' "$2"
    wait_for_success "$2"
    stages=(augment individual stack final report)
elif [[ $# -eq 0 ]]; then
    stages=(prepare baseline augment individual stack final report)
else
    echo "usage: $0 [--resume-after JOB_ID [--start-at STAGE]]" >&2
    exit 2
fi

for stage in "${stages[@]}"; do
    submitted="$(sbatch --parsable --export="ALL,REGBENCH_STAGE=${stage}" \
        scripts/run_pipeline_stage.slurm)"
    job_id="${submitted%%;*}"
    if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
        echo "could not parse Slurm job id for stage ${stage}: ${submitted}" >&2
        exit 1
    fi
    printf '%-12s %s\n' "$stage" "$job_id"
    wait_for_success "$job_id"
done

printf 'All pipeline stages completed successfully.\n'