#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 || ! "$1" =~ ^[0-9]+$ || ! "${2:-60}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: bash scripts/monitor_lora_study.sh JOB_ID [POLL_SECONDS]" >&2
    exit 2
fi
job_id="$1"
interval="${2:-60}"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
printf 'Monitoring LoRA job %s every %s seconds.\n' "$job_id" "$interval"
missing=0
while true; do
    if ! accounting="$(sacct -n -P -j "$job_id" --format=JobIDRaw,State,ExitCode)"; then
        printf 'Unable to query accounting for job %s.\n' "$job_id" >&2
        exit 1
    fi
    state="$(printf '%s\n' "$accounting" | awk -F'|' -v id="$job_id" '$1 == id {print $2; exit}' | xargs)"
    case "$state" in
        COMPLETED)
            for file in outputs/lora_study/selection.json outputs/lora_study/environment.json \
                        outputs/lora_study/report/report.tex outputs/lora_study/report/report.pdf \
                        outputs/lora_study/report/lora_comparison.png; do
                if [[ ! -s "$file" ]]; then
                    printf 'Job %s completed but artifact is missing or empty: %s\n' "$job_id" "$file" >&2
                    exit 1
                fi
            done
            printf 'Job %s completed successfully. Report: %s/outputs/lora_study/report/report.pdf\n' "$job_id" "$root"
            exit 0
            ;;
        FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|PREEMPTED*|BOOT_FAIL*|DEADLINE*|REVOKED*|SPECIAL_EXIT*)
            printf 'Job %s ended with state %s; inspect outputs/slurm/rte-lora-%s.err.\n' "$job_id" "$state" "$job_id" >&2
            exit 1
            ;;
        "")
            missing=$((missing + 1))
            if (( missing >= 10 )); then
                printf 'No accounting record for job %s after %s checks.\n' "$job_id" "$missing" >&2
                exit 1
            fi
            ;;
        *) missing=0 ;;
    esac
    sleep "$interval"
done
