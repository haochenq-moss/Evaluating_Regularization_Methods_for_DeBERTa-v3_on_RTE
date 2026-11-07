#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 || ! "$1" =~ ^[0-9]+$ || ! "${3:-60}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: bash scripts/record_slurm_timing.sh JOB_ID OUTPUT_FILE [POLL_SECONDS]" >&2
    exit 2
fi

job_id="$1"
output="$2"
interval="${3:-60}"
temporary="${output}.tmp"
trap 'rm -f -- "$temporary"' EXIT

while true; do
    if ! accounting="$(sacct -n -P -j "$job_id" --format=JobIDRaw,State,Start,End,Elapsed,ElapsedRaw,ExitCode)"; then
        printf 'Unable to query accounting for job %s.\n' "$job_id" >&2
        exit 1
    fi
    row="$(printf '%s\n' "$accounting" | awk -F'|' -v id="$job_id" '$1 == id {print; exit}')"
    state="$(printf '%s\n' "$row" | cut -d '|' -f 2 | xargs)"
    case "$state" in
        COMPLETED|FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|PREEMPTED*|BOOT_FAIL*|DEADLINE*|REVOKED*|SPECIAL_EXIT*)
            printf 'JobIDRaw|State|Start|End|Elapsed|ElapsedRaw|ExitCode\n%s\n' "$row" > "$temporary"
            mv -f -- "$temporary" "$output"
            printf 'Final Slurm timing for job %s recorded in %s (state %s).\n' "$job_id" "$output" "$state"
            exit 0
            ;;
        *) sleep "$interval" ;;
    esac
done
