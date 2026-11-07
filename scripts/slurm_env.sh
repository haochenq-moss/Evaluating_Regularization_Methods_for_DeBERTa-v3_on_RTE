#!/usr/bin/env bash
# Source from a Slurm script after changing to the project root.

if [[ -n "${REGBENCH_PYTHON:-}" ]]; then
    python_executable="$REGBENCH_PYTHON"
elif [[ -x "$PWD/.venv/bin/python" ]]; then
    python_executable="$PWD/.venv/bin/python"
else
    python_executable="python3"
fi

if ! command -v "$python_executable" >/dev/null 2>&1; then
    printf 'Python executable not found: %s. Set REGBENCH_PYTHON.\n' "$python_executable" >&2
    exit 1
fi
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
