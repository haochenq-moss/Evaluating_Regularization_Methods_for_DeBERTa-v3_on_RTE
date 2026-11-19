import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_slurm_python_override(tmp_path):
    env = {**os.environ, "REGBENCH_PYTHON": "/bin/true"}
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; printf "%s" "$python_executable"', "_",
         str(ROOT / "scripts/slurm_env.sh")],
        cwd=tmp_path, env=env, text=True, capture_output=True, check=True,
    )
    assert result.stdout == "/bin/true"


def test_slurm_missing_python_fails(tmp_path):
    env = {**os.environ, "REGBENCH_PYTHON": str(tmp_path / "nonexistent-python")}
    result = subprocess.run(
        ["bash", "-c", 'source "$1"', "_", str(ROOT / "scripts/slurm_env.sh")],
        cwd=tmp_path, env=env, text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert "Python executable not found" in result.stderr