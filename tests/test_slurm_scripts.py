import os
from pathlib import Path
import shutil
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


def test_monitor_reports_success_and_missing_artifacts(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy(ROOT / "scripts/monitor_epoch_study.sh", scripts)
    binary = tmp_path / "bin"
    binary.mkdir()
    sacct = binary / "sacct"
    sacct.write_text("#!/usr/bin/env bash\nprintf '123|COMPLETED|0:0\\n'\n")
    sacct.chmod(0o755)
    env = {**os.environ, "PATH": f"{binary}:{os.environ['PATH']}"}
    command = ["bash", str(scripts / "monitor_epoch_study.sh"), "123", "1"]
    missing = subprocess.run(command, env=env, text=True, capture_output=True)
    assert missing.returncode != 0
    assert "missing or empty" in missing.stderr
    for path in ("selection.json", "final_metrics.json", "report/report.tex",
                 "report/learning_curves.png", "report/paired_accuracy.png"):
        artifact = tmp_path / "outputs/epoch_study" / path
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("test")
    completed = subprocess.run(command, env=env, text=True, capture_output=True)
    assert completed.returncode == 0
    assert "completed successfully" in completed.stdout


def test_monitor_reports_failed_job(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy(ROOT / "scripts/monitor_epoch_study.sh", scripts)
    binary = tmp_path / "bin"
    binary.mkdir()
    sacct = binary / "sacct"
    sacct.write_text("#!/usr/bin/env bash\nprintf '123|FAILED|1:0\\n'\n")
    sacct.chmod(0o755)
    result = subprocess.run(
        ["bash", str(scripts / "monitor_epoch_study.sh"), "123", "1"],
        env={**os.environ, "PATH": f"{binary}:{os.environ['PATH']}"},
        text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert "FAILED" in result.stderr


def test_followup_monitor_checks_artifacts_and_failure(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy(ROOT / "scripts/monitor_followup_study.sh", scripts)
    binary = tmp_path / "bin"
    binary.mkdir()
    sacct = binary / "sacct"
    sacct.write_text("#!/usr/bin/env bash\nprintf '123|COMPLETED|0:0\\n'\n")
    sacct.chmod(0o755)
    env = {**os.environ, "PATH": f"{binary}:{os.environ['PATH']}"}
    cmd = ["bash", str(scripts / "monitor_followup_study.sh"), "123", "1"]
    result = subprocess.run(cmd, env=env, text=True, capture_output=True)
    assert result.returncode != 0 and "missing or empty" in result.stderr
    for path in ("environment.json", "report/report.tex", "report/report.pdf", "report/followup_curves.png"):
        artifact = tmp_path / "outputs/followup_study" / path
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("test")
    result = subprocess.run(cmd, env=env, text=True, capture_output=True)
    assert result.returncode == 0 and "completed successfully" in result.stdout
    sacct.write_text("#!/usr/bin/env bash\nprintf '123|FAILED|1:0\\n'\n")
    result = subprocess.run(cmd, env=env, text=True, capture_output=True)
    assert result.returncode != 0 and "FAILED" in result.stderr


def test_lora_monitor_reports_success_and_missing_files(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy(ROOT / "scripts/monitor_lora_study.sh", scripts)
    binary = tmp_path / "bin"
    binary.mkdir()
    sacct = binary / "sacct"
    sacct.write_text("#!/usr/bin/env bash\nprintf '123|COMPLETED|0:0\\n'\n")
    sacct.chmod(0o755)
    env = {**os.environ, "PATH": f"{binary}:{os.environ['PATH']}"}
    cmd = ["bash", str(scripts / "monitor_lora_study.sh"), "123", "1"]
    result = subprocess.run(cmd, env=env, text=True, capture_output=True)
    assert result.returncode != 0 and "missing or empty" in result.stderr
    for path in ("selection.json", "environment.json", "report/report.tex",
                 "report/report.pdf", "report/lora_comparison.png"):
        artifact = tmp_path / "outputs/lora_study" / path
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("test")
    result = subprocess.run(cmd, env=env, text=True, capture_output=True)
    assert result.returncode == 0 and "completed successfully" in result.stdout


def test_record_slurm_timing_writes_final_accounting(tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    sacct = binary / "sacct"
    sacct.write_text("#!/usr/bin/env bash\nprintf '123|COMPLETED|start|end|01:02:03|3723|0:0\\n'\n")
    sacct.chmod(0o755)
    target = tmp_path / "timing.txt"
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/record_slurm_timing.sh"), "123", str(target), "1"],
        env={**os.environ, "PATH": f"{binary}:{os.environ['PATH']}"},
        text=True, capture_output=True,
    )
    assert result.returncode == 0
    assert target.read_text().splitlines() == [
        "JobIDRaw|State|Start|End|Elapsed|ElapsedRaw|ExitCode",
        "123|COMPLETED|start|end|01:02:03|3723|0:0",
    ]
