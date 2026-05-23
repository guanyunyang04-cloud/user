from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PYTHON = "C:/Users/ASUS/miniconda3/envs/yolos/python.exe"


def _run_help(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [PYTHON, *args, "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )


def test_run_alpha_path20_protocol_supports_module_help() -> None:
    result = _run_help("-m", "daily_research.path_policy.run_alpha_path20_protocol")

    assert result.returncode == 0
    assert "forecast-walkforward-study" in result.stdout
    assert "--stage" in result.stdout


def test_run_alpha_path20_protocol_supports_direct_script_help() -> None:
    result = _run_help("daily_research/path_policy/run_alpha_path20_protocol.py")

    assert result.returncode == 0
    assert "forecast-walkforward-study" in result.stdout
    assert "--stage" in result.stdout
