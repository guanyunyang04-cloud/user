from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PYTHON = "C:/Users/ASUS/miniconda3/envs/yolos/python.exe"


def run_cli(*args: str) -> dict:
    result = subprocess.run(
        [PYTHON, "-m", "tools.brain.workflow", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return json.loads(result.stdout)


def run_cli_failure(*args: str) -> dict:
    result = subprocess.run(
        [PYTHON, "-m", "tools.brain.workflow", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode != 0
    return json.loads(result.stdout)


def run_script_cli(*args: str) -> dict:
    result = subprocess.run(
        [PYTHON, "tools/brain/workflow.py", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return json.loads(result.stdout)
