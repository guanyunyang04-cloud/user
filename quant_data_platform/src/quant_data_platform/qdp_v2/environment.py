from __future__ import annotations

import os
import sys
from importlib import metadata
from typing import Any


PRODUCTION_ENV_NAME = "yolos"
CORE_PACKAGES = ("mootdx", "duckdb", "baostock", "pyarrow", "sklearn")
PACKAGE_DISTRIBUTIONS = {
    "sklearn": "scikit-learn",
}


def runtime_environment() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in CORE_PACKAGES:
        try:
            packages[name] = metadata.version(PACKAGE_DISTRIBUTIONS.get(name, name))
        except Exception:
            packages[name] = ""
    return {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "effective_env": effective_environment_name(),
        "virtual_env": os.environ.get("VIRTUAL_ENV", ""),
        "packages": packages,
    }


def is_yolos_environment() -> bool:
    return effective_environment_name() == PRODUCTION_ENV_NAME


def effective_environment_name() -> str:
    conda_env = str(os.environ.get("CONDA_DEFAULT_ENV", "") or "").strip().lower()
    if conda_env == PRODUCTION_ENV_NAME:
        return PRODUCTION_ENV_NAME
    executable = sys.executable.replace("/", "\\").lower()
    if f"\\envs\\{PRODUCTION_ENV_NAME}\\python.exe" in executable:
        return PRODUCTION_ENV_NAME
    return conda_env


def assert_yolos_environment(*, command: str) -> None:
    if is_yolos_environment() or os.environ.get("QDP_ALLOW_NON_YOLOS", "").strip() == "1":
        return
    env = runtime_environment()
    raise RuntimeError(
        "qdp_v2_requires_yolos_environment:"
        f" command={command}; current_env={env.get('conda_env') or 'unknown'};"
        " run `conda run -n yolos python -m quant_data_platform.cli ...`"
    )
