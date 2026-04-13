from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_FOREGROUND_TIMEOUT_HOURS = 10
DEFAULT_EXECUTION_DISCIPLINE = "foreground_only"
DEFAULT_RESUME_DISCIPLINE = "strict_same_run_dir_only"
RUN_CONTRACT_FILENAME = "run_contract.json"
YOLOS_PROJECT_PYTHON = Path(r"C:\Users\ASUS\miniconda3\envs\yolos\python.exe")
PREFERRED_PROJECT_PYTHONS = (
    YOLOS_PROJECT_PYTHON,
)


def runtime_metadata() -> dict[str, Any]:
    return {
        "execution_discipline": DEFAULT_EXECUTION_DISCIPLINE,
        "resume_discipline": DEFAULT_RESUME_DISCIPLINE,
        "foreground_timeout_budget_hours": DEFAULT_FOREGROUND_TIMEOUT_HOURS,
    }


def _normalize_python_path(raw: str | Path | None) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    try:
        return str(Path(text).expanduser().resolve())
    except Exception:
        return text


def _is_yolos_python(raw: str | Path | None) -> bool:
    normalized = _normalize_python_path(raw).replace("/", "\\").lower()
    expected = _normalize_python_path(YOLOS_PROJECT_PYTHON).replace("/", "\\").lower()
    return bool(normalized) and (normalized == expected or "\\envs\\yolos\\" in normalized)


def resolve_project_python_executable(explicit: str | None = None) -> str:
    raw = str(explicit or "").strip()
    if raw and _is_yolos_python(raw):
        return raw
    for candidate in PREFERRED_PROJECT_PYTHONS:
        if candidate.exists():
            return str(candidate)
    if raw:
        return raw
    return str(sys.executable)


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _normalize_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_normalize_value(item) for item in value)
    return value


def stable_json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(_normalize_value(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def contract_fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def load_run_metadata(run_dir: Path) -> dict[str, Any]:
    contract_path = Path(run_dir) / RUN_CONTRACT_FILENAME
    if contract_path.exists():
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    metrics_path = Path(run_dir) / "metrics.json"
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    return {}


def compare_expected_fields(observed: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    normalized_observed = _normalize_value(observed)
    for key, expected_value in _normalize_value(expected).items():
        observed_value = normalized_observed.get(key, None)
        if observed_value != expected_value:
            mismatches.append(f"{key}: expected={expected_value!r} observed={observed_value!r}")
    return mismatches


def assert_expected_run_fields(run_dir: Path, expected: dict[str, Any], *, label: str) -> dict[str, Any]:
    observed = load_run_metadata(run_dir)
    if not observed:
        raise FileNotFoundError(f"{label} is missing both run_contract.json and metrics.json: {run_dir}")
    mismatches = compare_expected_fields(observed, expected)
    if mismatches:
        preview = "; ".join(mismatches[:8])
        raise ValueError(
            f"{label} protocol mismatch under {run_dir}. "
            f"Use a new root-tag or --force-rerun. Mismatches: {preview}"
        )
    return observed
