from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
CONTINUOUS_POLICY_ROOT = PROJECT_ROOT / "output" / "continuous_policy"
MODELS_ROOT = CONTINUOUS_POLICY_ROOT / "models"
EVALUATIONS_ROOT = CONTINUOUS_POLICY_ROOT / "evaluations"
EXPORTS_ROOT = CONTINUOUS_POLICY_ROOT / "exports"
PROTOCOLS_ROOT = CONTINUOUS_POLICY_ROOT / "protocols"
RUNTIME_ROOT = CONTINUOUS_POLICY_ROOT / "runtime"
LATEST_TRAIN_SUMMARY_PATH = CONTINUOUS_POLICY_ROOT / "latest_train_summary.json"
LATEST_EVALUATION_SUMMARY_PATH = CONTINUOUS_POLICY_ROOT / "latest_evaluation_summary.json"
LATEST_EXPORT_SUMMARY_PATH = CONTINUOUS_POLICY_ROOT / "latest_export_summary.json"
LATEST_PROTOCOL_SUMMARY_PATH = CONTINUOUS_POLICY_ROOT / "latest_protocol_summary.json"
LATEST_BEHAVIOR_AUDIT_SUMMARY_PATH = CONTINUOUS_POLICY_ROOT / "latest_behavior_audit_summary.json"
LATEST_CONCLUSION_LEDGER_PATH = CONTINUOUS_POLICY_ROOT / "latest_conclusion_ledger.json"
RUNTIME_STATE_PATH = RUNTIME_ROOT / "portfolio_state.json"


def ensure_layout() -> None:
    CONTINUOUS_POLICY_ROOT.mkdir(parents=True, exist_ok=True)
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    EVALUATIONS_ROOT.mkdir(parents=True, exist_ok=True)
    EXPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    PROTOCOLS_ROOT.mkdir(parents=True, exist_ok=True)
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def timestamp_tag(prefix: str) -> str:
    return f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    ensure_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)
    return path


def update_latest_summary(kind: str, payload: dict[str, Any]) -> Path:
    normalized = str(kind or "").strip().lower()
    if normalized == "train":
        return write_json(LATEST_TRAIN_SUMMARY_PATH, payload)
    if normalized == "evaluation":
        return write_json(LATEST_EVALUATION_SUMMARY_PATH, payload)
    if normalized == "export":
        return write_json(LATEST_EXPORT_SUMMARY_PATH, payload)
    if normalized == "protocol":
        return write_json(LATEST_PROTOCOL_SUMMARY_PATH, payload)
    if normalized == "behavior_audit":
        return write_json(LATEST_BEHAVIOR_AUDIT_SUMMARY_PATH, payload)
    if normalized == "conclusion_ledger":
        return write_json(LATEST_CONCLUSION_LEDGER_PATH, payload)
    raise ValueError(f"Unsupported continuous-policy latest summary kind: {kind}")


def resolve_latest_model_artifact(explicit_path: str = "") -> Path:
    text = str(explicit_path or "").strip()
    if text:
        path = Path(text).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Continuous-policy model artifact not found: {path}")
        return path

    summary = read_json(LATEST_TRAIN_SUMMARY_PATH)
    model_path = Path(str(summary.get("model_artifact_path", "") or "")).expanduser()
    if model_path and str(model_path) and model_path.exists():
        return model_path.resolve()

    candidates = sorted(
        [
            *MODELS_ROOT.glob("*/continuous_policy_v3_seq_artifact.pt"),
            *MODELS_ROOT.glob("*/continuous_policy_v2_artifact.pt"),
            *MODELS_ROOT.glob("*/continuous_policy_artifact.pkl"),
        ],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0].resolve()
    raise FileNotFoundError("No continuous-policy model artifact has been trained yet.")


def load_runtime_state() -> dict[str, Any]:
    return read_json(RUNTIME_STATE_PATH)


def save_runtime_state(payload: dict[str, Any]) -> Path:
    return write_json(RUNTIME_STATE_PATH, payload)
