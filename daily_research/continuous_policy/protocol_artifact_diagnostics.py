from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ABNORMAL_WINDOWS_EXIT_CODES = {3221225477, 3221225786, 3221226505}


def _preview(text: str, *, limit: int = 500) -> str:
    cleaned = text.replace("\x00", "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[-limit:]


def inspect_json_artifact(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    payload: dict[str, Any] = {
        "kind": "json",
        "path": str(artifact_path.resolve()) if artifact_path.exists() else str(artifact_path),
        "exists": artifact_path.exists(),
        "parse_ok": False,
        "size_bytes": 0,
        "top_level_keys": [],
    }
    if not artifact_path.exists():
        payload["missing_reason"] = "not_found"
        return payload

    raw = artifact_path.read_text(encoding="utf-8", errors="replace")
    payload["size_bytes"] = int(artifact_path.stat().st_size)
    payload["tail_preview"] = _preview(raw)
    try:
        parsed = json.loads(raw)
    except Exception as exc:
        payload["parse_error"] = f"{type(exc).__name__}: {exc}"
        return payload

    if not isinstance(parsed, dict):
        payload["parse_error"] = f"top_level_type={type(parsed).__name__}"
        return payload

    payload["parse_ok"] = True
    payload["top_level_keys"] = sorted(str(key) for key in parsed.keys())
    payload["run_tag"] = str(parsed.get("run_tag", ""))
    payload["promotion_status"] = str((parsed.get("promotion_gate", {}) or {}).get("status", ""))
    return payload


def summarize_protocol_artifacts(protocol_root: str | Path, *, exit_code: int = 0) -> dict[str, Any]:
    root = Path(protocol_root)
    protocol_summary_path = root / "protocol_summary.json"
    runtime_failure_summary_path = root / "runtime_failure_summary.json"
    protocol_health = inspect_json_artifact(protocol_summary_path)
    runtime_failure_health = inspect_json_artifact(runtime_failure_summary_path)
    protocol_summary: dict[str, Any] = {}
    if protocol_health.get("parse_ok"):
        protocol_summary = json.loads(protocol_summary_path.read_text(encoding="utf-8"))
    runtime_failure_summary: dict[str, Any] = {}
    if runtime_failure_health.get("parse_ok"):
        runtime_failure_summary = json.loads(runtime_failure_summary_path.read_text(encoding="utf-8"))

    training_evidence = dict(protocol_summary.get("training_evidence", {}) or {})
    promotion_gate = dict(protocol_summary.get("promotion_gate", {}) or {})
    diagnostic_flags: list[str] = []
    if int(exit_code) != 0:
        diagnostic_flags.append("nonzero_exit")
    if int(exit_code) in ABNORMAL_WINDOWS_EXIT_CODES:
        diagnostic_flags.append("abnormal_exit")
    if not protocol_health.get("parse_ok"):
        diagnostic_flags.append("protocol_summary_unparseable")
    if runtime_failure_summary:
        diagnostic_flags.append("runtime_failure_summary_present")

    return {
        "protocol_root": str(root.resolve()) if root.exists() else str(root),
        "exit_code": int(exit_code),
        "completed_evidence": int(exit_code) == 0 and bool(protocol_health.get("parse_ok")),
        "diagnostic_flags": diagnostic_flags,
        "protocol_summary_health": protocol_health,
        "runtime_failure_summary_health": runtime_failure_health,
        "runtime_failure_summary": runtime_failure_summary,
        "protocol_summary": protocol_summary,
        "training_evidence_status": str(training_evidence.get("status", "")),
        "best_epoch": int(training_evidence.get("best_epoch", 0) or 0),
        "completed_epochs": int(training_evidence.get("completed_epochs", 0) or 0),
        "promotion_status": str(promotion_gate.get("status", "")),
        "promotion_failed_checks": list(promotion_gate.get("failed_checks", []) or []),
    }
