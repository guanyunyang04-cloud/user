from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from reflection_learning import build_proposal_payload


def find_brain_root(cwd: Path) -> Path | None:
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "brain" / "brain_manifest.json").exists():
            return candidate
    return None


def owner_brain(cwd: Path, fallback: str) -> str:
    brain_root = find_brain_root(cwd.resolve())
    manifest = (brain_root or cwd.resolve()) / "brain" / "brain_manifest.json"
    if not manifest.exists():
        return fallback
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
    except Exception:
        return fallback
    return str(payload.get("brain_id") or payload.get("brain_type") or fallback)


def agent_learning_dir(cwd: Path) -> Path:
    resolved = cwd.resolve()
    brain_root = find_brain_root(resolved) or resolved
    return brain_root / "brain" / "output" / "agent_learning"


def index_path(cwd: Path) -> Path:
    return agent_learning_dir(cwd) / "agent_learning_index.json"


def load_learning_index(cwd: Path) -> dict[str, Any]:
    path = index_path(cwd)
    if not path.exists():
        return {"schema_version": 1, "proposals": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {"schema_version": 1, "proposals": []}
    if not isinstance(payload, dict):
        return {"schema_version": 1, "proposals": []}
    proposals = payload.get("proposals", [])
    if not isinstance(proposals, list):
        proposals = []
    return {"schema_version": int(payload.get("schema_version", 1) or 1), "proposals": proposals}


def proposal_status_counts(proposals: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in proposals:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "") or item.get("lifecycle_status", "") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def write_learning_index(cwd: Path, payload: dict[str, Any]) -> Path:
    out_dir = agent_learning_dir(cwd)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "agent_learning_index.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def create_proposal(
    cwd: Path,
    title: str,
    trigger: str,
    evidence: str,
    recommendation: str,
    *,
    severity: str = "info",
    owner_brain_name: str = "",
    writeback_target: str = "brain/references/",
    requires_user_confirmation: bool = True,
    target_layer: str = "brain_docs",
    status: str = "proposed",
    related_task: str = "",
    suggested_tests: list[str] | None = None,
    lesson: str = "",
    root_cause: str = "",
    supporting_events: list[dict[str, Any]] | None = None,
    anti_overfit_check: str = "",
) -> dict[str, Any]:
    resolved = cwd.resolve()
    out_dir = agent_learning_dir(resolved)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in title.lower()).strip("_") or "proposal"
    proposal_id = f"{stamp}_{safe_title}"
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    payload = build_proposal_payload(
        proposal_id=proposal_id,
        title=title,
        created_at=created_at,
        status=status,
        severity=severity,
        owner_brain=owner_brain_name or owner_brain(resolved, "workspace"),
        writeback_target=writeback_target,
        target_layer=target_layer,
        related_task=related_task,
        trigger=trigger,
        evidence=evidence,
        recommendation=recommendation,
        requires_user_confirmation=requires_user_confirmation,
        suggested_tests=list(suggested_tests or []),
        lesson=lesson,
        root_cause=root_cause,
        supporting_events=supporting_events,
        anti_overfit_check=anti_overfit_check,
    )
    json_path = out_dir / f"{proposal_id}.json"
    md_path = out_dir / f"{proposal_id}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    md_path.write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                f"- Authority: `{payload['authority']}`",
                f"- Proposal creation policy: `{payload['proposal_creation_policy']}`",
                f"- Implementation requires user confirmation: `{payload['implementation_requires_user_confirmation']}`",
                f"- Severity: `{payload['severity']}`",
                f"- Owner brain: `{payload['owner_brain']}`",
                f"- Writeback target: `{payload['writeback_target']}`",
                f"- Trigger: {trigger}",
                f"- Evidence: {evidence}",
                f"- Recommendation: {recommendation}",
                f"- Lesson: {payload['lesson']}",
                f"- Root cause: {payload['root_cause']}",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    index = load_learning_index(resolved)
    proposals = [item for item in index.get("proposals", []) if isinstance(item, dict) and item.get("proposal_id") != proposal_id]
    proposals.append(
        {
            "proposal_id": proposal_id,
            "title": title,
            "status": status,
            "lifecycle_status": payload["lifecycle_status"],
            "severity": severity,
            "target_layer": target_layer,
            "writeback_route": payload["writeback_route"],
            "verification_required": payload["verification_required"],
            "proposal_creation_policy": payload["proposal_creation_policy"],
            "implementation_requires_user_confirmation": payload["implementation_requires_user_confirmation"],
            "json_path": str(json_path.resolve()),
            "markdown_path": str(md_path.resolve()),
            "created_at": payload["created_at"],
        }
    )
    index["proposals"] = proposals
    write_learning_index(resolved, index)
    return {
        "status": "ok",
        "proposal_id": proposal_id,
        "json_path": str(json_path.resolve()),
        "markdown_path": str(md_path.resolve()),
    }


def list_proposals(cwd: Path) -> dict[str, Any]:
    index = load_learning_index(cwd)
    return {"status": "ok", "proposals": list(index.get("proposals", []) or [])}


def mark_proposal(cwd: Path, proposal_id: str, status: str) -> dict[str, Any]:
    normalized = str(status or "").strip().lower()
    if normalized not in {"proposed", "approved", "implemented", "verified", "rejected", "superseded"}:
        return {"status": "error", "error": f"unsupported proposal status: {status}"}
    index = load_learning_index(cwd)
    proposals = list(index.get("proposals", []) or [])
    updated: dict[str, Any] | None = None
    for item in proposals:
        if isinstance(item, dict) and item.get("proposal_id") == proposal_id:
            item["status"] = normalized
            item["lifecycle_status"] = normalized
            item["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            updated = item
            json_path = Path(str(item.get("json_path", "")))
            if json_path.exists():
                try:
                    payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
                    if isinstance(payload, dict):
                        payload["status"] = normalized
                        payload["lifecycle_status"] = normalized
                        payload["updated_at"] = item["updated_at"]
                        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
                except Exception:
                    pass
            break
    if updated is None:
        return {"status": "error", "error": f"proposal not found: {proposal_id}"}
    index["proposals"] = proposals
    write_learning_index(cwd, index)
    return {"status": "ok", "proposal": updated}
