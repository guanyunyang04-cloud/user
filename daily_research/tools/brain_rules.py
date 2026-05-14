from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from daily_research.tools.brain_platform import WORKSPACE_ROOT, resolve_artifact_freshness


ACTIVE_ARTIFACT = Path("daily_research/output/active_execution_strategy.json")
CONTROL_PLANE_DOC_LIMITS = {
    Path("daily_research/brain/state_center.md"): 180,
    Path("daily_research/brain/knowledge_center.md"): 180,
    Path("daily_research/brain/operations_center.md"): 150,
    Path("daily_research/brain/continuous_policy_design_contract.md"): 180,
}


@dataclass(frozen=True)
class BrainRuleFinding:
    severity: str
    code: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _run_git_diff_name(path: Path) -> str:
    result = subprocess.run(
        ["git", "diff", "--", path.as_posix()],
        cwd=str(WORKSPACE_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return result.stdout or result.stderr or ""


def check_active_artifact_diff() -> BrainRuleFinding | None:
    diff = _run_git_diff_name(ACTIVE_ARTIFACT)
    if diff.strip():
        return BrainRuleFinding("error", "active_artifact_diff", f"{ACTIVE_ARTIFACT.as_posix()} has uncommitted diff")
    return None


def findings_for_failed_completed_trials(study_evidence: Mapping[str, Any]) -> list[BrainRuleFinding]:
    findings: list[BrainRuleFinding] = []
    trials = study_evidence.get("trials", [])
    if not isinstance(trials, list):
        return findings
    for trial in trials:
        if not isinstance(trial, Mapping):
            continue
        status = str(trial.get("status", "") or "").lower()
        evidence_status = str(trial.get("training_evidence_status", "") or "").lower()
        trial_id = str(trial.get("trial_id", "") or "?")
        if status in {"failed", "error", "timeout"} and evidence_status in {"completed", "sufficient", "ok"}:
            findings.append(
                BrainRuleFinding(
                    "error",
                    "failed_trial_marked_completed_evidence",
                    f"trial {trial_id} is {status} but training_evidence_status={evidence_status}",
                )
            )
    return findings


def finding_for_stale_latest_without_explicit_tag(*, has_explicit_study_tag: bool) -> BrainRuleFinding | None:
    freshness = resolve_artifact_freshness()
    if freshness.is_stale_risk and not has_explicit_study_tag:
        return BrainRuleFinding(
            "warning",
            "loose_latest_stale_requires_explicit_tag",
            freshness.mismatch_reason or "latest artifacts are stale-risk; use an explicit study tag",
        )
    return None


def finding_for_full_gold_claim_without_catalog(text: str, *, data_lake_root: str | Path | None = None) -> BrainRuleFinding | None:
    lower = str(text or "").lower()
    if "full gold" not in lower and "全量 gold" not in lower and "full-universe gold" not in lower:
        return None
    try:
        from daily_research.data_lake import ResearchDataLake

        lake = ResearchDataLake(data_lake_root)
        rows = lake.list_datasets(dataset_kind="continuous_policy_training_matrices", zone="strict_train")
        has_gold = any("learned_all_a" in str(row.get("universe_name", "")) for _, row in rows.iterrows())
    except Exception:
        has_gold = False
    if not has_gold:
        return BrainRuleFinding(
            "error",
            "full_gold_claim_without_catalog_entry",
            "full Gold training-set claims require a registered Gold strict_train catalog entry",
        )
    return None


def findings_for_realtime_completed_evidence(label_summary: Mapping[str, Any]) -> list[BrainRuleFinding]:
    zone = str(label_summary.get("zone", "") or "")
    unobserved = int(label_summary.get("unobserved_label_rows", 0) or 0)
    is_training_safe = bool(label_summary.get("is_training_safe", False))
    if zone == "realtime_research" and (unobserved > 0 or is_training_safe):
        return [
            BrainRuleFinding(
                "error",
                "realtime_tail_counted_as_training_evidence",
                "realtime_research labels may be unobserved and must not be completed training evidence",
            )
        ]
    return []


def findings_for_control_plane_doc_lengths(
    limits: Mapping[Path, int] | None = None,
) -> list[BrainRuleFinding]:
    findings: list[BrainRuleFinding] = []
    for rel_path, max_lines in (limits or CONTROL_PLANE_DOC_LIMITS).items():
        path = WORKSPACE_ROOT / rel_path
        if not path.exists():
            continue
        line_count = len(path.read_text(encoding="utf-8-sig").splitlines())
        if line_count > max_lines:
            findings.append(
                BrainRuleFinding(
                    "warning",
                    "brain_control_plane_doc_too_long",
                    f"{rel_path.as_posix()} has {line_count} lines; target <= {max_lines}",
                )
            )
    return findings


def run_brain_rules(
    *,
    study_evidence: Mapping[str, Any] | None = None,
    has_explicit_study_tag: bool = False,
    claim_text: str = "",
    label_summaries: Iterable[Mapping[str, Any]] = (),
    check_control_plane_lengths: bool = True,
) -> dict[str, Any]:
    findings: list[BrainRuleFinding] = []
    active = check_active_artifact_diff()
    if active is not None:
        findings.append(active)
    stale = finding_for_stale_latest_without_explicit_tag(has_explicit_study_tag=has_explicit_study_tag)
    if stale is not None:
        findings.append(stale)
    if study_evidence:
        findings.extend(findings_for_failed_completed_trials(study_evidence))
    gold = finding_for_full_gold_claim_without_catalog(claim_text)
    if gold is not None:
        findings.append(gold)
    for summary in label_summaries:
        findings.extend(findings_for_realtime_completed_evidence(summary))
    if check_control_plane_lengths:
        findings.extend(findings_for_control_plane_doc_lengths())
    errors = [finding for finding in findings if finding.severity == "error"]
    warnings = [finding for finding in findings if finding.severity == "warning"]
    return {
        "status": "failed" if errors else "ok",
        "error_count": len(errors),
        "warning_count": len(warnings),
        "findings": [finding.to_dict() for finding in findings],
    }


def print_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(dict(payload), ensure_ascii=False, indent=2))
