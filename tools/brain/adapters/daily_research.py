from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from tools.brain.adapters.daily_research_frontier import (
    SUMMARY_FILENAMES,
    STUDY_ROOTS,
    _dataset_ids,
    _model_families,
    build_frontier_report,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
ACTIVE_ARTIFACT = Path("daily_research/output/active_execution_strategy.json")
CONTROL_PLANE_DOC_LIMITS = {
    Path("daily_research/brain/state_center.md"): 180,
    Path("daily_research/brain/knowledge_center.md"): 180,
    Path("daily_research/brain/operations_center.md"): 150,
    Path("daily_research/brain/continuous_policy_design_contract.md"): 180,
}
LATEST_ARTIFACTS = {
    "latest_study_summary": Path("daily_research/output/continuous_policy/latest_study_summary.json"),
    "latest_protocol_summary": Path("daily_research/output/continuous_policy/latest_protocol_summary.json"),
    "latest_behavior_audit_summary": Path("daily_research/output/continuous_policy/latest_behavior_audit_summary.json"),
    "latest_conclusion_ledger": Path("daily_research/output/continuous_policy/latest_conclusion_ledger.json"),
}
CONTINUOUS_POLICY_OUTPUT_ROOT = Path("daily_research/output/continuous_policy")
STUDY_SUMMARY_METRIC_KEYS = (
    "annual_return",
    "max_drawdown",
    "monthly_return_mean",
    "cash_timing_quality_1d",
    "portfolio_daily_source_target_count",
    "portfolio_daily_source_realized_sell_rate",
    "portfolio_daily_exposure_utilization",
    "portfolio_daily_receiver_unrealized_deploy_share",
    "allocation_layer_native_target_used",
    "allocation_layer_native_fallback_used",
    "native_target_valid",
    "native_receiver_executable_mask_count",
    "native_positive_delta_count",
    "native_positive_delta_unsupported_share",
    "native_receiver_mask_mismatch_count",
    "native_target_constraint_violations",
    "native_target_invalid_sum_count",
    "native_target_invalid_turnover_count",
    "native_target_invalid_cap_count",
    "native_target_invalid_negative_weight_count",
    "native_target_invalid_unsupported_receiver_count",
    "native_target_invalid_sell_nonheld_count",
    "native_source_target_count",
    "training_evidence_status",
)


@dataclass(frozen=True)
class ArtifactRecord:
    name: str
    path: str
    exists: bool
    tag: str
    executed_at: str
    generated_at: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactFreshnessReport:
    artifacts: dict[str, ArtifactRecord]
    is_stale_risk: bool
    mismatch_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifacts": {key: value.to_dict() for key, value in self.artifacts.items()},
            "is_stale_risk": self.is_stale_risk,
            "mismatch_reason": self.mismatch_reason,
        }


@dataclass(frozen=True)
class StudyTrialEvidence:
    trial_id: int
    trial_tag: str
    status: str
    phase: str
    protocol_summary_json: str
    evaluation_summary_json: str
    training_diagnostics_json: str
    loss_profile: str
    sample_model_type: str
    training_evidence_status: str
    metrics: dict[str, Any]
    training_terms: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StudyEvidenceReport:
    study_tag: str
    study_summary_json: str
    exists: bool
    coherent: bool
    workflow: str
    summary_kind: str
    status: str
    stage: str
    evidence_verdict: str
    dataset_ids: list[str]
    model_families: list[str]
    searched_paths: list[str]
    artifact_freshness: dict[str, Any]
    trial_count: int
    completed_trial_count: int
    failed_trial_count: int
    search_profile: str
    objective_profile: str
    confirmatory_enabled: bool
    resource_limits: dict[str, Any]
    trials: list[StudyTrialEvidence]
    evidence_gaps: list[str]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["trials"] = [trial.to_dict() for trial in self.trials]
        return payload


def workspace_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        return WORKSPACE_ROOT / candidate
    try:
        if candidate.resolve().is_relative_to(WORKSPACE_ROOT.resolve()):
            return candidate
    except OSError:
        pass
    parts = candidate.parts
    if "daily_research" in parts:
        rel = Path(*parts[parts.index("daily_research") :])
        return WORKSPACE_ROOT / rel
    return candidate


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    full = workspace_path(path)
    if not full.exists():
        return {}
    try:
        payload = json.loads(full.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_json_path_text(path_text: str) -> dict[str, Any]:
    if not path_text:
        return {}
    return _read_json_if_exists(Path(path_text))


def _extract_artifact_tag(payload: dict[str, Any]) -> str:
    for key in ("study_tag", "protocol_tag", "run_tag", "tag"):
        value = str(payload.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _artifact_record(name: str, path: Path) -> ArtifactRecord:
    payload = _read_json_if_exists(path)
    promotion_gate = payload.get("promotion_gate") if isinstance(payload.get("promotion_gate"), dict) else {}
    return ArtifactRecord(
        name=name,
        path=path.as_posix(),
        exists=workspace_path(path).exists(),
        tag=_extract_artifact_tag(payload),
        executed_at=str(payload.get("executed_at", "") or ""),
        generated_at=str(payload.get("generated_at", "") or ""),
        status=str(payload.get("status", "") or promotion_gate.get("status", "") or payload.get("phase", "") or ""),
    )


def resolve_artifact_freshness() -> ArtifactFreshnessReport:
    records = {name: _artifact_record(name, path) for name, path in LATEST_ARTIFACTS.items()}
    study_tag = records["latest_study_summary"].tag
    protocol_tag = records["latest_protocol_summary"].tag
    mismatch = bool(study_tag and protocol_tag and study_tag != protocol_tag)
    reason = ""
    if mismatch:
        reason = f"latest study tag differs from latest protocol tag: {study_tag} != {protocol_tag}"
    return ArtifactFreshnessReport(records, mismatch, reason)


def artifact_freshness_evidence_gaps(freshness: ArtifactFreshnessReport | Mapping[str, Any]) -> list[str]:
    gaps: list[str] = []
    if isinstance(freshness, ArtifactFreshnessReport):
        if freshness.is_stale_risk:
            gaps.append("latest_study_summary and latest_protocol_summary tags differ")
        study = freshness.artifacts["latest_study_summary"]
        protocol = freshness.artifacts["latest_protocol_summary"]
        if not study.exists:
            gaps.append("latest study summary is missing")
        if not protocol.exists:
            gaps.append("latest protocol summary is missing")
        return gaps

    if bool(freshness.get("is_stale_risk", False)):
        gaps.append("latest_study_summary and latest_protocol_summary tags differ")
    artifacts = freshness.get("artifacts", {})
    if isinstance(artifacts, Mapping):
        study = artifacts.get("latest_study_summary", {})
        protocol = artifacts.get("latest_protocol_summary", {})
        if isinstance(study, Mapping) and not bool(study.get("exists", False)):
            gaps.append("latest study summary is missing")
        if isinstance(protocol, Mapping) and not bool(protocol.get("exists", False)):
            gaps.append("latest protocol summary is missing")
    return gaps


def _study_summary_path(study_tag: str) -> Path:
    return CONTINUOUS_POLICY_OUTPUT_ROOT / "studies" / study_tag / "study_summary.json"


def _summary_path_candidates(study_tag: str, workflow: str | None = None) -> list[tuple[str, Path]]:
    workflow_key = str(workflow or "").strip()
    workflows = [workflow_key] if workflow_key else list(STUDY_ROOTS)
    candidates: list[tuple[str, Path]] = []
    for candidate_workflow in workflows:
        root = STUDY_ROOTS.get(candidate_workflow)
        if root is None:
            continue
        for filename in SUMMARY_FILENAMES:
            candidates.append((candidate_workflow, root / study_tag / filename))
    return candidates


def _empty_study_evidence_report(
    *,
    study_tag: str,
    workflow: str = "",
    searched_paths: list[str] | None = None,
    evidence_gaps: list[str] | None = None,
) -> StudyEvidenceReport:
    paths = list(searched_paths or [])
    gaps = list(evidence_gaps or [])
    return StudyEvidenceReport(
        study_tag=study_tag,
        study_summary_json="",
        exists=False,
        coherent=False,
        workflow=workflow,
        summary_kind="",
        status="",
        stage="",
        evidence_verdict="",
        dataset_ids=[],
        model_families=[],
        searched_paths=paths,
        artifact_freshness=resolve_artifact_freshness().to_dict(),
        trial_count=0,
        completed_trial_count=0,
        failed_trial_count=0,
        search_profile="",
        objective_profile="",
        confirmatory_enabled=False,
        resource_limits={},
        trials=[],
        evidence_gaps=gaps,
    )


def _nested_dict(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return current if isinstance(current, dict) else {}


def _trial_training_diagnostics(protocol_payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    embedded = _nested_dict(protocol_payload, "train", "training_diagnostics")
    path_text = str(
        embedded.get("training_diagnostics_json", "")
        or protocol_payload.get("training_diagnostics_json", "")
        or protocol_payload.get("model_training_diagnostics_json", "")
        or ""
    )
    path_payload = _read_json_path_text(path_text)
    if path_payload:
        return path_text, path_payload
    return path_text, embedded


def _trial_evaluation_payload(protocol_payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    embedded = _nested_dict(protocol_payload, "evaluation")
    path_text = str(
        embedded.get("evaluation_summary_json", "")
        or protocol_payload.get("evaluation_summary_json", "")
        or ""
    )
    path_payload = _read_json_path_text(path_text)
    if path_payload:
        return path_text, path_payload
    return path_text, embedded


def _metric_from_sources(key: str, *sources: dict[str, Any]) -> Any:
    for source in sources:
        if key in source:
            return source[key]
    for source in sources:
        for nested_key in ("continuity_metrics", "continuous_policy_metrics", "primary_metrics", "training_evidence"):
            nested = source.get(nested_key) if isinstance(source, dict) else None
            if isinstance(nested, dict) and key in nested:
                return nested[key]
    return None


def _build_trial_evidence(trial_payload: dict[str, Any]) -> StudyTrialEvidence:
    protocol_path = str(trial_payload.get("protocol_summary_json", "") or "")
    protocol_payload = _read_json_path_text(protocol_path)
    evaluation_path, evaluation_payload = _trial_evaluation_payload(protocol_payload)
    diagnostics_path, diagnostics_payload = _trial_training_diagnostics(protocol_payload)
    primary_metrics = trial_payload.get("primary_metrics") if isinstance(trial_payload.get("primary_metrics"), dict) else {}
    continuity_metrics = _nested_dict(evaluation_payload, "continuity_metrics")
    training_evidence = _nested_dict(protocol_payload, "training_evidence")
    metrics: dict[str, Any] = {}
    for key in STUDY_SUMMARY_METRIC_KEYS:
        value = _metric_from_sources(
            key,
            trial_payload,
            primary_metrics,
            continuity_metrics,
            evaluation_payload,
            training_evidence,
            diagnostics_payload,
        )
        if value is not None:
            metrics[key] = value
    terms = diagnostics_payload.get("portfolio_day_set_native_allocation_vector_terms", {})
    if not isinstance(terms, dict):
        terms = {}
    trial_config = trial_payload.get("trial_config") if isinstance(trial_payload.get("trial_config"), dict) else {}
    return StudyTrialEvidence(
        trial_id=int(trial_payload.get("trial_id", 0) or 0),
        trial_tag=str(trial_payload.get("trial_tag", "") or protocol_payload.get("run_tag", "") or ""),
        status=str(trial_payload.get("status", "") or ""),
        phase=str(trial_payload.get("phase", "") or ""),
        protocol_summary_json=protocol_path,
        evaluation_summary_json=evaluation_path,
        training_diagnostics_json=diagnostics_path,
        loss_profile=str(trial_config.get("loss_profile", "") or protocol_payload.get("loss_profile", "") or diagnostics_payload.get("loss_profile", "") or ""),
        sample_model_type=str(diagnostics_payload.get("sample_model_type", "") or ""),
        training_evidence_status=str(
            _metric_from_sources("training_evidence_status", trial_payload, primary_metrics, training_evidence, diagnostics_payload)
            or training_evidence.get("status", "")
            or ""
        ),
        metrics=metrics,
        training_terms=dict(terms),
    )


def resolve_study_evidence(study_tag: str, workflow: str | None = None) -> StudyEvidenceReport:
    tag = str(study_tag or "").strip()
    if not tag:
        raise ValueError("study_tag is required")
    workflow_key = str(workflow or "").strip()
    if workflow_key and workflow_key not in STUDY_ROOTS:
        return _empty_study_evidence_report(
            study_tag=tag,
            workflow=workflow_key,
            evidence_gaps=[f"unsupported_study_workflow: {workflow_key}"],
        )

    candidates = _summary_path_candidates(tag, workflow=workflow_key or None)
    searched_paths = [path.as_posix() for _, path in candidates]
    hits: list[tuple[str, Path]] = []
    workflows_with_hits: set[str] = set()
    for candidate_workflow, path in candidates:
        if candidate_workflow in workflows_with_hits:
            continue
        if workspace_path(path).is_file():
            hits.append((candidate_workflow, path))
            workflows_with_hits.add(candidate_workflow)
    if len(hits) > 1:
        hit_paths = [path.as_posix() for _, path in hits]
        return _empty_study_evidence_report(
            study_tag=tag,
            searched_paths=hit_paths,
            evidence_gaps=[f"ambiguous_study_tag: {tag}"],
        )
    if not hits:
        return _empty_study_evidence_report(
            study_tag=tag,
            workflow=workflow_key,
            searched_paths=searched_paths,
            evidence_gaps=[f"study summary not found: {path}" for path in searched_paths],
        )

    resolved_workflow, path = hits[0]
    summary = _read_json_if_exists(path)
    gaps: list[str] = []
    trials: list[StudyTrialEvidence] = []
    if not summary:
        gaps.append(f"study summary not found: {path.as_posix()}")
    for trial_payload in summary.get("screening_trials", []) if isinstance(summary.get("screening_trials"), list) else []:
        if isinstance(trial_payload, dict):
            evidence = _build_trial_evidence(trial_payload)
            trials.append(evidence)
            if not evidence.protocol_summary_json:
                gaps.append(f"trial {evidence.trial_id} missing protocol_summary_json")
            if not evidence.training_diagnostics_json:
                gaps.append(f"trial {evidence.trial_id} missing training diagnostics path")
    if summary and str(summary.get("study_tag", "") or summary.get("run_tag", "") or "") != tag:
        gaps.append("study summary tag differs from requested tag")
    return StudyEvidenceReport(
        study_tag=tag,
        study_summary_json=path.as_posix(),
        exists=bool(summary),
        coherent=bool(summary) and not gaps,
        workflow=resolved_workflow,
        summary_kind=path.name,
        status=str(summary.get("status", "") or ""),
        stage=str(summary.get("stage", "") or ""),
        evidence_verdict=str(summary.get("evidence_verdict", "") or ""),
        dataset_ids=_dataset_ids(summary),
        model_families=_model_families(summary),
        searched_paths=searched_paths,
        artifact_freshness=resolve_artifact_freshness().to_dict(),
        trial_count=int(summary.get("trial_count", 0) or 0),
        completed_trial_count=int(summary.get("completed_trial_count", 0) or 0),
        failed_trial_count=int(summary.get("failed_trial_count", 0) or 0),
        search_profile=str(summary.get("search_profile", "") or ""),
        objective_profile=str(summary.get("objective_profile", "") or ""),
        confirmatory_enabled=bool(summary.get("confirmatory_enabled", False)),
        resource_limits=dict(summary.get("resource_limits", {}) or {}),
        trials=trials,
        evidence_gaps=gaps,
    )


def active_artifact_diff_status() -> dict[str, Any]:
    result = subprocess.run(
        ["git", "diff", "--", ACTIVE_ARTIFACT.as_posix()],
        cwd=str(WORKSPACE_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    diff_text = result.stdout or ""
    return {
        "path": ACTIVE_ARTIFACT.as_posix(),
        "status": "clean" if result.returncode == 0 and not diff_text.strip() else "dirty",
        "returncode": result.returncode,
        "diff_line_count": len(diff_text.splitlines()),
    }


def read_child_context_excerpt(path: str, *, max_lines: int = 18) -> list[str]:
    try:
        lines = workspace_path(path).read_text(encoding="utf-8-sig").splitlines()
    except Exception:
        return []
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped.startswith("-"):
            out.append(stripped)
        if len(out) >= max_lines:
            break
    return out


def child_context_additions() -> dict[str, Any]:
    return {
        "state_summary": read_child_context_excerpt("daily_research/brain/state_center.md"),
        "hard_rules": read_child_context_excerpt("daily_research/brain/knowledge_center.md"),
    }


def capsule_guard_additions(rule_report: Mapping[str, Any]) -> dict[str, Any]:
    findings = rule_report.get("findings", []) if isinstance(rule_report, Mapping) else []
    return {
        "active_artifact_guard": {
            "path": ACTIVE_ARTIFACT.as_posix(),
            "status": "clean" if not any(f["code"] == "active_artifact_diff" for f in findings) else "dirty",
        },
        "frontier_report": build_frontier_report(),
    }


def control_plane_doc_limits() -> dict[Path, int]:
    return dict(CONTROL_PLANE_DOC_LIMITS)
