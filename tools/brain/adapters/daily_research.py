from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from tools.brain.adapters.daily_research_frontier import (
    SUMMARY_FILENAMES,
    RUN_ROOTS,
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

EXPERIMENT_REVIEW_TERMS = (
    "multi-horizon",
    "multi horizon",
    "path_policy",
    "path20",
    "stage gate",
    "model quality",
    "training",
    "experiment",
    "evidence",
    "预算",
    "实验",
    "训练",
    "模型",
    "证据",
    "质量结论",
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
class RunEvidenceReport:
    run_tag: str
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
        payload["run_summary_json"] = payload.pop("study_summary_json", "")
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


def _task_is_experiment_review(task: str) -> bool:
    text = str(task or "").lower()
    return any(term in text for term in EXPERIMENT_REVIEW_TERMS)


def _walk_mappings(value: Any) -> list[Mapping[str, Any]]:
    mappings: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        mappings.append(value)
        for child in value.values():
            mappings.extend(_walk_mappings(child))
    elif isinstance(value, list):
        for child in value:
            mappings.extend(_walk_mappings(child))
    return mappings


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _training_summary_gaps(training_summary: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(training_summary, Mapping):
        return []
    gaps: list[str] = []
    config = training_summary.get("training_config") if isinstance(training_summary.get("training_config"), Mapping) else {}
    configured_epochs = _as_int(config.get("epochs") or config.get("max_epochs") or training_summary.get("epochs"))
    if configured_epochs is not None and configured_epochs <= 2:
        gaps.append("epochs<=2")

    seed_values: set[str] = set()
    has_learning_curve = False
    for mapping in _walk_mappings(training_summary):
        seed = mapping.get("seed")
        if seed is not None:
            seed_values.add(str(seed))
        if isinstance(mapping.get("learning_curve"), list) and mapping.get("learning_curve"):
            has_learning_curve = True
        epochs_ran = _as_int(mapping.get("epochs_ran") or mapping.get("epoch_count"))
        best_epoch = _as_int(mapping.get("best_epoch"))
        if epochs_ran is not None and epochs_ran <= 2:
            gaps.append("epochs_ran<=2")
        if epochs_ran is not None and best_epoch is not None and epochs_ran == best_epoch:
            gaps.append("best_epoch_at_last_epoch")
        stopped_reason = str(mapping.get("stopped_reason", "") or mapping.get("stop_reason", "") or "").lower()
        if stopped_reason == "max_epochs_reached":
            gaps.append("max_epochs_reached")
    if len(seed_values) == 1:
        gaps.append("single_seed")
    if not has_learning_curve:
        gaps.append("missing_learning_curve")
    return sorted(set(gaps))


def _heuristic_low_budget_task(task: str) -> bool:
    text = str(task or "").lower()
    return any(term in text for term in ("低预算", "2 epoch", "2epoch", "epochs=2", "scout", "smoke")) and any(
        term in text for term in ("质量结论", "模型质量", "实验", "证据", "review", "审阅")
    )


def detect_low_budget_evidence(
    *,
    task: str,
    training_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not _task_is_experiment_review(task):
        return {
            "status": "clear",
            "signals": [],
            "learning_opportunities": [],
            "evidence_grade": "",
            "next_actions": ["no_learning_needed"],
        }

    gaps = _training_summary_gaps(training_summary)
    low_budget = bool(gaps) or _heuristic_low_budget_task(task)
    if not low_budget:
        return {
            "status": "clear",
            "signals": [],
            "learning_opportunities": [],
            "evidence_grade": "",
            "next_actions": ["no_learning_needed"],
        }

    confidence = "high" if gaps else "medium"
    evidence_grade = "scout_only" if gaps or _heuristic_low_budget_task(task) or "scout" in str(task or "").lower() else "smoke_only"
    action = (
        "downgrade low-budget runs to smoke_only/scout_only, require enough epochs, effective early stopping, multi-seed checks, "
        "and validation convergence before using results as model-quality conclusions"
    )
    return {
        "status": "opportunity",
        "signals": ["low_budget_evidence_pollution"],
        "learning_opportunities": [
            {
                "target_layer": "experiment_governance",
                "owner_brain": "daily_research",
                "confidence": confidence,
                "recommended_action": action,
                "writeback_route": "daily_research/brain/knowledge_center.md",
                "verification_required": True,
                "detector_id": "daily_research_low_budget_detector",
                "source_signal": "low_budget_evidence_pollution",
                "evidence_gaps": gaps or ["task_reports_low_budget_evidence"],
            }
        ],
        "evidence_grade": evidence_grade,
        "next_actions": ["create_proposal"],
    }


def _extract_artifact_tag(payload: dict[str, Any]) -> str:
    for key in ("run_tag", "protocol_tag", "tag"):
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
    run_tag = records["latest_study_summary"].tag
    protocol_tag = records["latest_protocol_summary"].tag
    mismatch = bool(run_tag and protocol_tag and run_tag != protocol_tag)
    reason = ""
    if mismatch:
        reason = f"latest run tag differs from latest protocol tag: {run_tag} != {protocol_tag}"
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


def _study_summary_path(run_tag: str) -> Path:
    return CONTINUOUS_POLICY_OUTPUT_ROOT / "studies" / run_tag / "study_summary.json"


def _summary_path_candidates(run_tag: str, workflow: str | None = None) -> list[tuple[str, Path]]:
    workflow_key = str(workflow or "").strip()
    workflows = [workflow_key] if workflow_key else list(RUN_ROOTS)
    candidates: list[tuple[str, Path]] = []
    for candidate_workflow in workflows:
        root = RUN_ROOTS.get(candidate_workflow)
        if root is None:
            continue
        for filename in SUMMARY_FILENAMES:
            candidates.append((candidate_workflow, root / run_tag / filename))
    return candidates


def _empty_run_evidence_report(
    *,
    run_tag: str,
    workflow: str = "",
    searched_paths: list[str] | None = None,
    evidence_gaps: list[str] | None = None,
) -> RunEvidenceReport:
    paths = list(searched_paths or [])
    gaps = list(evidence_gaps or [])
    return RunEvidenceReport(
        run_tag=run_tag,
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


def resolve_run_evidence(run_tag: str, workflow: str | None = None) -> RunEvidenceReport:
    tag = str(run_tag or "").strip()
    if not tag:
        raise ValueError("run_tag is required")
    workflow_key = str(workflow or "").strip()
    if workflow_key and workflow_key not in RUN_ROOTS:
        return _empty_run_evidence_report(
            run_tag=tag,
            workflow=workflow_key,
            evidence_gaps=[f"unsupported_run_workflow: {workflow_key}"],
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
        return _empty_run_evidence_report(
            run_tag=tag,
            searched_paths=hit_paths,
            evidence_gaps=[f"ambiguous_run_tag: {tag}"],
        )
    if not hits:
        return _empty_run_evidence_report(
            run_tag=tag,
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
    summary_run_tag = str(summary.get("run_tag", "") or "") if summary else ""
    if summary and not summary_run_tag:
        gaps.append("run summary missing run_tag")
    elif summary and summary_run_tag != tag:
        gaps.append("run summary tag differs from requested tag")
    return RunEvidenceReport(
        run_tag=tag,
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
    path_text = ACTIVE_ARTIFACT.as_posix()
    absolute_path = WORKSPACE_ROOT / ACTIVE_ARTIFACT

    def run_git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    tracked_result = run_git("ls-files", "--error-unmatch", "--", path_text)
    tracked = tracked_result.returncode == 0
    exists = absolute_path.is_file()
    ignored_result = run_git("check-ignore", "-q", "--", path_text)
    ignored = ignored_result.returncode == 0
    diff_result: subprocess.CompletedProcess[str] | None = None
    diff_text = ""
    if tracked:
        # HEAD covers both staged and unstaged changes; a tracked deletion is dirty too.
        diff_result = run_git("diff", "--no-ext-diff", "HEAD", "--", path_text)
        diff_text = (diff_result.stdout or "") + (diff_result.stderr or "")
        status = "tracked_clean" if diff_result.returncode == 0 and not diff_text.strip() else "tracked_dirty"
    elif not exists:
        status = "missing"
    elif ignored:
        status = "ignored_untracked"
    else:
        status = "untracked"

    return {
        "path": path_text,
        "status": status,
        "exists": exists,
        "tracked": tracked,
        "ignored": ignored,
        "clean": status == "tracked_clean",
        "returncode": diff_result.returncode if diff_result is not None else tracked_result.returncode,
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
    del rule_report  # The artifact state is a direct sensor, not inferred from rule findings.
    return {
        "active_artifact_guard": active_artifact_diff_status(),
        "frontier_report": build_frontier_report(),
    }


def control_plane_doc_limits() -> dict[Path, int]:
    return dict(CONTROL_PLANE_DOC_LIMITS)
