from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.brain.platform import PYTHON_EXECUTABLE, WORKSPACE_ROOT
from tools.brain.project_profiles import infer_project_id_from_paths, load_project_profile


ACTIVE_ARTIFACT = "daily_research/output/active_execution_strategy.json"
BIG_ARTIFACT_PATHS = (
    "daily_research/output/**",
    "daily_research/cache/**",
    "daily_research/archive/**",
)

ALWAYS_COMMANDS = (
    f"git diff -- {ACTIVE_ARTIFACT}",
    "git diff --check",
    f"{PYTHON_EXECUTABLE} -m tools.brain.doc_guard check --scope changed",
    f"{PYTHON_EXECUTABLE} -m tools.brain.integrity_check --json",
)

BRAIN_TOOL_FAST_TESTS = (
    "tools/brain/tests/test_selective_verification.py",
    "tools/brain/tests/test_project_commit.py",
    "tools/brain/tests/test_platform.py",
)

BRAIN_TOOL_CONTRACT_TESTS = (
    "tools/brain/tests/test_capsule.py",
    "tools/brain/tests/test_doc_guard.py",
    "tools/brain/tests/test_evidence_registry.py",
    "tools/brain/tests/test_platform.py",
    "tools/brain/tests/test_rules.py",
    "tools/brain/tests/test_workflow_cli.py::BrainWorkflowCliTest::test_bootstrap_cli_outputs_valid_json_capsule",
    "tools/brain/tests/test_workflow_cli.py::BrainWorkflowCliTest::test_route_cli_outputs_structured_workspace_target",
    "tools/brain/tests/test_workflow_cli.py::BrainWorkflowCliTest::test_verify_plan_cli_delegates_to_selective_verification",
    "tools/brain/tests/test_workspace_brain_skill_contract.py",
)

BRAIN_TOOL_PROCESS_TESTS = (
    "tools/brain/tests/test_agent_run.py",
    "tools/brain/tests/test_long_task_monitor.py",
)

BRAIN_TOOL_VERIFY_PLAN_TEST = "tools/brain/tests/test_workflow_cli.py::BrainWorkflowCliTest::test_verify_plan_cli_delegates_to_selective_verification"

DAILY_TOOL_HEALTH_GUARD_TESTS = (
    "tools/brain/tests/test_platform.py::BrainPlatformTest::test_health_default_skips_daily_strict_lanes",
    "tools/brain/tests/test_platform.py::BrainPlatformTest::test_health_full_daily_runs_project_and_openmp_strict_lanes",
    "tools/brain/tests/test_platform.py::BrainPlatformTest::test_health_standard_daily_doc_guard_stays_in_project_brain",
    "tools/brain/tests/test_workflow_cli.py::BrainWorkflowCliTest::test_health_cli_defaults_to_compact_checks",
)

BRIDGE_MATRIX_TESTS = (
    "daily_research/path_policy/tests/test_v2_score_backtest_bridge.py",
    "daily_research/path_policy/tests/test_v2_candidate_review_matrix.py",
)

HIGH_RETURN_DISCOVERY_TESTS = (
    "daily_research/path_policy/tests/test_v2_high_return_model_discovery.py",
    *BRIDGE_MATRIX_TESTS,
)

FORECAST_TESTS = (
    "daily_research/path_policy/tests/test_forecast_features.py",
    "daily_research/path_policy/tests/test_forecast_dataset.py::test_forecast_sequence_dataset_accepts_custom_horizon_grid_with_horizon_specific_risk",
    "daily_research/path_policy/tests/test_forecast_training.py::test_train_forecast_models_accepts_custom_horizon_decision_utility_contract",
    "daily_research/path_policy/tests/test_models.py",
)

FORECAST_DEFERRED_LONG_TESTS = (
    "daily_research/path_policy/tests/test_forecast_dataset.py",
    "daily_research/path_policy/tests/test_forecast_training.py",
)

RL_PROTOCOL_GROUPS = (
    (
        "daily_research/path_policy/tests/test_rl_protocol.py::test_rl_protocol_parser_accepts_sequence_stage_and_rejects_latest_in_main",
        "daily_research/path_policy/tests/test_rl_protocol.py::test_protocol_parser_defaults_to_current_neural_mainline_dataset_stage",
        "daily_research/path_policy/tests/test_rl_protocol.py::test_protocol_parser_accepts_forecast_walkforward_stage_with_neural_policy_defaults",
        "daily_research/path_policy/tests/test_rl_protocol.py::test_protocol_rejects_full_universe_forecast_eager_dataset_mode",
        "daily_research/path_policy/tests/test_rl_protocol.py::test_protocol_parser_accepts_walkforward_matrix_stage",
        "daily_research/path_policy/tests/test_rl_protocol.py::test_protocol_parser_accepts_v4_validation_repair_stage",
        "daily_research/path_policy/tests/test_rl_protocol.py::test_protocol_parser_accepts_v5_dt_validation_stage",
    ),
    (
        "daily_research/path_policy/tests/test_rl_protocol.py::test_forecast_walkforward_study_contract_fixture",
        "daily_research/path_policy/tests/test_rl_protocol.py::test_forecast_walkforward_study_memmap_contract_fixture",
        "daily_research/path_policy/tests/test_rl_protocol.py::test_current_neural_mainline_stage_no_longer_requires_legacy_allow_flag",
        "daily_research/path_policy/tests/test_rl_protocol.py::test_rl_stage_rejects_loose_latest_dataset_id",
        "daily_research/path_policy/tests/test_rl_protocol.py::test_protocol_parser_rejects_unknown_rl_model_family",
    ),
    (
        "daily_research/path_policy/tests/test_rl_protocol.py::test_multiyear_aggregate_excludes_incomplete_years",
        "daily_research/path_policy/tests/test_rl_protocol.py::test_walkforward_summary_separates_train_validation_test",
        "daily_research/path_policy/tests/test_rl_protocol.py::test_episode_artifact_reuse_and_hash_mismatch_failure",
        "daily_research/path_policy/tests/test_rl_protocol.py::test_walkforward_verdict_downgrades_on_projection_mismatch",
        "daily_research/path_policy/tests/test_rl_protocol.py::test_walkforward_verdict_requires_complete_validation_before_test_success",
        "daily_research/path_policy/tests/test_rl_protocol.py::test_walkforward_verdict_negative_validation_can_be_diagnosed_failure",
        "daily_research/path_policy/tests/test_rl_protocol.py::test_walkforward_evidence_diagnostics_flags_projection_and_exposure",
        "daily_research/path_policy/tests/test_rl_protocol.py::test_walkforward_matrix_outputs_two_model_families",
        "daily_research/path_policy/tests/test_matrix_evidence_diagnostics_selects_best_validation_family",
    ),
    (
        "daily_research/path_policy/tests/test_rl_protocol.py::test_v4_checkpoint_selection_ignores_test_metrics",
        "daily_research/path_policy/tests/test_rl_protocol.py::test_v4_verdict_requires_positive_validation_and_baseline_win",
        "daily_research/path_policy/tests/test_rl_protocol.py::test_v4_projection_mismatch_downgrades_verdict",
        "daily_research/path_policy/tests/test_rl_protocol.py::test_v4_multiseed_summary_calculates_distribution",
        "daily_research/path_policy/tests/test_rl_protocol.py::test_v4_baseline_targets_do_not_use_oracle_or_future_columns",
        "daily_research/path_policy/tests/test_rl_protocol.py::test_v4_v3_final_checkpoint_baseline_skips_without_explicit_model",
        "daily_research/path_policy/tests/test_rl_protocol.py::test_v4_v3_final_checkpoint_baseline_loads_explicit_checkpoint",
        "daily_research/path_policy/tests/test_rl_protocol.py::test_v4_validation_repair_study_contract_fixture",
        "daily_research/path_policy/tests/test_rl_protocol.py::test_v5_dt_validation_study_contract_fixture",
        "daily_research/path_policy/tests/test_rl_protocol.py::test_v5_long_context_incomplete_downgrades_verdict",
    ),
)

CONTINUOUS_SHARED_CORE = {
    "daily_research/continuous_policy/portfolio_simulator.py",
    "daily_research/continuous_policy/model_seq_v3.py",
    "daily_research/continuous_policy/pipeline_utils.py",
    "daily_research/continuous_policy/train_policy.py",
    "daily_research/continuous_policy/run_continuous_policy_protocol.py",
    "daily_research/continuous_policy/run_self_optimizing_study.py",
    "daily_research/continuous_policy/research_profile_registry.py",
    "daily_research/continuous_policy/training_contracts.py",
    "daily_research/continuous_policy/model_portfolio_set_v5.py",
    "daily_research/continuous_policy/decision_core_v6.py",
}

CONTINUOUS_MEDIUM_TESTS = (
    "daily_research/continuous_policy/tests/test_portfolio_cashflow_decision.py",
    "daily_research/continuous_policy/tests/test_portfolio_decision_features.py",
    "daily_research/continuous_policy/tests/test_protocol_profile_binding.py",
    "daily_research/continuous_policy/tests/test_lake_evaluator_cli_contract.py",
)

CONTINUOUS_LONG_TESTS = (
    "daily_research/continuous_policy/tests/test_portfolio_daily_strategy_contracts.py",
    "daily_research/continuous_policy/tests/test_allocation_closure_diagnostics.py",
    "daily_research/continuous_policy/tests/test_decision_core_v6.py",
)

RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
WORKSPACE_AREAS = (
    "brain",
    "tools/brain",
    "daily_research",
    "traditional_quant_research",
    "daily_stock_analysis-main",
    "t0_project",
)

TEST_LANE_BUDGETS = {
    "smoke": {
        "target_minutes": 2,
        "role": "fast changed-surface confidence for ordinary local loops",
        "default": True,
    },
    "project": {
        "target_minutes": 8,
        "role": "normal project gate excluding slow, research, external, and benchmark work",
        "default": False,
    },
    "full": {
        "target_minutes": None,
        "role": "all deterministic project checks, including deferred deterministic suites",
        "default": False,
    },
    "research": {
        "target_minutes": None,
        "role": "training, backtest, model, memmap-build, and research-regression checks",
        "default": False,
    },
    "external": {
        "target_minutes": None,
        "role": "network, live provider, third-party service, and benchmark checks",
        "default": False,
    },
}

TEST_BURDEN_RULES = (
    "default gates use changed-surface commands, not whole-project suites",
    "large memmap builds, training, full backtests, and live provider probes stay out of blocking_commands",
    "tests should protect contracts and safety boundaries with synthetic or minimal fixtures",
    "research evidence belongs in reports or references; tests only protect the generator contract",
    "stale tests for removed mechanisms should be archived or deleted during project-local cleanup",
)


def _pytest_command(paths: Iterable[str]) -> str:
    return f"{PYTHON_EXECUTABLE} -m pytest {' '.join(paths)} -q"


def _py_compile_command(paths: Iterable[str]) -> str:
    return f"{PYTHON_EXECUTABLE} -m py_compile {' '.join(_quote_command_arg(path) for path in paths)}"


def _normalize_path(path: str) -> str:
    normalized = str(path or "").strip().strip('"').strip("'").replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _unique(paths: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in paths:
        path = _normalize_path(raw)
        if not path or path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def _run_git(args: list[str]) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=str(WORKSPACE_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]


def _safe_pathspecs() -> list[str]:
    return [".", *(f":(exclude){pattern}" for pattern in BIG_ARTIFACT_PATHS)]


def _git_changed_paths(base: str | None = None) -> list[str]:
    paths: list[str] = []
    pathspecs = _safe_pathspecs()
    if base:
        paths.extend(_run_git(["diff", "--name-only", str(base), "--", *pathspecs]))
        paths.extend(_run_git(["diff", "--name-only", str(base), "--", ACTIVE_ARTIFACT]))
    else:
        paths.extend(_run_git(["diff", "--name-only", "--", *pathspecs]))
        paths.extend(_run_git(["diff", "--name-only", "--cached", "--", *pathspecs]))
        paths.extend(_run_git(["diff", "--name-only", "--", ACTIVE_ARTIFACT]))
        paths.extend(_run_git(["diff", "--name-only", "--cached", "--", ACTIVE_ARTIFACT]))
    paths.extend(_run_git(["ls-files", "--others", "--exclude-standard", "--", *pathspecs]))
    return _unique(sorted(paths))


def _risk_max(current: str, candidate: str) -> str:
    return candidate if RISK_ORDER[candidate] > RISK_ORDER[current] else current


def _add_command(commands: list[str], command: str) -> None:
    new_targets = _pytest_targets(command)
    if new_targets:
        for index, existing in enumerate(list(commands)):
            existing_targets = _pytest_targets(existing)
            if not existing_targets:
                continue
            if new_targets <= existing_targets:
                return
            if existing_targets < new_targets:
                commands[index] = command
                return
    if command not in commands:
        commands.append(command)


def _pytest_targets(command: str) -> set[str]:
    prefix = f"{PYTHON_EXECUTABLE} -m pytest "
    suffix = " -q"
    if not command.startswith(prefix) or not command.endswith(suffix):
        return set()
    target_text = command[len(prefix) : -len(suffix)].strip()
    return set(target_text.split()) if target_text else set()


def _is_docs_only_path(path: str) -> bool:
    return path.endswith((".md", ".txt")) or path in {"README.md", "AGENTS.md", "CLAUDE.md", "SKILL.md"}


def _area_for_path(path: str) -> str:
    normalized = _normalize_path(path)
    if normalized.startswith("tools/brain/"):
        return "tools/brain"
    if normalized.startswith("brain/"):
        return "brain"
    for area in WORKSPACE_AREAS:
        if normalized == area or normalized.startswith(f"{area}/"):
            return area
    return normalized.split("/", 1)[0] if normalized else "workspace"


def _is_brain_doc_path(path: str) -> bool:
    normalized = _normalize_path(path)
    return (
        normalized.startswith("brain/")
        or "/brain/" in normalized
        or normalized.startswith("brain/workflows/")
    ) and normalized.endswith((".md", ".json", ".txt"))


def _quote_command_arg(value: str) -> str:
    if not value:
        return '""'
    if any(char.isspace() for char in value):
        return '"' + value.replace('"', '\\"') + '"'
    return value


def _doc_guard_files_command(changed_paths: Iterable[str]) -> str | None:
    doc_paths = [_normalize_path(path) for path in changed_paths if _is_brain_doc_path(path)]
    if not doc_paths:
        return None
    files = " ".join(_quote_command_arg(path) for path in _unique(doc_paths))
    return f"{PYTHON_EXECUTABLE} -m tools.brain.doc_guard check --files {files}"


def _minimal_blocking_guards(changed_paths: list[str], *, active_artifact_blocked: bool) -> list[str]:
    if active_artifact_blocked or not changed_paths:
        return []
    commands = ["git diff --check"]
    doc_guard_command = _doc_guard_files_command(changed_paths)
    if doc_guard_command:
        commands.append(doc_guard_command)
    return commands


def _build_lane_commands(
    *,
    blocking_commands: list[str],
    deferred_commands: list[str],
    active_artifact_blocked: bool,
) -> dict[str, list[str]]:
    if active_artifact_blocked:
        return {lane: [] for lane in TEST_LANE_BUDGETS}
    return {
        "smoke": list(blocking_commands),
        "project": list(blocking_commands),
        "full": _unique([*blocking_commands, *deferred_commands]),
        "research": list(deferred_commands),
        "external": [],
    }


def _test_strategy_summary(project_id: str, risk_level: str, manual_review_required: bool) -> dict[str, Any]:
    return {
        "default_lane": "smoke",
        "coverage_policy": "changed_surface_only",
        "project_id": project_id,
        "risk_level": risk_level,
        "manual_review_required": manual_review_required,
        "lane_budgets": TEST_LANE_BUDGETS,
        "burden_rules": list(TEST_BURDEN_RULES),
        "slow_work_policy": "defer research, data-heavy, external, and benchmark checks unless explicitly requested or required by release risk",
    }


def _command_mentions_area(command: str, area: str) -> bool:
    if area == "brain":
        return " brain/" in command or "tools.brain.doc_guard" in command or "tools.brain.integrity_check" in command
    return area in command


def _skipped_reason_by_area(
    changed_paths: list[str],
    selected_commands: list[str],
    deferred_commands: list[str],
    warnings: list[str],
) -> dict[str, str]:
    touched_paths_by_area: dict[str, list[str]] = {}
    for path in changed_paths:
        touched_paths_by_area.setdefault(_area_for_path(path), []).append(path)
    reasons: dict[str, str] = {}
    for area in WORKSPACE_AREAS:
        paths = touched_paths_by_area.get(area, [])
        if not paths:
            reasons[area] = "unchanged_area_not_tested"
            continue
        if any(path == ACTIVE_ARTIFACT for path in paths):
            reasons[area] = "critical_active_artifact_blocker"
        elif all(_is_docs_only_path(path) or _is_brain_doc_path(path) for path in paths):
            reasons[area] = "docs_only_minimal_guards"
        elif any(_command_mentions_area(command, area) for command in selected_commands):
            reasons[area] = "changed_surface_selected"
        elif any(_command_mentions_area(command, area) for command in deferred_commands):
            reasons[area] = "changed_surface_deferred_only"
        elif any(path in warning for path in paths for warning in warnings):
            reasons[area] = "manual_review_required_no_direct_test"
        else:
            reasons[area] = "changed_surface_no_direct_test"
    return reasons


def _project_changed_surface_test_command(project_id: str, body_root: str, path: str) -> str | None:
    normalized = _normalize_path(path)
    test_root = f"{body_root}/tests"
    if normalized.startswith(f"{test_root}/") and normalized.endswith(".py"):
        return _pytest_command([normalized])

    stem = Path(normalized).stem
    if not stem:
        return None
    candidate = f"{test_root}/test_{stem}.py"
    if (WORKSPACE_ROOT / candidate).exists():
        return _pytest_command([candidate])
    return None


def build_verification_plan(*, paths: list[str] | None = None, base: str | None = None) -> dict[str, Any]:
    changed_paths = _unique(paths if paths is not None else _git_changed_paths(base=base))
    project_id = infer_project_id_from_paths(changed_paths)
    project_profile = load_project_profile(project_id if project_id != "cross_project" else "workspace")
    verification_profile = (
        project_profile.get("verification_profile", {}) if isinstance(project_profile.get("verification_profile"), dict) else {}
    )
    always_commands = list(verification_profile.get("always_commands", []) or ALWAYS_COMMANDS)
    body_root = str(project_profile.get("body_root", "") or "").strip().replace("\\", "/")
    selected_commands: list[str] = []
    deferred_long_commands: list[str] = []
    warnings: list[str] = []
    risk_level = "low"
    manual_review_required = False
    active_artifact_blocked = False

    for path in changed_paths:
        if project_id not in {"daily_research", "workspace", "cross_project"} and body_root and path.startswith(f"{body_root}/"):
            if path.endswith(".py"):
                risk_level = _risk_max(risk_level, "medium")
                command = _project_changed_surface_test_command(project_id, body_root, path)
                if command:
                    _add_command(selected_commands, command)
                else:
                    manual_review_required = True
                    warnings.append(f"unmapped_project_python_change: {path} has no same-surface test mapping")
            continue

        if path == ACTIVE_ARTIFACT:
            risk_level = "critical"
            manual_review_required = True
            active_artifact_blocked = True
            warnings.append("active_artifact_diff_blocker: active execution artifact changed; stop and obtain promotion authority")
            continue

        if path.startswith("daily_research/execution/"):
            risk_level = _risk_max(risk_level, "high")
            manual_review_required = True
            warnings.append(f"execution_or_active_boundary: {path} requires manual review before completion")
            name = Path(path).name
            test_name = f"daily_research/execution/tests/test_{name}"
            if test_name.endswith(".py") and (WORKSPACE_ROOT / test_name).exists():
                _add_command(selected_commands, _pytest_command([test_name]))
            continue

        if path.startswith("daily_research/deep_alpha/") or path.startswith("daily_research/baseline/"):
            risk_level = _risk_max(risk_level, "high")
            manual_review_required = True
            warnings.append(f"limited_test_coverage_boundary: {path} requires manual review or added tests")
            if path.startswith("daily_research/baseline/"):
                name = Path(path).name
                test_name = f"daily_research/baseline/tests/test_{name}"
                if test_name.endswith(".py") and (WORKSPACE_ROOT / test_name).exists():
                    _add_command(selected_commands, _pytest_command([test_name]))
            continue

        if path.startswith("daily_research/data_lake/"):
            risk_level = _risk_max(risk_level, "medium")
            _add_command(selected_commands, _pytest_command(["daily_research/data_lake/tests"]))
            continue

        if path.startswith("daily_research/data_platform/"):
            risk_level = _risk_max(risk_level, "medium")
            _add_command(selected_commands, _pytest_command(["daily_research/data_platform/tests"]))
            continue

        if path.startswith("tools/brain/"):
            risk_level = _risk_max(risk_level, "medium")
            if "selective_verification" in path:
                _add_command(
                    selected_commands,
                    _pytest_command(
                        [
                            *BRAIN_TOOL_FAST_TESTS,
                            BRAIN_TOOL_VERIFY_PLAN_TEST,
                        ]
                    ),
                )
            elif Path(path).name in {"agent_run.py", "long_task_monitor.py"}:
                _add_command(selected_commands, _pytest_command(BRAIN_TOOL_PROCESS_TESTS))
            elif Path(path).name in {"capsule.py", "workflow.py", "platform.py", "rules.py", "doc_guard.py", "integrity_check.py"}:
                _add_command(selected_commands, _pytest_command(BRAIN_TOOL_CONTRACT_TESTS))
            else:
                _add_command(selected_commands, _pytest_command(BRAIN_TOOL_FAST_TESTS))
            continue

        if path.startswith("brain/workflows/") or path.startswith("daily_research/brain/workflows/"):
            risk_level = _risk_max(risk_level, "medium")
            _add_command(selected_commands, _pytest_command(BRAIN_TOOL_CONTRACT_TESTS))
            continue

        if path in CONTINUOUS_SHARED_CORE:
            risk_level = _risk_max(risk_level, "high")
            warnings.append(f"continuous_policy_shared_core: {path} changed; defer full evidence suite unless release/merge risk requires it")
            _add_command(selected_commands, _pytest_command(CONTINUOUS_MEDIUM_TESTS))
            _add_command(deferred_long_commands, _pytest_command(CONTINUOUS_LONG_TESTS))
            continue

        if path.startswith("daily_research/continuous_policy/"):
            risk_level = _risk_max(risk_level, "medium")
            _add_command(
                selected_commands,
                _pytest_command(
                    [
                        "daily_research/continuous_policy/tests/test_research_registry_simplification.py",
                        "daily_research/continuous_policy/tests/test_protocol_profile_binding.py",
                    ]
                ),
            )
            continue

        if path == "daily_research/path_policy/run_alpha_path20_protocol.py":
            risk_level = _risk_max(risk_level, "high")
            for group in RL_PROTOCOL_GROUPS:
                _add_command(selected_commands, _pytest_command(group))
            _add_command(
                selected_commands,
                _pytest_command(["daily_research/path_policy/tests/test_run_alpha_path20_protocol_entrypoint.py"]),
            )
            _add_command(
                deferred_long_commands,
                _pytest_command(["daily_research/path_policy/tests/test_rl_protocol.py"]),
            )
            continue

        if path.startswith("daily_research/path_policy/"):
            risk_level = _risk_max(risk_level, "medium")
            name = Path(path).name
            if name.startswith("forecast_") or "forecast" in name:
                _add_command(selected_commands, _pytest_command(FORECAST_TESTS))
                _add_command(deferred_long_commands, _pytest_command(FORECAST_DEFERRED_LONG_TESTS))
            elif name == "v2_score_backtest_bridge.py":
                _add_command(selected_commands, _pytest_command(BRIDGE_MATRIX_TESTS))
            elif name == "v2_candidate_review_matrix.py":
                _add_command(selected_commands, _pytest_command(["daily_research/path_policy/tests/test_v2_candidate_review_matrix.py"]))
            elif name == "v2_high_return_model_discovery.py":
                _add_command(selected_commands, _pytest_command(HIGH_RETURN_DISCOVERY_TESTS))
            elif name.startswith("rl_"):
                _add_command(
                    selected_commands,
                    _pytest_command(
                        [
                            "daily_research/path_policy/tests/test_rl_dataset.py",
                            "daily_research/path_policy/tests/test_rl_episode.py",
                            "daily_research/path_policy/tests/test_rl_models.py",
                            "daily_research/path_policy/tests/test_rl_replay.py",
                        ]
                    ),
                )
            else:
                test_name = f"daily_research/path_policy/tests/test_{name}"
                if test_name.endswith(".py") and (WORKSPACE_ROOT / test_name).exists():
                    _add_command(selected_commands, _pytest_command([test_name]))
                else:
                    manual_review_required = True
                    warnings.append(f"unmapped_python_change: {path} has no selective verification rule")
            continue

        if path.startswith("daily_research/tools/"):
            risk_level = _risk_max(risk_level, "medium")
            name = Path(path).name
            _add_command(selected_commands, _py_compile_command([path]))
            if name in {"project_consistency_check.py", "openmp_runtime_check.py"}:
                _add_command(selected_commands, _pytest_command(DAILY_TOOL_HEALTH_GUARD_TESTS))
            elif name in {"workspace_maintenance.py"}:
                warnings.append(f"maintenance_tool_smoke_only: {path} is maintenance-only; run explicit maintenance review before destructive cleanup")
            continue

        if _is_docs_only_path(path):
            continue

        if path.endswith(".py"):
            risk_level = _risk_max(risk_level, "medium")
            manual_review_required = True
            warnings.append(f"unmapped_python_change: {path} has no selective verification rule")

    blocking_commands = (
        []
        if active_artifact_blocked
        else [
            *_minimal_blocking_guards(changed_paths, active_artifact_blocked=active_artifact_blocked),
            *selected_commands,
        ]
    )
    deferred_commands = list(deferred_long_commands)
    lane_commands = _build_lane_commands(
        blocking_commands=blocking_commands,
        deferred_commands=deferred_commands,
        active_artifact_blocked=active_artifact_blocked,
    )
    return {
        "schema_version": 2,
        "mode": "recommend_only",
        "coverage_policy": "changed_surface_only",
        "base": str(base or ""),
        "project_id": project_id,
        "project_profile": project_profile,
        "changed_paths": changed_paths,
        "always_commands": always_commands,
        "selected_commands": selected_commands,
        "deferred_long_commands": deferred_long_commands,
        "blocking_commands": blocking_commands,
        "deferred_commands": deferred_commands,
        "lane_commands": lane_commands,
        "test_strategy": _test_strategy_summary(project_id, risk_level, manual_review_required),
        "skipped_reason_by_area": _skipped_reason_by_area(changed_paths, selected_commands, deferred_commands, warnings),
        "risk_level": risk_level,
        "warnings": warnings,
        "manual_review_required": manual_review_required,
    }


def _print_text(payload: dict[str, Any]) -> None:
    print("Selective verification plan")
    print(f"risk_level: {payload['risk_level']}")
    print(f"manual_review_required: {str(payload['manual_review_required']).lower()}")
    print("\nChanged paths:")
    for path in payload["changed_paths"] or ["<none>"]:
        print(f"- {path}")
    print("\nAlways-run guards:")
    for command in payload["always_commands"]:
        print(f"- {command}")
    print("\nSelected commands:")
    for command in payload["selected_commands"] or ["<none>"]:
        print(f"- {command}")
    print("\nBlocking commands:")
    for command in payload["blocking_commands"] or ["<none>"]:
        print(f"- {command}")
    print("\nDeferred long commands:")
    for command in payload["deferred_long_commands"] or ["<none>"]:
        print(f"- {command}")
    print("\nDeferred commands:")
    for command in payload["deferred_commands"] or ["<none>"]:
        print(f"- {command}")
    print("\nTest lanes:")
    for lane, commands in payload.get("lane_commands", {}).items():
        budget = payload.get("test_strategy", {}).get("lane_budgets", {}).get(lane, {})
        target = budget.get("target_minutes")
        target_text = "unbounded" if target is None else f"<= {target} min"
        print(f"- {lane}: {len(commands)} command(s), target {target_text}")
    if payload["warnings"]:
        print("\nWarnings:")
        for warning in payload["warnings"]:
            print(f"- {warning}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recommend daily_research verification commands from changed paths.")
    parser.add_argument("--base", default="", help="Compare changed paths against this revision when --paths is absent.")
    parser.add_argument("--paths", nargs="*", default=None, help="Explicit changed paths; takes priority over --base.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = build_verification_plan(
        paths=list(args.paths) if args.paths is not None else None,
        base=str(args.base or "") or None,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
