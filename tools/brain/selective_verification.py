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


ACTIVE_ARTIFACT = "daily_research/output/active_execution_strategy.json"
BIG_ARTIFACT_PATHS = (
    "daily_research/output/**",
    "daily_research/cache/**",
    "daily_research/archive/**",
)

ALWAYS_COMMANDS = (
    f"git diff -- {ACTIVE_ARTIFACT}",
    "git diff --check",
    f"{PYTHON_EXECUTABLE} -m tools.brain.doc_guard check",
    f"{PYTHON_EXECUTABLE} -m tools.brain.integrity_check --json",
)

BRAIN_TOOL_TESTS = (
    "tools/brain/tests/test_capsule.py",
    "tools/brain/tests/test_evidence_registry.py",
    "tools/brain/tests/test_rules.py",
    "tools/brain/tests/test_workflow_cli.py",
    "tools/brain/tests/test_workspace_brain_skill.py",
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


def _pytest_command(paths: Iterable[str]) -> str:
    return f"{PYTHON_EXECUTABLE} -m pytest {' '.join(paths)} -q"


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
    if command not in commands:
        commands.append(command)


def _is_docs_only_path(path: str) -> bool:
    return path.endswith((".md", ".txt")) or path in {"README.md", "AGENTS.md", "CLAUDE.md", "SKILL.md"}


def build_verification_plan(*, paths: list[str] | None = None, base: str | None = None) -> dict[str, Any]:
    changed_paths = _unique(paths if paths is not None else _git_changed_paths(base=base))
    selected_commands: list[str] = []
    deferred_long_commands: list[str] = []
    warnings: list[str] = []
    risk_level = "low"
    manual_review_required = False

    for path in changed_paths:
        if path == ACTIVE_ARTIFACT:
            risk_level = "critical"
            manual_review_required = True
            warnings.append("active_artifact_diff_blocker: active execution artifact changed; stop and obtain promotion authority")
            continue

        if path.startswith("daily_research/execution/"):
            risk_level = _risk_max(risk_level, "high")
            manual_review_required = True
            warnings.append(f"execution_or_active_boundary: {path} requires manual review before completion")
            continue

        if path.startswith("daily_research/deep_alpha/") or path.startswith("daily_research/baseline/"):
            risk_level = _risk_max(risk_level, "high")
            manual_review_required = True
            warnings.append(f"limited_test_coverage_boundary: {path} requires manual review or added tests")
            continue

        if path.startswith("daily_research/data_lake/"):
            risk_level = _risk_max(risk_level, "medium")
            _add_command(selected_commands, _pytest_command(["daily_research/data_lake/tests"]))
            continue

        if path.startswith("tools/brain/"):
            risk_level = _risk_max(risk_level, "medium")
            if "selective_verification" in path:
                _add_command(
                    selected_commands,
                    _pytest_command(
                        [
                            "tools/brain/tests/test_selective_verification.py",
                            "tools/brain/tests/test_workflow_cli.py",
                        ]
                    ),
                )
            elif Path(path).name in {"capsule.py", "workflow.py", "platform.py", "rules.py", "doc_guard.py", "integrity_check.py"}:
                _add_command(selected_commands, _pytest_command(BRAIN_TOOL_TESTS))
            else:
                _add_command(selected_commands, _pytest_command(["tools/brain/tests"]))
            continue

        if path.startswith("brain/workflows/") or path.startswith("daily_research/brain/workflows/"):
            risk_level = _risk_max(risk_level, "medium")
            _add_command(selected_commands, _pytest_command(BRAIN_TOOL_TESTS))
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
                    _add_command(selected_commands, _pytest_command(["daily_research/path_policy/tests"]))
            continue

        if _is_docs_only_path(path):
            continue

        if path.endswith(".py"):
            risk_level = _risk_max(risk_level, "medium")
            manual_review_required = True
            warnings.append(f"unmapped_python_change: {path} has no selective verification rule")

    return {
        "schema_version": 1,
        "mode": "recommend_only",
        "base": str(base or ""),
        "changed_paths": changed_paths,
        "always_commands": list(ALWAYS_COMMANDS),
        "selected_commands": selected_commands,
        "deferred_long_commands": deferred_long_commands,
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
    print("\nDeferred long commands:")
    for command in payload["deferred_long_commands"] or ["<none>"]:
        print(f"- {command}")
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

