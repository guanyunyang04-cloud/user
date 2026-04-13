from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = WORKSPACE_ROOT / "daily_research"
OUTPUT_ROOT = PROJECT_ROOT / "output"

EXPECTED_TARGET_WEIGHT_SEMANTICS = "research_raw_target_weight"
EXPECTED_TARGET_WEIGHT_CAP_MODE = "follow_research_raw_no_global_cap"
MIN_DEFAULT_EPOCHS = 32


@dataclass(frozen=True)
class CheckResult:
    code: str
    detail: str


def _read_text(relative_path: str) -> str:
    return (WORKSPACE_ROOT / relative_path).read_text(encoding="utf-8")


def _read_json(relative_path: str) -> dict[str, Any]:
    payload = json.loads((WORKSPACE_ROOT / relative_path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _parse_default_int(text: str, flag: str) -> int:
    pattern = rf'add_argument\("{re.escape(flag)}".*?default=([0-9]+)'
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not parse default integer for {flag}")
    return int(match.group(1))


def _has_default_expression(text: str, flag: str, snippet_regex: str) -> bool:
    pattern = rf'add_argument\("{re.escape(flag)}".*?default={snippet_regex}'
    return re.search(pattern, text, flags=re.DOTALL) is not None


def _parse_default_budget_start(text: str, flag: str) -> int:
    pattern = rf'add_argument\("{re.escape(flag)}".*?default="([^"]+)"'
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not parse default budget list for {flag}")
    first = str(match.group(1)).split(",", 1)[0].strip()
    return int(first)


def _require(condition: bool, failures: list[CheckResult], code: str, detail: str) -> None:
    if not condition:
        failures.append(CheckResult(code=code, detail=detail))


def _check_train_entrypoints(failures: list[CheckResult]) -> None:
    train_text = _read_text("daily_research/deep_alpha/run_deep_alpha_research.py")
    pretrain_text = _read_text("daily_research/deep_alpha/pretrain_deep_alpha_encoder.py")

    _require(
        _has_default_expression(train_text, "--epochs", "DEFAULT_MIN_START_EPOCH_BUDGET"),
        failures,
        "train_default_epochs_too_low",
        "run_deep_alpha_research.py default --epochs must start at >= 32.",
    )
    _require(
        _has_default_expression(
            train_text,
            "--min-epochs",
            "default_min_epochs_for_budget\\(DEFAULT_MIN_START_EPOCH_BUDGET\\)",
        ),
        failures,
        "train_default_min_epochs_too_low",
        "run_deep_alpha_research.py default --min-epochs must follow the 32-start policy.",
    )
    _require(
        "No CUDA device is available" in train_text and 'torch.device("cuda")' in train_text,
        failures,
        "train_gpu_guard_missing",
        "run_deep_alpha_research.py must require CUDA.",
    )
    _require(
        'torch.device("cuda" if torch.cuda.is_available() else "cpu")' not in train_text,
        failures,
        "train_cpu_fallback_present",
        "run_deep_alpha_research.py still contains a CPU fallback path.",
    )

    _require(
        _has_default_expression(pretrain_text, "--epochs", "DEFAULT_MIN_START_EPOCH_BUDGET"),
        failures,
        "pretrain_default_epochs_too_low",
        "pretrain_deep_alpha_encoder.py default --epochs must start at >= 32.",
    )
    _require(
        _has_default_expression(
            pretrain_text,
            "--min-epochs",
            "default_min_epochs_for_budget\\(DEFAULT_MIN_START_EPOCH_BUDGET\\)",
        ),
        failures,
        "pretrain_default_min_epochs_too_low",
        "pretrain_deep_alpha_encoder.py default --min-epochs must follow the 32-start policy.",
    )
    _require(
        _parse_default_int(pretrain_text, "--max-total-epochs") >= MIN_DEFAULT_EPOCHS,
        failures,
        "pretrain_max_total_epochs_too_low",
        "pretrain_deep_alpha_encoder.py default --max-total-epochs must not cap below the start budget.",
    )
    _require(
        "No CUDA device is available" in pretrain_text and 'torch.device("cuda")' in pretrain_text,
        failures,
        "pretrain_gpu_guard_missing",
        "pretrain_deep_alpha_encoder.py must require CUDA.",
    )
    _require(
        'torch.device("cuda" if torch.cuda.is_available() else "cpu")' not in pretrain_text,
        failures,
        "pretrain_cpu_fallback_present",
        "pretrain_deep_alpha_encoder.py still contains a CPU fallback path.",
    )


def _check_config_dataclass_defaults(failures: list[CheckResult]) -> None:
    config_text = _read_text("daily_research/deep_alpha/config.py")
    _require(
        "epochs: int = DEFAULT_MIN_START_EPOCH_BUDGET" in config_text,
        failures,
        "config_default_epochs_too_low",
        "DeepAlphaConfig must inherit the 32-start policy at the dataclass level.",
    )
    _require(
        "min_epochs: int = default_min_epochs_for_budget(DEFAULT_MIN_START_EPOCH_BUDGET)" in config_text,
        failures,
        "config_default_min_epochs_too_low",
        "DeepAlphaConfig must inherit the current min-epoch floor at the dataclass level.",
    )


def _check_execution_profile_defaults(failures: list[CheckResult]) -> None:
    train_text = _read_text("daily_research/deep_alpha/run_deep_alpha_research.py")
    frequency_matrix_text = _read_text("daily_research/deep_alpha/run_retrain_frequency_formal_matrix.py")
    _require(
        'default="regoff_k1_5d_ensemble_native_anchor"' in train_text,
        failures,
        "train_execution_profile_default_stale",
        "run_deep_alpha_research.py still defaults to a stale execution-alignment profile.",
    )
    _require(
        'metrics.get("execution_alignment_profile", "regoff_k1_5d_ensemble_native_anchor")' in frequency_matrix_text,
        failures,
        "frequency_matrix_execution_profile_default_stale",
        "run_retrain_frequency_formal_matrix.py still falls back to a stale execution-alignment profile.",
    )


def _check_aux_defaults(failures: list[CheckResult]) -> None:
    scripts = {
        "daily_research/deep_alpha/run_minimal_matrix.py": (
            ("--epochs", "DEFAULT_MIN_START_EPOCH_BUDGET"),
            ("--min-epochs", "default_min_epochs_for_budget\\(DEFAULT_MIN_START_EPOCH_BUDGET\\)"),
            ("--pretrain-epochs", "DEFAULT_MIN_START_EPOCH_BUDGET"),
            ("--pretrain-min-epochs", "default_min_epochs_for_budget\\(DEFAULT_MIN_START_EPOCH_BUDGET\\)"),
            ("--max-total-epochs", MIN_DEFAULT_EPOCHS),
        ),
        "daily_research/deep_alpha/run_epoch_budget_formal_matrix.py": (("--epoch-budgets", MIN_DEFAULT_EPOCHS),),
        "daily_research/deep_alpha/run_graph_off_plain_budget_review.py": (("--epoch-budgets", MIN_DEFAULT_EPOCHS),),
        "daily_research/deep_alpha/run_encoder_transformer_stability_review.py": (("--epoch-budgets", MIN_DEFAULT_EPOCHS),),
        "daily_research/deep_alpha/run_short_alpha_production_epoch_extension.py": (("--epoch-budgets", MIN_DEFAULT_EPOCHS),),
    }
    for relative_path, checks in scripts.items():
        text = _read_text(relative_path)
        for flag, minimum in checks:
            if flag == "--epoch-budgets":
                value = _parse_default_budget_start(text, flag)
                _require(
                    value >= int(minimum),
                    failures,
                    "aux_default_too_low",
                    f"{relative_path} has stale default for {flag}: {value} < {minimum}.",
                )
            elif isinstance(minimum, str):
                _require(
                    _has_default_expression(text, flag, minimum),
                    failures,
                    "aux_default_too_low",
                    f"{relative_path} is missing the current shared default expression for {flag}.",
                )
            else:
                value = _parse_default_int(text, flag)
                _require(
                    value >= int(minimum),
                    failures,
                    "aux_default_too_low",
                    f"{relative_path} has stale default for {flag}: {value} < {minimum}.",
                )


def _check_execution_semantics(failures: list[CheckResult]) -> None:
    manifest = _read_json("daily_research/output/active_execution_strategy.json")
    _require(
        str(manifest.get("target_weight_semantics", "")).strip() == EXPECTED_TARGET_WEIGHT_SEMANTICS,
        failures,
        "active_manifest_target_weight_semantics_mismatch",
        "active_execution_strategy.json is missing research_raw_target_weight semantics.",
    )
    _require(
        str(manifest.get("target_weight_cap_mode", "")).strip() == EXPECTED_TARGET_WEIGHT_CAP_MODE,
        failures,
        "active_manifest_target_weight_cap_mode_mismatch",
        "active_execution_strategy.json is missing follow_research_raw_no_global_cap semantics.",
    )
    _require(
        float(manifest.get("transaction_cost_bps", 0.0) or 0.0) > 0.0
        and float(manifest.get("slippage_bps", 0.0) or 0.0) > 0.0
        and float(manifest.get("sell_tax_bps", 0.0) or 0.0) > 0.0,
        failures,
        "active_manifest_cost_semantics_missing",
        "active_execution_strategy.json must carry non-zero transaction/slippage/tax settings.",
    )
    _require(
        str(manifest.get("score_panel_role", "")).strip() == "execution_preweight_score_panel",
        failures,
        "active_manifest_score_role_mismatch",
        "active_execution_strategy.json must expose an execution pre-weight score panel role, not a stale static reference role.",
    )
    _require(
        str(manifest.get("effective_live_target_weight_mode", "")).strip() != "",
        failures,
        "active_manifest_effective_live_mode_missing",
        "active_execution_strategy.json must expose the current effective live target-weight mode.",
    )
    _require(
        str(manifest.get("effective_live_execution_profile", "")).strip() != "",
        failures,
        "active_manifest_effective_live_profile_missing",
        "active_execution_strategy.json must expose the current effective live execution profile.",
    )
    _require(
        str(manifest.get("effective_live_weight_generation_note", "")).strip() != "",
        failures,
        "active_manifest_effective_weight_note_missing",
        "active_execution_strategy.json must explain how current live weights are generated.",
    )
    production_root = Path(str(manifest.get("production_root", "")).strip())
    if production_root.exists():
        live_monitor_path = production_root / "live_trigger_monitor.json"
        score_reference_path = production_root / "daily_live_score_reference.json"
        if live_monitor_path.exists():
            live_monitor = _read_json(str(live_monitor_path.relative_to(WORKSPACE_ROOT)))
            _require(
                str(manifest.get("effective_live_target_weight_mode", "")).strip()
                == str(live_monitor.get("live_target_weight_mode", "")).strip(),
                failures,
                "active_manifest_effective_mode_out_of_sync",
                "active_execution_strategy.json current live mode is out of sync with live_trigger_monitor.json.",
            )
            _require(
                str(manifest.get("effective_live_execution_profile", "")).strip()
                == str(
                    live_monitor.get("effective_execution_profile", "")
                    or live_monitor.get("latest_selected_profile", "")
                ).strip(),
                failures,
                "active_manifest_effective_profile_out_of_sync",
                "active_execution_strategy.json current live profile is out of sync with live_trigger_monitor.json.",
            )
        if score_reference_path.exists():
            score_reference = _read_json(str(score_reference_path.relative_to(WORKSPACE_ROOT)))
            _require(
                str(manifest.get("effective_live_weight_generation_note", "")).strip()
                == str(score_reference.get("weight_generation_note", "")).strip(),
                failures,
                "active_manifest_weight_note_out_of_sync",
                "active_execution_strategy.json current live weight-generation note is out of sync with daily_live_score_reference.json.",
            )


def _check_no_hardcoded_operational_roots(failures: list[CheckResult]) -> None:
    files = (
        "daily_research/execution/run_short_alpha_execution_single_mapping_candidate_pipeline.py",
        "daily_research/execution/activate_execution_single_mapping_candidate.py",
    )
    forbidden_patterns = (
        "short_alpha_targeted_weak_month_repair_regime_firstweek_combo_expand_stable_topk3_review_20",
        "short_alpha_production_execution_policy_audit_20",
        "short_alpha_execution_single_mapping_candidate_pipeline_20",
    )
    for relative_path in files:
        text = _read_text(relative_path)
        for pattern in forbidden_patterns:
            _require(
                pattern not in text,
                failures,
                "hardcoded_operational_root_present",
                f"{relative_path} still hardcodes a dated operational root containing `{pattern}`.",
            )


def _check_execution_pipeline_consistency(failures: list[CheckResult]) -> None:
    production_refresh_text = _read_text("daily_research/execution/update_default_candidate_production.py")
    pipeline_text = _read_text("daily_research/execution/run_short_alpha_execution_single_mapping_candidate_pipeline.py")
    candidate_profiles_text = _read_text("daily_research/execution/research_candidate_profiles.py")
    trade_plan_text = _read_text("daily_research/baseline/generate_daily_trade_plan.py")

    _require(
        '"--min-epochs", 1' not in production_refresh_text,
        failures,
        "production_refresh_min_epochs_stale",
        "update_default_candidate_production.py still hardcodes --min-epochs 1.",
    )
    _require(
        "default_min_epochs_for_budget(epoch_budget)" in production_refresh_text,
        failures,
        "production_refresh_min_epochs_not_aligned",
        "update_default_candidate_production.py must align min epochs with the current 32-start policy.",
    )
    _require(
        "static_fallback_daily_live_target_weight_panel.csv" in production_refresh_text,
        failures,
        "production_refresh_static_fallback_missing",
        "update_default_candidate_production.py must materialize static_fallback_daily_live_target_weight_panel.csv.",
    )
    _require(
        "portfolio_capped_daily_live_target_weight_panel.csv" in production_refresh_text,
        failures,
        "production_refresh_portfolio_capped_reference_missing",
        "update_default_candidate_production.py must preserve the capped reference panel explicitly.",
    )
    _require(
        'DEFAULT_FALLBACK_LIVE_TARGET_WEIGHT_PANEL = STATIC_PRODUCTION_ROOT / "static_fallback_daily_live_target_weight_panel.csv"'
        in pipeline_text,
        failures,
        "single_mapping_pipeline_fallback_not_unified",
        "run_short_alpha_execution_single_mapping_candidate_pipeline.py must use the static fallback panel.",
    )
    _require(
        "fallback_target_weight_semantics" in pipeline_text and "follow_research_raw_no_global_cap" in pipeline_text,
        failures,
        "single_mapping_pipeline_fallback_semantics_missing",
        "run_short_alpha_execution_single_mapping_candidate_pipeline.py must record unified raw-weight fallback semantics.",
    )
    _require(
        '"role": "execution_preweight_score_panel"' in pipeline_text
        and "_build_hybrid_preweight_score_panel" in pipeline_text,
        failures,
        "single_mapping_pipeline_companion_score_missing",
        "run_short_alpha_execution_single_mapping_candidate_pipeline.py must materialize the pre-weight raw score panel for the selected execution path.",
    )
    _require(
        '"weight_generation_note"' in pipeline_text
        and '"effective_execution_bridge_meta"' in pipeline_text,
        failures,
        "single_mapping_pipeline_effective_state_missing",
        "run_short_alpha_execution_single_mapping_candidate_pipeline.py must describe the current effective execution path, not only the selected score file.",
    )
    _require(
        "_seed_live_only_formal_reference" in pipeline_text,
        failures,
        "single_mapping_pipeline_live_only_seed_missing",
        "run_short_alpha_execution_single_mapping_candidate_pipeline.py must support seeding formal references into a fresh live-only root.",
    )
    _require(
        "--transaction-cost-bps" in candidate_profiles_text
        and "--slippage-bps" in candidate_profiles_text
        and "--sell-tax-bps" in candidate_profiles_text,
        failures,
        "candidate_profile_cost_defaults_missing",
        "research_candidate_profiles.py must inject default cost settings for active candidate backtests.",
    )
    _require(
        '--transaction-cost-bps", type=float, default=None' in trade_plan_text
        and '--slippage-bps", type=float, default=None' in trade_plan_text
        and '--sell-tax-bps", type=float, default=None' in trade_plan_text,
        failures,
        "trade_plan_cost_flags_missing",
        "generate_daily_trade_plan.py must accept manifest-driven cost flags so run_trade_plan.py cannot drift and fail.",
    )
    _require(
        "_merge_score_reference_metadata" in trade_plan_text
        and "effective_live_weight_generation_note" in trade_plan_text,
        failures,
        "trade_plan_effective_execution_explanation_missing",
        "generate_daily_trade_plan.py must expose the effective live execution explanation when displaying external target-weight plans.",
    )


def _check_environment_source_of_truth(failures: list[CheckResult]) -> None:
    env_path = PROJECT_ROOT / "environment.yml"
    _require(
        env_path.exists(),
        failures,
        "environment_source_missing",
        "daily_research/environment.yml must exist as the authoritative dependency environment source.",
    )
    if not env_path.exists():
        return

    env_text = env_path.read_text(encoding="utf-8")
    required_snippets = (
        "name: yolos",
        "- python=3.11",
        "- numpy",
        "- pandas",
        "- scipy",
        "- scikit-learn",
        "- joblib",
        "- lightgbm",
        "- pytorch",
        "- pytorch-cuda=12.4",
        "- fastapi",
        "- uvicorn",
        "- jinja2",
    )
    for snippet in required_snippets:
        _require(
            snippet in env_text,
            failures,
            "environment_source_incomplete",
            f"daily_research/environment.yml is missing required runtime dependency marker: {snippet}",
        )


def _check_project_python_runtime_contract(failures: list[CheckResult]) -> None:
    guardrails_text = _read_text("daily_research/deep_alpha/experiment_guardrails.py")
    _require(
        r"C:\Users\ASUS\miniconda3\envs\yolos\python.exe" in guardrails_text,
        failures,
        "project_python_yolos_anchor_missing",
        "experiment_guardrails.py must anchor the project runtime to the yolos python.",
    )
    _require(
        r"C:\Users\ASUS\miniconda3\envs\quant\python.exe" not in guardrails_text,
        failures,
        "project_python_quant_fallback_present",
        "experiment_guardrails.py must not keep the old quant python fallback.",
    )

    raw_runtime_patterns = (
        re.compile(r"default\s*=\s*sys\.executable"),
        re.compile(r"\[\s*sys\.executable\s*,", flags=re.DOTALL),
    )
    offenders: list[str] = []
    for path in PROJECT_ROOT.rglob("*.py"):
        relative_path = path.relative_to(WORKSPACE_ROOT).as_posix()
        if relative_path == "daily_research/deep_alpha/experiment_guardrails.py":
            continue
        text = path.read_text(encoding="utf-8")
        if any(pattern.search(text) for pattern in raw_runtime_patterns):
            offenders.append(relative_path)
    _require(
        not offenders,
        failures,
        "project_python_default_not_locked_to_yolos",
        "daily_research scripts still contain raw sys.executable defaults or subprocess launches: "
        + ", ".join(offenders[:12])
        + (" ..." if len(offenders) > 12 else ""),
    )


def _check_execution_application_contract(failures: list[CheckResult]) -> None:
    required_files = (
        "daily_research/execution/app.py",
        "daily_research/execution/app_runtime.py",
        "daily_research/execution/app_service.py",
        "daily_research/execution/app_tasks.py",
        "daily_research/execution/run_execution_app.py",
        "daily_research/execution/__main__.py",
        "daily_research/execution/web_server.py",
        "daily_research/execution/web_service.py",
        "daily_research/execution/web_models.py",
        "daily_research/execution/run_execution_web.py",
        "daily_research/execution/web/templates/base.html",
        "daily_research/execution/web/templates/dashboard.html",
        "daily_research/execution/web/templates/tasks.html",
        "daily_research/execution/web/templates/jobs.html",
        "daily_research/execution/web/templates/job_detail.html",
        "daily_research/execution/web/templates/doctor.html",
        "daily_research/execution/web/templates/trade_plan.html",
        "daily_research/execution/web/templates/guide.html",
        "daily_research/execution/web/templates/runtime.html",
        "daily_research/execution/web/templates/account.html",
        "daily_research/execution/web/static/execution_console.css",
        "daily_research/execution/web/static/execution_console.js",
        "daily_research/execution/使用教程.md",
    )
    for relative_path in required_files:
        _require(
            (WORKSPACE_ROOT / relative_path).exists(),
            failures,
            "execution_app_file_missing",
            f"{relative_path} must exist as part of the unified execution application.",
        )

    app_text = _read_text("daily_research/execution/app.py")
    for snippet in ('"tasks"', '"status"', '"doctor"', '"run"', '"resume"', '"tail"', '"unlock"', '"web"'):
        _require(
            snippet in app_text,
            failures,
            "execution_app_command_missing",
            f"execution app is missing required command registration: {snippet}",
        )


def _check_memory_sync(failures: list[CheckResult]) -> None:
    required_strings = {
        "daily_research/brain/identity_layer.md": (
            "Agent 无状态，项目大脑有状态",
            "正式生产研究与执行主线",
            "strict resume",
            "前台执行",
            "默认追求最有效",
        ),
        "daily_research/brain/state_center.md": (
            EXPECTED_TARGET_WEIGHT_SEMANTICS,
            "当前接管摘要",
            "当前状态",
            "当前主问题",
            "当前优先级",
            "当前边界",
            "当前时态",
            "任何程序都必须在 `yolos` 环境下运行",
            "run_execution_app.py",
            "Web 控制台",
            "Start-Job",
        ),
        "daily_research/brain/knowledge_center.md": (
            EXPECTED_TARGET_WEIGHT_SEMANTICS,
            EXPECTED_TARGET_WEIGHT_CAP_MODE,
            "strict resume",
            "formal 验证采用滚动窗口协议",
            "recent 验证现在是 strongest-model 研究闭环必备伴随证据",
            "recent 胜利不能直接当 promotion 结论",
            "任何程序都必须在 `yolos` 环境下运行",
            "run_execution_app.py",
            "execution_app",
            "FastAPI",
            "简体中文",
            "Start-Job",
        ),
        "daily_research/brain/governance_layer.md": (
            "目标一致性检查",
            "规则冲突检查",
            "经验教训检查",
            "依赖完整性检查",
            "project_consistency_check.py",
        ),
        "daily_research/brain/operations_center.md": (
            EXPECTED_TARGET_WEIGHT_SEMANTICS,
            EXPECTED_TARGET_WEIGHT_CAP_MODE,
            "项目地图",
            "高频命令",
            "环境基线",
            "写回路由",
            "任何程序都必须在 `yolos` 环境下运行",
            "run_execution_app.py",
            "execution app 运行时",
            "run_execution_web.py",
            "使用教程",
            "Start-Job",
        ),
    }
    for relative_path, snippets in required_strings.items():
        text = _read_text(relative_path)
        for snippet in snippets:
            _require(
                snippet in text,
                failures,
                "brain_memory_not_synced",
                f"{relative_path} is missing current consistency marker: {snippet}",
            )


def run_checks() -> list[CheckResult]:
    failures: list[CheckResult] = []
    _check_train_entrypoints(failures)
    _check_config_dataclass_defaults(failures)
    _check_execution_profile_defaults(failures)
    _check_aux_defaults(failures)
    _check_execution_semantics(failures)
    _check_no_hardcoded_operational_roots(failures)
    _check_execution_pipeline_consistency(failures)
    _check_environment_source_of_truth(failures)
    _check_project_python_runtime_contract(failures)
    _check_execution_application_contract(failures)
    _check_memory_sync(failures)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Check project-wide consistency rules that must not regress.")
    parser.parse_args()
    failures = run_checks()
    payload = {
        "status": "ok" if not failures else "failed",
        "failure_count": len(failures),
        "failures": [failure.__dict__ for failure in failures],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
