from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from daily_research.baseline.data_provider import get_latest_completed_trading_date
from daily_research.continuous_policy.analyze_behavior_gap import main as behavior_audit_main
from daily_research.continuous_policy.conclusion_ledger import main as conclusion_ledger_main
from daily_research.continuous_policy.label_builder import LABEL_CONFIGS
from daily_research.continuous_policy.evaluate_policy import main as evaluate_main
from daily_research.continuous_policy.export_action_panel import main as export_main
from daily_research.continuous_policy.model import load_artifact
from daily_research.continuous_policy.model_seq_v3 import (
    DAILY_HEAD_LAYOUT_CHOICES,
    DAILY_HEAD_LAYOUT_MONOLITHIC_V1,
    DEFAULT_LOSS_PROFILE,
    LOSS_PROFILE_NAMES,
)
from daily_research.continuous_policy.model_v2 import DECODER_PROFILE_NAMES
from daily_research.continuous_policy.pipeline_utils import (
    BUDGET_OBJECTIVE_CHOICES,
    DEFAULT_BUDGET_OBJECTIVE,
    run_policy_rollout,
)
from daily_research.continuous_policy.portfolio_simulator import (
    BUDGET_CALIBRATION_CHOICES,
    BUDGET_SEMANTICS_CHOICES,
    DEFAULT_BUDGET_CALIBRATION,
    DEFAULT_BUDGET_SEMANTICS,
    DEFAULT_EXECUTION_SEMANTICS,
    EXECUTION_SEMANTICS_CHOICES,
    PortfolioState,
)
from daily_research.continuous_policy.runtime import (
    EVALUATIONS_ROOT,
    EXPORTS_ROOT,
    MODELS_ROOT,
    PROTOCOLS_ROOT,
    now_iso,
    safe_print_json,
    timestamp_tag,
    update_latest_summary,
    write_json,
)
from daily_research.continuous_policy.state_builder import DEFAULT_ALPHA_PRIOR_SOURCE, prepare_policy_inputs, resolve_active_policy_defaults
from daily_research.continuous_policy.train_policy import main as train_main
from daily_research.continuous_policy.training_contracts import TRAINER_BACKENDS, TRAINER_BACKEND_FORMAL_V2
from daily_research.execution import app_service


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = WORKSPACE_ROOT / "daily_research" / "output"


PROMOTION_THRESHOLDS = {
    "open_win_rate_5d": 0.45,
    "reduce_success_rate_5d": 0.52,
    "exit_timeliness_rate_5d": 0.52,
    "cash_timing_quality_1d": 0.08,
    "hold_share": 0.15,
    "avg_turnover_max": 0.25,
    "max_drawdown_min": -0.06,
}

TRAINING_EVIDENCE_THRESHOLDS = {
    "min_train_day_count": 180,
    "min_teacher_action_rows": 10000,
    "min_adaptive_teacher_action_rows": 1200,
    "teacher_action_rows_per_train_day": 6.0,
    "best_epoch_edge_margin": 2,
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _call_stage(label: str, fn: Any, argv: list[str]) -> None:
    code = int(fn(argv))
    if code != 0:
        raise RuntimeError(f"continuous_policy {label} stage failed with exit code {code}.")


def _discover_policy_v5b_reference_panel() -> Path | None:
    search_patterns = (
        "short_alpha_recent_model_protocol_*__short_expert_policy_v5b/execution_aligned_daily_live_target_weight_panel.csv",
        "short_alpha_policy_v5_family_formal_review_*/runs/short_expert_policy_v5b/execution_aligned_daily_live_target_weight_panel.csv",
        "short_alpha_recent_model_protocol_*__short_expert_policy_v5b/daily_live_target_weight_panel.csv",
        "short_alpha_policy_v5_family_formal_review_*/runs/short_expert_policy_v5b/daily_live_target_weight_panel.csv",
    )
    candidates: list[Path] = []
    for pattern in search_patterns:
        candidates.extend((OUTPUT_ROOT).glob(pattern))
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0].resolve()


def _default_shadow_start(eval_start_date: str, eval_end_date: str) -> str:
    start_ts = pd.Timestamp(eval_start_date).normalize()
    end_ts = pd.Timestamp(eval_end_date).normalize()
    recent_start = end_ts - pd.Timedelta(days=14)
    return max(start_ts, recent_start).strftime("%Y%m%d")


def _build_promotion_gate(
    *,
    train_summary: dict[str, Any],
    evaluation_summary: dict[str, Any],
    shadow_summary: dict[str, Any],
    training_evidence: dict[str, Any],
) -> dict[str, Any]:
    training_contract = dict(
        train_summary.get("training_contract", {})
        or evaluation_summary.get("training_contract", {})
        or {}
    )
    metrics = dict(evaluation_summary.get("continuous_policy_metrics", {}) or {})
    continuity = dict(evaluation_summary.get("continuity_metrics", {}) or {})
    reference_panels = evaluation_summary.get("reference_panels", []) or []
    active_reference = dict(evaluation_summary.get("active_manifest_reference", {}) or {})
    if active_reference:
        reference_panels = [active_reference, *reference_panels]
    active_metrics = {}
    for item in reference_panels:
        if str((item or {}).get("label", "") or "") == "active_manifest_reference":
            active_metrics = dict((item or {}).get("metrics", {}) or {})
            break
    checks = {
        "contract_promotable": bool(training_contract.get("promotable", False)),
        "training_evidence_sufficient": str(training_evidence.get("status", "") or "") == "sufficient",
        "open_win_rate_5d": float(continuity.get("open_win_rate_5d", 0.0) or 0.0) >= PROMOTION_THRESHOLDS["open_win_rate_5d"],
        "reduce_success_rate_5d": float(continuity.get("reduce_success_rate_5d", 0.0) or 0.0) >= PROMOTION_THRESHOLDS["reduce_success_rate_5d"],
        "exit_timeliness_rate_5d": float(continuity.get("exit_timeliness_rate_5d", 0.0) or 0.0) >= PROMOTION_THRESHOLDS["exit_timeliness_rate_5d"],
        "cash_timing_quality_1d": float(continuity.get("cash_timing_quality_1d", 0.0) or 0.0) >= PROMOTION_THRESHOLDS["cash_timing_quality_1d"],
        "hold_share": float(continuity.get("hold_share", 0.0) or 0.0) >= PROMOTION_THRESHOLDS["hold_share"],
        "avg_turnover": float(metrics.get("avg_turnover", 0.0) or 0.0) <= PROMOTION_THRESHOLDS["avg_turnover_max"],
        "max_drawdown": float(metrics.get("max_drawdown", 0.0) or 0.0) >= PROMOTION_THRESHOLDS["max_drawdown_min"],
        "shadow_reversal": float(shadow_summary.get("continuity_metrics", {}).get("immediate_reversal_rate_3d", 0.0) or 0.0) <= 0.18,
    }
    if active_metrics:
        checks["annual_return_vs_active"] = float(metrics.get("annual_return", 0.0) or 0.0) >= float(active_metrics.get("annual_return", 0.0) or 0.0)
        checks["sharpe_vs_active"] = float(metrics.get("sharpe", 0.0) or 0.0) >= float(active_metrics.get("sharpe", 0.0) or 0.0)
    failed_checks = [name for name, ok in checks.items() if not ok]
    return {
        "status": "eligible" if not failed_checks else "shadow_only",
        "failed_checks": failed_checks,
        "checks": checks,
    }


def _build_training_evidence_assessment(train_summary: dict[str, Any]) -> dict[str, Any]:
    diagnostics = dict(train_summary.get("training_diagnostics", {}) or {})
    teacher_summary = dict(train_summary.get("teacher_summary", {}) or {})
    completed_epochs = int(diagnostics.get("completed_epochs", 0) or 0)
    best_epoch = int(diagnostics.get("best_epoch", 0) or 0)
    train_day_count = int(
        diagnostics.get("train_day_count", train_summary.get("daily_rows", 0)) or 0
    )
    validation_day_count = int(diagnostics.get("validation_day_count", 0) or 0)
    train_sample_rows = int(
        diagnostics.get("train_sample_rows", teacher_summary.get("train_sample_rows", train_summary.get("sample_rows", 0)))
        or 0
    )
    teacher_action_rows = int(teacher_summary.get("action_rows", 0) or 0)
    raw_min_teacher_action_rows = int(TRAINING_EVIDENCE_THRESHOLDS["min_teacher_action_rows"])
    adaptive_min_teacher_action_rows = int(
        max(
            float(TRAINING_EVIDENCE_THRESHOLDS["min_adaptive_teacher_action_rows"]),
            train_day_count * float(TRAINING_EVIDENCE_THRESHOLDS["teacher_action_rows_per_train_day"]),
        )
    )
    # Daily execution teachers are capped by active lifecycle slots; a fixed 10k
    # action-row gate can be unreachable even when the day/sample coverage is real.
    effective_min_teacher_action_rows = min(raw_min_teacher_action_rows, adaptive_min_teacher_action_rows)
    edge_margin = int(TRAINING_EVIDENCE_THRESHOLDS["best_epoch_edge_margin"])
    best_epoch_not_at_edge = completed_epochs > 0 and best_epoch <= max(completed_epochs - edge_margin, 0)
    checks = {
        "train_day_count": train_day_count >= int(TRAINING_EVIDENCE_THRESHOLDS["min_train_day_count"]),
        "teacher_action_rows": teacher_action_rows >= effective_min_teacher_action_rows,
        "best_epoch_not_at_edge": bool(best_epoch_not_at_edge),
    }
    failed_checks = [name for name, ok in checks.items() if not ok]
    status = "sufficient" if not failed_checks else "insufficient"
    recommended_actions: list[str] = []
    if not checks["best_epoch_not_at_edge"] and completed_epochs > 0:
        recommended_actions.append(
            f"best_epoch={best_epoch} 仍贴近 completed_epochs={completed_epochs} 边缘；先沿同一 run_dir 做 strict resume，把预算至少加到 {completed_epochs + 16} epoch。"
        )
    if not checks["train_day_count"]:
        recommended_actions.append(
            f"当前 train_day_count={train_day_count} 低于 {TRAINING_EVIDENCE_THRESHOLDS['min_train_day_count']}；下一轮应显式向前扩训练窗口。"
        )
    if not checks["teacher_action_rows"]:
        recommended_actions.append(
            f"当前 teacher_action_rows={teacher_action_rows} 低于有效下限 {effective_min_teacher_action_rows}；原始参考阈值为 {raw_min_teacher_action_rows}，有效下限已按日频持仓容量自适应。"
        )
    if not recommended_actions:
        recommended_actions.append("当前训练预算与样本覆盖已达到本轮正式证据下限。")
    return {
        "status": status,
        "failed_checks": failed_checks,
        "checks": checks,
        "thresholds": {
            **dict(TRAINING_EVIDENCE_THRESHOLDS),
            "raw_min_teacher_action_rows": raw_min_teacher_action_rows,
            "adaptive_min_teacher_action_rows": adaptive_min_teacher_action_rows,
            "effective_min_teacher_action_rows": effective_min_teacher_action_rows,
        },
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch,
        "train_day_count": train_day_count,
        "validation_day_count": validation_day_count,
        "train_sample_rows": train_sample_rows,
        "teacher_action_rows": teacher_action_rows,
        "recommended_actions": recommended_actions,
    }


def build_parser() -> argparse.ArgumentParser:
    defaults = resolve_active_policy_defaults()
    latest_completed = get_latest_completed_trading_date()
    parser = argparse.ArgumentParser(description="Run the continuous-policy protocol: train -> evaluate -> shadow continuity -> export.")
    parser.add_argument(
        "--pool-name",
        default=defaults["pool_name"] or "liquid500",
        help="Rolling liquidity pool name, or `all_a` / `learned_all_a` to run learned selection over the whole A-share universe.",
    )
    parser.add_argument("--benchmark", default=defaults["benchmark"] or "000300.SH")
    parser.add_argument("--data-source", default="tq", choices=("tq", "csv"))
    parser.add_argument("--csv-folder", default="")
    parser.add_argument("--pool-rebalance-days", type=int, default=21)
    parser.add_argument("--pool-adv-window", type=int, default=20)
    parser.add_argument("--max-universe-size", type=int, default=0)
    parser.add_argument("--train-start-date", default="20240102")
    parser.add_argument("--train-end-date", default="20251231")
    parser.add_argument("--eval-start-date", default="20260102")
    parser.add_argument("--eval-end-date", default=latest_completed)
    parser.add_argument("--shadow-start-date", default="")
    parser.add_argument("--shadow-end-date", default=latest_completed)
    parser.add_argument("--transaction-cost-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=7.0)
    parser.add_argument("--sell-tax-bps", type=float, default=10.0)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--skip-multiplier", type=float, default=2.0)
    parser.add_argument(
        "--execution-semantics",
        default=DEFAULT_EXECUTION_SEMANTICS,
        choices=EXECUTION_SEMANTICS_CHOICES,
        help="semantic_preserving_v1 keeps lifecycle intent separate from actual weight-change orders.",
    )
    parser.add_argument(
        "--budget-semantics",
        default=DEFAULT_BUDGET_SEMANTICS,
        choices=BUDGET_SEMANTICS_CHOICES,
        help="action_budget_split_v1 protects existing lifecycle actions and budgets new entries separately.",
    )
    parser.add_argument(
        "--budget-calibration",
        default=DEFAULT_BUDGET_CALIBRATION,
        choices=BUDGET_CALIBRATION_CHOICES,
        help="Optional portfolio-level gross/candidate/turnover calibration.",
    )
    parser.add_argument(
        "--budget-objective",
        default=DEFAULT_BUDGET_OBJECTIVE,
        choices=BUDGET_OBJECTIVE_CHOICES,
        help="Daily budget target objective propagated to training and teacher-oracle evaluation.",
    )
    parser.add_argument(
        "--alpha-prior-source",
        default=DEFAULT_ALPHA_PRIOR_SOURCE,
        help="Alpha prior source propagated to train/evaluate/shadow/export state builders.",
    )
    parser.add_argument("--alpha-prior-score-panel", default="")
    parser.add_argument("--alpha-prior-target-weight-panel", default="")
    parser.add_argument(
        "--label-preset",
        default="balanced_v2",
        choices=tuple(sorted(LABEL_CONFIGS)),
    )
    parser.add_argument(
        "--trainer-backend",
        default=TRAINER_BACKEND_FORMAL_V2,
        choices=TRAINER_BACKENDS,
    )
    parser.add_argument(
        "--decoder-profile",
        default="default_v2",
        choices=DECODER_PROFILE_NAMES,
    )
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--min-epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1.5e-3)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--sequence-layers", type=int, default=1)
    parser.add_argument("--daily-hidden-dim", type=int, default=96)
    parser.add_argument("--daily-head-layout", default=DAILY_HEAD_LAYOUT_MONOLITHIC_V1, choices=DAILY_HEAD_LAYOUT_CHOICES)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--daily-dropout", type=float, default=0.05)
    parser.add_argument("--loss-profile", default=DEFAULT_LOSS_PROFILE, choices=LOSS_PROFILE_NAMES)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--resume-mode", default="strict", choices=("strict", "fresh"))
    parser.add_argument("--force-bootstrap-from-account", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--tag", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol_tag = str(args.tag or timestamp_tag("protocol"))
    protocol_root = PROTOCOLS_ROOT / protocol_tag
    protocol_root.mkdir(parents=True, exist_ok=True)

    train_tag = f"{protocol_tag}__train"
    train_args = [
        "--pool-name",
        args.pool_name,
        "--start-date",
        args.train_start_date,
        "--end-date",
        args.train_end_date,
        "--benchmark",
        args.benchmark,
        "--data-source",
        args.data_source,
        "--pool-rebalance-days",
        str(args.pool_rebalance_days),
        "--pool-adv-window",
        str(args.pool_adv_window),
        "--max-universe-size",
        str(args.max_universe_size),
        "--random-seed",
        str(args.random_seed),
        "--skip-multiplier",
        str(args.skip_multiplier),
        "--execution-semantics",
        str(args.execution_semantics),
        "--budget-semantics",
        str(args.budget_semantics),
        "--budget-calibration",
        str(args.budget_calibration),
        "--budget-objective",
        str(args.budget_objective),
        "--alpha-prior-source",
        str(args.alpha_prior_source),
        "--alpha-prior-score-panel",
        str(args.alpha_prior_score_panel),
        "--alpha-prior-target-weight-panel",
        str(args.alpha_prior_target_weight_panel),
        "--label-preset",
        args.label_preset,
        "--trainer-backend",
        args.trainer_backend,
        "--decoder-profile",
        args.decoder_profile,
        "--loss-profile",
        args.loss_profile,
        "--transaction-cost-bps",
        str(args.transaction_cost_bps),
        "--slippage-bps",
        str(args.slippage_bps),
        "--sell-tax-bps",
        str(args.sell_tax_bps),
        "--epochs",
        str(args.epochs),
        "--min-epochs",
        str(args.min_epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--hidden-dim",
        str(args.hidden_dim),
        "--sequence-layers",
        str(args.sequence_layers),
        "--daily-hidden-dim",
        str(args.daily_hidden_dim),
        "--daily-head-layout",
        str(args.daily_head_layout),
        "--dropout",
        str(args.dropout),
        "--daily-dropout",
        str(args.daily_dropout),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--resume-mode",
        args.resume_mode,
        "--tag",
        train_tag,
    ]
    if str(args.csv_folder or "").strip():
        train_args.extend(["--csv-folder", str(args.csv_folder)])
    if args.refresh_cache:
        train_args.append("--refresh-cache")
    _call_stage("train", train_main, train_args)
    train_summary = _read_json(MODELS_ROOT / train_tag / "train_summary.json")
    artifact_path = Path(str(train_summary.get("model_artifact_path", "") or "")).expanduser().resolve()
    if not artifact_path.exists():
        raise FileNotFoundError(f"continuous_policy protocol did not produce a model artifact: {artifact_path}")

    eval_tag = f"{protocol_tag}__evaluate"
    eval_args = [
        "--model-path",
        str(artifact_path),
        "--pool-name",
        args.pool_name,
        "--start-date",
        args.eval_start_date,
        "--end-date",
        args.eval_end_date,
        "--benchmark",
        args.benchmark,
        "--data-source",
        args.data_source,
        "--pool-rebalance-days",
        str(args.pool_rebalance_days),
        "--pool-adv-window",
        str(args.pool_adv_window),
        "--max-universe-size",
        str(args.max_universe_size),
        "--transaction-cost-bps",
        str(args.transaction_cost_bps),
        "--slippage-bps",
        str(args.slippage_bps),
        "--sell-tax-bps",
        str(args.sell_tax_bps),
        "--execution-semantics",
        str(args.execution_semantics),
        "--budget-semantics",
        str(args.budget_semantics),
        "--budget-calibration",
        str(args.budget_calibration),
        "--budget-objective",
        str(args.budget_objective),
        "--alpha-prior-source",
        str(args.alpha_prior_source),
        "--alpha-prior-score-panel",
        str(args.alpha_prior_score_panel),
        "--alpha-prior-target-weight-panel",
        str(args.alpha_prior_target_weight_panel),
        "--label-preset",
        args.label_preset,
        "--tag",
        eval_tag,
    ]
    if str(args.csv_folder or "").strip():
        eval_args.extend(["--csv-folder", str(args.csv_folder)])
    if args.refresh_cache:
        eval_args.append("--refresh-cache")
    discovered_references: list[dict[str, str]] = []
    v5b_panel = _discover_policy_v5b_reference_panel()
    if v5b_panel is not None:
        eval_args.extend(["--reference-panel", f"policy_v5b={v5b_panel}"])
        discovered_references.append({"label": "policy_v5b", "panel_path": str(v5b_panel)})
    _call_stage("evaluate", evaluate_main, eval_args)
    evaluation_summary = _read_json(EVALUATIONS_ROOT / eval_tag / "evaluation_summary.json")

    shadow_start_date = str(args.shadow_start_date or "").strip() or _default_shadow_start(args.eval_start_date, args.shadow_end_date)
    shadow_end_date = str(args.shadow_end_date or "").strip() or str(args.eval_end_date)
    account_snapshot = app_service.load_account_snapshot()
    holdings = [str(item.get("stock", "")).strip().upper() for item in account_snapshot.get("positions", [])]
    shadow_prepared = prepare_policy_inputs(
        pool_name=args.pool_name,
        start_date=shadow_start_date,
        end_date=shadow_end_date,
        benchmark=args.benchmark,
        data_source=args.data_source,
        csv_folder=args.csv_folder,
        extra_stocks=holdings,
        max_universe_size=args.max_universe_size,
        pool_rebalance_days=args.pool_rebalance_days,
        pool_adv_window=args.pool_adv_window,
        alpha_prior_source=args.alpha_prior_source,
        alpha_prior_score_panel=args.alpha_prior_score_panel,
        alpha_prior_target_weight_panel=args.alpha_prior_target_weight_panel,
        refresh_cache=args.refresh_cache,
        progress_desc="continuous policy protocol shadow",
    )
    shadow_dates = [dt for dt in shadow_prepared.close.index if dt >= pd.Timestamp(shadow_start_date) and dt <= pd.Timestamp(shadow_end_date)]
    if len(shadow_dates) < 2:
        raise ValueError("continuous_policy protocol shadow window must contain at least two signal dates.")
    initial_portfolio = PortfolioState.from_account_snapshot(
        account_snapshot=account_snapshot,
        latest_prices=shadow_prepared.close.loc[shadow_dates[0]],
    )
    shadow_rollout = run_policy_rollout(
        prepared=shadow_prepared,
        artifact=load_artifact(artifact_path),
        start_date=shadow_start_date,
        end_date=shadow_end_date,
        initial_portfolio=initial_portfolio,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        sell_tax_bps=args.sell_tax_bps,
        source_label="continuous_policy_shadow",
        execution_semantics=args.execution_semantics,
        budget_semantics=args.budget_semantics,
        budget_calibration=args.budget_calibration,
        budget_objective=args.budget_objective,
    )

    shadow_action_panel_path = protocol_root / "shadow_daily_action_panel.csv"
    shadow_action_outcomes_path = protocol_root / "shadow_daily_action_outcomes.csv"
    shadow_turnover_path = protocol_root / "shadow_daily_turnover.csv"
    shadow_position_history_path = protocol_root / "shadow_daily_position_history.csv"
    shadow_returns_path = protocol_root / "shadow_daily_returns.csv"
    shadow_monthly_returns_path = protocol_root / "shadow_monthly_returns.csv"
    shadow_summary_path = protocol_root / "shadow_window_summary.json"
    shadow_rollout["action_panel"].to_csv(shadow_action_panel_path, index=False, encoding="utf-8-sig")
    shadow_rollout["action_outcomes"].to_csv(shadow_action_outcomes_path, index=False, encoding="utf-8-sig")
    shadow_rollout["turnover_frame"].to_csv(shadow_turnover_path, index=False, encoding="utf-8-sig")
    shadow_rollout["position_history"].to_csv(shadow_position_history_path, index=False, encoding="utf-8-sig")
    shadow_rollout["returns"].rename("daily_return").to_csv(shadow_returns_path, encoding="utf-8-sig")
    shadow_rollout["monthly_returns"].to_csv(shadow_monthly_returns_path, index=False, encoding="utf-8-sig")
    shadow_summary = {
        "start_date": shadow_start_date,
        "end_date": shadow_end_date,
        "signal_date_count": len(shadow_rollout["dates"]),
        "metrics": shadow_rollout["metrics"],
        "continuity_metrics": shadow_rollout["continuity_metrics"],
        "execution_semantics": str(args.execution_semantics),
        "budget_semantics": str(args.budget_semantics),
        "budget_calibration": str(args.budget_calibration),
        "budget_objective": str(args.budget_objective),
        "daily_head_layout": str(args.daily_head_layout),
        "alpha_prior_source": str(args.alpha_prior_source),
        "alpha_prior_score_panel": str(args.alpha_prior_score_panel),
        "alpha_prior_target_weight_panel": str(args.alpha_prior_target_weight_panel),
        "action_panel_csv": str(shadow_action_panel_path.resolve()),
        "action_outcomes_csv": str(shadow_action_outcomes_path.resolve()),
        "turnover_csv": str(shadow_turnover_path.resolve()),
        "position_history_csv": str(shadow_position_history_path.resolve()),
        "returns_csv": str(shadow_returns_path.resolve()),
        "monthly_returns_csv": str(shadow_monthly_returns_path.resolve()),
    }
    write_json(shadow_summary_path, shadow_summary)

    export_tag = f"{protocol_tag}__export"
    export_args = [
        "--model-path",
        str(artifact_path),
        "--pool-name",
        args.pool_name,
        "--signal-date",
        shadow_end_date,
        "--benchmark",
        args.benchmark,
        "--data-source",
        args.data_source,
        "--pool-rebalance-days",
        str(args.pool_rebalance_days),
        "--pool-adv-window",
        str(args.pool_adv_window),
        "--max-universe-size",
        str(args.max_universe_size),
        "--transaction-cost-bps",
        str(args.transaction_cost_bps),
        "--slippage-bps",
        str(args.slippage_bps),
        "--sell-tax-bps",
        str(args.sell_tax_bps),
        "--execution-semantics",
        str(args.execution_semantics),
        "--budget-semantics",
        str(args.budget_semantics),
        "--budget-calibration",
        str(args.budget_calibration),
        "--alpha-prior-source",
        str(args.alpha_prior_source),
        "--alpha-prior-score-panel",
        str(args.alpha_prior_score_panel),
        "--alpha-prior-target-weight-panel",
        str(args.alpha_prior_target_weight_panel),
        "--tag",
        export_tag,
    ]
    if str(args.csv_folder or "").strip():
        export_args.extend(["--csv-folder", str(args.csv_folder)])
    if args.refresh_cache:
        export_args.append("--refresh-cache")
    if args.force_bootstrap_from_account:
        export_args.append("--force-bootstrap-from-account")
    _call_stage("export", export_main, export_args)
    export_summary = _read_json(EXPORTS_ROOT / export_tag / "export_summary.json")

    summary_payload = {
        "run_tag": protocol_tag,
        "executed_at": now_iso(),
        "pool_name": args.pool_name,
        "benchmark": args.benchmark,
        "label_preset": args.label_preset,
        "trainer_backend": str(train_summary.get("trainer_backend", args.trainer_backend) or args.trainer_backend),
        "decoder_profile": str(train_summary.get("decoder_profile", args.decoder_profile) or args.decoder_profile),
        "loss_profile": str(train_summary.get("loss_profile", args.loss_profile) or args.loss_profile),
        "daily_head_layout": str(train_summary.get("daily_head_layout", args.daily_head_layout) or args.daily_head_layout),
        "execution_semantics": str(args.execution_semantics),
        "budget_semantics": str(args.budget_semantics),
        "budget_calibration": str(args.budget_calibration),
        "budget_objective": str(args.budget_objective),
        "alpha_prior_source": str(args.alpha_prior_source),
        "alpha_prior_score_panel": str(args.alpha_prior_score_panel),
        "alpha_prior_target_weight_panel": str(args.alpha_prior_target_weight_panel),
        "training_contract": dict(train_summary.get("training_contract", {}) or {}),
        "model_artifact_path": str(artifact_path),
        "protocol_summary_json": str((protocol_root / "protocol_summary.json").resolve()),
        "train_window": {"start_date": args.train_start_date, "end_date": args.train_end_date, "run_tag": train_tag},
        "evaluation_window": {"start_date": args.eval_start_date, "end_date": args.eval_end_date, "run_tag": eval_tag},
        "shadow_window": {"start_date": shadow_start_date, "end_date": shadow_end_date},
        "discovered_reference_panels": discovered_references,
        "train": {
            "run_tag": train_summary.get("run_tag", train_tag),
            "label_preset": train_summary.get("label_preset", args.label_preset),
            "trainer_backend": train_summary.get("trainer_backend", args.trainer_backend),
            "decoder_profile": train_summary.get("decoder_profile", args.decoder_profile),
            "loss_profile": train_summary.get("loss_profile", args.loss_profile),
            "daily_head_layout": train_summary.get("daily_head_layout", args.daily_head_layout),
            "execution_semantics": train_summary.get("execution_semantics", args.execution_semantics),
            "budget_semantics": train_summary.get("budget_semantics", args.budget_semantics),
            "budget_calibration": train_summary.get("budget_calibration", args.budget_calibration),
            "budget_objective": train_summary.get("budget_objective", args.budget_objective),
            "alpha_prior_source": train_summary.get("alpha_prior_source", args.alpha_prior_source),
            "training_contract": train_summary.get("training_contract", {}),
            "training_diagnostics": train_summary.get("training_diagnostics", {}),
            "teacher_summary": train_summary.get("teacher_summary", {}),
            "sample_rows": train_summary.get("sample_rows"),
            "feature_count": train_summary.get("feature_count"),
            "daily_feature_count": train_summary.get("daily_feature_count"),
            "model_artifact_path": str(artifact_path),
        },
        "evaluation": {
            "run_tag": evaluation_summary.get("run_tag", eval_tag),
            "label_preset": evaluation_summary.get("label_preset", args.label_preset),
            "trainer_backend": evaluation_summary.get("trainer_backend", train_summary.get("trainer_backend", args.trainer_backend)),
            "training_contract": evaluation_summary.get("training_contract", train_summary.get("training_contract", {})),
            "training_diagnostics": evaluation_summary.get("training_diagnostics", train_summary.get("training_diagnostics", {})),
            "execution_semantics": evaluation_summary.get("execution_semantics", args.execution_semantics),
            "budget_semantics": evaluation_summary.get("budget_semantics", args.budget_semantics),
            "budget_calibration": evaluation_summary.get("budget_calibration", args.budget_calibration),
            "budget_objective": evaluation_summary.get("budget_objective", args.budget_objective),
            "daily_head_layout": train_summary.get("daily_head_layout", args.daily_head_layout),
            "alpha_prior_source": evaluation_summary.get("alpha_prior_source", args.alpha_prior_source),
            "continuous_policy_metrics": evaluation_summary.get("continuous_policy_metrics", {}),
            "continuity_metrics": evaluation_summary.get("continuity_metrics", {}),
            "teacher_oracle_metrics": evaluation_summary.get("teacher_oracle_metrics", {}),
            "reference_panels": evaluation_summary.get("reference_panels", []),
            "evaluation_summary_json": str((EVALUATIONS_ROOT / eval_tag / "evaluation_summary.json").resolve()),
        },
        "shadow": {
            "metrics": shadow_summary.get("metrics", {}),
            "continuity_metrics": shadow_summary.get("continuity_metrics", {}),
            "execution_semantics": shadow_summary.get("execution_semantics", args.execution_semantics),
            "budget_semantics": shadow_summary.get("budget_semantics", args.budget_semantics),
            "budget_calibration": shadow_summary.get("budget_calibration", args.budget_calibration),
            "budget_objective": shadow_summary.get("budget_objective", args.budget_objective),
            "daily_head_layout": train_summary.get("daily_head_layout", args.daily_head_layout),
            "alpha_prior_source": shadow_summary.get("alpha_prior_source", args.alpha_prior_source),
            "shadow_summary_json": str(shadow_summary_path.resolve()),
        },
        "latest_export": {
            "signal_date": export_summary.get("signal_date", shadow_end_date),
            "trainer_backend": export_summary.get("trainer_backend", train_summary.get("trainer_backend", args.trainer_backend)),
            "action_counts": export_summary.get("action_counts", {}),
            "runtime_alignment_gap": export_summary.get("runtime_alignment_gap"),
            "execution_semantics": export_summary.get("execution_semantics", args.execution_semantics),
            "budget_semantics": export_summary.get("budget_semantics", args.budget_semantics),
            "budget_calibration": export_summary.get("budget_calibration", args.budget_calibration),
            "budget_objective": export_summary.get("budget_objective", args.budget_objective),
            "daily_head_layout": train_summary.get("daily_head_layout", args.daily_head_layout),
            "alpha_prior_source": export_summary.get("alpha_prior_source", args.alpha_prior_source),
            "export_summary_json": str((EXPORTS_ROOT / export_tag / "export_summary.json").resolve()),
        },
    }
    summary_payload["training_evidence"] = _build_training_evidence_assessment(train_summary)
    summary_payload["promotion_gate"] = _build_promotion_gate(
        train_summary=train_summary,
        evaluation_summary=evaluation_summary,
        shadow_summary=shadow_summary,
        training_evidence=summary_payload["training_evidence"],
    )
    protocol_summary_path = protocol_root / "protocol_summary.json"
    write_json(protocol_summary_path, summary_payload)
    audit_tag = f"{protocol_tag}__audit"
    _call_stage(
        "behavior-audit",
        behavior_audit_main,
        [
            "--evaluation-summary",
            str((EVALUATIONS_ROOT / eval_tag / "evaluation_summary.json").resolve()),
            "--tag",
            audit_tag,
        ],
    )
    latest_behavior_audit = _read_json(
        OUTPUT_ROOT / "continuous_policy" / "analysis" / "behavior_audits" / f"{audit_tag}.json"
    )
    summary_payload["latest_behavior_audit"] = latest_behavior_audit
    write_json(protocol_summary_path, summary_payload)
    ledger_tag = f"{protocol_tag}__ledger"
    _call_stage(
        "conclusion-ledger",
        conclusion_ledger_main,
        [
            "--protocol-summary",
            str(protocol_summary_path.resolve()),
            "--tag",
            ledger_tag,
        ],
    )
    summary_payload["latest_conclusion_ledger"] = _read_json(
        OUTPUT_ROOT / "continuous_policy" / "analysis" / "conclusion_ledgers" / f"{ledger_tag}.json"
    )
    write_json(protocol_summary_path, summary_payload)
    update_latest_summary("protocol", summary_payload)
    safe_print_json(summary_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
