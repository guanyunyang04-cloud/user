from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.continuous_policy.model import load_artifact, predict_policy
from daily_research.baseline.data_provider import get_latest_completed_trading_date
from daily_research.continuous_policy.pipeline_utils import build_reason_summary, translate_target_weights_to_share_actions
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
    EXPORTS_ROOT,
    now_iso,
    resolve_latest_model_artifact,
    save_runtime_state,
    timestamp_tag,
    update_latest_summary,
    write_json,
    load_runtime_state,
)
from daily_research.continuous_policy.state_builder import build_cross_section_state, build_daily_state_features, prepare_policy_inputs, resolve_active_policy_defaults
from daily_research.execution import app_service
from daily_research.execution.strategy_manifest import load_strategy_manifest


def build_parser() -> argparse.ArgumentParser:
    defaults = resolve_active_policy_defaults()
    parser = argparse.ArgumentParser(description="Export the latest continuous-policy shadow action panel.")
    parser.add_argument("--model-path", default="")
    parser.add_argument(
        "--pool-name",
        default=defaults["pool_name"] or "liquid500",
        help="Rolling liquidity pool name, or `all_a` / `learned_all_a` to export whole-A learned selection actions.",
    )
    parser.add_argument("--signal-date", default=get_latest_completed_trading_date())
    parser.add_argument("--benchmark", default=defaults["benchmark"] or "000300.SH")
    parser.add_argument("--data-source", default="tq", choices=("tq", "csv"))
    parser.add_argument("--csv-folder", default="")
    parser.add_argument("--pool-rebalance-days", type=int, default=21)
    parser.add_argument("--pool-adv-window", type=int, default=20)
    parser.add_argument("--max-universe-size", type=int, default=0)
    parser.add_argument("--transaction-cost-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=7.0)
    parser.add_argument("--sell-tax-bps", type=float, default=10.0)
    parser.add_argument(
        "--execution-semantics",
        default=DEFAULT_EXECUTION_SEMANTICS,
        choices=EXECUTION_SEMANTICS_CHOICES,
        help="Use semantic_preserving_v1 to keep lifecycle intent separate from weight-change orders.",
    )
    parser.add_argument(
        "--budget-semantics",
        default=DEFAULT_BUDGET_SEMANTICS,
        choices=BUDGET_SEMANTICS_CHOICES,
        help="How candidate budget is applied during export.",
    )
    parser.add_argument(
        "--budget-calibration",
        default=DEFAULT_BUDGET_CALIBRATION,
        choices=BUDGET_CALIBRATION_CHOICES,
        help="Optional portfolio-level gross/candidate/turnover calibration.",
    )
    parser.add_argument("--force-bootstrap-from-account", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--tag", default="")
    return parser


def _portfolio_state_from_runtime_or_account(
    *,
    runtime_state: dict[str, object],
    account_snapshot: dict[str, object],
    price_row: pd.Series,
    force_bootstrap_from_account: bool,
) -> PortfolioState:
    if runtime_state and (not force_bootstrap_from_account):
        try:
            return PortfolioState.from_snapshot(runtime_state)
        except Exception:
            pass
    return PortfolioState.from_account_snapshot(account_snapshot=account_snapshot, latest_prices=price_row)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact_path = resolve_latest_model_artifact(args.model_path)
    artifact = load_artifact(artifact_path)
    account_snapshot = app_service.load_account_snapshot()
    holdings = [str(item.get("stock", "")).strip().upper() for item in account_snapshot.get("positions", [])]
    runtime_state = load_runtime_state()
    signal_dt = pd.Timestamp(args.signal_date).normalize()
    runtime_last_signal_date = str(runtime_state.get("last_signal_date", "") or "").strip()
    runtime_is_usable = False
    if runtime_last_signal_date:
        try:
            runtime_is_usable = pd.Timestamp(runtime_last_signal_date).normalize() <= signal_dt
        except Exception:
            runtime_is_usable = False
    last_signal_date = runtime_last_signal_date if runtime_is_usable else ""
    requested_start_date = last_signal_date or str(args.signal_date)
    prepared = prepare_policy_inputs(
        pool_name=args.pool_name,
        start_date=requested_start_date,
        end_date=args.signal_date,
        benchmark=args.benchmark,
        data_source=args.data_source,
        csv_folder=args.csv_folder,
        extra_stocks=holdings,
        max_universe_size=args.max_universe_size,
        pool_rebalance_days=args.pool_rebalance_days,
        pool_adv_window=args.pool_adv_window,
        refresh_cache=args.refresh_cache,
        progress_desc="continuous policy export",
    )
    if signal_dt not in prepared.close.index:
        raise KeyError(f"Signal date is not available in prepared close frame: {signal_dt:%Y-%m-%d}")

    process_dates = [dt for dt in prepared.close.index if dt <= signal_dt and (not last_signal_date or dt > pd.Timestamp(last_signal_date))]
    if not process_dates:
        process_dates = [signal_dt]
    portfolio = _portfolio_state_from_runtime_or_account(
        runtime_state=runtime_state if runtime_is_usable else {},
        account_snapshot=account_snapshot,
        price_row=prepared.close.loc[process_dates[0]],
        force_bootstrap_from_account=bool(args.force_bootstrap_from_account or (not runtime_is_usable)),
    )

    final_state_frame = pd.DataFrame()
    final_policy_frame = pd.DataFrame()
    final_global_targets: dict[str, float] = {}
    final_step_result = None
    for idx, current_dt in enumerate(process_dates):
        state_frame = build_cross_section_state(prepared, date=current_dt, portfolio_state=portfolio)
        daily_features = build_daily_state_features(state_frame)
        policy_frame, global_targets = predict_policy(
            artifact,
            state_frame=state_frame,
            daily_features=daily_features,
        )
        step_result = portfolio.step(
            date=current_dt,
            prices=prepared.close.loc[current_dt],
            policy_frame=policy_frame,
            global_targets=global_targets,
            source_label="continuous_policy",
            execution_semantics=args.execution_semantics,
            budget_semantics=args.budget_semantics,
            budget_calibration=args.budget_calibration,
        )
        if idx + 1 < len(process_dates):
            next_dt = process_dates[idx + 1]
            next_returns = prepared.close.loc[next_dt].div(prepared.close.loc[current_dt]).sub(1.0).fillna(0.0)
            gross_return = float(step_result.weights.reindex(next_returns.index).fillna(0.0).mul(next_returns).sum())
            trading_cost = (
                float(step_result.diagnostics.get("buy_turnover", 0.0))
                * (float(args.transaction_cost_bps) + float(args.slippage_bps))
                / 10_000.0
                + float(step_result.diagnostics.get("sell_turnover", 0.0))
                * (float(args.transaction_cost_bps) + float(args.slippage_bps) + float(args.sell_tax_bps))
                / 10_000.0
            )
            portfolio.record_realized_return(gross_return - trading_cost)
        final_state_frame = state_frame
        final_policy_frame = policy_frame
        final_global_targets = global_targets
        final_step_result = step_result

    if final_step_result is None:
        raise RuntimeError("Continuous-policy export did not produce a final step result.")

    share_actions = translate_target_weights_to_share_actions(
        account_snapshot=account_snapshot,
        target_weights=final_step_result.weights,
        price_row=prepared.close.loc[signal_dt],
    )
    action_panel = pd.DataFrame(final_step_result.actions)
    if action_panel.empty:
        action_panel = pd.DataFrame(columns=["stock"])
    merged = share_actions.merge(action_panel, how="left", on="stock", suffixes=("", "_model"))
    merged["signal_date"] = signal_dt.strftime("%Y-%m-%d")
    merged["reason_summary"] = merged.apply(build_reason_summary, axis=1)
    merged = merged.sort_values(["share_action", "target_weight", "stock"], ascending=[True, False, True]).reset_index(drop=True)

    latest_model_artifact = str(artifact_path.resolve())
    export_tag = str(args.tag or signal_dt.strftime("%Y%m%d"))
    export_root = EXPORTS_ROOT / export_tag
    export_root.mkdir(parents=True, exist_ok=True)
    action_panel_path = export_root / "daily_live_action_panel.csv"
    reasoning_path = export_root / "daily_execution_reasoning.json"
    runtime_snapshot_path = export_root / "portfolio_state.json"
    merged.to_csv(action_panel_path, index=False, encoding="utf-8-sig")

    current_prices = prepared.close.loc[signal_dt]
    account_runtime = PortfolioState.from_account_snapshot(account_snapshot=account_snapshot, latest_prices=current_prices)
    runtime_alignment_gap = float(
        sum(
            abs(final_step_result.weights.get(stock, 0.0) - account_runtime.weight_map().get(stock, 0.0))
            for stock in set(final_step_result.weights.index.astype(str)) | set(account_runtime.weight_map())
        )
    )
    runtime_snapshot = portfolio.snapshot()
    runtime_snapshot.update(
        {
            "latest_model_artifact": latest_model_artifact,
            "latest_export_dir": str(export_root.resolve()),
            "runtime_alignment_gap": runtime_alignment_gap,
        }
    )
    save_runtime_state(runtime_snapshot)
    write_json(runtime_snapshot_path, runtime_snapshot)

    reasoning_payload = {
        "signal_date": signal_dt.strftime("%Y-%m-%d"),
        "generated_at": now_iso(),
        "model_artifact_path": latest_model_artifact,
        "trainer_backend": str(
            artifact.train_summary.get("trainer_backend", "")
            or getattr(artifact, "training_contract", {}).get("trainer_backend", "")
            or ""
        ),
        "training_contract": dict(artifact.train_summary.get("training_contract", {}) or getattr(artifact, "training_contract", {}) or {}),
        "training_diagnostics": dict(artifact.train_summary.get("training_diagnostics", {}) or getattr(artifact, "training_diagnostics", {}) or {}),
        "label_preset": str(
            artifact.train_summary.get("label_preset")
            or artifact.train_summary.get("teacher_summary", {}).get("label_preset")
            or "balanced_v2"
        ),
        "global_targets": final_global_targets,
        "execution_semantics": str(args.execution_semantics),
        "budget_semantics": str(args.budget_semantics),
        "budget_calibration": str(args.budget_calibration),
        "portfolio_runtime_snapshot": runtime_snapshot,
        "current_account_path": account_snapshot.get("path", ""),
        "action_panel_csv": str(action_panel_path.resolve()),
        "rows": merged.to_dict(orient="records"),
    }
    write_json(reasoning_path, reasoning_payload)

    manifest = load_strategy_manifest()
    summary_payload = {
        "signal_date": signal_dt.strftime("%Y-%m-%d"),
        "generated_at": now_iso(),
        "model_artifact_path": latest_model_artifact,
        "trainer_backend": str(
            artifact.train_summary.get("trainer_backend", "")
            or getattr(artifact, "training_contract", {}).get("trainer_backend", "")
            or ""
        ),
        "training_contract": dict(artifact.train_summary.get("training_contract", {}) or getattr(artifact, "training_contract", {}) or {}),
        "training_diagnostics": dict(artifact.train_summary.get("training_diagnostics", {}) or getattr(artifact, "training_diagnostics", {}) or {}),
        "label_preset": str(
            artifact.train_summary.get("label_preset")
            or artifact.train_summary.get("teacher_summary", {}).get("label_preset")
            or "balanced_v2"
        ),
        "execution_semantics": str(args.execution_semantics),
        "budget_semantics": str(args.budget_semantics),
        "budget_calibration": str(args.budget_calibration),
        "pool_name": prepared.pool_name,
        "benchmark": prepared.benchmark,
        "export_dir": str(export_root.resolve()),
        "action_panel_csv": str(action_panel_path.resolve()),
        "reasoning_json": str(reasoning_path.resolve()),
        "runtime_state_json": str(runtime_snapshot_path.resolve()),
        "action_counts": {
            str(key): int(value)
            for key, value in merged["share_action"].astype(str).value_counts().sort_index().items()
        },
        "runtime_alignment_gap": runtime_alignment_gap,
        "manifest_reference": {
            "strategy_name": str(manifest.get("strategy_name", "") or ""),
            "candidate_label": str(manifest.get("candidate_label", "") or ""),
        },
    }
    update_latest_summary("export", summary_payload)
    write_json(export_root / "export_summary.json", summary_payload)
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
