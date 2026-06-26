from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.continuous_policy.label_builder import build_future_path_metrics
from daily_research.continuous_policy.pipeline_utils import BUDGET_OBJECTIVE_CHOICES, build_training_matrices
from daily_research.continuous_policy.run_self_optimizing_study import _score_protocol_summary
from daily_research.continuous_policy.runtime import timestamp_tag
from daily_research.continuous_policy.state_builder import prepare_policy_inputs
from quant_data_platform.lake import DEFAULT_POLICY_INPUT_LAKE_DATASET_ID


TARGET_COLUMNS: tuple[str, ...] = (
    "budget_risk_signal_target",
    "budget_deploy_signal_target",
    "budget_cash_timing_signal_target",
    "budget_alpha_focus_signal_target",
    "gross_exposure_target",
    "candidate_budget",
    "turnover_budget",
    "max_position_weight_target",
    "hold_bias_target",
    "reduce_bias_target",
    "exit_patience_target",
    "reentry_guard_target",
)


def _diff_map(after: dict[str, Any], before: dict[str, Any]) -> dict[str, float]:
    diff: dict[str, float] = {}
    for key in sorted(set(before) | set(after)):
        if key not in before or key not in after:
            continue
        try:
            diff[key] = float(after[key]) - float(before[key])
        except (TypeError, ValueError):
            continue
    return diff


def _series_mean(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame.get(column, pd.Series(dtype=float)), errors="coerce").mean())


def _series_std(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame.get(column, pd.Series(dtype=float)), errors="coerce").std())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare budget-objective contract behavior on a bounded teacher-rollout window.")
    parser.add_argument("--pool-name", default="learned_all_a")
    parser.add_argument("--prepare-start-date", default="20250701")
    parser.add_argument("--prepare-end-date", default="20251231")
    parser.add_argument("--compare-start-date", default="20251008")
    parser.add_argument("--compare-end-date", default="20251231")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--data-source", default="lake", choices=("lake",))
    parser.add_argument("--csv-folder", default="")
    parser.add_argument("--lake-dataset-id", default=DEFAULT_POLICY_INPUT_LAKE_DATASET_ID)
    parser.add_argument("--data-lake-root", default="")
    parser.add_argument("--max-universe-size", type=int, default=1200)
    parser.add_argument("--pool-rebalance-days", type=int, default=21)
    parser.add_argument("--pool-adv-window", type=int, default=20)
    parser.add_argument("--alpha-prior-source", default="active_execution_strategy")
    parser.add_argument("--alpha-prior-score-panel", default="")
    parser.add_argument("--alpha-prior-target-weight-panel", default="")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--label-preset", default="holdcash_v3")
    parser.add_argument("--execution-semantics", default="semantic_preserving_v1")
    parser.add_argument("--budget-semantics", default="action_budget_split_v1")
    parser.add_argument("--budget-calibration", default="cash_constraint_sell_source_guard_v7")
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=["result_value_v9", "result_value_v10"],
        choices=BUDGET_OBJECTIVE_CHOICES,
    )
    parser.add_argument("--baseline-objective", default="result_value_v9", choices=BUDGET_OBJECTIVE_CHOICES)
    parser.add_argument("--transaction-cost-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=7.0)
    parser.add_argument("--sell-tax-bps", type=float, default=10.0)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--skip-multiplier", type=float, default=2.0)
    parser.add_argument("--protocol-summary", default="")
    parser.add_argument("--objective-profile", default="sell_source_contract_v1")
    parser.add_argument("--protocol-score-output", default="")
    parser.add_argument("--output-json", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.objectives:
        raise ValueError("At least one budget objective is required.")
    baseline_objective = str(args.baseline_objective or args.objectives[0])
    if baseline_objective not in args.objectives:
        raise ValueError(f"Baseline objective {baseline_objective} must be included in --objectives.")

    output_json = Path(
        args.output_json
        or (
            Path("daily_research")
            / "output"
            / "continuous_policy"
            / "analysis"
            / "budget_objective_checks"
            / f"{timestamp_tag('budget_objective_contract_check')}.json"
        )
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)

    prepare_kwargs = {
        "pool_name": args.pool_name,
        "start_date": args.prepare_start_date,
        "end_date": args.prepare_end_date,
        "benchmark": args.benchmark,
        "data_source": args.data_source,
        "csv_folder": args.csv_folder,
        "lake_dataset_id": args.lake_dataset_id,
        "data_lake_root": args.data_lake_root,
        "max_universe_size": int(args.max_universe_size),
        "pool_rebalance_days": int(args.pool_rebalance_days),
        "pool_adv_window": int(args.pool_adv_window),
        "alpha_prior_source": args.alpha_prior_source,
        "alpha_prior_score_panel": args.alpha_prior_score_panel,
        "alpha_prior_target_weight_panel": args.alpha_prior_target_weight_panel,
        "refresh_cache": bool(args.refresh_cache),
        "progress_desc": "budget objective contract check",
    }
    prepared = prepare_policy_inputs(**prepare_kwargs)
    future_metrics = build_future_path_metrics(prepared)

    rollout_kwargs = {
        "prepared": prepared,
        "future_metrics": future_metrics,
        "start_date": args.compare_start_date,
        "end_date": args.compare_end_date,
        "transaction_cost_bps": float(args.transaction_cost_bps),
        "slippage_bps": float(args.slippage_bps),
        "sell_tax_bps": float(args.sell_tax_bps),
        "random_seed": int(args.random_seed),
        "skip_multiplier": float(args.skip_multiplier),
        "label_preset": args.label_preset,
        "execution_semantics": args.execution_semantics,
        "budget_semantics": args.budget_semantics,
        "budget_calibration": args.budget_calibration,
    }

    results: dict[str, Any] = {}
    for objective in args.objectives:
        train_frame, daily_frame, summary = build_training_matrices(
            budget_objective=objective,
            **rollout_kwargs,
        )
        results[str(objective)] = {
            "summary": summary,
            "daily_target_means": {
                column: _series_mean(daily_frame, column)
                for column in TARGET_COLUMNS
                if column in daily_frame.columns
            },
            "daily_target_std": {
                column: _series_std(daily_frame, column)
                for column in TARGET_COLUMNS
                if column in daily_frame.columns
            },
            "train_sample_rows": int(len(train_frame)),
            "daily_rows": int(len(daily_frame)),
        }

    baseline_result = results[baseline_objective]
    diff_vs_baseline: dict[str, Any] = {}
    for objective in args.objectives:
        objective = str(objective)
        if objective == baseline_objective:
            continue
        current_result = results[objective]
        diff_vs_baseline[objective] = {
            "daily_target_means": _diff_map(
                current_result.get("daily_target_means", {}),
                baseline_result.get("daily_target_means", {}),
            ),
            "daily_target_std": _diff_map(
                current_result.get("daily_target_std", {}),
                baseline_result.get("daily_target_std", {}),
            ),
            "budget_objective_diagnostics_mean": _diff_map(
                current_result.get("summary", {}).get("budget_objective_diagnostics_mean", {}),
                baseline_result.get("summary", {}).get("budget_objective_diagnostics_mean", {}),
            ),
            "teacher_rollout_metrics": _diff_map(
                current_result.get("summary", {}).get("teacher_rollout_metrics", {}),
                baseline_result.get("summary", {}).get("teacher_rollout_metrics", {}),
            ),
            "teacher_recent_turnover_mean": float(
                current_result.get("summary", {}).get("teacher_recent_turnover_mean", 0.0)
            ) - float(
                baseline_result.get("summary", {}).get("teacher_recent_turnover_mean", 0.0)
            ),
        }

    payload: dict[str, Any] = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "verification_kind": "budget_objective_contract_check",
        "baseline_objective": baseline_objective,
        "prepare_kwargs": prepare_kwargs,
        "rollout_kwargs": {
            key: value
            for key, value in rollout_kwargs.items()
            if key not in {"prepared", "future_metrics"}
        },
        "prepared_summary": prepared.to_summary(),
        "results": results,
        "diff_vs_baseline": diff_vs_baseline,
    }

    if str(args.protocol_summary or "").strip():
        protocol_summary_path = Path(str(args.protocol_summary).strip()).resolve()
        protocol_summary = json.loads(protocol_summary_path.read_text(encoding="utf-8"))
        score_payload = {
            "summary_path": str(protocol_summary_path),
            "objective_profile": str(args.objective_profile),
            "score_result": _score_protocol_summary(
                protocol_summary,
                objective_profile=str(args.objective_profile),
            ),
        }
        payload["protocol_score_smoke"] = score_payload
        if str(args.protocol_score_output or "").strip():
            protocol_score_output = Path(str(args.protocol_score_output).strip())
            protocol_score_output.parent.mkdir(parents=True, exist_ok=True)
            protocol_score_output.write_text(json.dumps(score_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_json.resolve())
    print(json.dumps(payload.get("diff_vs_baseline", {}), ensure_ascii=False, indent=2))
    if "protocol_score_smoke" in payload:
        print(json.dumps(payload["protocol_score_smoke"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
