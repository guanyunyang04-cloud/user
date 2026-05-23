from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.baseline.data_provider import get_latest_completed_trading_date
from daily_research.continuous_policy.label_builder import LABEL_CONFIGS, LABEL_HORIZONS, build_future_path_metrics
from daily_research.continuous_policy.pipeline_utils import build_training_matrices
from daily_research.continuous_policy.portfolio_simulator import (
    DEFAULT_BUDGET_CALIBRATION,
    DEFAULT_BUDGET_SEMANTICS,
    DEFAULT_EXECUTION_SEMANTICS,
)
from daily_research.continuous_policy.state_builder import (
    DEFAULT_ALPHA_PRIOR_SOURCE,
    prepare_policy_inputs,
)
from daily_research.data_lake import ResearchDataLake, build_label_completeness_summary
from daily_research.data_lake.policy_input_loader import DEFAULT_POLICY_INPUT_LAKE_DATASET_ID


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the local DuckDB/Parquet research data lake.")
    parser.add_argument("--universe", default="learned_all_a", help="Continuous-policy pool/universe name.")
    parser.add_argument(
        "--max-universe-size",
        type=int,
        default=0,
        help="Maximum universe size. 0 means full resolved universe with no cap.",
    )
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--start-date", default="auto")
    parser.add_argument("--end-date", default="latest")
    parser.add_argument("--zones", default="strict_train,realtime_research")
    parser.add_argument("--data-source", default="lake", choices=("lake",))
    parser.add_argument("--csv-folder", default="")
    parser.add_argument("--lake-dataset-id", default=DEFAULT_POLICY_INPUT_LAKE_DATASET_ID)
    parser.add_argument("--label-preset", default="holdcash_v3", choices=tuple(sorted(LABEL_CONFIGS)))
    parser.add_argument("--execution-semantics", default=DEFAULT_EXECUTION_SEMANTICS)
    parser.add_argument("--budget-semantics", default=DEFAULT_BUDGET_SEMANTICS)
    parser.add_argument("--budget-calibration", default=DEFAULT_BUDGET_CALIBRATION)
    parser.add_argument("--budget-objective", default="result_value_v10")
    parser.add_argument("--alpha-prior-source", default=DEFAULT_ALPHA_PRIOR_SOURCE)
    parser.add_argument("--alpha-prior-score-panel", default="")
    parser.add_argument("--alpha-prior-target-weight-panel", default="")
    parser.add_argument("--transaction-cost-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=7.0)
    parser.add_argument("--sell-tax-bps", type=float, default=10.0)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--skip-multiplier", type=float, default=2.0)
    parser.add_argument("--pool-rebalance-days", type=int, default=21)
    parser.add_argument("--pool-adv-window", type=int, default=20)
    parser.add_argument("--data-lake-root", default="")
    parser.add_argument("--refresh", action="store_true", help="Refresh upstream prepared caches and overwrite lake entries.")
    parser.add_argument("--reuse", action="store_true", help="Reuse lake entries with the same fingerprint when present.")
    parser.add_argument(
        "--market-only",
        action="store_true",
        help="Register Bronze/Silver market and feature panels, then exit without constructing Gold training matrices.",
    )
    parser.add_argument(
        "--skip-market",
        action="store_true",
        help="Skip Bronze/Silver registration and only construct requested Gold training dataset zones.",
    )
    return parser


def _resolve_start_date(value: str) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"", "auto", "earliest"}:
        return "20100101"
    return raw


def _resolve_end_date(value: str) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"", "latest", "auto"}:
        return get_latest_completed_trading_date()
    return raw


def _zone_list(value: str) -> list[str]:
    zones = [item.strip() for item in str(value or "").split(",") if item.strip()]
    return zones or ["strict_train"]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    requested_start = _resolve_start_date(args.start_date)
    requested_end = _resolve_end_date(args.end_date)
    zones = _zone_list(args.zones)
    lake = ResearchDataLake(str(args.data_lake_root or "").strip() or None)

    prepared = prepare_policy_inputs(
        pool_name=args.universe,
        start_date=requested_start,
        end_date=requested_end,
        benchmark=args.benchmark,
        data_source=args.data_source,
        csv_folder=args.csv_folder,
        lake_dataset_id=args.lake_dataset_id,
        data_lake_root=args.data_lake_root,
        max_universe_size=args.max_universe_size,
        pool_rebalance_days=args.pool_rebalance_days,
        pool_adv_window=args.pool_adv_window,
        alpha_prior_source=args.alpha_prior_source,
        alpha_prior_score_panel=args.alpha_prior_score_panel,
        alpha_prior_target_weight_panel=args.alpha_prior_target_weight_panel,
        refresh_cache=bool(args.refresh),
        auto_trim_history=False,
        progress_desc="data lake prepare",
    )
    prepared_start = prepared.close.index.min().strftime("%Y-%m-%d")
    prepared_end = prepared.close.index.max().strftime("%Y-%m-%d")
    base_spec = {
        "dataset": "continuous_policy_training_matrices",
        "pool_name": prepared.pool_name,
        "universe": args.universe,
        "benchmark": prepared.benchmark,
        "data_source": prepared.data_source,
        "csv_folder": prepared.csv_folder,
        "start_date": prepared_start,
        "end_date": prepared_end,
        "max_universe_size": int(args.max_universe_size),
        "label_preset": args.label_preset,
        "execution_semantics": args.execution_semantics,
        "budget_semantics": args.budget_semantics,
        "budget_calibration": args.budget_calibration,
        "budget_objective": args.budget_objective,
        "alpha_prior_source": args.alpha_prior_source,
        "alpha_prior_score_panel": args.alpha_prior_score_panel,
        "alpha_prior_target_weight_panel": args.alpha_prior_target_weight_panel,
        "random_seed": int(args.random_seed),
        "skip_multiplier": float(args.skip_multiplier),
    }

    market_record = None
    if not bool(args.skip_market):
        market_record = lake.save_market_data_bundle(
            spec={
                **base_spec,
                "dataset": "policy_input_bundle",
                "source": prepared.data_source,
                "benchmark_fields": ["open", "close"],
            },
            market_frames={
                "Open": prepared.open_,
                "High": prepared.high,
                "Low": prepared.low,
                "Close": prepared.close,
                "Volume": prepared.volume,
                "Amount": prepared.amount,
            },
            benchmark_close=prepared.benchmark_close,
            benchmark_open=prepared.benchmark_open,
            membership_frame=prepared.membership_frame,
            feature_frames={
                "score_none": prepared.score_none,
                "score_v2": prepared.score_v2,
                "score_blend": prepared.score_blend,
                **dict(prepared.feature_frames),
                **dict(prepared.derived_frames),
            },
            source=prepared.data_source,
            reuse=not bool(args.refresh),
        )

    training_records: list[dict[str, Any]] = []
    if not bool(args.market_only):
        future_metrics = build_future_path_metrics(prepared)
        max_forward_horizon = max(int(item) for item in getattr(future_metrics, "horizons", LABEL_HORIZONS))
        for zone in zones:
            trim_horizon = max_forward_horizon if zone == "strict_train" else 0
            sample_frame, daily_frame, teacher_summary = build_training_matrices(
                prepared=prepared,
                future_metrics=future_metrics,
                start_date=prepared_start,
                end_date=prepared_end,
                transaction_cost_bps=args.transaction_cost_bps,
                slippage_bps=args.slippage_bps,
                sell_tax_bps=args.sell_tax_bps,
                random_seed=args.random_seed,
                skip_multiplier=args.skip_multiplier,
                label_preset=args.label_preset,
                execution_semantics=args.execution_semantics,
                budget_semantics=args.budget_semantics,
                budget_calibration=args.budget_calibration,
                budget_objective=args.budget_objective,
                max_forward_horizon=trim_horizon,
            )
            label_summary = build_label_completeness_summary(
                sample_frame=sample_frame,
                daily_frame=daily_frame,
                zone=zone,
                available_trade_dates=list(prepared.close.index),
                max_forward_horizon=max_forward_horizon,
            )
            record = lake.save_training_dataset(
                spec={**base_spec, "zone": zone},
                sample_frame=sample_frame,
                daily_frame=daily_frame,
                teacher_summary=teacher_summary,
                zone=zone,
                label_completeness_summary=label_summary,
                source_cache={
                    "raw_cache_meta": prepared.raw_cache_meta,
                    "prepared_cache_meta": prepared.prepared_cache_meta,
                },
                reuse=bool(args.reuse) and not bool(args.refresh),
            )
            training_records.append(
                {
                    "dataset_id": record.dataset_id,
                    "zone": record.zone,
                    "status": record.status,
                    "sample_rows": int(len(record.sample_frame)),
                    "daily_rows": int(len(record.daily_frame)),
                    "label_completeness_summary": record.label_completeness_summary,
                }
            )

    manifest = {
        "status": "ok",
        "data_lake_root": str(lake.root.resolve()),
        "catalog_path": str(lake.catalog_path.resolve()),
        "prepared_start_date": prepared_start,
        "prepared_end_date": prepared_end,
        "universe_size": int(len(prepared.universe)),
        "market_dataset": (
            {
                "dataset_id": market_record.dataset_id,
                "status": market_record.status,
                "row_counts": market_record.row_counts,
            }
            if market_record is not None
            else {}
        ),
        "training_datasets": training_records,
    }
    manifest_path = lake.root / "manifest_latest.json"
    manifest_path.write_text(json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
