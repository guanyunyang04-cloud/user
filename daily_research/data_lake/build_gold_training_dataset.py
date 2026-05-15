from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.baseline.data_provider import get_latest_completed_trading_date
from daily_research.continuous_policy.label_builder import LABEL_CONFIGS, build_future_path_metrics
from daily_research.continuous_policy.portfolio_simulator import (
    DEFAULT_BUDGET_CALIBRATION,
    DEFAULT_BUDGET_SEMANTICS,
    DEFAULT_EXECUTION_SEMANTICS,
)
from daily_research.continuous_policy.state_builder import (
    DEFAULT_ALPHA_PRIOR_SOURCE,
    prepare_policy_inputs,
)
from daily_research.data_lake import ResearchDataLake
from daily_research.data_lake.audit_gold_dataset import audit_gold_dataset
from daily_research.data_lake.gold_training_builder import GoldBuildSpec, build_sharded_gold_training_dataset
from daily_research.data_lake.policy_input_loader import load_policy_inputs_from_lake


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _resolve_start_date(value: str) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"", "auto", "earliest"}:
        return "20100101"
    return pd.Timestamp(raw).strftime("%Y%m%d")


def _resolve_end_date(value: str) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"", "latest", "auto"}:
        return get_latest_completed_trading_date()
    return pd.Timestamp(raw).strftime("%Y%m%d")


def _zone_list(args: argparse.Namespace) -> list[str]:
    if bool(getattr(args, "strict_only", False)):
        return ["strict_train"]
    if bool(getattr(args, "realtime_only", False)):
        return ["realtime_research"]
    zones = [item.strip() for item in str(args.zones or "").split(",") if item.strip()]
    return zones or ["strict_train", "realtime_research"]


def _progress_writer(progress_jsonl: str):
    path = Path(progress_jsonl) if str(progress_jsonl or "").strip() else None
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def emit(event: str, payload: Mapping[str, Any]) -> None:
        record = {"event": event, **dict(payload)}
        text = json.dumps(_json_safe(record), ensure_ascii=False)
        if path is not None:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(text + "\n")
        print(text, flush=True)

    return emit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build resumable sharded Gold continuous-policy training datasets.")
    parser.add_argument("--universe", default="learned_all_a")
    parser.add_argument("--max-universe-size", type=int, default=0)
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--start-date", default="2010-01-04")
    parser.add_argument("--end-date", default="latest")
    parser.add_argument("--zones", default="strict_train,realtime_research")
    parser.add_argument("--strict-only", action="store_true")
    parser.add_argument("--realtime-only", action="store_true")
    parser.add_argument("--source-market-dataset-id", default="")
    parser.add_argument("--shard-frequency", default="quarter", choices=("month", "quarter", "year", "all"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--data-source", default="tq", choices=("tq", "csv", "lake"))
    parser.add_argument("--csv-folder", default="")
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
    parser.add_argument("--progress-jsonl", default="")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument(
        "--min-signal-dates",
        type=int,
        default=25,
        help="Minimum signal dates required after strict forward-horizon trimming. Keep default for real builds; lower only for smoke tests.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    requested_start = _resolve_start_date(args.start_date)
    requested_end = _resolve_end_date(args.end_date)
    zones = _zone_list(args)
    emit = _progress_writer(args.progress_jsonl)
    lake = ResearchDataLake(str(args.data_lake_root or "").strip() or None)
    if str(args.source_market_dataset_id or "").strip():
        prepared = load_policy_inputs_from_lake(
            lake=lake,
            dataset_id=str(args.source_market_dataset_id).strip(),
            start_date=requested_start,
            end_date=requested_end,
            pool_name=args.universe,
            benchmark=args.benchmark,
            max_universe_size=args.max_universe_size,
            min_trading_days=2,
            alpha_prior_source=args.alpha_prior_source,
            alpha_prior_score_panel=args.alpha_prior_score_panel,
            alpha_prior_target_weight_panel=args.alpha_prior_target_weight_panel,
        )
    else:
        prepared = prepare_policy_inputs(
            pool_name=args.universe,
            start_date=requested_start,
            end_date=requested_end,
            benchmark=args.benchmark,
            data_source=args.data_source,
            csv_folder=args.csv_folder,
            lake_dataset_id=args.source_market_dataset_id,
            data_lake_root=args.data_lake_root,
            max_universe_size=args.max_universe_size,
            pool_rebalance_days=args.pool_rebalance_days,
            pool_adv_window=args.pool_adv_window,
            alpha_prior_source=args.alpha_prior_source,
            alpha_prior_score_panel=args.alpha_prior_score_panel,
            alpha_prior_target_weight_panel=args.alpha_prior_target_weight_panel,
            refresh_cache=False,
            auto_trim_history=False,
            progress_desc="r64 gold data lake prepare",
        )
    prepared_start = prepared.close.index.min().strftime("%Y-%m-%d")
    prepared_end = prepared.close.index.max().strftime("%Y-%m-%d")
    future_metrics = build_future_path_metrics(prepared)
    build_spec = GoldBuildSpec(
        universe=args.universe,
        benchmark=args.benchmark,
        start_date=prepared_start if str(args.start_date).lower() in {"", "auto", "earliest"} else requested_start,
        end_date=prepared_end if str(args.end_date).lower() in {"", "auto", "latest"} else requested_end,
        max_universe_size=int(args.max_universe_size),
        data_source=args.data_source,
        label_preset=args.label_preset,
        execution_semantics=args.execution_semantics,
        budget_semantics=args.budget_semantics,
        budget_calibration=args.budget_calibration,
        budget_objective=args.budget_objective,
        alpha_prior_source=args.alpha_prior_source,
        alpha_prior_score_panel=args.alpha_prior_score_panel,
        alpha_prior_target_weight_panel=args.alpha_prior_target_weight_panel,
        transaction_cost_bps=float(args.transaction_cost_bps),
        slippage_bps=float(args.slippage_bps),
        sell_tax_bps=float(args.sell_tax_bps),
        random_seed=int(args.random_seed),
        skip_multiplier=float(args.skip_multiplier),
        source_market_dataset_id=args.source_market_dataset_id,
        shard_frequency=args.shard_frequency,
    )
    emit(
        "gold_build_prepare_complete",
        {
            "prepared_start_date": prepared_start,
            "prepared_end_date": prepared_end,
            "universe_size": len(prepared.universe),
            "zones": zones,
        },
    )
    results: list[dict[str, Any]] = []
    audit_reports: list[dict[str, Any]] = []
    for zone in zones:
        result = build_sharded_gold_training_dataset(
            lake=lake,
            prepared=prepared,
            future_metrics=future_metrics,
            build_spec=build_spec,
            zone=zone,
            resume=bool(args.resume),
            refresh=bool(args.refresh),
            min_signal_dates=int(args.min_signal_dates),
            progress=emit,
        )
        results.append(dict(result))
        if not bool(args.skip_audit):
            report = audit_gold_dataset(lake=lake, dataset_id=str(result["dataset_id"]), writeback=True)
            audit_reports.append(report)
            emit("gold_dataset_audit_complete", report)
    manifest = {
        "status": "ok" if all(str(report.get("status", "ok")) == "ok" for report in audit_reports) else "audit_failed",
        "data_lake_root": str(lake.root.resolve()),
        "catalog_path": str(lake.catalog_path.resolve()),
        "prepared_start_date": prepared_start,
        "prepared_end_date": prepared_end,
        "universe_size": len(prepared.universe),
        "source_market_dataset_id": str(args.source_market_dataset_id or ""),
        "training_datasets": results,
        "audit_reports": audit_reports,
    }
    manifest_path = lake.root / "gold_build_manifest_latest.json"
    manifest_path.write_text(json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    lake.write_catalog_manifest()
    print(json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
