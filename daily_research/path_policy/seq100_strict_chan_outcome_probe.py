"""Fast, causal outcome probe for strict Chan buy-point candidates."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_strict_chan_outcome_probe_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_strict_chan_outcome_probe_v1"
)
SCHEMA_VERSION = "seq100_strict_chan_outcome_probe/1"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _load_study(path: str | Path) -> dict[str, Any]:
    study = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
    if study.get("study_id") != "seq100_strict_chan_outcome_probe_v1":
        raise ValueError("strict_chan_outcome_probe_study_id")
    if int(study["source"]["forbidden_year"]) != 2026:
        raise ValueError("strict_chan_outcome_probe_forbidden_year")
    return study


def _cost_multipliers(
    price: float,
    date: str,
    *,
    stress: bool,
    costs: Mapping[str, Any],
) -> tuple[float, float]:
    lot_size = float(costs["lot_size"])
    commission = max(
        float(costs["commission_bps"]) / 10_000.0,
        float(costs["minimum_commission_cny"]) / max(price * lot_size, 1e-12),
    )
    transfer = float(costs["transfer_fee_bps"]) / 10_000.0
    slip = float(costs["slippage_bps"]) / 10_000.0
    if stress:
        slip *= float(costs["stress_slippage_multiplier"])
    stamp_bps = (
        float(costs["stamp_tax_bps_before_2023_08_28"])
        if date < "2023-08-28"
        else float(costs["stamp_tax_bps_from_2023_08_28"])
    )
    buy = (1.0 + slip) * (1.0 + commission + transfer)
    sell = (1.0 - slip) * (1.0 - commission - transfer - stamp_bps / 10_000.0)
    return buy, sell


def _daily_frame(bars: pd.DataFrame) -> pd.DataFrame:
    bars = bars.copy()
    bars["trade_date"] = bars["trade_date"].astype(str)
    return (
        bars.sort_values("timestamp")
        .groupby("trade_date", sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
        )
        .reset_index()
    )


def _event_candidates(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    payload = events["payload_json"].map(json.loads).apply(pd.Series)
    result = events.copy()
    for column in ("side", "point_type", "price", "variant", "level"):
        if column in payload:
            result[column] = payload[column]
    result = result[result["side"].eq("buy")].copy()
    result["confirmed_date"] = result["confirmed_time"].astype(str).str[:10]
    result = result.sort_values(["confirmed_date", "confirmed_time", "id"])
    return result.drop_duplicates(
        ["case_id", "profile", "point_type", "confirmed_date"], keep="first"
    )


def _outcome_rows(
    *,
    case_id: str,
    events: pd.DataFrame,
    daily: pd.DataFrame,
    horizons: Sequence[int],
    costs: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if events.empty or daily.empty:
        return [], {"candidate_count": 0, "eligible_count": 0}
    dates = daily["trade_date"].tolist()
    date_index = {date: index for index, date in enumerate(dates)}
    rows: list[dict[str, Any]] = []
    candidate_count = len(events)
    eligible_count = 0
    for _, event in events.iterrows():
        signal_date = str(event["confirmed_date"])
        if signal_date not in date_index:
            continue
        entry_index = date_index[signal_date] + 1
        if entry_index >= len(daily):
            continue
        eligible_count += 1
        entry_date = str(daily.iloc[entry_index]["trade_date"])
        entry_price = float(daily.iloc[entry_index]["open"])
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue
        for horizon in horizons:
            exit_index = entry_index + int(horizon) - 1
            if exit_index >= len(daily):
                continue
            window = daily.iloc[entry_index : exit_index + 1]
            exit_price = float(daily.iloc[exit_index]["close"])
            if not np.isfinite(exit_price) or exit_price <= 0:
                continue
            raw_return = exit_price / entry_price - 1.0
            row: dict[str, Any] = {
                "schema": SCHEMA_VERSION,
                "case_id": case_id,
                "symbol": str(event["symbol"]),
                "profile": str(event["profile"]),
                "point_type": int(event["point_type"]),
                "signal_id": str(event["id"]),
                "confirmed_time": str(event["confirmed_time"]),
                "entry_date": entry_date,
                "horizon_sessions": int(horizon),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "raw_return": raw_return,
                "mfe": float(window["high"].max() / entry_price - 1.0),
                "mae": float(window["low"].min() / entry_price - 1.0),
            }
            for stress in (False, True):
                buy, sell = _cost_multipliers(
                    entry_price,
                    str(daily.iloc[exit_index]["trade_date"]),
                    stress=stress,
                    costs=costs,
                )
                row["net_return_stress" if stress else "net_return_base"] = (
                    sell * (1.0 + raw_return) / buy - 1.0
                )
            rows.append(row)
    return rows, {
        "candidate_count": candidate_count,
        "eligible_count": eligible_count,
    }


def _baseline_rows(
    *,
    case_id: str,
    symbol: str,
    daily: pd.DataFrame,
    horizons: Sequence[int],
    costs: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry_index in range(len(daily)):
        entry_price = float(daily.iloc[entry_index]["open"])
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue
        for horizon in horizons:
            exit_index = entry_index + int(horizon) - 1
            if exit_index >= len(daily):
                continue
            exit_price = float(daily.iloc[exit_index]["close"])
            for stress in (False, True):
                buy, sell = _cost_multipliers(
                    entry_price,
                    str(daily.iloc[exit_index]["trade_date"]),
                    stress=stress,
                    costs=costs,
                )
                rows.append(
                    {
                        "case_id": case_id,
                        "symbol": symbol,
                        "horizon_sessions": int(horizon),
                        "scenario": "stress" if stress else "base",
                        "net_return": sell * (exit_price / entry_price) / buy - 1.0,
                    }
                )
    return rows


def _aggregate(outcomes: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    if outcomes.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in outcomes.groupby(
        ["profile", "point_type", "horizon_sessions"], sort=True
    ):
        profile, point_type, horizon = keys
        base = baseline[baseline["horizon_sessions"].eq(horizon)]
        case_means = group.groupby("case_id")["net_return_base"].mean()
        stress_case_means = group.groupby("case_id")["net_return_stress"].mean()
        base_case_means = base.groupby("case_id")["net_return"].mean()
        rows.append(
            {
                "schema": SCHEMA_VERSION,
                "profile": profile,
                "point_type": int(point_type),
                "horizon_sessions": int(horizon),
                "event_count": len(group),
                "case_count": len(case_means),
                "event_mean_net_base": group["net_return_base"].mean(),
                "event_median_net_base": group["net_return_base"].median(),
                "event_win_rate_base": (group["net_return_base"] > 0).mean(),
                "case_mean_net_base": case_means.mean(),
                "case_positive_fraction_base": (case_means > 0).mean(),
                "case_mean_net_stress": stress_case_means.mean(),
                "case_positive_fraction_stress": (stress_case_means > 0).mean(),
                "baseline_case_mean_net_base": base_case_means.mean(),
                "excess_case_mean_net_base": (
                    case_means.mean() - base_case_means.mean()
                    if len(base_case_means)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def run_probe(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    resume: bool = True,
) -> dict[str, Any]:
    study = _load_study(study_path)
    root = Path(output_root).resolve()
    summary_path = root / "summary.json"
    required = [
        summary_path,
        root / "event_outcomes.parquet",
        root / "aggregate.parquet",
    ]
    if resume and all(path.is_file() for path in required):
        return json.loads(summary_path.read_text(encoding="utf-8"))
    source_root = WORKSPACE_ROOT / str(study["source"]["stratified_audit_output"])
    horizons = [int(value) for value in study["execution"]["horizons_sessions"]]
    costs = dict(study["execution"]["costs"])
    profiles = {str(value) for value in study["source"]["profiles"]}
    outcome_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    case_stats: list[dict[str, Any]] = []
    for case_root in sorted((source_root / "cases").iterdir()):
        if not case_root.is_dir():
            continue
        events = pd.read_parquet(case_root / "profile_events.parquet")
        events = events[events["profile"].isin(profiles)]
        events = events[events["event_type"].eq("trade_point")]
        if events["confirmed_time"].astype(str).str[:10].ge("2026-01-01").any():
            raise ValueError("strict_chan_outcome_probe_forbidden_event_date")
        candidates = _event_candidates(events)
        bars = pd.read_parquet(case_root / "input_bars.parquet")
        if bars["trade_date"].astype(str).ge("2026-01-01").any():
            raise ValueError("strict_chan_outcome_probe_forbidden_bar_date")
        daily = _daily_frame(bars)
        rows, stats = _outcome_rows(
            case_id=case_root.name,
            events=candidates,
            daily=daily,
            horizons=horizons,
            costs=costs,
        )
        outcome_rows.extend(rows)
        baseline_rows.extend(
            _baseline_rows(
                case_id=case_root.name,
                symbol=str(bars["symbol"].iloc[0]),
                daily=daily,
                horizons=horizons,
                costs=costs,
            )
        )
        case_stats.append({"case_id": case_root.name, **stats})
    outcomes = pd.DataFrame(outcome_rows)
    baseline_all = pd.DataFrame(baseline_rows)
    baseline = baseline_all[baseline_all["scenario"].eq("base")]
    aggregate = _aggregate(outcomes, baseline)
    _write_parquet(root / "event_outcomes.parquet", outcomes)
    _write_parquet(root / "aggregate.parquet", aggregate)
    summary = {
        "schema": SCHEMA_VERSION,
        "status": "completed",
        "study_id": study["study_id"],
        "cases": len(case_stats),
        "candidate_count": int(sum(item["candidate_count"] for item in case_stats)),
        "eligible_candidate_count": int(
            sum(item["eligible_count"] for item in case_stats)
        ),
        "outcome_rows": len(outcomes),
        "baseline_rows": len(baseline_all),
        "case_stats": case_stats,
        "event_outcomes_path": str((root / "event_outcomes.parquet").resolve()),
        "aggregate_path": str((root / "aggregate.parquet").resolve()),
        "profit_claim": False,
    }
    _write_json(summary_path, summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_probe(
        study_path=args.study,
        output_root=args.output_root,
        resume=not args.no_resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
