"""Path, entry, exit, and intraday-T diagnostics for the exact value policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import psutil
from scipy import stats

from daily_research.path_policy import seq100_exact_value_growth_account as account
from daily_research.path_policy import seq100_exact_value_growth_policy as exact
from daily_research.path_policy import seq100_stock_distribution as base

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_exact_value_growth_path_exit_v1"
SUMMARY_SCHEMA = "seq100_exact_value_growth_path_exit_summary/1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT
    / "daily_research/studies/seq100_exact_value_growth_path_exit_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_exact_value_growth_path_exit_v1"
)
EXPECTED_5M_TIMES = tuple(
    pd.date_range("09:35", "11:30", freq="5min").strftime("%H%M00000")
) + tuple(pd.date_range("13:05", "15:00", freq="5min").strftime("%H%M00000"))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    partial.unlink(missing_ok=True)
    frame.to_parquet(partial, index=False)
    os.replace(partial, path)


def _payload_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def load_study(
    path: str | Path = DEFAULT_STUDY_PATH,
) -> tuple[dict[str, Any], Path]:
    study_path = base._resolve_path(path)
    study = _read_json(study_path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("study_id_mismatch")
    source = dict(study["source"])
    if int(source.get("forbidden_year", -1)) != base.FORBIDDEN_YEAR:
        raise ValueError("forbidden_year_contract_mismatch")
    if tuple(study["path"]["fixed_checkpoints"]) != (5, 10, 20, 40, 60, 80, 120):
        raise ValueError("path_checkpoint_contract_mismatch")
    if tuple(study["exit"]["fixed_legal_horizons"]) != (10, 20, 40, 60, 80, 120):
        raise ValueError("exit_horizon_contract_mismatch")
    if tuple(study["entry_variants"]["fixed_open_offsets_from_signal"]) != (
        1,
        2,
        3,
        5,
    ):
        raise ValueError("entry_offset_contract_mismatch")
    if float(study["selection"]["round_trip_cost"]) != 0.006:
        raise ValueError("round_trip_cost_contract_mismatch")
    if bool(study["decision_boundary"].get("account_rule_modified", True)):
        raise ValueError("account_rule_must_remain_frozen")
    return study, study_path


def _source_contract(study: Mapping[str, Any]) -> dict[str, Path]:
    source = dict(study["source"])
    keys = (
        "exact_policy_summary",
        "candidate_features",
        "selections",
        "exact_account_summary",
        "risk_budget_trades",
        "input_manifest",
        "label_manifest",
        "pack_manifest",
        "qdp_active_manifest",
        "intraday_manifest",
    )
    paths: dict[str, Path] = {}
    for key in keys:
        path = base._resolve_path(source[key])
        expected = str(source[f"{key}_sha256"]).lower()
        if not path.is_file() or base._sha256_file(path).lower() != expected:
            raise ValueError(f"source_invalid:{key}")
        paths[key] = path
    exact.validate_summary(paths["exact_policy_summary"])
    account.validate_summary(paths["exact_account_summary"])
    active = _read_json(paths["qdp_active_manifest"])
    if active["datasets"].get("market_intraday_5m") != source["intraday_dataset_id"]:
        raise ValueError("intraday_dataset_not_active")
    return paths


def _period_name(year: int, study: Mapping[str, Any]) -> str:
    for name, years in study["evaluation"]["periods"].items():
        if int(year) in {int(value) for value in years}:
            return str(name)
    raise ValueError(f"year_outside_period_contract:{year}")


def _inference(
    values: np.ndarray,
    study: Mapping[str, Any],
    *,
    seed_add: int,
) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    return {
        **base._hac_mean(x, lag=int(study["evaluation"]["hac_lag_months"])),
        "block": base._block_interval(
            x,
            block_length=int(study["evaluation"]["block_length_months"]),
            repetitions=int(study["evaluation"]["bootstrap_repetitions"]),
            seed=int(study["evaluation"]["seed"]) + int(seed_add),
        ),
    }


def _selected_features(
    panel: base.StockPanel,
    *,
    feature_path: Path,
    selection_path: Path,
    study: Mapping[str, Any],
) -> pd.DataFrame:
    features = pd.read_parquet(feature_path)
    selections = pd.read_parquet(selection_path)
    selections = selections.loc[
        selections["policy"].eq(str(study["selection"]["policy"]))
    ].copy()
    selected = selections[
        [
            "candidate_id",
            "row_position",
            "date_idx",
            "trade_date",
            "symbol",
            "evaluation_year",
            "industry_code",
            "selection_rank",
            "top_k",
        ]
    ].merge(
        features.drop(
            columns=[
                column
                for column in (
                    "row_position",
                    "date_idx",
                    "trade_date",
                    "symbol",
                    "evaluation_year",
                    "industry_code",
                )
                if column in features
            ]
        ),
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    identity = panel.row_index[["candidate_id", "date_idx", "symbol_idx", "symbol"]]
    selected = selected.merge(
        identity,
        on=["candidate_id", "date_idx", "symbol"],
        how="left",
        validate="one_to_one",
    )
    if bool(selected["symbol_idx"].isna().any()):
        raise ValueError("selected_symbol_alignment_failed")
    selected["symbol_idx"] = selected["symbol_idx"].astype(np.int64)
    return selected.sort_values(
        ["date_idx", "selection_rank"], kind="stable"
    ).reset_index(drop=True)


def _legal_exit(
    *,
    signal_idx: int,
    symbol_idx: int,
    horizon: int,
    cutoff: int,
    close: np.ndarray,
    sellable: np.ndarray,
) -> tuple[int, float]:
    start = int(signal_idx) + int(horizon)
    if start > int(cutoff):
        return -1, math.nan
    for date_idx in range(start, int(cutoff) + 1):
        price = float(close[date_idx, int(symbol_idx)])
        if (
            bool(sellable[date_idx, int(symbol_idx)])
            and math.isfinite(price)
            and price > 0.0
        ):
            return int(date_idx), price
    return -1, math.nan


def _shape_labels(row: Mapping[str, Any]) -> tuple[dict[str, bool], str]:
    r5 = float(row["return_d5"])
    r20 = float(row["return_d20"])
    r60 = float(row["return_d60"])
    mfe20 = float(row["mfe_d20"])
    mae10 = float(row["mae_d10"])
    peak60 = float(row["peak_day_d60"])
    labels = {
        "early_spike_fade": bool(
            np.isfinite(mfe20)
            and mfe20 >= 0.08
            and np.isfinite(r60)
            and r60 <= 0.25 * mfe20
        ),
        "dip_then_recover": bool(
            np.isfinite(mae10) and mae10 <= -0.05 and np.isfinite(r60) and r60 > 0.0
        ),
        "late_breakout": bool(
            np.isfinite(r20) and r20 <= 0.0 and np.isfinite(r60) and r60 >= 0.08
        ),
        "persistent_trend": bool(
            np.isfinite([r5, r20, r60, peak60]).all()
            and r5 > 0.0
            and r20 > 0.0
            and r60 > 0.0
            and peak60 >= 40.0
        ),
        "persistent_loss": bool(
            np.isfinite([r5, r20, r60]).all() and r5 < 0.0 and r20 < 0.0 and r60 < 0.0
        ),
    }
    precedence = (
        "early_spike_fade",
        "dip_then_recover",
        "late_breakout",
        "persistent_trend",
        "persistent_loss",
    )
    primary = next((name for name in precedence if labels[name]), "mixed_range")
    return labels, primary


def _path_frame(
    panel: base.StockPanel,
    selected: pd.DataFrame,
    study: Mapping[str, Any],
) -> pd.DataFrame:
    pack = panel.pack_manifest
    dates = np.asarray(pack["date_values"], dtype=str)
    cutoff_positions = np.flatnonzero(
        dates == str(study["source"]["maximum_outcome_date"])
    )
    if len(cutoff_positions) != 1:
        raise ValueError("maximum_outcome_date_missing")
    cutoff = int(cutoff_positions[0])
    raw = base._open_pack_array(pack, "feature_channels", "daily_raw", dtype=np.float32)
    observed = base._open_pack_array(pack, "masks", "price_observed", dtype=np.bool_)
    buyable = base._open_pack_array(pack, "masks", "entry_buyable", dtype=np.bool_)
    sellable = base._open_pack_array(pack, "masks", "exit_sellable", dtype=np.bool_)
    close = np.asarray(raw[:, :, 3])
    checkpoints = tuple(int(value) for value in study["path"]["fixed_checkpoints"])
    thresholds = tuple(float(value) for value in study["path"]["profit_thresholds_net"])
    maximum = int(study["path"]["maximum_open_days"])
    cost = float(study["selection"]["round_trip_cost"])
    records: list[dict[str, Any]] = []
    for source in selected.to_dict(orient="records"):
        signal_idx = int(source["date_idx"])
        symbol_idx = int(source["symbol_idx"])
        entry_idx = signal_idx + 1
        entry_price = (
            float(raw[entry_idx, symbol_idx, 0]) if entry_idx <= cutoff else math.nan
        )
        entry_valid = bool(
            entry_idx <= cutoff
            and buyable[entry_idx, symbol_idx]
            and observed[entry_idx, symbol_idx]
            and math.isfinite(entry_price)
            and entry_price > 0.0
        )
        record: dict[str, Any] = {
            **source,
            "entry_date_idx": entry_idx,
            "entry_date": dates[entry_idx] if entry_idx < len(dates) else None,
            "entry_adjusted_open": entry_price,
            "entry_valid": entry_valid,
            "signal_adjusted_close": float(raw[signal_idx, symbol_idx, 3]),
            "next_open_gap": (
                entry_price / float(raw[signal_idx, symbol_idx, 3]) - 1.0
                if entry_valid and float(raw[signal_idx, symbol_idx, 3]) > 0.0
                else math.nan
            ),
        }
        highs = np.full(maximum + 1, np.nan, dtype=np.float64)
        lows = np.full(maximum + 1, np.nan, dtype=np.float64)
        closes = np.full(maximum + 1, np.nan, dtype=np.float64)
        legal = np.zeros(maximum + 1, dtype=bool)
        if entry_valid:
            for offset in range(1, maximum + 1):
                date_idx = signal_idx + offset
                if date_idx > cutoff:
                    break
                if bool(observed[date_idx, symbol_idx]):
                    high = float(raw[date_idx, symbol_idx, 1])
                    low = float(raw[date_idx, symbol_idx, 2])
                    day_close = float(raw[date_idx, symbol_idx, 3])
                    if math.isfinite(high) and high > 0.0:
                        highs[offset] = high / entry_price - 1.0
                    if math.isfinite(low) and low > 0.0:
                        lows[offset] = low / entry_price - 1.0
                    if math.isfinite(day_close) and day_close > 0.0:
                        closes[offset] = day_close / entry_price - 1.0
                legal[offset] = bool(sellable[date_idx, symbol_idx])
        for horizon in checkpoints:
            window = slice(1, horizon + 1)
            record[f"return_d{horizon}"] = closes[horizon]
            record[f"mfe_d{horizon}"] = (
                float(np.nanmax(highs[window]))
                if np.isfinite(highs[window]).any()
                else math.nan
            )
            record[f"mae_d{horizon}"] = (
                float(np.nanmin(lows[window]))
                if np.isfinite(lows[window]).any()
                else math.nan
            )
            record[f"peak_day_d{horizon}"] = (
                int(np.nanargmax(highs[window])) + 1
                if np.isfinite(highs[window]).any()
                else -1
            )
            for threshold in thresholds:
                key = f"profit_{round(threshold * 100):02d}_by_d{horizon}"
                qualifying = np.flatnonzero(
                    legal[2 : horizon + 1]
                    & np.isfinite(closes[2 : horizon + 1])
                    & (closes[2 : horizon + 1] - cost >= threshold)
                )
                record[key] = bool(len(qualifying))
                if threshold == 0.0:
                    record[f"first_net_profit_day_d{horizon}"] = (
                        int(qualifying[0]) + 2 if len(qualifying) else -1
                    )
        for horizon in study["exit"]["fixed_legal_horizons"]:
            exit_idx, exit_price = _legal_exit(
                signal_idx=signal_idx,
                symbol_idx=symbol_idx,
                horizon=int(horizon),
                cutoff=cutoff,
                close=close,
                sellable=sellable,
            )
            record[f"legal_exit_idx_d{horizon}"] = exit_idx
            record[f"legal_exit_date_d{horizon}"] = (
                dates[exit_idx] if exit_idx >= 0 else None
            )
            record[f"legal_net_return_d{horizon}"] = (
                exit_price / entry_price - 1.0 - cost
                if entry_valid and math.isfinite(exit_price)
                else math.nan
            )
        labels, primary = _shape_labels(record)
        record.update({f"shape__{key}": value for key, value in labels.items()})
        record["primary_shape"] = primary
        records.append(record)
    return pd.DataFrame(records)


def _evaluation_groups(
    frame: pd.DataFrame, study: Mapping[str, Any]
) -> list[tuple[str, pd.DataFrame]]:
    local = frame.copy()
    if "period" not in local:
        local["period"] = local["evaluation_year"].map(
            lambda value: _period_name(int(value), study)
        )
    groups = [(str(name), group) for name, group in local.groupby("period", sort=True)]
    groups.append(("full_history", local))
    return groups


def _entry_results(
    panel: base.StockPanel,
    paths: pd.DataFrame,
    study: Mapping[str, Any],
) -> pd.DataFrame:
    """Evaluate frozen causal entry requests against the same signal-D60 exit."""

    pack = panel.pack_manifest
    dates = np.asarray(pack["date_values"], dtype=str)
    cutoff = int(
        np.flatnonzero(dates == str(study["source"]["maximum_outcome_date"]))[0]
    )
    raw = base._open_pack_array(pack, "feature_channels", "daily_raw", dtype=np.float32)
    observed = base._open_pack_array(pack, "masks", "price_observed", dtype=np.bool_)
    buyable = base._open_pack_array(pack, "masks", "entry_buyable", dtype=np.bool_)
    cost = float(study["selection"]["round_trip_cost"])
    variants: list[tuple[str, str, float | int | None]] = [
        (f"open_d{offset}", "fixed", int(offset))
        for offset in study["entry_variants"]["fixed_open_offsets_from_signal"]
    ]
    variants.extend(
        (
            f"pullback_{abs(round(float(threshold) * 100))}pct",
            "pullback",
            float(threshold),
        )
        for threshold in study["entry_variants"]["pullback_close_thresholds"]
    )
    variants.append(("avoid_gap_gt2_wait_to_plus1", "overextension", None))
    records: list[dict[str, Any]] = []
    observation_days = int(study["entry_variants"]["pullback_observation_days"])
    for source in paths.to_dict(orient="records"):
        signal_idx = int(source["date_idx"])
        symbol_idx = int(source["symbol_idx"])
        signal_close = float(source["signal_adjusted_close"])
        exit_idx = int(source["legal_exit_idx_d60"])
        exit_price = float(raw[exit_idx, symbol_idx, 3]) if exit_idx >= 0 else math.nan
        d60_evaluable = bool(signal_idx + 60 <= cutoff)
        for name, kind, parameter in variants:
            trigger_day = -1
            entry_offset = -1
            rule_triggered = False
            if kind == "fixed":
                entry_offset = int(parameter)
                trigger_day = 0
                rule_triggered = True
            elif kind == "pullback":
                threshold = float(parameter)
                for offset in range(1, observation_days + 1):
                    idx = signal_idx + offset
                    if idx > cutoff or not bool(observed[idx, symbol_idx]):
                        continue
                    day_close = float(raw[idx, symbol_idx, 3])
                    if math.isfinite(day_close) and day_close <= signal_close * (
                        1.0 + threshold
                    ):
                        trigger_day = offset
                        entry_offset = offset + 1
                        rule_triggered = True
                        break
            else:
                baseline_idx = signal_idx + 1
                baseline_price = (
                    float(raw[baseline_idx, symbol_idx, 0])
                    if baseline_idx <= cutoff
                    else math.nan
                )
                gap = (
                    baseline_price / signal_close - 1.0
                    if math.isfinite(baseline_price)
                    and baseline_price > 0.0
                    and signal_close > 0.0
                    else math.nan
                )
                if np.isfinite(gap) and gap <= 0.02:
                    trigger_day = 0
                    entry_offset = 1
                    rule_triggered = True
                elif np.isfinite(gap) and gap > 0.02:
                    for offset in range(1, observation_days + 1):
                        idx = signal_idx + offset
                        if idx > cutoff or not bool(observed[idx, symbol_idx]):
                            continue
                        day_close = float(raw[idx, symbol_idx, 3])
                        if (
                            math.isfinite(day_close)
                            and day_close <= signal_close * 1.01
                        ):
                            trigger_day = offset
                            entry_offset = offset + 1
                            rule_triggered = True
                            break
            entry_idx = signal_idx + entry_offset if entry_offset >= 0 else -1
            entry_price = (
                float(raw[entry_idx, symbol_idx, 0])
                if 0 <= entry_idx <= cutoff
                else math.nan
            )
            entry_filled = bool(
                rule_triggered
                and 0 <= entry_idx <= cutoff
                and bool(observed[entry_idx, symbol_idx])
                and bool(buyable[entry_idx, symbol_idx])
                and math.isfinite(entry_price)
                and entry_price > 0.0
            )
            outcome_valid = bool(
                d60_evaluable
                and entry_filled
                and exit_idx >= entry_idx
                and math.isfinite(exit_price)
                and exit_price > 0.0
            )
            records.append(
                {
                    "candidate_id": source["candidate_id"],
                    "trade_date": source["trade_date"],
                    "evaluation_year": int(source["evaluation_year"]),
                    "top_k": int(source["top_k"]),
                    "symbol": source["symbol"],
                    "variant": name,
                    "variant_kind": kind,
                    "rule_triggered": rule_triggered,
                    "trigger_day": trigger_day,
                    "entry_offset": entry_offset,
                    "entry_date": dates[entry_idx] if entry_idx >= 0 else None,
                    "entry_adjusted_open": entry_price,
                    "entry_filled": entry_filled,
                    "d60_evaluable": d60_evaluable,
                    "outcome_valid": outcome_valid,
                    "common_exit_date": dates[exit_idx] if exit_idx >= 0 else None,
                    "net_return": (
                        exit_price / entry_price - 1.0 - cost
                        if outcome_valid
                        else math.nan
                    ),
                }
            )
    return pd.DataFrame(records)


def _entry_monthly(entry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    eligible = entry.loc[entry["d60_evaluable"]].copy()
    for (variant, trade_date), group in eligible.groupby(
        ["variant", "trade_date"], sort=True
    ):
        top_k = int(group["top_k"].iloc[0])
        valid = group["outcome_valid"].to_numpy(dtype=bool)
        values = group["net_return"].to_numpy(dtype=float)
        rows.append(
            {
                "variant": str(variant),
                "trade_date": str(trade_date),
                "evaluation_year": int(group["evaluation_year"].iloc[0]),
                "top_k": top_k,
                "selected_count": len(group),
                "trigger_fraction": float(group["rule_triggered"].mean()),
                "fill_fraction": float(group["entry_filled"].mean()),
                "outcome_fraction": float(valid.sum() / top_k),
                "cash_denominator_net_return": float(np.nansum(values[valid]) / top_k),
            }
        )
    frame = pd.DataFrame(rows)
    baseline = frame.loc[
        frame["variant"].eq("open_d1"), ["trade_date", "cash_denominator_net_return"]
    ].rename(columns={"cash_denominator_net_return": "baseline_net_return"})
    frame = frame.merge(baseline, on="trade_date", how="left", validate="many_to_one")
    frame["paired_delta_vs_open_d1"] = (
        frame["cash_denominator_net_return"] - frame["baseline_net_return"]
    )
    return frame


def _entry_summaries(
    monthly: pd.DataFrame, study: Mapping[str, Any]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for period, period_frame in _evaluation_groups(monthly, study):
        for variant, group in period_frame.groupby("variant", sort=True):
            annual = group.groupby("evaluation_year", sort=True)[
                "cash_denominator_net_return"
            ].mean()
            results.append(
                {
                    "period": period,
                    "variant": str(variant),
                    "month_count": len(group),
                    "mean_trigger_fraction": float(group["trigger_fraction"].mean()),
                    "mean_fill_fraction": float(group["fill_fraction"].mean()),
                    "net_return": _inference(
                        group["cash_denominator_net_return"].to_numpy(),
                        study,
                        seed_add=100 + len(results),
                    ),
                    "paired_delta_vs_open_d1": _inference(
                        group["paired_delta_vs_open_d1"].to_numpy(),
                        study,
                        seed_add=500 + len(results),
                    ),
                    "positive_year_count": int((annual > 0.0).sum()),
                    "year_count": len(annual),
                }
            )
    return results


def _exit_monthly(paths: pd.DataFrame, study: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon in study["exit"]["fixed_legal_horizons"]:
        column = f"legal_net_return_d{int(horizon)}"
        eligible = paths.loc[paths[column].notna()].copy()
        for trade_date, group in eligible.groupby("trade_date", sort=True):
            top_k = int(group["top_k"].iloc[0])
            values = group[column].to_numpy(dtype=float)
            rows.append(
                {
                    "horizon": int(horizon),
                    "trade_date": str(trade_date),
                    "evaluation_year": int(group["evaluation_year"].iloc[0]),
                    "top_k": top_k,
                    "observed_count": int(np.isfinite(values).sum()),
                    "cash_denominator_net_return": float(np.nansum(values) / top_k),
                }
            )
    frame = pd.DataFrame(rows)
    baseline_horizon = int(study["exit"]["baseline_horizon"])
    baseline = frame.loc[
        frame["horizon"].eq(baseline_horizon),
        ["trade_date", "cash_denominator_net_return"],
    ].rename(columns={"cash_denominator_net_return": "baseline_d60_net_return"})
    frame = frame.merge(baseline, on="trade_date", how="left", validate="many_to_one")
    frame["paired_delta_vs_d60"] = (
        frame["cash_denominator_net_return"] - frame["baseline_d60_net_return"]
    )
    return frame


def _exit_summaries(
    monthly: pd.DataFrame, study: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for period, period_frame in _evaluation_groups(monthly, study):
        for horizon, group in period_frame.groupby("horizon", sort=True):
            paired = group.loc[group["baseline_d60_net_return"].notna()]
            annual = group.groupby("evaluation_year", sort=True)[
                "cash_denominator_net_return"
            ].mean()
            results.append(
                {
                    "period": period,
                    "horizon": int(horizon),
                    "month_count": len(group),
                    "observed_fraction": float(
                        group["observed_count"].sum() / group["top_k"].sum()
                    ),
                    "net_return": _inference(
                        group["cash_denominator_net_return"].to_numpy(),
                        study,
                        seed_add=1000 + len(results),
                    ),
                    "paired_month_count": len(paired),
                    "paired_delta_vs_d60": _inference(
                        paired["paired_delta_vs_d60"].to_numpy(),
                        study,
                        seed_add=1500 + len(results),
                    ),
                    "positive_year_count": int((annual > 0.0).sum()),
                    "year_count": len(annual),
                }
            )
    challengers: list[dict[str, Any]] = []
    baseline = int(study["exit"]["baseline_horizon"])
    for horizon in sorted(monthly["horizon"].unique()):
        if int(horizon) == baseline:
            continue
        full = next(
            row
            for row in results
            if row["period"] == "full_history" and row["horizon"] == int(horizon)
        )
        period_rows = [
            row
            for row in results
            if row["period"] != "full_history" and row["horizon"] == int(horizon)
        ]
        delta = full["paired_delta_vs_d60"]
        positive_periods = sum(
            float(row["paired_delta_vs_d60"]["mean"]) > 0.0 for row in period_rows
        )
        passed = bool(
            float(delta["lcb_95"]) > 0.0
            and float(delta["block"]["lcb_95"]) > 0.0
            and positive_periods >= 2
        )
        challengers.append(
            {
                "horizon": int(horizon),
                "positive_period_count": positive_periods,
                "passed": passed,
            }
        )
    decision = {
        "d60_robustly_challenged": any(row["passed"] for row in challengers),
        "passing_challengers": [row["horizon"] for row in challengers if row["passed"]],
        "challengers": challengers,
        "interpretation": "retain D60 as the least-disproved frozen baseline"
        if not any(row["passed"] for row in challengers)
        else "a frozen fixed-horizon challenger dominates D60 historically",
    }
    return results, decision


def _path_summaries(
    paths: pd.DataFrame, study: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checkpoint_results: list[dict[str, Any]] = []
    valid_paths = paths.loc[paths["entry_valid"]].copy()
    for period, period_frame in _evaluation_groups(valid_paths, study):
        for horizon in study["path"]["fixed_checkpoints"]:
            return_column = f"return_d{int(horizon)}"
            group = period_frame.loc[period_frame[return_column].notna()]
            if group.empty:
                continue
            first = group[f"first_net_profit_day_d{int(horizon)}"].to_numpy(dtype=int)
            record: dict[str, Any] = {
                "period": period,
                "horizon": int(horizon),
                "candidate_count": len(group),
                "mean_close_return": float(group[return_column].mean()),
                "median_close_return": float(group[return_column].median()),
                "positive_close_fraction": float(group[return_column].gt(0.0).mean()),
                "mean_mfe": float(group[f"mfe_d{int(horizon)}"].mean()),
                "median_mfe": float(group[f"mfe_d{int(horizon)}"].median()),
                "mean_mae": float(group[f"mae_d{int(horizon)}"].mean()),
                "median_mae": float(group[f"mae_d{int(horizon)}"].median()),
                "median_first_net_profit_day": float(np.median(first[first > 0]))
                if bool((first > 0).any())
                else math.nan,
            }
            for threshold in study["path"]["profit_thresholds_net"]:
                label = round(float(threshold) * 100)
                record[f"legal_profit_{label:02d}_fraction"] = float(
                    group[f"profit_{label:02d}_by_d{int(horizon)}"].mean()
                )
            checkpoint_results.append(record)
    shape_results: list[dict[str, Any]] = []
    shape_valid = valid_paths.loc[valid_paths["return_d60"].notna()]
    for period, period_frame in _evaluation_groups(shape_valid, study):
        for shape, group in period_frame.groupby("primary_shape", sort=True):
            shape_results.append(
                {
                    "period": period,
                    "primary_shape": str(shape),
                    "candidate_count": len(group),
                    "fraction": float(len(group) / len(period_frame)),
                    "median_return_d5": float(group["return_d5"].median()),
                    "median_return_d20": float(group["return_d20"].median()),
                    "median_return_d60": float(group["return_d60"].median()),
                    "median_mfe_d60": float(group["mfe_d60"].median()),
                    "median_mae_d60": float(group["mae_d60"].median()),
                }
            )
    return checkpoint_results, shape_results


def _continuation_states(
    paths: pd.DataFrame, study: Mapping[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frame = paths.loc[
        paths[["return_d10", "return_d20", "return_d60"]].notna().all(axis=1)
    ].copy()
    bins = [float(value) for value in study["exit"]["continuation_return_bins"]]
    labels = [
        f"[{bins[index]:g},{bins[index + 1]:g})" for index in range(len(bins) - 1)
    ]
    frame["d20_return_bin"] = pd.cut(
        frame["return_d20"], bins=bins, labels=labels, right=False
    ).astype(str)
    frame["recent_d10_to_d20_trend"] = np.where(
        frame["return_d20"] >= frame["return_d10"], "improving", "weakening"
    )
    frame["d20_giveback_fraction"] = np.where(
        frame["mfe_d20"] > 0.0,
        (frame["mfe_d20"] - frame["return_d20"]) / frame["mfe_d20"],
        np.nan,
    )
    frame["large_d20_giveback"] = frame["d20_giveback_fraction"].ge(
        float(study["exit"]["giveback_threshold"])
    )
    frame["d20_to_d60_return"] = (1.0 + frame["return_d60"]) / (
        1.0 + frame["return_d20"]
    ) - 1.0
    frame["d60_to_d120_return"] = np.where(
        frame["return_d120"].notna(),
        (1.0 + frame["return_d120"]) / (1.0 + frame["return_d60"]) - 1.0,
        np.nan,
    )
    summaries: list[dict[str, Any]] = []
    dimensions = ("d20_return_bin", "recent_d10_to_d20_trend", "large_d20_giveback")
    for period, period_frame in _evaluation_groups(frame, study):
        for dimension in dimensions:
            for state, group in period_frame.groupby(dimension, sort=True):
                summaries.append(
                    {
                        "period": period,
                        "dimension": dimension,
                        "state": str(state),
                        "candidate_count": len(group),
                        "mean_d20_to_d60_return": float(
                            group["d20_to_d60_return"].mean()
                        ),
                        "median_d20_to_d60_return": float(
                            group["d20_to_d60_return"].median()
                        ),
                        "positive_d20_to_d60_fraction": float(
                            group["d20_to_d60_return"].gt(0.0).mean()
                        ),
                        "mean_d60_to_d120_return": float(
                            group["d60_to_d120_return"].mean()
                        ),
                        "median_d60_to_d120_return": float(
                            group["d60_to_d120_return"].median()
                        ),
                    }
                )
    columns = [
        "candidate_id",
        "trade_date",
        "evaluation_year",
        "symbol",
        "return_d10",
        "return_d20",
        "return_d60",
        "return_d120",
        "d20_return_bin",
        "recent_d10_to_d20_trend",
        "d20_giveback_fraction",
        "large_d20_giveback",
        "d20_to_d60_return",
        "d60_to_d120_return",
    ]
    return frame[columns], summaries


def _attach_signal_features(
    panel: base.StockPanel,
    paths: pd.DataFrame,
    study: Mapping[str, Any],
) -> pd.DataFrame:
    names = [str(name) for name in study["signal_feature_screen"]["features"]]
    missing = [name for name in names if name not in paths]
    if not missing:
        return paths
    positions = base._feature_positions(panel, missing)
    rows = paths["row_position"].to_numpy(dtype=np.int64)
    matrix = base._feature_matrix(panel, rows, positions).astype(np.float64)
    result = paths.copy()
    for index, name in enumerate(missing):
        result[name] = matrix[:, index]
    return result


def _bh_qvalues(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return result
    order = valid[np.argsort(values[valid], kind="stable")]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.minimum(ranked, 1.0)
    return result


def _feature_screen(
    frame: pd.DataFrame, study: Mapping[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    targets = {
        "d60_legal_net_return": "legal_net_return_d60",
        "d20_net_profit_opportunity": "profit_00_by_d20",
        "d10_drawdown_less_severe": "mae_d10",
    }
    features = [str(name) for name in study["signal_feature_screen"]["features"]]
    deltas: list[dict[str, Any]] = []
    for target_name, target_column in targets.items():
        target_dates = int(
            frame.loc[frame[target_column].notna(), "trade_date"].nunique()
        )
        for feature in features:
            date_rows: list[dict[str, Any]] = []
            for trade_date, group in frame.loc[
                frame[[feature, target_column]].notna().all(axis=1)
            ].groupby("trade_date", sort=True):
                ordered = group.sort_values([feature, "candidate_id"], kind="stable")
                half = len(ordered) // 2
                if half < 2:
                    continue
                lower = ordered.iloc[:half]
                upper = ordered.iloc[-half:]
                date_rows.append(
                    {
                        "target": target_name,
                        "feature": feature,
                        "trade_date": str(trade_date),
                        "evaluation_year": int(group["evaluation_year"].iloc[0]),
                        "upper_minus_lower": float(
                            upper[target_column].astype(float).mean()
                            - lower[target_column].astype(float).mean()
                        ),
                    }
                )
            if date_rows:
                coverage = len(date_rows) / max(target_dates, 1)
                for row in date_rows:
                    row["date_coverage"] = coverage
                deltas.extend(date_rows)
    delta_frame = pd.DataFrame(deltas)
    summaries: list[dict[str, Any]] = []
    if delta_frame.empty:
        return delta_frame, summaries
    for (target, feature), group in delta_frame.groupby(
        ["target", "feature"], sort=True
    ):
        period_metrics: dict[str, dict[str, Any]] = {}
        for period, period_frame in _evaluation_groups(group, study):
            inference = _inference(
                period_frame["upper_minus_lower"].to_numpy(),
                study,
                seed_add=2500 + len(summaries) * 10 + len(period_metrics),
            )
            standard_error = float(inference["standard_error"])
            mean = float(inference["mean"])
            p_value = (
                float(2.0 * stats.norm.sf(abs(mean / standard_error)))
                if np.isfinite(standard_error) and standard_error > 0.0
                else (0.0 if np.isfinite(mean) and mean != 0.0 else 1.0)
            )
            period_metrics[period] = {**inference, "p_value_two_sided": p_value}
        summaries.append(
            {
                "target": str(target),
                "feature": str(feature),
                "date_coverage": float(group["date_coverage"].iloc[0]),
                "period_metrics": period_metrics,
            }
        )
    fdr = float(study["signal_feature_screen"]["false_discovery_rate"])
    minimum_coverage = float(study["signal_feature_screen"]["minimum_date_coverage"])
    for target in targets:
        target_rows = [row for row in summaries if row["target"] == target]
        q_values = _bh_qvalues(
            [
                row["period_metrics"]["full_history"]["p_value_two_sided"]
                for row in target_rows
            ]
        )
        for row, q_value in zip(target_rows, q_values, strict=True):
            full = row["period_metrics"]["full_history"]
            orientation = 1.0 if float(full["mean"]) >= 0.0 else -1.0
            oriented = _inference(
                orientation
                * delta_frame.loc[
                    delta_frame["target"].eq(target)
                    & delta_frame["feature"].eq(row["feature"]),
                    "upper_minus_lower",
                ].to_numpy(),
                study,
                seed_add=4000 + len(summaries),
            )
            period_positive = sum(
                orientation * float(metrics["mean"]) > 0.0
                for name, metrics in row["period_metrics"].items()
                if name != "full_history"
            )
            row["favored_half"] = "upper" if orientation > 0.0 else "lower"
            row["full_history_q_value"] = float(q_value)
            row["oriented_full_history"] = oriented
            row["oriented_positive_period_count"] = int(period_positive)
            row["robust_diagnostic"] = bool(
                row["date_coverage"] >= minimum_coverage
                and float(q_value) <= fdr
                and float(oriented["lcb_95"]) > 0.0
                and float(oriented["block"]["lcb_95"]) > 0.0
                and period_positive == 3
            )
    return delta_frame, summaries


def _duckdb_connection(
    temporary_root: Path,
) -> tuple[duckdb.DuckDBPyConnection, dict[str, Any]]:
    available = int(psutil.virtual_memory().available)
    reserve = min(max(int(available * 0.25), 2 * 1024**3), 8 * 1024**3)
    usable = max(available - reserve, 512 * 1024**2)
    memory_limit = min(max(int(usable * 0.60), 512 * 1024**2), 16 * 1024**3)
    cpu_count = max(int(os.cpu_count() or 1), 1)
    threads = max(1, min(8, cpu_count, int(max(usable, 1) / 1024**3) + 1))
    temporary_root.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute(f"SET threads={threads}")
    connection.execute(f"SET memory_limit='{memory_limit // 1024**2}MB'")
    connection.execute("SET preserve_insertion_order=false")
    connection.execute("SET enable_progress_bar=false")
    escaped = str(temporary_root.resolve()).replace("'", "''")
    connection.execute(f"SET temp_directory='{escaped}'")
    return connection, {
        "available_memory_gb_at_start": available / 1024**3,
        "duckdb_memory_limit_gb": memory_limit / 1024**3,
        "duckdb_threads": threads,
    }


def _intraday_paths_for_year(
    manifest: Mapping[str, Any], *, year: int, maximum_date: str
) -> list[Path]:
    root = WORKSPACE_ROOT / "quant_data_platform/data/qdp_v2"
    paths: list[Path] = []
    for shard in manifest["shards"]:
        start = str(shard.get("start_date", ""))
        end = str(shard.get("end_date", ""))
        start_year = int(start[:4]) if start[:4].isdigit() else 0
        end_year = int(end[:4]) if end[:4].isdigit() else int(maximum_date[:4])
        if start_year > int(maximum_date[:4]) or not start_year <= year <= end_year:
            continue
        path = root / str(shard["path"])
        if not path.is_file():
            raise ValueError(f"intraday_shard_missing:{path}")
        paths.append(path.resolve())
    if not paths:
        raise ValueError(f"intraday_shards_empty:{year}")
    return paths


def _parquet_scan(paths: Sequence[Path]) -> str:
    quoted = ", ".join(
        "'" + str(path).replace("\\", "/").replace("'", "''") + "'" for path in paths
    )
    return f"read_parquet([{quoted}], union_by_name=true, filename=true)"


def _complete_intraday_days(
    trades: pd.DataFrame,
    *,
    manifest_path: Path,
    maximum_date: str,
    temporary_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    windows = trades[
        ["trade_id", "symbol", "entry_date", "exit_date", "buy_notional_cny"]
    ].copy()
    for column in ("entry_date", "exit_date"):
        windows[column] = windows[column].astype(str)
    manifest = _read_json(manifest_path)
    connection, runtime = _duckdb_connection(temporary_root)
    connection.register("holding_windows", windows)
    frames: list[pd.DataFrame] = []
    year_runtime: list[dict[str, Any]] = []
    maximum_year = int(maximum_date[:4])
    try:
        for year in range(2012, maximum_year + 1):
            year_start = f"{year}-01-01"
            year_end = min(f"{year}-12-31", maximum_date)
            paths = _intraday_paths_for_year(
                manifest, year=year, maximum_date=maximum_date
            )
            scan = _parquet_scan(paths)
            started = time.perf_counter()
            year_frame = connection.execute(
                f"""
                WITH matched AS (
                    SELECT h.trade_id, h.symbol, h.buy_notional_cny,
                           m.trade_date, m.bar_time, m.open, m.high, m.low, m.close,
                           m.volume, m.amount,
                           row_number() OVER (
                               PARTITION BY h.trade_id, m.trade_date, m.bar_time
                               ORDER BY m.source DESC, m.filename DESC
                           ) AS duplicate_rank,
                           count(*) OVER (
                               PARTITION BY h.trade_id, m.trade_date, m.bar_time
                           ) AS duplicate_count
                    FROM {scan} AS m
                    INNER JOIN holding_windows AS h
                      ON m.symbol = h.symbol
                     AND m.trade_date > h.entry_date
                     AND m.trade_date < h.exit_date
                    WHERE m.trade_date BETWEEN ? AND ?
                ), deduplicated AS (
                    SELECT * FROM matched WHERE duplicate_rank = 1
                )
                SELECT trade_id, any_value(symbol) AS symbol,
                       any_value(buy_notional_cny) AS buy_notional_cny,
                       trade_date,
                       count(*) AS bar_count,
                       max(duplicate_count) AS maximum_duplicate_count,
                       list(bar_time ORDER BY bar_time) AS bar_times,
                       list(open ORDER BY bar_time) AS opens,
                       list(high ORDER BY bar_time) AS highs,
                       list(low ORDER BY bar_time) AS lows,
                       list(close ORDER BY bar_time) AS closes,
                       list(volume ORDER BY bar_time) AS volumes,
                       list(amount ORDER BY bar_time) AS amounts
                FROM deduplicated
                GROUP BY trade_id, trade_date
                ORDER BY trade_date, trade_id
                """,
                [year_start, year_end],
            ).fetchdf()
            frames.append(year_frame)
            year_runtime.append(
                {
                    "year": year,
                    "shard_count": len(paths),
                    "matched_stock_day_count": len(year_frame),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
    finally:
        connection.close()
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    expected = list(EXPECTED_5M_TIMES)

    def complete(row: pd.Series) -> bool:
        if int(row["bar_count"]) != len(expected):
            return False
        times = [str(value) for value in row["bar_times"]]
        if times != expected:
            return False
        arrays = (row["opens"], row["highs"], row["lows"], row["closes"])
        return all(
            len(values) == len(expected)
            and bool(np.isfinite(np.asarray(values, dtype=float)).all())
            and bool((np.asarray(values, dtype=float) > 0.0).all())
            for values in arrays
        )

    if frame.empty:
        frame["complete"] = pd.Series(dtype=bool)
    else:
        frame["complete"] = frame.apply(complete, axis=1)
    runtime.update(
        {
            "year_partitions": year_runtime,
            "matched_stock_day_count": len(frame),
            "complete_stock_day_count": int(frame["complete"].sum()),
            "stock_days_with_duplicate_source_rows": int(
                frame["maximum_duplicate_count"].gt(1).sum()
            ),
        }
    )
    return frame.loc[frame["complete"]].reset_index(drop=True), runtime


def _t_rule_day(
    opens: Sequence[float],
    closes: Sequence[float],
    rule: Mapping[str, Any],
    *,
    cost: float,
    tranche: float,
) -> dict[str, Any]:
    open_values = np.asarray(opens, dtype=float)
    close_values = np.asarray(closes, dtype=float)
    if (
        len(open_values) != len(close_values)
        or len(open_values) < 3
        or not np.isfinite(open_values).all()
        or not np.isfinite(close_values).all()
        or bool((open_values <= 0.0).any())
        or bool((close_values <= 0.0).any())
    ):
        raise ValueError("intraday_rule_input_invalid")
    direction = str(rule["direction"])
    trigger = float(rule["trigger_from_day_open"])
    paired_move = float(rule["paired_move"])
    day_open = float(open_values[0])
    trigger_signal = -1
    first_fill = -1
    second_fill = -1
    paired_completed = False
    first_price = math.nan
    second_price = math.nan
    if direction == "buy_then_sell_old_inventory":
        for index in range(len(close_values) - 1):
            if close_values[index] <= day_open * (1.0 - trigger):
                trigger_signal = index
                first_fill = index + 1
                first_price = float(open_values[first_fill])
                break
        if first_fill >= 0:
            for index in range(first_fill, len(close_values) - 1):
                if close_values[index] >= first_price * (1.0 + paired_move):
                    second_fill = index + 1
                    second_price = float(open_values[second_fill])
                    paired_completed = True
                    break
            if second_fill < 0:
                second_fill = len(close_values) - 1
                second_price = float(close_values[-1])
            gross_return = second_price / first_price - 1.0
        else:
            gross_return = 0.0
    elif direction == "sell_old_then_buy_back":
        for index in range(len(close_values) - 1):
            if close_values[index] >= day_open * (1.0 + trigger):
                trigger_signal = index
                first_fill = index + 1
                first_price = float(open_values[first_fill])
                break
        if first_fill >= 0:
            for index in range(first_fill, len(close_values) - 1):
                if close_values[index] <= first_price * (1.0 - paired_move):
                    second_fill = index + 1
                    second_price = float(open_values[second_fill])
                    paired_completed = True
                    break
            if second_fill < 0:
                second_fill = len(close_values) - 1
                second_price = float(close_values[-1])
            gross_return = first_price / second_price - 1.0
        else:
            gross_return = 0.0
    else:
        raise ValueError(f"unknown_t_direction:{direction}")
    triggered = first_fill >= 0
    net_return = gross_return - float(cost) if triggered else 0.0
    return {
        "triggered": triggered,
        "paired_completed": paired_completed,
        "trigger_signal_bar": trigger_signal,
        "first_fill_bar": first_fill,
        "second_fill_bar": second_fill,
        "first_fill_price": first_price,
        "second_fill_price": second_price,
        "gross_tranche_return": float(gross_return),
        "net_tranche_return": float(net_return),
        "position_return_contribution": float(tranche * net_return),
    }


def _t_oracle_day(
    opens: Sequence[float], *, cost: float, tranche: float
) -> dict[str, Any]:
    values = np.asarray(opens, dtype=float)
    best = 0.0
    for first in range(len(values) - 1):
        later = values[first + 1 :]
        best = max(
            best,
            float(np.max(later) / values[first] - 1.0),
            float(values[first] / np.min(later) - 1.0),
        )
    net = max(best - float(cost), 0.0)
    return {
        "triggered": net > 0.0,
        "paired_completed": True,
        "trigger_signal_bar": -1,
        "first_fill_bar": -1,
        "second_fill_bar": -1,
        "first_fill_price": math.nan,
        "second_fill_price": math.nan,
        "gross_tranche_return": best,
        "net_tranche_return": net,
        "position_return_contribution": float(tranche * net),
    }


def _t_results(
    trades: pd.DataFrame,
    days: pd.DataFrame,
    study: Mapping[str, Any],
) -> pd.DataFrame:
    metadata = trades[
        ["trade_id", "signal_date", "entry_date", "exit_date", "net_return_on_buy_cash"]
    ]
    frame = days.merge(metadata, on="trade_id", how="left", validate="many_to_one")
    cost = float(study["intraday_t"]["round_trip_cost"])
    tranche = float(study["intraday_t"]["t_tranche_fraction_of_position"])
    rules = [dict(rule) for rule in study["intraday_t"]["rules"]]
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        common = {
            "trade_id": int(row["trade_id"]),
            "symbol": str(row["symbol"]),
            "trade_date": str(row["trade_date"]),
            "evaluation_year": int(str(row["trade_date"])[:4]),
            "buy_notional_cny": float(row["buy_notional_cny"]),
            "holding_net_return": float(row["net_return_on_buy_cash"]),
        }
        for rule in rules:
            result = _t_rule_day(
                row["opens"], row["closes"], rule, cost=cost, tranche=tranche
            )
            records.append(
                {
                    **common,
                    "rule": str(rule["name"]),
                    "oracle": False,
                    **result,
                    "approximate_incremental_cny": float(
                        common["buy_notional_cny"]
                        * result["position_return_contribution"]
                    ),
                }
            )
        oracle = _t_oracle_day(row["opens"], cost=cost, tranche=tranche)
        records.append(
            {
                **common,
                "rule": "chronological_oracle",
                "oracle": True,
                **oracle,
                "approximate_incremental_cny": float(
                    common["buy_notional_cny"] * oracle["position_return_contribution"]
                ),
            }
        )
    return pd.DataFrame(records)


def _t_summaries(
    results: pd.DataFrame, study: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    daily = (
        results.groupby(["rule", "evaluation_year", "trade_date"], as_index=False)
        .agg(
            holding_count=("trade_id", "size"),
            trigger_fraction=("triggered", "mean"),
            mean_position_return_contribution=("position_return_contribution", "mean"),
            approximate_incremental_cny=("approximate_incremental_cny", "sum"),
        )
        .sort_values(["rule", "trade_date"], kind="stable")
    )
    summaries: list[dict[str, Any]] = []
    for period, period_frame in _evaluation_groups(daily, study):
        for rule, group in period_frame.groupby("rule", sort=True):
            detail = results.loc[
                results["rule"].eq(rule)
                & results["trade_date"].isin(group["trade_date"])
            ]
            triggered = detail.loc[detail["triggered"]]
            annual = group.groupby("evaluation_year", sort=True)[
                "mean_position_return_contribution"
            ].mean()
            summaries.append(
                {
                    "period": period,
                    "rule": str(rule),
                    "holding_day_count": len(detail),
                    "calendar_day_count": len(group),
                    "trigger_fraction": float(detail["triggered"].mean()),
                    "paired_completion_fraction_given_trigger": float(
                        triggered["paired_completed"].mean()
                    )
                    if len(triggered)
                    else math.nan,
                    "mean_gross_tranche_return_given_trigger": float(
                        triggered["gross_tranche_return"].mean()
                    )
                    if len(triggered)
                    else math.nan,
                    "mean_net_tranche_return_given_trigger": float(
                        triggered["net_tranche_return"].mean()
                    )
                    if len(triggered)
                    else math.nan,
                    "positive_triggered_fraction": float(
                        triggered["net_tranche_return"].gt(0.0).mean()
                    )
                    if len(triggered)
                    else math.nan,
                    "daily_equal_position_contribution": _inference(
                        group["mean_position_return_contribution"].to_numpy(),
                        study,
                        seed_add=5000 + len(summaries),
                    ),
                    "approximate_incremental_cny": float(
                        group["approximate_incremental_cny"].sum()
                    ),
                    "positive_year_count": int((annual > 0.0).sum()),
                    "year_count": len(annual),
                }
            )
    rule_decisions: list[dict[str, Any]] = []
    for rule in sorted(results.loc[~results["oracle"], "rule"].unique()):
        period_rows = [
            row
            for row in summaries
            if row["rule"] == rule and row["period"] != "full_history"
        ]
        full = next(
            row
            for row in summaries
            if row["rule"] == rule and row["period"] == "full_history"
        )
        every_period_lcb = len(period_rows) == 3 and all(
            float(row["daily_equal_position_contribution"]["lcb_95"]) > 0.0
            and float(row["daily_equal_position_contribution"]["block"]["lcb_95"]) > 0.0
            for row in period_rows
        )
        passed = bool(every_period_lcb and int(full["positive_year_count"]) >= 10)
        rule_decisions.append(
            {
                "rule": str(rule),
                "every_period_hac_and_block_lcb_positive": every_period_lcb,
                "positive_year_count": int(full["positive_year_count"]),
                "passed_stable_gate": passed,
            }
        )
    decision = {
        "stable_t_rule_found": any(row["passed_stable_gate"] for row in rule_decisions),
        "passing_rules": [
            row["rule"] for row in rule_decisions if row["passed_stable_gate"]
        ],
        "rules": rule_decisions,
        "account_replay_required_if_any_rule_passes": True,
        "buy_first_cash_feasibility_modeled": False,
    }
    return summaries, decision


def _entry_decision(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    variants = sorted(
        {str(row["variant"]) for row in summaries if str(row["variant"]) != "open_d1"}
    )
    candidates: list[dict[str, Any]] = []
    for variant in variants:
        full = next(
            row
            for row in summaries
            if row["variant"] == variant and row["period"] == "full_history"
        )
        periods = [
            row
            for row in summaries
            if row["variant"] == variant and row["period"] != "full_history"
        ]
        delta = full["paired_delta_vs_open_d1"]
        positive_periods = sum(
            float(row["paired_delta_vs_open_d1"]["mean"]) > 0.0 for row in periods
        )
        passed = bool(
            float(delta["lcb_95"]) > 0.0
            and float(delta["block"]["lcb_95"]) > 0.0
            and positive_periods >= 2
        )
        candidates.append(
            {
                "variant": variant,
                "positive_period_count": positive_periods,
                "passed": passed,
            }
        )
    return {
        "robust_entry_improvement_found": any(row["passed"] for row in candidates),
        "passing_variants": [row["variant"] for row in candidates if row["passed"]],
        "variants": candidates,
    }


def run_study(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    study, frozen_path = load_study(study_path)
    sources = _source_contract(study)
    root = base._resolve_path(output_root)
    summary_path = root / "summary.json"
    fingerprint = _payload_hash(
        {
            "study_sha256": base._sha256_file(frozen_path),
            "implementation_sha256": base._sha256_file(Path(__file__)),
            "source_hashes": {
                key: base._sha256_file(path) for key, path in sources.items()
            },
        }
    )
    if summary_path.is_file() and not force:
        current = _read_json(summary_path)
        if current.get("experiment_fingerprint") != fingerprint:
            raise ValueError("existing_output_fingerprint_mismatch")
        if all(
            base._record_valid(record, verify_hash=True)
            for record in current.get("files", {}).values()
        ):
            return current
    panel = base.load_panel(
        input_manifest_path=sources["input_manifest"],
        label_manifest_path=sources["label_manifest"],
    )
    if bool((panel.years == base.FORBIDDEN_YEAR).any()):
        raise ValueError("forbidden_2026_panel_row")
    selected = _selected_features(
        panel,
        feature_path=sources["candidate_features"],
        selection_path=sources["selections"],
        study=study,
    )
    if int(selected["evaluation_year"].max()) >= base.FORBIDDEN_YEAR:
        raise ValueError("forbidden_2026_selection")
    paths = _path_frame(panel, selected, study)
    enriched = _attach_signal_features(panel, paths, study)
    entry = _entry_results(panel, paths, study)
    entry_monthly = _entry_monthly(entry)
    entry_summaries = _entry_summaries(entry_monthly, study)
    exit_monthly = _exit_monthly(paths, study)
    exit_summaries, exit_decision = _exit_summaries(exit_monthly, study)
    checkpoint_summaries, shape_summaries = _path_summaries(paths, study)
    continuation, continuation_summaries = _continuation_states(paths, study)
    feature_deltas, feature_summaries = _feature_screen(enriched, study)
    trades = pd.read_parquet(sources["risk_budget_trades"]).reset_index(drop=True)
    trades.insert(0, "trade_id", np.arange(len(trades), dtype=np.int64))
    if bool(
        trades["exit_date"]
        .astype(str)
        .str[:4]
        .astype(int)
        .ge(base.FORBIDDEN_YEAR)
        .any()
    ):
        raise ValueError("forbidden_2026_account_trade")
    intraday_days, intraday_runtime = _complete_intraday_days(
        trades,
        manifest_path=sources["intraday_manifest"],
        maximum_date=str(study["source"]["maximum_outcome_date"]),
        temporary_root=root / "duckdb_tmp",
    )
    t_results = _t_results(trades, intraday_days, study)
    t_summaries, t_decision = _t_summaries(t_results, study)
    outputs = {
        "paths": (root / "selected_paths.parquet", paths),
        "entry_candidate_results": (root / "entry_candidate_results.parquet", entry),
        "entry_monthly_returns": (
            root / "entry_monthly_returns.parquet",
            entry_monthly,
        ),
        "exit_monthly_returns": (root / "exit_monthly_returns.parquet", exit_monthly),
        "continuation_states": (root / "continuation_states.parquet", continuation),
        "feature_date_deltas": (root / "feature_date_deltas.parquet", feature_deltas),
        "intraday_complete_days": (
            root / "intraday_complete_days.parquet",
            intraday_days,
        ),
        "intraday_t_results": (root / "intraday_t_results.parquet", t_results),
    }
    for path, frame in outputs.values():
        _write_parquet(path, frame)
    robust_features = [
        {
            "target": row["target"],
            "feature": row["feature"],
            "favored_half": row["favored_half"],
        }
        for row in feature_summaries
        if row["robust_diagnostic"]
    ]
    decision = {
        "entry": _entry_decision(entry_summaries),
        "signal_feature_screen": {
            "robust_diagnostic_found": bool(robust_features),
            "robust_features": robust_features,
            "data_mined_within_consumed_history": True,
        },
        "exit": exit_decision,
        "intraday_t": t_decision,
        "production_rule_change_authorized": False,
        "profit_claim_allowed": False,
    }
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "study_id": STUDY_ID,
        "experiment_fingerprint": fingerprint,
        "study_sha256": base._sha256_file(frozen_path),
        "implementation_sha256": base._sha256_file(Path(__file__)),
        "selected_candidate_count": len(selected),
        "selected_month_count": int(selected["trade_date"].nunique()),
        "path_d60_evaluable_count": int(paths["return_d60"].notna().sum()),
        "intraday_runtime": intraday_runtime,
        "path_checkpoint_summaries": checkpoint_summaries,
        "path_shape_summaries": shape_summaries,
        "entry_summaries": entry_summaries,
        "exit_summaries": exit_summaries,
        "continuation_summaries": continuation_summaries,
        "feature_summaries": feature_summaries,
        "intraday_t_summaries": t_summaries,
        "decision": decision,
        "forbidden_2026_read_count": 0,
        "retrospective_only": True,
        "files": {name: base._file_record(path) for name, (path, _) in outputs.items()},
    }
    base._write_json(summary_path, summary)
    return summary


def validate_summary(path: str | Path) -> dict[str, Any]:
    summary = _read_json(base._resolve_path(path))
    checks = {
        "schema": summary.get("schema") == SUMMARY_SCHEMA,
        "study": summary.get("study_id") == STUDY_ID,
        "completed": summary.get("status") == "completed",
        "retrospective": summary.get("retrospective_only") is True,
        "no_2026": int(summary.get("forbidden_2026_read_count", -1)) == 0,
        "files": all(
            base._record_valid(record, verify_hash=True)
            for record in summary.get("files", {}).values()
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"summary_validation_failed:{checks}")
    return {"status": "ok", "checks": checks}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_study(
        study_path=args.study,
        output_root=args.output_root,
        force=args.force,
    )
    print(json.dumps(summary["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
