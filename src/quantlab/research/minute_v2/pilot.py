"""Data-only audit for the retained minute-v2 pilot month."""

from __future__ import annotations

import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from quantlab.core.io import read_json, write_json
from quantlab.data.qdp_v2.duckdb_resources import GIB, MIB, open_guarded_duckdb

from .builder import verify_month
from .contracts import (
    DAILY_WINDOWS,
    EXPECTED_DECISION_BARS,
    MODEL_FEATURE_COLUMNS,
    OPTIONAL_STORAGE_COLUMNS,
    SIXTY_MINUTE_WINDOWS,
    MinuteV2Error,
)
from .features import STOCK_DAY_COLUMNS, build_feature_frame
from .labels import DAILY_LABEL_HORIZONS, MINUTE_LABEL_HORIZONS
from .source import overlapping_minute_paths, resolve_source_snapshot


def _literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _scan(path: Path) -> str:
    return f"read_parquet({_literal(path)})"


def _real_mutation_probe(
    connection: Any,
    base_scan: str,
    *,
    workspace_root: Path,
) -> dict[str, Any]:
    scan = base_scan
    first_date = connection.execute(
        f"SELECT trade_date FROM {scan} GROUP BY trade_date ORDER BY trade_date LIMIT 1"
    ).fetchone()[0]
    if first_date is None:
        raise MinuteV2Error("minute_v2_pilot_base_empty")
    trade_date = first_date.isoformat() if hasattr(first_date, "isoformat") else str(first_date)
    symbols = [
        str(row[0])
        for row in connection.execute(
            "SELECT symbol FROM ("
            f"SELECT symbol,count(*) AS row_count,bool_and(valid_open) AND bool_and(valid_high) "
            f"AND bool_and(valid_low) AND bool_and(valid_close) AS fully_valid FROM {scan} "
            f"WHERE trade_date={_literal(trade_date)} GROUP BY symbol) "
            f"WHERE row_count={EXPECTED_DECISION_BARS} AND fully_valid ORDER BY symbol LIMIT 16"
        ).fetchall()
    ]
    if len(symbols) < 5:
        raise MinuteV2Error(f"minute_v2_pilot_mutation_support_too_small:{len(symbols)}")
    symbol_values = ",".join(_literal(value) for value in symbols)
    frame = connection.execute(
        f"SELECT * FROM {scan} WHERE trade_date={_literal(trade_date)} "
        f"AND symbol IN ({symbol_values}) ORDER BY symbol,bar_time"
    ).fetchdf()
    frame["trade_date"] = frame["trade_date"].astype(str).str[:10]
    frame["bar_time"] = frame["bar_time"].astype(str).str.zfill(9)
    snapshot = resolve_source_snapshot(workspace_root)
    calendar_frames = [
        pd.read_parquet(path, columns=["trade_date", "is_open"])
        for path in snapshot.shard_paths["trading_calendar"]
    ]
    calendar = pd.concat(calendar_frames, ignore_index=True)
    prior_dates = sorted(
        calendar.loc[
            calendar["is_open"].fillna(False)
            & calendar["trade_date"].astype(str).lt(trade_date),
            "trade_date",
        ]
        .astype(str)
        .unique()
        .tolist()
    )[-2:]
    raw_dates = [*prior_dates, trade_date]
    minute_paths = overlapping_minute_paths(
        snapshot,
        start_date=raw_dates[0],
        end_date=trade_date,
    )
    path_values = ",".join(_literal(path) for path in minute_paths)
    date_values = ",".join(_literal(value) for value in raw_dates)
    bars = connection.execute(
        "SELECT symbol,trade_date,bar_time,open,high,low,close,volume,amount "
        f"FROM read_parquet([{path_values}],union_by_name=true) "
        f"WHERE symbol IN ({symbol_values}) AND trade_date IN ({date_values}) "
        "ORDER BY symbol,trade_date,bar_time"
    ).fetchdf()
    first = frame.drop_duplicates(["symbol", "trade_date"], keep="first").copy()
    previous_close = pd.to_numeric(first["close"], errors="coerce") / (
        1.0 + pd.to_numeric(first["return_from_previous_close"], errors="coerce")
    )
    previous_amount = np.expm1(pd.to_numeric(first["previous_log_amount_20d"], errors="coerce"))
    auction_gap = pd.to_numeric(first["auction_gap"], errors="coerce")
    auction_ratio = pd.to_numeric(first["auction_amount_to_daily20"], errors="coerce")
    stock_day_values: dict[str, Any] = {
            "symbol": first["symbol"].astype(str),
            "trade_date": first["trade_date"].astype(str),
            "industry_name": first["industry_name"].astype(str),
            "adjust_factor": first["adjust_factor"],
            "previous_adjust_factor": first["adjust_factor"],
            "previous_close": previous_close,
            "auction_price": previous_close * (1.0 + auction_gap),
            "auction_amount": previous_amount * auction_ratio,
            "previous_return_1d": first["previous_return_1d"],
            "previous_amount_20d": previous_amount,
            "history_120d_available": first["history_120d_available"],
            "history_240d_available": first["history_240d_available"],
            "previous_total_share": 2_000_000_000.0,
            "previous_float_share": 1_000_000_000.0,
            "previous_total_mv": np.expm1(first["previous_log_total_market_value"]),
            "previous_circ_mv": np.expm1(first["previous_log_circulating_market_value"]),
            "previous_turnover_rate": first["previous_turnover_rate"],
            "previous_pe": first["previous_pe"],
            "previous_pb": first["previous_pb"],
            "corporate_action_today": first["corporate_action_today"],
            "cash_dividend_per_10": first["cash_dividend_per_10"],
            "bonus_share_per_10": first["bonus_share_per_10"],
            "transfer_share_per_10": first["transfer_share_per_10"],
            "daily_liquidity_rank": first["daily_liquidity_rank"],
            "exclude_open": ~first["valid_open"].astype(bool),
            "exclude_high": ~first["valid_high"].astype(bool),
            "exclude_low": ~first["valid_low"].astype(bool),
            "exclude_close": ~first["valid_close"].astype(bool),
        }
    for window in DAILY_WINDOWS:
        for prefix in (
            "previous_return",
            "previous_close_to_sma",
            "previous_volatility",
            "previous_amount_ratio",
        ):
            name = f"{prefix}_{window}d"
            stock_day_values[name] = first[name]
    stock_days = pd.DataFrame(stock_day_values).loc[:, list(STOCK_DAY_COLUMNS)]
    original = build_feature_frame(
        bars,
        stock_days,
        expected_session_bars=240,
    )
    mutated_bars = bars.copy()
    future = mutated_bars["trade_date"].astype(str).eq(trade_date) & mutated_bars[
        "bar_time"
    ].astype(str).gt("100000000")
    mutated_bars.loc[future, ["open", "high", "low", "close"]] *= 1.7
    mutated_bars.loc[future, "volume"] *= 2.0
    mutated_bars.loc[future, "amount"] *= 3.4
    mutated = build_feature_frame(
        mutated_bars,
        stock_days,
        expected_session_bars=240,
    )
    columns = ["symbol", "trade_date", "bar_time", *MODEL_FEATURE_COLUMNS]
    before = original.loc[original["bar_time"] <= "100000000", columns].reset_index(drop=True)
    after = mutated.loc[mutated["bar_time"] <= "100000000", columns].reset_index(drop=True)
    assert_frame_equal(before, after, check_exact=False, rtol=1.0e-12, atol=1.0e-14)
    before_values = before.loc[:, list(MODEL_FEATURE_COLUMNS)].to_numpy(dtype=float)
    after_values = after.loc[:, list(MODEL_FEATURE_COLUMNS)].to_numpy(dtype=float)
    finite = np.isfinite(before_values) & np.isfinite(after_values)
    maximum_absolute_difference = (
        float(np.max(np.abs(before_values[finite] - after_values[finite])))
        if finite.any()
        else 0.0
    )
    return {
        "status": "ok",
        "trade_date": trade_date,
        "symbol_count": len(symbols),
        "input_rows": int(len(bars)),
        "input_dates": raw_dates,
        "compared_rows": int(len(before)),
        "maximum_absolute_difference": maximum_absolute_difference,
        "numeric_tolerance": {"relative": 1.0e-12, "absolute": 1.0e-14},
        "mutation_boundary": "all raw bars after 10:00 were changed; features through 10:00 were invariant",
    }


def audit_pilot_month(
    manifest_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).resolve()
    manifest = read_json(manifest_file)
    verify = verify_month(manifest_file)
    artifacts = dict(manifest.get("artifacts", {}))
    if "base" not in artifacts:
        raise MinuteV2Error("minute_v2_pilot_audit_requires_retained_base")
    base_path = Path(str(artifacts["base"]["path"])).resolve()
    optional_path = (
        Path(str(artifacts["optional_features"]["path"])).resolve()
        if "optional_features" in artifacts
        else None
    )
    event_path = Path(str(artifacts["events"]["path"])).resolve()
    label_path = Path(str(artifacts["labels"]["path"])).resolve()
    month_directory = manifest_file.parent
    with TemporaryDirectory(prefix="_audit_", dir=month_directory) as temporary:
        connection = open_guarded_duckdb(
            ":memory:",
            temp_directory=temporary,
            threads=2,
            floor_bytes=4 * GIB,
            minimum_limit_bytes=256 * MIB,
        )
        try:
            base_scan = _scan(base_path)
            if optional_path is not None:
                optional_columns = [
                    name
                    for name in OPTIONAL_STORAGE_COLUMNS
                    if name not in {"symbol", "trade_date", "bar_time"}
                ]
                connection.execute(
                    "CREATE OR REPLACE TEMP VIEW base_with_optional AS SELECT b.*," 
                    + ",".join(f"o.{name}" for name in optional_columns)
                    + " FROM "
                    + _scan(base_path)
                    + " b JOIN "
                    + _scan(optional_path)
                    + " o USING(symbol,trade_date,bar_time)"
                )
                base_scan = "base_with_optional"
            finite_expressions = [
                f"count(*) FILTER(WHERE {name} IS NOT NULL AND isfinite(CAST({name} AS DOUBLE))) "
                f"AS {name}"
                for name in MODEL_FEATURE_COLUMNS
            ]
            finite_row = connection.execute(
                "SELECT count(*) AS total," + ",".join(finite_expressions) + f" FROM {base_scan}"
            ).fetchone()
            total_rows = int(finite_row[0])
            finite_fraction = {
                name: float(finite_row[index + 1]) / total_rows
                for index, name in enumerate(MODEL_FEATURE_COLUMNS)
            }
            event_row = connection.execute(
                "SELECT count(*),"
                "count(*) FILTER(WHERE event_extreme_return),"
                "count(*) FILTER(WHERE event_volume_shock),"
                "count(*) FILTER(WHERE event_industry_leadership),"
                "count(*) FILTER(WHERE event_vwap_cross),"
                "count(*) FILTER(WHERE event_liquidity_anchor),"
                "count(*) FILTER(WHERE event_background_control) "
                f"FROM {_scan(event_path)}"
            ).fetchone()
            group_row = connection.execute(
                "WITH base_groups AS (SELECT trade_date,bar_time,count(*) AS rows "
                f"FROM {base_scan} GROUP BY trade_date,bar_time),"
                "event_groups AS (SELECT trade_date,bar_time,count(*) AS rows "
                f"FROM {_scan(event_path)} GROUP BY trade_date,bar_time) "
                "SELECT count(*),count(e.rows),count(*) FILTER(WHERE e.rows IS NULL),"
                "min(e.rows),max(e.rows) FROM base_groups b LEFT JOIN event_groups e "
                "USING(trade_date,bar_time)"
            ).fetchone()
            label_row = connection.execute(
                "SELECT count(*),count(*) FILTER(WHERE entry_executable),"
                "count(*) FILTER(WHERE label_observed),"
                "count(*) FILTER(WHERE delayed_exit_days=0),"
                "count(*) FILTER(WHERE delayed_exit_days BETWEEN 1 AND 5) "
                f"FROM {_scan(label_path)}"
            ).fetchone()
            label_coverage_expressions = []
            for horizon in MINUTE_LABEL_HORIZONS:
                label_coverage_expressions.extend(
                    [
                        f"count(*) FILTER(WHERE label_{horizon}m_observed)",
                        f"count(label_return_{horizon}m)",
                        f"count(label_mfe_{horizon}m)",
                        f"count(label_mae_{horizon}m)",
                    ]
                )
            for horizon in MINUTE_LABEL_HORIZONS:
                label_coverage_expressions.extend(
                    [
                        f"count(*) FILTER(WHERE label_session_{horizon}m_observed)",
                        f"count(label_session_return_{horizon}m)",
                        f"count(label_session_mfe_{horizon}m)",
                        f"count(label_session_mae_{horizon}m)",
                    ]
                )
            for horizon in DAILY_LABEL_HORIZONS:
                label_coverage_expressions.extend(
                    [
                        f"count(*) FILTER(WHERE label_{horizon}d_observed)",
                        f"count(label_return_{horizon}d)",
                        f"count(label_mfe_{horizon}d)",
                        f"count(label_mae_{horizon}d)",
                    ]
                )
            label_coverage_row = connection.execute(
                "SELECT " + ",".join(label_coverage_expressions) + f" FROM {_scan(label_path)}"
            ).fetchone()
            high_mask_violation = " OR ".join(
                f"(l.label_mfe_{horizon}m IS NOT NULL "
                f"AND l.label_{horizon}m_mfe_invalid_reason <> '')"
                for horizon in MINUTE_LABEL_HORIZONS
            )
            low_mask_violation = " OR ".join(
                f"(l.label_mae_{horizon}m IS NOT NULL "
                f"AND l.label_{horizon}m_mae_invalid_reason <> '')"
                for horizon in MINUTE_LABEL_HORIZONS
            )
            close_mask_violation = " OR ".join(
                f"(l.label_return_{horizon}m IS NOT NULL "
                f"AND l.label_{horizon}m_invalid_reason = 'endpoint_close_excluded')"
                for horizon in MINUTE_LABEL_HORIZONS
            )
            quality_row = connection.execute(
                "SELECT "
                f"count(*) FILTER(WHERE {high_mask_violation}),"
                f"count(*) FILTER(WHERE {low_mask_violation}),"
                f"count(*) FILTER(WHERE {close_mask_violation}),"
                "count(*) FILTER(WHERE NOT e.valid_high AND e.valid_close "
                "AND l.entry_executable AND l.label_return_5m IS NOT NULL) "
                f"FROM {_scan(event_path)} e JOIN {_scan(label_path)} l "
                "USING(symbol,trade_date,bar_time)"
            ).fetchone()
            boundary_expressions = [
                "count(*) FILTER(WHERE minute_index % 60 <> 0 "
                "AND bar_time <> '130100000' "
                f"AND previous_m60_{window} IS NOT NULL "
                f"AND m60_close_to_sma_{window}bar IS DISTINCT FROM previous_m60_{window})"
                for window in SIXTY_MINUTE_WINDOWS
            ]
            boundary_projection = ",".join(
                f"LAG(m60_close_to_sma_{window}bar) OVER ("
                "PARTITION BY symbol,trade_date ORDER BY bar_time) "
                f"AS previous_m60_{window}"
                for window in SIXTY_MINUTE_WINDOWS
            )
            boundary_row = connection.execute(
                "SELECT "
                + ",".join(boundary_expressions)
                + " FROM (SELECT *,"
                + boundary_projection
                + f" FROM {base_scan})"
            ).fetchone()
            crossnight_row = connection.execute(
                "SELECT count(*) FILTER(WHERE bar_time='145500000'),"
                "count(*) FILTER(WHERE bar_time='145500000' AND label_5m_observed "
                "AND label_5m_crossed_overnight AND label_end_bar_time_5m='093500000'),"
                "count(*) FILTER(WHERE bar_time='145500000' "
                "AND label_session_5m_observed) "
                f"FROM {_scan(label_path)}"
            ).fetchone()
            mutation = _real_mutation_probe(
                connection,
                base_scan,
                workspace_root=Path(str(manifest["source"]["workspace_root"])),
            )
        finally:
            connection.close()
    event_total = int(event_row[0])
    label_total = int(label_row[0])
    label_coverage: dict[str, dict[str, float | int]] = {}
    coverage_index = 0
    for suffix, horizons in (
        ("continuous", MINUTE_LABEL_HORIZONS),
        ("session", MINUTE_LABEL_HORIZONS),
        ("holding", DAILY_LABEL_HORIZONS),
    ):
        unit = "m" if suffix in {"continuous", "session"} else "d"
        for horizon in horizons:
            observed, returns, mfe, mae = (
                int(label_coverage_row[coverage_index + offset]) for offset in range(4)
            )
            coverage_index += 4
            label_coverage[f"{suffix}_{horizon}{unit}"] = {
                "observed_rows": observed,
                "observed_fraction": observed / label_total,
                "return_rows": returns,
                "mfe_rows": mfe,
                "mae_rows": mae,
            }
    result = {
        "schema": "quantlab.minute_v2_pilot_audit/2",
        "status": "ok",
        "manifest": str(manifest_file),
        "scope": "data correctness only; no strategy score or return was inspected",
        "artifact_verification": verify,
        "base": {
            "row_count": total_rows,
            "feature_finite_fraction": finite_fraction,
            "minimum_feature_finite_fraction": min(finite_fraction.values()),
            "sixty_minute_non_boundary_change_violations": {
                f"{window}bar": int(boundary_row[index])
                for index, window in enumerate(SIXTY_MINUTE_WINDOWS)
            },
        },
        "events": {
            "row_count": event_total,
            "fraction_of_decision_rows": event_total / total_rows,
            "flag_counts": {
                "extreme_return": int(event_row[1]),
                "volume_shock": int(event_row[2]),
                "industry_leadership": int(event_row[3]),
                "vwap_cross": int(event_row[4]),
                "liquidity_anchor": int(event_row[5]),
                "background_control": int(event_row[6]),
            },
            "all_decision_minutes": {
                "base_group_count": int(group_row[0]),
                "candidate_group_count": int(group_row[1]),
                "missing_group_count": int(group_row[2]),
                "minimum_candidate_rows": int(group_row[3]),
                "maximum_candidate_rows": int(group_row[4]),
            },
        },
        "labels": {
            "row_count": label_total,
            "entry_executable_fraction": int(label_row[1]) / label_total,
            "observed_fraction": int(label_row[2]) / label_total,
            "next_day_exit_fraction": int(label_row[3]) / label_total,
            "delayed_exit_fraction": int(label_row[4]) / label_total,
            "horizon_coverage": label_coverage,
            "field_mask_violations": {
                "high_mask_with_mfe": int(quality_row[0]),
                "low_mask_with_mae": int(quality_row[1]),
                "close_mask_with_return": int(quality_row[2]),
            },
            "high_masked_but_return_still_observed_rows": int(quality_row[3]),
            "crossnight_1455_contract": {
                "candidate_rows": int(crossnight_row[0]),
                "continuous_5m_to_next_0935_rows": int(crossnight_row[1]),
                "same_session_5m_rows": int(crossnight_row[2]),
            },
        },
        "future_mutation_probe": mutation,
    }
    if not all(math.isfinite(float(value)) for value in finite_fraction.values()):
        raise MinuteV2Error("minute_v2_pilot_finite_fraction_invalid")
    if int(group_row[0]) != int(group_row[1]) or int(group_row[2]) != 0:
        raise MinuteV2Error("minute_v2_pilot_decision_minutes_missing")
    if any(int(value) != 0 for value in boundary_row):
        raise MinuteV2Error("minute_v2_pilot_sixty_minute_boundary_violation")
    if int(crossnight_row[1]) <= 0:
        raise MinuteV2Error("minute_v2_pilot_crossnight_1455_missing")
    resource_limit = int(manifest.get("effective_duckdb_memory_limit_bytes", 0))
    configured_value = manifest["config"].get("duckdb_memory_limit_gib", "auto")
    configured_limit = (
        None
        if isinstance(configured_value, str)
        and configured_value.strip().lower() == "auto"
        else int(float(configured_value) * GIB)
    )
    if resource_limit <= 0 or (
        configured_limit is not None and resource_limit > configured_limit
    ):
        raise MinuteV2Error("minute_v2_pilot_memory_limit_invalid")
    target = Path(output_path).resolve() if output_path is not None else month_directory / "pilot_audit.json"
    write_json(target, result)
    return result


__all__ = ["audit_pilot_month"]
