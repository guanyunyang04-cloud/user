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
from .contracts import EXPECTED_DECISION_BARS, MODEL_FEATURE_COLUMNS, MinuteV2Error
from .features import BAR_COLUMNS, STOCK_DAY_COLUMNS, build_feature_frame


def _literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _scan(path: Path) -> str:
    return f"read_parquet({_literal(path)})"


def _real_mutation_probe(connection: Any, base_path: Path) -> dict[str, Any]:
    scan = _scan(base_path)
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
    bars = frame.loc[:, list(BAR_COLUMNS)].copy()
    first = frame.drop_duplicates(["symbol", "trade_date"], keep="first").copy()
    previous_close = pd.to_numeric(first["close"], errors="coerce") / (
        1.0 + pd.to_numeric(first["return_from_previous_close"], errors="coerce")
    )
    previous_amount = np.exp(pd.to_numeric(first["previous_log_amount_20d"], errors="coerce"))
    auction_gap = pd.to_numeric(first["auction_gap"], errors="coerce")
    auction_ratio = pd.to_numeric(first["auction_amount_to_daily20"], errors="coerce")
    stock_days = pd.DataFrame(
        {
            "symbol": first["symbol"].astype(str),
            "trade_date": first["trade_date"].astype(str),
            "industry_name": first["industry_name"].astype(str),
            "adjust_factor": first["adjust_factor"],
            "previous_close": previous_close,
            "auction_price": previous_close * (1.0 + auction_gap),
            "auction_amount": previous_amount * auction_ratio,
            "previous_return_1d": first["previous_return_1d"],
            "previous_return_5d": first["previous_return_5d"],
            "previous_return_20d": first["previous_return_20d"],
            "previous_volatility_20d": first["previous_volatility_20d"],
            "previous_amount_20d": previous_amount,
            "daily_liquidity_rank": first["daily_liquidity_rank"],
            "exclude_open": ~first["valid_open"].astype(bool),
            "exclude_high": ~first["valid_high"].astype(bool),
            "exclude_low": ~first["valid_low"].astype(bool),
            "exclude_close": ~first["valid_close"].astype(bool),
        }
    ).loc[:, list(STOCK_DAY_COLUMNS)]
    original = build_feature_frame(
        bars,
        stock_days,
        expected_session_bars=EXPECTED_DECISION_BARS,
    )
    mutated_bars = bars.copy()
    future = mutated_bars["bar_time"].astype(str) > "100000000"
    mutated_bars.loc[future, ["open", "high", "low", "close"]] *= 1.7
    mutated_bars.loc[future, "volume"] *= 2.0
    mutated_bars.loc[future, "amount"] *= 3.4
    mutated = build_feature_frame(
        mutated_bars,
        stock_days,
        expected_session_bars=EXPECTED_DECISION_BARS,
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
                "count(*) FILTER(WHERE event_periodic),"
                "count(*) FILTER(WHERE event_extreme_return),"
                "count(*) FILTER(WHERE event_volume_shock),"
                "count(*) FILTER(WHERE event_industry_leadership),"
                "count(*) FILTER(WHERE event_vwap_cross),"
                "count(*) FILTER(WHERE event_random_negative) "
                f"FROM {_scan(event_path)}"
            ).fetchone()
            label_row = connection.execute(
                "SELECT count(*),count(*) FILTER(WHERE entry_executable),"
                "count(*) FILTER(WHERE label_observed),"
                "count(*) FILTER(WHERE delayed_exit_days=0),"
                "count(*) FILTER(WHERE delayed_exit_days BETWEEN 1 AND 5) "
                f"FROM {_scan(label_path)}"
            ).fetchone()
            mutation = _real_mutation_probe(connection, base_path)
        finally:
            connection.close()
    event_total = int(event_row[0])
    label_total = int(label_row[0])
    result = {
        "schema": "quantlab.minute_v2_pilot_audit/1",
        "status": "ok",
        "manifest": str(manifest_file),
        "scope": "data correctness only; no strategy score or return was inspected",
        "artifact_verification": verify,
        "base": {
            "row_count": total_rows,
            "feature_finite_fraction": finite_fraction,
            "minimum_feature_finite_fraction": min(finite_fraction.values()),
        },
        "events": {
            "row_count": event_total,
            "fraction_of_decision_rows": event_total / total_rows,
            "flag_counts": {
                "periodic": int(event_row[1]),
                "extreme_return": int(event_row[2]),
                "volume_shock": int(event_row[3]),
                "industry_leadership": int(event_row[4]),
                "vwap_cross": int(event_row[5]),
                "random_negative": int(event_row[6]),
            },
        },
        "labels": {
            "row_count": label_total,
            "entry_executable_fraction": int(label_row[1]) / label_total,
            "observed_fraction": int(label_row[2]) / label_total,
            "next_day_exit_fraction": int(label_row[3]) / label_total,
            "delayed_exit_fraction": int(label_row[4]) / label_total,
        },
        "future_mutation_probe": mutation,
    }
    if not all(math.isfinite(float(value)) for value in finite_fraction.values()):
        raise MinuteV2Error("minute_v2_pilot_finite_fraction_invalid")
    target = Path(output_path).resolve() if output_path is not None else month_directory / "pilot_audit.json"
    write_json(target, result)
    return result


__all__ = ["audit_pilot_month"]
