from __future__ import annotations

"""Event study for observable hot-money and breakout/retest states.

This is deliberately a descriptive, execution-aware study rather than a
strategy optimizer.  A signal is known at the close of day ``t`` and can be
entered at the next observed open.  All rolling levels use observations at or
before ``t``; future highs/lows are used only as outcome labels.
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import duckdb
import numpy as np
import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_hot_money_event_study_v1"
DEFAULT_STUDY_PATH = WORKSPACE_ROOT / "daily_research/studies/seq100_hot_money_event_study_v1.json"
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "daily_research/research_records/seq100/seq100_hot_money_event_study_v1_20260806"
DEFAULT_QDP_ROOT = WORKSPACE_ROOT / "quant_data_platform/data/qdp_v2"
FORBIDDEN_YEAR = 2026
MAX_OUTCOME_DATE = "2025-12-31"
HORIZONS = (1, 3, 5, 10, 20)
# A-share T+1 means a position bought at the next observed open cannot be
# sold on that same observed day.  H1 remains in the schema as a diagnostic
# horizon, but it is never treated as an executable terminal return.
LEGAL_HORIZONS = (3, 5, 10, 20)
EVENT_COLUMNS = (
    "participation_shock",
    "positive_participation_shock",
    "range_expansion",
    "breakout20",
    "breakout60",
    "breakout_confirmed",
    "retest20",
    "rebreakout20",
    "climax_fade",
)
PERIODS = {
    "development_2012_2020": (2012, 2020),
    "validation_2021_2022": (2021, 2022),
    "oos_2023_2025": (2023, 2025),
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported_json_value:{type(value).__name__}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (WORKSPACE_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _load_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _active_paths(root: Path, active: Mapping[str, Any], domain: str, *, end_year: int) -> list[Path]:
    dataset_id = str(dict(active.get("datasets", {}) or {}).get(domain, "") or "")
    if not dataset_id:
        raise KeyError(f"active_dataset_missing:{domain}")
    manifest_path = root / "datasets" / domain / dataset_id / "dataset.json"
    manifest = _load_json(manifest_path)
    paths: list[Path] = []
    for shard in list(manifest.get("shards", []) or []):
        if str(shard.get("status", "stored")) not in {"stored", "completed"}:
            continue
        start = str(shard.get("start_date", ""))[:4]
        if start and start.isdigit() and int(start) > int(end_year):
            continue
        path = Path(str(shard.get("path", "") or ""))
        if not path.is_absolute():
            path = root / path
        if path.is_file():
            paths.append(path.resolve())
    if not paths:
        raise FileNotFoundError(f"no_active_shards:{domain}")
    return paths


def _parquet_list(paths: Iterable[Path]) -> str:
    escaped = ["'" + str(path).replace("'", "''") + "'" for path in paths]
    return "[" + ",".join(escaped) + "]"


def _connect(output_root: Path, memory_limit: str = "512MB") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA threads=2")
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    temp_dir = output_root / "duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute("PRAGMA temp_directory=?", [str(temp_dir)])
    return con


def _future_columns() -> str:
    expressions: list[str] = [
        "lead(open, 1) OVER w AS entry_open_next",
        "lead(trade_date, 1) OVER w AS entry_date_next",
        "lead(high, 1) OVER w AS entry_high_next",
        "lead(low, 1) OVER w AS entry_low_next",
    ]
    for horizon in HORIZONS:
        expressions.extend(
            [
                f"lead(close, {horizon}) OVER w AS exit_close_{horizon}",
                f"lead(trade_date, {horizon}) OVER w AS exit_date_{horizon}",
                f"max(high) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 1 FOLLOWING AND {horizon} FOLLOWING) AS future_high_{horizon}",
                f"min(low) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 1 FOLLOWING AND {horizon} FOLLOWING) AS future_low_{horizon}",
            ]
        )
        if horizon >= 2:
            expressions.extend(
                [
                    f"max(high) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 2 FOLLOWING AND {horizon} FOLLOWING) AS legal_future_high_{horizon}",
                    f"min(low) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 2 FOLLOWING AND {horizon} FOLLOWING) AS legal_future_low_{horizon}",
                ]
            )
        else:
            expressions.extend(
                [
                    "CAST(NULL AS DOUBLE) AS legal_future_high_1",
                    "CAST(NULL AS DOUBLE) AS legal_future_low_1",
                ]
            )
    return ",\n        ".join(expressions)


def _daily_event_query(daily_paths: list[Path], study: Mapping[str, Any], *, candidate_only: bool = True) -> str:
    source = _parquet_list(daily_paths)
    period = dict(study.get("period", {}) or {})
    signal_start = str(period.get("signal_start", "2012-01-01"))
    signal_end = str(period.get("signal_end", MAX_OUTCOME_DATE))
    input_start = str(period.get("input_start", "2011-11-22"))
    input_end = str(period.get("input_end", MAX_OUTCOME_DATE))
    thresholds = dict(study.get("event_thresholds", {}) or {})
    amount_shock = float(thresholds.get("amount_ratio20", 2.0))
    volume_shock = float(thresholds.get("volume_ratio20", 1.5))
    range_shock = float(thresholds.get("range_ratio20", 1.5))
    breakout_buffer = float(thresholds.get("breakout_buffer", 0.001))
    retest_upper = float(thresholds.get("retest_upper", 0.010))
    retest_lower = float(thresholds.get("retest_lower", 0.005))
    retest_volume_max = float(thresholds.get("retest_amount_ratio_max", 1.20))
    selection_filter = "AND event_any" if candidate_only else ""
    return f"""
WITH raw AS (
    SELECT symbol, trade_date, open, high, low, close, volume, amount
    FROM read_parquet({source})
    WHERE trade_date >= '{input_start}' AND trade_date <= '{input_end}'
), ordered AS (
    SELECT *,
        lag(close, 1) OVER w AS prev_close,
        lag(close, 3) OVER w AS close_lag3,
        lag(close, 5) OVER w AS close_lag5,
        lag(close, 10) OVER w AS close_lag10,
        lag(close, 20) OVER w AS close_lag20,
        lag(close, 60) OVER w AS close_lag60
    FROM raw
    WINDOW w AS (PARTITION BY symbol ORDER BY trade_date)
), returns AS (
    SELECT *,
        close / NULLIF(prev_close, 0) - 1.0 AS ret_1d,
        close / NULLIF(close_lag3, 0) - 1.0 AS ret_3d,
        close / NULLIF(close_lag5, 0) - 1.0 AS ret_5d,
        close / NULLIF(close_lag10, 0) - 1.0 AS ret_10d,
        close / NULLIF(close_lag20, 0) - 1.0 AS ret_20d,
        (high / NULLIF(low, 0) - 1.0) AS range_1d,
        (close - low) / NULLIF(high - low, 0) AS close_location_1d,
        (close - open) / NULLIF(high - low, 0) AS body_location_1d
    FROM ordered
), rolling AS (
    SELECT *,
        median(amount) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS amount_median20_prev,
        median(volume) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS volume_median20_prev,
        median(range_1d) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS range_median20_prev,
        stddev_samp(ret_1d) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS volatility20_prev,
        stddev_samp(ret_1d) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS volatility5_prev,
        max(high) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS prev20_high,
        max(high) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING) AS prev60_high,
        min(low) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS prev20_low,
        avg(close) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS ma20_prev,
        avg(close) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING) AS ma60_prev
    FROM returns
), break_state AS (
    SELECT *,
        close > prev20_high * (1.0 + {breakout_buffer}) AS breakout20,
        close > prev60_high * (1.0 + {breakout_buffer}) AS breakout60,
        amount / NULLIF(amount_median20_prev, 0) AS amount_ratio20,
        volume / NULLIF(volume_median20_prev, 0) AS volume_ratio20,
        range_1d / NULLIF(range_median20_prev, 0) AS range_ratio20,
        volatility5_prev / NULLIF(volatility20_prev, 0) AS volatility_ratio5_20
    FROM rolling
), state_one AS (
    SELECT *,
        max(CASE WHEN breakout20 THEN 1 ELSE 0 END) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS prior_breakout5
    FROM break_state
), state_two AS (
    SELECT *,
        (prior_breakout5 = 1
         AND low <= prev20_high * (1.0 + {retest_upper})
         AND close >= prev20_high * (1.0 - {retest_lower})
         AND amount_ratio20 <= {retest_volume_max}) AS retest20
    FROM state_one
), state_three AS (
    SELECT *,
        max(CASE WHEN retest20 THEN 1 ELSE 0 END) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS prior_retest5
    FROM state_two
), future AS (
    SELECT *,
        {_future_columns()}
    FROM state_three
    WINDOW w AS (PARTITION BY symbol ORDER BY trade_date)
), market AS (
    SELECT trade_date,
        avg(ret_1d) AS market_equal_weight_ret,
        avg(CASE WHEN ret_1d > 0 THEN 1.0 ELSE 0.0 END) AS market_breadth_up,
        count(*) AS market_stock_count
    FROM future
    WHERE ret_1d IS NOT NULL
    GROUP BY trade_date
), labelled AS (
    SELECT f.*, m.market_equal_weight_ret, m.market_breadth_up, m.market_stock_count,
        date_diff('day', CAST(f.trade_date AS DATE), CAST(f.entry_date_next AS DATE)) AS entry_calendar_gap,
        (f.close > f.prev20_high * (1.0 + {breakout_buffer}) AND f.amount_ratio20 >= {amount_shock} AND f.volume_ratio20 >= {volume_shock} AND f.close_location_1d >= 0.65) AS breakout_confirmed,
        (f.close > f.prev20_high * (1.0 + {breakout_buffer}) AND f.amount_ratio20 >= {amount_shock} AND f.volume_ratio20 >= {volume_shock} AND f.close_location_1d >= 0.65) AS _dummy
    FROM future f LEFT JOIN market m USING (trade_date)
), flags AS (
    SELECT *,
        (amount_ratio20 >= {amount_shock} AND volume_ratio20 >= {volume_shock}) AS participation_shock,
        (amount_ratio20 >= {amount_shock} AND volume_ratio20 >= {volume_shock} AND ret_1d >= 0.02 AND close_location_1d >= 0.65) AS positive_participation_shock,
        (range_ratio20 >= {range_shock} AND ret_1d >= 0.015 AND close_location_1d >= 0.60) AS range_expansion,
        (close > prev60_high * (1.0 + {breakout_buffer})) AS breakout60_final,
        (prior_retest5 = 1 AND close > prev20_high * (1.0 + {breakout_buffer}) AND close_location_1d >= 0.60 AND amount_ratio20 >= 1.20) AS rebreakout20,
        (amount_ratio20 >= {amount_shock} AND volume_ratio20 >= {volume_shock} AND ret_1d >= 0.05 AND close_location_1d <= 0.45) AS climax_fade
    FROM labelled
), selected AS (
    SELECT *,
        (participation_shock OR positive_participation_shock OR range_expansion OR breakout20 OR breakout60_final OR retest20 OR rebreakout20 OR climax_fade) AS event_any
    FROM flags
)
SELECT symbol, trade_date, open, high, low, close, amount, volume,
    ret_1d, ret_3d, ret_5d, ret_10d, ret_20d, range_1d, close_location_1d, body_location_1d,
    amount_ratio20, volume_ratio20, range_ratio20, volatility_ratio5_20, volatility20_prev,
    prev20_high, prev60_high, prev20_low, ma20_prev, ma60_prev,
    market_equal_weight_ret, market_breadth_up, market_stock_count, entry_date_next, entry_open_next,
    entry_high_next, entry_low_next, entry_calendar_gap,
    participation_shock, positive_participation_shock, range_expansion, breakout20, breakout60_final AS breakout60,
    breakout_confirmed, retest20, rebreakout20, climax_fade, event_any,
    exit_close_1, exit_close_3, exit_close_5, exit_close_10, exit_close_20,
    future_high_1, future_high_3, future_high_5, future_high_10, future_high_20,
    future_low_1, future_low_3, future_low_5, future_low_10, future_low_20,
    legal_future_high_1, legal_future_high_3, legal_future_high_5, legal_future_high_10, legal_future_high_20,
    legal_future_low_1, legal_future_low_3, legal_future_low_5, legal_future_low_10, legal_future_low_20,
    exit_date_1, exit_date_3, exit_date_5, exit_date_10, exit_date_20
FROM selected
WHERE trade_date >= '{signal_start}' AND trade_date <= '{signal_end}'
  {selection_filter}
  AND entry_open_next IS NOT NULL
  AND entry_calendar_gap <= 10
"""


def _minute_query(minute_paths: list[Path], event_path: Path, year: int) -> str:
    source = _parquet_list(minute_paths)
    event_file = str(event_path.resolve()).replace("'", "''")
    return f"""
WITH keys AS (
    SELECT DISTINCT symbol, trade_date
    FROM read_parquet('{event_file}')
    WHERE substr(trade_date, 1, 4) = '{int(year):04d}'
), raw AS (
    SELECT r.symbol, r.trade_date, r.bar_time, r.open, r.high, r.low, r.close, r.volume, r.amount
    FROM read_parquet({source}) r
    INNER JOIN keys k ON k.symbol = r.symbol AND k.trade_date = r.trade_date
    WHERE r.trade_date >= '{int(year):04d}-01-01' AND r.trade_date <= '{int(year):04d}-12-31'
), ordered AS (
    SELECT *,
        lag(close) OVER (PARTITION BY symbol, trade_date ORDER BY bar_time) AS prev_bar_close,
        row_number() OVER (PARTITION BY symbol, trade_date ORDER BY bar_time) AS rn,
        count(*) OVER (PARTITION BY symbol, trade_date) AS nbar
    FROM raw
), totals AS (
    SELECT symbol, trade_date, sum(amount) AS day_amount, sum(volume) AS day_volume,
        min_by(open, bar_time) AS first_open, max_by(close, bar_time) AS last_close,
        max(high) AS day_high, min(low) AS day_low
    FROM ordered GROUP BY symbol, trade_date
), enriched AS (
    SELECT o.*, t.day_amount, t.day_volume, t.first_open, t.last_close, t.day_high, t.day_low
    FROM ordered o INNER JOIN totals t USING (symbol, trade_date)
), features AS (
    SELECT symbol, trade_date,
        count(*) AS minute_bar_count,
        first_open AS minute_first_open,
        last_close AS minute_last_close,
        (last_close / NULLIF(first_open, 0) - 1.0) AS minute_open_close_return,
        (day_high / NULLIF(day_low, 0) - 1.0) AS minute_range,
        (last_close - day_low) / NULLIF(day_high - day_low, 0) AS minute_close_location,
        sqrt(sum(CASE WHEN prev_bar_close > 0 AND close > 0 THEN pow(ln(close / prev_bar_close), 2) ELSE 0 END)) AS minute_realized_volatility,
        sqrt(sum(CASE WHEN prev_bar_close > 0 AND close > 0 AND close < prev_bar_close THEN pow(ln(close / prev_bar_close), 2) ELSE 0 END)) AS minute_downside_semivolatility,
        sqrt(sum(CASE WHEN prev_bar_close > 0 AND close > 0 AND close > prev_bar_close THEN pow(ln(close / prev_bar_close), 2) ELSE 0 END)) AS minute_upside_semivolatility,
        sum(CASE WHEN rn <= 6 THEN ln(close / NULLIF(open, 0)) ELSE 0 END) AS minute_first_30m_return,
        sum(CASE WHEN rn > nbar - 6 THEN ln(close / NULLIF(open, 0)) ELSE 0 END) AS minute_last_30m_return,
        sum(CASE WHEN rn <= 12 THEN amount ELSE 0 END) / NULLIF(day_amount, 0) AS minute_first_hour_amount_share,
        sum(CASE WHEN rn > nbar - 12 THEN amount ELSE 0 END) / NULLIF(day_amount, 0) AS minute_last_hour_amount_share,
        max(amount) / NULLIF(day_amount, 0) AS minute_max_bar_amount_share,
        sum(CASE WHEN close > open THEN 1.0 ELSE 0.0 END) / NULLIF(count(*), 0) AS minute_positive_bar_fraction,
        abs(ln(last_close / NULLIF(first_open, 0))) / NULLIF(sum(CASE
            WHEN rn = 1 AND open > 0 AND close > 0 THEN abs(ln(close / open))
            WHEN prev_bar_close > 0 AND close > 0 THEN abs(ln(close / prev_bar_close))
            ELSE 0 END), 0) AS minute_trend_efficiency,
        (sum(amount) / NULLIF(day_volume, 0)) AS minute_vwap,
        sum(CASE WHEN close > (day_amount / NULLIF(day_volume, 0)) THEN 1.0 ELSE 0.0 END) / NULLIF(count(*), 0) AS minute_price_above_vwap_share,
        max(high) / NULLIF(first_open, 0) - 1.0 AS minute_mfe_from_open,
        min(low) / NULLIF(first_open, 0) - 1.0 AS minute_mae_from_open,
        sum(pow(amount / NULLIF(day_amount, 0), 2)) AS minute_amount_concentration_hhi,
        -sum(CASE WHEN amount > 0 AND day_amount > 0 THEN (amount / day_amount) * ln(amount / day_amount) ELSE 0 END) AS minute_amount_entropy,
        (sum(CASE WHEN rn <= nbar / 2 THEN amount ELSE 0 END) - sum(CASE WHEN rn > nbar / 2 THEN amount ELSE 0 END)) / NULLIF(day_amount, 0) AS minute_volume_half_imbalance,
        avg(CASE WHEN high = day_high THEN rn / NULLIF(nbar, 0) ELSE NULL END) AS minute_high_time_fraction,
        avg(CASE WHEN low = day_low THEN rn / NULLIF(nbar, 0) ELSE NULL END) AS minute_low_time_fraction
    FROM enriched
    GROUP BY symbol, trade_date, first_open, last_close, day_high, day_low, day_amount, day_volume
)
SELECT *, minute_last_close / NULLIF(minute_vwap, 0) - 1.0 AS minute_vwap_close_deviation
FROM features
"""


def _copy_query(con: duckdb.DuckDBPyConnection, query: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    con.execute(f"COPY ({query}) TO '{str(temporary).replace(chr(39), chr(39) * 2)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    os.replace(temporary, path)


def _hac_se(values: pd.Series, lag: int = 10) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 2:
        return float("nan")
    x = x - x.mean()
    lag = min(int(lag), n - 1)
    gamma0 = float(np.dot(x, x) / n)
    variance = gamma0
    for j in range(1, lag + 1):
        gamma = float(np.dot(x[j:], x[:-j]) / n)
        variance += 2.0 * (1.0 - j / (lag + 1.0)) * gamma
    return math.sqrt(max(variance, 0.0) / n)


def _period(year: int) -> str:
    for name, (start, end) in PERIODS.items():
        if int(start) <= int(year) <= int(end):
            return name
    return "outside"


def _return_column(horizon: int) -> str:
    return f"gross_return_{horizon}"


def _prepare_analysis(frame: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    frame = frame.copy()
    frame["year"] = frame["trade_date"].astype(str).str.slice(0, 4).astype(int)
    frame["period"] = frame["year"].map(_period)
    entry = frame["entry_open_next"].astype(float)
    if "entry_high_next" in frame:
        frame["entry_day_mfe"] = frame["entry_high_next"].astype(float).div(entry).sub(1.0)
    if "entry_low_next" in frame:
        frame["entry_day_mae"] = frame["entry_low_next"].astype(float).div(entry).sub(1.0)
    for horizon in HORIZONS:
        # The signal-to-entry-day close is a shadow diagnostic only.  It is
        # not a legal sell for a new long position under T+1.
        frame[f"shadow_return_{horizon}"] = frame[f"exit_close_{horizon}"].astype(float).div(entry).sub(1.0)
        if horizon == 1:
            frame[f"gross_return_{horizon}"] = np.nan
            frame[f"mfe_{horizon}"] = np.nan
            frame[f"mae_{horizon}"] = frame.get("entry_day_mae", pd.Series(np.nan, index=frame.index)).astype(float)
        else:
            frame[f"gross_return_{horizon}"] = frame[f"exit_close_{horizon}"].astype(float).div(entry).sub(1.0)
            legal_high = frame.get(f"legal_future_high_{horizon}", pd.Series(np.nan, index=frame.index))
            frame[f"mfe_{horizon}"] = legal_high.astype(float).div(entry).sub(1.0)
            # MAE measures exposure and therefore includes the entry day even
            # though a T+1 sale cannot be sent until D2.
            frame[f"mae_{horizon}"] = frame[f"future_low_{horizon}"].astype(float).div(entry).sub(1.0)
        frame[f"net_return_{horizon}"] = frame[f"gross_return_{horizon}"] - float(cost_bps) / 10_000.0
    minute = frame.get("minute_bar_count", pd.Series(index=frame.index, dtype=float))
    frame["minute_covered"] = minute.fillna(0).astype(float) >= 30
    frame["minute_trend_confirmed"] = (
        frame.get("minute_trend_efficiency", pd.Series(np.nan, index=frame.index)).astype(float) >= 0.35
    ) & (
        frame.get("minute_close_location", pd.Series(np.nan, index=frame.index)).astype(float) >= 0.65
    ) & (
        frame.get("minute_positive_bar_fraction", pd.Series(np.nan, index=frame.index)).astype(float) >= 0.55
    )
    frame["minute_closing_pressure"] = (
        frame.get("minute_last_30m_return", pd.Series(np.nan, index=frame.index)).astype(float) >= 0.005
    ) & (
        frame.get("minute_last_hour_amount_share", pd.Series(np.nan, index=frame.index)).astype(float) >= 0.20
    )
    frame["minute_intraday_fade"] = (
        frame.get("minute_last_30m_return", pd.Series(np.nan, index=frame.index)).astype(float) <= -0.01
    ) | (
        frame.get("minute_vwap_close_deviation", pd.Series(np.nan, index=frame.index)).astype(float) <= -0.005
    )
    frame["hot_confirmed"] = frame["positive_participation_shock"].astype(bool) & frame["minute_trend_confirmed"]
    frame["breakout_confirmed_intraday"] = frame["breakout_confirmed"].astype(bool) & frame["minute_trend_confirmed"]
    frame["retest_confirmed"] = frame["retest20"].astype(bool) & frame["minute_closing_pressure"] & frame["minute_covered"]
    return frame


def _summary_rows(frame: pd.DataFrame, event_names: Iterable[str], *, cost_bps: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event_name in event_names:
        if event_name not in frame.columns:
            continue
        event_mask = frame[event_name].astype(bool)
        for period_name, group in frame.loc[event_mask].groupby("period", sort=False):
            if period_name == "outside":
                continue
            for horizon in HORIZONS:
                ret_col = f"gross_return_{horizon}"
                net_col = f"net_return_{horizon}"
                valid = group[[ret_col, net_col, f"mfe_{horizon}", f"mae_{horizon}"]].dropna()
                date_mean = group.dropna(subset=[ret_col]).groupby("trade_date", sort=True)[ret_col].mean()
                net_date_mean = group.dropna(subset=[net_col]).groupby("trade_date", sort=True)[net_col].mean()
                rows.append({
                    "event": event_name,
                    "period": period_name,
                    "horizon": int(horizon),
                    "rows": int(len(valid)),
                    "dates": int(len(date_mean)),
                    "symbols": int(group["symbol"].nunique()),
                    "mean_gross_return": float(valid[ret_col].mean()) if len(valid) else np.nan,
                    "median_gross_return": float(valid[ret_col].median()) if len(valid) else np.nan,
                    "mean_net_return": float(valid[net_col].mean()) if len(valid) else np.nan,
                    "positive_rate": float((valid[ret_col] > 0).mean()) if len(valid) else np.nan,
                    "net_positive_rate": float((valid[net_col] > 0).mean()) if len(valid) else np.nan,
                    "mean_mfe": float(valid[f"mfe_{horizon}"].mean()) if len(valid) else np.nan,
                    "median_mfe": float(valid[f"mfe_{horizon}"].median()) if len(valid) else np.nan,
                    "mean_mae": float(valid[f"mae_{horizon}"].mean()) if len(valid) else np.nan,
                    "median_mae": float(valid[f"mae_{horizon}"].median()) if len(valid) else np.nan,
                    "hac_se_gross_date_mean": _hac_se(date_mean),
                    "hac_lcb_gross_date_mean": float(date_mean.mean() - 1.96 * _hac_se(date_mean)) if len(date_mean) else np.nan,
                    "hac_se_net_date_mean": _hac_se(net_date_mean),
                    "hac_lcb_net_date_mean": float(net_date_mean.mean() - 1.96 * _hac_se(net_date_mean)) if len(net_date_mean) else np.nan,
                    "cost_bps_round_trip": float(cost_bps),
                })
    return pd.DataFrame(rows)


def _baseline_excess(frame: pd.DataFrame, baseline: pd.DataFrame, event_names: Iterable[str]) -> pd.DataFrame:
    if baseline.empty:
        return pd.DataFrame()
    base = baseline.copy()
    base["trade_date"] = base["trade_date"].astype(str)
    joined = frame.merge(base, on="trade_date", how="left", suffixes=("", "_baseline"))
    rows: list[dict[str, Any]] = []
    for event_name in event_names:
        if event_name not in joined.columns:
            continue
        g0 = joined.loc[joined[event_name].astype(bool)].copy()
        for period_name, group in g0.groupby("period", sort=False):
            if period_name == "outside":
                continue
            for horizon in HORIZONS:
                ret = f"gross_return_{horizon}"
                base_col = f"baseline_gross_return_{horizon}"
                if base_col not in group:
                    continue
                excess = group[ret] - group[base_col]
                date_mean = pd.DataFrame({"date": group["trade_date"], "x": excess}).dropna().groupby("date")["x"].mean()
                rows.append({
                    "event": event_name,
                    "period": period_name,
                    "horizon": int(horizon),
                    "mean_excess_vs_date_universe": float(excess.mean()),
                    "hac_se_excess_date_mean": _hac_se(date_mean),
                    "hac_lcb_excess_date_mean": float(date_mean.mean() - 1.96 * _hac_se(date_mean)) if len(date_mean) else np.nan,
                })
    return pd.DataFrame(rows)


def _bin_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    breakout = frame.loc[frame["breakout20"].astype(bool)].copy()
    if not breakout.empty:
        breakout["breakout_gap_fraction"] = (breakout["close"] - breakout["prev20_high"]) / breakout["close"].abs().replace(0, np.nan)
        breakout["breakout_gap_bin"] = pd.cut(breakout["breakout_gap_fraction"], [-np.inf, 0.0, 0.005, 0.015, np.inf], labels=["below_level", "0-0.5pct", "0.5-1.5pct", ">1.5pct"])
        outputs["breakout_location_bins"] = _bin_summary(breakout, "breakout_gap_bin")
    retest = frame.loc[frame["retest20"].astype(bool)].copy()
    if not retest.empty:
        retest["retest_depth"] = (retest["prev20_high"] - retest["low"]) / retest["close"].abs().replace(0, np.nan)
        retest["retest_depth_bin"] = pd.cut(retest["retest_depth"], [-np.inf, 0.0, 0.005, 0.015, np.inf], labels=["above_level", "0-0.5pct", "0.5-1.5pct", ">1.5pct"])
        outputs["retest_depth_bins"] = _bin_summary(retest, "retest_depth_bin")
    return outputs


def _bin_summary(frame: pd.DataFrame, bin_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (period_name, bucket), group in frame.groupby(["period", bin_col], observed=True, sort=False):
        row: dict[str, Any] = {"period": period_name, "bucket": str(bucket), "rows": int(len(group))}
        for horizon in HORIZONS:
            col = f"gross_return_{horizon}"
            row[f"mean_gross_return_{horizon}"] = float(group[col].mean()) if col in group else np.nan
            row[f"positive_rate_{horizon}"] = (
                float((group[col] > 0).mean())
                if col in group and horizon in LEGAL_HORIZONS
                else np.nan
            )
            row[f"mean_mfe_{horizon}"] = float(group[f"mfe_{horizon}"].mean())
            row[f"mean_mae_{horizon}"] = float(group[f"mae_{horizon}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _markdown(summary: Mapping[str, Any], event_summary: pd.DataFrame, excess: pd.DataFrame, bins: Mapping[str, pd.DataFrame]) -> str:
    lines = [
        "# Hot-Money and Breakout Event Study",
        "",
        "This is a pre-specified, descriptive study of observable participation shocks and price-location states. It does not identify proprietary main-player orders from OHLCV.",
        "",
        "## Mathematical object",
        "",
        "At signal close t, define an observable state X_t from scale-normalized turnover, range, recent resistance and intraday path statistics. The study estimates E[R_{t->t+h}|X_t in A] and P(MFE_h >= a, MAE_h >= -b | X_t in A), with entry at the next observed open. Date-equal means and HAC intervals treat a trading date as the sampling cluster.",
        "",
        "## Data and execution",
        "",
        f"- Signal window: {summary.get('signal_start')} to {summary.get('signal_end')}; outcomes stop at {MAX_OUTCOME_DATE}; 2026 is forbidden.",
        f"- Daily candidate rows: {summary.get('daily_event_rows')}; five-minute candidate rows with at least 30 bars: {summary.get('minute_covered_rows')}",
        f"- Round-trip cost assumption: {summary.get('cost_bps')} bp, subtracted from gross arithmetic return.",
        "- Entry uses the next observed open. D1 terminal return is a shadow diagnostic only because a new long cannot be sold on its entry day under T+1. Legal terminal returns are D3/D5/D10/D20; MFE uses sellable path days D2..DH, while MAE includes entry-day exposure. The table reports the calendar gap to the entry open so suspensions are visible rather than silently treated as normal fills.",
        "",
        "## Interpretation guardrails",
        "",
        "- A participation shock is an abnormal amount/volume observation relative to the stock's preceding 20 observations. It is a proxy for attention or trading pressure, not proof of institutional or游资 ownership.",
        "- Breakout and retest levels are computed only from prior highs/lows. Future extrema appear only in MFE/MAE labels.",
        "- Positive MFE is not a tradable profit claim: the path may reverse before an executable exit, and high volatility mechanically raises MFE.",
        "- A first-hit continuation label also starts at path day 2. Reaching a target on the entry day is not counted as a legal sale under T+1.",
        "",
        "## Main results",
        "",
    ]
    if event_summary.empty:
        lines.append("No valid events were produced.")
    else:
        focus = event_summary[(event_summary["horizon"] == 5) & (event_summary["period"].isin(PERIODS))].copy()
        cols = ["event", "period", "rows", "mean_gross_return", "mean_net_return", "positive_rate", "mean_mfe", "mean_mae", "hac_lcb_net_date_mean"]
        lines.append(focus[cols].to_markdown(index=False, floatfmt=".4f"))
    if not excess.empty:
        lines.extend(["", "## Excess versus same-date universe", "", excess[excess["horizon"].isin([3, 5, 20])].to_markdown(index=False, floatfmt=".4f")])
    for name, table in bins.items():
        if table.empty:
            continue
        lines.extend(["", f"## {name}", "", table.to_markdown(index=False, floatfmt=".4f")])
    lines.extend([
        "",
        "## Decision rule for the next phase",
        "",
        "The event table can establish a conditional distribution and a risk veto, but it cannot by itself prove a profitable long-only policy. A candidate rule is admissible only if its OOS net-return HAC lower bound is positive, its result survives year-by-year checks, and entry gaps/turnover are executable. Otherwise the event remains a descriptive state for a later hurdle-plus-ranking model.",
    ])
    return "\n".join(lines) + "\n"


def run_study(*, study_path: Path = DEFAULT_STUDY_PATH, qdp_root: Path = DEFAULT_QDP_ROOT, output_root: Path = DEFAULT_OUTPUT_ROOT, force: bool = False) -> dict[str, Any]:
    study = _load_json(study_path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("study_id_mismatch")
    if int(study.get("source", {}).get("forbidden_year", -1)) != FORBIDDEN_YEAR:
        raise ValueError("forbidden_year_contract_missing")
    output_root = _resolve(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    if summary_path.is_file() and not force:
        return _load_json(summary_path)
    active = _load_json(_resolve(qdp_root) / "active/active.json")
    daily_paths = _active_paths(_resolve(qdp_root), active, "market_daily_raw", end_year=2025)
    minute_paths = _active_paths(_resolve(qdp_root), active, "market_intraday_5m", end_year=2025)
    con = _connect(output_root)
    daily_base_path = output_root / "daily_base.parquet"
    if force or not daily_base_path.is_file():
        source = _parquet_list(daily_paths)
        _copy_query(
            con,
            f"SELECT symbol, trade_date, open, high, low, close, volume, amount FROM read_parquet({source}) WHERE trade_date >= '2011-11-22' AND trade_date <= '{MAX_OUTCOME_DATE}'",
            daily_base_path,
        )
    daily_year_files: list[Path] = []
    baseline_year_files: list[Path] = []
    for year in range(2012, 2026):
        year_study = json.loads(json.dumps(study))
        year_study["period"]["signal_start"] = f"{year}-01-01"
        year_study["period"]["signal_end"] = f"{year}-12-31"
        year_study["period"]["input_start"] = "2011-11-22" if year == 2012 else f"{year - 1}-01-01"
        year_study["period"]["input_end"] = MAX_OUTCOME_DATE if year == 2025 else f"{year + 1}-03-31"
        out = output_root / f"daily_events_{year}.parquet"
        _copy_query(con, _daily_event_query([daily_base_path], year_study), out)
        daily_year_files.append(out)
        baseline_out = output_root / f"same_date_universe_baseline_{year}.parquet"
        baseline_query = _daily_event_query([daily_base_path], year_study, candidate_only=False)
        baseline_select = [
            "trade_date",
            "count(*) AS baseline_count",
        ]
        for horizon in HORIZONS:
            if horizon in LEGAL_HORIZONS:
                baseline_select.append(f"avg(exit_close_{horizon} / NULLIF(entry_open_next, 0) - 1.0) AS baseline_gross_return_{horizon}")
            else:
                baseline_select.append(f"CAST(NULL AS DOUBLE) AS baseline_gross_return_{horizon}")
            baseline_select.append(f"avg(future_high_{horizon} / NULLIF(entry_open_next, 0) - 1.0) AS baseline_mfe_{horizon}")
            baseline_select.append(f"avg(future_low_{horizon} / NULLIF(entry_open_next, 0) - 1.0) AS baseline_mae_{horizon}")
        _copy_query(con, f"SELECT {', '.join(baseline_select)} FROM ({baseline_query}) q GROUP BY trade_date", baseline_out)
        baseline_year_files.append(baseline_out)
    daily_event_path = output_root / "daily_events.parquet"
    year_list = _parquet_list(daily_year_files)
    _copy_query(con, f"SELECT * FROM read_parquet({year_list})", daily_event_path)
    baseline_path = output_root / "same_date_universe_baseline.parquet"
    baseline_list = _parquet_list(baseline_year_files)
    _copy_query(con, f"SELECT * FROM read_parquet({baseline_list})", baseline_path)
    daily_count = int(con.execute(f"SELECT count(*) FROM read_parquet('{str(daily_event_path).replace(chr(39), chr(39)*2)}')").fetchone()[0])
    minute_paths_by_year: dict[int, list[Path]] = {}
    manifest_path = _resolve(qdp_root) / "datasets/market_intraday_5m" / str(active["datasets"]["market_intraday_5m"]) / "dataset.json"
    minute_manifest = _load_json(manifest_path)
    for year in range(2012, 2026):
        paths: list[Path] = []
        for shard in list(minute_manifest.get("shards", []) or []):
            if str(shard.get("status", "stored")) not in {"stored", "completed"}:
                continue
            start, end = str(shard.get("start_date", "")), str(shard.get("end_date", ""))
            if start and start > f"{year}-12-31":
                continue
            if end and end < f"{year}-01-01":
                continue
            path = Path(str(shard.get("path", "") or ""))
            if not path.is_absolute():
                path = _resolve(qdp_root) / path
            if path.is_file():
                paths.append(path.resolve())
        minute_paths_by_year[year] = paths
    minute_files: list[Path] = []
    for year, paths in minute_paths_by_year.items():
        if not paths:
            continue
        out = output_root / f"minute_candidate_features_{year}.parquet"
        year_event_path = output_root / f"daily_events_{year}.parquet"
        _copy_query(con, _minute_query(paths, year_event_path, year), out)
        minute_files.append(out)
    con.close()
    daily = pd.read_parquet(daily_event_path)
    if minute_files:
        minute = pd.concat([pd.read_parquet(path) for path in minute_files], ignore_index=True)
        daily = daily.merge(minute, on=["symbol", "trade_date"], how="left", validate="one_to_one")
    daily = _prepare_analysis(daily, float(study.get("execution", {}).get("round_trip_cost_bps", 60.0)))
    baseline = pd.read_parquet(baseline_path)
    event_names = list(EVENT_COLUMNS) + ["minute_trend_confirmed", "minute_closing_pressure", "minute_intraday_fade", "hot_confirmed", "breakout_confirmed_intraday", "retest_confirmed"]
    event_summary = _summary_rows(daily, event_names, cost_bps=float(study.get("execution", {}).get("round_trip_cost_bps", 60.0)))
    excess = _baseline_excess(daily, baseline, event_names)
    bins = _bin_tables(daily)
    _write_csv(output_root / "event_summary.csv", event_summary)
    if not excess.empty:
        _write_csv(output_root / "event_excess.csv", excess)
    for name, table in bins.items():
        _write_csv(output_root / f"{name}.csv", table)
    minute_covered_rows = int(daily["minute_covered"].sum()) if "minute_covered" in daily else 0
    result: dict[str, Any] = {
        "study_id": STUDY_ID,
        "status": "completed",
        "signal_start": str(study.get("period", {}).get("signal_start", "2012-01-01")),
        "signal_end": str(study.get("period", {}).get("signal_end", MAX_OUTCOME_DATE)),
        "forbidden_year": FORBIDDEN_YEAR,
        "daily_event_rows": daily_count,
        "minute_covered_rows": minute_covered_rows,
        "minute_feature_files": [str(path.resolve()) for path in minute_files],
        "cost_bps": float(study.get("execution", {}).get("round_trip_cost_bps", 60.0)),
        "event_columns": event_names,
        "outputs": {
            "daily_events": str(daily_event_path.resolve()),
            "event_summary": str((output_root / "event_summary.csv").resolve()),
            "research_record": str((output_root / "research_record.md").resolve()),
        },
    }
    report = _markdown(result, event_summary, excess, bins)
    (output_root / "research_record.md").write_text(report, encoding="utf-8")
    _write_json(summary_path, result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an execution-aware hot-money event study.")
    parser.add_argument("--study-path", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--qdp-root", default=str(DEFAULT_QDP_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    result = run_study(
        study_path=_resolve(args.study_path),
        qdp_root=_resolve(args.qdp_root),
        output_root=_resolve(args.output_root),
        force=bool(args.force),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default) if args.json else result["outputs"]["research_record"])
    return result


if __name__ == "__main__":
    main()
