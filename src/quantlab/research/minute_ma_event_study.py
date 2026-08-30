"""Causal forward outcomes and compact statistics for minute-MA signals.

This module is intentionally an event study, not an inventory backtest. Entry
is the next available minute open after a signal, forward minute horizons use
future closes, and T+1 uses the first minute open of the next trading day.
Every result carries observed/usable flags so missing bars are not silently
treated as zero returns.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from quantlab.research.minute_ma import normalize_bar_time, session_minute_ordinal


class EventStudyError(ValueError):
    """Raised when event-study inputs violate the explicit contract."""


OUTCOME_COLUMNS = (
    "signal_id",
    "strategy_id",
    "strategy_family",
    "symbol",
    "signal_date",
    "signal_time",
    "sixty_minute_bucket",
    "ma_period",
    "event_trigger",
    "causal_only",
    "diagnostic_only",
    "signal_executable",
    "signal_adjusted_close",
    "entry_date",
    "entry_time",
    "entry_price",
    "entry_adjusted_price",
    "entry_observed",
    "entry_executable",
    "entry_reason",
    "entry_bar_index",
    "mfe_same_day",
    "mae_same_day",
    "same_day_observed",
    "t1_exit_date",
    "t1_exit_time",
    "t1_exit_adjusted_price",
    "t1_exit_observed",
    "t1_exit_reason",
    "t1_gross_return",
    "t1_net_return",
)


@dataclass(frozen=True)
class EventStudyConfig:
    """Fixed horizons and percentage cost assumptions for event comparisons."""

    minute_horizons: tuple[int, ...] = (5, 15, 30, 60)
    day_horizons: tuple[int, ...] = (1, 2, 3, 5)
    commission_bps: float = 3.0
    transfer_fee_bps: float = 0.1
    slippage_bps: float = 7.0
    stamp_tax_before_bps: float = 10.0
    stamp_tax_after_bps: float = 5.0
    stamp_tax_change_date: str = "2023-08-28"
    require_contiguous: bool = True

    def validate(self) -> None:
        for name, values in (("minute_horizons", self.minute_horizons), ("day_horizons", self.day_horizons)):
            if not values or tuple(sorted(set(values))) != values:
                raise EventStudyError(f"event_study_{name}_invalid")
            if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0 for value in values):
                raise EventStudyError(f"event_study_{name}_invalid")
        for name, value in (
            ("commission_bps", self.commission_bps),
            ("transfer_fee_bps", self.transfer_fee_bps),
            ("slippage_bps", self.slippage_bps),
            ("stamp_tax_before_bps", self.stamp_tax_before_bps),
            ("stamp_tax_after_bps", self.stamp_tax_after_bps),
        ):
            if not np.isfinite(float(value)) or float(value) < 0:
                raise EventStudyError(f"event_study_{name}_invalid")
        try:
            date.fromisoformat(self.stamp_tax_change_date)
        except ValueError as exc:
            raise EventStudyError("event_study_stamp_tax_change_date_invalid") from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "minute_horizons": list(self.minute_horizons),
            "day_horizons": list(self.day_horizons),
            "commission_bps": float(self.commission_bps),
            "transfer_fee_bps": float(self.transfer_fee_bps),
            "slippage_bps": float(self.slippage_bps),
            "stamp_tax_before_bps": float(self.stamp_tax_before_bps),
            "stamp_tax_after_bps": float(self.stamp_tax_after_bps),
            "stamp_tax_change_date": self.stamp_tax_change_date,
            "require_contiguous": bool(self.require_contiguous),
        }


_BAR_COLUMNS = {
    "symbol",
    "trade_date",
    "bar_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
}


def _require_columns(frame: pd.DataFrame, required: Iterable[str], *, prefix: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise EventStudyError(f"{prefix}_columns_missing:{','.join(missing)}")


def _normalise_date_series(values: pd.Series) -> pd.Series:
    raw = values.astype("string").str.strip()
    compact = raw.str.fullmatch(r"\d{8}", na=False)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if compact.any():
        parsed.loc[compact] = pd.to_datetime(raw.loc[compact], format="%Y%m%d", errors="coerce")
    if (~compact).any():
        parsed.loc[~compact] = pd.to_datetime(raw.loc[~compact], errors="coerce")
    if parsed.isna().any():
        bad = raw.loc[parsed.isna()].iloc[0]
        raise EventStudyError(f"event_study_trade_date_invalid:{bad!r}")
    return parsed.dt.strftime("%Y-%m-%d")


def _normalise_trading_dates(values: Sequence[str] | pd.Series) -> list[str]:
    """Normalise an explicit open-market calendar and reject duplicates."""

    series = values if isinstance(values, pd.Series) else pd.Series(list(values), dtype="object")
    if series.empty:
        raise EventStudyError("event_study_trading_dates_empty")
    normalised = _normalise_date_series(series)
    if normalised.duplicated().any():
        raise EventStudyError("event_study_trading_dates_duplicate")
    return sorted(normalised.astype(str).tolist())


def prepare_outcome_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Normalise raw minute bars while retaining incomplete days for outcomes."""

    _require_columns(bars, _BAR_COLUMNS, prefix="event_study_bar")
    frame = bars.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.strip()
    if frame["symbol"].eq("").any() or frame["symbol"].str.lower().isin({"nan", "none", "<na>"}).any():
        raise EventStudyError("event_study_symbol_invalid")
    frame["trade_date"] = _normalise_date_series(frame["trade_date"])
    frame["bar_time"] = frame["bar_time"].map(normalize_bar_time).astype("string")
    for column in ("open", "high", "low", "close", "volume", "amount"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "adjust_factor" not in frame:
        frame["adjust_factor"] = 1.0
    frame["adjust_factor"] = pd.to_numeric(frame["adjust_factor"], errors="coerce")
    finite = np.isfinite(frame.loc[:, ["open", "high", "low", "close", "volume", "amount", "adjust_factor"]]).all(axis=1)
    valid = (
        finite
        & frame.loc[:, ["open", "high", "low", "close"]].gt(0).all(axis=1)
        & frame.loc[:, ["volume", "amount"]].ge(0).all(axis=1)
        & frame["adjust_factor"].gt(0)
        & frame["high"].ge(frame["low"])
        & frame["high"].ge(frame[["open", "close"]].max(axis=1))
        & frame["low"].le(frame[["open", "close"]].min(axis=1))
    )
    if not bool(valid.all()):
        raise EventStudyError(f"event_study_invalid_bar_count:{int((~valid).sum())}")
    frame["session_minute_ordinal"] = frame["bar_time"].map(session_minute_ordinal)
    if frame["session_minute_ordinal"].isna().any():
        bad = frame.loc[frame["session_minute_ordinal"].isna(), "bar_time"].iloc[0]
        raise EventStudyError(f"event_study_out_of_session_bar:{bad}")
    frame["session_minute_ordinal"] = frame["session_minute_ordinal"].astype("int16")
    if frame.duplicated(["symbol", "trade_date", "bar_time"]).any():
        raise EventStudyError("event_study_duplicate_minute_key")
    factor_counts = frame.groupby(["symbol", "trade_date"], sort=False)["adjust_factor"].nunique()
    if factor_counts.gt(1).any():
        raise EventStudyError("event_study_intraday_adjust_factor_changed")
    for column in ("open", "high", "low", "close"):
        frame[f"adjusted_{column}"] = frame[column] * frame["adjust_factor"]
    frame.sort_values(["symbol", "trade_date", "session_minute_ordinal", "bar_time"], inplace=True, kind="stable", ignore_index=True)
    frame["bar_index_internal"] = np.arange(len(frame), dtype=np.int64)
    return frame


def _truthy(row: pd.Series, names: Sequence[str]) -> bool:
    for name in names:
        if name in row.index:
            value = row[name]
            if pd.notna(value) and bool(value):
                return True
    return False


def _bar_is_executable(row: pd.Series, *, side: str) -> tuple[bool, str]:
    if _truthy(row, ("is_suspended", "suspended", "exclude_open", "exclude_close")):
        return False, "suspended_or_quality_excluded"
    if side == "buy" and _truthy(row, ("is_limit_up", "limit_up", "buy_blocked")):
        return False, "buy_limit_blocked"
    if side == "sell" and _truthy(row, ("is_limit_down", "limit_down", "sell_blocked")):
        return False, "sell_limit_blocked"
    names = ("entry_executable", "executable") if side == "buy" else ("exit_executable", "executable")
    for name in names:
        if name in row.index and pd.notna(row[name]) and not bool(row[name]):
            return False, f"{name}_false"
    return True, "ok"


def _next_date_map(dates: Sequence[str]) -> dict[str, str]:
    ordered = sorted({str(value) for value in dates})
    return {current: following for current, following in zip(ordered, ordered[1:], strict=False)}


def _is_adjacent(frame: pd.DataFrame, left: int, right: int, next_dates: dict[str, str]) -> bool:
    first = frame.iloc[int(left)]
    second = frame.iloc[int(right)]
    if str(first["symbol"]) != str(second["symbol"]):
        return False
    if str(first["trade_date"]) == str(second["trade_date"]):
        return int(second["session_minute_ordinal"]) == int(first["session_minute_ordinal"]) + 1
    return (
        next_dates.get(str(first["trade_date"])) == str(second["trade_date"])
        and int(first["session_minute_ordinal"]) == 239
        and int(second["session_minute_ordinal"]) == 0
    )


def _stamp_tax_bps(exit_date: str, config: EventStudyConfig) -> float:
    return (
        float(config.stamp_tax_before_bps)
        if str(exit_date) < str(config.stamp_tax_change_date)
        else float(config.stamp_tax_after_bps)
    )


def _net_return(entry: float, exit_price: float, exit_date: str, config: EventStudyConfig) -> float:
    buy_rate = (float(config.commission_bps) + float(config.transfer_fee_bps)) / 10_000.0
    sell_rate = (
        float(config.commission_bps)
        + float(config.transfer_fee_bps)
        + _stamp_tax_bps(exit_date, config)
    ) / 10_000.0
    slip = float(config.slippage_bps) / 10_000.0
    entry_exec = float(entry) * (1.0 + slip)
    exit_exec = float(exit_price) * (1.0 - slip)
    return exit_exec * (1.0 - sell_rate) / (entry_exec * (1.0 + buy_rate)) - 1.0


def _gross_return(entry: float, exit_price: float) -> float:
    return float(exit_price) / float(entry) - 1.0


def _metric_names(config: EventStudyConfig) -> tuple[str, ...]:
    return (
        tuple(f"gross_return_{h}m" for h in config.minute_horizons)
        + tuple(f"net_return_{h}m" for h in config.minute_horizons)
        + tuple(f"gross_return_{h}d" for h in config.day_horizons)
        + tuple(f"net_return_{h}d" for h in config.day_horizons)
        + ("mfe_same_day", "mae_same_day", "t1_gross_return", "t1_net_return")
    )


def compute_event_outcomes(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    config: EventStudyConfig | None = None,
    trading_dates: Sequence[str] | pd.Series | None = None,
) -> pd.DataFrame:
    """Attach executable entry and causal forward outcomes to signal rows.

    ``trading_dates`` is the explicit open-market calendar used for cross-day
    adjacency and 1/2/3/5-day targets.  When omitted, the observed dates for
    each symbol are used for backwards-compatible small fixtures; callers
    evaluating real stocks should provide the market calendar so a suspended
    stock cannot be treated as if it had traded on a later day.
    """

    selected = config or EventStudyConfig()
    selected.validate()
    _require_columns(signals, ("signal_id", "strategy_id", "symbol", "signal_date", "signal_time"), prefix="event_study_signal")
    if signals.empty:
        return pd.DataFrame({name: pd.Series(dtype="object") for name in (*OUTCOME_COLUMNS, *_metric_names(selected))})
    signal_frame = signals.copy()
    signal_frame["signal_date"] = _normalise_date_series(signal_frame["signal_date"])
    signal_frame["symbol"] = signal_frame["symbol"].astype(str).str.strip()
    if signal_frame["signal_id"].duplicated().any():
        raise EventStudyError("event_study_signal_id_duplicate")
    frame = prepare_outcome_bars(bars)
    explicit_calendar = None if trading_dates is None else _normalise_trading_dates(trading_dates)
    explicit_next_dates = None if explicit_calendar is None else _next_date_map(explicit_calendar)
    by_symbol_date_time = {
        (str(row.symbol), str(row.trade_date), str(row.bar_time)): int(row.bar_index_internal)
        for row in frame.itertuples(index=False)
    }
    symbol_positions: dict[str, list[int]] = {}
    symbol_position_lookup: dict[str, dict[int, int]] = {}
    symbol_next_dates: dict[str, dict[str, str]] = {}
    adjacent_next: dict[int, bool] = {}
    day_stats: dict[tuple[str, str], dict[str, Any]] = {}
    suffix_high = np.full(len(frame), np.nan, dtype=np.float64)
    suffix_low = np.full(len(frame), np.nan, dtype=np.float64)
    daily_rows: list[dict[str, Any]] = []
    for symbol, group in frame.groupby("symbol", sort=False):
        positions = group["bar_index_internal"].astype(int).tolist()
        symbol_name = str(symbol)
        symbol_positions[symbol_name] = positions
        symbol_position_lookup[symbol_name] = {value: index for index, value in enumerate(positions)}
        symbol_next_dates[symbol_name] = explicit_next_dates or _next_date_map(group["trade_date"].astype(str).unique().tolist())
        adjacent_next.update(
            {
                left: _is_adjacent(frame, left, right, symbol_next_dates[symbol_name])
                for left, right in zip(positions, positions[1:], strict=False)
            }
        )
        for trade_date, day_group in group.groupby("trade_date", sort=False):
            day_indices = day_group["bar_index_internal"].astype(int).tolist()
            day_ordinals = set(day_group["session_minute_ordinal"].astype(int).tolist())
            day_high = day_group["adjusted_high"].to_numpy(dtype=np.float64)
            day_low = day_group["adjusted_low"].to_numpy(dtype=np.float64)
            suffix_high[day_indices] = np.maximum.accumulate(day_high[::-1])[::-1]
            suffix_low[day_indices] = np.minimum.accumulate(day_low[::-1])[::-1]
            day_adjacent = all(
                adjacent_next.get(left, False)
                for left in day_indices[:-1]
            )
            day_stats[(symbol_name, str(trade_date))] = {
                "adjacent": bool(day_adjacent),
                "complete": len(day_group) == 240 and day_ordinals == set(range(240)),
            }
            daily_rows.append(
                {
                    "symbol": symbol_name,
                    "trade_date": str(trade_date),
                    "first_bar_index": int(day_group["bar_index_internal"].min()),
                    "first_bar_time": str(day_group["bar_time"].iloc[0]),
                    "adjusted_open": float(day_group["adjusted_open"].iloc[0]),
                    "adjusted_high": float(day_group["adjusted_high"].max()),
                    "adjusted_low": float(day_group["adjusted_low"].min()),
                    "adjusted_close": float(day_group["adjusted_close"].iloc[-1]),
                }
            )
    daily = pd.DataFrame(daily_rows).sort_values(["symbol", "trade_date"], kind="stable", ignore_index=True)
    daily_lookup: dict[tuple[str, str], int] = {
        (str(row.symbol), str(row.trade_date)): int(index)
        for index, row in enumerate(daily.itertuples(index=False))
    }
    daily_by_symbol: dict[str, list[int]] = {}
    daily_position_lookup: dict[str, dict[int, int]] = {}
    for symbol, group in daily.groupby("symbol", sort=False):
        values = group.index.astype(int).tolist()
        daily_by_symbol[str(symbol)] = values
        daily_position_lookup[str(symbol)] = {value: index for index, value in enumerate(values)}

    metric_names = _metric_names(selected)
    output: list[dict[str, Any]] = []
    for signal in signal_frame.itertuples(index=False):
        signal_dict = signal._asdict()
        symbol = str(signal_dict["symbol"])
        signal_date = str(signal_dict["signal_date"])
        signal_time = normalize_bar_time(signal_dict["signal_time"])
        base: dict[str, Any] = {
            name: signal_dict.get(name)
            for name in OUTCOME_COLUMNS
            if name in signal_dict
        }
        base.update(
            {
                "signal_id": str(signal_dict["signal_id"]),
                "strategy_id": str(signal_dict["strategy_id"]),
                "strategy_family": signal_dict.get("strategy_family"),
                "symbol": symbol,
                "signal_date": signal_date,
                "signal_time": signal_time,
                "sixty_minute_bucket": signal_dict.get("sixty_minute_bucket"),
                "ma_period": signal_dict.get("ma_period"),
                "event_trigger": signal_dict.get("event_trigger"),
                "causal_only": bool(signal_dict.get("causal_only", True)),
                "diagnostic_only": bool(signal_dict.get("diagnostic_only", False)),
                "signal_executable": bool(signal_dict.get("signal_executable", True)),
                "signal_adjusted_close": signal_dict.get("signal_adjusted_close"),
                "entry_date": None,
                "entry_time": None,
                "entry_price": np.nan,
                "entry_adjusted_price": np.nan,
                "entry_observed": False,
                "entry_executable": False,
                "entry_reason": "signal_bar_missing",
                "entry_bar_index": None,
                "mfe_same_day": np.nan,
                "mae_same_day": np.nan,
                "same_day_observed": False,
                "t1_exit_date": None,
                "t1_exit_time": None,
                "t1_exit_adjusted_price": np.nan,
                "t1_exit_observed": False,
                "t1_exit_reason": "entry_unobserved",
                "t1_gross_return": np.nan,
                "t1_net_return": np.nan,
            }
        )
        for name in metric_names:
            base[name] = np.nan
        signal_index = by_symbol_date_time.get((symbol, signal_date, signal_time))
        if signal_index is None:
            output.append(base)
            continue
        positions = symbol_positions[symbol]
        local_position = symbol_position_lookup[symbol][signal_index]
        if local_position + 1 >= len(positions):
            base["entry_reason"] = "no_next_minute"
            output.append(base)
            continue
        entry_index = positions[local_position + 1]
        if selected.require_contiguous and not adjacent_next.get(signal_index, False):
            base["entry_reason"] = "next_minute_gap"
            output.append(base)
            continue
        entry_row = frame.iloc[entry_index]
        entry_ok, entry_reason = _bar_is_executable(entry_row, side="buy")
        base.update(
            {
                "entry_date": str(entry_row["trade_date"]),
                "entry_time": str(entry_row["bar_time"]),
                "entry_price": float(entry_row["open"]),
                "entry_adjusted_price": float(entry_row["adjusted_open"]),
                "entry_observed": True,
                "entry_executable": bool(entry_ok and base["signal_executable"]),
                "entry_reason": entry_reason if base["signal_executable"] else "signal_marked_non_executable",
                "entry_bar_index": int(entry_index),
            }
        )
        # Keep hypothetical diagnostics for non-causal signals, but never
        # score a bar that the execution contract marks as unfilled.
        if not entry_ok:
            output.append(base)
            continue
        entry_price = float(entry_row["adjusted_open"])
        entry_date = str(entry_row["trade_date"])
        stats = day_stats.get((symbol, entry_date))
        if stats is not None and stats["adjacent"] and stats["complete"]:
            base["mfe_same_day"] = _gross_return(entry_price, float(suffix_high[entry_index]))
            base["mae_same_day"] = _gross_return(entry_price, float(suffix_low[entry_index]))
            base["same_day_observed"] = True

        for horizon in selected.minute_horizons:
            target_local = local_position + 1 + int(horizon)
            gross_name = f"gross_return_{horizon}m"
            net_name = f"net_return_{horizon}m"
            if target_local >= len(positions):
                continue
            target_index = positions[target_local]
            if selected.require_contiguous:
                adjacent = all(
                    adjacent_next.get(positions[offset], False)
                    for offset in range(local_position + 1, target_local)
                )
                if not adjacent:
                    continue
            target_row = frame.iloc[target_index]
            target_price = float(target_row["adjusted_close"])
            target_date = str(target_row["trade_date"])
            base[gross_name] = _gross_return(entry_price, target_price)
            base[net_name] = _net_return(entry_price, target_price, target_date, selected)

        daily_index = daily_lookup.get((symbol, entry_date))
        symbol_daily = daily_by_symbol.get(symbol, [])
        if daily_index is not None and symbol_daily:
            daily_local = daily_position_lookup[symbol][daily_index]
            if explicit_calendar is None:
                target_dates = [
                    str(daily.iloc[symbol_daily[daily_local + int(horizon)]]['trade_date'])
                    if daily_local + int(horizon) < len(symbol_daily)
                    else None
                    for horizon in selected.day_horizons
                ]
                next_day = (
                    str(daily.iloc[symbol_daily[daily_local + 1]]["trade_date"])
                    if daily_local + 1 < len(symbol_daily)
                    else None
                )
            else:
                calendar_position = {value: index for index, value in enumerate(explicit_calendar)}
                calendar_local = calendar_position.get(entry_date)
                target_dates = (
                    [
                        explicit_calendar[calendar_local + int(horizon)]
                        if calendar_local is not None and calendar_local + int(horizon) < len(explicit_calendar)
                        else None
                        for horizon in selected.day_horizons
                    ]
                    if calendar_local is not None
                    else [None] * len(selected.day_horizons)
                )
                next_day = (
                    explicit_calendar[calendar_local + 1]
                    if calendar_local is not None and calendar_local + 1 < len(explicit_calendar)
                    else None
                )
            for horizon, target_date in zip(selected.day_horizons, target_dates, strict=True):
                if target_date is None:
                    continue
                target_index = daily_lookup.get((symbol, target_date))
                if target_index is None:
                    continue
                target_daily = daily.iloc[target_index]
                target_price = float(target_daily["adjusted_close"])
                base[f"gross_return_{horizon}d"] = _gross_return(entry_price, target_price)
                base[f"net_return_{horizon}d"] = _net_return(entry_price, target_price, target_date, selected)
            if next_day is not None:
                base["t1_exit_date"] = next_day
                t1_index = daily_lookup.get((symbol, next_day))
                if t1_index is None:
                    base["t1_exit_reason"] = "next_trading_day_bar_missing"
                else:
                    t1_daily = daily.iloc[t1_index]
                    sell_ok, sell_reason = _bar_is_executable(
                        frame.iloc[int(t1_daily["first_bar_index"])],
                        side="sell",
                    )
                    base.update(
                        {
                            "t1_exit_time": str(t1_daily["first_bar_time"]),
                            "t1_exit_adjusted_price": float(t1_daily["adjusted_open"]),
                            "t1_exit_observed": bool(sell_ok),
                            "t1_exit_reason": sell_reason,
                            "t1_gross_return": _gross_return(entry_price, float(t1_daily["adjusted_open"])) if sell_ok else np.nan,
                            "t1_net_return": _net_return(entry_price, float(t1_daily["adjusted_open"]), next_day, selected)
                            if sell_ok
                            else np.nan,
                        }
                    )
            else:
                base["t1_exit_reason"] = "no_next_trading_day"
        output.append(base)

    result = pd.DataFrame(output)
    ordered = list(dict.fromkeys(name for name in (*OUTCOME_COLUMNS, *metric_names) if name in result.columns))
    result = result.loc[:, ordered]
    result.sort_values(["signal_date", "signal_time", "strategy_id", "symbol", "ma_period"], kind="stable", inplace=True, ignore_index=True)
    return result


def _summary_for_values(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if not len(numeric):
        return {
            "observed_count": 0,
            "mean": None,
            "median": None,
            "win_rate": None,
            "profit_factor": None,
            "trimmed_mean_top1pct_removed": None,
        }
    positive = numeric[numeric > 0]
    negative = numeric[numeric < 0]
    profit_factor = float(positive.sum() / abs(negative.sum())) if len(negative) else None
    positive_indices = np.flatnonzero(numeric > 0.0)
    remove_count = max(1, int(np.ceil(len(positive_indices) * 0.01))) if len(positive_indices) else 0
    if remove_count:
        remove_indices = positive_indices[np.argsort(numeric[positive_indices])[::-1][:remove_count]]
        trimmed = np.delete(numeric, remove_indices)
    else:
        trimmed = numeric
    return {
        "observed_count": int(len(numeric)),
        "mean": float(numeric.mean()),
        "median": float(np.median(numeric)),
        "win_rate": float(np.mean(numeric > 0)),
        "profit_factor": profit_factor,
        "trimmed_mean_top1pct_removed": float(trimmed.mean()) if len(trimmed) else None,
    }


def _with_regimes(outcomes: pd.DataFrame, regime_frame: pd.DataFrame | None) -> pd.DataFrame:
    frame = outcomes.copy()
    frame["year"] = pd.to_numeric(frame["signal_date"].astype(str).str[:4], errors="coerce").astype("Int64")
    if regime_frame is None:
        return frame
    _require_columns(regime_frame, ("trade_date",), prefix="event_study_regime")
    regime = regime_frame.copy()
    regime["trade_date"] = _normalise_date_series(regime["trade_date"])
    if regime.duplicated(["trade_date"]).any():
        raise EventStudyError("event_study_regime_duplicate_date")
    rename = {"trade_date": "signal_date"}
    regime = regime.rename(columns=rename)
    overlap = set(frame.columns).intersection(regime.columns).difference({"signal_date"})
    if overlap:
        raise EventStudyError(f"event_study_regime_column_collision:{','.join(sorted(overlap))}")
    return frame.merge(regime, on="signal_date", how="left", validate="many_to_one")


def summarize_event_study(
    outcomes: pd.DataFrame,
    *,
    group_by: Sequence[str] = ("strategy_id", "ma_period"),
    regime_frame: pd.DataFrame | None = None,
    include_diagnostic: bool = False,
) -> pd.DataFrame:
    """Summarise counts, distribution, profit factor and winner-trimmed means."""

    if outcomes.empty:
        return pd.DataFrame()
    frame = _with_regimes(outcomes, regime_frame)
    if not include_diagnostic and "diagnostic_only" in frame.columns:
        frame = frame.loc[~frame["diagnostic_only"].fillna(False)].copy()
    keys = tuple(group_by)
    missing = sorted(set(keys).difference(frame.columns))
    if missing:
        raise EventStudyError(f"event_study_group_columns_missing:{','.join(missing)}")
    metric_names = [
        name
        for name in frame.columns
        if name.startswith(("gross_return_", "net_return_")) or name in {"mfe_same_day", "mae_same_day", "t1_gross_return", "t1_net_return"}
    ]
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(list(keys), dropna=False, sort=True, observed=True)
    for values, group in grouped:
        if not isinstance(values, tuple):
            values = (values,)
        row = {key: value for key, value in zip(keys, values, strict=True)}
        row["signal_count"] = int(len(group))
        row["entry_observed_count"] = int(group.get("entry_observed", pd.Series(False, index=group.index)).fillna(False).sum())
        row["entry_executable_count"] = int(group.get("entry_executable", pd.Series(False, index=group.index)).fillna(False).sum())
        row["diagnostic_excluded"] = not bool(include_diagnostic)
        for metric in metric_names:
            summary = _summary_for_values(group[metric])
            for name, value in summary.items():
                row[f"{metric}_{name}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def compare_to_control(
    outcomes: pd.DataFrame,
    *,
    strategy_id: str,
    control_id: str,
    metric: str = "net_return_60m",
    match_keys: Sequence[str] = ("symbol", "signal_date", "sixty_minute_bucket", "ma_period"),
) -> pd.DataFrame:
    """Return paired strategy-minus-control differences on local matched keys."""

    required = set(match_keys) | {"strategy_id", metric}
    _require_columns(outcomes, required, prefix="event_study_outcome")
    left = outcomes.loc[outcomes["strategy_id"].eq(strategy_id), list(match_keys) + [metric]].copy()
    right = outcomes.loc[outcomes["strategy_id"].eq(control_id), list(match_keys) + [metric]].copy()
    if left.empty or right.empty:
        return pd.DataFrame(
            columns=[
                *match_keys,
                "strategy_id",
                "control_id",
                "metric",
                "strategy_value",
                "control_value",
                "paired_difference",
            ]
        )
    left = left.drop_duplicates(list(match_keys), keep="first").rename(columns={metric: "strategy_value"})
    right = right.drop_duplicates(list(match_keys), keep="first").rename(columns={metric: "control_value"})
    merged = left.merge(right, on=list(match_keys), how="inner", validate="one_to_one")
    merged["paired_difference"] = merged["strategy_value"] - merged["control_value"]
    merged["strategy_id"] = str(strategy_id)
    merged["control_id"] = str(control_id)
    merged["metric"] = str(metric)
    return merged.loc[
        :,
        [
            *match_keys,
            "strategy_id",
            "control_id",
            "metric",
            "strategy_value",
            "control_value",
            "paired_difference",
        ],
    ]


def summarize_control_comparison(paired: pd.DataFrame) -> dict[str, Any]:
    """Summarise a paired comparison without treating missing outcomes as zero."""

    if paired.empty:
        return {"matched_count": 0, "observed_count": 0, "mean_difference": None, "median_difference": None, "positive_fraction": None}
    values = pd.to_numeric(paired["paired_difference"], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return {
        "matched_count": int(len(paired)),
        "observed_count": int(len(values)),
        "mean_difference": float(values.mean()) if len(values) else None,
        "median_difference": float(np.median(values)) if len(values) else None,
        "positive_fraction": float(np.mean(values > 0)) if len(values) else None,
    }


def build_representative_regime_frame() -> pd.DataFrame:
    """Return the fixed categories used by the minute-MA representative checks."""

    rows = [
        ("2022-04-25", "broad_down"),
        ("2022-04-27", "high_dispersion"),
        ("2022-04-29", "broad_up"),
        ("2022-06-15", "high_liquidity"),
        ("2022-09-30", "low_liquidity"),
        ("2022-11-08", "low_dispersion"),
        ("2022-11-29", "sector_concentration_proxy"),
        ("2023-08-29", "broad_up"),
        ("2023-09-28", "industry_rotation_proxy"),
        ("2024-02-05", "broad_down"),
        ("2024-02-07", "high_dispersion"),
        ("2024-09-30", "broad_up"),
    ]
    return pd.DataFrame(rows, columns=["trade_date", "representative_category"])


__all__ = [
    "EventStudyConfig",
    "EventStudyError",
    "OUTCOME_COLUMNS",
    "compare_to_control",
    "build_representative_regime_frame",
    "compute_event_outcomes",
    "prepare_outcome_bars",
    "summarize_control_comparison",
    "summarize_event_study",
]
