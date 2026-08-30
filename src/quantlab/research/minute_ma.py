"""Causal 60-minute moving-average states and diagnostic event summaries.

The module is intentionally separate from ``minute_v2``.  It implements the
chart periods used by the minute strategy research and keeps live, causal
features distinct from end-of-hour diagnostic outcomes.  Any full-hour value
exposed in the minute state table is prefixed with ``diagnostic_`` so it cannot
be mistaken for a real-time signal input.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from quantlab.data.qdp_v2.active import resolve_active_domain

MA_PERIODS = (10, 20, 40, 60, 120, 240)
EXPECTED_HOUR_BARS = 60

BAR_COLUMNS = (
    "symbol",
    "trade_date",
    "bar_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
)

EVENT_KEY_COLUMNS = ("symbol", "trade_date", "sixty_minute_bucket", "ma_period")


class MinuteMAError(ValueError):
    """Raised when the minute-MA research contract is violated."""


@dataclass(frozen=True)
class MinuteMAConfig:
    """Fixed event definitions; thresholds are descriptive, not optimized."""

    periods: tuple[int, ...] = MA_PERIODS
    near_touch_bps: float = 25.0
    posthoc_catchup_bps: float = 25.0
    prior_touch_window_hours: int = 20

    def validate(self) -> None:
        if not self.periods or tuple(sorted(set(self.periods))) != self.periods:
            raise MinuteMAError("minute_ma_periods_must_be_unique_and_sorted")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 1
            for value in self.periods
        ):
            raise MinuteMAError("minute_ma_period_must_exceed_one")
        for name, value in (
            ("near_touch_bps", self.near_touch_bps),
            ("posthoc_catchup_bps", self.posthoc_catchup_bps),
        ):
            if not np.isfinite(float(value)) or float(value) < 0:
                raise MinuteMAError(f"minute_ma_{name}_invalid")
        if (
            isinstance(self.prior_touch_window_hours, bool)
            or not isinstance(self.prior_touch_window_hours, (int, np.integer))
            or int(self.prior_touch_window_hours) <= 0
        ):
            raise MinuteMAError("minute_ma_prior_touch_window_invalid")

    def as_dict(self) -> dict[str, Any]:
        """Return the immutable event configuration in JSON-safe form."""

        return {
            "periods": [int(value) for value in self.periods],
            "near_touch_bps": float(self.near_touch_bps),
            "posthoc_catchup_bps": float(self.posthoc_catchup_bps),
            "prior_touch_window_hours": int(self.prior_touch_window_hours),
        }


def normalize_bar_time(value: Any) -> str:
    """Normalize common QDP/provider time encodings to ``HHMMSSmmm``."""

    if isinstance(value, pd.Timestamp):
        return value.strftime("%H%M%S") + f"{value.microsecond // 1000:03d}"
    if hasattr(value, "strftime") and not isinstance(value, str):
        try:
            return value.strftime("%H%M%S") + "000"
        except (AttributeError, ValueError):
            pass
    raw = str(value).strip()
    if raw.endswith(".0") and raw[:-2].isdigit():
        raw = raw[:-2]
    digits = "".join(character for character in raw if character.isdigit())
    if len(digits) == 4:
        digits += "00000"
    elif len(digits) == 6:
        digits += "000"
    elif len(digits) < 9:
        digits = digits.zfill(9)
    if len(digits) != 9:
        raise MinuteMAError(f"minute_ma_bar_time_invalid:{value!r}")
    return digits


def session_minute_ordinal(value: Any) -> int | None:
    """Return 0..239 for the four Chinese A-share trading hours."""

    bar_time = normalize_bar_time(value)
    hour = int(bar_time[:2])
    minute = int(bar_time[2:4])
    minute_of_day = hour * 60 + minute
    morning_start = 9 * 60 + 31
    morning_end = 11 * 60 + 30
    afternoon_start = 13 * 60 + 1
    afternoon_end = 15 * 60
    if morning_start <= minute_of_day <= morning_end:
        return minute_of_day - morning_start
    if afternoon_start <= minute_of_day <= afternoon_end:
        return 120 + minute_of_day - afternoon_start
    return None


def sixty_minute_bucket(value: Any) -> int | None:
    ordinal = session_minute_ordinal(value)
    return None if ordinal is None else ordinal // EXPECTED_HOUR_BARS + 1


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise MinuteMAError(f"minute_ma_columns_missing:{','.join(missing)}")


def prepare_minute_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Validate bars, apply the optional price factor, and drop incomplete hours."""

    _require_columns(bars, BAR_COLUMNS)
    frame = bars.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.strip()
    if frame["symbol"].eq("").any() or frame["symbol"].str.lower().isin({"nan", "none", "<na>"}).any():
        raise MinuteMAError("minute_ma_symbol_invalid")
    raw_dates = frame["trade_date"].astype("string").str.strip()
    compact_dates = raw_dates.str.fullmatch(r"\d{8}", na=False)
    parsed_dates = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    if compact_dates.any():
        parsed_dates.loc[compact_dates] = pd.to_datetime(raw_dates.loc[compact_dates], format="%Y%m%d", errors="coerce")
    if (~compact_dates).any():
        parsed_dates.loc[~compact_dates] = pd.to_datetime(raw_dates.loc[~compact_dates], errors="coerce")
    if parsed_dates.isna().any():
        bad = raw_dates.loc[parsed_dates.isna()].iloc[0]
        raise MinuteMAError(f"minute_ma_trade_date_invalid:{bad!r}")
    frame["trade_date"] = parsed_dates.dt.strftime("%Y-%m-%d")
    frame["bar_time"] = frame["bar_time"].map(normalize_bar_time).astype("string")
    for column in ("open", "high", "low", "close", "volume", "amount"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "adjust_factor" not in frame:
        frame["adjust_factor"] = 1.0
    frame["adjust_factor"] = pd.to_numeric(frame["adjust_factor"], errors="coerce")

    finite_prices = np.isfinite(frame.loc[:, ["open", "high", "low", "close"]]).all(axis=1)
    finite_activity = np.isfinite(frame.loc[:, ["volume", "amount"]]).all(axis=1)
    valid = (
        finite_prices
        & finite_activity
        & frame.loc[:, ["open", "high", "low", "close"]].gt(0).all(axis=1)
        & frame.loc[:, ["volume", "amount"]].ge(0).all(axis=1)
        & frame["high"].ge(frame["low"])
        & frame["high"].ge(frame[["open", "close"]].max(axis=1))
        & frame["low"].le(frame[["open", "close"]].min(axis=1))
        & np.isfinite(frame["adjust_factor"])
        & frame["adjust_factor"].gt(0)
    )
    if not bool(valid.all()):
        invalid_activity = (~finite_activity | frame.loc[:, ["volume", "amount"]].lt(0).any(axis=1)).sum()
        if int(invalid_activity):
            raise MinuteMAError(f"minute_ma_invalid_activity:{int(invalid_activity)}")
        raise MinuteMAError(f"minute_ma_invalid_ohlc_or_factor:{int((~valid).sum())}")

    frame["session_minute_ordinal"] = frame["bar_time"].map(session_minute_ordinal)
    if frame["session_minute_ordinal"].isna().any():
        bad = frame.loc[frame["session_minute_ordinal"].isna(), "bar_time"].iloc[0]
        raise MinuteMAError(f"minute_ma_out_of_session_bar:{bad}")
    frame["session_minute_ordinal"] = frame["session_minute_ordinal"].astype("int16")
    frame["sixty_minute_bucket"] = frame["session_minute_ordinal"].floordiv(EXPECTED_HOUR_BARS).add(1).astype("int8")
    key = ["symbol", "trade_date", "bar_time"]
    if frame.duplicated(key).any():
        raise MinuteMAError("minute_ma_duplicate_minute_key")
    frame.sort_values(key, inplace=True, ignore_index=True)

    hour_key = ["symbol", "trade_date", "sixty_minute_bucket"]
    expected_start = (frame["sixty_minute_bucket"].astype(int) - 1) * EXPECTED_HOUR_BARS
    frame["_expected_hour_start"] = expected_start
    completeness = frame.groupby(hour_key, sort=False).agg(
        bar_count=("bar_time", "size"),
        distinct_ordinals=("session_minute_ordinal", "nunique"),
        first_ordinal=("session_minute_ordinal", "min"),
        last_ordinal=("session_minute_ordinal", "max"),
        expected_start=("_expected_hour_start", "first"),
    )
    complete = (
        completeness["bar_count"].eq(EXPECTED_HOUR_BARS)
        & completeness["distinct_ordinals"].eq(EXPECTED_HOUR_BARS)
        & completeness["first_ordinal"].eq(completeness["expected_start"])
        & completeness["last_ordinal"].eq(completeness["expected_start"] + EXPECTED_HOUR_BARS - 1)
    )
    complete_keys = completeness.index[complete]
    indexed = frame.set_index(hour_key)
    frame = indexed.loc[indexed.index.isin(complete_keys)].reset_index()
    frame.drop(columns="_expected_hour_start", inplace=True)
    frame.sort_values(key, inplace=True, ignore_index=True)
    frame.attrs["incomplete_hour_count"] = int((~complete).sum())

    factor_counts = frame.groupby(["symbol", "trade_date"], sort=False)["adjust_factor"].nunique()
    if factor_counts.gt(1).any():
        raise MinuteMAError("minute_ma_intraday_adjust_factor_changed")
    for column in ("open", "high", "low", "close"):
        frame[f"adjusted_{column}"] = frame[column] * frame["adjust_factor"]
    return frame


def build_hourly_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Aggregate complete minute bars into the chart's four hourly bars."""

    prepared = bars if "adjusted_close" in bars.columns else prepare_minute_bars(bars)
    hour_key = ["symbol", "trade_date", "sixty_minute_bucket"]
    hourly = (
        prepared.groupby(hour_key, sort=False, observed=True)
        .agg(
            start_bar_time=("bar_time", "first"),
            end_bar_time=("bar_time", "last"),
            bar_count=("bar_time", "size"),
            adjust_factor=("adjust_factor", "first"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            amount=("amount", "sum"),
            adjusted_open=("adjusted_open", "first"),
            adjusted_high=("adjusted_high", "max"),
            adjusted_low=("adjusted_low", "min"),
            adjusted_close=("adjusted_close", "last"),
        )
        .reset_index()
    )
    hourly.sort_values(hour_key, inplace=True, ignore_index=True)
    hourly["hour_sequence"] = hourly.groupby("symbol", sort=False).cumcount().astype("int32")
    return hourly


def _hour_history(hourly: pd.DataFrame, config: MinuteMAConfig) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    keys = [
        "symbol",
        "trade_date",
        "sixty_minute_bucket",
        "hour_sequence",
        "adjust_factor",
        "adjusted_open",
        "adjusted_high",
        "adjusted_low",
        "adjusted_close",
        "close",
    ]
    for period in config.periods:
        current = hourly.loc[:, keys].copy()
        current["ma_period"] = int(period)
        prior_sum = pd.Series(index=current.index, dtype="float64")
        prior_ma = pd.Series(index=current.index, dtype="float64")
        prior_ma_slope = pd.Series(index=current.index, dtype="float64")
        previous_close = pd.Series(index=current.index, dtype="float64")
        previous_high = pd.Series(index=current.index, dtype="float64")
        for _, index in current.groupby("symbol", sort=False).groups.items():
            values = current.loc[index, "adjusted_close"]
            sums = values.rolling(period - 1, min_periods=period - 1).sum().shift(1)
            ma = values.rolling(period, min_periods=period).mean().shift(1)
            previous_ma = ma.shift(1)
            prior_sum.loc[index] = sums
            prior_ma.loc[index] = ma
            prior_ma_slope.loc[index] = np.where(previous_ma.gt(0), (ma / previous_ma - 1.0) * 10_000.0, np.nan)
            previous_close.loc[index] = values.shift(1)
            previous_high.loc[index] = current.loc[index, "adjusted_high"].shift(1)
        current["prior_close_sum_adjusted"] = prior_sum
        current["prior_completed_ma_adjusted"] = prior_ma
        current["prior_ma_slope_bps"] = prior_ma_slope
        current["previous_hour_close_adjusted"] = previous_close
        current["previous_hour_high_adjusted"] = previous_high
        current["causal_intersection_adjusted"] = prior_sum / float(period - 1)
        current["causal_intersection"] = current["causal_intersection_adjusted"] / current["adjust_factor"]
        current["final_ma_adjusted"] = (prior_sum + current["adjusted_close"]) / float(period)
        current["final_ma"] = current["final_ma_adjusted"] / current["adjust_factor"]
        parts.append(current)
    history = pd.concat(parts, ignore_index=True)
    history = history.loc[history["prior_close_sum_adjusted"].notna()].copy()
    history.sort_values(
        ["symbol", "trade_date", "sixty_minute_bucket", "ma_period"],
        inplace=True,
        ignore_index=True,
    )
    return history


def _add_prior_touch_counts(
    prepared: pd.DataFrame,
    history: pd.DataFrame,
    config: MinuteMAConfig,
) -> pd.DataFrame:
    minimal_bars = prepared.loc[
        :,
        [
            "symbol",
            "trade_date",
            "sixty_minute_bucket",
            "adjusted_low",
            "adjusted_high",
        ],
    ]
    parts: list[pd.DataFrame] = []
    for period in config.periods:
        levels = history.loc[
            history["ma_period"].eq(period),
            [
                "symbol",
                "trade_date",
                "sixty_minute_bucket",
                "ma_period",
                "hour_sequence",
                "causal_intersection_adjusted",
            ],
        ]
        merged = minimal_bars.merge(
            levels,
            on=["symbol", "trade_date", "sixty_minute_bucket"],
            how="inner",
            validate="many_to_one",
        )
        merged["true_touch"] = merged["adjusted_low"].le(merged["causal_intersection_adjusted"]) & merged[
            "adjusted_high"
        ].ge(merged["causal_intersection_adjusted"])
        touches = (
            merged.groupby(
                [
                    "symbol",
                    "trade_date",
                    "sixty_minute_bucket",
                    "ma_period",
                    "hour_sequence",
                ],
                sort=False,
                observed=True,
            )["true_touch"]
            .any()
            .reset_index()
        )
        touches.sort_values(["symbol", "hour_sequence"], inplace=True)
        prior_window = pd.Series(index=touches.index, dtype="float64")
        prior_loaded = pd.Series(index=touches.index, dtype="float64")
        for _, index in touches.groupby("symbol", sort=False).groups.items():
            values = touches.loc[index, "true_touch"].astype(int)
            prior_window.loc[index] = (
                values.rolling(
                    int(config.prior_touch_window_hours),
                    min_periods=1,
                )
                .sum()
                .shift(1)
                .fillna(0)
            )
            prior_loaded.loc[index] = values.cumsum().shift(1).fillna(0)
        touches["prior_true_touch_count_window"] = prior_window.astype("int16")
        touches["prior_true_touch_count_loaded"] = prior_loaded.astype("int32")
        parts.append(touches.drop(columns="true_touch"))
    return pd.concat(parts, ignore_index=True)


def _target_mask(
    frame: pd.DataFrame,
    target_dates: Sequence[str] | None,
) -> pd.Series:
    if target_dates is None:
        return pd.Series(True, index=frame.index)
    values = {date.fromisoformat(str(value)).isoformat() for value in target_dates}
    return frame["trade_date"].isin(values)


def build_minute_ma_states(
    bars: pd.DataFrame,
    *,
    config: MinuteMAConfig | None = None,
    target_dates: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build long causal minute states for each configured hourly MA period.

    Historical bars are retained for the MA seed and prior-touch counters.  If
    ``target_dates`` is supplied, only those dates are expanded to minute x MA
    rows, keeping representative validation runs small.
    """

    selected = config or MinuteMAConfig()
    selected.validate()
    prepared = prepare_minute_bars(bars)
    hourly = build_hourly_bars(prepared)
    history = _hour_history(hourly, selected)
    touch_counts = _add_prior_touch_counts(prepared, history, selected)
    target = prepared.loc[_target_mask(prepared, target_dates)].copy()
    if target.empty:
        return pd.DataFrame()

    history_columns = [
        "symbol",
        "trade_date",
        "sixty_minute_bucket",
        "ma_period",
        "hour_sequence",
        "prior_close_sum_adjusted",
        "prior_completed_ma_adjusted",
        "prior_ma_slope_bps",
        "previous_hour_close_adjusted",
        "previous_hour_high_adjusted",
        "causal_intersection_adjusted",
        "causal_intersection",
        "final_ma_adjusted",
        "final_ma",
        "adjusted_open",
        "adjusted_high",
        "adjusted_low",
        "adjusted_close",
    ]
    history_target = history.loc[history["trade_date"].isin(target["trade_date"].unique()), history_columns].rename(
        columns={
            "final_ma_adjusted": "diagnostic_final_ma_adjusted",
            "final_ma": "diagnostic_final_ma",
            "adjusted_open": "diagnostic_hour_adjusted_open",
            "adjusted_high": "diagnostic_hour_adjusted_high",
            "adjusted_low": "diagnostic_hour_adjusted_low",
            "adjusted_close": "diagnostic_hour_adjusted_close",
        }
    )
    states = target.merge(
        history_target,
        on=["symbol", "trade_date", "sixty_minute_bucket"],
        how="inner",
        validate="many_to_many",
    )
    states = states.merge(
        touch_counts,
        on=[
            "symbol",
            "trade_date",
            "sixty_minute_bucket",
            "ma_period",
            "hour_sequence",
        ],
        how="left",
        validate="many_to_one",
    )
    states.sort_values(
        ["symbol", "trade_date", "sixty_minute_bucket", "ma_period", "bar_time"],
        inplace=True,
        ignore_index=True,
    )
    level = states["causal_intersection_adjusted"]
    states["live_ma_adjusted"] = (states["prior_close_sum_adjusted"] + states["adjusted_close"]) / states["ma_period"]
    states["live_ma"] = states["live_ma_adjusted"] / states["adjust_factor"]
    states["close_to_live_ma_bps"] = (states["adjusted_close"] / states["live_ma_adjusted"] - 1.0) * 10_000.0
    states["close_to_intersection_bps"] = (states["adjusted_close"] / level - 1.0) * 10_000.0
    states["range_distance_to_intersection_bps"] = np.select(
        [states["adjusted_low"].gt(level), states["adjusted_high"].lt(level)],
        [
            (states["adjusted_low"] / level - 1.0) * 10_000.0,
            (states["adjusted_high"] / level - 1.0) * 10_000.0,
        ],
        default=0.0,
    )
    states["touched_now"] = states["adjusted_low"].le(level) & states["adjusted_high"].ge(level)
    states["close_below_intersection"] = states["adjusted_close"].lt(level)

    group_key = list(EVENT_KEY_COLUMNS)
    grouped = states.groupby(group_key, sort=False, observed=True)
    states["partial_hour_high_adjusted"] = grouped["adjusted_high"].cummax()
    states["partial_hour_low_adjusted"] = grouped["adjusted_low"].cummin()
    states["partial_hour_open_adjusted"] = grouped["adjusted_open"].transform("first")
    states["minutes_below_so_far"] = grouped["close_below_intersection"].cumsum().astype("int16")
    states["rebound_from_partial_low_bps"] = (
        states["adjusted_close"] / states["partial_hour_low_adjusted"] - 1.0
    ) * 10_000.0
    states["partial_hour_range_bps"] = (
        states["partial_hour_high_adjusted"] / states["partial_hour_low_adjusted"] - 1.0
    ) * 10_000.0
    range_value = states["partial_hour_high_adjusted"] - states["partial_hour_low_adjusted"]
    states["partial_hour_close_location"] = np.where(
        range_value.gt(0),
        (states["adjusted_close"] - states["partial_hour_low_adjusted"]) / range_value,
        0.5,
    )
    body_low = np.minimum(states["partial_hour_open_adjusted"], states["adjusted_close"])
    states["partial_lower_wick_bps"] = (body_low / states["partial_hour_low_adjusted"] - 1.0) * 10_000.0

    previous_close = grouped["adjusted_close"].shift(1).fillna(states["previous_hour_close_adjusted"])
    direction = np.sign(states["adjusted_close"] - previous_close)
    states["up_amount"] = np.where(direction > 0, states["amount"], 0.0)
    states["down_amount"] = np.where(direction < 0, states["amount"], 0.0)
    states["cumulative_up_amount"] = states.groupby(group_key, sort=False)["up_amount"].cumsum()
    states["cumulative_down_amount"] = states.groupby(group_key, sort=False)["down_amount"].cumsum()
    states["up_down_amount_ratio"] = np.where(
        states["cumulative_down_amount"].gt(0),
        states["cumulative_up_amount"] / states["cumulative_down_amount"],
        np.nan,
    )

    alignment_key = ["symbol", "trade_date", "bar_time"]
    pivot = states.pivot_table(
        index=alignment_key,
        columns="ma_period",
        values="live_ma_adjusted",
        aggfunc="first",
    )
    score = pd.Series(0.0, index=pivot.index)
    comparisons = pd.Series(0, index=pivot.index, dtype="int16")
    for short, long in zip(selected.periods[:-1], selected.periods[1:], strict=True):
        if short not in pivot or long not in pivot:
            continue
        available = pivot[short].notna() & pivot[long].notna()
        score.loc[available] += np.sign(pivot.loc[available, short] - pivot.loc[available, long])
        comparisons.loc[available] += 1
    alignment = (score / comparisons.replace(0, np.nan)).rename("ma_alignment_score").reset_index()
    states = states.merge(alignment, on=alignment_key, how="left", validate="many_to_one")
    states["bullish_ma_stack"] = states["ma_alignment_score"].eq(1.0)
    states["bearish_ma_stack"] = states["ma_alignment_score"].eq(-1.0)
    states.drop(columns=["up_amount", "down_amount"], inplace=True)
    states.attrs["causal_columns"] = [column for column in states.columns if not column.startswith("diagnostic_")]
    states.attrs["diagnostic_columns"] = [
        "diagnostic_final_ma_adjusted",
        "diagnostic_final_ma",
        "diagnostic_hour_adjusted_open",
        "diagnostic_hour_adjusted_high",
        "diagnostic_hour_adjusted_low",
        "diagnostic_hour_adjusted_close",
    ]
    return states


def _first_true_index(values: np.ndarray) -> int | None:
    indices = np.flatnonzero(values)
    return None if len(indices) == 0 else int(indices[0])


def _max_true_run(values: np.ndarray) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return best


def _event_summary(group: pd.DataFrame, config: MinuteMAConfig) -> dict[str, Any] | None:
    current = group.sort_values("bar_time").reset_index(drop=True)
    level = float(current["causal_intersection_adjusted"].iloc[0])
    touched = current["touched_now"].to_numpy(dtype=bool)
    below = current["close_below_intersection"].to_numpy(dtype=bool)
    sides = (~below).astype(np.int8)
    opening_above = bool(float(current["previous_hour_close_adjusted"].iloc[0]) >= level)
    path = np.concatenate(([int(opening_above)], sides))
    close_cross_count = int(np.count_nonzero(path[1:] != path[:-1]))
    touch_count = int(np.count_nonzero(touched & ~np.concatenate(([False], touched[:-1]))))
    first_touch = _first_true_index(touched)
    first_below = _first_true_index(below)
    reclaim = None
    if first_below is not None:
        reclaim_relative = _first_true_index(~below[first_below + 1 :])
        if reclaim_relative is not None:
            reclaim = first_below + 1 + reclaim_relative
    elif not opening_above:
        reclaim = _first_true_index(~below)

    true_touch = bool(touched.any())
    final_above = bool(not below[-1])
    hour_low = float(current["diagnostic_hour_adjusted_low"].iloc[0])
    final_ma = float(current["diagnostic_final_ma_adjusted"].iloc[0])
    final_ma_to_low_bps = (final_ma / hour_low - 1.0) * 10_000.0
    closest_index = int(current["range_distance_to_intersection_bps"].abs().idxmin())
    posthoc_catchup = bool(
        not true_touch and hour_low > level and abs(final_ma_to_low_bps) <= float(config.posthoc_catchup_bps)
    )
    near_touch = bool(
        not true_touch
        and abs(float(current.loc[closest_index, "range_distance_to_intersection_bps"])) <= float(config.near_touch_bps)
    )

    if true_touch and close_cross_count >= 3:
        kind = "repeated_cross"
    elif true_touch and opening_above and first_below is not None and reclaim is not None and final_above:
        kind = "break_reclaim"
    elif true_touch and not opening_above and reclaim is not None and final_above:
        kind = "reclaim_from_below"
    elif true_touch and below.any() and not final_above:
        kind = "failed_break"
    elif true_touch and final_above:
        kind = "touch_hold"
    elif posthoc_catchup:
        kind = "posthoc_catchup"
    elif near_touch:
        kind = "near_touch"
    else:
        return None

    confirmation_index: int | None
    if kind in {"break_reclaim", "reclaim_from_below", "repeated_cross"}:
        confirmation_index = reclaim
    elif kind in {"touch_hold", "failed_break"}:
        confirmation_index = first_touch
    else:
        confirmation_index = None
    reference_index = first_touch if first_touch is not None else closest_index
    reference = current.iloc[reference_index]
    confirmation = current.iloc[confirmation_index] if confirmation_index is not None else None
    first_below_time = None if first_below is None else str(current.loc[first_below, "bar_time"])
    first_reclaim_time = None if reclaim is None else str(current.loc[reclaim, "bar_time"])

    base = current.iloc[0]
    final = current.iloc[-1]
    return {
        "symbol": str(base["symbol"]),
        "trade_date": str(base["trade_date"]),
        "sixty_minute_bucket": int(base["sixty_minute_bucket"]),
        "ma_period": int(base["ma_period"]),
        "hour_sequence": int(base["hour_sequence"]),
        "event_kind": kind,
        "true_touch": true_touch,
        "near_touch": near_touch,
        "posthoc_catchup": posthoc_catchup,
        "opening_above_intersection": opening_above,
        "final_above_intersection": final_above,
        "touch_episode_count": touch_count,
        "close_cross_count": close_cross_count,
        "minutes_below": int(below.sum()),
        "max_consecutive_minutes_below": _max_true_run(below),
        "prior_true_touch_count_window": int(base["prior_true_touch_count_window"]),
        "prior_touch_window_hours": int(config.prior_touch_window_hours),
        "prior_true_touch_count_loaded": int(base["prior_true_touch_count_loaded"]),
        "first_touch_time": None if first_touch is None else str(current.loc[first_touch, "bar_time"]),
        "first_below_close_time": first_below_time,
        "first_reclaim_time": first_reclaim_time,
        "causal_confirmation_time": (None if confirmation is None else str(confirmation["bar_time"])),
        "reference_time": str(reference["bar_time"]),
        "causal_intersection": float(base["causal_intersection"]),
        "causal_intersection_adjusted": level,
        "live_ma_at_reference": float(reference["live_ma"]),
        "final_ma": float(base["diagnostic_final_ma"]),
        "final_ma_to_hour_low_bps": final_ma_to_low_bps,
        "closest_range_distance_bps": float(current.loc[closest_index, "range_distance_to_intersection_bps"]),
        "max_break_depth_bps": float((current["adjusted_low"].min() / level - 1.0) * 10_000.0),
        "rebound_at_reference_bps": float(reference["rebound_from_partial_low_bps"]),
        "rebound_at_confirmation_bps": (
            np.nan if confirmation is None else float(confirmation["rebound_from_partial_low_bps"])
        ),
        "final_rebound_from_hour_low_bps": float((float(final["adjusted_close"]) / hour_low - 1.0) * 10_000.0),
        "partial_lower_wick_at_reference_bps": float(reference["partial_lower_wick_bps"]),
        "up_down_amount_ratio_at_reference": float(reference["up_down_amount_ratio"]),
        "up_down_amount_ratio_final": float(final["up_down_amount_ratio"]),
        "prior_completed_ma_adjusted": float(base["prior_completed_ma_adjusted"]),
        "prior_ma_slope_bps": float(base["prior_ma_slope_bps"]),
        "ma_alignment_score_at_reference": float(reference["ma_alignment_score"]),
        "bullish_ma_stack_at_reference": bool(reference["bullish_ma_stack"]),
        "bearish_ma_stack_at_reference": bool(reference["bearish_ma_stack"]),
        "hour_open": float(base["diagnostic_hour_adjusted_open"] / base["adjust_factor"]),
        "hour_high": float(base["diagnostic_hour_adjusted_high"] / base["adjust_factor"]),
        "hour_low": float(base["diagnostic_hour_adjusted_low"] / base["adjust_factor"]),
        "hour_close": float(base["diagnostic_hour_adjusted_close"] / base["adjust_factor"]),
        "diagnostic_uses_hour_end": True,
    }


def summarize_ma_events(
    states: pd.DataFrame,
    *,
    config: MinuteMAConfig | None = None,
) -> pd.DataFrame:
    """Summarize causal touches and explicitly non-causal chart catch-ups."""

    selected = config or MinuteMAConfig()
    selected.validate()
    if states.empty:
        return pd.DataFrame()
    _require_columns(states, (*EVENT_KEY_COLUMNS, "touched_now", "bar_time"))
    rows = [
        summary
        for _, group in states.groupby(list(EVENT_KEY_COLUMNS), sort=False, observed=True)
        if (summary := _event_summary(group, selected)) is not None
    ]
    if not rows:
        return pd.DataFrame()
    events = pd.DataFrame(rows)
    events.sort_values(
        ["trade_date", "symbol", "sixty_minute_bucket", "ma_period"],
        inplace=True,
        ignore_index=True,
    )
    return events


def build_ma_event_table(
    bars: pd.DataFrame,
    *,
    config: MinuteMAConfig | None = None,
    target_dates: Sequence[str] | None = None,
) -> pd.DataFrame:
    states = build_minute_ma_states(bars, config=config, target_dates=target_dates)
    return summarize_ma_events(states, config=config)


def _sql_scan(paths: Sequence[Path]) -> str:
    if not paths:
        raise MinuteMAError("minute_ma_source_paths_empty")
    values = ",".join("'" + str(path).replace("'", "''") + "'" for path in paths)
    return f"read_parquet([{values}], union_by_name=true)"


def _overlapping_paths(
    workspace_root: Path,
    domain: str,
    start_date: str,
    end_date: str,
) -> list[Path]:
    context = resolve_active_domain(domain, workspace_root=workspace_root)
    return [
        path
        for shard, path in zip(context.manifest.shards, context.shard_paths, strict=True)
        if str(shard.end_date) >= start_date and str(shard.start_date) <= end_date
    ]


def _excluded_stock_days(
    workspace_root: Path,
    *,
    start_date: str,
    end_date: str,
) -> set[tuple[str, str]]:
    years = range(int(start_date[:4]), int(end_date[:4]) + 1)
    root = workspace_root / "data" / "qdp" / "source_archives" / "minute" / "quality"
    excluded: set[tuple[str, str]] = set()
    for year in years:
        directory = root / f"year={year}"
        minute_path = directory / "minute_feature_exclusions.parquet"
        if minute_path.is_file():
            frame = pd.read_parquet(minute_path)
            flags = [name for name in ("exclude_open", "exclude_high", "exclude_low", "exclude_close") if name in frame]
            if flags:
                bad = frame.loc[frame[flags].fillna(False).any(axis=1), ["symbol", "trade_date"]]
                excluded.update(zip(bad["symbol"].astype(str), bad["trade_date"].astype(str), strict=False))
        session_path = directory / "session_feature_exclusions.parquet"
        if session_path.is_file():
            frame = pd.read_parquet(session_path, columns=["symbol", "trade_date"])
            excluded.update(zip(frame["symbol"].astype(str), frame["trade_date"].astype(str), strict=False))
    return {key for key in excluded if start_date <= key[1] <= end_date}


def load_qdp_minute_bars(
    workspace_root: str | Path,
    *,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load a bounded, quality-filtered local QDP minute slice."""

    root = Path(workspace_root).resolve()
    start = date.fromisoformat(str(start_date)).isoformat()
    end = date.fromisoformat(str(end_date)).isoformat()
    if start > end:
        raise MinuteMAError("minute_ma_source_date_range_invalid")
    selected_symbols = tuple(
        sorted({str(value).strip() for value in symbols if value is not None and str(value).strip()})
    )
    if not selected_symbols:
        raise MinuteMAError("minute_ma_source_symbols_empty")

    minute_paths = _overlapping_paths(root, "market_intraday_1m", start, end)
    factor_context = resolve_active_domain("adjust_factor", workspace_root=root)
    connection = duckdb.connect()
    connection.execute("PRAGMA threads=4")
    symbol_frame = pd.DataFrame({"symbol": selected_symbols})
    connection.register("selected_symbols", symbol_frame)
    query = f"""
        SELECT
            CAST(m.symbol AS VARCHAR) AS symbol,
            CAST(m.trade_date AS VARCHAR) AS trade_date,
            CAST(m.bar_time AS VARCHAR) AS bar_time,
            TRY_CAST(m.open AS DOUBLE) AS open,
            TRY_CAST(m.high AS DOUBLE) AS high,
            TRY_CAST(m.low AS DOUBLE) AS low,
            TRY_CAST(m.close AS DOUBLE) AS close,
            TRY_CAST(m.volume AS DOUBLE) AS volume,
            TRY_CAST(m.amount AS DOUBLE) AS amount,
            TRY_CAST(f.adjust_factor AS DOUBLE) AS adjust_factor
        FROM {_sql_scan(minute_paths)} m
        JOIN selected_symbols s ON s.symbol = m.symbol
        LEFT JOIN {_sql_scan(factor_context.shard_paths)} f
          ON f.symbol = m.symbol AND f.trade_date = m.trade_date
        WHERE m.trade_date BETWEEN ? AND ?
        ORDER BY m.symbol, m.trade_date, m.bar_time
    """
    try:
        frame = connection.execute(query, [start, end]).df()
    finally:
        connection.close()
    if frame.empty:
        raise MinuteMAError(f"minute_ma_source_slice_empty:{start}:{end}")
    if frame.duplicated(["symbol", "trade_date", "bar_time"]).any():
        raise MinuteMAError("minute_ma_source_duplicate_minute_key")
    if frame["adjust_factor"].isna().any():
        missing = frame.loc[frame["adjust_factor"].isna(), ["symbol", "trade_date"]].iloc[0]
        raise MinuteMAError(f"minute_ma_adjust_factor_missing:{missing['symbol']}:{missing['trade_date']}")
    excluded = _excluded_stock_days(root, start_date=start, end_date=end)
    source_keys = set(zip(frame["symbol"].astype(str), frame["trade_date"].astype(str), strict=False))
    applied_excluded = excluded.intersection(source_keys)
    if excluded:
        keys = pd.MultiIndex.from_frame(frame.loc[:, ["symbol", "trade_date"]])
        frame = frame.loc[~keys.isin(pd.MultiIndex.from_tuples(sorted(excluded)))].copy()
    if frame.empty:
        raise MinuteMAError(f"minute_ma_source_slice_empty_after_quality:{start}:{end}")
    frame.attrs["quality_excluded_stock_days"] = len(applied_excluded)
    frame.attrs["quality_excluded_stock_days_available_in_range"] = len(excluded)
    return frame


__all__ = [
    "BAR_COLUMNS",
    "EVENT_KEY_COLUMNS",
    "EXPECTED_HOUR_BARS",
    "MA_PERIODS",
    "MinuteMAConfig",
    "MinuteMAError",
    "build_hourly_bars",
    "build_ma_event_table",
    "build_minute_ma_states",
    "load_qdp_minute_bars",
    "normalize_bar_time",
    "prepare_minute_bars",
    "session_minute_ordinal",
    "sixty_minute_bucket",
    "summarize_ma_events",
]
