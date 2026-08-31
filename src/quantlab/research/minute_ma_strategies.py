"""Finite, causal rule variants for the minute-MA event layer.

The module deliberately contains rules rather than a portfolio simulator. A
rule can only inspect the current row and rows before it in its
symbol/date/hour/MA group. End-of-hour diagnostic columns are never used for
executable signals.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantlab.research.minute_ma import (
    EVENT_KEY_COLUMNS,
    MA_PERIODS,
    normalize_bar_time,
    session_minute_ordinal,
    summarize_ma_events,
)


class StrategyError(ValueError):
    """Raised when a strategy definition or signal input is invalid."""


DEFAULT_LIQUIDITY_COLUMN = "prior_20d_median_amount"


SIGNAL_COLUMNS = (
    "signal_id",
    "strategy_id",
    "strategy_family",
    "symbol",
    "signal_date",
    "signal_time",
    "entry_time",
    "sixty_minute_bucket",
    "ma_period",
    "hour_sequence",
    "event_trigger",
    "causal_only",
    "diagnostic_only",
    "signal_executable",
    "reference_signal_id",
    "reference_symbol",
    "liquidity_match_ratio",
    "close_to_intersection_bps",
    "prior_ma_slope_bps",
    "ma_alignment_score",
    "prior_true_touch_count_window",
    "up_down_amount_ratio",
    "bullish_ma_stack",
    "previous_hour_close_adjusted",
    "previous_hour_high_adjusted",
    "signal_adjusted_close",
    "signal_amount",
    "author_id",
    "hypothesis_id",
    "market_regime",
    "sector_strength_rank",
    "sector_breadth",
    "leader_relative_return",
    "auction_gap",
    "volume_acceleration_5_20",
    "amount_curve_surprise",
    "vwap_deviation",
    "recent_high_breakout",
    "prior_acceleration",
    "breakout_recent",
    "daily_trend_positive",
    "market_supportive",
    "sector_strong",
    "leader_sync",
    "auction_confirmed",
    "volume_normal",
    "not_repeated_cross",
)


@dataclass(frozen=True)
class StrategySpec:
    """A named rule version kept immutable for reproducible comparisons."""

    strategy_id: str
    family: str
    description: str
    signal_rule: str
    ma_periods: tuple[int, ...] = MA_PERIODS
    stable_minutes: int = 0
    near_touch_bps: float = 25.0
    causal: bool = True
    executable: bool = True
    control: bool = False
    implemented: bool = True
    required_columns: tuple[str, ...] = ()
    filters: tuple[str, ...] = ()
    reference_control: bool = False
    author_id: str | None = None
    hypothesis_id: str | None = None

    def validate(self) -> None:
        if not self.strategy_id or self.strategy_id.strip() != self.strategy_id:
            raise StrategyError("strategy_id_invalid")
        if not self.family or not self.signal_rule:
            raise StrategyError("strategy_definition_incomplete")
        if not self.ma_periods or tuple(sorted(set(self.ma_periods))) != self.ma_periods:
            raise StrategyError("strategy_ma_periods_invalid")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 1
            for value in self.ma_periods
        ):
            raise StrategyError("strategy_ma_period_invalid")
        if not np.isfinite(float(self.near_touch_bps)) or float(self.near_touch_bps) < 0:
            raise StrategyError("strategy_near_touch_bps_invalid")
        if (
            isinstance(self.stable_minutes, bool)
            or not isinstance(self.stable_minutes, (int, np.integer))
            or int(self.stable_minutes) < 0
        ):
            raise StrategyError("strategy_stable_minutes_invalid")
        if self.signal_rule == "break_reclaim_stable" and int(self.stable_minutes) < 1:
            raise StrategyError("strategy_stable_minutes_required")
        if not self.causal and self.executable:
            raise StrategyError("noncausal_strategy_cannot_be_executable")
        if self.reference_control and not self.control:
            raise StrategyError("reference_control_must_be_control")
        if self.reference_control and self.signal_rule != "liquidity_matched_stock":
            raise StrategyError("reference_control_rule_invalid")
        if (self.author_id is None) != (self.hypothesis_id is None):
            raise StrategyError("strategy_author_hypothesis_pair_invalid")
        unknown_filters = sorted(set(self.filters).difference(_FILTER_COLUMNS))
        if unknown_filters:
            raise StrategyError(f"unknown_strategy_filter:{','.join(unknown_filters)}")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["ma_periods"] = list(self.ma_periods)
        value["required_columns"] = list(self.required_columns)
        value["filters"] = list(self.filters)
        return value


_BASE_COLUMNS = {
    "symbol",
    "trade_date",
    "bar_time",
    "sixty_minute_bucket",
    "ma_period",
    "hour_sequence",
    "session_minute_ordinal",
    "adjusted_close",
    "adjusted_high",
    "adjusted_low",
    "amount",
    "causal_intersection_adjusted",
    "close_to_intersection_bps",
    "range_distance_to_intersection_bps",
    "touched_now",
    "close_below_intersection",
    "previous_hour_close_adjusted",
    "prior_ma_slope_bps",
    "ma_alignment_score",
    "prior_true_touch_count_window",
    "up_down_amount_ratio",
    "bullish_ma_stack",
}


_FILTER_COLUMNS = {
    "positive_slope": {"prior_ma_slope_bps"},
    "bullish_stack": {"bullish_ma_stack"},
    "first_touch": {"prior_true_touch_count_window"},
    "amount_confirm": {"up_down_amount_ratio"},
    "recent_high_breakout": {"recent_high_breakout"},
    "prior_acceleration": {"prior_acceleration"},
    "breakout_recent": {"breakout_recent"},
    "daily_trend_positive": {"daily_trend_positive"},
    "market_supportive": {"market_supportive"},
    "sector_strong": {"sector_strong"},
    "leader_sync": {"leader_sync"},
    "auction_confirmed": {"auction_confirmed"},
    "volume_normal": {"volume_normal"},
    "not_repeated_cross": {"not_repeated_cross"},
    "vwap_supportive": {"vwap_supportive"},
    "amount_acceleration_positive": {"amount_acceleration_positive"},
}


def strategy_catalog(*, include_unavailable: bool = True) -> tuple[StrategySpec, ...]:
    """Return the finite strategy registry.

    Unavailable entries document the next data-dependent layer without
    fabricating market or sector features. The signal builder refuses these
    entries until their required point-in-time fields are supplied.
    """

    ma = MA_PERIODS
    specs = (
        StrategySpec(
            "s0_random_matched",
            "S0",
            "Within-stock random-time control; it matches the same stock/hour, not cross-sectional liquidity.",
            "random_minute",
            ma_periods=ma,
            control=True,
        ),
        StrategySpec(
            "s0_liquidity_matched",
            "S0",
            "Cross-sectional control: deterministic random choice among same-minute stocks within a causal turnover band.",
            "liquidity_matched_stock",
            ma_periods=ma,
            control=True,
            reference_control=True,
            required_columns=("prior_20d_median_amount",),
        ),
        StrategySpec(
            "s0_strong_no_ma",
            "S0",
            "First minute closing at or above the previous completed hour high; a price-strength proxy without MA inputs.",
            "strong_without_ma",
            ma_periods=ma,
            control=True,
            required_columns=("previous_hour_high_adjusted",),
        ),
        StrategySpec(
            "s0_distance_only",
            "S0",
            "First close within the configured distance of the causal MA intersection; no reversal condition.",
            "distance_only",
            ma_periods=ma,
            control=True,
        ),
        StrategySpec(
            "s1_touch_immediate",
            "S1",
            "First actual intrabar touch, entered at the next minute open.",
            "touch_immediate",
            ma_periods=ma,
        ),
        StrategySpec(
            "s1_touch_reclaim",
            "S1",
            "Touch or break followed by the first causal close back above the intersection.",
            "touch_reclaim",
            ma_periods=ma,
        ),
        StrategySpec(
            "s1_near_reversal",
            "S1",
            "Near-touch followed by a causal upward reversal above the intersection.",
            "near_reversal",
            ma_periods=ma,
        ),
        StrategySpec(
            "s1_break_reclaim_stable3",
            "S1",
            "Break/touch followed by three consecutive closes above the intersection.",
            "break_reclaim_stable",
            ma_periods=ma,
            stable_minutes=3,
        ),
        StrategySpec(
            "s1_posthoc_catchup_diagnostic",
            "S1",
            "End-of-hour MA catch-up, retained only as a non-causal diagnostic control.",
            "posthoc_catchup",
            ma_periods=ma,
            causal=False,
            executable=False,
            control=True,
        ),
        StrategySpec(
            "s2_reclaim_positive_slope",
            "S2",
            "S1 reclaim with a positive prior completed-MA slope.",
            "touch_reclaim",
            ma_periods=ma,
            filters=("positive_slope",),
        ),
        StrategySpec(
            "s2_reclaim_bull_stack",
            "S2",
            "S1 reclaim while the causal moving averages are bullishly stacked.",
            "touch_reclaim",
            ma_periods=ma,
            filters=("bullish_stack",),
        ),
        StrategySpec(
            "s2_first_touch_reclaim",
            "S2",
            "S1 reclaim when no true touch occurred in the preceding configured window.",
            "touch_reclaim",
            ma_periods=ma,
            filters=("first_touch",),
        ),
        StrategySpec(
            "s2_reclaim_recent_high",
            "S2",
            "S1 reclaim after a causal recent-high breakout in the same session.",
            "touch_reclaim",
            ma_periods=ma,
            filters=("recent_high_breakout",),
        ),
        StrategySpec(
            "s2_reclaim_acceleration",
            "S2",
            "S1 reclaim after a positive short-over-long price acceleration.",
            "touch_reclaim",
            ma_periods=ma,
            filters=("prior_acceleration",),
        ),
        StrategySpec(
            "s2_reclaim_breakout_recent",
            "S2",
            "S1 reclaim while the latest breakout remains recent.",
            "touch_reclaim",
            ma_periods=ma,
            filters=("breakout_recent",),
        ),
        StrategySpec(
            "s2_reclaim_daily_trend",
            "S2",
            "S1 reclaim in a positive prior daily trend.",
            "touch_reclaim",
            ma_periods=ma,
            filters=("daily_trend_positive",),
        ),
        StrategySpec(
            "s4_reclaim_volume_proxy",
            "S4",
            "S1 reclaim with cumulative up-amount at least as large as down-amount.",
            "touch_reclaim",
            ma_periods=ma,
            filters=("amount_confirm",),
        ),
        StrategySpec(
            "s4_reclaim_amount_acceleration",
            "S4",
            "S1 reclaim with a causal minute amount curve accelerating against its prior baseline.",
            "touch_reclaim",
            ma_periods=ma,
            filters=("amount_acceleration_positive",),
        ),
        StrategySpec(
            "s4_reclaim_vwap_support",
            "S4",
            "S1 reclaim while price has regained its causal session VWAP proxy.",
            "touch_reclaim",
            ma_periods=ma,
            filters=("vwap_supportive",),
        ),
        StrategySpec(
            "s4_reclaim_volume_normal",
            "S4",
            "S1 reclaim with neither extreme contraction nor abnormal amount expansion.",
            "touch_reclaim",
            ma_periods=ma,
            filters=("volume_normal",),
        ),
        StrategySpec(
            "s4_auction_confirmed_reclaim",
            "S4",
            "S1 reclaim after an opening-auction and first-minutes confirmation.",
            "touch_reclaim",
            ma_periods=ma,
            filters=("auction_confirmed",),
        ),
        StrategySpec(
            "fu_r3_core_pullback",
            "S3",
            "Fu-Ge R3 proxy: core pullback only with supportive market, sector and leader context.",
            "touch_reclaim",
            ma_periods=ma,
            filters=("market_supportive", "sector_strong", "leader_sync", "not_repeated_cross"),
            author_id="fu_ge_long_kong_long",
            hypothesis_id="FU-R3",
        ),
        StrategySpec(
            "ht_r3_core_pullback",
            "S3",
            "Hai-Tang R3 proxy: core pullback with sector breadth, leader sync and normal volume.",
            "touch_reclaim",
            ma_periods=ma,
            filters=("sector_strong", "leader_sync", "volume_normal"),
            author_id="hai_tang_jun",
            hypothesis_id="HT-R3",
        ),
        StrategySpec(
            "cs_r3_leader_confirmation",
            "S3",
            "Chong-Sheng R3 proxy: require core/sector confirmation before following a reclaim.",
            "touch_reclaim",
            ma_periods=ma,
            filters=("leader_sync", "sector_strong"),
            author_id="chong_sheng_san_hu_yi_ge",
            hypothesis_id="CS-R3",
        ),
        StrategySpec(
            "ng_r1_auction_reclaim",
            "S3",
            "Niu-Ge R1/R2 proxy: auction confirmation followed by a causal MA reclaim.",
            "touch_reclaim",
            ma_periods=ma,
            filters=("auction_confirmed", "market_supportive"),
            author_id="niu_ge_xun_shi",
            hypothesis_id="NG-R1",
        ),
        StrategySpec(
            "ng_r3_volume_quality",
            "S4",
            "Niu-Ge R3 proxy: normal volume quality and a non-repeated MA interaction.",
            "touch_reclaim",
            ma_periods=ma,
            filters=("volume_normal", "not_repeated_cross"),
            author_id="niu_ge_xun_shi",
            hypothesis_id="NG-R3",
        ),
        StrategySpec(
            "s3_market_regime_gate",
            "S3",
            "Reserved for a point-in-time market-regime gate; source fields are not in the MA table.",
            "touch_reclaim",
            ma_periods=ma,
            implemented=False,
            required_columns=("market_regime",),
        ),
        StrategySpec(
            "s3_sector_sync_gate",
            "S3",
            "Reserved for point-in-time sector breadth and leader synchronisation fields.",
            "touch_reclaim",
            ma_periods=ma,
            implemented=False,
            required_columns=("sector_breadth", "sector_leader_sync"),
        ),
    )
    selected = tuple(spec for spec in specs if include_unavailable or spec.implemented)
    for spec in selected:
        spec.validate()
    return selected


def strategy_spec(strategy_id: str) -> StrategySpec:
    for spec in strategy_catalog(include_unavailable=True):
        if spec.strategy_id == str(strategy_id):
            return spec
    raise StrategyError(f"unknown_strategy:{strategy_id}")


def strategy_catalog_frame(*, include_unavailable: bool = True) -> pd.DataFrame:
    """Return a serialisable registry table for research records."""

    return pd.DataFrame([spec.as_dict() for spec in strategy_catalog(include_unavailable=include_unavailable)])


def _empty_signal_frame() -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype="object") for name in SIGNAL_COLUMNS})


def _require_state_columns(states: pd.DataFrame, required: Iterable[str]) -> None:
    missing = sorted(set(required).difference(states.columns))
    if missing:
        raise StrategyError(f"strategy_state_columns_missing:{','.join(missing)}")


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _stable_key_number(value: Any) -> int:
    """Map a key to a stable integer without relying on process-randomised hash()."""

    number = 0
    for byte in str(value).encode("utf-8"):
        number = (number * 131 + int(byte)) & 0xFFFFFFFF
    return number


def _stable_random_position(rows: pd.DataFrame, *, seed: int) -> int:
    """Choose a reproducible position independent of group iteration order."""

    if rows.empty:
        raise StrategyError("random_control_group_empty")
    first = rows.iloc[0]
    sequence = np.random.SeedSequence(
        [
            int(seed) & 0xFFFFFFFF,
            _stable_key_number(first["symbol"]),
            _stable_key_number(first["trade_date"]),
            int(first["sixty_minute_bucket"]) & 0xFFFFFFFF,
            int(first["ma_period"]) & 0xFFFFFFFF,
        ]
    )
    return int(np.random.default_rng(sequence).integers(0, len(rows)))


def _filter_passes(row: pd.Series, spec: StrategySpec) -> bool:
    for name in spec.filters:
        if name == "positive_slope" and (
            not _finite(row["prior_ma_slope_bps"]) or float(row["prior_ma_slope_bps"]) <= 0.0
        ):
            return False
        if name == "bullish_stack" and not bool(row["bullish_ma_stack"]):
            return False
        if name == "first_touch" and int(row["prior_true_touch_count_window"]) != 0:
            return False
        if name == "amount_confirm":
            ratio = row["up_down_amount_ratio"]
            if not _finite(ratio) or float(ratio) < 1.0:
                return False
        if name in {
            "recent_high_breakout",
            "prior_acceleration",
            "breakout_recent",
            "daily_trend_positive",
            "market_supportive",
            "sector_strong",
            "leader_sync",
            "auction_confirmed",
            "volume_normal",
            "not_repeated_cross",
            "vwap_supportive",
            "amount_acceleration_positive",
        }:
            value = row.get(name)
            if pd.isna(value) or not bool(value):
                return False
        if name not in _FILTER_COLUMNS:
            raise StrategyError(f"unknown_strategy_filter:{name}")
    return True


def _causal_signal_candidates(rows: pd.DataFrame, spec: StrategySpec, *, random_seed: int = 7):
    """Yield causal candidates in time order, one per trigger episode.

    Filters are evaluated by the caller.  Yielding episode candidates instead
    of returning only the first base-rule match lets a filtered rule reject an
    early reclaim and still evaluate a later, independent reclaim.
    """

    rule = spec.signal_rule
    if rule == "random_minute":
        if len(rows):
            yield _stable_random_position(rows, seed=int(random_seed)), "random_minute"
        return

    if rule == "strong_without_ma":
        close = pd.to_numeric(rows["adjusted_close"], errors="coerce").to_numpy(dtype=float)
        previous = pd.to_numeric(rows["previous_hour_high_adjusted"], errors="coerce").to_numpy(dtype=float)
        for position in np.flatnonzero(np.isfinite(close) & np.isfinite(previous) & (close >= previous)):
            yield int(position), "strong_without_ma"
        return

    if rule == "distance_only":
        distance = pd.to_numeric(rows["close_to_intersection_bps"], errors="coerce").abs().to_numpy(dtype=float)
        for position in np.flatnonzero(np.isfinite(distance) & (distance <= float(spec.near_touch_bps))):
            yield int(position), "distance_only"
        return

    if rule == "touch_immediate":
        touched = rows["touched_now"].fillna(False).to_numpy(dtype=bool)
        for position in np.flatnonzero(touched):
            yield int(position), "touch_immediate"
        return

    touched_values = rows["touched_now"].fillna(False).to_numpy(dtype=bool)
    below_values = rows["close_below_intersection"].fillna(False).to_numpy(dtype=bool)

    if rule == "touch_reclaim":
        trigger_seen = False
        for position, (touched, below) in enumerate(zip(touched_values, below_values, strict=True)):
            # Keep the episode open after the first causal reclaim.  This lets
            # a later minute satisfy a directional/volume filter without
            # requiring a second artificial touch.  The caller still emits at
            # most one signal per stock/hour/MA group.
            trigger_seen = trigger_seen or touched or below
            if trigger_seen and not below:
                yield position, "touch_reclaim"
        return

    if rule == "near_reversal":
        distances = pd.to_numeric(rows["range_distance_to_intersection_bps"], errors="coerce").to_numpy(dtype=float)
        closes = pd.to_numeric(rows["adjusted_close"], errors="coerce").to_numpy(dtype=float)
        near_seen = False
        for position, (touched, below, distance) in enumerate(
            zip(touched_values, below_values, distances, strict=True)
        ):
            near = not touched and np.isfinite(distance) and abs(float(distance)) <= float(spec.near_touch_bps)
            near_seen = near_seen or near
            rising = position > 0 and np.isfinite(closes[position]) and np.isfinite(closes[position - 1]) and closes[position] > closes[position - 1]
            if near_seen and not below and rising:
                yield position, "near_reversal"
                near_seen = False
        return

    if rule == "break_reclaim_stable":
        trigger_seen = False
        stable_run = 0
        for position, (touched, below) in enumerate(zip(touched_values, below_values, strict=True)):
            trigger_seen = trigger_seen or touched or below
            if trigger_seen and not below:
                stable_run += 1
                if stable_run >= int(spec.stable_minutes):
                    yield position, f"break_reclaim_stable{int(spec.stable_minutes)}"
                    # Do not emit every subsequent bar in the same run.
                    trigger_seen = False
                    stable_run = 0
            else:
                stable_run = 0
        return

    raise StrategyError(f"unknown_strategy_rule:{rule}")


def _causal_signal_position(rows: pd.DataFrame, spec: StrategySpec) -> tuple[int, str] | None:
    """Find the first causal candidate using only the current prefix."""

    return next(_causal_signal_candidates(rows, spec), None)


def _signal_from_mapping(row: Mapping[str, Any], spec: StrategySpec, trigger: str) -> dict[str, Any]:
    symbol = str(row["symbol"])
    trade_date = str(row["trade_date"])
    bar_time = str(row["bar_time"])
    period = int(row["ma_period"])
    bucket = int(row["sixty_minute_bucket"])
    signal_id = f"{spec.strategy_id}|{symbol}|{trade_date}|{bucket}|{period}|{bar_time}"
    signal: dict[str, Any] = {
        "signal_id": signal_id,
        "strategy_id": spec.strategy_id,
        "strategy_family": spec.family,
        "symbol": symbol,
        "signal_date": trade_date,
        "signal_time": bar_time,
        "entry_time": None,
        "sixty_minute_bucket": bucket,
        "ma_period": period,
        "hour_sequence": int(row["hour_sequence"]),
        "event_trigger": trigger,
        "causal_only": bool(spec.causal),
        "diagnostic_only": not bool(spec.executable),
        "signal_executable": bool(spec.executable),
        "reference_signal_id": None,
        "reference_symbol": None,
        "liquidity_match_ratio": np.nan,
        "close_to_intersection_bps": float(row["close_to_intersection_bps"]),
        "prior_ma_slope_bps": float(row["prior_ma_slope_bps"]),
        "ma_alignment_score": float(row["ma_alignment_score"]),
        "prior_true_touch_count_window": int(row["prior_true_touch_count_window"]),
        "up_down_amount_ratio": float(row["up_down_amount_ratio"]),
        "bullish_ma_stack": bool(row["bullish_ma_stack"]),
        "previous_hour_close_adjusted": float(row["previous_hour_close_adjusted"]),
        "previous_hour_high_adjusted": (
            float(row["previous_hour_high_adjusted"])
            if _finite(row.get("previous_hour_high_adjusted"))
            else np.nan
        ),
        "signal_adjusted_close": float(row["adjusted_close"]),
        "signal_amount": float(row["amount"]),
        "author_id": spec.author_id,
        "hypothesis_id": spec.hypothesis_id,
    }
    # Preserve causal context when the enriched full-universe runner supplies
    # it; the minimal MA pilot remains compatible without these optional fields.
    for name in SIGNAL_COLUMNS:
        if name in signal or name not in row:
            continue
        value = row[name]
        if pd.isna(value):
            signal[name] = None
        elif isinstance(value, (np.bool_, bool)):
            signal[name] = bool(value)
        elif isinstance(value, (np.integer, int)):
            signal[name] = int(value)
        elif isinstance(value, (np.floating, float)):
            signal[name] = float(value)
        else:
            signal[name] = value
    return signal


def _signal_from_series(row: pd.Series, spec: StrategySpec, trigger: str) -> dict[str, Any]:
    return _signal_from_mapping(row, spec, trigger)


def _signal_row(rows: pd.DataFrame, position: int, spec: StrategySpec, trigger: str) -> dict[str, Any]:
    return _signal_from_series(rows.iloc[int(position)], spec, trigger)


def _iter_period_groups(states: pd.DataFrame, periods: Sequence[int]):
    selected = states.loc[states["ma_period"].isin([int(value) for value in periods])]
    return selected.groupby(list(EVENT_KEY_COLUMNS), sort=False, observed=True)


def _posthoc_signals(states: pd.DataFrame, spec: StrategySpec) -> list[dict[str, Any]]:
    events = summarize_ma_events(states)
    if events.empty:
        return []
    available_periods = {int(value) for value in states["ma_period"].dropna().unique()}
    periods = tuple(period for period in spec.ma_periods if int(period) in available_periods)
    selected = events.loc[
        events["event_kind"].eq("posthoc_catchup") & events["ma_period"].isin(periods)
    ]
    rows: list[dict[str, Any]] = []
    for event in selected.itertuples(index=False):
        group = states.loc[
            (states["symbol"].astype(str) == str(event.symbol))
            & (states["trade_date"].astype(str) == str(event.trade_date))
            & (states["sixty_minute_bucket"] == int(event.sixty_minute_bucket))
            & (states["ma_period"] == int(event.ma_period))
        ].sort_values(["session_minute_ordinal", "bar_time"], kind="stable")
        if group.empty:
            continue
        # The catch-up is only knowable after the hour closes. Anchor its
        # diagnostic timestamp at the final bar instead of backdating it to
        # the visually attractive low/reference minute.
        final_row = group.tail(1).reset_index(drop=True)
        rows.append(_signal_row(final_row, 0, spec, "posthoc_catchup"))
    return rows


def build_strategy_signals(
    states: pd.DataFrame,
    *,
    strategy_ids: Sequence[str] | None = None,
    include_diagnostic: bool = False,
    random_seed: int = 7,
) -> pd.DataFrame:
    """Build one first signal per stock/hour/MA group for each rule.

    The input must be the causal state table from minute_ma. The output is an
    event signal table; it does not claim that an order was filled. The
    event-study module resolves the next-minute entry separately.
    """

    if states.empty:
        return _empty_signal_frame()
    catalog = {spec.strategy_id: spec for spec in strategy_catalog(include_unavailable=True)}
    selected_ids = tuple(strategy_ids) if strategy_ids is not None else tuple(
        spec.strategy_id
        for spec in strategy_catalog(include_unavailable=False)
        if (include_diagnostic or spec.executable) and not spec.reference_control
    )
    unknown = sorted(set(selected_ids).difference(catalog))
    if unknown:
        raise StrategyError(f"unknown_strategy:{','.join(unknown)}")
    selected_specs = [catalog[strategy_id] for strategy_id in selected_ids]
    required = set(_BASE_COLUMNS)
    for spec in selected_specs:
        required.update(spec.required_columns)
        for filter_name in spec.filters:
            required.update(_FILTER_COLUMNS.get(filter_name, set()))
        spec.validate()
        if spec.reference_control:
            raise StrategyError(f"strategy_requires_reference_signals:{spec.strategy_id}")
        if not spec.implemented:
            raise StrategyError(f"strategy_not_implemented:{spec.strategy_id}")
    _require_state_columns(states, required)

    output: list[dict[str, Any]] = []
    available_periods = {int(value) for value in states["ma_period"].dropna().unique()}
    sorted_states = states.sort_values(
        [
            "symbol",
            "trade_date",
            "sixty_minute_bucket",
            "ma_period",
            "session_minute_ordinal",
            "bar_time",
        ],
        kind="stable",
    ).reset_index(drop=True)
    groups_by_period = {
        int(period): tuple(
            group
            for _, group in sorted_states.loc[sorted_states["ma_period"].eq(int(period))].groupby(
                list(EVENT_KEY_COLUMNS), sort=False, observed=True
            )
        )
        for period in sorted(available_periods)
    }
    for spec in selected_specs:
        if spec.signal_rule == "posthoc_catchup":
            output.extend(_posthoc_signals(states, spec))
            continue
        periods = tuple(period for period in spec.ma_periods if int(period) in available_periods)
        groups = (group for period in periods for group in groups_by_period.get(int(period), ()))
        for group in groups:
            rows = group.reset_index(drop=True)
            if rows.empty:
                continue
            for position, trigger in _causal_signal_candidates(rows, spec, random_seed=int(random_seed)):
                row = rows.iloc[int(position)]
                if not _filter_passes(row, spec):
                    continue
                output.append(_signal_row(rows, position, spec, trigger))
                break

    if not output:
        return _empty_signal_frame()
    result = pd.DataFrame(output)
    for name in SIGNAL_COLUMNS:
        if name not in result.columns:
            result[name] = None
    result = result.loc[:, list(SIGNAL_COLUMNS)]
    result.sort_values(
        ["signal_date", "signal_time", "strategy_id", "symbol", "ma_period"],
        inplace=True,
        kind="stable",
        ignore_index=True,
    )
    if result["signal_id"].duplicated().any():
        raise StrategyError("strategy_signal_id_duplicate")
    return result


def build_strategy_signals_many(
    states: pd.DataFrame,
    *,
    strategy_ids: Sequence[str],
    include_diagnostic: bool = False,
    random_seed: int = 7,
) -> pd.DataFrame:
    """Build several rules while scanning each stock/hour group once.

    The ordinary public builder is intentionally simple and useful for small
    probes.  Full-universe runs often evaluate the same groups against many
    independent hypotheses; caching the base-rule candidate positions keeps
    that work bounded without sharing any future outcome fields.
    """

    selected_ids = tuple(str(value) for value in strategy_ids)
    if not selected_ids:
        return _empty_signal_frame()
    if len(selected_ids) == 1:
        return build_strategy_signals(
            states,
            strategy_ids=selected_ids,
            include_diagnostic=include_diagnostic,
            random_seed=random_seed,
        )
    if states.empty:
        return _empty_signal_frame()
    catalog = {spec.strategy_id: spec for spec in strategy_catalog(include_unavailable=True)}
    unknown = sorted(set(selected_ids).difference(catalog))
    if unknown:
        raise StrategyError(f"unknown_strategy:{','.join(unknown)}")
    specs = [catalog[value] for value in selected_ids]
    required = set(_BASE_COLUMNS)
    for spec in specs:
        spec.validate()
        if spec.reference_control:
            raise StrategyError(f"strategy_requires_reference_signals:{spec.strategy_id}")
        if not spec.implemented:
            raise StrategyError(f"strategy_not_implemented:{spec.strategy_id}")
        required.update(spec.required_columns)
        for filter_name in spec.filters:
            required.update(_FILTER_COLUMNS.get(filter_name, set()))
    _require_state_columns(states, required)
    available_periods = {int(value) for value in states["ma_period"].dropna().unique()}
    sorted_states = states.sort_values(
        [
            "symbol",
            "trade_date",
            "sixty_minute_bucket",
            "ma_period",
            "session_minute_ordinal",
            "bar_time",
        ],
        kind="stable",
    ).reset_index(drop=True)
    group_items = []
    for period in sorted(available_periods):
        group_items.extend(
            (key, group.reset_index(drop=True))
            for key, group in sorted_states.loc[sorted_states["ma_period"].eq(period)].groupby(
                list(EVENT_KEY_COLUMNS), sort=False, observed=True
            )
        )
    rules = {spec.signal_rule for spec in specs if spec.signal_rule != "posthoc_catchup"}
    candidate_cache: dict[tuple[Any, ...], dict[str, list[tuple[int, str]]]] = {}
    for key, rows in group_items:
        cached: dict[str, list[tuple[int, str]]] = {}
        for rule in rules:
            rule_specs = [spec for spec in specs if spec.signal_rule == rule]
            if not rule_specs:
                continue
            # All specs using one rule share the same near-touch/stability
            # settings in the finite registry.  If a future version differs,
            # fall back to the ordinary candidate generator for that spec.
            first = rule_specs[0]
            same_settings = all(
                spec.near_touch_bps == first.near_touch_bps and spec.stable_minutes == first.stable_minutes
                for spec in rule_specs
            )
            if same_settings:
                cached[rule] = list(
                    _causal_signal_candidates(rows, first, random_seed=int(random_seed))
                )
        candidate_cache[key if isinstance(key, tuple) else (key,)] = cached
    output: list[dict[str, Any]] = []
    for spec in specs:
        if spec.signal_rule == "posthoc_catchup":
            output.extend(_posthoc_signals(states, spec))
            continue
        for key, rows in group_items:
            cache_key = key if isinstance(key, tuple) else (key,)
            candidates = candidate_cache.get(cache_key, {}).get(spec.signal_rule)
            if candidates is None:
                candidates = list(
                    _causal_signal_candidates(rows, spec, random_seed=int(random_seed))
                )
            for position, trigger in candidates:
                if _filter_passes(rows.iloc[int(position)], spec):
                    output.append(_signal_row(rows, position, spec, trigger))
                    break
    if not output:
        return _empty_signal_frame()
    result = pd.DataFrame(output)
    for name in SIGNAL_COLUMNS:
        if name not in result.columns:
            result[name] = None
    result = result.loc[:, list(SIGNAL_COLUMNS)]
    result.sort_values(
        ["signal_date", "signal_time", "strategy_id", "symbol", "ma_period"],
        inplace=True,
        kind="stable",
        ignore_index=True,
    )
    if result["signal_id"].duplicated().any():
        raise StrategyError("strategy_signal_id_duplicate")
    return result


def build_strategy_signals_vectorized(
    states: pd.DataFrame,
    *,
    strategy_ids: Sequence[str],
    include_diagnostic: bool = False,
    random_seed: int = 7,
) -> pd.DataFrame:
    """Build finite rules with one vectorised scan of the state table.

    This is the production path for a full-universe day.  Base event masks are
    computed once per row and filters are applied before selecting the first
    row in each stock/hour/MA group.  No outcome or end-of-hour field is read.
    """

    selected_ids = tuple(str(value) for value in strategy_ids)
    if not selected_ids or states.empty:
        return _empty_signal_frame()
    catalog = {spec.strategy_id: spec for spec in strategy_catalog(include_unavailable=True)}
    unknown = sorted(set(selected_ids).difference(catalog))
    if unknown:
        raise StrategyError(f"unknown_strategy:{','.join(unknown)}")
    specs = [catalog[value] for value in selected_ids]
    regular_specs = [spec for spec in specs if spec.signal_rule != "posthoc_catchup"]
    diagnostic_specs = [spec for spec in specs if spec.signal_rule == "posthoc_catchup"]
    required = set(_BASE_COLUMNS)
    for spec in specs:
        spec.validate()
        if spec.reference_control:
            raise StrategyError(f"strategy_requires_reference_signals:{spec.strategy_id}")
        if not spec.implemented:
            raise StrategyError(f"strategy_not_implemented:{spec.strategy_id}")
        required.update(spec.required_columns)
        for filter_name in spec.filters:
            required.update(_FILTER_COLUMNS.get(filter_name, set()))
    _require_state_columns(states, required)
    ordered = states.sort_values(
        [
            "symbol",
            "trade_date",
            "sixty_minute_bucket",
            "ma_period",
            "session_minute_ordinal",
            "bar_time",
        ],
        kind="stable",
    ).reset_index(drop=True)
    group_columns = list(EVENT_KEY_COLUMNS)
    groups = ordered.groupby(group_columns, sort=False, observed=True)
    group_id = groups.ngroup()
    group_id_values = group_id.to_numpy(dtype=np.int64, copy=False)
    touched = ordered["touched_now"].fillna(False).astype(bool)
    below = ordered["close_below_intersection"].fillna(False).astype(bool)
    trigger = (touched | below).groupby(group_id, sort=False).cummax().astype(bool)
    close = pd.to_numeric(ordered["adjusted_close"], errors="coerce")
    previous_close = close.groupby(group_id, sort=False).shift(1)
    rising = close.gt(previous_close).fillna(False)
    distance = pd.to_numeric(ordered["range_distance_to_intersection_bps"], errors="coerce").abs()
    max_near_touch = float(max((spec.near_touch_bps for spec in specs), default=0.0))
    group_start_mask = np.r_[True, group_id_values[1:] != group_id_values[:-1]]
    near = (~touched) & distance.le(max_near_touch)

    def near_reversal_candidates(near_values: np.ndarray) -> np.ndarray:
        """Mark the first rising close in each near-touch episode."""

        result = np.zeros(len(near_values), dtype=bool)
        near_array = near_values.astype(bool, copy=False)
        below_array = below.to_numpy(dtype=bool, copy=False)
        rising_array = rising.to_numpy(dtype=bool, copy=False)
        starts = np.flatnonzero(group_start_mask)
        ends = np.r_[starts[1:], len(ordered)]
        for start, end in zip(starts, ends, strict=True):
            seen = False
            for position in range(int(start), int(end)):
                seen = seen or bool(near_array[position])
                if seen and not below_array[position] and rising_array[position]:
                    result[position] = True
                    seen = False
        return result

    base_masks: dict[str, pd.Series] = {
        "touch_immediate": touched,
        "touch_reclaim": trigger & ~below,
        "near_reversal": pd.Series(near_reversal_candidates(near.to_numpy()), index=ordered.index),
        "strong_without_ma": close.ge(pd.to_numeric(ordered["previous_hour_high_adjusted"], errors="coerce")).fillna(False),
        "distance_only": pd.to_numeric(ordered["close_to_intersection_bps"], errors="coerce").abs().le(max_near_touch).fillna(False),
    }
    if any(spec.signal_rule == "break_reclaim_stable" for spec in regular_specs):
        positions = np.arange(len(ordered), dtype=np.int64)
        below_values = below.to_numpy(dtype=bool, copy=False)
        reset_anchor = np.where(
            below_values,
            positions,
            np.where(group_start_mask, positions - 1, -1),
        )
        last_reset = np.maximum.accumulate(reset_anchor)
        stable_run = pd.Series(positions - last_reset, index=ordered.index, dtype="int64")
        base_masks["break_reclaim_stable"] = stable_run.ge(
            min(max(1, int(spec.stable_minutes)) for spec in regular_specs if spec.signal_rule == "break_reclaim_stable")
        ) & trigger & ~below
    if any(spec.signal_rule == "random_minute" for spec in regular_specs):
        group_starts = np.flatnonzero(group_start_mask)
        group_ends = np.r_[group_starts[1:], len(ordered)]
        group_sizes = group_ends - group_starts
        first_rows = ordered.iloc[group_starts]
        offsets = np.fromiter(
            (
                int(
                    np.random.default_rng(
                        np.random.SeedSequence(
                            [
                                int(random_seed) & 0xFFFFFFFF,
                                _stable_key_number(symbol),
                                _stable_key_number(trade_date),
                                int(bucket) & 0xFFFFFFFF,
                                int(period) & 0xFFFFFFFF,
                            ]
                        )
                    ).integers(0, int(size))
                )
                for symbol, trade_date, bucket, period, size in zip(
                    first_rows["symbol"],
                    first_rows["trade_date"],
                    first_rows["sixty_minute_bucket"],
                    first_rows["ma_period"],
                    group_sizes,
                    strict=True,
                )
            ),
            dtype=np.int64,
            count=len(group_starts),
        )
        random_mask_values = np.zeros(len(ordered), dtype=bool)
        random_mask_values[group_starts + offsets] = True
        random_mask = pd.Series(random_mask_values, index=ordered.index)
        base_masks["random_minute"] = random_mask

    filter_arrays: dict[str, np.ndarray] = {
        "positive_slope": pd.to_numeric(ordered["prior_ma_slope_bps"], errors="coerce").gt(0).fillna(False).to_numpy(dtype=bool),
        "bullish_stack": ordered["bullish_ma_stack"].fillna(False).astype(bool).to_numpy(dtype=bool),
        "first_touch": pd.to_numeric(ordered["prior_true_touch_count_window"], errors="coerce").eq(0).fillna(False).to_numpy(dtype=bool),
        "amount_confirm": pd.to_numeric(ordered["up_down_amount_ratio"], errors="coerce").ge(1.0).fillna(False).to_numpy(dtype=bool),
    }
    for name in _FILTER_COLUMNS:
        if name not in filter_arrays and name in ordered.columns:
            filter_arrays[name] = ordered[name].fillna(False).astype(bool).to_numpy(dtype=bool)

    output_frames: list[pd.DataFrame] = []

    def signal_frame(selected_positions: np.ndarray, spec: StrategySpec) -> pd.DataFrame:
        selected = ordered.iloc[selected_positions].copy()
        result = pd.DataFrame(index=selected.index)
        result["strategy_id"] = spec.strategy_id
        result["strategy_family"] = spec.family
        result["symbol"] = selected["symbol"].astype(str)
        result["signal_date"] = selected["trade_date"].astype(str)
        result["signal_time"] = selected["bar_time"].astype(str)
        result["entry_time"] = None
        result["sixty_minute_bucket"] = selected["sixty_minute_bucket"].astype(int)
        result["ma_period"] = selected["ma_period"].astype(int)
        result["hour_sequence"] = selected["hour_sequence"].astype(int)
        result["event_trigger"] = spec.signal_rule
        result["causal_only"] = bool(spec.causal)
        result["diagnostic_only"] = not bool(spec.executable)
        result["signal_executable"] = bool(spec.executable)
        result["reference_signal_id"] = None
        result["reference_symbol"] = None
        result["liquidity_match_ratio"] = np.nan
        direct = {
            "close_to_intersection_bps": "close_to_intersection_bps",
            "prior_ma_slope_bps": "prior_ma_slope_bps",
            "ma_alignment_score": "ma_alignment_score",
            "prior_true_touch_count_window": "prior_true_touch_count_window",
            "up_down_amount_ratio": "up_down_amount_ratio",
            "bullish_ma_stack": "bullish_ma_stack",
            "previous_hour_close_adjusted": "previous_hour_close_adjusted",
            "previous_hour_high_adjusted": "previous_hour_high_adjusted",
            "signal_adjusted_close": "adjusted_close",
            "signal_amount": "amount",
        }
        for target, source in direct.items():
            result[target] = selected[source].to_numpy()
        result["author_id"] = spec.author_id
        result["hypothesis_id"] = spec.hypothesis_id
        reserved = set(result.columns) | {"signal_id"}
        for name in SIGNAL_COLUMNS:
            if name in reserved:
                continue
            result[name] = selected[name].to_numpy() if name in selected.columns else None
        result["signal_id"] = (
            spec.strategy_id
            + "|"
            + result["symbol"]
            + "|"
            + result["signal_date"]
            + "|"
            + result["sixty_minute_bucket"].astype(str)
            + "|"
            + result["ma_period"].astype(str)
            + "|"
            + result["signal_time"]
        )
        return result.loc[:, list(SIGNAL_COLUMNS)].reset_index(drop=True)

    for spec in regular_specs:
        candidate = base_masks.get(spec.signal_rule)
        if candidate is None:
            raise StrategyError(f"unknown_strategy_rule:{spec.signal_rule}")
        # A rule with a narrower near-touch threshold gets its own mask.
        if spec.signal_rule in {"near_reversal", "distance_only"} and float(spec.near_touch_bps) != float(max((item.near_touch_bps for item in specs), default=0.0)):
            threshold = float(spec.near_touch_bps)
            narrow = distance.le(threshold).fillna(False)
            if spec.signal_rule == "distance_only":
                candidate = pd.to_numeric(ordered["close_to_intersection_bps"], errors="coerce").abs().le(threshold).fillna(False)
            else:
                candidate = pd.Series(near_reversal_candidates((~touched & distance.le(threshold)).to_numpy()), index=ordered.index)
                candidate &= narrow
        # ``Series.to_numpy(copy=False)`` can expose the mask backing array.
        # Filters are strategy-specific; mutating that view would silently
        # narrow the shared base mask for every later strategy in the loop.
        selected_mask = candidate.to_numpy(dtype=bool, copy=True)
        for name in spec.filters:
            try:
                selected_mask &= filter_arrays[name]
            except KeyError as exc:
                raise StrategyError(f"strategy_filter_column_missing:{name}") from exc
        candidate_positions = np.flatnonzero(selected_mask)
        if len(candidate_positions):
            _, first_positions = np.unique(
                group_id_values[candidate_positions], return_index=True
            )
            selected_positions = candidate_positions[np.sort(first_positions)]
            output_frames.append(signal_frame(selected_positions, spec))
    diagnostic_rows: list[dict[str, Any]] = []
    for spec in diagnostic_specs:
        diagnostic_rows.extend(_posthoc_signals(states, spec))
    if diagnostic_rows:
        diagnostic = pd.DataFrame(diagnostic_rows)
        for name in SIGNAL_COLUMNS:
            if name not in diagnostic.columns:
                diagnostic[name] = None
        output_frames.append(diagnostic.loc[:, list(SIGNAL_COLUMNS)])
    if not output_frames:
        return _empty_signal_frame()
    result = pd.concat(output_frames, ignore_index=True)
    result.sort_values(
        ["signal_date", "signal_time", "strategy_id", "symbol", "ma_period"],
        inplace=True,
        kind="stable",
        ignore_index=True,
    )
    if result["signal_id"].duplicated().any():
        raise StrategyError("strategy_signal_id_duplicate")
    return result


def build_matched_random_control(
    states: pd.DataFrame,
    reference_signals: pd.DataFrame,
    *,
    strategy_id: str = "s0_random_matched",
    random_seed: int = 7,
) -> pd.DataFrame:
    """Choose one random minute in each reference signal's same-stock hour.

    This is a timing control for asking whether a signal's exact minute adds
    value within the same stock/hour. It is deliberately distinct from the
    cross-sectional liquidity control built by
    :func:`build_liquidity_matched_control`.
    """

    spec = strategy_spec(strategy_id)
    if not spec.control or spec.signal_rule != "random_minute":
        raise StrategyError("matched_control_strategy_invalid")
    _require_state_columns(states, _BASE_COLUMNS)
    required_ref = {"symbol", "signal_date", "sixty_minute_bucket", "ma_period"}
    missing = sorted(required_ref.difference(reference_signals.columns))
    if missing:
        raise StrategyError(f"strategy_reference_columns_missing:{','.join(missing)}")
    if reference_signals.empty:
        return _empty_signal_frame()
    rows: list[dict[str, Any]] = []
    for ref in reference_signals.itertuples(index=False):
        group = states.loc[
            (states["symbol"].astype(str) == str(ref.symbol))
            & (states["trade_date"].astype(str) == str(ref.signal_date))
            & (states["sixty_minute_bucket"] == int(ref.sixty_minute_bucket))
            & (states["ma_period"] == int(ref.ma_period))
        ].sort_values(["session_minute_ordinal", "bar_time"], kind="stable")
        if group.empty:
            continue
        normalised = group.reset_index(drop=True)
        position = _stable_random_position(normalised, seed=int(random_seed))
        rows.append(_signal_row(normalised, position, spec, "matched_random_minute"))
    if not rows:
        return _empty_signal_frame()
    result = pd.DataFrame(rows)
    for name in SIGNAL_COLUMNS:
        if name not in result.columns:
            result[name] = None
    result = result.loc[:, list(SIGNAL_COLUMNS)]
    result.sort_values(
        ["signal_date", "signal_time", "symbol", "ma_period"],
        inplace=True,
        kind="stable",
        ignore_index=True,
    )
    return result


def _normalise_strategy_dates(values: pd.Series) -> pd.Series:
    raw = values.astype("string").str.strip()
    if bool(raw.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False).all()):
        return raw
    compact = raw.str.fullmatch(r"\d{8}", na=False)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if compact.any():
        parsed.loc[compact] = pd.to_datetime(raw.loc[compact], format="%Y%m%d", errors="coerce")
    if (~compact).any():
        parsed.loc[~compact] = pd.to_datetime(raw.loc[~compact], errors="coerce")
    if parsed.isna().any():
        bad = raw.loc[parsed.isna()].iloc[0]
        raise StrategyError(f"strategy_trade_date_invalid:{bad!r}")
    return parsed.dt.strftime("%Y-%m-%d")


def _normalise_strategy_times(values: pd.Series) -> pd.Series:
    raw = values.astype("string").str.strip()
    if bool(raw.str.fullmatch(r"\d{9}", na=False).all()):
        return raw
    return raw.map(normalize_bar_time).astype("string")


def attach_prior_daily_liquidity(
    states: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    lookback_days: int = 20,
    column: str = DEFAULT_LIQUIDITY_COLUMN,
) -> pd.DataFrame:
    """Attach a point-in-time turnover proxy to minute states.

    The value is the median daily traded amount over the preceding
    ``lookback_days`` complete sessions, shifted by one session.  It therefore
    cannot use the signal day's amount.  Incomplete sessions are excluded from
    the history rather than silently treated as low liquidity.
    """

    if (
        isinstance(lookback_days, bool)
        or not isinstance(lookback_days, (int, np.integer))
        or int(lookback_days) < 1
    ):
        raise StrategyError("liquidity_lookback_days_invalid")
    if not column or str(column).strip() != str(column):
        raise StrategyError("liquidity_column_invalid")
    _require_state_columns(states, ("symbol", "trade_date"))
    required = {"symbol", "trade_date", "bar_time", "amount"}
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise StrategyError(f"liquidity_bar_columns_missing:{','.join(missing)}")
    if column in states.columns:
        raise StrategyError(f"liquidity_column_already_present:{column}")

    frame = bars.loc[:, ["symbol", "trade_date", "bar_time", "amount"]].copy()
    frame["symbol"] = frame["symbol"].astype(str).str.strip()
    if frame["symbol"].eq("").any() or frame["symbol"].str.lower().isin({"nan", "none", "<na>"}).any():
        raise StrategyError("liquidity_symbol_invalid")
    frame["trade_date"] = _normalise_strategy_dates(frame["trade_date"])
    frame["bar_time"] = frame["bar_time"].map(normalize_bar_time)
    frame["session_minute_ordinal"] = frame["bar_time"].map(session_minute_ordinal)
    frame = frame.loc[frame["session_minute_ordinal"].notna()].copy()
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
    valid = np.isfinite(frame["amount"]) & frame["amount"].ge(0)
    if not bool(valid.all()):
        raise StrategyError("liquidity_amount_invalid")
    if frame.duplicated(["symbol", "trade_date", "bar_time"]).any():
        raise StrategyError("liquidity_duplicate_minute_key")

    daily = (
        frame.groupby(["symbol", "trade_date"], sort=True, observed=True)
        .agg(
            daily_amount=("amount", "sum"),
            bar_count=("session_minute_ordinal", "size"),
            distinct_minute_count=("session_minute_ordinal", "nunique"),
        )
        .reset_index()
    )
    daily = daily.loc[
        daily["bar_count"].eq(240) & daily["distinct_minute_count"].eq(240)
    ].copy()
    if daily.empty:
        raise StrategyError("liquidity_complete_daily_history_empty")
    daily.sort_values(["symbol", "trade_date"], inplace=True, kind="stable", ignore_index=True)
    prior = pd.Series(np.nan, index=daily.index, dtype="float64")
    for _, indices in daily.groupby("symbol", sort=False).groups.items():
        values = daily.loc[indices, "daily_amount"]
        prior.loc[indices] = values.rolling(
            int(lookback_days), min_periods=int(lookback_days)
        ).median().shift(1)
    daily[column] = prior
    liquidity = daily.loc[:, ["symbol", "trade_date", column]]
    result = states.copy()
    result["trade_date"] = _normalise_strategy_dates(result["trade_date"])
    result = result.merge(
        liquidity,
        on=["symbol", "trade_date"],
        how="left",
        validate="many_to_one",
    )
    result.attrs.update(
        {
            "liquidity_column": str(column),
            "liquidity_lookback_days": int(lookback_days),
            "liquidity_complete_stock_days": int(len(daily)),
        }
    )
    return result


def _stable_choice_position(key: str, *, seed: int, size: int) -> int:
    if size <= 0:
        raise StrategyError("liquidity_candidate_pool_empty")
    sequence = np.random.SeedSequence(
        [int(seed) & 0xFFFFFFFF, _stable_key_number(key)]
    )
    return int(np.random.default_rng(sequence).integers(0, int(size)))


def build_liquidity_matched_control(
    states: pd.DataFrame,
    reference_signals: pd.DataFrame,
    *,
    strategy_id: str = "s0_liquidity_matched",
    liquidity_column: str = DEFAULT_LIQUIDITY_COLUMN,
    liquidity_band_ratio: float = 2.0,
    nearest_candidates: int = 5,
    random_seed: int = 7,
) -> pd.DataFrame:
    """Choose a same-minute stock with comparable *prior* liquidity.

    A reference signal is matched on date, minute, 60-minute bucket and MA
    period.  Candidates must be a different symbol and have a prior-turnover
    ratio within ``liquidity_band_ratio``.  The nearest
    ``nearest_candidates`` are ordered by absolute log-ratio and one is chosen
    deterministically at random.  The reference id is carried on the control
    row so outcomes can be paired even though the symbols differ.

    The function only sees the supplied state universe.  Callers must declare
    that universe separately; this is not a survivorship-free universe builder.
    """

    spec = strategy_spec(strategy_id)
    if not spec.reference_control or spec.signal_rule != "liquidity_matched_stock":
        raise StrategyError("liquidity_control_strategy_invalid")
    if (
        not np.isfinite(float(liquidity_band_ratio))
        or float(liquidity_band_ratio) < 1.0
    ):
        raise StrategyError("liquidity_band_ratio_invalid")
    if (
        isinstance(nearest_candidates, bool)
        or not isinstance(nearest_candidates, (int, np.integer))
        or int(nearest_candidates) < 1
    ):
        raise StrategyError("liquidity_nearest_candidates_invalid")
    _require_state_columns(
        states,
        set(_BASE_COLUMNS) | {str(liquidity_column)},
    )
    required_ref = {
        "signal_id",
        "symbol",
        "signal_date",
        "signal_time",
        "sixty_minute_bucket",
        "ma_period",
    }
    missing = sorted(required_ref.difference(reference_signals.columns))
    if missing:
        raise StrategyError(f"strategy_reference_columns_missing:{','.join(missing)}")
    if reference_signals.empty:
        empty = _empty_signal_frame()
        empty.attrs.update({"reference_count": 0, "matched_count": 0, "unmatched_count": 0, "unmatched_reasons": {}})
        return empty
    if reference_signals["signal_id"].duplicated().any():
        raise StrategyError("strategy_reference_signal_id_duplicate")

    working = states.copy()
    working["symbol"] = working["symbol"].astype(str).str.strip()
    working["trade_date"] = _normalise_strategy_dates(working["trade_date"])
    working["bar_time"] = working["bar_time"].map(normalize_bar_time)
    working["ma_period"] = pd.to_numeric(working["ma_period"], errors="coerce")
    working["sixty_minute_bucket"] = pd.to_numeric(working["sixty_minute_bucket"], errors="coerce")
    working["liquidity_value"] = pd.to_numeric(working[liquidity_column], errors="coerce")
    # Only rows at reference timestamps can be selected.  Filtering before
    # grouping is material for a full-universe day (the state table otherwise
    # contains every minute of every stock).
    reference_dates = _normalise_strategy_dates(reference_signals["signal_date"])
    reference_times = reference_signals["signal_time"].map(normalize_bar_time)
    reference_keys = set(
        zip(
            reference_dates.astype(str),
            reference_times.astype(str),
            reference_signals["sixty_minute_bucket"].astype(int),
            reference_signals["ma_period"].astype(int),
            strict=False,
        )
    )
    working["_match_key"] = list(
        zip(
            working["trade_date"].astype(str),
            working["bar_time"].astype(str),
            working["sixty_minute_bucket"].astype(int),
            working["ma_period"].astype(int),
            strict=False,
        )
    )
    working = working.loc[working["_match_key"].isin(reference_keys)].copy()
    grouped = {
        key if isinstance(key, tuple) else (key,): group
        for key, group in working.groupby("_match_key", sort=False, observed=True)
    }

    rows: list[dict[str, Any]] = []
    unmatched = Counter()
    for reference, reference_date, reference_time in zip(
        reference_signals.itertuples(index=False),
        reference_dates.astype(str),
        reference_times.astype(str),
        strict=True,
    ):
        ref = reference._asdict()
        ref_id = str(ref["signal_id"])
        key = (
            str(reference_date),
            str(reference_time),
            int(ref["sixty_minute_bucket"]),
            int(ref["ma_period"]),
        )
        group = grouped.get(key)
        if group is None or group.empty:
            unmatched["same_minute_state_missing"] += 1
            continue
        ref_symbol = str(ref["symbol"]).strip()
        reference_rows = group.loc[group["symbol"].eq(ref_symbol)]
        if reference_rows.empty:
            unmatched["reference_state_missing"] += 1
            continue
        ref_liquidity = float(reference_rows.iloc[0]["liquidity_value"])
        if not np.isfinite(ref_liquidity) or ref_liquidity <= 0:
            unmatched["reference_liquidity_missing"] += 1
            continue
        symbols = group["symbol"].astype(str).to_numpy()
        liquidity = group["liquidity_value"].to_numpy(dtype=float)
        candidate_mask = (symbols != ref_symbol) & np.isfinite(liquidity) & (liquidity > 0)
        if not candidate_mask.any():
            unmatched["candidate_liquidity_missing"] += 1
            continue
        candidate_indices = np.flatnonzero(candidate_mask)
        ratios = liquidity[candidate_indices] / ref_liquidity
        band = float(liquidity_band_ratio)
        in_band = (ratios >= 1.0 / band) & (ratios <= band)
        candidate_indices = candidate_indices[in_band]
        ratios = ratios[in_band]
        if not len(candidate_indices):
            unmatched["no_candidate_in_liquidity_band"] += 1
            continue
        distances = np.abs(np.log(ratios))
        order = np.lexsort((symbols[candidate_indices], distances))[: int(nearest_candidates)]
        candidate_indices = candidate_indices[order]
        ratios = ratios[order]
        choice = _stable_choice_position(ref_id, seed=int(random_seed), size=len(candidate_indices))
        chosen_position = int(candidate_indices[choice])
        chosen = group.iloc[[chosen_position]].copy().reset_index(drop=True)
        signal = _signal_row(chosen, 0, spec, "liquidity_matched_stock")
        chosen_symbol = str(chosen.iloc[0]["symbol"])
        signal["signal_id"] = f"{strategy_id}|ref:{ref_id}|{chosen_symbol}"
        signal["reference_signal_id"] = ref_id
        signal["reference_symbol"] = ref_symbol
        signal["liquidity_match_ratio"] = float(liquidity[chosen_position] / ref_liquidity)
        rows.append(signal)

    if not rows:
        result = _empty_signal_frame()
    else:
        result = pd.DataFrame(rows)
        for name in SIGNAL_COLUMNS:
            if name not in result.columns:
                result[name] = None
        result = result.loc[:, list(SIGNAL_COLUMNS)]
        result.sort_values(
            ["signal_date", "signal_time", "strategy_id", "symbol", "ma_period", "reference_signal_id"],
            inplace=True,
            kind="stable",
            ignore_index=True,
        )
        if result["signal_id"].duplicated().any():
            raise StrategyError("liquidity_control_signal_id_duplicate")
    result.attrs.update(
        {
            "reference_count": int(len(reference_signals)),
            "matched_count": int(len(rows)),
            "unmatched_count": int(len(reference_signals) - len(rows)),
            "unmatched_reasons": dict(unmatched),
            "liquidity_column": str(liquidity_column),
            "liquidity_band_ratio": float(liquidity_band_ratio),
            "nearest_candidates": int(nearest_candidates),
        }
    )
    return result


def build_liquidity_matched_controls_many(
    states: pd.DataFrame,
    reference_signals: pd.DataFrame,
    *,
    strategy_id: str = "s0_liquidity_matched",
    liquidity_column: str = DEFAULT_LIQUIDITY_COLUMN,
    liquidity_band_ratio: float = 2.0,
    nearest_candidates: int = 5,
    random_seed: int = 7,
) -> pd.DataFrame:
    """Build one shared cross-sectional control for every reference signal key.

    Several rule variants can signal the same stock/minute/MA key.  Matching
    that key once and cloning the chosen control row keeps the comparison
    independent for each strategy without repeating the expensive candidate
    search.  Each clone retains its own ``reference_signal_id`` so
    :func:`compare_to_reference_control` can pair it unambiguously.
    """

    if reference_signals.empty:
        empty = _empty_signal_frame()
        empty.attrs.update({"reference_count": 0, "matched_count": 0, "unmatched_count": 0, "unmatched_reasons": {}})
        return empty
    spec = strategy_spec(strategy_id)
    if not spec.reference_control or spec.signal_rule != "liquidity_matched_stock":
        raise StrategyError("liquidity_control_strategy_invalid")
    if not np.isfinite(float(liquidity_band_ratio)) or float(liquidity_band_ratio) < 1.0:
        raise StrategyError("liquidity_band_ratio_invalid")
    if isinstance(nearest_candidates, bool) or not isinstance(nearest_candidates, (int, np.integer)) or int(nearest_candidates) < 1:
        raise StrategyError("liquidity_nearest_candidates_invalid")
    _require_state_columns(states, set(_BASE_COLUMNS) | {str(liquidity_column)})
    required = {"signal_id", "symbol", "signal_date", "signal_time", "sixty_minute_bucket", "ma_period"}
    missing = sorted(required.difference(reference_signals.columns))
    if missing:
        raise StrategyError(f"strategy_reference_columns_missing:{','.join(missing)}")
    if reference_signals["signal_id"].duplicated().any():
        raise StrategyError("strategy_reference_signal_id_duplicate")

    references = reference_signals.copy()
    references["symbol"] = references["symbol"].astype(str).str.strip()
    references["signal_date"] = _normalise_strategy_dates(references["signal_date"])
    references["signal_time"] = _normalise_strategy_times(references["signal_time"])
    references["sixty_minute_bucket"] = pd.to_numeric(references["sixty_minute_bucket"], errors="coerce").astype("int16")
    references["ma_period"] = pd.to_numeric(references["ma_period"], errors="coerce").astype("int16")
    control_columns = [
        "symbol",
        "signal_date",
        "signal_time",
        "sixty_minute_bucket",
        "ma_period",
    ]
    canonical = references.drop_duplicates(control_columns, keep="first")

    working = states.copy().reset_index(drop=True)
    working["symbol"] = working["symbol"].astype(str).str.strip()
    working["trade_date"] = _normalise_strategy_dates(working["trade_date"])
    working["bar_time"] = _normalise_strategy_times(working["bar_time"])
    working["sixty_minute_bucket"] = pd.to_numeric(working["sixty_minute_bucket"], errors="coerce").astype("int16")
    working["ma_period"] = pd.to_numeric(working["ma_period"], errors="coerce").astype("int16")
    working["liquidity_value"] = pd.to_numeric(working[liquidity_column], errors="coerce")
    match_columns = ["trade_date", "bar_time", "sixty_minute_bucket", "ma_period"]
    wanted = canonical.loc[
        :, ["signal_date", "signal_time", "sixty_minute_bucket", "ma_period"]
    ].rename(columns={"signal_date": "trade_date", "signal_time": "bar_time"})
    wanted = wanted.drop_duplicates(match_columns)
    wanted_index = pd.MultiIndex.from_frame(wanted)
    working_index = pd.MultiIndex.from_frame(working.loc[:, match_columns])
    working = working.loc[working_index.isin(wanted_index)].reset_index(drop=True)

    # Materialise each same-minute group as arrays once.  The previous path
    # repeatedly filtered a pandas DataFrame for every reference event; on a
    # full-universe day that dominated control construction time and created
    # thousands of short-lived objects.
    working_symbols = working["symbol"].astype(str).to_numpy()
    working_liquidity = working["liquidity_value"].to_numpy(dtype=float)
    grouped: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]] = {}
    for key, positions in working.groupby(match_columns, sort=False, observed=True).indices.items():
        state_key = key if isinstance(key, tuple) else (key,)
        state_positions = np.asarray(positions, dtype=np.int64)
        symbols = working_symbols[state_positions]
        liquidity = working_liquidity[state_positions]
        symbol_positions = {str(symbol): int(position) for position, symbol in enumerate(symbols)}
        grouped[state_key] = (state_positions, symbols, liquidity, symbol_positions)

    matched_by_key: dict[tuple[Any, ...], dict[str, Any] | None] = {}
    reason_by_key: dict[tuple[Any, ...], str] = {}
    band = float(liquidity_band_ratio)
    canonical_values = canonical.loc[:, [*control_columns, "signal_id"]]
    for ref_symbol, ref_date, ref_time, bucket, period, ref_signal_id in canonical_values.itertuples(
        index=False, name=None
    ):
        ref_key = (str(ref_symbol), str(ref_date), str(ref_time), int(bucket), int(period))
        state_key = ref_key[1:]
        group = grouped.get(state_key)
        if group is None:
            reason_by_key[ref_key] = "same_minute_state_missing"
            matched_by_key[ref_key] = None
            continue
        state_positions, symbols, liquidity, symbol_positions = group
        reference_position = symbol_positions.get(str(ref_symbol))
        if reference_position is None:
            reason_by_key[ref_key] = "reference_state_missing"
            matched_by_key[ref_key] = None
            continue
        ref_liquidity = float(liquidity[reference_position])
        if not np.isfinite(ref_liquidity) or ref_liquidity <= 0:
            reason_by_key[ref_key] = "reference_liquidity_missing"
            matched_by_key[ref_key] = None
            continue
        candidate_mask = (symbols != str(ref_symbol)) & np.isfinite(liquidity) & (liquidity > 0)
        candidate_indices = np.flatnonzero(candidate_mask)
        if not len(candidate_indices):
            reason_by_key[ref_key] = "candidate_liquidity_missing"
            matched_by_key[ref_key] = None
            continue
        ratios = liquidity[candidate_indices] / ref_liquidity
        in_band = (ratios >= 1.0 / band) & (ratios <= band)
        candidate_indices = candidate_indices[in_band]
        ratios = ratios[in_band]
        if not len(candidate_indices):
            reason_by_key[ref_key] = "no_candidate_in_liquidity_band"
            matched_by_key[ref_key] = None
            continue
        distances = np.abs(np.log(ratios))
        order = np.lexsort((symbols[candidate_indices], distances))[: int(nearest_candidates)]
        candidate_indices = candidate_indices[order]
        ratios = ratios[order]
        choice = _stable_choice_position(str(ref_signal_id), seed=int(random_seed), size=len(candidate_indices))
        chosen_position = int(candidate_indices[choice])
        state_row = working.iloc[int(state_positions[chosen_position])]
        signal = _signal_from_series(state_row, spec, "liquidity_matched_stock")
        signal["liquidity_match_ratio"] = float(liquidity[chosen_position] / ref_liquidity)
        matched_by_key[ref_key] = signal

    rows: list[dict[str, Any]] = []
    unmatched_reasons = Counter()
    reference_values = references.loc[:, [*control_columns, "signal_id"]]
    for ref_symbol, ref_date, ref_time, bucket, period, ref_signal_id in reference_values.itertuples(
        index=False, name=None
    ):
        ref_key = (str(ref_symbol), str(ref_date), str(ref_time), int(bucket), int(period))
        matched = matched_by_key.get(ref_key)
        if matched is None:
            unmatched_reasons[reason_by_key.get(ref_key, "no_candidate_in_liquidity_band")] += 1
            continue
        signal = dict(matched)
        ref_id = str(ref_signal_id)
        chosen_symbol = str(signal["symbol"])
        signal["signal_id"] = f"{strategy_id}|ref:{ref_id}|{chosen_symbol}"
        signal["reference_signal_id"] = ref_id
        signal["reference_symbol"] = str(ref_symbol)
        rows.append(signal)
    if rows:
        result = pd.DataFrame(rows)
        for name in SIGNAL_COLUMNS:
            if name not in result.columns:
                result[name] = None
        result = result.loc[:, list(SIGNAL_COLUMNS)]
        result.sort_values(
            ["signal_date", "signal_time", "strategy_id", "symbol", "ma_period", "reference_signal_id"],
            inplace=True,
            kind="stable",
            ignore_index=True,
        )
        if result["signal_id"].duplicated().any():
            raise StrategyError("liquidity_control_signal_id_duplicate")
    else:
        result = _empty_signal_frame()
    result.attrs.update(
        {
            "reference_count": int(len(references)),
            "matched_count": int(len(rows)),
            "unmatched_count": int(len(references) - len(rows)),
            "unmatched_reasons": dict(unmatched_reasons),
            "liquidity_column": str(liquidity_column),
            "liquidity_band_ratio": float(liquidity_band_ratio),
            "nearest_candidates": int(nearest_candidates),
        }
    )
    return result


__all__ = [
    "DEFAULT_LIQUIDITY_COLUMN",
    "SIGNAL_COLUMNS",
    "StrategyError",
    "StrategySpec",
    "attach_prior_daily_liquidity",
    "build_liquidity_matched_control",
    "build_liquidity_matched_controls_many",
    "build_matched_random_control",
    "build_strategy_signals",
    "build_strategy_signals_many",
    "build_strategy_signals_vectorized",
    "strategy_catalog",
    "strategy_catalog_frame",
    "strategy_spec",
]
