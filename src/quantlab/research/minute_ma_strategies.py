"""Finite, causal rule variants for the minute-MA event layer.

The module deliberately contains rules rather than a portfolio simulator. A
rule can only inspect the current row and rows before it in its
symbol/date/hour/MA group. End-of-hour diagnostic columns are never used for
executable signals.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantlab.research.minute_ma import EVENT_KEY_COLUMNS, MA_PERIODS, summarize_ma_events


class StrategyError(ValueError):
    """Raised when a strategy definition or signal input is invalid."""


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
    "close_to_intersection_bps",
    "prior_ma_slope_bps",
    "ma_alignment_score",
    "prior_true_touch_count_window",
    "up_down_amount_ratio",
    "bullish_ma_stack",
    "previous_hour_close_adjusted",
    "signal_adjusted_close",
    "signal_amount",
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
            "Deterministic random minute within each eligible stock-hour; control only.",
            "random_minute",
            ma_periods=ma,
            control=True,
        ),
        StrategySpec(
            "s0_strong_no_ma",
            "S0",
            "First minute closing at or above the previous completed hour close; no MA condition.",
            "strong_without_ma",
            ma_periods=ma,
            control=True,
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
            "s4_reclaim_volume_proxy",
            "S4",
            "S1 reclaim with cumulative up-amount at least as large as down-amount.",
            "touch_reclaim",
            ma_periods=ma,
            filters=("amount_confirm",),
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
        if name not in {"positive_slope", "bullish_stack", "first_touch", "amount_confirm"}:
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
        previous = pd.to_numeric(rows["previous_hour_close_adjusted"], errors="coerce").to_numpy(dtype=float)
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
        below_seen = False
        for position, (touched, below) in enumerate(zip(touched_values, below_values, strict=True)):
            if below:
                trigger_seen = True
                below_seen = True
                continue
            if touched or (trigger_seen and (below_seen or not below)):
                yield position, "touch_reclaim"
                # The first reclaim consumes this episode.  A later touch or
                # break must start a new episode before another signal.
                trigger_seen = False
                below_seen = False
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


def _signal_row(rows: pd.DataFrame, position: int, spec: StrategySpec, trigger: str) -> dict[str, Any]:
    row = rows.iloc[int(position)]
    symbol = str(row["symbol"])
    trade_date = str(row["trade_date"])
    bar_time = str(row["bar_time"])
    period = int(row["ma_period"])
    bucket = int(row["sixty_minute_bucket"])
    signal_id = f"{spec.strategy_id}|{symbol}|{trade_date}|{bucket}|{period}|{bar_time}"
    return {
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
        "close_to_intersection_bps": float(row["close_to_intersection_bps"]),
        "prior_ma_slope_bps": float(row["prior_ma_slope_bps"]),
        "ma_alignment_score": float(row["ma_alignment_score"]),
        "prior_true_touch_count_window": int(row["prior_true_touch_count_window"]),
        "up_down_amount_ratio": float(row["up_down_amount_ratio"]),
        "bullish_ma_stack": bool(row["bullish_ma_stack"]),
        "previous_hour_close_adjusted": float(row["previous_hour_close_adjusted"]),
        "signal_adjusted_close": float(row["adjusted_close"]),
        "signal_amount": float(row["amount"]),
    }


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
        if include_diagnostic or spec.executable
    )
    unknown = sorted(set(selected_ids).difference(catalog))
    if unknown:
        raise StrategyError(f"unknown_strategy:{','.join(unknown)}")
    selected_specs = [catalog[strategy_id] for strategy_id in selected_ids]
    required = set(_BASE_COLUMNS)
    for spec in selected_specs:
        required.update(spec.required_columns)
        spec.validate()
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
    result = pd.DataFrame(output).loc[:, list(SIGNAL_COLUMNS)]
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
    """Choose one random minute in each reference signal's stock-hour group.

    Matching the symbol/date/hour/MA keys keeps the control comparison local;
    the chosen minute is independent of the reference signal time. The
    control is a return comparison, not a claim that the random minute was
    selected by a separate screening process.
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
    result = pd.DataFrame(rows).loc[:, list(SIGNAL_COLUMNS)]
    result.sort_values(
        ["signal_date", "signal_time", "symbol", "ma_period"],
        inplace=True,
        kind="stable",
        ignore_index=True,
    )
    return result


__all__ = [
    "SIGNAL_COLUMNS",
    "StrategyError",
    "StrategySpec",
    "build_matched_random_control",
    "build_strategy_signals",
    "strategy_catalog",
    "strategy_catalog_frame",
    "strategy_spec",
]
