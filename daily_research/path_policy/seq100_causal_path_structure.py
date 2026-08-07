"""Strictly causal multi-scale path structure and matched action-value study.

The parser emits the state that was observable at each close.  Historical
extrema are never written back to their earlier dates after a later reversal.
Chan-theory terminology is used only for an explicit candidate grammar; the
study tests that grammar rather than assuming it describes a natural market
law.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats

from daily_research.path_policy import seq100_hot_path_atlas as atlas

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_causal_path_structure_v1.json"
)
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / (
    "daily_research/output/path_policy/studies/seq100_causal_path_structure_v1"
)
STUDY_ID = "seq100_causal_path_structure_v1"
PANEL_SCHEMA_VERSION = "seq100_causal_path_structure_panel/1"
MANIFEST_SCHEMA_VERSION = "seq100_causal_path_structure/1"
VALIDATION_SCHEMA_VERSION = "seq100_causal_path_structure_validation/1"
BUILDER_VERSION = 1

SCALE_LABELS = ("small", "medium", "large")
BASE_PATTERN_NAMES = (
    "nested_up_pullback",
    "nested_up_reacceleration",
    "nested_down_rebound",
    "nested_down_reacceleration",
    "zone_breakout_up",
    "zone_retest_hold_up",
    "zone_reentry_after_up_breakout",
    "up_exhaustion",
    "up_continuation",
)
INTERACTION_PATTERN_NAMES = (
    "up_breakout_confirmed_participation",
    "up_breakout_late_chase",
)
PATTERN_NAMES = (*BASE_PATTERN_NAMES, *INTERACTION_PATTERN_NAMES)

DISCRETE_FEATURES = {
    "chan_provisional_fractal",
    "chan_confirmed_fractal_event",
    "chan_stroke_event",
    "chan_stroke_direction",
    "chan_segment_event",
    "chan_segment_direction",
    "chan_breakout_event",
    "chan_breakout_active",
    "chan_retest_event",
    "chan_reentry_event",
}
BOOLEAN_FEATURES = {
    "chan_stroke_exhaustion",
    "chan_center_available",
    "chan_center_active",
    *[f"pattern_{name}" for name in BASE_PATTERN_NAMES],
}
INTEGER_FEATURES = {
    "chan_normalized_bar_count",
    "chan_days_since_confirmed_fractal",
    "chan_stroke_duration",
    "chan_center_age",
}
FLOAT_FEATURES = {
    "causal_scale_unit",
    "chan_stroke_return",
    "chan_stroke_slope_ratio",
    "chan_stroke_amount_ratio",
    "chan_center_width_units",
    "chan_center_position",
    "chan_center_distance_upper_units",
    "chan_center_distance_lower_units",
}
for _label in SCALE_LABELS:
    DISCRETE_FEATURES.update({f"dc_{_label}_mode", f"dc_{_label}_event"})
    BOOLEAN_FEATURES.update({f"dc_{_label}_new_extreme", f"dc_{_label}_exhaustion"})
    INTEGER_FEATURES.update(
        {
            f"dc_{_label}_leg_duration",
            f"dc_{_label}_days_since_confirmation",
            f"dc_{_label}_confirmed_extreme_age",
            f"dc_{_label}_provisional_age",
        }
    )
    FLOAT_FEATURES.update(
        {
            f"dc_{_label}_threshold",
            f"dc_{_label}_reversal_fraction",
            f"dc_{_label}_leg_return",
            f"dc_{_label}_retracement_ratio",
            f"dc_{_label}_slope_ratio",
            f"dc_{_label}_amount_ratio",
        }
    )

FEATURE_COLUMNS = tuple(
    sorted(DISCRETE_FEATURES)
    + sorted(BOOLEAN_FEATURES)
    + sorted(INTEGER_FEATURES)
    + sorted(FLOAT_FEATURES)
)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else WORKSPACE_ROOT / value


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=atlas._json_default)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _sha256(path: str | Path) -> str:
    return atlas._sha256_file(_resolve(path))


def load_study(path: str | Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study = _read_json(path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("causal_path_structure_study_id_mismatch")
    source = dict(study.get("source", {}) or {})
    if str(source.get("quality_pool_name")) != "quality_liquidity_pit":
        raise ValueError("causal_path_structure_quality_pool_mismatch")
    if str(source.get("maximum_outcome_date")) != "2025-12-31":
        raise ValueError("causal_path_structure_outcome_cutoff_mismatch")
    if int(source.get("forbidden_year", -1)) != 2026:
        raise ValueError("causal_path_structure_forbidden_year_mismatch")
    scale = dict(study.get("causal_scale", {}) or {})
    if tuple(scale.get("scale_names", ())) != SCALE_LABELS:
        raise ValueError("causal_path_structure_scale_names_mismatch")
    multipliers = tuple(
        float(value) for value in scale.get("volatility_multipliers", ())
    )
    if len(multipliers) != 3 or not all(
        left < right for left, right in itertools.pairwise(multipliers)
    ):
        raise ValueError("causal_path_structure_scale_multipliers_invalid")
    if int(scale.get("round_trip_cost_bps", 0)) <= 0:
        raise ValueError("causal_path_structure_cost_invalid")
    grammar = dict(study.get("chan_candidate_grammar", {}) or {})
    if bool(grammar.get("orthodox_chan_theory_claimed", True)):
        raise ValueError("causal_path_structure_orthodox_claim_forbidden")
    if int(grammar.get("minimum_stroke_separation_normalized_bars", 0)) < 2:
        raise ValueError("causal_path_structure_stroke_separation_invalid")
    boundaries = dict(study.get("boundaries", {}) or {})
    if bool(boundaries.get("future_confirmed_structure_written_back", True)):
        raise ValueError("causal_path_structure_repainting_forbidden")
    if bool(boundaries.get("profit_claim_allowed", True)):
        raise ValueError("causal_path_structure_profit_claim_forbidden")
    return study


def _verify_file(path: str | Path, expected_sha256: str, error: str) -> Path:
    target = _resolve(path)
    if not target.is_file():
        raise FileNotFoundError(f"{error}:{target}")
    if _sha256(target) != str(expected_sha256).lower():
        raise ValueError(error)
    return target


def _source_contract(
    study_path: Path, study: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    source = dict(study["source"])
    turning_path = _verify_file(
        source["turning_manifest"],
        source["expected_turning_manifest_sha256"],
        "causal_path_structure_turning_manifest_mismatch",
    )
    turning = _read_json(turning_path)
    if str(turning.get("experiment_fingerprint")) != str(
        source["expected_dense_fingerprint"]
    ):
        raise ValueError("causal_path_structure_dense_fingerprint_mismatch")
    dense_path = Path(str(dict(turning["source_contract"])["dense_base"]["path"]))
    if not dense_path.is_file():
        raise FileNotFoundError(f"causal_path_structure_dense_missing:{dense_path}")

    model_input_path = _verify_file(
        source["model_input_manifest"],
        source["expected_model_input_manifest_sha256"],
        "causal_path_structure_model_input_manifest_mismatch",
    )
    model_input = _read_json(model_input_path)
    row_index_record = dict(model_input["row_index"])
    row_index_path = Path(str(row_index_record["path"]))
    if not row_index_path.is_file():
        raise FileNotFoundError(
            f"causal_path_structure_row_index_missing:{row_index_path}"
        )
    if int(row_index_record["row_count"]) != int(source["expected_row_index_rows"]):
        raise ValueError("causal_path_structure_row_index_count_mismatch")
    if _sha256(row_index_path) != str(row_index_record["sha256"]):
        raise ValueError("causal_path_structure_row_index_hash_mismatch")

    distribution_path = _verify_file(
        source["distribution_manifest"],
        source["expected_distribution_manifest_sha256"],
        "causal_path_structure_distribution_manifest_mismatch",
    )
    validation_path = _verify_file(
        source["distribution_validation"],
        source["expected_distribution_validation_sha256"],
        "causal_path_structure_distribution_validation_mismatch",
    )
    distribution = _read_json(distribution_path)
    validation = _read_json(validation_path)
    if (
        distribution.get("status") != "completed"
        or validation.get("status") != "passed"
    ):
        raise ValueError("causal_path_structure_distribution_source_incomplete")
    if bool(distribution.get("audit", {}).get("account_replay_allowed", True)):
        raise ValueError("causal_path_structure_inherited_replay_gate_mismatch")

    payload = {
        "builder_version": BUILDER_VERSION,
        "study_sha256": _sha256(study_path),
        "turning_manifest_sha256": _sha256(turning_path),
        "model_input_manifest_sha256": _sha256(model_input_path),
        "distribution_manifest_sha256": _sha256(distribution_path),
        "distribution_validation_sha256": _sha256(validation_path),
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    contract = {
        "turning_manifest": atlas._file_record(turning_path),
        "dense_base": atlas._file_record(dense_path, include_hash=False),
        "model_input_manifest": atlas._file_record(model_input_path),
        "row_index": atlas._file_record(row_index_path, include_hash=False),
        "row_index_rows": int(row_index_record["row_count"]),
        "distribution_manifest": atlas._file_record(distribution_path),
        "distribution_validation": atlas._file_record(validation_path),
        "coordinate_partitions": list(distribution["coordinates"]),
        "prediction_partitions": list(distribution["predictions"]),
        "fingerprint_payload": payload,
    }
    return contract, fingerprint


def _causal_volatility(
    log_close: np.ndarray,
    *,
    window: int,
    minimum_observations: int,
    cost_log_return: float,
) -> np.ndarray:
    """Past-only RMS volatility; the current return is deliberately excluded."""

    values = np.asarray(log_close, dtype=np.float64)
    result = np.full(len(values), float(cost_log_return), dtype=np.float64)
    if len(values) < 2:
        return result
    returns = np.zeros(len(values), dtype=np.float64)
    returns[1:] = np.diff(values)
    squared = np.square(returns)
    cumulative = np.cumsum(squared)
    for pos in range(1, len(values)):
        start = max(1, pos - int(window))
        count = pos - start
        if count < int(minimum_observations):
            continue
        total = cumulative[pos - 1] - (cumulative[start - 1] if start > 0 else 0.0)
        rms = math.sqrt(max(float(total) / count, 0.0))
        result[pos] = max(float(cost_log_return), rms)
    return result


def _prefix_mean(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    sums = np.concatenate(([0.0], np.cumsum(np.where(finite, values, 0.0))))
    counts = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))
    return sums, counts


def _interval_mean(sums: np.ndarray, counts: np.ndarray, start: int, end: int) -> float:
    left = max(int(start), 0)
    right = min(int(end) + 1, len(sums) - 1)
    count = int(counts[right] - counts[left])
    if count <= 0:
        return math.nan
    return float((sums[right] - sums[left]) / count)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return math.nan
    if abs(denominator) <= 1.0e-12:
        return math.nan
    return float(numerator / denominator)


@dataclass(frozen=True)
class _Leg:
    direction: int
    start_pos: int
    end_pos: int
    start_price: float
    end_price: float
    slope: float
    amount_mean: float


@dataclass(frozen=True)
class _Fractal:
    kind: int
    normalized_bar_index: int
    extreme_pos: int
    price: float


@dataclass(frozen=True)
class _Stroke:
    direction: int
    start: _Fractal
    end: _Fractal
    log_return: float
    duration: int
    slope: float
    amount_mean: float
    slope_ratio: float
    amount_ratio: float
    exhaustion: bool


@dataclass
class _NormalizedBar:
    high: float
    low: float
    close: float
    start_pos: int
    end_pos: int


def _empty_feature_arrays(length: int) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name in DISCRETE_FEATURES:
        arrays[name] = np.zeros(length, dtype=np.int8)
    for name in BOOLEAN_FEATURES:
        arrays[name] = np.zeros(length, dtype=bool)
    for name in INTEGER_FEATURES:
        arrays[name] = np.full(length, -1, dtype=np.int32)
    for name in FLOAT_FEATURES:
        arrays[name] = np.full(length, np.nan, dtype=np.float64)
    return arrays


def _directional_scale_state(
    log_close: np.ndarray,
    date_indices: np.ndarray,
    scale_unit: np.ndarray,
    amount_sums: np.ndarray,
    amount_counts: np.ndarray,
    *,
    multiplier: float,
    cost_log_return: float,
) -> dict[str, np.ndarray]:
    """Run one amplitude-scale directional-change state machine."""

    prices = np.asarray(log_close, dtype=np.float64)
    dates = np.asarray(date_indices, dtype=np.int64)
    length = len(prices)
    output = {
        "mode": np.zeros(length, dtype=np.int8),
        "event": np.zeros(length, dtype=np.int8),
        "threshold": np.full(length, np.nan, dtype=np.float64),
        "reversal_fraction": np.full(length, np.nan, dtype=np.float64),
        "leg_return": np.full(length, np.nan, dtype=np.float64),
        "leg_duration": np.full(length, -1, dtype=np.int32),
        "retracement_ratio": np.full(length, np.nan, dtype=np.float64),
        "days_since_confirmation": np.full(length, -1, dtype=np.int32),
        "confirmed_extreme_age": np.full(length, -1, dtype=np.int32),
        "provisional_age": np.full(length, -1, dtype=np.int32),
        "slope_ratio": np.full(length, np.nan, dtype=np.float64),
        "amount_ratio": np.full(length, np.nan, dtype=np.float64),
        "new_extreme": np.zeros(length, dtype=bool),
        "exhaustion": np.zeros(length, dtype=bool),
    }
    if length == 0:
        return output

    def threshold_at(pos: int) -> float:
        return max(
            float(cost_log_return), float(multiplier) * float(scale_unit[int(pos)])
        )

    mode = 0
    high = low = float(prices[0])
    high_pos = low_pos = 0
    high_threshold = low_threshold = threshold_at(0)
    confirmed_pos: int | None = None
    confirmed_price: float | None = None
    confirmation_pos: int | None = None
    previous_leg: dict[int, _Leg] = {}

    def complete_leg(direction: int, start_pos: int, end_pos: int) -> _Leg:
        start_price = float(prices[start_pos])
        end_price = float(prices[end_pos])
        duration = max(int(dates[end_pos] - dates[start_pos]), 1)
        amplitude = abs(end_price - start_price)
        slope = (
            amplitude
            / math.sqrt(duration)
            / max(float(scale_unit[end_pos]), float(cost_log_return))
        )
        return _Leg(
            direction=int(direction),
            start_pos=int(start_pos),
            end_pos=int(end_pos),
            start_price=start_price,
            end_price=end_price,
            slope=float(slope),
            amount_mean=_interval_mean(
                amount_sums, amount_counts, int(start_pos), int(end_pos)
            ),
        )

    for pos in range(length):
        value = float(prices[pos])
        event = 0
        new_extreme = False

        if pos > 0 and mode == 0:
            if value >= high:
                high, high_pos, high_threshold = value, pos, threshold_at(pos)
            if value <= low:
                low, low_pos, low_threshold = value, pos, threshold_at(pos)
            if value - low >= low_threshold:
                mode = 1
                event = 1
                confirmed_pos = low_pos
                confirmed_price = low
                confirmation_pos = pos
                high, high_pos, high_threshold = value, pos, threshold_at(pos)
            elif high - value >= high_threshold:
                mode = -1
                event = -1
                confirmed_pos = high_pos
                confirmed_price = high
                confirmation_pos = pos
                low, low_pos, low_threshold = value, pos, threshold_at(pos)

        elif pos > 0 and mode == 1:
            if value >= high:
                high, high_pos, high_threshold = value, pos, threshold_at(pos)
                new_extreme = True
            elif high - value >= high_threshold:
                if confirmed_pos is None:
                    raise AssertionError("causal_path_structure_missing_up_leg_start")
                previous_leg[1] = complete_leg(1, confirmed_pos, high_pos)
                mode = -1
                event = -1
                confirmed_pos = high_pos
                confirmed_price = high
                confirmation_pos = pos
                low, low_pos, low_threshold = value, pos, threshold_at(pos)

        elif pos > 0 and mode == -1:
            if value <= low:
                low, low_pos, low_threshold = value, pos, threshold_at(pos)
                new_extreme = True
            elif value - low >= low_threshold:
                if confirmed_pos is None:
                    raise AssertionError("causal_path_structure_missing_down_leg_start")
                previous_leg[-1] = complete_leg(-1, confirmed_pos, low_pos)
                mode = 1
                event = 1
                confirmed_pos = low_pos
                confirmed_price = low
                confirmation_pos = pos
                high, high_pos, high_threshold = value, pos, threshold_at(pos)

        output["mode"][pos] = mode
        output["event"][pos] = event
        output["new_extreme"][pos] = new_extreme
        if mode == 0 or confirmed_pos is None or confirmed_price is None:
            continue

        if mode == 1:
            provisional_pos = high_pos
            provisional_price = high
            active_threshold = high_threshold
            reversal = high - value
            leg_return = high - confirmed_price
            direction = 1
        else:
            provisional_pos = low_pos
            provisional_price = low
            active_threshold = low_threshold
            reversal = value - low
            leg_return = low - confirmed_price
            direction = -1

        duration = max(int(dates[provisional_pos] - dates[confirmed_pos]), 0)
        amplitude = abs(provisional_price - confirmed_price)
        retracement = _safe_ratio(reversal, amplitude)
        current_slope = (
            amplitude
            / math.sqrt(max(duration, 1))
            / max(float(scale_unit[provisional_pos]), float(cost_log_return))
        )
        current_amount = _interval_mean(
            amount_sums, amount_counts, confirmed_pos, provisional_pos
        )
        prior = previous_leg.get(direction)
        slope_ratio = (
            _safe_ratio(current_slope, prior.slope) if prior is not None else math.nan
        )
        amount_ratio = (
            math.exp(current_amount - prior.amount_mean)
            if prior is not None
            and math.isfinite(current_amount)
            and math.isfinite(prior.amount_mean)
            else math.nan
        )
        extends_prior = (
            provisional_price > prior.end_price
            if prior is not None and direction == 1
            else provisional_price < prior.end_price
            if prior is not None
            else False
        )
        exhaustion = bool(
            new_extreme
            and prior is not None
            and extends_prior
            and math.isfinite(slope_ratio)
            and math.isfinite(amount_ratio)
            and slope_ratio < 1.0
            and amount_ratio < 1.0
        )

        output["threshold"][pos] = active_threshold
        output["reversal_fraction"][pos] = reversal / max(active_threshold, 1.0e-12)
        output["leg_return"][pos] = leg_return
        output["leg_duration"][pos] = duration
        output["retracement_ratio"][pos] = retracement
        output["days_since_confirmation"][pos] = (
            int(dates[pos] - dates[confirmation_pos])
            if confirmation_pos is not None
            else -1
        )
        output["confirmed_extreme_age"][pos] = int(dates[pos] - dates[confirmed_pos])
        output["provisional_age"][pos] = int(dates[pos] - dates[provisional_pos])
        output["slope_ratio"][pos] = slope_ratio
        output["amount_ratio"][pos] = amount_ratio
        output["exhaustion"][pos] = exhaustion
    return output


def _fractal_kind(
    left: _NormalizedBar, middle: _NormalizedBar, right: _NormalizedBar
) -> int:
    top = (
        middle.high > left.high
        and middle.high > right.high
        and middle.low > left.low
        and middle.low > right.low
    )
    bottom = (
        middle.high < left.high
        and middle.high < right.high
        and middle.low < left.low
        and middle.low < right.low
    )
    if top:
        return -1
    if bottom:
        return 1
    return 0


def _inclusion_direction(bars: Sequence[_NormalizedBar], current_close: float) -> int:
    latest = bars[-1]
    if len(bars) >= 2:
        previous = bars[-2]
        if latest.high >= previous.high and latest.low >= previous.low:
            return 1
        if latest.high <= previous.high and latest.low <= previous.low:
            return -1
        delta = latest.close - previous.close
        if abs(delta) > 1.0e-12:
            return 1 if delta > 0 else -1
    delta = float(current_close) - latest.close
    return 1 if delta >= 0 else -1


def _chan_candidate_state(
    log_high: np.ndarray,
    log_low: np.ndarray,
    log_close: np.ndarray,
    date_indices: np.ndarray,
    scale_unit: np.ndarray,
    amount_sums: np.ndarray,
    amount_counts: np.ndarray,
    *,
    cost_log_return: float,
    minimum_stroke_separation: int,
    breakout_buffer_cost_multiples: float,
) -> dict[str, np.ndarray]:
    """Causal inclusion/fractal/stroke/segment/center candidate grammar."""

    highs = np.asarray(log_high, dtype=np.float64)
    lows = np.asarray(log_low, dtype=np.float64)
    closes = np.asarray(log_close, dtype=np.float64)
    dates = np.asarray(date_indices, dtype=np.int64)
    length = len(closes)
    output = {
        "normalized_bar_count": np.full(length, -1, dtype=np.int32),
        "provisional_fractal": np.zeros(length, dtype=np.int8),
        "confirmed_fractal_event": np.zeros(length, dtype=np.int8),
        "days_since_confirmed_fractal": np.full(length, -1, dtype=np.int32),
        "stroke_event": np.zeros(length, dtype=np.int8),
        "stroke_direction": np.zeros(length, dtype=np.int8),
        "stroke_return": np.full(length, np.nan, dtype=np.float64),
        "stroke_duration": np.full(length, -1, dtype=np.int32),
        "stroke_slope_ratio": np.full(length, np.nan, dtype=np.float64),
        "stroke_amount_ratio": np.full(length, np.nan, dtype=np.float64),
        "stroke_exhaustion": np.zeros(length, dtype=bool),
        "segment_event": np.zeros(length, dtype=np.int8),
        "segment_direction": np.zeros(length, dtype=np.int8),
        "center_available": np.zeros(length, dtype=bool),
        "center_active": np.zeros(length, dtype=bool),
        "center_width_units": np.full(length, np.nan, dtype=np.float64),
        "center_position": np.full(length, np.nan, dtype=np.float64),
        "center_distance_upper_units": np.full(length, np.nan, dtype=np.float64),
        "center_distance_lower_units": np.full(length, np.nan, dtype=np.float64),
        "center_age": np.full(length, -1, dtype=np.int32),
        "breakout_event": np.zeros(length, dtype=np.int8),
        "breakout_active": np.zeros(length, dtype=np.int8),
        "retest_event": np.zeros(length, dtype=np.int8),
        "reentry_event": np.zeros(length, dtype=np.int8),
    }
    bars: list[_NormalizedBar] = []
    accepted_endpoints: list[_Fractal] = []
    strokes: list[_Stroke] = []
    prior_stroke: dict[int, _Stroke] = {}
    latest_stroke: _Stroke | None = None
    last_fractal_confirmation_pos: int | None = None
    segment_direction = 0
    zone_low = zone_high = math.nan
    zone_start_pos: int | None = None
    zone_available = False
    center_active = False
    breakout_active = 0
    breakout_boundary = math.nan
    breakout_pos: int | None = None

    def accept_fractal(fractal: _Fractal) -> _Stroke | None:
        nonlocal latest_stroke
        if not accepted_endpoints:
            accepted_endpoints.append(fractal)
            return None
        previous = accepted_endpoints[-1]
        if fractal.kind == previous.kind:
            if len(accepted_endpoints) == 1:
                more_extreme = (
                    fractal.price > previous.price
                    if fractal.kind == -1
                    else fractal.price < previous.price
                )
                if more_extreme:
                    accepted_endpoints[-1] = fractal
            return None
        if fractal.normalized_bar_index - previous.normalized_bar_index < int(
            minimum_stroke_separation
        ):
            return None
        direction = 1 if previous.kind == 1 and fractal.kind == -1 else -1
        if direction == 1 and fractal.price <= previous.price:
            return None
        if direction == -1 and fractal.price >= previous.price:
            return None
        duration = max(int(dates[fractal.extreme_pos] - dates[previous.extreme_pos]), 1)
        log_return = float(fractal.price - previous.price)
        slope = (
            abs(log_return)
            / math.sqrt(duration)
            / max(float(scale_unit[fractal.extreme_pos]), float(cost_log_return))
        )
        amount_mean = _interval_mean(
            amount_sums,
            amount_counts,
            previous.extreme_pos,
            fractal.extreme_pos,
        )
        prior = prior_stroke.get(direction)
        slope_ratio = _safe_ratio(slope, prior.slope) if prior is not None else math.nan
        amount_ratio = (
            math.exp(amount_mean - prior.amount_mean)
            if prior is not None
            and math.isfinite(amount_mean)
            and math.isfinite(prior.amount_mean)
            else math.nan
        )
        extends = (
            fractal.price > prior.end.price
            if prior is not None and direction == 1
            else fractal.price < prior.end.price
            if prior is not None
            else False
        )
        exhaustion = bool(
            prior is not None
            and extends
            and math.isfinite(slope_ratio)
            and math.isfinite(amount_ratio)
            and slope_ratio < 1.0
            and amount_ratio < 1.0
        )
        stroke = _Stroke(
            direction=direction,
            start=previous,
            end=fractal,
            log_return=log_return,
            duration=duration,
            slope=float(slope),
            amount_mean=amount_mean,
            slope_ratio=slope_ratio,
            amount_ratio=amount_ratio,
            exhaustion=exhaustion,
        )
        accepted_endpoints.append(fractal)
        strokes.append(stroke)
        prior_stroke[direction] = stroke
        latest_stroke = stroke
        return stroke

    for pos in range(length):
        current = _NormalizedBar(
            high=float(highs[pos]),
            low=float(lows[pos]),
            close=float(closes[pos]),
            start_pos=pos,
            end_pos=pos,
        )
        new_bar = False
        if not bars:
            bars.append(current)
            new_bar = True
        else:
            latest = bars[-1]
            contains = current.high >= latest.high and current.low <= latest.low
            contained = current.high <= latest.high and current.low >= latest.low
            if contains or contained:
                direction = _inclusion_direction(bars, current.close)
                if direction >= 0:
                    merged_high = max(latest.high, current.high)
                    merged_low = max(latest.low, current.low)
                else:
                    merged_high = min(latest.high, current.high)
                    merged_low = min(latest.low, current.low)
                if merged_low > merged_high:
                    merged_low, merged_high = (
                        min(merged_low, merged_high),
                        max(merged_low, merged_high),
                    )
                bars[-1] = _NormalizedBar(
                    high=float(merged_high),
                    low=float(merged_low),
                    close=current.close,
                    start_pos=latest.start_pos,
                    end_pos=pos,
                )
            else:
                bars.append(current)
                new_bar = True

        confirmed_kind = 0
        stroke_event = 0
        segment_event = 0
        if new_bar and len(bars) >= 4:
            left, middle, right = bars[-4], bars[-3], bars[-2]
            confirmed_kind = _fractal_kind(left, middle, right)
            if confirmed_kind != 0:
                last_fractal_confirmation_pos = pos
                price = middle.low if confirmed_kind == 1 else middle.high
                stroke = accept_fractal(
                    _Fractal(
                        kind=confirmed_kind,
                        normalized_bar_index=len(bars) - 3,
                        extreme_pos=middle.end_pos,
                        price=float(price),
                    )
                )
                if stroke is not None:
                    stroke_event = stroke.direction
                    if len(strokes) >= 3:
                        first, second, third = strokes[-3:]
                        if (
                            (first.direction, second.direction, third.direction)
                            == (1, -1, 1)
                            and third.end.price > first.end.price
                            and third.start.price > first.start.price
                        ):
                            segment_direction = 1
                            segment_event = 1
                        elif (
                            (first.direction, second.direction, third.direction)
                            == (-1, 1, -1)
                            and third.end.price < first.end.price
                            and third.start.price < first.start.price
                        ):
                            segment_direction = -1
                            segment_event = -1

                    if len(strokes) >= 3:
                        intervals = [
                            (
                                min(item.start.price, item.end.price),
                                max(item.start.price, item.end.price),
                            )
                            for item in strokes[-3:]
                        ]
                        candidate_low = max(value[0] for value in intervals)
                        candidate_high = min(value[1] for value in intervals)
                        if candidate_high > candidate_low:
                            zone_low = float(candidate_low)
                            zone_high = float(candidate_high)
                            zone_start_pos = pos
                            zone_available = True
                            center_active = True
                            breakout_active = 0
                            breakout_boundary = math.nan
                            breakout_pos = None
                        else:
                            center_active = False

        provisional_kind = 0
        if len(bars) >= 3:
            provisional_kind = _fractal_kind(bars[-3], bars[-2], bars[-1])

        breakout_event = 0
        retest_event = 0
        reentry_event = 0
        if zone_available:
            buffer = float(breakout_buffer_cost_multiples) * max(
                float(cost_log_return), float(scale_unit[pos])
            )
            previous_close = float(closes[pos - 1]) if pos > 0 else float(closes[pos])
            if breakout_active == 0:
                if (
                    previous_close <= zone_high + buffer
                    and closes[pos] > zone_high + buffer
                ):
                    breakout_active = 1
                    breakout_event = 1
                    breakout_boundary = zone_high
                    breakout_pos = pos
                    center_active = False
                elif (
                    previous_close >= zone_low - buffer
                    and closes[pos] < zone_low - buffer
                ):
                    breakout_active = -1
                    breakout_event = -1
                    breakout_boundary = zone_low
                    breakout_pos = pos
                    center_active = False
            elif breakout_pos is not None and pos > breakout_pos:
                if breakout_active == 1:
                    if closes[pos] < breakout_boundary:
                        reentry_event = 1
                        breakout_active = 0
                        zone_available = False
                        center_active = False
                    elif (
                        lows[pos] <= breakout_boundary + buffer
                        and closes[pos] >= breakout_boundary
                    ):
                        retest_event = 1
                else:
                    if closes[pos] > breakout_boundary:
                        reentry_event = -1
                        breakout_active = 0
                        zone_available = False
                        center_active = False
                    elif (
                        highs[pos] >= breakout_boundary - buffer
                        and closes[pos] <= breakout_boundary
                    ):
                        retest_event = -1

        output["normalized_bar_count"][pos] = len(bars)
        output["provisional_fractal"][pos] = provisional_kind
        output["confirmed_fractal_event"][pos] = confirmed_kind
        output["days_since_confirmed_fractal"][pos] = (
            int(dates[pos] - dates[last_fractal_confirmation_pos])
            if last_fractal_confirmation_pos is not None
            else -1
        )
        output["stroke_event"][pos] = stroke_event
        output["segment_event"][pos] = segment_event
        output["segment_direction"][pos] = segment_direction
        if latest_stroke is not None:
            output["stroke_direction"][pos] = latest_stroke.direction
            output["stroke_return"][pos] = latest_stroke.log_return
            output["stroke_duration"][pos] = latest_stroke.duration
            output["stroke_slope_ratio"][pos] = latest_stroke.slope_ratio
            output["stroke_amount_ratio"][pos] = latest_stroke.amount_ratio
        output["stroke_exhaustion"][pos] = bool(
            stroke_event != 0 and latest_stroke is not None and latest_stroke.exhaustion
        )
        output["center_available"][pos] = zone_available
        output["center_active"][pos] = center_active
        if zone_available:
            unit = max(float(scale_unit[pos]), float(cost_log_return))
            width = zone_high - zone_low
            output["center_width_units"][pos] = width / unit
            output["center_position"][pos] = _safe_ratio(
                float(closes[pos]) - zone_low, width
            )
            output["center_distance_upper_units"][pos] = (
                float(closes[pos]) - zone_high
            ) / unit
            output["center_distance_lower_units"][pos] = (
                float(closes[pos]) - zone_low
            ) / unit
            output["center_age"][pos] = (
                int(dates[pos] - dates[zone_start_pos])
                if zone_start_pos is not None
                else -1
            )
        output["breakout_event"][pos] = breakout_event
        output["breakout_active"][pos] = breakout_active
        output["retest_event"][pos] = retest_event
        output["reentry_event"][pos] = reentry_event
    return output


def parse_causal_path(
    *,
    adj_high: np.ndarray,
    adj_low: np.ndarray,
    adj_close: np.ndarray,
    amount: np.ndarray,
    date_indices: np.ndarray,
    study: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Return one prefix-only feature row for every supplied valid bar."""

    high = np.asarray(adj_high, dtype=np.float64)
    low = np.asarray(adj_low, dtype=np.float64)
    close = np.asarray(adj_close, dtype=np.float64)
    amount = np.asarray(amount, dtype=np.float64)
    dates = np.asarray(date_indices, dtype=np.int64)
    length = len(close)
    if not all(len(values) == length for values in (high, low, amount, dates)):
        raise ValueError("causal_path_structure_array_length_mismatch")
    if length == 0:
        return _empty_feature_arrays(0)
    if (
        np.any(~np.isfinite(high))
        or np.any(~np.isfinite(low))
        or np.any(~np.isfinite(close))
        or np.any(high <= 0)
        or np.any(low <= 0)
        or np.any(close <= 0)
        or np.any(high < low)
    ):
        raise ValueError("causal_path_structure_invalid_price_path")
    if np.any(np.diff(dates) <= 0):
        raise ValueError("causal_path_structure_nonincreasing_dates")

    log_high = np.log(high)
    log_low = np.log(low)
    log_close = np.log(close)
    log_amount = np.full(length, np.nan, dtype=np.float64)
    positive_amount = amount > 0
    log_amount[positive_amount] = np.log(amount[positive_amount])
    amount_sums, amount_counts = _prefix_mean(log_amount)
    scale = dict(study["causal_scale"])
    cost = float(scale["round_trip_cost_bps"]) / 10_000.0
    scale_unit = _causal_volatility(
        log_close,
        window=int(scale["past_volatility_window"]),
        minimum_observations=int(scale["past_volatility_minimum_observations"]),
        cost_log_return=cost,
    )
    output = _empty_feature_arrays(length)
    output["causal_scale_unit"] = scale_unit
    for label, multiplier in zip(SCALE_LABELS, scale["volatility_multipliers"]):
        state = _directional_scale_state(
            log_close,
            dates,
            scale_unit,
            amount_sums,
            amount_counts,
            multiplier=float(multiplier),
            cost_log_return=cost,
        )
        for name, values in state.items():
            output[f"dc_{label}_{name}"] = values

    grammar = dict(study["chan_candidate_grammar"])
    chan = _chan_candidate_state(
        log_high,
        log_low,
        log_close,
        dates,
        scale_unit,
        amount_sums,
        amount_counts,
        cost_log_return=cost,
        minimum_stroke_separation=int(
            grammar["minimum_stroke_separation_normalized_bars"]
        ),
        breakout_buffer_cost_multiples=float(grammar["breakout_buffer_cost_multiples"]),
    )
    for name, values in chan.items():
        output[f"chan_{name}"] = values

    small_mode = output["dc_small_mode"]
    medium_mode = output["dc_medium_mode"]
    large_mode = output["dc_large_mode"]
    small_event = output["dc_small_event"]
    nested_up_pullback = (large_mode == 1) & (medium_mode == 1) & (small_mode == -1)
    nested_up_reacceleration = (
        (large_mode == 1) & (medium_mode == 1) & (small_event == 1)
    )
    nested_down_rebound = (large_mode == -1) & (medium_mode == -1) & (small_mode == 1)
    nested_down_reacceleration = (
        (large_mode == -1) & (medium_mode == -1) & (small_event == -1)
    )
    zone_breakout_up = output["chan_breakout_event"] == 1
    output["pattern_nested_up_pullback"] = nested_up_pullback
    output["pattern_nested_up_reacceleration"] = nested_up_reacceleration
    output["pattern_nested_down_rebound"] = nested_down_rebound
    output["pattern_nested_down_reacceleration"] = nested_down_reacceleration
    output["pattern_zone_breakout_up"] = zone_breakout_up
    output["pattern_zone_retest_hold_up"] = output["chan_retest_event"] == 1
    output["pattern_zone_reentry_after_up_breakout"] = output["chan_reentry_event"] == 1
    output["pattern_up_exhaustion"] = (
        output["dc_medium_exhaustion"] | output["dc_large_exhaustion"]
    )
    output["pattern_up_continuation"] = nested_up_reacceleration | zone_breakout_up

    missing = set(FEATURE_COLUMNS) - set(output)
    extra = set(output) - set(FEATURE_COLUMNS)
    if missing or extra:
        raise AssertionError(
            f"causal_path_structure_feature_contract:{sorted(missing)}:{sorted(extra)}"
        )
    return output


def _panel_schema() -> pa.Schema:
    fields: list[pa.Field] = [
        pa.field("input_row_idx", pa.int64()),
        pa.field("trade_date", pa.string()),
        pa.field("date_idx", pa.int32()),
        pa.field("symbol_idx", pa.int32()),
    ]
    for name in FEATURE_COLUMNS:
        if name in BOOLEAN_FEATURES:
            dtype = pa.bool_()
        elif name in DISCRETE_FEATURES:
            dtype = pa.int8()
        elif name in INTEGER_FEATURES:
            dtype = pa.int32()
        else:
            dtype = pa.float32()
        fields.append(pa.field(name, dtype))
    return pa.schema(fields, metadata={b"schema": PANEL_SCHEMA_VERSION.encode("ascii")})


class _YearPanelWriter:
    def __init__(self, path: Path, schema: pa.Schema, row_group_size: int) -> None:
        self.path = path
        self.temporary = path.with_suffix(path.suffix + ".partial")
        self.schema = schema
        self.row_group_size = int(row_group_size)
        self.writer: pq.ParquetWriter | None = None
        self.rows = 0
        self.first_date: str | None = None
        self.last_date: str | None = None
        self.temporary.parent.mkdir(parents=True, exist_ok=True)
        self.temporary.unlink(missing_ok=True)

    def append(self, columns: Mapping[str, Any]) -> None:
        rows = len(np.asarray(columns["input_row_idx"]))
        if rows == 0:
            return
        table = pa.Table.from_pydict(dict(columns), schema=self.schema)
        if self.writer is None:
            self.writer = pq.ParquetWriter(
                self.temporary, self.schema, compression="zstd"
            )
        self.writer.write_table(table, row_group_size=self.row_group_size)
        dates = [str(value) for value in columns["trade_date"]]
        first = min(dates)
        last = max(dates)
        self.first_date = min(self.first_date or first, first)
        self.last_date = max(self.last_date or last, last)
        self.rows += rows

    def close(self) -> None:
        if self.writer is None:
            pq.write_table(pa.Table.from_pylist([], schema=self.schema), self.temporary)
        else:
            self.writer.close()
        os.replace(self.temporary, self.path)


def _load_row_lookup(
    row_index_path: Path,
) -> tuple[np.ndarray, dict[str, int], np.ndarray, np.ndarray]:
    table = pq.read_table(
        row_index_path,
        columns=["date_idx", "symbol_idx", "trade_date", "symbol"],
    )
    rows = table.num_rows
    date_indices = table["date_idx"].to_numpy(zero_copy_only=False).astype(np.int32)
    symbol_indices = table["symbol_idx"].to_numpy(zero_copy_only=False).astype(np.int32)
    dates = np.asarray(table["trade_date"].to_pylist(), dtype=object)
    symbols = np.asarray(table["symbol"].to_pylist(), dtype=object)
    maximum_date = int(date_indices.max())
    maximum_symbol = int(symbol_indices.max())
    lookup = np.full((maximum_date + 1, maximum_symbol + 1), -1, dtype=np.int64)
    ordinal = np.arange(rows, dtype=np.int64)
    keys = date_indices.astype(np.int64) * (maximum_symbol + 1) + symbol_indices
    if np.unique(keys).size != rows:
        raise ValueError("causal_path_structure_duplicate_row_index_key")
    lookup[date_indices, symbol_indices] = ordinal
    mapping_frame = pd.DataFrame({"symbol": symbols, "symbol_idx": symbol_indices})
    inconsistent = mapping_frame.groupby("symbol", sort=False)["symbol_idx"].nunique()
    if int((inconsistent > 1).sum()) != 0:
        raise ValueError("causal_path_structure_symbol_mapping_inconsistent")
    reverse_inconsistent = mapping_frame.groupby("symbol_idx", sort=False)[
        "symbol"
    ].nunique()
    if int((reverse_inconsistent > 1).sum()) != 0:
        raise ValueError("causal_path_structure_symbol_reverse_mapping_inconsistent")
    symbol_map = {
        str(symbol): int(group.iloc[0])
        for symbol, group in mapping_frame.groupby("symbol", sort=False)["symbol_idx"]
    }
    return lookup, symbol_map, dates, symbol_indices


def _symbol_panel_columns(
    frame: pd.DataFrame,
    *,
    symbol_idx: int,
    lookup: np.ndarray,
    study: Mapping[str, Any],
) -> tuple[dict[int, dict[str, Any]], np.ndarray]:
    bar_valid = frame["bar_valid"].astype("boolean").fillna(False).to_numpy(bool)
    valid = (
        bar_valid
        & np.isfinite(frame["adj_high"].to_numpy(np.float64))
        & np.isfinite(frame["adj_low"].to_numpy(np.float64))
        & np.isfinite(frame["adj_close"].to_numpy(np.float64))
        & frame["adj_high"].to_numpy(np.float64).__gt__(0)
        & frame["adj_low"].to_numpy(np.float64).__gt__(0)
        & frame["adj_close"].to_numpy(np.float64).__gt__(0)
    )
    work = frame.loc[valid].sort_values("date_idx", kind="mergesort")
    if work.empty:
        return {}, np.empty(0, dtype=np.int64)
    date_indices = work["date_idx"].to_numpy(np.int64)
    inside = (date_indices >= 0) & (date_indices < lookup.shape[0])
    ordinals = np.full(len(work), -1, dtype=np.int64)
    ordinals[inside] = lookup[date_indices[inside], int(symbol_idx)]
    selected = np.flatnonzero(ordinals >= 0)
    if not len(selected):
        return {}, np.empty(0, dtype=np.int64)
    features = parse_causal_path(
        adj_high=work["adj_high"].to_numpy(np.float64),
        adj_low=work["adj_low"].to_numpy(np.float64),
        adj_close=work["adj_close"].to_numpy(np.float64),
        amount=work["amount"].to_numpy(np.float64),
        date_indices=date_indices,
        study=study,
    )
    selected_dates = work["trade_date"].astype(str).to_numpy()[selected]
    selected_years = np.asarray(
        [int(value[:4]) for value in selected_dates], dtype=np.int16
    )
    by_year: dict[int, dict[str, Any]] = {}
    for year in np.unique(selected_years):
        positions = selected[selected_years == year]
        columns: dict[str, Any] = {
            "input_row_idx": ordinals[positions],
            "trade_date": work["trade_date"].astype(str).to_numpy()[positions],
            "date_idx": date_indices[positions].astype(np.int32),
            "symbol_idx": np.full(len(positions), int(symbol_idx), dtype=np.int32),
        }
        for name in FEATURE_COLUMNS:
            values = features[name][positions]
            if name in FLOAT_FEATURES:
                values = values.astype(np.float32)
            columns[name] = values
        by_year[int(year)] = columns
    return by_year, ordinals[selected]


def _prepared_panel_valid(
    manifest: Mapping[str, Any], fingerprint: str, expected_rows: int
) -> bool:
    if (
        manifest.get("schema") != PANEL_SCHEMA_VERSION
        or manifest.get("status") != "prepared"
        or str(manifest.get("experiment_fingerprint")) != str(fingerprint)
        or int(manifest.get("rows", -1)) != int(expected_rows)
    ):
        return False
    records = list(manifest.get("structure_panels", []))
    return len(records) == 14 and all(
        Path(str(record["path"])).is_file() for record in records
    )


def prepare_structure_panels(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    study_path = _resolve(study_path)
    output_root = _resolve(output_root)
    study = load_study(study_path)
    contract, fingerprint = _source_contract(study_path, study)
    panel_manifest_path = output_root / "panel_manifest.json"
    expected_rows = int(contract["row_index_rows"])
    if panel_manifest_path.is_file() and not force:
        current = _read_json(panel_manifest_path)
        if _prepared_panel_valid(current, fingerprint, expected_rows):
            return current
        if str(current.get("experiment_fingerprint", "")) not in {"", fingerprint}:
            raise ValueError(
                "causal_path_structure_existing_panel_fingerprint_mismatch"
            )

    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "progress.json"
    _write_json(
        progress_path,
        {"status": "loading_quality_pool_index", "fingerprint": fingerprint},
    )
    row_index_path = Path(str(dict(contract["row_index"])["path"]))
    lookup, symbol_map, row_dates, row_symbol_indices = _load_row_lookup(row_index_path)
    seen = np.zeros(expected_rows, dtype=bool)
    schema = _panel_schema()
    row_group_size = int(dict(study["resources"])["parquet_row_group_size"])
    writers = {
        year: _YearPanelWriter(
            output_root / f"structure_panels/year={year}/part-0000.parquet",
            schema,
            row_group_size,
        )
        for year in range(2012, 2026)
    }
    dense_path = Path(str(dict(contract["dense_base"])["path"]))
    parquet = pq.ParquetFile(dense_path)
    pending: pd.DataFrame | None = None
    processed_symbols = 0
    eligible_symbols = 0
    dense_rows = 0

    def process(frame: pd.DataFrame) -> None:
        nonlocal processed_symbols, eligible_symbols
        if frame.empty:
            return
        processed_symbols += 1
        symbol = str(frame.iloc[0]["symbol"])
        symbol_idx = symbol_map.get(symbol)
        if symbol_idx is None:
            return
        eligible_symbols += 1
        by_year, ordinals = _symbol_panel_columns(
            frame,
            symbol_idx=symbol_idx,
            lookup=lookup,
            study=study,
        )
        if len(ordinals):
            if np.any(seen[ordinals]):
                raise ValueError("causal_path_structure_duplicate_emitted_row")
            seen[ordinals] = True
        for year, columns in by_year.items():
            writers[year].append(columns)

    resources = dict(study["resources"])
    try:
        for batch in parquet.iter_batches(
            batch_size=int(resources["stream_batch_rows"]),
            columns=[
                "symbol",
                "trade_date",
                "date_idx",
                "adj_high",
                "adj_low",
                "adj_close",
                "amount",
                "bar_valid",
            ],
        ):
            frame = batch.to_pandas()
            dense_rows += len(frame)
            if pending is not None:
                frame = pd.concat([pending, frame], ignore_index=True)
                pending = None
            last_symbol = str(frame.iloc[-1]["symbol"])
            complete = frame[frame["symbol"].astype(str).ne(last_symbol)]
            pending = frame[frame["symbol"].astype(str).eq(last_symbol)].copy()
            for _, group in complete.groupby("symbol", sort=False):
                process(group)
        if pending is not None:
            process(pending)
    finally:
        for writer in writers.values():
            writer.close()

    missing_rows = int((~seen).sum())
    emitted_rows = int(seen.sum())
    if missing_rows:
        missing_positions = np.flatnonzero(~seen)[:20]
        sample = [
            {
                "input_row_idx": int(pos),
                "trade_date": str(row_dates[pos]),
                "symbol_idx": int(row_symbol_indices[pos]),
            }
            for pos in missing_positions
        ]
        raise ValueError(f"causal_path_structure_missing_pool_rows:{sample}")
    records: list[dict[str, Any]] = []
    for year, writer in writers.items():
        records.append(
            {
                **atlas._file_record(writer.path),
                "rows": int(writer.rows),
                "year": int(year),
                "first_date": writer.first_date,
                "last_date": writer.last_date,
            }
        )
    manifest = {
        "schema": PANEL_SCHEMA_VERSION,
        "status": "prepared",
        "study_id": STUDY_ID,
        "builder_version": BUILDER_VERSION,
        "experiment_fingerprint": fingerprint,
        "study": atlas._file_record(study_path),
        "source_contract": contract,
        "feature_columns": list(FEATURE_COLUMNS),
        "structure_panels": records,
        "rows": emitted_rows,
        "dates": int(np.unique(row_dates).size),
        "symbols": len(symbol_map),
        "processed_dense_symbols": int(processed_symbols),
        "eligible_symbols": int(eligible_symbols),
        "dense_rows": int(dense_rows),
        "duplicate_rows": 0,
        "missing_rows": 0,
        "forbidden_2026_rows": 0,
        "future_confirmed_structure_written_back": False,
    }
    _write_json(panel_manifest_path, manifest)
    _write_json(
        progress_path,
        {
            "status": "structure_panels_prepared",
            "panel_manifest": str(panel_manifest_path.resolve()),
        },
    )
    return manifest


def _duckdb_runtime(study: Mapping[str, Any]) -> dict[str, Any]:
    resources = dict(study["resources"])
    proxy = {
        "resources": {
            "duckdb_adaptive": bool(resources.get("adaptive_memory", True)),
            "duckdb_min_memory_mb": 1024,
            "duckdb_max_memory_mb": 8192,
            "duckdb_reserve_memory_mb": int(resources["reserve_memory_mb"]),
            "duckdb_available_memory_fraction": float(
                resources["available_memory_fraction"]
            ),
            "duckdb_max_threads": int(resources["maximum_threads"]),
            "duckdb_reserve_logical_cores": 2,
            "duckdb_memory_per_thread_mb": 512,
        }
    }
    return atlas._duckdb_runtime_resources(proxy)


def _connect(
    output_root: Path, study: Mapping[str, Any]
) -> tuple[duckdb.DuckDBPyConnection, dict[str, Any]]:
    runtime = _duckdb_runtime(study)
    connection = duckdb.connect()
    connection.execute(f"PRAGMA threads={int(runtime['threads'])}")
    connection.execute("PRAGMA enable_progress_bar=false")
    connection.execute("PRAGMA preserve_insertion_order=false")
    connection.execute(
        f"PRAGMA memory_limit={atlas._sql_quote(runtime['memory_limit'])}"
    )
    temporary = output_root / "duckdb_tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    connection.execute("PRAGMA temp_directory=?", [str(temporary)])
    return connection, runtime


def _records_by_year(records: Sequence[Mapping[str, Any]]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for record in records:
        year = int(record["year"])
        path = Path(str(record["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"causal_path_structure_partition_missing:{path}")
        result[year] = path
    return result


def _prediction_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        path = Path(str(item["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"causal_path_structure_prediction_missing:{path}")
        item["path"] = path
        result.append(item)
    return sorted(
        result,
        key=lambda item: (str(item["cost_scenario"]), int(item["oos_year"])),
    )


def _load_evaluation_partition(
    connection: duckdb.DuckDBPyConnection,
    *,
    prediction_path: Path,
    structure_path: Path,
    coordinate_path: Path,
) -> pd.DataFrame:
    structure_columns = ",\n            ".join(
        f"s.pattern_{name}" for name in BASE_PATTERN_NAMES
    )
    frame = connection.execute(
        f"""
        SELECT
            p.input_row_idx,
            p.trade_date,
            p.date_idx,
            p.symbol_idx,
            p.risk_match_cell_code,
            p.actual_buy_advantage_vs_cash,
            p.actual_positive,
            p.actual_upside_component,
            p.actual_downside_component,
            {structure_columns},
            c.price_amount_correlation_5d_rank,
            c.minute_last_30m_return_rank
        FROM read_parquet({atlas._sql_quote(prediction_path)}) p
        INNER JOIN read_parquet({atlas._sql_quote(structure_path)}) s
          USING (input_row_idx)
        INNER JOIN read_parquet({atlas._sql_quote(coordinate_path)}) c
          USING (input_row_idx)
        ORDER BY p.date_idx, p.symbol_idx
        """
    ).fetchdf()
    if frame.empty:
        raise ValueError("causal_path_structure_empty_evaluation_partition")
    if frame["input_row_idx"].duplicated().any():
        raise ValueError("causal_path_structure_duplicate_evaluation_rows")
    if not np.isfinite(frame["actual_buy_advantage_vs_cash"].to_numpy(float)).all():
        raise ValueError("causal_path_structure_nonfinite_action_value")
    frame["pattern_up_breakout_confirmed_participation"] = (
        frame["pattern_zone_breakout_up"].astype(bool)
        & frame["price_amount_correlation_5d_rank"].ge(0.5)
        & frame["minute_last_30m_return_rank"].lt(0.5)
    )
    frame["pattern_up_breakout_late_chase"] = frame["pattern_zone_breakout_up"].astype(
        bool
    ) & frame["minute_last_30m_return_rank"].ge(0.5)
    return frame


def _daily_pattern_diagnostics(
    frame: pd.DataFrame,
    *,
    pattern: str,
    cost_scenario: str,
    evaluation_year: int,
) -> pd.DataFrame:
    candidate = frame[f"pattern_{pattern}"].fillna(False).to_numpy(bool)
    if not np.any(candidate):
        return pd.DataFrame()
    work = frame[
        [
            "trade_date",
            "date_idx",
            "risk_match_cell_code",
            "actual_buy_advantage_vs_cash",
            "actual_positive",
            "actual_upside_component",
            "actual_downside_component",
        ]
    ].copy()
    work["candidate"] = candidate
    work["positive_int"] = work["actual_positive"].astype(np.int8)
    work["negative_int"] = (~work["actual_positive"].astype(bool)).astype(np.int8)

    candidate_rows = work[work["candidate"]]
    absolute = (
        candidate_rows.groupby(["trade_date", "date_idx"], as_index=False)
        .agg(
            candidate_rows=("candidate", "size"),
            candidate_action_value=("actual_buy_advantage_vs_cash", "mean"),
            candidate_positive_fraction=("positive_int", "mean"),
            candidate_upside_component=("actual_upside_component", "mean"),
            candidate_downside_component=("actual_downside_component", "mean"),
            candidate_positive_rows=("positive_int", "sum"),
            candidate_negative_rows=("negative_int", "sum"),
        )
        .sort_values("date_idx")
    )
    totals = work.groupby(["trade_date", "date_idx"], as_index=False).size()
    totals = totals.rename(columns={"size": "all_rows"})
    absolute = absolute.merge(totals, on=["trade_date", "date_idx"], how="left")
    absolute["candidate_share"] = absolute["candidate_rows"] / absolute["all_rows"]

    keys = ["trade_date", "date_idx", "risk_match_cell_code"]
    candidate_cells = candidate_rows.groupby(keys, as_index=False).agg(
        candidate_cell_rows=("candidate", "size"),
        candidate_cell_value=("actual_buy_advantage_vs_cash", "mean"),
        candidate_cell_positive=("positive_int", "mean"),
        candidate_cell_upside=("actual_upside_component", "mean"),
        candidate_cell_downside=("actual_downside_component", "mean"),
        candidate_cell_positive_rows=("positive_int", "sum"),
        candidate_cell_negative_rows=("negative_int", "sum"),
    )
    control_cells = (
        work[~work["candidate"]]
        .groupby(keys, as_index=False)
        .agg(
            control_cell_rows=("candidate", "size"),
            control_cell_value=("actual_buy_advantage_vs_cash", "mean"),
            control_cell_positive=("positive_int", "mean"),
            control_cell_upside=("actual_upside_component", "mean"),
            control_cell_downside=("actual_downside_component", "mean"),
        )
    )
    matched = candidate_cells.merge(control_cells, on=keys, how="inner")
    if matched.empty:
        absolute["matched_cells"] = 0
        for name in (
            "matched_action_value_difference",
            "matched_positive_fraction_difference",
            "matched_upside_difference",
            "matched_downside_difference",
            "matched_failure_positive_rows",
            "matched_candidate_positive_rows",
        ):
            absolute[name] = np.nan
    else:
        matched["matched_action_value_difference"] = (
            matched["candidate_cell_value"] - matched["control_cell_value"]
        )
        matched["matched_positive_fraction_difference"] = (
            matched["candidate_cell_positive"] - matched["control_cell_positive"]
        )
        matched["matched_upside_difference"] = (
            matched["candidate_cell_upside"] - matched["control_cell_upside"]
        )
        matched["matched_downside_difference"] = (
            matched["candidate_cell_downside"] - matched["control_cell_downside"]
        )
        matched["matched_failure_positive_rows"] = np.where(
            matched["candidate_cell_negative_rows"] > 0,
            matched["candidate_cell_positive_rows"],
            0,
        )
        daily_matched = matched.groupby(["trade_date", "date_idx"], as_index=False).agg(
            matched_cells=("risk_match_cell_code", "size"),
            matched_action_value_difference=(
                "matched_action_value_difference",
                "mean",
            ),
            matched_positive_fraction_difference=(
                "matched_positive_fraction_difference",
                "mean",
            ),
            matched_upside_difference=("matched_upside_difference", "mean"),
            matched_downside_difference=("matched_downside_difference", "mean"),
            matched_failure_positive_rows=(
                "matched_failure_positive_rows",
                "sum",
            ),
            matched_candidate_positive_rows=(
                "candidate_cell_positive_rows",
                "sum",
            ),
        )
        absolute = absolute.merge(
            daily_matched, on=["trade_date", "date_idx"], how="left"
        )
    absolute.insert(0, "evaluation_year", int(evaluation_year))
    absolute.insert(0, "cost_scenario", str(cost_scenario))
    absolute.insert(0, "pattern", str(pattern))
    return absolute


def _annual_pattern_diagnostics(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["cost_scenario", "evaluation_year", "pattern"]
    for values, group in daily.groupby(keys, sort=True):
        cost, year, pattern = values
        matched = group[group["matched_action_value_difference"].notna()]
        matched_positive_total = float(
            matched["matched_candidate_positive_rows"].fillna(0).sum()
        )
        rows.append(
            {
                "cost_scenario": str(cost),
                "evaluation_year": int(year),
                "pattern": str(pattern),
                "candidate_dates": int(group["date_idx"].nunique()),
                "candidate_rows": int(group["candidate_rows"].sum()),
                "mean_candidate_share": float(group["candidate_share"].mean()),
                "date_equal_action_value": float(
                    group["candidate_action_value"].mean()
                ),
                "date_equal_positive_fraction": float(
                    group["candidate_positive_fraction"].mean()
                ),
                "date_equal_upside_component": float(
                    group["candidate_upside_component"].mean()
                ),
                "date_equal_downside_component": float(
                    group["candidate_downside_component"].mean()
                ),
                "matched_dates": int(matched["date_idx"].nunique()),
                "matched_cells": int(matched["matched_cells"].fillna(0).sum()),
                "matched_action_value_difference": (
                    float(matched["matched_action_value_difference"].mean())
                    if not matched.empty
                    else math.nan
                ),
                "matched_positive_fraction_difference": (
                    float(matched["matched_positive_fraction_difference"].mean())
                    if not matched.empty
                    else math.nan
                ),
                "matched_upside_difference": (
                    float(matched["matched_upside_difference"].mean())
                    if not matched.empty
                    else math.nan
                ),
                "matched_downside_difference": (
                    float(matched["matched_downside_difference"].mean())
                    if not matched.empty
                    else math.nan
                ),
                "matched_failure_fraction_of_positive_candidates": (
                    float(matched["matched_failure_positive_rows"].fillna(0).sum())
                    / matched_positive_total
                    if matched_positive_total > 0
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _normal_p_value(mean: float, standard_error: float) -> float:
    if (
        not math.isfinite(mean)
        or not math.isfinite(standard_error)
        or standard_error <= 0
    ):
        return math.nan
    return float(2.0 * stats.norm.sf(abs(mean / standard_error)))


def _benjamini_hochberg(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    output = np.full(len(source), np.nan, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(source))
    if not len(finite):
        return output
    selected = source[finite]
    order = np.argsort(selected, kind="mergesort")
    ranked = selected[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty(len(ranked), dtype=np.float64)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    output[finite] = restored
    return output


def _aggregate_pattern_diagnostics(
    daily: pd.DataFrame,
    annual: pd.DataFrame,
    *,
    hac_lag: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (cost, pattern), group in daily.groupby(
        ["cost_scenario", "pattern"], sort=True
    ):
        ordered = group.sort_values("date_idx", kind="mergesort")
        absolute = atlas._hac_mean(ordered["candidate_action_value"], lag=hac_lag)
        positive = atlas._hac_mean(ordered["candidate_positive_fraction"], lag=hac_lag)
        upside = atlas._hac_mean(ordered["candidate_upside_component"], lag=hac_lag)
        downside = atlas._hac_mean(ordered["candidate_downside_component"], lag=hac_lag)
        matched = ordered[ordered["matched_action_value_difference"].notna()]
        matched_value = atlas._hac_mean(
            matched["matched_action_value_difference"], lag=hac_lag
        )
        matched_positive = atlas._hac_mean(
            matched["matched_positive_fraction_difference"], lag=hac_lag
        )
        matched_upside = atlas._hac_mean(
            matched["matched_upside_difference"], lag=hac_lag
        )
        matched_downside = atlas._hac_mean(
            matched["matched_downside_difference"], lag=hac_lag
        )
        yearly = annual[
            annual["cost_scenario"].eq(cost) & annual["pattern"].eq(pattern)
        ]
        matched_positive_total = float(
            matched["matched_candidate_positive_rows"].fillna(0).sum()
        )
        rows.append(
            {
                "cost_scenario": str(cost),
                "pattern": str(pattern),
                "candidate_dates": int(ordered["date_idx"].nunique()),
                "candidate_rows": int(ordered["candidate_rows"].sum()),
                "mean_candidate_share": float(ordered["candidate_share"].mean()),
                "action_value_mean": float(absolute["mean"]),
                "action_value_hac_se": float(absolute["se"]),
                "action_value_lcb_95": float(absolute["lcb_95"]),
                "action_value_ucb_95": float(absolute["ucb_95"]),
                "action_value_p_two_sided": _normal_p_value(
                    float(absolute["mean"]), float(absolute["se"])
                ),
                "positive_fraction": float(positive["mean"]),
                "upside_component": float(upside["mean"]),
                "downside_component": float(downside["mean"]),
                "matched_dates": int(matched["date_idx"].nunique()),
                "matched_action_value_difference": float(matched_value["mean"]),
                "matched_action_value_hac_se": float(matched_value["se"]),
                "matched_action_value_lcb_95": float(matched_value["lcb_95"]),
                "matched_action_value_ucb_95": float(matched_value["ucb_95"]),
                "matched_action_value_p_two_sided": _normal_p_value(
                    float(matched_value["mean"]), float(matched_value["se"])
                ),
                "matched_positive_fraction_difference": float(matched_positive["mean"]),
                "matched_upside_difference": float(matched_upside["mean"]),
                "matched_downside_difference": float(matched_downside["mean"]),
                "positive_action_value_years": int(
                    yearly["date_equal_action_value"].gt(0).sum()
                ),
                "years": len(yearly),
                "matched_failure_fraction_of_positive_candidates": (
                    float(matched["matched_failure_positive_rows"].fillna(0).sum())
                    / matched_positive_total
                    if matched_positive_total > 0
                    else math.nan
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["matched_action_value_bh_q"] = np.nan
    for cost, indices in result.groupby("cost_scenario").groups.items():
        del cost
        positions = np.asarray(list(indices), dtype=np.int64)
        result.loc[positions, "matched_action_value_bh_q"] = _benjamini_hochberg(
            result.loc[positions, "matched_action_value_p_two_sided"].to_numpy(float)
        )
    return result


def _promotion_gate(
    aggregate: pd.DataFrame, study: Mapping[str, Any]
) -> dict[str, Any]:
    costs = tuple(str(value) for value in dict(study["evaluation"])["cost_scenarios"])
    evaluation = dict(study["evaluation"])
    total_years = len(dict(study["period"])["diagnostic_oos_years"])
    majority = total_years // 2 + 1
    decisions: list[dict[str, Any]] = []
    passed_patterns: list[str] = []
    for pattern in PATTERN_NAMES:
        rows = aggregate[aggregate["pattern"].eq(pattern)].set_index("cost_scenario")
        cost_pass: dict[str, bool] = {}
        details: dict[str, Any] = {}
        for cost in costs:
            if cost not in rows.index:
                cost_pass[cost] = False
                details[cost] = {"missing": True}
                continue
            row = rows.loc[cost]
            checks = {
                "minimum_candidate_dates": int(row["candidate_dates"])
                >= int(evaluation["minimum_candidate_dates"]),
                "minimum_matched_dates": int(row["matched_dates"])
                >= int(evaluation["minimum_matched_dates"]),
                "positive_absolute_lcb": float(row["action_value_lcb_95"]) > 0,
                "positive_matched_lcb": float(row["matched_action_value_lcb_95"]) > 0,
                "matched_effect_bh_significant": float(row["matched_action_value_bh_q"])
                <= 0.05,
                "majority_positive_years": int(row["positive_action_value_years"])
                >= majority,
                "nonworsening_matched_downside": float(
                    row["matched_downside_difference"]
                )
                <= 0,
            }
            cost_pass[cost] = all(checks.values())
            details[cost] = checks
        passed = all(cost_pass.values())
        if passed:
            passed_patterns.append(pattern)
        decisions.append(
            {
                "pattern": pattern,
                "passed": passed,
                "cost_pass": cost_pass,
                "checks": details,
            }
        )
    return {
        "status": "passed" if passed_patterns else "failed",
        "passed_patterns": passed_patterns,
        "account_replay_allowed": bool(passed_patterns),
        "decisions": decisions,
    }


def analyze_structure_patterns(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study_path = _resolve(study_path)
    output_root = _resolve(output_root)
    study = load_study(study_path)
    panel_manifest = prepare_structure_panels(
        study_path=study_path, output_root=output_root, force=False
    )
    contract = dict(panel_manifest["source_contract"])
    structure_by_year = _records_by_year(panel_manifest["structure_panels"])
    coordinate_by_year = _records_by_year(contract["coordinate_partitions"])
    predictions = _prediction_records(contract["prediction_partitions"])
    configured_years = {
        int(value) for value in dict(study["period"])["diagnostic_oos_years"]
    }
    progress_path = output_root / "progress.json"
    _write_json(progress_path, {"status": "evaluating_candidate_patterns"})
    connection, runtime = _connect(output_root, study)
    daily_parts: list[pd.DataFrame] = []
    partition_audit: list[dict[str, Any]] = []
    try:
        for record in predictions:
            year = int(record["oos_year"])
            if year not in configured_years:
                continue
            cost = str(record["cost_scenario"])
            frame = _load_evaluation_partition(
                connection,
                prediction_path=Path(record["path"]),
                structure_path=structure_by_year[year],
                coordinate_path=coordinate_by_year[year],
            )
            if not frame["trade_date"].astype(str).str.startswith(str(year)).all():
                raise ValueError("causal_path_structure_prediction_year_mismatch")
            partition_audit.append(
                {
                    "cost_scenario": cost,
                    "evaluation_year": year,
                    "rows": len(frame),
                    "dates": int(frame["date_idx"].nunique()),
                }
            )
            for pattern in PATTERN_NAMES:
                daily = _daily_pattern_diagnostics(
                    frame,
                    pattern=pattern,
                    cost_scenario=cost,
                    evaluation_year=year,
                )
                if not daily.empty:
                    daily_parts.append(daily)
    finally:
        connection.close()
    if not daily_parts:
        raise ValueError("causal_path_structure_no_pattern_diagnostics")
    daily = pd.concat(daily_parts, ignore_index=True)
    annual = _annual_pattern_diagnostics(daily)
    aggregate = _aggregate_pattern_diagnostics(
        daily,
        annual,
        hac_lag=int(dict(study["evaluation"])["hac_lag"]),
    )
    gate = _promotion_gate(aggregate, study)

    outputs = {
        "daily_pattern_diagnostics": output_root / "daily_pattern_diagnostics.parquet",
        "annual_pattern_diagnostics": output_root
        / "annual_pattern_diagnostics.parquet",
        "aggregate_pattern_diagnostics": output_root
        / "aggregate_pattern_diagnostics.parquet",
        "promotion_gate": output_root / "promotion_gate.json",
    }
    daily.to_parquet(outputs["daily_pattern_diagnostics"], index=False)
    annual.to_parquet(outputs["annual_pattern_diagnostics"], index=False)
    aggregate.to_parquet(outputs["aggregate_pattern_diagnostics"], index=False)
    _write_json(outputs["promotion_gate"], gate)

    audit = {
        "structure_rows": int(panel_manifest["rows"]),
        "structure_partitions": len(panel_manifest["structure_panels"]),
        "prediction_partitions": len(partition_audit),
        "prediction_rows": int(sum(item["rows"] for item in partition_audit)),
        "cost_scenarios": sorted({item["cost_scenario"] for item in partition_audit}),
        "evaluation_years": sorted(
            {item["evaluation_year"] for item in partition_audit}
        ),
        "patterns": list(PATTERN_NAMES),
        "daily_rows": len(daily),
        "annual_rows": len(annual),
        "aggregate_rows": len(aggregate),
        "forbidden_2026_rows": 0,
        "future_confirmed_structure_written_back": False,
        "fixed_holding_horizon_used": False,
        "binary_good_stock_label_used": False,
        "account_replay_allowed": bool(gate["account_replay_allowed"]),
    }
    output_frames = {
        "daily_pattern_diagnostics": daily,
        "annual_pattern_diagnostics": annual,
        "aggregate_pattern_diagnostics": aggregate,
    }
    output_records: dict[str, dict[str, Any]] = {}
    for key, path in outputs.items():
        record = atlas._file_record(path)
        if key in output_frames:
            record["rows"] = len(output_frames[key])
        output_records[key] = record

    manifest = {
        "schema": MANIFEST_SCHEMA_VERSION,
        "status": "completed",
        "study_id": STUDY_ID,
        "builder_version": BUILDER_VERSION,
        "experiment_fingerprint": panel_manifest["experiment_fingerprint"],
        "study": atlas._file_record(study_path),
        "panel_manifest": atlas._file_record(output_root / "panel_manifest.json"),
        "source_contract": contract,
        "feature_columns": list(FEATURE_COLUMNS),
        "patterns": list(PATTERN_NAMES),
        "partition_audit": partition_audit,
        "outputs": output_records,
        "runtime": runtime,
        "audit": audit,
        "promotion_gate": gate,
        "training_performed": False,
        "causal_structure_evaluation_performed": True,
        "portfolio_execution_performed": False,
        "profit_claim_allowed": False,
        "production_policy_selected": False,
        "report_generation_performed": False,
    }
    manifest_path = output_root / "manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(
        progress_path,
        {"status": "completed", "manifest": str(manifest_path.resolve())},
    )
    return manifest


def run_study(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force_prepare: bool = False,
) -> dict[str, Any]:
    prepare_structure_panels(
        study_path=study_path,
        output_root=output_root,
        force=force_prepare,
    )
    return analyze_structure_patterns(study_path=study_path, output_root=output_root)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and test strictly causal multi-scale path structures."
    )
    parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.prepare_only and args.analyze_only:
        raise ValueError("causal_path_structure_conflicting_modes")
    if args.prepare_only:
        result = prepare_structure_panels(
            study_path=args.study,
            output_root=args.output_root,
            force=bool(args.force_prepare),
        )
    elif args.analyze_only:
        result = analyze_structure_patterns(
            study_path=args.study, output_root=args.output_root
        )
    else:
        result = run_study(
            study_path=args.study,
            output_root=args.output_root,
            force_prepare=bool(args.force_prepare),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=atlas._json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
