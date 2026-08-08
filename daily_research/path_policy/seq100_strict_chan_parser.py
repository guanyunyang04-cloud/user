"""Strictly causal Chan-style path parser.

This module is intentionally separate from ``seq100_causal_path_structure``.
It implements the frozen definition/ambiguity contract and emits immutable
event records with distinct event and confirmation times.  The parser is a
market-path representation, not a profitability claim.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEFINITION_PATH = (
    WORKSPACE_ROOT
    / "daily_research"
    / "studies"
    / "seq100_strict_chan_definition_v1.json"
)
STUDY_ID = "seq100_strict_chan_definition_v1"
PARSER_SCHEMA_VERSION = "seq100_strict_chan_parser/1"

REQUIRED_INPUT_COLUMNS = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
)


@dataclass(frozen=True)
class NormalizedBar:
    id: str
    position: int
    start_index: int
    end_index: int
    high_index: int
    low_index: int
    start_time: str
    end_time: str
    confirmed_index: int
    confirmed_time: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    direction: int


@dataclass(frozen=True)
class Fractal:
    id: str
    kind: int
    kind_name: str
    left_bar_id: str
    middle_bar_id: str
    right_bar_id: str
    middle_position: int
    range_low: float
    range_high: float
    extreme_index: int
    extreme_time: str
    extreme_price: float
    confirmed_index: int
    confirmed_time: str


@dataclass(frozen=True)
class Stroke:
    id: str
    direction: int
    start_fractal_id: str
    end_fractal_id: str
    start_normalized_position: int
    end_normalized_position: int
    start_index: int
    end_index: int
    start_time: str
    end_time: str
    start_price: float
    end_price: float
    low: float
    high: float
    accepted_index: int
    accepted_time: str
    locked_index: int
    locked_time: str
    normalized_bar_span: int
    raw_bar_span: int


@dataclass(frozen=True)
class SegmentStateEvent:
    id: str
    candidate_id: str
    action: str
    direction: int
    segment_start_stroke_position: int
    endpoint_stroke_position: int
    middle_yin_stroke_id: str
    break_case: int
    event_index: int
    event_time: str
    confirmed_index: int
    confirmed_time: str


@dataclass(frozen=True)
class Segment:
    id: str
    direction: int
    start_stroke_position: int
    end_stroke_position: int
    stroke_ids: tuple[str, ...]
    start_index: int
    end_index: int
    start_time: str
    end_time: str
    start_price: float
    end_price: float
    low: float
    high: float
    event_index: int
    event_time: str
    confirmed_index: int
    confirmed_time: str
    break_case: int
    feature_fractal_id: str
    raw_bar_span: int
    price_change: float
    duration_normalized_price_change: float
    macd_histogram_area: float
    amount_per_bar: float


@dataclass(frozen=True)
class Center:
    id: str
    level: int
    component_kind: str
    component_ids: tuple[str, str, str]
    start_component_position: int
    end_component_position: int
    core_low: float
    core_high: float
    fluctuation_low: float
    fluctuation_high: float
    event_index: int
    event_time: str
    confirmed_index: int
    confirmed_time: str
    construction_variant: str


@dataclass(frozen=True)
class CenterExtension:
    id: str
    center_id: str
    level: int
    component_id: str
    component_position: int
    component_low: float
    component_high: float
    confirmed_index: int
    confirmed_time: str


@dataclass(frozen=True)
class CenterRelation:
    id: str
    level: int
    previous_center_id: str
    current_center_id: str
    relation: str
    direction: int
    comparison_variant: str
    confirmed_index: int
    confirmed_time: str


@dataclass(frozen=True)
class Movement:
    id: str
    level: int
    direction: int
    start_center_id: str
    end_center_id: str
    component_ids: tuple[str, ...]
    start_index: int
    end_index: int
    start_time: str
    end_time: str
    low: float
    high: float
    event_index: int
    event_time: str
    confirmed_index: int
    confirmed_time: str
    decomposition_variant: str


@dataclass(frozen=True)
class TrendType:
    id: str
    level: int
    kind: str
    direction: int
    center_ids: tuple[str, ...]
    event_index: int
    event_time: str
    confirmed_index: int
    confirmed_time: str


@dataclass(frozen=True)
class Divergence:
    id: str
    level: int
    direction: int
    metric: str
    trend_id: str
    previous_segment_id: str
    current_segment_id: str
    previous_strength: float
    current_strength: float
    strength_ratio: float
    event_index: int
    event_time: str
    confirmed_index: int
    confirmed_time: str


@dataclass(frozen=True)
class TradePoint:
    id: str
    point_type: int
    side: str
    level: int
    variant: str
    basis_id: str
    price: float
    event_index: int
    event_time: str
    confirmed_index: int
    confirmed_time: str


@dataclass(frozen=True)
class _WorkingNormalizedBar:
    start_index: int
    end_index: int
    high_index: int
    low_index: int
    start_time: str
    end_time: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    direction: int


@dataclass(frozen=True)
class _WorkingStroke:
    direction: int
    start: Fractal
    end: Fractal


@dataclass(frozen=True)
class _FeatureElement:
    id: str
    position: int
    low: float
    high: float
    low_stroke_position: int
    high_stroke_position: int
    source_stroke_positions: tuple[int, ...]
    confirmed_index: int
    confirmed_time: str
    direction: int


@dataclass(frozen=True)
class _WorkingFeatureElement:
    low: float
    high: float
    low_stroke_position: int
    high_stroke_position: int
    source_stroke_positions: tuple[int, ...]
    direction: int


@dataclass(frozen=True)
class _FeatureFractalCandidate:
    id: str
    direction: int
    endpoint_stroke_position: int
    endpoint_price: float
    event_index: int
    event_time: str
    detected_index: int
    detected_time: str
    break_case: int


@dataclass(frozen=True)
class _Component:
    id: str
    position: int
    direction: int
    low: float
    high: float
    start_index: int
    end_index: int
    start_time: str
    end_time: str
    confirmed_index: int
    confirmed_time: str


@dataclass(frozen=True)
class ChanParseResult:
    schema: str
    variant_selection: Mapping[str, str]
    normalized_bars: tuple[NormalizedBar, ...]
    provisional_normalized_bar: Mapping[str, Any] | None
    fractals: tuple[Fractal, ...]
    strokes: tuple[Stroke, ...]
    provisional_stroke: Mapping[str, Any] | None
    segment_state_events: tuple[SegmentStateEvent, ...]
    segments: tuple[Segment, ...]
    centers: tuple[Center, ...]
    center_extensions: tuple[CenterExtension, ...]
    center_relations: tuple[CenterRelation, ...]
    movements: tuple[Movement, ...]
    trend_types: tuple[TrendType, ...]
    divergences: tuple[Divergence, ...]
    trade_points: tuple[TradePoint, ...]

    def event_records(self) -> list[dict[str, Any]]:
        groups: tuple[tuple[str, Iterable[Any]], ...] = (
            ("normalized_bar", self.normalized_bars),
            ("fractal", self.fractals),
            ("stroke", self.strokes),
            ("segment_state", self.segment_state_events),
            ("segment", self.segments),
            ("center", self.centers),
            ("center_extension", self.center_extensions),
            ("center_relation", self.center_relations),
            ("movement", self.movements),
            ("trend_type", self.trend_types),
            ("divergence", self.divergences),
            ("trade_point", self.trade_points),
        )
        records: list[dict[str, Any]] = []
        for event_type, values in groups:
            for value in values:
                record = asdict(value)
                if event_type == "stroke":
                    record["event_index"] = int(record["accepted_index"])
                    record["event_time"] = str(record["accepted_time"])
                    record["confirmed_index"] = int(record["locked_index"])
                    record["confirmed_time"] = str(record["locked_time"])
                elif event_type == "fractal":
                    record["event_index"] = int(record["extreme_index"])
                    record["event_time"] = str(record["extreme_time"])
                elif event_type == "normalized_bar":
                    record["event_index"] = int(record["start_index"])
                    record["event_time"] = str(record["start_time"])
                record.setdefault("event_index", int(record["confirmed_index"]))
                record.setdefault("event_time", str(record["confirmed_time"]))
                record["event_type"] = event_type
                records.append(record)
        return sorted(
            records,
            key=lambda row: (
                int(row["confirmed_index"]),
                str(row["event_type"]),
                str(row["id"]),
            ),
        )

    def event_frames(self) -> dict[str, pd.DataFrame]:
        records = self.event_records()
        names = sorted({str(row["event_type"]) for row in records})
        return {
            name: pd.DataFrame(
                [row for row in records if str(row["event_type"]) == name]
            )
            for name in names
        }

    def summary(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "variant_selection": dict(self.variant_selection),
            "normalized_bars": len(self.normalized_bars),
            "fractals": len(self.fractals),
            "locked_strokes": len(self.strokes),
            "segments": len(self.segments),
            "pending_segment_breaks": sum(
                event.action == "opened" for event in self.segment_state_events
            )
            - sum(
                event.action in {"confirmed", "invalidated"}
                for event in self.segment_state_events
            ),
            "centers_by_level": _count_by(self.centers, "level"),
            "movements_by_level": _count_by(self.movements, "level"),
            "trend_types": len(self.trend_types),
            "divergences": len(self.divergences),
            "trade_points": len(self.trade_points),
        }


def _count_by(values: Iterable[Any], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(getattr(value, field))
        counts[key] = counts.get(key, 0) + 1
    return counts


def load_definition_spec(
    path: str | Path = DEFAULT_DEFINITION_PATH,
) -> dict[str, Any]:
    resolved = Path(path).resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("study_id") != STUDY_ID:
        raise ValueError("strict_chan_definition_study_id_mismatch")
    definitions = list(payload.get("definitions", []) or [])
    ids = [str(item.get("id", "")) for item in definitions]
    if not ids or len(ids) != len(set(ids)) or any(not item for item in ids):
        raise ValueError("strict_chan_definition_ids_invalid")
    valid_classes = set(dict(payload.get("epistemic_classes", {})))
    by_id = {str(item["id"]): dict(item) for item in definitions}
    for item in definitions:
        if str(item.get("class")) not in valid_classes:
            raise ValueError(f"strict_chan_definition_class_invalid:{item['id']}")
        missing = set(map(str, item.get("depends_on", []) or [])) - set(by_id)
        if missing:
            raise ValueError(
                f"strict_chan_definition_dependency_missing:{item['id']}:{sorted(missing)}"
            )
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(definition_id: str) -> None:
        if definition_id in visiting:
            raise ValueError(f"strict_chan_definition_cycle:{definition_id}")
        if definition_id in visited:
            return
        visiting.add(definition_id)
        for dependency in by_id[definition_id].get("depends_on", []) or []:
            visit(str(dependency))
        visiting.remove(definition_id)
        visited.add(definition_id)

    for definition_id in ids:
        visit(definition_id)
    variants = dict(payload.get("variants", {}) or {})
    for name, contract in variants.items():
        contract = dict(contract)
        options = list(contract.get("options", []) or [])
        if contract.get("primary") not in options:
            raise ValueError(f"strict_chan_variant_primary_invalid:{name}")
    boundaries = dict(payload.get("boundaries", {}) or {})
    if bool(boundaries.get("replaces_weak_grammar", True)):
        raise ValueError("strict_chan_must_not_replace_weak_grammar")
    if bool(boundaries.get("profit_claim_allowed", True)):
        raise ValueError("strict_chan_profit_claim_must_be_false")
    return payload


def primary_variants(spec: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(name): str(dict(contract)["primary"])
        for name, contract in dict(spec["variants"]).items()
        if name != "divergence_metrics"
    }


def _variant_selection(
    spec: Mapping[str, Any], overrides: Mapping[str, str] | None
) -> dict[str, str]:
    selected = primary_variants(spec)
    for name, value in dict(overrides or {}).items():
        if name == "divergence_metrics":
            continue
        contract = dict(dict(spec["variants"])[name])
        if value not in set(contract["options"]):
            raise ValueError(f"strict_chan_variant_invalid:{name}:{value}")
        selected[name] = str(value)
    return selected


def _iso(value: Any) -> str:
    return pd.Timestamp(value).isoformat()


def _validate_bars(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED_INPUT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"strict_chan_input_columns_missing:{sorted(missing)}")
    bars = frame.loc[:, REQUIRED_INPUT_COLUMNS].copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], errors="raise")
    if bars["timestamp"].duplicated().any():
        raise ValueError("strict_chan_input_timestamp_duplicate")
    if not bars["timestamp"].is_monotonic_increasing:
        raise ValueError("strict_chan_input_timestamp_not_sorted")
    numeric = [name for name in REQUIRED_INPUT_COLUMNS if name != "timestamp"]
    for name in numeric:
        bars[name] = pd.to_numeric(bars[name], errors="coerce")
    values = bars[numeric].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("strict_chan_input_nonfinite")
    if (
        (bars[["open", "high", "low", "close"]] <= 0).any().any()
        or (bars[["volume", "amount"]] < 0).any().any()
        or (bars["high"] < bars[["open", "close", "low"]].max(axis=1)).any()
        or (bars["low"] > bars[["open", "close", "high"]].min(axis=1)).any()
    ):
        raise ValueError("strict_chan_input_ohlcv_invalid")
    return bars.reset_index(drop=True)


def _contains(
    low_a: float,
    high_a: float,
    low_b: float,
    high_b: float,
    *,
    mode: str,
) -> bool:
    if mode == "closed_interval":
        return (low_a <= low_b and high_a >= high_b) or (
            low_b <= low_a and high_b >= high_a
        )
    if mode == "strict_containment":
        return (low_a < low_b and high_a > high_b) or (
            low_b < low_a and high_b > high_a
        )
    raise ValueError(f"strict_chan_inclusion_mode_unknown:{mode}")


def _relation_direction(
    previous_low: float,
    previous_high: float,
    current_low: float,
    current_high: float,
    previous_close: float,
    current_close: float,
) -> int:
    if current_high >= previous_high and current_low >= previous_low:
        return 1
    if current_high <= previous_high and current_low <= previous_low:
        return -1
    if current_close > previous_close:
        return 1
    if current_close < previous_close:
        return -1
    return 1 if current_high - previous_high >= previous_low - current_low else -1


def _merge_working_bar(
    current: _WorkingNormalizedBar,
    *,
    index: int,
    timestamp: str,
    high: float,
    low: float,
    close: float,
    volume: float,
    amount: float,
    direction: int,
) -> _WorkingNormalizedBar:
    if direction > 0:
        merged_high = max(current.high, high)
        merged_low = max(current.low, low)
        high_index = current.high_index if current.high >= high else index
        low_index = current.low_index if current.low >= low else index
    else:
        merged_high = min(current.high, high)
        merged_low = min(current.low, low)
        high_index = current.high_index if current.high <= high else index
        low_index = current.low_index if current.low <= low else index
    return replace(
        current,
        end_index=index,
        high_index=high_index,
        low_index=low_index,
        end_time=timestamp,
        high=float(merged_high),
        low=float(merged_low),
        close=float(close),
        volume=float(current.volume + volume),
        amount=float(current.amount + amount),
        direction=int(direction),
    )


def _normalize_bars(
    bars: pd.DataFrame, *, inclusion_mode: str, initial_direction_mode: str
) -> tuple[list[NormalizedBar], Mapping[str, Any] | None]:
    locked: list[NormalizedBar] = []
    current: _WorkingNormalizedBar | None = None
    lookahead_direction = 0
    if initial_direction_mode not in {
        "previous_noninclusive_then_close",
        "first_noninclusive_lookahead",
    }:
        raise ValueError(
            f"strict_chan_initial_direction_unknown:{initial_direction_mode}"
        )
    if initial_direction_mode == "first_noninclusive_lookahead" and len(bars) >= 2:
        for position in range(1, len(bars)):
            previous = bars.iloc[position - 1]
            following = bars.iloc[position]
            if not _contains(
                float(previous.low),
                float(previous.high),
                float(following.low),
                float(following.high),
                mode=inclusion_mode,
            ):
                lookahead_direction = _relation_direction(
                    float(previous.low),
                    float(previous.high),
                    float(following.low),
                    float(following.high),
                    float(previous.close),
                    float(following.close),
                )
                break
    for index, row in enumerate(bars.itertuples(index=False)):
        timestamp = _iso(row.timestamp)
        if current is None:
            current = _WorkingNormalizedBar(
                start_index=index,
                end_index=index,
                high_index=index,
                low_index=index,
                start_time=timestamp,
                end_time=timestamp,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
                amount=float(row.amount),
                direction=0,
            )
            continue
        if _contains(
            current.low,
            current.high,
            float(row.low),
            float(row.high),
            mode=inclusion_mode,
        ):
            if locked:
                previous = locked[-1]
                direction = _relation_direction(
                    previous.low,
                    previous.high,
                    current.low,
                    current.high,
                    previous.close,
                    current.close,
                )
            elif lookahead_direction:
                direction = lookahead_direction
            else:
                direction = _relation_direction(
                    current.low,
                    current.high,
                    float(row.low),
                    float(row.high),
                    current.close,
                    float(row.close),
                )
            current = _merge_working_bar(
                current,
                index=index,
                timestamp=timestamp,
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
                amount=float(row.amount),
                direction=direction,
            )
            continue
        if locked:
            previous = locked[-1]
            direction = _relation_direction(
                previous.low,
                previous.high,
                current.low,
                current.high,
                previous.close,
                current.close,
            )
        elif lookahead_direction:
            direction = lookahead_direction
        else:
            direction = _relation_direction(
                current.low,
                current.high,
                float(row.low),
                float(row.high),
                current.close,
                float(row.close),
            )
        position = len(locked)
        locked.append(
            NormalizedBar(
                id=f"normalized:{current.start_index}:{current.end_index}",
                position=position,
                start_index=current.start_index,
                end_index=current.end_index,
                high_index=current.high_index,
                low_index=current.low_index,
                start_time=current.start_time,
                end_time=current.end_time,
                confirmed_index=index,
                confirmed_time=timestamp,
                open=current.open,
                high=current.high,
                low=current.low,
                close=current.close,
                volume=current.volume,
                amount=current.amount,
                direction=direction,
            )
        )
        current = _WorkingNormalizedBar(
            start_index=index,
            end_index=index,
            high_index=index,
            low_index=index,
            start_time=timestamp,
            end_time=timestamp,
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=float(row.volume),
            amount=float(row.amount),
            direction=direction,
        )
    provisional = asdict(current) if current is not None else None
    return locked, provisional


def _fractal_kind(left: Any, middle: Any, right: Any, *, equality_mode: str) -> int:
    if equality_mode == "strict":
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
    elif equality_mode == "outer_bar_tiebreak":
        top = (
            middle.high >= max(left.high, right.high)
            and middle.low >= max(left.low, right.low)
            and (
                middle.high > min(left.high, right.high)
                or middle.low > min(left.low, right.low)
            )
        )
        bottom = (
            middle.high <= min(left.high, right.high)
            and middle.low <= min(left.low, right.low)
            and (
                middle.high < max(left.high, right.high)
                or middle.low < max(left.low, right.low)
            )
        )
    else:
        raise ValueError(f"strict_chan_fractal_equality_unknown:{equality_mode}")
    if top and not bottom:
        return 1
    if bottom and not top:
        return -1
    return 0


def _build_fractals(
    normalized: Sequence[NormalizedBar],
    bars: pd.DataFrame,
    *,
    equality_mode: str,
) -> list[Fractal]:
    fractals: list[Fractal] = []
    for middle_position in range(1, len(normalized) - 1):
        left = normalized[middle_position - 1]
        middle = normalized[middle_position]
        right = normalized[middle_position + 1]
        kind = _fractal_kind(left, middle, right, equality_mode=equality_mode)
        if kind == 0:
            continue
        extreme_index = middle.high_index if kind > 0 else middle.low_index
        extreme_price = middle.high if kind > 0 else middle.low
        fractals.append(
            Fractal(
                id=f"fractal:{'top' if kind > 0 else 'bottom'}:{middle.id}",
                kind=kind,
                kind_name="top" if kind > 0 else "bottom",
                left_bar_id=left.id,
                middle_bar_id=middle.id,
                right_bar_id=right.id,
                middle_position=middle_position,
                range_low=middle.low,
                range_high=middle.high,
                extreme_index=extreme_index,
                extreme_time=_iso(bars.iloc[extreme_index]["timestamp"]),
                extreme_price=float(extreme_price),
                confirmed_index=right.confirmed_index,
                confirmed_time=right.confirmed_time,
            )
        )
    return fractals


def _more_extreme(candidate: Fractal, reference: Fractal) -> bool:
    return (
        candidate.extreme_price > reference.extreme_price
        if candidate.kind > 0
        else candidate.extreme_price < reference.extreme_price
    )


def _stroke_valid(start: Fractal, end: Fractal, *, rule: str) -> bool:
    if start.kind == end.kind:
        return False
    if end.middle_position - start.middle_position < 3:
        return False
    direction = 1 if start.kind < 0 and end.kind > 0 else -1
    if direction > 0 and end.extreme_price <= start.extreme_price:
        return False
    if direction < 0 and end.extreme_price >= start.extreme_price:
        return False
    if rule == "disjoint_fractals_and_price_ranges":
        if direction > 0 and not start.range_high < end.range_low:
            return False
        if direction < 0 and not end.range_high < start.range_low:
            return False
    elif rule != "disjoint_fractals_only":
        raise ValueError(f"strict_chan_stroke_rule_unknown:{rule}")
    return True


def _working_stroke(start: Fractal, end: Fractal) -> _WorkingStroke:
    return _WorkingStroke(
        direction=1 if start.kind < 0 and end.kind > 0 else -1,
        start=start,
        end=end,
    )


def _lock_stroke(stroke: _WorkingStroke, *, locked_by: Fractal) -> Stroke:
    return Stroke(
        id=f"stroke:{stroke.start.id}:{stroke.end.id}",
        direction=stroke.direction,
        start_fractal_id=stroke.start.id,
        end_fractal_id=stroke.end.id,
        start_normalized_position=stroke.start.middle_position,
        end_normalized_position=stroke.end.middle_position,
        start_index=stroke.start.extreme_index,
        end_index=stroke.end.extreme_index,
        start_time=stroke.start.extreme_time,
        end_time=stroke.end.extreme_time,
        start_price=stroke.start.extreme_price,
        end_price=stroke.end.extreme_price,
        low=min(stroke.start.extreme_price, stroke.end.extreme_price),
        high=max(stroke.start.extreme_price, stroke.end.extreme_price),
        accepted_index=stroke.end.confirmed_index,
        accepted_time=stroke.end.confirmed_time,
        locked_index=locked_by.confirmed_index,
        locked_time=locked_by.confirmed_time,
        normalized_bar_span=(
            stroke.end.middle_position - stroke.start.middle_position + 1
        ),
        raw_bar_span=abs(stroke.end.extreme_index - stroke.start.extreme_index) + 1,
    )


def _build_strokes(
    fractals: Sequence[Fractal], *, rule: str
) -> tuple[list[Stroke], Mapping[str, Any] | None]:
    anchor: Fractal | None = None
    current: _WorkingStroke | None = None
    locked: list[Stroke] = []
    for fractal in sorted(fractals, key=lambda item: item.confirmed_index):
        if anchor is None:
            anchor = fractal
            continue
        if fractal.kind == anchor.kind:
            if _more_extreme(fractal, anchor):
                anchor = fractal
                if current is not None:
                    current = _working_stroke(current.start, fractal)
            continue
        if not _stroke_valid(anchor, fractal, rule=rule):
            continue
        next_stroke = _working_stroke(anchor, fractal)
        if current is not None:
            locked.append(_lock_stroke(current, locked_by=fractal))
        current = next_stroke
        anchor = fractal
    provisional = None
    if current is not None:
        provisional = {
            "id": f"stroke:{current.start.id}:{current.end.id}",
            "direction": current.direction,
            "start_fractal_id": current.start.id,
            "end_fractal_id": current.end.id,
            "accepted_index": current.end.confirmed_index,
            "accepted_time": current.end.confirmed_time,
            "locked": False,
        }
    return locked, provisional


def _overlap(low_a: float, high_a: float, low_b: float, high_b: float) -> bool:
    return max(low_a, low_b) <= min(high_a, high_b)


def _feature_input_positions(
    strokes: Sequence[Stroke], start: int, direction: int
) -> list[int]:
    return [
        position
        for position in range(start + 1, len(strokes))
        if strokes[position].direction == -direction
    ]


def _merge_feature(
    current: _WorkingFeatureElement,
    stroke: Stroke,
    position: int,
    *,
    direction: int,
) -> _WorkingFeatureElement:
    if direction > 0:
        high = max(current.high, stroke.high)
        low = max(current.low, stroke.low)
        high_pos = (
            current.high_stroke_position if current.high >= stroke.high else position
        )
        low_pos = current.low_stroke_position if current.low >= stroke.low else position
    else:
        high = min(current.high, stroke.high)
        low = min(current.low, stroke.low)
        high_pos = (
            current.high_stroke_position if current.high <= stroke.high else position
        )
        low_pos = current.low_stroke_position if current.low <= stroke.low else position
    return _WorkingFeatureElement(
        low=float(low),
        high=float(high),
        low_stroke_position=low_pos,
        high_stroke_position=high_pos,
        source_stroke_positions=(*current.source_stroke_positions, position),
        direction=direction,
    )


def _iter_standard_feature_sequence(
    strokes: Sequence[Stroke], *, start: int, direction: int, inclusion_mode: str
) -> Iterator[_FeatureElement]:
    locked: list[_FeatureElement] = []
    current: _WorkingFeatureElement | None = None
    for stroke_position in range(start + 1, len(strokes)):
        stroke = strokes[stroke_position]
        if stroke.direction != -direction:
            continue
        if current is None:
            current = _WorkingFeatureElement(
                low=stroke.low,
                high=stroke.high,
                low_stroke_position=stroke_position,
                high_stroke_position=stroke_position,
                source_stroke_positions=(stroke_position,),
                direction=0,
            )
            continue
        if _contains(
            current.low,
            current.high,
            stroke.low,
            stroke.high,
            mode=inclusion_mode,
        ):
            if locked:
                previous = locked[-1]
                merge_direction = _relation_direction(
                    previous.low,
                    previous.high,
                    current.low,
                    current.high,
                    (previous.low + previous.high) / 2.0,
                    (current.low + current.high) / 2.0,
                )
            else:
                merge_direction = _relation_direction(
                    current.low,
                    current.high,
                    stroke.low,
                    stroke.high,
                    (current.low + current.high) / 2.0,
                    (stroke.low + stroke.high) / 2.0,
                )
            current = _merge_feature(
                current,
                stroke,
                stroke_position,
                direction=merge_direction,
            )
            continue
        if locked:
            previous = locked[-1]
            current_direction = _relation_direction(
                previous.low,
                previous.high,
                current.low,
                current.high,
                (previous.low + previous.high) / 2.0,
                (current.low + current.high) / 2.0,
            )
        else:
            current_direction = _relation_direction(
                current.low,
                current.high,
                stroke.low,
                stroke.high,
                (current.low + current.high) / 2.0,
                (stroke.low + stroke.high) / 2.0,
            )
        position = len(locked)
        feature = _FeatureElement(
            id=(
                f"feature:{start}:{direction}:"
                + ",".join(map(str, current.source_stroke_positions))
            ),
            position=position,
            low=current.low,
            high=current.high,
            low_stroke_position=current.low_stroke_position,
            high_stroke_position=current.high_stroke_position,
            source_stroke_positions=current.source_stroke_positions,
            confirmed_index=stroke.locked_index,
            confirmed_time=stroke.locked_time,
            direction=current_direction,
        )
        locked.append(feature)
        yield feature
        current = _WorkingFeatureElement(
            low=stroke.low,
            high=stroke.high,
            low_stroke_position=stroke_position,
            high_stroke_position=stroke_position,
            source_stroke_positions=(stroke_position,),
            direction=current_direction,
        )


def _standard_feature_sequence(
    strokes: Sequence[Stroke], *, start: int, direction: int, inclusion_mode: str
) -> list[_FeatureElement]:
    return list(
        _iter_standard_feature_sequence(
            strokes,
            start=start,
            direction=direction,
            inclusion_mode=inclusion_mode,
        )
    )


def _iter_feature_fractal_candidates(
    strokes: Sequence[Stroke],
    *,
    start: int,
    direction: int,
    inclusion_mode: str,
    equality_mode: str,
) -> Iterator[_FeatureFractalCandidate]:
    features: list[_FeatureElement] = []
    for feature in _iter_standard_feature_sequence(
        strokes,
        start=start,
        direction=direction,
        inclusion_mode=inclusion_mode,
    ):
        features.append(feature)
        if len(features) < 3:
            continue
        left, middle, right = features[-3:]
        kind = _fractal_kind(left, middle, right, equality_mode=equality_mode)
        expected_kind = 1 if direction > 0 else -1
        if kind != expected_kind:
            continue
        endpoint_stroke_position = (
            middle.high_stroke_position if direction > 0 else middle.low_stroke_position
        )
        if endpoint_stroke_position <= start:
            continue
        endpoint_stroke = strokes[endpoint_stroke_position]
        has_gap = not _overlap(left.low, left.high, middle.low, middle.high)
        yield _FeatureFractalCandidate(
            id=(
                f"feature_fractal:{start}:{direction}:{left.id}:{middle.id}:{right.id}"
            ),
            direction=direction,
            endpoint_stroke_position=endpoint_stroke_position,
            endpoint_price=float(endpoint_stroke.start_price),
            event_index=endpoint_stroke.start_index,
            event_time=endpoint_stroke.start_time,
            detected_index=right.confirmed_index,
            detected_time=right.confirmed_time,
            break_case=2 if has_gap else 1,
        )


def _feature_fractal_candidates(
    strokes: Sequence[Stroke],
    *,
    start: int,
    direction: int,
    inclusion_mode: str,
    equality_mode: str,
) -> list[_FeatureFractalCandidate]:
    return list(
        _iter_feature_fractal_candidates(
            strokes,
            start=start,
            direction=direction,
            inclusion_mode=inclusion_mode,
            equality_mode=equality_mode,
        )
    )


def _three_stroke_overlap(strokes: Sequence[Stroke], start: int) -> bool:
    if start + 2 >= len(strokes):
        return False
    group = strokes[start : start + 3]
    return max(item.low for item in group) <= min(item.high for item in group)


def _candidate_invalidation(
    strokes: Sequence[Stroke], candidate: _FeatureFractalCandidate
) -> tuple[int, str] | None:
    for stroke in strokes[candidate.endpoint_stroke_position + 1 :]:
        if stroke.direction != candidate.direction:
            continue
        breaks = (
            stroke.end_price > candidate.endpoint_price
            if candidate.direction > 0
            else stroke.end_price < candidate.endpoint_price
        )
        if breaks:
            if stroke.locked_index < candidate.detected_index:
                return candidate.detected_index, candidate.detected_time
            return stroke.locked_index, stroke.locked_time
    return None


def _latest_confirmation(
    *values: tuple[int, str] | None,
) -> tuple[int, str]:
    available = [value for value in values if value is not None]
    if not available:
        raise ValueError("strict_chan_confirmation_floor_empty")
    return max(available, key=lambda value: (value[0], value[1]))


def _macd_histogram(close: np.ndarray) -> np.ndarray:
    series = pd.Series(close, dtype="float64")
    fast = series.ewm(span=12, adjust=False, min_periods=1).mean()
    slow = series.ewm(span=26, adjust=False, min_periods=1).mean()
    difference = fast - slow
    signal = difference.ewm(span=9, adjust=False, min_periods=1).mean()
    return (difference - signal).to_numpy(dtype=np.float64)


def _make_segment(
    strokes: Sequence[Stroke],
    *,
    start: int,
    candidate: _FeatureFractalCandidate,
    confirmed_index: int,
    confirmed_time: str,
    bars: pd.DataFrame,
    macd_histogram: np.ndarray,
) -> Segment:
    end = candidate.endpoint_stroke_position - 1
    selected = tuple(strokes[start : end + 1])
    start_index = selected[0].start_index
    end_index = selected[-1].end_index
    lo = min(start_index, end_index)
    hi = max(start_index, end_index)
    amount_values = bars.iloc[lo : hi + 1]["amount"].to_numpy(dtype=np.float64)
    histogram = macd_histogram[lo : hi + 1]
    if candidate.direction > 0:
        macd_area = float(np.clip(histogram, 0.0, None).sum())
    else:
        macd_area = float(np.clip(-histogram, 0.0, None).sum())
    if macd_area <= 0:
        macd_area = float(np.abs(histogram).sum())
    start_price = selected[0].start_price
    end_price = selected[-1].end_price
    price_change = float(math.log(end_price / start_price))
    raw_span = hi - lo + 1
    return Segment(
        id=(f"segment:{selected[0].id}:{selected[-1].id}:case{candidate.break_case}"),
        direction=candidate.direction,
        start_stroke_position=start,
        end_stroke_position=end,
        stroke_ids=tuple(item.id for item in selected),
        start_index=start_index,
        end_index=end_index,
        start_time=selected[0].start_time,
        end_time=selected[-1].end_time,
        start_price=start_price,
        end_price=end_price,
        low=min(item.low for item in selected),
        high=max(item.high for item in selected),
        event_index=candidate.event_index,
        event_time=candidate.event_time,
        confirmed_index=confirmed_index,
        confirmed_time=confirmed_time,
        break_case=candidate.break_case,
        feature_fractal_id=candidate.id,
        raw_bar_span=raw_span,
        price_change=price_change,
        duration_normalized_price_change=abs(price_change) / max(raw_span, 1),
        macd_histogram_area=macd_area,
        amount_per_bar=float(amount_values.mean()) if len(amount_values) else 0.0,
    )


def build_segments(
    strokes: Sequence[Stroke],
    bars: pd.DataFrame,
    *,
    inclusion_mode: str = "closed_interval",
    equality_mode: str = "strict",
) -> tuple[list[Segment], list[SegmentStateEvent]]:
    if not strokes:
        return [], []
    close = bars["close"].to_numpy(dtype=np.float64)
    histogram = _macd_histogram(close)
    segments: list[Segment] = []
    state_events: list[SegmentStateEvent] = []
    start = 0
    while start + 2 < len(strokes) and not _three_stroke_overlap(strokes, start):
        start += 1
    causal_floor: tuple[int, str] | None = None
    while start + 2 < len(strokes):
        if not _three_stroke_overlap(strokes, start):
            break
        direction = strokes[start].direction
        candidates = _iter_feature_fractal_candidates(
            strokes,
            start=start,
            direction=direction,
            inclusion_mode=inclusion_mode,
            equality_mode=equality_mode,
        )
        accepted: tuple[_FeatureFractalCandidate, int, str] | None = None
        decision_floor = causal_floor
        for candidate in candidates:
            if candidate.endpoint_stroke_position - 1 < start + 2:
                continue
            detected = (candidate.detected_index, candidate.detected_time)
            if candidate.break_case == 1:
                confirmed = _latest_confirmation(detected, decision_floor)
                accepted = (
                    candidate,
                    confirmed[0],
                    confirmed[1],
                )
                break
            candidate_id = f"segment_break:{start}:{candidate.id}"
            opened_at = _latest_confirmation(detected, decision_floor)
            open_event = SegmentStateEvent(
                id=f"{candidate_id}:opened",
                candidate_id=candidate_id,
                action="opened",
                direction=direction,
                segment_start_stroke_position=start,
                endpoint_stroke_position=candidate.endpoint_stroke_position,
                middle_yin_stroke_id=strokes[candidate.endpoint_stroke_position].id,
                break_case=2,
                event_index=candidate.event_index,
                event_time=candidate.event_time,
                confirmed_index=opened_at[0],
                confirmed_time=opened_at[1],
            )
            confirmation = next(
                (
                    item
                    for item in _iter_feature_fractal_candidates(
                        strokes,
                        start=candidate.endpoint_stroke_position,
                        direction=-direction,
                        inclusion_mode=inclusion_mode,
                        equality_mode=equality_mode,
                    )
                    if item.detected_index >= candidate.detected_index
                ),
                None,
            )
            confirmation_at = (
                _latest_confirmation(
                    (confirmation.detected_index, confirmation.detected_time),
                    opened_at,
                )
                if confirmation is not None
                else None
            )
            invalidation = _candidate_invalidation(strokes, candidate)
            if invalidation is not None:
                invalidation = _latest_confirmation(invalidation, opened_at)
            if invalidation is not None and (
                confirmation_at is None or invalidation[0] <= confirmation_at[0]
            ):
                state_events.extend(
                    [
                        open_event,
                        SegmentStateEvent(
                            id=f"{candidate_id}:invalidated",
                            candidate_id=candidate_id,
                            action="invalidated",
                            direction=direction,
                            segment_start_stroke_position=start,
                            endpoint_stroke_position=(
                                candidate.endpoint_stroke_position
                            ),
                            middle_yin_stroke_id=strokes[
                                candidate.endpoint_stroke_position
                            ].id,
                            break_case=2,
                            event_index=candidate.event_index,
                            event_time=candidate.event_time,
                            confirmed_index=invalidation[0],
                            confirmed_time=invalidation[1],
                        ),
                    ]
                )
                decision_floor = invalidation
                continue
            state_events.append(open_event)
            if confirmation_at is None:
                break
            state_events.append(
                SegmentStateEvent(
                    id=f"{candidate_id}:confirmed",
                    candidate_id=candidate_id,
                    action="confirmed",
                    direction=direction,
                    segment_start_stroke_position=start,
                    endpoint_stroke_position=candidate.endpoint_stroke_position,
                    middle_yin_stroke_id=strokes[candidate.endpoint_stroke_position].id,
                    break_case=2,
                    event_index=candidate.event_index,
                    event_time=candidate.event_time,
                    confirmed_index=confirmation_at[0],
                    confirmed_time=confirmation_at[1],
                )
            )
            accepted = (
                candidate,
                confirmation_at[0],
                confirmation_at[1],
            )
            break
        if accepted is None:
            break
        candidate, confirmed_index, confirmed_time = accepted
        segment = _make_segment(
            strokes,
            start=start,
            candidate=candidate,
            confirmed_index=confirmed_index,
            confirmed_time=confirmed_time,
            bars=bars,
            macd_histogram=histogram,
        )
        segments.append(segment)
        next_start = candidate.endpoint_stroke_position
        if next_start <= start:
            raise ValueError("strict_chan_segment_parser_no_progress")
        start = next_start
        causal_floor = (confirmed_index, confirmed_time)
    return segments, state_events


def _components_from_segments(segments: Sequence[Segment]) -> list[_Component]:
    return [
        _Component(
            id=item.id,
            position=position,
            direction=item.direction,
            low=item.low,
            high=item.high,
            start_index=item.start_index,
            end_index=item.end_index,
            start_time=item.start_time,
            end_time=item.end_time,
            confirmed_index=item.confirmed_index,
            confirmed_time=item.confirmed_time,
        )
        for position, item in enumerate(segments)
    ]


def _components_from_strokes(strokes: Sequence[Stroke]) -> list[_Component]:
    return [
        _Component(
            id=item.id,
            position=position,
            direction=item.direction,
            low=item.low,
            high=item.high,
            start_index=item.start_index,
            end_index=item.end_index,
            start_time=item.start_time,
            end_time=item.end_time,
            confirmed_index=item.locked_index,
            confirmed_time=item.locked_time,
        )
        for position, item in enumerate(strokes)
    ]


def _center_relation(
    previous: Center,
    current: Center,
    *,
    mode: str,
    previous_extensions: Sequence[CenterExtension],
) -> tuple[str, int]:
    if mode == "core_interval_overlap":
        previous_low, previous_high = previous.core_low, previous.core_high
        current_low, current_high = current.core_low, current.core_high
    elif mode == "fluctuation_interval_overlap":
        previous_low = min(
            [previous.fluctuation_low]
            + [item.component_low for item in previous_extensions]
        )
        previous_high = max(
            [previous.fluctuation_high]
            + [item.component_high for item in previous_extensions]
        )
        current_low, current_high = (
            current.fluctuation_low,
            current.fluctuation_high,
        )
    else:
        raise ValueError(f"strict_chan_center_relation_unknown:{mode}")
    if current_low > previous_high:
        return "new_birth_up", 1
    if current_high < previous_low:
        return "new_birth_down", -1
    return "expansion", 0


def _build_center_level(
    components: Sequence[_Component],
    *,
    level: int,
    component_kind: str,
    construction_variant: str,
    relation_mode: str,
) -> tuple[list[Center], list[CenterExtension], list[CenterRelation]]:
    centers: list[Center] = []
    extensions: list[CenterExtension] = []
    position = 0
    while position + 2 < len(components):
        triple = components[position : position + 3]
        core_low = max(item.low for item in triple)
        core_high = min(item.high for item in triple)
        if core_low > core_high:
            position += 1
            continue
        center = Center(
            id=(
                f"center:{level}:{component_kind}:"
                + ":".join(item.id for item in triple)
            ),
            level=level,
            component_kind=component_kind,
            component_ids=(triple[0].id, triple[1].id, triple[2].id),
            start_component_position=position,
            end_component_position=position + 2,
            core_low=float(core_low),
            core_high=float(core_high),
            fluctuation_low=min(item.low for item in triple),
            fluctuation_high=max(item.high for item in triple),
            event_index=triple[0].start_index,
            event_time=triple[0].start_time,
            confirmed_index=triple[2].confirmed_index,
            confirmed_time=triple[2].confirmed_time,
            construction_variant=construction_variant,
        )
        centers.append(center)
        next_position = position + 3
        while next_position < len(components) and _overlap(
            center.core_low,
            center.core_high,
            components[next_position].low,
            components[next_position].high,
        ):
            component = components[next_position]
            extensions.append(
                CenterExtension(
                    id=f"center_extension:{center.id}:{component.id}",
                    center_id=center.id,
                    level=level,
                    component_id=component.id,
                    component_position=next_position,
                    component_low=component.low,
                    component_high=component.high,
                    confirmed_index=component.confirmed_index,
                    confirmed_time=component.confirmed_time,
                )
            )
            next_position += 1
        position = next_position
    relations: list[CenterRelation] = []
    extensions_by_center: dict[str, list[CenterExtension]] = {}
    for item in extensions:
        extensions_by_center.setdefault(item.center_id, []).append(item)
    for previous, current in pairwise(centers):
        relation, direction = _center_relation(
            previous,
            current,
            mode=relation_mode,
            previous_extensions=extensions_by_center.get(previous.id, []),
        )
        relations.append(
            CenterRelation(
                id=f"center_relation:{previous.id}:{current.id}:{relation}",
                level=level,
                previous_center_id=previous.id,
                current_center_id=current.id,
                relation=relation,
                direction=direction,
                comparison_variant=relation_mode,
                confirmed_index=current.confirmed_index,
                confirmed_time=current.confirmed_time,
            )
        )
    return centers, extensions, relations


def _build_movements(
    centers: Sequence[Center],
    components: Sequence[_Component],
    *,
    level: int,
) -> list[Movement]:
    movements: list[Movement] = []
    for previous, current in pairwise(centers):
        start = previous.start_component_position
        end = max(current.start_component_position - 1, start)
        selected = tuple(components[start : end + 1])
        if not selected:
            continue
        if current.core_low > previous.core_high:
            direction = 1
        elif current.core_high < previous.core_low:
            direction = -1
        else:
            direction = 0
        movements.append(
            Movement(
                id=f"movement:{level}:{previous.id}:{current.id}",
                level=level,
                direction=direction,
                start_center_id=previous.id,
                end_center_id=current.id,
                component_ids=tuple(item.id for item in selected),
                start_index=selected[0].start_index,
                end_index=selected[-1].end_index,
                start_time=selected[0].start_time,
                end_time=selected[-1].end_time,
                low=min(item.low for item in selected),
                high=max(item.high for item in selected),
                event_index=selected[0].start_index,
                event_time=selected[0].start_time,
                confirmed_index=current.confirmed_index,
                confirmed_time=current.confirmed_time,
                decomposition_variant="center_first_same_level",
            )
        )
    return movements


def _components_from_movements(
    movements: Sequence[Movement],
) -> list[_Component]:
    return [
        _Component(
            id=item.id,
            position=position,
            direction=item.direction,
            low=item.low,
            high=item.high,
            start_index=item.start_index,
            end_index=item.end_index,
            start_time=item.start_time,
            end_time=item.end_time,
            confirmed_index=item.confirmed_index,
            confirmed_time=item.confirmed_time,
        )
        for position, item in enumerate(movements)
    ]


def build_recursive_centers(
    segments: Sequence[Segment],
    strokes: Sequence[Stroke],
    *,
    base_component: str,
    relation_mode: str,
    maximum_levels: int = 5,
) -> tuple[list[Center], list[CenterExtension], list[CenterRelation], list[Movement]]:
    if base_component == "segment":
        components = _components_from_segments(segments)
        component_kind = "segment"
    elif base_component == "stroke_diagnostic":
        components = _components_from_strokes(strokes)
        component_kind = "stroke"
    else:
        raise ValueError(f"strict_chan_base_component_unknown:{base_component}")
    all_centers: list[Center] = []
    all_extensions: list[CenterExtension] = []
    all_relations: list[CenterRelation] = []
    all_movements: list[Movement] = []
    level = 1
    while level <= maximum_levels and len(components) >= 3:
        centers, extensions, relations = _build_center_level(
            components,
            level=level,
            component_kind=component_kind,
            construction_variant=base_component,
            relation_mode=relation_mode,
        )
        if not centers:
            break
        movements = _build_movements(centers, components, level=level)
        all_centers.extend(centers)
        all_extensions.extend(extensions)
        all_relations.extend(relations)
        all_movements.extend(movements)
        if len(movements) < 3:
            break
        components = _components_from_movements(movements)
        component_kind = "movement"
        level += 1
    return all_centers, all_extensions, all_relations, all_movements


def _trend_types(
    centers: Sequence[Center], relations: Sequence[CenterRelation]
) -> list[TrendType]:
    centers_by_level: dict[int, list[Center]] = {}
    relations_by_level: dict[int, list[CenterRelation]] = {}
    for center in centers:
        centers_by_level.setdefault(center.level, []).append(center)
    for relation in relations:
        relations_by_level.setdefault(relation.level, []).append(relation)
    output: list[TrendType] = []
    for level, level_centers in centers_by_level.items():
        center_map = {item.id: item for item in level_centers}
        for center in level_centers:
            output.append(
                TrendType(
                    id=f"trend_type:{level}:consolidation:{center.id}",
                    level=level,
                    kind="consolidation",
                    direction=0,
                    center_ids=(center.id,),
                    event_index=center.event_index,
                    event_time=center.event_time,
                    confirmed_index=center.confirmed_index,
                    confirmed_time=center.confirmed_time,
                )
            )
        run: list[str] = []
        run_direction = 0
        for relation in relations_by_level.get(level, []):
            if relation.direction == 0:
                run = []
                run_direction = 0
                continue
            if run_direction == relation.direction and run:
                if run[-1] != relation.previous_center_id:
                    run = [relation.previous_center_id, relation.current_center_id]
                else:
                    run.append(relation.current_center_id)
            else:
                run = [relation.previous_center_id, relation.current_center_id]
                run_direction = relation.direction
            current = center_map[relation.current_center_id]
            output.append(
                TrendType(
                    id=(f"trend_type:{level}:trend:{run_direction}:" + ":".join(run)),
                    level=level,
                    kind="trend",
                    direction=run_direction,
                    center_ids=tuple(run),
                    event_index=center_map[run[0]].event_index,
                    event_time=center_map[run[0]].event_time,
                    confirmed_index=current.confirmed_index,
                    confirmed_time=current.confirmed_time,
                )
            )
    return output


def _transition_segment(
    segments: Sequence[Segment],
    left: Center,
    right: Center,
    direction: int,
) -> Segment | None:
    candidates = [
        segment
        for position, segment in enumerate(segments)
        if left.end_component_position < position <= right.start_component_position
        and segment.direction == direction
    ]
    return candidates[-1] if candidates else None


def _divergences(
    trends: Sequence[TrendType],
    centers: Sequence[Center],
    segments: Sequence[Segment],
    metrics: Sequence[str],
) -> list[Divergence]:
    center_map = {item.id: item for item in centers}
    output: list[Divergence] = []
    for trend in trends:
        if trend.kind != "trend" or trend.level != 1 or len(trend.center_ids) < 3:
            continue
        first, second, third = [center_map[item] for item in trend.center_ids[-3:]]
        if any(item.component_kind != "segment" for item in (first, second, third)):
            continue
        previous = _transition_segment(segments, first, second, trend.direction)
        current = _transition_segment(segments, second, third, trend.direction)
        if previous is None or current is None:
            continue
        confirmed_at = _latest_confirmation(
            (trend.confirmed_index, trend.confirmed_time),
            (previous.confirmed_index, previous.confirmed_time),
            (current.confirmed_index, current.confirmed_time),
        )
        for metric in metrics:
            previous_strength = float(getattr(previous, metric))
            current_strength = float(getattr(current, metric))
            if not (
                np.isfinite(previous_strength)
                and np.isfinite(current_strength)
                and previous_strength > 0
                and current_strength < previous_strength
            ):
                continue
            output.append(
                Divergence(
                    id=f"divergence:{trend.id}:{metric}:{current.id}",
                    level=trend.level,
                    direction=trend.direction,
                    metric=metric,
                    trend_id=trend.id,
                    previous_segment_id=previous.id,
                    current_segment_id=current.id,
                    previous_strength=previous_strength,
                    current_strength=current_strength,
                    strength_ratio=current_strength / previous_strength,
                    event_index=current.end_index,
                    event_time=current.end_time,
                    confirmed_index=confirmed_at[0],
                    confirmed_time=confirmed_at[1],
                )
            )
    unique = {item.id: item for item in output}
    return sorted(unique.values(), key=lambda item: item.confirmed_index)


def _trade_points(
    divergences: Sequence[Divergence],
    centers: Sequence[Center],
    extensions: Sequence[CenterExtension],
    segments: Sequence[Segment],
) -> list[TradePoint]:
    segment_map = {item.id: item for item in segments}
    segment_position = {item.id: position for position, item in enumerate(segments)}
    output: list[TradePoint] = []
    first_points: list[TradePoint] = []
    for divergence in divergences:
        segment = segment_map[divergence.current_segment_id]
        side = "sell" if divergence.direction > 0 else "buy"
        point = TradePoint(
            id=f"point:1:{side}:{divergence.id}",
            point_type=1,
            side=side,
            level=divergence.level,
            variant=divergence.metric,
            basis_id=divergence.id,
            price=segment.end_price,
            event_index=segment.end_index,
            event_time=segment.end_time,
            confirmed_index=divergence.confirmed_index,
            confirmed_time=divergence.confirmed_time,
        )
        first_points.append(point)
        output.append(point)
    for point in first_points:
        basis = next(item for item in divergences if item.id == point.basis_id)
        basis_position = segment_position[basis.current_segment_id]
        if basis_position + 2 >= len(segments):
            continue
        retest = segments[basis_position + 2]
        valid = (
            retest.direction < 0 and retest.end_price > point.price
            if point.side == "buy"
            else retest.direction > 0 and retest.end_price < point.price
        )
        if valid:
            confirmed_at = _latest_confirmation(
                (point.confirmed_index, point.confirmed_time),
                (retest.confirmed_index, retest.confirmed_time),
            )
            output.append(
                TradePoint(
                    id=f"point:2:{point.side}:{point.id}:{retest.id}",
                    point_type=2,
                    side=point.side,
                    level=point.level,
                    variant=point.variant,
                    basis_id=point.id,
                    price=retest.end_price,
                    event_index=retest.end_index,
                    event_time=retest.end_time,
                    confirmed_index=confirmed_at[0],
                    confirmed_time=confirmed_at[1],
                )
            )
    extension_end: dict[str, int] = {
        center.id: center.end_component_position
        for center in centers
        if center.level == 1 and center.component_kind == "segment"
    }
    for extension in extensions:
        if extension.center_id in extension_end:
            extension_end[extension.center_id] = max(
                extension_end[extension.center_id], extension.component_position
            )
    for center in centers:
        if center.level != 1 or center.component_kind != "segment":
            continue
        start = extension_end.get(center.id, center.end_component_position) + 1
        for departure_position in range(start, len(segments)):
            departure = segments[departure_position]
            if departure.end_price > center.core_high:
                side = "buy"
                return_direction = -1
            elif departure.end_price < center.core_low:
                side = "sell"
                return_direction = 1
            else:
                continue
            retest = next(
                (
                    item
                    for item in segments[departure_position + 1 :]
                    if item.direction == return_direction
                ),
                None,
            )
            remains_outside = retest is not None and (
                retest.end_price > center.core_high
                if side == "buy"
                else retest.end_price < center.core_low
            )
            if retest is not None and remains_outside:
                output.append(
                    TradePoint(
                        id=f"point:3:{side}:{center.id}:{retest.id}",
                        point_type=3,
                        side=side,
                        level=center.level,
                        variant="center_core_return",
                        basis_id=center.id,
                        price=retest.end_price,
                        event_index=retest.end_index,
                        event_time=retest.end_time,
                        confirmed_index=retest.confirmed_index,
                        confirmed_time=retest.confirmed_time,
                    )
                )
            break
    unique = {item.id: item for item in output}
    return sorted(unique.values(), key=lambda item: item.confirmed_index)


def _validate_causal_dependencies(result: ChanParseResult) -> None:
    def require(
        event_type: str,
        event_id: str,
        confirmed_index: int,
        dependencies: Sequence[tuple[str, int]],
    ) -> None:
        if not dependencies:
            return
        dependency_id, dependency_index = max(dependencies, key=lambda item: item[1])
        if confirmed_index < dependency_index:
            raise ValueError(
                "strict_chan_dependency_confirmation_violation:"
                f"{event_type}:{event_id}:{confirmed_index}:"
                f"{dependency_id}:{dependency_index}"
            )

    normalized = {item.id: item.confirmed_index for item in result.normalized_bars}
    fractals = {item.id: item.confirmed_index for item in result.fractals}
    strokes = {item.id: item.locked_index for item in result.strokes}
    segments = {item.id: item.confirmed_index for item in result.segments}
    movements = {item.id: item.confirmed_index for item in result.movements}
    components = {**strokes, **segments, **movements}
    centers = {item.id: item.confirmed_index for item in result.centers}
    trends = {item.id: item.confirmed_index for item in result.trend_types}
    divergences = {item.id: item.confirmed_index for item in result.divergences}
    points = {item.id: item.confirmed_index for item in result.trade_points}

    for item in result.fractals:
        require(
            "fractal",
            item.id,
            item.confirmed_index,
            [
                (dependency, normalized[dependency])
                for dependency in (
                    item.left_bar_id,
                    item.middle_bar_id,
                    item.right_bar_id,
                )
            ],
        )
    for item in result.strokes:
        require(
            "stroke",
            item.id,
            item.locked_index,
            [
                (item.start_fractal_id, fractals[item.start_fractal_id]),
                (item.end_fractal_id, fractals[item.end_fractal_id]),
            ],
        )
    for item in result.segments:
        require(
            "segment",
            item.id,
            item.confirmed_index,
            [(dependency, strokes[dependency]) for dependency in item.stroke_ids],
        )

    states: dict[str, list[SegmentStateEvent]] = {}
    for item in result.segment_state_events:
        states.setdefault(item.candidate_id, []).append(item)
    for candidate_id, values in states.items():
        opened = next((item for item in values if item.action == "opened"), None)
        if opened is None:
            raise ValueError(f"strict_chan_segment_state_open_missing:{candidate_id}")
        require(
            "segment_state",
            candidate_id,
            min(item.confirmed_index for item in values if item.action != "opened")
            if any(item.action != "opened" for item in values)
            else opened.confirmed_index,
            [(opened.id, opened.confirmed_index)],
        )

    for item in result.centers:
        require(
            "center",
            item.id,
            item.confirmed_index,
            [(dependency, components[dependency]) for dependency in item.component_ids],
        )
    for item in result.center_extensions:
        require(
            "center_extension",
            item.id,
            item.confirmed_index,
            [
                (item.center_id, centers[item.center_id]),
                (item.component_id, components[item.component_id]),
            ],
        )
    for item in result.center_relations:
        require(
            "center_relation",
            item.id,
            item.confirmed_index,
            [
                (item.previous_center_id, centers[item.previous_center_id]),
                (item.current_center_id, centers[item.current_center_id]),
            ],
        )
    for item in result.movements:
        require(
            "movement",
            item.id,
            item.confirmed_index,
            [
                (item.start_center_id, centers[item.start_center_id]),
                (item.end_center_id, centers[item.end_center_id]),
                *[
                    (dependency, components[dependency])
                    for dependency in item.component_ids
                ],
            ],
        )
    for item in result.trend_types:
        require(
            "trend_type",
            item.id,
            item.confirmed_index,
            [(dependency, centers[dependency]) for dependency in item.center_ids],
        )
    for item in result.divergences:
        require(
            "divergence",
            item.id,
            item.confirmed_index,
            [
                (item.trend_id, trends[item.trend_id]),
                (
                    item.previous_segment_id,
                    segments[item.previous_segment_id],
                ),
                (item.current_segment_id, segments[item.current_segment_id]),
            ],
        )
    for item in result.trade_points:
        basis = divergences.get(item.basis_id)
        if basis is None:
            basis = points.get(item.basis_id)
        if basis is None:
            basis = centers.get(item.basis_id)
        if basis is None:
            raise ValueError(f"strict_chan_trade_point_basis_missing:{item.id}")
        require(
            "trade_point",
            item.id,
            item.confirmed_index,
            [(item.basis_id, basis)],
        )


def parse_strict_chan(
    frame: pd.DataFrame,
    *,
    definition_path: str | Path = DEFAULT_DEFINITION_PATH,
    variant_overrides: Mapping[str, str] | None = None,
) -> ChanParseResult:
    spec = load_definition_spec(definition_path)
    selected = _variant_selection(spec, variant_overrides)
    bars = _validate_bars(frame)
    normalized, provisional_normalized = _normalize_bars(
        bars,
        inclusion_mode=selected["inclusion_equality"],
        initial_direction_mode=selected["initial_inclusion_direction"],
    )
    fractals = _build_fractals(
        normalized,
        bars,
        equality_mode=selected["fractal_equality"],
    )
    strokes, provisional_stroke = _build_strokes(
        fractals,
        rule=selected["stroke_rule"],
    )
    segments, segment_state_events = build_segments(
        strokes,
        bars,
        inclusion_mode=selected["inclusion_equality"],
        equality_mode=selected["fractal_equality"],
    )
    centers, extensions, relations, movements = build_recursive_centers(
        segments,
        strokes,
        base_component=selected["lowest_center_component"],
        relation_mode=selected["center_relation"],
    )
    trends = _trend_types(centers, relations)
    divergence_metrics = list(
        dict(dict(spec["variants"])["divergence_metrics"])["options"]
    )
    divergences = _divergences(
        trends,
        centers,
        segments,
        divergence_metrics,
    )
    points = _trade_points(divergences, centers, extensions, segments)
    result = ChanParseResult(
        schema=PARSER_SCHEMA_VERSION,
        variant_selection=selected,
        normalized_bars=tuple(normalized),
        provisional_normalized_bar=provisional_normalized,
        fractals=tuple(fractals),
        strokes=tuple(strokes),
        provisional_stroke=provisional_stroke,
        segment_state_events=tuple(segment_state_events),
        segments=tuple(segments),
        centers=tuple(centers),
        center_extensions=tuple(extensions),
        center_relations=tuple(relations),
        movements=tuple(movements),
        trend_types=tuple(trends),
        divergences=tuple(divergences),
        trade_points=tuple(points),
    )
    _validate_causal_dependencies(result)
    return result


__all__ = [
    "DEFAULT_DEFINITION_PATH",
    "Center",
    "CenterExtension",
    "CenterRelation",
    "ChanParseResult",
    "Divergence",
    "Fractal",
    "Movement",
    "NormalizedBar",
    "Segment",
    "SegmentStateEvent",
    "Stroke",
    "TradePoint",
    "TrendType",
    "build_recursive_centers",
    "build_segments",
    "load_definition_spec",
    "parse_strict_chan",
    "primary_variants",
]
