"""Strict-Chan structure increment and same-date matched-control study.

The study is deliberately outcome blind at sample selection.  It treats named
Chan points as path annotations, not profitable trades, and asks two narrower
questions: whether rich confirmed structure improves chronological path
prediction beyond existing transparent coordinates, and whether annotated
states differ from same-date risk/activity-matched non-pattern states.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from daily_research.path_policy import (
    seq100_strict_chan_coordinate_screen as coordinate,
)
from daily_research.path_policy import seq100_strict_chan_intraday as intraday
from daily_research.path_policy import seq100_strict_chan_outcome_probe as outcome
from daily_research.path_policy import seq100_strict_chan_parser as chan
from daily_research.path_policy import seq100_strict_chan_stratified_audit as stratified

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT
    / "daily_research/studies/seq100_strict_chan_structure_increment_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT / "daily_research/output/path_policy/studies/"
    "seq100_strict_chan_structure_increment_v1"
)
STUDY_ID = "seq100_strict_chan_structure_increment_v1"
SAMPLE_STUDY_ID = "seq100_strict_chan_structure_increment_sample_v1"
SCHEMA_VERSION = "seq100_strict_chan_structure_increment/1"


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (WORKSPACE_ROOT / path).resolve()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _stable_seed(seed: int, *values: Any) -> int:
    encoded = "|".join((str(seed), *(str(value) for value in values))).encode()
    return int(hashlib.sha256(encoded).hexdigest()[:16], 16) % (2**32)


def load_study(path: str | Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study_path = _resolve(path)
    study = json.loads(study_path.read_text(encoding="utf-8"))
    if study.get("study_id") != STUDY_ID:
        raise ValueError("strict_chan_structure_increment_study_id")
    if int(study["source"]["forbidden_year"]) != 2026:
        raise ValueError("strict_chan_structure_increment_forbidden_year")
    if set(map(int, study["source"]["point_types"])) != {1, 2, 3}:
        raise ValueError("strict_chan_structure_increment_point_types")
    if int(study["path"]["maximum_sessions"]) != 40:
        raise ValueError("strict_chan_structure_increment_path_length")
    if not set(study["matching"]["axes"]).issubset(study["coordinates"]):
        raise ValueError("strict_chan_structure_increment_match_axis")
    if study["evaluation"]["primary_target"] not in study["evaluation"]["targets"]:
        raise ValueError("strict_chan_structure_increment_primary_target")
    study["_study_path"] = str(study_path)
    return study


def _sample_spec(study: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    path = _resolve(str(study["source"]["sample_study"]))
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("study_id") != SAMPLE_STUDY_ID:
        raise ValueError("strict_chan_structure_increment_sample_id")
    if spec.get("parent_study_id") != chan.STUDY_ID:
        raise ValueError("strict_chan_structure_increment_sample_parent")
    selection = spec["selection"]
    if tuple(selection.get("selection_columns", ())) != stratified.SELECTION_COLUMNS:
        raise ValueError("strict_chan_structure_increment_selection_columns")
    for name in (
        "future_path_test_performed",
        "return_test_performed",
        "profit_claim_allowed",
        "account_replay_allowed",
        "model_training_allowed",
    ):
        if bool(spec["boundaries"].get(name)):
            raise ValueError(f"strict_chan_structure_increment_sample_boundary:{name}")
    spec["_spec_path"] = str(path)
    manifest = _resolve(str(spec["quality_pool_manifest"]))
    if not manifest.is_file():
        raise ValueError("strict_chan_structure_increment_pool_manifest")
    spec["_quality_pool_manifest_path"] = str(manifest)
    return spec, path


def _freeze_sample(
    study: Mapping[str, Any],
    *,
    root: Path,
    refresh: bool,
) -> dict[str, Any]:
    spec, spec_path = _sample_spec(study)
    frames, partitions, manifest_sha256 = stratified._load_pool_frames(spec)
    excluded_symbols: set[str] = set()
    exclusions: list[dict[str, Any]] = []
    values = spec["selection"].get("exclude_selected_cases", ())
    if isinstance(values, (str, Path)):
        values = [values]
    for value in values:
        path = _resolve(str(value))
        if not path.is_file():
            raise ValueError("strict_chan_structure_increment_exclusion_missing")
        frame = pd.read_parquet(path, columns=["symbol"])
        symbols = set(frame["symbol"].astype(str))
        excluded_symbols.update(symbols)
        exclusions.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "symbol_count": len(symbols),
                "selection_columns_read": ["symbol"],
                "outcomes_read": False,
            }
        )
    if excluded_symbols:
        frames = {
            year: frame[~frame["symbol"].astype(str).isin(excluded_symbols)].copy()
            for year, frame in frames.items()
        }
    cases, statistics = stratified.select_cases_from_frames(spec, frames)
    payload = {
        "schema": SCHEMA_VERSION,
        "study_id": SAMPLE_STUDY_ID,
        "spec_sha256": _sha256(spec_path),
        "runner_sha256": _sha256(Path(__file__)),
        "definition_sha256": _sha256(chan.DEFAULT_DEFINITION_PATH),
        "quality_pool_manifest_sha256": manifest_sha256,
        "source_partitions": partitions,
        "exclusions": exclusions,
        "excluded_symbol_count": len(excluded_symbols),
        "cases": cases,
    }
    fingerprint = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    for case in cases:
        case["selection_fingerprint"] = fingerprint
    manifest = {
        **payload,
        "status": "frozen_sample",
        "selection_fingerprint": fingerprint,
        "outcome_blind_selection": True,
        "returns_used_for_selection": False,
        "parser_outcomes_used_for_selection": False,
        "cases_configured": len(cases),
        "strata_statistics": statistics,
        "cases": cases,
    }
    sample_root = root / "sample"
    manifest_path = sample_root / "sample_manifest.json"
    if manifest_path.is_file() and not refresh:
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current.get("selection_fingerprint") != fingerprint:
            raise ValueError("strict_chan_structure_increment_frozen_sample_mismatch")
        return current
    _write_json(manifest_path, manifest)
    _write_parquet(
        sample_root / "selected_cases.parquet",
        pd.DataFrame(
            [
                {
                    "case_id": case["case_id"],
                    "stratum_id": case["stratum_id"],
                    "sample_index": case["sample_index"],
                    "symbol": case["symbol"],
                    "focal_date": case["focal_date"],
                    "selection_rank": case["selection_rank"],
                    "selection_fingerprint": fingerprint,
                }
                for case in cases
            ]
        ),
    )
    return manifest


def _evaluation_split(signal_date: str, study: Mapping[str, Any]) -> str:
    year = int(signal_date[:4])
    evaluation = study["evaluation"]
    if year in set(map(int, evaluation["development_signal_years"])):
        return "development"
    if year in set(map(int, evaluation["validation_signal_years"])):
        return "validation"
    return "pre_evaluation"


def _period_id(case: Mapping[str, Any]) -> str:
    return "_".join(map(str, case["stratum_years"]))


def _safe_log(value: float, *, floor: float = 1e-12) -> float:
    return float(math.log(max(float(value), floor)))


def _ratio_log(numerator: float, denominator: float) -> float:
    if not (np.isfinite(numerator) and np.isfinite(denominator)):
        return math.nan
    return _safe_log(max(abs(numerator), 1e-12) / max(abs(denominator), 1e-12))


def _pending_features(
    events: Sequence[chan.SegmentStateEvent],
    *,
    confirmed_index: int,
    recent_bars: int = 240,
) -> dict[str, float]:
    visible = [event for event in events if event.confirmed_index <= confirmed_index]
    visible.sort(key=lambda item: (item.confirmed_index, item.id))
    latest: dict[str, chan.SegmentStateEvent] = {}
    opened_at: dict[str, int] = {}
    for event in visible:
        latest[event.candidate_id] = event
        if event.action == "opened":
            opened_at[event.candidate_id] = event.confirmed_index
    active = [event for event in latest.values() if event.action == "opened"]
    ages = [
        confirmed_index - opened_at[event.candidate_id]
        for event in active
        if event.candidate_id in opened_at
    ]
    recent = [
        event
        for event in visible
        if event.confirmed_index >= confirmed_index - recent_bars
    ]
    recent_resolved = [
        event for event in recent if event.action in {"confirmed", "invalidated"}
    ]
    invalidated = sum(event.action == "invalidated" for event in recent_resolved)
    return {
        "active_pending_count_log": math.log1p(len(active)),
        "active_pending_same_down_count_log": math.log1p(
            sum(event.direction < 0 for event in active)
        ),
        "active_pending_max_age_log": math.log1p(max(ages, default=0)),
        "recent_pending_open_count_log": math.log1p(
            sum(event.action == "opened" for event in recent)
        ),
        "recent_pending_resolution_count_log": math.log1p(len(recent_resolved)),
        "recent_pending_invalidation_fraction": (
            invalidated / len(recent_resolved) if recent_resolved else 0.0
        ),
    }


def _signal_segment(
    result: chan.ChanParseResult,
    point: chan.TradePoint,
) -> chan.Segment | None:
    candidates = [
        segment
        for segment in result.segments
        if segment.end_index <= point.event_index
        and segment.confirmed_index <= point.confirmed_index
    ]
    exact = [
        segment for segment in candidates if segment.end_index == point.event_index
    ]
    values = exact or candidates
    return (
        max(values, key=lambda item: (item.end_index, item.confirmed_index))
        if values
        else None
    )


def _point_context(
    result: chan.ChanParseResult,
    point: chan.TradePoint,
) -> tuple[
    chan.Segment | None,
    chan.Segment | None,
    chan.Divergence | None,
    chan.Center | None,
    int,
]:
    segment_map = {item.id: item for item in result.segments}
    divergence_map = {item.id: item for item in result.divergences}
    point_map = {item.id: item for item in result.trade_points}
    trend_map = {item.id: item for item in result.trend_types}
    center_map = {item.id: item for item in result.centers}
    signal = _signal_segment(result, point)
    comparison: chan.Segment | None = None
    divergence: chan.Divergence | None = None
    center: chan.Center | None = None
    trend_center_count = 0
    if point.point_type == 1:
        divergence = divergence_map.get(point.basis_id)
    elif point.point_type == 2:
        first_point = point_map.get(point.basis_id)
        divergence = (
            divergence_map.get(first_point.basis_id)
            if first_point is not None
            else None
        )
    elif point.point_type == 3:
        center = center_map.get(point.basis_id)
    if divergence is not None:
        comparison = segment_map.get(divergence.previous_segment_id)
        if point.point_type == 1:
            signal = segment_map.get(divergence.current_segment_id, signal)
        trend = trend_map.get(divergence.trend_id)
        if trend is not None:
            trend_center_count = len(trend.center_ids)
            if trend.center_ids:
                center = center_map.get(trend.center_ids[-1])
    if comparison is None and signal is not None:
        ordered = list(result.segments)
        try:
            position = ordered.index(signal)
        except ValueError:
            position = -1
        if position > 0:
            comparison = ordered[position - 1]
    return signal, comparison, divergence, center, trend_center_count


def _structure_record(
    result: chan.ChanParseResult,
    point: chan.TradePoint,
) -> dict[str, Any]:
    signal, comparison, divergence, center, trend_center_count = _point_context(
        result, point
    )
    record: dict[str, Any] = {
        "confirmation_delay_bars_log": math.log1p(
            max(point.confirmed_index - point.event_index, 0)
        ),
        "divergence_strength_ratio": (
            float(divergence.strength_ratio) if divergence is not None else math.nan
        ),
        "trend_center_count_log": math.log1p(trend_center_count),
        "confirmed_segment_count_log": math.log1p(
            sum(
                item.confirmed_index <= point.confirmed_index
                for item in result.segments
            )
        ),
        "confirmed_center_count_log": math.log1p(
            sum(
                item.confirmed_index <= point.confirmed_index for item in result.centers
            )
        ),
    }
    if signal is None:
        record.update(
            {
                "signal_segment_span_log": math.nan,
                "signal_segment_abs_price_change": math.nan,
                "signal_segment_speed_log": math.nan,
                "signal_segment_macd_per_bar_price_log": math.nan,
                "signal_segment_amount_per_bar_log": math.nan,
                "signal_segment_break_case": math.nan,
            }
        )
    else:
        price_scale = math.sqrt(max(signal.start_price * signal.end_price, 1e-12))
        macd_scaled = signal.macd_histogram_area / max(
            signal.raw_bar_span * price_scale, 1e-12
        )
        record.update(
            {
                "signal_segment_span_log": math.log1p(signal.raw_bar_span),
                "signal_segment_abs_price_change": abs(signal.price_change),
                "signal_segment_speed_log": _safe_log(
                    signal.duration_normalized_price_change
                ),
                "signal_segment_macd_per_bar_price_log": _safe_log(macd_scaled),
                "signal_segment_amount_per_bar_log": _safe_log(signal.amount_per_bar),
                "signal_segment_break_case": float(signal.break_case),
            }
        )
    if signal is None or comparison is None:
        record.update(
            {
                "comparison_span_ratio_log": math.nan,
                "comparison_price_change_ratio_log": math.nan,
                "comparison_macd_ratio_log": math.nan,
                "comparison_amount_ratio_log": math.nan,
            }
        )
    else:
        record.update(
            {
                "comparison_span_ratio_log": _ratio_log(
                    signal.raw_bar_span, comparison.raw_bar_span
                ),
                "comparison_price_change_ratio_log": _ratio_log(
                    signal.price_change, comparison.price_change
                ),
                "comparison_macd_ratio_log": _ratio_log(
                    signal.macd_histogram_area, comparison.macd_histogram_area
                ),
                "comparison_amount_ratio_log": _ratio_log(
                    signal.amount_per_bar, comparison.amount_per_bar
                ),
            }
        )
    if center is None or center.core_low <= 0 or center.core_high <= center.core_low:
        record.update(
            {
                "center_core_width_log": math.nan,
                "center_fluctuation_width_log": math.nan,
                "center_extension_count_log": math.nan,
                "center_point_location": math.nan,
                "center_component_span_log": math.nan,
            }
        )
    else:
        extensions = sum(
            item.center_id == center.id
            and item.confirmed_index <= point.confirmed_index
            for item in result.center_extensions
        )
        core_width = math.log(center.core_high / center.core_low)
        fluctuation_width = (
            math.log(center.fluctuation_high / center.fluctuation_low)
            if center.fluctuation_low > 0
            and center.fluctuation_high > center.fluctuation_low
            else 0.0
        )
        record.update(
            {
                "center_core_width_log": core_width,
                "center_fluctuation_width_log": fluctuation_width,
                "center_extension_count_log": math.log1p(extensions),
                "center_point_location": (
                    math.log(max(point.price, 1e-12) / center.core_low)
                    / max(core_width, 1e-12)
                ),
                "center_component_span_log": math.log1p(
                    center.end_component_position - center.start_component_position + 1
                ),
            }
        )
    record.update(
        _pending_features(
            result.segment_state_events,
            confirmed_index=point.confirmed_index,
        )
    )
    return record


def _candidate_frame(
    case: Mapping[str, Any],
    episodes: intraday.IntradayEpisodes,
    study: Mapping[str, Any],
) -> pd.DataFrame:
    point_types = set(map(int, study["source"]["point_types"]))
    priority = {
        str(value): index
        for index, value in enumerate(study["source"]["variant_priority"])
    }
    rows: list[dict[str, Any]] = []
    for episode_id, frame in episodes.frame.groupby("episode_id", sort=True):
        parse_frame = frame.loc[:, chan.REQUIRED_INPUT_COLUMNS].reset_index(drop=True)
        result = chan.parse_strict_chan(parse_frame)
        for point in result.trade_points:
            if point.side != "buy" or point.point_type not in point_types:
                continue
            confirmed_date = str(point.confirmed_time)[:10]
            if not (
                str(case["display_start_date"])
                <= confirmed_date
                <= str(case["display_end_date"])
            ):
                continue
            record = asdict(point)
            record.update(_structure_record(result, point))
            record.update(
                {
                    "schema": SCHEMA_VERSION,
                    "case_id": str(case["case_id"]),
                    "symbol": str(case["symbol"]),
                    "symbol_suffix": Path(str(case["symbol"])).suffix,
                    "stratum_id": str(case["stratum_id"]),
                    "period_id": _period_id(case),
                    "focal_date": str(case["focal_date"]),
                    "episode_id": int(episode_id),
                    "signal_date": confirmed_date,
                    "signal_year": int(confirmed_date[:4]),
                    "evaluation_split": _evaluation_split(confirmed_date, study),
                    "variant_priority": priority.get(point.variant, len(priority)),
                }
            )
            rows.append(record)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).sort_values(
        [
            "case_id",
            "point_type",
            "signal_date",
            "variant_priority",
            "confirmed_time",
            "id",
        ]
    )
    frame = frame.drop_duplicates(
        ["case_id", "episode_id", "point_type", "signal_date"], keep="first"
    ).copy()
    frame["signal_key"] = (
        frame["case_id"].astype(str)
        + "|"
        + frame["episode_id"].astype(str)
        + "|"
        + frame["point_type"].astype(str)
        + "|"
        + frame["signal_date"].astype(str)
    )
    return frame


def _path_record(
    *,
    case: Mapping[str, Any],
    episode_id: int,
    daily: pd.DataFrame,
    signal_index: int,
    maximum_sessions: int,
    costs: Mapping[str, Any],
) -> dict[str, Any] | None:
    entry_index = signal_index + 1
    exit_index = entry_index + maximum_sessions - 1
    if entry_index >= len(daily) or exit_index >= len(daily):
        return None
    signal_date = str(daily.iloc[signal_index]["trade_date"])
    entry_date = str(daily.iloc[entry_index]["trade_date"])
    exit_date = str(daily.iloc[exit_index]["trade_date"])
    if exit_date >= "2026-01-01":
        raise ValueError("strict_chan_structure_increment_forbidden_path")
    entry_price = float(daily.iloc[entry_index]["open"])
    raw_entry = float(daily.iloc[entry_index].get("raw_open", entry_price))
    window = daily.iloc[entry_index : exit_index + 1]
    close = window["close"].to_numpy(dtype=np.float64)
    high = window["high"].to_numpy(dtype=np.float64)
    low = window["low"].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(entry_price)
        or entry_price <= 0
        or not np.isfinite(close).all()
        or not np.isfinite(high).all()
        or not np.isfinite(low).all()
    ):
        return None
    close_return = close / entry_price - 1.0
    high_return = high / entry_price - 1.0
    low_return = low / entry_price - 1.0
    terminal = float(close_return[-1])
    raw_exit = float(window.iloc[-1].get("raw_close", close[-1]))
    prices = np.concatenate(([entry_price], close))
    total_variation = float(np.abs(np.diff(np.log(prices))).sum())
    record: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "daily_key": f"{case['case_id']}|{episode_id}|{signal_date}",
        "case_id": str(case["case_id"]),
        "symbol": str(case["symbol"]),
        "symbol_suffix": Path(str(case["symbol"])).suffix,
        "stratum_id": str(case["stratum_id"]),
        "period_id": _period_id(case),
        "focal_date": str(case["focal_date"]),
        "episode_id": int(episode_id),
        "signal_date": signal_date,
        "signal_year": int(signal_date[:4]),
        "entry_date": entry_date,
        "exit_date": exit_date,
        "entry_price": entry_price,
        "raw_entry_price": raw_entry,
        "mean_close_return_40": float(close_return.mean()),
        "median_close_return_40": float(np.median(close_return)),
        "terminal_close_return_40": terminal,
        "mfe_close_40": float(close_return.max()),
        "mae_close_40": float(close_return.min()),
        "mfe_high_40": float(high_return.max()),
        "mae_low_40": float(low_return.min()),
        "positive_close_fraction_40": float((close_return > 0).mean()),
        "upside_area_40": float(np.clip(close_return, 0.0, None).mean()),
        "downside_area_40": float(np.clip(close_return, None, 0.0).mean()),
        "path_efficiency_40": (
            float(math.log(close[-1] / entry_price) / total_variation)
            if total_variation > 0
            else 0.0
        ),
        "mfe_close_session": int(np.argmax(close_return) + 1),
        "mae_close_session": int(np.argmin(close_return) + 1),
    }
    for stress in (False, True):
        buy, sell = outcome._cost_multipliers(
            raw_entry,
            exit_date,
            stress=stress,
            costs=costs,
            exit_price=raw_exit,
        )
        record[
            "terminal_net_return_stress_40" if stress else "terminal_net_return_base_40"
        ] = sell * (1.0 + terminal) / buy - 1.0
    return record


def _daily_path_frame(
    case: Mapping[str, Any],
    episodes: intraday.IntradayEpisodes,
    study: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    maximum_sessions = int(study["path"]["maximum_sessions"])
    costs = study["path"]["costs"]
    for episode_id, frame in episodes.frame.groupby("episode_id", sort=True):
        daily = outcome._daily_frame(frame)
        for signal_index, signal_date in enumerate(daily["trade_date"].astype(str)):
            if not (
                str(case["display_start_date"])
                <= signal_date
                <= str(case["display_end_date"])
            ):
                continue
            record = _path_record(
                case=case,
                episode_id=int(episode_id),
                daily=daily,
                signal_index=signal_index,
                maximum_sessions=maximum_sessions,
                costs=costs,
            )
            if record is not None:
                rows.append(record)
    return pd.DataFrame(rows)


def _partition_fingerprint(
    *,
    study: Mapping[str, Any],
    selection_fingerprint: str,
    cases: Sequence[Mapping[str, Any]],
) -> str:
    sources = (
        Path(__file__).resolve(),
        Path(chan.__file__).resolve(),
        Path(intraday.__file__).resolve(),
        Path(outcome.__file__).resolve(),
        chan.DEFAULT_DEFINITION_PATH.resolve(),
        Path(str(study["_study_path"])),
    )
    payload = {
        "schema": SCHEMA_VERSION,
        "selection_fingerprint": selection_fingerprint,
        "cases": list(cases),
        "source_sha256": {path.name: _sha256(path) for path in sources},
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


def _run_partition(
    *,
    period_id: str,
    cases: Sequence[Mapping[str, Any]],
    study: Mapping[str, Any],
    selection_fingerprint: str,
    output_root: Path,
    resume: bool,
) -> dict[str, Any]:
    part_root = output_root / "partitions" / f"period={period_id}"
    done_path = part_root / "done.json"
    fingerprint = _partition_fingerprint(
        study=study,
        selection_fingerprint=selection_fingerprint,
        cases=cases,
    )
    required = (part_root / "signals.parquet", part_root / "daily_paths.parquet")
    if resume and done_path.is_file() and all(path.is_file() for path in required):
        completed = json.loads(done_path.read_text(encoding="utf-8"))
        if completed.get("fingerprint") == fingerprint:
            return completed
    windows = {
        str(case["symbol"]): (str(case["start_date"]), str(case["end_date"]))
        for case in cases
    }
    if len(windows) != len(cases):
        raise ValueError("strict_chan_structure_increment_duplicate_symbol")
    loaded = intraday.load_symbol_windows_episodes(
        windows,
        temporary_root=part_root / "duckdb_tmp",
        input_provenance={
            "mode": "batched_structure_increment",
            "period_id": period_id,
        },
    )
    signal_frames: list[pd.DataFrame] = []
    path_frames: list[pd.DataFrame] = []
    case_stats: list[dict[str, Any]] = []
    for case in cases:
        episodes = loaded[str(case["symbol"])]
        signals = _candidate_frame(case, episodes, study)
        paths = _daily_path_frame(case, episodes, study)
        if not signals.empty:
            signal_frames.append(signals)
        if not paths.empty:
            path_frames.append(paths)
        case_stats.append(
            {
                "case_id": str(case["case_id"]),
                "symbol": str(case["symbol"]),
                "signal_count": len(signals),
                "daily_path_count": len(paths),
                "usable_days": int(episodes.audit["usable_adjusted_days"]),
                "usable_rows": int(episodes.audit["usable_rows"]),
            }
        )
    signals_all = (
        pd.concat(signal_frames, ignore_index=True)
        if signal_frames
        else pd.DataFrame(columns=["signal_key", "case_id", "point_type"])
    )
    paths_all = (
        pd.concat(path_frames, ignore_index=True)
        if path_frames
        else pd.DataFrame(columns=["daily_key", "case_id", "signal_date"])
    )
    if not signals_all.empty:
        paths_for_signals = paths_all.rename(columns={"daily_key": "path_daily_key"})
        signals_all["path_daily_key"] = (
            signals_all["case_id"].astype(str)
            + "|"
            + signals_all["episode_id"].astype(str)
            + "|"
            + signals_all["signal_date"].astype(str)
        )
        path_columns = [
            column
            for column in paths_for_signals.columns
            if column
            not in {
                "path_daily_key",
                "case_id",
                "symbol",
                "symbol_suffix",
                "stratum_id",
                "period_id",
                "focal_date",
                "episode_id",
                "signal_date",
                "signal_year",
            }
        ]
        signals_all = signals_all.merge(
            paths_for_signals[["path_daily_key", *path_columns]],
            on="path_daily_key",
            how="inner",
            validate="many_to_one",
        ).drop(columns=["path_daily_key"])
    if signals_all["signal_key"].duplicated().any():
        raise ValueError("strict_chan_structure_increment_signal_duplicate")
    if paths_all["daily_key"].duplicated().any():
        raise ValueError("strict_chan_structure_increment_daily_duplicate")
    _write_parquet(part_root / "signals.parquet", signals_all)
    _write_parquet(part_root / "daily_paths.parquet", paths_all)
    result = {
        "schema": SCHEMA_VERSION,
        "status": "completed",
        "period_id": period_id,
        "fingerprint": fingerprint,
        "case_count": len(cases),
        "signal_count": len(signals_all),
        "daily_path_count": len(paths_all),
        "case_stats": case_stats,
    }
    _write_json(done_path, result)
    return result


def _coordinate_join(
    frame: pd.DataFrame,
    *,
    study: Mapping[str, Any],
    sources: Mapping[str, Any],
    temporary_root: Path,
    key_column: str,
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    if frame.empty:
        return frame.copy(), {}
    prepared = frame.copy()
    if key_column != "signal_key":
        prepared = prepared.rename(columns={key_column: "signal_key"})
    joined, runtime = coordinate._join_coordinates(
        prepared,
        study=study,
        sources=sources,
        temporary_root=temporary_root,
    )
    if key_column != "signal_key":
        joined = joined.rename(columns={"signal_key": key_column})
    return joined, runtime


def _attach_point_flags(
    daily: pd.DataFrame,
    signals: pd.DataFrame,
) -> pd.DataFrame:
    pattern = signals[["case_id", "episode_id", "signal_date"]].drop_duplicates()
    pattern["has_strict_buy_point"] = True
    result = daily.merge(
        pattern,
        on=["case_id", "episode_id", "signal_date"],
        how="left",
        validate="one_to_one",
    )
    result["has_strict_buy_point"] = result["has_strict_buy_point"].fillna(False)
    return result


def _match_controls(
    signals: pd.DataFrame,
    daily: pd.DataFrame,
    study: Mapping[str, Any],
) -> pd.DataFrame:
    axes = list(study["matching"]["axes"])
    targets = list(study["evaluation"]["targets"])
    maximum_distance = float(study["matching"]["maximum_distance"])
    eligible_signals = signals[
        signals["coordinate_available"]
        & signals[axes].notna().all(axis=1)
        & signals[targets].notna().all(axis=1)
    ].copy()
    eligible_daily = daily[
        daily["coordinate_available"]
        & ~daily["has_strict_buy_point"]
        & daily[axes].notna().all(axis=1)
        & daily[targets].notna().all(axis=1)
    ].copy()
    rows: list[dict[str, Any]] = []
    group_columns = ["signal_date", "symbol_suffix"]
    control_groups = {
        key: group.sort_values("daily_key").reset_index(drop=True)
        for key, group in eligible_daily.groupby(group_columns, sort=True)
    }
    for key, signal_group in eligible_signals.groupby(group_columns, sort=True):
        controls = control_groups.get(key)
        if controls is None or controls.empty:
            continue
        signal_group = signal_group.sort_values("signal_key").reset_index(drop=True)
        signal_values = signal_group[axes].to_numpy(dtype=np.float64)
        control_values = controls[axes].to_numpy(dtype=np.float64)
        distance = np.sqrt(
            np.mean(
                (signal_values[:, None, :] - control_values[None, :, :]) ** 2,
                axis=2,
            )
        )
        same_symbol = (
            signal_group["symbol"].astype(str).to_numpy()[:, None]
            == controls["symbol"].astype(str).to_numpy()[None, :]
        )
        cost = distance.copy()
        cost[same_symbol] = 1e6
        row_positions, column_positions = linear_sum_assignment(cost)
        for signal_position, control_position in zip(
            row_positions, column_positions, strict=True
        ):
            match_distance = float(distance[signal_position, control_position])
            if (
                cost[signal_position, control_position] >= 1e6
                or match_distance > maximum_distance
            ):
                continue
            signal = signal_group.iloc[signal_position]
            control = controls.iloc[control_position]
            record: dict[str, Any] = {
                "schema": SCHEMA_VERSION,
                "signal_key": str(signal["signal_key"]),
                "control_key": str(control["daily_key"]),
                "case_id": str(signal["case_id"]),
                "control_case_id": str(control["case_id"]),
                "symbol": str(signal["symbol"]),
                "control_symbol": str(control["symbol"]),
                "signal_date": str(signal["signal_date"]),
                "signal_year": int(signal["signal_year"]),
                "evaluation_split": str(signal["evaluation_split"]),
                "point_type": int(signal["point_type"]),
                "match_distance": match_distance,
            }
            for axis in axes:
                record[f"signal__{axis}"] = float(signal[axis])
                record[f"control__{axis}"] = float(control[axis])
            for target in targets:
                signal_value = float(signal[target])
                control_value = float(control[target])
                record[f"signal__{target}"] = signal_value
                record[f"control__{target}"] = control_value
                record[f"difference__{target}"] = signal_value - control_value
            rows.append(record)
    return pd.DataFrame(rows)


def _rank_block(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_output = np.empty((len(train), len(columns)), dtype=np.float64)
    test_output = np.empty((len(test), len(columns)), dtype=np.float64)
    for position, column in enumerate(columns):
        train_values = pd.to_numeric(train[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        test_values = pd.to_numeric(test[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        finite = np.sort(train_values[np.isfinite(train_values)])
        if not len(finite):
            train_output[:, position] = 0.5
            test_output[:, position] = 0.5
            continue
        train_rank = np.searchsorted(finite, train_values, side="right") / (
            len(finite) + 1.0
        )
        test_rank = np.searchsorted(finite, test_values, side="right") / (
            len(finite) + 1.0
        )
        train_rank[~np.isfinite(train_values)] = 0.5
        test_rank[~np.isfinite(test_values)] = 0.5
        train_output[:, position] = train_rank
        test_output[:, position] = test_rank
    return train_output, test_output


def _neighbor_predictions(
    signals: pd.DataFrame,
    study: Mapping[str, Any],
) -> pd.DataFrame:
    coordinates = list(study["coordinates"])
    structures = list(study["structure_features"])
    targets = list(study["evaluation"]["targets"])
    first_year = int(study["evaluation"]["first_prediction_year"])
    minimum = int(study["evaluation"]["minimum_historical_signals_per_point_type"])
    neighbor_count = int(study["evaluation"]["neighbor_count"])
    eligible = signals[
        signals["coordinate_available"]
        & signals[targets].notna().all(axis=1)
        & signals["signal_year"].ge(first_year - 4)
    ].copy()
    rows: list[dict[str, Any]] = []
    for year in sorted(
        value for value in eligible["signal_year"].unique() if value >= first_year
    ):
        test_year = eligible[eligible["signal_year"].eq(year)]
        for point_type, test_type in test_year.groupby("point_type", sort=True):
            historical = eligible[
                eligible["signal_year"].lt(year) & eligible["point_type"].eq(point_type)
            ]
            if len(historical) < minimum:
                continue
            for _, test_row in test_type.iterrows():
                resolved = historical[
                    historical["exit_date"].astype(str).lt(str(test_row["signal_date"]))
                    & historical["case_id"].astype(str).ne(str(test_row["case_id"]))
                ]
                if len(resolved) < minimum:
                    continue
                test_frame = test_row.to_frame().T
                coordinate_train, coordinate_test = _rank_block(
                    resolved, test_frame, coordinates
                )
                structure_train, structure_test = _rank_block(
                    resolved, test_frame, structures
                )
                target_values = resolved[targets].to_numpy(dtype=np.float64)
                coordinate_distance = np.mean(
                    (coordinate_train - coordinate_test[0][None, :]) ** 2,
                    axis=1,
                )
                structure_distance = np.mean(
                    (structure_train - structure_test[0][None, :]) ** 2,
                    axis=1,
                )
                distances = {
                    "coordinates": coordinate_distance,
                    "structure": structure_distance,
                    "combined": 0.5 * coordinate_distance + 0.5 * structure_distance,
                }
                for representation, values in distances.items():
                    available = np.flatnonzero(np.isfinite(values))
                    count = min(neighbor_count, len(available))
                    if count < minimum:
                        continue
                    nearest = available[
                        np.argpartition(values[available], count - 1)[:count]
                    ]
                    predictions = target_values[nearest].mean(axis=0)
                    for target_position, target in enumerate(targets):
                        rows.append(
                            {
                                "schema": SCHEMA_VERSION,
                                "signal_key": str(test_row["signal_key"]),
                                "case_id": str(test_row["case_id"]),
                                "symbol": str(test_row["symbol"]),
                                "signal_date": str(test_row["signal_date"]),
                                "signal_year": int(year),
                                "evaluation_split": str(test_row["evaluation_split"]),
                                "point_type": int(point_type),
                                "representation": representation,
                                "target": target,
                                "actual": float(test_row[target]),
                                "prediction": float(predictions[target_position]),
                                "absolute_error": abs(
                                    float(predictions[target_position])
                                    - float(test_row[target])
                                ),
                                "neighbor_count": int(count),
                                "historical_signal_count": len(resolved),
                                "latest_historical_exit_date": str(
                                    resolved["exit_date"].max()
                                ),
                            }
                        )
    return pd.DataFrame(rows)


def _bootstrap_interval(
    values: pd.Series,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if not len(array):
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    sampled = rng.choice(array, size=(resamples, len(array)), replace=True).mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def _spearman(frame: pd.DataFrame, prediction: str) -> float:
    if len(frame) < 2:
        return math.nan
    actual = frame["actual"].to_numpy(dtype=np.float64)
    predicted = frame[prediction].to_numpy(dtype=np.float64)
    if np.ptp(actual) <= 0 or np.ptp(predicted) <= 0:
        return math.nan
    return float(pd.Series(actual).corr(pd.Series(predicted), method="spearman"))


def _prediction_contrasts(
    predictions: pd.DataFrame,
    study: Mapping[str, Any],
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    key_columns = [
        "signal_key",
        "case_id",
        "symbol",
        "signal_date",
        "signal_year",
        "evaluation_split",
        "point_type",
        "target",
        "actual",
    ]
    wide = predictions.pivot_table(
        index=key_columns,
        columns="representation",
        values=["prediction", "absolute_error"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}__{representation}" for metric, representation in wide]
    wide = wide.reset_index()
    required = [
        "prediction__coordinates",
        "prediction__structure",
        "prediction__combined",
        "absolute_error__coordinates",
        "absolute_error__structure",
        "absolute_error__combined",
    ]
    wide = wide.dropna(subset=required)
    wide["combined_improvement"] = (
        wide["absolute_error__coordinates"] - wide["absolute_error__combined"]
    )
    wide["structure_improvement"] = (
        wide["absolute_error__coordinates"] - wide["absolute_error__structure"]
    )
    rows: list[dict[str, Any]] = []
    resamples = int(study["evaluation"]["bootstrap_case_resamples"])
    base_seed = int(study["evaluation"]["bootstrap_seed"])
    for (split, target), group in wide.groupby(
        ["evaluation_split", "target"], sort=True
    ):
        if split not in {"development", "validation"}:
            continue
        case = group.groupby("case_id", sort=True).agg(
            actual=("actual", "mean"),
            prediction__coordinates=("prediction__coordinates", "mean"),
            prediction__structure=("prediction__structure", "mean"),
            prediction__combined=("prediction__combined", "mean"),
            absolute_error__coordinates=("absolute_error__coordinates", "mean"),
            absolute_error__structure=("absolute_error__structure", "mean"),
            absolute_error__combined=("absolute_error__combined", "mean"),
            combined_improvement=("combined_improvement", "mean"),
            structure_improvement=("structure_improvement", "mean"),
        )
        low, high = _bootstrap_interval(
            case["combined_improvement"],
            resamples=resamples,
            seed=_stable_seed(base_seed, split, target, "combined"),
        )
        structure_low, structure_high = _bootstrap_interval(
            case["structure_improvement"],
            resamples=resamples,
            seed=_stable_seed(base_seed, split, target, "structure"),
        )
        annual = group.groupby("signal_year", sort=True)["combined_improvement"].mean()
        rows.append(
            {
                "schema": SCHEMA_VERSION,
                "evaluation_split": str(split),
                "target": str(target),
                "event_count": len(group),
                "case_count": len(case),
                "coordinate_mae": float(case["absolute_error__coordinates"].mean()),
                "structure_mae": float(case["absolute_error__structure"].mean()),
                "combined_mae": float(case["absolute_error__combined"].mean()),
                "structure_mae_improvement": float(
                    case["structure_improvement"].mean()
                ),
                "structure_improvement_low": structure_low,
                "structure_improvement_high": structure_high,
                "combined_mae_improvement": float(case["combined_improvement"].mean()),
                "combined_improvement_low": low,
                "combined_improvement_high": high,
                "coordinate_spearman": _spearman(case, "prediction__coordinates"),
                "structure_spearman": _spearman(case, "prediction__structure"),
                "combined_spearman": _spearman(case, "prediction__combined"),
                "annual_count": len(annual),
                "positive_improvement_year_fraction": float((annual > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _matched_contrasts(
    pairs: pd.DataFrame,
    study: Mapping[str, Any],
) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    targets = list(study["evaluation"]["targets"])
    resamples = int(study["evaluation"]["bootstrap_case_resamples"])
    base_seed = int(study["evaluation"]["bootstrap_seed"])
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("all", pairs)]
    scopes.extend(
        (f"point_type_{int(point_type)}", group)
        for point_type, group in pairs.groupby("point_type", sort=True)
    )
    for scope, scope_frame in scopes:
        for split, split_frame in scope_frame.groupby("evaluation_split", sort=True):
            if split not in {"development", "validation"}:
                continue
            for target in targets:
                column = f"difference__{target}"
                case = split_frame.groupby("case_id", sort=True)[column].mean()
                low, high = _bootstrap_interval(
                    case,
                    resamples=resamples,
                    seed=_stable_seed(base_seed, scope, split, target, "matched"),
                )
                annual = split_frame.groupby("signal_year", sort=True)[column].mean()
                rows.append(
                    {
                        "schema": SCHEMA_VERSION,
                        "scope": scope,
                        "evaluation_split": str(split),
                        "target": target,
                        "pair_count": len(split_frame),
                        "case_count": len(case),
                        "mean_signal_minus_control": float(case.mean()),
                        "bootstrap_low": low,
                        "bootstrap_high": high,
                        "annual_count": len(annual),
                        "positive_year_fraction": float((annual > 0).mean()),
                        "mean_match_distance": float(
                            split_frame["match_distance"].mean()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _increment_gate(
    contrasts: pd.DataFrame,
    study: Mapping[str, Any],
) -> dict[str, Any]:
    evaluation = study["evaluation"]
    primary = str(evaluation["primary_target"])
    minimum_cases = int(evaluation["minimum_cases_per_split"])
    minimum_year_fraction = float(evaluation["minimum_positive_year_fraction"])
    reasons: list[str] = []
    evidence: list[dict[str, Any]] = []
    for split in ("development", "validation"):
        rows = contrasts[
            contrasts["evaluation_split"].eq(split) & contrasts["target"].eq(primary)
        ]
        if len(rows) != 1:
            reasons.append(f"{split}:missing")
            continue
        row = rows.iloc[0]
        evidence.append(row.to_dict())
        checks = {
            "case_count": int(row["case_count"]) >= minimum_cases,
            "mae_improvement": float(row["combined_mae_improvement"]) > 0,
            "bootstrap_lower": float(row["combined_improvement_low"]) > 0,
            "spearman_increment": float(row["combined_spearman"])
            > float(row["coordinate_spearman"]),
            "positive_year_fraction": float(row["positive_improvement_year_fraction"])
            >= minimum_year_fraction,
        }
        reasons.extend(
            f"{split}:{name}" for name, passed in checks.items() if not passed
        )
    passed = not reasons and len(evidence) == 2
    return {
        "schema": SCHEMA_VERSION,
        "primary_target": primary,
        "strict_increment_gate_passed": passed,
        "full_market_panel_authorized": passed,
        "account_replay_authorized": False,
        "failure_reasons": reasons,
        "evidence": evidence,
    }


def run_study(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    resume: bool = True,
    refresh_sample: bool = False,
    aggregate_only: bool = False,
) -> dict[str, Any]:
    study = load_study(study_path)
    root = _resolve(output_root)
    summary_path = root / "summary.json"
    if resume and summary_path.is_file() and not aggregate_only:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    sample_path = root / "sample" / "sample_manifest.json"
    if aggregate_only:
        if not sample_path.is_file():
            raise ValueError("strict_chan_structure_increment_aggregate_sample_missing")
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
    else:
        sample = _freeze_sample(study, root=root, refresh=refresh_sample)
    cases = list(sample["cases"])
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        grouped.setdefault(_period_id(case), []).append(case)
    partition_results: list[dict[str, Any]] = []
    if aggregate_only:
        for period_id in sorted(grouped):
            done_path = root / "partitions" / f"period={period_id}" / "done.json"
            if not done_path.is_file():
                raise ValueError(
                    "strict_chan_structure_increment_aggregate_partition_missing:"
                    f"{period_id}"
                )
            partition_results.append(json.loads(done_path.read_text(encoding="utf-8")))
    else:
        for period_id, period_cases in sorted(grouped.items()):
            partition_results.append(
                _run_partition(
                    period_id=period_id,
                    cases=period_cases,
                    study=study,
                    selection_fingerprint=str(sample["selection_fingerprint"]),
                    output_root=root,
                    resume=resume,
                )
            )
    part_roots = [
        root / "partitions" / f"period={period_id}" for period_id in sorted(grouped)
    ]
    signals = pd.concat(
        [pd.read_parquet(path / "signals.parquet") for path in part_roots],
        ignore_index=True,
    )
    daily = pd.concat(
        [pd.read_parquet(path / "daily_paths.parquet") for path in part_roots],
        ignore_index=True,
    )
    if signals["signal_date"].astype(str).ge("2026-01-01").any():
        raise ValueError("strict_chan_structure_increment_forbidden_signal")
    if daily["signal_date"].astype(str).ge("2026-01-01").any():
        raise ValueError("strict_chan_structure_increment_forbidden_daily")
    daily = _attach_point_flags(daily, signals)
    coordinate_sources = coordinate._coordinate_sources(study)
    signals_joined, signal_runtime = _coordinate_join(
        signals,
        study=study,
        sources=coordinate_sources,
        temporary_root=root / "coordinate_tmp" / "signals",
        key_column="signal_key",
    )
    daily_joined, daily_runtime = _coordinate_join(
        daily,
        study=study,
        sources=coordinate_sources,
        temporary_root=root / "coordinate_tmp" / "daily",
        key_column="daily_key",
    )
    pairs = _match_controls(signals_joined, daily_joined, study)
    predictions = _neighbor_predictions(signals_joined, study)
    prediction_contrasts = _prediction_contrasts(predictions, study)
    matched_contrasts = _matched_contrasts(pairs, study)
    gate = _increment_gate(prediction_contrasts, study)
    output_frames = {
        "signals": signals_joined,
        "daily_paths": daily_joined,
        "matched_pairs": pairs,
        "neighbor_predictions": predictions,
        "prediction_contrasts": prediction_contrasts,
        "matched_contrasts": matched_contrasts,
    }
    output_records: dict[str, Any] = {}
    for name, frame in output_frames.items():
        path = root / f"{name}.parquet"
        _write_parquet(path, frame)
        output_records[name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "rows": len(frame),
        }
    gate_path = root / "increment_gate.json"
    _write_json(gate_path, gate)
    point_counts = (
        signals_joined.groupby("point_type", sort=True)
        .size()
        .rename("count")
        .reset_index()
    )
    summary = {
        "schema": SCHEMA_VERSION,
        "status": "completed",
        "study_id": STUDY_ID,
        "study": {
            "path": str(Path(str(study["_study_path"])).resolve()),
            "sha256": _sha256(str(study["_study_path"])),
        },
        "sample": {
            "study_id": sample["study_id"],
            "selection_fingerprint": sample["selection_fingerprint"],
            "cases": len(cases),
            "excluded_prior_symbols": int(sample["excluded_symbol_count"]),
            "outcomes_used_for_selection": False,
        },
        "partitions": partition_results,
        "signal_count": len(signals_joined),
        "signal_case_count": int(signals_joined["case_id"].nunique()),
        "point_type_counts": point_counts.to_dict("records"),
        "coordinate_matched_signal_count": int(
            signals_joined["coordinate_available"].sum()
        ),
        "daily_path_count": len(daily_joined),
        "non_pattern_daily_count": int((~daily_joined["has_strict_buy_point"]).sum()),
        "matched_pair_count": len(pairs),
        "matched_signal_fraction": (
            float(pairs["signal_key"].nunique() / len(signals_joined))
            if len(signals_joined)
            else 0.0
        ),
        "prediction_rows": len(predictions),
        "prediction_signal_count": (
            int(predictions["signal_key"].nunique()) if len(predictions) else 0
        ),
        "coordinate_runtime": {
            "signals": dict(signal_runtime),
            "daily": dict(daily_runtime),
        },
        "increment_gate": gate,
        "full_market_panel_authorized": bool(gate["full_market_panel_authorized"]),
        "account_replay_authorized": False,
        "stable_profit_claim_allowed": False,
        "outputs": output_records,
        "gate_path": str(gate_path),
        "source_evidence": {
            "coordinate_manifest": {
                "path": str(coordinate_sources["manifest_path"]),
                "sha256": coordinate_sources["manifest_sha256"],
            },
            "coordinate_validation": {
                "path": str(coordinate_sources["validation_path"]),
                "sha256": coordinate_sources["validation_sha256"],
            },
            "parser": {
                "path": str(Path(chan.__file__)),
                "sha256": _sha256(chan.__file__),
            },
        },
    }
    _write_json(summary_path, summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--refresh-sample", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_study(
        study_path=args.study,
        output_root=args.output_root,
        resume=not args.no_resume,
        refresh_sample=bool(args.refresh_sample),
        aggregate_only=bool(args.aggregate_only),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
