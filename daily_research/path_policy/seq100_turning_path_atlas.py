"""Continuous-path turning atlas with causal directional-change confirmation.

Fixed return horizons remain useful diagnostics, but they are not used to
define a bottom, top, pullback, or failed rebound here.  Historical extrema are
labelled only after a scale-dependent reversal confirms them.  The confirmation
time is a stopping time; the earlier extreme date is explicitly retrospective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from daily_research.path_policy import seq100_hot_path_atlas as atlas
from daily_research.path_policy import seq100_hot_path_neutral as neutral
from daily_research.path_policy import seq100_hot_path_rules as rules

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_turning_path_atlas_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_turning_path_atlas_v1"
)
STUDY_ID = "seq100_turning_path_atlas_v1"
PREPARED_SCHEMA = "seq100_turning_path_atlas_manifest/1"
ANALYSIS_SCHEMA = "seq100_turning_path_atlas_analysis/1"
BUILDER_VERSION = 1

EVENT_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("threshold_multiplier", pa.int16()),
        ("threshold_log_return", pa.float64()),
        ("threshold_bps", pa.int32()),
        ("event_order", pa.int32()),
        ("event_type", pa.string()),
        ("extreme_date", pa.string()),
        ("extreme_date_idx", pa.int32()),
        ("confirmation_date", pa.string()),
        ("confirmation_date_idx", pa.int32()),
        ("confirmation_delay_market_days", pa.int32()),
        ("confirmation_delay_observations", pa.int32()),
        ("extreme_log_price", pa.float64()),
        ("confirmation_log_price", pa.float64()),
        ("confirmation_reversal_log_return", pa.float64()),
        ("previous_extreme_date_idx", pa.int32()),
        ("incoming_leg_log_return", pa.float64()),
        ("incoming_leg_market_days", pa.int32()),
        ("next_extreme_date_idx", pa.int32()),
        ("outgoing_leg_log_return", pa.float64()),
        ("outgoing_leg_market_days", pa.int32()),
    ]
)

EPISODE_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("episode_order", pa.int32()),
        ("episode_type", pa.string()),
        ("outcome", pa.string()),
        ("right_censored", pa.bool_()),
        ("major_threshold_multiplier", pa.int16()),
        ("major_threshold_log_return", pa.float64()),
        ("probe_multiplier", pa.int16()),
        ("probe_log_return", pa.float64()),
        ("anchor_type", pa.string()),
        ("anchor_date", pa.string()),
        ("anchor_date_idx", pa.int32()),
        ("anchor_log_price", pa.float64()),
        ("onset_date", pa.string()),
        ("onset_date_idx", pa.int32()),
        ("onset_log_price", pa.float64()),
        ("onset_reversal_log_return", pa.float64()),
        ("anchor_age_market_days", pa.int32()),
        ("last_major_extreme_date_idx", pa.int32()),
        ("last_major_confirmation_date_idx", pa.int32()),
        ("signed_leg_move_to_anchor", pa.float64()),
        ("days_since_major_confirmation", pa.int32()),
        ("resolution_date", pa.string()),
        ("resolution_date_idx", pa.int32()),
        ("resolution_log_price", pa.float64()),
        ("resolution_delay_market_days", pa.int32()),
    ]
)

PROMINENCE_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("extremum_type", pa.string()),
        ("extreme_date", pa.string()),
        ("extreme_date_idx", pa.int32()),
        ("extreme_log_price", pa.float64()),
        ("prominence_log_return", pa.float64()),
        ("prominence_cost_multiple", pa.float64()),
        ("past_vol20", pa.float64()),
        ("prominence_vol_multiple", pa.float64()),
        ("left_base_date_idx", pa.int32()),
        ("right_base_date_idx", pa.int32()),
        ("left_span_market_days", pa.int32()),
        ("right_span_market_days", pa.int32()),
    ]
)

EPISODE_STATE_FEATURES = (
    "onset_reversal_log_return",
    "anchor_age_market_days",
    "signed_leg_move_to_anchor",
    "days_since_major_confirmation",
)


def load_study(path: str | Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study_path = atlas._resolve_path(path)
    study = atlas._read_json(study_path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("turning_path_study_id_mismatch")
    source = dict(study.get("source", {}) or {})
    if int(source.get("forbidden_year", -1)) != 2026:
        raise ValueError("turning_path_forbidden_year_contract_missing")
    if str(source.get("maximum_outcome_date")) != "2025-12-31":
        raise ValueError("turning_path_outcome_cutoff_mismatch")
    turning = dict(study.get("turning_definition", {}) or {})
    multipliers = [int(value) for value in turning.get("threshold_multipliers", [])]
    if not multipliers or multipliers != sorted(set(multipliers)):
        raise ValueError("turning_path_threshold_multipliers_invalid")
    pairs = list(dict(study.get("episode_definition", {}) or {}).get("scale_pairs", []))
    if not pairs:
        raise ValueError("turning_path_scale_pairs_missing")
    for pair in pairs:
        major = int(pair.get("major_multiplier", -1))
        probe = int(pair.get("probe_multiplier", -1))
        if major not in multipliers or probe not in multipliers or not probe < major:
            raise ValueError("turning_path_scale_pair_invalid")
    if int(turning.get("round_trip_cost_bps", 0)) <= 0:
        raise ValueError("turning_path_cost_contract_invalid")
    return study


def _records_by_year(records: Sequence[Mapping[str, Any]]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for record in records:
        year = int(record["year"])
        path = Path(str(record["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"turning_path_source_panel_missing:{path}")
        result[year] = path
    return result


def _source_contract(
    study_path: Path, study: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    source = dict(study["source"])
    atlas_manifest_path = atlas._resolve_path(str(source["atlas_manifest"]))
    atlas_manifest = atlas._read_json(atlas_manifest_path)
    if atlas_manifest.get("schema") != atlas.MANIFEST_SCHEMA:
        raise ValueError("turning_path_atlas_manifest_schema_mismatch")
    if str(atlas_manifest.get("experiment_fingerprint")) != str(
        source["expected_atlas_fingerprint"]
    ):
        raise ValueError("turning_path_atlas_fingerprint_mismatch")
    dense_path = Path(str(dict(atlas_manifest["dense_base"])["path"]))
    if not dense_path.is_file():
        raise FileNotFoundError("turning_path_dense_base_missing")

    rule_manifest_path = atlas._resolve_path(str(source["rule_manifest"]))
    rule_manifest = atlas._read_json(rule_manifest_path)
    if rule_manifest.get("schema") != rules.MANIFEST_SCHEMA:
        raise ValueError("turning_path_rule_manifest_schema_mismatch")
    if str(rule_manifest.get("experiment_fingerprint")) != str(
        source["expected_rule_fingerprint"]
    ):
        raise ValueError("turning_path_rule_fingerprint_mismatch")

    neutral_manifest_path = atlas._resolve_path(str(source["neutral_manifest"]))
    neutral_manifest = atlas._read_json(neutral_manifest_path)
    if neutral_manifest.get("schema") != neutral.MANIFEST_SCHEMA:
        raise ValueError("turning_path_neutral_manifest_schema_mismatch")
    if str(neutral_manifest.get("experiment_fingerprint")) != str(
        source["expected_neutral_fingerprint"]
    ):
        raise ValueError("turning_path_neutral_fingerprint_mismatch")

    rule_paths = _records_by_year(rule_manifest["compact_panels"])
    neutral_paths = _records_by_year(neutral_manifest["neutral_panels"])
    payload = {
        "builder_version": BUILDER_VERSION,
        "study_sha256": atlas._sha256_file(study_path),
        "atlas_manifest_sha256": atlas._sha256_file(atlas_manifest_path),
        "rule_manifest_sha256": atlas._sha256_file(rule_manifest_path),
        "neutral_manifest_sha256": atlas._sha256_file(neutral_manifest_path),
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    contract = {
        "atlas_manifest": atlas._file_record(atlas_manifest_path),
        "dense_base": atlas._file_record(dense_path, include_hash=False),
        "dense_base_rows": int(dict(atlas_manifest["dense_base"])["row_count"]),
        "rule_manifest": atlas._file_record(rule_manifest_path),
        "rule_panels": {str(year): str(path) for year, path in rule_paths.items()},
        "neutral_manifest": atlas._file_record(neutral_manifest_path),
        "neutral_panels": {
            str(year): str(path) for year, path in neutral_paths.items()
        },
        "fingerprint_payload": payload,
    }
    return contract, fingerprint


def _append_event(
    events: list[dict[str, Any]],
    *,
    event_type: str,
    extreme_pos: int,
    confirmation_pos: int,
    prices: np.ndarray,
    dates: np.ndarray,
    date_indices: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    record = {
        "event_type": event_type,
        "extreme_pos": int(extreme_pos),
        "confirmation_pos": int(confirmation_pos),
        "extreme_date": str(dates[extreme_pos]),
        "extreme_date_idx": int(date_indices[extreme_pos]),
        "confirmation_date": str(dates[confirmation_pos]),
        "confirmation_date_idx": int(date_indices[confirmation_pos]),
        "confirmation_delay_market_days": int(
            date_indices[confirmation_pos] - date_indices[extreme_pos]
        ),
        "confirmation_delay_observations": int(confirmation_pos - extreme_pos),
        "extreme_log_price": float(prices[extreme_pos]),
        "confirmation_log_price": float(prices[confirmation_pos]),
        "confirmation_reversal_log_return": float(
            prices[confirmation_pos] - prices[extreme_pos]
        ),
        "threshold_log_return": float(threshold),
    }
    events.append(record)
    return record


def _start_episode(
    *,
    episode_type: str,
    anchor_type: str,
    anchor_pos: int,
    onset_pos: int,
    prices: np.ndarray,
    dates: np.ndarray,
    date_indices: np.ndarray,
    threshold: float,
    probe: float,
    last_event: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "episode_type": episode_type,
        "anchor_type": anchor_type,
        "anchor_pos": int(anchor_pos),
        "anchor_date": str(dates[anchor_pos]),
        "anchor_date_idx": int(date_indices[anchor_pos]),
        "anchor_log_price": float(prices[anchor_pos]),
        "onset_date": str(dates[onset_pos]),
        "onset_date_idx": int(date_indices[onset_pos]),
        "onset_log_price": float(prices[onset_pos]),
        "onset_reversal_log_return": float(prices[onset_pos] - prices[anchor_pos]),
        "anchor_age_market_days": int(
            date_indices[onset_pos] - date_indices[anchor_pos]
        ),
        "last_major_extreme_date_idx": int(last_event["extreme_date_idx"]),
        "last_major_confirmation_date_idx": int(last_event["confirmation_date_idx"]),
        "signed_leg_move_to_anchor": float(
            prices[anchor_pos] - float(last_event["extreme_log_price"])
        ),
        "days_since_major_confirmation": int(
            date_indices[onset_pos] - int(last_event["confirmation_date_idx"])
        ),
        "major_threshold_log_return": float(threshold),
        "probe_log_return": float(probe),
    }


def _resolve_episode(
    active: Mapping[str, Any],
    *,
    outcome: str,
    resolution_pos: int | None,
    prices: np.ndarray,
    dates: np.ndarray,
    date_indices: np.ndarray,
) -> dict[str, Any]:
    record = dict(active)
    record["outcome"] = outcome
    record["right_censored"] = resolution_pos is None
    if resolution_pos is None:
        record.update(
            {
                "resolution_date": None,
                "resolution_date_idx": None,
                "resolution_log_price": None,
                "resolution_delay_market_days": None,
            }
        )
    else:
        record.update(
            {
                "resolution_date": str(dates[resolution_pos]),
                "resolution_date_idx": int(date_indices[resolution_pos]),
                "resolution_log_price": float(prices[resolution_pos]),
                "resolution_delay_market_days": int(
                    date_indices[resolution_pos] - int(record["onset_date_idx"])
                ),
            }
        )
    return record


def detect_directional_changes(
    prices: np.ndarray,
    dates: np.ndarray,
    date_indices: np.ndarray,
    *,
    threshold: float,
    probe: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return alternating confirmed extrema and optional causal probe episodes."""

    values = np.asarray(prices, dtype=np.float64)
    dates = np.asarray(dates)
    date_indices = np.asarray(date_indices, dtype=np.int64)
    if len(values) != len(dates) or len(values) != len(date_indices):
        raise ValueError("turning_path_array_length_mismatch")
    if len(values) < 2 or threshold <= 0:
        return [], []
    if probe is not None and not (0 < probe < threshold):
        raise ValueError("turning_path_probe_must_be_below_threshold")

    events: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    mode = 0  # 0 unknown, 1 major up state, -1 major down state
    high_pos = low_pos = 0
    high = low = float(values[0])
    last_event: dict[str, Any] | None = None
    active: dict[str, Any] | None = None

    for pos in range(1, len(values)):
        value = float(values[pos])
        if mode == 0:
            if value >= high:
                high, high_pos = value, pos
            if value <= low:
                low, low_pos = value, pos
            if value - low >= threshold:
                last_event = _append_event(
                    events,
                    event_type="trough",
                    extreme_pos=low_pos,
                    confirmation_pos=pos,
                    prices=values,
                    dates=dates,
                    date_indices=date_indices,
                    threshold=threshold,
                )
                mode = 1
                high, high_pos = value, pos
            elif high - value >= threshold:
                last_event = _append_event(
                    events,
                    event_type="peak",
                    extreme_pos=high_pos,
                    confirmation_pos=pos,
                    prices=values,
                    dates=dates,
                    date_indices=date_indices,
                    threshold=threshold,
                )
                mode = -1
                low, low_pos = value, pos
            continue

        if mode == 1:
            if value >= high:
                if active is not None:
                    episodes.append(
                        _resolve_episode(
                            active,
                            outcome="pullback_recovered",
                            resolution_pos=pos,
                            prices=values,
                            dates=dates,
                            date_indices=date_indices,
                        )
                    )
                    active = None
                high, high_pos = value, pos
                continue
            drawdown = high - value
            if probe is not None and active is None and drawdown >= probe:
                if last_event is None:
                    raise AssertionError("turning_path_missing_last_event")
                active = _start_episode(
                    episode_type="up_pullback",
                    anchor_type="running_high",
                    anchor_pos=high_pos,
                    onset_pos=pos,
                    prices=values,
                    dates=dates,
                    date_indices=date_indices,
                    threshold=threshold,
                    probe=probe,
                    last_event=last_event,
                )
            if drawdown >= threshold:
                if active is not None:
                    episodes.append(
                        _resolve_episode(
                            active,
                            outcome="terminal_top",
                            resolution_pos=pos,
                            prices=values,
                            dates=dates,
                            date_indices=date_indices,
                        )
                    )
                    active = None
                last_event = _append_event(
                    events,
                    event_type="peak",
                    extreme_pos=high_pos,
                    confirmation_pos=pos,
                    prices=values,
                    dates=dates,
                    date_indices=date_indices,
                    threshold=threshold,
                )
                mode = -1
                low, low_pos = value, pos
            continue

        if value <= low:
            if active is not None:
                episodes.append(
                    _resolve_episode(
                        active,
                        outcome="downtrend_continued",
                        resolution_pos=pos,
                        prices=values,
                        dates=dates,
                        date_indices=date_indices,
                    )
                )
                active = None
            low, low_pos = value, pos
            continue
        rebound = value - low
        if probe is not None and active is None and rebound >= probe:
            if last_event is None:
                raise AssertionError("turning_path_missing_last_event")
            active = _start_episode(
                episode_type="down_rebound",
                anchor_type="running_low",
                anchor_pos=low_pos,
                onset_pos=pos,
                prices=values,
                dates=dates,
                date_indices=date_indices,
                threshold=threshold,
                probe=probe,
                last_event=last_event,
            )
        if rebound >= threshold:
            if active is not None:
                episodes.append(
                    _resolve_episode(
                        active,
                        outcome="major_bottom_confirmed",
                        resolution_pos=pos,
                        prices=values,
                        dates=dates,
                        date_indices=date_indices,
                    )
                )
                active = None
            last_event = _append_event(
                events,
                event_type="trough",
                extreme_pos=low_pos,
                confirmation_pos=pos,
                prices=values,
                dates=dates,
                date_indices=date_indices,
                threshold=threshold,
            )
            mode = 1
            high, high_pos = value, pos

    if active is not None:
        episodes.append(
            _resolve_episode(
                active,
                outcome="right_censored",
                resolution_pos=None,
                prices=values,
                dates=dates,
                date_indices=date_indices,
            )
        )

    for order, event in enumerate(events):
        event["event_order"] = int(order)
        previous = events[order - 1] if order else None
        following = events[order + 1] if order + 1 < len(events) else None
        event["previous_extreme_date_idx"] = (
            int(previous["extreme_date_idx"]) if previous is not None else None
        )
        event["incoming_leg_log_return"] = (
            float(event["extreme_log_price"] - previous["extreme_log_price"])
            if previous is not None
            else None
        )
        event["incoming_leg_market_days"] = (
            int(event["extreme_date_idx"] - previous["extreme_date_idx"])
            if previous is not None
            else None
        )
        event["next_extreme_date_idx"] = (
            int(following["extreme_date_idx"]) if following is not None else None
        )
        event["outgoing_leg_log_return"] = (
            float(following["extreme_log_price"] - event["extreme_log_price"])
            if following is not None
            else None
        )
        event["outgoing_leg_market_days"] = (
            int(following["extreme_date_idx"] - event["extreme_date_idx"])
            if following is not None
            else None
        )
        event.pop("extreme_pos", None)
        event.pop("confirmation_pos", None)
    for order, episode in enumerate(episodes):
        episode["episode_order"] = int(order)
        episode.pop("anchor_pos", None)
    return events, episodes


class _BufferedParquetWriter:
    def __init__(self, path: Path, schema: pa.Schema, buffer_rows: int) -> None:
        self.path = path
        self.temporary = path.with_suffix(path.suffix + ".partial")
        self.schema = schema
        self.buffer_rows = int(buffer_rows)
        self.records: list[dict[str, Any]] = []
        self.writer: pq.ParquetWriter | None = None
        self.rows = 0
        self.temporary.unlink(missing_ok=True)

    def append(self, records: Sequence[Mapping[str, Any]]) -> None:
        self.records.extend(dict(record) for record in records)
        if len(self.records) >= self.buffer_rows:
            self.flush()

    def flush(self) -> None:
        if not self.records:
            return
        table = pa.Table.from_pylist(self.records, schema=self.schema)
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(
                self.temporary, self.schema, compression="zstd"
            )
        self.writer.write_table(table, row_group_size=self.buffer_rows)
        self.rows += len(self.records)
        self.records.clear()

    def close(self) -> None:
        self.flush()
        if self.writer is None:
            pq.write_table(pa.Table.from_pylist([], schema=self.schema), self.temporary)
        else:
            self.writer.close()
        os.replace(self.temporary, self.path)


def detect_path_prominence(
    prices: np.ndarray,
    dates: np.ndarray,
    date_indices: np.ndarray,
    *,
    cost_log_return: float,
) -> list[dict[str, Any]]:
    """Measure continuous topographic prominence for every local extremum."""

    from scipy.signal import find_peaks, peak_prominences

    values = np.asarray(prices, dtype=np.float64)
    dates = np.asarray(dates)
    date_indices = np.asarray(date_indices, dtype=np.int64)
    if len(values) < 3:
        return []
    returns = np.empty(len(values), dtype=np.float64)
    returns[0] = np.nan
    returns[1:] = np.diff(values)
    past_vol20 = (
        pd.Series(returns)
        .rolling(20, min_periods=10)
        .std(ddof=1)
        .shift(1)
        .to_numpy(dtype=np.float64)
    )
    records: list[dict[str, Any]] = []
    for extremum_type, signed in (("peak", values), ("trough", -values)):
        positions, _ = find_peaks(signed)
        if not len(positions):
            continue
        prominences, left_bases, right_bases = peak_prominences(signed, positions)
        for position, prominence, left_base, right_base in zip(
            positions, prominences, left_bases, right_bases
        ):
            if not math.isfinite(float(prominence)) or float(prominence) <= 0:
                continue
            volatility = float(past_vol20[int(position)])
            records.append(
                {
                    "extremum_type": extremum_type,
                    "extreme_date": str(dates[int(position)]),
                    "extreme_date_idx": int(date_indices[int(position)]),
                    "extreme_log_price": float(values[int(position)]),
                    "prominence_log_return": float(prominence),
                    "prominence_cost_multiple": float(prominence / cost_log_return),
                    "past_vol20": volatility if math.isfinite(volatility) else None,
                    "prominence_vol_multiple": (
                        float(prominence / volatility)
                        if math.isfinite(volatility) and volatility > 0
                        else None
                    ),
                    "left_base_date_idx": int(date_indices[int(left_base)]),
                    "right_base_date_idx": int(date_indices[int(right_base)]),
                    "left_span_market_days": int(
                        date_indices[int(position)] - date_indices[int(left_base)]
                    ),
                    "right_span_market_days": int(
                        date_indices[int(right_base)] - date_indices[int(position)]
                    ),
                }
            )
    return records


def _symbol_records(
    frame: pd.DataFrame,
    study: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    bar_valid = frame["bar_valid"].astype("boolean").fillna(False)
    valid = (
        bar_valid
        & frame["adj_close"].notna()
        & np.isfinite(frame["adj_close"].to_numpy(dtype=np.float64))
        & frame["adj_close"].gt(0)
    )
    work = frame.loc[valid].sort_values("date_idx", kind="mergesort")
    if len(work) < 2:
        return [], [], []
    symbol = str(work.iloc[0]["symbol"])
    prices = np.log(work["adj_close"].to_numpy(dtype=np.float64))
    dates = work["trade_date"].astype(str).to_numpy()
    date_indices = work["date_idx"].to_numpy(dtype=np.int64)
    turning = dict(study["turning_definition"])
    cost = float(turning["round_trip_cost_bps"]) / 10_000.0
    all_events: list[dict[str, Any]] = []
    all_episodes: list[dict[str, Any]] = []
    prominence = detect_path_prominence(
        prices,
        dates,
        date_indices,
        cost_log_return=cost,
    )
    for record in prominence:
        record["symbol"] = symbol
    for multiplier in [int(value) for value in turning["threshold_multipliers"]]:
        threshold = float(multiplier) * cost
        events, _ = detect_directional_changes(
            prices,
            dates,
            date_indices,
            threshold=threshold,
        )
        for event in events:
            event.update(
                {
                    "symbol": symbol,
                    "threshold_multiplier": int(multiplier),
                    "threshold_bps": round(threshold * 10_000),
                }
            )
        all_events.extend(events)
    for pair in list(dict(study["episode_definition"])["scale_pairs"]):
        major = int(pair["major_multiplier"])
        probe_multiplier = int(pair["probe_multiplier"])
        _, episodes = detect_directional_changes(
            prices,
            dates,
            date_indices,
            threshold=float(major) * cost,
            probe=float(probe_multiplier) * cost,
        )
        for episode in episodes:
            episode.update(
                {
                    "symbol": symbol,
                    "major_threshold_multiplier": major,
                    "probe_multiplier": probe_multiplier,
                }
            )
        all_episodes.extend(episodes)
    return all_events, all_episodes, prominence


def _prepared_valid(manifest: Mapping[str, Any], fingerprint: str) -> bool:
    return (
        manifest.get("schema") == PREPARED_SCHEMA
        and manifest.get("status") == "prepared"
        and str(manifest.get("experiment_fingerprint")) == str(fingerprint)
        and Path(str(dict(manifest.get("outputs", {})).get("events", ""))).is_file()
        and Path(str(dict(manifest.get("outputs", {})).get("episodes", ""))).is_file()
        and Path(str(dict(manifest.get("outputs", {})).get("prominence", ""))).is_file()
    )


def prepare_turning_events(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    study_path = atlas._resolve_path(study_path)
    output_root = atlas._resolve_path(output_root)
    study = load_study(study_path)
    contract, fingerprint = _source_contract(study_path, study)
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file() and not force:
        current = atlas._read_json(manifest_path)
        if _prepared_valid(current, fingerprint):
            return current
        if str(current.get("experiment_fingerprint", "")) not in {"", fingerprint}:
            raise ValueError("turning_path_existing_fingerprint_mismatch")

    output_root.mkdir(parents=True, exist_ok=True)
    events_path = output_root / "turning_events.parquet"
    episodes_path = output_root / "turning_episodes.parquet"
    prominence_path = output_root / "turning_prominence.parquet"
    progress_path = output_root / "progress.json"
    resources = dict(study["resources"])
    event_writer = _BufferedParquetWriter(
        events_path, EVENT_SCHEMA, int(resources["writer_buffer_rows"])
    )
    episode_writer = _BufferedParquetWriter(
        episodes_path, EPISODE_SCHEMA, int(resources["writer_buffer_rows"])
    )
    prominence_writer = _BufferedParquetWriter(
        prominence_path, PROMINENCE_SCHEMA, int(resources["writer_buffer_rows"])
    )
    dense_path = Path(str(contract["dense_base"]["path"]))
    parquet = pq.ParquetFile(dense_path)
    pending: pd.DataFrame | None = None
    processed_symbols: set[str] = set()
    input_rows = 0

    def process(frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        symbol = str(frame.iloc[0]["symbol"])
        if symbol in processed_symbols:
            raise ValueError(f"turning_path_noncontiguous_symbol:{symbol}")
        processed_symbols.add(symbol)
        events, episodes, prominence = _symbol_records(frame, study)
        event_writer.append(events)
        episode_writer.append(episodes)
        prominence_writer.append(prominence)
        if len(processed_symbols) % 200 == 0:
            atlas._write_json(
                progress_path,
                {
                    "status": "streaming_turning_events",
                    "processed_symbols": len(processed_symbols),
                    "input_rows": input_rows,
                    "event_rows_flushed": event_writer.rows,
                    "episode_rows_flushed": episode_writer.rows,
                    "prominence_rows_flushed": prominence_writer.rows,
                    "experiment_fingerprint": fingerprint,
                },
            )

    try:
        for batch in parquet.iter_batches(
            batch_size=int(resources["stream_batch_rows"]),
            columns=["symbol", "trade_date", "date_idx", "adj_close", "bar_valid"],
        ):
            frame = batch.to_pandas()
            input_rows += len(frame)
            if pending is not None:
                frame = pd.concat([pending, frame], ignore_index=True)
            last_symbol = str(frame.iloc[-1]["symbol"])
            complete = frame[frame["symbol"].ne(last_symbol)]
            pending = frame[frame["symbol"].eq(last_symbol)].copy()
            for _, group in complete.groupby("symbol", sort=False):
                process(group)
        if pending is not None:
            process(pending)
    finally:
        event_writer.close()
        episode_writer.close()
        prominence_writer.close()

    if input_rows != int(contract["dense_base_rows"]):
        raise ValueError(
            f"turning_path_dense_row_count_mismatch:{input_rows}:{contract['dense_base_rows']}"
        )
    manifest = {
        "schema": PREPARED_SCHEMA,
        "status": "prepared",
        "study_id": STUDY_ID,
        "builder_version": BUILDER_VERSION,
        "experiment_fingerprint": fingerprint,
        "study": atlas._file_record(study_path),
        "source_contract": contract,
        "processed_symbols": len(processed_symbols),
        "input_rows": int(input_rows),
        "event_rows": int(event_writer.rows),
        "episode_rows": int(episode_writer.rows),
        "prominence_rows": int(prominence_writer.rows),
        "outputs": {
            "events": str(events_path.resolve()),
            "episodes": str(episodes_path.resolve()),
            "prominence": str(prominence_path.resolve()),
        },
        "training_performed": False,
        "portfolio_selection_performed": False,
    }
    atlas._write_json(manifest_path, manifest)
    atlas._write_json(
        progress_path,
        {"status": "prepared", "manifest": str(manifest_path.resolve())},
    )
    return manifest


def _connect(output_root: Path, study: Mapping[str, Any]) -> duckdb.DuckDBPyConnection:
    return atlas._connect(output_root, study)


def _feature_panel_query(
    episodes_path: Path,
    rule_paths: Sequence[Path],
    neutral_paths: Sequence[Path],
    features: Sequence[str],
) -> str:
    feature_columns = ",\n        ".join(
        f"r.{name}_rank AS {name}_rank" for name in features
    )
    return f"""
    SELECT
        e.*,
        r.signal_year,
        n.industry,
        n.circ_mv,
        n.total_mv,
        n.turnover_proxy_rank,
        n.size_bin_5,
        n.size_bin_10,
        {feature_columns}
    FROM read_parquet({atlas._sql_quote(episodes_path)}) e
    INNER JOIN {atlas._parquet_scan(rule_paths)} r
      ON e.symbol = r.symbol AND e.onset_date = r.trade_date
    INNER JOIN {atlas._parquet_scan(neutral_paths)} n
      ON e.symbol = n.symbol AND e.onset_date = n.trade_date
    WHERE e.onset_date >= '2012-01-01'
      AND e.onset_date <= '2025-12-31'
    """


def _event_summary(
    connection: duckdb.DuckDBPyConnection, events_path: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary = connection.execute(
        f"""
        SELECT
            threshold_multiplier,
            event_type,
            CAST(left(extreme_date, 4) AS INTEGER) AS event_year,
            count(*) AS events,
            median(confirmation_delay_market_days) AS median_confirmation_days,
            median(abs(incoming_leg_log_return)) AS median_incoming_abs_move,
            median(abs(outgoing_leg_log_return)) AS median_outgoing_abs_move
        FROM read_parquet({atlas._sql_quote(events_path)})
        WHERE extreme_date >= '2012-01-01' AND extreme_date <= '2025-12-31'
        GROUP BY threshold_multiplier, event_type, event_year
        ORDER BY threshold_multiplier, event_type, event_year
        """
    ).fetchdf()
    audit = (
        connection.execute(
            f"""
        SELECT
            count(*) AS rows,
            count(*) - count(DISTINCT
                symbol || '|' || threshold_multiplier::VARCHAR || '|'
                || event_order::VARCHAR
            ) AS duplicate_keys,
            count(*) FILTER (
                WHERE confirmation_date_idx < extreme_date_idx
            ) AS confirmation_before_extreme,
            count(*) FILTER (WHERE left(confirmation_date, 4) = '2026')
                AS forbidden_confirmation_rows,
            min(extreme_date) AS minimum_extreme_date,
            max(confirmation_date) AS maximum_confirmation_date
        FROM read_parquet({atlas._sql_quote(events_path)})
        """
        )
        .fetchdf()
        .iloc[0]
        .to_dict()
    )
    return summary, {
        key: (int(value) if isinstance(value, (np.integer, int)) else str(value))
        for key, value in audit.items()
    }


def _prominence_summary(
    connection: duckdb.DuckDBPyConnection,
    prominence_path: Path,
    study: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    summary = connection.execute(
        f"""
        SELECT
            extremum_type,
            CAST(left(extreme_date, 4) AS INTEGER) AS event_year,
            count(*) AS extrema,
            quantile_cont(prominence_cost_multiple, 0.25) AS cost_multiple_q25,
            median(prominence_cost_multiple) AS cost_multiple_median,
            quantile_cont(prominence_cost_multiple, 0.75) AS cost_multiple_q75,
            quantile_cont(prominence_cost_multiple, 0.90) AS cost_multiple_q90,
            quantile_cont(prominence_cost_multiple, 0.99) AS cost_multiple_q99,
            median(prominence_vol_multiple) AS vol_multiple_median,
            median(left_span_market_days + right_span_market_days)
                AS median_total_span_days
        FROM read_parquet({atlas._sql_quote(prominence_path)})
        WHERE extreme_date >= '2012-01-01' AND extreme_date <= '2025-12-31'
        GROUP BY extremum_type, event_year
        ORDER BY extremum_type, event_year
        """
    ).fetchdf()
    multipliers = [
        int(value)
        for value in dict(study["turning_definition"])["threshold_multipliers"]
    ]
    values = ",".join(f"({value})" for value in multipliers)
    survival = connection.execute(
        f"""
        WITH scales(multiplier) AS (VALUES {values}), base AS (
            SELECT *
            FROM read_parquet({atlas._sql_quote(prominence_path)})
            WHERE extreme_date >= '2012-01-01' AND extreme_date <= '2025-12-31'
        )
        SELECT
            b.extremum_type,
            s.multiplier,
            count(*) FILTER (
                WHERE b.prominence_cost_multiple >= s.multiplier
            ) AS surviving_extrema,
            count(*) AS total_extrema,
            surviving_extrema / NULLIF(total_extrema, 0)::DOUBLE AS survival_share
        FROM base b CROSS JOIN scales s
        GROUP BY b.extremum_type, s.multiplier
        ORDER BY b.extremum_type, s.multiplier
        """
    ).fetchdf()
    audit_row = (
        connection.execute(
            f"""
        SELECT
            count(*) AS rows,
            count(*) - count(DISTINCT
                symbol || '|' || extremum_type || '|' || extreme_date_idx::VARCHAR
            ) AS duplicate_keys,
            count(*) FILTER (
                WHERE prominence_log_return <= 0
                   OR NOT isfinite(prominence_log_return)
            ) AS invalid_prominence,
            count(*) FILTER (WHERE left(extreme_date, 4) = '2026')
                AS forbidden_rows,
            min(extreme_date) AS minimum_date,
            max(extreme_date) AS maximum_date
        FROM read_parquet({atlas._sql_quote(prominence_path)})
        """
        )
        .fetchdf()
        .iloc[0]
        .to_dict()
    )
    audit = {
        key: (int(value) if isinstance(value, (np.integer, int)) else str(value))
        for key, value in audit_row.items()
    }
    return summary, survival, audit


def _episode_summary(features: pd.DataFrame) -> pd.DataFrame:
    resolved = features[~features["right_censored"].astype(bool)].copy()
    return (
        resolved.groupby(
            [
                "probe_multiplier",
                "major_threshold_multiplier",
                "episode_type",
                "outcome",
                "signal_year",
            ],
            sort=True,
        )
        .agg(
            episodes=("symbol", "size"),
            dates=("onset_date", "nunique"),
            median_resolution_days=("resolution_delay_market_days", "median"),
            median_onset_reversal=("onset_reversal_log_return", "median"),
            median_leg_move=("signed_leg_move_to_anchor", "median"),
        )
        .reset_index()
        .rename(columns={"signal_year": "evaluation_year"})
    )


def _target_expression(episode_type: str) -> str:
    if episode_type == "up_pullback":
        return "CASE WHEN outcome = 'terminal_top' THEN 1.0 ELSE 0.0 END"
    if episode_type == "down_rebound":
        return "CASE WHEN outcome = 'major_bottom_confirmed' THEN 1.0 ELSE 0.0 END"
    raise ValueError(f"turning_path_unknown_episode_type:{episode_type}")


def _feature_contrasts(
    connection: duckdb.DuckDBPyConnection,
    feature_path: Path,
    feature_names: Sequence[str],
    *,
    hac_lag: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for episode_type in ("up_pullback", "down_rebound"):
        target = _target_expression(episode_type)
        for feature in feature_names:
            date_cells = connection.execute(
                f"""
                WITH labeled AS (
                    SELECT *, {target} AS target
                    FROM read_parquet({atlas._sql_quote(feature_path)})
                    WHERE episode_type = '{episode_type}'
                      AND NOT right_censored
                      AND {feature} IS NOT NULL
                      AND isfinite({feature})
                )
                SELECT
                    probe_multiplier,
                    major_threshold_multiplier,
                    onset_date,
                    avg({feature}) FILTER (WHERE target = 1.0) AS positive_mean,
                    avg({feature}) FILTER (WHERE target = 0.0) AS negative_mean,
                    count(*) FILTER (WHERE target = 1.0) AS positive_rows,
                    count(*) FILTER (WHERE target = 0.0) AS negative_rows
                FROM labeled
                GROUP BY probe_multiplier, major_threshold_multiplier, onset_date
                HAVING positive_rows > 0 AND negative_rows > 0
                ORDER BY probe_multiplier, major_threshold_multiplier, onset_date
                """
            ).fetchdf()
            for keys, group in date_cells.groupby(
                ["probe_multiplier", "major_threshold_multiplier"], sort=True
            ):
                probe_multiplier, major_multiplier = keys
                difference = (group["positive_mean"] - group["negative_mean"]).to_numpy(
                    dtype=np.float64
                )
                estimate = atlas._hac_mean(difference, lag=int(hac_lag))
                rows.append(
                    {
                        "probe_multiplier": int(probe_multiplier),
                        "major_threshold_multiplier": int(major_multiplier),
                        "episode_type": episode_type,
                        "feature": feature,
                        "matched_dates": len(group),
                        "positive_rows": int(group["positive_rows"].sum()),
                        "negative_rows": int(group["negative_rows"].sum()),
                        **{
                            f"difference_{key}": value
                            for key, value in estimate.items()
                        },
                        "p_two_sided": float(
                            min(
                                1.0,
                                2.0
                                * rules._one_sided_positive_p(
                                    abs(float(estimate["mean"])),
                                    float(estimate["se"]),
                                ),
                            )
                        ),
                    }
                )
    result = pd.DataFrame(rows)
    result["bh_q"] = np.nan
    for positions in result.groupby(
        ["episode_type", "probe_multiplier", "major_threshold_multiplier"],
        sort=True,
    ).groups.values():
        index = list(positions)
        result.loc[index, "bh_q"] = rules._benjamini_hochberg(
            result.loc[index, "p_two_sided"].to_numpy(float)
        )
    return result


def _path_profile_query(
    episode_features_path: Path,
    dense_path: Path,
    study: Mapping[str, Any],
) -> str:
    analysis = dict(study["analysis"])
    start = int(analysis["profile_offset_start"])
    end = int(analysis["profile_offset_end"])
    cap = int(analysis["maximum_profile_events_per_date_outcome"])
    reference = dict(analysis["profile_reference_scale_pair"])
    probe = int(reference["probe_multiplier"])
    major = int(reference["major_multiplier"])
    return f"""
    WITH eligible AS (
        SELECT *,
            row_number() OVER (
                PARTITION BY episode_type, outcome, onset_date
                ORDER BY hash(symbol || '|' || onset_date || '|' || outcome)
            ) AS sample_order
        FROM read_parquet({atlas._sql_quote(episode_features_path)})
        WHERE NOT right_censored
          AND probe_multiplier = {probe}
          AND major_threshold_multiplier = {major}
    ), sampled AS (
        SELECT * FROM eligible WHERE sample_order <= {cap}
    ), paths AS (
        SELECT
            e.episode_type,
            e.outcome,
            d.date_idx - e.onset_date_idx AS relative_market_day,
            ln(d.adj_close) - e.onset_log_price AS relative_log_price,
            ln(d.adj_close) - e.anchor_log_price AS anchor_relative_log_price
        FROM sampled e
        INNER JOIN read_parquet({atlas._sql_quote(dense_path)}) d
          ON e.symbol = d.symbol
         AND d.date_idx BETWEEN e.onset_date_idx + {start}
                            AND e.onset_date_idx + {end}
        WHERE d.bar_valid AND d.adj_close > 0 AND isfinite(d.adj_close)
    )
    SELECT
        episode_type,
        outcome,
        relative_market_day,
        count(*) AS rows,
        quantile_cont(relative_log_price, 0.25) AS relative_log_price_q25,
        median(relative_log_price) AS relative_log_price_median,
        quantile_cont(relative_log_price, 0.75) AS relative_log_price_q75,
        median(anchor_relative_log_price) AS anchor_relative_log_price_median
    FROM paths
    GROUP BY episode_type, outcome, relative_market_day
    ORDER BY episode_type, outcome, relative_market_day
    """


def _choose_examples(features: pd.DataFrame, study: Mapping[str, Any]) -> pd.DataFrame:
    analysis = dict(study["analysis"])
    years = {int(value) for value in analysis["representative_example_years"]}
    reference = dict(analysis["profile_reference_scale_pair"])
    work = features[
        features["signal_year"].isin(years)
        & ~features["right_censored"].astype(bool)
        & features["probe_multiplier"].eq(int(reference["probe_multiplier"]))
        & features["major_threshold_multiplier"].eq(int(reference["major_multiplier"]))
    ].copy()
    chosen: list[pd.Series] = []
    for keys, group in work.groupby(["episode_type", "outcome"], sort=True):
        if group.empty:
            continue
        columns = [
            "onset_reversal_log_return",
            "signed_leg_move_to_anchor",
            "resolution_delay_market_days",
        ]
        distance = np.zeros(len(group), dtype=np.float64)
        for column in columns:
            values = group[column].to_numpy(dtype=np.float64)
            median = float(np.nanmedian(values))
            mad = float(np.nanmedian(np.abs(values - median)))
            scale = mad if mad > 1e-9 else float(np.nanstd(values)) + 1e-9
            distance += np.abs(values - median) / scale
        ordered = group.assign(_distance=distance).sort_values(
            ["_distance", "onset_date", "symbol"], kind="mergesort"
        )
        chosen.append(ordered.iloc[0])
    return pd.DataFrame(chosen).drop(columns=["_distance"], errors="ignore")


def _plot_profiles(profile: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    palette = {"positive": "#2563eb", "negative": "#d97706"}
    mapping = {
        "up_pullback": [
            ("pullback_recovered", "Recovered to a new high", "positive", "-"),
            ("terminal_top", "Became a confirmed top", "negative", "--"),
        ],
        "down_rebound": [
            ("major_bottom_confirmed", "Confirmed a major bottom", "positive", "-"),
            ("downtrend_continued", "Made a new low", "negative", "--"),
        ],
    }
    titles = {
        "up_pullback": "Path after the first probe-scale pullback",
        "down_rebound": "Path after the first probe-scale rebound",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharex=True)
    fig.patch.set_facecolor("#fafafa")
    for axis, episode_type in zip(axes, ("up_pullback", "down_rebound")):
        axis.set_facecolor("#fafafa")
        for outcome, label, color_key, linestyle in mapping[episode_type]:
            data = profile[
                profile["episode_type"].eq(episode_type)
                & profile["outcome"].eq(outcome)
            ].sort_values("relative_market_day")
            x = data["relative_market_day"].to_numpy(float)
            median = np.expm1(data["relative_log_price_median"].to_numpy(float)) * 100
            low = np.expm1(data["relative_log_price_q25"].to_numpy(float)) * 100
            high = np.expm1(data["relative_log_price_q75"].to_numpy(float)) * 100
            color = palette[color_key]
            axis.fill_between(x, low, high, color=color, alpha=0.12, linewidth=0)
            axis.plot(
                x, median, color=color, linewidth=2.0, linestyle=linestyle, label=label
            )
        axis.axvline(0, color="#374151", linewidth=1.0, linestyle=":")
        axis.axhline(0, color="#9ca3af", linewidth=0.8)
        axis.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        axis.set_title(titles[episode_type], loc="left", fontsize=12, color="#111827")
        axis.set_xlabel("Market days from causal probe onset")
        axis.legend(frameon=False, fontsize=9, loc="best")
    axes[0].set_ylabel("Adjusted close relative to onset (%)")
    fig.suptitle(
        "Directional-change episode paths",
        x=0.07,
        y=1.01,
        ha="left",
        fontsize=15,
        color="#111827",
    )
    fig.text(
        0.07,
        0.955,
        "Median and interquartile range; 2012-2025; alignment offsets are descriptive only",
        ha="left",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    plt.close(fig)


def _plot_examples(
    connection: duckdb.DuckDBPyConnection,
    examples: pd.DataFrame,
    dense_path: Path,
    events_path: Path,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    if examples.empty:
        raise ValueError("turning_path_no_representative_examples")
    rows = examples.to_dict("records")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    fig.patch.set_facecolor("#fafafa")
    axes_flat = axes.ravel()
    colors = {"up_pullback": "#2563eb", "down_rebound": "#d97706"}
    for axis, record in zip(axes_flat, rows):
        start = int(record["onset_date_idx"]) - 35
        end = int(record["onset_date_idx"]) + 55
        path = connection.execute(
            f"""
            SELECT trade_date, date_idx, adj_close
            FROM read_parquet({atlas._sql_quote(dense_path)})
            WHERE symbol = ? AND date_idx BETWEEN ? AND ?
              AND bar_valid AND adj_close > 0
            ORDER BY date_idx
            """,
            [str(record["symbol"]), start, end],
        ).fetchdf()
        path["relative_day"] = path["date_idx"] - int(record["onset_date_idx"])
        path["normalized"] = path["adj_close"] / math.exp(
            float(record["onset_log_price"])
        )
        color = colors[str(record["episode_type"])]
        axis.plot(path["relative_day"], path["normalized"], color=color, linewidth=1.8)
        markers = [
            (int(record["anchor_date_idx"]), "Anchor", "#374151", "o"),
            (int(record["onset_date_idx"]), "Probe", "#7c3aed", "s"),
            (int(record["resolution_date_idx"]), "Resolved", "#111827", "X"),
        ]
        for date_idx, label, marker_color, marker in markers:
            point = path[path["date_idx"].eq(date_idx)]
            if point.empty:
                continue
            x = float(point.iloc[0]["relative_day"])
            y = float(point.iloc[0]["normalized"])
            axis.scatter([x], [y], s=45, color=marker_color, marker=marker, zorder=3)
            axis.annotate(
                label, (x, y), xytext=(4, 7), textcoords="offset points", fontsize=8
            )
        nested = connection.execute(
            f"""
            SELECT threshold_multiplier, event_type, extreme_date_idx
            FROM read_parquet({atlas._sql_quote(events_path)})
            WHERE symbol = ? AND extreme_date_idx BETWEEN ? AND ?
            ORDER BY threshold_multiplier, extreme_date_idx
            """,
            [str(record["symbol"]), start, end],
        ).fetchdf()
        for _, event in nested[nested["threshold_multiplier"].eq(8)].iterrows():
            point = path[path["date_idx"].eq(int(event["extreme_date_idx"]))]
            if not point.empty:
                axis.scatter(
                    [float(point.iloc[0]["relative_day"])],
                    [float(point.iloc[0]["normalized"])],
                    s=22,
                    facecolors="none",
                    edgecolors="#111827",
                    linewidths=1.0,
                    zorder=2,
                )
        axis.axvline(0, color="#9ca3af", linewidth=0.9, linestyle=":")
        axis.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        axis.set_facecolor("#fafafa")
        axis.set_title(
            f"{record['symbol']} | {record['episode_type']} -> {record['outcome']}",
            loc="left",
            fontsize=10,
            color="#111827",
        )
        axis.set_xlabel("Market days from probe onset")
        axis.set_ylabel("Adjusted close / onset close")
    for axis in axes_flat[len(rows) :]:
        axis.set_visible(False)
    fig.suptitle(
        "Deterministic median-state episode examples",
        x=0.07,
        y=0.995,
        ha="left",
        fontsize=15,
        color="#111827",
    )
    fig.text(
        0.07,
        0.955,
        "Cases are selected by distance to each outcome group's median state, not by visual appeal",
        ha="left",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    plt.close(fig)


def _research_record(
    event_summary: pd.DataFrame,
    prominence_summary: pd.DataFrame,
    prominence_survival: pd.DataFrame,
    episode_summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    audit: Mapping[str, Any],
    prominence_audit: Mapping[str, Any],
    feature_join: Mapping[str, Any],
    study: Mapping[str, Any],
) -> str:
    totals = (
        event_summary.groupby(["threshold_multiplier", "event_type"], sort=True)[
            "events"
        ]
        .sum()
        .reset_index()
    )
    episode_totals = (
        episode_summary.groupby(
            [
                "probe_multiplier",
                "major_threshold_multiplier",
                "episode_type",
                "outcome",
            ],
            sort=True,
        )["episodes"]
        .sum()
        .reset_index()
    )
    reference = dict(dict(study["analysis"])["profile_reference_scale_pair"])
    reference_contrasts = contrasts[
        contrasts["probe_multiplier"].eq(int(reference["probe_multiplier"]))
        & contrasts["major_threshold_multiplier"].eq(int(reference["major_multiplier"]))
    ]
    leading = (
        reference_contrasts.sort_values(
            ["episode_type", "bh_q", "difference_mean"],
            ascending=[True, True, False],
            kind="mergesort",
        )
        .groupby("episode_type", sort=True)
        .head(10)
    )
    lines = [
        "# Continuous Turning-Path Atlas V1",
        "",
        "This study does not define the market by D5 or D20, and it does not declare one reversal threshold to be the true market scale. Every local peak and trough receives a continuous topographic log-price prominence. Directional-change thresholds form a 1/2/3/4/6/8/12/16/24-times-cost sensitivity lattice used only to measure when a historical extreme becomes causally confirmable.",
        "",
        "Pullback/top and rebound/bottom comparisons are evaluated over every frozen probe/major scale pair. The 2x/8x pair is used only for representative figures; it is not privileged as ground truth. No episode has a fixed resolution horizon.",
        "",
        "## Continuous extremum prominence by year",
        "",
        prominence_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Extrema surviving each cost scale",
        "",
        prominence_survival.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Multi-scale confirmed extrema",
        "",
        totals.to_markdown(index=False),
        "",
        "## Causal probe outcomes",
        "",
        episode_totals.to_markdown(index=False),
        "",
        "## Strongest same-date feature differences at the reference figure scale",
        "",
        leading[
            [
                "episode_type",
                "probe_multiplier",
                "major_threshold_multiplier",
                "feature",
                "matched_dates",
                "difference_mean",
                "difference_lcb_95",
                "difference_ucb_95",
                "bh_q",
            ]
        ].to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Mechanical audit",
        "",
        f"- Event rows: {int(audit['rows']):,}",
        f"- Duplicate event keys: {int(audit['duplicate_keys']):,}",
        f"- Confirmations before extrema: {int(audit['confirmation_before_extreme']):,}",
        f"- 2026 confirmations: {int(audit['forbidden_confirmation_rows']):,}",
        f"- Continuous-prominence rows: {int(prominence_audit['rows']):,}",
        f"- Invalid prominence rows: {int(prominence_audit['invalid_prominence']):,}",
        f"- Duplicate prominence keys: {int(prominence_audit['duplicate_keys']):,}",
        f"- Resolved formal episodes: {int(feature_join['resolved_formal_episodes']):,}",
        f"- Resolved episodes with complete signal-time features: {int(feature_join['resolved_feature_rows']):,}",
        f"- Feature join coverage: {float(feature_join['coverage']):.2%}",
        "",
        "## Boundary",
        "",
        "Extreme dates, outcome classes, and aligned post-onset paths use future information and are discovery labels only. Real-time use begins at the probe onset and may use only signal-time features. Feature differences are descriptive until a chronological model trained only on already resolved episodes predicts later years. The scale hierarchy is tied to cost, but it is not proof that any one scale is optimal or profitable.",
        "",
    ]
    return "\n".join(lines)


def analyze_turning_atlas(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study_path = atlas._resolve_path(study_path)
    output_root = atlas._resolve_path(output_root)
    study = load_study(study_path)
    prepared = prepare_turning_events(
        study_path=study_path, output_root=output_root, force=False
    )
    contract = dict(prepared["source_contract"])
    events_path = Path(str(dict(prepared["outputs"])["events"]))
    episodes_path = Path(str(dict(prepared["outputs"])["episodes"]))
    prominence_path = Path(str(dict(prepared["outputs"])["prominence"]))
    dense_path = Path(str(contract["dense_base"]["path"]))
    rule_paths = [Path(value) for _, value in sorted(contract["rule_panels"].items())]
    neutral_paths = [
        Path(value) for _, value in sorted(contract["neutral_panels"].items())
    ]
    rule_manifest = atlas._read_json(Path(str(contract["rule_manifest"]["path"])))
    features = [str(value) for value in rule_manifest["feature_ranks"]]
    feature_columns = [f"{name}_rank" for name in features]
    output_root.mkdir(parents=True, exist_ok=True)
    feature_path = output_root / "episode_features.parquet"
    connection = _connect(output_root, study)
    try:
        atlas._copy_query(
            connection,
            _feature_panel_query(episodes_path, rule_paths, neutral_paths, features),
            feature_path,
        )
        resolved_formal = int(
            connection.execute(
                f"""
                SELECT count(*)
                FROM read_parquet({atlas._sql_quote(episodes_path)})
                WHERE NOT right_censored
                  AND onset_date >= '2012-01-01' AND onset_date <= '2025-12-31'
                """
            ).fetchone()[0]
        )
        feature_rows = int(
            connection.execute(
                f"SELECT count(*) FROM read_parquet({atlas._sql_quote(feature_path)})"
            ).fetchone()[0]
        )
        resolved_feature_rows = int(
            connection.execute(
                f"""
                SELECT count(*)
                FROM read_parquet({atlas._sql_quote(feature_path)})
                WHERE NOT right_censored
                """
            ).fetchone()[0]
        )
        duplicate_features = int(
            connection.execute(
                f"""
                SELECT count(*) - count(DISTINCT
                    symbol || '|' || major_threshold_multiplier::VARCHAR || '|'
                    || probe_multiplier::VARCHAR || '|' || episode_order::VARCHAR
                )
                FROM read_parquet({atlas._sql_quote(feature_path)})
                """
            ).fetchone()[0]
        )
        if duplicate_features != 0:
            raise ValueError("turning_path_duplicate_episode_features")
        event_summary, audit = _event_summary(connection, events_path)
        prominence_summary, prominence_survival, prominence_audit = _prominence_summary(
            connection, prominence_path, study
        )
        features_frame = connection.execute(
            f"""
            SELECT probe_multiplier, major_threshold_multiplier,
                   episode_type, outcome, right_censored, signal_year,
                   symbol, onset_date, onset_date_idx, onset_log_price,
                   anchor_date, anchor_date_idx, anchor_log_price,
                   resolution_date, resolution_date_idx,
                   onset_reversal_log_return, signed_leg_move_to_anchor,
                   resolution_delay_market_days
            FROM read_parquet({atlas._sql_quote(feature_path)})
            """
        ).fetchdf()
        episode_summary = _episode_summary(features_frame)
        contrast_features = [*EPISODE_STATE_FEATURES, *feature_columns]
        contrasts = _feature_contrasts(
            connection,
            feature_path,
            contrast_features,
            hac_lag=int(dict(study["analysis"])["hac_lag"]),
        )
        profiles = connection.execute(
            _path_profile_query(feature_path, dense_path, study)
        ).fetchdf()
        examples = _choose_examples(features_frame, study)
        profile_path = output_root / "path_profiles.csv"
        event_summary_path = output_root / "event_summary.csv"
        prominence_summary_path = output_root / "prominence_summary.csv"
        prominence_survival_path = output_root / "prominence_scale_survival.csv"
        episode_summary_path = output_root / "episode_summary.csv"
        contrasts_path = output_root / "feature_contrasts.csv"
        examples_path = output_root / "representative_examples.csv"
        profile_figure = output_root / "figures/path_profiles.png"
        examples_figure = output_root / "figures/representative_examples.png"
        profiles.to_csv(profile_path, index=False)
        event_summary.to_csv(event_summary_path, index=False)
        prominence_summary.to_csv(prominence_summary_path, index=False)
        prominence_survival.to_csv(prominence_survival_path, index=False)
        episode_summary.to_csv(episode_summary_path, index=False)
        contrasts.to_csv(contrasts_path, index=False)
        examples.to_csv(examples_path, index=False)
        _plot_profiles(profiles, profile_figure)
        _plot_examples(
            connection,
            examples,
            dense_path,
            events_path,
            examples_figure,
        )
    finally:
        connection.close()

    feature_join = {
        "resolved_formal_episodes": resolved_formal,
        "feature_rows": feature_rows,
        "resolved_feature_rows": resolved_feature_rows,
        "coverage": resolved_feature_rows / max(resolved_formal, 1),
        "duplicate_keys": duplicate_features,
    }
    record_path = output_root / "research_record.md"
    record_path.write_text(
        _research_record(
            event_summary,
            prominence_summary,
            prominence_survival,
            episode_summary,
            contrasts,
            audit,
            prominence_audit,
            feature_join,
            study,
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema": ANALYSIS_SCHEMA,
        "status": "completed",
        "study_id": STUDY_ID,
        "builder_version": BUILDER_VERSION,
        "experiment_fingerprint": str(prepared["experiment_fingerprint"]),
        "study": atlas._file_record(study_path),
        "source_manifest": atlas._file_record(output_root / "manifest.json"),
        "event_audit": audit,
        "prominence_audit": prominence_audit,
        "feature_join": feature_join,
        "outputs": {
            "episode_features": str(feature_path.resolve()),
            "event_summary": str(event_summary_path.resolve()),
            "prominence_summary": str(prominence_summary_path.resolve()),
            "prominence_scale_survival": str(prominence_survival_path.resolve()),
            "episode_summary": str(episode_summary_path.resolve()),
            "feature_contrasts": str(contrasts_path.resolve()),
            "path_profiles": str(profile_path.resolve()),
            "representative_examples": str(examples_path.resolve()),
            "path_profile_figure": str(profile_figure.resolve()),
            "representative_examples_figure": str(examples_figure.resolve()),
            "research_record": str(record_path.resolve()),
        },
        "training_performed": False,
        "portfolio_selection_performed": False,
        "profit_claim_allowed": False,
    }
    atlas._write_json(output_root / "analysis_manifest.json", manifest)
    atlas._write_json(
        output_root / "progress.json",
        {
            "status": "completed",
            "analysis_manifest": str(
                (output_root / "analysis_manifest.json").resolve()
            ),
        },
    )
    return manifest


def run_study(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force_prepare: bool = False,
) -> dict[str, Any]:
    prepare_turning_events(
        study_path=study_path,
        output_root=output_root,
        force=force_prepare,
    )
    return analyze_turning_atlas(study_path=study_path, output_root=output_root)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a continuous-path directional-change atlas."
    )
    parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--force-prepare", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    result = run_study(
        study_path=args.study,
        output_root=args.output_root,
        force_prepare=bool(args.force_prepare),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=atlas._json_default))
    return result


if __name__ == "__main__":
    main()
