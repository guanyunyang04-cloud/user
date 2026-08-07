"""Continuous causal breakout-retest state and action-value study.

The study keeps the boundary and normalization scale fixed when an upward
center breakout occurs, then updates only with bars observed through the
current close.  It evaluates the resulting state against causally available
dynamic action-value experience; it does not define a fixed holding horizon or
declare a retest to be a profitable label.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import SplineTransformer

from daily_research.path_policy import seq100_causal_path_structure as structure
from daily_research.path_policy import (
    seq100_dynamic_action_distribution as distribution,
)
from daily_research.path_policy import seq100_dynamic_action_value_baselines as baseline
from daily_research.path_policy import seq100_hot_path_atlas as atlas

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_causal_retest_state_v1.json"
)
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / (
    "daily_research/output/path_policy/studies/seq100_causal_retest_state_v1"
)
STUDY_ID = "seq100_causal_retest_state_v1"
PANEL_SCHEMA_VERSION = "seq100_causal_retest_state_panel/1"
MANIFEST_SCHEMA_VERSION = "seq100_causal_retest_state/1"
VALIDATION_SCHEMA_VERSION = "seq100_causal_retest_state_validation/1"
BUILDER_VERSION = 1

SCOPE_NAMES = (
    "up_retest_event",
    "up_price_amount_confirmed_breakout",
)
MODEL_NAMES = (
    "inherited_top20_additive",
    "causal_rank_spline",
    "historical_neighbor",
)
MOMENT_NAMES = (
    "positive_probability",
    "upside_component",
    "downside_component",
    "continuous_action_value",
)

STATE_BOOLEAN_FEATURES = {
    "up_breakout_event",
    "up_breakout_active",
    "up_retest_event",
    "up_retest_history",
    "up_repeated_retest",
    "up_intraday_boundary_penetration",
    "up_reentry_event",
}
STATE_INTEGER_FEATURES = {
    "sessions_since_up_breakout",
    "market_days_since_up_breakout",
    "up_retest_bar_count",
    "up_retest_episode_count",
    "sessions_since_first_up_retest",
    "sessions_since_last_up_retest",
    "center_age_at_breakout",
}
STATE_FLOAT_FEATURES = {
    "locked_breakout_scale_unit",
    "center_width_units_at_breakout",
    "breakout_strength_units",
    "boundary_close_distance_units",
    "boundary_low_distance_units",
    "boundary_high_distance_units",
    "peak_expansion_units",
    "trough_distance_units",
    "drawdown_from_peak_units",
    "drawdown_fraction_of_peak_expansion",
    "close_change_from_breakout_units",
    "mfe_from_breakout_units",
    "mae_from_breakout_units",
    "path_efficiency_since_breakout",
    "amount_log_ratio_to_breakout",
    "amount_log_ratio_to_prebreakout_mean5",
    "amount_log_ratio_to_postbreakout_peak",
    "amount_log_ratio_to_previous_bar",
}
CONTEXT_FEATURES = (
    "dc_small_mode",
    "dc_medium_mode",
    "dc_large_mode",
    "dc_small_retracement_ratio",
    "dc_medium_retracement_ratio",
    "dc_large_retracement_ratio",
    "dc_small_slope_ratio",
    "dc_medium_slope_ratio",
    "dc_large_slope_ratio",
    "dc_small_amount_ratio",
    "dc_medium_amount_ratio",
    "dc_large_amount_ratio",
    "chan_stroke_slope_ratio",
    "chan_stroke_amount_ratio",
)
PANEL_FEATURE_COLUMNS = tuple(
    sorted(STATE_BOOLEAN_FEATURES)
    + sorted(STATE_INTEGER_FEATURES)
    + sorted(STATE_FLOAT_FEATURES)
    + list(CONTEXT_FEATURES)
)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else WORKSPACE_ROOT / value


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = _resolve(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_frame(path: Path, frame: pd.DataFrame, *, row_group_size: int) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(
        temporary,
        index=False,
        compression="zstd",
        row_group_size=int(row_group_size),
    )
    os.replace(temporary, path)
    record = atlas._file_record(path)
    record["rows"] = len(frame)
    return record


def load_study(path: str | Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study = _read_json(path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("causal_retest_state_study_id")
    source = dict(study["source"])
    if source.get("quality_pool_name") != "quality_liquidity_pit":
        raise ValueError("causal_retest_state_pool")
    if int(source.get("forbidden_year", -1)) != 2026:
        raise ValueError("causal_retest_state_forbidden_year")
    period = dict(study["period"])
    if tuple(int(value) for value in period["formal_years"]) != tuple(range(2012, 2026)):
        raise ValueError("causal_retest_state_period")
    if tuple(int(value) for value in period["strict_oos_years"]) != tuple(range(2013, 2026)):
        raise ValueError("causal_retest_state_oos_period")
    state = dict(study["state"])
    if bool(state["fixed_holding_horizon_used"]):
        raise ValueError("causal_retest_state_fixed_horizon")
    if bool(state["future_turning_confirmation_written_back"]):
        raise ValueError("causal_retest_state_future_writeback")
    target = dict(study["target"])
    if bool(target["binary_good_stock_label_used"]):
        raise ValueError("causal_retest_state_binary_label")
    if bool(target["final_2025_oracle_row_labels_allowed"]):
        raise ValueError("causal_retest_state_final_oracle")
    if int(target["minimum_sessions_after_resolution"]) != 10:
        raise ValueError("causal_retest_state_maturity")
    if tuple(dict(study["scopes"])) != SCOPE_NAMES:
        raise ValueError("causal_retest_state_scopes")
    estimators = dict(study["estimators"])
    if tuple(estimators["models"]) != MODEL_NAMES:
        raise ValueError("causal_retest_state_models")
    if bool(estimators["hyperparameter_selection_performed"]):
        raise ValueError("causal_retest_state_hyperparameter_selection")
    features = dict(study["features"])
    state_names = [str(value) for value in features["continuous_state_coordinates"]]
    if not set(state_names).issubset(set(PANEL_FEATURE_COLUMNS)):
        missing = sorted(set(state_names) - set(PANEL_FEATURE_COLUMNS))
        raise ValueError(f"causal_retest_state_missing_panel_features:{missing}")
    return study


def _verified_file(source: Mapping[str, Any], key: str, hash_key: str) -> Path:
    path = _resolve(source[key])
    digest = _sha256(path)
    if digest != str(source[hash_key]):
        raise ValueError(f"causal_retest_state_source_hash:{key}")
    return path


def _source_contract(
    study_path: Path, study: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = dict(study["source"])
    requests = (
        ("path_structure_study", "expected_path_structure_study_sha256"),
        ("path_structure_module", "expected_path_structure_module_sha256"),
        ("path_structure_manifest", "expected_path_structure_manifest_sha256"),
        ("path_structure_validation", "expected_path_structure_validation_sha256"),
        ("distribution_study", "expected_distribution_study_sha256"),
        ("distribution_manifest", "expected_distribution_manifest_sha256"),
        ("distribution_validation", "expected_distribution_validation_sha256"),
    )
    paths = {
        key: _verified_file(source, key, hash_key)
        for key, hash_key in requests
    }
    path_manifest = _read_json(paths["path_structure_manifest"])
    path_validation = _read_json(paths["path_structure_validation"])
    distribution_manifest = _read_json(paths["distribution_manifest"])
    distribution_validation = _read_json(paths["distribution_validation"])
    if path_manifest.get("status") != "completed" or path_validation.get("status") != "passed":
        raise ValueError("causal_retest_state_path_source_incomplete")
    if distribution_manifest.get("status") != "completed" or distribution_validation.get("status") != "passed":
        raise ValueError("causal_retest_state_distribution_source_incomplete")
    if bool(path_manifest.get("audit", {}).get("account_replay_allowed", True)):
        raise ValueError("causal_retest_state_path_replay_claim")
    if bool(distribution_manifest.get("audit", {}).get("final_oracle_row_labels_used", True)):
        raise ValueError("causal_retest_state_distribution_leak")
    expected_rows = int(source["expected_panel_rows"])
    if int(path_manifest.get("audit", {}).get("structure_rows", -1)) != expected_rows:
        raise ValueError("causal_retest_state_path_rows")
    path_contract = dict(path_manifest["source_contract"])
    coordinate_records = list(distribution_manifest["coordinates"])
    prediction_records = list(distribution_manifest["predictions"])
    if len(coordinate_records) != 14 or len(prediction_records) != 26:
        raise ValueError("causal_retest_state_distribution_partitions")
    for record in [*coordinate_records, *prediction_records]:
        path = Path(str(record["path"]))
        if not path.is_file() or _sha256(path) != str(record["sha256"]):
            raise ValueError(f"causal_retest_state_partition_hash:{path}")
    contract = {
        key: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for key, path in paths.items()
    }
    contract.update(
        {
            "dense_base": dict(path_contract["dense_base"]),
            "row_index": dict(path_contract["row_index"]),
            "coordinate_partitions": coordinate_records,
            "prediction_partitions": prediction_records,
            "path_structure_panels": list(path_manifest["panel_manifest"] and _read_json(path_manifest["panel_manifest"]["path"])["structure_panels"]),
            "expected_rows": expected_rows,
        }
    )
    fingerprint_payload = {
        "study": _sha256(study_path),
        "sources": {key: value["sha256"] for key, value in contract.items() if isinstance(value, dict) and "sha256" in value},
        "builder_version": BUILDER_VERSION,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return contract, path_manifest, distribution_manifest, {"fingerprint": fingerprint, "paths": paths}


def _empty_state(length: int) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for name in STATE_BOOLEAN_FEATURES:
        output[name] = np.zeros(length, dtype=bool)
    for name in STATE_INTEGER_FEATURES:
        output[name] = np.full(length, -1, dtype=np.int32)
    for name in STATE_FLOAT_FEATURES:
        output[name] = np.full(length, np.nan, dtype=np.float64)
    output["up_retest_bar_count"] = np.zeros(length, dtype=np.int32)
    output["up_retest_episode_count"] = np.zeros(length, dtype=np.int32)
    return output


def _safe_log_amount(amount: np.ndarray) -> np.ndarray:
    values = np.asarray(amount, dtype=np.float64)
    result = np.full(len(values), np.nan, dtype=np.float64)
    valid = np.isfinite(values) & (values > 0.0)
    result[valid] = np.log(values[valid])
    return result


def derive_causal_retest_state(
    *,
    adj_high: np.ndarray,
    adj_low: np.ndarray,
    adj_close: np.ndarray,
    amount: np.ndarray,
    date_indices: np.ndarray,
    path_features: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Derive one prefix-only upward breakout state for every valid bar."""

    high = np.asarray(adj_high, dtype=np.float64)
    low = np.asarray(adj_low, dtype=np.float64)
    close = np.asarray(adj_close, dtype=np.float64)
    dates = np.asarray(date_indices, dtype=np.int64)
    length = len(close)
    if not all(len(values) == length for values in (high, low, amount, dates)):
        raise ValueError("causal_retest_state_array_length")
    output = _empty_state(length)
    if length == 0:
        return output
    if bool((high <= 0.0).any()) or bool((low <= 0.0).any()) or bool((close <= 0.0).any()):
        raise ValueError("causal_retest_state_nonpositive_price")
    log_high = np.log(high)
    log_low = np.log(low)
    log_close = np.log(close)
    log_amount = _safe_log_amount(np.asarray(amount, dtype=np.float64))
    event = np.asarray(path_features["chan_breakout_event"], dtype=np.int8)
    active = np.asarray(path_features["chan_breakout_active"], dtype=np.int8)
    retest = np.asarray(path_features["chan_retest_event"], dtype=np.int8)
    reentry = np.asarray(path_features["chan_reentry_event"], dtype=np.int8)
    scale = np.asarray(path_features["causal_scale_unit"], dtype=np.float64)
    upper_distance = np.asarray(
        path_features["chan_center_distance_upper_units"], dtype=np.float64
    )
    center_width = np.asarray(path_features["chan_center_width_units"], dtype=np.float64)
    center_age = np.asarray(path_features["chan_center_age"], dtype=np.int32)

    state: dict[str, Any] | None = None
    for pos in range(length):
        if event[pos] == 1:
            if not (
                math.isfinite(scale[pos])
                and scale[pos] > 0.0
                and math.isfinite(upper_distance[pos])
            ):
                raise ValueError(f"causal_retest_state_breakout_geometry:{pos}")
            locked_scale = float(scale[pos])
            boundary = float(log_close[pos] - upper_distance[pos] * locked_scale)
            pre_start = max(0, pos - 5)
            pre_amount = log_amount[pre_start:pos]
            finite_pre = pre_amount[np.isfinite(pre_amount)]
            state = {
                "breakout_pos": pos,
                "breakout_date": int(dates[pos]),
                "boundary": boundary,
                "scale": locked_scale,
                "breakout_close": float(log_close[pos]),
                "breakout_amount": float(log_amount[pos]),
                "pre_amount_mean": (
                    float(finite_pre.mean()) if len(finite_pre) else math.nan
                ),
                "center_width": float(center_width[pos]),
                "center_age": int(center_age[pos]),
                "peak": float(log_high[pos]),
                "trough": float(log_low[pos]),
                "amount_peak": float(log_amount[pos]),
                "variation": 0.0,
                "touch_bars": 0,
                "touch_episodes": 0,
                "first_touch_pos": None,
                "last_touch_pos": None,
                "previous_touch": False,
            }
        elif state is not None and active[pos] != 1 and reentry[pos] != 1:
            state = None

        if state is None:
            continue

        if pos > int(state["breakout_pos"]):
            state["variation"] += abs(float(log_close[pos] - log_close[pos - 1]))
        state["peak"] = max(float(state["peak"]), float(log_high[pos]))
        state["trough"] = min(float(state["trough"]), float(log_low[pos]))
        if math.isfinite(log_amount[pos]):
            if not math.isfinite(float(state["amount_peak"])):
                state["amount_peak"] = float(log_amount[pos])
            else:
                state["amount_peak"] = max(
                    float(state["amount_peak"]), float(log_amount[pos])
                )

        current_touch = bool(retest[pos] == 1)
        if current_touch:
            state["touch_bars"] += 1
            if not bool(state["previous_touch"]):
                state["touch_episodes"] += 1
            if state["first_touch_pos"] is None:
                state["first_touch_pos"] = pos
            state["last_touch_pos"] = pos
        state["previous_touch"] = current_touch

        unit = float(state["scale"])
        boundary = float(state["boundary"])
        breakout_close = float(state["breakout_close"])
        peak = float(state["peak"])
        trough = float(state["trough"])
        peak_expansion = (peak - boundary) / unit
        close_distance = (float(log_close[pos]) - boundary) / unit
        variation = float(state["variation"])
        net_change = float(log_close[pos] - breakout_close)

        output["up_breakout_event"][pos] = event[pos] == 1
        output["up_breakout_active"][pos] = active[pos] == 1
        output["up_retest_event"][pos] = current_touch
        output["up_retest_history"][pos] = int(state["touch_bars"]) > 0
        output["up_repeated_retest"][pos] = int(state["touch_episodes"]) >= 2
        output["up_intraday_boundary_penetration"][pos] = log_low[pos] < boundary
        output["up_reentry_event"][pos] = reentry[pos] == 1
        output["sessions_since_up_breakout"][pos] = pos - int(state["breakout_pos"])
        output["market_days_since_up_breakout"][pos] = int(dates[pos]) - int(state["breakout_date"])
        output["up_retest_bar_count"][pos] = int(state["touch_bars"])
        output["up_retest_episode_count"][pos] = int(state["touch_episodes"])
        output["sessions_since_first_up_retest"][pos] = (
            pos - int(state["first_touch_pos"])
            if state["first_touch_pos"] is not None
            else -1
        )
        output["sessions_since_last_up_retest"][pos] = (
            pos - int(state["last_touch_pos"])
            if state["last_touch_pos"] is not None
            else -1
        )
        output["center_age_at_breakout"][pos] = int(state["center_age"])
        output["locked_breakout_scale_unit"][pos] = unit
        output["center_width_units_at_breakout"][pos] = float(state["center_width"])
        output["breakout_strength_units"][pos] = (breakout_close - boundary) / unit
        output["boundary_close_distance_units"][pos] = close_distance
        output["boundary_low_distance_units"][pos] = (float(log_low[pos]) - boundary) / unit
        output["boundary_high_distance_units"][pos] = (float(log_high[pos]) - boundary) / unit
        output["peak_expansion_units"][pos] = peak_expansion
        output["trough_distance_units"][pos] = (trough - boundary) / unit
        output["drawdown_from_peak_units"][pos] = (peak - float(log_close[pos])) / unit
        output["drawdown_fraction_of_peak_expansion"][pos] = (
            (peak - float(log_close[pos])) / max(peak - boundary, np.finfo(float).eps)
        )
        output["close_change_from_breakout_units"][pos] = net_change / unit
        output["mfe_from_breakout_units"][pos] = (peak - breakout_close) / unit
        output["mae_from_breakout_units"][pos] = (breakout_close - trough) / unit
        output["path_efficiency_since_breakout"][pos] = (
            net_change / variation if variation > 0.0 else 0.0
        )
        if math.isfinite(log_amount[pos]):
            if math.isfinite(float(state["breakout_amount"])):
                output["amount_log_ratio_to_breakout"][pos] = (
                    float(log_amount[pos]) - float(state["breakout_amount"])
                )
            if math.isfinite(float(state["pre_amount_mean"])):
                output["amount_log_ratio_to_prebreakout_mean5"][pos] = (
                    float(log_amount[pos]) - float(state["pre_amount_mean"])
                )
            if math.isfinite(float(state["amount_peak"])):
                output["amount_log_ratio_to_postbreakout_peak"][pos] = (
                    float(log_amount[pos]) - float(state["amount_peak"])
                )
            if pos > 0 and math.isfinite(log_amount[pos - 1]):
                output["amount_log_ratio_to_previous_bar"][pos] = (
                    float(log_amount[pos]) - float(log_amount[pos - 1])
                )

        if reentry[pos] == 1 or active[pos] != 1:
            state = None
    return output


def _panel_schema() -> pa.Schema:
    fields: list[pa.Field] = [
        pa.field("input_row_idx", pa.int64()),
        pa.field("trade_date", pa.string()),
        pa.field("date_idx", pa.int32()),
        pa.field("symbol_idx", pa.int32()),
    ]
    for name in PANEL_FEATURE_COLUMNS:
        if name in STATE_BOOLEAN_FEATURES:
            dtype = pa.bool_()
        elif name in STATE_INTEGER_FEATURES:
            dtype = pa.int32()
        elif name in structure.DISCRETE_FEATURES:
            dtype = pa.int8()
        else:
            dtype = pa.float32()
        fields.append(pa.field(name, dtype))
    return pa.schema(fields, metadata={b"schema": PANEL_SCHEMA_VERSION.encode("ascii")})


def _prepared_panel_valid(
    manifest: Mapping[str, Any], *, fingerprint: str, expected_rows: int
) -> bool:
    return bool(
        manifest.get("schema") == PANEL_SCHEMA_VERSION
        and manifest.get("status") == "prepared"
        and str(manifest.get("experiment_fingerprint")) == fingerprint
        and int(manifest.get("rows", -1)) == expected_rows
        and len(manifest.get("state_panels", [])) == 14
        and all(Path(str(item["path"])).is_file() for item in manifest.get("state_panels", []))
    )


def prepare_retest_panels(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    study_file = _resolve(study_path)
    root = _resolve(output_root)
    study = load_study(study_file)
    contract, _path_manifest, _distribution_manifest, source_meta = _source_contract(
        study_file, study
    )
    fingerprint = str(source_meta["fingerprint"])
    expected_rows = int(contract["expected_rows"])
    manifest_path = root / "panel_manifest.json"
    if manifest_path.is_file() and not force:
        current = _read_json(manifest_path)
        if _prepared_panel_valid(
            current, fingerprint=fingerprint, expected_rows=expected_rows
        ):
            return current
        if str(current.get("experiment_fingerprint", "")) not in {"", fingerprint}:
            raise ValueError("causal_retest_state_existing_panel_fingerprint")

    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "progress.json", {"status": "preparing_continuous_state_panel"})
    row_index_path = Path(str(contract["row_index"]["path"]))
    lookup, symbol_map, row_dates, row_symbol_indices = structure._load_row_lookup(
        row_index_path
    )
    source_study = structure.load_study(
        Path(str(contract["path_structure_study"]["path"]))
    )
    seen = np.zeros(expected_rows, dtype=bool)
    schema = _panel_schema()
    row_group_size = int(dict(study["resources"])["parquet_row_group_size"])
    writers = {
        year: structure._YearPanelWriter(
            root / f"state_panels/year={year}/part-0000.parquet",
            schema,
            row_group_size,
        )
        for year in range(2012, 2026)
    }
    counters = {
        "up_breakout_events": 0,
        "up_retest_events": 0,
        "up_reentry_events": 0,
        "up_breakout_active_rows": 0,
    }
    processed_symbols = 0
    eligible_symbols = 0

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
        valid = (
            frame["bar_valid"].astype("boolean").fillna(False).to_numpy(bool)
            & np.isfinite(frame["adj_high"].to_numpy(float))
            & np.isfinite(frame["adj_low"].to_numpy(float))
            & np.isfinite(frame["adj_close"].to_numpy(float))
            & frame["adj_high"].to_numpy(float).__gt__(0.0)
            & frame["adj_low"].to_numpy(float).__gt__(0.0)
            & frame["adj_close"].to_numpy(float).__gt__(0.0)
        )
        work = frame.loc[valid].sort_values("date_idx", kind="mergesort")
        if work.empty:
            return
        date_indices = work["date_idx"].to_numpy(np.int64)
        inside = (date_indices >= 0) & (date_indices < lookup.shape[0])
        ordinals = np.full(len(work), -1, dtype=np.int64)
        ordinals[inside] = lookup[date_indices[inside], int(symbol_idx)]
        selected = np.flatnonzero(ordinals >= 0)
        if not len(selected):
            return
        parsed = structure.parse_causal_path(
            adj_high=work["adj_high"].to_numpy(float),
            adj_low=work["adj_low"].to_numpy(float),
            adj_close=work["adj_close"].to_numpy(float),
            amount=work["amount"].to_numpy(float),
            date_indices=date_indices,
            study=source_study,
        )
        state = derive_causal_retest_state(
            adj_high=work["adj_high"].to_numpy(float),
            adj_low=work["adj_low"].to_numpy(float),
            adj_close=work["adj_close"].to_numpy(float),
            amount=work["amount"].to_numpy(float),
            date_indices=date_indices,
            path_features=parsed,
        )
        selected_ordinals = ordinals[selected]
        if bool(seen[selected_ordinals].any()):
            raise ValueError("causal_retest_state_duplicate_emitted_row")
        seen[selected_ordinals] = True
        selected_dates = work["trade_date"].astype(str).to_numpy()[selected]
        selected_years = np.asarray([int(value[:4]) for value in selected_dates])
        counters["up_breakout_events"] += int(state["up_breakout_event"][selected].sum())
        counters["up_retest_events"] += int(state["up_retest_event"][selected].sum())
        counters["up_reentry_events"] += int(state["up_reentry_event"][selected].sum())
        counters["up_breakout_active_rows"] += int(
            state["up_breakout_active"][selected].sum()
        )
        for year in np.unique(selected_years):
            positions = selected[selected_years == year]
            columns: dict[str, Any] = {
                "input_row_idx": ordinals[positions],
                "trade_date": work["trade_date"].astype(str).to_numpy()[positions],
                "date_idx": date_indices[positions].astype(np.int32),
                "symbol_idx": np.full(len(positions), int(symbol_idx), dtype=np.int32),
            }
            for name in sorted(STATE_BOOLEAN_FEATURES | STATE_INTEGER_FEATURES | STATE_FLOAT_FEATURES):
                values = state[name][positions]
                columns[name] = (
                    values.astype(np.float32)
                    if name in STATE_FLOAT_FEATURES
                    else values
                )
            for name in CONTEXT_FEATURES:
                values = parsed[name][positions]
                columns[name] = (
                    values.astype(np.float32)
                    if name in structure.FLOAT_FEATURES
                    else values
                )
            writers[int(year)].append(columns)

    dense_path = Path(str(contract["dense_base"]["path"]))
    parquet = pq.ParquetFile(dense_path)
    resources = dict(study["resources"])
    pending: pd.DataFrame | None = None
    dense_rows = 0
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

    missing = np.flatnonzero(~seen)
    if len(missing):
        sample = [
            {
                "input_row_idx": int(pos),
                "trade_date": str(row_dates[pos]),
                "symbol_idx": int(row_symbol_indices[pos]),
            }
            for pos in missing[:20]
        ]
        raise ValueError(f"causal_retest_state_missing_pool_rows:{sample}")
    records = []
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
        "study": atlas._file_record(study_file),
        "source_contract": contract,
        "feature_columns": list(PANEL_FEATURE_COLUMNS),
        "state_panels": records,
        "rows": int(seen.sum()),
        "dates": int(np.unique(row_dates).size),
        "symbols": len(symbol_map),
        "processed_dense_symbols": int(processed_symbols),
        "eligible_symbols": int(eligible_symbols),
        "dense_rows": int(dense_rows),
        "duplicate_rows": 0,
        "missing_rows": 0,
        "forbidden_2026_rows": 0,
        "future_confirmed_structure_written_back": False,
        **counters,
    }
    _write_json(manifest_path, manifest)
    _write_json(root / "progress.json", {"status": "continuous_state_panel_prepared"})
    return manifest


def _records_by_year(records: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for raw in records:
        record = dict(raw)
        year = int(record["year"])
        if year in result:
            raise ValueError("causal_retest_state_duplicate_year_partition")
        path = Path(str(record["path"]))
        if not path.is_file() or _sha256(path) != str(record["sha256"]):
            raise ValueError(f"causal_retest_state_year_partition_hash:{year}")
        result[year] = record
    return result


@dataclass(frozen=True)
class RetestYear:
    year: int
    frame: pd.DataFrame


class RetestStateStore:
    def __init__(
        self,
        *,
        records: Sequence[Mapping[str, Any]],
        cache_years: int,
    ) -> None:
        self.records = _records_by_year(records)
        self.cache_years = max(int(cache_years), 1)
        self.cache: dict[int, RetestYear] = {}
        self.order: list[int] = []

    def load(self, year: int, coordinate_year: baseline.CoordinateYear) -> RetestYear:
        requested = int(year)
        cached = self.cache.get(requested)
        if cached is not None:
            if requested in self.order:
                self.order.remove(requested)
            self.order.append(requested)
            return cached
        record = self.records[requested]
        frame = pd.read_parquet(record["path"])
        if len(frame) != int(record["rows"]):
            raise ValueError("causal_retest_state_panel_partition_rows")
        frame = frame.sort_values("input_row_idx", kind="mergesort").reset_index(drop=True)
        coordinate_ids = coordinate_year.frame["input_row_idx"].to_numpy(np.int64)
        panel_ids = frame["input_row_idx"].to_numpy(np.int64)
        if not np.array_equal(panel_ids, coordinate_ids):
            raise ValueError(f"causal_retest_state_panel_coordinate_alignment:{year}")
        result = RetestYear(year=requested, frame=frame)
        self.cache[requested] = result
        self.order.append(requested)
        while len(self.order) > self.cache_years:
            removed = self.order.pop(0)
            self.cache.pop(removed, None)
        return result

    def clear(self) -> None:
        self.cache.clear()
        self.order.clear()
        gc.collect()


@dataclass(frozen=True)
class ScopeData:
    scope: str
    feature_names: tuple[str, ...]
    input_row_idx: np.ndarray
    trade_date: np.ndarray
    date_idx: np.ndarray
    symbol_idx: np.ndarray
    features: np.ndarray
    moments: np.ndarray


def _target_moments(target: np.ndarray) -> np.ndarray:
    values = np.asarray(target, dtype=np.float64)
    return np.column_stack(
        [
            values > 0.0,
            np.maximum(values, 0.0),
            np.maximum(-values, 0.0),
            values,
        ]
    ).astype(np.float64, copy=False)


def _date_equal_weights(date_idx: np.ndarray) -> np.ndarray:
    dates = np.asarray(date_idx, dtype=np.int64)
    if len(dates) == 0:
        return np.empty(0, dtype=np.float64)
    _, inverse, counts = np.unique(dates, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse].astype(np.float64)


def _model_feature_names(study: Mapping[str, Any]) -> tuple[str, ...]:
    features = dict(study["features"])
    names = tuple(
        [str(value) for value in features["base_coordinates"]]
        + [str(value) for value in features["continuous_state_coordinates"]]
    )
    if len(names) != len(set(names)):
        raise ValueError("causal_retest_state_duplicate_model_feature")
    return names


def _feature_matrix(
    *,
    coordinate_values: np.ndarray,
    coordinate_names: Sequence[str],
    panel_frame: pd.DataFrame,
    positions: np.ndarray,
    study: Mapping[str, Any],
) -> np.ndarray:
    features = dict(study["features"])
    base_names = [str(value) for value in features["base_coordinates"]]
    state_names = [str(value) for value in features["continuous_state_coordinates"]]
    name_to_index = {str(name): index for index, name in enumerate(coordinate_names)}
    missing = sorted(set(base_names) - set(name_to_index))
    if missing:
        raise ValueError(f"causal_retest_state_missing_base_coordinates:{missing}")
    base_indices = np.asarray([name_to_index[name] for name in base_names], dtype=np.int64)
    base_values = np.asarray(coordinate_values, dtype=np.float64)[:, base_indices]
    state_values = panel_frame.iloc[np.asarray(positions, dtype=np.int64)][
        state_names
    ].to_numpy(np.float64, copy=True)
    return np.column_stack([base_values, state_values]).astype(np.float64, copy=False)


def _scope_masks(
    *,
    panel_frame: pd.DataFrame,
    positions: np.ndarray,
    coordinate_values: np.ndarray,
    coordinate_names: Sequence[str],
    activity_rank: np.ndarray,
    threshold: float,
) -> dict[str, np.ndarray]:
    selected_panel = panel_frame.iloc[np.asarray(positions, dtype=np.int64)]
    names = {str(name): index for index, name in enumerate(coordinate_names)}
    correlation = np.asarray(coordinate_values, dtype=np.float64)[
        :, names["price_amount_correlation_5d_rank"]
    ]
    late_return = np.asarray(coordinate_values, dtype=np.float64)[
        :, names["minute_last_30m_return_rank"]
    ]
    active = np.isfinite(activity_rank) & (np.asarray(activity_rank) >= float(threshold))
    return {
        "up_retest_event": active
        & selected_panel["up_retest_event"].astype(bool).to_numpy(),
        "up_price_amount_confirmed_breakout": active
        & selected_panel["up_breakout_event"].astype(bool).to_numpy()
        & np.isfinite(correlation)
        & np.isfinite(late_return)
        & (correlation >= 0.5)
        & (late_return < 0.5),
    }


def _training_scope_data(
    *,
    experience: baseline.AlignedExperience,
    coordinate_year: baseline.CoordinateYear,
    panel_year: RetestYear,
    coordinate_names: Sequence[str],
    activity_rank: np.ndarray,
    study: Mapping[str, Any],
) -> dict[str, ScopeData]:
    positions = np.asarray(experience.coordinate_positions, dtype=np.int64)
    activity = np.asarray(activity_rank, dtype=np.float64)[positions]
    matrix = _feature_matrix(
        coordinate_values=experience.coordinates,
        coordinate_names=coordinate_names,
        panel_frame=panel_year.frame,
        positions=positions,
        study=study,
    )
    masks = _scope_masks(
        panel_frame=panel_year.frame,
        positions=positions,
        coordinate_values=experience.coordinates,
        coordinate_names=coordinate_names,
        activity_rank=activity,
        threshold=float(dict(study["evaluation"])["activity_threshold"]),
    )
    input_ids = coordinate_year.frame.iloc[positions]["input_row_idx"].to_numpy(np.int64)
    moments = _target_moments(experience.target)
    feature_names = _model_feature_names(study)
    result: dict[str, ScopeData] = {}
    for scope, mask in masks.items():
        result[scope] = ScopeData(
            scope=scope,
            feature_names=feature_names,
            input_row_idx=input_ids[mask].copy(),
            trade_date=np.asarray(experience.trade_date)[mask].copy(),
            date_idx=np.asarray(experience.date_idx)[mask].copy(),
            symbol_idx=np.asarray(experience.symbol_idx)[mask].copy(),
            features=matrix[mask].copy(),
            moments=moments[mask].copy(),
        )
    return result


def _combine_scope_data(parts: Sequence[ScopeData], *, scope: str) -> ScopeData:
    valid = [part for part in parts if len(part.features)]
    if not valid:
        return ScopeData(
            scope=scope,
            feature_names=(),
            input_row_idx=np.empty(0, dtype=np.int64),
            trade_date=np.empty(0, dtype=object),
            date_idx=np.empty(0, dtype=np.int32),
            symbol_idx=np.empty(0, dtype=np.int32),
            features=np.empty((0, 0), dtype=np.float64),
            moments=np.empty((0, len(MOMENT_NAMES)), dtype=np.float64),
        )
    names = valid[0].feature_names
    if any(part.feature_names != names for part in valid):
        raise ValueError("causal_retest_state_feature_contract_mismatch")
    order = np.argsort(
        np.concatenate([part.input_row_idx for part in valid]), kind="mergesort"
    )
    return ScopeData(
        scope=scope,
        feature_names=names,
        input_row_idx=np.concatenate([part.input_row_idx for part in valid])[order],
        trade_date=np.concatenate([part.trade_date for part in valid])[order],
        date_idx=np.concatenate([part.date_idx for part in valid])[order],
        symbol_idx=np.concatenate([part.symbol_idx for part in valid])[order],
        features=np.concatenate([part.features for part in valid], axis=0)[order],
        moments=np.concatenate([part.moments for part in valid], axis=0)[order],
    )


class EmpiricalMarginalRank:
    def __init__(self) -> None:
        self.sorted_values: list[np.ndarray] = []

    def fit(self, values: np.ndarray) -> EmpiricalMarginalRank:
        source = np.asarray(values, dtype=np.float64)
        self.sorted_values = []
        for column in range(source.shape[1]):
            finite = source[:, column][np.isfinite(source[:, column])]
            self.sorted_values.append(np.sort(finite, kind="mergesort"))
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        source = np.asarray(values, dtype=np.float64)
        if source.shape[1] != len(self.sorted_values):
            raise ValueError("causal_retest_state_rank_dimension")
        result = np.full(source.shape, 0.5, dtype=np.float64)
        for column, history in enumerate(self.sorted_values):
            finite = np.isfinite(source[:, column])
            if not len(history) or not finite.any():
                continue
            current = source[finite, column]
            left = np.searchsorted(history, current, side="left")
            right = np.searchsorted(history, current, side="right")
            result[finite, column] = (left + right) / (2.0 * len(history))
        return np.clip(result, 0.0, 1.0)

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        self.fit(values)
        return self.transform(values)


@dataclass
class FittedScopeModels:
    ranker: EmpiricalMarginalRank
    spline_transformer: SplineTransformer
    spline_model: Ridge
    neighbor_model: NearestNeighbors
    ranked_training: np.ndarray
    training_moments: np.ndarray
    training_weights: np.ndarray
    training_dates: np.ndarray
    prior: np.ndarray
    neighbor_count: int
    effective_dates: int
    spline_alpha: float


def _project_moments(raw: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(raw, dtype=np.float64)
    probability = np.clip(values[:, 0], 1.0e-6, 1.0 - 1.0e-6)
    upside = np.maximum(values[:, 1], 0.0)
    downside = np.maximum(values[:, 2], 0.0)
    direct = values[:, 3]
    expected = upside - downside
    return {
        "positive_probability": probability,
        "upside_component": upside,
        "downside_component": downside,
        "direct_expected_value": direct,
        "expected_value": expected,
        "reconstruction_gap": expected - direct,
    }


def fit_scope_models(
    data: ScopeData,
    *,
    study: Mapping[str, Any],
    threads: int,
) -> FittedScopeModels | None:
    minimum_rows = int(dict(study["evaluation"])["minimum_scope_training_rows"])
    if len(data.features) < minimum_rows:
        return None
    ranker = EmpiricalMarginalRank()
    ranked = ranker.fit_transform(data.features)
    estimators = dict(study["estimators"])
    knots = int(estimators["spline_knots"])
    degree = int(estimators["spline_degree"])
    knot_values = np.tile(
        np.linspace(0.0, 1.0, knots, dtype=np.float64)[:, None],
        (1, ranked.shape[1]),
    )
    transformer = SplineTransformer(
        n_knots=knots,
        degree=degree,
        knots=knot_values,
        extrapolation="constant",
        include_bias=False,
        order="C",
    )
    basis = np.asarray(transformer.fit_transform(ranked), dtype=np.float64)
    weights = _date_equal_weights(data.date_idx)
    effective_dates = int(np.unique(data.date_idx).size)
    alpha = float(estimators["spline_ridge_penalty_per_effective_date"]) * max(
        effective_dates, 1
    )
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(basis, data.moments, sample_weight=weights)
    neighbor_count = min(
        len(ranked),
        max(32, min(256, math.ceil(math.sqrt(len(ranked))))),
    )
    neighbors = NearestNeighbors(
        n_neighbors=neighbor_count,
        metric="euclidean",
        algorithm="auto",
        n_jobs=max(int(threads), 1),
    )
    neighbors.fit(ranked)
    prior = np.average(data.moments, axis=0, weights=weights)
    return FittedScopeModels(
        ranker=ranker,
        spline_transformer=transformer,
        spline_model=ridge,
        neighbor_model=neighbors,
        ranked_training=ranked,
        training_moments=data.moments,
        training_weights=weights,
        training_dates=np.asarray(data.date_idx, dtype=np.int64),
        prior=np.asarray(prior, dtype=np.float64),
        neighbor_count=neighbor_count,
        effective_dates=effective_dates,
        spline_alpha=alpha,
    )


def predict_scope_models(
    fitted: FittedScopeModels | None,
    values: np.ndarray,
    *,
    study: Mapping[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    rows = len(values)
    if fitted is None:
        missing = np.full((rows, len(MOMENT_NAMES)), np.nan, dtype=np.float64)
        projected = _project_moments(missing)
        return {
            "causal_rank_spline": projected,
            "historical_neighbor": projected,
        }
    ranked = fitted.ranker.transform(values)
    basis = np.asarray(fitted.spline_transformer.transform(ranked), dtype=np.float64)
    spline_raw = np.asarray(fitted.spline_model.predict(basis), dtype=np.float64)
    distances, indices = fitted.neighbor_model.kneighbors(
        ranked, return_distance=True
    )
    floor = float(dict(study["estimators"])["neighbor_distance_floor"])
    prior_dates = float(
        dict(study["estimators"])["neighbor_prior_date_equivalents"]
    )
    neighbor_raw = np.empty((rows, len(MOMENT_NAMES)), dtype=np.float64)
    for row in range(rows):
        selected = indices[row]
        weights = (
            fitted.training_weights[selected]
            / np.maximum(distances[row] + floor, np.finfo(float).eps)
        )
        if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
            local = fitted.prior
        else:
            local = np.average(
                fitted.training_moments[selected], axis=0, weights=weights
            )
        neighbor_dates = int(np.unique(fitted.training_dates[selected]).size)
        shrink = neighbor_dates / (neighbor_dates + prior_dates)
        neighbor_raw[row] = shrink * local + (1.0 - shrink) * fitted.prior
    return {
        "causal_rank_spline": _project_moments(spline_raw),
        "historical_neighbor": _project_moments(neighbor_raw),
    }


def _prediction_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for raw in records:
        record = dict(raw)
        key = (str(record["cost_scenario"]), int(record["oos_year"]))
        if key in result:
            raise ValueError("causal_retest_state_duplicate_prediction_partition")
        result[key] = record
    return result


def _evaluation_scope_frames(
    *,
    prediction_record: Mapping[str, Any],
    coordinate_year: baseline.CoordinateYear,
    panel_year: RetestYear,
    coordinate_names: Sequence[str],
    study: Mapping[str, Any],
) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    columns = [
        "cost_scenario",
        "oos_year",
        "training_cutoff_year",
        "maximum_training_label_as_of_year",
        "maximum_training_signal_year",
        "evaluation_as_of_year",
        "input_row_idx",
        "trade_date",
        "date_idx",
        "symbol_idx",
        "activity_rank",
        "risk_match_cell_code",
        "actual_buy_advantage_vs_cash",
        "actual_positive",
        "actual_upside_component",
        "actual_downside_component",
        "evaluation_sessions_after_resolution",
        "predicted_top20_additive_positive_probability",
        "predicted_top20_additive_upside_component",
        "predicted_top20_additive_downside_component",
        "predicted_top20_additive_direct_expected_value",
        "predicted_top20_additive_expected_value",
        "predicted_top20_additive_reconstruction_gap",
    ]
    frame = pd.read_parquet(prediction_record["path"], columns=columns)
    if len(frame) != int(prediction_record["rows"]):
        raise ValueError("causal_retest_state_prediction_rows")
    coordinate_ids = coordinate_year.frame["input_row_idx"].to_numpy(np.int64)
    wanted = frame["input_row_idx"].to_numpy(np.int64)
    positions = np.searchsorted(coordinate_ids, wanted)
    if bool((positions >= len(coordinate_ids)).any()) or not np.array_equal(
        coordinate_ids[positions], wanted
    ):
        raise ValueError("causal_retest_state_prediction_coordinate_alignment")
    coordinate_values = coordinate_year.values[positions].astype(np.float64, copy=False)
    activity = frame["activity_rank"].to_numpy(np.float64)
    masks = _scope_masks(
        panel_frame=panel_year.frame,
        positions=positions,
        coordinate_values=coordinate_values,
        coordinate_names=coordinate_names,
        activity_rank=activity,
        threshold=float(dict(study["evaluation"])["activity_threshold"]),
    )
    matrix = _feature_matrix(
        coordinate_values=coordinate_values,
        coordinate_names=coordinate_names,
        panel_frame=panel_year.frame,
        positions=positions,
        study=study,
    )
    coordinate_index = {str(name): index for index, name in enumerate(coordinate_names)}
    diagnostic_names = [
        str(value) for value in dict(study["features"])["diagnostic_coordinates"]
    ]
    selected_panel = panel_year.frame.iloc[positions].reset_index(drop=True)
    result: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    for scope, mask in masks.items():
        output = frame.loc[mask].reset_index(drop=True).copy()
        output.insert(2, "scope", scope)
        for name in diagnostic_names:
            if name in coordinate_index:
                output[name] = coordinate_values[mask, coordinate_index[name]].astype(
                    np.float32
                )
            elif name in selected_panel.columns:
                output[name] = selected_panel.loc[mask, name].to_numpy()
            else:
                raise ValueError(f"causal_retest_state_diagnostic_coordinate:{name}")
        inherited_map = {
            "positive_probability": "predicted_top20_additive_positive_probability",
            "upside_component": "predicted_top20_additive_upside_component",
            "downside_component": "predicted_top20_additive_downside_component",
            "direct_expected_value": "predicted_top20_additive_direct_expected_value",
            "expected_value": "predicted_top20_additive_expected_value",
            "reconstruction_gap": "predicted_top20_additive_reconstruction_gap",
        }
        for moment, source_name in inherited_map.items():
            output[f"predicted_inherited_top20_additive_{moment}"] = output[
                source_name
            ].to_numpy(np.float64)
        output = output.drop(columns=list(inherited_map.values()))
        result[scope] = (output, matrix[mask].copy())
    return result


def _append_model_predictions(
    frame: pd.DataFrame,
    predictions: Mapping[str, Mapping[str, np.ndarray]],
) -> pd.DataFrame:
    output = frame.copy()
    for model, moments in predictions.items():
        for name, values in moments.items():
            output[f"predicted_{model}_{name}"] = np.asarray(
                values, dtype=np.float64
            ).astype(np.float32)
    return output


def _binary_auc(actual: np.ndarray, predicted: np.ndarray) -> float:
    y = np.asarray(actual, dtype=bool)
    p = np.asarray(predicted, dtype=np.float64)
    finite = np.isfinite(p)
    y = y[finite]
    p = p[finite]
    if len(y) < 2 or np.unique(y).size < 2:
        return math.nan
    return float(distribution._binary_auc(y.astype(np.int8), p))


def _safe_spearman(actual: np.ndarray, predicted: np.ndarray) -> float:
    y = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    finite = np.isfinite(y) & np.isfinite(p)
    if int(finite.sum()) < 2:
        return math.nan
    if np.unique(y[finite]).size < 2 or np.unique(p[finite]).size < 2:
        return math.nan
    return float(stats.spearmanr(y[finite], p[finite]).statistic)


def build_daily_model_evaluation(
    prediction_records: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in prediction_records:
        frame = pd.read_parquet(record["path"])
        for (trade_date, date_idx), group in frame.groupby(
            ["trade_date", "date_idx"], sort=True
        ):
            actual = group["actual_buy_advantage_vs_cash"].to_numpy(np.float64)
            actual_positive = group["actual_positive"].astype(bool).to_numpy()
            upside = group["actual_upside_component"].to_numpy(np.float64)
            downside = group["actual_downside_component"].to_numpy(np.float64)
            for model in MODEL_NAMES:
                expected = group[f"predicted_{model}_expected_value"].to_numpy(
                    np.float64
                )
                probability = group[
                    f"predicted_{model}_positive_probability"
                ].to_numpy(np.float64)
                predicted_upside = group[
                    f"predicted_{model}_upside_component"
                ].to_numpy(np.float64)
                predicted_downside = group[
                    f"predicted_{model}_downside_component"
                ].to_numpy(np.float64)
                finite = np.isfinite(expected)
                if not finite.any():
                    continue
                eligible_positions = np.flatnonzero(finite)
                ranked = np.lexsort(
                    (
                        group["symbol_idx"].to_numpy(np.int64)[eligible_positions],
                        -expected[eligible_positions],
                    )
                )
                top = int(eligible_positions[int(ranked[0])])
                selects = bool(expected[top] > 0.0)
                clipped_probability = np.clip(
                    probability[finite], 1.0e-6, 1.0 - 1.0e-6
                )
                binary = actual_positive[finite].astype(np.float64)
                rows.append(
                    {
                        "cost_scenario": str(record["cost_scenario"]),
                        "evaluation_year": int(record["oos_year"]),
                        "scope": str(record["scope"]),
                        "model": model,
                        "trade_date": str(trade_date),
                        "date_idx": int(date_idx),
                        "candidate_rows": len(group),
                        "selection_date": selects,
                        "policy_realized_action_value": float(actual[top])
                        if selects
                        else 0.0,
                        "policy_upside_component": float(upside[top])
                        if selects
                        else 0.0,
                        "policy_downside_component": float(downside[top])
                        if selects
                        else 0.0,
                        "top1_realized_action_value": float(actual[top]),
                        "top1_positive": bool(actual_positive[top]),
                        "top1_upside_component": float(upside[top]),
                        "top1_downside_component": float(downside[top]),
                        "top1_predicted_expected_value": float(expected[top]),
                        "daily_spearman": _safe_spearman(actual, expected),
                        "daily_binary_auc": _binary_auc(
                            actual_positive, probability
                        ),
                        "brier": float(
                            np.mean(np.square(clipped_probability - binary))
                        ),
                        "log_loss": float(
                            -np.mean(
                                binary * np.log(clipped_probability)
                                + (1.0 - binary) * np.log(1.0 - clipped_probability)
                            )
                        ),
                        "upside_mae": float(
                            np.mean(
                                np.abs(predicted_upside[finite] - upside[finite])
                            )
                        ),
                        "downside_mae": float(
                            np.mean(
                                np.abs(predicted_downside[finite] - downside[finite])
                            )
                        ),
                    }
                )
    if not rows:
        raise ValueError("causal_retest_state_no_model_evaluation")
    return pd.DataFrame(rows).sort_values(
        ["cost_scenario", "scope", "model", "date_idx"], kind="mergesort"
    )


def _annual_model_evaluation(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["cost_scenario", "evaluation_year", "scope", "model"]
    for values, group in daily.groupby(keys, sort=True):
        cost, year, scope, model = values
        selected = group[group["selection_date"].astype(bool)]
        rows.append(
            {
                "cost_scenario": str(cost),
                "evaluation_year": int(year),
                "scope": str(scope),
                "model": str(model),
                "dates": int(group["date_idx"].nunique()),
                "selection_dates": int(group["selection_date"].sum()),
                "policy_realized_action_value": float(
                    group["policy_realized_action_value"].mean()
                ),
                "selected_realized_action_value": float(
                    selected["policy_realized_action_value"].mean()
                )
                if len(selected)
                else math.nan,
                "top1_realized_action_value": float(
                    group["top1_realized_action_value"].mean()
                ),
                "top1_positive_fraction": float(group["top1_positive"].mean()),
                "top1_downside_component": float(
                    group["top1_downside_component"].mean()
                ),
                "mean_daily_spearman": float(group["daily_spearman"].mean()),
                "mean_daily_binary_auc": float(group["daily_binary_auc"].mean()),
                "brier": float(group["brier"].mean()),
                "log_loss": float(group["log_loss"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _normal_p_value(mean: float, standard_error: float) -> float:
    if not math.isfinite(mean) or not math.isfinite(standard_error) or standard_error <= 0.0:
        return math.nan
    return float(2.0 * stats.norm.sf(abs(mean / standard_error)))


def _aggregate_model_evaluation(
    daily: pd.DataFrame,
    annual: pd.DataFrame,
    *,
    hac_lag: int,
) -> pd.DataFrame:
    baseline_rows = daily[
        daily["model"].eq("inherited_top20_additive")
    ][
        [
            "cost_scenario",
            "scope",
            "date_idx",
            "top1_realized_action_value",
            "top1_downside_component",
        ]
    ].rename(
        columns={
            "top1_realized_action_value": "baseline_top1_realized_action_value",
            "top1_downside_component": "baseline_top1_downside_component",
        }
    )
    enriched = daily.merge(
        baseline_rows,
        on=["cost_scenario", "scope", "date_idx"],
        how="left",
        validate="many_to_one",
    )
    enriched["top1_value_increment_vs_inherited"] = (
        enriched["top1_realized_action_value"]
        - enriched["baseline_top1_realized_action_value"]
    )
    enriched["top1_downside_increment_vs_inherited"] = (
        enriched["top1_downside_component"]
        - enriched["baseline_top1_downside_component"]
    )
    rows: list[dict[str, Any]] = []
    for (cost, scope, model), group in enriched.groupby(
        ["cost_scenario", "scope", "model"], sort=True
    ):
        group = group.sort_values("date_idx", kind="mergesort")
        policy = atlas._hac_mean(group["policy_realized_action_value"], lag=hac_lag)
        top1 = atlas._hac_mean(group["top1_realized_action_value"], lag=hac_lag)
        increment = atlas._hac_mean(
            group["top1_value_increment_vs_inherited"], lag=hac_lag
        )
        yearly = annual[
            annual["cost_scenario"].eq(cost)
            & annual["scope"].eq(scope)
            & annual["model"].eq(model)
        ]
        selected = group[group["selection_date"].astype(bool)]
        rows.append(
            {
                "cost_scenario": str(cost),
                "scope": str(scope),
                "model": str(model),
                "dates": int(group["date_idx"].nunique()),
                "selection_dates": int(group["selection_date"].sum()),
                "policy_action_value_mean": float(policy["mean"]),
                "policy_action_value_hac_se": float(policy["se"]),
                "policy_action_value_lcb_95": float(policy["lcb_95"]),
                "policy_action_value_ucb_95": float(policy["ucb_95"]),
                "selected_action_value_mean": float(
                    selected["policy_realized_action_value"].mean()
                )
                if len(selected)
                else math.nan,
                "top1_action_value_mean": float(top1["mean"]),
                "top1_action_value_hac_se": float(top1["se"]),
                "top1_action_value_lcb_95": float(top1["lcb_95"]),
                "top1_positive_fraction": float(group["top1_positive"].mean()),
                "top1_value_increment_vs_inherited": float(increment["mean"]),
                "top1_value_increment_hac_se": float(increment["se"]),
                "top1_value_increment_lcb_95": float(increment["lcb_95"]),
                "top1_downside_increment_vs_inherited": float(
                    group["top1_downside_increment_vs_inherited"].mean()
                ),
                "positive_policy_years": int(
                    yearly["policy_realized_action_value"].gt(0.0).sum()
                ),
                "years": len(yearly),
                "mean_daily_spearman": float(group["daily_spearman"].mean()),
                "mean_daily_binary_auc": float(group["daily_binary_auc"].mean()),
                "brier": float(group["brier"].mean()),
                "log_loss": float(group["log_loss"].mean()),
                "upside_mae": float(group["upside_mae"].mean()),
                "downside_mae": float(group["downside_mae"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _daily_coordinate_diagnostics(
    frame: pd.DataFrame,
    *,
    coordinate: str,
    record: Mapping[str, Any],
    low_cut: float,
    high_cut: float,
) -> pd.DataFrame:
    work = frame[
        [
            "trade_date",
            "date_idx",
            "risk_match_cell_code",
            "actual_buy_advantage_vs_cash",
            "actual_positive",
            "actual_upside_component",
            "actual_downside_component",
            coordinate,
        ]
    ].copy()
    work["coordinate_rank"] = work.groupby("date_idx", sort=False)[coordinate].rank(
        method="average", pct=True
    )
    rows: list[dict[str, Any]] = []
    for (trade_date, date_idx), day in work.groupby(
        ["trade_date", "date_idx"], sort=True
    ):
        winner_differences: list[float] = []
        value_differences: list[float] = []
        positive_differences: list[float] = []
        upside_differences: list[float] = []
        downside_differences: list[float] = []
        positive_with_failure = 0
        positive_rows = int(day["actual_positive"].astype(bool).sum())
        matched_cells = 0
        rank_cells = 0
        for _, cell in day.groupby("risk_match_cell_code", sort=False):
            positive = cell[cell["actual_positive"].astype(bool)]
            failure = cell[~cell["actual_positive"].astype(bool)]
            if len(positive) and len(failure):
                positive_coordinate = positive[coordinate].to_numpy(float)
                failure_coordinate = failure[coordinate].to_numpy(float)
                if np.isfinite(positive_coordinate).any() and np.isfinite(
                    failure_coordinate
                ).any():
                    winner_differences.append(
                        float(np.nanmean(positive_coordinate))
                        - float(np.nanmean(failure_coordinate))
                    )
                    matched_cells += 1
                positive_with_failure += len(positive)
            low = cell[cell["coordinate_rank"].le(low_cut)]
            high = cell[cell["coordinate_rank"].ge(high_cut)]
            if len(low) and len(high):
                value_differences.append(
                    float(high["actual_buy_advantage_vs_cash"].mean())
                    - float(low["actual_buy_advantage_vs_cash"].mean())
                )
                positive_differences.append(
                    float(high["actual_positive"].mean())
                    - float(low["actual_positive"].mean())
                )
                upside_differences.append(
                    float(high["actual_upside_component"].mean())
                    - float(low["actual_upside_component"].mean())
                )
                downside_differences.append(
                    float(high["actual_downside_component"].mean())
                    - float(low["actual_downside_component"].mean())
                )
                rank_cells += 1
        rows.append(
            {
                "cost_scenario": str(record["cost_scenario"]),
                "evaluation_year": int(record["oos_year"]),
                "scope": str(record["scope"]),
                "coordinate": coordinate,
                "trade_date": str(trade_date),
                "date_idx": int(date_idx),
                "rows": len(day),
                "positive_rows": positive_rows,
                "matched_failure_positive_rows": positive_with_failure,
                "winner_failure_cells": matched_cells,
                "winner_minus_failure_coordinate": float(
                    np.mean(winner_differences)
                )
                if winner_differences
                else math.nan,
                "rank_contrast_cells": rank_cells,
                "high_minus_low_action_value": float(np.mean(value_differences))
                if value_differences
                else math.nan,
                "high_minus_low_positive_fraction": float(
                    np.mean(positive_differences)
                )
                if positive_differences
                else math.nan,
                "high_minus_low_upside_component": float(
                    np.mean(upside_differences)
                )
                if upside_differences
                else math.nan,
                "high_minus_low_downside_component": float(
                    np.mean(downside_differences)
                )
                if downside_differences
                else math.nan,
            }
        )
    return pd.DataFrame(rows)


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


def build_coordinate_diagnostics(
    *,
    study: Mapping[str, Any],
    prediction_records: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    diagnostic_names = [
        str(value) for value in dict(study["features"])["diagnostic_coordinates"]
    ]
    low_cut, high_cut = [
        float(value) for value in dict(study["evaluation"])["diagnostic_rank_splits"]
    ]
    daily_parts: list[pd.DataFrame] = []
    for record in prediction_records:
        frame = pd.read_parquet(record["path"])
        for coordinate in diagnostic_names:
            daily_parts.append(
                _daily_coordinate_diagnostics(
                    frame,
                    coordinate=coordinate,
                    record=record,
                    low_cut=low_cut,
                    high_cut=high_cut,
                )
            )
    daily = pd.concat(daily_parts, ignore_index=True)
    annual_rows: list[dict[str, Any]] = []
    keys = ["cost_scenario", "evaluation_year", "scope", "coordinate"]
    for values, group in daily.groupby(keys, sort=True):
        cost, year, scope, coordinate = values
        annual_rows.append(
            {
                "cost_scenario": str(cost),
                "evaluation_year": int(year),
                "scope": str(scope),
                "coordinate": str(coordinate),
                "dates": int(group["date_idx"].nunique()),
                "rows": int(group["rows"].sum()),
                "winner_failure_dates": int(
                    group["winner_minus_failure_coordinate"].notna().sum()
                ),
                "winner_minus_failure_coordinate": float(
                    group["winner_minus_failure_coordinate"].mean()
                ),
                "rank_contrast_dates": int(
                    group["high_minus_low_action_value"].notna().sum()
                ),
                "high_minus_low_action_value": float(
                    group["high_minus_low_action_value"].mean()
                ),
                "high_minus_low_positive_fraction": float(
                    group["high_minus_low_positive_fraction"].mean()
                ),
                "high_minus_low_upside_component": float(
                    group["high_minus_low_upside_component"].mean()
                ),
                "high_minus_low_downside_component": float(
                    group["high_minus_low_downside_component"].mean()
                ),
            }
        )
    annual = pd.DataFrame(annual_rows)
    lag = int(dict(study["evaluation"])["hac_lag"])
    aggregate_rows: list[dict[str, Any]] = []
    for (cost, scope, coordinate), group in daily.groupby(
        ["cost_scenario", "scope", "coordinate"], sort=True
    ):
        ordered = group.sort_values("date_idx", kind="mergesort")
        winner = ordered[ordered["winner_minus_failure_coordinate"].notna()]
        contrast = ordered[ordered["high_minus_low_action_value"].notna()]
        winner_hac = atlas._hac_mean(
            winner["winner_minus_failure_coordinate"], lag=lag
        )
        value_hac = atlas._hac_mean(contrast["high_minus_low_action_value"], lag=lag)
        yearly = annual[
            annual["cost_scenario"].eq(cost)
            & annual["scope"].eq(scope)
            & annual["coordinate"].eq(coordinate)
        ]
        positive_total = float(ordered["positive_rows"].sum())
        aggregate_rows.append(
            {
                "cost_scenario": str(cost),
                "scope": str(scope),
                "coordinate": str(coordinate),
                "dates": int(ordered["date_idx"].nunique()),
                "rows": int(ordered["rows"].sum()),
                "winner_failure_dates": len(winner),
                "winner_minus_failure_coordinate": float(winner_hac["mean"]),
                "winner_minus_failure_hac_se": float(winner_hac["se"]),
                "winner_minus_failure_p": _normal_p_value(
                    float(winner_hac["mean"]), float(winner_hac["se"])
                ),
                "rank_contrast_dates": len(contrast),
                "high_minus_low_action_value": float(value_hac["mean"]),
                "high_minus_low_action_value_hac_se": float(value_hac["se"]),
                "high_minus_low_action_value_lcb_95": float(value_hac["lcb_95"]),
                "high_minus_low_action_value_p": _normal_p_value(
                    float(value_hac["mean"]), float(value_hac["se"])
                ),
                "high_minus_low_positive_fraction": float(
                    contrast["high_minus_low_positive_fraction"].mean()
                ),
                "high_minus_low_upside_component": float(
                    contrast["high_minus_low_upside_component"].mean()
                ),
                "high_minus_low_downside_component": float(
                    contrast["high_minus_low_downside_component"].mean()
                ),
                "positive_years_high_minus_low_value": int(
                    yearly["high_minus_low_action_value"].gt(0.0).sum()
                ),
                "years": len(yearly),
                "matched_failure_fraction_of_positive": float(
                    ordered["matched_failure_positive_rows"].sum() / positive_total
                )
                if positive_total > 0.0
                else math.nan,
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate["winner_minus_failure_bh_q"] = np.nan
    aggregate["high_minus_low_action_value_bh_q"] = np.nan
    for indices in aggregate.groupby(["cost_scenario", "scope"]).groups.values():
        positions = np.asarray(list(indices), dtype=np.int64)
        aggregate.loc[positions, "winner_minus_failure_bh_q"] = _benjamini_hochberg(
            aggregate.loc[positions, "winner_minus_failure_p"].to_numpy(float)
        )
        aggregate.loc[
            positions, "high_minus_low_action_value_bh_q"
        ] = _benjamini_hochberg(
            aggregate.loc[positions, "high_minus_low_action_value_p"].to_numpy(
                float
            )
        )
    return daily, annual, aggregate


def _promotion_gate(
    aggregate: pd.DataFrame,
    *,
    study: Mapping[str, Any],
) -> dict[str, Any]:
    costs = [str(value) for value in dict(study["evaluation"])["cost_scenarios"]]
    minimum_dates = int(dict(study["evaluation"])["minimum_selection_dates"])
    majority = len(dict(study["period"])["strict_oos_years"]) // 2 + 1
    decisions: list[dict[str, Any]] = []
    passed: list[str] = []
    for scope in SCOPE_NAMES:
        for model in ("causal_rank_spline", "historical_neighbor"):
            key = f"{scope}:{model}"
            rows = aggregate[
                aggregate["scope"].eq(scope) & aggregate["model"].eq(model)
            ].set_index("cost_scenario")
            cost_checks: dict[str, Any] = {}
            cost_pass: dict[str, bool] = {}
            for cost in costs:
                if cost not in rows.index:
                    cost_checks[cost] = {"missing": True}
                    cost_pass[cost] = False
                    continue
                row = rows.loc[cost]
                checks = {
                    "minimum_selection_dates": int(row["selection_dates"])
                    >= minimum_dates,
                    "positive_policy_lcb": float(row["policy_action_value_lcb_95"])
                    > 0.0,
                    "majority_positive_years": int(row["positive_policy_years"])
                    >= majority,
                    "nonworsening_top1_downside": float(
                        row["top1_downside_increment_vs_inherited"]
                    )
                    <= 0.0,
                }
                cost_checks[cost] = checks
                cost_pass[cost] = all(checks.values())
            did_pass = all(cost_pass.values())
            if did_pass:
                passed.append(key)
            decisions.append(
                {
                    "candidate": key,
                    "passed": did_pass,
                    "cost_pass": cost_pass,
                    "checks": cost_checks,
                }
            )
    return {
        "status": "passed" if passed else "failed",
        "passed_candidates": passed,
        "account_replay_allowed": bool(passed),
        "decisions": decisions,
    }


def _runtime_resources(study: Mapping[str, Any]) -> dict[str, int]:
    resources = dict(study["resources"])
    available_mb = int(psutil.virtual_memory().available / 1024**2)
    reserve_mb = int(resources["reserve_memory_mb"])
    usable_mb = max(available_mb - reserve_mb, 512)
    logical = int(psutil.cpu_count() or 1)
    threads = max(
        1,
        min(
            int(resources["maximum_threads"]),
            max(logical - 2, 1),
            max(usable_mb // 512, 1),
        ),
    )
    return {
        "available_memory_mb": available_mb,
        "reserve_memory_mb": reserve_mb,
        "usable_memory_mb": usable_mb,
        "logical_cpu_count": logical,
        "threads": threads,
    }


def run_study(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    study_file = _resolve(study_path)
    study = load_study(study_file)
    root = _resolve(output_root or study["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    runtime = _runtime_resources(study)
    minimum_available = int(runtime["available_memory_mb"])
    panel_manifest = prepare_retest_panels(
        study_path=study_file,
        output_root=root,
        force=False,
    )
    contract, _path_manifest, _distribution_manifest, source_meta = _source_contract(
        study_file, study
    )
    if str(panel_manifest["experiment_fingerprint"]) != str(source_meta["fingerprint"]):
        raise ValueError("causal_retest_state_panel_fingerprint")

    distribution_study = distribution.load_study(
        Path(str(contract["distribution_study"]["path"]))
    )
    (
        _distribution_contract,
        _prefix_manifest,
        _input_manifest,
        version_records,
        _feature_index,
    ) = distribution._source_contract(distribution_study)
    _, coordinate_names, _, _, _ = baseline._coordinate_metadata(
        distribution_study
    )
    coordinate_records = list(contract["coordinate_partitions"])
    state_records = list(panel_manifest["state_panels"])
    resources = dict(study["resources"])
    coordinates = baseline.CoordinateStore(
        records=coordinate_records,
        coordinate_names=coordinate_names,
        cache_years=int(resources["coordinate_cache_years"]),
    )
    states = RetestStateStore(
        records=state_records,
        cache_years=int(resources["coordinate_cache_years"]),
    )
    coordinate_index = {name: index for index, name in enumerate(coordinate_names)}
    gate = dict(distribution_study["activity_gate"])
    gate_indices = np.asarray(
        [coordinate_index[str(name)] for name in gate["coordinates"]], dtype=np.int16
    )
    activity_store = distribution.ActivityRankStore(
        coordinates=coordinates,
        coordinate_indices=gate_indices,
        minimum_finite=int(gate["minimum_finite_coordinates"]),
        cache_years=int(resources["coordinate_cache_years"]),
    )
    source_predictions = _prediction_records(contract["prediction_partitions"])
    maturity_sessions = int(dict(study["target"])["minimum_sessions_after_resolution"])
    row_group_size = int(resources["parquet_row_group_size"])
    training_cache: dict[str, dict[str, ScopeData]] = {}
    generated_predictions: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    costs = [str(value) for value in dict(study["evaluation"])["cost_scenarios"]]
    oos_years = [int(value) for value in dict(study["period"])["strict_oos_years"]]
    _write_json(root / "progress.json", {"status": "fitting_causal_oos_models"})

    def cached_training(record: Mapping[str, Any]) -> dict[str, ScopeData]:
        cache_key = str(record["sha256"])
        cached = training_cache.get(cache_key)
        if cached is not None:
            return cached
        signal_year = int(record["signal_year"])
        coordinate_year = coordinates.load(signal_year)
        panel_year = states.load(signal_year, coordinate_year)
        experience = baseline._read_aligned_experience(
            record=record,
            coordinates=coordinate_year,
            maturity_sessions=maturity_sessions,
        )
        activity = activity_store.load(signal_year)
        built = _training_scope_data(
            experience=experience,
            coordinate_year=coordinate_year,
            panel_year=panel_year,
            coordinate_names=coordinate_names,
            activity_rank=activity,
            study=study,
        )
        training_cache[cache_key] = built
        return built

    try:
        for cost in costs:
            for oos_year in oos_years:
                fold_started = time.perf_counter()
                available = int(psutil.virtual_memory().available / 1024**2)
                minimum_available = min(minimum_available, available)
                if available <= int(runtime["reserve_memory_mb"]):
                    raise MemoryError(f"causal_retest_state_memory_reserve:{available}")
                cutoff = oos_year - 1
                desired = baseline.select_latest_records(
                    version_records=version_records,
                    cost_scenario=cost,
                    cutoff_year=cutoff,
                    maximum_signal_year=cutoff,
                )
                if not desired:
                    raise ValueError("causal_retest_state_empty_training_versions")
                maximum_label_as_of = max(
                    int(record["as_of_year"]) for record in desired.values()
                )
                maximum_signal_year = max(int(value) for value in desired)
                if maximum_label_as_of > cutoff or maximum_signal_year >= oos_year:
                    raise ValueError("causal_retest_state_training_cutoff")
                training_by_scope: dict[str, ScopeData] = {}
                fitted_by_scope: dict[str, FittedScopeModels | None] = {}
                for scope in SCOPE_NAMES:
                    parts = [
                        cached_training(record)[scope]
                        for _, record in sorted(desired.items())
                    ]
                    combined = _combine_scope_data(parts, scope=scope)
                    training_by_scope[scope] = combined
                    fitted_by_scope[scope] = fit_scope_models(
                        combined,
                        study=study,
                        threads=int(runtime["threads"]),
                    )

                source_record = source_predictions[(cost, oos_year)]
                if int(source_record["training_cutoff_year"]) != cutoff:
                    raise ValueError("causal_retest_state_source_prediction_cutoff")
                coordinate_year = coordinates.load(oos_year)
                panel_year = states.load(oos_year, coordinate_year)
                evaluation = _evaluation_scope_frames(
                    prediction_record=source_record,
                    coordinate_year=coordinate_year,
                    panel_year=panel_year,
                    coordinate_names=coordinate_names,
                    study=study,
                )
                fold_scope_rows: dict[str, int] = {}
                for scope in SCOPE_NAMES:
                    frame, matrix = evaluation[scope]
                    fitted = fitted_by_scope[scope]
                    predictions = predict_scope_models(fitted, matrix, study=study)
                    frame = _append_model_predictions(frame, predictions)
                    training = training_by_scope[scope]
                    frame["model_training_rows"] = len(training.features)
                    frame["model_training_dates"] = int(
                        np.unique(training.date_idx).size
                    )
                    frame["model_training_cutoff_year"] = cutoff
                    frame["model_maximum_label_as_of_year"] = maximum_label_as_of
                    frame["model_maximum_signal_year"] = maximum_signal_year
                    frame["spline_alpha"] = (
                        float(fitted.spline_alpha) if fitted is not None else np.nan
                    )
                    frame["neighbor_count"] = (
                        int(fitted.neighbor_count) if fitted is not None else 0
                    )
                    path = (
                        root
                        / "predictions"
                        / f"cost={cost}"
                        / f"scope={scope}"
                        / f"oos_year={oos_year}"
                        / "part-0000.parquet"
                    )
                    record = _write_frame(
                        path,
                        frame,
                        row_group_size=row_group_size,
                    )
                    record.update(
                        {
                            "cost_scenario": cost,
                            "scope": scope,
                            "oos_year": oos_year,
                            "training_cutoff_year": cutoff,
                            "maximum_training_label_as_of_year": maximum_label_as_of,
                            "maximum_training_signal_year": maximum_signal_year,
                            "training_rows": len(training.features),
                            "training_dates": int(np.unique(training.date_idx).size),
                            "model_available": fitted is not None,
                        }
                    )
                    generated_predictions.append(record)
                    fold_scope_rows[scope] = len(frame)
                fold_rows.append(
                    {
                        "cost_scenario": cost,
                        "oos_year": oos_year,
                        "training_cutoff_year": cutoff,
                        "maximum_training_label_as_of_year": maximum_label_as_of,
                        "maximum_training_signal_year": maximum_signal_year,
                        "training_version_count": len(desired),
                        "retest_training_rows": len(
                            training_by_scope["up_retest_event"].features
                        ),
                        "breakout_training_rows": len(
                            training_by_scope[
                                "up_price_amount_confirmed_breakout"
                            ].features
                        ),
                        "retest_evaluation_rows": fold_scope_rows[
                            "up_retest_event"
                        ],
                        "breakout_evaluation_rows": fold_scope_rows[
                            "up_price_amount_confirmed_breakout"
                        ],
                        "elapsed_seconds": float(time.perf_counter() - fold_started),
                    }
                )
                print(
                    json.dumps(
                        {
                            "cost_scenario": cost,
                            "oos_year": oos_year,
                            "retest_train": fold_rows[-1]["retest_training_rows"],
                            "breakout_train": fold_rows[-1][
                                "breakout_training_rows"
                            ],
                            "retest_eval": fold_rows[-1]["retest_evaluation_rows"],
                            "breakout_eval": fold_rows[-1][
                                "breakout_evaluation_rows"
                            ],
                            "elapsed_seconds": fold_rows[-1]["elapsed_seconds"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                del evaluation, training_by_scope, fitted_by_scope
                gc.collect()
    finally:
        activity_store.clear()
        coordinates.clear()
        states.clear()

    fold_summary = pd.DataFrame(fold_rows).sort_values(
        ["cost_scenario", "oos_year"], kind="mergesort"
    )
    daily_model = build_daily_model_evaluation(generated_predictions)
    annual_model = _annual_model_evaluation(daily_model)
    aggregate_model = _aggregate_model_evaluation(
        daily_model,
        annual_model,
        hac_lag=int(dict(study["evaluation"])["hac_lag"]),
    )
    daily_coordinate, annual_coordinate, aggregate_coordinate = (
        build_coordinate_diagnostics(
            study=study,
            prediction_records=generated_predictions,
        )
    )
    gate_result = _promotion_gate(aggregate_model, study=study)
    outputs = {
        "fold_summary": _write_frame(
            root / "fold_summary.parquet",
            fold_summary,
            row_group_size=row_group_size,
        ),
        "daily_model_evaluation": _write_frame(
            root / "daily_model_evaluation.parquet",
            daily_model,
            row_group_size=row_group_size,
        ),
        "annual_model_evaluation": _write_frame(
            root / "annual_model_evaluation.parquet",
            annual_model,
            row_group_size=row_group_size,
        ),
        "aggregate_model_evaluation": _write_frame(
            root / "aggregate_model_evaluation.parquet",
            aggregate_model,
            row_group_size=row_group_size,
        ),
        "daily_coordinate_diagnostics": _write_frame(
            root / "daily_coordinate_diagnostics.parquet",
            daily_coordinate,
            row_group_size=row_group_size,
        ),
        "annual_coordinate_diagnostics": _write_frame(
            root / "annual_coordinate_diagnostics.parquet",
            annual_coordinate,
            row_group_size=row_group_size,
        ),
        "aggregate_coordinate_diagnostics": _write_frame(
            root / "aggregate_coordinate_diagnostics.parquet",
            aggregate_coordinate,
            row_group_size=row_group_size,
        ),
    }
    gate_path = root / "promotion_gate.json"
    _write_json(gate_path, gate_result)
    outputs["promotion_gate"] = atlas._file_record(gate_path)
    minimum_available = min(
        minimum_available, int(psutil.virtual_memory().available / 1024**2)
    )
    audit = {
        "panel_rows": int(panel_manifest["rows"]),
        "panel_partitions": len(panel_manifest["state_panels"]),
        "prediction_partitions": len(generated_predictions),
        "prediction_rows": int(
            sum(int(record["rows"]) for record in generated_predictions)
        ),
        "cost_scenarios": costs,
        "oos_years": oos_years,
        "scopes": list(SCOPE_NAMES),
        "models": list(MODEL_NAMES),
        "training_cutoff_violations": 0,
        "forbidden_2026_rows": 0,
        "final_oracle_row_labels_used": False,
        "fixed_holding_horizon_used": False,
        "binary_good_stock_label_used": False,
        "future_confirmed_structure_written_back": False,
        "account_replay_allowed": bool(gate_result["account_replay_allowed"]),
    }
    manifest = {
        "schema": MANIFEST_SCHEMA_VERSION,
        "status": "completed",
        "study_id": STUDY_ID,
        "builder_version": BUILDER_VERSION,
        "experiment_fingerprint": panel_manifest["experiment_fingerprint"],
        "study": atlas._file_record(study_file),
        "panel_manifest": atlas._file_record(root / "panel_manifest.json"),
        "source_contract": contract,
        "feature_columns": list(PANEL_FEATURE_COLUMNS),
        "model_feature_names": list(_model_feature_names(study)),
        "diagnostic_coordinates": list(
            dict(study["features"])["diagnostic_coordinates"]
        ),
        "predictions": generated_predictions,
        "outputs": outputs,
        "runtime": {
            **runtime,
            "elapsed_seconds": float(time.perf_counter() - started),
            "minimum_available_memory_mb_observed": minimum_available,
            "available_memory_mb_at_end": int(
                psutil.virtual_memory().available / 1024**2
            ),
            "adaptive_memory": bool(resources["adaptive_memory"]),
        },
        "audit": audit,
        "promotion_gate": gate_result,
        "training_performed": True,
        "prediction_performed": True,
        "portfolio_execution_performed": False,
        "profit_claim_allowed": False,
        "production_policy_selected": False,
        "deep_model_selected": False,
        "report_generation_performed": False,
    }
    _write_json(root / "manifest.json", manifest)
    _write_json(root / "progress.json", {"status": "completed"})
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the continuous causal breakout-retest state study."
    )
    parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force-panel", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.prepare_only:
        manifest = prepare_retest_panels(
            study_path=args.study,
            output_root=args.output_root or DEFAULT_OUTPUT_ROOT,
            force=bool(args.force_panel),
        )
    else:
        manifest = run_study(
            study_path=args.study,
            output_root=args.output_root,
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
