from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats

from daily_research.path_policy import seq100_path_label_learnability as base


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_short_horizon_target_reaudit_v1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_short_horizon_target_reaudit_v1.json"
)
DEFAULT_INPUT_ROOT = (
    WORKSPACE_ROOT / "tmp/seq100_short_horizon_target_reaudit/attempt_001"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_short_horizon_target_reaudit_v1"
)
OLD_CONTRACT_PATH = (
    WORKSPACE_ROOT
    / "daily_research/research_records/seq100/seq100_path_label_learnability_v1/contract.json"
)
OLD_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_path_label_learnability_v1"
)
HORIZON_ATLAS_SCRIPT = WORKSPACE_ROOT / "tmp/seq100_future_horizon_atlas.py"
INPUT_PREP_SCRIPT = WORKSPACE_ROOT / "tmp/seq100_learnability_inputs.py"
PRE2023_ATLAS_ROOT = WORKSPACE_ROOT / "tmp/seq100_future_path_atlas/attempt_001"

FOLD_YEARS = (2023, 2024, 2025)
SHORT_TARGETS = {1: ("g",), 3: ("g", "mfe", "pre_peak_mae", "state")}
SHORT_LABEL_COLUMNS = ("g_1", "g_3", "mfe_3", "pre_peak_mae_3")
SHORT_LABEL_INDEX = {name: index for index, name in enumerate(SHORT_LABEL_COLUMNS)}
RESULT_SCHEMA = "seq100_short_horizon_target_reaudit_task/v1"
GROUP_SCHEMA = "seq100_short_horizon_target_reaudit_group/v1"
PROGRESS_SCHEMA = "seq100_short_horizon_target_reaudit_progress/v1"

FLAG_OUTCOME_WITHIN_CUTOFF = np.uint16(1 << 0)
FLAG_ENTRY_FILLED = np.uint16(1 << 1)
FLAG_ANCHOR_VALID = np.uint16(1 << 2)
FLAG_PATH_COMPLETE = np.uint16(1 << 3)
FLAG_G_VALID = np.uint16(1 << 4)
FLAG_HAS_SELLABLE_D2_TO_D3 = np.uint16(1 << 5)
FLAG_MFE_VALID = np.uint16(1 << 6)
FLAG_STATE_ASSIGNED = np.uint16(1 << 7)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _emit(event: str, **payload: Any) -> None:
    print(
        json.dumps(
            {"event": event, "at": _now(), **payload},
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )


def _row_slices(count: int, batch_size: int) -> Iterable[slice]:
    for start in range(0, int(count), int(batch_size)):
        yield slice(start, min(start + int(batch_size), int(count)))


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


def _load_file_module(path: Path, name: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _file_record(
    path: Path,
    *,
    shape: Sequence[int] | None = None,
    dtype: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "size": int(path.stat().st_size),
        "sha256": _file_sha256(path),
    }
    if shape is not None:
        record["shape"] = [int(value) for value in shape]
    if dtype is not None:
        record["dtype"] = str(dtype)
    return record


def _contingency_metrics(left: np.ndarray, right: np.ndarray, size: int = 3) -> dict[str, Any]:
    left_values = np.asarray(left, dtype=np.int16)
    right_values = np.asarray(right, dtype=np.int16)
    valid = (
        (left_values >= 0)
        & (left_values < int(size))
        & (right_values >= 0)
        & (right_values < int(size))
    )
    encoded = left_values[valid].astype(np.int64) * int(size) + right_values[valid]
    matrix = np.bincount(encoded, minlength=int(size) ** 2).reshape(size, size)
    total = int(matrix.sum())
    if total <= 1:
        raise ValueError("contingency comparison has fewer than two rows")

    def choose_two(values: np.ndarray | int) -> np.ndarray | float:
        array = np.asarray(values, dtype=np.float64)
        result = array * (array - 1.0) / 2.0
        return float(result) if result.ndim == 0 else result

    row_sum = matrix.sum(axis=1)
    column_sum = matrix.sum(axis=0)
    pair_total = float(choose_two(total))
    pair_same = float(np.asarray(choose_two(matrix)).sum())
    pair_rows = float(np.asarray(choose_two(row_sum)).sum())
    pair_columns = float(np.asarray(choose_two(column_sum)).sum())
    expected = pair_rows * pair_columns / max(pair_total, 1.0)
    maximum = 0.5 * (pair_rows + pair_columns)
    ari = (pair_same - expected) / max(maximum - expected, 1.0e-12)

    probabilities = matrix.astype(np.float64) / float(total)
    row_probability = probabilities.sum(axis=1)
    column_probability = probabilities.sum(axis=0)
    positive = probabilities > 0.0
    denominator = row_probability[:, None] * column_probability[None, :]
    mutual_information = float(
        np.sum(probabilities[positive] * np.log(probabilities[positive] / denominator[positive]))
    )
    row_entropy = float(-np.sum(row_probability[row_probability > 0.0] * np.log(row_probability[row_probability > 0.0])))
    column_entropy = float(-np.sum(column_probability[column_probability > 0.0] * np.log(column_probability[column_probability > 0.0])))
    nmi = mutual_information / max(0.5 * (row_entropy + column_entropy), 1.0e-12)
    purity = float(matrix.max(axis=1).sum() / total)
    conditional_entropy = max(row_entropy + column_entropy - mutual_information - row_entropy, 0.0)
    normalized_conditional_entropy = conditional_entropy / max(column_entropy, 1.0e-12)
    return {
        "row_count": total,
        "contingency": matrix.tolist(),
        "adjusted_rand_index": float(ari),
        "normalized_mutual_information": float(nmi),
        "right_given_left_purity": purity,
        "normalized_right_given_left_entropy": float(normalized_conditional_entropy),
    }


def _daily_tertiles(
    *,
    date_idx: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    dates = np.asarray(date_idx, dtype=np.int32)
    truth = np.asarray(values, dtype=np.float64)
    allowed = np.asarray(valid, dtype=bool)
    result = np.full(len(dates), -1, dtype=np.int8)
    boundaries = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        local = np.flatnonzero(allowed[start:stop] & np.isfinite(truth[start:stop]))
        if len(local) < 3:
            continue
        order = local[np.argsort(truth[start:stop][local], kind="mergesort")]
        for label, part in enumerate(np.array_split(order, 3)):
            result[start + part] = int(label)
    return result


def prepare_d3_atlas(input_root: Path) -> dict[str, Any]:
    output_root = input_root / "d3_atlas"
    summary_path = output_root / "d3_atlas_summary.json"
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if payload.get("status") == "completed":
            _emit("d3_atlas_already_complete", output_root=output_root)
            return payload

    atlas = _load_file_module(HORIZON_ATLAS_SCRIPT, "seq100_d3_horizon_atlas")
    atlas.HORIZONS = (3,)
    output_root.mkdir(parents=True, exist_ok=True)
    audit_payload = atlas.audit(output_root)
    source_module = atlas._load_source_module()
    source = source_module.PathAtlasSource()
    paths, status, macro_labels = atlas._open_arrays(source_module, source)
    units: list[dict[str, Any]] = []
    for window_year in atlas.WINDOW_YEARS:
        units.append(
            atlas._fit_window_horizon(
                module=source_module,
                source=source,
                paths=paths,
                status=status,
                horizon=3,
                window_year=int(window_year),
                output_root=output_root,
            )
        )
    stability = atlas.calendar_stability(
        module=source_module,
        paths=paths,
        output_root=output_root,
    )
    selected, fixed, candidate_map = atlas.assign_final_labels(
        module=source_module,
        source=source,
        paths=paths,
        status=status,
        macro_labels=macro_labels,
        output_root=output_root,
    )
    selected_profile = atlas.summarize_horizon_clusters(
        paths=paths,
        labels=selected,
        family="selected",
        output_root=output_root,
    )
    fixed_profile = atlas.summarize_horizon_clusters(
        paths=paths,
        labels=fixed,
        family="fixed_k3",
        output_root=output_root,
    )
    fixed_map = atlas._semantic_map(fixed_profile, 3)
    state = atlas._remap(np.asarray(fixed[0], dtype=np.int16), fixed_map).astype(np.int8)
    g3 = np.asarray(paths[:, 2], dtype=np.float32)
    valid = (state >= 0) & np.isfinite(g3)
    global_thresholds = np.quantile(g3[valid], (1.0 / 3.0, 2.0 / 3.0))
    global_tertile = np.full(len(g3), -1, dtype=np.int8)
    global_tertile[valid] = np.digitize(g3[valid], global_thresholds).astype(np.int8)
    daily_tertile = _daily_tertiles(
        date_idx=np.asarray(source.candidate_date_idx[: len(g3)], dtype=np.int32),
        values=g3,
        valid=valid,
    )
    model = atlas._load_model(
        source_module, output_root, 3, max(atlas.WINDOW_YEARS)
    )
    raw_centers = atlas._raw_centers(model, "fixed_k3_centers")
    semantic_centers = np.empty_like(raw_centers)
    for raw_label, semantic_label in fixed_map.items():
        semantic_centers[int(semantic_label)] = raw_centers[int(raw_label)]
    time_axis = np.arange(1, 4, dtype=np.float64) / 3.0
    linear_endpoint = semantic_centers[:, -1, None] * time_axis[None, :]
    shape_residual = semantic_centers - linear_endpoint
    state_profiles: list[dict[str, Any]] = []
    for label in range(3):
        current = valid & (state == label)
        state_profiles.append(
            {
                "state": label,
                "row_count": int(np.count_nonzero(current)),
                "g3_q10": float(np.quantile(g3[current], 0.10)),
                "g3_median": float(np.median(g3[current])),
                "g3_q90": float(np.quantile(g3[current], 0.90)),
                "raw_center": semantic_centers[label].tolist(),
                "endpoint_detrended_center": shape_residual[label].tolist(),
            }
        )
    final_unit = next(
        item for item in units if int(item["window_year"]) == max(atlas.WINDOW_YEARS)
    )
    payload = {
        "status": "completed",
        "completed_at": _now(),
        "scope": "pre_2023_full_candidate_d1_d3_path_clustering",
        "candidate_count": int(source.boundary.candidate_count),
        "assignable_path_count": int(audit_payload["assignable_path_count"]),
        "signal_end_date": str(source.date_values[source.boundary.signal_end_idx]),
        "outcome_cutoff_date": str(source.date_values[source.boundary.outcome_cutoff_idx]),
        "forbidden_years": [2023, 2024, 2025, 2026],
        "k_candidates": list(atlas.K_CANDIDATES),
        "window_years": list(atlas.WINDOW_YEARS),
        "selected_k_by_window": [
            {"window_year": int(item["window_year"]), "selected_k": int(item["selected_k"])}
            for item in units
        ],
        "final_selected_k": int(final_unit["selected_k"]),
        "fixed_k3_calendar_stability": {
            "minimum_ari": float(
                stability.loc[stability["family"] == "fixed_k3", "reference_assignment_ari"].min()
            ),
            "mean_ari": float(
                stability.loc[stability["family"] == "fixed_k3", "reference_assignment_ari"].mean()
            ),
            "minimum_centroid_correlation": float(
                stability.loc[
                    stability["family"] == "fixed_k3",
                    "matched_centroid_correlation_mean",
                ].min()
            ),
        },
        "state_vs_g3_global_tertile": _contingency_metrics(state, global_tertile),
        "state_vs_g3_daily_tertile": _contingency_metrics(state, daily_tertile),
        "global_g3_tertile_thresholds": global_thresholds.tolist(),
        "state_profiles": state_profiles,
        "center_endpoint_range": float(np.ptp(semantic_centers[:, -1])),
        "maximum_endpoint_detrended_center_range": float(
            np.max(np.ptp(shape_residual, axis=0))
        ),
        "files": {
            "fixed_model": _file_record(
                output_root / "models/h03/through_2022/model.npz"
            ),
            "fixed_labels": _file_record(
                output_root / "fixed_k3_horizon_labels.int16.dat",
                shape=(1, int(source.boundary.candidate_count)),
                dtype="int16",
            ),
            "selected_profile": _file_record(
                output_root / "selected_cluster_profiles.parquet"
            ),
            "fixed_profile": _file_record(
                output_root / "fixed_k3_cluster_profiles.parquet"
            ),
            "calendar_stability": _file_record(
                output_root / "calendar_stability.parquet"
            ),
            "candidate_map": _file_record(candidate_map),
        },
        "source": {
            "horizon_atlas_script": str(HORIZON_ATLAS_SCRIPT.resolve()),
            "horizon_atlas_script_sha256": _file_sha256(HORIZON_ATLAS_SCRIPT),
            "pre2023_path_root": str(PRE2023_ATLAS_ROOT.resolve()),
            "source_audit_sha256": _file_sha256(output_root / "source_audit.json"),
        },
    }
    _write_json(summary_path, payload)
    _emit(
        "d3_atlas_complete",
        selected_k=payload["final_selected_k"],
        fixed_k3_minimum_ari=payload["fixed_k3_calendar_stability"]["minimum_ari"],
    )
    return payload


def _load_d3_state_model(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as source:
        model = {name: np.asarray(source[name]).copy() for name in source.files}
    if int(np.asarray(model["horizon"]).reshape(-1)[0]) != 3:
        raise ValueError("D3 state model horizon changed")
    if int(np.asarray(model["window_year"]).reshape(-1)[0]) != 2022:
        raise ValueError("D3 state model is not frozen through 2022")
    centers = np.asarray(model["fixed_k3_centers"], dtype=np.float64)
    raw_scaled = (
        centers @ np.asarray(model["pca_components"], dtype=np.float64)
        + np.asarray(model["pca_mean"], dtype=np.float64)
    )
    raw_centers = (
        raw_scaled * np.asarray(model["scale"], dtype=np.float64)
        + np.asarray(model["center"], dtype=np.float64)
    )
    order = np.argsort(raw_centers[:, -1], kind="stable")
    raw_to_state = np.empty(3, dtype=np.int8)
    raw_to_state[order] = np.arange(3, dtype=np.int8)
    model["raw_to_state"] = raw_to_state
    return model


def _assign_d3_state(values: np.ndarray, model: Mapping[str, Any]) -> np.ndarray:
    prefix = np.asarray(values[:, :3], dtype=np.float32)
    scaled = (
        np.clip(prefix, model["clip_low"], model["clip_high"])
        - model["center"]
    ) / model["scale"]
    transformed = (scaled - model["pca_mean"]) @ np.asarray(
        model["pca_components"], dtype=np.float32
    ).T
    centers = np.asarray(model["fixed_k3_centers"], dtype=np.float32)
    squared = (
        np.square(transformed).sum(axis=1, keepdims=True)
        - 2.0 * transformed @ centers.T
        + np.square(centers).sum(axis=1)[None, :]
    )
    raw = np.argmin(squared, axis=1)
    return np.asarray(model["raw_to_state"], dtype=np.int8)[raw]


def _numeric_path_labels(
    close: np.ndarray,
    sellable: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(close, dtype=np.float64)
    legal = np.asarray(sellable, dtype=bool)
    base_valid = np.asarray(valid, dtype=bool).reshape(-1)
    if values.ndim != 2 or values.shape[1] < 2 or legal.shape != values.shape:
        raise ValueError("path labels require aligned D1-DH close and sellable arrays")
    g = np.full(len(values), np.nan, dtype=np.float32)
    mfe = np.full(len(values), np.nan, dtype=np.float32)
    pre_peak_mae = np.full(len(values), np.nan, dtype=np.float32)
    g[base_valid] = values[base_valid, -1].astype(np.float32)
    legal_exit = legal[:, 1:]
    has_sellable = legal_exit.any(axis=1)
    mfe_valid = base_valid & has_sellable
    if np.any(mfe_valid):
        masked = np.where(legal_exit, values[:, 1:], -np.inf)
        peak_idx = np.argmax(masked, axis=1) + 1
        rows = np.arange(len(values), dtype=np.int64)
        peak_value = values[rows, peak_idx]
        cumulative_min = np.minimum.accumulate(values, axis=1)
        adverse = np.minimum(0.0, cumulative_min[rows, peak_idx])
        mfe[mfe_valid] = peak_value[mfe_valid].astype(np.float32)
        pre_peak_mae[mfe_valid] = adverse[mfe_valid].astype(np.float32)
    return g, mfe, pre_peak_mae, has_sellable


def _derive_short_batch(
    *,
    source: Any,
    d3_model: Mapping[str, Any],
    date_idx: int,
    symbol_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    symbols = np.asarray(symbol_idx, dtype=np.int64)
    count = len(symbols)
    labels = np.full((count, len(SHORT_LABEL_COLUMNS)), np.nan, dtype=np.float32)
    states = np.full(count, -1, dtype=np.int8)
    flags = np.zeros((count, 2), dtype=np.uint16)
    maximum_horizon = min(3, int(source.cutoff_idx) - int(date_idx))
    if maximum_horizon < 1:
        return labels, states, flags
    filled = np.asarray(source.entry_filled[int(date_idx), symbols], dtype=bool)
    future = source.future_ohlc_prefix(
        int(date_idx), symbols, int(maximum_horizon)
    )
    anchor_denominator = 1.0 + future[:, 0, 0]
    anchor_valid = np.isfinite(anchor_denominator) & (anchor_denominator > 1.0e-8)
    close = np.divide(
        1.0 + future[:, :, 3],
        anchor_denominator[:, None],
        out=np.full((count, maximum_horizon), np.nan, dtype=np.float64),
        where=anchor_valid[:, None],
    ) - 1.0
    low = np.divide(
        1.0 + future[:, :, 2],
        anchor_denominator[:, None],
        out=np.full((count, maximum_horizon), np.nan, dtype=np.float64),
        where=anchor_valid[:, None],
    ) - 1.0
    complete_day = np.isfinite(future).all(axis=2) & np.isfinite(low) & (low > -1.0)
    complete_prefix = np.logical_and.accumulate(complete_day, axis=1)
    base_flags = np.where(filled, FLAG_ENTRY_FILLED, 0).astype(np.uint16)
    base_flags |= np.where(anchor_valid, FLAG_ANCHOR_VALID, 0).astype(np.uint16)
    flags[:] = base_flags[:, None]

    flags[:, 0] |= FLAG_OUTCOME_WITHIN_CUTOFF
    complete1 = complete_prefix[:, 0]
    flags[:, 0] |= np.where(complete1, FLAG_PATH_COMPLETE, 0).astype(np.uint16)
    valid1 = filled & anchor_valid & complete1
    labels[valid1, SHORT_LABEL_INDEX["g_1"]] = close[valid1, 0].astype(np.float32)
    flags[:, 0] |= np.where(valid1, FLAG_G_VALID, 0).astype(np.uint16)

    if maximum_horizon >= 3:
        flags[:, 1] |= FLAG_OUTCOME_WITHIN_CUTOFF
        complete3 = complete_prefix[:, 2]
        flags[:, 1] |= np.where(complete3, FLAG_PATH_COMPLETE, 0).astype(np.uint16)
        valid3 = filled & anchor_valid & complete3
        start = int(date_idx) + 1
        sellable = np.asarray(
            source.exit_sellable[start : start + 3, symbols], dtype=bool
        ).T
        g3, mfe3, mae3, has_sellable = _numeric_path_labels(
            close[:, :3], sellable[:, :3], valid3
        )
        labels[:, SHORT_LABEL_INDEX["g_3"]] = g3
        labels[:, SHORT_LABEL_INDEX["mfe_3"]] = mfe3
        labels[:, SHORT_LABEL_INDEX["pre_peak_mae_3"]] = mae3
        flags[:, 1] |= np.where(valid3, FLAG_G_VALID, 0).astype(np.uint16)
        flags[:, 1] |= np.where(
            has_sellable, FLAG_HAS_SELLABLE_D2_TO_D3, 0
        ).astype(np.uint16)
        mfe_valid = valid3 & has_sellable
        flags[:, 1] |= np.where(mfe_valid, FLAG_MFE_VALID, 0).astype(np.uint16)
        if np.any(valid3):
            states[valid3] = _assign_d3_state(close[valid3, :3], d3_model)
            flags[:, 1] |= np.where(
                valid3, FLAG_STATE_ASSIGNED, 0
            ).astype(np.uint16)
    return labels, states, flags


def prepare_short_labels(input_root: Path) -> dict[str, Any]:
    output_root = input_root / "labels"
    manifest_path = output_root / "short_label_manifest.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") == "completed":
            _emit("short_labels_already_complete", output_root=output_root)
            return payload
    input_module = _load_file_module(INPUT_PREP_SCRIPT, "seq100_short_label_source")
    source = input_module.LearnabilitySource()
    d3_model_path = input_root / "d3_atlas/models/h03/through_2022/model.npz"
    d3_model = _load_d3_state_model(d3_model_path)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "labels": output_root / "short_labels.float32.dat",
        "states": output_root / "state_3.int8.dat",
        "flags": output_root / "short_label_flags.uint16.dat",
        "progress": output_root / "generation_progress.json",
    }
    config = {
        "schema": "seq100_short_horizon_labels/config/v1",
        "candidate_count": int(source.candidate_count),
        "outcome_cutoff": "2025-12-31",
        "label_columns": list(SHORT_LABEL_COLUMNS),
        "state_column": "state_3",
        "pack_manifest_sha256": str(source.pack_sha256),
        "feature_view_manifest_sha256": str(source.feature_view_sha256),
        "d3_model_sha256": _file_sha256(d3_model_path),
        "input_prep_script_sha256": _file_sha256(INPUT_PREP_SCRIPT),
    }
    config_sha = _canonical_sha256(config)
    if paths["progress"].is_file():
        progress = json.loads(paths["progress"].read_text(encoding="utf-8"))
        if str(progress.get("config_sha256")) != config_sha:
            raise ValueError("existing short-label output uses another config")
        mode = "r+"
    else:
        unexpected = [path for key, path in paths.items() if key != "progress" and path.exists()]
        if unexpected:
            raise ValueError(f"short-label outputs exist without recovery metadata: {unexpected}")
        progress = {
            "status": "initialized",
            "config": config,
            "config_sha256": config_sha,
            "next_date_idx": int(source.signal_start_idx),
            "processed_candidates": 0,
            "created_at": _now(),
        }
        _write_json(paths["progress"], progress)
        mode = "w+"
    labels = np.memmap(
        paths["labels"],
        dtype=np.float32,
        mode=mode,
        shape=(int(source.candidate_count), len(SHORT_LABEL_COLUMNS)),
    )
    states = np.memmap(
        paths["states"],
        dtype=np.int8,
        mode=mode,
        shape=(int(source.candidate_count),),
    )
    flags = np.memmap(
        paths["flags"],
        dtype=np.uint16,
        mode=mode,
        shape=(int(source.candidate_count), 2),
    )
    if mode == "w+":
        labels[:] = np.nan
        states[:] = -1
        flags[:] = 0
        labels.flush()
        states.flush()
        flags.flush()
    if progress.get("status") != "completed":
        started = time.monotonic()
        last_emit = started
        start_date_idx = int(progress.get("next_date_idx", source.signal_start_idx))
        processed_before = int(progress.get("processed_candidates", 0))
        for date_idx in range(start_date_idx, int(source.cutoff_idx) + 1):
            left, right = source.candidate_bounds(date_idx)
            if right > left:
                symbols = np.asarray(source.candidate_symbol_idx[left:right], dtype=np.int64)
                current_labels, current_states, current_flags = _derive_short_batch(
                    source=source,
                    d3_model=d3_model,
                    date_idx=date_idx,
                    symbol_idx=symbols,
                )
                labels[left:right] = current_labels
                states[left:right] = current_states
                flags[left:right] = current_flags
            now = time.monotonic()
            if now - last_emit >= 30.0:
                labels.flush()
                states.flush()
                flags.flush()
                processed = int(
                    np.searchsorted(source.candidate_date_idx, date_idx, side="right")
                )
                elapsed = max(now - started, 1.0e-9)
                rate = max(processed - processed_before, 0) / elapsed
                progress = {
                    **progress,
                    "status": "running",
                    "next_date_idx": int(date_idx) + 1,
                    "processed_candidates": processed,
                    "processed_signal_date": str(source.date_values[date_idx]),
                    "candidate_rate_per_second": float(rate),
                    "updated_at": _now(),
                }
                _write_json(paths["progress"], progress)
                _emit(
                    "short_label_progress",
                    processed_candidates=processed,
                    signal_date=str(source.date_values[date_idx]),
                    candidate_rate_per_second=rate,
                )
                last_emit = now
        labels.flush()
        states.flush()
        flags.flush()
        progress = {
            **progress,
            "status": "completed",
            "next_date_idx": int(source.cutoff_idx) + 1,
            "processed_candidates": int(source.candidate_count_through_cutoff),
            "completed_at": _now(),
            "updated_at": _now(),
        }
        _write_json(paths["progress"], progress)

    payload: dict[str, Any] = {
        "schema": "seq100_short_horizon_label_manifest/v1",
        "status": "completed",
        "completed_at": _now(),
        "candidate_count": int(source.candidate_count),
        "label_columns": list(SHORT_LABEL_COLUMNS),
        "state_columns": ["state_3"],
        "maximum_outcome_date_read": "2025-12-31",
        "forbidden_outcome_year": 2026,
        "training_performed": False,
        "semantics": {
            "g_1": "gross simple return from legal next-open anchor to D1 close",
            "g_3": "gross simple return from legal next-open anchor to D3 close",
            "mfe_3": "maximum D2-D3 close return on source exit_sellable days; not floored at zero",
            "pre_peak_mae_3": "minimum of entry baseline zero and D1-to-best-sellable-close close returns",
            "state_3": "through-2022 fixed-K3 D1-D3 close-path assignment ordered by raw center endpoint",
        },
        "source": {
            "pack_manifest": str(source.pack_path.resolve()),
            "pack_manifest_sha256": str(source.pack_sha256),
            "feature_view_manifest": str(source.feature_view_path.resolve()),
            "feature_view_manifest_sha256": str(source.feature_view_sha256),
            "input_prep_script": str(INPUT_PREP_SCRIPT.resolve()),
            "input_prep_script_sha256": _file_sha256(INPUT_PREP_SCRIPT),
            "d3_state_model": str(d3_model_path.resolve()),
            "d3_state_model_sha256": _file_sha256(d3_model_path),
        },
        "files": {
            "short_labels": _file_record(
                paths["labels"],
                shape=(int(source.candidate_count), len(SHORT_LABEL_COLUMNS)),
                dtype="float32",
            ),
            "state_3": _file_record(
                paths["states"], shape=(int(source.candidate_count),), dtype="int8"
            ),
            "flags": _file_record(
                paths["flags"], shape=(int(source.candidate_count), 2), dtype="uint16"
            ),
            "generation_progress": _file_record(paths["progress"]),
        },
    }
    payload["resolved_short_label_manifest_sha256"] = _canonical_sha256(payload)
    _write_json(manifest_path, payload)
    _emit("short_labels_complete", candidate_count=source.candidate_count)
    return payload


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    truth = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(truth) & np.isfinite(weight) & (weight > 0.0)
    if not np.any(valid):
        raise ValueError("weighted quantile has no valid rows")
    order = np.argsort(truth[valid], kind="mergesort")
    sorted_values = truth[valid][order]
    sorted_weight = weight[valid][order]
    cumulative = np.cumsum(sorted_weight)
    target = float(quantile) * float(cumulative[-1])
    return float(sorted_values[min(int(np.searchsorted(cumulative, target, side="left")), len(sorted_values) - 1)])


def _load_old_inputs(*, verify_large_hashes: bool = False) -> base.LearnabilityInputs:
    study = base.load_study(OLD_CONTRACT_PATH)
    return base.LearnabilityInputs(study, verify_large_hashes=verify_large_hashes)


def _open_short_arrays(manifest: Mapping[str, Any]) -> tuple[np.memmap, np.memmap, np.memmap]:
    count = int(manifest["candidate_count"])
    files = dict(manifest["files"])
    labels = np.memmap(
        Path(str(files["short_labels"]["path"])),
        dtype=np.float32,
        mode="r",
        shape=(count, len(SHORT_LABEL_COLUMNS)),
    )
    states = np.memmap(
        Path(str(files["state_3"]["path"])),
        dtype=np.int8,
        mode="r",
        shape=(count,),
    )
    flags = np.memmap(
        Path(str(files["flags"]["path"])),
        dtype=np.uint16,
        mode="r",
        shape=(count, 2),
    )
    return labels, states, flags


def write_thresholds(input_root: Path) -> dict[str, Any]:
    path = input_root / "mfe_thresholds.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    manifest = json.loads(
        (input_root / "labels/short_label_manifest.json").read_text(encoding="utf-8")
    )
    short_labels, _states, _flags = _open_short_arrays(manifest)
    old = _load_old_inputs(verify_large_hashes=False)
    thresholds: list[dict[str, Any]] = []
    for horizon in (3, 5, 10, 20, 40, 60):
        values = (
            short_labels[:, SHORT_LABEL_INDEX["mfe_3"]]
            if horizon == 3
            else old.label_values("mfe", horizon)
        )
        fold = base.build_fold_rows(
            candidate_date_idx=old.candidate_date_idx,
            date_values=old.date_values,
            values=(
                short_labels[:, SHORT_LABEL_INDEX["g_3"]]
                if horizon == 3
                else old.label_values("g", horizon)
            ),
            year=2023,
            dependency_days=horizon,
            valid=np.isfinite,
        )
        current = np.asarray(values[fold.train_rows], dtype=np.float32)
        valid = np.isfinite(current)
        rows = fold.train_rows[valid]
        weights = base.date_equal_weights(old.candidate_date_idx[rows])
        thresholds.append(
            {
                "horizon": horizon,
                "pre_2023_train_row_count": int(len(rows)),
                "date_equal_q80": _weighted_quantile(current[valid], weights, 0.80),
            }
        )
    payload = {
        "schema": "seq100_short_horizon_target_reaudit_mfe_thresholds/v1",
        "created_at": _now(),
        "definition": "date-equal weighted q80 on the purged pre-2023 training rows",
        "thresholds": thresholds,
        "forbidden_years": [2026],
        "forbidden_years_consumed": [],
    }
    _write_json(path, payload)
    return payload


def audit_prepared_inputs(input_root: Path) -> dict[str, Any]:
    manifest_path = input_root / "labels/short_label_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    labels, states, flags = _open_short_arrays(manifest)
    old = _load_old_inputs(verify_large_hashes=False)
    if int(manifest["candidate_count"]) != old.candidate_count:
        raise ValueError("short labels no longer align with the base candidate universe")
    post_dates = np.flatnonzero(np.char.startswith(old.date_values, "2026-"))
    post_start = (
        int(np.searchsorted(old.candidate_date_idx, int(post_dates[0]), side="left"))
        if len(post_dates)
        else old.candidate_count
    )
    if not bool(np.isnan(np.asarray(labels[post_start:])).all()):
        raise AssertionError("2026 candidate has a short-horizon label")
    if not bool((np.asarray(states[post_start:]) == -1).all()):
        raise AssertionError("2026 candidate has a D3 state")
    atlas_summary = json.loads(
        (input_root / "d3_atlas/d3_atlas_summary.json").read_text(encoding="utf-8")
    )
    pre_count = int(atlas_summary["candidate_count"])
    atlas_raw = np.memmap(
        input_root / "d3_atlas/fixed_k3_horizon_labels.int16.dat",
        dtype=np.int16,
        mode="r",
        shape=(1, pre_count),
    )
    profile = pd.read_parquet(
        input_root / "d3_atlas/fixed_k3_cluster_profiles.parquet"
    )
    mapping = {
        int(row.raw_cluster): int(row.semantic_rank)
        for row in profile.itertuples(index=False)
    }
    expected = np.full(pre_count, -1, dtype=np.int8)
    for raw, semantic in mapping.items():
        expected[np.asarray(atlas_raw[0]) == raw] = semantic
    comparable = (expected >= 0) & (np.asarray(states[:pre_count]) >= 0)
    mismatch = int(
        np.count_nonzero(expected[comparable] != np.asarray(states[:pre_count])[comparable])
    )
    if mismatch:
        raise AssertionError("full-history D3 state assignment disagrees with pre-2023 atlas")
    payload = {
        "schema": "seq100_short_horizon_target_reaudit_input_audit/v1",
        "status": "passed",
        "completed_at": _now(),
        "candidate_count": old.candidate_count,
        "post_2025_candidate_count": int(old.candidate_count - post_start),
        "finite_counts": {
            name: int(np.count_nonzero(np.isfinite(np.asarray(labels[:, index]))))
            for index, name in enumerate(SHORT_LABEL_COLUMNS)
        },
        "state_3_count": int(np.count_nonzero(np.asarray(states) >= 0)),
        "pre_2023_state_crosscheck_count": int(np.count_nonzero(comparable)),
        "pre_2023_state_mismatch_count": mismatch,
        "flag_counts": {
            "g1_valid": int(np.count_nonzero(np.asarray(flags[:, 0]) & FLAG_G_VALID)),
            "g3_valid": int(np.count_nonzero(np.asarray(flags[:, 1]) & FLAG_G_VALID)),
            "mfe3_valid": int(np.count_nonzero(np.asarray(flags[:, 1]) & FLAG_MFE_VALID)),
        },
        "forbidden_years_consumed": [],
        "files": {
            "short_label_manifest": _file_record(manifest_path),
            "d3_atlas_summary": _file_record(
                input_root / "d3_atlas/d3_atlas_summary.json"
            ),
            "mfe_thresholds": _file_record(input_root / "mfe_thresholds.json"),
        },
    }
    _write_json(input_root / "input_audit.json", payload)
    _emit("prepared_input_audit_complete", **payload["finite_counts"])
    return payload


def prepare_all(input_root: Path) -> dict[str, Any]:
    self_test()
    atlas = prepare_d3_atlas(input_root)
    labels = prepare_short_labels(input_root)
    thresholds = write_thresholds(input_root)
    audit = audit_prepared_inputs(input_root)
    summary = {
        "status": "completed",
        "completed_at": _now(),
        "d3_atlas": atlas,
        "short_labels": labels,
        "mfe_thresholds": thresholds,
        "audit": audit,
    }
    _write_json(input_root / "preparation_summary.json", summary)
    return summary


def load_study(path: Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("study_id")) != STUDY_ID:
        raise ValueError(f"study_id must be {STUDY_ID}")
    contract = payload.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("study contract is missing")
    if str(contract.get("contract_id")) != STUDY_ID:
        raise ValueError("contract_id changed")
    declared = str(payload.get("contract_sha256", ""))
    computed = _canonical_sha256(contract)
    if declared != computed:
        raise ValueError("study contract SHA-256 mismatch")
    protocol = dict(contract.get("protocol", {}) or {})
    if tuple(int(value) for value in protocol.get("fold_years", [])) != FOLD_YEARS:
        raise ValueError("fold years changed")
    if dict(protocol.get("targets_by_horizon", {})) != {
        "1": ["g"],
        "3": ["g", "mfe", "pre_peak_mae", "state"],
    }:
        raise ValueError("short-horizon target schedule changed")
    firewall = dict(contract.get("scientific_firewall", {}) or {})
    if tuple(int(value) for value in firewall.get("forbidden_years", [])) != (2026,):
        raise ValueError("2026 firewall changed")
    for section_name in ("inputs", "implementation"):
        section = dict(contract.get(section_name, {}) or {})
        for name, record in section.items():
            if not isinstance(record, Mapping) or "path" not in record or "sha256" not in record:
                continue
            bound_path = _resolve_path(str(record["path"]))
            if not bound_path.is_file():
                raise FileNotFoundError(bound_path)
            if _file_sha256(bound_path) != str(record["sha256"]):
                raise ValueError(f"bound {section_name}.{name} changed")
    return payload


class ShortHorizonInputs:
    def __init__(self, study: Mapping[str, Any], *, verify_large_hashes: bool) -> None:
        bindings = dict(study["contract"]["inputs"])
        old_contract = _resolve_path(str(bindings["old_learnability_contract"]["path"]))
        if _file_sha256(old_contract) != str(bindings["old_learnability_contract"]["sha256"]):
            raise ValueError("old learnability contract changed")
        old_study = base.load_study(old_contract)
        self.old = base.LearnabilityInputs(
            old_study, verify_large_hashes=verify_large_hashes
        )
        manifest_path = _resolve_path(str(bindings["short_label_manifest"]["path"]))
        if _file_sha256(manifest_path) != str(bindings["short_label_manifest"]["sha256"]):
            raise ValueError("short label manifest changed")
        self.short_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_resolved = str(bindings["short_label_manifest"]["resolved_sha256"])
        if str(self.short_manifest.get("resolved_short_label_manifest_sha256")) != expected_resolved:
            raise ValueError("short label resolved manifest changed")
        files = dict(self.short_manifest["files"])
        for record in files.values():
            file_path = Path(str(record["path"])).resolve()
            if file_path.stat().st_size != int(record["size"]):
                raise ValueError(f"short input size changed: {file_path}")
            if verify_large_hashes and _file_sha256(file_path) != str(record["sha256"]):
                raise ValueError(f"short input SHA-256 changed: {file_path}")
        self.candidate_count = self.old.candidate_count
        if int(self.short_manifest["candidate_count"]) != self.candidate_count:
            raise ValueError("short labels and base inputs have different row counts")
        self.short_labels = np.memmap(
            Path(str(files["short_labels"]["path"])),
            dtype=np.float32,
            mode="r",
            shape=(self.candidate_count, len(SHORT_LABEL_COLUMNS)),
        )
        self.state_3 = np.memmap(
            Path(str(files["state_3"]["path"])),
            dtype=np.int8,
            mode="r",
            shape=(self.candidate_count,),
        )
        for name in (
            "continuous",
            "categorical",
            "candidate_date_idx",
            "candidate_symbol_idx",
            "date_values",
            "continuous_columns",
            "categorical_columns",
            "feature_names",
        ):
            setattr(self, name, getattr(self.old, name))
        self._assert_firewall()

    def _assert_firewall(self) -> None:
        post_dates = np.flatnonzero(np.char.startswith(self.date_values, "2026-"))
        post_start = (
            int(np.searchsorted(self.candidate_date_idx, int(post_dates[0]), side="left"))
            if len(post_dates)
            else self.candidate_count
        )
        if not bool(np.isnan(np.asarray(self.short_labels[post_start:])).all()):
            raise ValueError("2026 candidate has a short label")
        if not bool((np.asarray(self.state_3[post_start:]) == -1).all()):
            raise ValueError("2026 candidate has a D3 state")

    def label_values(self, target: str, horizon: int) -> np.ndarray:
        if int(horizon) == 1 and target == "g":
            return self.short_labels[:, SHORT_LABEL_INDEX["g_1"]]
        if int(horizon) == 3:
            if target == "state":
                return self.state_3
            return self.short_labels[:, SHORT_LABEL_INDEX[f"{target}_3"]]
        raise ValueError(f"unsupported short target: {target} D{horizon}")

    def all_label_values(self, target: str, horizon: int) -> np.ndarray:
        if int(horizon) in SHORT_TARGETS:
            return self.label_values(target, horizon)
        return self.old.label_values(target, horizon)

    def common_rows(self, year: int, horizon: int) -> base.FoldRows:
        return base.build_fold_rows(
            candidate_date_idx=self.candidate_date_idx,
            date_values=self.date_values,
            values=self.label_values("g", horizon),
            year=year,
            dependency_days=horizon,
            valid=np.isfinite,
        )


def _task_dir(output_root: Path, year: int, horizon: int, target: str) -> Path:
    return output_root / "folds" / f"fold_{year}" / f"h{horizon:02d}" / target


def _task_complete(path: Path, contract_sha256: str) -> bool:
    return base._task_result_complete(
        path,
        contract_sha256,
        result_schema=RESULT_SCHEMA,
    )


def _completed_task_count(output_root: Path, contract_sha256: str) -> int:
    return sum(
        _task_complete(path, contract_sha256)
        for path in output_root.glob("folds/fold_*/h*/**/task_result.json")
    )


def preflight(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    started = time.perf_counter()
    inputs = ShortHorizonInputs(study, verify_large_hashes=True)
    folds: list[dict[str, Any]] = []
    for year in FOLD_YEARS:
        for horizon in SHORT_TARGETS:
            fold = inputs.common_rows(year, horizon)
            folds.append(
                {
                    "fold_year": year,
                    "horizon": horizon,
                    "targets": list(SHORT_TARGETS[horizon]),
                    "dependency_days": horizon,
                    "train_row_count": int(len(fold.train_rows)),
                    "evaluation_row_count": int(len(fold.evaluation_rows)),
                    "maximum_train_signal_date_idx": int(fold.maximum_train_signal_date_idx),
                    "maximum_train_outcome_date_idx": int(
                        fold.maximum_train_signal_date_idx + horizon
                    ),
                }
            )
    payload = {
        "schema": "seq100_short_horizon_target_reaudit_preflight/v1",
        "status": "passed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "contract_sha256": str(study["contract_sha256"]),
        "large_input_hashes_verified": True,
        "elapsed_seconds": float(time.perf_counter() - started),
        "candidate_count": inputs.candidate_count,
        "continuous_feature_count": len(inputs.continuous_columns),
        "categorical_feature_count": len(inputs.categorical_columns),
        "maximum_consumed_outcome_date": "2025-12-31",
        "forbidden_years_consumed": [],
        "folds": folds,
        "parameters": {
            target: base._model_parameters(study, target=target)[0]
            for target in ("g", "mfe", "pre_peak_mae", "state")
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "preflight.json", payload)
    return payload


def _require_preflight(study: Mapping[str, Any], output_root: Path) -> None:
    path = output_root / "preflight.json"
    if not path.is_file():
        raise RuntimeError("full-hash preflight must pass before training")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise RuntimeError("preflight did not pass")
    if str(payload.get("contract_sha256")) != str(study["contract_sha256"]):
        raise RuntimeError("preflight belongs to another contract")
    if not bool(payload.get("large_input_hashes_verified")):
        raise RuntimeError("preflight skipped large hashes")


def _run_group(
    *,
    study: Mapping[str, Any],
    inputs: ShortHorizonInputs,
    output_root: Path,
    events_path: Path,
    year: int,
    horizon: int,
) -> int:
    pending = [
        target
        for target in SHORT_TARGETS[horizon]
        if not _task_complete(
            _task_dir(output_root, year, horizon, target) / "task_result.json",
            str(study["contract_sha256"]),
        )
    ]
    if not pending:
        return 0
    fold = inputs.common_rows(year, horizon)
    initial_train = np.asarray(inputs.label_values("g", horizon)[fold.train_rows])
    initial_evaluation = np.asarray(inputs.label_values("g", horizon)[fold.evaluation_rows])
    initial_train_weight = base.date_equal_weights(
        inputs.candidate_date_idx[fold.train_rows]
    )
    initial_evaluation_weight = base.date_equal_weights(
        inputs.candidate_date_idx[fold.evaluation_rows]
    )
    datasets = base.build_lgb_datasets(
        inputs=inputs,
        train_rows=fold.train_rows,
        evaluation_rows=fold.evaluation_rows,
        initial_train_label=initial_train,
        initial_evaluation_label=initial_evaluation,
        initial_train_weight=initial_train_weight,
        initial_evaluation_weight=initial_evaluation_weight,
        study=study,
    )
    group_dir = output_root / "folds" / f"fold_{year}" / f"h{horizon:02d}"
    base._write_group_material(
        output_dir=group_dir,
        study=study,
        inputs=inputs,
        fold=fold,
        datasets=datasets,
        study_id=STUDY_ID,
        group_schema=GROUP_SCHEMA,
    )
    completed = 0
    for target in pending:
        values = inputs.label_values(target, horizon)
        train_raw = np.asarray(values[fold.train_rows])
        evaluation_raw = np.asarray(values[fold.evaluation_rows])
        train_valid = base._valid_target(train_raw, target)
        evaluation_valid = base._valid_target(evaluation_raw, target)
        train_label, train_weight = base.aligned_target_weights(
            values=train_raw,
            date_idx=inputs.candidate_date_idx[fold.train_rows],
            valid=train_valid,
        )
        evaluation_label, evaluation_weight = base.aligned_target_weights(
            values=evaluation_raw,
            date_idx=inputs.candidate_date_idx[fold.evaluation_rows],
            valid=evaluation_valid,
        )
        base.train_one_model(
            study=study,
            inputs=inputs,
            datasets=datasets,
            target=target,
            horizon=horizon,
            year=year,
            train_label=train_label,
            evaluation_label=evaluation_label,
            metric_evaluation_label=evaluation_raw,
            train_weight=train_weight,
            evaluation_weight=evaluation_weight,
            output_dir=_task_dir(output_root, year, horizon, target),
            events_path=events_path,
            study_id=STUDY_ID,
            result_schema=RESULT_SCHEMA,
        )
        completed += 1
    base._release_training_memory(datasets)
    return completed


def run_study(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    _require_preflight(study, output_root)
    inputs = ShortHorizonInputs(study, verify_large_hashes=False)
    output_root.mkdir(parents=True, exist_ok=True)
    events_path = output_root / "events.jsonl"
    progress_path = output_root / "progress.json"
    started_at = _now()
    try:
        for year in FOLD_YEARS:
            for horizon in SHORT_TARGETS:
                current = f"fold_{year}_h{horizon:02d}"
                _write_json(
                    progress_path,
                    {
                        "schema": PROGRESS_SCHEMA,
                        "study_id": STUDY_ID,
                        "contract_sha256": str(study["contract_sha256"]),
                        "status": "running",
                        "phase": "training",
                        "started_at": started_at,
                        "updated_at": _now(),
                        "task_count": 15,
                        "completed_task_count": _completed_task_count(
                            output_root, str(study["contract_sha256"])
                        ),
                        "current_task": current,
                        **base._memory_snapshot(),
                    },
                )
                _run_group(
                    study=study,
                    inputs=inputs,
                    output_root=output_root,
                    events_path=events_path,
                    year=year,
                    horizon=horizon,
                )
        results = []
        for year in FOLD_YEARS:
            for horizon, targets in SHORT_TARGETS.items():
                for target in targets:
                    path = _task_dir(output_root, year, horizon, target) / "task_result.json"
                    if not _task_complete(path, str(study["contract_sha256"])):
                        raise RuntimeError(f"short-horizon task incomplete: {path}")
                    results.append(json.loads(path.read_text(encoding="utf-8")))
        summary = {
            "schema": "seq100_short_horizon_target_reaudit_training_summary/v1",
            "status": "completed",
            "completed_at": _now(),
            "study_id": STUDY_ID,
            "contract_sha256": str(study["contract_sha256"]),
            "task_count": len(results),
            "fold_years": list(FOLD_YEARS),
            "targets_by_horizon": {
                str(horizon): list(targets) for horizon, targets in SHORT_TARGETS.items()
            },
            "maximum_consumed_outcome_date": "2025-12-31",
            "forbidden_years_consumed": [],
            "results": [
                {
                    "fold_year": int(item["fold_year"]),
                    "horizon": int(item["horizon"]),
                    "target": str(item["target"]),
                    "best_iteration": int(item["best_iteration"]),
                    "training_seconds": float(item["training_seconds"]),
                    "metrics": item["metrics"],
                }
                for item in results
            ],
        }
        _write_json(output_root / "training_summary.json", summary)
        _write_json(
            progress_path,
            {
                "schema": PROGRESS_SCHEMA,
                "study_id": STUDY_ID,
                "contract_sha256": str(study["contract_sha256"]),
                "status": "completed",
                "phase": "completed",
                "started_at": started_at,
                "updated_at": _now(),
                "task_count": 15,
                "completed_task_count": 15,
                "current_task": None,
                **base._memory_snapshot(),
            },
        )
        return summary
    except BaseException as exc:
        _write_json(
            progress_path,
            {
                "schema": PROGRESS_SCHEMA,
                "study_id": STUDY_ID,
                "contract_sha256": str(study["contract_sha256"]),
                "status": "failed",
                "phase": "failed",
                "started_at": started_at,
                "updated_at": _now(),
                "task_count": 15,
                "completed_task_count": _completed_task_count(
                    output_root, str(study["contract_sha256"])
                ),
                "error_type": type(exc).__name__,
                "error": str(exc),
                **base._memory_snapshot(),
            },
        )
        raise


@dataclass(frozen=True)
class PredictionBundle:
    year: int
    horizon: int
    target: str
    rows: np.ndarray
    date_idx: np.ndarray
    actual: np.ndarray
    prediction: np.ndarray
    result: dict[str, Any]


def _prediction_bundle(
    *,
    inputs: ShortHorizonInputs,
    output_root: Path,
    year: int,
    horizon: int,
    target: str,
) -> PredictionBundle:
    root = output_root if int(horizon) in SHORT_TARGETS else OLD_OUTPUT_ROOT
    group_dir = root / "folds" / f"fold_{int(year)}" / f"h{int(horizon):02d}"
    task_dir = group_dir / target
    result = json.loads((task_dir / "task_result.json").read_text(encoding="utf-8"))
    rows = np.load(group_dir / "evaluation_rows.npy", allow_pickle=False)
    prediction = np.load(task_dir / "prediction.npy", allow_pickle=False)
    actual = np.asarray(inputs.all_label_values(target, horizon)[rows])
    if len(rows) != len(actual) or len(rows) != len(prediction):
        raise ValueError(f"prediction alignment changed for {target} D{horizon} {year}")
    for name in ("prediction", "daily_metrics", "model"):
        record = dict(result["files"])[name]
        path = Path(str(record["path"]))
        if path.stat().st_size != int(record["size"]):
            raise ValueError(f"prediction evidence size changed: {path}")
        if _file_sha256(path) != str(record["sha256"]):
            raise ValueError(f"prediction evidence SHA-256 changed: {path}")
    return PredictionBundle(
        year=int(year),
        horizon=int(horizon),
        target=str(target),
        rows=np.asarray(rows, dtype=np.int64),
        date_idx=np.asarray(inputs.candidate_date_idx[rows], dtype=np.int32),
        actual=actual,
        prediction=np.asarray(prediction),
        result=result,
    )


def _score_from_bundle(bundle: PredictionBundle) -> np.ndarray:
    if bundle.target == "state":
        probability = np.asarray(bundle.prediction, dtype=np.float64)
        return probability @ np.arange(probability.shape[1], dtype=np.float64)
    return np.asarray(bundle.prediction, dtype=np.float64)


def _daily_event_enrichment(
    *,
    date_idx: np.ndarray,
    actual: np.ndarray,
    score: np.ndarray,
    absolute_threshold: float,
    horizon: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dates = np.asarray(date_idx, dtype=np.int32)
    truth = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(score, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    boundaries = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        valid = np.isfinite(truth[start:stop]) & np.isfinite(predicted[start:stop])
        if int(np.count_nonzero(valid)) < 20:
            continue
        current_actual = truth[start:stop][valid]
        current_score = predicted[start:stop][valid]
        count = len(current_actual)
        score_order = np.argsort(current_score, kind="mergesort")
        top1 = score_order[-max(1, int(math.ceil(0.01 * count))) :]
        top5 = score_order[-max(1, int(math.ceil(0.05 * count))) :]
        actual_order = np.argsort(current_actual, kind="mergesort")
        daily_tail = np.zeros(count, dtype=bool)
        daily_tail[actual_order[-max(1, int(math.ceil(0.20 * count))) :]] = True
        absolute_tail = current_actual >= float(absolute_threshold)
        rows.append(
            {
                "date_idx": int(dates[start]),
                "candidate_count": int(count),
                "daily_tail_base_rate": float(daily_tail.mean()),
                "daily_tail_top1_rate": float(daily_tail[top1].mean()),
                "daily_tail_top5_rate": float(daily_tail[top5].mean()),
                "daily_tail_top1_lift": float(daily_tail[top1].mean() - daily_tail.mean()),
                "daily_tail_top5_lift": float(daily_tail[top5].mean() - daily_tail.mean()),
                "absolute_tail_base_rate": float(absolute_tail.mean()),
                "absolute_tail_top1_rate": float(absolute_tail[top1].mean()),
                "absolute_tail_top5_rate": float(absolute_tail[top5].mean()),
                "absolute_tail_top1_lift": float(absolute_tail[top1].mean() - absolute_tail.mean()),
                "absolute_tail_top5_lift": float(absolute_tail[top5].mean() - absolute_tail.mean()),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("MFE upper-tail audit has no valid dates")
    summary = {
        "date_count": int(len(frame)),
        "pre_2023_absolute_threshold": float(absolute_threshold),
        "daily_tail_top1_lift": float(frame["daily_tail_top1_lift"].mean()),
        "daily_tail_top5_lift": float(frame["daily_tail_top5_lift"].mean()),
        "absolute_tail_top1_lift": float(frame["absolute_tail_top1_lift"].mean()),
        "absolute_tail_top5_lift": float(frame["absolute_tail_top5_lift"].mean()),
        "daily_tail_top5_lift_hac": base._hac_mean_test(
            frame["daily_tail_top5_lift"].to_numpy(), max(int(horizon) - 1, 0)
        ),
        "absolute_tail_top5_lift_hac": base._hac_mean_test(
            frame["absolute_tail_top5_lift"].to_numpy(), max(int(horizon) - 1, 0)
        ),
    }
    return frame, summary


def _regression_gate(
    annual: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    *,
    require_upper_tail: bool,
) -> dict[str, Any]:
    minimum_worst = float(thresholds.get("minimum_worst_year_rank_ic", 0.01))
    maximum_hac = float(thresholds.get("maximum_hac_p_value", 0.05))
    minimum_hac_years = int(thresholds.get("minimum_hac_supported_years", 2))
    minimum_top1_years = int(thresholds.get("minimum_top1_positive_years", 2))
    minimum_monotone_years = int(thresholds.get("minimum_monotone_years", 2))
    monotone_cutoff = float(thresholds.get("minimum_decile_spearman", 0.5))
    reversal_floor = float(thresholds.get("minimum_no_reversal_decile_spearman", -0.2))
    rank_ic = [float(item["rank_ic"]) for item in annual]
    top1 = [float(item["top_1pct_lift"]) for item in annual]
    top5 = [float(item["top_5pct_lift"]) for item in annual]
    top_bottom = [float(item["top_bottom_spread"]) for item in annual]
    decile = [float(item["decile_spearman"]) for item in annual]
    hac = [float(item["rank_ic_hac_p_value"]) for item in annual]
    gates = {
        "all_years_positive_rank_ic": all(math.isfinite(value) and value > 0.0 for value in rank_ic),
        "all_years_positive_top5_lift": all(value > 0.0 for value in top5),
        "all_years_positive_top_bottom_spread": all(value > 0.0 for value in top_bottom),
        "worst_year_rank_ic": min(rank_ic),
        "worst_year_rank_ic_pass": min(rank_ic) >= minimum_worst,
        "top1_positive_years": sum(value > 0.0 for value in top1),
        "top1_positive_years_pass": sum(value > 0.0 for value in top1) >= minimum_top1_years,
        "hac_supported_years": sum(
            rank > 0.0 and p_value < maximum_hac for rank, p_value in zip(rank_ic, hac)
        ),
        "hac_supported_years_pass": sum(
            rank > 0.0 and p_value < maximum_hac for rank, p_value in zip(rank_ic, hac)
        )
        >= minimum_hac_years,
        "monotone_years": sum(value >= monotone_cutoff for value in decile),
        "monotone_years_pass": sum(value >= monotone_cutoff for value in decile)
        >= minimum_monotone_years,
        "no_material_decile_reversal": all(value > reversal_floor for value in decile),
    }
    if require_upper_tail:
        daily_tail = [float(item["daily_tail_top5_lift"]) for item in annual]
        gates["all_years_daily_top_quintile_enrichment"] = all(
            value > 0.0 for value in daily_tail
        )
    required = [
        "all_years_positive_rank_ic",
        "all_years_positive_top5_lift",
        "all_years_positive_top_bottom_spread",
        "worst_year_rank_ic_pass",
        "top1_positive_years_pass",
        "hac_supported_years_pass",
        "monotone_years_pass",
        "no_material_decile_reversal",
    ]
    if require_upper_tail:
        required.append("all_years_daily_top_quintile_enrichment")
    gates["qualified"] = bool(all(bool(gates[name]) for name in required))
    return gates


def _ranking_fields(bundle: PredictionBundle) -> dict[str, Any]:
    ranking = dict(bundle.result["metrics"]["ranking"])
    deciles = [float(value) for value in ranking["decile_means"]]
    return {
        "year": bundle.year,
        "rank_ic": float(ranking["rank_ic_mean"]),
        "top_1pct_lift": float(ranking["top_1pct_lift"]),
        "top_5pct_lift": float(ranking["top_5pct_lift"]),
        "top_bottom_spread": float(deciles[-1] - deciles[0]),
        "decile_spearman": float(ranking["decile_spearman"]),
        "rank_ic_hac_p_value": float(ranking["rank_ic_hac"]["p_value_two_sided"]),
        "best_iteration": int(bundle.result["best_iteration"]),
    }


def _date_demean(date_idx: np.ndarray, values: np.ndarray) -> np.ndarray:
    dates = np.asarray(date_idx, dtype=np.int32)
    result = np.asarray(values, dtype=np.float64).copy()
    boundaries = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        current = result[start:stop]
        valid = np.isfinite(current)
        if np.any(valid):
            current[valid] -= float(current[valid].mean())
        result[start:stop] = current
    return result


def _prediction_component_audit(bundle: PredictionBundle) -> dict[str, Any]:
    score = _score_from_bundle(bundle)
    actual = np.asarray(bundle.actual, dtype=np.float64)
    valid = np.isfinite(actual) & np.isfinite(score)
    dates = bundle.date_idx[valid]
    truth = actual[valid]
    predicted = score[valid]
    daily_rows: list[dict[str, Any]] = []
    boundaries = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        current_score = predicted[start:stop]
        current_truth = truth[start:stop]
        if len(current_score) < 10:
            continue
        rank_ic = stats.spearmanr(current_score, current_truth).statistic
        daily_rows.append(
            {
                "date_idx": int(dates[start]),
                "score_mean": float(current_score.mean()),
                "actual_mean": float(current_truth.mean()),
                "score_variance": float(np.var(current_score)),
                "unique_score_count": int(len(np.unique(current_score))),
                "rank_ic": float(rank_ic) if math.isfinite(float(rank_ic)) else math.nan,
            }
        )
    frame = pd.DataFrame(daily_rows)
    grand_mean = float(frame["score_mean"].mean())
    between = float(np.mean(np.square(frame["score_mean"] - grand_mean)))
    within = float(frame["score_variance"].mean())
    market_spearman = (
        stats.spearmanr(frame["score_mean"], frame["actual_mean"]).statistic
        if frame["score_mean"].nunique() > 1 and frame["actual_mean"].nunique() > 1
        else math.nan
    )
    market_pearson = (
        stats.pearsonr(frame["score_mean"], frame["actual_mean"]).statistic
        if frame["score_mean"].nunique() > 1 and frame["actual_mean"].nunique() > 1
        else math.nan
    )
    return {
        "year": bundle.year,
        "horizon": bundle.horizon,
        "target": bundle.target,
        "date_count": int(len(frame)),
        "between_date_score_variance": between,
        "within_date_score_variance": within,
        "within_date_variance_share": float(within / max(between + within, 1.0e-18)),
        "constant_cross_section_fraction": float((frame["unique_score_count"] <= 1).mean()),
        "mean_daily_cross_sectional_rank_ic": float(frame["rank_ic"].mean()),
        "market_component_spearman": (
            float(market_spearman) if math.isfinite(float(market_spearman)) else math.nan
        ),
        "market_component_pearson": (
            float(market_pearson) if math.isfinite(float(market_pearson)) else math.nan
        ),
    }


def _temperature_scale(probability: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1.0e-9, 1.0)
    logits = np.log(clipped) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _fit_temperature(
    probability: np.ndarray,
    actual: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    truth = np.asarray(actual, dtype=np.int64)
    prob = np.asarray(probability, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    valid = np.isin(truth, [0, 1, 2]) & np.isfinite(prob).all(axis=1) & (weight > 0.0)
    truth = truth[valid]
    prob = prob[valid]
    weight = weight[valid]
    weight /= weight.sum()

    def objective(log_temperature: float) -> float:
        calibrated = _temperature_scale(prob, math.exp(float(log_temperature)))
        return float(
            -np.sum(weight * np.log(np.clip(calibrated[np.arange(len(truth)), truth], 1.0e-12, 1.0)))
        )

    result = optimize.minimize_scalar(
        objective,
        bounds=(-4.0, 4.0),
        method="bounded",
        options={"xatol": 1.0e-5},
    )
    if not bool(result.success):
        raise RuntimeError("temperature scaling failed")
    return {
        "temperature": float(math.exp(float(result.x))),
        "calibration_row_count": int(len(truth)),
        "weighted_logloss": float(result.fun),
    }


def _state_probability_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    high = dict(metrics["high_state_probability"])
    return {
        "rank_ic": float(metrics["ordinal_ranking"]["rank_ic_mean"]),
        "decile_spearman": float(metrics["ordinal_ranking"]["decile_spearman"]),
        "top_5pct_high_state_lift": float(high["ranking"]["top_5pct_lift"]),
        "binary_brier_skill": float(high["brier_skill"]),
        "binary_logloss_skill": float(high["logloss_skill"]),
        "binary_expected_calibration_error": float(
            high["calibration"]["expected_calibration_error"]
        ),
        "multiclass_brier_skill": float(metrics["multiclass_brier_skill"]),
        "multiclass_logloss_skill": float(metrics["multiclass_logloss_skill"]),
    }


def _calibration_audit(
    *,
    inputs: ShortHorizonInputs,
    output_root: Path,
) -> dict[str, Any]:
    horizons: list[dict[str, Any]] = []
    for horizon in (3, 5, 10, 20, 40, 60):
        bundles = {
            year: _prediction_bundle(
                inputs=inputs,
                output_root=output_root,
                year=year,
                horizon=horizon,
                target="state",
            )
            for year in FOLD_YEARS
        }
        annual: list[dict[str, Any]] = []
        calibration_history: list[PredictionBundle] = []
        for year in FOLD_YEARS:
            current = bundles[year]
            raw = _state_probability_summary(current.result["metrics"])
            row: dict[str, Any] = {
                "year": year,
                "raw": raw,
                "calibration_source_years": [item.year for item in calibration_history],
                "calibrated": None,
            }
            if calibration_history:
                history_probability = np.concatenate(
                    [np.asarray(item.prediction, dtype=np.float64) for item in calibration_history]
                )
                history_actual = np.concatenate(
                    [np.asarray(item.actual, dtype=np.int8) for item in calibration_history]
                )
                history_dates = np.concatenate([item.date_idx for item in calibration_history])
                history_weights = base.date_equal_weights(history_dates)
                fit = _fit_temperature(
                    history_probability, history_actual, history_weights
                )
                calibrated_probability = _temperature_scale(
                    current.prediction, fit["temperature"]
                )
                valid = np.isin(np.asarray(current.actual), [0, 1, 2])
                _daily, metrics = base.state_metrics(
                    date_idx=current.date_idx[valid],
                    actual=np.asarray(current.actual)[valid],
                    probability=calibrated_probability[valid],
                    date_values=inputs.date_values,
                    horizon=horizon,
                )
                row["calibrator"] = fit
                row["calibrated"] = _state_probability_summary(metrics)
            annual.append(row)
            calibration_history.append(current)
        horizons.append({"horizon": horizon, "annual": annual})
    return {
        "method": "single-temperature scaling fitted only on preceding OOS fold predictions",
        "first_calibrated_year": 2024,
        "horizons": horizons,
    }


def _common_rows(bundles: Sequence[PredictionBundle]) -> tuple[np.ndarray, list[np.ndarray]]:
    common = np.asarray(bundles[0].rows, dtype=np.int64)
    for bundle in bundles[1:]:
        common = np.intersect1d(common, bundle.rows, assume_unique=True)
    indices = [np.searchsorted(bundle.rows, common) for bundle in bundles]
    for bundle, current in zip(bundles, indices):
        if not np.array_equal(bundle.rows[current], common):
            raise AssertionError("bundle row intersection drifted")
    return common, indices


def _pair_prediction_redundancy(
    left: PredictionBundle,
    right: PredictionBundle,
) -> dict[str, Any]:
    common, indices = _common_rows((left, right))
    left_score = _score_from_bundle(left)[indices[0]]
    right_score = _score_from_bundle(right)[indices[1]]
    dates = left.date_idx[indices[0]]
    valid = np.isfinite(left_score) & np.isfinite(right_score)
    left_residual = _date_demean(dates[valid], left_score[valid])
    right_residual = _date_demean(dates[valid], right_score[valid])
    weights = base.date_equal_weights(dates[valid]).astype(np.float64)
    weights /= weights.sum()
    left_mean = float(np.sum(weights * left_residual))
    right_mean = float(np.sum(weights * right_residual))
    covariance = float(
        np.sum(weights * (left_residual - left_mean) * (right_residual - right_mean))
    )
    weighted_pearson = covariance / max(
        math.sqrt(
            float(np.sum(weights * np.square(left_residual - left_mean)))
            * float(np.sum(weights * np.square(right_residual - right_mean)))
        ),
        1.0e-18,
    )
    daily: list[float] = []
    boundaries = np.flatnonzero(
        np.r_[True, dates[valid][1:] != dates[valid][:-1], True]
    )
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if stop - start < 10:
            continue
        correlation = stats.spearmanr(
            left_score[valid][start:stop], right_score[valid][start:stop]
        ).statistic
        if math.isfinite(float(correlation)):
            daily.append(float(correlation))
    return {
        "year": left.year,
        "left_horizon": left.horizon,
        "right_horizon": right.horizon,
        "common_row_count": int(len(common)),
        "date_equal_within_date_pearson": float(weighted_pearson),
        "mean_daily_spearman": float(np.mean(daily)),
    }


def _residual_novelty(
    *,
    target_bundle: PredictionBundle,
    predictor_bundles: Sequence[PredictionBundle],
    date_values: np.ndarray,
) -> dict[str, Any]:
    bundles = (*predictor_bundles, target_bundle)
    common, indices = _common_rows(bundles)
    dates = target_bundle.date_idx[indices[-1]]
    predictors = np.column_stack(
        [
            _date_demean(dates, _score_from_bundle(bundle)[index])
            for bundle, index in zip(predictor_bundles, indices[:-1])
        ]
    )
    target_score = _date_demean(
        dates, _score_from_bundle(target_bundle)[indices[-1]]
    )
    actual = np.asarray(target_bundle.actual[indices[-1]], dtype=np.float64)
    valid = (
        np.isfinite(predictors).all(axis=1)
        & np.isfinite(target_score)
        & np.isfinite(actual)
    )
    dates = dates[valid]
    predictors = predictors[valid]
    target_score = target_score[valid]
    actual = actual[valid]
    weights = base.date_equal_weights(dates).astype(np.float64)
    predictor_scale = np.sqrt(
        np.average(np.square(predictors), axis=0, weights=weights)
    )
    active = predictor_scale > 1.0e-10
    standardized = predictors[:, active] / predictor_scale[active]
    square_root = np.sqrt(weights)
    ridge = 1.0e-3
    if standardized.shape[1]:
        design = standardized * square_root[:, None]
        response = target_score * square_root
        coefficients = np.linalg.solve(
            design.T @ design + ridge * np.eye(design.shape[1]),
            design.T @ response,
        )
        fitted = standardized @ coefficients
    else:
        coefficients = np.empty(0, dtype=np.float64)
        fitted = np.zeros_like(target_score)
    residual = target_score - fitted
    weighted_mean = float(np.average(target_score, weights=weights))
    total = float(np.sum(weights * np.square(target_score - weighted_mean)))
    unexplained = float(np.sum(weights * np.square(residual)))
    _daily, ranking = base.daily_score_metrics(
        date_idx=dates,
        actual=actual,
        score=residual,
        date_values=date_values,
        horizon=target_bundle.horizon,
    )
    return {
        "year": target_bundle.year,
        "target_horizon": target_bundle.horizon,
        "predictor_horizons": [bundle.horizon for bundle in predictor_bundles],
        "common_row_count": int(len(common)),
        "score_explained_r2": float(1.0 - unexplained / max(total, 1.0e-18)),
        "active_predictor_horizons": [
            bundle.horizon
            for bundle, is_active in zip(predictor_bundles, active)
            if bool(is_active)
        ],
        "standardized_coefficients": coefficients.tolist(),
        "predictor_scales": predictor_scale.tolist(),
        "ridge_penalty": ridge,
        "residual_rank_ic": float(ranking["rank_ic_mean"]),
        "residual_top_5pct_lift": float(ranking["top_5pct_lift"]),
        "residual_rank_ic_hac_p_value": float(
            ranking["rank_ic_hac"]["p_value_two_sided"]
        ),
    }


def _state_redundancy_audit(
    *,
    inputs: ShortHorizonInputs,
    output_root: Path,
) -> dict[str, Any]:
    pair_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    actual_transitions: list[dict[str, Any]] = []
    pairs = ((3, 5), (3, 10), (3, 20), (5, 10), (5, 20), (10, 20))
    for year in FOLD_YEARS:
        bundles = {
            horizon: _prediction_bundle(
                inputs=inputs,
                output_root=output_root,
                year=year,
                horizon=horizon,
                target="state",
            )
            for horizon in (3, 5, 10, 20)
        }
        for left_horizon, right_horizon in pairs:
            left = bundles[left_horizon]
            right = bundles[right_horizon]
            pair_rows.append(_pair_prediction_redundancy(left, right))
            common, indices = _common_rows((left, right))
            metrics = _contingency_metrics(
                np.asarray(left.actual[indices[0]], dtype=np.int8),
                np.asarray(right.actual[indices[1]], dtype=np.int8),
            )
            actual_transitions.append(
                {
                    "scope": str(year),
                    "left_horizon": left_horizon,
                    "right_horizon": right_horizon,
                    **metrics,
                }
            )
        residual_rows.append(
            _residual_novelty(
                target_bundle=bundles[5],
                predictor_bundles=(bundles[3],),
                date_values=inputs.date_values,
            )
        )
        residual_rows.append(
            _residual_novelty(
                target_bundle=bundles[10],
                predictor_bundles=(bundles[3], bundles[5]),
                date_values=inputs.date_values,
            )
        )
        residual_rows.append(
            _residual_novelty(
                target_bundle=bundles[20],
                predictor_bundles=(bundles[3], bundles[5], bundles[10]),
                date_values=inputs.date_values,
            )
        )
    first_2023 = int(np.flatnonzero(np.char.startswith(inputs.date_values, "2023-"))[0])
    pre_mask = np.asarray(inputs.candidate_date_idx) < first_2023
    for left_horizon, right_horizon in pairs:
        left = np.asarray(inputs.all_label_values("state", left_horizon), dtype=np.int8)
        right = np.asarray(inputs.all_label_values("state", right_horizon), dtype=np.int8)
        valid = pre_mask & np.isin(left, [0, 1, 2]) & np.isin(right, [0, 1, 2])
        actual_transitions.append(
            {
                "scope": "pre_2023",
                "left_horizon": left_horizon,
                "right_horizon": right_horizon,
                **_contingency_metrics(left[valid], right[valid]),
            }
        )
    return {
        "prediction_pair_metrics": pair_rows,
        "residual_novelty": residual_rows,
        "actual_state_transitions": actual_transitions,
    }


def _g60_novelty_audit(
    *,
    inputs: ShortHorizonInputs,
    output_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for year in FOLD_YEARS:
        bundles = {
            horizon: _prediction_bundle(
                inputs=inputs,
                output_root=output_root,
                year=year,
                horizon=horizon,
                target="g",
            )
            for horizon in (5, 10, 20, 40, 60)
        }
        rows.append(
            _residual_novelty(
                target_bundle=bundles[60],
                predictor_bundles=(bundles[5], bundles[10], bundles[20], bundles[40]),
                date_values=inputs.date_values,
            )
        )
    return rows


def _state_gate(
    *,
    horizon: int,
    calibration: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    current = next(
        item for item in calibration["horizons"] if int(item["horizon"]) == int(horizon)
    )
    annual = list(current["annual"])
    rank_ic = [float(item["raw"]["rank_ic"]) for item in annual]
    top5 = [float(item["raw"]["top_5pct_high_state_lift"]) for item in annual]
    probability = [
        item["raw"] if item["calibrated"] is None else item["calibrated"]
        for item in annual
    ]
    minimum_rank = float(thresholds.get("minimum_worst_year_rank_ic", 0.01))
    gates = {
        "all_years_positive_rank_ic": all(value > 0.0 for value in rank_ic),
        "worst_year_rank_ic": min(rank_ic),
        "worst_year_rank_ic_pass": min(rank_ic) >= minimum_rank,
        "all_years_positive_top5_high_state_lift": all(value > 0.0 for value in top5),
        "sequential_probability_brier_skill_positive": all(
            float(item["binary_brier_skill"]) > 0.0
            and float(item["multiclass_brier_skill"]) > 0.0
            for item in probability
        ),
    }
    gates["qualified"] = bool(
        gates["all_years_positive_rank_ic"]
        and gates["worst_year_rank_ic_pass"]
        and gates["all_years_positive_top5_high_state_lift"]
        and gates["sequential_probability_brier_skill_positive"]
    )
    return gates


def run_posthoc_audit(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    input_root: Path = DEFAULT_INPUT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    training_summary_path = output_root / "training_summary.json"
    if not training_summary_path.is_file():
        raise RuntimeError("short-horizon training must complete before post-hoc audit")
    training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    if training_summary.get("status") != "completed":
        raise RuntimeError("short-horizon training is incomplete")
    inputs = ShortHorizonInputs(study, verify_large_hashes=False)
    decision_thresholds = dict(study["contract"]["decision"]["qualification_thresholds"])
    threshold_payload = json.loads(
        (input_root / "mfe_thresholds.json").read_text(encoding="utf-8")
    )
    threshold_map = {
        int(item["horizon"]): float(item["date_equal_q80"])
        for item in threshold_payload["thresholds"]
    }
    audit_root = output_root / "posthoc"
    audit_root.mkdir(parents=True, exist_ok=True)

    mfe_candidates: list[dict[str, Any]] = []
    for horizon in (3, 5, 10, 20, 40, 60):
        annual: list[dict[str, Any]] = []
        for year in FOLD_YEARS:
            bundle = _prediction_bundle(
                inputs=inputs,
                output_root=output_root,
                year=year,
                horizon=horizon,
                target="mfe",
            )
            row = _ranking_fields(bundle)
            daily, event = _daily_event_enrichment(
                date_idx=bundle.date_idx,
                actual=bundle.actual,
                score=bundle.prediction,
                absolute_threshold=threshold_map[horizon],
                horizon=horizon,
            )
            daily.to_parquet(
                audit_root / f"mfe_h{horizon:02d}_{year}_upper_tail.parquet",
                index=False,
            )
            row.update(event)
            annual.append(row)
        gate = _regression_gate(
            annual, decision_thresholds, require_upper_tail=True
        )
        mfe_candidates.append({"horizon": horizon, "annual": annual, "gate": gate})

    g_candidates: list[dict[str, Any]] = []
    for horizon in (1, 3, 5, 10, 20, 40, 60):
        annual = [
            _ranking_fields(
                _prediction_bundle(
                    inputs=inputs,
                    output_root=output_root,
                    year=year,
                    horizon=horizon,
                    target="g",
                )
            )
            for year in FOLD_YEARS
        ]
        gate = _regression_gate(
            annual, decision_thresholds, require_upper_tail=False
        )
        g_candidates.append({"horizon": horizon, "annual": annual, "gate": gate})

    calibration = _calibration_audit(inputs=inputs, output_root=output_root)
    state_candidates = [
        {
            "horizon": horizon,
            "gate": _state_gate(
                horizon=horizon,
                calibration=calibration,
                thresholds=decision_thresholds,
            ),
        }
        for horizon in (3, 5, 10, 20, 40, 60)
    ]
    state_redundancy = _state_redundancy_audit(
        inputs=inputs, output_root=output_root
    )
    g60_novelty = _g60_novelty_audit(inputs=inputs, output_root=output_root)

    component_rows: list[dict[str, Any]] = []
    component_targets = (
        *(("g", horizon) for horizon in (1, 3, 60)),
        *(("mfe", horizon) for horizon in (3, 5, 10, 20, 40, 60)),
        *(("state", horizon) for horizon in (3, 5, 10, 20, 40, 60)),
        *(("pre_peak_mae", horizon) for horizon in (3, 5, 10, 20, 40)),
    )
    for target, horizon in component_targets:
        for year in FOLD_YEARS:
            component_rows.append(
                _prediction_component_audit(
                    _prediction_bundle(
                        inputs=inputs,
                        output_root=output_root,
                        year=year,
                        horizon=horizon,
                        target=target,
                    )
                )
            )

    d3_atlas = json.loads(
        (input_root / "d3_atlas/d3_atlas_summary.json").read_text(encoding="utf-8")
    )
    mfe_qualified = [
        int(item["horizon"]) for item in mfe_candidates if bool(item["gate"]["qualified"])
    ]
    g_qualified = [
        int(item["horizon"]) for item in g_candidates if bool(item["gate"]["qualified"])
    ]
    state_qualified = [
        int(item["horizon"]) for item in state_candidates if bool(item["gate"]["qualified"])
    ]
    d3_shape_nmi = float(
        d3_atlas["state_vs_g3_daily_tertile"]["normalized_mutual_information"]
    )
    d3_redundancy_cutoff = float(
        decision_thresholds.get("maximum_state_endpoint_tertile_nmi", 0.80)
    )
    g60_residual_supported = sum(
        float(item["residual_rank_ic"])
        >= float(decision_thresholds.get("minimum_residual_rank_ic", 0.01))
        for item in g60_novelty
    ) >= 2
    decision = {
        "mfe_opportunity_target_supported": bool(mfe_qualified),
        "qualified_mfe_horizons": mfe_qualified,
        "qualified_endpoint_return_horizons": g_qualified,
        "qualified_state_horizons_after_sequential_calibration": state_qualified,
        "g1_supported": 1 in g_qualified,
        "g3_supported": 3 in g_qualified,
        "g60_supported": 60 in g_qualified,
        "g60_retains_residual_information_beyond_shorter_g_scores": bool(
            g60_residual_supported
        ),
        "d3_state_not_merely_endpoint_tertiles": bool(d3_shape_nmi < d3_redundancy_cutoff),
        "d3_state_vs_daily_g3_tertile_nmi": d3_shape_nmi,
        "next_feature_audit_targets": {
            "opportunity": [f"mfe_{horizon}" for horizon in mfe_qualified],
            "endpoint": [f"g_{horizon}" for horizon in g_qualified],
            "state": [f"state_{horizon}" for horizon in state_qualified],
            "risk": ["pre_peak_mae_3", "pre_peak_mae_5", "pre_peak_mae_10", "pre_peak_mae_20", "pre_peak_mae_40"],
        },
        "does_not_select": [
            "exit_rule",
            "holding_period",
            "slot_count",
            "leverage",
            "stop_loss",
            "successor_architecture",
        ],
    }
    summary = {
        "schema": "seq100_short_horizon_target_reaudit_summary/v1",
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "contract_sha256": str(study["contract_sha256"]),
        "scope": {
            "fold_years": list(FOLD_YEARS),
            "fold_role": "user-approved reused confirmation/decision window; not a pristine holdout",
            "maximum_consumed_outcome_date": "2025-12-31",
            "forbidden_years_consumed": [],
            "new_booster_count": 15,
            "old_boosters_retrained": 0,
        },
        "d3_path_atlas": d3_atlas,
        "mfe_reassessment": mfe_candidates,
        "endpoint_return_reassessment": g_candidates,
        "state_probability_calibration": calibration,
        "state_candidates": state_candidates,
        "state_horizon_redundancy": state_redundancy,
        "g60_residual_novelty": g60_novelty,
        "market_vs_stock_information": component_rows,
        "decision": decision,
        "disclosure": {
            "2023_2025_reuse": "These years were already used in prior target research. The owner explicitly authorized them as this study's final confirmation/decision window; claims are conditional on that reuse.",
            "2026": "No 2026 row, outcome, feature screening result, calibration fit, evaluation metric, or decision input was read.",
            "economics": "Label learnability does not establish an exit rule, account policy, or tradable return.",
        },
    }
    _write_json(audit_root / "calibration.json", calibration)
    _write_json(audit_root / "state_redundancy.json", state_redundancy)
    _write_json(audit_root / "market_vs_stock.json", component_rows)
    _write_json(output_root / "decision.json", decision)
    _write_json(output_root / "summary.json", summary)
    return summary


def self_test() -> dict[str, Any]:
    left = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int8)
    right = np.asarray([2, 2, 1, 1, 0, 0], dtype=np.int8)
    metrics = _contingency_metrics(left, right)
    if not math.isclose(metrics["adjusted_rand_index"], 1.0, abs_tol=1.0e-12):
        raise AssertionError("contingency ARI drifted")
    values = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    weights = np.ones(4, dtype=np.float64)
    if _weighted_quantile(values, weights, 0.75) != 2.0:
        raise AssertionError("weighted quantile drifted")
    fake_model = {
        "clip_low": np.full(3, -1.0, dtype=np.float32),
        "clip_high": np.full(3, 1.0, dtype=np.float32),
        "center": np.zeros(3, dtype=np.float32),
        "scale": np.ones(3, dtype=np.float32),
        "pca_mean": np.zeros(3, dtype=np.float32),
        "pca_components": np.eye(3, dtype=np.float32),
        "fixed_k3_centers": np.asarray(
            [[-0.1, -0.1, -0.1], [0.0, 0.0, 0.0], [0.1, 0.1, 0.1]],
            dtype=np.float32,
        ),
        "raw_to_state": np.asarray([0, 1, 2], dtype=np.int8),
    }
    assigned = _assign_d3_state(
        np.asarray(
            [[-0.09, -0.09, -0.09], [0.0, 0.0, 0.0], [0.09, 0.09, 0.09]],
            dtype=np.float32,
        ),
        fake_model,
    )
    if assigned.tolist() != [0, 1, 2]:
        raise AssertionError("D3 state assignment drifted")
    payload = {"status": "passed", "test_count": 3}
    _emit("self_test_complete", **payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-audit short-horizon Seq100 targets without reading 2026."
    )
    parser.add_argument("--study-contract", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "command",
        choices=("self-test", "prepare", "preflight", "run", "audit"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "self-test":
        result = self_test()
    elif args.command == "prepare":
        result = prepare_all(args.input_root.resolve())
    elif args.command == "preflight":
        result = preflight(
            study_path=args.study_contract.resolve(),
            output_root=args.output_root.resolve(),
        )
    elif args.command == "run":
        result = run_study(
            study_path=args.study_contract.resolve(),
            output_root=args.output_root.resolve(),
        )
    else:
        result = run_posthoc_audit(
            study_path=args.study_contract.resolve(),
            input_root=args.input_root.resolve(),
            output_root=args.output_root.resolve(),
        )
    print(json.dumps(result, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
