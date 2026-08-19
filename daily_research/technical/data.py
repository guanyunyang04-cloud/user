from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MODEL_INPUT_MANIFEST = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies"
    / "seq100_quality_liquidity_training_ready/model_inputs/manifest.json"
)
FEATURE_VIEW_MANIFEST = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies"
    / "seq100_full_market_multitask_forecast_v1"
    / "feature_views/price_path_core_183/manifest.json"
)
EXACT_TARGET_MANIFEST = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies"
    / "seq100_full_market_multitask_forecast_v1/exact_net_targets/manifest.json"
)
PATH_TARGET_MANIFEST = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies"
    / "seq100_full_market_multitask_forecast_v1/targets/manifest.json"
)
PACK_MANIFEST = (
    WORKSPACE_ROOT
    / "daily_research/data/research_store/seq100_pit_l35v2_v1/pack/manifest.json"
)
OUTPUT_ROOT = WORKSPACE_ROOT / "daily_research/output/technical"
MAXIMUM_OUTCOME_DATE = "2025-12-31"

FEATURE_FAMILIES: dict[int, tuple[str, ...]] = {
    158: ("daily_price_volume_technical", "market_state"),
    183: (
        "daily_price_volume_technical",
        "market_state",
        "same_day_5m",
    ),
}


class TechnicalDataError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TechnicalDataError(f"missing JSON: {path}")
    return dict(json.loads(path.read_text(encoding="utf-8")))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    partial.replace(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(type(value).__name__)


def stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path, *, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def open_array(record: Mapping[str, Any], dtype: np.dtype[Any] | type) -> np.memmap:
    shape = tuple(int(value) for value in record["shape"])
    path = Path(str(record["path"]))
    expected = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    if not path.is_file() or path.stat().st_size != expected:
        raise TechnicalDataError(f"array size mismatch: {path}")
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


@dataclass(frozen=True)
class TechnicalData:
    model_manifest: dict[str, Any]
    view_manifest: dict[str, Any]
    target_manifest: dict[str, Any]
    path_target_manifest: dict[str, Any]
    pack: dict[str, Any]
    row_index: pd.DataFrame
    matrix: np.memmap
    feature_names: tuple[str, ...]
    feature_families: tuple[str, ...]

    @property
    def row_count(self) -> int:
        return len(self.row_index)

    @property
    def dates(self) -> np.ndarray:
        return self.row_index["date_idx"].to_numpy(dtype=np.int32, copy=False)

    @property
    def date_values(self) -> np.ndarray:
        return np.asarray(self.pack["date_values"], dtype=str)

    @property
    def cutoff_idx(self) -> int:
        matches = np.flatnonzero(self.date_values == MAXIMUM_OUTCOME_DATE)
        if len(matches) != 1:
            raise TechnicalDataError("2025-12-31 is not unique in pack calendar")
        return int(matches[0])

    def feature_count(self, count: int) -> int:
        positions = self.feature_positions(count)
        return len(positions)

    def feature_positions(self, count: int) -> np.ndarray:
        if int(count) not in FEATURE_FAMILIES:
            raise TechnicalDataError(f"unsupported feature count: {count}")
        families = set(FEATURE_FAMILIES[int(count)])
        positions = np.asarray(
            [
                index
                for index, family in enumerate(self.feature_families)
                if family in families
            ],
            dtype=np.int32,
        )
        expected = int(count)
        if len(positions) != expected:
            raise TechnicalDataError(
                f"feature family count mismatch: {count} -> {len(positions)}"
            )
        # The current view is deliberately ordered as daily 104, market 54,
        # minute 25.  Requiring a prefix avoids building another large matrix.
        if not np.array_equal(positions, np.arange(expected, dtype=np.int32)):
            raise TechnicalDataError(f"feature set {count} is not a matrix prefix")
        return positions

    def target(self, name: str) -> tuple[np.memmap, np.memmap, int]:
        columns = list(self.target_manifest["target_columns"])
        if name not in columns:
            raise TechnicalDataError(f"missing exact target: {name}")
        values = open_array(self.target_manifest["files"]["values"], np.float32)
        valid = open_array(self.target_manifest["files"]["valid"], np.uint8)
        return values, valid, columns.index(name)


def load_data() -> TechnicalData:
    model = read_json(MODEL_INPUT_MANIFEST)
    view = read_json(FEATURE_VIEW_MANIFEST)
    targets = read_json(EXACT_TARGET_MANIFEST)
    path_targets = read_json(PATH_TARGET_MANIFEST)
    pack = read_json(PACK_MANIFEST)

    artifacts = {
        "model input": model,
        "183 view": view,
        "exact targets": targets,
        "path targets": path_targets,
    }
    incomplete = [
        name for name, artifact in artifacts.items() if artifact.get("status") != "completed"
    ]
    if incomplete:
        raise TechnicalDataError(f"current artifacts are incomplete: {incomplete}")
    input_fingerprint = str(model["input_fingerprint"])
    target_input = dict(targets["sources"]["model_inputs"])
    path_input = dict(path_targets["sources"]["model_inputs"])
    exact_path_source = dict(targets["sources"]["path_targets"])
    if (
        target_input.get("input_fingerprint") != input_fingerprint
        or path_input.get("input_fingerprint") != input_fingerprint
    ):
        raise TechnicalDataError("targets are bound to stale model inputs")
    if exact_path_source.get("target_fingerprint") != path_targets.get(
        "fingerprint"
    ):
        raise TechnicalDataError("exact targets are bound to stale path targets")
    pack_sha = sha256_file(PACK_MANIFEST)
    if any(
        artifact["sources"]["pack"].get("sha256") != pack_sha
        for artifact in (targets, path_targets)
    ):
        raise TechnicalDataError("targets are bound to a stale execution pack")
    if any(
        int(artifact["contract"].get("forbidden_2026_read_count", -1)) != 0
        for artifact in (targets, path_targets)
    ):
        raise TechnicalDataError("targets read forbidden 2026 outcomes")

    row_index = pd.read_parquet(Path(model["row_index"]["path"]))
    required = {
        "candidate_id",
        "date_idx",
        "symbol_idx",
        "trade_date",
        "symbol",
    }
    if not required.issubset(row_index.columns):
        raise TechnicalDataError("row index columns are incomplete")
    row_count = int(model["row_count"])
    if (
        len(row_index) != row_count
        or int(view["shape"][0]) != row_count
        or int(targets["row_count"]) != row_count
        or int(path_targets["row_count"]) != row_count
    ):
        raise TechnicalDataError("row counts disagree")
    dates = row_index["date_idx"].to_numpy(dtype=np.int32, copy=False)
    if np.any(dates[1:] < dates[:-1]):
        raise TechnicalDataError("row index is not date sorted")
    trade_dates = row_index["trade_date"].astype(str)
    if (
        trade_dates.max() > MAXIMUM_OUTCOME_DATE
        or trade_dates.str.startswith("2026").any()
    ):
        raise TechnicalDataError("research rows extend beyond 2025-12-31")
    if row_index.duplicated(["date_idx", "symbol_idx"]).any():
        raise TechnicalDataError("duplicate date/symbol rows")

    records_by_column = {
        int(record["column_index"]): dict(record) for record in model["features"]
    }
    source_columns = [int(value) for value in view["source_columns"]]
    try:
        selected = [records_by_column[column] for column in source_columns]
    except KeyError as exc:
        raise TechnicalDataError(f"view source column is missing: {exc}") from exc
    feature_names = tuple(str(record["feature_name"]) for record in selected)
    families = tuple(str(record["analytic_family"]) for record in selected)
    if len(feature_names) != 183 or len(set(feature_names)) != 183:
        raise TechnicalDataError("183 feature names are not unique and complete")
    expected_view_fingerprint = stable_hash(
        {
            "schema": "seq100_full_market_feature_view/1",
            "model_input_fingerprint": input_fingerprint,
            "source_sample_hash": model["storage"]["compact"]["sample_hash"],
            "feature_variant": view["feature_variant"],
            "feature_names": feature_names,
            "source_columns": source_columns,
        }
    )
    if view.get("fingerprint") != expected_view_fingerprint:
        raise TechnicalDataError("183 view is bound to stale model inputs")

    view_record = dict(view["file"])
    view_record["shape"] = list(view["shape"])
    matrix = open_array(view_record, np.float32)
    compact = open_array(model["storage"]["compact"], np.float32)
    sample_rows = np.asarray(view["sample_rows"], dtype=np.int64)
    sample = np.asarray(matrix[sample_rows], dtype=np.float32)
    projected = np.asarray(compact[sample_rows][:, source_columns], dtype=np.float32)
    if not np.array_equal(sample, projected, equal_nan=True):
        raise TechnicalDataError("183 sample does not match current 557 projection")
    sample_hash = hashlib.sha256(sample.tobytes()).hexdigest()
    if sample_hash != view.get("sample_hash"):
        raise TechnicalDataError("183 sample hash mismatch")
    data = TechnicalData(
        model_manifest=model,
        view_manifest=view,
        target_manifest=targets,
        path_target_manifest=path_targets,
        pack=pack,
        row_index=row_index,
        matrix=matrix,
        feature_names=feature_names,
        feature_families=families,
    )
    data.feature_positions(158)
    data.feature_positions(183)
    return data


def build_forward_folds(
    *,
    date_idx: np.ndarray,
    trade_date: np.ndarray,
    validation_start_date: str = "2020-01-01",
    validation_end_date: str = MAXIMUM_OUTCOME_DATE,
    fold_count: int = 5,
    purge_days: int = 30,
) -> list[dict[str, Any]]:
    dates = np.asarray(date_idx, dtype=np.int32)
    labels = np.asarray(trade_date, dtype=str)
    unique_dates, first = np.unique(dates, return_index=True)
    unique_labels = labels[first]
    selected = unique_dates[
        (unique_labels >= str(validation_start_date))
        & (unique_labels <= str(validation_end_date))
    ]
    blocks = np.array_split(selected, int(fold_count))
    folds: list[dict[str, Any]] = []
    for fold_number, block in enumerate(blocks, start=1):
        if not len(block):
            raise TechnicalDataError("empty validation fold")
        start_idx = int(block[0])
        end_idx = int(block[-1])
        train_max = start_idx - int(purge_days) - 1
        train_rows = int(np.sum(dates <= train_max))
        evaluation_rows = int(np.sum((dates >= start_idx) & (dates <= end_idx)))
        if train_rows <= 0 or evaluation_rows <= 0:
            raise TechnicalDataError("fold support is empty")
        train_label_position = (
            np.searchsorted(unique_dates, train_max, side="right") - 1
        )
        folds.append(
            {
                "fold": fold_number,
                "validation_start_date_idx": start_idx,
                "validation_end_date_idx": end_idx,
                "validation_start_date": str(
                    unique_labels[np.searchsorted(unique_dates, start_idx)]
                ),
                "validation_end_date": str(
                    unique_labels[np.searchsorted(unique_dates, end_idx)]
                ),
                "validation_date_count": len(block),
                "training_maximum_date_idx": train_max,
                "training_maximum_date": str(unique_labels[train_label_position]),
                "training_row_count": train_rows,
                "validation_row_count": evaluation_rows,
                "purge_days": int(purge_days),
            }
        )
    return folds


def fold_rows(dates: np.ndarray, fold: Mapping[str, Any], split: str) -> np.ndarray:
    current = np.asarray(dates, dtype=np.int32)
    if split == "train":
        mask = current <= int(fold["training_maximum_date_idx"])
    elif split == "validation":
        mask = (current >= int(fold["validation_start_date_idx"])) & (
            current <= int(fold["validation_end_date_idx"])
        )
    else:
        raise TechnicalDataError(f"unknown split: {split}")
    rows = np.flatnonzero(mask).astype(np.int64)
    if not len(rows) or int(rows[-1]) - int(rows[0]) + 1 != len(rows):
        raise TechnicalDataError(f"{split} rows are not one contiguous range")
    return rows


def equal_date_weights(dates: np.ndarray, valid: np.ndarray) -> np.ndarray:
    current_dates = np.asarray(dates, dtype=np.int32)
    current_valid = np.asarray(valid, dtype=bool)
    output = np.zeros(len(current_dates), dtype=np.float32)
    if not current_valid.any():
        return output
    _, inverse, counts = np.unique(
        current_dates[current_valid], return_inverse=True, return_counts=True
    )
    weights = 1.0 / counts[inverse].astype(np.float64)
    weights *= len(weights) / weights.sum()
    output[np.flatnonzero(current_valid)] = weights.astype(np.float32)
    return output


def date_group_sizes(dates: np.ndarray) -> np.ndarray:
    current = np.asarray(dates, dtype=np.int32)
    if current.ndim != 1 or not len(current):
        raise TechnicalDataError("ranking dates are empty")
    if np.any(current[1:] < current[:-1]):
        raise TechnicalDataError("ranking dates are not sorted")
    boundaries = np.flatnonzero(np.r_[True, current[1:] != current[:-1], True])
    return np.diff(boundaries).astype(np.int32)


def date_relevance_labels(
    dates: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    *,
    levels: int = 10,
) -> np.ndarray:
    current_dates = np.asarray(dates, dtype=np.int32)
    current_values = np.asarray(values, dtype=np.float64)
    current_valid = np.asarray(valid, dtype=bool) & np.isfinite(current_values)
    output = np.zeros(len(current_dates), dtype=np.float32)
    boundaries = np.flatnonzero(
        np.r_[True, current_dates[1:] != current_dates[:-1], True]
    )
    for left, right in pairwise(boundaries):
        positions = left + np.flatnonzero(current_valid[left:right])
        if len(positions) < int(levels):
            continue
        order = np.argsort(current_values[positions], kind="stable")
        relevance = np.floor(
            np.arange(len(positions), dtype=np.float64) * int(levels) / len(positions)
        ).astype(np.float32)
        output[positions[order]] = relevance
    return output


def daily_rank_metrics(
    *, dates: np.ndarray, actual: np.ndarray, prediction: np.ndarray
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.DataFrame(
        {
            "date_idx": np.asarray(dates, dtype=np.int32),
            "actual": np.asarray(actual, dtype=np.float64),
            "prediction": np.asarray(prediction, dtype=np.float64),
        }
    ).dropna()
    frame["prediction_rank"] = frame.groupby("date_idx")["prediction"].rank(pct=True)
    frame["actual_rank"] = frame.groupby("date_idx")["actual"].rank(pct=True)
    daily = frame.groupby("date_idx").agg(
        baseline_actual=("actual", "mean"), row_count=("actual", "size")
    )
    for name, fraction in (("top1pct", 0.01), ("top5pct", 0.05)):
        selected = frame[frame["prediction_rank"] >= 1.0 - fraction]
        daily = daily.join(
            selected.groupby("date_idx")["actual"].mean().rename(f"{name}_actual"),
            how="left",
        )
    rank_ic = frame.groupby("date_idx")[["prediction_rank", "actual_rank"]].corr()
    rank_ic = rank_ic.iloc[0::2, -1]
    daily["rank_ic"] = rank_ic.to_numpy(dtype=np.float64)
    error = frame["prediction"].to_numpy() - frame["actual"].to_numpy()
    metrics = {
        "row_count": len(frame),
        "date_count": len(daily),
        "daily_rank_ic_mean": float(daily["rank_ic"].mean()),
        "daily_rank_ic_positive_fraction": float(daily["rank_ic"].gt(0.0).mean()),
        "top1pct_actual_mean": float(daily["top1pct_actual"].mean()),
        "top5pct_actual_mean": float(daily["top5pct_actual"].mean()),
        "baseline_actual_mean": float(daily["baseline_actual"].mean()),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
    }
    return daily.reset_index(), metrics


def verify_current_data(*, chunk_rows: int = 32_768) -> dict[str, Any]:
    data = load_data()
    compact_record = dict(data.model_manifest["storage"]["compact"])
    compact = open_array(compact_record, np.float32)
    source_columns = np.asarray(data.view_manifest["source_columns"], dtype=np.int32)
    finite_counts = np.zeros(183, dtype=np.int64)
    inf_counts = np.zeros(183, dtype=np.int64)
    all_empty_rows = 0
    projection_equal = True
    for left in range(0, data.row_count, int(chunk_rows)):
        right = min(left + int(chunk_rows), data.row_count)
        current = np.asarray(data.matrix[left:right], dtype=np.float32)
        projected = np.asarray(compact[left:right, source_columns], dtype=np.float32)
        if not np.array_equal(current, projected, equal_nan=True):
            projection_equal = False
        finite = np.isfinite(current)
        finite_counts += finite.sum(axis=0)
        inf_counts += np.isinf(current).sum(axis=0)
        all_empty_rows += int((~finite).all(axis=1).sum())

    fixed_ranges: dict[str, tuple[float, float]] = {
        "minute_trend_efficiency": (0.0, 1.0),
        "minute_high_time_fraction": (0.0, 1.0),
        "minute_low_time_fraction": (0.0, 1.0),
    }
    ranges: dict[str, Any] = {}
    for name, (lower, upper) in fixed_ranges.items():
        column = data.feature_names.index(name)
        values = np.asarray(data.matrix[:, column], dtype=np.float32)
        finite = values[np.isfinite(values)]
        outside = int(((finite < lower - 1.0e-6) | (finite > upper + 1.0e-6)).sum())
        ranges[name] = {
            "minimum": float(finite.min()),
            "maximum": float(finite.max()),
            "outside_count": outside,
        }

    matrix_path = Path(data.view_manifest["file"]["path"])
    result = {
        "status": "ok",
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_fingerprint": data.model_manifest["input_fingerprint"],
        "view_fingerprint": data.view_manifest["fingerprint"],
        "target_fingerprint": data.target_manifest["fingerprint"],
        "row_count": data.row_count,
        "feature_count": 183,
        "matrix_sha256": sha256_file(matrix_path),
        "row_key_unique": True,
        "maximum_trade_date": str(data.row_index["trade_date"].astype(str).max()),
        "forbidden_2026_read_count": 0,
        "projection_557_to_183_equal": projection_equal,
        "infinite_value_count": int(inf_counts.sum()),
        "all_empty_row_count": int(all_empty_rows),
        "minimum_feature_coverage": float(finite_counts.min() / data.row_count),
        "minimum_coverage_feature": data.feature_names[int(np.argmin(finite_counts))],
        "fixed_ranges": ranges,
        "feature_sets": {
            str(count): {
                "families": list(FEATURE_FAMILIES[count]),
                "feature_count": data.feature_count(count),
                "is_prefix_of_183": True,
            }
            for count in FEATURE_FAMILIES
        },
    }
    if (
        not projection_equal
        or result["infinite_value_count"]
        or result["all_empty_row_count"]
        or any(item["outside_count"] for item in ranges.values())
    ):
        result["status"] = "failed"
        raise TechnicalDataError(json.dumps(result, ensure_ascii=False))
    write_json(OUTPUT_ROOT / "quality.json", result)
    return result
