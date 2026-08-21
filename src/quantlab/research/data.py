from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd

from quantlab.core.io import (
    DataContractError,
    read_json,
    sha256_file,
    write_json,
)
from quantlab.core.io import (
    open_array as _open_array,
)
from quantlab.core.paths import paths

DATA_MANIFEST = paths().research_data / "daily" / "manifest.json"
OUTPUT_ROOT = paths().runs / "daily"
DEFAULT_VALIDATION_START_DATE = "2020-01-01"
CUTOFF_VIOLATION_FIELD = "cutoff_violation_count"
LEGACY_CUTOFF_VIOLATION_FIELD = "forbidden_2026_read_count"

FEATURE_FAMILIES: dict[int, tuple[str, ...]] = {
    158: ("daily_price_volume_technical", "market_state"),
    183: (
        "daily_price_volume_technical",
        "market_state",
        "same_day_5m",
    ),
}


class ResearchDataError(DataContractError):
    pass


def _date_text(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ResearchDataError(f"{field} is missing")
    try:
        return pd.Timestamp(text).strftime("%Y-%m-%d")
    except (TypeError, ValueError) as exc:
        raise ResearchDataError(f"{field} is not a date: {text}") from exc


def cutoff_violation_count(record: Mapping[str, Any]) -> int:
    """Read the generic cutoff audit field, with old-run compatibility."""

    if CUTOFF_VIOLATION_FIELD in record:
        value = record[CUTOFF_VIOLATION_FIELD]
    elif LEGACY_CUTOFF_VIOLATION_FIELD in record:
        value = record[LEGACY_CUTOFF_VIOLATION_FIELD]
    else:
        return -1
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ResearchDataError("cutoff violation count is not an integer") from exc


def cutoff_audit_fields(count: int = 0) -> dict[str, int]:
    """Emit the current, calendar-agnostic cutoff audit field.

    Historical run files may still contain the old year-specific field; readers
    accept that alias through :func:`cutoff_violation_count`, but new artifacts
    should not perpetuate it.
    """

    value = int(count)
    return {CUTOFF_VIOLATION_FIELD: value}


def open_array(
    record: Mapping[str, Any], dtype: np.dtype[Any] | type | None = None
) -> np.memmap:
    return _open_array(record, dtype, base=DATA_MANIFEST.parent)


@dataclass(frozen=True)
class ResearchData:
    manifest: dict[str, Any]
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
        return np.asarray(self.manifest["execution"]["date_values"], dtype=str)

    @property
    def maximum_outcome_date(self) -> str:
        return _date_text(
            self.manifest["scope"].get("maximum_outcome_date"),
            field="maximum_outcome_date",
        )

    @property
    def symbol_values(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.manifest["execution"]["symbol_values"])

    @property
    def execution(self) -> dict[str, Any]:
        return dict(self.manifest["execution"])

    @property
    def fingerprints(self) -> dict[str, str]:
        return {
            str(key): str(value)
            for key, value in dict(self.manifest["fingerprints"]).items()
        }

    @property
    def cutoff_idx(self) -> int:
        matches = np.flatnonzero(self.date_values == self.maximum_outcome_date)
        if len(matches) != 1:
            raise ResearchDataError(
                f"{self.maximum_outcome_date} is not unique in the calendar"
            )
        return int(matches[0])

    def feature_count(self, count: int) -> int:
        positions = self.feature_positions(count)
        return len(positions)

    def feature_positions(self, count: int) -> np.ndarray:
        if int(count) not in FEATURE_FAMILIES:
            raise ResearchDataError(f"unsupported feature count: {count}")
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
            raise ResearchDataError(
                f"feature family count mismatch: {count} -> {len(positions)}"
            )
        # The current view is deliberately ordered as daily 104, market 54,
        # minute 25.  Requiring a prefix avoids building another large matrix.
        if not np.array_equal(positions, np.arange(expected, dtype=np.int32)):
            raise ResearchDataError(f"feature set {count} is not a matrix prefix")
        return positions

    def target(self, name: str) -> tuple[np.memmap, np.memmap, int]:
        target = self.manifest["targets"]["exact"]
        columns = list(target["columns"])
        if name not in columns:
            raise ResearchDataError(f"missing exact target: {name}")
        values = open_array(target["values"])
        valid = open_array(target["valid"])
        return values, valid, columns.index(name)

    def legal_exit(
        self, horizon: int
    ) -> tuple[np.memmap, np.memmap, np.memmap, int]:
        target = self.manifest["targets"]["legal_exit"]
        horizons = [int(value) for value in target["horizons"]]
        if int(horizon) not in horizons:
            raise ResearchDataError(f"missing legal exit horizon: {horizon}")
        return (
            open_array(target["values"]),
            open_array(target["valid"]),
            open_array(target["fill_days"]),
            horizons.index(int(horizon)),
        )


def load_data() -> ResearchData:
    manifest = read_json(DATA_MANIFEST)
    if manifest.get("schema") != "quantlab.research_dataset/1":
        raise ResearchDataError("unsupported research dataset schema")
    if manifest.get("status") != "ready":
        raise ResearchDataError("research dataset is not ready")
    scope = dict(manifest["scope"])
    maximum_outcome_date = _date_text(
        scope.get("maximum_outcome_date"), field="maximum_outcome_date"
    )
    if cutoff_violation_count(scope) != 0:
        raise ResearchDataError("research dataset has an invalid outcome boundary")

    row_record = dict(manifest["row_index"])
    row_path = DATA_MANIFEST.parent / str(row_record["path"])
    row_index = pd.read_parquet(row_path)
    required = {
        "candidate_id",
        "date_idx",
        "symbol_idx",
        "trade_date",
        "symbol",
    }
    if not required.issubset(row_index.columns):
        raise ResearchDataError("row index columns are incomplete")
    row_count = int(scope["row_count"])
    feature = dict(manifest["features"])
    matrix_record = dict(feature["matrix"])
    if len(row_index) != row_count or int(matrix_record["shape"][0]) != row_count:
        raise ResearchDataError("row counts disagree")
    dates = row_index["date_idx"].to_numpy(dtype=np.int32, copy=False)
    if np.any(dates[1:] < dates[:-1]):
        raise ResearchDataError("row index is not date sorted")
    trade_dates = row_index["trade_date"].astype(str)
    if trade_dates.max() > maximum_outcome_date:
        raise ResearchDataError(
            f"research rows extend beyond {maximum_outcome_date}"
        )
    if row_index.duplicated(["date_idx", "symbol_idx"]).any():
        raise ResearchDataError("duplicate date/symbol rows")

    feature_names = tuple(str(value) for value in feature["names"])
    families = tuple(str(value) for value in feature["families"])
    if len(feature_names) != 183 or len(set(feature_names)) != 183:
        raise ResearchDataError("183 feature names are not unique and complete")
    matrix = open_array(matrix_record)
    data = ResearchData(
        manifest=manifest,
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
    validation_end_date: str | None = None,
    fold_count: int = 5,
    purge_days: int = 30,
) -> list[dict[str, Any]]:
    dates = np.asarray(date_idx, dtype=np.int32)
    labels = np.asarray(trade_date, dtype=str)
    unique_dates, first = np.unique(dates, return_index=True)
    unique_labels = labels[first]
    if validation_end_date is None:
        validation_end_date = max(str(value) for value in unique_labels)
    selected = unique_dates[
        (unique_labels >= str(validation_start_date))
        & (unique_labels <= str(validation_end_date))
    ]
    blocks = np.array_split(selected, int(fold_count))
    folds: list[dict[str, Any]] = []
    for fold_number, block in enumerate(blocks, start=1):
        if not len(block):
            raise ResearchDataError("empty validation fold")
        start_idx = int(block[0])
        end_idx = int(block[-1])
        train_max = start_idx - int(purge_days) - 1
        train_rows = int(np.sum(dates <= train_max))
        evaluation_rows = int(np.sum((dates >= start_idx) & (dates <= end_idx)))
        if train_rows <= 0 or evaluation_rows <= 0:
            raise ResearchDataError("fold support is empty")
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
        raise ResearchDataError(f"unknown split: {split}")
    rows = np.flatnonzero(mask).astype(np.int64)
    if not len(rows) or int(rows[-1]) - int(rows[0]) + 1 != len(rows):
        raise ResearchDataError(f"{split} rows are not one contiguous range")
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
        raise ResearchDataError("ranking dates are empty")
    if np.any(current[1:] < current[:-1]):
        raise ResearchDataError("ranking dates are not sorted")
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
    finite_counts = np.zeros(183, dtype=np.int64)
    inf_counts = np.zeros(183, dtype=np.int64)
    all_empty_rows = 0
    for left in range(0, data.row_count, int(chunk_rows)):
        right = min(left + int(chunk_rows), data.row_count)
        current = np.asarray(data.matrix[left:right], dtype=np.float32)
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

    matrix_path = DATA_MANIFEST.parent / data.manifest["features"]["matrix"]["path"]
    matrix_sha256 = sha256_file(matrix_path)
    result = {
        "status": "ok",
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_fingerprint": data.fingerprints["input"],
        "view_fingerprint": data.fingerprints["features"],
        "target_fingerprint": data.fingerprints["exact_targets"],
        "row_count": data.row_count,
        "feature_count": 183,
        "matrix_sha256": matrix_sha256,
        "row_key_unique": True,
        "maximum_trade_date": str(data.row_index["trade_date"].astype(str).max()),
        **cutoff_audit_fields(),
        "source_projection_evidence": bool(
            data.manifest["quality"].get("source_557_projection_equal")
        ),
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
        matrix_sha256 != data.manifest["features"]["matrix"]["sha256"]
        or result["infinite_value_count"]
        or result["all_empty_row_count"]
        or any(item["outside_count"] for item in ranges.values())
    ):
        result["status"] = "failed"
        raise ResearchDataError(f"research data quality failed: {result}")
    write_json(OUTPUT_ROOT / "quality.json", result)
    return result
