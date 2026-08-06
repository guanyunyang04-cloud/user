from __future__ import annotations

"""Certify the canonical 2012-2025 QDP and 557-feature model input."""

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from quant_data_platform.qdp_v2.database_audit import audit_database_as_of
from quant_data_platform.qdp_v2.manifest import (
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
)
from quant_data_platform.qdp_v2.status import active_dataset_map

from daily_research.path_policy import seq100_quality_liquidity_model as model
from daily_research.path_policy import seq100_quality_liquidity_training_ready as ready

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_quality_liquidity_data_certification"
BUILDER_VERSION = "seq100_quality_liquidity_data_certification/2"
AS_OF_DATE = "2025-12-31"
RESEARCH_START = "2012-01-01"
YEARS = tuple(range(2012, 2026))
EXPECTED_ROW_COUNT = 4_191_476
EXPECTED_FEATURE_COUNT = 557
QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
UNSTABLE_BALANCE_FEATURES = (
    "balance_notes_receivable",
    "balance_accounts_receivable",
    "balance_receivables_ratio",
    "balance_fixed_assets",
    "balance_construction_in_progress",
    "balance_notes_payable",
    "balance_accounts_payable",
)
STABLE_BALANCE_FEATURES = (
    "balance_trade_receivables_total",
    "balance_trade_receivables_to_current_assets",
    "balance_trade_receivables_to_total_assets",
    "balance_fixed_assets_measure",
    "balance_construction_in_progress_measure",
    "balance_trade_payables_total",
    "balance_trade_payables_to_current_liabilities",
    "balance_trade_payables_to_total_assets",
)
SOURCE_ERA_UNSTABLE_FEATURES = ("income_discontinued_net_income",)
NEXT_OPEN_DOMAINS = (
    "announcement",
    "balance_sheet_quarterly",
    "cash_flow_statement_quarterly",
    "income_statement_quarterly",
    "margin_detail",
    "margin_eligibility",
    "margin_market",
    "moneyflow_raw",
    "research_report",
    "research_report_forecast",
)
DEFAULT_READY_ROOT = ready.DEFAULT_OUTPUT_ROOT
DEFAULT_OUTPUT_ROOT = DEFAULT_READY_ROOT / "certification"


class CertificationError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CertificationError(f"required_json_missing:{path}")
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path, **extra: Any) -> dict[str, Any]:
    result = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size": int(path.stat().st_size),
    }
    result.update(extra)
    return result


def _q(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _paths_for_dataset(
    workspace: Path, dataset_ids: Mapping[str, str], domain: str
) -> list[Path]:
    root = qdp_v2_root(workspace)
    dataset_id = str(dataset_ids.get(domain, ""))
    manifest_path = dataset_manifest_for_id(root, dataset_id, domain)
    if manifest_path is None:
        raise CertificationError(f"bound_qdp_dataset_missing:{domain}:{dataset_id}")
    manifest = read_dataset_manifest(manifest_path)
    paths = [resolve_manifest_path(item.path, root=root) for item in manifest.shards]
    if not paths or not all(path.is_file() for path in paths):
        raise CertificationError(f"bound_qdp_shards_missing:{domain}")
    return paths


def _source_bindings(
    *, workspace: Path, ready_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    bound = {
        str(domain): str(dataset_id)
        for domain, dataset_id in dict(
            ready_manifest.get("qdp_dataset_ids", {}) or {}
        ).items()
    }
    active = active_dataset_map(read_active_manifest(qdp_v2_root(workspace)))
    mismatches = {
        domain: {"bound": dataset_id, "active": active.get(domain, "")}
        for domain, dataset_id in bound.items()
        if active.get(domain) != dataset_id
    }
    return {
        "bound_dataset_count": len(bound),
        "all_bound_datasets_still_active": not mismatches,
        "mismatches": mismatches,
        "dataset_ids": bound,
    }


def _row_index_checks(
    *, row_index: pd.DataFrame, ready_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    dates = row_index["trade_date"].astype(str)
    years = dates.str.slice(0, 4).astype(int)
    actual_by_year = {
        str(year): int((years == year).sum()) for year in sorted(years.unique())
    }
    expected_by_year = {
        str(year): int(ready_manifest["row_spine"][str(year)]["row_count"])
        for year in YEARS
    }
    candidate_ids = row_index["candidate_id"].to_numpy(dtype=np.int64)
    date_idx = row_index["date_idx"].to_numpy(dtype=np.int64)
    return {
        "row_count": len(row_index),
        "row_count_exact": len(row_index) == EXPECTED_ROW_COUNT,
        "columns_exact": list(row_index.columns)
        == [
            "candidate_id",
            "date_idx",
            "symbol_idx",
            "trade_date",
            "symbol",
            "security_id",
            "legacy_candidate_row",
        ],
        "candidate_id_unique": bool(row_index["candidate_id"].is_unique),
        "candidate_id_strictly_increasing": bool(
            len(candidate_ids) < 2 or np.all(candidate_ids[1:] > candidate_ids[:-1])
        ),
        "date_idx_nondecreasing": bool(
            len(date_idx) < 2 or np.all(date_idx[1:] >= date_idx[:-1])
        ),
        "min_date": str(dates.min()),
        "max_date": str(dates.max()),
        "starts_in_2012": str(dates.min()) == "2012-01-04",
        "ends_at_cutoff": str(dates.max()) == AS_OF_DATE,
        "formal_years_exact": tuple(sorted(years.unique())) == YEARS,
        "burn_in_formal_row_count": int(years.isin([2010, 2011]).sum()),
        "forbidden_2026_row_count": int((years >= 2026).sum()),
        "rows_by_year": actual_by_year,
        "rows_by_year_match_spines": actual_by_year == expected_by_year,
    }


def _feature_contract_checks(input_manifest: Mapping[str, Any]) -> dict[str, Any]:
    names = list(input_manifest["feature_groups"][model.COMPACT_VARIANT])
    forbidden_exact = set(ready.FORBIDDEN_INHERITED_METADATA_COLUMNS) | {
        "candidate_id",
        "date_idx",
        "symbol_idx",
        "trade_date",
        "symbol",
        "security_id",
        "legacy_candidate_row",
    }
    forbidden_prefixes = ("future_", "label_", "entry_", "outcome_")
    forbidden = sorted(
        name
        for name in names
        if name in forbidden_exact or name.lower().startswith(forbidden_prefixes)
    )
    adjusted = sorted(
        name
        for name in names
        if "qfq" in name.lower() or "hfq" in name.lower()
    )
    unstable_balance = sorted(set(names).intersection(UNSTABLE_BALANCE_FEATURES))
    stable_balance_missing = sorted(set(STABLE_BALANCE_FEATURES).difference(names))
    source_era_unstable = sorted(
        set(names).intersection(SOURCE_ERA_UNSTABLE_FEATURES)
    )
    return {
        "feature_count": len(names),
        "feature_count_exact": len(names) == EXPECTED_FEATURE_COUNT,
        "feature_names_unique": len(names) == len(set(names)),
        "forbidden_metadata_features": forbidden,
        "future_or_label_metadata_absent": not forbidden,
        "qfq_hfq_features": adjusted,
        "qfq_hfq_features_absent": not adjusted,
        "unstable_balance_component_features": unstable_balance,
        "unstable_balance_component_features_absent": not unstable_balance,
        "stable_balance_features_missing": stable_balance_missing,
        "stable_balance_features_present": not stable_balance_missing,
        "source_era_unstable_features": source_era_unstable,
        "source_era_unstable_features_absent": not source_era_unstable,
        "training_performed": bool(input_manifest.get("training_performed", True)),
        "feature_set_selected": bool(input_manifest.get("feature_set_selected", True)),
    }


def _bernoulli_js_distance(left: float, right: float) -> float:
    if not np.isfinite(left) or not np.isfinite(right):
        return float("nan")
    left = float(np.clip(left, 0.0, 1.0))
    right = float(np.clip(right, 0.0, 1.0))
    middle = 0.5 * (left + right)

    def divergence(value: float) -> float:
        result = 0.0
        if value > 0.0:
            result += value * np.log(value / middle)
        if value < 1.0:
            result += (1.0 - value) * np.log((1.0 - value) / (1.0 - middle))
        return result

    return float(np.sqrt(max(0.0, 0.5 * (divergence(left) + divergence(right)))))


def _maximum_adjacent(
    values: Sequence[float], years: Sequence[int]
) -> tuple[float, str]:
    best_value = float("nan")
    best_years = ""
    for index in range(1, len(values)):
        previous = float(values[index - 1])
        current = float(values[index])
        if not np.isfinite(previous) or not np.isfinite(current):
            continue
        difference = abs(current - previous)
        if not np.isfinite(best_value) or difference > best_value:
            best_value = difference
            best_years = f"{int(years[index - 1])}->{int(years[index])}"
    return best_value, best_years


def _maximum_metric(
    values: Sequence[float], labels: Sequence[str]
) -> tuple[float, str]:
    best_value = float("nan")
    best_label = ""
    for value, label in zip(values, labels, strict=True):
        current = float(value)
        if not np.isfinite(current):
            continue
        if not np.isfinite(best_value) or current > best_value:
            best_value = current
            best_label = str(label)
    return best_value, best_label


def _global_robust_scale(ordered: pd.DataFrame, *, prefix: str = "") -> float:
    q05 = ordered[f"{prefix}q05"].to_numpy(dtype=np.float64)
    q25 = ordered[f"{prefix}q25"].to_numpy(dtype=np.float64)
    q50 = ordered[f"{prefix}q50"].to_numpy(dtype=np.float64)
    q75 = ordered[f"{prefix}q75"].to_numpy(dtype=np.float64)
    q95 = ordered[f"{prefix}q95"].to_numpy(dtype=np.float64)
    iqr = q75 - q25
    central_90_equivalent_iqr = (q95 - q05) / 2.44
    candidates = np.concatenate(
        [
            iqr[np.isfinite(iqr) & (iqr > 0.0)],
            central_90_equivalent_iqr[
                np.isfinite(central_90_equivalent_iqr)
                & (central_90_equivalent_iqr > 0.0)
            ],
        ]
    )
    distribution_scale = float(np.nanmedian(candidates)) if len(candidates) else 0.0
    finite_level = np.concatenate(
        [
            np.abs(q05[np.isfinite(q05)]),
            np.abs(q50[np.isfinite(q50)]),
            np.abs(q95[np.isfinite(q95)]),
        ]
    )
    level = float(np.nanmedian(finite_level)) if len(finite_level) else 0.0
    relative_floor = max(level, 1.0) * 0.01
    return max(distribution_scale, relative_floor)


def _feature_matrix_quality(
    *,
    values: np.memmap,
    row_index: pd.DataFrame,
    feature_names: Sequence[str],
    chunk_rows: int = 8192,
    quantile_sample_rows_per_year: int = 50_000,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    feature_count = len(feature_names)
    annual_rows: list[dict[str, Any]] = []
    all_missing_row_count = 0
    any_infinite_row_count = 0
    for year in YEARS:
        positions = np.flatnonzero(
            row_index["trade_date"]
            .astype(str)
            .str.startswith(f"{year}-")
            .to_numpy()
        )
        if not len(positions):
            raise CertificationError(f"feature_year_empty:{year}")
        if not np.array_equal(
            positions,
            np.arange(positions[0], positions[-1] + 1, dtype=np.int64),
        ):
            raise CertificationError(f"feature_year_not_contiguous:{year}")
        finite_count = np.zeros(feature_count, dtype=np.int64)
        nan_count = np.zeros(feature_count, dtype=np.int64)
        posinf_count = np.zeros(feature_count, dtype=np.int64)
        neginf_count = np.zeros(feature_count, dtype=np.int64)
        zero_count = np.zeros(feature_count, dtype=np.int64)
        non_binary_count = np.zeros(feature_count, dtype=np.int64)
        minimum = np.full(feature_count, np.inf, dtype=np.float64)
        maximum = np.full(feature_count, -np.inf, dtype=np.float64)
        for left in range(int(positions[0]), int(positions[-1]) + 1, chunk_rows):
            right = min(left + chunk_rows, int(positions[-1]) + 1)
            block = np.asarray(values[left:right], dtype=np.float32)
            finite = np.isfinite(block)
            nan_count += np.count_nonzero(np.isnan(block), axis=0)
            posinf_count += np.count_nonzero(np.isposinf(block), axis=0)
            neginf_count += np.count_nonzero(np.isneginf(block), axis=0)
            finite_count += np.count_nonzero(finite, axis=0)
            zero_count += np.count_nonzero(finite & (block == 0), axis=0)
            non_binary_count += np.count_nonzero(
                finite & (block != 0) & (block != 1), axis=0
            )
            safe_min = np.where(finite, block, np.inf).min(axis=0)
            safe_max = np.where(finite, block, -np.inf).max(axis=0)
            minimum = np.minimum(minimum, safe_min)
            maximum = np.maximum(maximum, safe_max)
            all_missing_row_count += int(np.count_nonzero(~finite.any(axis=1)))
            any_infinite_row_count += int(np.count_nonzero(np.isinf(block).any(axis=1)))

        sample_count = min(len(positions), int(quantile_sample_rows_per_year))
        sample_local = np.linspace(
            0, len(positions) - 1, num=sample_count, dtype=np.int64
        )
        sample_positions = positions[sample_local]
        quantile_values = np.full(
            (len(QUANTILES), feature_count), np.nan, dtype=np.float64
        )
        nonzero_quantile_values = np.full(
            (len(QUANTILES), feature_count), np.nan, dtype=np.float64
        )
        nonzero_sample_count = np.zeros(feature_count, dtype=np.int64)
        for start_column in range(0, feature_count, 32):
            stop_column = min(start_column + 32, feature_count)
            sample = np.asarray(
                values[
                    np.ix_(
                        sample_positions,
                        np.arange(start_column, stop_column, dtype=np.int64),
                    )
                ],
                dtype=np.float64,
            )
            sample[~np.isfinite(sample)] = np.nan
            nonzero_sample = sample.copy()
            nonzero_sample[nonzero_sample == 0.0] = np.nan
            nonzero_sample_count[start_column:stop_column] = np.count_nonzero(
                np.isfinite(nonzero_sample), axis=0
            )
            finite_columns = np.isfinite(sample).any(axis=0)
            if bool(finite_columns.any()):
                with np.errstate(all="ignore"):
                    quantile_values[
                        :, start_column + np.flatnonzero(finite_columns)
                    ] = np.nanquantile(
                        sample[:, finite_columns], QUANTILES, axis=0
                    )
            nonzero_columns = np.isfinite(nonzero_sample).any(axis=0)
            if bool(nonzero_columns.any()):
                with np.errstate(all="ignore"):
                    nonzero_quantile_values[
                        :, start_column + np.flatnonzero(nonzero_columns)
                    ] = np.nanquantile(
                        nonzero_sample[:, nonzero_columns], QUANTILES, axis=0
                    )
        row_count = len(positions)
        for column, name in enumerate(feature_names):
            finite = int(finite_count[column])
            annual_rows.append(
                {
                    "year": year,
                    "feature": str(name),
                    "row_count": row_count,
                    "finite_count": finite,
                    "finite_rate": float(finite / row_count),
                    "nan_count": int(nan_count[column]),
                    "positive_infinity_count": int(posinf_count[column]),
                    "negative_infinity_count": int(neginf_count[column]),
                    "zero_count": int(zero_count[column]),
                    "non_binary_count": int(non_binary_count[column]),
                    "zero_rate_among_finite": float(zero_count[column] / finite)
                    if finite
                    else float("nan"),
                    "nonzero_rate_among_finite": float(
                        1.0 - zero_count[column] / finite
                    )
                    if finite
                    else float("nan"),
                    "minimum": float(minimum[column]) if finite else float("nan"),
                    "maximum": float(maximum[column]) if finite else float("nan"),
                    "quantile_sample_row_count": sample_count,
                    "nonzero_quantile_sample_count": int(
                        nonzero_sample_count[column]
                    ),
                    **{
                        f"q{int(q * 100):02d}": float(quantile_values[offset, column])
                        for offset, q in enumerate(QUANTILES)
                    },
                    **{
                        f"nz_q{int(q * 100):02d}": float(
                            nonzero_quantile_values[offset, column]
                        )
                        for offset, q in enumerate(QUANTILES)
                    },
                }
            )
    annual = pd.DataFrame(annual_rows)
    drift_rows: list[dict[str, Any]] = []
    for feature, group in annual.groupby("feature", sort=False):
        ordered = group.sort_values("year").reset_index(drop=True)
        years = ordered["year"].to_numpy(dtype=np.int64)
        pair_labels = [
            f"{int(years[index - 1])}->{int(years[index])}"
            for index in range(1, len(years))
        ]
        missing = 1.0 - ordered["finite_rate"].to_numpy(dtype=np.float64)
        event_rates = ordered["nonzero_rate_among_finite"].to_numpy(
            dtype=np.float64
        )
        finite_total = int(ordered["finite_count"].sum())
        zero_total = int(ordered["zero_count"].sum())
        non_binary_total = int(ordered["non_binary_count"].sum())
        minimum = float(ordered["minimum"].min())
        maximum = float(ordered["maximum"].max())
        constant = finite_total > 0 and np.isclose(
            minimum, maximum, equal_nan=False
        )
        overall_zero_rate = float(zero_total / finite_total) if finite_total else float("nan")
        if constant:
            feature_type = "constant"
        elif non_binary_total == 0:
            feature_type = "binary"
        elif overall_zero_rate >= 0.80:
            feature_type = "sparse"
        else:
            feature_type = "continuous"

        quantile_prefix = "nz_" if feature_type == "sparse" else ""
        quantile_matrix = ordered[
            [f"{quantile_prefix}q{int(q * 100):02d}" for q in QUANTILES]
        ].to_numpy(dtype=np.float64)
        medians = ordered[f"{quantile_prefix}q50"].to_numpy(dtype=np.float64)
        scale = _global_robust_scale(ordered, prefix=quantile_prefix)
        location_shifts: list[float] = []
        tail_shifts: list[float] = []
        quantile_wasserstein: list[float] = []
        for index in range(1, len(ordered)):
            sufficient = feature_type not in {"binary", "constant"}
            if feature_type == "sparse":
                sufficient = bool(
                    int(ordered.loc[index - 1, "nonzero_quantile_sample_count"])
                    >= 50
                    and int(ordered.loc[index, "nonzero_quantile_sample_count"])
                    >= 50
                )
            if not sufficient:
                location_shifts.append(float("nan"))
                tail_shifts.append(float("nan"))
                quantile_wasserstein.append(float("nan"))
                continue
            left_quantiles = quantile_matrix[index - 1]
            right_quantiles = quantile_matrix[index]
            valid_quantiles = np.isfinite(left_quantiles) & np.isfinite(
                right_quantiles
            )
            location_shifts.append(
                abs(medians[index] - medians[index - 1]) / scale
                if np.isfinite(medians[index]) and np.isfinite(medians[index - 1])
                else float("nan")
            )
            tail_candidates = [
                abs(right_quantiles[offset] - left_quantiles[offset])
                for offset in (1, 5)
                if valid_quantiles[offset]
            ]
            tail_shifts.append(
                max(tail_candidates) / scale if tail_candidates else float("nan")
            )
            quantile_wasserstein.append(
                float(np.mean(np.abs(right_quantiles[valid_quantiles] - left_quantiles[valid_quantiles])) / scale)
                if bool(valid_quantiles.any())
                else float("nan")
            )

        max_missing_jump, missing_years = _maximum_adjacent(missing, years)
        max_event_jump, event_years = _maximum_adjacent(event_rates, years)
        event_js = [
            _bernoulli_js_distance(event_rates[index - 1], event_rates[index])
            for index in range(1, len(event_rates))
        ]
        max_event_js, event_js_years = _maximum_metric(event_js, pair_labels)
        max_location, location_years = _maximum_metric(
            location_shifts, pair_labels
        )
        max_tail, tail_years = _maximum_metric(tail_shifts, pair_labels)
        max_wasserstein, wasserstein_years = _maximum_metric(
            quantile_wasserstein, pair_labels
        )
        coverage_shift = bool(max_missing_jump >= 0.20)
        event_rate_shift = bool(
            feature_type == "binary"
            and max_event_jump >= 0.10
            and max_event_js >= 0.075
        ) or bool(
            feature_type == "sparse"
            and max_event_jump >= 0.05
            and max_event_js >= 0.075
        )
        conditional_shift = bool(
            feature_type in {"continuous", "sparse"}
            and (
                max_location >= 3.0
                or max_tail >= 4.0
                or max_wasserstein >= 1.5
            )
        )
        value_shift = event_rate_shift or conditional_shift
        if coverage_shift and value_shift:
            drift_class = "coverage_and_value_shift"
        elif coverage_shift:
            drift_class = "source_coverage_shift"
        elif event_rate_shift:
            drift_class = "event_rate_shift"
        elif conditional_shift:
            drift_class = "conditional_distribution_shift"
        else:
            drift_class = "stable_or_below_effect_threshold"
        drift_rows.append(
            {
                "feature": feature,
                "feature_type": feature_type,
                "minimum_finite_rate": float(ordered["finite_rate"].min()),
                "maximum_finite_rate": float(ordered["finite_rate"].max()),
                "maximum_adjacent_missing_rate_jump": max_missing_jump,
                "maximum_missing_rate_jump_years": missing_years,
                "overall_zero_rate_among_finite": overall_zero_rate,
                "maximum_adjacent_event_rate_jump": max_event_jump,
                "maximum_event_rate_jump_years": event_years,
                "maximum_adjacent_event_rate_js_distance": max_event_js,
                "maximum_event_rate_js_years": event_js_years,
                "global_robust_scale": scale,
                "conditional_quantiles": "nonzero" if feature_type == "sparse" else "all_finite",
                "maximum_adjacent_median_shift_robust_scale": max_location,
                "maximum_median_shift_years": location_years,
                "maximum_adjacent_tail_shift_robust_scale": max_tail,
                "maximum_tail_shift_years": tail_years,
                "maximum_adjacent_quantile_wasserstein_robust_scale": max_wasserstein,
                "maximum_quantile_wasserstein_years": wasserstein_years,
                "all_missing_year_count": int((ordered["finite_count"] == 0).sum()),
                "source_coverage_shift_flag": coverage_shift,
                "structural_coverage_shift_flag": coverage_shift,
                "event_rate_shift_flag": event_rate_shift,
                "conditional_distribution_shift_flag": conditional_shift,
                "candidate_value_nonstationarity_flag": value_shift,
                "robust_distribution_shift_flag": value_shift,
                "drift_class": drift_class,
            }
        )
    drift = pd.DataFrame(drift_rows)
    aggregate = annual.groupby("feature", sort=False).agg(
        finite_count=("finite_count", "sum"),
        nan_count=("nan_count", "sum"),
        positive_infinity_count=("positive_infinity_count", "sum"),
        negative_infinity_count=("negative_infinity_count", "sum"),
        minimum=("minimum", "min"),
        maximum=("maximum", "max"),
    )
    all_missing_features = aggregate.index[aggregate["finite_count"].eq(0)].tolist()
    constant_features = aggregate.index[
        aggregate["finite_count"].gt(0)
        & np.isclose(aggregate["minimum"], aggregate["maximum"], equal_nan=False)
    ].tolist()
    summary = {
        "full_population_row_count": len(row_index),
        "feature_count": feature_count,
        "all_missing_row_count": all_missing_row_count,
        "any_infinite_row_count": any_infinite_row_count,
        "positive_infinity_value_count": int(
            annual["positive_infinity_count"].sum()
        ),
        "negative_infinity_value_count": int(
            annual["negative_infinity_count"].sum()
        ),
        "all_missing_features": all_missing_features,
        "constant_features": constant_features,
        "all_missing_year_feature_count": int(drift["all_missing_year_count"].sum()),
        "structural_coverage_shift_feature_count": int(
            drift["structural_coverage_shift_flag"].sum()
        ),
        "event_rate_shift_feature_count": int(drift["event_rate_shift_flag"].sum()),
        "conditional_distribution_shift_feature_count": int(
            drift["conditional_distribution_shift_flag"].sum()
        ),
        "robust_distribution_shift_feature_count": int(
            drift["robust_distribution_shift_flag"].sum()
        ),
        "feature_type_counts": {
            str(key): int(value)
            for key, value in drift["feature_type"].value_counts().items()
        },
        "distribution_method": {
            "quantiles": list(QUANTILES),
            "sampling": "deterministic_evenly_spaced_within_each_year",
            "maximum_rows_per_year": int(quantile_sample_rows_per_year),
            "missing_zero_binary_and_infinity_counts": "full_population_exact",
            "observation_process": "annual_finite_rate_with_20pp_effect_threshold",
            "binary_and_sparse_process": (
                "annual_nonzero_event_rate_with_bernoulli_jensen_shannon_distance"
            ),
            "sparse_conditional_distribution": (
                "nonzero_quantiles_when_each_adjacent_year_has_at_least_50_sampled_events"
            ),
            "continuous_distribution": "annual_quantile_wasserstein_approximation",
            "drift_scale": (
                "all_year_robust_scale=max(median_positive_iqr,"
                "median_positive_central90_equivalent_iqr,one_percent_level_floor)"
            ),
            "effect_thresholds": {
                "binary_event_rate_absolute_jump": 0.10,
                "sparse_event_rate_absolute_jump": 0.05,
                "bernoulli_js_distance": 0.075,
                "median_robust_scale": 3.0,
                "tail_robust_scale": 4.0,
                "quantile_wasserstein_robust_scale": 1.5,
            },
            "interpretation": (
                "coverage shifts describe the observation process; value shifts are "
                "candidates for economic or definition nonstationarity, not proof of either"
            ),
        },
    }
    return annual, drift, summary


def _balance_semantic_checks(
    *, workspace: Path, dataset_ids: Mapping[str, str]
) -> dict[str, Any]:
    paths = _paths_for_dataset(
        workspace, dataset_ids, "balance_sheet_quarterly"
    )
    required = {
        *ready.BALANCE_EXTENSION_NUMERIC_FIELDS,
        *ready.BALANCE_SEMANTIC_NUMERIC_FIELDS,
        *ready.BALANCE_EXTENSION_STATE_FIELDS,
        *ready.BALANCE_SEMANTIC_STATE_FIELDS,
        ready.BALANCE_EXTENSION_CONFLICT_FIELD,
        ready.BALANCE_SEMANTIC_CONFLICT_FIELD,
        "accounts_receivable_and_notes_reported",
        "accounts_payable_and_notes_reported",
        "receivables_financing",
        "fixed_assets_total_reported",
        "construction_in_progress_total_reported",
    }
    with duckdb.connect() as connection:
        columns = {
            str(row[0])
            for row in connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?, union_by_name=true)",
                [[str(path) for path in paths]],
            ).fetchall()
        }
        missing = sorted(required.difference(columns))
        if missing:
            return {
                "required_columns_present": False,
                "missing_columns": missing,
                "passed": False,
            }
        row = connection.execute(
            """
            SELECT count(*),
                   count(trade_receivables_total),
                   count(trade_payables_total),
                   count(fixed_assets_measure),
                   count(construction_in_progress_measure),
                   count(*) FILTER (
                     WHERE trade_receivables_field_state NOT IN (
                       'combined_plus_financing_observed',
                       'combined_observed_financing_unreported',
                       'components_plus_financing_observed',
                       'components_observed_financing_unreported',
                       'unknown','not_applicable'
                     ) OR trade_receivables_field_state IS NULL
                     OR fixed_assets_measure_field_state NOT IN (
                       'total_observed','component_fallback','unknown','not_applicable'
                     ) OR fixed_assets_measure_field_state IS NULL
                     OR construction_in_progress_measure_field_state NOT IN (
                       'total_observed','component_fallback','unknown','not_applicable'
                     ) OR construction_in_progress_measure_field_state IS NULL
                     OR trade_payables_field_state NOT IN (
                       'combined_observed','components_observed','unknown','not_applicable'
                     ) OR trade_payables_field_state IS NULL
                   ),
                   count(*) FILTER (
                     WHERE year(cast(report_date AS DATE))=2018
                       AND accounts_receivable_and_notes_reported IS NOT NULL
                   ),
                   count(*) FILTER (
                     WHERE year(cast(report_date AS DATE))=2018
                       AND accounts_payable_and_notes_reported IS NOT NULL
                   ),
                   count(*) FILTER (
                     WHERE year(cast(report_date AS DATE))=2019
                       AND accounts_receivable_and_notes_reported IS NOT NULL
                   ),
                   count(*) FILTER (
                     WHERE year(cast(report_date AS DATE))=2019
                       AND accounts_payable_and_notes_reported IS NOT NULL
                   )
            FROM read_parquet(?, union_by_name=true)
            """,
            [[str(path) for path in paths]],
        ).fetchone()
        year_counts = connection.execute(
            """
            SELECT year(cast(report_date AS DATE)) AS report_year,count(*)
            FROM read_parquet(?, union_by_name=true)
            WHERE year(cast(report_date AS DATE)) IN (2018,2019)
            GROUP BY 1 ORDER BY 1
            """,
            [[str(path) for path in paths]],
        ).fetchall()
    year_total = {int(year): int(count) for year, count in year_counts}
    coverage_2018 = {
        "receivables": float(row[6] or 0) / max(year_total.get(2018, 0), 1),
        "payables": float(row[7] or 0) / max(year_total.get(2018, 0), 1),
    }
    coverage_2019 = {
        "receivables": float(row[8] or 0) / max(year_total.get(2019, 0), 1),
        "payables": float(row[9] or 0) / max(year_total.get(2019, 0), 1),
    }
    return {
        "required_columns_present": True,
        "missing_columns": [],
        "row_count": int(row[0] or 0),
        "semantic_measure_nonnull_counts": {
            "trade_receivables_total": int(row[1] or 0),
            "trade_payables_total": int(row[2] or 0),
            "fixed_assets_measure": int(row[3] or 0),
            "construction_in_progress_measure": int(row[4] or 0),
        },
        "invalid_semantic_state_count": int(row[5] or 0),
        "combined_provider_coverage": {
            "2018": coverage_2018,
            "2019": coverage_2019,
        },
        "passed": (
            int(row[0] or 0) > 0
            and all(int(value or 0) > 0 for value in row[1:5])
            and int(row[5] or 0) == 0
            and min(*coverage_2018.values(), *coverage_2019.values()) >= 0.90
        ),
    }


def _training_block_pit_checks(
    *, ready_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    with duckdb.connect() as con:
        for block_name, block in dict(ready_manifest["blocks"]).items():
            paths = [
                str(Path(record["path"]))
                for _, record in sorted(dict(block["partitions"]).items())
                if int(_) in YEARS
            ]
            columns = set(
                con.execute(
                    "DESCRIBE SELECT * FROM read_parquet(?, union_by_name=true)",
                    [paths],
                ).fetchdf()["column_name"].astype(str)
            )
            checks: dict[str, Any] = {}
            if {"source_date", "feature_available_date"}.issubset(columns):
                row = con.execute(
                    """
                    SELECT count(*) FILTER (
                               WHERE coalesce(feature_available_date,'')<>''
                                 AND try_cast(feature_available_date AS DATE)>
                                     try_cast(trade_date AS DATE)
                           ),
                           count(*) FILTER (
                               WHERE coalesce(source_date,'')<>''
                                 AND try_cast(source_date AS DATE)>
                                     try_cast(trade_date AS DATE)
                           ),
                           count(*) FILTER (
                               WHERE coalesce(feature_available_date,'')<>''
                                 AND try_cast(feature_available_date AS DATE)<>
                                     try_cast(trade_date AS DATE)
                           ),
                           count(*) FILTER (
                               WHERE coverage_state='observed'
                                 AND coalesce(source_date,'')=''
                           )
                    FROM read_parquet(?, union_by_name=true)
                    """,
                    [paths],
                ).fetchone()
                checks.update(
                    {
                        "feature_available_after_signal_count": int(row[0] or 0),
                        "source_after_signal_count": int(row[1] or 0),
                        "feature_available_not_signal_date_count": int(row[2] or 0),
                        "observed_without_source_date_count": int(row[3] or 0),
                    }
                )
            for prefix in ("margin_detail", "margin_market"):
                source_column = f"{prefix}_source_date"
                available_column = f"{prefix}_feature_available_date"
                if {source_column, available_column}.issubset(columns):
                    row = con.execute(
                        f"""
                        SELECT count(*) FILTER (
                                   WHERE coalesce({_q(available_column)},'')<>''
                                     AND try_cast({_q(available_column)} AS DATE)>
                                         try_cast(trade_date AS DATE)
                               ),
                               count(*) FILTER (
                                   WHERE coalesce({_q(source_column)},'')<>''
                                     AND try_cast({_q(source_column)} AS DATE)>
                                         try_cast(trade_date AS DATE)
                               ),
                               count(*) FILTER (
                                   WHERE coalesce({_q(available_column)},'')<>''
                                     AND try_cast({_q(available_column)} AS DATE)<>
                                         try_cast(trade_date AS DATE)
                               )
                        FROM read_parquet(?, union_by_name=true)
                        """,
                        [paths],
                    ).fetchone()
                    checks[f"{prefix}_feature_available_after_signal_count"] = int(
                        row[0] or 0
                    )
                    checks[f"{prefix}_source_after_signal_count"] = int(row[1] or 0)
                    checks[f"{prefix}_available_not_signal_date_count"] = int(
                        row[2] or 0
                    )
            age_columns = sorted(
                column
                for column in columns
                if column.endswith("_age_days") or column.startswith("announcement_days_since_")
            )
            negative_age_count = 0
            if age_columns:
                predicate = " OR ".join(
                    f"try_cast({_q(column)} AS DOUBLE)<0" for column in age_columns
                )
                negative_age_count = int(
                    con.execute(
                        f"SELECT count(*) FROM read_parquet(?, union_by_name=true) "
                        f"WHERE {predicate}",
                        [paths],
                    ).fetchone()[0]
                )
            checks["negative_age_row_count"] = negative_age_count
            nonblocking = (
                {"feature_available_not_signal_date_count"}
                if block_name == "financial_statement_extensions"
                else set()
            )
            results[block_name] = {
                "date_checks": checks,
                "nonblocking_asof_carry_fields": sorted(nonblocking),
                "passed": all(
                    int(value) == 0
                    for key, value in checks.items()
                    if key not in nonblocking
                ),
            }
    return results


def _qdp_next_open_checks(
    *, workspace: Path, dataset_ids: Mapping[str, str]
) -> dict[str, Any]:
    calendar_paths = _paths_for_dataset(workspace, dataset_ids, "trading_calendar")
    results: dict[str, Any] = {}
    with duckdb.connect() as con:
        for domain in NEXT_OPEN_DOMAINS:
            paths = _paths_for_dataset(workspace, dataset_ids, domain)
            row = con.execute(
                """
                WITH open_dates AS (
                  SELECT DISTINCT cast(trade_date AS VARCHAR) AS trade_date
                  FROM read_parquet(?, union_by_name=true)
                  WHERE coalesce(try_cast(is_open AS BOOLEAN), false)
                    AND cast(trade_date AS VARCHAR)<=?
                ), source_rows AS (
                  SELECT cast(source_date AS VARCHAR) AS source_date,
                         cast(feature_available_date AS VARCHAR) AS available_date
                  FROM read_parquet(?, union_by_name=true)
                  WHERE cast(trade_date AS VARCHAR) BETWEEN '2010-01-01' AND ?
                ), source_dates AS (
                  SELECT DISTINCT source_date FROM source_rows
                ), next_open AS (
                  SELECT s.source_date,min(o.trade_date) AS next_date
                  FROM source_dates s
                  LEFT JOIN open_dates o ON o.trade_date>s.source_date
                  GROUP BY s.source_date
                )
                SELECT count(*) AS rows,
                       count(*) FILTER (
                         WHERE try_cast(s.source_date AS DATE)>
                               try_cast(s.available_date AS DATE)
                       ) AS source_after_available,
                       count(*) FILTER (
                         WHERE coalesce(s.available_date,'')<>''
                           AND s.available_date>?
                       ) AS available_after_cutoff,
                       count(*) FILTER (
                         WHERE n.next_date IS NOT NULL AND s.available_date<>n.next_date
                       ) AS wrong_next_open,
                       count(*) FILTER (
                         WHERE n.next_date IS NULL AND coalesce(s.available_date,'')<>''
                       ) AS unexpected_available_without_in_scope_next_open,
                       count(*) FILTER (
                         WHERE n.next_date IS NOT NULL AND coalesce(s.available_date,'')=''
                       ) AS missing_available_with_in_scope_next_open
                FROM source_rows s LEFT JOIN next_open n USING(source_date)
                """,
                [
                    [str(path) for path in calendar_paths],
                    AS_OF_DATE,
                    [str(path) for path in paths],
                    AS_OF_DATE,
                    AS_OF_DATE,
                ],
            ).fetchone()
            results[domain] = {
                "row_count": int(row[0] or 0),
                "source_after_available_count": int(row[1] or 0),
                "available_after_cutoff_count": int(row[2] or 0),
                "wrong_next_open_count": int(row[3] or 0),
                "unexpected_available_without_in_scope_next_open_count": int(
                    row[4] or 0
                ),
                "missing_available_with_in_scope_next_open_count": int(row[5] or 0),
            }
            results[domain]["passed"] = all(
                int(value) == 0
                for key, value in results[domain].items()
                if key.endswith("_count") and key != "row_count"
            )
    return results


def _price_and_suspension_checks(
    *,
    workspace: Path,
    dataset_ids: Mapping[str, str],
    row_index_path: Path,
    ready_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    factor = _paths_for_dataset(workspace, dataset_ids, "stk_factor_pro_raw")
    daily = _paths_for_dataset(workspace, dataset_ids, "market_daily_raw")
    status = _paths_for_dataset(workspace, dataset_ids, "security_status")
    with duckdb.connect() as con:
        price = con.execute(
            """
            WITH pairs AS (
              SELECT f.symbol,f.trade_date,
                     greatest(
                       abs(try_cast(f.open AS DOUBLE)/nullif(try_cast(d.open AS DOUBLE),0)-1),
                       abs(try_cast(f.high AS DOUBLE)/nullif(try_cast(d.high AS DOUBLE),0)-1),
                       abs(try_cast(f.low AS DOUBLE)/nullif(try_cast(d.low AS DOUBLE),0)-1),
                       abs(try_cast(f.close AS DOUBLE)/nullif(try_cast(d.close AS DOUBLE),0)-1)
                     ) AS relative_mismatch,
                     d.adjusted_flag
              FROM read_parquet(?, union_by_name=true) f
              JOIN read_parquet(?, union_by_name=true) d USING(symbol,trade_date)
              WHERE cast(f.trade_date AS VARCHAR) BETWEEN ? AND ?
            )
            SELECT count(*) AS joined_rows,
                   count(*) FILTER (
                     WHERE relative_mismatch>0.005
                   ) AS material_mismatches,
                   max(relative_mismatch) AS maximum_relative_mismatch,
                   count(*) FILTER (
                     WHERE lower(trim(coalesce(cast(adjusted_flag AS VARCHAR),'')))
                           NOT IN ('','none','false','0','raw','unadjusted')
                   ) AS invalid_adjustment_flags
            FROM pairs
            """,
            [
                [str(path) for path in factor],
                [str(path) for path in daily],
                RESEARCH_START,
                AS_OF_DATE,
            ],
        ).fetchone()
        technical_paths = [
            str(record["path"])
            for year, record in sorted(
                dict(
                    ready_manifest["blocks"]["tushare_technical_candidates"][
                        "partitions"
                    ]
                ).items()
            )
            if int(year) in YEARS
        ]
        ungated = int(
            con.execute(
                """
                WITH bad AS (
                  SELECT f.symbol,f.trade_date
                  FROM read_parquet(?, union_by_name=true) f
                  JOIN read_parquet(?, union_by_name=true) d USING(symbol,trade_date)
                  WHERE cast(f.trade_date AS VARCHAR) BETWEEN ? AND ?
                    AND greatest(
                      abs(try_cast(f.open AS DOUBLE)/nullif(try_cast(d.open AS DOUBLE),0)-1),
                      abs(try_cast(f.high AS DOUBLE)/nullif(try_cast(d.high AS DOUBLE),0)-1),
                      abs(try_cast(f.low AS DOUBLE)/nullif(try_cast(d.low AS DOUBLE),0)-1),
                      abs(try_cast(f.close AS DOUBLE)/nullif(try_cast(d.close AS DOUBLE),0)-1)
                    )>0.005
                )
                SELECT count(*)
                FROM bad b
                JOIN read_parquet(?) r USING(symbol,trade_date)
                JOIN read_parquet(?, union_by_name=true) t USING(candidate_id)
                WHERE t.coverage_state<>'source_price_mismatch'
                """,
                [
                    [str(path) for path in factor],
                    [str(path) for path in daily],
                    RESEARCH_START,
                    AS_OF_DATE,
                    str(row_index_path),
                    technical_paths,
                ],
            ).fetchone()[0]
        )
        eligibility = con.execute(
            """
            WITH spine AS (
              SELECT symbol,trade_date FROM read_parquet(?)
            ), joined AS (
              SELECT s.symbol,s.trade_date,st.is_st,st.is_suspended,st.is_delisted,
                     d.volume
              FROM spine s
              LEFT JOIN read_parquet(?, union_by_name=true) st USING(symbol,trade_date)
              LEFT JOIN read_parquet(?, union_by_name=true) d USING(symbol,trade_date)
            )
            SELECT count(*),
                   count(*) FILTER (WHERE is_st IS NULL),
                   count(*) FILTER (WHERE coalesce(is_st,false)),
                   count(*) FILTER (WHERE coalesce(is_suspended,false)),
                   count(*) FILTER (WHERE coalesce(is_delisted,false)),
                   count(*) FILTER (WHERE volume IS NULL OR try_cast(volume AS DOUBLE)<=0)
            FROM joined
            """,
            [
                str(row_index_path),
                [str(path) for path in status],
                [str(path) for path in daily],
            ],
        ).fetchone()
    return {
        "raw_bfq_price_anchor": {
            "joined_row_count": int(price[0] or 0),
            "material_ohlc_mismatch_count": int(price[1] or 0),
            "material_ohlc_mismatch_rate": float(
                int(price[1] or 0) / max(int(price[0] or 0), 1)
            ),
            "maximum_relative_mismatch": float(price[2] or 0.0),
            "invalid_adjustment_flag_count": int(price[3] or 0),
            "ungated_material_mismatch_in_formal_pool_count": ungated,
            "relative_tolerance": 0.005,
            "passed": int(price[0] or 0) > 0
            and float(int(price[1] or 0) / max(int(price[0] or 0), 1)) <= 0.005
            and int(price[3] or 0) == 0
            and ungated == 0,
        },
        "formal_pool_tradeability": {
            "row_count": int(eligibility[0] or 0),
            "missing_status_row_count": int(eligibility[1] or 0),
            "st_row_count": int(eligibility[2] or 0),
            "suspended_row_count": int(eligibility[3] or 0),
            "delisted_row_count": int(eligibility[4] or 0),
            "missing_or_nonpositive_daily_volume_row_count": int(eligibility[5] or 0),
            "passed": all(int(value or 0) == 0 for value in eligibility[1:]),
        },
    }


def _minute_recompute_check(
    *,
    workspace: Path,
    dataset_ids: Mapping[str, str],
    ready_manifest: Mapping[str, Any],
    row_index: pd.DataFrame,
) -> dict[str, Any]:
    intraday = _paths_for_dataset(workspace, dataset_ids, "market_intraday_5m")
    dates: list[str] = []
    date_series = row_index["trade_date"].astype(str)
    for year in YEARS:
        available = date_series[date_series.str.startswith(f"{year}-")].unique()
        for index in sorted({0, len(available) // 2, len(available) - 1}):
            dates.append(str(available[index]))
    minute_paths = [
        str(ready_manifest["feature_partitions"][str(year)]["minute"]["path"])
        if "feature_partitions" in ready_manifest
        else ""
        for year in YEARS
    ]
    if not all(minute_paths):
        scope_manifest = _read_json(ready.DEFAULT_SCOPE_OUTPUT_ROOT / "manifest.json")
        minute_paths = [
            str(scope_manifest["feature_partitions"][str(year)]["minute"]["path"])
            for year in YEARS
        ]
    quoted_dates = ",".join("'" + value.replace("'", "''") + "'" for value in dates)
    with duckdb.connect() as con:
        row = con.execute(
            f"""
            WITH bars AS (
              SELECT symbol,cast(trade_date AS VARCHAR) AS trade_date,
                     count(*) AS bars,
                     count(DISTINCT cast(bar_time AS VARCHAR)) AS distinct_times,
                     max(cast(bar_time AS VARCHAR)) AS max_bar_time,
                     arg_min(try_cast(open AS DOUBLE),cast(bar_time AS VARCHAR)) AS day_open,
                     max(try_cast(high AS DOUBLE)) AS day_high,
                     min(try_cast(low AS DOUBLE)) AS day_low,
                     max(try_cast(close AS DOUBLE)) AS max_close,
                     min(try_cast(close AS DOUBLE)) AS min_close,
                     arg_max(try_cast(close AS DOUBLE),cast(bar_time AS VARCHAR)) AS day_close
              FROM read_parquet(?, union_by_name=true)
              WHERE cast(trade_date AS VARCHAR) IN ({quoted_dates})
              GROUP BY symbol,trade_date
            ), expected AS (
              SELECT symbol,cast(trade_date AS VARCHAR) AS trade_date,
                     try_cast(minute_open_close_return AS DOUBLE) AS open_close,
                     try_cast(minute_range AS DOUBLE) AS day_range,
                     try_cast(minute_close_location AS DOUBLE) AS close_location,
                     try_cast(minute_mfe_from_open AS DOUBLE) AS mfe_open,
                     try_cast(minute_mae_from_open AS DOUBLE) AS mae_open
              FROM read_parquet(?, union_by_name=true)
              WHERE cast(trade_date AS VARCHAR) IN ({quoted_dates})
            ), joined AS (
              SELECT e.*,b.* EXCLUDE(symbol,trade_date),
                     day_close/nullif(day_open,0)-1 AS calc_open_close,
                     (day_high-day_low)/nullif(day_open,0) AS calc_range,
                     (day_close-day_low)/nullif(day_high-day_low,0) AS calc_location,
                     max_close/nullif(day_open,0)-1 AS calc_mfe,
                     min_close/nullif(day_open,0)-1 AS calc_mae
              FROM expected e LEFT JOIN bars b USING(symbol,trade_date)
            )
            SELECT count(*),
                   count(*) FILTER (WHERE bars<>48 OR distinct_times<>48),
                   count(*) FILTER (WHERE max_bar_time>'150000000'),
                   count(*) FILTER (
                     WHERE abs(open_close-calc_open_close)>1e-6
                        OR abs(day_range-calc_range)>1e-6
                        OR abs(close_location-calc_location)>1e-6
                        OR abs(mfe_open-calc_mfe)>1e-6
                        OR abs(mae_open-calc_mae)>1e-6
                   )
            FROM joined
            """,
            [
                [str(path) for path in intraday],
                minute_paths,
            ],
        ).fetchone()
    return {
        "sample_date_count": len(dates),
        "sample_stock_day_count": int(row[0] or 0),
        "invalid_48_bar_stock_day_count": int(row[1] or 0),
        "post_close_bar_stock_day_count": int(row[2] or 0),
        "recomputed_feature_mismatch_count": int(row[3] or 0),
        "passed": int(row[0] or 0) > 0 and all(int(value or 0) == 0 for value in row[1:]),
        "sampling": "first_middle_last_formal_signal_date_per_year",
    }


def prepare(
    *,
    ready_root: Path = DEFAULT_READY_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    workspace: Path = WORKSPACE_ROOT,
    qdp_deep: bool = True,
    qdp_runtime: str = "balanced",
    qdp_threads: int | None = None,
) -> dict[str, Any]:
    if YEARS != ready.RESEARCH_YEARS or AS_OF_DATE != ready.END_DATE:
        raise CertificationError("certification_boundary_not_canonical")
    ready_manifest_path = ready_root / "manifest.json"
    readiness_path = ready_root / "readiness.json"
    input_manifest_path = ready_root / model.MODEL_INPUT_DIR_NAME / "manifest.json"
    ready_manifest = _read_json(ready_manifest_path)
    readiness = _read_json(readiness_path)
    input_manifest = _read_json(input_manifest_path)
    verified = model._verify_model_input_files(input_manifest, full_hash=True)
    row_index = verified["row_index"]
    compact = dict(input_manifest["storage"]["compact"])
    values = np.memmap(
        Path(compact["path"]),
        dtype=np.dtype(compact["dtype"]),
        mode="r",
        shape=tuple(int(value) for value in compact["shape"]),
    )
    feature_names = list(input_manifest["feature_groups"][model.COMPACT_VARIANT])
    output_root.mkdir(parents=True, exist_ok=True)

    row_checks = _row_index_checks(
        row_index=row_index, ready_manifest=ready_manifest
    )
    contract_checks = _feature_contract_checks(input_manifest)
    annual, drift, matrix_summary = _feature_matrix_quality(
        values=values,
        row_index=row_index,
        feature_names=feature_names,
    )
    del values
    annual_path = output_root / "feature_quality_by_year.parquet"
    drift_path = output_root / "feature_drift.parquet"
    _write_frame(annual_path, annual)
    _write_frame(drift_path, drift)

    bindings = _source_bindings(workspace=workspace, ready_manifest=ready_manifest)
    dataset_ids = dict(bindings["dataset_ids"])
    balance_semantics = _balance_semantic_checks(
        workspace=workspace, dataset_ids=dataset_ids
    )
    block_pit = _training_block_pit_checks(ready_manifest=ready_manifest)
    qdp_pit = _qdp_next_open_checks(workspace=workspace, dataset_ids=dataset_ids)
    price_suspension = _price_and_suspension_checks(
        workspace=workspace,
        dataset_ids=dataset_ids,
        row_index_path=Path(input_manifest["row_index"]["path"]),
        ready_manifest=ready_manifest,
    )
    minute = _minute_recompute_check(
        workspace=workspace,
        dataset_ids=dataset_ids,
        ready_manifest=ready_manifest,
        row_index=row_index,
    )
    qdp_path = output_root / "qdp_as_of_2025-12-31.json"
    qdp = audit_database_as_of(
        as_of_date=AS_OF_DATE,
        workspace_root=workspace,
        deep=qdp_deep,
        runtime=qdp_runtime,
        threads=qdp_threads,
        write_path=qdp_path,
    )
    pit = {
        "training_blocks": block_pit,
        "qdp_next_open": qdp_pit,
        "balance_semantics": balance_semantics,
        "price_and_suspension": price_suspension,
        "minute_recomputation": minute,
    }
    pit_path = output_root / "pit_checks.json"
    _write_json(pit_path, pit)

    source_manifest_record = dict(input_manifest["source"]["training_ready_manifest"])
    hash_checks = {
        "training_ready_manifest_bound": source_manifest_record.get("sha256")
        == _sha256(ready_manifest_path),
        "row_index_full_hash_matches": True,
        "compact_full_hash_matches": True,
        "feature_contract_full_hash_matches": True,
    }
    blocking_checks = {
        "qdp_as_of_audit_ok": qdp.get("status") == "ok",
        "qdp_forbidden_rows_zero": int(
            dict(qdp.get("as_of", {})).get(
                "forbidden_after_cutoff_row_count_in_audit_snapshot", -1
            )
        )
        == 0,
        "source_dataset_bindings_current": bool(
            bindings["all_bound_datasets_still_active"]
        ),
        "balance_semantic_contract_passed": bool(balance_semantics["passed"]),
        "row_index_contract_passed": all(
            bool(row_checks[key])
            for key in (
                "row_count_exact",
                "columns_exact",
                "candidate_id_unique",
                "candidate_id_strictly_increasing",
                "date_idx_nondecreasing",
                "starts_in_2012",
                "ends_at_cutoff",
                "formal_years_exact",
                "rows_by_year_match_spines",
            )
        )
        and int(row_checks["burn_in_formal_row_count"]) == 0
        and int(row_checks["forbidden_2026_row_count"]) == 0,
        "feature_contract_passed": bool(contract_checks["feature_count_exact"])
        and bool(contract_checks["feature_names_unique"])
        and bool(contract_checks["future_or_label_metadata_absent"])
        and bool(contract_checks["qfq_hfq_features_absent"])
        and bool(contract_checks["unstable_balance_component_features_absent"])
        and bool(contract_checks["stable_balance_features_present"])
        and bool(contract_checks["source_era_unstable_features_absent"])
        and contract_checks["training_performed"] is False
        and contract_checks["feature_set_selected"] is False,
        "feature_matrix_finite_contract_passed": int(
            matrix_summary["any_infinite_row_count"]
        )
        == 0
        and int(matrix_summary["all_missing_row_count"]) == 0
        and not matrix_summary["all_missing_features"],
        "training_block_pit_passed": all(
            bool(record["passed"]) for record in block_pit.values()
        ),
        "qdp_next_open_pit_passed": all(
            bool(record["passed"]) for record in qdp_pit.values()
        ),
        "raw_price_and_tradeability_passed": all(
            bool(record["passed"]) for record in price_suspension.values()
        ),
        "minute_same_day_recomputation_passed": bool(minute["passed"]),
        "all_input_hashes_match": all(hash_checks.values()),
    }
    status = "certified" if all(blocking_checks.values()) else "failed"
    manifest = {
        "schema": "seq100_quality_liquidity_data_certification/v1",
        "study_id": STUDY_ID,
        "builder_version": BUILDER_VERSION,
        "status": status,
        "created_at": _now(),
        "scope": {
            "research_start_date": RESEARCH_START,
            "end_date": AS_OF_DATE,
            "burn_in_years": [2010, 2011],
            "research_years": list(YEARS),
            "expected_row_count": EXPECTED_ROW_COUNT,
            "expected_feature_count": EXPECTED_FEATURE_COUNT,
            "training_performed": False,
        },
        "blocking_checks": blocking_checks,
        "row_index": row_checks,
        "feature_contract": contract_checks,
        "feature_matrix": matrix_summary,
        "source_bindings": bindings,
        "pit_checks": _record(pit_path),
        "qdp_as_of_audit": _record(
            qdp_path,
            status=qdp.get("status"),
            finding_count=int(qdp.get("finding_count", 0)),
        ),
        "hash_checks": hash_checks,
        "outputs": {
            "feature_quality_by_year": _record(
                annual_path, row_count=len(annual)
            ),
            "feature_drift": _record(drift_path, row_count=len(drift)),
        },
        "sources": {
            "training_ready_manifest": _record(ready_manifest_path),
            "training_readiness": _record(readiness_path, status=readiness.get("status")),
            "model_input_manifest": _record(
                input_manifest_path,
                input_fingerprint=input_manifest.get("input_fingerprint"),
            ),
        },
        "training_performed": False,
        "feature_set_selected": False,
    }
    manifest_path = output_root / "manifest.json"
    _write_json(manifest_path, manifest)
    if status != "certified":
        raise CertificationError(f"data_certification_failed:{blocking_checks}")
    return manifest


def evaluate(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    manifest = _read_json(output_root / "manifest.json")
    checks = dict(manifest.get("blocking_checks", {}) or {})
    output_hashes = all(
        Path(record["path"]).is_file()
        and _sha256(Path(record["path"])) == record["sha256"]
        for record in dict(manifest.get("outputs", {}) or {}).values()
    )
    return {
        "status": "ok"
        if manifest.get("status") == "certified"
        and all(checks.values())
        and output_hashes
        else "failed",
        "study_id": manifest.get("study_id"),
        "blocking_checks": checks,
        "output_hashes_match": output_hashes,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready-root", type=Path, default=DEFAULT_READY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE_ROOT)
    parser.add_argument("--qdp-runtime", choices=("safe", "balanced", "fast"), default="balanced")
    parser.add_argument("--qdp-threads", type=int, default=0)
    parser.add_argument("--qdp-shallow", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = (
            evaluate(args.output_root)
            if args.evaluate
            else prepare(
                ready_root=args.ready_root,
                output_root=args.output_root,
                workspace=args.workspace,
                qdp_deep=not args.qdp_shallow,
                qdp_runtime=args.qdp_runtime,
                qdp_threads=int(args.qdp_threads) or None,
            )
        )
    except CertificationError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("status") in {"certified", "ok"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
