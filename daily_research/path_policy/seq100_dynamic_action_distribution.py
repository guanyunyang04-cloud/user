"""Causal opportunity, upside, and downside decomposition for dynamic actions.

The study keeps the continuous dynamic buy advantage as the scientific target.
Its binary positive indicator is only one moment of that distribution, never a
standalone "good stock" label.  Every out-of-sample fold learns from annual
prefix-oracle experience versions available before the fold, and no portfolio
execution is permitted unless the reconstructed expected action value passes a
strict promotion gate under both transaction-cost scenarios.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
from scipy.stats import norm, rankdata

from daily_research.path_policy import (
    seq100_dynamic_action_value_baselines as baseline,
)
from daily_research.path_policy import seq100_market_replay as replay

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = WORKSPACE_ROOT / (
    "daily_research/studies/seq100_dynamic_action_distribution_v1.json"
)
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / (
    "daily_research/output/path_policy/studies/"
    "seq100_dynamic_action_distribution_v1"
)
STUDY_ID = "seq100_dynamic_action_distribution_v1"
SCHEMA_VERSION = 1

MOMENT_NAMES = (
    "positive_probability",
    "upside_component",
    "downside_component",
    "direct_expected_value",
)
MODEL_NAMES = ("prior", "market", "additive", "matched_history")
SELECTION_NAMES = (
    "maximum_expected_value",
    "maximum_positive_probability",
    "maximum_upside_component",
    "minimum_downside_component",
)


def _resolve(path: str | Path) -> Path:
    return replay.resolve_path(path)


def _read_json(path: str | Path) -> dict[str, Any]:
    return replay.read_json(path)


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    baseline._write_json(path, payload)


def _write_frame(
    path: str | Path, frame: pd.DataFrame, *, row_group_size: int
) -> dict[str, Any]:
    return baseline._write_frame(path, frame, row_group_size=row_group_size)


def _write_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return baseline._write_npz(path, arrays)


def load_study(path: str | Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study = _read_json(path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("dynamic_action_distribution_study_id")
    source = dict(study["source"])
    if str(source["quality_pool_name"]) != "quality_liquidity_pit":
        raise ValueError("dynamic_action_distribution_pool")
    if int(source["expected_feature_count"]) != 557:
        raise ValueError("dynamic_action_distribution_feature_count")
    if int(source["forbidden_year"]) != 2026:
        raise ValueError("dynamic_action_distribution_forbidden_year")

    period = dict(study["period"])
    years = tuple(int(value) for value in period["formal_years"])
    oos_years = tuple(int(value) for value in period["strict_oos_years"])
    if years != tuple(range(2012, 2026)) or oos_years != tuple(range(2013, 2026)):
        raise ValueError("dynamic_action_distribution_period")
    if bool(period["daily_first_availability_claimed"]):
        raise ValueError("dynamic_action_distribution_daily_availability")
    if bool(period["pristine_confirmation_claimed"]):
        raise ValueError("dynamic_action_distribution_pristine_claim")

    target = dict(study["target"])
    if not bool(target["continuous_target"]):
        raise ValueError("dynamic_action_distribution_continuous_target")
    if int(target["minimum_sessions_after_resolution"]) != 10:
        raise ValueError("dynamic_action_distribution_maturity")
    if bool(target["fixed_holding_horizon_used"]):
        raise ValueError("dynamic_action_distribution_fixed_horizon")
    if bool(target["binary_good_stock_label_used"]):
        raise ValueError("dynamic_action_distribution_binary_label")
    if bool(target["final_2025_oracle_row_labels_allowed"]):
        raise ValueError("dynamic_action_distribution_final_oracle")
    components = dict(target["distribution_components"])
    if not bool(components["binary_component_is_diagnostic_not_policy_label"]):
        raise ValueError("dynamic_action_distribution_binary_component")

    coordinates = dict(study["coordinates"])
    specs = [dict(item) for item in coordinates["specifications"]]
    names = [str(item["name"]) for item in specs]
    if not names or len(names) != len(set(names)):
        raise ValueError("dynamic_action_distribution_coordinate_names")
    if not set(coordinates["matched_coordinates"]).issubset(names):
        raise ValueError("dynamic_action_distribution_matched_coordinates")
    if not set(coordinates["risk_match_coordinates"]).issubset(names):
        raise ValueError("dynamic_action_distribution_risk_coordinates")
    if int(coordinates["matched_bins"]) != 2:
        raise ValueError("dynamic_action_distribution_matched_bins")
    if int(coordinates["risk_match_bins"]) != 2:
        raise ValueError("dynamic_action_distribution_risk_bins")

    gate = dict(study["activity_gate"])
    if not set(gate["coordinates"]).issubset(names):
        raise ValueError("dynamic_action_distribution_gate_coordinates")
    thresholds = {str(key): float(value) for key, value in gate["thresholds"].items()}
    if thresholds != {"all": 0.0, "top30": 0.7, "top20": 0.8, "top10": 0.9}:
        raise ValueError("dynamic_action_distribution_gate_thresholds")
    if str(gate["primary_training_scope"]) != "top20":
        raise ValueError("dynamic_action_distribution_primary_gate")
    if bool(gate["threshold_selection_performed"]):
        raise ValueError("dynamic_action_distribution_gate_selection")
    if not bool(gate["gate_is_not_a_buy_rule"]):
        raise ValueError("dynamic_action_distribution_gate_rule")

    estimators = dict(study["estimators"])
    if tuple(estimators["training_scopes"]) != ("all", "top20"):
        raise ValueError("dynamic_action_distribution_training_scopes")
    if tuple(estimators["models"]) != MODEL_NAMES:
        raise ValueError("dynamic_action_distribution_models")
    if bool(estimators["hyperparameter_selection_performed"]):
        raise ValueError("dynamic_action_distribution_hyperparameter_selection")
    return study


def _source_contract(
    study: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[tuple[str, int, int], dict[str, Any]],
    dict[str, int],
]:
    source = dict(study["source"])
    requests = (
        ("baseline_study", "expected_baseline_study_sha256"),
        ("baseline_manifest", "expected_baseline_manifest_sha256"),
        ("baseline_validation", "expected_baseline_validation_sha256"),
    )
    records: dict[str, Any] = {}
    for path_key, hash_key in requests:
        path = _resolve(source[path_key])
        digest = replay.sha256(path)
        if digest != str(source[hash_key]):
            raise ValueError(f"dynamic_action_distribution_source_hash:{path_key}")
        records[path_key] = {"path": str(path.resolve()), "sha256": digest}

    baseline_study_path = _resolve(source["baseline_study"])
    baseline_study = baseline.load_study(baseline_study_path)
    baseline_manifest = _read_json(source["baseline_manifest"])
    baseline_validation = _read_json(source["baseline_validation"])
    if baseline_manifest.get("status") != "completed":
        raise ValueError("dynamic_action_distribution_baseline_incomplete")
    if baseline_validation.get("status") != "passed":
        raise ValueError("dynamic_action_distribution_baseline_validation")
    if bool(baseline_manifest.get("portfolio_execution_performed")):
        raise ValueError("dynamic_action_distribution_baseline_execution")
    if bool(baseline_manifest.get("profit_claim_allowed")):
        raise ValueError("dynamic_action_distribution_baseline_profit_claim")
    if str(dict(baseline_manifest["study"])["sha256"]) != replay.sha256(
        baseline_study_path
    ):
        raise ValueError("dynamic_action_distribution_baseline_study_link")

    (
        inherited_contract,
        prefix_manifest,
        input_manifest,
        version_records,
        feature_index,
    ) = baseline._source_contract(baseline_study)
    if int(input_manifest["row_count"]) != int(
        dict(baseline_manifest["source_audit"])["model_input_rows"]
    ):
        raise ValueError("dynamic_action_distribution_input_rows")
    records["inherited_source_contract"] = inherited_contract
    return records, prefix_manifest, input_manifest, version_records, feature_index


def _activity_rank(
    values: np.ndarray,
    dates: np.ndarray,
    *,
    coordinate_indices: np.ndarray,
    minimum_finite: int,
) -> np.ndarray:
    selected = np.asarray(values, dtype=np.float64)[:, coordinate_indices]
    finite = np.isfinite(selected)
    counts = finite.sum(axis=1)
    totals = np.where(finite, selected, 0.0).sum(axis=1)
    score = np.divide(
        totals,
        counts,
        out=np.full(len(selected), np.nan, dtype=np.float64),
        where=counts >= int(minimum_finite),
    )
    return baseline._same_date_percentile(score, np.asarray(dates))


def _gate_mask(activity_rank: np.ndarray, threshold: float) -> np.ndarray:
    rank = np.asarray(activity_rank, dtype=np.float64)
    return np.isfinite(rank) & (rank >= float(threshold))


class ActivityRankStore:
    def __init__(
        self,
        *,
        coordinates: baseline.CoordinateStore,
        coordinate_indices: np.ndarray,
        minimum_finite: int,
        cache_years: int,
    ) -> None:
        self.coordinates = coordinates
        self.coordinate_indices = np.asarray(coordinate_indices, dtype=np.int16)
        self.minimum_finite = int(minimum_finite)
        self.cache_years = max(int(cache_years), 1)
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def load(self, year: int) -> np.ndarray:
        requested = int(year)
        cached = self.cache.get(requested)
        if cached is not None:
            self.cache.move_to_end(requested)
            return cached
        coordinate_year = self.coordinates.load(requested)
        result = _activity_rank(
            coordinate_year.values,
            coordinate_year.frame["date_idx"].to_numpy(np.int64),
            coordinate_indices=self.coordinate_indices,
            minimum_finite=self.minimum_finite,
        )
        self.cache[requested] = result
        self.cache.move_to_end(requested)
        while len(self.cache) > self.cache_years:
            self.cache.popitem(last=False)
        return result

    def clear(self) -> None:
        self.cache.clear()
        gc.collect()


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


class DistributionAccumulator:
    def __init__(
        self,
        *,
        market_indices: np.ndarray,
        stock_indices: np.ndarray,
        matched_indices: np.ndarray,
        univariate_bins: int,
        matched_bins: int,
    ) -> None:
        self.market_indices = np.asarray(market_indices, dtype=np.int16)
        self.stock_indices = np.asarray(stock_indices, dtype=np.int16)
        self.matched_indices = np.asarray(matched_indices, dtype=np.int16)
        self.univariate_bins = int(univariate_bins)
        self.matched_bins = int(matched_bins)
        self.bin_size = self.univariate_bins + 1
        self.cell_size = (self.matched_bins + 1) ** len(self.matched_indices)
        moment_count = len(MOMENT_NAMES)

        self.global_date_count = 0.0
        self.global_sum = np.zeros(moment_count, dtype=np.float64)
        self.global_sum2 = np.zeros(moment_count, dtype=np.float64)
        market_shape = (len(self.market_indices), self.bin_size)
        stock_shape = (len(self.stock_indices), self.bin_size)
        self.market_count = np.zeros(market_shape, dtype=np.float64)
        self.market_sum = np.zeros((moment_count, *market_shape), dtype=np.float64)
        self.market_sum2 = np.zeros((moment_count, *market_shape), dtype=np.float64)
        self.stock_weight = np.zeros(stock_shape, dtype=np.float64)
        self.stock_sum = np.zeros((moment_count, *stock_shape), dtype=np.float64)
        self.stock_sum2 = np.zeros((moment_count, *stock_shape), dtype=np.float64)
        self.stock_date_count = np.zeros(stock_shape, dtype=np.float64)
        self.cell_weight = np.zeros(self.cell_size, dtype=np.float64)
        self.cell_sum = np.zeros((moment_count, self.cell_size), dtype=np.float64)
        self.cell_sum2 = np.zeros((moment_count, self.cell_size), dtype=np.float64)
        self.cell_date_count = np.zeros(self.cell_size, dtype=np.float64)
        self.training_rows = 0

    def apply(
        self,
        experience: baseline.AlignedExperience,
        *,
        sign: int,
        mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        direction = int(sign)
        if direction not in (-1, 1):
            raise ValueError("dynamic_action_distribution_accumulator_sign")
        if mask is None:
            selected = np.ones(len(experience.target), dtype=bool)
        else:
            selected = np.asarray(mask, dtype=bool)
            if len(selected) != len(experience.target):
                raise ValueError("dynamic_action_distribution_mask_alignment")
        y = np.asarray(experience.target, dtype=np.float64)[selected]
        x = np.asarray(experience.coordinates, dtype=np.float64)[selected]
        dates = np.asarray(experience.date_idx, dtype=np.int64)[selected]
        if not len(y):
            return {"rows": 0, "dates": 0, "target_mean": math.nan}
        if bool((dates[1:] < dates[:-1]).any()):
            raise ValueError("dynamic_action_distribution_date_order")

        moments = _target_moments(y)
        _unique_dates, starts, counts = np.unique(
            dates, return_index=True, return_counts=True
        )
        group_index = np.repeat(np.arange(len(starts), dtype=np.int64), counts)
        daily_mean = np.vstack(
            [
                np.add.reduceat(moments[:, index], starts)
                / counts.astype(np.float64)
                for index in range(moments.shape[1])
            ]
        ).T
        residual = moments - daily_mean[group_index]
        weights = 1.0 / counts[group_index].astype(np.float64)

        self.global_date_count += direction * len(starts)
        self.global_sum += direction * daily_mean.sum(axis=0)
        self.global_sum2 += direction * np.square(daily_mean).sum(axis=0)

        market_bins = baseline._univariate_bins(
            x[starts][:, self.market_indices], bin_count=self.univariate_bins
        )
        for column in range(market_bins.shape[1]):
            code = market_bins[:, column]
            self.market_count[column] += direction * np.bincount(
                code, minlength=self.bin_size
            )
            for moment in range(moments.shape[1]):
                self.market_sum[moment, column] += direction * np.bincount(
                    code, weights=daily_mean[:, moment], minlength=self.bin_size
                )
                self.market_sum2[moment, column] += direction * np.bincount(
                    code,
                    weights=np.square(daily_mean[:, moment]),
                    minlength=self.bin_size,
                )

        stock_bins = baseline._univariate_bins(
            x[:, self.stock_indices], bin_count=self.univariate_bins
        )
        for column in range(stock_bins.shape[1]):
            code = stock_bins[:, column]
            self.stock_weight[column] += direction * np.bincount(
                code, weights=weights, minlength=self.bin_size
            )
            self.stock_date_count[column] += direction * baseline._date_presence(
                code, group_index, self.bin_size
            )
            for moment in range(moments.shape[1]):
                self.stock_sum[moment, column] += direction * np.bincount(
                    code,
                    weights=weights * residual[:, moment],
                    minlength=self.bin_size,
                )
                self.stock_sum2[moment, column] += direction * np.bincount(
                    code,
                    weights=weights * np.square(residual[:, moment]),
                    minlength=self.bin_size,
                )

        cell_code = baseline._matched_codes(
            x,
            matched_indices=self.matched_indices,
            bin_count=self.matched_bins,
        )
        self.cell_weight += direction * np.bincount(
            cell_code, weights=weights, minlength=self.cell_size
        )
        self.cell_date_count += direction * baseline._date_presence(
            cell_code, group_index, self.cell_size
        )
        for moment in range(moments.shape[1]):
            self.cell_sum[moment] += direction * np.bincount(
                cell_code,
                weights=weights * moments[:, moment],
                minlength=self.cell_size,
            )
            self.cell_sum2[moment] += direction * np.bincount(
                cell_code,
                weights=weights * np.square(moments[:, moment]),
                minlength=self.cell_size,
            )
        self.training_rows += direction * len(y)
        return {
            "rows": len(y),
            "dates": len(starts),
            "target_mean": float(y.mean()),
            "positive_fraction": float(np.mean(y > 0.0)),
        }

    def assert_valid(self) -> None:
        tolerance = 1.0e-8
        arrays = (
            self.market_count,
            self.stock_weight,
            self.stock_date_count,
            self.cell_weight,
            self.cell_date_count,
        )
        if self.global_date_count <= 0.0 or self.training_rows <= 0:
            raise ValueError("dynamic_action_distribution_empty_training")
        if any(bool((values < -tolerance).any()) for values in arrays):
            raise ValueError("dynamic_action_distribution_negative_stat")
        if bool((self.cell_sum[0] - self.cell_weight > tolerance).any()):
            raise ValueError("dynamic_action_distribution_probability_weight")


def _moment_mean_and_se(
    *,
    count: np.ndarray,
    total: np.ndarray,
    total2: np.ndarray,
    date_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    expanded_count = np.broadcast_to(count, total.shape)
    expanded_dates = np.broadcast_to(date_count, total.shape)
    return baseline._mean_and_se(
        count=expanded_count,
        total=total,
        total2=total2,
        date_count=expanded_dates,
    )


def build_snapshot(
    *,
    accumulator: DistributionAccumulator,
    study: Mapping[str, Any],
    coordinate_names: Sequence[str],
    market_indices: np.ndarray,
    stock_indices: np.ndarray,
    families: Mapping[str, Sequence[int]],
    matched_indices: np.ndarray,
    scope: str,
    training_cutoff_year: int,
    maximum_label_as_of_year: int,
    maximum_signal_year: int,
) -> dict[str, np.ndarray]:
    accumulator.assert_valid()
    estimators = dict(study["estimators"])
    market_alpha = float(
        estimators["market_univariate_smoothing_date_equivalents"]
    )
    stock_alpha = float(
        estimators["stock_univariate_smoothing_date_equivalents"]
    )
    matched_alpha = float(estimators["matched_cell_smoothing_date_equivalents"])

    prior = accumulator.global_sum / accumulator.global_date_count
    prior_variance = np.maximum(
        accumulator.global_sum2 / accumulator.global_date_count - np.square(prior),
        0.0,
    )
    prior_se = np.sqrt(prior_variance / accumulator.global_date_count)

    market_mean, market_raw_se = _moment_mean_and_se(
        count=accumulator.market_count,
        total=accumulator.market_sum,
        total2=accumulator.market_sum2,
        date_count=accumulator.market_count,
    )
    market_lambda = accumulator.market_count / (
        accumulator.market_count + market_alpha
    )
    market_effect = market_lambda[None, :, :] * (
        market_mean - prior[:, None, None]
    )
    market_se = market_lambda[None, :, :] * np.nan_to_num(
        market_raw_se, nan=prior_se[:, None, None]
    )

    stock_mean, stock_raw_se = _moment_mean_and_se(
        count=accumulator.stock_weight,
        total=accumulator.stock_sum,
        total2=accumulator.stock_sum2,
        date_count=accumulator.stock_date_count,
    )
    stock_lambda = accumulator.stock_weight / (
        accumulator.stock_weight + stock_alpha
    )
    stock_effect = stock_lambda[None, :, :] * stock_mean
    stock_se = stock_lambda[None, :, :] * np.nan_to_num(
        stock_raw_se, nan=prior_se[:, None, None]
    )

    cell_mean, cell_raw_se = _moment_mean_and_se(
        count=accumulator.cell_weight,
        total=accumulator.cell_sum,
        total2=accumulator.cell_sum2,
        date_count=accumulator.cell_date_count,
    )
    cell_lambda = accumulator.cell_weight / (
        accumulator.cell_weight + matched_alpha
    )
    family_names = list(families)
    family_index_by_stock = np.full(len(stock_indices), -1, dtype=np.int16)
    for family_index, family in enumerate(family_names):
        family_index_by_stock[np.asarray(families[family], dtype=np.int64)] = family_index
    if bool((family_index_by_stock < 0).any()):
        raise ValueError("dynamic_action_distribution_family_coverage")

    return {
        "scope": np.asarray([scope], dtype="U16"),
        "moment_names": np.asarray(MOMENT_NAMES, dtype="U32"),
        "prior": prior.astype(np.float64),
        "prior_se": prior_se.astype(np.float64),
        "market_effect": market_effect.astype(np.float64),
        "market_se": market_se.astype(np.float64),
        "stock_effect": stock_effect.astype(np.float64),
        "stock_se": stock_se.astype(np.float64),
        "cell_mean": cell_mean.astype(np.float64),
        "cell_se": np.nan_to_num(
            cell_raw_se, nan=prior_se[:, None]
        ).astype(np.float64),
        "cell_lambda": cell_lambda.astype(np.float64),
        "cell_weight": accumulator.cell_weight.astype(np.float64),
        "cell_date_count": accumulator.cell_date_count.astype(np.float64),
        "coordinate_names": np.asarray(list(coordinate_names), dtype="U64"),
        "market_indices": np.asarray(market_indices, dtype=np.int16),
        "stock_indices": np.asarray(stock_indices, dtype=np.int16),
        "matched_indices": np.asarray(matched_indices, dtype=np.int16),
        "family_names": np.asarray(family_names, dtype="U32"),
        "family_index_by_stock": family_index_by_stock,
        "univariate_bins": np.asarray([accumulator.univariate_bins], dtype=np.int16),
        "matched_bins": np.asarray([accumulator.matched_bins], dtype=np.int16),
        "training_rows": np.asarray([accumulator.training_rows], dtype=np.int64),
        "training_dates": np.asarray(
            [round(accumulator.global_date_count)], dtype=np.int32
        ),
        "training_cutoff_year": np.asarray([training_cutoff_year], dtype=np.int16),
        "maximum_label_as_of_year": np.asarray(
            [maximum_label_as_of_year], dtype=np.int16
        ),
        "maximum_signal_year": np.asarray([maximum_signal_year], dtype=np.int16),
    }


def _lookup_mean_effect(table: np.ndarray, bins: np.ndarray) -> np.ndarray:
    effects = np.zeros((len(bins), table.shape[0]), dtype=np.float64)
    for column in range(bins.shape[1]):
        effects += table[:, column, bins[:, column]].T
    return effects / max(bins.shape[1], 1)


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


def predict_from_snapshot(
    values: np.ndarray, snapshot: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    x = np.asarray(values, dtype=np.float64)
    prior = np.asarray(snapshot["prior"], dtype=np.float64)
    univariate_bins = int(np.asarray(snapshot["univariate_bins"])[0])
    matched_bins = int(np.asarray(snapshot["matched_bins"])[0])
    market_indices = np.asarray(snapshot["market_indices"], dtype=np.int64)
    stock_indices = np.asarray(snapshot["stock_indices"], dtype=np.int64)
    matched_indices = np.asarray(snapshot["matched_indices"], dtype=np.int64)
    family_names = [str(value) for value in np.asarray(snapshot["family_names"])]
    family_index_by_stock = np.asarray(
        snapshot["family_index_by_stock"], dtype=np.int64
    )

    raw_prior = np.broadcast_to(prior, (len(x), len(prior))).copy()
    market_bins = baseline._univariate_bins(
        x[:, market_indices], bin_count=univariate_bins
    )
    raw_market = raw_prior + _lookup_mean_effect(
        np.asarray(snapshot["market_effect"], dtype=np.float64), market_bins
    )

    stock_bins = baseline._univariate_bins(
        x[:, stock_indices], bin_count=univariate_bins
    )
    stock_table = np.asarray(snapshot["stock_effect"], dtype=np.float64)
    family_effects: list[np.ndarray] = []
    for family_index, _family in enumerate(family_names):
        columns = np.flatnonzero(family_index_by_stock == family_index)
        effect = np.zeros((len(x), len(MOMENT_NAMES)), dtype=np.float64)
        for column in columns:
            effect += stock_table[:, column, stock_bins[:, column]].T
        family_effects.append(effect / max(len(columns), 1))
    raw_additive = raw_market + np.mean(np.stack(family_effects, axis=0), axis=0)

    cell_code = baseline._matched_codes(
        x, matched_indices=matched_indices, bin_count=matched_bins
    )
    cell_lambda = np.asarray(snapshot["cell_lambda"], dtype=np.float64)[cell_code]
    cell_mean = np.asarray(snapshot["cell_mean"], dtype=np.float64)[:, cell_code].T
    raw_matched = (
        cell_lambda[:, None] * cell_mean
        + (1.0 - cell_lambda[:, None]) * raw_additive
    )

    output: dict[str, np.ndarray] = {
        "matched_cell_code": cell_code.astype(np.int32),
        "matched_cell_weight": np.asarray(
            snapshot["cell_weight"], dtype=np.float64
        )[cell_code],
        "matched_cell_date_count": np.asarray(
            snapshot["cell_date_count"], dtype=np.float64
        )[cell_code],
    }
    for model, raw in (
        ("prior", raw_prior),
        ("market", raw_market),
        ("additive", raw_additive),
        ("matched_history", raw_matched),
    ):
        projected = _project_moments(raw)
        for name, result in projected.items():
            output[f"predicted_{model}_{name}"] = result
    return output


def _snapshot_record(
    *,
    root: Path,
    cost_scenario: str,
    oos_year: int,
    scope: str,
    snapshot: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    path = (
        root
        / "model_states"
        / f"cost={cost_scenario}"
        / f"oos_year={int(oos_year)}"
        / f"scope={scope}"
        / "state.npz"
    )
    record = _write_npz(path, snapshot)
    record.update(
        {
            "cost_scenario": cost_scenario,
            "oos_year": int(oos_year),
            "scope": scope,
        }
    )
    return record


def _prediction_frame(
    *,
    experience: baseline.AlignedExperience,
    coordinate_year: baseline.CoordinateYear,
    hot_mask: np.ndarray,
    activity_rank: np.ndarray,
    risk_cell_code: np.ndarray,
    predictions_by_scope: Mapping[str, Mapping[str, np.ndarray]],
    training_cutoff_year: int,
    maximum_label_as_of_year: int,
    maximum_signal_year: int,
) -> pd.DataFrame:
    positions = experience.coordinate_positions[hot_mask]
    target = experience.target[hot_mask]
    output: dict[str, Any] = {
        "cost_scenario": np.full(len(positions), experience.cost_scenario, dtype=object),
        "oos_year": np.full(len(positions), experience.signal_year, dtype=np.int16),
        "training_cutoff_year": np.full(
            len(positions), int(training_cutoff_year), dtype=np.int16
        ),
        "maximum_training_label_as_of_year": np.full(
            len(positions), int(maximum_label_as_of_year), dtype=np.int16
        ),
        "maximum_training_signal_year": np.full(
            len(positions), int(maximum_signal_year), dtype=np.int16
        ),
        "evaluation_as_of_year": np.full(
            len(positions), experience.as_of_year, dtype=np.int16
        ),
        "input_row_idx": coordinate_year.frame["input_row_idx"].to_numpy(np.int64)[
            positions
        ],
        "trade_date": experience.trade_date[hot_mask],
        "date_idx": experience.date_idx[hot_mask],
        "symbol_idx": experience.symbol_idx[hot_mask],
        "activity_rank": np.asarray(activity_rank[hot_mask], dtype=np.float32),
        "risk_match_cell_code": np.asarray(risk_cell_code, dtype=np.int16),
        "actual_buy_advantage_vs_cash": target,
        "actual_positive": target > 0.0,
        "actual_upside_component": np.maximum(target, 0.0),
        "actual_downside_component": np.maximum(-target, 0.0),
        "prefix_oracle_cash_buy_selected": experience.selected[hot_mask],
        "evaluation_sessions_after_resolution": experience.sessions_after_resolution[
            hot_mask
        ],
    }
    for scope, predictions in predictions_by_scope.items():
        for name, values in predictions.items():
            if name in {
                "matched_cell_code",
                "matched_cell_weight",
                "matched_cell_date_count",
            }:
                output[f"{scope}_{name}"] = np.asarray(values)
            else:
                suffix = name.removeprefix("predicted_")
                output[f"predicted_{scope}_{suffix}"] = np.asarray(
                    values, dtype=np.float32
                )
    return pd.DataFrame(output)


def _gate_daily_rows(
    *,
    experience: baseline.AlignedExperience,
    activity_rank: np.ndarray,
    thresholds: Mapping[str, float],
) -> list[dict[str, Any]]:
    dates = np.asarray(experience.date_idx, dtype=np.int64)
    trade_dates = np.asarray(experience.trade_date)
    target = np.asarray(experience.target, dtype=np.float64)
    selected = np.asarray(experience.selected, dtype=bool)
    unique_dates, starts, counts = np.unique(dates, return_index=True, return_counts=True)
    rows: list[dict[str, Any]] = []
    for date, start, count in zip(unique_dates, starts, counts, strict=True):
        stop = int(start + count)
        for scope, threshold in thresholds.items():
            mask = _gate_mask(activity_rank[start:stop], threshold)
            scoped = target[start:stop][mask]
            if not len(scoped):
                continue
            rows.append(
                {
                    "cost_scenario": experience.cost_scenario,
                    "oos_year": experience.signal_year,
                    "trade_date": str(trade_dates[start]),
                    "date_idx": int(date),
                    "gate_scope": scope,
                    "rows": len(scoped),
                    "actual_mean": float(scoped.mean()),
                    "positive_fraction": float(np.mean(scoped > 0.0)),
                    "upside_component": float(np.maximum(scoped, 0.0).mean()),
                    "downside_component": float(np.maximum(-scoped, 0.0).mean()),
                    "oracle_selected_rate": float(selected[start:stop][mask].mean()),
                }
            )
    return rows


def _binary_auc(actual: np.ndarray, predicted: np.ndarray) -> float:
    labels = np.asarray(actual, dtype=bool)
    scores = np.asarray(predicted, dtype=np.float64)
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = rankdata(scores, method="average")
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def _model_prefixes(study: Mapping[str, Any]) -> list[str]:
    scopes = [str(value) for value in dict(study["estimators"])["training_scopes"]]
    return [f"{scope}_{model}" for scope in scopes for model in MODEL_NAMES]


def _evaluate_prediction_partition(
    frame: pd.DataFrame, *, model_prefix: str
) -> list[dict[str, Any]]:
    dates = frame["date_idx"].to_numpy(np.int64)
    if bool((dates[1:] < dates[:-1]).any()):
        raise ValueError("dynamic_action_distribution_prediction_order")
    trade_dates = frame["trade_date"].astype(str).to_numpy()
    actual = frame["actual_buy_advantage_vs_cash"].to_numpy(np.float64)
    positive = actual > 0.0
    actual_up = np.maximum(actual, 0.0)
    actual_down = np.maximum(-actual, 0.0)
    probability = frame[
        f"predicted_{model_prefix}_positive_probability"
    ].to_numpy(np.float64)
    upside = frame[f"predicted_{model_prefix}_upside_component"].to_numpy(
        np.float64
    )
    downside = frame[f"predicted_{model_prefix}_downside_component"].to_numpy(
        np.float64
    )
    expected = frame[f"predicted_{model_prefix}_expected_value"].to_numpy(
        np.float64
    )
    direct = frame[f"predicted_{model_prefix}_direct_expected_value"].to_numpy(
        np.float64
    )
    gap = frame[f"predicted_{model_prefix}_reconstruction_gap"].to_numpy(
        np.float64
    )
    unique_dates, starts, counts = np.unique(dates, return_index=True, return_counts=True)
    rows: list[dict[str, Any]] = []
    for date, start, count in zip(unique_dates, starts, counts, strict=True):
        stop = int(start + count)
        local = slice(start, stop)
        clipped_probability = np.clip(probability[local], 1.0e-6, 1.0 - 1.0e-6)
        row: dict[str, Any] = {
            "cost_scenario": str(frame["cost_scenario"].iloc[start]),
            "oos_year": int(frame["oos_year"].iloc[start]),
            "trade_date": str(trade_dates[start]),
            "date_idx": int(date),
            "model": model_prefix,
            "rows": int(count),
            "brier": float(np.square(clipped_probability - positive[local]).mean()),
            "log_loss": float(
                -np.mean(
                    positive[local] * np.log(clipped_probability)
                    + (~positive[local]) * np.log1p(-clipped_probability)
                )
            ),
            "binary_rank_auc": _binary_auc(positive[local], probability[local]),
            "upside_mae": float(np.abs(upside[local] - actual_up[local]).mean()),
            "downside_mae": float(
                np.abs(downside[local] - actual_down[local]).mean()
            ),
            "direct_value_mae": float(np.abs(direct[local] - actual[local]).mean()),
            "reconstruction_gap_abs": float(np.abs(gap[local]).mean()),
            "spearman": baseline._safe_spearman(actual[local], expected[local]),
        }
        selection_indices = {
            "maximum_expected_value": int(np.argmax(expected[local])),
            "maximum_positive_probability": int(np.argmax(probability[local])),
            "maximum_upside_component": int(np.argmax(upside[local])),
            "minimum_downside_component": int(np.argmin(downside[local])),
        }
        for selection_name, local_index in selection_indices.items():
            index = start + local_index
            row[f"{selection_name}_actual"] = float(actual[index])
            row[f"{selection_name}_actual_positive"] = bool(positive[index])
            row[f"{selection_name}_actual_upside"] = float(actual_up[index])
            row[f"{selection_name}_actual_downside"] = float(actual_down[index])
        chosen = start + selection_indices["maximum_expected_value"]
        action = bool(expected[chosen] > 0.0)
        row["maximum_expected_value_prediction"] = float(expected[chosen])
        row["expected_value_action"] = action
        row["expected_value_policy_value"] = float(actual[chosen]) if action else 0.0
        rows.append(row)
    return rows


def evaluate_predictions(
    *,
    study: Mapping[str, Any],
    prediction_records: Sequence[Mapping[str, Any]],
    output_root: Path,
    row_group_size: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model_prefixes = _model_prefixes(study)
    daily_rows: list[dict[str, Any]] = []
    for record in prediction_records:
        frame = pd.read_parquet(record["path"])
        for model_prefix in model_prefixes:
            daily_rows.extend(
                _evaluate_prediction_partition(frame, model_prefix=model_prefix)
            )
        del frame
        gc.collect()
    daily = pd.DataFrame(daily_rows).sort_values(
        ["cost_scenario", "model", "trade_date"], kind="mergesort"
    )
    metric_columns = [
        "brier",
        "log_loss",
        "binary_rank_auc",
        "upside_mae",
        "downside_mae",
        "direct_value_mae",
        "reconstruction_gap_abs",
        "spearman",
        "maximum_expected_value_actual",
        "maximum_expected_value_actual_positive",
        "maximum_positive_probability_actual",
        "maximum_positive_probability_actual_positive",
        "maximum_upside_component_actual",
        "maximum_upside_component_actual_positive",
        "minimum_downside_component_actual",
        "minimum_downside_component_actual_positive",
        "expected_value_policy_value",
        "expected_value_action",
    ]
    annual = (
        daily.groupby(["cost_scenario", "oos_year", "model"], sort=True)[
            metric_columns
        ]
        .mean()
        .reset_index()
    )
    annual["dates"] = (
        daily.groupby(["cost_scenario", "oos_year", "model"], sort=True)
        .size()
        .to_numpy()
    )

    aggregate_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for (cost, model), group in daily.groupby(["cost_scenario", "model"], sort=True):
        policy_mean, policy_se, policy_lower = baseline._hac_mean(
            group["expected_value_policy_value"].to_numpy(np.float64), lag=20
        )
        _top_mean, top_se, top_lower = baseline._hac_mean(
            group["maximum_expected_value_actual"].to_numpy(np.float64), lag=20
        )
        scoped_annual = annual[
            annual["cost_scenario"].eq(cost) & annual["model"].eq(model)
        ]
        aggregate_rows.append(
            {
                "cost_scenario": cost,
                "model": model,
                "dates": len(group),
                "oos_years": scoped_annual["oos_year"].nunique(),
                **{
                    name: float(group[name].mean())
                    for name in metric_columns
                    if name not in {"expected_value_policy_value"}
                },
                "expected_value_policy_value": policy_mean,
                "policy_hac_standard_error": policy_se,
                "policy_hac_lower_95": policy_lower,
                "top_expected_hac_standard_error": top_se,
                "top_expected_hac_lower_95": top_lower,
                "positive_policy_years": int(
                    (scoped_annual["expected_value_policy_value"] > 0.0).sum()
                ),
                "positive_spearman_years": int(
                    (scoped_annual["spearman"] > 0.0).sum()
                ),
            }
        )
        for selection in SELECTION_NAMES:
            actual_name = f"{selection}_actual"
            positive_name = f"{selection}_actual_positive"
            mean, standard_error, lower = baseline._hac_mean(
                group[actual_name].to_numpy(np.float64), lag=20
            )
            selection_rows.append(
                {
                    "cost_scenario": cost,
                    "model": model,
                    "selection": selection,
                    "dates": len(group),
                    "mean_realized_advantage": mean,
                    "hac_standard_error": standard_error,
                    "hac_lower_95": lower,
                    "positive_fraction": float(group[positive_name].mean()),
                    "mean_realized_upside": float(
                        group[f"{selection}_actual_upside"].mean()
                    ),
                    "mean_realized_downside": float(
                        group[f"{selection}_actual_downside"].mean()
                    ),
                }
            )
        passed = bool(
            policy_lower > 0.0
            and int((scoped_annual["expected_value_policy_value"] > 0.0).sum()) >= 7
            and float(group["spearman"].mean()) > 0.0
            and float(group["reconstruction_gap_abs"].mean()) <= 5.0e-4
        )
        gate_rows.append(
            {
                "cost_scenario": cost,
                "model": model,
                "passed_cost_gate": passed,
                "policy_hac_lower_95": policy_lower,
                "positive_policy_years": int(
                    (scoped_annual["expected_value_policy_value"] > 0.0).sum()
                ),
                "mean_spearman": float(group["spearman"].mean()),
                "mean_reconstruction_gap_abs": float(
                    group["reconstruction_gap_abs"].mean()
                ),
            }
        )

    aggregate = pd.DataFrame(aggregate_rows).sort_values(
        ["cost_scenario", "model"], kind="mergesort"
    )
    selection = pd.DataFrame(selection_rows).sort_values(
        ["cost_scenario", "model", "selection"], kind="mergesort"
    )
    gate_frame = pd.DataFrame(gate_rows)
    passed_models = sorted(
        model
        for model, group in gate_frame.groupby("model", sort=True)
        if set(group["cost_scenario"]) == {"base", "double_slippage"}
        and bool(group["passed_cost_gate"].all())
    )
    promotion_gate = {
        "schema": "seq100_dynamic_action_distribution_promotion_gate/1",
        "passed_models": passed_models,
        "account_replay_allowed": bool(passed_models),
        "profit_claim_allowed": False,
        "cost_model_results": json.loads(gate_frame.to_json(orient="records")),
    }

    outputs = {
        "daily_model_metrics": _write_frame(
            output_root / "daily_model_metrics.parquet",
            daily,
            row_group_size=row_group_size,
        ),
        "annual_model_metrics": _write_frame(
            output_root / "annual_model_metrics.parquet",
            annual,
            row_group_size=row_group_size,
        ),
        "aggregate_model_metrics": _write_frame(
            output_root / "aggregate_model_metrics.parquet",
            aggregate,
            row_group_size=row_group_size,
        ),
        "selection_decomposition": _write_frame(
            output_root / "selection_decomposition.parquet",
            selection,
            row_group_size=row_group_size,
        ),
    }
    _write_json(output_root / "promotion_gate.json", promotion_gate)
    outputs["promotion_gate"] = baseline._file_record(
        output_root / "promotion_gate.json"
    )
    audit = {
        "daily_model_metric_rows": len(daily),
        "annual_model_metric_rows": len(annual),
        "aggregate_model_metric_rows": len(aggregate),
        "passed_models": passed_models,
        "account_replay_allowed": bool(passed_models),
    }
    return outputs, audit, promotion_gate


def build_gate_diagnostics(
    *,
    daily: pd.DataFrame,
    output_root: Path,
    row_group_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metric_columns = [
        "actual_mean",
        "positive_fraction",
        "upside_component",
        "downside_component",
        "oracle_selected_rate",
        "rows",
    ]
    annual = (
        daily.groupby(["cost_scenario", "oos_year", "gate_scope"], sort=True)[
            metric_columns
        ]
        .mean()
        .reset_index()
    )
    aggregate_rows: list[dict[str, Any]] = []
    for (cost, scope), group in daily.groupby(
        ["cost_scenario", "gate_scope"], sort=True
    ):
        mean, standard_error, lower = baseline._hac_mean(
            group["actual_mean"].to_numpy(np.float64), lag=20
        )
        aggregate_rows.append(
            {
                "cost_scenario": cost,
                "gate_scope": scope,
                "dates": len(group),
                "actual_mean": mean,
                "actual_mean_hac_standard_error": standard_error,
                "actual_mean_hac_lower_95": lower,
                "positive_fraction": float(group["positive_fraction"].mean()),
                "upside_component": float(group["upside_component"].mean()),
                "downside_component": float(group["downside_component"].mean()),
                "oracle_selected_rate": float(group["oracle_selected_rate"].mean()),
                "mean_rows": float(group["rows"].mean()),
                "positive_mean_years": int(
                    (
                        annual["cost_scenario"].eq(cost)
                        & annual["gate_scope"].eq(scope)
                        & annual["actual_mean"].gt(0.0)
                    ).sum()
                ),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows).sort_values(
        ["cost_scenario", "gate_scope"], kind="mergesort"
    )
    return (
        {
            "daily_gate_diagnostics": _write_frame(
                output_root / "daily_gate_diagnostics.parquet",
                daily,
                row_group_size=row_group_size,
            ),
            "annual_gate_diagnostics": _write_frame(
                output_root / "annual_gate_diagnostics.parquet",
                annual,
                row_group_size=row_group_size,
            ),
            "aggregate_gate_diagnostics": _write_frame(
                output_root / "aggregate_gate_diagnostics.parquet",
                aggregate,
                row_group_size=row_group_size,
            ),
        },
        {
            "daily_gate_diagnostic_rows": len(daily),
            "aggregate_gate_diagnostic_rows": len(aggregate),
        },
    )


def _collapse_group_values_to_dates(
    values: np.ndarray, *, group_dates: np.ndarray, date_count: int
) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(source)
    result = np.full(int(date_count), np.nan, dtype=np.float64)
    if not finite.any():
        return result
    sums = np.bincount(
        group_dates[finite], weights=source[finite], minlength=int(date_count)
    )
    counts = np.bincount(group_dates[finite], minlength=int(date_count))
    np.divide(sums, counts, out=result, where=counts > 0)
    return result


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    result = np.full(len(p), np.nan, dtype=np.float64)
    finite_positions = np.flatnonzero(np.isfinite(p))
    if not len(finite_positions):
        return result
    order = finite_positions[np.argsort(p[finite_positions], kind="mergesort")]
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def _normal_two_sided_p(mean: float, standard_error: float) -> float:
    if not np.isfinite(mean) or not np.isfinite(standard_error):
        return math.nan
    if standard_error <= 0.0:
        return 0.0 if abs(mean) > 0.0 else 1.0
    return float(2.0 * norm.sf(abs(mean / standard_error)))


def build_matched_failure_diagnostics(
    *,
    study: Mapping[str, Any],
    prediction_records: Sequence[Mapping[str, Any]],
    coordinate_records: Sequence[Mapping[str, Any]],
    coordinate_names: Sequence[str],
    output_root: Path,
    row_group_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    specs = [dict(item) for item in dict(study["coordinates"])["specifications"]]
    name_to_index = {name: index for index, name in enumerate(coordinate_names)}
    risk_names = [
        str(value) for value in dict(study["coordinates"])["risk_match_coordinates"]
    ]
    risk_indices = np.asarray([name_to_index[name] for name in risk_names], dtype=np.int16)
    discriminator_names = [
        str(item["name"])
        for item in specs
        if str(item["family"]) != "market"
        and str(item["name"]) not in set(risk_names)
    ]
    discriminator_indices = [name_to_index[name] for name in discriminator_names]
    risk_bins = int(dict(study["coordinates"])["risk_match_bins"])
    cell_size = (risk_bins + 1) ** len(risk_indices)
    coordinate_by_year = {
        int(record["year"]): dict(record) for record in coordinate_records
    }

    daily_rows: list[dict[str, Any]] = []
    availability_rows: list[dict[str, Any]] = []
    for record in prediction_records:
        cost = str(record["cost_scenario"])
        year = int(record["oos_year"])
        prediction = pd.read_parquet(
            record["path"],
            columns=[
                "trade_date",
                "date_idx",
                "input_row_idx",
                "actual_buy_advantage_vs_cash",
                "actual_positive",
                "risk_match_cell_code",
            ],
        )
        coordinate_record = coordinate_by_year[year]
        coordinate = pd.read_parquet(
            coordinate_record["path"],
            columns=["input_row_idx", *coordinate_names],
        )
        first_input = int(coordinate["input_row_idx"].iloc[0])
        local = prediction["input_row_idx"].to_numpy(np.int64) - first_input
        if bool((local < 0).any()) or bool((local >= len(coordinate)).any()):
            raise ValueError("dynamic_action_distribution_matched_alignment")
        values = coordinate.iloc[local][list(coordinate_names)].to_numpy(np.float64)
        stored_risk = prediction["risk_match_cell_code"].to_numpy(np.int64)
        recomputed_risk = baseline._matched_codes(
            values, matched_indices=risk_indices, bin_count=risk_bins
        ).astype(np.int64)
        if not np.array_equal(stored_risk, recomputed_risk):
            raise ValueError("dynamic_action_distribution_risk_cell")

        dates = prediction["date_idx"].to_numpy(np.int64)
        trade_dates = prediction["trade_date"].astype(str).to_numpy()
        target = prediction["actual_buy_advantage_vs_cash"].to_numpy(np.float64)
        positive = prediction["actual_positive"].astype(bool).to_numpy()
        unique_dates, date_first_positions, date_inverse = np.unique(
            dates, return_index=True, return_inverse=True
        )
        packed = date_inverse.astype(np.int64) * int(cell_size) + stored_risk
        unique_groups, group_inverse = np.unique(packed, return_inverse=True)
        group_dates = (unique_groups // int(cell_size)).astype(np.int64)
        group_count = len(unique_groups)
        positive_count = np.bincount(
            group_inverse, weights=positive.astype(np.float64), minlength=group_count
        )
        failure_count = np.bincount(
            group_inverse, weights=(~positive).astype(np.float64), minlength=group_count
        )
        positive_with_failure = positive & (failure_count[group_inverse] > 0.0)
        availability_rows.append(
            {
                "cost_scenario": cost,
                "oos_year": year,
                "hot_rows": len(prediction),
                "positive_rows": int(positive.sum()),
                "positive_rows_with_same_date_risk_matched_failure": int(
                    positive_with_failure.sum()
                ),
                "matched_failure_fraction": float(
                    positive_with_failure.sum() / max(int(positive.sum()), 1)
                ),
                "date_risk_cells": group_count,
                "mixed_date_risk_cells": int(
                    ((positive_count > 0.0) & (failure_count > 0.0)).sum()
                ),
            }
        )

        for coordinate_name, column in zip(
            discriminator_names, discriminator_indices, strict=True
        ):
            coordinate_value = values[:, column]
            finite = np.isfinite(coordinate_value)
            winner = finite & positive
            failure = finite & (~positive)
            winner_sum = np.bincount(
                group_inverse[winner],
                weights=coordinate_value[winner],
                minlength=group_count,
            )
            winner_count = np.bincount(
                group_inverse[winner], minlength=group_count
            )
            failure_sum = np.bincount(
                group_inverse[failure],
                weights=coordinate_value[failure],
                minlength=group_count,
            )
            finite_failure_count = np.bincount(
                group_inverse[failure], minlength=group_count
            )
            winner_failure = np.full(group_count, np.nan, dtype=np.float64)
            mixed = (winner_count > 0) & (finite_failure_count > 0)
            winner_failure[mixed] = (
                winner_sum[mixed] / winner_count[mixed]
                - failure_sum[mixed] / finite_failure_count[mixed]
            )

            high = finite & (coordinate_value >= 0.5)
            low = finite & (coordinate_value < 0.5)
            high_count = np.bincount(group_inverse[high], minlength=group_count)
            low_count = np.bincount(group_inverse[low], minlength=group_count)
            high_target = np.bincount(
                group_inverse[high], weights=target[high], minlength=group_count
            )
            low_target = np.bincount(
                group_inverse[low], weights=target[low], minlength=group_count
            )
            high_positive = np.bincount(
                group_inverse[high],
                weights=positive[high].astype(np.float64),
                minlength=group_count,
            )
            low_positive = np.bincount(
                group_inverse[low],
                weights=positive[low].astype(np.float64),
                minlength=group_count,
            )
            high_low_value = np.full(group_count, np.nan, dtype=np.float64)
            high_low_positive = np.full(group_count, np.nan, dtype=np.float64)
            split = (high_count > 0) & (low_count > 0)
            high_low_value[split] = (
                high_target[split] / high_count[split]
                - low_target[split] / low_count[split]
            )
            high_low_positive[split] = (
                high_positive[split] / high_count[split]
                - low_positive[split] / low_count[split]
            )

            daily_winner_failure = _collapse_group_values_to_dates(
                winner_failure, group_dates=group_dates, date_count=len(unique_dates)
            )
            daily_high_low_value = _collapse_group_values_to_dates(
                high_low_value, group_dates=group_dates, date_count=len(unique_dates)
            )
            daily_high_low_positive = _collapse_group_values_to_dates(
                high_low_positive,
                group_dates=group_dates,
                date_count=len(unique_dates),
            )
            for date_position, date in enumerate(unique_dates):
                daily_rows.append(
                    {
                        "cost_scenario": cost,
                        "oos_year": year,
                        "trade_date": str(trade_dates[date_first_positions[date_position]]),
                        "date_idx": int(date),
                        "coordinate": coordinate_name,
                        "winner_minus_failure_coordinate": daily_winner_failure[
                            date_position
                        ],
                        "high_minus_low_action_value": daily_high_low_value[
                            date_position
                        ],
                        "high_minus_low_positive_probability": daily_high_low_positive[
                            date_position
                        ],
                    }
                )
        del prediction, coordinate, values
        gc.collect()

    daily = pd.DataFrame(daily_rows).sort_values(
        ["cost_scenario", "coordinate", "trade_date"], kind="mergesort"
    )
    contrast_columns = [
        "winner_minus_failure_coordinate",
        "high_minus_low_action_value",
        "high_minus_low_positive_probability",
    ]
    annual = (
        daily.groupby(["cost_scenario", "oos_year", "coordinate"], sort=True)[
            contrast_columns
        ]
        .mean()
        .reset_index()
    )
    aggregate_rows: list[dict[str, Any]] = []
    for (cost, coordinate_name), group in daily.groupby(
        ["cost_scenario", "coordinate"], sort=True
    ):
        row: dict[str, Any] = {
            "cost_scenario": cost,
            "coordinate": coordinate_name,
            "dates": len(group),
        }
        for contrast in contrast_columns:
            values = group[contrast].dropna().to_numpy(np.float64)
            mean, standard_error, lower = baseline._hac_mean(values, lag=20)
            row[contrast] = mean
            row[f"{contrast}_hac_standard_error"] = standard_error
            row[f"{contrast}_hac_lower_95"] = lower
            row[f"{contrast}_p_value"] = _normal_two_sided_p(
                mean, standard_error
            )
            scoped_annual = annual[
                annual["cost_scenario"].eq(cost)
                & annual["coordinate"].eq(coordinate_name)
            ]
            row[f"{contrast}_positive_years"] = int(
                (scoped_annual[contrast] > 0.0).sum()
            )
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows).sort_values(
        ["cost_scenario", "coordinate"], kind="mergesort"
    )
    for cost in aggregate["cost_scenario"].unique():
        mask = aggregate["cost_scenario"].eq(cost)
        for contrast in contrast_columns:
            aggregate.loc[mask, f"{contrast}_q_value"] = _benjamini_hochberg(
                aggregate.loc[mask, f"{contrast}_p_value"].to_numpy(np.float64)
            )
    availability = pd.DataFrame(availability_rows).sort_values(
        ["cost_scenario", "oos_year"], kind="mergesort"
    )
    return (
        {
            "daily_matched_failure_diagnostics": _write_frame(
                output_root / "daily_matched_failure_diagnostics.parquet",
                daily,
                row_group_size=row_group_size,
            ),
            "annual_matched_failure_diagnostics": _write_frame(
                output_root / "annual_matched_failure_diagnostics.parquet",
                annual,
                row_group_size=row_group_size,
            ),
            "aggregate_matched_failure_diagnostics": _write_frame(
                output_root / "aggregate_matched_failure_diagnostics.parquet",
                aggregate,
                row_group_size=row_group_size,
            ),
            "matched_failure_availability": _write_frame(
                output_root / "matched_failure_availability.parquet",
                availability,
                row_group_size=row_group_size,
            ),
        },
        {
            "matched_discriminator_count": len(discriminator_names),
            "daily_matched_failure_rows": len(daily),
            "mean_matched_failure_fraction": float(
                availability["matched_failure_fraction"].mean()
            ),
            "bh_significant_positive_value_coordinates": sorted(
                set(
                    aggregate.loc[
                        aggregate["high_minus_low_action_value"].gt(0.0)
                        & aggregate["high_minus_low_action_value_q_value"].le(0.05),
                        "coordinate",
                    ].astype(str)
                )
            ),
        },
    )


@dataclass(frozen=True)
class FoldSource:
    cost_scenario: str
    oos_year: int
    training_cutoff_year: int
    maximum_label_as_of_year: int
    maximum_signal_year: int
    training_versions: list[dict[str, Any]]


def run_study(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    memory_start = int(psutil.virtual_memory().available / 1024**2)
    minimum_available = memory_start
    study_file = _resolve(study_path)
    study = load_study(study_file)
    root = _resolve(output_root or study["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    resources = dict(study["resources"])
    reserve_memory = int(resources["reserve_memory_mb"])
    row_group_size = int(resources["parquet_row_group_size"])
    maturity_sessions = int(dict(study["target"])["minimum_sessions_after_resolution"])

    source_contract, prefix_manifest, input_manifest, version_records, feature_index = (
        _source_contract(study)
    )
    coordinate_records, coordinate_audit = baseline.build_coordinate_partitions(
        study=study,
        input_manifest=input_manifest,
        feature_index=feature_index,
        output_root=root,
    )
    minimum_available = min(
        minimum_available, int(psutil.virtual_memory().available / 1024**2)
    )
    specs, coordinate_names, market_names, families, matched_indices = (
        baseline._coordinate_metadata(study)
    )
    name_to_index = {name: index for index, name in enumerate(coordinate_names)}
    market_indices = np.asarray(
        [name_to_index[name] for name in market_names], dtype=np.int16
    )
    stock_indices = np.asarray(
        [
            name_to_index[str(item["name"])]
            for item in specs
            if str(item["family"]) != "market"
        ],
        dtype=np.int16,
    )
    gate = dict(study["activity_gate"])
    gate_indices = np.asarray(
        [name_to_index[str(name)] for name in gate["coordinates"]], dtype=np.int16
    )
    thresholds = {str(key): float(value) for key, value in gate["thresholds"].items()}
    primary_scope = str(gate["primary_training_scope"])
    primary_threshold = thresholds[primary_scope]
    risk_indices = np.asarray(
        [
            name_to_index[str(name)]
            for name in dict(study["coordinates"])["risk_match_coordinates"]
        ],
        dtype=np.int16,
    )
    risk_bins = int(dict(study["coordinates"])["risk_match_bins"])
    training_scopes = [
        str(value) for value in dict(study["estimators"])["training_scopes"]
    ]
    coordinates = baseline.CoordinateStore(
        records=coordinate_records,
        coordinate_names=coordinate_names,
        cache_years=int(resources["coordinate_cache_years"]),
    )
    activity_store = ActivityRankStore(
        coordinates=coordinates,
        coordinate_indices=gate_indices,
        minimum_finite=int(gate["minimum_finite_coordinates"]),
        cache_years=int(resources["coordinate_cache_years"]),
    )
    univariate_bins = int(dict(study["coordinates"])["univariate_bins"])
    matched_bins = int(dict(study["coordinates"])["matched_bins"])
    oos_years = [int(value) for value in dict(study["period"])["strict_oos_years"]]
    cost_scenarios = [str(value) for value in dict(study["evaluation"])["cost_scenarios"]]

    prediction_records: list[dict[str, Any]] = []
    state_records: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    gate_daily_rows: list[dict[str, Any]] = []
    for cost_scenario in cost_scenarios:
        accumulators = {
            scope: DistributionAccumulator(
                market_indices=market_indices,
                stock_indices=stock_indices,
                matched_indices=matched_indices,
                univariate_bins=univariate_bins,
                matched_bins=matched_bins,
            )
            for scope in training_scopes
        }
        active_records: dict[int, dict[str, Any]] = {}
        for oos_year in oos_years:
            available = int(psutil.virtual_memory().available / 1024**2)
            if available <= reserve_memory:
                raise MemoryError(
                    f"dynamic_action_distribution_memory_reserve:{available}"
                )
            fold_started = time.perf_counter()
            training_cutoff = oos_year - 1
            desired_records = baseline.select_latest_records(
                version_records=version_records,
                cost_scenario=cost_scenario,
                cutoff_year=training_cutoff,
                maximum_signal_year=training_cutoff,
            )
            for signal_year, previous in list(active_records.items()):
                desired = desired_records.get(signal_year)
                if desired is not None and int(desired["as_of_year"]) == int(
                    previous["as_of_year"]
                ):
                    continue
                coordinate_year = coordinates.load(signal_year)
                experience = baseline._read_aligned_experience(
                    record=previous,
                    coordinates=coordinate_year,
                    maturity_sessions=maturity_sessions,
                )
                activity = activity_store.load(signal_year)[
                    experience.coordinate_positions
                ]
                scope_masks = {
                    "all": np.ones(len(activity), dtype=bool),
                    primary_scope: _gate_mask(activity, primary_threshold),
                }
                for scope in training_scopes:
                    audit = accumulators[scope].apply(
                        experience, sign=-1, mask=scope_masks[scope]
                    )
                    update_rows.append(
                        {
                            "cost_scenario": cost_scenario,
                            "oos_year": oos_year,
                            "scope": scope,
                            "operation": "remove",
                            "signal_year": signal_year,
                            "as_of_year": int(previous["as_of_year"]),
                            **audit,
                        }
                    )
                del active_records[signal_year]
            for signal_year, desired in desired_records.items():
                previous = active_records.get(signal_year)
                if previous is not None and int(previous["as_of_year"]) == int(
                    desired["as_of_year"]
                ):
                    continue
                coordinate_year = coordinates.load(signal_year)
                experience = baseline._read_aligned_experience(
                    record=desired,
                    coordinates=coordinate_year,
                    maturity_sessions=maturity_sessions,
                )
                activity = activity_store.load(signal_year)[
                    experience.coordinate_positions
                ]
                scope_masks = {
                    "all": np.ones(len(activity), dtype=bool),
                    primary_scope: _gate_mask(activity, primary_threshold),
                }
                for scope in training_scopes:
                    audit = accumulators[scope].apply(
                        experience, sign=1, mask=scope_masks[scope]
                    )
                    update_rows.append(
                        {
                            "cost_scenario": cost_scenario,
                            "oos_year": oos_year,
                            "scope": scope,
                            "operation": "add",
                            "signal_year": signal_year,
                            "as_of_year": int(desired["as_of_year"]),
                            **audit,
                        }
                    )
                active_records[signal_year] = dict(desired)

            maximum_label_as_of = max(
                int(record["as_of_year"]) for record in active_records.values()
            )
            maximum_signal_year = max(active_records)
            if maximum_label_as_of > training_cutoff or maximum_signal_year >= oos_year:
                raise ValueError("dynamic_action_distribution_training_cutoff")

            snapshots: dict[str, dict[str, np.ndarray]] = {}
            fold_states: list[dict[str, Any]] = []
            for scope in training_scopes:
                snapshot = build_snapshot(
                    accumulator=accumulators[scope],
                    study=study,
                    coordinate_names=coordinate_names,
                    market_indices=market_indices,
                    stock_indices=stock_indices,
                    families=families,
                    matched_indices=matched_indices,
                    scope=scope,
                    training_cutoff_year=training_cutoff,
                    maximum_label_as_of_year=maximum_label_as_of,
                    maximum_signal_year=maximum_signal_year,
                )
                state_record = _snapshot_record(
                    root=root,
                    cost_scenario=cost_scenario,
                    oos_year=oos_year,
                    scope=scope,
                    snapshot=snapshot,
                )
                state_records.append(state_record)
                fold_states.append(state_record)
                snapshots[scope] = snapshot

            evaluation_record = baseline.select_evaluation_record(
                version_records=version_records,
                cost_scenario=cost_scenario,
                signal_year=oos_year,
            )
            coordinate_year = coordinates.load(oos_year)
            evaluation = baseline._read_aligned_experience(
                record=evaluation_record,
                coordinates=coordinate_year,
                maturity_sessions=maturity_sessions,
            )
            activity = activity_store.load(oos_year)[evaluation.coordinate_positions]
            gate_daily_rows.extend(
                _gate_daily_rows(
                    experience=evaluation,
                    activity_rank=activity,
                    thresholds=thresholds,
                )
            )
            hot_mask = _gate_mask(activity, primary_threshold)
            if not hot_mask.any():
                raise ValueError("dynamic_action_distribution_empty_hot_evaluation")
            hot_values = evaluation.coordinates[hot_mask]
            risk_cell_code = baseline._matched_codes(
                hot_values, matched_indices=risk_indices, bin_count=risk_bins
            )
            predictions_by_scope = {
                scope: predict_from_snapshot(hot_values, snapshots[scope])
                for scope in training_scopes
            }
            prediction = _prediction_frame(
                experience=evaluation,
                coordinate_year=coordinate_year,
                hot_mask=hot_mask,
                activity_rank=activity,
                risk_cell_code=risk_cell_code,
                predictions_by_scope=predictions_by_scope,
                training_cutoff_year=training_cutoff,
                maximum_label_as_of_year=maximum_label_as_of,
                maximum_signal_year=maximum_signal_year,
            )
            prediction_path = (
                root
                / "predictions"
                / f"cost={cost_scenario}"
                / f"oos_year={oos_year}"
                / "part-0000.parquet"
            )
            prediction_record = _write_frame(
                prediction_path, prediction, row_group_size=row_group_size
            )
            prediction_record.update(
                {
                    "cost_scenario": cost_scenario,
                    "oos_year": oos_year,
                    "training_cutoff_year": training_cutoff,
                    "maximum_training_label_as_of_year": maximum_label_as_of,
                    "maximum_training_signal_year": maximum_signal_year,
                    "evaluation_as_of_year": int(evaluation_record["as_of_year"]),
                }
            )
            prediction_records.append(prediction_record)
            fold_elapsed = float(time.perf_counter() - fold_started)
            fold_records.append(
                {
                    "cost_scenario": cost_scenario,
                    "oos_year": oos_year,
                    "training_cutoff_year": training_cutoff,
                    "maximum_training_label_as_of_year": maximum_label_as_of,
                    "maximum_training_signal_year": maximum_signal_year,
                    "training_version_count": len(active_records),
                    "training_versions": [
                        {
                            "signal_year": signal_year,
                            "as_of_year": int(record["as_of_year"]),
                            "version_role": str(record["version_role"]),
                            "path": str(record["path"]),
                            "sha256": str(record["sha256"]),
                        }
                        for signal_year, record in sorted(active_records.items())
                    ],
                    "training_rows_all": int(snapshots["all"]["training_rows"][0]),
                    "training_rows_top20": int(
                        snapshots[primary_scope]["training_rows"][0]
                    ),
                    "training_dates_all": int(
                        snapshots["all"]["training_dates"][0]
                    ),
                    "training_dates_top20": int(
                        snapshots[primary_scope]["training_dates"][0]
                    ),
                    "evaluation_as_of_year": int(evaluation_record["as_of_year"]),
                    "evaluation_rows": len(evaluation.target),
                    "hot_evaluation_rows": len(prediction),
                    "elapsed_seconds": fold_elapsed,
                    "states": fold_states,
                    "prediction": prediction_record,
                }
            )
            print(
                json.dumps(
                    {
                        "cost_scenario": cost_scenario,
                        "oos_year": oos_year,
                        "training_rows_all": int(snapshots["all"]["training_rows"][0]),
                        "training_rows_top20": int(
                            snapshots[primary_scope]["training_rows"][0]
                        ),
                        "hot_prediction_rows": len(prediction),
                        "elapsed_seconds": fold_elapsed,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            minimum_available = min(
                minimum_available,
                int(psutil.virtual_memory().available / 1024**2),
            )
            del evaluation, prediction, predictions_by_scope, snapshots, hot_values
            gc.collect()
        activity_store.clear()
        coordinates.clear()

    update_frame = pd.DataFrame(update_rows).sort_values(
        ["cost_scenario", "oos_year", "scope", "operation", "signal_year"],
        kind="mergesort",
    )
    fold_summary = pd.DataFrame(
        [
            {
                key: value
                for key, value in record.items()
                if key not in {"training_versions", "states", "prediction"}
            }
            for record in fold_records
        ]
    ).sort_values(["cost_scenario", "oos_year"], kind="mergesort")
    outputs: dict[str, Any] = {
        "training_version_updates": _write_frame(
            root / "training_version_updates.parquet",
            update_frame,
            row_group_size=row_group_size,
        ),
        "fold_summary": _write_frame(
            root / "fold_summary.parquet",
            fold_summary,
            row_group_size=row_group_size,
        ),
    }
    gate_daily = pd.DataFrame(gate_daily_rows).sort_values(
        ["cost_scenario", "gate_scope", "trade_date"], kind="mergesort"
    )
    gate_outputs, gate_audit = build_gate_diagnostics(
        daily=gate_daily, output_root=root, row_group_size=row_group_size
    )
    outputs.update(gate_outputs)
    evaluation_outputs, evaluation_audit, promotion_gate = evaluate_predictions(
        study=study,
        prediction_records=prediction_records,
        output_root=root,
        row_group_size=row_group_size,
    )
    outputs.update(evaluation_outputs)
    matched_outputs, matched_audit = build_matched_failure_diagnostics(
        study=study,
        prediction_records=prediction_records,
        coordinate_records=coordinate_records,
        coordinate_names=coordinate_names,
        output_root=root,
        row_group_size=row_group_size,
    )
    outputs.update(matched_outputs)

    memory_end = int(psutil.virtual_memory().available / 1024**2)
    manifest = {
        "schema": f"seq100_dynamic_action_distribution/{SCHEMA_VERSION}",
        "status": "completed",
        "study_id": STUDY_ID,
        "study": {
            "path": str(study_file.resolve()),
            "sha256": replay.sha256(study_file),
        },
        "source_contract": source_contract,
        "source_audit": {
            "prefix_experience_version_rows": int(
                dict(prefix_manifest["audit"])["experience_version_rows"]
            ),
            "model_input_rows": int(input_manifest["row_count"]),
            "forbidden_2026_rows": 0,
        },
        "coordinates": coordinate_records,
        "coordinate_audit": coordinate_audit,
        "model_states": state_records,
        "predictions": prediction_records,
        "folds": fold_records,
        "outputs": outputs,
        "audit": {
            "cost_scenarios": cost_scenarios,
            "oos_years": oos_years,
            "fold_count": len(fold_records),
            "state_count": len(state_records),
            "prediction_partitions": len(prediction_records),
            "hot_prediction_rows": int(
                sum(int(record["rows"]) for record in prediction_records)
            ),
            "minimum_sessions_after_resolution": maturity_sessions,
            "training_cutoff_violations": 0,
            "forbidden_2026_rows": 0,
            "final_oracle_row_labels_used": False,
            "fixed_holding_horizon_used": False,
            "binary_good_stock_label_used": False,
            "binary_distribution_component_used": True,
            "activity_gate_is_buy_rule": False,
            "pristine_confirmation_claimed": False,
            **gate_audit,
            **evaluation_audit,
            **matched_audit,
        },
        "runtime": {
            "elapsed_seconds": float(time.perf_counter() - started),
            "available_memory_mb_at_start": memory_start,
            "available_memory_mb_at_end": memory_end,
            "minimum_available_memory_mb_observed": minimum_available,
            "reserve_memory_mb": reserve_memory,
            "logical_cpu_count": int(psutil.cpu_count() or 1),
            "adaptive_memory": bool(resources["adaptive_memory"]),
        },
        "promotion_gate": promotion_gate,
        "training_performed": True,
        "prediction_performed": True,
        "causal_distribution_evaluation_performed": True,
        "portfolio_execution_performed": False,
        "profit_claim_allowed": False,
        "production_policy_selected": False,
        "deep_model_selected": False,
        "report_generation_performed": False,
    }
    _write_json(root / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run causal dynamic action distribution decomposition."
    )
    parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)
    result = run_study(study_path=args.study, output_root=args.output_root)
    print(
        json.dumps(
            {
                "status": result["status"],
                "fold_count": result["audit"]["fold_count"],
                "hot_prediction_rows": result["audit"]["hot_prediction_rows"],
                "passed_models": result["audit"]["passed_models"],
                "elapsed_seconds": result["runtime"]["elapsed_seconds"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
