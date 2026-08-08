"""Hierarchical D20 residual policy research for Seq100.

This study asks a deliberately narrower question than the preceding stock
distribution work: can the supported cross-sectional residual ranking become
positive absolute value after separately forecasting the market and PIT
industry components, vetoing the worst predicted loss tail, and allowing cash?

The development years reuse frozen stock forecasts.  The reserved 2023-2025
years cannot be trained or read by this module unless the predeclared
development gate passes.  This remains an overlapping-cohort forecast study;
finite-capital account replay is a later, separately gated step.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from daily_research.path_policy import seq100_stock_bad_tail as bad_tail
from daily_research.path_policy import seq100_stock_distribution as base

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT
    / "daily_research/studies/seq100_hierarchical_residual_policy_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/"
    / "seq100_hierarchical_residual_policy_v1"
)

STUDY_ID = "seq100_hierarchical_residual_policy_v1"
SUMMARY_SCHEMA = "seq100_hierarchical_residual_policy_summary/1"
PREFLIGHT_SCHEMA = "seq100_hierarchical_residual_policy_preflight/1"
HORIZON = 20
DEVELOPMENT_YEARS = (2017, 2018, 2019, 2020, 2021, 2022)
CONFIRMATION_YEARS = (2023, 2024, 2025)
PRIMARY_POLICY = "cash_gate_residual_veto10_top48"
POLICIES = (
    "residual_top48",
    "residual_veto10_top48",
    "hierarchical_veto10_top48",
    PRIMARY_POLICY,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    base._write_json(path, value)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    partial.unlink(missing_ok=True)
    frame.to_parquet(partial, index=False)
    os.replace(partial, path)


def _sha256_payload(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def load_study(
    study_path: str | Path = DEFAULT_STUDY_PATH,
) -> tuple[dict[str, Any], Path]:
    path = base._resolve_path(study_path)
    study = _read_json(path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("study_id_mismatch")
    source = dict(study.get("source", {}) or {})
    if str(source.get("expected_input_fingerprint")) != base.EXPECTED_INPUT_FINGERPRINT:
        raise ValueError("input_fingerprint_contract_mismatch")
    if (
        str(source.get("expected_label_fingerprint"))
        != bad_tail.EXPECTED_LABEL_FINGERPRINT
    ):
        raise ValueError("label_fingerprint_contract_mismatch")
    if int(source.get("expected_row_count", -1)) != base.EXPECTED_ROW_COUNT:
        raise ValueError("row_count_contract_mismatch")
    target = dict(study.get("target", {}) or {})
    if int(target.get("horizon", -1)) != HORIZON:
        raise ValueError("horizon_contract_mismatch")
    if (
        tuple(int(v) for v in study["evaluation"]["development_years"])
        != DEVELOPMENT_YEARS
    ):
        raise ValueError("development_year_contract_mismatch")
    if (
        tuple(int(v) for v in study["evaluation"]["retrospective_confirmation_years"])
        != CONFIRMATION_YEARS
    ):
        raise ValueError("confirmation_year_contract_mismatch")
    if str(study["policy"]["primary"]) != PRIMARY_POLICY:
        raise ValueError("primary_policy_contract_mismatch")
    if bool(study["decision_boundary"].get("portfolio_selected", True)):
        raise ValueError("study_must_start_without_portfolio_selection")
    if bool(study["decision_boundary"].get("account_optimization_performed", True)):
        raise ValueError("study_must_start_without_account_optimization")
    return study, path


def _validate_source_contract(study: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(study["source"])
    input_manifest_path = base._resolve_path(source["input_manifest"])
    label_manifest_path = base._resolve_path(source["label_manifest"])
    development_path = base._resolve_path(source["development_run_summary"])
    if base._sha256_file(development_path) != source["development_run_summary_sha256"]:
        raise ValueError("frozen_development_summary_hash_mismatch")
    input_manifest = _read_json(input_manifest_path)
    label_manifest = _read_json(label_manifest_path)
    development = _read_json(development_path)
    if input_manifest.get("input_fingerprint") != source["expected_input_fingerprint"]:
        raise ValueError("input_manifest_fingerprint_mismatch")
    if (
        label_manifest.get("experiment_fingerprint")
        != source["expected_label_fingerprint"]
    ):
        raise ValueError("label_manifest_fingerprint_mismatch")
    if development.get("input_fingerprint") != source["expected_input_fingerprint"]:
        raise ValueError("development_input_fingerprint_mismatch")
    if development.get("label_fingerprint") != source["expected_label_fingerprint"]:
        raise ValueError("development_label_fingerprint_mismatch")
    if (
        tuple(int(v) for v in development.get("evaluation_years", ()))
        != DEVELOPMENT_YEARS
    ):
        raise ValueError("development_source_year_mismatch")
    if int(development.get("primary_horizon", -1)) != HORIZON:
        raise ValueError("development_source_horizon_mismatch")
    return {
        "input_manifest_path": input_manifest_path,
        "label_manifest_path": label_manifest_path,
        "development_summary_path": development_path,
        "development_summary": development,
    }


def _feature_names_for_family(
    input_manifest: Mapping[str, Any], family: str
) -> tuple[str, ...]:
    names = tuple(
        str(record["feature_name"])
        for record in input_manifest["features"]
        if str(record.get("analytic_family")) == str(family)
    )
    if not names or len(names) != len(set(names)):
        raise ValueError(f"feature_family_invalid:{family}")
    return names


@dataclass(frozen=True)
class GroupedFactorPanel:
    date_idx: np.ndarray
    group_code: np.ndarray
    features: np.ndarray
    target: np.ndarray
    observation_count: np.ndarray
    feature_names: tuple[str, ...]
    lookup: np.ndarray
    lookup_width: int


@dataclass
class FittedFactorModel:
    name: str
    feature_names: tuple[str, ...]
    center: np.ndarray
    spread: np.ndarray
    model: Ridge
    training_mean: float
    training_group_count: int
    training_date_count: int
    maximum_train_date_idx: int


@dataclass(frozen=True)
class FactorPanels:
    market: GroupedFactorPanel
    industry: GroupedFactorPanel


def _mean_by_key(
    values: np.ndarray, keys: np.ndarray, *, size: int
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(values)
    finite = np.isfinite(raw)
    count = np.bincount(keys[finite], minlength=int(size)).astype(np.int32)
    total = np.bincount(
        keys[finite], weights=raw[finite].astype(np.float64), minlength=int(size)
    )
    mean = np.divide(
        total,
        count,
        out=np.full(int(size), np.nan, dtype=np.float64),
        where=count > 0,
    )
    return mean, count


def _build_factor_panels(panel: base.StockPanel) -> FactorPanels:
    """Build bounded date and date-industry panels once for all folds."""

    date_idx = np.asarray(panel.date_idx, dtype=np.int64)
    date_count = int(date_idx.max()) + 1
    market_names = _feature_names_for_family(panel.input_manifest, "market_state")
    industry_names = _feature_names_for_family(panel.input_manifest, "industry_context")
    market_positions = base._feature_positions(panel, market_names)
    industry_positions = base._feature_positions(panel, industry_names)

    market_target, market_count = _mean_by_key(
        panel.target(f"market_component_{HORIZON}"), date_idx, size=date_count
    )
    first_rows = np.r_[0, np.flatnonzero(date_idx[1:] != date_idx[:-1]) + 1]
    first_dates = date_idx[first_rows]
    if bool((first_dates[1:] <= first_dates[:-1]).any()):
        raise ValueError("date_index_not_strictly_increasing")
    market_features = base._feature_matrix(panel, first_rows, market_positions)
    market_lookup = np.full(date_count, -1, dtype=np.int64)
    market_lookup[first_dates] = np.arange(len(first_dates), dtype=np.int64)

    industry_code = np.asarray(panel.industry_code, dtype=np.int64)
    if bool((industry_code < 0).any()):
        raise ValueError("negative_pit_industry_code")
    industry_width = int(industry_code.max()) + 1
    group_space = date_count * industry_width
    group_key = date_idx * industry_width + industry_code
    present_count = np.bincount(group_key, minlength=group_space).astype(np.int32)
    present_keys = np.flatnonzero(present_count > 0).astype(np.int64, copy=False)
    industry_target_full, industry_target_count = _mean_by_key(
        panel.target(f"industry_component_{HORIZON}"),
        group_key,
        size=group_space,
    )
    industry_features = np.full(
        (len(present_keys), len(industry_names)), np.nan, dtype=np.float32
    )
    for column, position in enumerate(industry_positions):
        values = np.asarray(panel.compact[:, int(position)], dtype=np.float32)
        mean, _ = _mean_by_key(values, group_key, size=group_space)
        industry_features[:, column] = mean[present_keys].astype(np.float32)
    industry_lookup = np.full(group_space, -1, dtype=np.int64)
    industry_lookup[present_keys] = np.arange(len(present_keys), dtype=np.int64)

    return FactorPanels(
        market=GroupedFactorPanel(
            date_idx=first_dates.astype(np.int32),
            group_code=np.full(len(first_dates), -1, dtype=np.int32),
            features=market_features,
            target=market_target[first_dates],
            observation_count=market_count[first_dates],
            feature_names=market_names,
            lookup=market_lookup,
            lookup_width=1,
        ),
        industry=GroupedFactorPanel(
            date_idx=(present_keys // industry_width).astype(np.int32),
            group_code=(present_keys % industry_width).astype(np.int32),
            features=industry_features,
            target=industry_target_full[present_keys],
            observation_count=industry_target_count[present_keys],
            feature_names=industry_names,
            lookup=industry_lookup,
            lookup_width=industry_width,
        ),
    )


def _fit_factor_model(
    grouped: GroupedFactorPanel,
    *,
    maximum_train_date_idx: int,
    penalty: float,
    name: str,
) -> FittedFactorModel:
    selected = (
        (grouped.date_idx <= int(maximum_train_date_idx))
        & np.isfinite(grouped.target)
        & (grouped.observation_count > 0)
    )
    if int(selected.sum()) < 100:
        raise ValueError(f"too_few_factor_training_groups:{name}")
    x = grouped.features[selected]
    y = grouped.target[selected].astype(np.float64)
    weights = base._date_equal_weights(grouped.date_idx[selected])
    center, spread = base._fit_standardizer(x, weights)
    z = base._apply_standardizer(x, center, spread)
    alpha = float(penalty) * float(np.sum(weights))
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(z, y, sample_weight=weights)
    return FittedFactorModel(
        name=str(name),
        feature_names=grouped.feature_names,
        center=center,
        spread=spread,
        model=model,
        training_mean=float(np.average(y, weights=weights)),
        training_group_count=int(selected.sum()),
        training_date_count=int(np.unique(grouped.date_idx[selected]).size),
        maximum_train_date_idx=int(maximum_train_date_idx),
    )


def _predict_factor_groups(
    model: FittedFactorModel, grouped: GroupedFactorPanel
) -> np.ndarray:
    z = base._apply_standardizer(grouped.features, model.center, model.spread)
    return np.asarray(model.model.predict(z), dtype=np.float64)


def _factor_model_record(model: FittedFactorModel) -> dict[str, Any]:
    return {
        "name": model.name,
        "feature_names": list(model.feature_names),
        "center": model.center.tolist(),
        "spread": model.spread.tolist(),
        "coefficient": np.asarray(model.model.coef_, dtype=np.float64).tolist(),
        "intercept": float(model.model.intercept_),
        "training_mean": model.training_mean,
        "training_group_count": model.training_group_count,
        "training_date_count": model.training_date_count,
        "maximum_train_date_idx": model.maximum_train_date_idx,
    }


def _factor_diagnostic(
    grouped: GroupedFactorPanel,
    predicted: np.ndarray,
    *,
    model: FittedFactorModel,
    evaluation_year: int,
) -> dict[str, Any]:
    year = grouped.date_idx
    # The dense panel date index maps back to dates through the stock panel; the
    # caller passes a year-specific mask by encoding its first and last date.
    selected = np.isfinite(grouped.target) & np.isfinite(predicted)
    selected &= year >= 0
    y = grouped.target[selected]
    p = np.asarray(predicted, dtype=np.float64)[selected]
    dates = grouped.date_idx[selected]
    weights = base._date_equal_weights(dates)
    if not len(y):
        raise ValueError(f"empty_factor_diagnostic:{model.name}:{evaluation_year}")
    centered_y = y - float(np.average(y, weights=weights))
    centered_p = p - float(np.average(p, weights=weights))
    denominator = math.sqrt(
        float(np.average(np.square(centered_y), weights=weights))
        * float(np.average(np.square(centered_p), weights=weights))
    )
    correlation = (
        float(np.average(centered_y * centered_p, weights=weights) / denominator)
        if denominator > 0
        else float("nan")
    )
    return {
        "evaluation_year": int(evaluation_year),
        "model": model.name,
        "group_count": len(y),
        "date_count": int(np.unique(dates).size),
        "mse": float(np.average(np.square(y - p), weights=weights)),
        "constant_mse": float(
            np.average(np.square(y - model.training_mean), weights=weights)
        ),
        "correlation": correlation,
        "direction_accuracy": float(np.average((y > 0) == (p > 0), weights=weights)),
    }


def _source_forecast_paths(source_summary: Mapping[str, Any]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in source_summary["fold_summaries"]:
        summary_path = Path(str(value))
        summary = _read_json(summary_path)
        if int(summary["horizon"]) != HORIZON:
            continue
        year = int(summary["evaluation_year"])
        record = dict(summary["files"]["candidate_forecasts"])
        if not base._record_valid(record, verify_hash=True):
            raise ValueError(f"frozen_candidate_forecast_invalid:{year}")
        result[year] = Path(str(record["path"]))
    if tuple(sorted(result)) != DEVELOPMENT_YEARS:
        raise ValueError("frozen_candidate_years_incomplete")
    return result


def _load_frozen_stock_forecasts(
    panel: base.StockPanel, year: int, path: Path
) -> tuple[np.ndarray, np.ndarray]:
    columns = [
        "candidate_id",
        "date_idx",
        "symbol_idx",
        "trade_date",
        "ranking_score__lightgbm_mean_residual",
        "loss_probability__lightgbm_binary",
    ]
    frozen = pd.read_parquet(path, columns=columns)
    rows = panel.rows_for_year(year)
    identity = panel.row_index.iloc[rows]
    for column in ("candidate_id", "date_idx", "symbol_idx"):
        if not np.array_equal(frozen[column].to_numpy(), identity[column].to_numpy()):
            raise ValueError(f"frozen_candidate_identity_mismatch:{year}:{column}")
    if not np.array_equal(
        frozen["trade_date"].astype(str).to_numpy(),
        identity["trade_date"].astype(str).to_numpy(),
    ):
        raise ValueError(f"frozen_candidate_identity_mismatch:{year}:trade_date")
    residual = frozen["ranking_score__lightgbm_mean_residual"].to_numpy(
        dtype=np.float64
    )
    loss = frozen["loss_probability__lightgbm_binary"].to_numpy(dtype=np.float64)
    if not np.isfinite(residual).all() or not np.isfinite(loss).all():
        raise ValueError(f"frozen_forecast_nonfinite:{year}")
    return residual, loss


def _train_lightgbm_mean(
    panel: base.StockPanel,
    *,
    train_rows: np.ndarray,
    evaluation_rows: np.ndarray,
    target_values: np.ndarray,
    feature_names: Sequence[str],
    config: Mapping[str, Any],
    model_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    import lightgbm as lgb

    if str(lgb.__version__) != str(config["runtime_version"]):
        raise ValueError("lightgbm_runtime_version_mismatch")
    positions = base._feature_positions(panel, feature_names)
    x = base._feature_matrix(panel, train_rows, positions)
    y = np.asarray(target_values[train_rows], dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError("lightgbm_mean_target_nonfinite")
    weights = base._date_equal_weights(panel.date_idx[train_rows])
    dataset = lgb.Dataset(
        x,
        label=y,
        weight=weights,
        feature_name=list(feature_names),
        free_raw_data=False,
    )
    rounds = int(config["num_boost_round"])
    booster = lgb.train(
        bad_tail._lgb_parameters(config, objective="regression_l2"),
        dataset,
        num_boost_round=rounds,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_path))
    predicted = bad_tail._predict_boosters(
        panel,
        evaluation_rows,
        positions,
        {"mean": booster},
        iterations=rounds,
    )["mean"]
    importance = booster.feature_importance(importance_type="gain")
    top = np.argsort(importance)[::-1][:20]
    summary = {
        "training_row_count": len(train_rows),
        "training_date_count": int(np.unique(panel.date_idx[train_rows]).size),
        "feature_count": len(feature_names),
        "rounds": rounds,
        "top_gain_features": [
            {
                "feature": str(feature_names[int(position)]),
                "gain": float(importance[int(position)]),
            }
            for position in top
        ],
    }
    del x, dataset, booster
    gc.collect()
    return predicted, summary


def _train_stock_heads(
    panel: base.StockPanel,
    *,
    study: Mapping[str, Any],
    evaluation_year: int,
    model_root: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rows = panel.rows_for_year(evaluation_year)
    maximum_per_date = int(study["evaluation"]["maximum_train_rows_per_date"])
    config = dict(study["models"]["frozen_stock_residual"])
    feature_names = base.feature_names_for_block(
        panel.input_manifest, str(config["feature_block"])
    )
    residual_target = f"residual_log_return_{HORIZON}"
    gross_target = f"executable_log_return_{HORIZON}"
    residual_fold = base._build_fold(
        panel,
        evaluation_year=evaluation_year,
        horizon=HORIZON,
        target=residual_target,
    )
    residual_train = bad_tail._sample_fold_rows(
        panel,
        residual_fold["train_rows"],
        maximum_per_date=maximum_per_date,
    )
    residual_prediction, residual_summary = _train_lightgbm_mean(
        panel,
        train_rows=residual_train,
        evaluation_rows=rows,
        target_values=panel.target(residual_target),
        feature_names=feature_names,
        config=config,
        model_path=model_root / "residual_mean.txt",
    )
    gross_fold = base._build_fold(
        panel,
        evaluation_year=evaluation_year,
        horizon=HORIZON,
        target=gross_target,
    )
    gross_train = bad_tail._sample_fold_rows(
        panel,
        gross_fold["train_rows"],
        maximum_per_date=maximum_per_date,
    )
    gross_values = panel.target(gross_target)
    loss_prediction, loss_summary = bad_tail._fit_lgb_binary(
        panel,
        train_rows=gross_train,
        evaluation_rows=rows,
        target_values=(gross_values <= 0.0).astype(np.int8),
        feature_names=feature_names,
        config=config,
        model_path=model_root / "bad_tail_binary.txt",
    )
    return (
        residual_prediction,
        loss_prediction,
        {
            "evaluation_year": int(evaluation_year),
            "residual": residual_summary,
            "bad_tail": loss_summary,
            "residual_fold": {
                key: value
                for key, value in residual_fold.items()
                if key not in {"train_rows", "evaluation_rows"}
            },
            "gross_fold": {
                key: value
                for key, value in gross_fold.items()
                if key not in {"train_rows", "evaluation_rows"}
            },
        },
    )


def _rank_order(score: np.ndarray, identity: np.ndarray) -> np.ndarray:
    values = np.asarray(score, dtype=np.float64)
    ids = np.asarray(identity, dtype=np.int64)
    if len(values) != len(ids) or not np.isfinite(values).all():
        raise ValueError("ranking_input_invalid")
    return np.lexsort((ids, -values)).astype(np.int64, copy=False)


def _select_day(
    *,
    residual_score: np.ndarray,
    bad_tail_score: np.ndarray,
    hierarchical_score: np.ndarray,
    identity: np.ndarray,
    top_k: int,
    veto_fraction: float,
    stress_cost: float,
) -> dict[str, np.ndarray]:
    """Select from forecasts and identities only; no outcome is an input."""

    count = len(identity)
    if not (
        len(residual_score) == len(bad_tail_score) == len(hierarchical_score) == count
    ):
        raise ValueError("selection_length_mismatch")
    veto_count = min(math.ceil(count * float(veto_fraction)), count)
    bad_order = _rank_order(bad_tail_score, identity)
    allowed = np.ones(count, dtype=bool)
    allowed[bad_order[:veto_count]] = False
    allowed_positions = np.flatnonzero(allowed)
    residual_all = _rank_order(residual_score, identity)[: min(int(top_k), count)]
    residual_allowed_order = _rank_order(
        np.asarray(residual_score)[allowed_positions],
        np.asarray(identity)[allowed_positions],
    )
    residual_veto = allowed_positions[
        residual_allowed_order[: min(int(top_k), len(allowed_positions))]
    ]
    hierarchical_allowed_order = _rank_order(
        np.asarray(hierarchical_score)[allowed_positions],
        np.asarray(identity)[allowed_positions],
    )
    hierarchical_ranked = allowed_positions[hierarchical_allowed_order]
    hierarchical = hierarchical_ranked[
        np.asarray(hierarchical_score)[hierarchical_ranked] > float(stress_cost)
    ][: min(int(top_k), len(hierarchical_ranked))]
    primary = np.empty(0, dtype=np.int64)
    if len(residual_veto) == int(top_k):
        basket_prediction = float(
            np.mean(np.asarray(hierarchical_score)[residual_veto])
        )
        if basket_prediction > float(stress_cost):
            primary = residual_veto
    return {
        "residual_top48": residual_all,
        "residual_veto10_top48": residual_veto,
        "hierarchical_veto10_top48": hierarchical,
        PRIMARY_POLICY: primary,
    }


def _evaluate_year(
    panel: base.StockPanel,
    *,
    evaluation_year: int,
    residual_score: np.ndarray,
    bad_tail_score: np.ndarray,
    market_prediction: np.ndarray,
    industry_prediction: np.ndarray,
    study: Mapping[str, Any],
) -> pd.DataFrame:
    rows = panel.rows_for_year(evaluation_year)
    dates = panel.date_idx[rows]
    identities = panel.row_index.iloc[rows]["candidate_id"].to_numpy(dtype=np.int64)
    gross_full = panel.target(f"executable_log_return_{HORIZON}")
    residual_full = panel.target(f"residual_log_return_{HORIZON}")
    gross_valid = panel.target_valid(f"executable_log_return_{HORIZON}")[rows]
    residual_valid = panel.target_valid(f"residual_log_return_{HORIZON}")[rows]
    gross = gross_full[rows]
    residual = residual_full[rows]
    hierarchical = (
        np.asarray(market_prediction, dtype=np.float64)
        + np.asarray(industry_prediction, dtype=np.float64)
        + np.asarray(residual_score, dtype=np.float64)
    )
    for name, value in {
        "residual": residual_score,
        "bad_tail": bad_tail_score,
        "market": market_prediction,
        "industry": industry_prediction,
        "hierarchical": hierarchical,
    }.items():
        if len(value) != len(rows) or not np.isfinite(value).all():
            raise ValueError(f"evaluation_forecast_invalid:{evaluation_year}:{name}")

    evaluation = dict(study["evaluation"])
    policy = dict(study["policy"])
    top_k = int(policy["top_k"])
    veto_fraction = float(policy["bad_tail_veto_fraction"])
    base_cost = float(evaluation["cost_proxies"]["base"])
    stress_cost = float(evaluation["cost_proxies"]["stress"])
    horizon_column = base.HORIZONS.index(HORIZON) + 1
    within_cutoff = (
        panel.flags[rows, horizon_column] & base.FLAG_OUTCOME_WITHIN_CUTOFF
    ) != 0
    boundaries = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1, len(rows)]
    records: list[dict[str, Any]] = []
    for left, right in pairwise(boundaries):
        left_i = int(left)
        right_i = int(right)
        local = slice(left_i, right_i)
        chosen = _select_day(
            residual_score=np.asarray(residual_score)[local],
            bad_tail_score=np.asarray(bad_tail_score)[local],
            hierarchical_score=hierarchical[local],
            identity=identities[local],
            top_k=top_k,
            veto_fraction=veto_fraction,
            stress_cost=stress_cost,
        )
        local_calendar_evaluable = bool(np.all(within_cutoff[local]))
        local_gross_valid = gross_valid[local]
        universe = np.expm1(gross[local][local_gross_valid])
        universe_mean = float(np.mean(universe)) if len(universe) else float("nan")
        for policy_name, local_selected in chosen.items():
            selected = left_i + np.asarray(local_selected, dtype=np.int64)
            selected_gross_valid = gross_valid[selected]
            selected_residual_valid = residual_valid[selected]
            gross_simple = np.expm1(gross[selected][selected_gross_valid])
            residual_simple = np.expm1(residual[selected][selected_residual_valid])
            selected_count = len(selected)
            observed_count = int(selected_gross_valid.sum())
            gross_sum = float(np.sum(gross_simple))
            residual_sum = float(np.sum(residual_simple))
            record = {
                "date_idx": int(dates[left_i]),
                "trade_date": str(panel.trade_date[rows[left_i]]),
                "evaluation_year": int(evaluation_year),
                "policy": policy_name,
                "calendar_evaluable": local_calendar_evaluable,
                "candidate_count": right_i - left_i,
                "selected_count": selected_count,
                "invested": selected_count > 0,
                "observed_count": observed_count,
                "observed_fraction": (
                    float(observed_count / selected_count) if selected_count else 1.0
                ),
                "predicted_basket_log_return": (
                    float(np.mean(hierarchical[selected])) if selected_count else 0.0
                ),
                "gross_return": gross_sum / top_k,
                "gross_net_base": (gross_sum - base_cost * observed_count) / top_k,
                "gross_net_stress": (gross_sum - stress_cost * observed_count) / top_k,
                "hedged_residual_net_stress": (
                    residual_sum - stress_cost * int(selected_residual_valid.sum())
                )
                / top_k,
                "selected_observed_mean_gross": (
                    float(np.mean(gross_simple)) if len(gross_simple) else 0.0
                ),
                "universe_mean_gross": universe_mean,
                "selected_excess_gross": (
                    float(np.mean(gross_simple) - universe_mean)
                    if len(gross_simple) and np.isfinite(universe_mean)
                    else float("nan")
                ),
            }
            records.append(record)
    result = pd.DataFrame.from_records(records)
    if bool(result["trade_date"].str.startswith("2026-").any()):
        raise ValueError("forbidden_2026_evaluation_row")
    return result


def _summarize_policy(
    daily: pd.DataFrame,
    *,
    study: Mapping[str, Any],
    policy_name: str,
) -> dict[str, Any]:
    selected = daily.loc[
        (daily["policy"] == policy_name) & daily["calendar_evaluable"]
    ].copy()
    if not len(selected):
        raise ValueError(f"empty_policy_daily:{policy_name}")
    inference = dict(study["evaluation"]["inference"])
    stress = selected["gross_net_stress"].to_numpy(dtype=np.float64)
    residual_stress = selected["hedged_residual_net_stress"].to_numpy(dtype=np.float64)
    annual = (
        selected.groupby("evaluation_year", sort=True, as_index=False)
        .agg(
            gross_net_stress=("gross_net_stress", "mean"),
            gross_net_base=("gross_net_base", "mean"),
            hedged_residual_net_stress=("hedged_residual_net_stress", "mean"),
            invested_dates=("invested", "sum"),
            date_count=("date_idx", "size"),
        )
        .to_dict(orient="records")
    )
    invested = selected.loc[selected["invested"]]
    selected_total = float(invested["selected_count"].sum())
    observed_total = float(invested["observed_count"].sum())
    return {
        "policy": policy_name,
        "date_count": len(selected),
        "invested_date_count": int(selected["invested"].sum()),
        "invested_date_fraction": float(selected["invested"].mean()),
        "mean_selected_count_when_invested": (
            float(invested["selected_count"].mean()) if len(invested) else 0.0
        ),
        "observed_outcome_fraction": (
            observed_total / selected_total if selected_total > 0 else 1.0
        ),
        "gross": base._hac_mean(selected["gross_return"].to_numpy(), lag=HORIZON),
        "gross_net_base": base._hac_mean(
            selected["gross_net_base"].to_numpy(), lag=HORIZON
        ),
        "gross_net_stress": {
            **base._hac_mean(stress, lag=int(inference["hac_lag"])),
            "block": base._block_interval(
                stress,
                block_length=int(inference["moving_block_length"]),
                repetitions=int(inference["bootstrap_repetitions"]),
                seed=int(inference["seed"]),
            ),
        },
        "hedged_residual_net_stress": {
            **base._hac_mean(residual_stress, lag=int(inference["hac_lag"])),
            "block": base._block_interval(
                residual_stress,
                block_length=int(inference["moving_block_length"]),
                repetitions=int(inference["bootstrap_repetitions"]),
                seed=int(inference["seed"]) + 1,
            ),
        },
        "selected_excess_gross": base._hac_mean(
            selected["selected_excess_gross"].to_numpy(), lag=HORIZON
        ),
        "annual": annual,
    }


def _evaluate_gate(
    summary: Mapping[str, Any], *, gate: Mapping[str, Any]
) -> dict[str, Any]:
    annual = list(summary["annual"])
    positive_years = sum(float(row["gross_net_stress"]) > 0 for row in annual)
    checks = {
        "stress_net_hac_lcb_positive": float(summary["gross_net_stress"]["lcb_95"]) > 0,
        "stress_net_block_lcb_positive": float(
            summary["gross_net_stress"]["block"]["lcb_95"]
        )
        > 0,
        "minimum_positive_years": positive_years >= int(gate["minimum_positive_years"]),
    }
    if "minimum_mean_selected_count_when_invested" in gate:
        checks["minimum_mean_selected_count_when_invested"] = float(
            summary["mean_selected_count_when_invested"]
        ) >= float(gate["minimum_mean_selected_count_when_invested"])
    if "minimum_observed_outcome_fraction" in gate:
        checks["minimum_observed_outcome_fraction"] = float(
            summary["observed_outcome_fraction"]
        ) >= float(gate["minimum_observed_outcome_fraction"])
    if gate.get("hedged_residual_stress_net_hac_lower_bound_strictly_positive"):
        checks["hedged_residual_stress_net_hac_lcb_positive"] = (
            float(summary["hedged_residual_net_stress"]["lcb_95"]) > 0
        )
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "positive_year_count": positive_years,
        "required_positive_year_count": int(gate["minimum_positive_years"]),
    }


def _year_date_bounds(panel: base.StockPanel, year: int) -> tuple[int, int]:
    rows = panel.rows_for_year(year)
    return int(panel.date_idx[rows[0]]), int(panel.date_idx[rows[-1]])


def _fit_and_predict_factors(
    panel: base.StockPanel,
    factor_panels: FactorPanels,
    *,
    study: Mapping[str, Any],
    evaluation_year: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    fold = base._build_fold(
        panel,
        evaluation_year=evaluation_year,
        horizon=HORIZON,
        target=f"residual_log_return_{HORIZON}",
    )
    maximum_train_date_idx = int(fold["maximum_train_signal_date_idx"])
    market_model = _fit_factor_model(
        factor_panels.market,
        maximum_train_date_idx=maximum_train_date_idx,
        penalty=float(study["models"]["market_component"]["ridge_penalty"]),
        name="market_component",
    )
    industry_model = _fit_factor_model(
        factor_panels.industry,
        maximum_train_date_idx=maximum_train_date_idx,
        penalty=float(study["models"]["industry_component"]["ridge_penalty"]),
        name="industry_component",
    )
    market_group_prediction = _predict_factor_groups(market_model, factor_panels.market)
    industry_group_prediction = _predict_factor_groups(
        industry_model, factor_panels.industry
    )
    rows = panel.rows_for_year(evaluation_year)
    market_position = factor_panels.market.lookup[panel.date_idx[rows]]
    industry_key = panel.date_idx[rows].astype(
        np.int64
    ) * factor_panels.industry.lookup_width + panel.industry_code[rows].astype(np.int64)
    industry_position = factor_panels.industry.lookup[industry_key]
    if bool((market_position < 0).any()) or bool((industry_position < 0).any()):
        raise ValueError(f"factor_prediction_group_missing:{evaluation_year}")
    market_prediction = market_group_prediction[market_position]
    industry_prediction = industry_group_prediction[industry_position]

    start, stop = _year_date_bounds(panel, evaluation_year)
    diagnostics: list[dict[str, Any]] = []
    for grouped, prediction, model in (
        (factor_panels.market, market_group_prediction, market_model),
        (factor_panels.industry, industry_group_prediction, industry_model),
    ):
        year_mask = (grouped.date_idx >= start) & (grouped.date_idx <= stop)
        year_grouped = GroupedFactorPanel(
            date_idx=grouped.date_idx[year_mask],
            group_code=grouped.group_code[year_mask],
            features=grouped.features[year_mask],
            target=grouped.target[year_mask],
            observation_count=grouped.observation_count[year_mask],
            feature_names=grouped.feature_names,
            lookup=np.empty(0, dtype=np.int64),
            lookup_width=grouped.lookup_width,
        )
        diagnostics.append(
            _factor_diagnostic(
                year_grouped,
                prediction[year_mask],
                model=model,
                evaluation_year=evaluation_year,
            )
        )
    model_record = {
        "evaluation_year": evaluation_year,
        "fold": {
            key: value
            for key, value in fold.items()
            if key not in {"train_rows", "evaluation_rows"}
        },
        "market": _factor_model_record(market_model),
        "industry": _factor_model_record(industry_model),
    }
    return market_prediction, industry_prediction, diagnostics, model_record


def _phase_complete(
    path: Path, *, study_sha256: str, expected_years: Sequence[int]
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    summary = _read_json(path)
    if (
        summary.get("schema") != SUMMARY_SCHEMA
        or summary.get("study_sha256") != study_sha256
        or tuple(int(v) for v in summary.get("evaluation_years", ()))
        != tuple(int(v) for v in expected_years)
    ):
        raise ValueError(f"existing_phase_contract_mismatch:{path.parent.name}")
    if not all(
        base._record_valid(record, verify_hash=True)
        for record in dict(summary.get("files", {}) or {}).values()
    ):
        raise ValueError(f"existing_phase_file_invalid:{path.parent.name}")
    return summary


def run_phase(
    *,
    panel: base.StockPanel,
    factor_panels: FactorPanels,
    study: Mapping[str, Any],
    study_path: Path,
    source_contract: Mapping[str, Any],
    phase: str,
    output_root: Path,
    force: bool = False,
) -> dict[str, Any]:
    if phase not in {"development", "confirmation"}:
        raise ValueError(f"unknown_phase:{phase}")
    years = DEVELOPMENT_YEARS if phase == "development" else CONFIRMATION_YEARS
    phase_root = output_root / phase
    summary_path = phase_root / "summary.json"
    study_sha256 = base._sha256_file(study_path)
    if not force:
        current = _phase_complete(
            summary_path, study_sha256=study_sha256, expected_years=years
        )
        if current is not None:
            return current
    phase_root.mkdir(parents=True, exist_ok=True)
    source_paths = (
        _source_forecast_paths(source_contract["development_summary"])
        if phase == "development"
        else {}
    )
    daily_frames: list[pd.DataFrame] = []
    factor_diagnostics: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    for year in years:
        market_prediction, industry_prediction, diagnostic, factor_record = (
            _fit_and_predict_factors(
                panel,
                factor_panels,
                study=study,
                evaluation_year=year,
            )
        )
        if phase == "development":
            residual_score, bad_tail_score = _load_frozen_stock_forecasts(
                panel, year, source_paths[year]
            )
            stock_record: dict[str, Any] = {
                "source": str(source_paths[year]),
                "source_reused": True,
            }
        else:
            residual_score, bad_tail_score, stock_record = _train_stock_heads(
                panel,
                study=study,
                evaluation_year=year,
                model_root=phase_root / "models" / f"y{year}",
            )
            stock_record["source_reused"] = False
        daily_frames.append(
            _evaluate_year(
                panel,
                evaluation_year=year,
                residual_score=residual_score,
                bad_tail_score=bad_tail_score,
                market_prediction=market_prediction,
                industry_prediction=industry_prediction,
                study=study,
            )
        )
        factor_diagnostics.extend(diagnostic)
        fold_records.append(
            {
                "evaluation_year": year,
                "factor_models": factor_record,
                "stock_models": stock_record,
            }
        )
    daily = pd.concat(daily_frames, ignore_index=True)
    summaries = {
        policy: _summarize_policy(daily, study=study, policy_name=policy)
        for policy in POLICIES
    }
    gate_name = f"{phase}_gate"
    gate = _evaluate_gate(
        summaries[PRIMARY_POLICY], gate=dict(study["evaluation"][gate_name])
    )
    daily_path = phase_root / "policy_daily.parquet"
    diagnostic_path = phase_root / "factor_diagnostics.parquet"
    models_path = phase_root / "fold_records.json"
    _write_parquet(daily_path, daily)
    _write_parquet(diagnostic_path, pd.DataFrame(factor_diagnostics))
    _write_json(models_path, {"folds": fold_records})
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "study_id": STUDY_ID,
        "phase": phase,
        "study_sha256": study_sha256,
        "phase_fingerprint": _sha256_payload(
            {
                "study_sha256": study_sha256,
                "phase": phase,
                "years": list(years),
                "input_fingerprint": base.EXPECTED_INPUT_FINGERPRINT,
                "label_fingerprint": bad_tail.EXPECTED_LABEL_FINGERPRINT,
            }
        ),
        "input_fingerprint": base.EXPECTED_INPUT_FINGERPRINT,
        "label_fingerprint": bad_tail.EXPECTED_LABEL_FINGERPRINT,
        "evaluation_years": list(years),
        "primary_policy": PRIMARY_POLICY,
        "policy_summaries": summaries,
        "gate": gate,
        "eligible_for_next_phase": bool(gate["passed"]),
        "candidate_selection_precedes_outcome_masks": True,
        "calendar_cutoff_is_used_only_to_exclude_unobservable_end_dates": True,
        "retrospective_confirmation_consumed": phase == "confirmation",
        "account_replay_performed": False,
        "portfolio_optimization_performed": False,
        "forbidden_2026_read_count": 0,
        "profit_claim_allowed": False,
        "files": {
            "policy_daily": base._file_record(daily_path),
            "factor_diagnostics": base._file_record(diagnostic_path),
            "fold_records": base._file_record(models_path),
        },
    }
    _write_json(summary_path, summary)
    return summary


def run_preflight(
    *,
    panel: base.StockPanel,
    study: Mapping[str, Any],
    source_contract: Mapping[str, Any],
    output_root: Path,
    force: bool = False,
) -> dict[str, Any]:
    path = output_root / "preflight" / "summary.json"
    study_sha256 = base._sha256_file(DEFAULT_STUDY_PATH)
    if path.is_file() and not force:
        current = _read_json(path)
        if (
            current.get("schema") == PREFLIGHT_SCHEMA
            and current.get("study_sha256") == study_sha256
            and current.get("passed") is True
        ):
            return current
        raise ValueError("existing_preflight_contract_mismatch")
    year = int(study["evaluation"]["preflight_year"])
    source_path = _source_forecast_paths(source_contract["development_summary"])[year]
    frozen_residual, frozen_loss = _load_frozen_stock_forecasts(
        panel, year, source_path
    )
    residual, loss, fit = _train_stock_heads(
        panel,
        study=study,
        evaluation_year=year,
        model_root=output_root / "preflight" / "models",
    )
    residual_max = float(np.max(np.abs(residual - frozen_residual)))
    loss_max = float(np.max(np.abs(loss - frozen_loss)))
    tolerance = 2.0e-6
    passed = residual_max <= tolerance and loss_max <= tolerance
    summary = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "completed",
        "study_id": STUDY_ID,
        "study_sha256": study_sha256,
        "evaluation_year": year,
        "residual_max_absolute_difference": residual_max,
        "bad_tail_max_absolute_difference": loss_max,
        "tolerance": tolerance,
        "passed": passed,
        "fit": fit,
        "predictive_performance_used_for_tuning": False,
        "forbidden_2026_read_count": 0,
    }
    _write_json(path, summary)
    if not passed:
        raise ValueError("lean_stock_head_preflight_failed")
    return summary


def validate_summary(path: str | Path) -> dict[str, Any]:
    summary_path = base._resolve_path(path)
    summary = _read_json(summary_path)
    checks = {
        "schema": summary.get("schema") == SUMMARY_SCHEMA,
        "study": summary.get("study_id") == STUDY_ID,
        "candidate_selection_precedes_outcomes": summary.get(
            "candidate_selection_precedes_outcome_masks"
        )
        is True,
        "no_2026": int(summary.get("forbidden_2026_read_count", -1)) == 0,
        "no_account_replay": summary.get("account_replay_performed") is False,
        "files": all(
            base._record_valid(record, verify_hash=True)
            for record in dict(summary.get("files", {}) or {}).values()
        ),
        "primary_present": PRIMARY_POLICY in summary.get("policy_summaries", {}),
    }
    if not all(checks.values()):
        raise ValueError(f"summary_validation_failed:{checks}")
    return {"status": "ok", "checks": checks}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--phase",
        choices=("development", "confirmation", "all"),
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    study, study_path = load_study(args.study)
    source_contract = _validate_source_contract(study)
    panel = base.load_panel(
        input_manifest_path=source_contract["input_manifest_path"],
        label_manifest_path=source_contract["label_manifest_path"],
    )
    if bool((panel.years == base.FORBIDDEN_YEAR).any()):
        raise ValueError("forbidden_2026_panel_row")
    factor_panels = _build_factor_panels(panel)
    output_root = base._resolve_path(args.output_root)
    development: dict[str, Any] | None = None
    if args.phase in {"development", "all"}:
        development = run_phase(
            panel=panel,
            factor_panels=factor_panels,
            study=study,
            study_path=study_path,
            source_contract=source_contract,
            phase="development",
            output_root=output_root,
            force=args.force,
        )
        print(json.dumps(development["gate"], ensure_ascii=False, indent=2))
    if args.phase in {"confirmation", "all"}:
        if development is None:
            development_path = output_root / "development" / "summary.json"
            development = _phase_complete(
                development_path,
                study_sha256=base._sha256_file(study_path),
                expected_years=DEVELOPMENT_YEARS,
            )
        if development is None or not bool(development["gate"]["passed"]):
            print("confirmation_not_authorized:development_gate_failed")
            return
        run_preflight(
            panel=panel,
            study=study,
            source_contract=source_contract,
            output_root=output_root,
            force=args.force,
        )
        confirmation = run_phase(
            panel=panel,
            factor_panels=factor_panels,
            study=study,
            study_path=study_path,
            source_contract=source_contract,
            phase="confirmation",
            output_root=output_root,
            force=args.force,
        )
        print(json.dumps(confirmation["gate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
