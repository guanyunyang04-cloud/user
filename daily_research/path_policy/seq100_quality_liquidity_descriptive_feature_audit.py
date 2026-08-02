from __future__ import annotations

"""Full-population pre-training descriptive audit for the repaired Seq100 pool."""

import argparse
import gc
import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from daily_research.path_policy import seq100_quality_liquidity_data_prep as data_prep

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_quality_liquidity_descriptive_feature_audit"
SOURCE_STUDY_ID = "seq100_quality_liquidity_training_ready"
BUILDER_VERSION = "seq100_quality_liquidity_descriptive_feature_audit/1.0"
YEARS = tuple(range(2011, 2026))
HORIZONS = (10, 20)
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT
    / "daily_research/studies/seq100_quality_liquidity_descriptive_feature_audit.json"
)
DEFAULT_SOURCE_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_quality_liquidity_training_ready"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT / "daily_research/output/path_policy/studies" / STUDY_ID
)
BASE_FEATURE_MANIFEST = data_prep.BASE_FEATURE_MANIFEST
LABEL_MANIFEST = data_prep.LABEL_MANIFEST
LABEL_INDEX = {10: (4, 5, 1), 20: (7, 8, 2)}
FLAG_MFE_PRE_PEAK_MAE_VALID = 128
FLAG_STATE_ASSIGNED = 256


class DescriptiveAuditError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DescriptiveAuditError(f"required_json_missing:{path}")
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


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
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


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _record(path: Path, **extra: Any) -> dict[str, Any]:
    result = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size": int(path.stat().st_size),
    }
    result.update(extra)
    return result


def _chunks(values: Sequence[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not bool(valid.any()):
        return float("nan")
    return float(np.average(values[valid].astype(float), weights=weights[valid]))


def _weighted_median(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not bool(valid.any()):
        return float("nan")
    value_array = values[valid].to_numpy(dtype=np.float64)
    weight_array = weights[valid].to_numpy(dtype=np.float64)
    order = np.argsort(value_array, kind="stable")
    value_array = value_array[order]
    weight_array = weight_array[order]
    cutoff = weight_array.sum() / 2.0
    return float(value_array[np.searchsorted(np.cumsum(weight_array), cutoff)])


def period_for_year(year: int) -> str:
    if 2011 <= year <= 2015:
        return "2011_2015"
    if 2016 <= year <= 2020:
        return "2016_2020"
    if 2021 <= year <= 2025:
        return "2021_2025"
    raise DescriptiveAuditError(f"year_outside_research_period:{year}")


def recent_equal_or_larger_reversal(
    annual_effects: Mapping[int, float],
    *,
    recent_years: Sequence[int] = (2023, 2024, 2025),
) -> bool:
    recent_set = {int(year) for year in recent_years}
    history = np.asarray(
        [
            value
            for year, value in annual_effects.items()
            if int(year) not in recent_set and np.isfinite(value)
        ],
        dtype=np.float64,
    )
    recent = np.asarray(
        [
            annual_effects[year]
            for year in recent_years
            if year in annual_effects and np.isfinite(annual_effects[year])
        ],
        dtype=np.float64,
    )
    if history.size == 0 or recent.size == 0:
        return False
    historical_center = float(np.median(history))
    recent_center = float(np.mean(recent))
    return bool(
        historical_center != 0.0
        and recent_center != 0.0
        and np.sign(historical_center) != np.sign(recent_center)
        and abs(recent_center) >= abs(historical_center)
    )


def classify_family(
    *,
    stable_feature_count: int,
    stable_nonreversed_feature_count: int,
    availability_gated: bool,
    formal_eligible_feature_count: int,
    minimum_independent_representatives: int = 1,
) -> str:
    if formal_eligible_feature_count <= 0:
        return "diagnostic_only"
    if availability_gated:
        return "availability_gated_family"
    if (
        stable_feature_count > 0
        and stable_nonreversed_feature_count >= minimum_independent_representatives
    ):
        return "first_model_formal_family"
    return "defer_from_initial_model"


def _analytic_family(
    *, name: str, legacy_family: str, block: str, source_domain: str
) -> str:
    if block == "existing_seq100_base":
        return {
            "F1": "daily_price_volume_technical",
            "F2": "daily_cross_sectional_technical",
            "F3": "market_state",
            "F4": "industry_context",
            "F5": "size_liquidity_and_status",
        }.get(legacy_family, "existing_daily_other")
    if block == "minute":
        return "same_day_5m"
    if block == "fundamental":
        if name.startswith(("financial_", "performance_forecast_")):
            return "structured_financial_summary"
        return "financial_statements"
    if block == "event":
        return "research_reports" if name.startswith("research_") else "announcements"
    if block == "tushare_technical_candidates":
        return "traditional_technical_indicators"
    if block == "margin_features":
        return "margin_market" if source_domain == "margin_market" else "margin_detail"
    if block == "traditional_moneyflow_features":
        return "traditional_moneyflow"
    if block == "financial_statement_extensions":
        return "financial_statement_extensions"
    return block


def _feature_catalog(
    source_manifest: Mapping[str, Any], source_root: Path
) -> pd.DataFrame:
    legacy_path = Path(source_manifest["existing_atlas_518"]["manifest_path"])
    legacy_manifest = _read_json(legacy_path)
    catalog_path = Path(legacy_manifest["files"]["feature_catalog"]["path"])
    existing = pd.read_parquet(catalog_path).rename(columns={"name": "feature_name"})
    existing["physical_column"] = existing["feature_name"]
    existing["block"] = existing["source_block"]
    existing["source_domain"] = existing["source_block"]
    existing["source_field"] = existing["feature_name"]
    existing["eligibility"] = "formal_existing"
    existing["eligibility_reason"] = "frozen_existing_seq100_feature"
    existing["legacy_family"] = existing["family"]

    registry = pd.read_parquet(Path(source_manifest["feature_registry"]["path"]))
    registry["legacy_family"] = ""
    eligible = registry[
        registry["eligibility"].isin(["formal_candidate", "availability_gated"])
    ].copy()
    columns = [
        "feature_name",
        "physical_column",
        "block",
        "source_domain",
        "source_field",
        "eligibility",
        "eligibility_reason",
        "legacy_family",
    ]
    combined = pd.concat([existing[columns], eligible[columns]], ignore_index=True)
    combined["analytic_family"] = [
        _analytic_family(
            name=str(row.feature_name),
            legacy_family=str(row.legacy_family),
            block=str(row.block),
            source_domain=str(row.source_domain),
        )
        for row in combined.itertuples(index=False)
    ]
    combined = combined.drop_duplicates("feature_name", keep="first").reset_index(
        drop=True
    )
    if len(combined) != 628:
        raise DescriptiveAuditError(f"unexpected_numeric_feature_count:{len(combined)}")
    return combined


def _label_memmaps() -> tuple[dict[str, Any], np.memmap, np.memmap, np.memmap]:
    manifest = _read_json(LABEL_MANIFEST)
    labels_record = manifest["files"]["candidate_labels"]
    flags_record = manifest["files"]["label_flags"]
    states_record = manifest["files"]["state_labels"]
    labels = np.memmap(
        Path(labels_record["path"]),
        dtype=np.float32,
        mode="r",
        shape=tuple(labels_record["shape"]),
    )
    flags = np.memmap(
        Path(flags_record["path"]),
        dtype=np.uint16,
        mode="r",
        shape=tuple(flags_record["shape"]),
    )
    states = np.memmap(
        Path(states_record["path"]),
        dtype=np.int8,
        mode="r",
        shape=tuple(states_record["shape"]),
    )
    return manifest, labels, flags, states


def _rank_within_date(values: np.ndarray, date_idx: np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    ranks = series.groupby(date_idx, sort=False).rank(method="average", pct=True)
    return ranks.to_numpy(dtype=np.float64)


def _label_bundle(
    candidate_ids: np.ndarray,
    date_idx: np.ndarray,
    labels: np.memmap,
    flags: np.memmap,
    states: np.memmap,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {"date_idx": date_idx.astype(np.int32, copy=False)}
    for horizon in HORIZONS:
        mfe_index, mae_index, horizon_index = LABEL_INDEX[horizon]
        mfe = np.asarray(labels[candidate_ids, mfe_index], dtype=np.float64)
        mae = np.asarray(labels[candidate_ids, mae_index], dtype=np.float64)
        state = np.asarray(states[candidate_ids, horizon_index], dtype=np.int8)
        horizon_flags = np.asarray(flags[candidate_ids, horizon_index], dtype=np.uint16)
        path_valid = (
            ((horizon_flags & FLAG_MFE_PRE_PEAK_MAE_VALID) != 0)
            & np.isfinite(mfe)
            & np.isfinite(mae)
        )
        state_valid = ((horizon_flags & FLAG_STATE_ASSIGNED) != 0) & (state >= 0)
        masked_mfe = np.where(path_valid, mfe, np.nan)
        mfe_rank = _rank_within_date(masked_mfe, date_idx)
        result[f"mfe_{horizon}"] = mfe
        result[f"mae_{horizon}"] = mae
        result[f"state_{horizon}"] = state
        result[f"valid_{horizon}"] = path_valid
        result[f"state_valid_{horizon}"] = state_valid
        result[f"mfe_rank_{horizon}"] = mfe_rank
        result[f"top1_{horizon}"] = path_valid & (mfe_rank > 0.99)
        result[f"top5_{horizon}"] = path_valid & (mfe_rank > 0.95)
    return result


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _safe_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def _safe_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _target_summary_rows(
    *, year: int, bundle: Mapping[str, np.ndarray]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        valid = bundle[f"valid_{horizon}"]
        mfe = bundle[f"mfe_{horizon}"][valid]
        mae = bundle[f"mae_{horizon}"][valid]
        state = bundle[f"state_{horizon}"]
        state_valid = valid & bundle[f"state_valid_{horizon}"]
        quantiles = (
            np.quantile(mfe, [0.01, 0.05, 0.50, 0.95, 0.99])
            if mfe.size
            else np.full(5, np.nan)
        )
        rows.append(
            {
                "year": year,
                "horizon": horizon,
                "row_count": len(valid),
                "valid_count": int(valid.sum()),
                "top1_count": int(bundle[f"top1_{horizon}"].sum()),
                "top5_count": int(bundle[f"top5_{horizon}"].sum()),
                "top1_rate": _safe_rate(bundle[f"top1_{horizon}"].sum(), valid.sum()),
                "top5_rate": _safe_rate(bundle[f"top5_{horizon}"].sum(), valid.sum()),
                "mfe_mean": _safe_mean(mfe),
                "mfe_q01": float(quantiles[0]),
                "mfe_q05": float(quantiles[1]),
                "mfe_median": float(quantiles[2]),
                "mfe_q95": float(quantiles[3]),
                "mfe_q99": float(quantiles[4]),
                "mae_mean": _safe_mean(mae),
                "mae_median": _safe_median(mae),
                "state_valid_count": int(state_valid.sum()),
                "state_low_count": int((state_valid & (state == 0)).sum()),
                "state_mid_count": int((state_valid & (state == 1)).sum()),
                "state_high_count": int((state_valid & (state == 2)).sum()),
                "state_low_rate": _safe_rate(
                    (state[state_valid] == 0).sum(), state_valid.sum()
                ),
                "state_mid_rate": _safe_rate(
                    (state[state_valid] == 1).sum(), state_valid.sum()
                ),
                "state_high_rate": _safe_rate(
                    (state[state_valid] == 2).sum(), state_valid.sum()
                ),
            }
        )
    return rows


def _relation_rows_for_chunk(
    *,
    year: int,
    values: pd.DataFrame,
    bundle: Mapping[str, np.ndarray],
    catalog: pd.DataFrame,
    tail_fraction: float,
    minimum_valid_rows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    clean = values.replace([np.inf, -np.inf], np.nan)
    cross_sectional_ranks = clean.groupby(bundle["date_idx"], sort=False).rank(
        method="average", pct=True
    )
    metadata = catalog.set_index("feature_name").to_dict(orient="index")
    relation_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for name in clean.columns:
        raw = pd.to_numeric(clean[name], errors="coerce").to_numpy(dtype=np.float64)
        info = metadata[name]
        market_wide = info["analytic_family"] in {"market_state", "margin_market"}
        if market_wide:
            dates = pd.Series(bundle["date_idx"], name="date_idx")
            dated = pd.DataFrame({"date_idx": dates, "value": raw})
            daily = dated.groupby("date_idx", sort=False)["value"].median()
            daily_rank = daily.rank(method="average", pct=True)
            rank = dates.map(daily_rank).to_numpy(dtype=np.float64)
            relation_axis = "annual_trade_date_time_series"
        else:
            rank = cross_sectional_ranks[name].to_numpy(dtype=np.float64)
            relation_axis = "within_trade_date_cross_section"
        finite = np.isfinite(raw)
        finite_values = raw[finite]
        distinct = (
            int(pd.Series(finite_values).nunique(dropna=True))
            if finite_values.size
            else 0
        )
        quantiles = (
            np.quantile(finite_values, [0.01, 0.50, 0.99])
            if finite_values.size
            else np.full(3, np.nan)
        )
        coverage_rows.append(
            {
                "year": year,
                "feature": name,
                "analytic_family": info["analytic_family"],
                "eligibility": info["eligibility"],
                "row_count": len(raw),
                "nonnull_count": int(finite.sum()),
                "nonnull_rate": _safe_rate(finite.sum(), len(raw)),
                "distinct_count": distinct,
                "near_constant": bool(distinct <= 1),
                "q01": float(quantiles[0]),
                "q50": float(quantiles[1]),
                "q99": float(quantiles[2]),
                "mean": _safe_mean(finite_values),
                "std": float(np.std(finite_values))
                if finite_values.size
                else float("nan"),
                "minimum": float(np.min(finite_values))
                if finite_values.size
                else float("nan"),
                "maximum": float(np.max(finite_values))
                if finite_values.size
                else float("nan"),
                "infinite_count": int(
                    np.isinf(pd.to_numeric(values[name], errors="coerce")).sum()
                ),
                "relation_axis": relation_axis,
            }
        )
        high_feature = np.isfinite(rank) & (rank > 1.0 - tail_fraction)
        low_feature = np.isfinite(rank) & (rank <= tail_fraction)
        for horizon in HORIZONS:
            valid = finite & bundle[f"valid_{horizon}"] & np.isfinite(rank)
            high = valid & high_feature
            low = valid & low_feature
            valid_count = int(valid.sum())
            high_count = int(high.sum())
            low_count = int(low.sum())
            if valid_count < minimum_valid_rows:
                correlation = float("nan")
            else:
                outcome_rank = bundle[f"mfe_rank_{horizon}"]
                if np.std(rank[valid]) == 0 or np.std(outcome_rank[valid]) == 0:
                    correlation = float("nan")
                else:
                    correlation = float(
                        np.corrcoef(rank[valid], outcome_rank[valid])[0, 1]
                    )
            top1 = bundle[f"top1_{horizon}"]
            top5 = bundle[f"top5_{horizon}"]
            state = bundle[f"state_{horizon}"]
            state_valid = bundle[f"state_valid_{horizon}"]
            high_state = high & state_valid
            low_state = low & state_valid
            baseline_top1_hits = int((valid & top1).sum())
            baseline_top5_hits = int((valid & top5).sum())
            high_top1_hits = int((high & top1).sum())
            low_top1_hits = int((low & top1).sum())
            high_top5_hits = int((high & top5).sum())
            low_top5_hits = int((low & top5).sum())
            high_top1_rate = _safe_rate(high_top1_hits, high_count)
            low_top1_rate = _safe_rate(low_top1_hits, low_count)
            high_top5_rate = _safe_rate(high_top5_hits, high_count)
            low_top5_rate = _safe_rate(low_top5_hits, low_count)
            baseline_top1_rate = _safe_rate(baseline_top1_hits, valid_count)
            baseline_top5_rate = _safe_rate(baseline_top5_hits, valid_count)
            high_state_count = int(high_state.sum())
            low_state_count = int(low_state.sum())
            high_state_high_hits = int((high_state & (state == 2)).sum())
            low_state_high_hits = int((low_state & (state == 2)).sum())
            relation_rows.append(
                {
                    "year": year,
                    "horizon": horizon,
                    "feature": name,
                    "analytic_family": info["analytic_family"],
                    "eligibility": info["eligibility"],
                    "relation_axis": relation_axis,
                    "valid_count": valid_count,
                    "high_count": high_count,
                    "low_count": low_count,
                    "spearman_daily_rank": correlation,
                    "baseline_top1_hits": baseline_top1_hits,
                    "baseline_top5_hits": baseline_top5_hits,
                    "high_top1_hits": high_top1_hits,
                    "low_top1_hits": low_top1_hits,
                    "high_top5_hits": high_top5_hits,
                    "low_top5_hits": low_top5_hits,
                    "baseline_top1_rate": baseline_top1_rate,
                    "baseline_top5_rate": baseline_top5_rate,
                    "high_top1_rate": high_top1_rate,
                    "low_top1_rate": low_top1_rate,
                    "high_top5_rate": high_top5_rate,
                    "low_top5_rate": low_top5_rate,
                    "high_top1_lift": _safe_rate(high_top1_rate, baseline_top1_rate),
                    "low_top1_lift": _safe_rate(low_top1_rate, baseline_top1_rate),
                    "high_top5_lift": _safe_rate(high_top5_rate, baseline_top5_rate),
                    "low_top5_lift": _safe_rate(low_top5_rate, baseline_top5_rate),
                    "high_minus_low_top1_rate": high_top1_rate - low_top1_rate,
                    "high_minus_low_top5_rate": high_top5_rate - low_top5_rate,
                    "high_mfe_mean": _safe_mean(bundle[f"mfe_{horizon}"][high]),
                    "low_mfe_mean": _safe_mean(bundle[f"mfe_{horizon}"][low]),
                    "high_mfe_median": _safe_median(bundle[f"mfe_{horizon}"][high]),
                    "low_mfe_median": _safe_median(bundle[f"mfe_{horizon}"][low]),
                    "high_mae_mean": _safe_mean(bundle[f"mae_{horizon}"][high]),
                    "low_mae_mean": _safe_mean(bundle[f"mae_{horizon}"][low]),
                    "high_mae_median": _safe_median(bundle[f"mae_{horizon}"][high]),
                    "low_mae_median": _safe_median(bundle[f"mae_{horizon}"][low]),
                    "high_state_count": high_state_count,
                    "low_state_count": low_state_count,
                    "high_state_high_hits": high_state_high_hits,
                    "low_state_high_hits": low_state_high_hits,
                    "high_state_high_rate": _safe_rate(
                        high_state_high_hits, high_state_count
                    ),
                    "low_state_high_rate": _safe_rate(
                        low_state_high_hits, low_state_count
                    ),
                    "high_minus_low_state_high_rate": _safe_rate(
                        high_state_high_hits, high_state_count
                    )
                    - _safe_rate(low_state_high_hits, low_state_count),
                }
            )
    return relation_rows, coverage_rows


def _stratum_row(
    *,
    year: int,
    dimension: str,
    stratum: str,
    mask: np.ndarray,
    horizon: int,
    bundle: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    valid = mask & bundle[f"valid_{horizon}"]
    state_valid = valid & bundle[f"state_valid_{horizon}"]
    state = bundle[f"state_{horizon}"]
    return {
        "year": year,
        "dimension": dimension,
        "stratum": str(stratum),
        "horizon": horizon,
        "row_count": int(mask.sum()),
        "valid_count": int(valid.sum()),
        "top1_hits": int((valid & bundle[f"top1_{horizon}"]).sum()),
        "top5_hits": int((valid & bundle[f"top5_{horizon}"]).sum()),
        "top1_rate": _safe_rate((valid & bundle[f"top1_{horizon}"]).sum(), valid.sum()),
        "top5_rate": _safe_rate((valid & bundle[f"top5_{horizon}"]).sum(), valid.sum()),
        "mfe_mean": _safe_mean(bundle[f"mfe_{horizon}"][valid]),
        "mfe_median": _safe_median(bundle[f"mfe_{horizon}"][valid]),
        "mae_mean": _safe_mean(bundle[f"mae_{horizon}"][valid]),
        "mae_median": _safe_median(bundle[f"mae_{horizon}"][valid]),
        "state_low_hits": int((state_valid & (state == 0)).sum()),
        "state_mid_hits": int((state_valid & (state == 1)).sum()),
        "state_high_hits": int((state_valid & (state == 2)).sum()),
        "state_valid_count": int(state_valid.sum()),
        "state_low_rate": _safe_rate(
            (state_valid & (state == 0)).sum(), state_valid.sum()
        ),
        "state_mid_rate": _safe_rate(
            (state_valid & (state == 1)).sum(), state_valid.sum()
        ),
        "state_high_rate": _safe_rate(
            (state_valid & (state == 2)).sum(), state_valid.sum()
        ),
    }


def _context_and_strata(
    *,
    year: int,
    diagnostics_path: Path,
    spine: pd.DataFrame,
    bundle: Mapping[str, np.ndarray],
    base_values: np.memmap,
    turnover_column: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "candidate_id",
        "industry",
        "circ_mv",
        "amount_median_20",
        "circ_mv_rank",
        "amount_median_rank",
        "quality_liquidity_complete_keep",
    ]
    diagnostics = pd.read_parquet(diagnostics_path, columns=columns)
    diagnostics = diagnostics[diagnostics["quality_liquidity_complete_keep"]].copy()
    diagnostics = spine[["candidate_id"]].merge(
        diagnostics.drop(columns="quality_liquidity_complete_keep"),
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    candidate_ids = spine["candidate_id"].to_numpy(dtype=np.int64)
    turnover = np.asarray(base_values[candidate_ids, turnover_column], dtype=np.float64)
    diagnostics["turnover_daily_rank"] = _rank_within_date(
        turnover, spine["date_idx"].to_numpy(dtype=np.int32)
    )
    diagnostics["market_cap_band"] = pd.cut(
        diagnostics["circ_mv_rank"],
        bins=[-np.inf, 0.4, 0.6, 0.8, np.inf],
        labels=["rank_20_40", "rank_40_60", "rank_60_80", "rank_80_100"],
    ).astype("string")
    diagnostics["amount_band"] = pd.cut(
        diagnostics["amount_median_rank"],
        bins=[-np.inf, 0.5, 0.7, 0.85, np.inf],
        labels=["rank_30_50", "rank_50_70", "rank_70_85", "rank_85_100"],
    ).astype("string")
    diagnostics["turnover_band"] = pd.cut(
        diagnostics["turnover_daily_rank"],
        bins=[-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf],
        labels=["q1", "q2", "q3", "q4", "q5"],
    ).astype("string")
    diagnostics["industry"] = diagnostics["industry"].astype("string").fillna("Unknown")
    rows: list[dict[str, Any]] = []
    for dimension in ("industry", "market_cap_band", "amount_band", "turnover_band"):
        values = diagnostics[dimension].astype("string").fillna("Unknown")
        for stratum in sorted(values.unique().tolist()):
            mask = values.eq(stratum).to_numpy(dtype=bool)
            for horizon in HORIZONS:
                rows.append(
                    _stratum_row(
                        year=year,
                        dimension=dimension,
                        stratum=str(stratum),
                        mask=mask,
                        horizon=horizon,
                        bundle=bundle,
                    )
                )
    return diagnostics, pd.DataFrame(rows)


def _source_state_rows(
    *,
    year: int,
    values: pd.DataFrame,
    fields: Sequence[str],
    bundle: Mapping[str, np.ndarray],
    block: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in fields:
        states = values[field].astype("string").fillna("unknown")
        for state in sorted(states.unique().tolist()):
            mask = states.eq(state).to_numpy(dtype=bool)
            for horizon in HORIZONS:
                rows.append(
                    {
                        "year": year,
                        "block": block,
                        "field": field,
                        "state": str(state),
                        **_stratum_row(
                            year=year,
                            dimension=field,
                            stratum=str(state),
                            mask=mask,
                            horizon=horizon,
                            bundle=bundle,
                        ),
                    }
                )
    return rows


def _source_scope_manifest(source_manifest: Mapping[str, Any]) -> dict[str, Any]:
    atlas_manifest_path = Path(
        str(source_manifest["existing_atlas_518"]["manifest_path"])
    )
    scope_path = atlas_manifest_path.parent.parent / "manifest.json"
    return _read_json(scope_path)


def _physical_sources(source_manifest: Mapping[str, Any], year: int) -> dict[str, Path]:
    scope_manifest = _source_scope_manifest(source_manifest)
    sources = {
        block: Path(scope_manifest["feature_partitions"][str(year)][block]["path"])
        for block in ("minute", "fundamental", "event")
    }
    source_blocks = source_manifest["blocks"]
    for block in (
        "tushare_technical_candidates",
        "margin_features",
        "traditional_moneyflow_features",
        "financial_statement_extensions",
    ):
        sources[block] = Path(source_blocks[block]["partitions"][str(year)]["path"])
    return sources


def _diagnostics_path(source_manifest: Mapping[str, Any], year: int) -> Path:
    scope_manifest = _source_scope_manifest(source_manifest)
    minute_path = Path(
        scope_manifest["feature_partitions"][str(year)]["minute"]["path"]
    )
    data_prep_root = minute_path.parents[3]
    return data_prep_root / f"membership/year={year}/diagnostics.parquet"


def _state_fields_for_block(
    *, block: str, schema_names: Sequence[str], registry: pd.DataFrame
) -> list[str]:
    names = set(schema_names)
    fields: list[str] = []
    if "coverage_state" in names:
        fields.append("coverage_state")
    if block == "margin_features":
        fields.extend(
            name
            for name in (
                "margin_detail_coverage_state",
                "margin_market_coverage_state",
                "margin_eligibility_state",
            )
            if name in names
        )
    metadata = registry[
        (registry["block"] == block)
        & (registry["eligibility"] == "availability_metadata")
    ]["physical_column"].astype(str)
    fields.extend(name for name in metadata if name in names)
    if block == "event":
        fields.extend(
            name
            for name in (
                "announcement_source_covered",
                "research_metadata_source_covered",
                "research_forecast_source_complete",
                "research_forecast_source_partial",
                "research_forecast_source_unavailable",
            )
            if name in names
        )
    if block == "fundamental":
        fields.extend(
            name
            for name in schema_names
            if name.endswith(("_present", "_missing", "_source_conflict"))
        )
    return list(dict.fromkeys(fields))


def _excluded_pool_rows(
    *,
    year: int,
    daily_spine_path: Path,
    complete_candidate_ids: np.ndarray,
    diagnostics_path: Path,
    labels: np.memmap,
    flags: np.memmap,
    states: np.memmap,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_spine = pd.read_parquet(daily_spine_path)
    candidate_ids = daily_spine["candidate_id"].to_numpy(dtype=np.int64)
    date_idx = daily_spine["date_idx"].to_numpy(dtype=np.int32)
    bundle = _label_bundle(candidate_ids, date_idx, labels, flags, states)
    retained = np.isin(candidate_ids, complete_candidate_ids, assume_unique=True)
    diagnostics = pd.read_parquet(
        diagnostics_path,
        columns=[
            "candidate_id",
            "symbol",
            "trade_date",
            "industry",
            "circ_mv",
            "amount_median_20",
            "circ_mv_rank",
            "amount_median_rank",
            "minute_complete",
            "quality_liquidity_keep",
        ],
    )
    diagnostics = diagnostics[diagnostics["quality_liquidity_keep"]]
    daily_spine = daily_spine[["candidate_id"]].merge(
        diagnostics, on="candidate_id", how="left", validate="one_to_one"
    )
    daily_spine["support_group"] = np.where(retained, "retained", "excluded_missing_5m")
    detail = daily_spine.loc[~retained].copy()
    for horizon in HORIZONS:
        detail[f"mfe_{horizon}"] = bundle[f"mfe_{horizon}"][~retained]
        detail[f"pre_peak_mae_{horizon}"] = bundle[f"mae_{horizon}"][~retained]
        detail[f"state_{horizon}"] = bundle[f"state_{horizon}"][~retained]
        detail[f"top1_{horizon}"] = bundle[f"top1_{horizon}"][~retained]
        detail[f"top5_{horizon}"] = bundle[f"top5_{horizon}"][~retained]
    comparison: list[dict[str, Any]] = []
    for group, mask in (
        ("retained", retained),
        ("excluded_missing_5m", ~retained),
    ):
        for horizon in HORIZONS:
            comparison.append(
                _stratum_row(
                    year=year,
                    dimension="minute_common_support",
                    stratum=group,
                    mask=mask,
                    horizon=horizon,
                    bundle=bundle,
                )
            )
    return detail, pd.DataFrame(comparison)


def _prepare_year(
    *,
    year: int,
    source_manifest: Mapping[str, Any],
    catalog: pd.DataFrame,
    registry: pd.DataFrame,
    output_root: Path,
    labels: np.memmap,
    flags: np.memmap,
    states: np.memmap,
    base_values: np.memmap,
    base_catalog: Mapping[str, int],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    year_root = output_root / "annual" / f"year={year}"
    spine_path = Path(source_manifest["row_spine"][str(year)]["path"])
    daily_spine_path = Path(
        source_manifest["daily_quality_row_spine"][str(year)]["path"]
    )
    spine = pd.read_parquet(spine_path)
    if set(spine["year"].unique()) != {year}:
        raise DescriptiveAuditError(f"spine_year_mismatch:{year}")
    candidate_ids = spine["candidate_id"].to_numpy(dtype=np.int64)
    date_idx = spine["date_idx"].to_numpy(dtype=np.int32)
    bundle = _label_bundle(candidate_ids, date_idx, labels, flags, states)
    target_summary = pd.DataFrame(_target_summary_rows(year=year, bundle=bundle))
    _write_frame(target_summary, year_root / "target_summary.parquet")

    turnover_column = base_catalog["log_turnover_rate"]
    context, strata = _context_and_strata(
        year=year,
        diagnostics_path=_diagnostics_path(source_manifest, year),
        spine=spine,
        bundle=bundle,
        base_values=base_values,
        turnover_column=turnover_column,
    )
    _write_frame(strata, year_root / "strata.parquet")

    relation_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    source_state_rows: list[dict[str, Any]] = []
    sample_parts: list[pd.DataFrame] = [
        spine[["candidate_id"]][
            candidate_ids
            % int(config["redundancy"]["deterministic_candidate_id_modulus"])
            == 0
        ].reset_index(drop=True)
    ]
    chunk_size = int(config["feature_relation"]["feature_chunk_size"])
    tail_fraction = float(config["feature_relation"]["quantile_tail_fraction"])
    minimum_valid_rows = int(config["feature_relation"]["minimum_valid_rows"])

    base_features = catalog[catalog["block"] == "existing_seq100_base"]
    for names in _chunks(
        base_features["feature_name"].astype(str).tolist(), chunk_size
    ):
        indices = [base_catalog[name] for name in names]
        values = pd.DataFrame(
            np.asarray(
                base_values[candidate_ids[:, None], np.asarray(indices)[None, :]],
                dtype=np.float32,
            ),
            columns=names,
        )
        relations, coverage = _relation_rows_for_chunk(
            year=year,
            values=values,
            bundle=bundle,
            catalog=base_features,
            tail_fraction=tail_fraction,
            minimum_valid_rows=minimum_valid_rows,
        )
        relation_rows.extend(relations)
        coverage_rows.extend(coverage)
        sample_parts.append(
            values.loc[
                candidate_ids
                % int(config["redundancy"]["deterministic_candidate_id_modulus"])
                == 0
            ].reset_index(drop=True)
        )
        del values

    sources = _physical_sources(source_manifest, year)
    for block, path in sources.items():
        block_catalog = catalog[catalog["block"] == block]
        if block_catalog.empty:
            continue
        parquet_schema = pq.ParquetFile(path).schema_arrow.names
        state_fields = _state_fields_for_block(
            block=block, schema_names=parquet_schema, registry=registry
        )
        if state_fields:
            state_frame = pd.read_parquet(path, columns=["candidate_id", *state_fields])
            if not np.array_equal(
                state_frame["candidate_id"].to_numpy(dtype=np.int64), candidate_ids
            ):
                state_frame = spine[["candidate_id"]].merge(
                    state_frame, on="candidate_id", how="left", validate="one_to_one"
                )
            source_state_rows.extend(
                _source_state_rows(
                    year=year,
                    values=state_frame,
                    fields=state_fields,
                    bundle=bundle,
                    block=block,
                )
            )
        for feature_names in _chunks(
            block_catalog["feature_name"].astype(str).tolist(), chunk_size
        ):
            mapping = block_catalog.set_index("feature_name")["physical_column"].astype(
                str
            )
            physical = [mapping[name] for name in feature_names]
            frame = pd.read_parquet(path, columns=["candidate_id", *physical])
            if not np.array_equal(
                frame["candidate_id"].to_numpy(dtype=np.int64), candidate_ids
            ):
                frame = spine[["candidate_id"]].merge(
                    frame, on="candidate_id", how="left", validate="one_to_one"
                )
            values = frame[physical].rename(
                columns=dict(zip(physical, feature_names, strict=True))
            )
            relations, coverage = _relation_rows_for_chunk(
                year=year,
                values=values,
                bundle=bundle,
                catalog=block_catalog,
                tail_fraction=tail_fraction,
                minimum_valid_rows=minimum_valid_rows,
            )
            relation_rows.extend(relations)
            coverage_rows.extend(coverage)
            sample_parts.append(
                values.loc[
                    candidate_ids
                    % int(config["redundancy"]["deterministic_candidate_id_modulus"])
                    == 0
                ].reset_index(drop=True)
            )
            del frame, values

    relation = pd.DataFrame(relation_rows)
    coverage = pd.DataFrame(coverage_rows)
    source_states = pd.DataFrame(source_state_rows)
    if len(coverage) != len(catalog) or len(relation) != len(catalog) * len(HORIZONS):
        raise DescriptiveAuditError(
            f"annual_feature_result_count_mismatch:{year}:{len(coverage)}:{len(relation)}"
        )
    _write_frame(relation, year_root / "feature_relation.parquet")
    _write_frame(coverage, year_root / "feature_coverage_distribution.parquet")
    _write_frame(source_states, year_root / "source_state.parquet")
    redundancy_sample = pd.concat(sample_parts, axis=1)
    if redundancy_sample.columns.duplicated().any():
        raise DescriptiveAuditError(f"duplicate_redundancy_sample_columns:{year}")
    _write_frame(redundancy_sample, year_root / "redundancy_sample.parquet")

    excluded, pool_comparison = _excluded_pool_rows(
        year=year,
        daily_spine_path=daily_spine_path,
        complete_candidate_ids=candidate_ids,
        diagnostics_path=_diagnostics_path(source_manifest, year),
        labels=labels,
        flags=flags,
        states=states,
    )
    _write_frame(excluded, year_root / "excluded_minute_rows.parquet")
    _write_frame(pool_comparison, year_root / "pool_comparison.parquet")
    result = {
        "year": year,
        "status": "completed",
        "row_count": len(spine),
        "daily_pool_row_count": len(
            pd.read_parquet(daily_spine_path, columns=["candidate_id"])
        ),
        "excluded_minute_row_count": len(excluded),
        "numeric_feature_count": len(coverage),
        "relation_row_count": len(relation),
        "source_state_row_count": len(source_states),
        "redundancy_sample_row_count": len(redundancy_sample),
        "maximum_trade_date": str(spine["trade_date"].max()),
        "minimum_trade_date": str(spine["trade_date"].min()),
        "training_performed": False,
        "feature_set_selected": False,
    }
    _write_json(year_root / "manifest.json", result)
    del bundle, context, strata, relation, coverage, source_states, redundancy_sample
    gc.collect()
    return result


def _aggregate_relation(
    annual: pd.DataFrame, years: Sequence[int], period: str
) -> pd.DataFrame:
    frame = annual[annual["year"].isin(years)]
    rows: list[dict[str, Any]] = []
    for (feature, family, eligibility, horizon), group in frame.groupby(
        ["feature", "analytic_family", "eligibility", "horizon"], sort=False
    ):
        valid_count = int(group["valid_count"].sum())
        high_count = int(group["high_count"].sum())
        low_count = int(group["low_count"].sum())
        baseline_top1_hits = int(group["baseline_top1_hits"].sum())
        baseline_top5_hits = int(group["baseline_top5_hits"].sum())
        high_top1_hits = int(group["high_top1_hits"].sum())
        low_top1_hits = int(group["low_top1_hits"].sum())
        high_top5_hits = int(group["high_top5_hits"].sum())
        low_top5_hits = int(group["low_top5_hits"].sum())
        high_top1_rate = _safe_rate(high_top1_hits, high_count)
        low_top1_rate = _safe_rate(low_top1_hits, low_count)
        high_top5_rate = _safe_rate(high_top5_hits, high_count)
        low_top5_rate = _safe_rate(low_top5_hits, low_count)
        baseline_top1_rate = _safe_rate(baseline_top1_hits, valid_count)
        baseline_top5_rate = _safe_rate(baseline_top5_hits, valid_count)
        high_state_count = int(group["high_state_count"].sum())
        low_state_count = int(group["low_state_count"].sum())
        high_state_high_hits = int(group["high_state_high_hits"].sum())
        low_state_high_hits = int(group["low_state_high_hits"].sum())
        rows.append(
            {
                "period": period,
                "feature": feature,
                "analytic_family": family,
                "eligibility": eligibility,
                "horizon": int(horizon),
                "relation_axis": str(group["relation_axis"].iloc[0]),
                "year_count": int(group["year"].nunique()),
                "valid_count": valid_count,
                "high_count": high_count,
                "low_count": low_count,
                "spearman_daily_rank": _weighted_mean(
                    group["spearman_daily_rank"], group["valid_count"]
                ),
                "baseline_top1_rate": baseline_top1_rate,
                "baseline_top5_rate": baseline_top5_rate,
                "high_top1_rate": high_top1_rate,
                "low_top1_rate": low_top1_rate,
                "high_top5_rate": high_top5_rate,
                "low_top5_rate": low_top5_rate,
                "high_top1_lift": _safe_rate(high_top1_rate, baseline_top1_rate),
                "low_top1_lift": _safe_rate(low_top1_rate, baseline_top1_rate),
                "high_top5_lift": _safe_rate(high_top5_rate, baseline_top5_rate),
                "low_top5_lift": _safe_rate(low_top5_rate, baseline_top5_rate),
                "high_minus_low_top1_rate": high_top1_rate - low_top1_rate,
                "high_minus_low_top5_rate": high_top5_rate - low_top5_rate,
                "high_mfe_mean": _weighted_mean(
                    group["high_mfe_mean"], group["high_count"]
                ),
                "low_mfe_mean": _weighted_mean(
                    group["low_mfe_mean"], group["low_count"]
                ),
                "high_mfe_median": _weighted_median(
                    group["high_mfe_median"], group["high_count"]
                ),
                "low_mfe_median": _weighted_median(
                    group["low_mfe_median"], group["low_count"]
                ),
                "high_mae_mean": _weighted_mean(
                    group["high_mae_mean"], group["high_count"]
                ),
                "low_mae_mean": _weighted_mean(
                    group["low_mae_mean"], group["low_count"]
                ),
                "high_mae_median": _weighted_median(
                    group["high_mae_median"], group["high_count"]
                ),
                "low_mae_median": _weighted_median(
                    group["low_mae_median"], group["low_count"]
                ),
                "high_state_high_rate": _safe_rate(
                    high_state_high_hits, high_state_count
                ),
                "low_state_high_rate": _safe_rate(low_state_high_hits, low_state_count),
                "high_minus_low_state_high_rate": _safe_rate(
                    high_state_high_hits, high_state_count
                )
                - _safe_rate(low_state_high_hits, low_state_count),
                "median_semantics": "weighted_median_of_annual_group_medians",
            }
        )
    return pd.DataFrame(rows)


def _aggregate_strata(
    annual: pd.DataFrame, years: Sequence[int], period: str
) -> pd.DataFrame:
    frame = annual[annual["year"].isin(years)]
    rows: list[dict[str, Any]] = []
    for (dimension, stratum, horizon), group in frame.groupby(
        ["dimension", "stratum", "horizon"], sort=False
    ):
        valid_count = int(group["valid_count"].sum())
        state_valid_count = int(group["state_valid_count"].sum())
        rows.append(
            {
                "period": period,
                "dimension": dimension,
                "stratum": stratum,
                "horizon": int(horizon),
                "year_count": int(group["year"].nunique()),
                "row_count": int(group["row_count"].sum()),
                "valid_count": valid_count,
                "top1_rate": _safe_rate(group["top1_hits"].sum(), valid_count),
                "top5_rate": _safe_rate(group["top5_hits"].sum(), valid_count),
                "mfe_mean": _weighted_mean(group["mfe_mean"], group["valid_count"]),
                "mfe_median": _weighted_median(
                    group["mfe_median"], group["valid_count"]
                ),
                "mae_mean": _weighted_mean(group["mae_mean"], group["valid_count"]),
                "mae_median": _weighted_median(
                    group["mae_median"], group["valid_count"]
                ),
                "state_low_rate": _safe_rate(
                    group["state_low_hits"].sum(), state_valid_count
                ),
                "state_mid_rate": _safe_rate(
                    group["state_mid_hits"].sum(), state_valid_count
                ),
                "state_high_rate": _safe_rate(
                    group["state_high_hits"].sum(), state_valid_count
                ),
                "median_semantics": "weighted_median_of_annual_group_medians",
            }
        )
    return pd.DataFrame(rows)


def _aggregate_coverage(
    annual: pd.DataFrame, years: Sequence[int], period: str
) -> pd.DataFrame:
    frame = annual[annual["year"].isin(years)]
    rows: list[dict[str, Any]] = []
    for (feature, family, eligibility, relation_axis), group in frame.groupby(
        ["feature", "analytic_family", "eligibility", "relation_axis"], sort=False
    ):
        rows.append(
            {
                "period": period,
                "feature": feature,
                "analytic_family": family,
                "eligibility": eligibility,
                "relation_axis": relation_axis,
                "year_count": int(group["year"].nunique()),
                "row_count": int(group["row_count"].sum()),
                "nonnull_count": int(group["nonnull_count"].sum()),
                "nonnull_rate": _safe_rate(
                    group["nonnull_count"].sum(), group["row_count"].sum()
                ),
                "q01": _weighted_median(group["q01"], group["nonnull_count"]),
                "q50": _weighted_median(group["q50"], group["nonnull_count"]),
                "q99": _weighted_median(group["q99"], group["nonnull_count"]),
                "mean": _weighted_mean(group["mean"], group["nonnull_count"]),
                "near_constant_year_count": int(group["near_constant"].sum()),
                "quantile_semantics": "weighted_median_of_annual_quantiles",
            }
        )
    return pd.DataFrame(rows)


def _aggregate_source_states(
    annual: pd.DataFrame, years: Sequence[int], period: str
) -> pd.DataFrame:
    frame = annual[annual["year"].isin(years)]
    rows: list[dict[str, Any]] = []
    for (block, field, state, horizon), group in frame.groupby(
        ["block", "field", "state", "horizon"], sort=False
    ):
        valid_count = int(group["valid_count"].sum())
        state_valid_count = int(group["state_valid_count"].sum())
        rows.append(
            {
                "period": period,
                "block": block,
                "field": field,
                "state": state,
                "horizon": int(horizon),
                "year_count": int(group["year"].nunique()),
                "row_count": int(group["row_count"].sum()),
                "valid_count": valid_count,
                "top1_rate": _safe_rate(group["top1_hits"].sum(), valid_count),
                "top5_rate": _safe_rate(group["top5_hits"].sum(), valid_count),
                "mfe_mean": _weighted_mean(group["mfe_mean"], group["valid_count"]),
                "mfe_median": _weighted_median(
                    group["mfe_median"], group["valid_count"]
                ),
                "mae_mean": _weighted_mean(group["mae_mean"], group["valid_count"]),
                "mae_median": _weighted_median(
                    group["mae_median"], group["valid_count"]
                ),
                "state_low_rate": _safe_rate(
                    group["state_low_hits"].sum(), state_valid_count
                ),
                "state_mid_rate": _safe_rate(
                    group["state_mid_hits"].sum(), state_valid_count
                ),
                "state_high_rate": _safe_rate(
                    group["state_high_hits"].sum(), state_valid_count
                ),
                "median_semantics": "weighted_median_of_annual_group_medians",
            }
        )
    return pd.DataFrame(rows)


def _aggregate_targets(
    annual: pd.DataFrame, years: Sequence[int], period: str
) -> pd.DataFrame:
    frame = annual[annual["year"].isin(years)]
    rows: list[dict[str, Any]] = []
    for horizon, group in frame.groupby("horizon", sort=False):
        valid_count = int(group["valid_count"].sum())
        if "state_valid_count" in group:
            state_valid_count = int(group["state_valid_count"].sum())
            state_low_count = int(group["state_low_count"].sum())
            state_mid_count = int(group["state_mid_count"].sum())
            state_high_count = int(group["state_high_count"].sum())
        else:
            state_valid_count = valid_count
            state_low_count = int(
                np.rint((group["state_low_rate"] * group["valid_count"]).sum())
            )
            state_mid_count = int(
                np.rint((group["state_mid_rate"] * group["valid_count"]).sum())
            )
            state_high_count = int(
                np.rint((group["state_high_rate"] * group["valid_count"]).sum())
            )
        rows.append(
            {
                "period": period,
                "horizon": int(horizon),
                "year_count": int(group["year"].nunique()),
                "row_count": int(group["row_count"].sum()),
                "valid_count": valid_count,
                "top1_count": int(group["top1_count"].sum()),
                "top5_count": int(group["top5_count"].sum()),
                "top1_rate": _safe_rate(group["top1_count"].sum(), valid_count),
                "top5_rate": _safe_rate(group["top5_count"].sum(), valid_count),
                "mfe_mean": _weighted_mean(group["mfe_mean"], group["valid_count"]),
                "mfe_q01": _weighted_median(group["mfe_q01"], group["valid_count"]),
                "mfe_q05": _weighted_median(group["mfe_q05"], group["valid_count"]),
                "mfe_median": _weighted_median(
                    group["mfe_median"], group["valid_count"]
                ),
                "mfe_q95": _weighted_median(group["mfe_q95"], group["valid_count"]),
                "mfe_q99": _weighted_median(group["mfe_q99"], group["valid_count"]),
                "mae_mean": _weighted_mean(group["mae_mean"], group["valid_count"]),
                "mae_median": _weighted_median(
                    group["mae_median"], group["valid_count"]
                ),
                "state_low_rate": _safe_rate(state_low_count, state_valid_count),
                "state_mid_rate": _safe_rate(state_mid_count, state_valid_count),
                "state_high_rate": _safe_rate(state_high_count, state_valid_count),
                "quantile_semantics": "weighted_median_of_annual_quantiles",
            }
        )
    return pd.DataFrame(rows)


def _feature_stability(annual: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    gate = config["stability_gate"]
    thresholds = {
        "strong_opportunity": float(gate["minimum_top5_rate_difference"]),
        "risk": float(gate["minimum_risk_difference"]),
        "state": float(gate["minimum_state_high_rate_difference"]),
    }
    rows: list[dict[str, Any]] = []
    for (feature, family, eligibility, horizon), group in annual.groupby(
        ["feature", "analytic_family", "eligibility", "horizon"], sort=False
    ):
        effects = {
            "strong_opportunity": group.set_index("year")[
                "high_minus_low_top5_rate"
            ].to_dict(),
            "risk": (
                group.set_index("year")["high_mae_median"]
                - group.set_index("year")["low_mae_median"]
            ).to_dict(),
            "state": group.set_index("year")[
                "high_minus_low_state_high_rate"
            ].to_dict(),
        }
        candidates: list[tuple[int, str, str, dict[int, float]]] = []
        for dimension, annual_effects in effects.items():
            threshold = thresholds[dimension]
            positive = sum(
                1
                for value in annual_effects.values()
                if np.isfinite(value) and value >= threshold
            )
            negative = sum(
                1
                for value in annual_effects.values()
                if np.isfinite(value) and value <= -threshold
            )
            candidates.append((positive, dimension, "high", annual_effects))
            candidates.append((negative, dimension, "low", annual_effects))
        count, dimension, direction, selected = max(
            candidates, key=lambda item: item[0]
        )
        reversal = recent_equal_or_larger_reversal(
            selected, recent_years=tuple(gate["recent_years"])
        )
        stable = count >= int(gate["minimum_same_direction_years"])
        rows.append(
            {
                "feature": feature,
                "analytic_family": family,
                "eligibility": eligibility,
                "horizon": int(horizon),
                "best_relation_dimension": dimension,
                "preferred_feature_tail": direction,
                "same_direction_year_count": int(count),
                "stable_ten_year_relation": bool(stable),
                "recent_equal_or_larger_reversal": bool(reversal),
                "stable_without_recent_reversal": bool(stable and not reversal),
            }
        )
    return pd.DataFrame(rows)


def _redundancy_pairs(
    sample: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    threshold: float,
    minimum_rows: int,
) -> pd.DataFrame:
    feature_names = catalog["feature_name"].astype(str).tolist()
    values = sample[feature_names].replace([np.inf, -np.inf], np.nan)
    usable = values.loc[:, values.notna().sum() >= minimum_rows]
    usable = usable.loc[:, usable.nunique(dropna=True) >= 2]
    correlation = usable.rank(method="average", pct=True).corr(
        method="pearson", min_periods=minimum_rows
    )
    metadata = catalog.set_index("feature_name")["analytic_family"].to_dict()
    matrix = correlation.to_numpy(dtype=np.float64)
    columns = correlation.columns.astype(str).tolist()
    rows: list[dict[str, Any]] = []
    for left in range(len(columns)):
        for right in range(left + 1, len(columns)):
            value = matrix[left, right]
            if np.isfinite(value) and abs(value) >= threshold:
                rows.append(
                    {
                        "feature_left": columns[left],
                        "feature_right": columns[right],
                        "family_left": metadata[columns[left]],
                        "family_right": metadata[columns[right]],
                        "spearman": float(value),
                        "absolute_spearman": float(abs(value)),
                    }
                )
    return (
        pd.DataFrame(rows).sort_values(
            "absolute_spearman", ascending=False, ignore_index=True
        )
        if rows
        else pd.DataFrame(
            columns=[
                "feature_left",
                "feature_right",
                "family_left",
                "family_right",
                "spearman",
                "absolute_spearman",
            ]
        )
    )


def _independent_representatives(
    features: set[str], redundancy: pd.DataFrame
) -> list[str]:
    """Greedily retain stable representatives not joined by a redundancy edge."""

    if not features:
        return []
    neighbors: dict[str, set[str]] = {feature: set() for feature in features}
    for row in redundancy.itertuples(index=False):
        left = str(row.feature_left)
        right = str(row.feature_right)
        if left in features and right in features:
            neighbors[left].add(right)
            neighbors[right].add(left)
    selected: list[str] = []
    for feature in sorted(features, key=lambda name: (len(neighbors[name]), name)):
        if all(feature not in neighbors[existing] for existing in selected):
            selected.append(feature)
    return selected


def _recommendations(
    catalog: pd.DataFrame,
    registry: pd.DataFrame,
    stability: pd.DataFrame,
    redundancy: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    redundant_features = set(redundancy["feature_left"]) | set(
        redundancy["feature_right"]
    )
    diagnostic = registry[registry["eligibility"] == "diagnostic_only"].copy()
    diagnostic["analytic_family"] = [
        _analytic_family(
            name=str(row.feature_name),
            legacy_family="",
            block=str(row.block),
            source_domain=str(row.source_domain),
        )
        for row in diagnostic.itertuples(index=False)
    ]
    rows: list[dict[str, Any]] = []
    gated_families = {
        "research_reports",
        "margin_detail",
        "margin_market",
        "financial_statement_extensions",
    }
    all_families = sorted(
        set(catalog["analytic_family"].astype(str))
        | set(diagnostic["analytic_family"].astype(str))
    )
    gate = dict(config["stability_gate"])
    large_family_minimum = int(gate.get("large_family_minimum_formal_features", 10))
    large_family_representatives = int(
        gate.get("minimum_independent_stable_members_large_family", 2)
    )
    for family in all_families:
        family_catalog = catalog[catalog["analytic_family"] == family]
        family_stability = stability[stability["analytic_family"] == family]
        stable_features = set(
            family_stability.loc[
                family_stability["stable_ten_year_relation"], "feature"
            ].astype(str)
        )
        stable_nonreversed = set(
            family_stability.loc[
                family_stability["stable_without_recent_reversal"], "feature"
            ].astype(str)
        )
        independent_representatives = _independent_representatives(
            stable_nonreversed, redundancy
        )
        minimum_representatives = (
            large_family_representatives
            if len(family_catalog) >= large_family_minimum
            else 1
        )
        family_diagnostic_count = int((diagnostic["analytic_family"] == family).sum())
        availability_gated = bool(
            family in gated_families
            or (
                not family_catalog.empty
                and family_catalog["eligibility"].eq("availability_gated").all()
            )
        )
        classification = classify_family(
            stable_feature_count=len(stable_features),
            stable_nonreversed_feature_count=len(independent_representatives),
            availability_gated=availability_gated,
            formal_eligible_feature_count=len(family_catalog),
            minimum_independent_representatives=minimum_representatives,
        )
        if classification == "first_model_formal_family":
            reason = "pit_valid_and_independent_members_support_first_model_ablation"
        elif classification == "availability_gated_family":
            reason = "structural_source_availability_requires_explicit_gate"
        elif classification == "diagnostic_only":
            reason = "registry_excludes_formal_model_use"
        elif stable_nonreversed:
            reason = "stable_members_do_not_meet_the_independent_representative_gate"
        else:
            reason = "no_member_passed_the_preregistered_ten_year_stability_gate"
        rows.append(
            {
                "analytic_family": family,
                "classification": classification,
                "formal_eligible_feature_count": len(family_catalog),
                "diagnostic_only_feature_count": family_diagnostic_count,
                "stable_feature_count": len(stable_features),
                "stable_nonreversed_feature_count": len(stable_nonreversed),
                "independent_stable_representative_count": len(
                    independent_representatives
                ),
                "minimum_independent_representative_count": minimum_representatives,
                "redundant_feature_count": len(
                    set(family_catalog["feature_name"].astype(str)) & redundant_features
                ),
                "representative_stable_features": ",".join(
                    independent_representatives[:10]
                ),
                "reason": reason,
                "recommendation_semantics": (
                    "candidate_for_first_model_ablation_not_final_inclusion"
                ),
                "final_field_set_frozen": False,
            }
        )
    return pd.DataFrame(rows)


def _drift_summary(coverage: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (feature, family, eligibility), group in coverage.groupby(
        ["feature", "analytic_family", "eligibility"], sort=False
    ):
        ordered = group.sort_values("year")
        nonnull_range = float(
            ordered["nonnull_rate"].max() - ordered["nonnull_rate"].min()
        )
        spread = (ordered["q99"] - ordered["q01"]).replace(0, np.nan)
        typical_spread = (
            float(spread.median()) if spread.notna().any() else float("nan")
        )
        first_median = float(ordered.iloc[0]["q50"])
        last_median = float(ordered.iloc[-1]["q50"])
        normalized_shift = (
            (last_median - first_median) / typical_spread
            if np.isfinite(typical_spread) and typical_spread != 0
            else float("nan")
        )
        rows.append(
            {
                "feature": feature,
                "analytic_family": family,
                "eligibility": eligibility,
                "year_count": int(ordered["year"].nunique()),
                "minimum_nonnull_rate": float(ordered["nonnull_rate"].min()),
                "maximum_nonnull_rate": float(ordered["nonnull_rate"].max()),
                "nonnull_rate_range": nonnull_range,
                "first_year_median": first_median,
                "last_year_median": last_median,
                "median_shift_in_typical_98pct_spread": normalized_shift,
                "near_constant_year_count": int(ordered["near_constant"].sum()),
                "source_or_distribution_drift_flag": bool(
                    nonnull_range >= 0.30
                    or (np.isfinite(normalized_shift) and abs(normalized_shift) >= 2.0)
                ),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_outputs(
    *,
    output_root: Path,
    catalog: pd.DataFrame,
    registry: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    annual_relation = pd.concat(
        [
            pd.read_parquet(
                output_root / "annual" / f"year={year}" / "feature_relation.parquet"
            )
            for year in YEARS
        ],
        ignore_index=True,
    )
    annual_coverage = pd.concat(
        [
            pd.read_parquet(
                output_root
                / "annual"
                / f"year={year}"
                / "feature_coverage_distribution.parquet"
            )
            for year in YEARS
        ],
        ignore_index=True,
    )
    annual_target = pd.concat(
        [
            pd.read_parquet(
                output_root / "annual" / f"year={year}" / "target_summary.parquet"
            )
            for year in YEARS
        ],
        ignore_index=True,
    )
    annual_strata = pd.concat(
        [
            pd.read_parquet(output_root / "annual" / f"year={year}" / "strata.parquet")
            for year in YEARS
        ],
        ignore_index=True,
    )
    annual_source_state = pd.concat(
        [
            pd.read_parquet(
                output_root / "annual" / f"year={year}" / "source_state.parquet"
            )
            for year in YEARS
        ],
        ignore_index=True,
    )
    pool_comparison = pd.concat(
        [
            pd.read_parquet(
                output_root / "annual" / f"year={year}" / "pool_comparison.parquet"
            )
            for year in YEARS
        ],
        ignore_index=True,
    )
    excluded = pd.concat(
        [
            pd.read_parquet(
                output_root / "annual" / f"year={year}" / "excluded_minute_rows.parquet"
            )
            for year in YEARS
        ],
        ignore_index=True,
    )
    _write_frame(annual_relation, output_root / "feature_relation_by_year.parquet")
    _write_frame(annual_coverage, output_root / "coverage_distribution_by_year.parquet")
    _write_frame(annual_target, output_root / "target_summary_by_year.parquet")
    _write_frame(annual_strata, output_root / "strata_by_year.parquet")
    _write_frame(annual_source_state, output_root / "source_state_by_year.parquet")
    _write_frame(pool_comparison, output_root / "pool_bias_by_year.parquet")
    _write_frame(excluded, output_root / "excluded_minute_rows.parquet")

    periods: dict[str, Sequence[int]] = {
        "all_history": YEARS,
        **{
            str(name): tuple(int(year) for year in years)
            for name, years in config["period"]["eras"].items()
        },
        "decision_years_2023_2025": (2023, 2024, 2025),
    }
    period_relation = pd.concat(
        [
            _aggregate_relation(annual_relation, years, name)
            for name, years in periods.items()
        ],
        ignore_index=True,
    )
    period_strata = pd.concat(
        [
            _aggregate_strata(annual_strata, years, name)
            for name, years in periods.items()
        ],
        ignore_index=True,
    )
    period_pool = pd.concat(
        [
            _aggregate_strata(pool_comparison, years, name)
            for name, years in periods.items()
        ],
        ignore_index=True,
    )
    period_coverage = pd.concat(
        [
            _aggregate_coverage(annual_coverage, years, name)
            for name, years in periods.items()
        ],
        ignore_index=True,
    )
    period_source_state = pd.concat(
        [
            _aggregate_source_states(annual_source_state, years, name)
            for name, years in periods.items()
        ],
        ignore_index=True,
    )
    period_target = pd.concat(
        [
            _aggregate_targets(annual_target, years, name)
            for name, years in periods.items()
        ],
        ignore_index=True,
    )
    _write_frame(period_relation, output_root / "feature_relation_by_period.parquet")
    _write_frame(period_strata, output_root / "strata_by_period.parquet")
    _write_frame(period_pool, output_root / "pool_bias_by_period.parquet")
    _write_frame(
        period_coverage, output_root / "coverage_distribution_by_period.parquet"
    )
    _write_frame(period_source_state, output_root / "source_state_by_period.parquet")
    _write_frame(period_target, output_root / "target_summary_by_period.parquet")

    stability = _feature_stability(annual_relation, config)
    samples = pd.concat(
        [
            pd.read_parquet(
                output_root / "annual" / f"year={year}" / "redundancy_sample.parquet"
            )
            for year in YEARS
        ],
        ignore_index=True,
    )
    redundancy = _redundancy_pairs(
        samples,
        catalog,
        threshold=float(config["stability_gate"]["redundancy_absolute_spearman"]),
        minimum_rows=int(config["redundancy"]["minimum_pairwise_rows"]),
    )
    recommendations = _recommendations(catalog, registry, stability, redundancy, config)
    drift = _drift_summary(annual_coverage)
    _write_frame(stability, output_root / "feature_stability.parquet")
    _write_frame(redundancy, output_root / "redundancy_pairs.parquet")
    _write_frame(recommendations, output_root / "family_recommendations.parquet")
    _write_frame(drift, output_root / "source_and_distribution_drift.parquet")

    tail_columns = [
        "period",
        "feature",
        "analytic_family",
        "eligibility",
        "horizon",
        "relation_axis",
        "valid_count",
        "baseline_top1_rate",
        "baseline_top5_rate",
        "high_top1_rate",
        "low_top1_rate",
        "high_top5_rate",
        "low_top5_rate",
        "high_top1_lift",
        "low_top1_lift",
        "high_top5_lift",
        "low_top5_lift",
        "high_minus_low_top1_rate",
        "high_minus_low_top5_rate",
    ]
    risk_columns = [
        "period",
        "feature",
        "analytic_family",
        "eligibility",
        "horizon",
        "relation_axis",
        "high_mfe_median",
        "low_mfe_median",
        "high_mae_median",
        "low_mae_median",
        "high_state_high_rate",
        "low_state_high_rate",
        "high_minus_low_state_high_rate",
    ]
    _write_frame(
        period_relation[tail_columns], output_root / "tail_lift_by_period.parquet"
    )
    _write_frame(
        period_relation[risk_columns], output_root / "risk_state_by_period.parquet"
    )
    return {
        "annual_relation_rows": len(annual_relation),
        "annual_coverage_rows": len(annual_coverage),
        "period_relation_rows": len(period_relation),
        "source_state_rows": len(annual_source_state),
        "excluded_minute_rows": len(excluded),
        "redundancy_sample_rows": len(samples),
        "redundancy_pair_rows": len(redundancy),
        "family_recommendation_rows": len(recommendations),
        "classification_counts": {
            str(key): int(value)
            for key, value in recommendations["classification"].value_counts().items()
        },
    }


def prepare(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    config = _read_json(study_path)
    if config.get("study_id") != STUDY_ID:
        raise DescriptiveAuditError("study_id_mismatch")
    source_manifest_path = source_root / "manifest.json"
    source_manifest = _read_json(source_manifest_path)
    readiness = _read_json(source_root / "readiness.json")
    if source_manifest.get("study_id") != SOURCE_STUDY_ID:
        raise DescriptiveAuditError("source_training_ready_study_mismatch")
    if source_manifest.get("training_performed") or source_manifest.get(
        "feature_set_selected"
    ):
        raise DescriptiveAuditError("source_training_ready_contract_violated")
    if int(source_manifest.get("forbidden_2026_rows", -1)) != 0:
        raise DescriptiveAuditError("source_contains_forbidden_2026_rows")
    if tuple(int(year) for year in config["period"]["years"]) != YEARS:
        raise DescriptiveAuditError("research_year_contract_mismatch")
    if int(config["period"]["burn_in_year"]) != 2010:
        raise DescriptiveAuditError("burn_in_contract_mismatch")
    if int(config["period"]["forbidden_year"]) != 2026:
        raise DescriptiveAuditError("forbidden_year_contract_mismatch")
    if config["training"] != {
        "performed": False,
        "feature_set_selected": False,
        "model_variants_selected": False,
    }:
        raise DescriptiveAuditError("training_prohibition_contract_mismatch")

    label_manifest, labels, flags, states = _label_memmaps()
    if str(label_manifest.get("outcome_cutoff")) != "2025-12-31":
        raise DescriptiveAuditError("label_outcome_cutoff_mismatch")
    base_manifest = _read_json(BASE_FEATURE_MANIFEST)
    base_record = base_manifest["files"]["continuous"]
    base_values = np.memmap(
        Path(base_record["path"]),
        dtype=np.float32,
        mode="r",
        shape=tuple(base_manifest["continuous_shape"]),
    )
    base_catalog = {
        str(item["name"]): int(item["column_index"])
        for item in base_manifest["continuous_catalog"]
    }
    catalog = _feature_catalog(source_manifest, source_root)
    registry = pd.read_parquet(Path(source_manifest["feature_registry"]["path"]))
    input_fingerprint = _stable_hash(
        {
            "builder_version": BUILDER_VERSION,
            "study_sha256": _sha256(study_path),
            "source_manifest_sha256": _sha256(source_manifest_path),
            "source_readiness_sha256": _sha256(source_root / "readiness.json"),
            "base_feature_manifest_sha256": _sha256(BASE_FEATURE_MANIFEST),
            "label_manifest_sha256": _sha256(LABEL_MANIFEST),
        }
    )
    state_path = output_root / "state.json"
    if force and output_root.is_dir():
        raise DescriptiveAuditError("force_requires_an_empty_canonical_output_root")
    state = (
        _read_json(state_path)
        if state_path.is_file()
        else {
            "schema": "seq100_quality_liquidity_descriptive_feature_audit_state/v1",
            "study_id": STUDY_ID,
            "builder_version": BUILDER_VERSION,
            "input_fingerprint": input_fingerprint,
            "status": "running",
            "completed_years": {},
            "started_at": _now(),
            "training_performed": False,
            "feature_set_selected": False,
        }
    )
    if state.get("input_fingerprint") != input_fingerprint:
        raise DescriptiveAuditError("existing_output_input_fingerprint_mismatch")
    _write_frame(catalog, output_root / "numeric_feature_catalog.parquet")

    for year in YEARS:
        annual_manifest = output_root / "annual" / f"year={year}" / "manifest.json"
        if str(year) in state["completed_years"] and annual_manifest.is_file():
            continue
        result = _prepare_year(
            year=year,
            source_manifest=source_manifest,
            catalog=catalog,
            registry=registry,
            output_root=output_root,
            labels=labels,
            flags=flags,
            states=states,
            base_values=base_values,
            base_catalog=base_catalog,
            config=config,
        )
        state["completed_years"][str(year)] = result
        state["updated_at"] = _now()
        _write_json(state_path, state)

    aggregate = _aggregate_outputs(
        output_root=output_root,
        catalog=catalog,
        registry=registry,
        config=config,
    )
    excluded_expected = int(config["pools"]["excluded_minute_support_rows_expected"])
    checks = {
        "all_years_completed": set(state["completed_years"])
        == {str(year) for year in YEARS},
        "formal_years_are_2011_2025": min(YEARS) == 2011 and max(YEARS) == 2025,
        "burn_in_2010_not_read": "2010" not in state["completed_years"],
        "forbidden_2026_not_read": "2026" not in state["completed_years"],
        "numeric_feature_count_is_628": len(catalog) == 628,
        "excluded_minute_rows_match": aggregate["excluded_minute_rows"]
        == excluded_expected,
        "training_not_performed": True,
        "feature_set_not_selected": True,
        "source_training_ready": readiness.get("status")
        == "ready_with_documented_optional_gaps",
    }
    status = "completed" if all(checks.values()) else "failed"
    output_files = [
        "numeric_feature_catalog.parquet",
        "feature_relation_by_year.parquet",
        "feature_relation_by_period.parquet",
        "coverage_distribution_by_year.parquet",
        "coverage_distribution_by_period.parquet",
        "target_summary_by_year.parquet",
        "target_summary_by_period.parquet",
        "strata_by_year.parquet",
        "strata_by_period.parquet",
        "source_state_by_year.parquet",
        "source_state_by_period.parquet",
        "source_and_distribution_drift.parquet",
        "tail_lift_by_period.parquet",
        "risk_state_by_period.parquet",
        "pool_bias_by_year.parquet",
        "pool_bias_by_period.parquet",
        "excluded_minute_rows.parquet",
        "feature_stability.parquet",
        "redundancy_pairs.parquet",
        "family_recommendations.parquet",
    ]
    manifest = {
        "schema": "seq100_quality_liquidity_descriptive_feature_audit_manifest/v1",
        "study_id": STUDY_ID,
        "status": status,
        "builder_version": BUILDER_VERSION,
        "created_at": _now(),
        "input_fingerprint": input_fingerprint,
        "source_training_ready": {
            "study_id": SOURCE_STUDY_ID,
            "manifest": _record(source_manifest_path),
            "readiness": _record(source_root / "readiness.json"),
        },
        "scope": {
            "years": list(YEARS),
            "burn_in_year_read_count": 0,
            "forbidden_2026_read_count": 0,
            "common_support_row_count": int(
                sum(
                    int(source_manifest["row_spine"][str(year)]["row_count"])
                    for year in YEARS
                )
            ),
            "daily_reference_pool_row_count": int(
                sum(
                    int(
                        source_manifest["daily_quality_row_spine"][str(year)][
                            "row_count"
                        ]
                    )
                    for year in YEARS
                )
            ),
            "label_tail_definition": "within_trade_date_cross_section_true_mfe_rank",
            "feature_relation_population": "all_rows_with_valid_feature_and_label_per_year",
            "redundancy_population": f"candidate_id_mod_{config['redundancy']['deterministic_candidate_id_modulus']}_equals_zero",
        },
        "aggregation": aggregate,
        "checks": checks,
        "outputs": {name: _record(output_root / name) for name in output_files},
        "training_performed": False,
        "feature_set_selected": False,
        "model_variants_selected": False,
        "future_model_evaluation_semantics": "retrospective_rolling_oos",
        "config": _record(study_path),
    }
    _write_json(output_root / "manifest.json", manifest)
    state["status"] = status
    state["completed_at"] = _now()
    state["manifest_sha256"] = _sha256(output_root / "manifest.json")
    _write_json(state_path, state)
    if status != "completed":
        raise DescriptiveAuditError("descriptive_audit_checks_failed")
    return manifest


def evaluate(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    manifest = _read_json(output_root / "manifest.json")
    checks = dict(manifest.get("checks", {}))
    checks["manifest_status_completed"] = manifest.get("status") == "completed"
    checks["training_not_performed_manifest"] = not manifest.get(
        "training_performed", True
    )
    checks["feature_set_not_selected_manifest"] = not manifest.get(
        "feature_set_selected", True
    )
    checks["all_output_hashes_match"] = all(
        Path(record["path"]).is_file()
        and _sha256(Path(record["path"])) == record["sha256"]
        for record in manifest.get("outputs", {}).values()
    )
    return {
        "status": "ok" if all(checks.values()) else "failed",
        "study_id": manifest.get("study_id"),
        "checks": checks,
        "aggregation": manifest.get("aggregation", {}),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-path", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.evaluate:
        result = evaluate(args.output_root)
    else:
        result = prepare(
            study_path=args.study_path,
            source_root=args.source_root,
            output_root=args.output_root,
            force=args.force,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("status") in {"ok", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
