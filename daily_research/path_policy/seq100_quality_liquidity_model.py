from __future__ import annotations

"""Prepare and run the canonical rolling quality-liquidity LightGBM studies."""

import argparse
import gc
import hashlib
import itertools
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from quant_data_platform.qdp_v2.manifest import qdp_v2_root
from scipy import stats
from sklearn.metrics import average_precision_score

from daily_research.path_policy import seq100_path_label_learnability as learnability
from daily_research.path_policy import seq100_quality_liquidity_data_prep as data_prep
from daily_research.path_policy import (
    seq100_quality_liquidity_descriptive_feature_audit as descriptive,
)
from daily_research.path_policy import seq100_quality_liquidity_training_ready as ready

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_quality_liquidity_model"
YEARS = tuple(range(2011, 2026))
ROLLING_YEARS = (2023, 2024, 2025)
FORBIDDEN_YEAR = 2026
MAXIMUM_OUTCOME_DATE = "2025-12-31"
HORIZONS = (10, 20)
LABEL_HORIZON_INDEX = {5: 0, 10: 1, 20: 2}
STATE_HORIZON = 10
FLAG_G_VALID = 64
FLAG_MFE_PRE_PEAK_MAE_VALID = 128
FLAG_STATE_ASSIGNED = 256
TARGETS = ("mfe_10", "mfe_20", "risk_10", "risk_20", "state_10")
COMPACT_VARIANT = "compact_core"
MODEL_INPUT_DIR_NAME = "model_inputs"
FORMAL_TASK_SCOPE = "compact_core_only"

RETURN_HORIZONS = (1, 3, 5, 10, 20)
RETURN_TARGETS = tuple(f"return_{horizon}" for horizon in RETURN_HORIZONS)
RETURN_TASK_STAGE = "return_zscore"
RETURN_TASK_COUNT = len(RETURN_HORIZONS) * len(ROLLING_YEARS)
RETURN_WINSOR_LOWER = 0.01
RETURN_WINSOR_UPPER = 0.99
RETURN_MIN_CROSS_SECTION = 20
SHORT_FLAG_G_VALID = 16
DIRECT_RETURN_DIR_NAME = "direct_returns"
CLOSE_D1_DIR_NAME = "close_to_close_d1"
CLOSE_D1_TASK_STAGE = "close_d1_zscore"
CLOSE_D1_TARGET = "return_close_1"
CLOSE_D1_TASK_COUNT = len(ROLLING_YEARS)
FIRST_5M_BAR_TIME = "093500000"
LAST_5M_BAR_TIME = "150000000"
VWAP_OHLC_RELATIVE_TOLERANCE = 0.02
MINIMUM_VWAP_COVERAGE = 0.999

COMPACT_CONSTANT_DROPS = (
    "market_csi300__st_rate",
    "market_sse50__st_rate",
    "is_st_today",
    "is_suspended_today",
    "is_delisted_today",
    "financial_present",
    "income_report_type",
    "income_statement_present",
    "balance_report_type",
    "balance_sheet_present",
    "cashflow_report_type",
    "cash_flow_statement_present",
    "announcement_source_covered",
    "share_capital_missing",
)
COMPACT_REDUNDANCY_DROPS = (
    "income_period_type",
    "balance_period_type",
    "income_revenue_signed_log",
    "income_parent_net_income_signed_log",
    "balance_total_assets_log",
    "balance_total_liabilities_log",
    "cashflow_operating_signed_log",
    "cashflow_free_signed_log",
    "return_1d",
    "technical_boll_mid_bfq",
    "technical_xsii_td2_bfq",
)
COMPACT_COVERAGE_STABILITY_DROPS = (
    "income_ebitda",
    "cashflow_net_profit",
    "income_research_development_expense",
    "income_rd_intensity",
    "income_continuing_net_income",
)
COMPACT_FEATURE_COUNT = 557
COMPACT_FINANCIAL_REPLACEMENTS = {
    "balance_other_receivables": "balance_other_receivables_total",
    "balance_other_payables": "balance_other_payables_total",
    "balance_advances_from_customers": (
        "balance_customer_advances_and_contract_liabilities"
    ),
}
REFRESHED_BASE_FEATURES = (
    "industry_ret1_mean",
    "industry_ret5_mean",
    "industry_breadth_ret1_positive",
    "industry_ret1_dispersion",
    "industry_relative_ret5",
    "industry_relative_ret20",
    "industry_member_count_log",
    "industry_source_age_days",
    "industry_missing",
    "log_total_market_value",
    "log_circulating_market_value",
    "circulating_market_value_ratio",
    "signed_log_pe",
    "signed_log_pb",
    "log_turnover_rate",
    "log_total_share",
    "log_float_share",
    "float_share_ratio",
    "share_source_age_days",
    "valuation_missing",
)

DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_quality_liquidity_model.json"
)
DEFAULT_READY_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_quality_liquidity_training_ready"
)
DEFAULT_AUDIT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_quality_liquidity_descriptive_feature_audit"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_quality_liquidity_model"
)
DEFAULT_SHORT_LABEL_MANIFEST = (
    WORKSPACE_ROOT
    / "tmp/seq100_short_horizon_target_reaudit/attempt_001/labels/short_label_manifest.json"
)


class ModelError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ModelError(f"required_json_missing:{path}")
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
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
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_record(path: Path, **extra: Any) -> dict[str, Any]:
    result = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size": int(path.stat().st_size),
    }
    result.update(extra)
    return result


def _resolve_source_path(path: str | Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = WORKSPACE_ROOT / value
    return value.resolve()


def _load_config(study_path: Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    config = _read_json(study_path)
    if config.get("study_id") != STUDY_ID:
        raise ModelError("study_id_mismatch")
    period = dict(config.get("period", {}) or {})
    if tuple(int(year) for year in period.get("research_years", ())) != YEARS:
        raise ModelError("research_year_contract_mismatch")
    if (
        tuple(int(year) for year in period.get("rolling_prediction_years", ()))
        != ROLLING_YEARS
    ):
        raise ModelError("rolling_year_contract_mismatch")
    if int(period.get("burn_in_year", -1)) != 2010:
        raise ModelError("burn_in_contract_mismatch")
    if (
        int(dict(config.get("source", {}) or {}).get("forbidden_year", -1))
        != FORBIDDEN_YEAR
    ):
        raise ModelError("forbidden_year_contract_mismatch")
    if (
        str(dict(config.get("source", {}) or {}).get("maximum_outcome_date"))
        != MAXIMUM_OUTCOME_DATE
    ):
        raise ModelError("outcome_cutoff_contract_mismatch")
    if not bool(
        dict(config.get("execution", {}) or {}).get("training_performed", False)
    ):
        raise ModelError("model_execution_training_contract_disabled")
    return config


def _source_manifest(ready_root: Path = DEFAULT_READY_ROOT) -> dict[str, Any]:
    manifest = _read_json(ready_root / "manifest.json")
    readiness = _read_json(ready_root / "readiness.json")
    if manifest.get("study_id") != "seq100_quality_liquidity_training_ready":
        raise ModelError("training_ready_study_mismatch")
    if manifest.get("training_performed") or manifest.get("feature_set_selected"):
        raise ModelError("training_ready_already_trained")
    if int(manifest.get("forbidden_2026_rows", -1)) != 0:
        raise ModelError("training_ready_contains_2026_rows")
    if readiness.get("status") not in {"ready", "ready_with_documented_optional_gaps"}:
        raise ModelError("training_ready_not_ready")
    return manifest


def _audit_manifest(audit_root: Path = DEFAULT_AUDIT_ROOT) -> dict[str, Any]:
    manifest = _read_json(audit_root / "manifest.json")
    if manifest.get("study_id") != "seq100_quality_liquidity_descriptive_feature_audit":
        raise ModelError("descriptive_audit_study_mismatch")
    if manifest.get("status") != "completed":
        raise ModelError("descriptive_audit_not_completed")
    if manifest.get("training_performed") or manifest.get("feature_set_selected"):
        raise ModelError("descriptive_audit_training_contract_violated")
    if (
        int(dict(manifest.get("scope", {}) or {}).get("forbidden_2026_read_count", -1))
        != 0
    ):
        raise ModelError("descriptive_audit_contains_2026_reads")
    return manifest


def _catalog_and_registry(
    *, ready_root: Path, audit_root: Path, ready_manifest: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del audit_root
    catalog = descriptive._feature_catalog(ready_manifest, ready_root)
    registry = pd.read_parquet(Path(ready_manifest["feature_registry"]["path"]))
    required = {
        "feature_name",
        "physical_column",
        "block",
        "eligibility",
        "analytic_family",
    }
    if not required.issubset(catalog.columns):
        raise ModelError(
            f"numeric_catalog_columns_missing:{sorted(required - set(catalog.columns))}"
        )
    if len(catalog) != 628 or catalog["feature_name"].duplicated().any():
        raise ModelError("numeric_catalog_count_or_uniqueness_mismatch")
    if bool(catalog["feature_name"].isin(["listing_age_days"]).any()):
        raise ModelError("defective_listing_age_days_in_model_catalog")
    if int(catalog["feature_name"].eq("listing_age_open_days").sum()) != 1:
        raise ModelError("canonical_listing_age_open_days_missing")
    if bool(
        catalog["feature_name"]
        .astype(str)
        .str.contains("qfq|hfq", case=False, regex=True)
        .any()
    ):
        raise ModelError("adjusted_price_field_in_model_catalog")
    if bool(
        catalog["physical_column"]
        .astype(str)
        .isin(["net_mf_vol", "net_mf_amount"])
        .any()
    ):
        raise ModelError("unreconciled_moneyflow_field_in_model_catalog")
    return catalog, registry


def _base_manifest() -> dict[str, Any]:
    return _read_json(data_prep.BASE_FEATURE_MANIFEST)


def _label_manifest() -> dict[str, Any]:
    return _read_json(data_prep.LABEL_MANIFEST)


def _feature_contract(
    *, catalog: pd.DataFrame, registry: pd.DataFrame, base_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    del registry
    base_catalog = pd.DataFrame(base_manifest["continuous_catalog"])
    base_names = base_catalog["name"].astype(str).tolist()
    if ready.LEGACY_LISTING_AGE_FIELD not in base_names:
        raise ModelError("base_manifest_listing_age_field_missing")
    base_index = dict(
        zip(
            base_catalog["name"].astype(str),
            base_catalog["column_index"].astype(int),
            strict=True,
        )
    )

    def names_where(mask: pd.Series) -> list[str]:
        return catalog.loc[mask, "feature_name"].astype(str).tolist()

    legacy = names_where(
        catalog["eligibility"].eq("formal_existing")
        & ~catalog["analytic_family"].eq("research_reports")
    )
    technical = names_where(
        catalog["block"].eq("tushare_technical_candidates")
        & catalog["eligibility"].eq("formal_candidate")
    )
    moneyflow = names_where(
        catalog["block"].eq("traditional_moneyflow_features")
        & catalog["eligibility"].eq("formal_candidate")
    )
    full_core = [*legacy, *technical, *moneyflow]
    if len(full_core) != 587 or len(set(full_core)) != 587:
        raise ModelError("full_core_source_contract_mismatch")
    catalog_by_name = catalog.set_index("feature_name", drop=False)
    drops = (
        set(COMPACT_CONSTANT_DROPS)
        | set(COMPACT_REDUNDANCY_DROPS)
        | set(COMPACT_COVERAGE_STABILITY_DROPS)
    )
    if not drops.issubset(full_core):
        raise ModelError(f"compact_drop_missing:{sorted(drops - set(full_core))}")
    if not set(COMPACT_FINANCIAL_REPLACEMENTS).issubset(full_core):
        raise ModelError("compact_replacement_source_missing")
    replacement_names = set(COMPACT_FINANCIAL_REPLACEMENTS.values())
    if not replacement_names.issubset(catalog_by_name.index):
        raise ModelError(
            f"compact_replacement_target_missing:{sorted(replacement_names - set(catalog_by_name.index))}"
        )

    compact_names: list[str] = []
    compact_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, str]] = []
    for name in full_core:
        if name in drops:
            if name in COMPACT_CONSTANT_DROPS:
                reason = "constant_on_frozen_2011_2025_support"
            elif name in COMPACT_REDUNDANCY_DROPS:
                reason = "conservative_semantic_redundancy"
            else:
                reason = "insufficient_cross_year_coverage_stability"
            decisions.append({"feature_name": name, "action": "drop", "reason": reason})
            continue
        selected_name = COMPACT_FINANCIAL_REPLACEMENTS.get(name, name)
        row = dict(catalog_by_name.loc[selected_name])
        compact_names.append(selected_name)
        compact_rows.append(row)
        if selected_name != name:
            decisions.append(
                {
                    "feature_name": name,
                    "action": "replace",
                    "replacement": selected_name,
                    "reason": "complete_statement_semantics",
                }
            )
    compact_catalog = pd.DataFrame(compact_rows).reset_index(drop=True)
    if (
        len(compact_names) != COMPACT_FEATURE_COUNT
        or len(set(compact_names)) != COMPACT_FEATURE_COUNT
    ):
        raise ModelError(
            f"compact_core_count_or_uniqueness_mismatch:{len(compact_names)}:{len(set(compact_names))}"
        )
    if set(REFRESHED_BASE_FEATURES) - set(compact_names):
        raise ModelError(
            f"refreshed_base_feature_not_selected:{sorted(set(REFRESHED_BASE_FEATURES) - set(compact_names))}"
        )
    return {
        "base_index": base_index,
        "groups": {COMPACT_VARIANT: compact_names},
        "catalog": compact_catalog,
        "decisions": decisions,
        "source_full_core_count": len(full_core),
        "compact_feature_count": len(compact_names),
    }


def _row_index(ready_manifest: Mapping[str, Any]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in YEARS:
        path = Path(ready_manifest["row_spine"][str(year)]["path"])
        frame = pd.read_parquet(
            path,
            columns=["candidate_id", "date_idx", "trade_date", "symbol", "security_id"],
        )
        if frame.empty or not frame["candidate_id"].is_unique:
            raise ModelError(f"row_index_year_empty_or_duplicate:{year}")
        frame["trade_date"] = frame["trade_date"].astype(str)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if len(result) != int(ready_manifest["common_support"]["row_count"]):
        raise ModelError("row_index_count_mismatch")
    if not result["candidate_id"].is_unique:
        raise ModelError("row_index_candidate_id_duplicate")
    candidate_ids = result["candidate_id"].to_numpy(dtype=np.int64)
    if not bool((candidate_ids[1:] > candidate_ids[:-1]).all()):
        raise ModelError("row_index_candidate_id_not_strictly_increasing")
    date_idx = result["date_idx"].to_numpy(dtype=np.int32)
    if not bool((date_idx[1:] >= date_idx[:-1]).all()):
        raise ModelError("row_index_date_not_ordered")
    if bool(result["trade_date"].str.startswith("2026-").any()):
        raise ModelError("row_index_contains_2026")
    return result[["candidate_id", "date_idx", "trade_date", "symbol", "security_id"]]


def _source_paths(ready_manifest: Mapping[str, Any], year: int) -> dict[str, Path]:
    paths = descriptive._physical_sources(ready_manifest, year)
    for block in (
        "membership_context",
        "tushare_technical_candidates",
        "margin_features",
        "traditional_moneyflow_features",
        "financial_statement_extensions",
    ):
        paths[block] = Path(
            ready_manifest["blocks"][block]["partitions"][str(year)]["path"]
        )
    return paths


def _align_source_frame(frame: pd.DataFrame, expected_ids: np.ndarray) -> pd.DataFrame:
    ids = frame["candidate_id"].to_numpy(dtype=np.int64)
    if np.array_equal(ids, expected_ids):
        return frame.reset_index(drop=True)
    if not frame["candidate_id"].is_unique:
        raise ModelError("source_candidate_id_duplicate")
    if not bool(np.isin(expected_ids, ids, assume_unique=True).all()):
        raise ModelError("source_candidate_id_missing")
    indexed = frame.set_index("candidate_id")
    try:
        aligned = indexed.reindex(expected_ids)
    except Exception as exc:  # pragma: no cover - pandas error wording varies
        raise ModelError("source_candidate_id_alignment_failed") from exc
    aligned.insert(0, "candidate_id", expected_ids)
    return aligned.reset_index(drop=True)


def _as_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    values = []
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        values.append(series.to_numpy(dtype=np.float32, na_value=np.nan))
    return np.column_stack(values).astype(np.float32, copy=False)


def _safe_divide_array(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    left = np.asarray(numerator, dtype=np.float64)
    right = np.asarray(denominator, dtype=np.float64)
    output = np.full(left.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(left) & np.isfinite(right) & (np.abs(right) > 1e-12)
    output[valid] = (left[valid] / right[valid]).astype(np.float32)
    return output


def _signed_log1p_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return (np.sign(array) * np.log1p(np.abs(array))).astype(np.float32)


def _current_size_source_frame(
    *,
    ready_manifest: Mapping[str, Any],
    year: int,
    expected_ids: np.ndarray,
) -> pd.DataFrame:
    dataset_ids = dict(ready_manifest["qdp_dataset_ids"])
    spine_path = Path(ready_manifest["row_spine"][str(year)]["path"])
    valuation_scan = data_prep._scan(
        ready._active_paths(WORKSPACE_ROOT, "valuation", dataset_ids)
    )
    share_scan = data_prep._scan(
        ready._active_paths(WORKSPACE_ROOT, "share_capital", dataset_ids)
    )
    industry_scan = data_prep._scan(
        ready._active_paths(WORKSPACE_ROOT, "industry_concept", dataset_ids)
    )
    spine = str(spine_path).replace("'", "''")
    sql = f"""
      SELECT s.candidate_id,s.trade_date,
             try_cast(v.total_mv AS DOUBLE) AS total_mv,
             try_cast(v.circ_mv AS DOUBLE) AS circ_mv,
             try_cast(v.pe AS DOUBLE) AS pe,
             try_cast(v.pb AS DOUBLE) AS pb,
             try_cast(v.turnover_rate AS DOUBLE) AS turnover_rate,
             CASE WHEN sc.total_share_source_date<>''
                        AND sc.total_share_source_date<=s.trade_date
                  THEN try_cast(sc.total_share AS DOUBLE) END AS total_share,
             CASE WHEN sc.float_share_source_date<>''
                        AND sc.float_share_source_date<=s.trade_date
                  THEN try_cast(sc.float_share AS DOUBLE) END AS float_share,
             CASE WHEN sc.float_share_source_date<>''
                        AND sc.float_share_source_date<=s.trade_date
                  THEN datediff('day',try_cast(sc.float_share_source_date AS DATE),
                                      try_cast(s.trade_date AS DATE)) END
                  AS share_source_age_days,
             CASE WHEN i.industry_source_date<>''
                        AND i.industry_source_date<=s.trade_date
                  THEN datediff('day',try_cast(i.industry_source_date AS DATE),
                                      try_cast(s.trade_date AS DATE)) END
                  AS industry_source_age_days
      FROM read_parquet('{spine}') s
      LEFT JOIN {valuation_scan} v
        ON v.symbol=s.symbol AND v.trade_date=s.trade_date
      LEFT JOIN {share_scan} sc
        ON sc.symbol=s.symbol AND sc.trade_date=s.trade_date
      LEFT JOIN {industry_scan} i
        ON i.symbol=s.symbol AND i.trade_date=s.trade_date
      ORDER BY s.candidate_id
    """
    with duckdb.connect() as connection:
        frame = connection.execute(sql).fetchdf()
    return _align_source_frame(frame, expected_ids)


def _refreshed_industry_frame(
    *,
    ready_manifest: Mapping[str, Any],
    year: int,
    expected_ids: np.ndarray,
    base: np.memmap,
    base_index: Mapping[str, int],
) -> pd.DataFrame:
    diagnostics_path = descriptive._diagnostics_path(ready_manifest, year)
    diagnostics = pd.read_parquet(
        diagnostics_path,
        columns=["candidate_id", "date_idx", "industry"],
    )
    candidate_ids = diagnostics["candidate_id"].to_numpy(dtype=np.int64)
    return_names = ("return_1d", "return_5d", "return_20d")
    return_columns = np.asarray(
        [int(base_index[name]) for name in return_names], dtype=np.int32
    )
    returns = np.asarray(base[np.ix_(candidate_ids, return_columns)], dtype=np.float32)
    diagnostics["ret1"] = returns[:, 0]
    diagnostics["ret5"] = returns[:, 1]
    diagnostics["ret20"] = returns[:, 2]
    normalized = diagnostics["industry"].astype("string").str.strip()
    known = normalized.notna() & ~normalized.str.lower().isin(
        {"", "unknown", "unavailable", "unclassified", "nan", "none"}
    )
    work = diagnostics.loc[known].copy()
    work["industry"] = normalized.loc[known]
    ret1 = work["ret1"].to_numpy(dtype=np.float64)
    work["ret1_square"] = np.square(ret1)
    work["ret1_positive"] = np.where(
        np.isfinite(ret1), (ret1 > 0.0).astype(np.float64), np.nan
    )
    grouped = work.groupby(["date_idx", "industry"], sort=False, observed=True)
    aggregates = grouped.agg(
        industry_ret1_mean=("ret1", "mean"),
        industry_ret5_mean=("ret5", "mean"),
        industry_ret20_mean=("ret20", "mean"),
        industry_ret1_second_moment=("ret1_square", "mean"),
        industry_breadth_ret1_positive=("ret1_positive", "mean"),
        industry_member_count=("candidate_id", "size"),
    ).reset_index()
    variance = aggregates["industry_ret1_second_moment"].to_numpy(
        dtype=np.float64
    ) - np.square(aggregates["industry_ret1_mean"].to_numpy(dtype=np.float64))
    aggregates["industry_ret1_dispersion"] = np.sqrt(np.maximum(variance, 0.0))
    aggregates["industry_member_count_log"] = np.log1p(
        aggregates["industry_member_count"].to_numpy(dtype=np.float64)
    )

    selected = _align_source_frame(
        diagnostics[["candidate_id", "date_idx", "industry", "ret5", "ret20"]],
        expected_ids,
    )
    selected["industry"] = selected["industry"].astype("string").str.strip()
    selected = selected.merge(
        aggregates,
        on=["date_idx", "industry"],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    if not np.array_equal(
        selected["candidate_id"].to_numpy(dtype=np.int64), expected_ids
    ):
        raise ModelError(f"refreshed_industry_alignment_failed:{year}")
    selected["industry_relative_ret5"] = (
        selected["ret5"] - selected["industry_ret5_mean"]
    )
    selected["industry_relative_ret20"] = (
        selected["ret20"] - selected["industry_ret20_mean"]
    )
    selected["industry_missing"] = (
        selected["industry"].isna()
        | selected["industry"]
        .str.lower()
        .isin({"", "unknown", "unavailable", "unclassified", "nan", "none"})
    ).astype(np.float32)
    return selected[
        [
            "candidate_id",
            "industry_ret1_mean",
            "industry_ret5_mean",
            "industry_breadth_ret1_positive",
            "industry_ret1_dispersion",
            "industry_relative_ret5",
            "industry_relative_ret20",
            "industry_member_count_log",
            "industry_missing",
        ]
    ]


def _refreshed_base_frame(
    *,
    ready_manifest: Mapping[str, Any],
    year: int,
    expected_ids: np.ndarray,
    base: np.memmap,
    base_index: Mapping[str, int],
) -> pd.DataFrame:
    source = _current_size_source_frame(
        ready_manifest=ready_manifest,
        year=year,
        expected_ids=expected_ids,
    )
    total_mv = pd.to_numeric(source["total_mv"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    circ_mv = pd.to_numeric(source["circ_mv"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    pe = pd.to_numeric(source["pe"], errors="coerce").to_numpy(dtype=np.float64)
    pb = pd.to_numeric(source["pb"], errors="coerce").to_numpy(dtype=np.float64)
    turnover = pd.to_numeric(source["turnover_rate"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    total_share = pd.to_numeric(source["total_share"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    float_share = pd.to_numeric(source["float_share"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    refreshed = pd.DataFrame(
        {
            "candidate_id": expected_ids,
            "log_total_market_value": np.log1p(np.maximum(total_mv, 0.0)),
            "log_circulating_market_value": np.log1p(np.maximum(circ_mv, 0.0)),
            "circulating_market_value_ratio": _safe_divide_array(circ_mv, total_mv),
            "signed_log_pe": _signed_log1p_array(pe),
            "signed_log_pb": _signed_log1p_array(pb),
            "log_turnover_rate": np.log1p(np.maximum(turnover, 0.0)),
            "log_total_share": np.log1p(np.maximum(total_share, 0.0)),
            "log_float_share": np.log1p(np.maximum(float_share, 0.0)),
            "float_share_ratio": _safe_divide_array(float_share, total_share),
            "share_source_age_days": pd.to_numeric(
                source["share_source_age_days"], errors="coerce"
            ),
            "industry_source_age_days": pd.to_numeric(
                source["industry_source_age_days"], errors="coerce"
            ),
            "valuation_missing": ~(
                np.isfinite(total_mv)
                & np.isfinite(circ_mv)
                & np.isfinite(pe)
                & np.isfinite(pb)
            ),
        }
    )
    industry = _refreshed_industry_frame(
        ready_manifest=ready_manifest,
        year=year,
        expected_ids=expected_ids,
        base=base,
        base_index=base_index,
    )
    refreshed = refreshed.merge(
        industry,
        on="candidate_id",
        how="left",
        sort=False,
        validate="one_to_one",
    )
    if set(REFRESHED_BASE_FEATURES) - set(refreshed.columns):
        raise ModelError(f"refreshed_base_columns_missing:{year}")
    return refreshed[["candidate_id", *REFRESHED_BASE_FEATURES]]


def _write_feature_storage(
    *,
    ready_manifest: Mapping[str, Any],
    row_index: pd.DataFrame,
    contract: Mapping[str, Any],
    input_dir: Path,
) -> dict[str, Any]:
    row_count = len(row_index)
    compact_names = list(contract["groups"][COMPACT_VARIANT])
    compact_index = {name: index for index, name in enumerate(compact_names)}
    catalog = pd.DataFrame(contract["catalog"])
    compact_path = input_dir / "compact_core.float32.dat"
    compact_partial = input_dir / "compact_core.float32.dat.partial"
    compact_mm = np.memmap(
        compact_partial,
        dtype=np.float32,
        mode="w+",
        shape=(row_count, len(compact_names)),
    )
    base_manifest = _base_manifest()
    base_record = base_manifest["files"]["continuous"]
    base = np.memmap(
        Path(base_record["path"]),
        dtype=np.float32,
        mode="r",
        shape=tuple(int(value) for value in base_manifest["continuous_shape"]),
    )
    base_index = dict(contract["base_index"])
    refreshed_names = set(REFRESHED_BASE_FEATURES)
    base_rows = catalog[catalog["block"].eq("existing_seq100_base")]
    direct_base_names = [
        name
        for name in base_rows["feature_name"].astype(str)
        if name not in refreshed_names
    ]
    direct_source_columns = np.asarray(
        [int(base_index[name]) for name in direct_base_names], dtype=np.int32
    )
    direct_target_columns = np.asarray(
        [int(compact_index[name]) for name in direct_base_names], dtype=np.int32
    )
    block_order = [
        "minute",
        "fundamental",
        "event",
        "membership_context",
        "tushare_technical_candidates",
        "traditional_moneyflow_features",
        "financial_statement_extensions",
    ]
    yearly_profiles: dict[str, Any] = {}
    for year in YEARS:
        year_rows = row_index["trade_date"].str.startswith(f"{year}-").to_numpy()
        positions = np.flatnonzero(year_rows)
        if not positions.size or not np.array_equal(
            positions, np.arange(positions[0], positions[-1] + 1, dtype=np.int64)
        ):
            raise ModelError(f"model_input_year_rows_not_contiguous:{year}")
        expected_ids = row_index.loc[positions, "candidate_id"].to_numpy(dtype=np.int64)
        for start in range(0, len(positions), 65_536):
            stop = min(start + 65_536, len(positions))
            source_ids = expected_ids[start:stop]
            values = np.asarray(
                base[np.ix_(source_ids, direct_source_columns)], dtype=np.float32
            )
            target_rows = positions[start:stop]
            compact_mm[np.ix_(target_rows, direct_target_columns)] = values
        refreshed = _refreshed_base_frame(
            ready_manifest=ready_manifest,
            year=year,
            expected_ids=expected_ids,
            base=base,
            base_index=base_index,
        )
        for name in REFRESHED_BASE_FEATURES:
            compact_mm[positions, compact_index[name]] = pd.to_numeric(
                refreshed[name], errors="coerce"
            ).to_numpy(dtype=np.float32, na_value=np.nan)
        paths = _source_paths(ready_manifest, year)
        for block in block_order:
            block_features = catalog[catalog["block"].eq(block)]
            if block_features.empty:
                continue
            path = paths.get(block)
            if path is None:
                raise ModelError(f"source_block_path_missing:{year}:{block}")
            physical = block_features["physical_column"].astype(str).tolist()
            columns = list(dict.fromkeys(["candidate_id", *physical]))
            frame = _align_source_frame(
                pd.read_parquet(path, columns=columns), expected_ids
            )
            numeric = _as_numeric(frame, physical)
            for offset, name in enumerate(block_features["feature_name"].astype(str)):
                compact_mm[positions, compact_index[name]] = numeric[:, offset]
            del frame
        compact_mm.flush()
        yearly_profiles[str(year)] = {
            "row_count": len(positions),
            "refreshed_base_missing": {
                name: int(pd.to_numeric(refreshed[name], errors="coerce").isna().sum())
                for name in REFRESHED_BASE_FEATURES
            },
            "industry_missing_nonzero_count": int(
                (
                    pd.to_numeric(refreshed["industry_missing"], errors="coerce")
                    .fillna(1)
                    .ne(0)
                ).sum()
            ),
        }
        del refreshed
    sample_rows = np.asarray([0, row_count // 2, row_count - 1], dtype=np.int64)
    sample_hash = _stable_hash(compact_mm[sample_rows, :].astype(np.float32).tolist())
    del compact_mm, base
    gc.collect()
    return {
        "compact": {
            "path": str(compact_partial.resolve()),
            "final_path": str(compact_path.resolve()),
            "dtype": "float32",
            "layout": "row_major",
            "shape": [row_count, len(compact_names)],
            "sha256": _sha256(compact_partial),
            "sample_rows": sample_rows.tolist(),
            "sample_hash": sample_hash,
        },
        "yearly_profiles": yearly_profiles,
    }


def _feature_records(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    catalog = pd.DataFrame(contract["catalog"])
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(catalog.to_dict("records")):
        name = str(row["feature_name"])
        rows.append(
            {
                "feature_name": name,
                "storage": "compact",
                "column_index": index,
                "block": str(row["block"]),
                "analytic_family": str(row["analytic_family"]),
                "eligibility": str(row["eligibility"]),
                "physical_column": str(row["physical_column"]),
            }
        )
    return rows


def _model_input_fingerprint(
    *,
    config_path: Path,
    ready_root: Path,
    audit_root: Path,
    contract: Mapping[str, Any],
) -> str:
    del audit_root
    return _stable_hash(
        {
            "study_id": STUDY_ID,
            "config_sha256": _sha256(config_path),
            "ready_manifest_sha256": _sha256(ready_root / "manifest.json"),
            "ready_readiness_sha256": _sha256(ready_root / "readiness.json"),
            "base_feature_manifest_sha256": _sha256(data_prep.BASE_FEATURE_MANIFEST),
            "label_manifest_sha256": _sha256(data_prep.LABEL_MANIFEST),
            "numeric_feature_names": pd.DataFrame(contract["catalog"])["feature_name"]
            .astype(str)
            .tolist(),
            "groups": contract["groups"],
        }
    )


def _verify_model_input_files(
    manifest: Mapping[str, Any], *, full_hash: bool
) -> dict[str, Any]:
    row_record = dict(manifest["row_index"])
    row_path = Path(row_record["path"])
    if not row_path.is_file() or (
        full_hash and _sha256(row_path) != row_record["sha256"]
    ):
        raise ModelError("model_input_row_index_hash_mismatch")
    if int(row_path.stat().st_size) != int(row_record["size"]):
        raise ModelError("model_input_row_index_size_mismatch")
    row_index = pd.read_parquet(row_path)
    expected_columns = [
        "candidate_id",
        "date_idx",
        "trade_date",
        "symbol",
        "security_id",
    ]
    if list(row_index.columns) != expected_columns:
        raise ModelError("model_input_row_index_schema_mismatch")
    if (
        len(row_index) != int(manifest["row_count"])
        or not row_index["candidate_id"].is_unique
    ):
        raise ModelError("model_input_row_index_count_or_key_mismatch")
    record = dict(manifest["storage"]["compact"])
    path = Path(record["path"])
    if not path.is_file() or (full_hash and _sha256(path) != record["sha256"]):
        raise ModelError("model_input_compact_hash_mismatch")
    expected_size = int(np.prod(record["shape"])) * np.dtype(record["dtype"]).itemsize
    if int(path.stat().st_size) != expected_size:
        raise ModelError("model_input_compact_size_mismatch")
    values = np.memmap(
        path,
        mode="r",
        dtype=np.dtype(record["dtype"]),
        shape=tuple(int(value) for value in record["shape"]),
    )
    sample_rows = np.asarray(record["sample_rows"], dtype=np.int64)
    sample_hash = _stable_hash(values[sample_rows, :].tolist())
    del values
    if sample_hash != record["sample_hash"]:
        raise ModelError("model_input_compact_sample_hash_mismatch")
    contract_record = dict(manifest["compact_feature_contract"])
    contract_path = Path(contract_record["path"])
    if not contract_path.is_file() or (
        full_hash and _sha256(contract_path) != contract_record["sha256"]
    ):
        raise ModelError("model_input_feature_contract_hash_mismatch")
    contract = pd.read_parquet(contract_path)
    if (
        len(contract) != COMPACT_FEATURE_COUNT
        or contract["feature_name"].duplicated().any()
        or contract["feature_name"].astype(str).tolist()
        != list(manifest["feature_groups"][COMPACT_VARIANT])
    ):
        raise ModelError("model_input_feature_contract_content_mismatch")
    return {"row_index": row_index}


def prepare(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    ready_root: Path = DEFAULT_READY_ROOT,
    audit_root: Path = DEFAULT_AUDIT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    _load_config(study_path)
    ready_manifest = _source_manifest(ready_root)
    catalog, registry = _catalog_and_registry(
        ready_root=ready_root, audit_root=audit_root, ready_manifest=ready_manifest
    )
    base_manifest = _base_manifest()
    label_manifest = _label_manifest()
    if str(label_manifest.get("outcome_cutoff")) != MAXIMUM_OUTCOME_DATE:
        raise ModelError("label_manifest_cutoff_mismatch")
    contract = _feature_contract(
        catalog=catalog, registry=registry, base_manifest=base_manifest
    )
    fingerprint = _model_input_fingerprint(
        config_path=study_path,
        ready_root=ready_root,
        audit_root=audit_root,
        contract=contract,
    )
    input_dir = ready_root / MODEL_INPUT_DIR_NAME
    manifest_path = input_dir / "manifest.json"
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        if existing.get("input_fingerprint") == fingerprint:
            _verify_model_input_files(existing, full_hash=False)
            return existing
    input_dir.mkdir(parents=True, exist_ok=True)
    row_index = _row_index(ready_manifest)
    row_partial = input_dir / "row_index.parquet.partial"
    row_path = input_dir / "row_index.parquet"
    contract_partial = input_dir / "compact_feature_contract.parquet.partial"
    contract_path = input_dir / "compact_feature_contract.parquet"
    _write_parquet(row_index, row_partial)
    _write_parquet(pd.DataFrame(contract["catalog"]), contract_partial)
    storage = _write_feature_storage(
        ready_manifest=ready_manifest,
        row_index=row_index,
        contract=contract,
        input_dir=input_dir,
    )
    label_record = {
        "path": str(data_prep.LABEL_MANIFEST.resolve()),
        "sha256": _sha256(data_prep.LABEL_MANIFEST),
        "candidate_count": int(label_manifest["candidate_count"]),
    }
    features = _feature_records(contract)
    manifest = {
        "schema": "seq100_quality_liquidity_model_inputs/1",
        "status": "completed",
        "study_id": STUDY_ID,
        "created_at": _now(),
        "input_fingerprint": fingerprint,
        "row_count": len(row_index),
        "row_index": _file_record(
            row_partial,
            columns=list(row_index.columns),
            row_count=len(row_index),
        ),
        "label_manifest": label_record,
        "storage": storage,
        "features": features,
        "feature_groups": contract["groups"],
        "compact_feature_contract": _file_record(
            contract_partial,
            row_count=len(contract["catalog"]),
            decisions=contract["decisions"],
            source_full_core_count=contract["source_full_core_count"],
        ),
        "build_sources": {
            "legacy_base_feature_manifest": {
                "path": str(data_prep.BASE_FEATURE_MANIFEST.resolve()),
                "sha256": _sha256(data_prep.BASE_FEATURE_MANIFEST),
                "use": "copy_unaffected_columns_and_recompute_repaired_columns",
            }
        },
        "source": {
            "training_ready_manifest": _file_record(ready_root / "manifest.json"),
            "maximum_outcome_date": MAXIMUM_OUTCOME_DATE,
            "forbidden_year": FORBIDDEN_YEAR,
        },
        "training_performed": False,
        "feature_set_selected": False,
    }
    _verify_model_input_files(manifest, full_hash=False)
    os.replace(row_partial, row_path)
    manifest["row_index"]["path"] = str(row_path.resolve())
    os.replace(contract_partial, contract_path)
    manifest["compact_feature_contract"]["path"] = str(contract_path.resolve())
    record = manifest["storage"]["compact"]
    os.replace(Path(record["path"]), Path(record["final_path"]))
    record["path"] = record.pop("final_path")
    _write_json(manifest_path, manifest)
    return manifest


def _winsorized_zscore_by_date(
    *,
    values: np.ndarray,
    date_idx: np.ndarray,
    valid: np.ndarray,
    lower: float = RETURN_WINSOR_LOWER,
    upper: float = RETURN_WINSOR_UPPER,
    minimum_count: int = RETURN_MIN_CROSS_SECTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a target within each signal-date cross section."""

    raw = np.asarray(values, dtype=np.float64)
    dates = np.asarray(date_idx, dtype=np.int32)
    source_valid = np.asarray(valid, dtype=bool) & np.isfinite(raw)
    if raw.ndim != 1 or dates.shape != raw.shape or source_valid.shape != raw.shape:
        raise ModelError("return_normalization_shape_mismatch")
    if not 0.0 <= float(lower) < float(upper) <= 1.0:
        raise ModelError("return_winsor_quantiles_invalid")
    if int(minimum_count) < 3:
        raise ModelError("return_minimum_cross_section_invalid")
    boundaries = _date_boundaries(dates)
    normalized = np.full(len(raw), np.nan, dtype=np.float32)
    normalized_valid = np.zeros(len(raw), dtype=bool)
    for left, right in itertools.pairwise(boundaries):
        local_valid = source_valid[left:right]
        positions = np.flatnonzero(local_valid)
        if len(positions) < int(minimum_count):
            continue
        local = raw[left:right][positions]
        low, high = np.quantile(local, [float(lower), float(upper)], method="nearest")
        clipped = np.clip(local, low, high)
        mean = float(clipped.mean())
        scale = float(clipped.std(ddof=0))
        if not np.isfinite(scale) or scale <= 1.0e-12:
            continue
        global_positions = left + positions
        normalized[global_positions] = ((clipped - mean) / scale).astype(
            np.float32, copy=False
        )
        normalized_valid[global_positions] = True
    return normalized, normalized_valid


class ModelInputs:
    """Read the compact model inputs while retaining candidate-id label semantics."""

    def __init__(self, manifest: Mapping[str, Any]) -> None:
        self.manifest = dict(manifest)
        self.row_count = int(manifest["row_count"])
        self.row_index = pd.read_parquet(Path(manifest["row_index"]["path"]))
        self.candidate_ids = self.row_index["candidate_id"].to_numpy(dtype=np.int64)
        self.date_idx = self.row_index["date_idx"].to_numpy(dtype=np.int32)
        self.trade_date = np.asarray(
            self.row_index["trade_date"].astype(str).to_numpy(), dtype=str
        )
        self.years = (
            self.row_index["trade_date"].str.slice(0, 4).astype(np.int16).to_numpy()
        )
        compact_record = manifest["storage"]["compact"]
        self.compact = np.memmap(
            Path(compact_record["path"]),
            dtype=np.float32,
            mode="r",
            shape=tuple(int(value) for value in compact_record["shape"]),
        )
        label_manifest = _read_json(Path(manifest["label_manifest"]["path"]))
        self.label_manifest = label_manifest
        label_files = label_manifest["files"]
        candidate_count = int(label_manifest["candidate_count"])
        self.labels = np.memmap(
            Path(label_files["candidate_labels"]["path"]),
            dtype=np.float32,
            mode="r",
            shape=tuple(
                int(value) for value in label_files["candidate_labels"]["shape"]
            ),
        )
        self.flags = np.memmap(
            Path(label_files["label_flags"]["path"]),
            dtype=np.uint16,
            mode="r",
            shape=tuple(int(value) for value in label_files["label_flags"]["shape"]),
        )
        self.states = np.memmap(
            Path(label_files["state_labels"]["path"]),
            dtype=np.int8,
            mode="r",
            shape=tuple(int(value) for value in label_files["state_labels"]["shape"]),
        )
        if candidate_count <= int(self.candidate_ids.max()):
            raise ModelError("model_input_candidate_id_exceeds_label_memmap")
        pack_manifest = _read_json(Path(label_manifest["source"]["pack_manifest"]))
        self.date_values = np.asarray(pack_manifest["date_values"], dtype=str)
        self.feature_records = [dict(item) for item in manifest["features"]]
        self.feature_map = {
            str(item["feature_name"]): item for item in self.feature_records
        }
        self.feature_groups = {
            str(name): list(values)
            for name, values in dict(manifest["feature_groups"]).items()
        }
        self._target_cache: dict[str, np.ndarray] = {}
        self._valid_cache: dict[str, np.ndarray] = {}
        self._raw_return_cache: dict[int, np.ndarray] = {}
        self._raw_return_valid_cache: dict[int, np.ndarray] = {}
        self._return_cache: dict[int, np.ndarray] = {}
        self._return_valid_cache: dict[int, np.ndarray] = {}

    def rows_for_year(self, year: int) -> np.ndarray:
        rows = np.flatnonzero(self.years == int(year)).astype(np.int64, copy=False)
        if not rows.size:
            raise ModelError(f"model_input_year_missing:{year}")
        return rows

    def task_values(self, target: str) -> np.ndarray:
        cache = getattr(self, "_target_cache", None)
        if cache is None:
            cache = {}
            self._target_cache = cache
        if target in cache:
            return cache[target]
        if target.startswith("return_"):
            values = self.return_values(int(target.rsplit("_", 1)[1]))
        elif target.startswith("state"):
            values = np.asarray(
                self.states[self.candidate_ids, LABEL_HORIZON_INDEX[STATE_HORIZON]],
                dtype=np.int8,
            )
        else:
            horizon = int(target.rsplit("_", 1)[1])
            label = "mfe" if target.startswith("mfe") else "pre_peak_mae"
            column = self.label_manifest["label_columns"].index(f"{label}_{horizon}")
            values = np.asarray(
                self.labels[self.candidate_ids, column], dtype=np.float32
            )
        cache[target] = values
        return values

    def valid_mask(self, target: str) -> np.ndarray:
        cache = getattr(self, "_valid_cache", None)
        if cache is None:
            cache = {}
            self._valid_cache = cache
        if target in cache:
            return cache[target]
        if target.startswith("return_"):
            valid = self.return_valid_mask(int(target.rsplit("_", 1)[1]))
            cache[target] = valid
            return valid
        horizon_index = LABEL_HORIZON_INDEX[int(target.rsplit("_", 1)[1])]
        flags = np.asarray(
            self.flags[self.candidate_ids, horizon_index], dtype=np.uint16
        )
        if target.startswith("state"):
            state = self.task_values(target)
            valid = ((flags & FLAG_STATE_ASSIGNED) != 0) & np.isin(state, (0, 1, 2))
        else:
            values = self.task_values(target)
            valid = ((flags & FLAG_MFE_PRE_PEAK_MAE_VALID) != 0) & np.isfinite(values)
        cache[target] = valid
        return valid

    def _ensure_short_return_labels(self) -> None:
        if hasattr(self, "short_labels") and hasattr(self, "short_flags"):
            return
        manifest = _read_json(DEFAULT_SHORT_LABEL_MANIFEST)
        if (
            manifest.get("status") != "completed"
            or int(manifest.get("candidate_count", -1)) <= int(self.candidate_ids.max())
            or int(manifest.get("forbidden_outcome_year", -1)) != FORBIDDEN_YEAR
            or str(manifest.get("maximum_outcome_date_read")) != MAXIMUM_OUTCOME_DATE
        ):
            raise ModelError("short_return_label_contract_mismatch")
        files = dict(manifest["files"])
        labels = dict(files["short_labels"])
        flags = dict(files["flags"])
        self.short_label_manifest = manifest
        self.short_labels = np.memmap(
            Path(labels["path"]),
            dtype=np.dtype(labels["dtype"]),
            mode="r",
            shape=tuple(int(value) for value in labels["shape"]),
        )
        self.short_flags = np.memmap(
            Path(flags["path"]),
            dtype=np.dtype(flags["dtype"]),
            mode="r",
            shape=tuple(int(value) for value in flags["shape"]),
        )

    def raw_return_values(self, horizon: int) -> np.ndarray:
        horizon = int(horizon)
        cache = getattr(self, "_raw_return_cache", None)
        if cache is None:
            cache = {}
            self._raw_return_cache = cache
        if horizon in cache:
            return cache[horizon]
        if horizon in {1, 3}:
            self._ensure_short_return_labels()
            column = self.short_label_manifest["label_columns"].index(f"g_{horizon}")
            values = np.asarray(
                self.short_labels[self.candidate_ids, column], dtype=np.float32
            )
        elif horizon in {5, 10, 20}:
            column = self.label_manifest["label_columns"].index(f"g_{horizon}")
            values = np.asarray(
                self.labels[self.candidate_ids, column], dtype=np.float32
            )
        else:
            raise ModelError(f"unsupported_return_horizon:{horizon}")
        cache[horizon] = values
        return values

    def raw_task_values(self, target: str) -> np.ndarray:
        """Return the untransformed values used to evaluate a model target."""
        if target.startswith("return_"):
            return self.raw_return_values(int(target.rsplit("_", 1)[1]))
        return self.task_values(target)

    def raw_return_valid_mask(self, horizon: int) -> np.ndarray:
        horizon = int(horizon)
        cache = getattr(self, "_raw_return_valid_cache", None)
        if cache is None:
            cache = {}
            self._raw_return_valid_cache = cache
        if horizon in cache:
            return cache[horizon]
        values = self.raw_return_values(horizon)
        if horizon in {1, 3}:
            flag_column = 0 if horizon == 1 else 1
            flags = np.asarray(
                self.short_flags[self.candidate_ids, flag_column], dtype=np.uint16
            )
            valid = ((flags & SHORT_FLAG_G_VALID) != 0) & np.isfinite(values)
        else:
            flags = np.asarray(
                self.flags[self.candidate_ids, LABEL_HORIZON_INDEX[horizon]],
                dtype=np.uint16,
            )
            valid = ((flags & FLAG_G_VALID) != 0) & np.isfinite(values)
        cache[horizon] = valid
        return valid

    def return_values(self, horizon: int) -> np.ndarray:
        horizon = int(horizon)
        cache = getattr(self, "_return_cache", None)
        if cache is None:
            cache = {}
            self._return_cache = cache
        if horizon not in cache:
            values, valid = _winsorized_zscore_by_date(
                values=self.raw_return_values(horizon),
                date_idx=self.date_idx,
                valid=self.raw_return_valid_mask(horizon),
            )
            cache[horizon] = values
            valid_cache = getattr(self, "_return_valid_cache", None)
            if valid_cache is None:
                valid_cache = {}
                self._return_valid_cache = valid_cache
            valid_cache[horizon] = valid
        return cache[horizon]

    def return_valid_mask(self, horizon: int) -> np.ndarray:
        horizon = int(horizon)
        cache = getattr(self, "_return_valid_cache", None)
        if cache is None:
            cache = {}
            self._return_valid_cache = cache
        if horizon not in cache:
            self.return_values(horizon)
        return cache[horizon]

    def fold(self, *, year: int, horizon: int, target: str) -> dict[str, Any]:
        year_rows = self.rows_for_year(year)
        oos_start_date_idx = int(self.date_idx[year_rows[0]])
        oos_end_date_idx = int(self.date_idx[year_rows[-1]])
        maximum_train_signal_date_idx = oos_start_date_idx - int(horizon) - 1
        if maximum_train_signal_date_idx < int(self.date_idx[0]):
            raise ModelError("fold_purge_leaves_no_training_history")
        valid = self.valid_mask(target)
        train_rows = np.flatnonzero(
            (self.date_idx <= maximum_train_signal_date_idx) & valid
        ).astype(np.int64, copy=False)
        evaluation_rows = year_rows[valid[year_rows]]
        if not train_rows.size or not evaluation_rows.size:
            raise ModelError(f"fold_empty:{target}:{year}")
        if int(self.date_idx[train_rows[-1]]) + int(horizon) >= oos_start_date_idx:
            raise ModelError("horizon_purge_contract_failed")
        return {
            "year": int(year),
            "horizon": int(horizon),
            "target": target,
            "oos_start_date_idx": oos_start_date_idx,
            "oos_end_date_idx": oos_end_date_idx,
            "maximum_train_signal_date_idx": maximum_train_signal_date_idx,
            "train_rows": train_rows,
            "evaluation_rows": evaluation_rows,
        }

    def layout(self, feature_names: Sequence[str]) -> dict[str, Any]:
        names = [str(value) for value in feature_names]
        if not names or len(set(names)) != len(names):
            raise ModelError("model_feature_names_empty_or_duplicate")
        records = []
        for name in names:
            if name not in self.feature_map:
                raise ModelError(f"model_feature_not_registered:{name}")
            records.append(self.feature_map[name])
        positions: dict[str, list[tuple[int, int]]] = {
            "compact": [],
        }
        for position, record in enumerate(records):
            positions[str(record["storage"])].append(
                (position, int(record["column_index"]))
            )
        return {
            "feature_names": names,
            "records": records,
            "positions": positions,
            "categorical_positions": [],
        }


class CloseD1ModelInputs(ModelInputs):
    """Use the isolated close-to-close D1 cache as the horizon-one return target."""

    def __init__(
        self, manifest: Mapping[str, Any], label_contract: Mapping[str, Any]
    ) -> None:
        super().__init__(manifest)
        if int(label_contract.get("row_count", -1)) != self.row_count:
            raise ModelError("close_d1_label_row_count_mismatch")
        values_record = dict(label_contract["files"]["values"])
        valid_record = dict(label_contract["files"]["valid"])
        values_shape = tuple(int(value) for value in values_record["shape"])
        valid_shape = tuple(int(value) for value in valid_record["shape"])
        if values_shape != (self.row_count,) or valid_shape != (self.row_count,):
            raise ModelError("close_d1_label_shape_mismatch")
        values_path = Path(values_record["path"])
        valid_path = Path(valid_record["path"])
        if not values_path.is_file() or not valid_path.is_file():
            raise ModelError("close_d1_label_file_missing")
        self._raw_return_cache[1] = np.memmap(
            values_path,
            dtype=np.dtype(values_record["dtype"]),
            mode="r",
            shape=values_shape,
        )
        valid_values = np.memmap(
            valid_path,
            dtype=np.dtype(valid_record["dtype"]),
            mode="r",
            shape=valid_shape,
        )
        self._raw_return_valid_cache[1] = np.asarray(valid_values, dtype=bool)


def _make_sequence(
    *, inputs: ModelInputs, rows: np.ndarray, layout: Mapping[str, Any], batch_size: int
) -> Any:
    import lightgbm as lgb

    selected_rows = np.asarray(rows, dtype=np.int64)
    positions = dict(layout["positions"])

    class MatrixSequence(lgb.Sequence):
        def __init__(self) -> None:
            self.batch_size = int(batch_size)

        def __len__(self) -> int:
            return len(selected_rows)

        def _block(self, local: np.ndarray) -> np.ndarray:
            local = np.asarray(local, dtype=np.int64)
            global_rows = selected_rows[local]
            output = np.empty(
                (len(local), len(layout["feature_names"])), dtype=np.float64
            )
            for storage, pairs in positions.items():
                if not pairs:
                    continue
                target_positions = np.asarray(
                    [pair[0] for pair in pairs], dtype=np.int64
                )
                columns = np.asarray([pair[1] for pair in pairs], dtype=np.int32)
                if storage != "compact":
                    raise ModelError(f"unknown_model_storage:{storage}")
                values = np.asarray(
                    inputs.compact[np.ix_(global_rows, columns)],
                    dtype=np.float64,
                )
                output[:, target_positions] = values
            return output

        def __getitem__(self, index: Any) -> np.ndarray:
            if isinstance(index, slice):
                start = 0 if index.start is None else int(index.start)
                stop = len(selected_rows) if index.stop is None else int(index.stop)
                step = 1 if index.step is None else int(index.step)
                return self._block(np.arange(start, stop, step, dtype=np.int64))
            if isinstance(index, (list, tuple, np.ndarray)):
                return self._block(np.asarray(index, dtype=np.int64))
            if isinstance(index, (int, np.integer)):
                return self._block(np.asarray([int(index)], dtype=np.int64))[0]
            raise TypeError(f"unsupported_sequence_index:{type(index).__name__}")

    return MatrixSequence()


def _target_parameters(config: Mapping[str, Any], kind: str) -> dict[str, Any]:
    model = dict(config["model"])
    parameters: dict[str, Any] = {
        "boosting_type": "gbdt",
        "device_type": "cpu",
        "learning_rate": float(model["learning_rate"]),
        "num_leaves": int(model["num_leaves"]),
        "max_depth": int(model["max_depth"]),
        "min_data_in_leaf": int(model["min_data_in_leaf"]),
        "lambda_l2": float(model["lambda_l2"]),
        "feature_fraction": float(model["feature_fraction"]),
        "bagging_fraction": float(model["bagging_fraction"]),
        "bagging_freq": int(model["bagging_freq"]),
        "max_bin": int(model["max_bin"]),
        "num_threads": int(model["num_threads"]),
        "histogram_pool_size": int(model["histogram_pool_size_mb"]),
        "deterministic": True,
        "force_col_wise": True,
        "seed": int(model["seed"]),
        "feature_fraction_seed": int(model["seed"]),
        "bagging_seed": int(model["seed"]),
        "data_random_seed": int(model["seed"]),
        "feature_pre_filter": False,
        "verbosity": -1,
    }
    if kind in {"mfe", "risk"}:
        parameters.update(
            {
                "objective": "huber",
                "metric": "huber",
                "alpha": float(model["huber_alpha"]),
            }
        )
    elif kind == "return":
        parameters.update({"objective": "regression", "metric": "l2"})
    elif kind == "state":
        parameters.update(
            {"objective": "multiclass", "metric": "multi_logloss", "num_class": 3}
        )
    else:
        raise ModelError(f"unknown_model_kind:{kind}")
    return parameters


def _aligned_labels(
    *, inputs: ModelInputs, target: str, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(inputs.task_values(target)[rows])
    valid = np.asarray(inputs.valid_mask(target)[rows], dtype=bool)
    date_idx = inputs.date_idx[rows]
    weights = np.zeros(len(rows), dtype=np.float32)
    if not bool(valid.any()):
        raise ModelError(f"target_has_no_valid_rows:{target}")
    weights[valid] = learnability.date_equal_weights(date_idx[valid])
    labels = values.copy()
    if target.startswith("state"):
        labels[~valid] = 0
        labels = labels.astype(np.int32, copy=False)
    else:
        labels[~valid] = 0.0
        labels = labels.astype(np.float32, copy=False)
    return labels, weights, values


def _build_datasets(
    *,
    inputs: ModelInputs,
    config: Mapping[str, Any],
    target: str,
    feature_names: Sequence[str],
    train_rows: np.ndarray,
    evaluation_rows: np.ndarray,
) -> dict[str, Any]:
    import lightgbm as lgb

    layout = inputs.layout(feature_names)
    train_label, train_weight, _ = _aligned_labels(
        inputs=inputs, target=target, rows=train_rows
    )
    evaluation_label, evaluation_weight, evaluation_raw = _aligned_labels(
        inputs=inputs, target=target, rows=evaluation_rows
    )
    batch_size = int(config["model"]["sequence_batch_size"])
    train_sequence = _make_sequence(
        inputs=inputs, rows=train_rows, layout=layout, batch_size=batch_size
    )
    evaluation_sequence = _make_sequence(
        inputs=inputs, rows=evaluation_rows, layout=layout, batch_size=batch_size
    )
    train_sequence = learnability._memory_trimmed_sequence(train_sequence, config)
    evaluation_sequence = learnability._memory_trimmed_sequence(
        evaluation_sequence, config
    )
    construction = {
        "max_bin": int(config["model"]["max_bin"]),
        "data_random_seed": int(config["model"]["seed"]),
        "feature_pre_filter": False,
        "verbosity": -1,
    }
    categorical_positions = list(layout["categorical_positions"])
    train_set = lgb.Dataset(
        train_sequence,
        label=train_label,
        weight=train_weight,
        feature_name=list(feature_names),
        categorical_feature=categorical_positions,
        free_raw_data=True,
        params=construction,
    )
    evaluation_set = lgb.Dataset(
        evaluation_sequence,
        label=evaluation_label,
        weight=evaluation_weight,
        feature_name=list(feature_names),
        categorical_feature=categorical_positions,
        reference=train_set,
        free_raw_data=True,
        params=construction,
    )
    train_set.construct()
    evaluation_set.construct()
    return {
        "layout": layout,
        "train_sequence": train_sequence,
        "evaluation_sequence": evaluation_sequence,
        "train_set": train_set,
        "evaluation_set": evaluation_set,
        "train_label": train_label,
        "train_weight": train_weight,
        "evaluation_label": evaluation_label,
        "evaluation_weight": evaluation_weight,
        "evaluation_raw": evaluation_raw,
    }


def _predict(model: Any, sequence: Any, iterations: int) -> np.ndarray:
    output: np.ndarray | None = None
    for start in range(0, len(sequence), 250_000):
        stop = min(start + 250_000, len(sequence))
        part = np.asarray(
            model.predict(sequence[start:stop], num_iteration=int(iterations))
        )
        if output is None:
            shape = (
                (len(sequence),) if part.ndim == 1 else (len(sequence), part.shape[1])
            )
            output = np.empty(shape, dtype=np.float32)
        output[start:stop] = part.astype(np.float32, copy=False)
    if output is None:
        raise ModelError("empty_prediction_sequence")
    return output


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3:
        return float("nan")
    value = stats.spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else float("nan")


def _date_boundaries(date_idx: np.ndarray) -> np.ndarray:
    dates = np.asarray(date_idx, dtype=np.int32)
    if dates.ndim != 1 or bool(np.any(dates[1:] < dates[:-1])):
        raise ModelError("metric_rows_not_date_ordered")
    return np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])


def _mfe_daily_metrics(
    *,
    date_idx: np.ndarray,
    actual: np.ndarray,
    prediction: np.ndarray,
    risk: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    deciles: list[np.ndarray] = []
    boundaries = _date_boundaries(date_idx)
    for left, right in itertools.pairwise(boundaries):
        y = np.asarray(actual[left:right], dtype=np.float64)
        p = np.asarray(prediction[left:right], dtype=np.float64)
        valid = np.isfinite(y) & np.isfinite(p)
        if int(valid.sum()) < 20:
            continue
        y = y[valid]
        p = p[valid]
        risk_values = np.asarray(risk[left:right], dtype=np.float64)[valid]
        order_pred = np.argsort(p, kind="mergesort")
        order_true = np.argsort(y, kind="mergesort")
        count = len(y)
        top1_count = max(1, math.ceil(0.01 * count))
        top5_count = max(1, math.ceil(0.05 * count))
        top1_pred = order_pred[-top1_count:]
        top5_pred = order_pred[-top5_count:]
        top1_true = set(order_true[-top1_count:].tolist())
        top5_true = set(order_true[-top5_count:].tolist())
        top5_risk = -risk_values[top5_pred]
        top5_risk = top5_risk[np.isfinite(top5_risk)]
        decile_means = np.asarray(
            [float(y[part].mean()) for part in np.array_split(order_pred, 10)],
            dtype=np.float64,
        )
        deciles.append(decile_means)
        records.append(
            {
                "date_idx": int(date_idx[left]),
                "rank_ic": _safe_spearman(y, p),
                "top1_mfe_mean": float(y[top1_pred].mean()),
                "top5_mfe_mean": float(y[top5_pred].mean()),
                "top1_mfe_median": float(np.median(y[top1_pred])),
                "top5_mfe_median": float(np.median(y[top5_pred])),
                "top1_capture": float(
                    len(top1_true.intersection(set(top1_pred.tolist())))
                    / len(top1_true)
                ),
                "top5_capture": float(
                    len(top5_true.intersection(set(top5_pred.tolist())))
                    / len(top5_true)
                ),
                "top5_adverse_median": float(np.median(top5_risk))
                if len(top5_risk)
                else float("nan"),
                "mae": float(np.mean(np.abs(y - p))),
                "row_count": int(count),
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ModelError("mfe_metrics_no_valid_dates")
    decile = np.nanmean(np.asarray(deciles, dtype=np.float64), axis=0)
    summary = {
        "date_count": len(frame),
        "rank_ic": float(frame["rank_ic"].mean()),
        "top1_mfe_mean": float(frame["top1_mfe_mean"].mean()),
        "top5_mfe_mean": float(frame["top5_mfe_mean"].mean()),
        "top1_mfe_median": float(frame["top1_mfe_median"].mean()),
        "top5_mfe_median": float(frame["top5_mfe_median"].mean()),
        "top1_capture": float(frame["top1_capture"].mean()),
        "top5_capture": float(frame["top5_capture"].mean()),
        "top5_adverse_median": float(frame["top5_adverse_median"].mean()),
        "mae": float(frame["mae"].mean()),
        "decile_spearman": _safe_spearman(np.arange(10, dtype=np.float64), decile),
    }
    return frame, summary


def _return_daily_metrics(
    *,
    date_idx: np.ndarray,
    actual_raw: np.ndarray,
    actual_zscore: np.ndarray,
    prediction: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    boundaries = _date_boundaries(date_idx)
    decile_columns = [f"decile_{index}_return_mean" for index in range(1, 11)]
    for left, right in itertools.pairwise(boundaries):
        raw = np.asarray(actual_raw[left:right], dtype=np.float64)
        zscore = np.asarray(actual_zscore[left:right], dtype=np.float64)
        score = np.asarray(prediction[left:right], dtype=np.float64)
        valid = np.isfinite(raw) & np.isfinite(zscore) & np.isfinite(score)
        if int(valid.sum()) < RETURN_MIN_CROSS_SECTION:
            continue
        raw = raw[valid]
        zscore = zscore[valid]
        score = score[valid]
        order_pred = np.argsort(score, kind="mergesort")
        order_true = np.argsort(raw, kind="mergesort")
        count = len(raw)
        top1_count = max(1, math.ceil(0.01 * count))
        top5_count = max(1, math.ceil(0.05 * count))
        top1_pred = order_pred[-top1_count:]
        top5_pred = order_pred[-top5_count:]
        top1_true = set(order_true[-top1_count:].tolist())
        top5_true = set(order_true[-top5_count:].tolist())
        universe_mean = float(raw.mean())
        top1_mean = float(raw[top1_pred].mean())
        top5_mean = float(raw[top5_pred].mean())
        decile = np.asarray(
            [float(raw[part].mean()) for part in np.array_split(order_pred, 10)],
            dtype=np.float64,
        )
        record = {
            "date_idx": int(date_idx[left]),
            "rank_ic": _safe_spearman(raw, score),
            "universe_return_mean": universe_mean,
            "top1_return_mean": top1_mean,
            "top5_return_mean": top5_mean,
            "top1_return_median": float(np.median(raw[top1_pred])),
            "top5_return_median": float(np.median(raw[top5_pred])),
            "top1_excess_mean": top1_mean - universe_mean,
            "top5_excess_mean": top5_mean - universe_mean,
            "top1_capture": float(
                len(top1_true.intersection(set(top1_pred.tolist()))) / len(top1_true)
            ),
            "top5_capture": float(
                len(top5_true.intersection(set(top5_pred.tolist()))) / len(top5_true)
            ),
            "decile_spearman": _safe_spearman(np.arange(10, dtype=np.float64), decile),
            "zscore_mse": float(np.mean(np.square(zscore - score))),
            "row_count": int(count),
        }
        record.update(dict(zip(decile_columns, decile.tolist(), strict=True)))
        records.append(record)
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ModelError("return_metrics_no_valid_dates")
    metric_columns = [
        "rank_ic",
        "universe_return_mean",
        "top1_return_mean",
        "top5_return_mean",
        "top1_return_median",
        "top5_return_median",
        "top1_excess_mean",
        "top5_excess_mean",
        "top1_capture",
        "top5_capture",
        "decile_spearman",
        "zscore_mse",
        *decile_columns,
    ]
    summary = {"date_count": len(frame)}
    summary.update(
        {
            column: float(pd.to_numeric(frame[column], errors="coerce").mean())
            for column in metric_columns
        }
    )
    return frame, summary


def _risk_daily_metrics(
    *, date_idx: np.ndarray, actual: np.ndarray, prediction: np.ndarray
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    boundaries = _date_boundaries(date_idx)
    for left, right in itertools.pairwise(boundaries):
        y = np.asarray(actual[left:right], dtype=np.float64)
        p = np.asarray(prediction[left:right], dtype=np.float64)
        valid = np.isfinite(y) & np.isfinite(p)
        if int(valid.sum()) < 20:
            continue
        y = y[valid]
        p = p[valid]
        order = np.argsort(y, kind="mergesort")
        count = max(1, math.ceil(0.20 * len(y)))
        event = np.zeros(len(y), dtype=np.int8)
        event[order[:count]] = 1
        records.append(
            {
                "date_idx": int(date_idx[left]),
                "rank_ic": _safe_spearman(p, y),
                "mae": float(np.mean(np.abs(p - y))),
                "deep_adverse_pr_auc": float(average_precision_score(event, -p)),
                "row_count": len(y),
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ModelError("risk_metrics_no_valid_dates")
    return frame, {
        "date_count": len(frame),
        "rank_ic": float(frame["rank_ic"].mean()),
        "mae": float(frame["mae"].mean()),
        "deep_adverse_pr_auc": float(frame["deep_adverse_pr_auc"].mean()),
    }


def _state_daily_metrics(
    *, date_idx: np.ndarray, actual: np.ndarray, prediction: np.ndarray
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    boundaries = _date_boundaries(date_idx)
    for left, right in itertools.pairwise(boundaries):
        y = np.asarray(actual[left:right], dtype=np.int8)
        p = np.asarray(prediction[left:right], dtype=np.float64)
        valid = np.isin(y, (0, 1, 2)) & np.isfinite(p).all(axis=1)
        if int(valid.sum()) < 20:
            continue
        y = y[valid]
        p = np.clip(p[valid], 1.0e-7, 1.0)
        p /= p.sum(axis=1, keepdims=True)
        expected = p[:, 1] + 2.0 * p[:, 2]
        one_hot = np.eye(3, dtype=np.float64)[y]
        count = max(1, math.ceil(0.05 * len(y)))
        selected = np.argsort(p[:, 2], kind="mergesort")[-count:]
        high = y == 2
        records.append(
            {
                "date_idx": int(date_idx[left]),
                "ordinal_ic": _safe_spearman(expected, y.astype(np.float64)),
                "high_state_top5_lift": float(high[selected].mean() - high.mean()),
                "brier": float(np.mean(np.square(p - one_hot).sum(axis=1))),
                "logloss": float(
                    np.mean(-np.log(np.clip(p[np.arange(len(y)), y], 1.0e-12, 1.0)))
                ),
                "row_count": len(y),
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ModelError("state_metrics_no_valid_dates")
    return frame, {
        "date_count": len(frame),
        "ordinal_ic": float(frame["ordinal_ic"].mean()),
        "high_state_top5_lift": float(frame["high_state_top5_lift"].mean()),
        "brier": float(frame["brier"].mean()),
        "logloss": float(frame["logloss"].mean()),
    }


def _task_plan(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    targets = dict(config["targets"])
    tasks: list[dict[str, Any]] = []

    def add(
        *,
        stage: str,
        target: str,
        year: int,
        variant: str,
        gated_family: str | None = None,
    ) -> None:
        target_config = dict(targets[target])
        task_id = f"{stage}__{target}__{variant}__{year}"
        tasks.append(
            {
                "task_id": task_id,
                "stage": stage,
                "target": target,
                "kind": str(target_config["kind"]),
                "horizon": int(target_config["horizon"]),
                "year": int(year),
                "variant": variant,
                "gated_family": gated_family,
            }
        )

    adaptive_targets = ("risk_10", "risk_20", "state_10")
    for target in adaptive_targets:
        for year in ROLLING_YEARS:
            add(stage="tuning", target=target, year=year, variant=COMPACT_VARIANT)
    for target in ("mfe_10", "mfe_20"):
        for year in ROLLING_YEARS:
            add(stage="mfe_core", target=target, year=year, variant=COMPACT_VARIANT)
    for target in adaptive_targets:
        for year in ROLLING_YEARS:
            add(
                stage="risk_state_core",
                target=target,
                year=year,
                variant=COMPACT_VARIANT,
            )
    if len(tasks) != 24 or len({task["task_id"] for task in tasks}) != 24:
        raise ModelError(f"task_plan_contract_mismatch:{len(tasks)}")
    return tasks


def _direct_return_task_plan() -> list[dict[str, Any]]:
    tasks = [
        {
            "task_id": (
                f"{RETURN_TASK_STAGE}__g_{horizon:02d}__{COMPACT_VARIANT}__{year}"
            ),
            "stage": RETURN_TASK_STAGE,
            "target": f"return_{horizon}",
            "source_label": f"g_{horizon}",
            "kind": "return",
            "horizon": int(horizon),
            "year": int(year),
            "variant": COMPACT_VARIANT,
            "gated_family": None,
            "label_transform": "signal_date_winsor_01_99_zscore",
        }
        for horizon in RETURN_HORIZONS
        for year in ROLLING_YEARS
    ]
    if (
        len(tasks) != RETURN_TASK_COUNT
        or len({str(task["task_id"]) for task in tasks}) != RETURN_TASK_COUNT
    ):
        raise ModelError(f"direct_return_task_plan_contract_mismatch:{len(tasks)}")
    return tasks


def _close_d1_task_plan() -> list[dict[str, Any]]:
    tasks = [
        {
            "task_id": f"{CLOSE_D1_TASK_STAGE}__close_01__{COMPACT_VARIANT}__{year}",
            "stage": CLOSE_D1_TASK_STAGE,
            "target": CLOSE_D1_TARGET,
            "source_label": "future_ohlcva_path.day_0.close",
            "kind": "return",
            "horizon": 1,
            "year": int(year),
            "variant": COMPACT_VARIANT,
            "gated_family": None,
            "label_transform": "signal_date_winsor_01_99_zscore",
        }
        for year in ROLLING_YEARS
    ]
    if (
        len(tasks) != CLOSE_D1_TASK_COUNT
        or len({str(task["task_id"]) for task in tasks}) != CLOSE_D1_TASK_COUNT
    ):
        raise ModelError(f"close_d1_task_plan_contract_mismatch:{len(tasks)}")
    return tasks


def _save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(values), allow_pickle=False)
    os.replace(temporary, path)


def _task_result_path(output_root: Path, task_id: str) -> Path:
    return output_root / "tasks" / task_id / "task_result.json"


def _task_fingerprint(
    *,
    task: Mapping[str, Any],
    feature_names: Sequence[str],
    model_input_fingerprint: str,
    config_sha256: str,
    experiment_fingerprint: str | None = None,
) -> str:
    payload = {
        "task": dict(task),
        "feature_names": list(feature_names),
        "model_input_fingerprint": model_input_fingerprint,
        "config_sha256": config_sha256,
    }
    if experiment_fingerprint is not None:
        payload["experiment_fingerprint"] = str(experiment_fingerprint)
    return _stable_hash(payload)


def _task_complete(path: Path, *, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    try:
        result = _read_json(path)
        if (
            result.get("status") != "completed"
            or result.get("task_fingerprint") != fingerprint
        ):
            return False
        return all(
            Path(record["path"]).is_file()
            and _sha256(Path(record["path"])) == record["sha256"]
            for record in dict(result.get("files", {}) or {}).values()
        )
    except (ModelError, OSError, KeyError, ValueError, TypeError):
        return False


def _effective_features(
    *,
    inputs: ModelInputs,
    task: Mapping[str, Any],
    output_root: Path,
) -> tuple[list[str], str]:
    variant = str(task["variant"])
    stage = str(task["stage"])
    if stage in {
        "tuning",
        "mfe_core",
        "risk_state_core",
        RETURN_TASK_STAGE,
        CLOSE_D1_TASK_STAGE,
    }:
        return list(inputs.feature_groups[variant]), variant
    raise ModelError(f"unknown_task_stage:{stage}")


def _evaluate_prediction(
    *,
    inputs: ModelInputs,
    task: Mapping[str, Any],
    rows: np.ndarray,
    prediction: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    target = str(task["target"])
    kind = str(task["kind"])
    actual = inputs.task_values(target)[rows]
    if kind == "return":
        return _return_daily_metrics(
            date_idx=inputs.date_idx[rows],
            actual_raw=inputs.raw_task_values(target)[rows],
            actual_zscore=actual,
            prediction=prediction,
        )
    if kind == "mfe":
        risk = inputs.task_values(f"risk_{int(task['horizon'])}")[rows]
        return _mfe_daily_metrics(
            date_idx=inputs.date_idx[rows],
            actual=actual,
            prediction=prediction,
            risk=risk,
        )
    if kind == "risk":
        return _risk_daily_metrics(
            date_idx=inputs.date_idx[rows], actual=actual, prediction=prediction
        )
    return _state_daily_metrics(
        date_idx=inputs.date_idx[rows], actual=actual, prediction=prediction
    )


def _observed_subset_metrics(
    *,
    inputs: ModelInputs,
    task: Mapping[str, Any],
    rows: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    del inputs, task, rows, prediction
    return {}


def _importance_frame(
    *,
    model: Any,
    inputs: ModelInputs,
    feature_names: Sequence[str],
    task: Mapping[str, Any],
) -> pd.DataFrame:
    gain = np.asarray(
        model.feature_importance(importance_type="gain"), dtype=np.float64
    )
    split = np.asarray(
        model.feature_importance(importance_type="split"), dtype=np.float64
    )
    rows: list[dict[str, Any]] = []
    for name, gain_value, split_value in zip(feature_names, gain, split, strict=True):
        record = inputs.feature_map[str(name)]
        family = str(record["analytic_family"])
        rows.append(
            {
                "task_id": str(task["task_id"]),
                "target": str(task["target"]),
                "year": int(task["year"]),
                "variant": str(task["variant"]),
                "feature_family": family,
                "gain": float(gain_value),
                "split": float(split_value),
                "feature_count": 1,
            }
        )
    frame = pd.DataFrame(rows)
    return (
        frame.groupby(
            ["task_id", "target", "year", "variant", "feature_family"],
            as_index=False,
        )[["gain", "split", "feature_count"]]
        .sum()
        .sort_values(["task_id", "gain"], ascending=[True, False])
    )


def _release_datasets(datasets: dict[str, Any]) -> None:
    datasets["train_set"] = None
    datasets["evaluation_set"] = None
    datasets["train_sequence"] = None
    datasets["evaluation_sequence"] = None
    gc.collect()


def _run_training_task(
    *,
    task: Mapping[str, Any],
    inputs: ModelInputs,
    config: Mapping[str, Any],
    output_root: Path,
    config_sha256: str,
    experiment_fingerprint: str | None = None,
) -> dict[str, Any]:
    import lightgbm as lgb

    feature_names, base_variant = _effective_features(
        inputs=inputs, task=task, output_root=output_root
    )
    fingerprint = _task_fingerprint(
        task=task,
        feature_names=feature_names,
        model_input_fingerprint=str(inputs.manifest["input_fingerprint"]),
        config_sha256=config_sha256,
        experiment_fingerprint=experiment_fingerprint,
    )
    result_path = _task_result_path(output_root, str(task["task_id"]))
    if _task_complete(result_path, fingerprint=fingerprint):
        return _read_json(result_path)
    outer_year = int(task["year"])
    evaluation_year = outer_year - 1 if task["stage"] == "tuning" else outer_year
    fold = inputs.fold(
        year=evaluation_year,
        horizon=int(task["horizon"]),
        target=str(task["target"]),
    )
    datasets = _build_datasets(
        inputs=inputs,
        config=config,
        target=str(task["target"]),
        feature_names=feature_names,
        train_rows=fold["train_rows"],
        evaluation_rows=fold["evaluation_rows"],
    )
    parameters = _target_parameters(config, str(task["kind"]))
    started = time.perf_counter()
    if task["stage"] == "tuning":
        model = lgb.train(
            parameters,
            datasets["train_set"],
            num_boost_round=int(config["model"]["adaptive_max_rounds"]),
            valid_sets=[datasets["evaluation_set"]],
            valid_names=["inner_validation"],
            callbacks=[
                lgb.early_stopping(
                    int(config["model"]["adaptive_patience"]),
                    first_metric_only=True,
                    verbose=False,
                )
            ],
        )
        iterations = int(model.best_iteration)
    else:
        if task["kind"] in {"mfe", "return"}:
            iterations = int(config["model"]["fixed_mfe_rounds"])
        else:
            tuning_id = f"tuning__{task['target']}__{COMPACT_VARIANT}__{outer_year}"
            tuning = _read_json(_task_result_path(output_root, tuning_id))
            iterations = int(tuning["best_iteration"])
        model = lgb.train(
            parameters,
            datasets["train_set"],
            num_boost_round=iterations,
            valid_sets=[datasets["evaluation_set"]],
            valid_names=["outer_evaluation"],
        )
    training_seconds = float(time.perf_counter() - started)
    prediction_rows = inputs.rows_for_year(evaluation_year)
    prediction_sequence = _make_sequence(
        inputs=inputs,
        rows=prediction_rows,
        layout=datasets["layout"],
        batch_size=int(config["model"]["sequence_batch_size"]),
    )
    prediction = _predict(model, prediction_sequence, iterations)
    evaluation_positions = np.searchsorted(prediction_rows, fold["evaluation_rows"])
    if not np.array_equal(
        prediction_rows[evaluation_positions], fold["evaluation_rows"]
    ):
        raise ModelError("evaluation_prediction_row_alignment_failed")
    evaluation_prediction = prediction[evaluation_positions]
    daily, metrics = _evaluate_prediction(
        inputs=inputs,
        task=task,
        rows=fold["evaluation_rows"],
        prediction=evaluation_prediction,
    )
    daily.insert(
        1,
        "trade_date",
        [str(inputs.date_values[int(value)]) for value in daily["date_idx"]],
    )
    observed = _observed_subset_metrics(
        inputs=inputs,
        task=task,
        rows=fold["evaluation_rows"],
        prediction=evaluation_prediction,
    )
    task_dir = result_path.parent
    task_dir.mkdir(parents=True, exist_ok=True)
    model_path = task_dir / "model.txt"
    model_partial = model_path.with_suffix(".txt.partial")
    model.save_model(str(model_partial), num_iteration=iterations)
    os.replace(model_partial, model_path)
    prediction_path = task_dir / "prediction.npy"
    candidate_path = task_dir / "candidate_id.npy"
    daily_path = task_dir / "daily_metrics.parquet"
    importance_path = task_dir / "family_importance.parquet"
    _save_npy(prediction_path, prediction)
    _save_npy(candidate_path, inputs.candidate_ids[prediction_rows])
    _write_parquet(daily, daily_path)
    importance = _importance_frame(
        model=model, inputs=inputs, feature_names=feature_names, task=task
    )
    _write_parquet(importance, importance_path)
    result = {
        "schema": "seq100_quality_liquidity_model_task/1",
        "status": "completed",
        "completed_at": _now(),
        "study_id": str(task.get("study_id", STUDY_ID)),
        "task_id": str(task["task_id"]),
        "task_fingerprint": fingerprint,
        "stage": str(task["stage"]),
        "target": str(task["target"]),
        "kind": str(task["kind"]),
        "horizon": int(task["horizon"]),
        "model_year": outer_year,
        "evaluation_year": evaluation_year,
        "variant": str(task["variant"]),
        "base_variant": base_variant,
        "gated_family": task.get("gated_family"),
        "source_label": task.get("source_label"),
        "label_transform": task.get("label_transform"),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "availability_feature_count": 0,
        "train_row_count": len(fold["train_rows"]),
        "evaluation_row_count": len(fold["evaluation_rows"]),
        "candidate_prediction_row_count": len(prediction_rows),
        "maximum_train_signal_date_idx": int(fold["maximum_train_signal_date_idx"]),
        "maximum_train_signal_date": str(
            inputs.date_values[int(fold["maximum_train_signal_date_idx"])]
        ),
        "purge_days": int(task["horizon"]),
        "best_iteration": iterations,
        "training_seconds": training_seconds,
        "parameters": parameters,
        "metrics": metrics,
        "observed_subset_metrics": observed,
        "files": {
            "model": _file_record(model_path),
            "prediction": _file_record(
                prediction_path,
                shape=list(prediction.shape),
                dtype=str(prediction.dtype),
            ),
            "candidate_id": _file_record(
                candidate_path,
                shape=[len(prediction_rows)],
                dtype=str(inputs.candidate_ids.dtype),
            ),
            "daily_metrics": _file_record(daily_path, row_count=len(daily)),
            "family_importance": _file_record(
                importance_path, row_count=len(importance)
            ),
        },
    }
    _write_json(result_path, result)
    _release_datasets(datasets)
    del model, prediction, evaluation_prediction, prediction_sequence
    gc.collect()
    return result


def _completed_results(output_root: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for path in sorted((output_root / "tasks").glob("*/task_result.json")):
        result = _read_json(path)
        if result.get("status") == "completed":
            results[str(result["task_id"])] = result
    return results


def _partition_results(
    results: Mapping[str, Mapping[str, Any]], tasks: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    formal_ids = {str(task["task_id"]) for task in tasks}
    formal = {
        task_id: dict(result)
        for task_id, result in results.items()
        if task_id in formal_ids
    }
    diagnostic = {
        task_id: dict(result)
        for task_id, result in results.items()
        if task_id not in formal_ids
    }
    return formal, diagnostic


def _selected_head_contract(output_root: Path, inputs: ModelInputs) -> dict[str, Any]:
    del output_root
    targets = {
        target: {
            "selected_variant": COMPACT_VARIANT,
            "feature_count": len(inputs.feature_groups[COMPACT_VARIANT]),
            "gated_features_used": False,
        }
        for target in TARGETS
    }
    return {
        "schema": "seq100_quality_liquidity_selected_head_contract/1",
        "study_id": STUDY_ID,
        "formal_task_scope": FORMAL_TASK_SCOPE,
        "targets": targets,
        "training_semantics": "retrospective_rolling_oos",
        "training_performed": True,
    }


def _root_manifest(
    *,
    output_root: Path,
    config_path: Path,
    input_manifest: Mapping[str, Any],
    input_manifest_path: Path,
    tasks: Sequence[Mapping[str, Any]],
    status: str,
    files: Mapping[str, Any] | None = None,
    diagnostic_task_ids: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "schema": "seq100_quality_liquidity_model_manifest/1",
        "study_id": STUDY_ID,
        "status": status,
        "created_at": _now(),
        "config": _file_record(config_path),
        "model_inputs": _file_record(
            input_manifest_path,
            input_fingerprint=str(input_manifest.get("input_fingerprint", "")),
        ),
        "task_count": len(tasks),
        "tasks": [dict(task) for task in tasks],
        "formal_task_scope": FORMAL_TASK_SCOPE,
        "preserved_diagnostic_task_count": len(diagnostic_task_ids),
        "preserved_diagnostic_task_ids": sorted(
            str(value) for value in diagnostic_task_ids
        ),
        "outputs": dict(files or {}),
        "training_performed": status in {"training_completed", "evaluated", "audited"},
        "feature_set_selected": status in {"evaluated", "audited"},
        "evaluation_semantics": "retrospective_rolling_oos",
        "forbidden_year": FORBIDDEN_YEAR,
    }


def _update_ledger(
    *,
    output_root: Path,
    tasks: Sequence[Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Any]],
    diagnostic_task_ids: Sequence[str] = (),
) -> None:
    entries = []
    for task in tasks:
        result = results.get(str(task["task_id"]))
        entries.append(
            {
                **dict(task),
                "status": "completed" if result is not None else "pending",
                "result_path": str(
                    _task_result_path(output_root, str(task["task_id"])).resolve()
                )
                if result is not None
                else None,
                "best_iteration": result.get("best_iteration") if result else None,
                "updated_at": _now(),
            }
        )
    _write_json(
        output_root / "task_ledger.json",
        {
            "schema": "seq100_quality_liquidity_model_task_ledger/1",
            "study_id": STUDY_ID,
            "formal_task_scope": FORMAL_TASK_SCOPE,
            "task_count": len(tasks),
            "completed_count": sum(item["status"] == "completed" for item in entries),
            "preserved_diagnostic_task_count": len(diagnostic_task_ids),
            "preserved_diagnostic_task_ids": sorted(
                str(value) for value in diagnostic_task_ids
            ),
            "tasks": entries,
        },
    )


def run(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    ready_root: Path = DEFAULT_READY_ROOT,
    audit_root: Path = DEFAULT_AUDIT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    config = _load_config(study_path)
    _audit_manifest(audit_root)
    input_manifest = prepare(
        study_path=study_path,
        ready_root=ready_root,
        audit_root=audit_root,
        output_root=output_root,
    )
    inputs = ModelInputs(input_manifest)
    tasks = _task_plan(config)
    output_root.mkdir(parents=True, exist_ok=True)
    config_sha256 = _sha256(study_path)
    all_results = _completed_results(output_root)
    results, diagnostic_results = _partition_results(all_results, tasks)
    diagnostic_task_ids = tuple(sorted(diagnostic_results))
    _update_ledger(
        output_root=output_root,
        tasks=tasks,
        results=results,
        diagnostic_task_ids=diagnostic_task_ids,
    )
    stage_order = ("tuning", "mfe_core", "risk_state_core")
    completed_this_run = 0
    for stage in stage_order:
        for task in [item for item in tasks if item["stage"] == stage]:
            task_id = str(task["task_id"])
            if task_id in results:
                feature_names, _ = _effective_features(
                    inputs=inputs, task=task, output_root=output_root
                )
                fingerprint = _task_fingerprint(
                    task=task,
                    feature_names=feature_names,
                    model_input_fingerprint=str(input_manifest["input_fingerprint"]),
                    config_sha256=config_sha256,
                )
                if _task_complete(
                    _task_result_path(output_root, task_id), fingerprint=fingerprint
                ):
                    continue
                results.pop(task_id)
            if max_tasks is not None and completed_this_run >= int(max_tasks):
                break
            result = _run_training_task(
                task=task,
                inputs=inputs,
                config=config,
                output_root=output_root,
                config_sha256=config_sha256,
            )
            results[task_id] = result
            completed_this_run += 1
            _update_ledger(
                output_root=output_root,
                tasks=tasks,
                results=results,
                diagnostic_task_ids=diagnostic_task_ids,
            )
        stage_results = [item for item in tasks if item["stage"] == stage]
        if not all(str(item["task_id"]) in results for item in stage_results):
            break
    all_complete = len(results) == len(tasks) and all(
        str(task["task_id"]) in results for task in tasks
    )
    status = "training_completed" if all_complete else "training_in_progress"
    manifest = _root_manifest(
        output_root=output_root,
        config_path=study_path,
        input_manifest=input_manifest,
        input_manifest_path=ready_root / MODEL_INPUT_DIR_NAME / "manifest.json",
        tasks=tasks,
        status=status,
        diagnostic_task_ids=diagnostic_task_ids,
    )
    _write_json(output_root / "manifest.json", manifest)
    if all_complete:
        contract = _selected_head_contract(output_root, inputs)
        _write_json(output_root / "selected_head_contract.json", contract)
    return {
        "status": status,
        "study_id": STUDY_ID,
        "task_count": len(tasks),
        "completed_count": len(results),
        "completed_this_run": completed_this_run,
        "preserved_diagnostic_task_count": len(diagnostic_task_ids),
        "training_performed": all_complete,
    }


def _annual_metrics(results: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results.values():
        row = {
            "task_id": result["task_id"],
            "stage": result["stage"],
            "target": result["target"],
            "kind": result["kind"],
            "horizon": int(result["horizon"]),
            "model_year": int(result["model_year"]),
            "evaluation_year": int(result["evaluation_year"]),
            "variant": result["variant"],
            "base_variant": result.get("base_variant"),
            "gated_family": result.get("gated_family"),
            "feature_count": int(result["feature_count"]),
            "best_iteration": int(result["best_iteration"]),
            "train_row_count": int(result["train_row_count"]),
            "evaluation_row_count": int(result["evaluation_row_count"]),
        }
        row.update({str(key): value for key, value in dict(result["metrics"]).items()})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["target", "evaluation_year", "variant", "stage"]
    )


def _paired_variant_deltas(*, results: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    del results
    return pd.DataFrame(
        columns=[
            "target",
            "year",
            "candidate_variant",
            "reference_variant",
            "comparison_kind",
        ]
    )


def evaluate(
    *, output_root: Path = DEFAULT_OUTPUT_ROOT, study_path: Path = DEFAULT_STUDY_PATH
) -> dict[str, Any]:
    config = _load_config(study_path)
    manifest = _read_json(output_root / "manifest.json")
    if manifest.get("status") not in {"training_completed", "evaluated", "audited"}:
        raise ModelError("training_not_completed")
    tasks = _task_plan(config)
    all_results = _completed_results(output_root)
    results, diagnostic_results = _partition_results(all_results, tasks)
    if len(results) != len(tasks):
        raise ModelError(f"evaluation_task_count_mismatch:{len(results)}:{len(tasks)}")
    annual = _annual_metrics(results)
    paired = _paired_variant_deltas(results=results)
    importance_parts = []
    for result in results.values():
        importance_parts.append(
            pd.read_parquet(Path(result["files"]["family_importance"]["path"]))
        )
    importance = pd.concat(importance_parts, ignore_index=True)
    annual_path = output_root / "annual_metrics.parquet"
    paired_path = output_root / "paired_variant_deltas.parquet"
    importance_path = output_root / "family_importance.parquet"
    _write_parquet(annual, annual_path)
    _write_parquet(paired, paired_path)
    _write_parquet(importance, importance_path)
    inputs = ModelInputs(_read_json(Path(manifest["model_inputs"]["path"])))
    contract = _selected_head_contract(output_root, inputs)
    contract_path = output_root / "selected_head_contract.json"
    _write_json(contract_path, contract)
    files = {
        "annual_metrics": _file_record(annual_path, row_count=len(annual)),
        "paired_variant_deltas": _file_record(paired_path, row_count=len(paired)),
        "family_importance": _file_record(importance_path, row_count=len(importance)),
        "selected_head_contract": _file_record(contract_path),
    }
    manifest["status"] = "evaluated"
    manifest["outputs"] = files
    manifest["training_performed"] = True
    manifest["feature_set_selected"] = True
    _write_json(output_root / "manifest.json", manifest)
    return {
        "status": "evaluated",
        "task_count": len(results),
        "preserved_diagnostic_task_count": len(diagnostic_results),
        "annual_metric_rows": len(annual),
        "paired_delta_rows": len(paired),
        "family_importance_rows": len(importance),
        "selected_head_contract": str(contract_path.resolve()),
    }


def audit(
    *, output_root: Path = DEFAULT_OUTPUT_ROOT, study_path: Path = DEFAULT_STUDY_PATH
) -> dict[str, Any]:
    config = _load_config(study_path)
    manifest = _read_json(output_root / "manifest.json")
    input_manifest = _read_json(Path(manifest["model_inputs"]["path"]))
    input_checks = _verify_model_input_files(input_manifest, full_hash=True)
    del input_checks
    inputs = ModelInputs(input_manifest)
    tasks = _task_plan(config)
    all_results = _completed_results(output_root)
    results, diagnostic_results = _partition_results(all_results, tasks)
    task_ids = {str(task["task_id"]) for task in tasks}
    checks: dict[str, Any] = {
        "manifest_status": manifest.get("status")
        in {"training_completed", "evaluated", "audited"},
        "task_count_exact": set(results) == task_ids,
        "formal_task_scope_compact_core_only": manifest.get("formal_task_scope")
        == FORMAL_TASK_SCOPE,
        "input_fingerprint_present": bool(input_manifest.get("input_fingerprint")),
        "row_count_exact": inputs.row_count == int(input_manifest["row_count"]),
        "forbidden_2026_rows": not bool(
            np.char.startswith(inputs.trade_date, "2026-").any()
        ),
        "feature_contract_counts": len(
            input_manifest.get("feature_groups", {}).get(COMPACT_VARIANT, [])
        )
        == COMPACT_FEATURE_COUNT,
        "self_contained_feature_storage": set(input_manifest.get("storage", {}))
        == {"compact", "yearly_profiles"}
        and all(
            str(item.get("storage")) == "compact"
            for item in input_manifest.get("features", [])
        ),
        "repaired_quality_pool_context": all(
            int(profile["refreshed_base_missing"]["log_total_share"]) == 0
            and int(profile["industry_missing_nonzero_count"]) == 0
            for profile in input_manifest["storage"]["yearly_profiles"].values()
        ),
        "forbidden_listing_age": not any(
            str(item.get("feature_name")) == "listing_age_days"
            for item in input_manifest.get("features", [])
        ),
        "outputs_hashed": True,
        "task_files_valid": True,
        "purge_valid": True,
    }
    for task in tasks:
        result = results.get(str(task["task_id"]))
        if result is None:
            checks["task_files_valid"] = False
            continue
        try:
            feature_names, _ = _effective_features(
                inputs=inputs, task=task, output_root=output_root
            )
            fingerprint = _task_fingerprint(
                task=task,
                feature_names=feature_names,
                model_input_fingerprint=str(input_manifest["input_fingerprint"]),
                config_sha256=_sha256(study_path),
            )
            checks["task_files_valid"] &= _task_complete(
                _task_result_path(output_root, str(task["task_id"])),
                fingerprint=fingerprint,
            )
            prediction_path = Path(result["files"]["prediction"]["path"])
            candidate_path = Path(result["files"]["candidate_id"]["path"])
            prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
            candidate_ids = np.load(candidate_path, mmap_mode="r", allow_pickle=False)
            expected_rows = inputs.rows_for_year(int(result["evaluation_year"]))
            expected_ids = inputs.candidate_ids[expected_rows]
            checks["task_files_valid"] &= np.array_equal(candidate_ids, expected_ids)
            checks["task_files_valid"] &= prediction.shape[0] == len(expected_ids)
            fold = inputs.fold(
                year=int(result["evaluation_year"]),
                horizon=int(result["horizon"]),
                target=str(result["target"]),
            )
            checks["purge_valid"] &= int(
                result["maximum_train_signal_date_idx"]
            ) == int(fold["maximum_train_signal_date_idx"])
        except (ModelError, OSError, KeyError, ValueError, TypeError, IndexError):
            checks["task_files_valid"] = False
    for record in dict(manifest.get("outputs", {}) or {}).values():
        path = Path(record["path"])
        checks["outputs_hashed"] &= path.is_file() and _sha256(path) == record["sha256"]
    audit_payload = {
        "schema": "seq100_quality_liquidity_model_audit/1",
        "study_id": STUDY_ID,
        "status": "ok" if all(checks.values()) else "failed",
        "created_at": _now(),
        "checks": checks,
        "task_count": len(results),
        "preserved_diagnostic_task_count": len(diagnostic_results),
        "training_performed": True,
        "evaluation_semantics": "retrospective_rolling_oos",
    }
    _write_json(output_root / "audit.json", audit_payload)
    if audit_payload["status"] != "ok":
        raise ModelError("model_audit_failed")
    manifest["status"] = "audited"
    manifest["audit"] = _file_record(output_root / "audit.json")
    _write_json(output_root / "manifest.json", manifest)
    return audit_payload


def _direct_return_root(output_root: Path) -> Path:
    return output_root / DIRECT_RETURN_DIR_NAME


def _verify_return_label_sources(
    *, main_manifest: Mapping[str, Any], short_manifest: Mapping[str, Any]
) -> None:
    candidate_count = int(main_manifest.get("candidate_count", -1))
    if (
        candidate_count <= 0
        or int(short_manifest.get("candidate_count", -2)) != candidate_count
        or str(main_manifest.get("outcome_cutoff")) != MAXIMUM_OUTCOME_DATE
        or int(main_manifest.get("forbidden_outcome_year", -1)) != FORBIDDEN_YEAR
        or str(short_manifest.get("maximum_outcome_date_read")) != MAXIMUM_OUTCOME_DATE
        or int(short_manifest.get("forbidden_outcome_year", -1)) != FORBIDDEN_YEAR
        or short_manifest.get("status") != "completed"
    ):
        raise ModelError("direct_return_label_source_contract_mismatch")
    if not {"g_5", "g_10", "g_20"}.issubset(main_manifest["label_columns"]):
        raise ModelError("main_return_labels_missing")
    if not {"g_1", "g_3"}.issubset(short_manifest["label_columns"]):
        raise ModelError("short_return_labels_missing")
    records = [
        main_manifest["files"]["candidate_labels"],
        main_manifest["files"]["label_flags"],
        short_manifest["files"]["short_labels"],
        short_manifest["files"]["flags"],
    ]
    for record in records:
        path = Path(record["path"])
        expected_size = (
            int(np.prod(record["shape"])) * np.dtype(record["dtype"]).itemsize
        )
        if (
            not path.is_file()
            or int(path.stat().st_size) != expected_size
            or int(record.get("size", expected_size)) != expected_size
        ):
            raise ModelError(f"direct_return_label_file_invalid:{path}")


def _direct_return_fingerprint(
    *,
    input_manifest: Mapping[str, Any],
    study_path: Path,
    main_manifest_path: Path,
    short_manifest_path: Path,
) -> str:
    return _stable_hash(
        {
            "schema": "seq100_quality_liquidity_direct_returns/1",
            "model_input_fingerprint": str(input_manifest["input_fingerprint"]),
            "config_sha256": _sha256(study_path),
            "main_label_manifest_sha256": _sha256(main_manifest_path),
            "short_label_manifest_sha256": _sha256(short_manifest_path),
            "feature_variant": COMPACT_VARIANT,
            "feature_count": COMPACT_FEATURE_COUNT,
            "horizons": list(RETURN_HORIZONS),
            "rolling_years": list(ROLLING_YEARS),
            "objective": "regression_l2",
            "rounds": 512,
            "label_transform": {
                "group": "signal_trade_date",
                "winsor_lower": RETURN_WINSOR_LOWER,
                "winsor_upper": RETURN_WINSOR_UPPER,
                "quantile_method": "nearest",
                "zscore_ddof": 0,
                "minimum_cross_section": RETURN_MIN_CROSS_SECTION,
            },
        }
    )


def _label_file_contract(record: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(record["path"])
    return {
        "path": str(path.resolve()),
        "size": int(path.stat().st_size),
        "shape": [int(value) for value in record["shape"]],
        "dtype": str(record["dtype"]),
    }


def _return_label_coverage(inputs: ModelInputs) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for horizon in RETURN_HORIZONS:
        raw_valid = inputs.raw_return_valid_mask(horizon)
        normalized_valid = inputs.return_valid_mask(horizon)
        for year in YEARS:
            rows = inputs.rows_for_year(year)
            raw = raw_valid[rows]
            normalized = normalized_valid[rows]
            valid_dates = np.unique(inputs.date_idx[rows[normalized]])
            records.append(
                {
                    "horizon": int(horizon),
                    "year": int(year),
                    "row_count": len(rows),
                    "raw_valid_count": int(raw.sum()),
                    "normalized_valid_count": int(normalized.sum()),
                    "raw_coverage": float(raw.mean()),
                    "normalized_coverage": float(normalized.mean()),
                    "normalized_date_count": len(valid_dates),
                    "first_normalized_trade_date": (
                        str(inputs.date_values[int(valid_dates[0])])
                        if len(valid_dates)
                        else None
                    ),
                    "last_normalized_trade_date": (
                        str(inputs.date_values[int(valid_dates[-1])])
                        if len(valid_dates)
                        else None
                    ),
                }
            )
    return pd.DataFrame(records).sort_values(["horizon", "year"])


def prepare_returns(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    ready_root: Path = DEFAULT_READY_ROOT,
    audit_root: Path = DEFAULT_AUDIT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    config = _load_config(study_path)
    _audit_manifest(audit_root)
    input_manifest = prepare(
        study_path=study_path,
        ready_root=ready_root,
        audit_root=audit_root,
        output_root=output_root,
    )
    main_manifest_path = Path(input_manifest["label_manifest"]["path"])
    short_manifest_path = DEFAULT_SHORT_LABEL_MANIFEST.resolve()
    main_manifest = _read_json(main_manifest_path)
    short_manifest = _read_json(short_manifest_path)
    _verify_return_label_sources(
        main_manifest=main_manifest, short_manifest=short_manifest
    )
    fingerprint = _direct_return_fingerprint(
        input_manifest=input_manifest,
        study_path=study_path,
        main_manifest_path=main_manifest_path,
        short_manifest_path=short_manifest_path,
    )
    direct_root = _direct_return_root(output_root)
    direct_root.mkdir(parents=True, exist_ok=True)
    contract_path = direct_root / "label_contract.json"
    coverage_path = direct_root / "label_coverage.parquet"
    manifest_path = direct_root / "manifest.json"
    if contract_path.is_file() and coverage_path.is_file():
        existing_contract = _read_json(contract_path)
        if existing_contract.get("experiment_fingerprint") == fingerprint:
            manifest = _read_json(manifest_path)
            if manifest.get("experiment_fingerprint") != fingerprint:
                raise ModelError("direct_return_manifest_fingerprint_mismatch")
            return {
                "status": "prepared",
                "study_id": STUDY_ID,
                "stage": DIRECT_RETURN_DIR_NAME,
                "experiment_fingerprint": fingerprint,
                "task_count": RETURN_TASK_COUNT,
                "feature_count": COMPACT_FEATURE_COUNT,
                "label_coverage": str(coverage_path.resolve()),
                "existing_status": manifest.get("status"),
            }
    inputs = ModelInputs(input_manifest)
    coverage = _return_label_coverage(inputs)
    _write_parquet(coverage, coverage_path)
    label_contract = {
        "schema": "seq100_quality_liquidity_direct_return_labels/1",
        "status": "completed",
        "created_at": _now(),
        "study_id": STUDY_ID,
        "experiment_fingerprint": fingerprint,
        "feature_variant": COMPACT_VARIANT,
        "feature_count": COMPACT_FEATURE_COUNT,
        "horizons": list(RETURN_HORIZONS),
        "signal_timing": "after_signal_day_close",
        "entry": "next_trading_day_open",
        "exit": "D_h_close",
        "raw_label": "gross_simple_return_from_next_open_to_D_h_close",
        "training_transform": {
            "group": "signal_trade_date_cross_section",
            "winsor_quantiles": [RETURN_WINSOR_LOWER, RETURN_WINSOR_UPPER],
            "quantile_method": "nearest",
            "zscore_ddof": 0,
            "minimum_cross_section": RETURN_MIN_CROSS_SECTION,
        },
        "evaluation_target": "original_unstandardized_return",
        "sources": {
            "main_manifest": _file_record(main_manifest_path),
            "main_labels": _label_file_contract(
                main_manifest["files"]["candidate_labels"]
            ),
            "main_flags": _label_file_contract(main_manifest["files"]["label_flags"]),
            "short_manifest": _file_record(short_manifest_path),
            "short_labels": _label_file_contract(
                short_manifest["files"]["short_labels"]
            ),
            "short_flags": _label_file_contract(short_manifest["files"]["flags"]),
        },
        "coverage": _file_record(coverage_path, row_count=len(coverage)),
        "formal_years": list(YEARS),
        "rolling_years": list(ROLLING_YEARS),
        "burn_in_year": 2010,
        "forbidden_year": FORBIDDEN_YEAR,
        "maximum_outcome_date": MAXIMUM_OUTCOME_DATE,
        "training_performed": False,
    }
    _write_json(contract_path, label_contract)
    tasks = _direct_return_task_plan()
    manifest = {
        "schema": "seq100_quality_liquidity_direct_return_manifest/1",
        "study_id": STUDY_ID,
        "stage": DIRECT_RETURN_DIR_NAME,
        "status": "prepared",
        "created_at": _now(),
        "experiment_fingerprint": fingerprint,
        "config": _file_record(study_path),
        "model_inputs": _file_record(
            ready_root / MODEL_INPUT_DIR_NAME / "manifest.json",
            input_fingerprint=str(input_manifest["input_fingerprint"]),
        ),
        "label_contract": _file_record(contract_path),
        "label_coverage": _file_record(coverage_path, row_count=len(coverage)),
        "task_count": len(tasks),
        "tasks": tasks,
        "outputs": {},
        "training_performed": False,
        "feature_set_selected": False,
        "evaluation_semantics": "retrospective_rolling_oos",
        "forbidden_year": FORBIDDEN_YEAR,
        "model_parameters": {
            "objective": "regression",
            "metric": "l2",
            "rounds": int(config["model"]["fixed_mfe_rounds"]),
        },
    }
    _write_json(manifest_path, manifest)
    return {
        "status": "prepared",
        "study_id": STUDY_ID,
        "stage": DIRECT_RETURN_DIR_NAME,
        "experiment_fingerprint": fingerprint,
        "task_count": len(tasks),
        "feature_count": COMPACT_FEATURE_COUNT,
        "label_coverage": str(coverage_path.resolve()),
    }


def _update_return_ledger(
    *,
    direct_root: Path,
    tasks: Sequence[Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Any]],
) -> None:
    entries = []
    for task in tasks:
        task_id = str(task["task_id"])
        result = results.get(task_id)
        entries.append(
            {
                **dict(task),
                "status": "completed" if result is not None else "pending",
                "result_path": (
                    str(_task_result_path(direct_root, task_id).resolve())
                    if result is not None
                    else None
                ),
                "best_iteration": result.get("best_iteration") if result else None,
                "updated_at": _now(),
            }
        )
    _write_json(
        direct_root / "task_ledger.json",
        {
            "schema": "seq100_quality_liquidity_direct_return_ledger/1",
            "study_id": STUDY_ID,
            "task_count": len(tasks),
            "completed_count": sum(item["status"] == "completed" for item in entries),
            "tasks": entries,
        },
    )


def run_returns(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    ready_root: Path = DEFAULT_READY_ROOT,
    audit_root: Path = DEFAULT_AUDIT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    config = _load_config(study_path)
    prepare_returns(
        study_path=study_path,
        ready_root=ready_root,
        audit_root=audit_root,
        output_root=output_root,
    )
    direct_root = _direct_return_root(output_root)
    manifest_path = direct_root / "manifest.json"
    manifest = _read_json(manifest_path)
    input_manifest = _read_json(Path(manifest["model_inputs"]["path"]))
    inputs = ModelInputs(input_manifest)
    tasks = _direct_return_task_plan()
    results, diagnostic = _partition_results(_completed_results(direct_root), tasks)
    if diagnostic:
        raise ModelError(f"unexpected_direct_return_tasks:{sorted(diagnostic)}")
    _update_return_ledger(direct_root=direct_root, tasks=tasks, results=results)
    completed_this_run = 0
    fingerprint = str(manifest["experiment_fingerprint"])
    config_sha256 = _sha256(study_path)
    for task in tasks:
        task_id = str(task["task_id"])
        if task_id in results:
            feature_names, _ = _effective_features(
                inputs=inputs, task=task, output_root=direct_root
            )
            task_fingerprint = _task_fingerprint(
                task=task,
                feature_names=feature_names,
                model_input_fingerprint=str(input_manifest["input_fingerprint"]),
                config_sha256=config_sha256,
                experiment_fingerprint=fingerprint,
            )
            if _task_complete(
                _task_result_path(direct_root, task_id), fingerprint=task_fingerprint
            ):
                continue
            results.pop(task_id)
        if max_tasks is not None and completed_this_run >= int(max_tasks):
            break
        result = _run_training_task(
            task=task,
            inputs=inputs,
            config=config,
            output_root=direct_root,
            config_sha256=config_sha256,
            experiment_fingerprint=fingerprint,
        )
        results[task_id] = result
        completed_this_run += 1
        _update_return_ledger(direct_root=direct_root, tasks=tasks, results=results)
    all_complete = len(results) == len(tasks)
    manifest["status"] = (
        "training_completed" if all_complete else "training_in_progress"
    )
    manifest["updated_at"] = _now()
    manifest["training_performed"] = all_complete
    manifest["feature_set_selected"] = False
    if not all_complete:
        manifest["outputs"] = {}
    _write_json(manifest_path, manifest)
    return {
        "status": manifest["status"],
        "study_id": STUDY_ID,
        "stage": DIRECT_RETURN_DIR_NAME,
        "task_count": len(tasks),
        "completed_count": len(results),
        "completed_this_run": completed_this_run,
        "training_performed": all_complete,
    }


def _return_monthly_metrics(
    results: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results.values():
        daily = pd.read_parquet(Path(result["files"]["daily_metrics"]["path"]))
        daily["month"] = daily["trade_date"].astype(str).str.slice(0, 7)
        metric_columns = [
            column
            for column in daily.select_dtypes(include=[np.number]).columns
            if column not in {"date_idx", "row_count"}
        ]
        for month, part in daily.groupby("month", sort=True):
            row = {
                "task_id": str(result["task_id"]),
                "target": str(result["target"]),
                "horizon": int(result["horizon"]),
                "year": int(result["evaluation_year"]),
                "month": str(month),
                "date_count": len(part),
                "row_count": int(part["row_count"].sum()),
            }
            row.update(
                {
                    column: float(pd.to_numeric(part[column], errors="coerce").mean())
                    for column in metric_columns
                }
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["horizon", "month"])


def _return_horizon_summary(annual: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "rank_ic",
        "top1_return_mean",
        "top5_return_mean",
        "top1_excess_mean",
        "top5_excess_mean",
        "top1_capture",
        "top5_capture",
        "decile_spearman",
        "zscore_mse",
    ]
    rows: list[dict[str, Any]] = []
    for horizon, part in annual.groupby("horizon", sort=True):
        row: dict[str, Any] = {
            "horizon": int(horizon),
            "year_count": len(part),
            "positive_rank_ic_years": int(part["rank_ic"].gt(0.0).sum()),
        }
        for metric in metrics:
            values = pd.to_numeric(part[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_worst"] = float(
                values.max() if metric == "zscore_mse" else values.min()
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("horizon")


def evaluate_returns(
    *, output_root: Path = DEFAULT_OUTPUT_ROOT, study_path: Path = DEFAULT_STUDY_PATH
) -> dict[str, Any]:
    _load_config(study_path)
    direct_root = _direct_return_root(output_root)
    manifest_path = direct_root / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") not in {"training_completed", "evaluated", "audited"}:
        raise ModelError("direct_return_training_not_completed")
    tasks = _direct_return_task_plan()
    results, diagnostic = _partition_results(_completed_results(direct_root), tasks)
    if len(results) != len(tasks) or diagnostic:
        raise ModelError(
            f"direct_return_evaluation_task_count_mismatch:{len(results)}:{len(diagnostic)}"
        )
    annual = _annual_metrics(results)
    monthly = _return_monthly_metrics(results)
    horizon = _return_horizon_summary(annual)
    importance = pd.concat(
        [
            pd.read_parquet(Path(result["files"]["family_importance"]["path"]))
            for result in results.values()
        ],
        ignore_index=True,
    )
    paths = {
        "annual_metrics": direct_root / "annual_metrics.parquet",
        "monthly_metrics": direct_root / "monthly_metrics.parquet",
        "horizon_summary": direct_root / "horizon_summary.parquet",
        "family_importance": direct_root / "family_importance.parquet",
    }
    for frame, path in (
        (annual, paths["annual_metrics"]),
        (monthly, paths["monthly_metrics"]),
        (horizon, paths["horizon_summary"]),
        (importance, paths["family_importance"]),
    ):
        _write_parquet(frame, path)
    manifest["status"] = "evaluated"
    manifest["updated_at"] = _now()
    manifest["training_performed"] = True
    manifest["feature_set_selected"] = False
    manifest["selection_status"] = "pending_horizon_review"
    manifest["outputs"] = {
        name: _file_record(path, row_count=len(frame))
        for name, path, frame in (
            ("annual_metrics", paths["annual_metrics"], annual),
            ("monthly_metrics", paths["monthly_metrics"], monthly),
            ("horizon_summary", paths["horizon_summary"], horizon),
            ("family_importance", paths["family_importance"], importance),
        )
    }
    _write_json(manifest_path, manifest)
    return {
        "status": "evaluated",
        "task_count": len(results),
        "annual_metric_rows": len(annual),
        "monthly_metric_rows": len(monthly),
        "horizon_summary_rows": len(horizon),
        "family_importance_rows": len(importance),
    }


def audit_returns(
    *, output_root: Path = DEFAULT_OUTPUT_ROOT, study_path: Path = DEFAULT_STUDY_PATH
) -> dict[str, Any]:
    _load_config(study_path)
    direct_root = _direct_return_root(output_root)
    manifest_path = direct_root / "manifest.json"
    manifest = _read_json(manifest_path)
    input_manifest = _read_json(Path(manifest["model_inputs"]["path"]))
    _verify_model_input_files(input_manifest, full_hash=False)
    main_manifest_path = Path(input_manifest["label_manifest"]["path"])
    main_manifest = _read_json(main_manifest_path)
    short_manifest = _read_json(DEFAULT_SHORT_LABEL_MANIFEST)
    _verify_return_label_sources(
        main_manifest=main_manifest, short_manifest=short_manifest
    )
    expected_fingerprint = _direct_return_fingerprint(
        input_manifest=input_manifest,
        study_path=study_path,
        main_manifest_path=main_manifest_path,
        short_manifest_path=DEFAULT_SHORT_LABEL_MANIFEST,
    )
    inputs = ModelInputs(input_manifest)
    tasks = _direct_return_task_plan()
    results, diagnostic = _partition_results(_completed_results(direct_root), tasks)
    checks: dict[str, bool] = {
        "manifest_status": manifest.get("status")
        in {"training_completed", "evaluated", "audited"},
        "experiment_fingerprint": str(manifest.get("experiment_fingerprint"))
        == expected_fingerprint,
        "task_count_exact": len(results) == RETURN_TASK_COUNT and not diagnostic,
        "feature_count_exact": len(inputs.feature_groups[COMPACT_VARIANT])
        == COMPACT_FEATURE_COUNT,
        "formal_rows_exclude_2010": not bool((inputs.years == 2010).any()),
        "forbidden_2026_rows": not bool(
            np.char.startswith(inputs.trade_date, "2026-").any()
        ),
        "task_files_valid": True,
        "purge_valid": True,
        "outputs_hashed": True,
    }
    config_sha256 = _sha256(study_path)
    for task in tasks:
        task_id = str(task["task_id"])
        result = results.get(task_id)
        if result is None:
            checks["task_files_valid"] = False
            continue
        try:
            feature_names, _ = _effective_features(
                inputs=inputs, task=task, output_root=direct_root
            )
            fingerprint = _task_fingerprint(
                task=task,
                feature_names=feature_names,
                model_input_fingerprint=str(input_manifest["input_fingerprint"]),
                config_sha256=config_sha256,
                experiment_fingerprint=expected_fingerprint,
            )
            checks["task_files_valid"] &= _task_complete(
                _task_result_path(direct_root, task_id), fingerprint=fingerprint
            )
            prediction = np.load(
                Path(result["files"]["prediction"]["path"]),
                mmap_mode="r",
                allow_pickle=False,
            )
            candidate_ids = np.load(
                Path(result["files"]["candidate_id"]["path"]),
                mmap_mode="r",
                allow_pickle=False,
            )
            expected_rows = inputs.rows_for_year(int(result["evaluation_year"]))
            checks["task_files_valid"] &= np.array_equal(
                candidate_ids, inputs.candidate_ids[expected_rows]
            ) and prediction.shape[0] == len(expected_rows)
            fold = inputs.fold(
                year=int(result["evaluation_year"]),
                horizon=int(result["horizon"]),
                target=str(result["target"]),
            )
            checks["purge_valid"] &= int(
                result["maximum_train_signal_date_idx"]
            ) == int(fold["maximum_train_signal_date_idx"])
        except (ModelError, OSError, KeyError, ValueError, TypeError, IndexError):
            checks["task_files_valid"] = False
    for record in dict(manifest.get("outputs", {}) or {}).values():
        path = Path(record["path"])
        checks["outputs_hashed"] &= path.is_file() and _sha256(path) == record["sha256"]
    payload = {
        "schema": "seq100_quality_liquidity_direct_return_audit/1",
        "study_id": STUDY_ID,
        "stage": DIRECT_RETURN_DIR_NAME,
        "status": "ok" if all(checks.values()) else "failed",
        "created_at": _now(),
        "checks": checks,
        "task_count": len(results),
        "horizons": list(RETURN_HORIZONS),
        "rolling_years": list(ROLLING_YEARS),
        "training_performed": True,
        "evaluation_semantics": "retrospective_rolling_oos",
    }
    audit_path = direct_root / "audit.json"
    _write_json(audit_path, payload)
    if payload["status"] != "ok":
        raise ModelError("direct_return_audit_failed")
    manifest["status"] = "audited"
    manifest["audit"] = _file_record(audit_path)
    manifest["updated_at"] = _now()
    _write_json(manifest_path, manifest)
    return payload


def _close_d1_root(output_root: Path) -> Path:
    return _direct_return_root(output_root) / CLOSE_D1_DIR_NAME


def _close_d1_source_context(*, input_manifest: Mapping[str, Any]) -> dict[str, Any]:
    training_manifest_path = Path(
        input_manifest["source"]["training_ready_manifest"]["path"]
    ).resolve()
    training_manifest = _read_json(training_manifest_path)
    dataset_id = str(
        dict(training_manifest.get("qdp_dataset_ids", {}) or {}).get(
            "market_intraday_5m", ""
        )
    )
    if not dataset_id.startswith("market_intraday_5m__"):
        raise ModelError("close_d1_pinned_intraday_dataset_missing")
    qdp_root = qdp_v2_root(WORKSPACE_ROOT).resolve()
    intraday_manifest_path = (
        qdp_root / "datasets" / "market_intraday_5m" / dataset_id / "dataset.json"
    )
    intraday_manifest = _read_json(intraday_manifest_path)
    if str(intraday_manifest.get("dataset_id")) != dataset_id:
        raise ModelError("close_d1_intraday_dataset_id_mismatch")
    label_manifest_path = Path(input_manifest["label_manifest"]["path"]).resolve()
    label_manifest = _read_json(label_manifest_path)
    pack_manifest_path = Path(label_manifest["source"]["pack_manifest"]).resolve()
    pack_manifest = _read_json(pack_manifest_path)
    return {
        "training_manifest_path": training_manifest_path,
        "training_manifest": training_manifest,
        "label_manifest_path": label_manifest_path,
        "label_manifest": label_manifest,
        "pack_manifest_path": pack_manifest_path,
        "pack_manifest": pack_manifest,
        "qdp_root": qdp_root,
        "intraday_dataset_id": dataset_id,
        "intraday_manifest_path": intraday_manifest_path,
        "intraday_manifest": intraday_manifest,
    }


def _close_d1_experiment_fingerprint(
    *,
    input_manifest: Mapping[str, Any],
    study_path: Path,
    sources: Mapping[str, Any],
) -> str:
    return _stable_hash(
        {
            "schema": "seq100_quality_liquidity_close_d1/1",
            "model_input_fingerprint": str(input_manifest["input_fingerprint"]),
            "config_sha256": _sha256(study_path),
            "pack_manifest_sha256": _sha256(Path(sources["pack_manifest_path"])),
            "short_label_manifest_sha256": _sha256(DEFAULT_SHORT_LABEL_MANIFEST),
            "intraday_dataset_id": str(sources["intraday_dataset_id"]),
            "intraday_manifest_sha256": _sha256(
                Path(sources["intraday_manifest_path"])
            ),
            "feature_variant": COMPACT_VARIANT,
            "feature_count": COMPACT_FEATURE_COUNT,
            "rolling_years": list(ROLLING_YEARS),
            "objective": "regression_l2",
            "rounds": 512,
            "label": "signal_close_to_next_close_back_adjusted_return",
            "valid_support": "existing_g_1_flag_16",
            "label_transform": {
                "group": "signal_trade_date",
                "winsor_lower": RETURN_WINSOR_LOWER,
                "winsor_upper": RETURN_WINSOR_UPPER,
                "quantile_method": "nearest",
                "zscore_ddof": 0,
                "minimum_cross_section": RETURN_MIN_CROSS_SECTION,
            },
            "execution_outcomes": [
                "signal_close_to_next_close",
                "next_open_to_next_close",
                "next_first_5m_vwap_to_next_close",
            ],
        }
    )


def _record_file_valid(record: Mapping[str, Any], *, verify_hash: bool = True) -> bool:
    path = Path(record["path"])
    if not path.is_file() or int(path.stat().st_size) != int(record["size"]):
        return False
    return not verify_hash or _sha256(path) == str(record["sha256"])


def _build_close_d1_label_cache(
    *,
    inputs: ModelInputs,
    pack_manifest: Mapping[str, Any],
    output_root: Path,
    experiment_fingerprint: str,
) -> dict[str, Any]:
    arrays = dict(pack_manifest.get("label_arrays", {}) or {})
    path_contract = dict(arrays.get("future_ohlcva_path", {}) or {})
    fields = [str(value) for value in path_contract.get("fields", ())]
    expected_shape = (
        len(inputs.date_values),
        len(pack_manifest.get("symbol_values", ())),
        60,
        6,
    )
    if (
        tuple(int(value) for value in path_contract.get("shape", ())) != expected_shape
        or fields != ["open", "high", "low", "close", "volume", "amount"]
        or str(path_contract.get("anchor")) != "signal_day_close"
    ):
        raise ModelError("close_d1_future_path_contract_mismatch")
    price_semantics = str(
        dict(pack_manifest.get("label_semantics", {}) or {}).get("future_ohlc_path", "")
    )
    if "back_adjust" not in price_semantics:
        raise ModelError("close_d1_future_path_adjustment_mismatch")

    label_root = output_root / "labels"
    label_root.mkdir(parents=True, exist_ok=True)
    values_path = label_root / "close_to_close_d1.float32.dat"
    valid_path = label_root / "close_to_close_d1_valid.uint8.dat"
    values_partial = Path(str(values_path) + ".partial")
    valid_partial = Path(str(valid_path) + ".partial")
    values_partial.unlink(missing_ok=True)
    valid_partial.unlink(missing_ok=True)
    values = np.memmap(
        values_partial, dtype=np.float32, mode="w+", shape=(inputs.row_count,)
    )
    valid_store = np.memmap(
        valid_partial, dtype=np.uint8, mode="w+", shape=(inputs.row_count,)
    )
    values[:] = np.nan
    valid_store[:] = 0

    symbols = pd.Index([str(value) for value in pack_manifest["symbol_values"]])
    symbol_idx = symbols.get_indexer(inputs.row_index["symbol"].astype(str))
    if bool((symbol_idx < 0).any()):
        raise ModelError("close_d1_symbol_not_in_path_pack")
    support = np.asarray(inputs.raw_return_valid_mask(1), dtype=bool)
    g_1 = np.asarray(inputs.raw_return_values(1), dtype=np.float32)
    outcome_idx = inputs.date_idx[support] + 1
    if (
        not bool(support.any())
        or int(outcome_idx.max()) >= len(inputs.date_values)
        or str(inputs.date_values[int(outcome_idx.max())]) > MAXIMUM_OUTCOME_DATE
        or bool(
            np.char.startswith(
                inputs.date_values[outcome_idx].astype(str), "2026-"
            ).any()
        )
    ):
        raise ModelError("close_d1_outcome_boundary_violation")

    extracted_count = 0
    maximum_identity_error = 0.0
    source_shard_count = 0
    for shard in path_contract.get("shards", ()):
        shard = dict(shard)
        start = int(shard["date_start_idx"])
        end = int(shard["date_end_idx"])
        left = int(np.searchsorted(inputs.date_idx, start, side="left"))
        right = int(np.searchsorted(inputs.date_idx, end, side="right"))
        if left >= right:
            continue
        rows = np.arange(left, right, dtype=np.int64)
        rows = rows[support[rows]]
        if not len(rows):
            continue
        shard_path = Path(shard["path"])
        shape = tuple(int(value) for value in shard["shape"])
        if not shard_path.is_file() or int(shard_path.stat().st_size) != int(
            np.prod(shape) * np.dtype(np.float32).itemsize
        ):
            raise ModelError(f"close_d1_path_shard_invalid:{shard_path}")
        source = np.memmap(shard_path, dtype=np.float32, mode="r", shape=shape)
        local_date = inputs.date_idx[rows] - start
        local_symbol = symbol_idx[rows]
        open_return = np.asarray(
            source[local_date, local_symbol, 0, fields.index("open")],
            dtype=np.float32,
        )
        close_return = np.asarray(
            source[local_date, local_symbol, 0, fields.index("close")],
            dtype=np.float32,
        )
        reconstructed = (1.0 + close_return.astype(np.float64)) / (
            1.0 + open_return.astype(np.float64)
        ) - 1.0
        identity_error = np.abs(reconstructed - g_1[rows].astype(np.float64))
        valid_values = (
            np.isfinite(open_return)
            & np.isfinite(close_return)
            & np.isfinite(identity_error)
            & (open_return > -1.0)
            & (close_return > -1.0)
        )
        if not bool(valid_values.all()) or bool((identity_error > 1.0e-5).any()):
            raise ModelError(
                f"close_d1_path_identity_failed:{int((~valid_values).sum())}:"
                f"{float(np.nanmax(identity_error))}"
            )
        values[rows] = close_return
        valid_store[rows] = 1
        extracted_count += len(rows)
        maximum_identity_error = max(
            maximum_identity_error, float(identity_error.max(initial=0.0))
        )
        source_shard_count += 1
        del source

    if extracted_count != int(support.sum()) or int(valid_store.sum()) != int(
        support.sum()
    ):
        raise ModelError(
            f"close_d1_support_not_fully_extracted:{extracted_count}:"
            f"{int(support.sum())}"
        )
    values.flush()
    valid_store.flush()
    del values, valid_store
    os.replace(values_partial, values_path)
    os.replace(valid_partial, valid_path)
    contract = {
        "schema": "seq100_quality_liquidity_close_d1_labels/1",
        "status": "completed",
        "created_at": _now(),
        "experiment_fingerprint": experiment_fingerprint,
        "row_count": inputs.row_count,
        "valid_count": int(support.sum()),
        "invalid_count": int((~support).sum()),
        "valid_support": "short_label_g_1_flag_16_and_finite",
        "raw_label": "back_adjusted_next_close_over_signal_close_minus_one",
        "path_day_index": 0,
        "path_field_index": fields.index("close"),
        "source_shard_count": source_shard_count,
        "source_logical_row_read_count": extracted_count,
        "maximum_open_close_identity_error": maximum_identity_error,
        "maximum_signal_trade_date_read": str(inputs.trade_date[support][-1]),
        "maximum_outcome_date_read": str(inputs.date_values[int(outcome_idx.max())]),
        "forbidden_2026_read_count": 0,
        "files": {
            "values": _file_record(
                values_path,
                shape=[inputs.row_count],
                dtype=str(np.dtype(np.float32)),
            ),
            "valid": _file_record(
                valid_path,
                shape=[inputs.row_count],
                dtype=str(np.dtype(np.uint8)),
            ),
        },
    }
    return contract


def _intraday_paths_for_period(
    *, sources: Mapping[str, Any], start_date: str, end_date: str
) -> list[str]:
    paths: list[str] = []
    qdp_root = Path(sources["qdp_root"])
    for record in sources["intraday_manifest"].get("shards", ()):
        record = dict(record)
        if (
            str(record.get("end_date", "")) < start_date
            or str(record.get("start_date", "")) > end_date
        ):
            continue
        path = (qdp_root / str(record["path"])).resolve()
        if not path.is_file():
            raise ModelError(f"close_d1_intraday_shard_missing:{path}")
        paths.append(str(path))
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise ModelError("close_d1_intraday_period_has_no_shards")
    return paths


def _validate_vwap_execution_frame(
    frame: pd.DataFrame, *, existing_open_to_close: np.ndarray
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = frame.copy()
    first_count = pd.to_numeric(result["first_count"], errors="coerce").fillna(0)
    last_count = pd.to_numeric(result["last_count"], errors="coerce").fillna(0)
    unique_bars = first_count.eq(1) & last_count.eq(1)
    numeric_columns = [
        "first_open",
        "first_high",
        "first_low",
        "first_volume",
        "first_amount",
        "last_close",
    ]
    numeric = np.column_stack(
        [pd.to_numeric(result[column], errors="coerce") for column in numeric_columns]
    )
    finite = np.isfinite(numeric).all(axis=1)
    first_open = numeric[:, 0]
    first_high = numeric[:, 1]
    first_low = numeric[:, 2]
    first_volume = numeric[:, 3]
    first_amount = numeric[:, 4]
    last_close = numeric[:, 5]
    valid_numeric = (
        finite
        & (first_open > 0.0)
        & (first_low > 0.0)
        & (first_high >= first_low)
        & (first_open >= first_low * (1.0 - 1.0e-8))
        & (first_open <= first_high * (1.0 + 1.0e-8))
        & (first_volume > 0.0)
        & (first_amount > 0.0)
        & (last_close > 0.0)
    )
    vwap = np.divide(
        first_amount,
        first_volume,
        out=np.full(len(result), np.nan, dtype=np.float64),
        where=first_volume > 0.0,
    )
    strict_inside = (
        valid_numeric
        & (vwap >= first_low * (1.0 - 1.0e-8))
        & (vwap <= first_high * (1.0 + 1.0e-8))
    )
    within_tolerance = (
        valid_numeric
        & (vwap >= first_low * (1.0 - VWAP_OHLC_RELATIVE_TOLERANCE))
        & (vwap <= first_high * (1.0 + VWAP_OHLC_RELATIVE_TOLERANCE))
    )
    minute_open_to_close = (
        np.divide(
            last_close,
            first_open,
            out=np.full(len(result), np.nan, dtype=np.float64),
            where=first_open > 0.0,
        )
        - 1.0
    )
    existing = np.asarray(existing_open_to_close, dtype=np.float64)
    if existing.shape != (len(result),):
        raise ModelError("close_d1_open_return_reconciliation_shape_mismatch")
    reconciliation_error = np.abs(minute_open_to_close - existing)
    reconciled = np.isfinite(reconciliation_error) & (
        reconciliation_error <= VWAP_OHLC_RELATIVE_TOLERANCE
    )
    valid = (
        unique_bars.to_numpy(dtype=bool) & valid_numeric & within_tolerance & reconciled
    )
    state = np.full(len(result), "observed", dtype=object)
    state[~unique_bars.to_numpy(dtype=bool)] = "missing_or_duplicate_bar"
    state[unique_bars.to_numpy(dtype=bool) & ~valid_numeric] = "invalid_numeric_bar"
    state[unique_bars.to_numpy(dtype=bool) & valid_numeric & ~within_tolerance] = (
        "vwap_outside_ohlc_tolerance"
    )
    state[
        unique_bars.to_numpy(dtype=bool)
        & valid_numeric
        & within_tolerance
        & ~reconciled
    ] = "open_close_reconciliation_failed"
    vwap_return = (
        np.divide(
            last_close,
            vwap,
            out=np.full(len(result), np.nan, dtype=np.float64),
            where=valid,
        )
        - 1.0
    )
    result["first_5m_vwap"] = vwap.astype(np.float32)
    result["next_close_raw"] = last_close.astype(np.float32)
    result["minute_open_to_close_return"] = minute_open_to_close.astype(np.float32)
    result["existing_open_to_close_return"] = existing.astype(np.float32)
    result["open_close_reconciliation_abs_error"] = reconciliation_error.astype(
        np.float32
    )
    result["first_5m_vwap_to_close_return"] = vwap_return.astype(np.float32)
    result["vwap_strictly_inside_ohlc"] = strict_inside
    result["outcome_valid"] = valid
    result["source_state"] = state
    coverage = float(valid.mean()) if len(valid) else 0.0
    finite_errors = reconciliation_error[np.isfinite(reconciliation_error)]
    validation = {
        "row_count": len(result),
        "valid_count": int(valid.sum()),
        "coverage": coverage,
        "first_bar_duplicate_count": int((first_count > 1).sum()),
        "last_bar_duplicate_count": int((last_count > 1).sum()),
        "missing_or_duplicate_count": int((~unique_bars).sum()),
        "invalid_numeric_count": int((unique_bars & ~valid_numeric).sum()),
        "strict_inside_count": int(strict_inside.sum()),
        "vwap_outside_tolerance_count": int((valid_numeric & ~within_tolerance).sum()),
        "open_close_reconciliation_failure_count": int(
            (valid_numeric & within_tolerance & ~reconciled).sum()
        ),
        "maximum_open_close_reconciliation_abs_error": (
            float(finite_errors.max()) if len(finite_errors) else None
        ),
        "source_validation_passed": bool(
            coverage >= MINIMUM_VWAP_COVERAGE
            and not bool((first_count > 1).any())
            and not bool((last_count > 1).any())
            and not bool((valid_numeric & ~within_tolerance).any())
            and not bool((valid_numeric & within_tolerance & ~reconciled).any())
        ),
    }
    return result, validation


def _build_vwap_execution_cache(
    *,
    inputs: ModelInputs,
    sources: Mapping[str, Any],
    output_root: Path,
    experiment_fingerprint: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    support = np.asarray(inputs.raw_return_valid_mask(1), dtype=bool)
    oos = np.isin(inputs.years, np.asarray(ROLLING_YEARS, dtype=np.int16))
    rows = np.flatnonzero(support & oos).astype(np.int64, copy=False)
    entry_idx = inputs.date_idx[rows] + 1
    entry_dates = inputs.date_values[entry_idx].astype(str)
    if (
        not len(rows)
        or str(entry_dates[-1]) > MAXIMUM_OUTCOME_DATE
        or bool(np.char.startswith(entry_dates, "2026-").any())
    ):
        raise ModelError("close_d1_vwap_outcome_boundary_violation")
    keys = pd.DataFrame(
        {
            "local_row": rows,
            "candidate_id": inputs.candidate_ids[rows],
            "signal_trade_date": inputs.trade_date[rows],
            "signal_year": inputs.years[rows],
            "symbol": inputs.row_index.loc[rows, "symbol"].astype(str).to_numpy(),
            "entry_trade_date": entry_dates,
        }
    )
    start_date = str(entry_dates[0])
    end_date = str(entry_dates[-1])
    paths = _intraday_paths_for_period(
        sources=sources, start_date=start_date, end_date=end_date
    )
    connection = duckdb.connect()
    connection.execute("PRAGMA threads=8")
    connection.execute("PRAGMA memory_limit='8GB'")
    connection.register("close_d1_keys", keys)
    connection.read_parquet(paths).filter(
        f"trade_date BETWEEN '{start_date}' AND '{end_date}' "
        f"AND bar_time IN ('{FIRST_5M_BAR_TIME}','{LAST_5M_BAR_TIME}')"
    ).project(
        "symbol, trade_date, bar_time, open, high, low, close, volume, amount, source"
    ).create_view("close_d1_bars")
    bars = connection.execute(
        f"""
        SELECT
            k.local_row,
            count(*) FILTER (WHERE b.bar_time='{FIRST_5M_BAR_TIME}') AS first_count,
            count(*) FILTER (WHERE b.bar_time='{LAST_5M_BAR_TIME}') AS last_count,
            max(b.open) FILTER (WHERE b.bar_time='{FIRST_5M_BAR_TIME}') AS first_open,
            max(b.high) FILTER (WHERE b.bar_time='{FIRST_5M_BAR_TIME}') AS first_high,
            max(b.low) FILTER (WHERE b.bar_time='{FIRST_5M_BAR_TIME}') AS first_low,
            max(b.volume) FILTER (WHERE b.bar_time='{FIRST_5M_BAR_TIME}') AS first_volume,
            max(b.amount) FILTER (WHERE b.bar_time='{FIRST_5M_BAR_TIME}') AS first_amount,
            max(b.close) FILTER (WHERE b.bar_time='{LAST_5M_BAR_TIME}') AS last_close,
            max(b.source) FILTER (WHERE b.bar_time='{FIRST_5M_BAR_TIME}') AS first_source,
            max(b.source) FILTER (WHERE b.bar_time='{LAST_5M_BAR_TIME}') AS last_source
        FROM close_d1_keys k
        LEFT JOIN close_d1_bars b
          ON b.symbol=k.symbol AND b.trade_date=k.entry_trade_date
        GROUP BY k.local_row
        ORDER BY k.local_row
        """
    ).fetchdf()
    connection.close()
    if not np.array_equal(bars["local_row"].to_numpy(dtype=np.int64), rows):
        raise ModelError("close_d1_vwap_query_row_alignment_failed")
    frame = keys.merge(bars, on="local_row", how="left", validate="one_to_one")
    frame, validation = _validate_vwap_execution_frame(
        frame, existing_open_to_close=inputs.raw_return_values(1)[rows]
    )
    validation.update(
        {
            "schema": "seq100_quality_liquidity_close_d1_vwap_validation/1",
            "status": ("ok" if validation["source_validation_passed"] else "failed"),
            "created_at": _now(),
            "experiment_fingerprint": experiment_fingerprint,
            "intraday_dataset_id": str(sources["intraday_dataset_id"]),
            "logical_date_start": start_date,
            "logical_date_end": end_date,
            "first_bar_time": FIRST_5M_BAR_TIME,
            "last_bar_time": LAST_5M_BAR_TIME,
            "vwap_definition": "first_5m_amount_divided_by_first_5m_volume",
            "vwap_ohlc_relative_tolerance": VWAP_OHLC_RELATIVE_TOLERANCE,
            "minimum_coverage": MINIMUM_VWAP_COVERAGE,
            "parquet_shard_count": len(paths),
            "forbidden_2026_query_row_count": 0,
        }
    )
    if not validation["source_validation_passed"]:
        raise ModelError("close_d1_vwap_source_validation_failed")
    outcomes_path = output_root / "execution_outcomes.parquet"
    coverage_path = output_root / "execution_coverage.parquet"
    validation_path = output_root / "minute_source_validation.json"
    output_columns = [
        "local_row",
        "candidate_id",
        "signal_trade_date",
        "signal_year",
        "symbol",
        "entry_trade_date",
        "first_count",
        "last_count",
        "first_open",
        "first_high",
        "first_low",
        "first_volume",
        "first_amount",
        "first_5m_vwap",
        "next_close_raw",
        "minute_open_to_close_return",
        "existing_open_to_close_return",
        "open_close_reconciliation_abs_error",
        "first_5m_vwap_to_close_return",
        "vwap_strictly_inside_ohlc",
        "outcome_valid",
        "source_state",
        "first_source",
        "last_source",
    ]
    _write_parquet(frame[output_columns], outcomes_path)
    coverage = (
        frame.groupby("signal_year", as_index=False)
        .agg(
            row_count=("local_row", "size"),
            valid_count=("outcome_valid", "sum"),
            strict_inside_count=("vwap_strictly_inside_ohlc", "sum"),
        )
        .sort_values("signal_year")
    )
    coverage["coverage"] = coverage["valid_count"] / coverage["row_count"]
    _write_parquet(coverage, coverage_path)
    _write_json(validation_path, validation)
    files = {
        "execution_outcomes": _file_record(outcomes_path, row_count=len(frame)),
        "execution_coverage": _file_record(coverage_path, row_count=len(coverage)),
        "minute_source_validation": _file_record(validation_path),
    }
    return files, validation


def _close_d1_label_coverage(inputs: CloseD1ModelInputs) -> pd.DataFrame:
    raw_valid = np.asarray(inputs.raw_return_valid_mask(1), dtype=bool)
    normalized_valid = np.asarray(inputs.return_valid_mask(1), dtype=bool)
    records: list[dict[str, Any]] = []
    for year in YEARS:
        rows = inputs.rows_for_year(year)
        valid_dates = np.unique(inputs.date_idx[rows[normalized_valid[rows]]])
        records.append(
            {
                "year": int(year),
                "row_count": len(rows),
                "raw_valid_count": int(raw_valid[rows].sum()),
                "normalized_valid_count": int(normalized_valid[rows].sum()),
                "raw_coverage": float(raw_valid[rows].mean()),
                "normalized_coverage": float(normalized_valid[rows].mean()),
                "normalized_date_count": len(valid_dates),
                "first_normalized_trade_date": (
                    str(inputs.date_values[int(valid_dates[0])])
                    if len(valid_dates)
                    else None
                ),
                "last_normalized_trade_date": (
                    str(inputs.date_values[int(valid_dates[-1])])
                    if len(valid_dates)
                    else None
                ),
            }
        )
    return pd.DataFrame(records).sort_values("year")


def prepare_close_d1(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    ready_root: Path = DEFAULT_READY_ROOT,
    audit_root: Path = DEFAULT_AUDIT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    config = _load_config(study_path)
    _audit_manifest(audit_root)
    prepare(
        study_path=study_path,
        ready_root=ready_root,
        audit_root=audit_root,
        output_root=output_root,
    )
    input_manifest_path = ready_root / MODEL_INPUT_DIR_NAME / "manifest.json"
    input_manifest = _read_json(input_manifest_path)
    sources = _close_d1_source_context(input_manifest=input_manifest)
    fingerprint = _close_d1_experiment_fingerprint(
        input_manifest=input_manifest, study_path=study_path, sources=sources
    )
    close_root = _close_d1_root(output_root)
    close_root.mkdir(parents=True, exist_ok=True)
    manifest_path = close_root / "manifest.json"
    label_contract_path = close_root / "label_contract.json"
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        if str(existing.get("experiment_fingerprint")) != fingerprint:
            raise ModelError("close_d1_existing_experiment_fingerprint_mismatch")
        if label_contract_path.is_file():
            contract = _read_json(label_contract_path)
            preparation_files = dict(existing.get("preparation_files", {}) or {})
            if (
                str(contract.get("experiment_fingerprint")) == fingerprint
                and all(
                    _record_file_valid(record)
                    for record in dict(contract.get("files", {}) or {}).values()
                )
                and all(
                    _record_file_valid(record) for record in preparation_files.values()
                )
            ):
                return {
                    "status": "prepared",
                    "study_id": STUDY_ID,
                    "stage": CLOSE_D1_DIR_NAME,
                    "experiment_fingerprint": fingerprint,
                    "task_count": CLOSE_D1_TASK_COUNT,
                    "feature_count": COMPACT_FEATURE_COUNT,
                    "existing_status": existing.get("status"),
                }

    base_inputs = ModelInputs(input_manifest)
    label_contract = _build_close_d1_label_cache(
        inputs=base_inputs,
        pack_manifest=sources["pack_manifest"],
        output_root=close_root,
        experiment_fingerprint=fingerprint,
    )
    label_contract["sources"] = {
        "pack_manifest": _file_record(Path(sources["pack_manifest_path"])),
        "short_label_manifest": _file_record(DEFAULT_SHORT_LABEL_MANIFEST),
    }
    label_contract["training_transform"] = {
        "group": "signal_trade_date_cross_section",
        "winsor_quantiles": [RETURN_WINSOR_LOWER, RETURN_WINSOR_UPPER],
        "quantile_method": "nearest",
        "zscore_ddof": 0,
        "minimum_cross_section": RETURN_MIN_CROSS_SECTION,
    }
    _write_json(label_contract_path, label_contract)
    close_inputs = CloseD1ModelInputs(input_manifest, label_contract)
    coverage = _close_d1_label_coverage(close_inputs)
    coverage_path = close_root / "label_coverage.parquet"
    _write_parquet(coverage, coverage_path)
    execution_files, source_validation = _build_vwap_execution_cache(
        inputs=base_inputs,
        sources=sources,
        output_root=close_root,
        experiment_fingerprint=fingerprint,
    )
    preparation_files = {
        "label_coverage": _file_record(coverage_path, row_count=len(coverage)),
        **execution_files,
    }
    tasks = _close_d1_task_plan()
    manifest = {
        "schema": "seq100_quality_liquidity_close_d1_manifest/1",
        "study_id": STUDY_ID,
        "stage": CLOSE_D1_DIR_NAME,
        "status": "prepared",
        "created_at": _now(),
        "experiment_fingerprint": fingerprint,
        "config": _file_record(study_path),
        "model_inputs": _file_record(
            input_manifest_path,
            input_fingerprint=str(input_manifest["input_fingerprint"]),
        ),
        "label_contract": _file_record(label_contract_path),
        "preparation_files": preparation_files,
        "intraday_dataset": {
            "dataset_id": str(sources["intraday_dataset_id"]),
            "manifest": _file_record(Path(sources["intraday_manifest_path"])),
            "source_validation_passed": bool(
                source_validation["source_validation_passed"]
            ),
        },
        "baseline_direct_returns_root": str(_direct_return_root(output_root).resolve()),
        "task_count": len(tasks),
        "tasks": tasks,
        "outputs": {},
        "training_performed": False,
        "feature_set_selected": False,
        "evaluation_semantics": "retrospective_rolling_oos",
        "formal_years": list(YEARS),
        "rolling_years": list(ROLLING_YEARS),
        "burn_in_year": 2010,
        "forbidden_year": FORBIDDEN_YEAR,
        "maximum_outcome_date": MAXIMUM_OUTCOME_DATE,
        "model_parameters": {
            "objective": "regression",
            "metric": "l2",
            "rounds": int(config["model"]["fixed_mfe_rounds"]),
            "feature_count": COMPACT_FEATURE_COUNT,
        },
    }
    _write_json(manifest_path, manifest)
    return {
        "status": "prepared",
        "study_id": STUDY_ID,
        "stage": CLOSE_D1_DIR_NAME,
        "experiment_fingerprint": fingerprint,
        "task_count": len(tasks),
        "feature_count": COMPACT_FEATURE_COUNT,
        "label_valid_count": int(label_contract["valid_count"]),
        "vwap_valid_count": int(source_validation["valid_count"]),
    }


def _update_close_d1_ledger(
    *,
    close_root: Path,
    tasks: Sequence[Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Any]],
) -> None:
    entries = []
    for task in tasks:
        task_id = str(task["task_id"])
        result = results.get(task_id)
        entries.append(
            {
                **dict(task),
                "status": "completed" if result is not None else "pending",
                "result_path": (
                    str(_task_result_path(close_root, task_id).resolve())
                    if result is not None
                    else None
                ),
                "best_iteration": result.get("best_iteration") if result else None,
                "updated_at": _now(),
            }
        )
    _write_json(
        close_root / "task_ledger.json",
        {
            "schema": "seq100_quality_liquidity_close_d1_ledger/1",
            "study_id": STUDY_ID,
            "task_count": len(tasks),
            "completed_count": sum(item["status"] == "completed" for item in entries),
            "tasks": entries,
        },
    )


def run_close_d1(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    ready_root: Path = DEFAULT_READY_ROOT,
    audit_root: Path = DEFAULT_AUDIT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    config = _load_config(study_path)
    prepare_close_d1(
        study_path=study_path,
        ready_root=ready_root,
        audit_root=audit_root,
        output_root=output_root,
    )
    close_root = _close_d1_root(output_root)
    manifest_path = close_root / "manifest.json"
    manifest = _read_json(manifest_path)
    input_manifest = _read_json(Path(manifest["model_inputs"]["path"]))
    label_contract = _read_json(Path(manifest["label_contract"]["path"]))
    inputs = CloseD1ModelInputs(input_manifest, label_contract)
    tasks = _close_d1_task_plan()
    results, diagnostic = _partition_results(_completed_results(close_root), tasks)
    if diagnostic:
        raise ModelError(f"unexpected_close_d1_tasks:{sorted(diagnostic)}")
    _update_close_d1_ledger(close_root=close_root, tasks=tasks, results=results)
    completed_this_run = 0
    fingerprint = str(manifest["experiment_fingerprint"])
    config_sha256 = _sha256(study_path)
    for task in tasks:
        task_id = str(task["task_id"])
        if task_id in results:
            feature_names, _ = _effective_features(
                inputs=inputs, task=task, output_root=close_root
            )
            task_fingerprint = _task_fingerprint(
                task=task,
                feature_names=feature_names,
                model_input_fingerprint=str(input_manifest["input_fingerprint"]),
                config_sha256=config_sha256,
                experiment_fingerprint=fingerprint,
            )
            if _task_complete(
                _task_result_path(close_root, task_id), fingerprint=task_fingerprint
            ):
                continue
            results.pop(task_id)
        if max_tasks is not None and completed_this_run >= int(max_tasks):
            break
        result = _run_training_task(
            task=task,
            inputs=inputs,
            config=config,
            output_root=close_root,
            config_sha256=config_sha256,
            experiment_fingerprint=fingerprint,
        )
        results[task_id] = result
        completed_this_run += 1
        _update_close_d1_ledger(close_root=close_root, tasks=tasks, results=results)
    all_complete = len(results) == len(tasks)
    manifest["status"] = (
        "training_completed" if all_complete else "training_in_progress"
    )
    manifest["updated_at"] = _now()
    manifest["training_performed"] = all_complete
    if not all_complete:
        manifest["outputs"] = {}
    _write_json(manifest_path, manifest)
    return {
        "status": manifest["status"],
        "study_id": STUDY_ID,
        "stage": CLOSE_D1_DIR_NAME,
        "task_count": len(tasks),
        "completed_count": len(results),
        "completed_this_run": completed_this_run,
        "training_performed": all_complete,
    }


def _baseline_d1_results(
    *,
    output_root: Path,
    inputs: ModelInputs,
    study_path: Path,
) -> dict[int, dict[str, Any]]:
    direct_root = _direct_return_root(output_root)
    direct_manifest = _read_json(direct_root / "manifest.json")
    if direct_manifest.get("status") not in {
        "training_completed",
        "evaluated",
        "audited",
    }:
        raise ModelError("close_d1_baseline_direct_returns_not_completed")
    direct_tasks = _direct_return_task_plan()
    results, diagnostic = _partition_results(
        _completed_results(direct_root), direct_tasks
    )
    if len(results) != RETURN_TASK_COUNT or diagnostic:
        raise ModelError("close_d1_baseline_direct_return_task_count_mismatch")
    selected: dict[int, dict[str, Any]] = {}
    for task in direct_tasks:
        if int(task["horizon"]) != 1:
            continue
        result = results[str(task["task_id"])]
        feature_names, _ = _effective_features(
            inputs=inputs, task=task, output_root=direct_root
        )
        fingerprint = _task_fingerprint(
            task=task,
            feature_names=feature_names,
            model_input_fingerprint=str(inputs.manifest["input_fingerprint"]),
            config_sha256=_sha256(study_path),
            experiment_fingerprint=str(direct_manifest["experiment_fingerprint"]),
        )
        if not _task_complete(
            _task_result_path(direct_root, str(task["task_id"])),
            fingerprint=fingerprint,
        ):
            raise ModelError(f"close_d1_baseline_task_invalid:{task['task_id']}")
        selected[int(task["year"])] = result
    if set(selected) != set(ROLLING_YEARS):
        raise ModelError("close_d1_baseline_d1_years_missing")
    return selected


def _load_year_prediction(
    *, result: Mapping[str, Any], inputs: ModelInputs, year: int
) -> tuple[np.ndarray, np.ndarray]:
    rows = inputs.rows_for_year(year)
    candidate_ids = np.load(
        Path(result["files"]["candidate_id"]["path"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    prediction = np.load(
        Path(result["files"]["prediction"]["path"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    if not np.array_equal(candidate_ids, inputs.candidate_ids[rows]) or len(
        prediction
    ) != len(rows):
        raise ModelError(f"close_d1_prediction_alignment_failed:{year}")
    return rows, np.asarray(prediction)


def _cross_outcome_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["month"] = frame["trade_date"].astype(str).str.slice(0, 7)
    excluded = {"date_idx", "row_count", "evaluation_year"}
    metric_columns = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]
    rows: list[dict[str, Any]] = []
    group_columns = ["model_training_target", "outcome", "evaluation_year", "month"]
    for keys, part in frame.groupby(group_columns, sort=True):
        model_target, outcome, year, month = keys
        record: dict[str, Any] = {
            "model_training_target": str(model_target),
            "outcome": str(outcome),
            "evaluation_year": int(year),
            "month": str(month),
            "date_count": len(part),
            "row_count": int(part["row_count"].sum()),
        }
        record.update(
            {
                column: float(pd.to_numeric(part[column], errors="coerce").mean())
                for column in metric_columns
            }
        )
        rows.append(record)
    return pd.DataFrame(rows).sort_values(group_columns)


def _paired_model_deltas(annual: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "rank_ic",
        "top1_return_mean",
        "top5_return_mean",
        "top1_excess_mean",
        "top5_excess_mean",
        "top1_capture",
        "top5_capture",
        "decile_spearman",
        "zscore_mse",
    ]
    rows: list[dict[str, Any]] = []
    for (outcome, year), part in annual.groupby(
        ["outcome", "evaluation_year"], sort=True
    ):
        indexed = part.set_index("model_training_target")
        candidate = indexed.loc["close_to_close"]
        baseline = indexed.loc["next_open_to_close"]
        for metric in metrics:
            candidate_value = float(candidate[metric])
            baseline_value = float(baseline[metric])
            rows.append(
                {
                    "outcome": str(outcome),
                    "evaluation_year": int(year),
                    "metric": metric,
                    "close_trained_value": candidate_value,
                    "open_trained_value": baseline_value,
                    "close_minus_open": candidate_value - baseline_value,
                }
            )
    return pd.DataFrame(rows).sort_values(["outcome", "metric", "evaluation_year"])


def _cross_outcome_summary(annual: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "rank_ic",
        "top1_return_mean",
        "top5_return_mean",
        "top1_excess_mean",
        "top5_excess_mean",
        "top1_capture",
        "top5_capture",
        "decile_spearman",
        "zscore_mse",
    ]
    rows: list[dict[str, Any]] = []
    for (model_target, outcome), part in annual.groupby(
        ["model_training_target", "outcome"], sort=True
    ):
        record: dict[str, Any] = {
            "model_training_target": str(model_target),
            "outcome": str(outcome),
            "year_count": len(part),
            "positive_rank_ic_years": int(part["rank_ic"].gt(0.0).sum()),
            "positive_top5_excess_years": int(part["top5_excess_mean"].gt(0.0).sum()),
        }
        for metric in metrics:
            values = pd.to_numeric(part[metric], errors="coerce")
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_median"] = float(values.median())
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["outcome", "model_training_target"])


def evaluate_close_d1(
    *, output_root: Path = DEFAULT_OUTPUT_ROOT, study_path: Path = DEFAULT_STUDY_PATH
) -> dict[str, Any]:
    _load_config(study_path)
    close_root = _close_d1_root(output_root)
    manifest_path = close_root / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") not in {"training_completed", "evaluated", "audited"}:
        raise ModelError("close_d1_training_not_completed")
    input_manifest = _read_json(Path(manifest["model_inputs"]["path"]))
    label_contract = _read_json(Path(manifest["label_contract"]["path"]))
    base_inputs = ModelInputs(input_manifest)
    close_inputs = CloseD1ModelInputs(input_manifest, label_contract)
    tasks = _close_d1_task_plan()
    close_results, diagnostic = _partition_results(
        _completed_results(close_root), tasks
    )
    if len(close_results) != CLOSE_D1_TASK_COUNT or diagnostic:
        raise ModelError("close_d1_evaluation_task_count_mismatch")
    close_by_year = {
        int(result["evaluation_year"]): result for result in close_results.values()
    }
    baseline_by_year = _baseline_d1_results(
        output_root=output_root, inputs=base_inputs, study_path=study_path
    )

    vwap_frame = pd.read_parquet(
        Path(manifest["preparation_files"]["execution_outcomes"]["path"]),
        columns=[
            "local_row",
            "first_5m_vwap_to_close_return",
            "outcome_valid",
        ],
    )
    vwap_values = np.full(base_inputs.row_count, np.nan, dtype=np.float32)
    vwap_valid = np.zeros(base_inputs.row_count, dtype=bool)
    vwap_rows = vwap_frame["local_row"].to_numpy(dtype=np.int64)
    vwap_values[vwap_rows] = vwap_frame["first_5m_vwap_to_close_return"].to_numpy(
        dtype=np.float32
    )
    vwap_valid[vwap_rows] = vwap_frame["outcome_valid"].to_numpy(dtype=bool)
    outcomes = {
        "close_to_close": (
            np.asarray(close_inputs.raw_return_values(1)),
            np.asarray(close_inputs.raw_return_valid_mask(1), dtype=bool),
        ),
        "next_open_to_close": (
            np.asarray(base_inputs.raw_return_values(1)),
            np.asarray(base_inputs.raw_return_valid_mask(1), dtype=bool),
        ),
        "first_5m_vwap_to_close": (vwap_values, vwap_valid),
    }
    models = {
        "close_to_close": close_by_year,
        "next_open_to_close": baseline_by_year,
    }
    annual_records: list[dict[str, Any]] = []
    daily_parts: list[pd.DataFrame] = []
    baseline_references: list[dict[str, Any]] = []
    for model_target, results_by_year in models.items():
        for year in ROLLING_YEARS:
            result = results_by_year[year]
            year_rows, prediction = _load_year_prediction(
                result=result, inputs=base_inputs, year=year
            )
            if model_target == "next_open_to_close":
                baseline_references.append(
                    {
                        "year": int(year),
                        "task_id": str(result["task_id"]),
                        "task_result": _file_record(
                            _task_result_path(
                                _direct_return_root(output_root), str(result["task_id"])
                            )
                        ),
                    }
                )
            for outcome, (raw_values, raw_valid) in outcomes.items():
                local_raw = np.asarray(raw_values[year_rows], dtype=np.float32)
                local_valid = np.asarray(raw_valid[year_rows], dtype=bool)
                normalized, normalized_valid = _winsorized_zscore_by_date(
                    values=local_raw,
                    date_idx=base_inputs.date_idx[year_rows],
                    valid=local_valid,
                )
                positions = np.flatnonzero(
                    normalized_valid & np.isfinite(prediction)
                ).astype(np.int64, copy=False)
                if not len(positions):
                    raise ModelError(
                        f"close_d1_outcome_has_no_valid_rows:{model_target}:"
                        f"{outcome}:{year}"
                    )
                daily, metrics = _return_daily_metrics(
                    date_idx=base_inputs.date_idx[year_rows][positions],
                    actual_raw=local_raw[positions],
                    actual_zscore=normalized[positions],
                    prediction=prediction[positions],
                )
                daily.insert(
                    1,
                    "trade_date",
                    [
                        str(base_inputs.date_values[int(value)])
                        for value in daily["date_idx"]
                    ],
                )
                daily.insert(0, "evaluation_year", int(year))
                daily.insert(0, "outcome", outcome)
                daily.insert(0, "model_training_target", model_target)
                daily_parts.append(daily)
                record = {
                    "model_training_target": model_target,
                    "outcome": outcome,
                    "evaluation_year": int(year),
                    "task_id": str(result["task_id"]),
                    "feature_count": int(result["feature_count"]),
                    "best_iteration": int(result["best_iteration"]),
                    "year_row_count": len(year_rows),
                    "outcome_valid_count": int(normalized_valid.sum()),
                    "outcome_coverage": float(normalized_valid.mean()),
                }
                record.update(metrics)
                annual_records.append(record)
    annual = pd.DataFrame(annual_records).sort_values(
        ["outcome", "evaluation_year", "model_training_target"]
    )
    daily = pd.concat(daily_parts, ignore_index=True).sort_values(
        ["outcome", "evaluation_year", "trade_date", "model_training_target"]
    )
    monthly = _cross_outcome_monthly(daily)
    paired = _paired_model_deltas(annual)
    summary = _cross_outcome_summary(annual)
    importance = pd.concat(
        [
            pd.read_parquet(Path(result["files"]["family_importance"]["path"]))
            for result in close_results.values()
        ],
        ignore_index=True,
    )
    paths = {
        "annual_metrics": close_root / "annual_metrics.parquet",
        "monthly_metrics": close_root / "monthly_metrics.parquet",
        "daily_metrics": close_root / "cross_outcome_daily_metrics.parquet",
        "paired_model_deltas": close_root / "paired_model_deltas.parquet",
        "outcome_summary": close_root / "outcome_summary.parquet",
        "family_importance": close_root / "family_importance.parquet",
        "baseline_task_references": close_root / "baseline_task_references.json",
    }
    for frame, path in (
        (annual, paths["annual_metrics"]),
        (monthly, paths["monthly_metrics"]),
        (daily, paths["daily_metrics"]),
        (paired, paths["paired_model_deltas"]),
        (summary, paths["outcome_summary"]),
        (importance, paths["family_importance"]),
    ):
        _write_parquet(frame, path)
    _write_json(
        paths["baseline_task_references"],
        {
            "schema": "seq100_quality_liquidity_close_d1_baselines/1",
            "study_id": STUDY_ID,
            "source_stage": DIRECT_RETURN_DIR_NAME,
            "task_count": len(baseline_references),
            "tasks": baseline_references,
        },
    )
    manifest["status"] = "evaluated"
    manifest["updated_at"] = _now()
    manifest["training_performed"] = True
    manifest["feature_set_selected"] = False
    frames = {
        "annual_metrics": annual,
        "monthly_metrics": monthly,
        "daily_metrics": daily,
        "paired_model_deltas": paired,
        "outcome_summary": summary,
        "family_importance": importance,
    }
    manifest["outputs"] = {
        name: _file_record(paths[name], row_count=len(frame))
        for name, frame in frames.items()
    }
    manifest["outputs"]["baseline_task_references"] = _file_record(
        paths["baseline_task_references"]
    )
    _write_json(manifest_path, manifest)
    return {
        "status": "evaluated",
        "trained_task_count": len(close_results),
        "reused_baseline_task_count": len(baseline_references),
        "annual_metric_rows": len(annual),
        "monthly_metric_rows": len(monthly),
        "paired_delta_rows": len(paired),
    }


def audit_close_d1(
    *, output_root: Path = DEFAULT_OUTPUT_ROOT, study_path: Path = DEFAULT_STUDY_PATH
) -> dict[str, Any]:
    _load_config(study_path)
    close_root = _close_d1_root(output_root)
    manifest_path = close_root / "manifest.json"
    manifest = _read_json(manifest_path)
    input_manifest = _read_json(Path(manifest["model_inputs"]["path"]))
    _verify_model_input_files(input_manifest, full_hash=False)
    sources = _close_d1_source_context(input_manifest=input_manifest)
    expected_fingerprint = _close_d1_experiment_fingerprint(
        input_manifest=input_manifest, study_path=study_path, sources=sources
    )
    label_contract = _read_json(Path(manifest["label_contract"]["path"]))
    inputs = CloseD1ModelInputs(input_manifest, label_contract)
    tasks = _close_d1_task_plan()
    results, diagnostic = _partition_results(_completed_results(close_root), tasks)
    validation = _read_json(
        Path(manifest["preparation_files"]["minute_source_validation"]["path"])
    )
    checks: dict[str, bool] = {
        "manifest_status": manifest.get("status")
        in {"training_completed", "evaluated", "audited"},
        "experiment_fingerprint": str(manifest.get("experiment_fingerprint"))
        == expected_fingerprint,
        "label_fingerprint": str(label_contract.get("experiment_fingerprint"))
        == expected_fingerprint,
        "label_files_valid": all(
            _record_file_valid(record)
            for record in dict(label_contract.get("files", {}) or {}).values()
        ),
        "label_support_exact": int(label_contract.get("valid_count", -1))
        == int(inputs.raw_return_valid_mask(1).sum()),
        "task_count_exact": len(results) == CLOSE_D1_TASK_COUNT and not diagnostic,
        "feature_count_exact": len(inputs.feature_groups[COMPACT_VARIANT])
        == COMPACT_FEATURE_COUNT,
        "formal_rows_exclude_2010": not bool((inputs.years == 2010).any()),
        "forbidden_2026_rows": not bool(
            np.char.startswith(inputs.trade_date, "2026-").any()
        ),
        "forbidden_2026_label_reads": int(
            label_contract.get("forbidden_2026_read_count", -1)
        )
        == 0,
        "forbidden_2026_minute_query_rows": int(
            validation.get("forbidden_2026_query_row_count", -1)
        )
        == 0,
        "minute_source_validation_passed": bool(
            validation.get("source_validation_passed", False)
        ),
        "preparation_files_valid": all(
            _record_file_valid(record)
            for record in dict(manifest.get("preparation_files", {}) or {}).values()
        ),
        "task_files_valid": True,
        "purge_valid": True,
        "outputs_hashed": bool(manifest.get("outputs")),
        "existing_direct_return_plan_unchanged": len(_direct_return_task_plan())
        == RETURN_TASK_COUNT,
    }
    config_sha256 = _sha256(study_path)
    for task in tasks:
        task_id = str(task["task_id"])
        result = results.get(task_id)
        if result is None:
            checks["task_files_valid"] = False
            continue
        try:
            feature_names, _ = _effective_features(
                inputs=inputs, task=task, output_root=close_root
            )
            fingerprint = _task_fingerprint(
                task=task,
                feature_names=feature_names,
                model_input_fingerprint=str(input_manifest["input_fingerprint"]),
                config_sha256=config_sha256,
                experiment_fingerprint=expected_fingerprint,
            )
            checks["task_files_valid"] &= _task_complete(
                _task_result_path(close_root, task_id), fingerprint=fingerprint
            )
            prediction = np.load(
                Path(result["files"]["prediction"]["path"]),
                mmap_mode="r",
                allow_pickle=False,
            )
            candidate_ids = np.load(
                Path(result["files"]["candidate_id"]["path"]),
                mmap_mode="r",
                allow_pickle=False,
            )
            expected_rows = inputs.rows_for_year(int(result["evaluation_year"]))
            checks["task_files_valid"] &= np.array_equal(
                candidate_ids, inputs.candidate_ids[expected_rows]
            ) and len(prediction) == len(expected_rows)
            fold = inputs.fold(
                year=int(result["evaluation_year"]),
                horizon=1,
                target=CLOSE_D1_TARGET,
            )
            checks["purge_valid"] &= int(
                result["maximum_train_signal_date_idx"]
            ) == int(fold["maximum_train_signal_date_idx"])
        except (ModelError, OSError, KeyError, ValueError, TypeError, IndexError):
            checks["task_files_valid"] = False
    for record in dict(manifest.get("outputs", {}) or {}).values():
        checks["outputs_hashed"] &= _record_file_valid(record)
    payload = {
        "schema": "seq100_quality_liquidity_close_d1_audit/1",
        "study_id": STUDY_ID,
        "stage": CLOSE_D1_DIR_NAME,
        "status": "ok" if all(checks.values()) else "failed",
        "created_at": _now(),
        "checks": checks,
        "trained_task_count": len(results),
        "reused_existing_direct_d1_task_count": len(ROLLING_YEARS),
        "rolling_years": list(ROLLING_YEARS),
        "training_performed": True,
        "evaluation_semantics": "retrospective_rolling_oos",
    }
    audit_path = close_root / "audit.json"
    _write_json(audit_path, payload)
    if payload["status"] != "ok":
        raise ModelError("close_d1_audit_failed")
    manifest["status"] = "audited"
    manifest["audit"] = _file_record(audit_path)
    manifest["updated_at"] = _now()
    _write_json(manifest_path, manifest)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-path", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--ready-root", type=Path, default=DEFAULT_READY_ROOT)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--evaluate", action="store_true")
    modes.add_argument("--audit", action="store_true")
    modes.add_argument("--prepare-returns", action="store_true")
    modes.add_argument("--run-returns", action="store_true")
    modes.add_argument("--evaluate-returns", action="store_true")
    modes.add_argument("--audit-returns", action="store_true")
    modes.add_argument("--prepare-close-d1", action="store_true")
    modes.add_argument("--run-close-d1", action="store_true")
    modes.add_argument("--evaluate-close-d1", action="store_true")
    modes.add_argument("--audit-close-d1", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.prepare_close_d1:
        result = prepare_close_d1(
            study_path=args.study_path,
            ready_root=args.ready_root,
            audit_root=args.audit_root,
            output_root=args.output_root,
        )
    elif args.run_close_d1:
        result = run_close_d1(
            study_path=args.study_path,
            ready_root=args.ready_root,
            audit_root=args.audit_root,
            output_root=args.output_root,
            max_tasks=args.max_tasks,
        )
    elif args.evaluate_close_d1:
        result = evaluate_close_d1(
            output_root=args.output_root, study_path=args.study_path
        )
    elif args.audit_close_d1:
        result = audit_close_d1(
            output_root=args.output_root, study_path=args.study_path
        )
    elif args.prepare_returns:
        result = prepare_returns(
            study_path=args.study_path,
            ready_root=args.ready_root,
            audit_root=args.audit_root,
            output_root=args.output_root,
        )
    elif args.run_returns:
        result = run_returns(
            study_path=args.study_path,
            ready_root=args.ready_root,
            audit_root=args.audit_root,
            output_root=args.output_root,
            max_tasks=args.max_tasks,
        )
    elif args.evaluate_returns:
        result = evaluate_returns(
            output_root=args.output_root, study_path=args.study_path
        )
    elif args.audit_returns:
        result = audit_returns(output_root=args.output_root, study_path=args.study_path)
    elif args.prepare:
        result = prepare(
            study_path=args.study_path,
            ready_root=args.ready_root,
            audit_root=args.audit_root,
            output_root=args.output_root,
        )
    elif args.run:
        result = run(
            study_path=args.study_path,
            ready_root=args.ready_root,
            audit_root=args.audit_root,
            output_root=args.output_root,
            max_tasks=args.max_tasks,
        )
    elif args.evaluate:
        result = evaluate(output_root=args.output_root, study_path=args.study_path)
    else:
        result = audit(output_root=args.output_root, study_path=args.study_path)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return (
        0
        if result.get("status")
        in {
            "prepared",
            "completed",
            "training_completed",
            "training_in_progress",
            "evaluated",
            "ok",
            "audited",
        }
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
