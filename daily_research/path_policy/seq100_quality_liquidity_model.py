from __future__ import annotations

"""Prepare and run the first rolling quality-liquidity LightGBM comparison."""

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
LABEL_HORIZON_INDEX = {10: 1, 20: 2}
STATE_HORIZON = 10
FLAG_MFE_PRE_PEAK_MAE_VALID = 128
FLAG_STATE_ASSIGNED = 256
TARGETS = ("mfe_10", "mfe_20", "risk_10", "risk_20", "state_10")
COMPACT_VARIANT = "compact_core"
MODEL_INPUT_DIR_NAME = "model_inputs"
FORMAL_TASK_SCOPE = "compact_core_only"

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
        if target.startswith("state"):
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
) -> str:
    return _stable_hash(
        {
            "task": dict(task),
            "feature_names": list(feature_names),
            "model_input_fingerprint": model_input_fingerprint,
            "config_sha256": config_sha256,
        }
    )


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
    if stage in {"tuning", "mfe_core", "risk_state_core"}:
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
        if task["kind"] == "mfe":
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
        "study_id": STUDY_ID,
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
    parser.add_argument("--max-tasks", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.prepare:
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
