"""Retrospective effect audit for the seven-family QVER framework.

The score is frozen before outcomes are read.  It is a quantitative proxy for
the supplied research framework, not a replacement for the framework's
qualitative moat, management and discretionary valuation work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_causal_value_policy as causal
from daily_research.path_policy import seq100_exact_value_growth_account as old_account
from daily_research.path_policy import seq100_exact_value_growth_path_exit as path_audit
from daily_research.path_policy import seq100_exact_value_growth_policy as old_policy
from daily_research.path_policy import seq100_finite_capital_backtest as finite
from daily_research.path_policy import (
    seq100_margin_residual_account_feasibility as feasibility,
)
from daily_research.path_policy import seq100_stock_distribution as base
from daily_research.path_policy.seq100_exit_policy_audit import (
    CandidateCompleteAuditPack,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_qver_confirmation_effect_v1"
SUMMARY_SCHEMA = "seq100_qver_confirmation_effect_summary/1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_qver_confirmation_effect_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_qver_confirmation_effect_v1"
)

FRAMEWORK_WEIGHTS = {
    "normalized_valuation": 0.25,
    "actual_earnings_improvement": 0.20,
    "earnings_revision": 0.15,
    "company_quality": 0.20,
    "earnings_quality_cashflow": 0.10,
    "price_confirmation": 0.05,
    "governance_risk": 0.05,
}

EXTRA_FEATURES = (
    "financial_gross_profit_margin",
    "financial_asset_turnover",
    "cashflow_cfo_to_revenue",
    "relative_turnover_20d",
    "volume_ratio_20d",
)

SCORE_COLUMNS = (
    "normalized_valuation_score",
    "actual_earnings_improvement_score",
    "earnings_revision_framework_score",
    "company_quality_score",
    "earnings_quality_cashflow_score",
    "price_confirmation_framework_score",
    "governance_risk_score",
)

HORIZONS = (20, 60, 120)

READABLE_INDUSTRY_NAMES = {
    298: "非金属矿物制品业",
    301: "电气机械和器材制造业",
    302: "计算机、通信和其他电子设备制造业",
    303: "有色金属冶炼和压延加工业",
    307: "专用设备制造业",
    308: "汽车制造业",
    309: "通用设备制造业",
    312: "化学原料和化学制品制造业",
    313: "医药制造业",
    332: "铁路、船舶、航空航天和其他运输设备制造业",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    partial.unlink(missing_ok=True)
    frame.to_parquet(partial, index=False)
    os.replace(partial, path)


def _payload_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def load_study(
    path: str | Path = DEFAULT_STUDY_PATH,
) -> tuple[dict[str, Any], Path]:
    study_path = base._resolve_path(path)
    study = _read_json(study_path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("study_id_mismatch")
    if dict(study["score_contract"]["family_weights"]) != FRAMEWORK_WEIGHTS:
        raise ValueError("framework_weight_contract_mismatch")
    if tuple(int(value) for value in study["outcomes"]["fixed_open_day_horizons"]) != HORIZONS:
        raise ValueError("outcome_horizon_contract_mismatch")
    if int(study["source"]["forbidden_year"]) != 2026:
        raise ValueError("forbidden_year_contract_mismatch")
    if bool(study["decision_boundary"].get("score_or_threshold_optimization_performed", True)):
        raise ValueError("score_optimization_forbidden")
    if not bool(study["epistemic_contract"].get("candidate_selection_precedes_outcome_reads")):
        raise ValueError("selection_before_outcome_contract_missing")
    forbidden = " ".join(study["score_contract"]["explicitly_forbidden"]).lower()
    if "52_week_low" not in forbidden or "distance_from_52_week_low" not in forbidden:
        raise ValueError("low_price_reward_exclusion_missing")
    return study, study_path


def _source_contract(study: Mapping[str, Any]) -> dict[str, Path]:
    source = dict(study["source"])
    pairs = {
        "exact_policy_summary": "exact_policy_summary_sha256",
        "candidate_features": "candidate_features_sha256",
        "old_selections": "old_selections_sha256",
        "input_manifest": "input_manifest_sha256",
        "label_manifest": "label_manifest_sha256",
        "pack_manifest": "pack_manifest_sha256",
        "income_statement_quarterly": "income_statement_quarterly_sha256",
    }
    paths: dict[str, Path] = {}
    for key, hash_key in pairs.items():
        path = base._resolve_path(source[key])
        if not path.is_file():
            raise FileNotFoundError(path)
        if base._sha256_file(path).lower() != str(source[hash_key]).lower():
            raise ValueError(f"source_hash_mismatch:{key}")
        paths[key] = path
    membership = base._resolve_path(source["membership_root"])
    if not membership.is_dir():
        raise FileNotFoundError(membership)
    paths["membership_root"] = membership
    framework = dict(study["framework_source"])
    attachment = Path(str(framework["attachment_path"]))
    if attachment.is_file() and base._sha256_file(attachment) != str(
        framework["attachment_sha256"]
    ):
        raise ValueError("framework_attachment_hash_mismatch")
    old_policy.validate_summary(paths["exact_policy_summary"])
    return paths


def _period_name(year: int, study: Mapping[str, Any]) -> str:
    for name, years in study["evaluation"]["periods"].items():
        if int(year) in {int(value) for value in years}:
            return str(name)
    raise ValueError(f"evaluation_year_unassigned:{year}")


def _inference(
    values: Sequence[float] | np.ndarray,
    study: Mapping[str, Any],
    *,
    seed_add: int = 0,
) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {
            "mean": math.nan,
            "standard_error": math.nan,
            "lcb_95": math.nan,
            "ucb_95": math.nan,
            "n": 0,
            "block": {"lcb_95": math.nan, "ucb_95": math.nan},
        }
    config = study["evaluation"]["inference"]
    return {
        **base._hac_mean(x, lag=int(config["hac_lag_months"])),
        "block": base._block_interval(
            x,
            block_length=int(config["block_length_months"]),
            repetitions=int(config["bootstrap_repetitions"]),
            seed=int(config["seed"]) + int(seed_add),
        ),
    }


def _attach_extra_features(
    panel: base.StockPanel, features: pd.DataFrame
) -> pd.DataFrame:
    frame = features.copy()
    rows = frame["row_position"].to_numpy(dtype=np.int64)
    if bool((rows < 0).any()) or bool((rows >= panel.row_count).any()):
        raise ValueError("candidate_row_position_out_of_bounds")
    aligned = panel.row_index.iloc[rows]
    if not np.array_equal(
        aligned["candidate_id"].to_numpy(dtype=np.int64),
        frame["candidate_id"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("candidate_row_position_identity_mismatch")
    positions = base._feature_positions(panel, EXTRA_FEATURES)
    values = base._feature_matrix(panel, rows, positions)
    for column, value in zip(EXTRA_FEATURES, values.T, strict=True):
        frame[column] = value.astype(np.float64)
    frame["symbol_idx"] = panel.symbol_idx[rows].astype(np.int64)
    if int(frame["evaluation_year"].max()) >= 2026:
        raise ValueError("forbidden_2026_candidate")
    return frame


def _combine_score(frame: pd.DataFrame) -> np.ndarray:
    return (
        FRAMEWORK_WEIGHTS["normalized_valuation"]
        * frame["normalized_valuation_score"].to_numpy(dtype=np.float64)
        + FRAMEWORK_WEIGHTS["actual_earnings_improvement"]
        * frame["actual_earnings_improvement_score"].to_numpy(dtype=np.float64)
        + FRAMEWORK_WEIGHTS["earnings_revision"]
        * frame["earnings_revision_framework_score"].to_numpy(dtype=np.float64)
        + FRAMEWORK_WEIGHTS["company_quality"]
        * frame["company_quality_score"].to_numpy(dtype=np.float64)
        + FRAMEWORK_WEIGHTS["earnings_quality_cashflow"]
        * frame["earnings_quality_cashflow_score"].to_numpy(dtype=np.float64)
        + FRAMEWORK_WEIGHTS["price_confirmation"]
        * frame["price_confirmation_framework_score"].to_numpy(dtype=np.float64)
        + FRAMEWORK_WEIGHTS["governance_risk"]
        * frame["governance_risk_score"].to_numpy(dtype=np.float64)
    )


def _rating(score: float) -> str:
    value = float(score) * 100.0
    if not math.isfinite(value):
        return "unrated"
    if value >= 85.0:
        return "A+"
    if value >= 80.0:
        return "A"
    if value >= 75.0:
        return "A-"
    if value >= 68.0:
        return "B+"
    if value >= 60.0:
        return "B"
    return "C"


def _local_percentile(
    local: pd.DataFrame,
    industry: np.ndarray,
    column: str,
    *,
    direction: float = 1.0,
    minimum_group: int,
) -> np.ndarray:
    values = direction * local[column].to_numpy(dtype=np.float64)
    return causal._industry_percentile(
        values, industry, minimum_group_size=minimum_group
    )


def _score_framework(
    features: pd.DataFrame, study: Mapping[str, Any]
) -> pd.DataFrame:
    frame = features.copy()
    n = len(frame)
    arrays = {name: np.full(n, np.nan, dtype=np.float64) for name in SCORE_COLUMNS}
    minimum_group = int(study["score_contract"]["minimum_industry_rank_group_size"])
    minimum = dict(study["score_contract"]["minimum_nonmissing_components"])
    dates = frame["date_idx"].to_numpy(dtype=np.int64)
    boundaries = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1, n]
    for left, right in pairwise(boundaries):
        sl = slice(int(left), int(right))
        local = frame.iloc[sl]
        industry = local["industry_code"].to_numpy(dtype=np.int64)

        actual = causal._family_mean(
            tuple(
                _local_percentile(
                    local, industry, column, minimum_group=minimum_group
                )
                for column in (
                    "financial_net_profit_yoy",
                    "financial_revenue_yoy",
                    "performance_forecast_change_mid",
                )
            ),
            minimum_count=int(minimum["actual_earnings_improvement"]),
        )
        quality = causal._family_mean(
            (
                _local_percentile(
                    local, industry, "financial_roe_avg", minimum_group=minimum_group
                ),
                _local_percentile(
                    local,
                    industry,
                    "financial_net_profit_margin",
                    minimum_group=minimum_group,
                ),
                _local_percentile(
                    local,
                    industry,
                    "financial_gross_profit_margin",
                    minimum_group=minimum_group,
                ),
                _local_percentile(
                    local,
                    industry,
                    "financial_asset_turnover",
                    minimum_group=minimum_group,
                ),
                _local_percentile(
                    local,
                    industry,
                    "financial_debt_to_asset",
                    direction=-1.0,
                    minimum_group=minimum_group,
                ),
                _local_percentile(
                    local,
                    industry,
                    "balance_borrowing_ratio",
                    direction=-1.0,
                    minimum_group=minimum_group,
                ),
            ),
            minimum_count=int(minimum["company_quality"]),
        )
        earnings_quality = causal._family_mean(
            (
                _local_percentile(
                    local,
                    industry,
                    "cashflow_cfo_to_income",
                    minimum_group=minimum_group,
                ),
                _local_percentile(
                    local,
                    industry,
                    "cashflow_cfo_to_revenue",
                    minimum_group=minimum_group,
                ),
                _local_percentile(
                    local,
                    industry,
                    "cashflow_fcf_to_revenue",
                    minimum_group=minimum_group,
                ),
                _local_percentile(
                    local,
                    industry,
                    "balance_trade_receivables_to_total_assets",
                    direction=-1.0,
                    minimum_group=minimum_group,
                ),
                _local_percentile(
                    local,
                    industry,
                    "balance_inventory_ratio",
                    direction=-1.0,
                    minimum_group=minimum_group,
                ),
                _local_percentile(
                    local,
                    industry,
                    "balance_goodwill_ratio",
                    direction=-1.0,
                    minimum_group=minimum_group,
                ),
            ),
            minimum_count=int(minimum["earnings_quality_cashflow"]),
        )
        confirmation = causal._family_mean(
            (
                _local_percentile(
                    local,
                    industry,
                    "industry_relative_ret20",
                    minimum_group=minimum_group,
                ),
                _local_percentile(
                    local, industry, "trend_slope_20d", minimum_group=minimum_group
                ),
                _local_percentile(
                    local,
                    industry,
                    "relative_turnover_20d",
                    minimum_group=minimum_group,
                ),
                _local_percentile(
                    local, industry, "volume_ratio_20d", minimum_group=minimum_group
                ),
            ),
            minimum_count=int(minimum["price_confirmation"]),
        )
        values = {
            "normalized_valuation_score": local["valuation_score"].to_numpy(
                dtype=np.float64
            ),
            "actual_earnings_improvement_score": actual,
            "earnings_revision_framework_score": local[
                "revision_score_exact"
            ].to_numpy(dtype=np.float64),
            "company_quality_score": quality,
            "earnings_quality_cashflow_score": earnings_quality,
            "price_confirmation_framework_score": confirmation,
            "governance_risk_score": local["governance_score"].to_numpy(
                dtype=np.float64
            ),
        }
        for name, value in values.items():
            arrays[name][sl] = value
    for name, values in arrays.items():
        frame[name] = values
    frame["framework_score"] = _combine_score(frame)
    frame["framework_score_100"] = 100.0 * frame["framework_score"]
    frame["framework_rating"] = [
        _rating(value) for value in frame["framework_score"].to_numpy(dtype=float)
    ]
    minimum_institutions = 3
    analyst_current = (
        frame["current_np_n"].fillna(0).to_numpy(dtype=float)
        >= minimum_institutions
    ) & (frame["current_np"].to_numpy(dtype=float) > 0.0)
    revision_covered = (
        frame["current_eps_n"].fillna(0).to_numpy(dtype=float)
        >= minimum_institutions
    )
    for gap in (30, 90):
        revision_covered &= (
            frame[f"prior_{gap}_np_n"].fillna(0).to_numpy(dtype=float)
            >= minimum_institutions
        ) & (
            frame[f"prior_{gap}_eps_n"].fillna(0).to_numpy(dtype=float)
            >= minimum_institutions
        )
    frame["eligible__framework"] = (
        (frame["signed_log_pe"].to_numpy(dtype=float) > 0.0)
        & (frame["signed_log_pb"].to_numpy(dtype=float) > 0.0)
        & analyst_current
        & revision_covered
        & (frame["industry_relative_ret20"].to_numpy(dtype=float) >= 0.0)
        & (frame["trend_slope_20d"].to_numpy(dtype=float) >= 0.0)
        & (
            frame["listing_age_open_days"].to_numpy(dtype=float)
            >= int(study["universe"]["minimum_listing_open_days"])
        )
        & (
            frame["announcement_keyword_delisting_20d"].to_numpy(dtype=float)
            <= 0.0
        )
        & np.isfinite(frame["framework_score"].to_numpy(dtype=float))
    )
    return frame


def _select_indices(
    local: pd.DataFrame,
    *,
    top_k: int,
    industry_cap: int,
) -> np.ndarray:
    return causal._select_industry_capped(
        score=local["framework_score"].to_numpy(dtype=np.float64),
        eligible=local["eligible__framework"].to_numpy(dtype=bool),
        candidate_id=local["candidate_id"].to_numpy(dtype=np.int64),
        industry_code=local["industry_code"].to_numpy(dtype=np.int64),
        top_k=int(top_k),
        industry_cap=int(industry_cap),
    )


def _selection_frame(
    scored: pd.DataFrame,
    old_selection_path: Path,
    study: Mapping[str, Any],
) -> pd.DataFrame:
    selection = dict(study["selection"])
    top_k = int(selection["top_k"])
    cap = int(selection["maximum_names_per_pit_industry"])
    records: list[pd.DataFrame] = []
    for _, local in scored.groupby("date_idx", sort=True):
        chosen = _select_indices(local, top_k=top_k, industry_cap=cap)
        if not len(chosen):
            continue
        selected = local.iloc[chosen].copy()
        selected["selection_rank"] = np.arange(1, len(selected) + 1)
        records.append(selected)
    if not records:
        raise ValueError("framework_selection_empty")
    result = pd.concat(records, ignore_index=True)
    old = pd.read_parquet(old_selection_path)
    old = old.loc[old["policy"].eq("exact_full_top10"), ["candidate_id"]]
    result["old_exact_selected"] = result["candidate_id"].isin(
        old["candidate_id"].to_numpy(dtype=np.int64)
    )
    result["top_k"] = top_k
    result["industry_cap"] = cap
    return result.sort_values(
        ["date_idx", "selection_rank"], kind="stable"
    ).reset_index(drop=True)


def _candidate_returns(
    panel: base.StockPanel,
    scored: pd.DataFrame,
    study: Mapping[str, Any],
) -> pd.DataFrame:
    pack = panel.pack_manifest
    dates = np.asarray(pack["date_values"], dtype=str)
    cutoff_positions = np.flatnonzero(
        dates == str(study["source"]["maximum_outcome_date"])
    )
    if len(cutoff_positions) != 1:
        raise ValueError("maximum_outcome_date_missing")
    cutoff = int(cutoff_positions[0])
    raw = base._open_pack_array(pack, "feature_channels", "daily_raw", dtype=np.float32)
    observed = base._open_pack_array(pack, "masks", "price_observed", dtype=np.bool_)
    buyable = base._open_pack_array(pack, "masks", "entry_buyable", dtype=np.bool_)
    sellable = base._open_pack_array(pack, "masks", "exit_sellable", dtype=np.bool_)
    signal_idx = scored["date_idx"].to_numpy(dtype=np.int64)
    symbol_idx = scored["symbol_idx"].to_numpy(dtype=np.int64)
    entry_idx = signal_idx + 1
    safe_entry = np.clip(entry_idx, 0, cutoff)
    entry_price = np.asarray(raw[safe_entry, symbol_idx, 0], dtype=np.float64)
    entry_valid = (
        (entry_idx <= cutoff)
        & np.asarray(buyable[safe_entry, symbol_idx], dtype=bool)
        & np.asarray(observed[safe_entry, symbol_idx], dtype=bool)
        & np.isfinite(entry_price)
        & (entry_price > 0.0)
    )
    result = scored[
        [
            "candidate_id",
            "row_position",
            "date_idx",
            "trade_date",
            "evaluation_year",
            "symbol",
            "symbol_idx",
            "industry_code",
            "framework_score",
            "framework_score_100",
            "framework_rating",
            "eligible__framework",
        ]
    ].copy()
    result["entry_date_idx"] = np.where(entry_valid, entry_idx, -1)
    result["entry_date"] = np.where(entry_valid, dates[safe_entry], None)
    result["entry_adjusted_open"] = np.where(entry_valid, entry_price, np.nan)
    result["entry_valid"] = entry_valid
    retry_days = int(study["outcomes"]["sell_retry_open_days"])
    cost = float(study["outcomes"]["round_trip_cost_proxy"])
    for horizon in HORIZONS:
        planned = signal_idx + int(horizon)
        exit_idx = np.full(len(result), -1, dtype=np.int64)
        exit_price = np.full(len(result), np.nan, dtype=np.float64)
        pending = entry_valid & (planned <= cutoff)
        for delay in range(retry_days + 1):
            candidate_idx = planned + int(delay)
            in_bounds = candidate_idx <= cutoff
            safe = np.clip(candidate_idx, 0, cutoff)
            price = np.asarray(raw[safe, symbol_idx, 3], dtype=np.float64)
            hit = (
                pending
                & in_bounds
                & np.asarray(sellable[safe, symbol_idx], dtype=bool)
                & np.asarray(observed[safe, symbol_idx], dtype=bool)
                & np.isfinite(price)
                & (price > 0.0)
            )
            exit_idx[hit] = candidate_idx[hit]
            exit_price[hit] = price[hit]
            pending[hit] = False
        valid = (exit_idx >= 0) & entry_valid
        net_return = np.where(valid, exit_price / entry_price - 1.0 - cost, np.nan)
        safe_exit = np.clip(exit_idx, 0, cutoff)
        result[f"legal_exit_date_idx_d{horizon}"] = exit_idx
        result[f"legal_exit_date_d{horizon}"] = np.where(
            valid, dates[safe_exit], None
        )
        result[f"exit_delay_d{horizon}"] = np.where(
            valid, exit_idx - planned, np.nan
        )
        result[f"legal_net_return_d{horizon}"] = net_return
    return result


def _selected_paths(
    panel: base.StockPanel,
    selections: pd.DataFrame,
    study: Mapping[str, Any],
) -> pd.DataFrame:
    path_study = {
        "source": {"maximum_outcome_date": study["source"]["maximum_outcome_date"]},
        "selection": {
            "round_trip_cost": study["outcomes"]["round_trip_cost_proxy"]
        },
        "path": {
            "fixed_checkpoints": [5, 10, 20, 60, 120],
            "profit_thresholds_net": [0.0, 0.03, 0.05, 0.10],
            "maximum_open_days": int(study["outcomes"]["path_maximum_open_days"]),
        },
        "exit": {"fixed_legal_horizons": [20, 60, 120]},
    }
    keep = [
        "candidate_id",
        "date_idx",
        "trade_date",
        "symbol",
        "symbol_idx",
        "evaluation_year",
        "industry_code",
        "selection_rank",
    ]
    paths = path_audit._path_frame(panel, selections[keep], path_study)
    wanted = [
        "candidate_id",
        "next_open_gap",
        "return_d5",
        "return_d10",
        "return_d20",
        "return_d60",
        "return_d120",
        "mfe_d10",
        "mae_d10",
        "mfe_d20",
        "mae_d20",
        "mfe_d60",
        "mae_d60",
        "mfe_d120",
        "mae_d120",
        "peak_day_d60",
        "peak_day_d120",
        "primary_shape",
    ]
    return paths[wanted]


def _attach_industry_names(
    selections: pd.DataFrame, membership_root: Path
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for year, local in selections.groupby("evaluation_year", sort=True):
        path = membership_root / f"year={int(year)}" / "diagnostics.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        membership = pd.read_parquet(path, columns=["candidate_id", "industry"])
        membership = membership.loc[
            membership["candidate_id"].isin(local["candidate_id"])
        ]
        if bool(membership["candidate_id"].duplicated().any()):
            raise ValueError(f"industry_membership_duplicate:{year}")
        records.append(membership)
    names = pd.concat(records, ignore_index=True)
    result = selections.merge(
        names, on="candidate_id", how="left", validate="one_to_one"
    )
    result["industry_name_raw"] = result["industry"].fillna("Unknown").astype(str)
    readable = result["industry_code"].map(READABLE_INDUSTRY_NAMES)
    raw_is_readable = ~result["industry_name_raw"].str.contains("�", regex=False)
    result["industry_name"] = readable.where(
        readable.notna(),
        result["industry_name_raw"].where(
            raw_is_readable,
            "行业代码 " + result["industry_code"].astype(str),
        ),
    )
    return result.drop(columns="industry")


def _annual_actuals(path: Path, study: Mapping[str, Any]) -> pd.DataFrame:
    columns = [
        "symbol",
        "trade_date",
        "feature_available_date",
        "report_date",
        "fiscal_year",
        "fiscal_quarter",
        "update_flag",
        "parent_net_income",
        "basic_eps",
        "revenue",
        "source_conflict",
    ]
    frame = pd.read_parquet(path, columns=columns)
    available = pd.to_datetime(frame["feature_available_date"], errors="coerce")
    frame = frame.loc[
        frame["fiscal_quarter"].eq(4)
        & available.le(pd.Timestamp(str(study["source"]["maximum_outcome_date"])))
        & frame["fiscal_year"].le(int(study["source"]["maximum_realized_fiscal_year"]))
    ].copy()
    frame["feature_available_date"] = pd.to_datetime(
        frame["feature_available_date"], errors="coerce"
    )
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.sort_values(
        ["symbol", "fiscal_year", "feature_available_date", "trade_date", "update_flag"],
        kind="stable",
    ).drop_duplicates(["symbol", "fiscal_year"], keep="last")
    return frame.rename(
        columns={
            "parent_net_income": "actual_parent_net_income",
            "basic_eps": "actual_basic_eps",
            "revenue": "actual_revenue",
            "feature_available_date": "actual_available_date",
            "source_conflict": "actual_source_conflict",
        }
    )


def _attach_forecast_realization(
    frame: pd.DataFrame,
    actuals: pd.DataFrame,
    study: Mapping[str, Any],
) -> pd.DataFrame:
    result = frame.copy()
    result["target_fiscal_year"] = result["evaluation_year"].astype(int) + 1
    actual_columns = [
        "symbol",
        "fiscal_year",
        "actual_parent_net_income",
        "actual_basic_eps",
        "actual_revenue",
        "actual_available_date",
        "actual_source_conflict",
    ]
    target = actuals[actual_columns].rename(
        columns={"fiscal_year": "target_fiscal_year"}
    )
    prior = actuals[
        ["symbol", "fiscal_year", "actual_parent_net_income", "actual_basic_eps"]
    ].rename(
        columns={
            "fiscal_year": "prior_fiscal_year",
            "actual_parent_net_income": "prior_actual_parent_net_income",
            "actual_basic_eps": "prior_actual_basic_eps",
        }
    )
    result = result.merge(
        target,
        on=["symbol", "target_fiscal_year"],
        how="left",
        validate="many_to_one",
    )
    result["prior_fiscal_year"] = result["target_fiscal_year"] - 1
    result = result.merge(
        prior,
        on=["symbol", "prior_fiscal_year"],
        how="left",
        validate="many_to_one",
    )
    result["forecast_parent_net_income"] = (
        result["current_np"].to_numpy(dtype=float) * 10_000.0
    )
    result["forecast_eps"] = result["current_eps"].to_numpy(dtype=float)
    forecast = result["forecast_parent_net_income"].to_numpy(dtype=float)
    actual = result["actual_parent_net_income"].to_numpy(dtype=float)
    prior_actual = result["prior_actual_parent_net_income"].to_numpy(dtype=float)
    valid = np.isfinite(forecast) & (forecast > 0.0) & np.isfinite(actual)
    result["forecast_outcome_evaluable"] = valid
    result["forecast_error"] = np.where(valid, actual / forecast - 1.0, np.nan)
    denominator = np.abs(actual) + np.abs(forecast)
    result["forecast_smape"] = np.where(
        valid & (denominator > 0.0),
        2.0 * np.abs(actual - forecast) / denominator,
        np.nan,
    )
    tight = float(
        study["outcomes"]["fundamental_realization"]["forecast_tight_band"]
    )
    close = float(
        study["outcomes"]["fundamental_realization"]["forecast_close_band"]
    )
    result["forecast_within_10pct"] = valid & (
        np.abs(result["forecast_error"].to_numpy(dtype=float)) <= tight
    )
    result["forecast_within_20pct"] = valid & (
        np.abs(result["forecast_error"].to_numpy(dtype=float)) <= close
    )
    result["forecast_overpredicted"] = valid & (actual < forecast)
    prior_valid = valid & np.isfinite(prior_actual) & (prior_actual > 0.0)
    result["forecast_growth"] = np.where(
        prior_valid, forecast / prior_actual - 1.0, np.nan
    )
    result["actual_growth"] = np.where(
        prior_valid, actual / prior_actual - 1.0, np.nan
    )
    result["growth_direction_match"] = prior_valid & (
        (forecast > prior_actual) == (actual > prior_actual)
    )
    forecast_eps = result["forecast_eps"].to_numpy(dtype=float)
    actual_eps = result["actual_basic_eps"].to_numpy(dtype=float)
    prior_eps = result["prior_actual_basic_eps"].to_numpy(dtype=float)
    eps_valid = np.isfinite(forecast_eps) & (forecast_eps > 0.0) & np.isfinite(
        actual_eps
    )
    result["eps_forecast_outcome_evaluable"] = eps_valid
    result["eps_forecast_error"] = np.where(
        eps_valid, actual_eps / forecast_eps - 1.0, np.nan
    )
    eps_denominator = np.abs(actual_eps) + np.abs(forecast_eps)
    result["eps_forecast_smape"] = np.where(
        eps_valid & (eps_denominator > 0.0),
        2.0 * np.abs(actual_eps - forecast_eps) / eps_denominator,
        np.nan,
    )
    result["eps_forecast_within_10pct"] = eps_valid & (
        np.abs(result["eps_forecast_error"].to_numpy(dtype=float)) <= tight
    )
    result["eps_forecast_within_20pct"] = eps_valid & (
        np.abs(result["eps_forecast_error"].to_numpy(dtype=float)) <= close
    )
    result["eps_forecast_overpredicted"] = eps_valid & (
        actual_eps < forecast_eps
    )
    eps_growth_valid = (
        eps_valid & np.isfinite(prior_eps) & (np.abs(prior_eps) > 1e-12)
    )
    result["forecast_eps_growth"] = np.where(
        eps_growth_valid, forecast_eps / prior_eps - 1.0, np.nan
    )
    result["actual_eps_growth"] = np.where(
        eps_growth_valid, actual_eps / prior_eps - 1.0, np.nan
    )
    result["eps_growth_direction_match"] = eps_growth_valid & (
        (forecast_eps > prior_eps) == (actual_eps > prior_eps)
    )
    result["revision_np_mean"] = result[["revision_np_30", "revision_np_90"]].mean(
        axis=1, skipna=True
    )
    result["revision_eps_mean"] = result[
        ["revision_eps_30", "revision_eps_90"]
    ].mean(axis=1, skipna=True)
    result["positive_revision_at_signal"] = (
        result[["revision_np_mean", "revision_eps_mean"]].mean(axis=1, skipna=True)
        > 0.0
    )
    return result


def _future_snapshot_indices(
    scored: pd.DataFrame, selected: pd.DataFrame
) -> np.ndarray:
    source = scored.reset_index(drop=True)
    grouped: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for symbol, local in source.groupby("symbol", sort=False):
        indices = local.index.to_numpy(dtype=np.int64)
        dates = pd.to_datetime(local["trade_date"], errors="raise").to_numpy()
        order = np.argsort(dates, kind="stable")
        grouped[str(symbol)] = (dates[order], indices[order])
    result = np.full(len(selected), -1, dtype=np.int64)
    exits = pd.to_datetime(selected["legal_exit_date_d120"], errors="coerce")
    signals = pd.to_datetime(selected["trade_date"], errors="raise")
    for pos, (symbol, exit_date, signal_date) in enumerate(
        zip(selected["symbol"], exits, signals, strict=True)
    ):
        if pd.isna(exit_date):
            continue
        dates, indices = grouped[str(symbol)]
        offset = int(np.searchsorted(dates, exit_date.to_datetime64(), side="right") - 1)
        if offset >= 0 and dates[offset] > signal_date.to_datetime64():
            result[pos] = indices[offset]
    return result


def _valuation_quadrant(earnings_change: float, multiple_change: float) -> str:
    if not (math.isfinite(earnings_change) and math.isfinite(multiple_change)):
        return "snapshot_missing_or_nonpositive_earnings"
    threshold = 0.05
    if earnings_change > threshold and multiple_change > threshold:
        return "eps_and_multiple_double_hit"
    if earnings_change > threshold and multiple_change < -threshold:
        return "earnings_delivered_multiple_compressed"
    if earnings_change < -threshold and multiple_change > threshold:
        return "earnings_weakened_multiple_supported"
    if earnings_change < -threshold and multiple_change < -threshold:
        return "earnings_and_multiple_double_miss"
    return "mixed_or_stable"


def _attach_future_snapshots(
    selected: pd.DataFrame, scored: pd.DataFrame
) -> pd.DataFrame:
    result = selected.reset_index(drop=True).copy()
    indices = _future_snapshot_indices(scored, result)
    future_columns = [
        "trade_date",
        "ttm_parent_net_income",
        "log_total_market_value",
        "financial_net_profit_yoy",
        "revision_np_90",
        "revision_eps_90",
        "announcement_keyword_penalty_20d",
        "announcement_keyword_litigation_20d",
        "announcement_keyword_delisting_20d",
        "framework_score",
        "eligible__framework",
    ]
    future = pd.DataFrame(index=result.index)
    valid = indices >= 0
    for column in future_columns:
        values = np.full(len(result), np.nan, dtype=object)
        values[valid] = scored.iloc[indices[valid]][column].to_numpy()
        future[f"future_{column}"] = values
    result = pd.concat([result, future], axis=1)
    result["future_snapshot_available"] = valid
    for column in future_columns[1:]:
        result[f"future_{column}"] = pd.to_numeric(
            result[f"future_{column}"], errors="coerce"
        )
    initial_mcap = np.expm1(
        np.clip(result["log_total_market_value"].to_numpy(dtype=float), 0.0, 50.0)
    )
    future_mcap = np.expm1(
        np.clip(
            result["future_log_total_market_value"].to_numpy(dtype=float), 0.0, 50.0
        )
    )
    initial_earnings = result["ttm_parent_net_income"].to_numpy(dtype=float)
    future_earnings = result["future_ttm_parent_net_income"].to_numpy(dtype=float)
    valid_decomposition = (
        valid
        & np.isfinite(initial_mcap)
        & (initial_mcap > 0.0)
        & np.isfinite(future_mcap)
        & (future_mcap > 0.0)
        & np.isfinite(initial_earnings)
        & (initial_earnings > 0.0)
        & np.isfinite(future_earnings)
        & (future_earnings > 0.0)
    )
    earnings_change = np.where(
        valid_decomposition, future_earnings / initial_earnings - 1.0, np.nan
    )
    market_cap_change = np.where(
        valid_decomposition, future_mcap / initial_mcap - 1.0, np.nan
    )
    multiple_change = np.where(
        valid_decomposition,
        (future_mcap / initial_mcap) / (future_earnings / initial_earnings) - 1.0,
        np.nan,
    )
    result["ttm_earnings_change_to_d120_snapshot"] = earnings_change
    result["market_cap_change_to_d120_snapshot"] = market_cap_change
    result["implied_multiple_change_to_d120_snapshot"] = multiple_change
    result["valuation_outcome"] = [
        _valuation_quadrant(earnings, multiple)
        for earnings, multiple in zip(earnings_change, multiple_change, strict=True)
    ]
    result["delayed_price_realization"] = (
        result["legal_net_return_d60"].le(0.0)
        & result["legal_net_return_d120"].gt(0.0)
    )
    result["revision_reversed_by_d120_snapshot"] = (
        result[["future_revision_np_90", "future_revision_eps_90"]]
        .mean(axis=1, skipna=True)
        .lt(0.0)
    )
    result["governance_incident_by_d120_snapshot"] = (
        result[
            [
                "future_announcement_keyword_penalty_20d",
                "future_announcement_keyword_litigation_20d",
                "future_announcement_keyword_delisting_20d",
            ]
        ]
        .fillna(0.0)
        .gt(0.0)
        .any(axis=1)
    )
    return result


def _eligible_outcomes(
    scored: pd.DataFrame, returns: pd.DataFrame
) -> pd.DataFrame:
    component_columns = [*SCORE_COLUMNS, "exact_full_score"]
    frame = returns.merge(
        scored[["candidate_id", *component_columns]],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    frame = frame.loc[frame["eligible__framework"]].copy()
    frame["score_percentile"] = frame.groupby("date_idx")["framework_score"].rank(
        method="average", pct=True
    )
    frame["score_quintile"] = pd.cut(
        frame["score_percentile"],
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001],
        labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"],
        include_lowest=True,
        right=True,
    ).astype(str)
    return frame


def _score_bin_summary(eligible: pd.DataFrame) -> pd.DataFrame:
    order = {name: pos for pos, name in enumerate(["Q1_low", "Q2", "Q3", "Q4", "Q5_high"], 1)}
    records: list[dict[str, Any]] = []
    for quintile, group in eligible.groupby("score_quintile", sort=False):
        record: dict[str, Any] = {
            "score_quintile": str(quintile),
            "quintile_order": order[str(quintile)],
            "candidate_count": len(group),
            "month_count": int(group["date_idx"].nunique()),
            "mean_framework_score_100": float(group["framework_score_100"].mean()),
        }
        for horizon in HORIZONS:
            values = group[f"legal_net_return_d{horizon}"]
            record[f"evaluable_count_d{horizon}"] = int(values.notna().sum())
            record[f"mean_net_return_d{horizon}"] = float(values.mean())
            record[f"median_net_return_d{horizon}"] = float(values.median())
            observed = values.dropna()
            record[f"positive_fraction_d{horizon}"] = float(
                observed.gt(0.0).mean()
            )
        records.append(record)
    return pd.DataFrame(records).sort_values("quintile_order").reset_index(drop=True)


def _rank_ic_summary(
    eligible: pd.DataFrame, study: Mapping[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for date_idx, group in eligible.groupby("date_idx", sort=True):
        record: dict[str, Any] = {
            "date_idx": int(date_idx),
            "trade_date": str(group["trade_date"].iloc[0]),
            "evaluation_year": int(group["evaluation_year"].iloc[0]),
            "eligible_count": len(group),
        }
        for horizon in HORIZONS:
            local = group[["framework_score", f"legal_net_return_d{horizon}"]].dropna()
            record[f"evaluable_count_d{horizon}"] = len(local)
            record[f"rank_ic_d{horizon}"] = (
                float(
                    local["framework_score"].rank().corr(
                        local[f"legal_net_return_d{horizon}"].rank()
                    )
                )
                if len(local) >= 20
                else math.nan
            )
        records.append(record)
    frame = pd.DataFrame(records)
    summaries: list[dict[str, Any]] = []
    groups = [
        (name, frame.loc[frame["evaluation_year"].isin(years)])
        for name, years in study["evaluation"]["periods"].items()
    ]
    groups.append(("full_history", frame))
    for period, group in groups:
        for horizon in HORIZONS:
            summaries.append(
                {
                    "period": str(period),
                    "horizon": int(horizon),
                    "rank_ic": _inference(
                        group[f"rank_ic_d{horizon}"].to_numpy(),
                        study,
                        seed_add=100 + int(horizon),
                    ),
                }
            )
    return frame, summaries


def _component_rank_ic_summary(
    eligible: pd.DataFrame, study: Mapping[str, Any]
) -> pd.DataFrame:
    components = ["framework_score", "exact_full_score", *SCORE_COLUMNS]
    records: list[dict[str, Any]] = []
    period_groups = [
        (name, {int(value) for value in years})
        for name, years in study["evaluation"]["periods"].items()
    ]
    period_groups.append(("full_history", set(study["evaluation"]["years"])))
    for component in components:
        for horizon in HORIZONS:
            monthly: list[tuple[int, float]] = []
            for _, group in eligible.groupby("date_idx", sort=True):
                local = group[
                    ["evaluation_year", component, f"legal_net_return_d{horizon}"]
                ].dropna()
                if len(local) < 20:
                    continue
                value = local[component].rank().corr(
                    local[f"legal_net_return_d{horizon}"].rank()
                )
                monthly.append((int(local["evaluation_year"].iloc[0]), float(value)))
            for period, years in period_groups:
                values = np.asarray(
                    [value for year, value in monthly if year in years],
                    dtype=np.float64,
                )
                inference = _inference(
                    values,
                    study,
                    seed_add=500 + 10 * horizon + components.index(component),
                )
                records.append(
                    {
                        "component": component,
                        "horizon": horizon,
                        "period": period,
                        "mean_rank_ic": inference["mean"],
                        "hac_lcb_95": inference["lcb_95"],
                        "hac_ucb_95": inference["ucb_95"],
                        "block_lcb_95": inference["block"]["lcb_95"],
                        "block_ucb_95": inference["block"]["ucb_95"],
                        "month_count": inference["n"],
                    }
                )
    return pd.DataFrame(records)


def _selection_comparison(
    scored: pd.DataFrame,
    returns: pd.DataFrame,
    selections: pd.DataFrame,
    old_selection_path: Path,
    study: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    old = pd.read_parquet(old_selection_path)
    old = old.loc[old["policy"].eq("exact_full_top10"), ["candidate_id"]]
    framework_ids = set(selections["candidate_id"].astype(int))
    old_ids = set(old["candidate_id"].astype(int))
    union_ids = framework_ids | old_ids
    raw_columns = [
        "candidate_id",
        "financial_net_profit_yoy",
        "financial_revenue_yoy",
        "financial_roe_avg",
        "cashflow_cfo_to_income",
        "financial_debt_to_asset",
        "revision_np_90",
        "industry_relative_ret20",
        *SCORE_COLUMNS,
        "exact_full_score",
    ]
    members = returns.loc[returns["candidate_id"].isin(union_ids)].merge(
        scored[raw_columns],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    in_framework = members["candidate_id"].isin(framework_ids)
    in_old = members["candidate_id"].isin(old_ids)
    members["selection_membership"] = np.select(
        [in_framework & in_old, in_framework, in_old],
        ["both", "framework_only", "old_only"],
        default="neither",
    )
    if bool(members["selection_membership"].eq("neither").any()):
        raise ValueError("selection_union_membership_failed")
    records: list[dict[str, Any]] = []
    periods = [
        (name, {int(value) for value in years})
        for name, years in study["evaluation"]["periods"].items()
    ]
    periods.append(("full_history", set(study["evaluation"]["years"])))
    for membership, membership_frame in members.groupby(
        "selection_membership", sort=True
    ):
        for period, years in periods:
            group = membership_frame.loc[
                membership_frame["evaluation_year"].isin(years)
            ]
            record: dict[str, Any] = {
                "selection_membership": str(membership),
                "period": str(period),
                "decision_count": len(group),
                "unique_symbol_count": int(group["symbol"].nunique()),
                "median_framework_score_100": float(
                    100.0 * group["framework_score"].median()
                ),
                "median_old_score_100": float(
                    100.0 * group["exact_full_score"].median()
                ),
            }
            for horizon in HORIZONS:
                values = group[f"legal_net_return_d{horizon}"].dropna()
                record[f"evaluable_count_d{horizon}"] = len(values)
                record[f"mean_net_return_d{horizon}"] = float(values.mean())
                record[f"median_net_return_d{horizon}"] = float(values.median())
                record[f"positive_fraction_d{horizon}"] = float(
                    values.gt(0.0).mean()
                )
            for column in (
                "financial_net_profit_yoy",
                "financial_revenue_yoy",
                "financial_roe_avg",
                "cashflow_cfo_to_income",
                "financial_debt_to_asset",
                "revision_np_90",
                "industry_relative_ret20",
                *SCORE_COLUMNS,
            ):
                record[f"median__{column}"] = float(group[column].median())
            records.append(record)
    return members, pd.DataFrame(records)


def _cohort_returns(
    eligible: pd.DataFrame,
    selected: pd.DataFrame,
    old_selection_path: Path,
) -> pd.DataFrame:
    old = pd.read_parquet(old_selection_path)
    old = old.loc[old["policy"].eq("exact_full_top10"), ["candidate_id"]]
    selected_ids = {
        "framework_top10": set(selected["candidate_id"].astype(int)),
        "old_30_30_25_10_5_top10": set(old["candidate_id"].astype(int)),
    }
    industry_benchmark: dict[int, pd.Series] = {}
    for horizon in HORIZONS:
        industry_benchmark[horizon] = eligible.groupby(
            ["date_idx", "industry_code"]
        )[f"legal_net_return_d{horizon}"].mean()
    records: list[dict[str, Any]] = []
    for date_idx, universe in eligible.groupby("date_idx", sort=True):
        for policy, identifiers in selected_ids.items():
            local = universe.loc[universe["candidate_id"].isin(identifiers)]
            if not len(local):
                continue
            record: dict[str, Any] = {
                "policy": policy,
                "date_idx": int(date_idx),
                "trade_date": str(universe["trade_date"].iloc[0]),
                "evaluation_year": int(universe["evaluation_year"].iloc[0]),
                "selected_count": len(local),
                "eligible_count": len(universe),
            }
            for horizon in HORIZONS:
                column = f"legal_net_return_d{horizon}"
                observed = local[column].dropna()
                benchmark = universe[column].mean()
                residual_values: list[float] = []
                for row in local[["industry_code", column]].itertuples(index=False):
                    value = float(getattr(row, column))
                    key = (int(date_idx), int(row.industry_code))
                    industry_mean = float(industry_benchmark[horizon].get(key, np.nan))
                    if math.isfinite(value) and math.isfinite(industry_mean):
                        residual_values.append(value - industry_mean)
                selected_mean = float(observed.mean()) if len(observed) else math.nan
                record[f"observed_count_d{horizon}"] = len(observed)
                record[f"selected_mean_net_return_d{horizon}"] = selected_mean
                record[f"eligible_mean_net_return_d{horizon}"] = float(benchmark)
                record[f"excess_net_return_d{horizon}"] = (
                    selected_mean - float(benchmark)
                    if math.isfinite(selected_mean) and math.isfinite(float(benchmark))
                    else math.nan
                )
                record[f"industry_residual_d{horizon}"] = (
                    float(np.mean(residual_values)) if residual_values else math.nan
                )
            records.append(record)
    return pd.DataFrame(records)


def _cohort_summaries(
    cohorts: pd.DataFrame, study: Mapping[str, Any]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    period_groups = [
        (name, {int(value) for value in years})
        for name, years in study["evaluation"]["periods"].items()
    ]
    period_groups.append(("full_history", set(study["evaluation"]["years"])))
    for policy, policy_frame in cohorts.groupby("policy", sort=True):
        for period, years in period_groups:
            group = policy_frame.loc[policy_frame["evaluation_year"].isin(years)]
            for horizon in HORIZONS:
                selected = group[f"selected_mean_net_return_d{horizon}"].to_numpy()
                excess = group[f"excess_net_return_d{horizon}"].to_numpy()
                residual = group[f"industry_residual_d{horizon}"].to_numpy()
                annual = (
                    group.groupby("evaluation_year")[
                        f"selected_mean_net_return_d{horizon}"
                    ]
                    .mean()
                    .dropna()
                )
                records.append(
                    {
                        "policy": str(policy),
                        "period": str(period),
                        "horizon": int(horizon),
                        "month_count": int(np.isfinite(selected).sum()),
                        "selected_net_return": _inference(
                            selected, study, seed_add=200 + horizon
                        ),
                        "eligible_excess": _inference(
                            excess, study, seed_add=300 + horizon
                        ),
                        "industry_residual": _inference(
                            residual, study, seed_add=400 + horizon
                        ),
                        "positive_year_count": int(annual.gt(0.0).sum()),
                        "year_count": len(annual),
                        "best_year": int(annual.idxmax()) if len(annual) else None,
                        "excluding_best_year_mean": (
                            float(annual.drop(index=annual.idxmax()).mean())
                            if len(annual) > 1
                            else math.nan
                        ),
                    }
                )
    return records


def _feature_profile(
    scored: pd.DataFrame, selections: pd.DataFrame
) -> pd.DataFrame:
    frame = scored.copy()
    frame["current_pe"] = np.where(
        frame["signed_log_pe"].gt(0.0),
        np.expm1(frame["signed_log_pe"].clip(upper=50.0)),
        np.nan,
    )
    frame["current_pb"] = np.where(
        frame["signed_log_pb"].gt(0.0),
        np.expm1(frame["signed_log_pb"].clip(upper=50.0)),
        np.nan,
    )
    frame["market_cap_cny_bn"] = (
        np.expm1(frame["log_total_market_value"].clip(lower=0.0, upper=50.0)) / 1e9
    )
    metrics = [
        "framework_score_100",
        "current_pe",
        "current_pb",
        "market_cap_cny_bn",
        "forward_earnings_yield",
        "normalized_earnings_yield_base",
        "ttm_fcf_yield",
        "financial_net_profit_yoy",
        "financial_revenue_yoy",
        "performance_forecast_change_mid",
        "revision_np_30",
        "revision_np_90",
        "revision_eps_30",
        "revision_eps_90",
        "financial_roe_avg",
        "financial_net_profit_margin",
        "financial_gross_profit_margin",
        "financial_asset_turnover",
        "cashflow_cfo_to_income",
        "cashflow_cfo_to_revenue",
        "cashflow_fcf_to_revenue",
        "financial_debt_to_asset",
        "balance_trade_receivables_to_total_assets",
        "balance_inventory_ratio",
        "balance_goodwill_ratio",
        "industry_relative_ret20",
        "trend_slope_20d",
        "relative_turnover_20d",
        "volume_ratio_20d",
    ]
    selected_ids = set(selections["candidate_id"].astype(int))
    populations = {
        "selected_top10": frame.loc[frame["candidate_id"].isin(selected_ids)],
        "eligible_pool": frame.loc[frame["eligible__framework"]],
        "mainboard_quality_liquidity_monthly": frame,
    }
    records: list[dict[str, Any]] = []
    for metric in metrics:
        for population, local in populations.items():
            values = pd.to_numeric(local[metric], errors="coerce").dropna()
            records.append(
                {
                    "metric": metric,
                    "population": population,
                    "row_count": len(local),
                    "observed_count": len(values),
                    "observed_fraction": float(len(values) / len(local)),
                    "mean": float(values.mean()),
                    "p25": float(values.quantile(0.25)),
                    "median": float(values.median()),
                    "p75": float(values.quantile(0.75)),
                }
            )
    return pd.DataFrame(records)


def _market_cap_quartiles(
    scored: pd.DataFrame, selections: pd.DataFrame
) -> pd.DataFrame:
    eligible = scored.loc[scored["eligible__framework"]].copy()
    eligible["market_cap_percentile"] = eligible.groupby("date_idx")[
        "log_total_market_value"
    ].rank(method="average", pct=True)
    selected = eligible.loc[
        eligible["candidate_id"].isin(selections["candidate_id"])
    ].copy()
    selected["market_cap_quartile"] = pd.cut(
        selected["market_cap_percentile"],
        [0.0, 0.25, 0.50, 0.75, 1.0000001],
        labels=["Q1_small", "Q2", "Q3", "Q4_large"],
        include_lowest=True,
    ).astype(str)
    result = (
        selected.groupby("market_cap_quartile", as_index=False)
        .agg(selected_count=("candidate_id", "size"))
        .sort_values("market_cap_quartile")
    )
    result["selected_share"] = result["selected_count"] / len(selected)
    return result


def _forecast_summary(
    selected: pd.DataFrame, eligible: pd.DataFrame
) -> pd.DataFrame:
    selected_unique = selected.sort_values("trade_date").drop_duplicates(
        ["symbol", "target_fiscal_year"], keep="last"
    )
    populations = {
        "selected_decisions": selected,
        "selected_unique_symbol_fiscal_year": selected_unique,
        "eligible_decisions": eligible,
        "selected_positive_revision": selected.loc[
            selected["positive_revision_at_signal"]
        ],
    }
    records: list[dict[str, Any]] = []
    for population, frame in populations.items():
        evaluable = frame.loc[frame["forecast_outcome_evaluable"]]
        growth = evaluable.loc[evaluable["actual_growth"].notna()]
        eps_evaluable = frame.loc[frame["eps_forecast_outcome_evaluable"]]
        eps_growth = eps_evaluable.loc[eps_evaluable["actual_eps_growth"].notna()]
        records.append(
            {
                "population": population,
                "decision_count": len(frame),
                "unique_symbol_fiscal_year_count": int(
                    frame[["symbol", "target_fiscal_year"]].drop_duplicates().shape[0]
                ),
                "evaluable_count": len(evaluable),
                "coverage": float(len(evaluable) / len(frame)) if len(frame) else math.nan,
                "median_forecast_error": float(evaluable["forecast_error"].median()),
                "median_absolute_forecast_error": float(
                    evaluable["forecast_error"].abs().median()
                ),
                "median_smape": float(evaluable["forecast_smape"].median()),
                "within_10pct_fraction": float(
                    evaluable["forecast_within_10pct"].mean()
                ),
                "within_20pct_fraction": float(
                    evaluable["forecast_within_20pct"].mean()
                ),
                "overprediction_fraction": float(
                    evaluable["forecast_overpredicted"].mean()
                ),
                "growth_evaluable_count": len(growth),
                "growth_direction_accuracy": float(
                    growth["growth_direction_match"].mean()
                ),
                "actual_growth_positive_fraction": float(
                    growth["actual_growth"].gt(0.0).mean()
                ),
                "median_actual_growth": float(growth["actual_growth"].median()),
                "eps_evaluable_count": len(eps_evaluable),
                "eps_coverage": float(len(eps_evaluable) / len(frame))
                if len(frame)
                else math.nan,
                "median_eps_forecast_error": float(
                    eps_evaluable["eps_forecast_error"].median()
                ),
                "median_absolute_eps_forecast_error": float(
                    eps_evaluable["eps_forecast_error"].abs().median()
                ),
                "eps_within_10pct_fraction": float(
                    eps_evaluable["eps_forecast_within_10pct"].mean()
                ),
                "eps_within_20pct_fraction": float(
                    eps_evaluable["eps_forecast_within_20pct"].mean()
                ),
                "eps_overprediction_fraction": float(
                    eps_evaluable["eps_forecast_overpredicted"].mean()
                ),
                "eps_growth_evaluable_count": len(eps_growth),
                "eps_growth_direction_accuracy": float(
                    eps_growth["eps_growth_direction_match"].mean()
                ),
            }
        )
    return pd.DataFrame(records)


def _scenario_summary(selected: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for dimension in ("primary_shape", "valuation_outcome"):
        for scenario, group in selected.groupby(dimension, dropna=False, sort=True):
            observed_d120 = group["legal_net_return_d120"].dropna()
            records.append(
                {
                    "dimension": dimension,
                    "scenario": str(scenario),
                    "count": len(group),
                    "share": float(len(group) / len(selected)),
                    "mean_net_return_d60": float(group["legal_net_return_d60"].mean()),
                    "median_net_return_d60": float(
                        group["legal_net_return_d60"].median()
                    ),
                    "mean_net_return_d120": float(
                        group["legal_net_return_d120"].mean()
                    ),
                    "median_net_return_d120": float(
                        group["legal_net_return_d120"].median()
                    ),
                    "positive_fraction_d120": float(
                        observed_d120.gt(0.0).mean()
                    ),
                }
            )
    for column in (
        "delayed_price_realization",
        "revision_reversed_by_d120_snapshot",
        "governance_incident_by_d120_snapshot",
    ):
        group = selected.loc[selected[column]]
        observed_d120 = group["legal_net_return_d120"].dropna()
        records.append(
            {
                "dimension": "special_event",
                "scenario": column,
                "count": len(group),
                "share": float(len(group) / len(selected)),
                "mean_net_return_d60": float(group["legal_net_return_d60"].mean()),
                "median_net_return_d60": float(group["legal_net_return_d60"].median()),
                "mean_net_return_d120": float(group["legal_net_return_d120"].mean()),
                "median_net_return_d120": float(
                    group["legal_net_return_d120"].median()
                ),
                "positive_fraction_d120": float(
                    observed_d120.gt(0.0).mean()
                ),
            }
        )
    return pd.DataFrame(records)


def _industry_mix(selected: pd.DataFrame) -> pd.DataFrame:
    return (
        selected.groupby(["industry_code", "industry_name"], as_index=False)
        .agg(
            selected_count=("candidate_id", "size"),
            unique_symbol_count=("symbol", "nunique"),
            month_count=("date_idx", "nunique"),
            mean_score_100=("framework_score_100", "mean"),
            mean_net_return_d60=("legal_net_return_d60", "mean"),
            mean_net_return_d120=("legal_net_return_d120", "mean"),
        )
        .assign(selected_share=lambda frame: frame["selected_count"] / len(selected))
        .sort_values(["selected_count", "industry_name"], ascending=[False, True])
        .reset_index(drop=True)
    )


def _rating_summary(selected: pd.DataFrame) -> pd.DataFrame:
    order = {"A+": 1, "A": 2, "A-": 3, "B+": 4, "B": 5, "C": 6, "unrated": 7}
    result = (
        selected.groupby("framework_rating", as_index=False)
        .agg(
            selected_count=("candidate_id", "size"),
            mean_score_100=("framework_score_100", "mean"),
            mean_net_return_d60=("legal_net_return_d60", "mean"),
            mean_net_return_d120=("legal_net_return_d120", "mean"),
            positive_fraction_d120=(
                "legal_net_return_d120",
                lambda values: float(pd.Series(values).dropna().gt(0.0).mean()),
            ),
            within_20pct_forecast_fraction=(
                "forecast_error",
                lambda values: float(
                    pd.Series(values).dropna().abs().le(0.20).mean()
                ),
            ),
        )
        .assign(selected_share=lambda frame: frame["selected_count"] / len(selected))
    )
    result["rating_order"] = result["framework_rating"].map(order)
    return result.sort_values("rating_order").reset_index(drop=True)


def _example_outcomes(selected: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "symbol",
        "trade_date",
        "industry_name",
        "framework_score_100",
        "framework_rating",
        "selection_rank",
        "legal_net_return_d60",
        "legal_net_return_d120",
        "primary_shape",
        "valuation_outcome",
        "forecast_error",
        "actual_growth",
        "revision_reversed_by_d120_snapshot",
    ]
    evaluable = selected.loc[selected["legal_net_return_d120"].notna()].copy()
    winners = evaluable.nlargest(10, "legal_net_return_d120")[columns].copy()
    winners.insert(0, "example_type", "largest_d120_winner")
    losers = evaluable.nsmallest(10, "legal_net_return_d120")[columns].copy()
    losers.insert(0, "example_type", "largest_d120_loser")
    return pd.concat([winners, losers], ignore_index=True)


def _build_account_book(
    scored: pd.DataFrame,
    pack: CandidateCompleteAuditPack,
    study: Mapping[str, Any],
) -> tuple[finite.ForecastBook, pd.DataFrame, dict[str, Any]]:
    selection = dict(study["selection"])
    book = finite.ForecastBook(
        "qver_framework_top10",
        top_k=int(selection["top_k"]),
        candidate_scan_k=int(selection["candidate_scan_count"]),
    )
    cutoff = int(
        np.flatnonzero(
            np.asarray(pack.date_values, dtype=str)
            == str(study["source"]["maximum_account_mark_date"])
        )[0]
    )
    records: list[pd.DataFrame] = []
    excluded = 0
    for date_idx, local in scored.groupby("date_idx", sort=True):
        if int(date_idx) + int(pack.execution_days) > cutoff:
            excluded += 1
            continue
        chosen = _select_indices(
            local,
            top_k=int(selection["candidate_scan_count"]),
            industry_cap=int(selection["maximum_names_per_pit_industry"]),
        )
        if not len(chosen):
            continue
        scan = local.iloc[chosen].copy()
        scan["candidate_scan_rank"] = np.arange(1, len(scan) + 1)
        book.add_day(
            date_idx=int(date_idx),
            symbol_idx=scan["symbol_idx"].to_numpy(dtype=np.int64),
            score=scan["framework_score"].to_numpy(dtype=np.float64),
            planned_day=np.full(len(scan), 60, dtype=np.int16),
        )
        records.append(
            scan[
                [
                    "candidate_id",
                    "date_idx",
                    "trade_date",
                    "symbol",
                    "symbol_idx",
                    "industry_code",
                    "candidate_scan_rank",
                    "framework_score",
                ]
            ]
        )
    if not book.days or not records:
        raise ValueError("account_forecast_book_empty")
    candidates = pd.concat(records, ignore_index=True)
    top = candidates.loc[candidates["candidate_scan_rank"].le(int(selection["top_k"]))]
    return book, candidates, {
        "signal_date_count": len(book.days),
        "first_signal_date": str(pack.date_values[min(book.days)]),
        "last_signal_date": str(pack.date_values[max(book.days)]),
        "boundary_excluded_signal_date_count": excluded,
        "top_count_min": int(top.groupby("date_idx").size().min()),
        "top_count_max": int(top.groupby("date_idx").size().max()),
        "scan_count_min": int(candidates.groupby("date_idx").size().min()),
        "scan_count_max": int(candidates.groupby("date_idx").size().max()),
        "maximum_same_industry_top_names": int(
            top.groupby(["date_idx", "industry_code"])
            .size()
            .groupby("date_idx")
            .max()
            .max()
        ),
        "future_outcomes_used": False,
    }


def _simulate_account(
    *,
    market: finite.BacktestMarket,
    book: finite.ForecastBook,
    study: Mapping[str, Any],
    signal_amount: np.ndarray,
    terminal_recovery_date_idx: int,
    target_gross_fraction: float,
    years: tuple[int, ...],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    account = dict(study["account"])
    return finite.simulate_portfolio(
        market=market,
        book=book,
        raw_top3_paths={},
        policy=finite.PolicySpec(name="fixed_d60", kind="fixed", fixed_day=60),
        slots=int(account["position_slots"]),
        cost_scenario=str(account["cost_scenario"]),
        first_signal_date_idx=int(min(book.days)),
        last_signal_date_idx=int(max(book.days)),
        starting_cash=float(account["starting_cash_cny"]),
        allow_pyramiding=False,
        replace_rejected_from_ranked_candidates=bool(
            account["replace_duplicate_ranked_names"]
        ),
        calendar_years=years,
        top_k=int(study["selection"]["top_k"]),
        signal_amount_panel=signal_amount,
        maximum_signal_amount_fraction=float(
            account["maximum_signal_day_amount_fraction"]
        ),
        target_gross_fraction=float(target_gross_fraction),
        terminal_recovery_date_idx=int(terminal_recovery_date_idx),
    )


def _account_runs(
    panel: base.StockPanel,
    scored: pd.DataFrame,
    study: Mapping[str, Any],
    pack_manifest: Path,
) -> tuple[
    dict[str, Any],
    dict[str, tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]],
    pd.DataFrame,
]:
    pack = CandidateCompleteAuditPack(pack_manifest)
    feasibility._validate_pack_alignment(panel, pack)
    book, candidates, selection_audit = _build_account_book(scored, pack, study)
    adjust_factor, adjust_audit = feasibility._load_adjust_factor_panel(
        pack, study=study
    )
    market = old_account._market(pack, adjust_factor=adjust_factor)
    signal_amount = feasibility._signal_amount_panel(panel, pack)
    cutoff = int(
        np.flatnonzero(
            np.asarray(pack.date_values, dtype=str)
            == str(study["source"]["maximum_account_mark_date"])
        )[0]
    )
    years = tuple(int(value) for value in study["evaluation"]["years"])
    full = _simulate_account(
        market=market,
        book=book,
        study=study,
        signal_amount=signal_amount,
        terminal_recovery_date_idx=cutoff,
        target_gross_fraction=1.0,
        years=years,
    )
    risk_budget = feasibility._derive_risk_budget(
        full[1],
        starting_cash=float(study["account"]["starting_cash_cny"]),
        config=study["risk_budget"],
    )
    risk = _simulate_account(
        market=market,
        book=book,
        study=study,
        signal_amount=signal_amount,
        terminal_recovery_date_idx=cutoff,
        target_gross_fraction=float(risk_budget["frozen_target_gross_fraction"]),
        years=years,
    )
    confirmation_start = int(
        np.searchsorted(np.asarray(pack.date_values, dtype=str), "2023-01-01")
    )
    confirmation_book = feasibility._book_from_date(book, confirmation_start)
    confirmation = _simulate_account(
        market=market,
        book=confirmation_book,
        study=study,
        signal_amount=signal_amount,
        terminal_recovery_date_idx=cutoff,
        target_gross_fraction=float(risk_budget["frozen_target_gross_fraction"]),
        years=(2023, 2024, 2025),
    )
    capacity = feasibility._capacity_audit(
        panel,
        risk[2],
        maximum_fraction=float(study["account"]["maximum_signal_day_amount_fraction"]),
    )
    if not capacity["all_orders_within_limit"]:
        raise ValueError("account_capacity_limit_exceeded")
    positive_pnl = risk[2].loc[risk[2]["net_pnl_cny"].gt(0.0), "net_pnl_cny"].sort_values(
        ascending=False
    )
    concentration = {
        "positive_trade_count": len(positive_pnl),
        "positive_pnl_cny": float(positive_pnl.sum()),
        "top_10_positive_pnl_share": float(
            positive_pnl.head(10).sum() / positive_pnl.sum()
        ),
        "top_1pct_positive_pnl_share": float(
            positive_pnl.head(max(1, math.ceil(0.01 * len(positive_pnl)))).sum()
            / positive_pnl.sum()
        ),
    }
    audit = {
        "selection": selection_audit,
        "risk_budget": risk_budget,
        "capacity": capacity,
        "adjust_factor": adjust_audit,
        "winner_concentration": concentration,
    }
    return audit, {"full": full, "risk_budget": risk, "risk_2023_restart": confirmation}, candidates


def _summary_account_metrics(
    audit: Mapping[str, Any],
    runs: Mapping[
        str,
        tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, list[dict[str, Any]]],
    ],
    old_summary_path: Path,
) -> dict[str, Any]:
    old_summary = _read_json(old_summary_path.parent.parent / "seq100_exact_value_growth_account_v1" / "summary.json")
    result: dict[str, Any] = {"audit": dict(audit), "runs": {}}
    for name, run in runs.items():
        metric, _equity, _trades, annual = run
        result["runs"][name] = {
            "metric": metric,
            "annual": annual,
            "positive_year_count": int(
                sum(float(row["net_return"]) > 0.0 for row in annual)
            ),
            "year_count": len(annual),
        }
    result["old_frozen_30_30_25_10_5_comparator"] = {
        "risk_metric": old_summary["risk_metric"],
        "risk_annual": old_summary["risk_annual"],
        "risk_budget": old_summary["risk_budget"],
    }
    return result


def _compact_characteristic_summary(profile: pd.DataFrame) -> dict[str, Any]:
    table = profile.pivot(index="metric", columns="population", values="median")
    wanted = [
        "current_pe",
        "current_pb",
        "market_cap_cny_bn",
        "financial_net_profit_yoy",
        "financial_revenue_yoy",
        "financial_roe_avg",
        "cashflow_cfo_to_income",
        "financial_debt_to_asset",
        "revision_np_90",
        "industry_relative_ret20",
    ]
    return {
        metric: {
            "selected_median": float(table.loc[metric, "selected_top10"]),
            "eligible_median": float(table.loc[metric, "eligible_pool"]),
            "monthly_universe_median": float(
                table.loc[metric, "mainboard_quality_liquidity_monthly"]
            ),
        }
        for metric in wanted
    }


def run_study(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    study, frozen_path = load_study(study_path)
    sources = _source_contract(study)
    root = base._resolve_path(output_root)
    implementation_path = Path(__file__)
    fingerprint = _payload_hash(
        {
            "study_sha256": base._sha256_file(frozen_path),
            "implementation_sha256": base._sha256_file(implementation_path),
            "source_hashes": {
                key: base._sha256_file(path)
                for key, path in sources.items()
                if path.is_file()
            },
        }
    )
    summary_path = root / "summary.json"
    if summary_path.is_file() and not force:
        current = _read_json(summary_path)
        if current.get("experiment_fingerprint") != fingerprint:
            raise ValueError("existing_output_fingerprint_mismatch")
        if all(
            base._record_valid(record, verify_hash=True)
            for record in current.get("files", {}).values()
        ):
            return current
    panel = base.load_panel(
        input_manifest_path=sources["input_manifest"],
        label_manifest_path=sources["label_manifest"],
    )
    if bool((panel.years == int(study["source"]["forbidden_year"])).any()):
        raise ValueError("forbidden_2026_panel_row")
    source_features = pd.read_parquet(sources["candidate_features"])
    features = _attach_extra_features(panel, source_features)
    scored = _score_framework(features, study)
    selections = _selection_frame(scored, sources["old_selections"], study)
    selections = _attach_industry_names(selections, sources["membership_root"])

    # Outcome reads start only after the score and selected identities are frozen.
    returns = _candidate_returns(panel, scored, study)
    eligible = _eligible_outcomes(scored, returns)
    selected_returns = returns.loc[
        returns["candidate_id"].isin(selections["candidate_id"])
    ]
    selected = selections.merge(
        selected_returns.drop(
            columns=[
                column
                for column in selections.columns
                if column in selected_returns.columns and column != "candidate_id"
            ]
        ),
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    selected = selected.merge(
        _selected_paths(panel, selections, study),
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    selected = _attach_future_snapshots(selected, scored)

    actuals = _annual_actuals(sources["income_statement_quarterly"], study)
    selected = _attach_forecast_realization(selected, actuals, study)
    eligible_forecast_source = scored.loc[scored["eligible__framework"]].merge(
        eligible[
            [
                "candidate_id",
                "legal_net_return_d20",
                "legal_net_return_d60",
                "legal_net_return_d120",
            ]
        ],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    eligible_forecast = _attach_forecast_realization(
        eligible_forecast_source, actuals, study
    )

    score_bins = _score_bin_summary(eligible)
    rank_ic, rank_ic_summaries = _rank_ic_summary(eligible, study)
    component_rank_ic = _component_rank_ic_summary(eligible, study)
    cohorts = _cohort_returns(eligible, selections, sources["old_selections"])
    cohort_summaries = _cohort_summaries(cohorts, study)
    comparison_members, comparison_summary = _selection_comparison(
        scored,
        returns,
        selections,
        sources["old_selections"],
        study,
    )
    profile = _feature_profile(scored, selections)
    market_cap = _market_cap_quartiles(scored, selections)
    forecasts = _forecast_summary(selected, eligible_forecast)
    scenarios = _scenario_summary(selected)
    industries = _industry_mix(selected)
    ratings = _rating_summary(selected)
    examples = _example_outcomes(selected)

    account_audit, account_runs, account_candidates = _account_runs(
        panel, scored, study, sources["pack_manifest"]
    )
    account_summary = _summary_account_metrics(
        account_audit, account_runs, sources["exact_policy_summary"]
    )

    scored_output = scored[
        [
            "candidate_id",
            "row_position",
            "date_idx",
            "trade_date",
            "symbol",
            "symbol_idx",
            "evaluation_year",
            "industry_code",
            *SCORE_COLUMNS,
            "framework_score",
            "framework_score_100",
            "framework_rating",
            "eligible__framework",
            "exact_full_score",
        ]
    ]
    outputs: dict[str, tuple[Path, pd.DataFrame]] = {
        "scored_candidates": (root / "scored_candidates.parquet", scored_output),
        "selections": (root / "selections.parquet", selected),
        "eligible_outcomes": (root / "eligible_outcomes.parquet", eligible),
        "score_quintiles": (root / "score_quintiles.parquet", score_bins),
        "monthly_rank_ic": (root / "monthly_rank_ic.parquet", rank_ic),
        "component_rank_ic": (
            root / "component_rank_ic.parquet",
            component_rank_ic,
        ),
        "cohort_returns": (root / "cohort_returns.parquet", cohorts),
        "selection_comparison_members": (
            root / "selection_comparison_members.parquet",
            comparison_members,
        ),
        "selection_comparison_summary": (
            root / "selection_comparison_summary.parquet",
            comparison_summary,
        ),
        "feature_profile": (root / "feature_profile.parquet", profile),
        "market_cap_quartiles": (root / "market_cap_quartiles.parquet", market_cap),
        "forecast_realization": (root / "forecast_realization.parquet", forecasts),
        "scenario_summary": (root / "scenario_summary.parquet", scenarios),
        "industry_mix": (root / "industry_mix.parquet", industries),
        "rating_summary": (root / "rating_summary.parquet", ratings),
        "example_outcomes": (root / "example_outcomes.parquet", examples),
        "account_candidates": (root / "account_candidates.parquet", account_candidates),
    }
    for run_name, run in account_runs.items():
        outputs[f"account_{run_name}_equity"] = (
            root / f"account_{run_name}_equity.parquet",
            run[1],
        )
        outputs[f"account_{run_name}_trades"] = (
            root / f"account_{run_name}_trades.parquet",
            run[2],
        )
        outputs[f"account_{run_name}_annual"] = (
            root / f"account_{run_name}_annual.parquet",
            pd.DataFrame(run[3]),
        )
    for path, frame in outputs.values():
        _write_parquet(path, frame)

    overlap_count = int(selections["old_exact_selected"].sum())
    selected_d120 = selected["legal_net_return_d120"].dropna()
    positive_returns = selected_d120.loc[selected_d120.gt(0.0)].sort_values(
        ascending=False
    )
    full_framework_d120 = next(
        row
        for row in cohort_summaries
        if row["policy"] == "framework_top10"
        and row["period"] == "full_history"
        and row["horizon"] == 120
    )
    full_framework_d60 = next(
        row
        for row in cohort_summaries
        if row["policy"] == "framework_top10"
        and row["period"] == "full_history"
        and row["horizon"] == 60
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed_retrospective_framework_effect_audit",
        "study_id": STUDY_ID,
        "experiment_fingerprint": fingerprint,
        "study_sha256": base._sha256_file(frozen_path),
        "implementation_sha256": base._sha256_file(implementation_path),
        "candidate_selection_precedes_outcome_reads": True,
        "row_count": len(scored),
        "month_count": int(scored["date_idx"].nunique()),
        "start_date": str(scored["trade_date"].min()),
        "end_date": str(scored["trade_date"].max()),
        "eligible_count": int(scored["eligible__framework"].sum()),
        "selected_count": len(selections),
        "selected_month_count": int(selections["date_idx"].nunique()),
        "old_selection_overlap_count": overlap_count,
        "old_selection_overlap_fraction": float(overlap_count / len(selections)),
        "framework_coverage": {
            "quantified_families": list(FRAMEWORK_WEIGHTS),
            "explicitly_uncovered": study["score_contract"]["explicitly_uncovered"],
            "full_qualitative_framework_claim_allowed": False,
        },
        "characteristic_medians": _compact_characteristic_summary(profile),
        "market_cap_quartiles": market_cap.to_dict(orient="records"),
        "rating_summary": ratings.to_dict(orient="records"),
        "rank_ic_summaries": rank_ic_summaries,
        "component_rank_ic": component_rank_ic.loc[
            component_rank_ic["period"].eq("full_history")
        ].to_dict(orient="records"),
        "score_quintiles": score_bins.to_dict(orient="records"),
        "cohort_summaries": cohort_summaries,
        "selection_comparison": comparison_summary.loc[
            comparison_summary["period"].eq("full_history")
        ].to_dict(orient="records"),
        "primary_d60_summary": full_framework_d60,
        "primary_d120_summary": full_framework_d120,
        "forecast_realization": forecasts.to_dict(orient="records"),
        "scenario_summary": scenarios.to_dict(orient="records"),
        "industry_top10": industries.head(10).to_dict(orient="records"),
        "selected_return_concentration": {
            "d120_evaluable_count": len(selected_d120),
            "d120_positive_count": len(positive_returns),
            "top_1pct_positive_return_sum_share": float(
                positive_returns.head(
                    max(1, math.ceil(0.01 * len(positive_returns)))
                ).sum()
                / positive_returns.sum()
            ),
            "top_10_positive_return_sum_share": float(
                positive_returns.head(10).sum() / positive_returns.sum()
            ),
        },
        "account": account_summary,
        "forbidden_2026_read_count": 0,
        "retrospective_only": True,
        "profit_claim_allowed": False,
        "decision": {
            "score_is_return_magnitude_forecast": False,
            "target_price_accuracy_tested": False,
            "target_price_accuracy_reason": "the supplied framework defines scenario valuation work but no frozen company-level target-price series",
            "historical_rank_information_present": bool(
                float(full_framework_d120["eligible_excess"]["mean"]) > 0.0
            ),
            "historical_account_positive": bool(
                float(account_summary["runs"]["risk_budget"]["metric"]["liquidated_total_return"])
                > 0.0
            ),
            "live_or_stable_profit_claim_allowed": False,
            "reason": "all periods and framework proxy choices are retrospectively consumed; qualitative fields remain unmeasured",
        },
        "files": {
            name: base._file_record(path) for name, (path, _frame) in outputs.items()
        },
    }
    base._write_json(summary_path, summary)
    return summary


def validate_summary(path: str | Path) -> dict[str, Any]:
    summary = _read_json(base._resolve_path(path))
    checks = {
        "schema": summary.get("schema") == SUMMARY_SCHEMA,
        "study": summary.get("study_id") == STUDY_ID,
        "completed": summary.get("status")
        == "completed_retrospective_framework_effect_audit",
        "selection_before_outcomes": summary.get(
            "candidate_selection_precedes_outcome_reads"
        )
        is True,
        "no_2026": summary.get("forbidden_2026_read_count") == 0,
        "retrospective": summary.get("retrospective_only") is True,
        "no_profit_claim": summary.get("profit_claim_allowed") is False,
        "files": all(
            base._record_valid(record, verify_hash=True)
            for record in summary.get("files", {}).values()
        ),
    }
    return {"passed": bool(all(checks.values())), "checks": checks}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_study(
        study_path=args.study,
        output_root=args.output_root,
        force=bool(args.force),
    )
    if args.validate:
        validation = validate_summary(Path(args.output_root) / "summary.json")
        if not validation["passed"]:
            raise SystemExit(json.dumps(validation, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
