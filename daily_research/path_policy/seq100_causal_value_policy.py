"""Causal value, earnings-revision, quality and valuation-exit study."""

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

import duckdb
import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_stock_distribution as base

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_causal_value_policy_v1"
SUMMARY_SCHEMA = "seq100_causal_value_policy_summary/1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_causal_value_policy_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_causal_value_policy_v1"
)
EXPECTED_POLICIES = (
    "value_only",
    "value_analyst_covered_control",
    "value_actual",
    "value_actual_revision",
    "value_actual_revision_price_confirmed",
    "value_quality_actual_revision",
    "quality_only",
)
EXIT_POLICIES = (
    "d20_timeout",
    "d60_timeout",
    "quarter_of_peer_median_log_gap",
    "peer_industry_median",
    "peer_industry_right_edge_q75",
)
FEATURES = (
    "signed_log_pe",
    "signed_log_pb",
    "log_total_market_value",
    "cashflow_free_cash_flow",
    "financial_net_profit_yoy",
    "financial_revenue_yoy",
    "performance_forecast_change_mid",
    "cashflow_cfo_to_income",
    "financial_roe_avg",
    "financial_net_profit_margin",
    "financial_debt_to_asset",
    "industry_relative_ret20",
    "trend_slope_20d",
)
FORBIDDEN_FACTOR_TOKENS = ("52_week_low", "distance_from_52_week_low")


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
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def load_study(
    path: str | Path = DEFAULT_STUDY_PATH,
) -> tuple[dict[str, Any], Path]:
    study_path = base._resolve_path(path)
    study = _read_json(study_path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("study_id_mismatch")
    source = dict(study["source"])
    if source.get("expected_input_fingerprint") != base.EXPECTED_INPUT_FINGERPRINT:
        raise ValueError("input_fingerprint_contract_mismatch")
    if int(source.get("expected_row_count", -1)) != base.EXPECTED_ROW_COUNT:
        raise ValueError("input_row_count_contract_mismatch")
    if int(source.get("forbidden_year", -1)) != base.FORBIDDEN_YEAR:
        raise ValueError("forbidden_year_contract_mismatch")
    signals = dict(study["signals"])
    if tuple(signals.get("policies", ())) != EXPECTED_POLICIES:
        raise ValueError("policy_contract_mismatch")
    execution = dict(study["execution"])
    if tuple(execution.get("valuation_exit_policies", ())) != EXIT_POLICIES[1:]:
        raise ValueError("exit_policy_contract_mismatch")
    serialized = json.dumps(study["factor_contract"], ensure_ascii=False).lower()
    if any(
        token in serialized and token not in serialized.split("explicitly_excluded")[1]
        for token in FORBIDDEN_FACTOR_TOKENS
    ):
        raise ValueError("forbidden_low_price_reward_in_factor_contract")
    excluded = " ".join(study["factor_contract"]["explicitly_excluded"]).lower()
    if not all(token in excluded for token in FORBIDDEN_FACTOR_TOKENS):
        raise ValueError("forbidden_low_price_reward_not_explicit")
    if bool(study["decision_boundary"].get("account_replay_performed", True)):
        raise ValueError("study_must_start_without_account_replay")
    return study, study_path


def _source_contract(study: Mapping[str, Any]) -> dict[str, Path]:
    source = dict(study["source"])
    pairs = {
        "input_manifest": "input_manifest_sha256",
        "label_manifest": "label_manifest_sha256",
        "pack_manifest": "pack_manifest_sha256",
        "research_report": "research_report_sha256",
        "research_report_forecast": "research_report_forecast_sha256",
    }
    paths: dict[str, Path] = {}
    for key, hash_key in pairs.items():
        path = base._resolve_path(source[key])
        if not path.is_file():
            raise ValueError(f"source_missing:{key}")
        if base._sha256_file(path).lower() != str(source[hash_key]).lower():
            raise ValueError(f"source_hash_mismatch:{key}")
        paths[key] = path
    return paths


def _month_end_rows(panel: base.StockPanel) -> np.ndarray:
    identity = panel.row_index[["date_idx", "trade_date"]]
    dates = identity.drop_duplicates("date_idx").copy()
    dates["month"] = dates["trade_date"].astype(str).str[:7]
    month_end = dates.groupby("month", sort=True)["date_idx"].max().to_numpy()
    selected = np.isin(panel.date_idx, month_end)
    rows = np.flatnonzero(selected).astype(np.int64, copy=False)
    if not len(rows):
        raise ValueError("month_end_rows_empty")
    trade_dates = panel.trade_date[rows]
    if bool(np.char.startswith(trade_dates.astype(str), "2026-").any()):
        raise ValueError("forbidden_2026_month_end_row")
    return rows


def _analyst_consensus(
    signals: pd.DataFrame,
    *,
    report_path: Path,
    forecast_path: Path,
    lookback_days: int,
    revision_gap_days: int,
) -> pd.DataFrame:
    grid = signals[["candidate_id", "symbol", "signal_date"]].copy()
    grid["signal_date"] = pd.to_datetime(grid["signal_date"])
    grid["prior_date"] = grid["signal_date"] - pd.Timedelta(days=int(revision_gap_days))
    grid["target_year"] = grid["signal_date"].dt.year + 1
    con = duckdb.connect()
    try:
        con.register("signals", grid)
        forecast_sql = forecast_path.as_posix().replace("'", "''")
        report_sql = report_path.as_posix().replace("'", "''")
        query = f"""
        WITH asofs AS (
          SELECT candidate_id, symbol, target_year, 'current' AS stage,
                 signal_date AS asof_date
          FROM signals
          UNION ALL
          SELECT candidate_id, symbol, target_year, 'prior' AS stage,
                 prior_date AS asof_date
          FROM signals
        ), latest AS (
          SELECT a.candidate_id, a.stage, f.eps, f.net_profit,
                 coalesce(
                   nullif(trim(r.normalized_institution), ''), r.report_id
                 ) AS institution,
                 row_number() OVER (
                   PARTITION BY a.candidate_id, a.stage,
                     coalesce(nullif(trim(r.normalized_institution), ''), r.report_id)
                   ORDER BY try_cast(f.feature_available_date AS DATE) DESC,
                            f.report_id DESC
                 ) AS recency_rank
          FROM asofs a
          JOIN read_parquet('{forecast_sql}') f
            ON f.symbol = a.symbol
           AND cast(f.forecast_year AS INTEGER) = a.target_year
          JOIN read_parquet('{report_sql}') r ON r.report_id = f.report_id
          WHERE try_cast(f.feature_available_date AS DATE) <= a.asof_date
            AND try_cast(f.feature_available_date AS DATE)
                > a.asof_date - INTERVAL {int(lookback_days)} DAY
            AND try_cast(f.feature_available_date AS DATE) < DATE '2026-01-01'
        ), consensus AS (
          SELECT candidate_id, stage,
                 count(*) FILTER (
                   WHERE recency_rank = 1 AND eps IS NOT NULL
                 ) AS eps_institution_count,
                 count(*) FILTER (
                   WHERE recency_rank = 1 AND net_profit IS NOT NULL
                 ) AS net_profit_institution_count,
                 median(eps) FILTER (
                   WHERE recency_rank = 1 AND eps IS NOT NULL
                 ) AS eps_median,
                 median(net_profit) FILTER (
                   WHERE recency_rank = 1 AND net_profit IS NOT NULL
                 ) AS net_profit_median
          FROM latest
          GROUP BY candidate_id, stage
        )
        SELECT candidate_id,
               max(eps_institution_count) FILTER (WHERE stage = 'current')
                 AS current_eps_institution_count,
               max(eps_institution_count) FILTER (WHERE stage = 'prior')
                 AS prior_eps_institution_count,
               max(net_profit_institution_count) FILTER (WHERE stage = 'current')
                 AS current_net_profit_institution_count,
               max(net_profit_institution_count) FILTER (WHERE stage = 'prior')
                 AS prior_net_profit_institution_count,
               max(eps_median) FILTER (WHERE stage = 'current') AS current_eps,
               max(eps_median) FILTER (WHERE stage = 'prior') AS prior_eps,
               max(net_profit_median) FILTER (WHERE stage = 'current')
                 AS current_net_profit,
               max(net_profit_median) FILTER (WHERE stage = 'prior')
                 AS prior_net_profit
        FROM consensus
        GROUP BY candidate_id
        """
        result = con.execute(query).fetchdf()
    finally:
        con.close()
    if bool(result["candidate_id"].duplicated().any()):
        raise ValueError("analyst_consensus_candidate_duplicate")
    return result


def _rank_percentile(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    result = np.full(len(x), np.nan, dtype=np.float64)
    good = np.isfinite(x)
    if good.any():
        result[good] = (
            pd.Series(x[good]).rank(method="average", pct=True).to_numpy(dtype=float)
        )
    return result


def _industry_percentile(
    values: np.ndarray, industry: np.ndarray, *, minimum_group_size: int
) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    codes = np.asarray(industry, dtype=np.int64)
    result = _rank_percentile(x)
    for code in np.unique(codes[codes >= 0]):
        positions = np.flatnonzero(codes == int(code))
        good_count = int(np.isfinite(x[positions]).sum())
        if good_count < int(minimum_group_size):
            continue
        result[positions] = _rank_percentile(x[positions])
    return result


def _family_mean(parts: Sequence[np.ndarray], *, minimum_count: int) -> np.ndarray:
    matrix = np.column_stack(parts).astype(np.float64, copy=False)
    finite = np.isfinite(matrix)
    count = finite.sum(axis=1)
    total = np.where(finite, matrix, 0.0).sum(axis=1)
    result = np.full(len(matrix), np.nan, dtype=np.float64)
    good = count >= int(minimum_count)
    result[good] = total[good] / count[good]
    return result


def _peer_reference(
    values: np.ndarray,
    industries: np.ndarray,
    *,
    quantile: float,
    minimum_group_size: int,
) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    codes = np.asarray(industries, dtype=np.int64)
    positive = np.isfinite(x) & (x > 0.0)
    global_reference = (
        float(np.quantile(x[positive], quantile)) if positive.any() else np.nan
    )
    result = np.full(len(x), global_reference, dtype=np.float64)
    for code in np.unique(codes[codes >= 0]):
        positions = np.flatnonzero(codes == int(code))
        group = x[positions]
        good = np.isfinite(group) & (group > 0.0)
        if int(good.sum()) < int(minimum_group_size):
            continue
        result[positions] = float(np.quantile(group[good], quantile))
    return result


def _peer_target_ratios(
    pe: np.ndarray,
    pb: np.ndarray,
    industries: np.ndarray,
    *,
    minimum_group_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pe_values = np.asarray(pe, dtype=np.float64)
    pb_values = np.asarray(pb, dtype=np.float64)
    valid = (
        np.isfinite(pe_values)
        & (pe_values > 0.0)
        & np.isfinite(pb_values)
        & (pb_values > 0.0)
    )
    med_pe = _peer_reference(
        pe_values,
        industries,
        quantile=0.5,
        minimum_group_size=minimum_group_size,
    )
    med_pb = _peer_reference(
        pb_values,
        industries,
        quantile=0.5,
        minimum_group_size=minimum_group_size,
    )
    edge_pe = _peer_reference(
        pe_values,
        industries,
        quantile=0.75,
        minimum_group_size=minimum_group_size,
    )
    edge_pb = _peer_reference(
        pb_values,
        industries,
        quantile=0.75,
        minimum_group_size=minimum_group_size,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        median_gap = np.minimum(np.log(med_pe / pe_values), np.log(med_pb / pb_values))
        edge_gap = np.minimum(np.log(edge_pe / pe_values), np.log(edge_pb / pb_values))
    median_gap = np.maximum(median_gap, 0.0)
    edge_gap = np.maximum(edge_gap, 0.0)
    median_ratio = np.full(len(pe_values), np.nan, dtype=np.float64)
    edge_ratio = np.full(len(pe_values), np.nan, dtype=np.float64)
    quarter_ratio = np.full(len(pe_values), np.nan, dtype=np.float64)
    median_ratio[valid] = np.exp(np.minimum(median_gap[valid], math.log(20.0)))
    edge_ratio[valid] = np.exp(np.minimum(edge_gap[valid], math.log(20.0)))
    quarter_ratio[valid] = np.exp(np.minimum(0.25 * median_gap[valid], math.log(20.0)))
    return quarter_ratio, median_ratio, edge_ratio


def _feature_frame(
    panel: base.StockPanel,
    rows: np.ndarray,
    consensus: pd.DataFrame,
    *,
    minimum_group_size: int,
    minimum_current_institutions: int,
    minimum_prior_institutions: int,
) -> pd.DataFrame:
    positions = base._feature_positions(panel, FEATURES)
    matrix = base._feature_matrix(panel, rows, positions).astype(np.float64)
    raw = {name: matrix[:, pos] for pos, name in enumerate(FEATURES)}
    identity = panel.row_index.iloc[rows]
    frame = identity[["candidate_id", "date_idx", "trade_date", "symbol"]].copy()
    frame.insert(0, "row_position", rows)
    frame["signal_date"] = pd.to_datetime(frame["trade_date"])
    frame["evaluation_year"] = frame["signal_date"].dt.year.astype(int)
    frame["industry_code"] = panel.industry_code[rows]
    frame = frame.merge(consensus, on="candidate_id", how="left", validate="one_to_one")

    current_np_n = frame["current_net_profit_institution_count"].fillna(0)
    prior_np_n = frame["prior_net_profit_institution_count"].fillna(0)
    current_eps_n = frame["current_eps_institution_count"].fillna(0)
    prior_eps_n = frame["prior_eps_institution_count"].fillna(0)
    frame["analyst_covered"] = (
        (current_np_n >= int(minimum_current_institutions))
        & (prior_np_n >= int(minimum_prior_institutions))
        & (current_eps_n >= int(minimum_current_institutions))
        & (prior_eps_n >= int(minimum_prior_institutions))
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        np_revision = np.log(
            frame["current_net_profit"].to_numpy(dtype=float)
            / frame["prior_net_profit"].to_numpy(dtype=float)
        )
        eps_revision = np.log(
            frame["current_eps"].to_numpy(dtype=float)
            / frame["prior_eps"].to_numpy(dtype=float)
        )
    positive_np = (frame["current_net_profit"] > 0) & (frame["prior_net_profit"] > 0)
    positive_eps = (frame["current_eps"] > 0) & (frame["prior_eps"] > 0)
    np_revision[~positive_np.to_numpy()] = np.nan
    eps_revision[~positive_eps.to_numpy()] = np.nan
    covered = frame["analyst_covered"].to_numpy(dtype=bool)
    np_revision[~covered] = np.nan
    eps_revision[~covered] = np.nan
    frame["analyst_net_profit_revision_log"] = np.clip(np_revision, -0.5, 0.5)
    frame["analyst_eps_revision_log"] = np.clip(eps_revision, -0.5, 0.5)

    for name, values in raw.items():
        frame[name] = values
    score_columns = [
        "value_score",
        "actual_improvement_score",
        "quality_score",
        "revision_score",
        "price_confirmation_score",
        "quarter_gap_target_ratio",
        "peer_median_target_ratio",
        "peer_right_edge_target_ratio",
    ]
    for column in score_columns:
        frame[column] = np.nan
    frame["price_confirmed"] = False

    boundaries = np.r_[
        0,
        np.flatnonzero(
            frame["date_idx"].to_numpy()[1:] != frame["date_idx"].to_numpy()[:-1]
        )
        + 1,
        len(frame),
    ]
    for left, right in pairwise(boundaries):
        local = slice(int(left), int(right))
        industry = frame["industry_code"].to_numpy(dtype=np.int64)[local]
        signed_log_pe = frame["signed_log_pe"].to_numpy(dtype=float)[local]
        signed_log_pb = frame["signed_log_pb"].to_numpy(dtype=float)[local]
        pe = np.where(signed_log_pe > 0.0, np.expm1(signed_log_pe), np.nan)
        pb = np.where(signed_log_pb > 0.0, np.expm1(signed_log_pb), np.nan)
        log_mcap = frame["log_total_market_value"].to_numpy(dtype=float)[local]
        fcf = frame["cashflow_free_cash_flow"].to_numpy(dtype=float)[local]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            market_value = np.expm1(np.clip(log_mcap, 0.0, 50.0))
            fcf_yield = fcf / market_value
        pe_cheap = _industry_percentile(
            np.where(pe > 0.0, -pe, np.nan),
            industry,
            minimum_group_size=minimum_group_size,
        )
        pb_cheap = _industry_percentile(
            np.where(pb > 0.0, -pb, np.nan),
            industry,
            minimum_group_size=minimum_group_size,
        )
        fcf_score = _industry_percentile(
            fcf_yield, industry, minimum_group_size=minimum_group_size
        )
        value = _family_mean((pe_cheap, pb_cheap, fcf_score), minimum_count=2)
        value[~((pe > 0.0) & (pb > 0.0))] = np.nan

        actual_parts = tuple(
            _industry_percentile(
                frame[name].to_numpy(dtype=float)[local],
                industry,
                minimum_group_size=minimum_group_size,
            )
            for name in (
                "financial_net_profit_yoy",
                "financial_revenue_yoy",
                "performance_forecast_change_mid",
                "cashflow_cfo_to_income",
            )
        )
        actual = _family_mean(actual_parts, minimum_count=2)
        quality_parts = (
            _industry_percentile(
                frame["financial_roe_avg"].to_numpy(dtype=float)[local],
                industry,
                minimum_group_size=minimum_group_size,
            ),
            _industry_percentile(
                frame["financial_net_profit_margin"].to_numpy(dtype=float)[local],
                industry,
                minimum_group_size=minimum_group_size,
            ),
            _industry_percentile(
                -frame["financial_debt_to_asset"].to_numpy(dtype=float)[local],
                industry,
                minimum_group_size=minimum_group_size,
            ),
            _industry_percentile(
                frame["cashflow_cfo_to_income"].to_numpy(dtype=float)[local],
                industry,
                minimum_group_size=minimum_group_size,
            ),
        )
        quality = _family_mean(quality_parts, minimum_count=2)
        revision_raw = _family_mean(
            (
                frame["analyst_net_profit_revision_log"].to_numpy(dtype=float)[local],
                frame["analyst_eps_revision_log"].to_numpy(dtype=float)[local],
            ),
            minimum_count=1,
        )
        revision = _industry_percentile(
            revision_raw, industry, minimum_group_size=minimum_group_size
        )
        price_parts = (
            _industry_percentile(
                frame["industry_relative_ret20"].to_numpy(dtype=float)[local],
                industry,
                minimum_group_size=minimum_group_size,
            ),
            _industry_percentile(
                frame["trend_slope_20d"].to_numpy(dtype=float)[local],
                industry,
                minimum_group_size=minimum_group_size,
            ),
        )
        price_score = _family_mean(price_parts, minimum_count=2)
        price_confirmed = (
            frame["industry_relative_ret20"].to_numpy(dtype=float)[local] >= 0.0
        ) & (frame["trend_slope_20d"].to_numpy(dtype=float)[local] >= 0.0)
        quarter, median, edge = _peer_target_ratios(
            pe,
            pb,
            industry,
            minimum_group_size=minimum_group_size,
        )
        frame.loc[frame.index[local], "value_score"] = value
        frame.loc[frame.index[local], "actual_improvement_score"] = actual
        frame.loc[frame.index[local], "quality_score"] = quality
        frame.loc[frame.index[local], "revision_score"] = revision
        frame.loc[frame.index[local], "price_confirmation_score"] = price_score
        frame.loc[frame.index[local], "price_confirmed"] = price_confirmed
        frame.loc[frame.index[local], "quarter_gap_target_ratio"] = quarter
        frame.loc[frame.index[local], "peer_median_target_ratio"] = median
        frame.loc[frame.index[local], "peer_right_edge_target_ratio"] = edge
    if bool(frame["trade_date"].astype(str).str.startswith("2026-").any()):
        raise ValueError("forbidden_2026_feature_row")
    return frame


def _combine(*values: np.ndarray) -> np.ndarray:
    return _family_mean(values, minimum_count=len(values))


def _policy_score_and_eligibility(
    frame: pd.DataFrame, policy: str
) -> tuple[np.ndarray, np.ndarray]:
    value = frame["value_score"].to_numpy(dtype=float)
    actual = frame["actual_improvement_score"].to_numpy(dtype=float)
    revision = frame["revision_score"].to_numpy(dtype=float)
    quality = frame["quality_score"].to_numpy(dtype=float)
    covered = frame["analyst_covered"].to_numpy(dtype=bool)
    if policy == "value_only":
        score = value
        eligible = np.isfinite(value)
    elif policy == "value_analyst_covered_control":
        score = value
        eligible = np.isfinite(value) & covered
    elif policy == "value_actual":
        score = _combine(value, actual)
        eligible = np.isfinite(score)
    elif policy == "value_actual_revision":
        score = _combine(value, actual, revision)
        eligible = np.isfinite(score) & covered
    elif policy == "value_actual_revision_price_confirmed":
        score = _combine(value, actual, revision)
        eligible = (
            np.isfinite(score) & covered & frame["price_confirmed"].to_numpy(dtype=bool)
        )
    elif policy == "value_quality_actual_revision":
        score = _combine(value, quality, actual, revision)
        eligible = np.isfinite(score) & covered
    elif policy == "quality_only":
        score = quality
        eligible = np.isfinite(quality)
    else:
        raise ValueError(f"unknown_policy:{policy}")
    return score, eligible


def _select_industry_capped(
    *,
    score: np.ndarray,
    eligible: np.ndarray,
    candidate_id: np.ndarray,
    industry_code: np.ndarray,
    top_k: int,
    industry_cap: int,
) -> np.ndarray:
    positions = np.flatnonzero(np.asarray(eligible, dtype=bool) & np.isfinite(score))
    if not len(positions):
        return np.empty(0, dtype=np.int64)
    order = np.lexsort(
        (
            np.asarray(candidate_id, dtype=np.int64)[positions],
            -np.asarray(score, dtype=np.float64)[positions],
        )
    )
    selected: list[int] = []
    counts: dict[int, int] = {}
    industries = np.asarray(industry_code, dtype=np.int64)
    for position in positions[order]:
        code = int(industries[position])
        if counts.get(code, 0) >= int(industry_cap):
            continue
        selected.append(int(position))
        counts[code] = counts.get(code, 0) + 1
        if len(selected) == int(top_k):
            break
    return np.asarray(selected, dtype=np.int64)


def _selection_frame(
    features: pd.DataFrame,
    *,
    policies: Sequence[str],
    top_k: int,
    industry_cap: int,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    date_values = features["date_idx"].to_numpy(dtype=np.int64)
    boundaries = np.r_[
        0, np.flatnonzero(date_values[1:] != date_values[:-1]) + 1, len(features)
    ]
    for left, right in pairwise(boundaries):
        local = features.iloc[int(left) : int(right)]
        for policy in policies:
            score, eligible = _policy_score_and_eligibility(local, policy)
            chosen = _select_industry_capped(
                score=score,
                eligible=eligible,
                candidate_id=local["candidate_id"].to_numpy(dtype=np.int64),
                industry_code=local["industry_code"].to_numpy(dtype=np.int64),
                top_k=top_k,
                industry_cap=industry_cap,
            )
            if not len(chosen):
                continue
            selected = local.iloc[chosen][
                [
                    "row_position",
                    "candidate_id",
                    "date_idx",
                    "trade_date",
                    "symbol",
                    "evaluation_year",
                    "industry_code",
                    "analyst_covered",
                    "price_confirmed",
                    "value_score",
                    "actual_improvement_score",
                    "quality_score",
                    "revision_score",
                    "quarter_gap_target_ratio",
                    "peer_median_target_ratio",
                    "peer_right_edge_target_ratio",
                ]
            ].copy()
            selected.insert(0, "policy", policy)
            selected["policy_score"] = score[chosen]
            selected["selection_rank"] = np.arange(1, len(chosen) + 1)
            records.append(selected)
    if not records:
        raise ValueError("selection_frame_empty")
    return pd.concat(records, ignore_index=True)


def _d60_timeout(
    panel: base.StockPanel,
    rows: np.ndarray,
    *,
    retry_days: int,
) -> dict[str, np.ndarray]:
    pack = panel.pack_manifest
    dates = np.asarray(pack["date_values"], dtype=str)
    cutoff_positions = np.flatnonzero(dates == base.MAXIMUM_OUTCOME_DATE)
    if len(cutoff_positions) != 1:
        raise ValueError("maximum_outcome_date_missing")
    cutoff = int(cutoff_positions[0])
    date_idx = panel.date_idx[rows].astype(np.int64)
    symbol_idx = panel.symbol_idx[rows].astype(np.int64)
    raw = base._open_pack_array(pack, "feature_channels", "daily_raw", dtype=np.float32)
    price_observed = base._open_pack_array(
        pack, "masks", "price_observed", dtype=np.bool_
    )
    entry_buyable = base._open_pack_array(
        pack, "masks", "entry_buyable", dtype=np.bool_
    )
    exit_sellable = base._open_pack_array(
        pack, "masks", "exit_sellable", dtype=np.bool_
    )
    entry_idx = date_idx + 1
    within = date_idx + 60 + int(retry_days) <= cutoff
    safe_entry = np.minimum(entry_idx, cutoff)
    entry_price = np.asarray(raw[safe_entry, symbol_idx, 0], dtype=np.float64)
    entry_valid = (
        within
        & np.asarray(price_observed[safe_entry, symbol_idx], dtype=bool)
        & np.asarray(entry_buyable[safe_entry, symbol_idx], dtype=bool)
        & np.isfinite(entry_price)
        & (entry_price > 0.0)
    )
    exit_price = np.full(len(rows), np.nan, dtype=np.float64)
    exit_offset = np.full(len(rows), -1, dtype=np.int16)
    unresolved = entry_valid.copy()
    for offset in range(60, 60 + int(retry_days) + 1):
        path_idx = date_idx + int(offset)
        safe = np.minimum(path_idx, cutoff)
        close = np.asarray(raw[safe, symbol_idx, 3], dtype=np.float64)
        fill = (
            unresolved
            & np.asarray(price_observed[safe, symbol_idx], dtype=bool)
            & np.asarray(exit_sellable[safe, symbol_idx], dtype=bool)
            & np.isfinite(close)
            & (close > 0.0)
        )
        exit_price[fill] = close[fill]
        exit_offset[fill] = int(offset)
        unresolved[fill] = False
    valid = entry_valid & np.isfinite(exit_price)
    simple = np.full(len(rows), np.nan, dtype=np.float64)
    simple[valid] = exit_price[valid] / entry_price[valid] - 1.0
    return {
        "simple_return": simple,
        "valid": valid,
        "within": within,
        "entry_valid": entry_valid,
        "entry_price": entry_price,
        "exit_offset": exit_offset,
        "date_idx": date_idx,
        "symbol_idx": symbol_idx,
    }


def _target_exit(
    panel: base.StockPanel,
    rows: np.ndarray,
    timeout: Mapping[str, np.ndarray],
    target_ratio: np.ndarray,
) -> dict[str, np.ndarray]:
    pack = panel.pack_manifest
    dates = np.asarray(pack["date_values"], dtype=str)
    cutoff = int(np.flatnonzero(dates == base.MAXIMUM_OUTCOME_DATE)[0])
    raw = base._open_pack_array(pack, "feature_channels", "daily_raw", dtype=np.float32)
    price_observed = base._open_pack_array(
        pack, "masks", "price_observed", dtype=np.bool_
    )
    exit_sellable = base._open_pack_array(
        pack, "masks", "exit_sellable", dtype=np.bool_
    )
    date_idx = np.asarray(timeout["date_idx"], dtype=np.int64)
    symbol_idx = np.asarray(timeout["symbol_idx"], dtype=np.int64)
    entry_price = np.asarray(timeout["entry_price"], dtype=np.float64)
    ratio = np.asarray(target_ratio, dtype=np.float64)
    valid_target = np.asarray(timeout["entry_valid"], dtype=bool) & np.isfinite(ratio)
    valid_target &= ratio >= 1.0
    unresolved = valid_target.copy()
    hit = np.zeros(len(rows), dtype=bool)
    hit_offset = np.full(len(rows), -1, dtype=np.int16)
    result = np.asarray(timeout["simple_return"], dtype=np.float64).copy()
    target_price = entry_price * ratio
    for offset in range(2, 61):
        path_idx = date_idx + int(offset)
        safe = np.minimum(path_idx, cutoff)
        high = np.asarray(raw[safe, symbol_idx, 1], dtype=np.float64)
        fill = (
            unresolved
            & np.asarray(price_observed[safe, symbol_idx], dtype=bool)
            & np.asarray(exit_sellable[safe, symbol_idx], dtype=bool)
            & np.isfinite(high)
            & (high >= target_price)
        )
        result[fill] = ratio[fill] - 1.0
        hit[fill] = True
        hit_offset[fill] = int(offset)
        unresolved[fill] = False
    return {
        "simple_return": result,
        "valid": np.asarray(timeout["valid"], dtype=bool),
        "within": np.asarray(timeout["within"], dtype=bool),
        "hit": hit,
        "hit_offset": hit_offset,
    }


def _outcomes(
    panel: base.StockPanel, features: pd.DataFrame, *, retry_days: int
) -> dict[str, Any]:
    rows = features["row_position"].to_numpy(dtype=np.int64)
    d20_log = panel.target("executable_log_return_20")[rows]
    d20 = np.expm1(d20_log)
    d20_valid = panel.target_valid("executable_log_return_20")[rows]
    d20_within = (
        panel.flags[rows, base.HORIZONS.index(20) + 1] & base.FLAG_OUTCOME_WITHIN_CUTOFF
    ) != 0
    d20_residual = np.expm1(panel.target("residual_log_return_20")[rows])
    d20_residual_valid = panel.target_valid("residual_log_return_20")[rows]
    timeout = _d60_timeout(panel, rows, retry_days=retry_days)
    d60_residual, d60_market, d60_industry = base._leave_one_out_factors(
        np.asarray(timeout["simple_return"], dtype=float),
        np.asarray(timeout["valid"], dtype=bool),
        features["date_idx"].to_numpy(dtype=np.int64),
        features["industry_code"].to_numpy(dtype=np.int64),
    )
    targets = {
        "quarter_of_peer_median_log_gap": _target_exit(
            panel,
            rows,
            timeout,
            features["quarter_gap_target_ratio"].to_numpy(dtype=float),
        ),
        "peer_industry_median": _target_exit(
            panel,
            rows,
            timeout,
            features["peer_median_target_ratio"].to_numpy(dtype=float),
        ),
        "peer_industry_right_edge_q75": _target_exit(
            panel,
            rows,
            timeout,
            features["peer_right_edge_target_ratio"].to_numpy(dtype=float),
        ),
    }
    return {
        "d20_timeout": {
            "simple_return": d20,
            "valid": d20_valid,
            "within": d20_within,
            "residual": d20_residual,
            "residual_valid": d20_residual_valid,
        },
        "d60_timeout": {
            **timeout,
            "residual": d60_residual,
            "residual_valid": np.isfinite(d60_residual),
            "market": d60_market,
            "industry": d60_industry,
            "hit": np.zeros(len(features), dtype=bool),
        },
        **targets,
    }


def _cohort_frame(
    features: pd.DataFrame,
    selections: pd.DataFrame,
    outcomes: Mapping[str, Any],
    *,
    top_k: int,
    cost: float,
) -> pd.DataFrame:
    position_map = pd.Series(
        np.arange(len(features), dtype=np.int64),
        index=features["row_position"].to_numpy(dtype=np.int64),
    )
    selected = selections.copy()
    selected["local_position"] = selected["row_position"].map(position_map)
    if bool(selected["local_position"].isna().any()):
        raise ValueError("selection_outcome_alignment_failed")
    selected["local_position"] = selected["local_position"].astype(np.int64)
    date_values = features["date_idx"].to_numpy(dtype=np.int64)
    universe: dict[tuple[str, int], float] = {}
    for exit_name in ("d20_timeout", "d60_timeout"):
        outcome = outcomes[exit_name]
        values = np.asarray(outcome["simple_return"], dtype=float)
        valid = np.asarray(outcome["valid"], dtype=bool)
        for date_idx in np.unique(date_values):
            local = date_values == int(date_idx)
            good = local & valid & np.isfinite(values)
            universe[(exit_name, int(date_idx))] = (
                float(values[good].mean()) if good.any() else np.nan
            )
    records: list[dict[str, Any]] = []
    for (policy, date_idx), group in selected.groupby(
        ["policy", "date_idx"], sort=True
    ):
        local_positions = group["local_position"].to_numpy(dtype=np.int64)
        for exit_name in EXIT_POLICIES:
            outcome = outcomes[exit_name]
            values = np.asarray(outcome["simple_return"], dtype=float)
            valid = np.asarray(outcome["valid"], dtype=bool)
            within = np.asarray(outcome["within"], dtype=bool)
            selected_valid = valid[local_positions] & np.isfinite(
                values[local_positions]
            )
            selected_values = values[local_positions][selected_valid]
            selected_count = len(local_positions)
            observed_count = int(selected_valid.sum())
            stress_net = (
                float(selected_values.sum()) - float(cost) * observed_count
            ) / int(top_k)
            base_exit = "d20_timeout" if exit_name == "d20_timeout" else "d60_timeout"
            universe_mean = universe.get((base_exit, int(date_idx)), np.nan)
            selected_mean = (
                float(selected_values.mean()) if len(selected_values) else np.nan
            )
            if exit_name in {"d20_timeout", "d60_timeout"}:
                residual = np.asarray(outcome["residual"], dtype=float)
                residual_valid = np.asarray(outcome["residual_valid"], dtype=bool)
                good_residual = residual_valid[local_positions] & np.isfinite(
                    residual[local_positions]
                )
                residual_mean = (
                    float(residual[local_positions][good_residual].mean())
                    if good_residual.any()
                    else np.nan
                )
            else:
                residual_mean = np.nan
            hit = np.asarray(outcome.get("hit", np.zeros(len(features))), dtype=bool)
            hit_rate = (
                float(hit[local_positions].sum() / max(observed_count, 1))
                if exit_name not in {"d20_timeout", "d60_timeout"}
                else np.nan
            )
            records.append(
                {
                    "policy": str(policy),
                    "exit_policy": exit_name,
                    "horizon": 20 if exit_name == "d20_timeout" else 60,
                    "date_idx": int(date_idx),
                    "trade_date": str(group["trade_date"].iloc[0]),
                    "evaluation_year": int(group["evaluation_year"].iloc[0]),
                    "calendar_evaluable": bool(within[local_positions].all()),
                    "selected_count": selected_count,
                    "observed_count": observed_count,
                    "observed_fraction": observed_count / int(top_k),
                    "stress_net_return": stress_net,
                    "selected_mean_gross": selected_mean,
                    "universe_mean_gross": universe_mean,
                    "selected_excess_gross": (
                        selected_mean - universe_mean
                        if np.isfinite(selected_mean) and np.isfinite(universe_mean)
                        else np.nan
                    ),
                    "industry_residual_mean": residual_mean,
                    "target_hit_rate": hit_rate,
                }
            )
    result = pd.DataFrame(records)
    return result.loc[result["calendar_evaluable"]].reset_index(drop=True)


def _period_name(year: int, periods: Mapping[str, Sequence[int]]) -> str:
    for name, years in periods.items():
        if int(year) in {int(value) for value in years}:
            return str(name)
    raise ValueError(f"evaluation_year_not_in_period_contract:{year}")


def _inference(
    values: np.ndarray,
    *,
    lag: int,
    block_length: int,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    return {
        **base._hac_mean(x, lag=lag),
        "block": base._block_interval(
            x,
            block_length=block_length,
            repetitions=repetitions,
            seed=seed,
        ),
    }


def _summaries(cohorts: pd.DataFrame, study: Mapping[str, Any]) -> list[dict[str, Any]]:
    periods = dict(study["evaluation"]["periods"])
    inference = dict(study["evaluation"]["inference"])
    frame = cohorts.copy()
    frame["period"] = frame["evaluation_year"].map(
        lambda year: _period_name(int(year), periods)
    )
    results: list[dict[str, Any]] = []
    for (period, policy, exit_policy), group in frame.groupby(
        ["period", "policy", "exit_policy"], sort=True
    ):
        horizon = int(group["horizon"].iloc[0])
        prefix = "d20" if horizon == 20 else "d60"
        lag = int(inference[f"{prefix}_hac_lag_months"])
        block = int(inference[f"{prefix}_block_length_months"])
        seed = int(inference["seed"]) + len(results)
        stress = _inference(
            group["stress_net_return"].to_numpy(dtype=float),
            lag=lag,
            block_length=block,
            repetitions=int(inference["bootstrap_repetitions"]),
            seed=seed,
        )
        excess = _inference(
            group["selected_excess_gross"].to_numpy(dtype=float),
            lag=lag,
            block_length=block,
            repetitions=int(inference["bootstrap_repetitions"]),
            seed=seed + 1000,
        )
        residual = _inference(
            group["industry_residual_mean"].to_numpy(dtype=float),
            lag=lag,
            block_length=block,
            repetitions=int(inference["bootstrap_repetitions"]),
            seed=seed + 2000,
        )
        annual_frame = group.groupby("evaluation_year", sort=True, as_index=False).agg(
            stress_net_return=("stress_net_return", "mean"),
            selected_excess_gross=("selected_excess_gross", "mean"),
            industry_residual_mean=("industry_residual_mean", "mean"),
            mean_target_hit_rate=("target_hit_rate", "mean"),
            month_count=("date_idx", "size"),
        )
        annual = annual_frame.to_dict(orient="records")
        results.append(
            {
                "period": period,
                "policy": policy,
                "exit_policy": exit_policy,
                "horizon": horizon,
                "month_count": len(group),
                "mean_selected_count": float(group["selected_count"].mean()),
                "minimum_selected_count": int(group["selected_count"].min()),
                "observed_fraction": float(
                    group["observed_count"].sum()
                    / max(group["selected_count"].sum(), 1)
                ),
                "mean_target_hit_rate": float(group["target_hit_rate"].mean()),
                "stress_net": stress,
                "selected_excess_gross": excess,
                "industry_residual": residual,
                "positive_year_count": int(
                    (annual_frame["stress_net_return"] > 0.0).sum()
                ),
                "year_count": len(annual_frame),
                "annual": annual,
            }
        )
    return results


def _paired_exit_summaries(
    cohorts: pd.DataFrame, study: Mapping[str, Any]
) -> list[dict[str, Any]]:
    d60 = cohorts.loc[cohorts["horizon"].eq(60)].copy()
    baseline = d60.loc[d60["exit_policy"].eq("d60_timeout")][
        ["policy", "date_idx", "evaluation_year", "stress_net_return"]
    ].rename(columns={"stress_net_return": "baseline_return"})
    targets = d60.loc[~d60["exit_policy"].eq("d60_timeout")].merge(
        baseline,
        on=["policy", "date_idx", "evaluation_year"],
        how="inner",
        validate="many_to_one",
    )
    targets["paired_delta"] = targets["stress_net_return"] - targets["baseline_return"]
    periods = dict(study["evaluation"]["periods"])
    targets["period"] = targets["evaluation_year"].map(
        lambda year: _period_name(int(year), periods)
    )
    inference = dict(study["evaluation"]["inference"])
    results: list[dict[str, Any]] = []
    for (period, policy, exit_policy), group in targets.groupby(
        ["period", "policy", "exit_policy"], sort=True
    ):
        results.append(
            {
                "period": period,
                "policy": policy,
                "exit_policy": exit_policy,
                "paired_delta_vs_d60_timeout": _inference(
                    group["paired_delta"].to_numpy(dtype=float),
                    lag=int(inference["d60_hac_lag_months"]),
                    block_length=int(inference["d60_block_length_months"]),
                    repetitions=int(inference["bootstrap_repetitions"]),
                    seed=int(inference["seed"]) + len(results) + 4000,
                ),
            }
        )
    return results


def _coverage_summary(features: pd.DataFrame) -> list[dict[str, Any]]:
    frame = features.copy()
    frame["analyst_covered"] = frame["analyst_covered"].astype(bool)
    return (
        frame.groupby("evaluation_year", sort=True, as_index=False)
        .agg(
            candidate_count=("candidate_id", "size"),
            analyst_covered_count=("analyst_covered", "sum"),
            analyst_covered_fraction=("analyst_covered", "mean"),
            month_count=("date_idx", "nunique"),
        )
        .to_dict(orient="records")
    )


def _decision_summary(
    summaries: Sequence[Mapping[str, Any]], study: Mapping[str, Any]
) -> dict[str, Any]:
    primary = str(study["signals"]["primary_policy"])
    rows = [
        row
        for row in summaries
        if row["policy"] == primary
        and row["exit_policy"] in {"d20_timeout", "d60_timeout"}
    ]
    checks: list[dict[str, Any]] = []
    for row in rows:
        stress = row["stress_net"]
        residual = row["industry_residual"]
        checks.append(
            {
                "period": row["period"],
                "exit_policy": row["exit_policy"],
                "stress_net_hac_lcb_positive": float(stress["lcb_95"]) > 0.0,
                "stress_net_block_lcb_positive": float(stress["block"]["lcb_95"]) > 0.0,
                "industry_residual_hac_lcb_positive": float(residual["lcb_95"]) > 0.0,
                "majority_years_positive": int(row["positive_year_count"])
                >= math.ceil(int(row["year_count"]) / 2),
            }
        )
    return {
        "primary_policy": primary,
        "all_primary_period_horizon_checks_passed": bool(
            checks
            and all(
                all(value for key, value in row.items() if key.endswith("positive"))
                for row in checks
            )
        ),
        "checks": checks,
        "account_replay_authorized": False,
        "reason": "all historical periods are retrospectively consumed; forward-only confirmation is required even if historical checks pass",
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
    summary_path = root / "summary.json"
    fingerprint = _payload_hash(
        {
            "study_sha256": base._sha256_file(frozen_path),
            "source_hashes": {
                key: base._sha256_file(path) for key, path in sources.items()
            },
        }
    )
    if summary_path.is_file() and not force:
        current = _read_json(summary_path)
        if current.get("experiment_fingerprint") != fingerprint:
            raise ValueError("existing_output_fingerprint_mismatch")
        if all(
            base._record_valid(record, verify_hash=True)
            for record in dict(current.get("files", {})).values()
        ):
            return current
    panel = base.load_panel(
        input_manifest_path=sources["input_manifest"],
        label_manifest_path=sources["label_manifest"],
    )
    if bool((panel.years == base.FORBIDDEN_YEAR).any()):
        raise ValueError("forbidden_2026_panel_row")
    rows = _month_end_rows(panel)
    identity = panel.row_index.iloc[rows][
        ["candidate_id", "symbol", "trade_date"]
    ].copy()
    identity["signal_date"] = pd.to_datetime(identity["trade_date"])
    signal_contract = dict(study["signals"])
    consensus = _analyst_consensus(
        identity,
        report_path=sources["research_report"],
        forecast_path=sources["research_report_forecast"],
        lookback_days=int(signal_contract["analyst_consensus_lookback_calendar_days"]),
        revision_gap_days=int(signal_contract["analyst_revision_gap_calendar_days"]),
    )
    features = _feature_frame(
        panel,
        rows,
        consensus,
        minimum_group_size=int(signal_contract["minimum_industry_rank_group_size"]),
        minimum_current_institutions=int(
            signal_contract["minimum_current_institutions"]
        ),
        minimum_prior_institutions=int(signal_contract["minimum_prior_institutions"]),
    )
    selections = _selection_frame(
        features,
        policies=EXPECTED_POLICIES,
        top_k=int(signal_contract["top_k"]),
        industry_cap=int(signal_contract["maximum_names_per_pit_industry"]),
    )
    features_path = root / "candidate_features.parquet"
    selections_path = root / "selections.parquet"
    _write_parquet(features_path, features)
    _write_parquet(selections_path, selections)

    execution = dict(study["execution"])
    outcomes = _outcomes(
        panel,
        features,
        retry_days=int(execution["sell_retry_open_days"]),
    )
    cohorts = _cohort_frame(
        features,
        selections,
        outcomes,
        top_k=int(signal_contract["top_k"]),
        cost=float(execution["cost_proxy_round_trip"]),
    )
    cohorts_path = root / "cohort_returns.parquet"
    _write_parquet(cohorts_path, cohorts)
    summaries = _summaries(cohorts, study)
    paired = _paired_exit_summaries(cohorts, study)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "study_id": STUDY_ID,
        "experiment_fingerprint": fingerprint,
        "study_sha256": base._sha256_file(frozen_path),
        "row_count": len(features),
        "month_count": int(features["date_idx"].nunique()),
        "start_date": str(features["trade_date"].min()),
        "end_date": str(features["trade_date"].max()),
        "analyst_coverage": _coverage_summary(features),
        "policy_summaries": summaries,
        "paired_exit_summaries": paired,
        "decision": _decision_summary(summaries, study),
        "candidate_selection_precedes_outcome_reads": True,
        "52_week_low_reward_used": False,
        "forbidden_2026_read_count": 0,
        "account_replay_performed": False,
        "portfolio_optimization_performed": False,
        "profit_claim_allowed": False,
        "files": {
            "candidate_features": base._file_record(features_path),
            "selections": base._file_record(selections_path),
            "cohort_returns": base._file_record(cohorts_path),
        },
    }
    base._write_json(summary_path, summary)
    return summary


def validate_summary(path: str | Path) -> dict[str, Any]:
    summary = _read_json(base._resolve_path(path))
    checks = {
        "schema": summary.get("schema") == SUMMARY_SCHEMA,
        "study": summary.get("study_id") == STUDY_ID,
        "selection_before_outcomes": summary.get(
            "candidate_selection_precedes_outcome_reads"
        )
        is True,
        "no_low_price_reward": summary.get("52_week_low_reward_used") is False,
        "no_2026": int(summary.get("forbidden_2026_read_count", -1)) == 0,
        "no_account": summary.get("account_replay_performed") is False,
        "files": all(
            base._record_valid(record, verify_hash=True)
            for record in dict(summary.get("files", {})).values()
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"summary_validation_failed:{checks}")
    return {"status": "ok", "checks": checks}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_study(
        study_path=args.study,
        output_root=args.output_root,
        force=args.force,
    )
    print(json.dumps(summary["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
