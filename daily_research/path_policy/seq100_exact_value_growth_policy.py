"""Exact causal main-board value-growth policy information test.

The study translates the quantifiable part of the discretionary
Quality x Value x Earnings Revision x Confirmation workflow into a frozen
30/30/25/10/5 score.  It deliberately keeps qualitative moat and management
judgment out of historical data and reads outcomes only after selections are
materialized.
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

import duckdb
import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_causal_value_policy as causal
from daily_research.path_policy import seq100_stock_distribution as base

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_exact_value_growth_policy_v1"
SUMMARY_SCHEMA = "seq100_exact_value_growth_policy_summary/1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_exact_value_growth_policy_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_exact_value_growth_policy_v1"
)
POLICIES = (
    "exact_full_top10",
    "exact_full_top20",
    "exact_no_cycle_top10",
    "exact_no_governance_top10",
    "forward_value_only_top10",
    "ttm_fcf_only_top10",
)
EXTRA_FEATURES = (
    "listing_age_open_days",
    "balance_trade_receivables_to_total_assets",
    "balance_inventory_ratio",
    "balance_goodwill_ratio",
    "balance_borrowing_ratio",
    "balance_parent_equity",
    "cashflow_fcf_to_revenue",
    "announcement_keyword_penalty_20d",
    "announcement_keyword_litigation_20d",
    "announcement_keyword_delisting_20d",
    "income_source_conflict",
    "balance_source_conflict",
    "cashflow_source_conflict",
)
MAIN_BOARD_PREFIXES = ("600", "601", "603", "605", "000", "001", "002", "003")


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
    if tuple(signals.get("policies", ())) != POLICIES:
        raise ValueError("policy_contract_mismatch")
    weights = dict(study["score_contract"]["family_weights"])
    expected = {
        "valuation": 0.30,
        "earnings_and_revision": 0.30,
        "quality": 0.25,
        "price_confirmation": 0.10,
        "governance": 0.05,
    }
    if weights != expected or not math.isclose(sum(weights.values()), 1.0):
        raise ValueError("score_weight_contract_mismatch")
    serialized = json.dumps(study["score_contract"], ensure_ascii=False).lower()
    excluded = " ".join(study["score_contract"]["explicitly_excluded"]).lower()
    if "52_week_low" not in excluded or "distance_from_52_week_low" not in excluded:
        raise ValueError("forbidden_low_price_reward_not_explicit")
    prefix = serialized.split('"explicitly_excluded"', 1)[0]
    if "52_week_low" in prefix or "distance_from_52_week_low" in prefix:
        raise ValueError("forbidden_low_price_reward_in_score")
    return study, study_path


def _source_contract(study: Mapping[str, Any]) -> dict[str, Path]:
    source = dict(study["source"])
    keys = (
        "input_manifest",
        "label_manifest",
        "candidate_features",
        "ttm_candidate_features",
        "income_statement_quarterly",
        "balance_sheet_quarterly",
        "research_report",
        "research_report_forecast",
    )
    paths: dict[str, Path] = {}
    for key in keys:
        path = base._resolve_path(source[key])
        expected = str(source[f"{key}_sha256"]).lower()
        if not path.is_file() or base._sha256_file(path).lower() != expected:
            raise ValueError(f"source_invalid:{key}")
        paths[key] = path
    return paths


def _extra_compact_frame(
    panel: base.StockPanel, features: pd.DataFrame
) -> pd.DataFrame:
    rows = features["row_position"].to_numpy(dtype=np.int64)
    positions = base._feature_positions(panel, EXTRA_FEATURES)
    values = base._feature_matrix(panel, rows, positions).astype(np.float64)
    result = pd.DataFrame({"candidate_id": features["candidate_id"].to_numpy()})
    for index, name in enumerate(EXTRA_FEATURES):
        result[name] = values[:, index]
    return result


def _analyst_consensus_multi(
    signals: pd.DataFrame,
    *,
    report_path: Path,
    forecast_path: Path,
    lookback_days: int,
    revision_gaps: Sequence[int],
) -> pd.DataFrame:
    grid = signals[["candidate_id", "symbol", "trade_date"]].copy()
    grid["signal_date"] = pd.to_datetime(grid["trade_date"])
    grid["target_year"] = grid["signal_date"].dt.year + 1
    stages = [
        ("current", 0),
        *[(f"prior_{int(gap)}", int(gap)) for gap in revision_gaps],
    ]
    stage_sql = "\nUNION ALL\n".join(
        f"SELECT candidate_id, symbol, target_year, '{name}' AS stage, "
        f"signal_date - INTERVAL {gap} DAY AS asof_date FROM signals"
        for name, gap in stages
    )
    columns: list[str] = ["candidate_id"]
    expressions: list[str] = []
    for name, _ in stages:
        for metric in ("eps", "net_profit", "pe", "ev_ebitda"):
            prefix = (
                "np"
                if metric == "net_profit"
                else ("ev" if metric == "ev_ebitda" else metric)
            )
            expressions.extend(
                (
                    f"max({metric}_institution_count) FILTER (WHERE stage = '{name}') AS {name}_{prefix}_n",
                    f"max({metric}_median) FILTER (WHERE stage = '{name}') AS {name}_{prefix}",
                )
            )
            columns.extend((f"{name}_{prefix}_n", f"{name}_{prefix}"))
    forecast_sql = forecast_path.as_posix().replace("'", "''")
    report_sql = report_path.as_posix().replace("'", "''")
    query = f"""
    WITH asofs AS (
      {stage_sql}
    ), latest AS (
      SELECT a.candidate_id, a.stage, f.eps, f.net_profit, f.pe, f.ev_ebitda,
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
             count(*) FILTER (WHERE recency_rank = 1 AND eps IS NOT NULL) AS eps_institution_count,
             count(*) FILTER (WHERE recency_rank = 1 AND net_profit IS NOT NULL) AS net_profit_institution_count,
             count(*) FILTER (WHERE recency_rank = 1 AND pe > 0) AS pe_institution_count,
             count(*) FILTER (WHERE recency_rank = 1 AND ev_ebitda > 0) AS ev_ebitda_institution_count,
             median(eps) FILTER (WHERE recency_rank = 1 AND eps IS NOT NULL) AS eps_median,
             median(net_profit) FILTER (WHERE recency_rank = 1 AND net_profit IS NOT NULL) AS net_profit_median,
             median(pe) FILTER (WHERE recency_rank = 1 AND pe > 0) AS pe_median,
             median(ev_ebitda) FILTER (WHERE recency_rank = 1 AND ev_ebitda > 0) AS ev_ebitda_median
      FROM latest
      GROUP BY candidate_id, stage
    )
    SELECT {", ".join([columns[0], *expressions])}
    FROM consensus
    GROUP BY candidate_id
    """
    con = duckdb.connect()
    try:
        con.register("signals", grid)
        result = con.execute(query).fetchdf()
    finally:
        con.close()
    if bool(result["candidate_id"].duplicated().any()):
        raise ValueError("analyst_consensus_candidate_duplicate")
    missing = set(columns) - set(result.columns)
    if missing:
        raise ValueError(f"analyst_consensus_columns_missing:{sorted(missing)}")
    return result


def _cycle_normalization(
    signals: pd.DataFrame,
    *,
    income_path: Path,
    balance_path: Path,
    history_years: int,
) -> pd.DataFrame:
    grid = signals[["candidate_id", "symbol", "trade_date"]].copy()
    grid["signal_date"] = pd.to_datetime(grid["trade_date"])
    income_sql = income_path.as_posix().replace("'", "''")
    balance_sql = balance_path.as_posix().replace("'", "''")
    query = f"""
    WITH income_known AS (
      SELECT s.candidate_id, i.fiscal_year, i.fiscal_quarter,
             coalesce(i.total_revenue, i.revenue) AS revenue,
             i.parent_net_income, i.company_type,
             try_cast(i.feature_available_date AS DATE) AS available_date,
             row_number() OVER (
               PARTITION BY s.candidate_id, i.fiscal_year, i.fiscal_quarter
               ORDER BY try_cast(i.feature_available_date AS DATE) DESC,
                        (i.report_type = '1') DESC,
                        i.report_date DESC, i.update_flag DESC
             ) AS version_rank
      FROM signals s
      JOIN read_parquet('{income_sql}') i ON i.symbol = s.symbol
      WHERE try_cast(i.feature_available_date AS DATE) <= s.signal_date
        AND try_cast(i.feature_available_date AS DATE) < DATE '2026-01-01'
    ), income_periods AS (
      SELECT * FROM income_known WHERE version_rank = 1
    ), current_income AS (
      SELECT *, row_number() OVER (
        PARTITION BY candidate_id
        ORDER BY fiscal_year DESC, fiscal_quarter DESC, available_date DESC
      ) AS period_rank
      FROM income_periods
    ), current_only AS (
      SELECT * FROM current_income WHERE period_rank = 1
    ), income_annual_ranked AS (
      SELECT *, row_number() OVER (
        PARTITION BY candidate_id ORDER BY fiscal_year DESC
      ) AS annual_rank
      FROM income_periods WHERE fiscal_quarter = 4
    ), balance_known AS (
      SELECT s.candidate_id, b.fiscal_year, b.fiscal_quarter, b.parent_equity,
             row_number() OVER (
               PARTITION BY s.candidate_id, b.fiscal_year, b.fiscal_quarter
               ORDER BY try_cast(b.feature_available_date AS DATE) DESC,
                        (b.report_type = '1') DESC,
                        b.report_date DESC, b.update_flag DESC
             ) AS version_rank
      FROM signals s
      JOIN read_parquet('{balance_sql}') b ON b.symbol = s.symbol
      WHERE try_cast(b.feature_available_date AS DATE) <= s.signal_date
        AND try_cast(b.feature_available_date AS DATE) < DATE '2026-01-01'
    ), balance_periods AS (
      SELECT * FROM balance_known WHERE version_rank = 1
    ), annual_values AS (
      SELECT ia.candidate_id, ia.annual_rank,
             ia.parent_net_income / nullif(ia.revenue, 0) AS parent_margin,
             ia.parent_net_income / nullif(ba.parent_equity, 0) AS parent_roe
      FROM income_annual_ranked ia
      LEFT JOIN balance_periods ba
        ON ba.candidate_id = ia.candidate_id
       AND ba.fiscal_year = ia.fiscal_year
       AND ba.fiscal_quarter = 4
      WHERE ia.annual_rank <= {int(history_years)}
    ), annual_stats AS (
      SELECT candidate_id,
             count(parent_margin) AS margin_history_count,
             count(parent_roe) AS roe_history_count,
             quantile_cont(parent_margin, 0.25) AS margin_bear,
             median(parent_margin) AS margin_base,
             quantile_cont(parent_margin, 0.75) AS margin_bull,
             quantile_cont(parent_roe, 0.25) AS roe_bear,
             median(parent_roe) AS roe_base,
             quantile_cont(parent_roe, 0.75) AS roe_bull
      FROM annual_values GROUP BY candidate_id
    )
    SELECT c.candidate_id, c.fiscal_year, c.fiscal_quarter, c.company_type,
           c.available_date AS current_income_available_date,
           CASE WHEN c.fiscal_quarter = 4 THEN c.revenue
                ELSE c.revenue + py.revenue - pq.revenue END AS ttm_revenue,
           CASE WHEN c.fiscal_quarter = 4 THEN c.parent_net_income
                ELSE c.parent_net_income + py.parent_net_income - pq.parent_net_income END AS ttm_parent_net_income,
           a.margin_history_count, a.roe_history_count,
           a.margin_bear, a.margin_base, a.margin_bull,
           a.roe_bear, a.roe_base, a.roe_bull
    FROM current_only c
    LEFT JOIN income_periods py
      ON py.candidate_id = c.candidate_id
     AND py.fiscal_year = c.fiscal_year - 1 AND py.fiscal_quarter = 4
    LEFT JOIN income_periods pq
      ON pq.candidate_id = c.candidate_id
     AND pq.fiscal_year = c.fiscal_year - 1 AND pq.fiscal_quarter = c.fiscal_quarter
    LEFT JOIN annual_stats a ON a.candidate_id = c.candidate_id
    """
    con = duckdb.connect()
    try:
        con.register("signals", grid)
        result = con.execute(query).fetchdf()
    finally:
        con.close()
    if bool(result["candidate_id"].duplicated().any()):
        raise ValueError("cycle_normalization_candidate_duplicate")
    return result


def _covered_revision(
    current: np.ndarray,
    prior: np.ndarray,
    current_count: np.ndarray,
    prior_count: np.ndarray,
    *,
    minimum_institutions: int,
) -> np.ndarray:
    current_value = np.asarray(current, dtype=np.float64)
    prior_value = np.asarray(prior, dtype=np.float64)
    covered = (
        (np.asarray(current_count, dtype=np.float64) >= int(minimum_institutions))
        & (np.asarray(prior_count, dtype=np.float64) >= int(minimum_institutions))
        & np.isfinite(current_value)
        & np.isfinite(prior_value)
        & (current_value > 0.0)
        & (prior_value > 0.0)
    )
    result = np.full(len(current_value), np.nan, dtype=np.float64)
    result[covered] = np.clip(
        np.log(current_value[covered] / prior_value[covered]), -0.5, 0.5
    )
    return result


def _assemble_features(
    panel: base.StockPanel,
    sources: Mapping[str, Path],
    study: Mapping[str, Any],
) -> pd.DataFrame:
    features = pd.read_parquet(sources["candidate_features"])
    ttm = pd.read_parquet(sources["ttm_candidate_features"])[
        ["candidate_id", "ttm_fcf"]
    ]
    if len(features) != 207_444 or int(features["date_idx"].nunique()) != 168:
        raise ValueError("candidate_feature_scope_changed")
    if bool(features["trade_date"].astype(str).str.startswith("2026-").any()):
        raise ValueError("forbidden_2026_candidate_feature")
    codes = features["symbol"].astype(str).str[:3]
    if not bool(codes.isin(MAIN_BOARD_PREFIXES).all()):
        raise ValueError("non_main_board_candidate")
    features = features.merge(ttm, on="candidate_id", how="left", validate="one_to_one")
    features = features.merge(
        _extra_compact_frame(panel, features),
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    signal_contract = dict(study["signals"])
    analyst = _analyst_consensus_multi(
        features,
        report_path=sources["research_report"],
        forecast_path=sources["research_report_forecast"],
        lookback_days=int(signal_contract["analyst_consensus_lookback_calendar_days"]),
        revision_gaps=signal_contract["analyst_revision_gap_calendar_days"],
    )
    overlapping_analyst_columns = set(features.columns) & set(analyst.columns) - {
        "candidate_id"
    }
    if overlapping_analyst_columns:
        features = features.drop(columns=sorted(overlapping_analyst_columns))
    features = features.merge(
        analyst, on="candidate_id", how="left", validate="one_to_one"
    )
    cycle = _cycle_normalization(
        features,
        income_path=sources["income_statement_quarterly"],
        balance_path=sources["balance_sheet_quarterly"],
        history_years=int(
            study["score_contract"]["cycle_normalization"]["history_years"]
        ),
    )
    features = features.merge(
        cycle, on="candidate_id", how="left", validate="one_to_one"
    )
    current_equity = features["balance_parent_equity"].to_numpy(dtype=float)
    general = features["company_type"].astype(str).eq("1").to_numpy()
    ttm_revenue = features["ttm_revenue"].to_numpy(dtype=float)
    normalized: dict[str, np.ndarray] = {}
    for case in ("bear", "base", "bull"):
        margin = features[f"margin_{case}"].to_numpy(dtype=float)
        roe = features[f"roe_{case}"].to_numpy(dtype=float)
        value = np.where(general, ttm_revenue * margin, current_equity * roe)
        value[~np.isfinite(value)] = np.nan
        normalized[case] = value
        features[f"normalized_parent_net_income_{case}"] = value
    features["cycle_history_count"] = np.where(
        general,
        features["margin_history_count"].to_numpy(dtype=float),
        features["roe_history_count"].to_numpy(dtype=float),
    )
    minimum = int(signal_contract["minimum_institutions"])
    for gap in signal_contract["analyst_revision_gap_calendar_days"]:
        for metric in ("np", "eps"):
            features[f"revision_{metric}_{int(gap)}"] = _covered_revision(
                features[f"current_{metric}"].to_numpy(dtype=float),
                features[f"prior_{int(gap)}_{metric}"].to_numpy(dtype=float),
                features[f"current_{metric}_n"].fillna(0).to_numpy(dtype=float),
                features[f"prior_{int(gap)}_{metric}_n"]
                .fillna(0)
                .to_numpy(dtype=float),
                minimum_institutions=minimum,
            )
    return features.sort_values(
        ["date_idx", "candidate_id"], kind="stable"
    ).reset_index(drop=True)


def _score_features(features: pd.DataFrame, study: Mapping[str, Any]) -> pd.DataFrame:
    frame = features.copy()
    n = len(frame)
    columns = (
        "forward_earnings_yield",
        "ttm_fcf_yield",
        "normalized_earnings_yield_bear",
        "normalized_earnings_yield_base",
        "normalized_earnings_yield_bull",
        "valuation_score",
        "valuation_no_cycle_score",
        "actual_score",
        "revision_score_exact",
        "earnings_revision_score",
        "quality_score_exact",
        "price_confirmation_score_exact",
        "governance_score",
        "exact_full_score",
        "exact_no_cycle_score",
        "exact_no_governance_score",
        "forward_value_score",
        "ttm_fcf_score",
    )
    arrays = {name: np.full(n, np.nan, dtype=np.float64) for name in columns}
    eligibility = {
        policy: np.zeros(n, dtype=bool)
        for policy in POLICIES
        if policy != "exact_full_top20"
    }
    eligibility["exact_full_top20"] = eligibility["exact_full_top10"]
    minimum_group = int(study["signals"]["minimum_industry_rank_group_size"])
    minimum_institutions = int(study["signals"]["minimum_institutions"])
    minimum_listing = int(study["universe"]["minimum_listing_open_days"])
    date_values = frame["date_idx"].to_numpy(dtype=np.int64)
    boundaries = np.r_[0, np.flatnonzero(date_values[1:] != date_values[:-1]) + 1, n]
    for left, right in pairwise(boundaries):
        sl = slice(int(left), int(right))
        local = frame.iloc[sl]
        industry = local["industry_code"].to_numpy(dtype=np.int64)
        signed_pe = local["signed_log_pe"].to_numpy(dtype=float)
        signed_pb = local["signed_log_pb"].to_numpy(dtype=float)
        log_mcap = local["log_total_market_value"].to_numpy(dtype=float)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            market_value = np.expm1(np.clip(log_mcap, 0.0, 50.0))
            forward_yield = (
                local["current_np"].to_numpy(dtype=float) * 10_000.0 / market_value
            )
            ttm_fcf_yield = local["ttm_fcf"].to_numpy(dtype=float) / market_value
        normalized_yields: dict[str, np.ndarray] = {}
        for case in ("bear", "base", "bull"):
            normalized_yields[case] = (
                local[f"normalized_parent_net_income_{case}"].to_numpy(dtype=float)
                / market_value
            )
            arrays[f"normalized_earnings_yield_{case}"][sl] = normalized_yields[case]
        arrays["forward_earnings_yield"][sl] = forward_yield
        arrays["ttm_fcf_yield"][sl] = ttm_fcf_yield
        valuation_parts = (
            causal._industry_percentile(
                forward_yield, industry, minimum_group_size=minimum_group
            ),
            causal._industry_percentile(
                np.where(signed_pe > 0.0, -signed_pe, np.nan),
                industry,
                minimum_group_size=minimum_group,
            ),
            causal._industry_percentile(
                np.where(signed_pb > 0.0, -signed_pb, np.nan),
                industry,
                minimum_group_size=minimum_group,
            ),
            causal._industry_percentile(
                ttm_fcf_yield, industry, minimum_group_size=minimum_group
            ),
            causal._industry_percentile(
                normalized_yields["base"],
                industry,
                minimum_group_size=minimum_group,
            ),
        )
        valuation = causal._family_mean(valuation_parts, minimum_count=4)
        valuation_no_cycle = causal._family_mean(valuation_parts[:4], minimum_count=3)
        actual_parts = tuple(
            causal._industry_percentile(
                local[name].to_numpy(dtype=float),
                industry,
                minimum_group_size=minimum_group,
            )
            for name in (
                "financial_net_profit_yoy",
                "financial_revenue_yoy",
                "performance_forecast_change_mid",
                "cashflow_cfo_to_income",
            )
        )
        actual = causal._family_mean(actual_parts, minimum_count=2)
        revision_parts = tuple(
            causal._industry_percentile(
                local[f"revision_{metric}_{gap}"].to_numpy(dtype=float),
                industry,
                minimum_group_size=minimum_group,
            )
            for gap in study["signals"]["analyst_revision_gap_calendar_days"]
            for metric in ("np", "eps")
        )
        revision = causal._family_mean(revision_parts, minimum_count=2)
        earnings = causal._family_mean((actual, revision), minimum_count=2)
        quality_parts = (
            causal._industry_percentile(
                local["financial_roe_avg"], industry, minimum_group_size=minimum_group
            ),
            causal._industry_percentile(
                local["financial_net_profit_margin"],
                industry,
                minimum_group_size=minimum_group,
            ),
            causal._industry_percentile(
                local["cashflow_cfo_to_income"],
                industry,
                minimum_group_size=minimum_group,
            ),
            causal._industry_percentile(
                local["cashflow_fcf_to_revenue"],
                industry,
                minimum_group_size=minimum_group,
            ),
            causal._industry_percentile(
                -local["financial_debt_to_asset"],
                industry,
                minimum_group_size=minimum_group,
            ),
            causal._industry_percentile(
                -local["balance_trade_receivables_to_total_assets"],
                industry,
                minimum_group_size=minimum_group,
            ),
            causal._industry_percentile(
                -local["balance_inventory_ratio"],
                industry,
                minimum_group_size=minimum_group,
            ),
            causal._industry_percentile(
                -local["balance_goodwill_ratio"],
                industry,
                minimum_group_size=minimum_group,
            ),
            causal._industry_percentile(
                -local["balance_borrowing_ratio"],
                industry,
                minimum_group_size=minimum_group,
            ),
        )
        quality = causal._family_mean(quality_parts, minimum_count=5)
        price_parts = (
            causal._industry_percentile(
                local["industry_relative_ret20"],
                industry,
                minimum_group_size=minimum_group,
            ),
            causal._industry_percentile(
                local["trend_slope_20d"], industry, minimum_group_size=minimum_group
            ),
        )
        price = causal._family_mean(price_parts, minimum_count=2)
        clean_parts = np.column_stack(
            [
                local["announcement_keyword_penalty_20d"].to_numpy(dtype=float) <= 0.0,
                local["announcement_keyword_litigation_20d"].to_numpy(dtype=float)
                <= 0.0,
                local["announcement_keyword_delisting_20d"].to_numpy(dtype=float)
                <= 0.0,
                local["income_source_conflict"].to_numpy(dtype=float) <= 0.0,
                local["balance_source_conflict"].to_numpy(dtype=float) <= 0.0,
                local["cashflow_source_conflict"].to_numpy(dtype=float) <= 0.0,
            ]
        )
        governance = clean_parts.mean(axis=1)
        full = (
            0.30 * valuation
            + 0.30 * earnings
            + 0.25 * quality
            + 0.10 * price
            + 0.05 * governance
        )
        no_cycle = (
            0.30 * valuation_no_cycle
            + 0.30 * earnings
            + 0.25 * quality
            + 0.10 * price
            + 0.05 * governance
        )
        no_governance = (
            0.30 * valuation + 0.30 * earnings + 0.25 * quality + 0.10 * price
        ) / 0.95
        forward_score = valuation_parts[0]
        ttm_score = valuation_parts[3]
        for name, value in (
            ("valuation_score", valuation),
            ("valuation_no_cycle_score", valuation_no_cycle),
            ("actual_score", actual),
            ("revision_score_exact", revision),
            ("earnings_revision_score", earnings),
            ("quality_score_exact", quality),
            ("price_confirmation_score_exact", price),
            ("governance_score", governance),
            ("exact_full_score", full),
            ("exact_no_cycle_score", no_cycle),
            ("exact_no_governance_score", no_governance),
            ("forward_value_score", forward_score),
            ("ttm_fcf_score", ttm_score),
        ):
            arrays[name][sl] = value
        analyst_current = (
            local["current_np_n"].fillna(0).to_numpy(dtype=float)
            >= minimum_institutions
        ) & (local["current_np"].to_numpy(dtype=float) > 0.0)
        revision_covered = np.ones(len(local), dtype=bool)
        for gap in study["signals"]["analyst_revision_gap_calendar_days"]:
            revision_covered &= (
                local[f"prior_{gap}_np_n"].fillna(0).to_numpy(dtype=float)
                >= minimum_institutions
            ) & (
                local[f"prior_{gap}_eps_n"].fillna(0).to_numpy(dtype=float)
                >= minimum_institutions
            )
        revision_covered &= (
            local["current_eps_n"].fillna(0).to_numpy(dtype=float)
            >= minimum_institutions
        )
        base_conditions = (
            (signed_pe > 0.0)
            & (signed_pb > 0.0)
            & analyst_current
            & revision_covered
            & (local["industry_relative_ret20"].to_numpy(dtype=float) >= 0.0)
            & (local["trend_slope_20d"].to_numpy(dtype=float) >= 0.0)
            & (local["listing_age_open_days"].to_numpy(dtype=float) >= minimum_listing)
        )
        governance_ok = (
            local["announcement_keyword_delisting_20d"].to_numpy(dtype=float) <= 0.0
        )
        exact = base_conditions & governance_ok & np.isfinite(full)
        eligibility["exact_full_top10"][sl] = exact
        eligibility["exact_no_cycle_top10"][sl] = (
            base_conditions & governance_ok & np.isfinite(no_cycle)
        )
        eligibility["exact_no_governance_top10"][sl] = base_conditions & np.isfinite(
            no_governance
        )
        eligibility["forward_value_only_top10"][sl] = (
            (signed_pe > 0.0)
            & (signed_pb > 0.0)
            & analyst_current
            & governance_ok
            & (local["listing_age_open_days"].to_numpy(dtype=float) >= minimum_listing)
            & np.isfinite(forward_score)
        )
        eligibility["ttm_fcf_only_top10"][sl] = (
            (signed_pe > 0.0)
            & (signed_pb > 0.0)
            & governance_ok
            & (local["listing_age_open_days"].to_numpy(dtype=float) >= minimum_listing)
            & np.isfinite(ttm_score)
        )
    for name, values in arrays.items():
        frame[name] = values
    for policy in POLICIES:
        source = "exact_full_top10" if policy == "exact_full_top20" else policy
        frame[f"eligible__{policy}"] = eligibility[source]
    return frame


def _selection_frame(features: pd.DataFrame, study: Mapping[str, Any]) -> pd.DataFrame:
    score_columns = {
        "exact_full_top10": "exact_full_score",
        "exact_full_top20": "exact_full_score",
        "exact_no_cycle_top10": "exact_no_cycle_score",
        "exact_no_governance_top10": "exact_no_governance_score",
        "forward_value_only_top10": "forward_value_score",
        "ttm_fcf_only_top10": "ttm_fcf_score",
    }
    breadth = dict(study["signals"]["breadth"])
    records: list[pd.DataFrame] = []
    date_values = features["date_idx"].to_numpy(dtype=np.int64)
    boundaries = np.r_[
        0, np.flatnonzero(date_values[1:] != date_values[:-1]) + 1, len(features)
    ]
    keep = [
        "row_position",
        "candidate_id",
        "date_idx",
        "trade_date",
        "symbol",
        "evaluation_year",
        "industry_code",
        "valuation_score",
        "earnings_revision_score",
        "quality_score_exact",
        "price_confirmation_score_exact",
        "governance_score",
    ]
    for left, right in pairwise(boundaries):
        local = features.iloc[int(left) : int(right)]
        for policy in POLICIES:
            top_k, cap = (int(value) for value in breadth[policy])
            score = local[score_columns[policy]].to_numpy(dtype=float)
            eligible = local[f"eligible__{policy}"].to_numpy(dtype=bool)
            chosen = causal._select_industry_capped(
                score=score,
                eligible=eligible,
                candidate_id=local["candidate_id"].to_numpy(dtype=np.int64),
                industry_code=local["industry_code"].to_numpy(dtype=np.int64),
                top_k=top_k,
                industry_cap=cap,
            )
            if not len(chosen):
                continue
            selected = local.iloc[chosen][keep].copy()
            selected.insert(0, "policy", policy)
            selected["policy_score"] = score[chosen]
            selected["selection_rank"] = np.arange(1, len(chosen) + 1)
            selected["top_k"] = top_k
            selected["industry_cap"] = cap
            records.append(selected)
    if not records:
        raise ValueError("selection_frame_empty")
    return pd.concat(records, ignore_index=True)


def _cohort_frame(
    panel: base.StockPanel,
    features: pd.DataFrame,
    selections: pd.DataFrame,
    *,
    retry_days: int,
    cost: float,
) -> pd.DataFrame:
    rows = features["row_position"].to_numpy(dtype=np.int64)
    outcome = causal._d60_timeout(panel, rows, retry_days=retry_days)
    values = np.asarray(outcome["simple_return"], dtype=float)
    valid = np.asarray(outcome["valid"], dtype=bool)
    within = np.asarray(outcome["within"], dtype=bool)
    residual, _, _ = base._leave_one_out_factors(
        values,
        valid,
        features["date_idx"].to_numpy(dtype=np.int64),
        features["industry_code"].to_numpy(dtype=np.int64),
    )
    universe: dict[int, float] = {}
    date_values = features["date_idx"].to_numpy(dtype=np.int64)
    for date_idx in np.unique(date_values):
        good = (date_values == int(date_idx)) & valid & np.isfinite(values)
        universe[int(date_idx)] = float(values[good].mean()) if good.any() else np.nan
    position_map = pd.Series(
        np.arange(len(features), dtype=np.int64),
        index=features["row_position"].to_numpy(dtype=np.int64),
    )
    selected = selections.copy()
    selected["local_position"] = selected["row_position"].map(position_map)
    if bool(selected["local_position"].isna().any()):
        raise ValueError("selection_outcome_alignment_failed")
    selected["local_position"] = selected["local_position"].astype(np.int64)
    records: list[dict[str, Any]] = []
    for (policy, date_idx), group in selected.groupby(
        ["policy", "date_idx"], sort=True
    ):
        local = group["local_position"].to_numpy(dtype=np.int64)
        if not bool(within[local].all()):
            continue
        good = valid[local] & np.isfinite(values[local])
        observed = values[local][good]
        good_residual = np.isfinite(residual[local])
        top_k = int(group["top_k"].iloc[0])
        selected_mean = float(observed.mean()) if len(observed) else np.nan
        universe_mean = universe[int(date_idx)]
        records.append(
            {
                "policy": str(policy),
                "date_idx": int(date_idx),
                "trade_date": str(group["trade_date"].iloc[0]),
                "evaluation_year": int(group["evaluation_year"].iloc[0]),
                "top_k": top_k,
                "selected_count": len(local),
                "observed_count": int(good.sum()),
                "observed_fraction": float(good.sum() / top_k),
                "stress_net_return": (
                    float(observed.sum()) - float(cost) * int(good.sum())
                )
                / top_k,
                "selected_mean_gross": selected_mean,
                "universe_mean_gross": universe_mean,
                "selected_excess_gross": selected_mean - universe_mean
                if np.isfinite(selected_mean) and np.isfinite(universe_mean)
                else np.nan,
                "industry_residual_mean": float(residual[local][good_residual].mean())
                if good_residual.any()
                else np.nan,
            }
        )
    return pd.DataFrame(records)


def _inference(
    values: np.ndarray, study: Mapping[str, Any], *, seed_add: int
) -> dict[str, Any]:
    config = dict(study["evaluation"]["inference"])
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return {
        **base._hac_mean(x, lag=int(config["hac_lag_months"])),
        "block": base._block_interval(
            x,
            block_length=int(config["block_length_months"]),
            repetitions=int(config["bootstrap_repetitions"]),
            seed=int(config["seed"]) + int(seed_add),
        ),
    }


def _period_name(year: int, study: Mapping[str, Any]) -> str:
    for name, years in study["evaluation"]["periods"].items():
        if int(year) in {int(value) for value in years}:
            return str(name)
    raise ValueError(f"evaluation_year_outside_contract:{year}")


def _summaries(cohorts: pd.DataFrame, study: Mapping[str, Any]) -> list[dict[str, Any]]:
    frame = cohorts.copy()
    frame["period"] = frame["evaluation_year"].map(
        lambda value: _period_name(int(value), study)
    )
    frames = [(name, group) for name, group in frame.groupby("period", sort=True)]
    frames.append(("full_history", frame))
    results: list[dict[str, Any]] = []
    for period, period_frame in frames:
        for policy, group in period_frame.groupby("policy", sort=True):
            annual = group.groupby("evaluation_year", sort=True)[
                "stress_net_return"
            ].mean()
            best_year = int(annual.idxmax()) if len(annual) else -1
            excluding = group.loc[
                group["evaluation_year"].ne(best_year), "stress_net_return"
            ]
            results.append(
                {
                    "period": period,
                    "policy": policy,
                    "month_count": len(group),
                    "mean_selected_count": float(group["selected_count"].mean()),
                    "minimum_selected_count": int(group["selected_count"].min()),
                    "observed_fraction": float(
                        group["observed_count"].sum() / group["top_k"].sum()
                    ),
                    "stress_net": _inference(
                        group["stress_net_return"].to_numpy(),
                        study,
                        seed_add=len(results),
                    ),
                    "selected_excess_gross": _inference(
                        group["selected_excess_gross"].to_numpy(),
                        study,
                        seed_add=1000 + len(results),
                    ),
                    "industry_residual": _inference(
                        group["industry_residual_mean"].to_numpy(),
                        study,
                        seed_add=2000 + len(results),
                    ),
                    "positive_year_count": int((annual > 0.0).sum()),
                    "year_count": len(annual),
                    "best_year": best_year,
                    "excluding_best_year_mean": float(excluding.mean())
                    if len(excluding)
                    else np.nan,
                    "annual": [
                        {"year": int(year), "stress_net_return": float(value)}
                        for year, value in annual.items()
                    ],
                }
            )
    return results


def _paired_controls(
    cohorts: pd.DataFrame, study: Mapping[str, Any]
) -> list[dict[str, Any]]:
    primary = str(study["signals"]["primary_policy"])
    controls = ("forward_value_only_top10", "ttm_fcf_only_top10")
    left = cohorts.loc[
        cohorts["policy"].eq(primary),
        ["date_idx", "evaluation_year", "stress_net_return"],
    ].rename(columns={"stress_net_return": "primary_return"})
    results: list[dict[str, Any]] = []
    for control in controls:
        right = cohorts.loc[
            cohorts["policy"].eq(control), ["date_idx", "stress_net_return"]
        ].rename(columns={"stress_net_return": "control_return"})
        paired = left.merge(right, on="date_idx", validate="one_to_one")
        paired["delta"] = paired["primary_return"] - paired["control_return"]
        paired["period"] = paired["evaluation_year"].map(
            lambda value: _period_name(int(value), study)
        )
        groups = [(name, group) for name, group in paired.groupby("period", sort=True)]
        groups.append(("full_history", paired))
        for period, group in groups:
            results.append(
                {
                    "control": control,
                    "period": period,
                    "month_count": len(group),
                    "primary_minus_control": _inference(
                        group["delta"].to_numpy(), study, seed_add=3000 + len(results)
                    ),
                }
            )
    return results


def _coverage(features: pd.DataFrame) -> list[dict[str, Any]]:
    frame = features.copy()
    minimum = 3
    frame["forward_covered"] = frame["current_np_n"].fillna(0).ge(minimum)
    frame["revision_30_covered"] = frame["prior_30_np_n"].fillna(0).ge(minimum) & frame[
        "prior_30_eps_n"
    ].fillna(0).ge(minimum)
    frame["revision_90_covered"] = frame["prior_90_np_n"].fillna(0).ge(minimum) & frame[
        "prior_90_eps_n"
    ].fillna(0).ge(minimum)
    frame["cycle_covered"] = (
        frame["cycle_history_count"].fillna(0).ge(3)
        & frame["normalized_parent_net_income_base"].notna()
    )
    frame["ev_ebitda_covered"] = frame["current_ev_n"].fillna(0).ge(minimum)
    return (
        frame.groupby("evaluation_year", sort=True, as_index=False)
        .agg(
            candidate_count=("candidate_id", "size"),
            forward_covered_fraction=("forward_covered", "mean"),
            revision_30_covered_fraction=("revision_30_covered", "mean"),
            revision_90_covered_fraction=("revision_90_covered", "mean"),
            cycle_covered_fraction=("cycle_covered", "mean"),
            ev_ebitda_covered_fraction=("ev_ebitda_covered", "mean"),
            exact_eligible_fraction=("eligible__exact_full_top10", "mean"),
        )
        .to_dict(orient="records")
    )


def _decision(
    summaries: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    study: Mapping[str, Any],
) -> dict[str, Any]:
    primary = str(study["signals"]["primary_policy"])
    primary_rows = [row for row in summaries if row["policy"] == primary]
    full = next(row for row in primary_rows if row["period"] == "full_history")
    period_rows = [row for row in primary_rows if row["period"] != "full_history"]
    paired_counts: dict[str, int] = {}
    for control in ("forward_value_only_top10", "ttm_fcf_only_top10"):
        rows = [
            row
            for row in paired
            if row["control"] == control and row["period"] != "full_history"
        ]
        paired_counts[control] = sum(
            float(row["primary_minus_control"]["mean"]) > 0.0 for row in rows
        )
    checks = {
        "all_three_period_primary_point_means_positive": len(period_rows) == 3
        and all(float(row["stress_net"]["mean"]) > 0.0 for row in period_rows),
        "full_history_stress_net_hac_lcb_positive": float(full["stress_net"]["lcb_95"])
        > 0.0,
        "full_history_stress_net_block_lcb_positive": float(
            full["stress_net"]["block"]["lcb_95"]
        )
        > 0.0,
        "full_history_industry_residual_hac_lcb_positive": float(
            full["industry_residual"]["lcb_95"]
        )
        > 0.0,
        "full_history_industry_residual_block_lcb_positive": float(
            full["industry_residual"]["block"]["lcb_95"]
        )
        > 0.0,
        "minimum_positive_years": int(full["positive_year_count"])
        >= int(study["evaluation"]["information_gate"]["minimum_positive_years"]),
        "excluding_best_year_primary_mean_positive": float(
            full["excluding_best_year_mean"]
        )
        > 0.0,
        "paired_mean_beats_forward_value_in_two_periods": paired_counts[
            "forward_value_only_top10"
        ]
        >= 2,
        "paired_mean_beats_ttm_fcf_in_two_periods": paired_counts["ttm_fcf_only_top10"]
        >= 2,
    }
    passed = all(checks.values())
    return {
        "primary_policy": primary,
        "information_gate_passed": passed,
        "checks": checks,
        "paired_positive_period_counts": paired_counts,
        "account_replay_authorized": passed,
        "profit_claim_allowed": False,
        "reason": "run exact finite account next"
        if passed
        else "frozen information gate failed; do not spend account-replay degrees of freedom",
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
            for record in current.get("files", {}).values()
        ):
            return current
    panel = base.load_panel(
        input_manifest_path=sources["input_manifest"],
        label_manifest_path=sources["label_manifest"],
    )
    if bool((panel.years == base.FORBIDDEN_YEAR).any()):
        raise ValueError("forbidden_2026_panel_row")
    features = _assemble_features(panel, sources, study)
    features = _score_features(features, study)
    selections = _selection_frame(features, study)
    features_path = root / "candidate_features.parquet"
    selections_path = root / "selections.parquet"
    _write_parquet(features_path, features)
    _write_parquet(selections_path, selections)
    cohorts = _cohort_frame(
        panel,
        features,
        selections,
        retry_days=int(study["execution"]["sell_retry_open_days"]),
        cost=float(study["execution"]["round_trip_cost_proxy"]),
    )
    cohorts_path = root / "cohort_returns.parquet"
    _write_parquet(cohorts_path, cohorts)
    summaries = _summaries(cohorts, study)
    paired = _paired_controls(cohorts, study)
    decision = _decision(summaries, paired, study)
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
        "main_board_only": True,
        "candidate_selection_precedes_outcome_reads": True,
        "qualitative_moat_backfilled": False,
        "right_edge_valuation_exit_used": False,
        "52_week_low_reward_used": False,
        "forbidden_2026_read_count": 0,
        "coverage": _coverage(features),
        "policy_summaries": summaries,
        "paired_control_summaries": paired,
        "decision": decision,
        "historical_account_replay_performed": False,
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
        "main_board": summary.get("main_board_only") is True,
        "selection_before_outcomes": summary.get(
            "candidate_selection_precedes_outcome_reads"
        )
        is True,
        "no_qualitative_backfill": summary.get("qualitative_moat_backfilled") is False,
        "no_right_edge": summary.get("right_edge_valuation_exit_used") is False,
        "no_low_reward": summary.get("52_week_low_reward_used") is False,
        "no_2026": int(summary.get("forbidden_2026_read_count", -1)) == 0,
        "files": all(
            base._record_valid(record, verify_hash=True)
            for record in summary.get("files", {}).values()
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
