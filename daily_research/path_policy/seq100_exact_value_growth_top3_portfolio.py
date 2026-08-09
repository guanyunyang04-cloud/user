"""Adaptive retrospective Top3 concentration and holding-policy comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_exact_value_growth_account as account
from daily_research.path_policy import (
    seq100_exact_value_growth_exit_challenge as challenge,
)
from daily_research.path_policy import seq100_exact_value_growth_path_exit as diagnostic
from daily_research.path_policy import seq100_finite_capital_backtest as finite
from daily_research.path_policy import (
    seq100_margin_residual_account_feasibility as feasibility,
)
from daily_research.path_policy import seq100_stock_distribution as base
from daily_research.path_policy.seq100_exit_policy_audit import (
    CandidateCompleteAuditPack,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_exact_value_growth_top3_portfolio_v1"
SUMMARY_SCHEMA = "seq100_exact_value_growth_top3_portfolio_summary/1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT
    / "daily_research/studies/seq100_exact_value_growth_top3_portfolio_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT / "daily_research/output/path_policy/studies/"
    "seq100_exact_value_growth_top3_portfolio_v1"
)


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
    epistemic = dict(study["epistemic_contract"])
    required_true = (
        "top3_question_was_asked_after_all_2012_2025_top10_outcomes_were_read",
        "all_2012_2025_results_are_adaptive_retrospective_evidence",
        "no_untouched_confirmation_claim",
        "no_2026_outcome_may_be_read",
        "original_2026_shadow_arms_must_not_change",
    )
    if not all(bool(epistemic.get(key)) for key in required_true):
        raise ValueError("epistemic_contract_mismatch")
    if int(study["source"]["forbidden_year"]) != base.FORBIDDEN_YEAR:
        raise ValueError("forbidden_year_contract_mismatch")
    selection = dict(study["selection"])
    if tuple(selection["selected_ranks"]) != (1, 2, 3):
        raise ValueError("selected_rank_contract_mismatch")
    if int(selection["monthly_order_count"]) != 3:
        raise ValueError("monthly_order_count_mismatch")
    if bool(selection["rank4_or_lower_replacement_is_forbidden"]) is not True:
        raise ValueError("rank_replacement_contract_mismatch")
    if bool(selection["pyramiding"]):
        raise ValueError("pyramiding_forbidden")
    expected = {
        "monthly_rebalance_max3": ("model_plan_to_next_monthly_signal", 3, True),
        "fixed_d20_max3": ("fixed_d20", 3, True),
        "fixed_d60_max3": ("fixed_d60", 3, True),
        "fixed_d120_max3": ("fixed_d120", 3, True),
        "fixed_d20_overlap6": ("fixed_d20", 6, False),
        "fixed_d60_overlap12": ("fixed_d60", 12, False),
        "fixed_d120_overlap21": ("fixed_d120", 21, False),
    }
    actual = {
        str(row["name"]): (
            str(row["policy"]),
            int(row["slots"]),
            bool(row["literal_maximum_three_names"]),
        )
        for row in study["strategy_family"]
    }
    if actual != expected:
        raise ValueError("strategy_family_contract_mismatch")
    risk = dict(study["risk_budget"])
    expected_risk = {
        "derivation_end_date": "2019-12-31",
        "target_annualized_volatility": 0.15,
        "minimum_gross_fraction": 0.25,
        "maximum_gross_fraction": 1.0,
        "one_fraction_per_strategy_from_its_own_full_exposure_development_path": True,
        "parameter_grid_performed": False,
    }
    if risk != expected_risk:
        raise ValueError("risk_budget_contract_mismatch")
    if bool(study["decision_boundary"].get("production_claim_allowed", True)):
        raise ValueError("production_claim_forbidden")
    return study, study_path


def _source_contract(
    study: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, Path], dict[str, Any]]:
    source = dict(study["source"])
    source_keys = (
        "path_diagnostic_summary",
        "selected_paths",
        "exact_account_study",
        "exact_account_summary",
        "exit_challenge_study",
        "exit_challenge_summary",
    )
    sources: dict[str, Path] = {}
    for key in source_keys:
        path = base._resolve_path(source[key])
        expected = str(source[f"{key}_sha256"]).lower()
        if not path.is_file() or base._sha256_file(path).lower() != expected:
            raise ValueError(f"source_invalid:{key}")
        sources[key] = path
    diagnostic.validate_summary(sources["path_diagnostic_summary"])
    account.validate_summary(sources["exact_account_summary"])
    challenge.validate_summary(sources["exit_challenge_summary"])
    account_study, loaded_path = account.load_study(sources["exact_account_study"])
    if loaded_path != sources["exact_account_study"]:
        raise ValueError("account_study_path_drift")
    account_sources = account._source_contract(account_study)
    return sources, account_sources, account_study


def _period_years(study: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    periods = {
        str(name): tuple(int(value) for value in values)
        for name, values in study["evaluation"]["periods"].items()
    }
    periods = {
        "full_history": tuple(int(value) for value in study["evaluation"]["years"]),
        **periods,
    }
    return periods


def _cohort_monthly(paths: pd.DataFrame, study: Mapping[str, Any]) -> pd.DataFrame:
    if int(paths["evaluation_year"].max()) >= base.FORBIDDEN_YEAR:
        raise ValueError("forbidden_2026_path")
    if bool(paths.duplicated(["trade_date", "selection_rank"]).any()):
        raise ValueError("duplicate_monthly_selection_rank")
    groups = {
        "top3": (1, 3, 3),
        "ranks4_10": (4, 10, 7),
        "top10": (1, 10, 10),
    }
    records: list[dict[str, Any]] = []
    for horizon in study["evaluation"]["cohort_horizons"]:
        horizon = int(horizon)
        return_column = f"legal_net_return_d{horizon}"
        for group_name, (minimum_rank, maximum_rank, denominator) in groups.items():
            selected = paths.loc[
                paths["selection_rank"].between(minimum_rank, maximum_rank),
                ["trade_date", "evaluation_year", "selection_rank", return_column],
            ]
            for (trade_date, year), month in selected.groupby(
                ["trade_date", "evaluation_year"], sort=True
            ):
                values = month[return_column].to_numpy(dtype=np.float64)
                complete = len(month) == denominator and bool(np.isfinite(values).all())
                records.append(
                    {
                        "trade_date": str(trade_date),
                        "evaluation_year": int(year),
                        "horizon": horizon,
                        "rank_group": group_name,
                        "rank_minimum": minimum_rank,
                        "rank_maximum": maximum_rank,
                        "cash_denominator": denominator,
                        "observed_count": int(np.isfinite(values).sum()),
                        "complete_month": complete,
                        "cash_denominator_net_return": (
                            float(values.sum() / denominator) if complete else math.nan
                        ),
                    }
                )
    return pd.DataFrame(records).sort_values(
        ["horizon", "rank_group", "trade_date"], kind="stable"
    )


def _cohort_summaries(
    paths: pd.DataFrame,
    monthly: pd.DataFrame,
    study: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    periods = _period_years(study)
    group_summaries: list[dict[str, Any]] = []
    rank_summaries: list[dict[str, Any]] = []
    paired_summaries: list[dict[str, Any]] = []
    seed_add = 0
    for period, years in periods.items():
        year_set = set(years)
        for (horizon, rank_group), group in monthly.loc[
            monthly["evaluation_year"].isin(year_set) & monthly["complete_month"]
        ].groupby(["horizon", "rank_group"], sort=True):
            values = group["cash_denominator_net_return"].to_numpy(dtype=np.float64)
            annual = group.groupby("evaluation_year", sort=True)[
                "cash_denominator_net_return"
            ].mean()
            group_summaries.append(
                {
                    "period": period,
                    "horizon": int(horizon),
                    "rank_group": str(rank_group),
                    "month_count": len(group),
                    "mean_monthly_net_return": float(np.mean(values)),
                    "median_monthly_net_return": float(np.median(values)),
                    "positive_month_fraction": float(np.mean(values > 0.0)),
                    "inference": diagnostic._inference(
                        values, study, seed_add=seed_add
                    ),
                    "positive_year_count": int((annual > 0.0).sum()),
                    "year_count": len(annual),
                    "worst_annual_mean_monthly_return": float(annual.min()),
                }
            )
            seed_add += 1
        for horizon in study["evaluation"]["cohort_horizons"]:
            horizon = int(horizon)
            return_column = f"legal_net_return_d{horizon}"
            local = paths.loc[
                paths["evaluation_year"].isin(year_set)
                & paths[return_column].notna()
                & paths["selection_rank"].le(10)
            ]
            for rank, group in local.groupby("selection_rank", sort=True):
                values = group[return_column].to_numpy(dtype=np.float64)
                annual = group.groupby("evaluation_year", sort=True)[
                    return_column
                ].mean()
                rank_summaries.append(
                    {
                        "period": period,
                        "horizon": horizon,
                        "selection_rank": int(rank),
                        "candidate_count": len(group),
                        "mean_net_return": float(np.mean(values)),
                        "median_net_return": float(np.median(values)),
                        "winning_fraction": float(np.mean(values > 0.0)),
                        "inference": diagnostic._inference(
                            values, study, seed_add=1000 + seed_add
                        ),
                        "positive_year_count": int((annual > 0.0).sum()),
                        "year_count": len(annual),
                    }
                )
                seed_add += 1
        for horizon in study["evaluation"]["cohort_horizons"]:
            local = monthly.loc[
                monthly["evaluation_year"].isin(year_set)
                & monthly["complete_month"]
                & monthly["horizon"].eq(int(horizon))
                & monthly["rank_group"].isin(["top3", "ranks4_10"])
            ].pivot(
                index=["trade_date", "evaluation_year"],
                columns="rank_group",
                values="cash_denominator_net_return",
            )
            local = local.dropna(subset=["top3", "ranks4_10"])
            delta = (local["top3"] - local["ranks4_10"]).to_numpy(dtype=np.float64)
            annual_delta = (
                local.assign(delta=local["top3"] - local["ranks4_10"])
                .groupby(level="evaluation_year")["delta"]
                .mean()
            )
            paired_summaries.append(
                {
                    "period": period,
                    "horizon": int(horizon),
                    "comparison": "top3_minus_ranks4_10",
                    "paired_month_count": len(local),
                    "mean_delta": float(np.mean(delta)),
                    "median_delta": float(np.median(delta)),
                    "inference": diagnostic._inference(
                        delta, study, seed_add=2000 + seed_add
                    ),
                    "positive_delta_year_count": int((annual_delta > 0.0).sum()),
                    "year_count": len(annual_delta),
                }
            )
            seed_add += 1
    return group_summaries, rank_summaries, paired_summaries


def _strict_top3_book(
    source_book: finite.ForecastBook,
    *,
    maximum_date_idx: int,
    monthly_plan: bool,
    profile: str,
) -> tuple[finite.ForecastBook, dict[str, Any]]:
    source_dates = sorted(int(value) for value in source_book.days)
    included = [value for value in source_dates if value <= int(maximum_date_idx)]
    if not included:
        raise ValueError("top3_book_empty")
    date_position = {value: position for position, value in enumerate(source_dates)}
    result = finite.ForecastBook(profile, top_k=3, candidate_scan_k=3)
    plan_days: list[int] = []
    for date_idx in included:
        source_day = source_book.days[date_idx]
        ranked = tuple(int(value) for value in source_day.ranked_symbol_idx[:3])
        if len(ranked) != 3:
            raise ValueError(f"source_top3_incomplete:{date_idx}")
        scores: list[float] = []
        for symbol_idx in ranked:
            lookup = source_book.lookup(date_idx, symbol_idx)
            if lookup is None:
                raise ValueError("source_top3_lookup_failed")
            scores.append(float(lookup[0]))
        planned_day = 60
        if monthly_plan:
            position = date_position[date_idx]
            if position + 1 >= len(source_dates):
                raise ValueError("next_monthly_signal_missing")
            planned_day = int(source_dates[position + 1] - date_idx)
            if not 2 <= planned_day <= 60:
                raise ValueError(f"monthly_plan_day_invalid:{planned_day}")
        result.add_day(
            date_idx=date_idx,
            symbol_idx=np.asarray(ranked, dtype=np.int64),
            score=np.asarray(scores, dtype=np.float64),
            planned_day=np.full(3, planned_day, dtype=np.int16),
        )
        if result.days[date_idx].ranked_symbol_idx != ranked:
            raise ValueError(f"top3_rank_drift:{date_idx}")
        plan_days.append(planned_day)
    return result, {
        "signal_date_count": len(result.days),
        "first_signal_date_idx": min(result.days),
        "last_signal_date_idx": max(result.days),
        "selected_names_per_signal": 3,
        "candidate_scan_count": 3,
        "minimum_planned_day": min(plan_days),
        "maximum_planned_day": max(plan_days),
        "rank4_or_lower_present": False,
    }


def _unconstrained_cohort_slots(
    signal_dates: Sequence[int], *, horizon: int, names_per_signal: int = 3
) -> int:
    dates = tuple(sorted(int(value) for value in signal_dates))
    maximum = 0
    for current in dates:
        active = sum(
            1
            for signal in dates
            if signal <= current and signal + int(horizon) > current
        )
        maximum = max(maximum, active * int(names_per_signal))
    return maximum


def _policy(strategy: Mapping[str, Any]) -> finite.PolicySpec:
    name = str(strategy["name"])
    policy = str(strategy["policy"])
    if policy == "model_plan_to_next_monthly_signal":
        return finite.PolicySpec(name=name, kind="model_plan")
    if not policy.startswith("fixed_d"):
        raise ValueError(f"unknown_top3_policy:{policy}")
    horizon = int(policy.removeprefix("fixed_d"))
    return challenge.ExtendedFixedPolicy(name=name, kind="fixed", fixed_day=horizon)


def _run_account(
    *,
    market: finite.BacktestMarket,
    book: finite.ForecastBook,
    policy: finite.PolicySpec,
    slots: int,
    study: Mapping[str, Any],
    signal_amount: np.ndarray,
    years: Sequence[int],
    cutoff: int,
    target_gross_fraction: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    config = dict(study["account"])
    return finite.simulate_portfolio(
        market=market,
        book=book,
        raw_top3_paths={},
        policy=policy,
        slots=int(slots),
        cost_scenario=str(config["cost_scenario"]),
        first_signal_date_idx=int(min(book.days)),
        last_signal_date_idx=int(max(book.days)),
        starting_cash=float(config["starting_cash_cny"]),
        allow_pyramiding=False,
        replace_rejected_from_ranked_candidates=False,
        calendar_years=tuple(int(value) for value in years),
        top_k=3,
        signal_amount_panel=signal_amount,
        maximum_signal_amount_fraction=float(
            config["maximum_signal_day_amount_fraction"]
        ),
        target_gross_fraction=float(target_gross_fraction),
        terminal_recovery_date_idx=int(cutoff),
    )


def _annual_stability(
    annual: Sequence[Mapping[str, Any]], study: Mapping[str, Any]
) -> dict[str, Any]:
    rows = [dict(row) for row in annual]
    returns = np.asarray([float(row["net_return"]) for row in rows], dtype=np.float64)
    periods = _period_years(study)

    def positive_count(name: str) -> int:
        years = set(periods[name])
        return sum(
            int(row["year"]) in years and float(row["net_return"]) > 0.0 for row in rows
        )

    order = np.sort(returns)
    return {
        "positive_year_count": int(np.sum(returns > 0.0)),
        "year_count": len(rows),
        "development_positive_year_count": positive_count("development"),
        "post_development_positive_year_count": positive_count("post_development"),
        "late_positive_year_count": positive_count("late"),
        "mean_annual_return": float(np.mean(returns)),
        "median_annual_return": float(np.median(returns)),
        "worst_annual_return": float(np.min(returns)),
        "best_annual_return": float(np.max(returns)),
        "annual_return_standard_deviation": float(np.std(returns, ddof=1)),
        "mean_return_excluding_best_year": float(np.mean(order[:-1])),
        "worst_three_year_mean_return": float(np.mean(order[:3])),
    }


def _risk_gate(
    metric: Mapping[str, Any],
    stability: Mapping[str, Any],
    study: Mapping[str, Any],
) -> dict[str, Any]:
    contract = dict(study["evaluation"]["risk_account_gate"])
    checks = {
        "total_return_positive": float(metric["liquidated_total_return"]) > 0.0,
        "not_ruined": metric["ruined"] is False,
        "annualized_volatility_within_limit": float(
            metric["signal_period_annualized_volatility"]
        )
        <= float(contract["maximum_annualized_volatility"]),
        "maximum_drawdown_within_limit": float(metric["signal_period_maximum_drawdown"])
        >= -float(contract["maximum_drawdown"]),
        "minimum_positive_years": int(stability["positive_year_count"])
        >= int(contract["minimum_positive_years"]),
        "minimum_post_development_positive_years": int(
            stability["post_development_positive_year_count"]
        )
        >= int(contract["minimum_post_development_positive_years"]),
        "minimum_late_positive_years": int(stability["late_positive_year_count"])
        >= int(contract["minimum_late_positive_years"]),
    }
    return {"passed": bool(all(checks.values())), "checks": checks}


def _metric_summary(
    metric: Mapping[str, Any],
    annual: Sequence[Mapping[str, Any]],
    study: Mapping[str, Any],
) -> dict[str, Any]:
    stability = _annual_stability(annual, study)
    log_growth = float(metric["annualized_log_growth"])
    return {
        **dict(metric),
        "annualized_compound_return_from_log_growth": float(math.expm1(log_growth)),
        "stability": stability,
    }


def _decision(
    risk_metrics: Mapping[str, Mapping[str, Any]],
    gates: Mapping[str, Mapping[str, Any]],
    strategy_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    literal = [
        str(row["name"])
        for row in strategy_rows
        if bool(row["literal_maximum_three_names"])
    ]
    overlap = [
        str(row["name"])
        for row in strategy_rows
        if not bool(row["literal_maximum_three_names"])
    ]
    passing = [name for name in literal if bool(gates[name]["passed"])]
    primary = (
        max(
            passing, key=lambda name: float(risk_metrics[name]["annualized_log_growth"])
        )
        if passing
        else None
    )
    highest_return = max(
        literal, key=lambda name: float(risk_metrics[name]["annualized_log_growth"])
    )
    lowest_drawdown = min(
        literal,
        key=lambda name: abs(
            float(risk_metrics[name]["signal_period_maximum_drawdown"])
        ),
    )
    overlap_best = max(
        overlap, key=lambda name: float(risk_metrics[name]["annualized_log_growth"])
    )
    return {
        "stable_literal_max3_winner_found": primary is not None,
        "primary_literal_max3_choice": primary,
        "passing_literal_max3_variants": passing,
        "highest_return_literal_max3_variant": highest_return,
        "lowest_drawdown_literal_max3_variant": lowest_drawdown,
        "best_overlapping_monthly_top3_diagnostic": overlap_best,
        "all_results_are_adaptive_retrospective": True,
        "production_claim_allowed": False,
        "forward_confirmation_required": True,
    }


def _write_run(
    root: Path,
    prefix: str,
    result: tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for suffix, frame in (
        ("equity", result[1]),
        ("trades", result[2]),
        ("annual", pd.DataFrame(result[3])),
    ):
        path = root / f"{prefix}_{suffix}.parquet"
        _write_parquet(path, frame)
        files[f"{prefix}_{suffix}"] = base._file_record(path)
    return files


def run_study(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    study, frozen_path = load_study(study_path)
    sources, account_sources, account_study = _source_contract(study)
    root = base._resolve_path(output_root)
    summary_path = root / "summary.json"
    fingerprint = _payload_hash(
        {
            "study_sha256": base._sha256_file(frozen_path),
            "implementation_sha256": base._sha256_file(Path(__file__)),
            "source_hashes": {
                **{key: base._sha256_file(path) for key, path in sources.items()},
                **{
                    f"account_source__{key}": base._sha256_file(path)
                    for key, path in account_sources.items()
                    if path.is_file()
                },
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

    paths = pd.read_parquet(sources["selected_paths"])
    monthly = _cohort_monthly(paths, study)
    cohort_groups, cohort_ranks, cohort_paired = _cohort_summaries(
        paths, monthly, study
    )

    panel = base.load_panel(
        input_manifest_path=account_sources["input_manifest"],
        label_manifest_path=account_sources["label_manifest"],
    )
    pack = CandidateCompleteAuditPack(account_sources["pack_manifest"])
    feasibility._validate_pack_alignment(panel, pack)
    source_book, account_candidates, source_selection_audit = (
        account._account_selection(
            panel,
            pack,
            study=account_study,
            feature_path=account_sources["candidate_features"],
            selection_path=account_sources["selections"],
        )
    )
    adjust_factor, factor_audit = feasibility._load_adjust_factor_panel(
        pack, study=account_study
    )
    boundary = dict(study["common_account_boundary"])
    market = replace(
        account._market(pack, adjust_factor=adjust_factor),
        forward_days=int(boundary["extended_forward_days"]),
        execution_days=int(boundary["extended_execution_days"]),
    )
    signal_amount = feasibility._signal_amount_panel(panel, pack)
    dates = np.asarray(pack.date_values, dtype=str)
    cutoff_positions = np.flatnonzero(
        dates == str(account_study["source"]["maximum_account_mark_date"])
    )
    if len(cutoff_positions) != 1:
        raise ValueError("maximum_account_mark_date_missing")
    cutoff = int(cutoff_positions[0])
    common_last_signal = cutoff - int(boundary["extended_execution_days"])
    fixed_book, fixed_book_audit = _strict_top3_book(
        source_book,
        maximum_date_idx=common_last_signal,
        monthly_plan=False,
        profile="exact_value_growth_strict_top3_fixed",
    )
    monthly_book, monthly_book_audit = _strict_top3_book(
        source_book,
        maximum_date_idx=common_last_signal,
        monthly_plan=True,
        profile="exact_value_growth_strict_top3_monthly",
    )
    if tuple(fixed_book.days) != tuple(monthly_book.days):
        raise ValueError("top3_strategy_signal_dates_drift")
    common_candidates = account_candidates.loc[
        account_candidates["date_idx"].le(common_last_signal)
        & account_candidates["candidate_scan_rank"].le(3)
    ].copy()
    source_path_top3 = paths.loc[
        paths["date_idx"].le(common_last_signal) & paths["selection_rank"].le(3),
        ["date_idx", "selection_rank", "candidate_id"],
    ].sort_values(["date_idx", "selection_rank"])
    account_path_top3 = common_candidates[
        ["date_idx", "candidate_scan_rank", "candidate_id"]
    ].rename(columns={"candidate_scan_rank": "selection_rank"})
    account_path_top3 = account_path_top3.sort_values(["date_idx", "selection_rank"])
    if not np.array_equal(
        source_path_top3[["date_idx", "selection_rank", "candidate_id"]].to_numpy(),
        account_path_top3[["date_idx", "selection_rank", "candidate_id"]].to_numpy(),
    ):
        raise ValueError("top3_path_account_alignment_failed")

    overlap_requirements = {
        horizon: _unconstrained_cohort_slots(
            fixed_book.signal_date_indices, horizon=horizon
        )
        for horizon in (20, 60, 120)
    }
    strategy_rows = [dict(row) for row in study["strategy_family"]]
    for row in strategy_rows:
        policy_name = str(row["policy"])
        if not bool(row["literal_maximum_three_names"]) and policy_name.startswith(
            "fixed_d"
        ):
            horizon = int(policy_name.removeprefix("fixed_d"))
            if int(row["slots"]) < int(overlap_requirements[horizon]):
                raise ValueError(f"overlap_slots_insufficient:{row['name']}")

    years = tuple(int(value) for value in study["evaluation"]["years"])
    starting_cash = float(study["account"]["starting_cash_cny"])
    files: dict[str, dict[str, Any]] = {}
    full_metrics: dict[str, dict[str, Any]] = {}
    risk_metrics: dict[str, dict[str, Any]] = {}
    risk_budgets: dict[str, dict[str, Any]] = {}
    risk_gates: dict[str, dict[str, Any]] = {}
    for strategy in strategy_rows:
        name = str(strategy["name"])
        book = monthly_book if name == "monthly_rebalance_max3" else fixed_book
        policy = _policy(strategy)
        full = _run_account(
            market=market,
            book=book,
            policy=policy,
            slots=int(strategy["slots"]),
            study=study,
            signal_amount=signal_amount,
            years=years,
            cutoff=cutoff,
            target_gross_fraction=1.0,
        )
        budget = feasibility._derive_risk_budget(
            full[1], starting_cash=starting_cash, config=study["risk_budget"]
        )
        risk = _run_account(
            market=market,
            book=book,
            policy=policy,
            slots=int(strategy["slots"]),
            study=study,
            signal_amount=signal_amount,
            years=years,
            cutoff=cutoff,
            target_gross_fraction=float(budget["frozen_target_gross_fraction"]),
        )
        full_metrics[name] = _metric_summary(full[0], full[3], study)
        risk_metrics[name] = _metric_summary(risk[0], risk[3], study)
        risk_budgets[name] = budget
        risk_gates[name] = _risk_gate(
            risk_metrics[name], risk_metrics[name]["stability"], study
        )
        files.update(_write_run(root, f"full__{name}", full))
        files.update(_write_run(root, f"risk__{name}", risk))

    for name, frame in (
        ("cohort_monthly", monthly),
        ("common_top3_candidates", common_candidates),
    ):
        output_path = root / f"{name}.parquet"
        _write_parquet(output_path, frame)
        files[name] = base._file_record(output_path)

    exact_summary = _read_json(sources["exact_account_summary"])
    exit_summary = _read_json(sources["exit_challenge_summary"])
    prior_top10 = {
        "d60_original_boundary_risk_metric": exact_summary["risk_metric"],
        "d60_original_boundary_risk_budget": exact_summary["risk_budget"],
        "common_boundary_d60_risk_metric": exit_summary["account_metrics"][
            "risk_budget"
        ]["60"],
        "common_boundary_d120_risk_metric": exit_summary["account_metrics"][
            "risk_budget"
        ]["120"],
        "comparison_note": (
            "The common-boundary Top10 metrics use the previously frozen 0.686089 "
            "gross fraction; each Top3 risk metric uses its own 2012-2019 "
            "volatility-derived fraction. Return comparisons are therefore "
            "risk-oriented, not identical-exposure comparisons."
        ),
    }
    decision = _decision(risk_metrics, risk_gates, strategy_rows)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed_adaptive_retrospective_top3_comparison",
        "study_id": STUDY_ID,
        "experiment_fingerprint": fingerprint,
        "study_sha256": base._sha256_file(frozen_path),
        "implementation_sha256": base._sha256_file(Path(__file__)),
        "source_selection_audit": source_selection_audit,
        "top3_book_audit": {
            "fixed": fixed_book_audit,
            "monthly": monthly_book_audit,
            "common_first_signal_date": str(dates[min(fixed_book.days)]),
            "common_last_signal_date": str(dates[max(fixed_book.days)]),
            "common_signal_count": len(fixed_book.days),
            "path_account_alignment_count": len(source_path_top3),
            "overlap_slot_requirements_without_deferred_exits": overlap_requirements,
        },
        "cohort_group_summaries": cohort_groups,
        "cohort_rank_summaries": cohort_ranks,
        "cohort_paired_summaries": cohort_paired,
        "full_account_metrics": full_metrics,
        "risk_budgets": risk_budgets,
        "risk_account_metrics": risk_metrics,
        "risk_account_gates": risk_gates,
        "prior_top10_reference": prior_top10,
        "factor_audit": factor_audit,
        "decision": decision,
        "forbidden_2026_read_count": 0,
        "files": files,
    }
    base._write_json(summary_path, summary)
    return summary


def validate_summary(path: str | Path) -> dict[str, Any]:
    summary = _read_json(base._resolve_path(path))
    checks = {
        "schema": summary.get("schema") == SUMMARY_SCHEMA,
        "study": summary.get("study_id") == STUDY_ID,
        "completed": summary.get("status")
        == "completed_adaptive_retrospective_top3_comparison",
        "no_2026": int(summary.get("forbidden_2026_read_count", -1)) == 0,
        "not_production": summary.get("decision", {}).get("production_claim_allowed")
        is False,
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
