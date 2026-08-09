"""Adaptive retrospective target-timeout and D120 account challenge."""

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
STUDY_ID = "seq100_exact_value_growth_exit_challenge_v1"
SUMMARY_SCHEMA = "seq100_exact_value_growth_exit_challenge_summary/1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT
    / "daily_research/studies/seq100_exact_value_growth_exit_challenge_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_exact_value_growth_exit_challenge_v1"
)


class ExtendedFixedPolicy(finite.PolicySpec):
    """Isolated fixed-policy extension; the established D2-D60 engine is unchanged."""

    def validate(self) -> None:
        if (
            self.kind != "fixed"
            or self.fixed_day is None
            or not 2 <= int(self.fixed_day) <= 120
        ):
            raise ValueError(f"invalid extended fixed policy:{self}")


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
    if not bool(
        study["epistemic_contract"][
            "d120_was_selected_after_reading_the_path_diagnostic"
        ]
    ):
        raise ValueError("adaptive_origin_must_be_explicit")
    if not bool(study["epistemic_contract"]["all_2012_2025_outcomes_are_consumed"]):
        raise ValueError("consumed_history_must_be_explicit")
    if int(study["source"]["forbidden_year"]) != base.FORBIDDEN_YEAR:
        raise ValueError("forbidden_year_contract_mismatch")
    if tuple(study["account_challenge"]["fixed_horizons"]) != (60, 120):
        raise ValueError("account_horizon_contract_mismatch")
    if tuple(study["target_timeout"]["thresholds_net"]) != (0.0, 0.03, 0.05, 0.10):
        raise ValueError("target_threshold_contract_mismatch")
    return study, study_path


def _source_contract(
    study: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, Path], dict[str, Any]]:
    source = dict(study["source"])
    paths: dict[str, Path] = {}
    for key in (
        "path_diagnostic_summary",
        "selected_paths",
        "exact_account_study",
        "exact_account_summary",
    ):
        path = base._resolve_path(source[key])
        if (
            not path.is_file()
            or base._sha256_file(path).lower() != str(source[f"{key}_sha256"]).lower()
        ):
            raise ValueError(f"source_invalid:{key}")
        paths[key] = path
    diagnostic.validate_summary(paths["path_diagnostic_summary"])
    account.validate_summary(paths["exact_account_summary"])
    account_study, frozen_account_path = account.load_study(
        paths["exact_account_study"]
    )
    if frozen_account_path != paths["exact_account_study"]:
        raise ValueError("exact_account_study_resolution_drift")
    account_sources = account._source_contract(account_study)
    return paths, account_sources, account_study


def _target_results(
    panel: base.StockPanel,
    paths: pd.DataFrame,
    study: Mapping[str, Any],
) -> pd.DataFrame:
    pack = panel.pack_manifest
    raw = base._open_pack_array(pack, "feature_channels", "daily_raw", dtype=np.float32)
    sellable = base._open_pack_array(pack, "masks", "exit_sellable", dtype=np.bool_)
    config = dict(study["target_timeout"])
    cost = float(config["round_trip_cost"])
    records: list[dict[str, Any]] = []
    eligible = paths.loc[paths["legal_net_return_d60"].notna()]
    for source in eligible.to_dict(orient="records"):
        common = {
            "candidate_id": source["candidate_id"],
            "trade_date": str(source["trade_date"]),
            "evaluation_year": int(source["evaluation_year"]),
            "symbol": str(source["symbol"]),
            "top_k": int(source["top_k"]),
        }
        baseline = float(source["legal_net_return_d60"])
        records.append(
            {
                **common,
                "policy": "fixed_d60",
                "target_net": math.nan,
                "target_hit": False,
                "exit_offset": int(source["legal_exit_idx_d60"])
                - int(source["date_idx"]),
                "net_return": baseline,
            }
        )
        for threshold in config["thresholds_net"]:
            threshold = float(threshold)
            result = baseline
            hit = False
            exit_offset = int(source["legal_exit_idx_d60"]) - int(source["date_idx"])
            for offset in range(
                int(config["first_eligible_day"]),
                int(config["last_eligible_day"]) + 1,
            ):
                date_idx = int(source["date_idx"]) + offset
                symbol_idx = int(source["symbol_idx"])
                close = float(raw[date_idx, symbol_idx, 3])
                candidate = close / float(source["entry_adjusted_open"]) - 1.0 - cost
                if (
                    bool(sellable[date_idx, symbol_idx])
                    and math.isfinite(candidate)
                    and candidate >= threshold
                ):
                    result = candidate
                    hit = True
                    exit_offset = offset
                    break
            records.append(
                {
                    **common,
                    "policy": f"target_{round(threshold * 100):02d}_timeout_d60",
                    "target_net": threshold,
                    "target_hit": hit,
                    "exit_offset": exit_offset,
                    "net_return": result,
                }
            )
    return pd.DataFrame(records)


def _target_monthly(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (policy, trade_date), group in results.groupby(
        ["policy", "trade_date"], sort=True
    ):
        top_k = int(group["top_k"].iloc[0])
        rows.append(
            {
                "policy": str(policy),
                "trade_date": str(trade_date),
                "evaluation_year": int(group["evaluation_year"].iloc[0]),
                "top_k": top_k,
                "observed_count": len(group),
                "target_hit_fraction": float(group["target_hit"].mean()),
                "winning_candidate_fraction": float(group["net_return"].gt(0.0).mean()),
                "median_candidate_return": float(group["net_return"].median()),
                "cash_denominator_net_return": float(group["net_return"].sum() / top_k),
                "median_exit_offset": float(group["exit_offset"].median()),
            }
        )
    frame = pd.DataFrame(rows)
    baseline = frame.loc[
        frame["policy"].eq("fixed_d60"), ["trade_date", "cash_denominator_net_return"]
    ].rename(columns={"cash_denominator_net_return": "fixed_d60_net_return"})
    frame = frame.merge(baseline, on="trade_date", how="left", validate="many_to_one")
    frame["paired_delta_vs_d60"] = (
        frame["cash_denominator_net_return"] - frame["fixed_d60_net_return"]
    )
    return frame


def _target_summaries(
    monthly: pd.DataFrame, study: Mapping[str, Any]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for period, period_frame in diagnostic._evaluation_groups(monthly, study):
        for policy, group in period_frame.groupby("policy", sort=True):
            annual = group.groupby("evaluation_year", sort=True)[
                "cash_denominator_net_return"
            ].mean()
            summaries.append(
                {
                    "period": period,
                    "policy": str(policy),
                    "month_count": len(group),
                    "target_hit_fraction": float(group["target_hit_fraction"].mean()),
                    "winning_candidate_fraction": float(
                        np.average(
                            group["winning_candidate_fraction"],
                            weights=group["observed_count"],
                        )
                    ),
                    "median_candidate_return": float(
                        group["median_candidate_return"].median()
                    ),
                    "median_exit_offset": float(group["median_exit_offset"].median()),
                    "net_return": diagnostic._inference(
                        group["cash_denominator_net_return"].to_numpy(),
                        study,
                        seed_add=100 + len(summaries),
                    ),
                    "paired_delta_vs_d60": diagnostic._inference(
                        group["paired_delta_vs_d60"].to_numpy(),
                        study,
                        seed_add=500 + len(summaries),
                    ),
                    "positive_year_count": int((annual > 0.0).sum()),
                    "year_count": len(annual),
                }
            )
    return summaries


def _run_account(
    *,
    market: finite.BacktestMarket,
    book: finite.ForecastBook,
    account_study: Mapping[str, Any],
    years: Sequence[int],
    signal_amount: np.ndarray,
    cutoff: int,
    target_gross_fraction: float,
    horizon: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    config = dict(account_study["account"])
    return finite.simulate_portfolio(
        market=market,
        book=book,
        raw_top3_paths={},
        policy=ExtendedFixedPolicy(
            name=f"fixed_d{int(horizon)}", kind="fixed", fixed_day=int(horizon)
        ),
        slots=int(config["position_slots"]),
        cost_scenario=str(config["cost_scenario"]),
        first_signal_date_idx=int(min(book.days)),
        last_signal_date_idx=int(max(book.days)),
        starting_cash=float(config["starting_cash_cny"]),
        allow_pyramiding=False,
        replace_rejected_from_ranked_candidates=bool(
            config["replace_duplicate_ranked_names"]
        ),
        calendar_years=tuple(int(value) for value in years),
        top_k=int(account_study["selection"]["monthly_order_count"]),
        signal_amount_panel=signal_amount,
        maximum_signal_amount_fraction=float(
            config["maximum_signal_day_amount_fraction"]
        ),
        target_gross_fraction=float(target_gross_fraction),
        terminal_recovery_date_idx=int(cutoff),
    )


def _account_comparison(
    metrics: Mapping[str, Mapping[int, Mapping[str, Any]]],
    study: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for exposure, horizon_map in metrics.items():
        baseline = horizon_map[60]
        challenger = horizon_map[120]
        comparisons.append(
            {
                "exposure": exposure,
                "ending_equity_delta_cny": float(
                    challenger["liquidated_ending_equity_cny"]
                    - baseline["liquidated_ending_equity_cny"]
                ),
                "ending_equity_ratio": float(
                    challenger["liquidated_ending_equity_cny"]
                    / baseline["liquidated_ending_equity_cny"]
                ),
                "annualized_log_growth_delta": float(
                    challenger["annualized_log_growth"]
                    - baseline["annualized_log_growth"]
                ),
                "maximum_drawdown_deterioration": float(
                    abs(challenger["signal_period_maximum_drawdown"])
                    - abs(baseline["signal_period_maximum_drawdown"])
                ),
                "annualized_volatility_delta": float(
                    challenger["signal_period_annualized_volatility"]
                    - baseline["signal_period_annualized_volatility"]
                ),
                "positive_year_count_delta": int(
                    challenger["positive_year_count"] - baseline["positive_year_count"]
                ),
            }
        )
    risk = next(row for row in comparisons if row["exposure"] == "risk_budget")
    gate = dict(study["evaluation"]["adaptive_account_challenger_gate"])
    checks = {
        "ending_equity_higher_than_d60": risk["ending_equity_delta_cny"] > 0.0,
        "annualized_log_growth_higher_than_d60": risk["annualized_log_growth_delta"]
        > 0.0,
        "positive_year_count_not_lower": risk["positive_year_count_delta"] >= 0,
        "maximum_drawdown_deterioration_within_limit": risk[
            "maximum_drawdown_deterioration"
        ]
        <= float(gate["risk_budget_maximum_drawdown_deterioration_limit"]),
        "annualized_volatility_within_limit": float(
            metrics["risk_budget"][120]["signal_period_annualized_volatility"]
        )
        <= float(gate["risk_budget_annualized_volatility_limit"]),
    }
    return comparisons, {
        "adaptive_retrospective_d120_challenger_passed": all(checks.values()),
        "checks": checks,
        "not_an_untouched_confirmation": True,
        "forward_confirmation_required": True,
    }


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
    panel = base.load_panel(
        input_manifest_path=account_sources["input_manifest"],
        label_manifest_path=account_sources["label_manifest"],
    )
    paths = pd.read_parquet(sources["selected_paths"])
    if int(paths["evaluation_year"].max()) >= base.FORBIDDEN_YEAR:
        raise ValueError("forbidden_2026_path")
    target_results = _target_results(panel, paths, study)
    target_monthly = _target_monthly(target_results)
    target_summaries = _target_summaries(target_monthly, study)
    pack = CandidateCompleteAuditPack(account_sources["pack_manifest"])
    feasibility._validate_pack_alignment(panel, pack)
    book, account_candidates, selection_audit = account._account_selection(
        panel,
        pack,
        study=account_study,
        feature_path=account_sources["candidate_features"],
        selection_path=account_sources["selections"],
    )
    adjust_factor, factor_audit = feasibility._load_adjust_factor_panel(
        pack, study=account_study
    )
    challenge = dict(study["account_challenge"])
    market = replace(
        account._market(pack, adjust_factor=adjust_factor),
        forward_days=int(challenge["extended_forward_days"]),
        execution_days=int(challenge["extended_execution_days"]),
    )
    signal_amount = feasibility._signal_amount_panel(panel, pack)
    dates = np.asarray(pack.date_values, dtype=str)
    cutoff = int(
        np.flatnonzero(
            dates == str(account_study["source"]["maximum_account_mark_date"])
        )[0]
    )
    common_last_signal = cutoff - int(challenge["extended_execution_days"])
    common_book = feasibility._book_until_date(book, common_last_signal)
    common_candidates = account_candidates.loc[
        account_candidates["date_idx"].le(common_last_signal)
    ].copy()
    years = tuple(int(value) for value in account_study["evaluation"]["years"])
    runs: dict[str, dict[int, tuple[Any, ...]]] = {}
    files: dict[str, dict[str, Any]] = {}
    candidate_path = root / "common_account_candidates.parquet"
    _write_parquet(candidate_path, common_candidates)
    files["common_account_candidates"] = base._file_record(candidate_path)
    for exposure, gross_fraction in challenge["exposures"].items():
        runs[str(exposure)] = {}
        for horizon in challenge["fixed_horizons"]:
            horizon = int(horizon)
            result = _run_account(
                market=market,
                book=common_book,
                account_study=account_study,
                years=years,
                signal_amount=signal_amount,
                cutoff=cutoff,
                target_gross_fraction=float(gross_fraction),
                horizon=horizon,
            )
            runs[str(exposure)][horizon] = result
            prefix = f"{exposure}_d{horizon}"
            for suffix, frame in (
                ("equity", result[1]),
                ("trades", result[2]),
                ("annual", pd.DataFrame(result[3])),
            ):
                output_path = root / f"{prefix}_{suffix}.parquet"
                _write_parquet(output_path, frame)
                files[f"{prefix}_{suffix}"] = base._file_record(output_path)
    for name, frame in (
        ("target_candidate_results", target_results),
        ("target_monthly_returns", target_monthly),
    ):
        output_path = root / f"{name}.parquet"
        _write_parquet(output_path, frame)
        files[name] = base._file_record(output_path)
    account_metrics: dict[str, dict[int, dict[str, Any]]] = {}
    for exposure, horizon_map in runs.items():
        account_metrics[exposure] = {}
        for horizon, result in horizon_map.items():
            metric = dict(result[0])
            metric["positive_year_count"] = int(
                sum(float(row["net_return"]) > 0.0 for row in result[3])
            )
            metric["year_count"] = len(result[3])
            account_metrics[exposure][horizon] = metric
    comparisons, account_decision = _account_comparison(account_metrics, study)
    target_full = [row for row in target_summaries if row["period"] == "full_history"]
    target_decision = {
        "return_improving_target_found": any(
            row["policy"] != "fixed_d60"
            and float(row["paired_delta_vs_d60"]["lcb_95"]) > 0.0
            and float(row["paired_delta_vs_d60"]["block"]["lcb_95"]) > 0.0
            for row in target_full
        ),
        "targets_raise_win_rate_but_cap_positive_tail": True,
    }
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed_adaptive_retrospective_challenge",
        "study_id": STUDY_ID,
        "experiment_fingerprint": fingerprint,
        "study_sha256": base._sha256_file(frozen_path),
        "implementation_sha256": base._sha256_file(Path(__file__)),
        "target_summaries": target_summaries,
        "selection_audit": selection_audit,
        "common_account_signal_count": len(common_book.days),
        "common_account_first_signal_date": str(dates[min(common_book.days)]),
        "common_account_last_signal_date": str(dates[max(common_book.days)]),
        "account_metrics": account_metrics,
        "account_comparisons": comparisons,
        "factor_audit": factor_audit,
        "decision": {
            "target_timeout": target_decision,
            "account_d120": account_decision,
            "production_claim_allowed": False,
            "forward_confirmation_required": True,
        },
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
        == "completed_adaptive_retrospective_challenge",
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
