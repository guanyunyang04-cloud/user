"""Posthoc robustness of the D60 value account to transparent factor definitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_causal_value_policy as causal
from daily_research.path_policy import seq100_stock_distribution as base
from daily_research.path_policy import seq100_value_staggered_account as account

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_value_definition_robustness_v1"
SUMMARY_SCHEMA = "seq100_value_definition_robustness_summary/1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_value_definition_robustness_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_value_definition_robustness_v1"
)
VARIANTS = (
    "pe_pb_fcf_equal",
    "pe_pb_equal",
    "earnings_yield_only",
    "book_to_price_only",
    "fcf_yield_only",
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
    if tuple(study.get("variants", ())) != VARIANTS:
        raise ValueError("variant_contract_mismatch")
    source = dict(study["source"])
    if source.get("expected_input_fingerprint") != base.EXPECTED_INPUT_FINGERPRINT:
        raise ValueError("input_fingerprint_contract_mismatch")
    if int(source.get("expected_row_count", -1)) != base.EXPECTED_ROW_COUNT:
        raise ValueError("input_row_count_contract_mismatch")
    if int(source.get("forbidden_year", -1)) != base.FORBIDDEN_YEAR:
        raise ValueError("forbidden_year_contract_mismatch")
    serialized = json.dumps(study, ensure_ascii=False).lower()
    if "52_week_low_or_distance_from_low_reward_is_forbidden" not in serialized:
        raise ValueError("low_price_reward_exclusion_missing")
    return study, study_path


def _source_contract(study: Mapping[str, Any]) -> dict[str, Path]:
    source = dict(study["source"])
    paths: dict[str, Path] = {}
    for key in ("causal_value_summary", "candidate_features", "parent_selections"):
        path = base._resolve_path(source[key])
        expected = str(source[f"{key}_sha256"]).lower()
        if not path.is_file() or base._sha256_file(path).lower() != expected:
            raise ValueError(f"source_invalid:{key}")
        paths[key] = path
    paths["input_manifest"] = base._resolve_path(source["input_manifest"])
    paths["label_manifest"] = base._resolve_path(source["label_manifest"])
    causal.validate_summary(paths["causal_value_summary"])
    return paths


def _variant_selections(
    features: pd.DataFrame,
    *,
    top_k: int,
    industry_cap: int,
    minimum_group_size: int,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    dates = features["date_idx"].to_numpy(dtype=np.int64)
    boundaries = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1, len(features)]
    for left, right in pairwise(boundaries):
        local = features.iloc[int(left) : int(right)]
        industry = local["industry_code"].to_numpy(dtype=np.int64)
        pe_log = local["signed_log_pe"].to_numpy(dtype=float)
        pb_log = local["signed_log_pb"].to_numpy(dtype=float)
        log_market_value = local["log_total_market_value"].to_numpy(dtype=float)
        fcf = local["cashflow_free_cash_flow"].to_numpy(dtype=float)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            market_value = np.expm1(np.clip(log_market_value, 0.0, 50.0))
            fcf_yield = fcf / market_value
        pe_score = causal._industry_percentile(
            np.where(pe_log > 0.0, -pe_log, np.nan),
            industry,
            minimum_group_size=minimum_group_size,
        )
        pb_score = causal._industry_percentile(
            np.where(pb_log > 0.0, -pb_log, np.nan),
            industry,
            minimum_group_size=minimum_group_size,
        )
        fcf_score = causal._industry_percentile(
            fcf_yield,
            industry,
            minimum_group_size=minimum_group_size,
        )
        positive_both = (pe_log > 0.0) & (pb_log > 0.0)
        scores = {
            "pe_pb_fcf_equal": local["value_score"].to_numpy(dtype=float),
            "pe_pb_equal": causal._family_mean((pe_score, pb_score), minimum_count=2),
            "earnings_yield_only": pe_score,
            "book_to_price_only": pb_score,
            "fcf_yield_only": fcf_score,
        }
        eligibility = {
            "pe_pb_fcf_equal": np.isfinite(scores["pe_pb_fcf_equal"]),
            "pe_pb_equal": positive_both & np.isfinite(scores["pe_pb_equal"]),
            "earnings_yield_only": (pe_log > 0.0) & np.isfinite(pe_score),
            "book_to_price_only": (pb_log > 0.0) & np.isfinite(pb_score),
            "fcf_yield_only": positive_both & np.isfinite(fcf_score),
        }
        for variant in VARIANTS:
            chosen = causal._select_industry_capped(
                score=scores[variant],
                eligible=eligibility[variant],
                candidate_id=local["candidate_id"].to_numpy(dtype=np.int64),
                industry_code=industry,
                top_k=top_k,
                industry_cap=industry_cap,
            )
            selected = local.iloc[chosen][
                [
                    "row_position",
                    "candidate_id",
                    "date_idx",
                    "trade_date",
                    "symbol",
                    "evaluation_year",
                    "industry_code",
                ]
            ].copy()
            selected.insert(0, "variant", variant)
            selected["variant_score"] = scores[variant][chosen]
            selected["selection_rank"] = np.arange(1, len(chosen) + 1)
            records.append(selected)
    return pd.concat(records, ignore_index=True)


def _reproduction_audit(selections: pd.DataFrame, parent_path: Path) -> dict[str, Any]:
    current = selections.loc[selections["variant"].eq("pe_pb_fcf_equal")][
        ["date_idx", "selection_rank", "candidate_id"]
    ].sort_values(["date_idx", "selection_rank"])
    parent = pd.read_parquet(
        parent_path, columns=["policy", "date_idx", "selection_rank", "candidate_id"]
    )
    parent = parent.loc[parent["policy"].eq("value_only")][
        ["date_idx", "selection_rank", "candidate_id"]
    ].sort_values(["date_idx", "selection_rank"])
    exact = len(current) == len(parent) and np.array_equal(
        current.to_numpy(), parent.to_numpy()
    )
    if not exact:
        raise ValueError("parent_value_selection_reproduction_failed")
    return {"row_count": len(current), "exact_candidate_and_rank_match": True}


def _attach_outcomes(
    panel: base.StockPanel,
    features: pd.DataFrame,
    selections: pd.DataFrame,
) -> pd.DataFrame:
    rows = features["row_position"].to_numpy(dtype=np.int64)
    outcome = causal._d60_timeout(panel, rows, retry_days=20)
    outcome_frame = pd.DataFrame(
        {
            "row_position": rows,
            "calendar_evaluable": np.asarray(outcome["within"], dtype=bool),
            "entry_filled": np.asarray(outcome["entry_valid"], dtype=bool),
            "outcome_valid": np.asarray(outcome["valid"], dtype=bool),
            "entry_price": np.asarray(outcome["entry_price"], dtype=float),
            "exit_offset": np.asarray(outcome["exit_offset"], dtype=np.int16),
            "simple_return": np.asarray(outcome["simple_return"], dtype=float),
            "pack_date_idx": np.asarray(outcome["date_idx"], dtype=np.int64),
            "pack_symbol_idx": np.asarray(outcome["symbol_idx"], dtype=np.int64),
        }
    )
    ledger = selections.merge(
        outcome_frame, on="row_position", how="left", validate="many_to_one"
    )
    ledger["extended_exit"] = False
    ledger["terminal_writeoff"] = False
    trapped = (
        ledger["calendar_evaluable"] & ledger["entry_filled"] & ~ledger["outcome_valid"]
    )
    if bool(trapped.any()):
        account._resolve_censored_positions(panel, ledger, trapped.to_numpy(dtype=bool))
    unresolved = (
        ledger["calendar_evaluable"] & ledger["entry_filled"] & ~ledger["outcome_valid"]
    )
    if bool(unresolved.any()):
        raise ValueError("unresolved_selected_position")
    ledger["applied_cost"] = np.where(
        ledger["outcome_valid"],
        np.where(ledger["terminal_writeoff"], 0.003, 0.006),
        0.0,
    )
    return ledger


def _decision(metrics: list[Mapping[str, Any]]) -> dict[str, Any]:
    profitable = [float(row["terminal_multiple"]) > 1.0 for row in metrics]
    recent_positive = []
    for row in metrics:
        annual = pd.DataFrame(row["annual"])
        recent = annual.loc[annual["year"] >= 2023, "return"]
        recent_positive.append(bool(len(recent) and (recent.mean() > 0.0)))
    return {
        "all_value_definitions_profitable": bool(all(profitable)),
        "all_value_definitions_positive_in_2023_2025_mean": bool(all(recent_positive)),
        "retrospective_robustness_passed": bool(
            all(profitable) and all(recent_positive)
        ),
        "forward_confirmation_required": True,
        "production_strategy_claim_allowed": False,
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
    fingerprint = _payload_hash(
        {
            "study_sha256": base._sha256_file(frozen_path),
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
            for record in dict(current.get("files", {})).values()
        ):
            return current
    panel = base.load_panel(
        input_manifest_path=sources["input_manifest"],
        label_manifest_path=sources["label_manifest"],
    )
    features = pd.read_parquet(sources["candidate_features"])
    selection_contract = dict(study["selection"])
    selections = _variant_selections(
        features,
        top_k=int(selection_contract["top_k_before_account_breadth"]),
        industry_cap=int(selection_contract["maximum_names_per_pit_industry"]),
        minimum_group_size=int(selection_contract["minimum_industry_rank_group_size"]),
    )
    reproduction = _reproduction_audit(selections, sources["parent_selections"])
    ledger = _attach_outcomes(panel, features, selections)
    account_contract = dict(study["account"])
    curves: list[pd.DataFrame] = []
    cohorts: list[pd.DataFrame] = []
    cycles: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    for variant in VARIANTS:
        variant_ledger = ledger.loc[ledger["variant"].eq(variant)].copy()
        chosen, cohort = account._cohort_table(
            variant_ledger,
            breadth=int(selection_contract["account_breadth"]),
            sleeve_count=int(account_contract["sleeve_count"]),
            cost=float(account_contract["round_trip_cost_proxy"]),
        )
        curve, metric, cycle = account._simulate_account(
            panel,
            chosen,
            breadth=int(selection_contract["account_breadth"]),
            sleeve_count=int(account_contract["sleeve_count"]),
            initial_cash=float(account_contract["initial_cash"]),
            cost=float(account_contract["round_trip_cost_proxy"]),
        )
        curve.insert(0, "variant", variant)
        cohort.insert(0, "variant", variant)
        cycle.insert(0, "variant", variant)
        metric["variant"] = variant
        curves.append(curve)
        cohorts.append(cohort)
        cycles.append(cycle)
        metrics.append(metric)
    selection_path = root / "variant_selections.parquet"
    ledger_path = root / "selected_outcomes.parquet"
    curves_path = root / "daily_account_curves.parquet"
    cohorts_path = root / "cohort_returns.parquet"
    cycles_path = root / "sleeve_cycles.parquet"
    _write_parquet(selection_path, selections)
    _write_parquet(ledger_path, ledger)
    _write_parquet(curves_path, pd.concat(curves, ignore_index=True))
    _write_parquet(cohorts_path, pd.concat(cohorts, ignore_index=True))
    _write_parquet(cycles_path, pd.concat(cycles, ignore_index=True))
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "study_id": STUDY_ID,
        "experiment_fingerprint": fingerprint,
        "study_sha256": base._sha256_file(frozen_path),
        "parent_reproduction": reproduction,
        "account_metrics": metrics,
        "decision": _decision(metrics),
        "posthoc_robustness_test": True,
        "52_week_low_reward_used": False,
        "forbidden_2026_read_count": 0,
        "round_lots_and_minimum_commission_modeled": False,
        "production_strategy_claim_allowed": False,
        "files": {
            "variant_selections": base._file_record(selection_path),
            "selected_outcomes": base._file_record(ledger_path),
            "daily_account_curves": base._file_record(curves_path),
            "cohort_returns": base._file_record(cohorts_path),
            "sleeve_cycles": base._file_record(cycles_path),
        },
    }
    base._write_json(summary_path, summary)
    return summary


def validate_summary(path: str | Path) -> dict[str, Any]:
    summary = _read_json(base._resolve_path(path))
    checks = {
        "schema": summary.get("schema") == SUMMARY_SCHEMA,
        "study": summary.get("study_id") == STUDY_ID,
        "reproduced": summary.get("parent_reproduction", {}).get(
            "exact_candidate_and_rank_match"
        )
        is True,
        "posthoc": summary.get("posthoc_robustness_test") is True,
        "no_low_price_reward": summary.get("52_week_low_reward_used") is False,
        "no_2026": int(summary.get("forbidden_2026_read_count", -1)) == 0,
        "not_production": summary.get("production_strategy_claim_allowed") is False,
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
