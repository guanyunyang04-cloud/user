"""Causal same-day adaptive-line cross study inside margin-balance Top10."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_margin_top10_adaptive_entry as base
from daily_research.path_policy import seq100_stock_distribution as common
from daily_research.path_policy.seq100_exit_policy_audit import (
    CandidateCompleteAuditPack,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_margin_top10_current_cross_v1"
SUMMARY_SCHEMA = "seq100_margin_top10_current_cross_summary/1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_margin_top10_current_cross_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_margin_top10_current_cross_v1"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _payload_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def load_study(
    path: str | Path = DEFAULT_STUDY_PATH,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    study_path = common._resolve_path(path)
    study = _read_json(study_path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("study_id_mismatch")
    if study.get("epistemic_contract") != {
        "hypothesis_was_specified_after_the_prior_margin_top10_result": True,
        "all_2012_2025_results_are_adaptive_retrospective_evidence": True,
        "no_2026_outcome_may_be_read": True,
        "production_claim_allowed": False,
    }:
        raise ValueError("epistemic_contract_mismatch")
    if study.get("selection") != {
        "margin_balance_field": "rzye",
        "minimum_consecutive_increases": 2,
        "consecutive_source_trading_days_required": True,
        "daily_rank": "descending absolute CNY margin-balance increment",
        "daily_top_k": 10,
        "universe": "quality_liquidity_pit on the signal date",
        "margin_and_kline_alignment": "same source trading date",
        "availability_gate": "same-date margin data must have feature_available_date equal to the next trading session",
        "signal_information_time": "after the exchange publishes signal-date margin data before the next session opens",
        "entry": "next legal raw-price open on feature_available_date",
    }:
        raise ValueError("selection_contract_mismatch")
    source = dict(study["source"])
    for key in ("base_study", "base_implementation", "prior_summary"):
        source_path = common._resolve_path(source[key])
        if (
            not source_path.is_file()
            or common._sha256_file(source_path).lower()
            != str(source[f"{key}_sha256"]).lower()
        ):
            raise ValueError(f"source_invalid:{key}")
    base_study, _ = base.load_study(source["base_study"])
    if int(study["source"]["forbidden_year"]) != common.FORBIDDEN_YEAR:
        raise ValueError("forbidden_year_contract_mismatch")
    if tuple(study["evaluation"]["full_years"]) != tuple(range(2012, 2026)):
        raise ValueError("evaluation_year_contract_mismatch")
    if bool(study["decision_boundary"]["production_claim_allowed"]):
        raise ValueError("production_claim_forbidden")
    return study, study_path, base_study


def _attach_current_cross(
    candidates: pd.DataFrame,
    *,
    pack: CandidateCompleteAuditPack,
    factor: np.ndarray,
    adp: np.ndarray,
) -> pd.DataFrame:
    result = candidates.reset_index(drop=True).copy()
    current_cross = np.zeros(len(result), dtype=bool)
    current_cross_rising = np.zeros(len(result), dtype=bool)
    prior_close_to_adp = np.full(len(result), np.nan, dtype=np.float64)
    for symbol_idx, group in result.groupby("symbol_idx", sort=False):
        symbol_idx = int(symbol_idx)
        maximum_idx = int(group["date_idx"].max())
        raw_close = np.asarray(
            pack.exit_close_raw[: maximum_idx + 1, symbol_idx], dtype=np.float64
        )
        factors = np.asarray(factor[: maximum_idx + 1, symbol_idx], dtype=np.float64)
        line = np.asarray(adp[: maximum_idx + 1, symbol_idx], dtype=np.float64)
        valid_dates = np.flatnonzero(
            np.isfinite(raw_close)
            & (raw_close > 0.0)
            & np.isfinite(factors)
            & (factors > 0.0)
            & np.isfinite(line)
        )
        if not len(valid_dates):
            continue
        candidate_dates = group["date_idx"].to_numpy(dtype=np.int64)
        positions = np.searchsorted(valid_dates, candidate_dates)
        usable = (positions > 0) & (positions < len(valid_dates))
        usable &= (
            valid_dates[np.minimum(positions, len(valid_dates) - 1)] == candidate_dates
        )
        for row_position, date_idx, position, is_usable in zip(
            group.index.to_numpy(dtype=np.int64),
            candidate_dates,
            positions,
            usable,
            strict=True,
        ):
            if not is_usable:
                continue
            date_idx = int(date_idx)
            prior_idx = int(valid_dates[int(position) - 1])
            current_close = float(raw_close[date_idx] * factors[date_idx])
            prior_close = float(raw_close[prior_idx] * factors[prior_idx])
            current_line = float(line[date_idx])
            prior_line = float(line[prior_idx])
            crossed = current_close > current_line and prior_close <= prior_line
            current_cross[row_position] = crossed
            current_cross_rising[row_position] = crossed and current_line > prior_line
            prior_close_to_adp[row_position] = prior_close / prior_line - 1.0
    result["current_cross"] = current_cross
    result["current_cross_rising"] = current_cross_rising
    result["prior_close_to_adp"] = prior_close_to_adp
    return result


def _signal_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "all_margin_top10_min2": np.ones(len(frame), dtype=bool),
        "current_cross": frame["current_cross"].to_numpy(dtype=bool),
        "current_cross_rising": frame["current_cross_rising"].to_numpy(dtype=bool),
        "prior_cross_support_control": frame["cross_support"].to_numpy(dtype=bool),
    }


def _periods(study: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    return {
        "full_history": tuple(
            int(value) for value in study["evaluation"]["full_years"]
        ),
        **{
            str(name): tuple(int(value) for value in values)
            for name, values in study["evaluation"]["periods"].items()
        },
    }


def _daily_summary(frame: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    selected = frame.loc[mask].copy()
    selected["next_up_observed"] = selected["signal_close_return_d1"].notna()
    selected["entry_day_observed"] = selected["entry_to_close_return_d1"].notna()
    selected["next_up_value"] = selected["next_close_up"].where(
        selected["next_up_observed"]
    )
    selected["entry_day_up_value"] = (selected["entry_to_close_return_d1"] > 0.0).where(
        selected["entry_day_observed"]
    )
    selected["cash_net_return"] = selected["one_day_net_return"].fillna(0.0)
    return (
        selected.groupby(["date_idx", "trade_date", "evaluation_year"], as_index=False)
        .agg(
            selected_count=("candidate_id", "size"),
            filled_count=("one_day_net_return", "count"),
            next_up_fraction=("next_up_value", "mean"),
            entry_day_up_fraction=("entry_day_up_value", "mean"),
            entry_day_gross_return=("entry_to_close_return_d1", "mean"),
            legal_d2_gross_return=("one_day_gross_adjusted_return", "mean"),
            legal_d2_cash_net_return=("cash_net_return", "mean"),
        )
        .sort_values("date_idx", kind="stable")
    )


def _summaries(
    frame: pd.DataFrame, study: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    masks = _signal_masks(frame)
    baseline = _daily_summary(frame, masks["all_margin_top10_min2"]).rename(
        columns={
            "next_up_fraction": "baseline_next_up_fraction",
            "entry_day_up_fraction": "baseline_entry_day_up_fraction",
            "entry_day_gross_return": "baseline_entry_day_gross_return",
            "legal_d2_gross_return": "baseline_legal_d2_gross_return",
            "legal_d2_cash_net_return": "baseline_legal_d2_cash_net_return",
        }
    )
    baseline_columns = [
        "date_idx",
        "baseline_next_up_fraction",
        "baseline_entry_day_up_fraction",
        "baseline_entry_day_gross_return",
        "baseline_legal_d2_gross_return",
        "baseline_legal_d2_cash_net_return",
    ]
    records: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    seed_add = 0
    for rule, mask in masks.items():
        daily = _daily_summary(frame, mask).merge(
            baseline[baseline_columns],
            on="date_idx",
            how="left",
            validate="one_to_one",
        )
        for name in (
            "next_up_fraction",
            "entry_day_up_fraction",
            "entry_day_gross_return",
            "legal_d2_gross_return",
            "legal_d2_cash_net_return",
        ):
            daily[f"paired_{name}"] = daily[name] - daily[f"baseline_{name}"]
        daily["rule"] = rule
        daily_frames.append(daily)
        for period, years in _periods(study).items():
            local_daily = daily.loc[daily["evaluation_year"].isin(set(years))]
            local = frame.loc[
                mask & frame["evaluation_year"].isin(set(years)).to_numpy(dtype=bool)
            ]
            observed_next = local.loc[local["signal_close_return_d1"].notna()]
            observed_entry = local.loc[local["entry_to_close_return_d1"].notna()]
            observed_legal = local.loc[local["one_day_net_return"].notna()]
            if local_daily.empty:
                continue
            annual = local_daily.groupby("evaluation_year")[
                "legal_d2_cash_net_return"
            ].mean()
            records.append(
                {
                    "period": period,
                    "rule": rule,
                    "selected_candidate_count": len(local),
                    "signal_date_count": len(local_daily),
                    "filled_candidate_count": len(observed_legal),
                    "next_close_up_fraction": float(
                        observed_next["next_close_up"].mean()
                    ),
                    "entry_day_gross_up_fraction": float(
                        (observed_entry["entry_to_close_return_d1"] > 0.0).mean()
                    ),
                    "entry_day_gross_mean": float(
                        observed_entry["entry_to_close_return_d1"].mean()
                    ),
                    "legal_d2_gross_mean": float(
                        observed_legal["one_day_gross_adjusted_return"].mean()
                    ),
                    "legal_d2_net_mean": float(
                        observed_legal["one_day_net_return"].mean()
                    ),
                    "legal_d2_net_median": float(
                        observed_legal["one_day_net_return"].median()
                    ),
                    "legal_d2_net_winning_fraction": float(
                        (observed_legal["one_day_net_return"] > 0.0).mean()
                    ),
                    "entry_to_close_path_means": {
                        f"d{horizon}": float(
                            local[f"entry_to_close_return_d{horizon}"].mean()
                        )
                        for horizon in study["outcomes"]["path_horizons_from_signal"]
                    },
                    "entry_to_close_path_medians": {
                        f"d{horizon}": float(
                            local[f"entry_to_close_return_d{horizon}"].median()
                        )
                        for horizon in study["outcomes"]["path_horizons_from_signal"]
                    },
                    "daily_next_up_fraction": base._inference(
                        local_daily["next_up_fraction"].to_numpy(),
                        study,
                        seed_add=seed_add,
                    ),
                    "paired_next_up_vs_all_top10": base._inference(
                        local_daily["paired_next_up_fraction"].to_numpy(),
                        study,
                        seed_add=seed_add + 1000,
                    ),
                    "daily_legal_d2_cash_net_return": base._inference(
                        local_daily["legal_d2_cash_net_return"].to_numpy(),
                        study,
                        seed_add=seed_add + 2000,
                    ),
                    "paired_legal_net_vs_all_top10": base._inference(
                        local_daily["paired_legal_d2_cash_net_return"].to_numpy(),
                        study,
                        seed_add=seed_add + 3000,
                    ),
                    "positive_net_year_count": int((annual > 0.0).sum()),
                    "year_count": len(annual),
                }
            )
            seed_add += 1
    return records, pd.concat(daily_frames, ignore_index=True)


def run_study(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    study, frozen_path, base_study = load_study(study_path)
    root = common._resolve_path(output_root)
    summary_path = root / "summary.json"
    fingerprint = _payload_hash(
        {
            "study_sha256": common._sha256_file(frozen_path),
            "implementation_sha256": common._sha256_file(Path(__file__)),
            "base_study_sha256": study["source"]["base_study_sha256"],
            "base_implementation_sha256": study["source"]["base_implementation_sha256"],
            "prior_summary_sha256": study["source"]["prior_summary_sha256"],
        }
    )
    if summary_path.is_file() and not force:
        current = _read_json(summary_path)
        if current.get("experiment_fingerprint") != fingerprint:
            raise ValueError("existing_output_fingerprint_mismatch")
        if all(
            common._record_valid(record, verify_hash=True)
            for record in current.get("files", {}).values()
        ):
            return current

    sources = base._source_contract(base_study)
    pack = CandidateCompleteAuditPack(sources["pack_manifest"])
    qdp_root = Path(str(pack.manifest["qdp_root"])).resolve()
    selection_study = copy.deepcopy(base_study)
    selection_study["margin_selection"]["minimum_consecutive_increases"] = int(
        study["selection"]["minimum_consecutive_increases"]
    )
    candidates, timeline, selection_audit = base._margin_top10(
        pool_paths=base._pool_paths(sources["quality_liquidity_pit_manifest"]),
        margin_paths=base._qdp_shard_paths(
            sources["margin_detail_manifest"], qdp_root=qdp_root
        ),
        pack=pack,
        study=selection_study,
    )
    factor_fields, factor_audit = base._load_dense_fields(
        sources["adjust_factor_manifest"],
        pack=pack,
        fields=["adjust_factor"],
        maximum_date=str(study["source"]["maximum_outcome_date"]),
    )
    market_fields, market_audit = base._load_dense_fields(
        sources["market_daily_manifest"],
        pack=pack,
        fields=["high", "low"],
        maximum_date=str(study["source"]["maximum_outcome_date"]),
    )
    factor = factor_fields["adjust_factor"]
    cutoff = int(pack.date_to_idx[str(study["source"]["maximum_outcome_date"])])
    adp = base._adaptive_line(
        pack.exit_close_raw,
        factor,
        symbol_indices=candidates["symbol_idx"].unique(),
        maximum_date_idx=cutoff,
        study=base_study,
    )
    candidates = base._attach_technical(
        candidates,
        pack=pack,
        factor=factor,
        high_raw=market_fields["high"],
        low_raw=market_fields["low"],
        adp=adp,
    )
    candidates = _attach_current_cross(candidates, pack=pack, factor=factor, adp=adp)
    candidates, decrease_audit = base._attach_paths_and_exits(
        candidates,
        timeline,
        pack=pack,
        factor=factor,
        high_raw=market_fields["high"],
        low_raw=market_fields["low"],
        study=selection_study,
    )
    summaries, daily = _summaries(candidates, study)

    root.mkdir(parents=True, exist_ok=True)
    candidate_path = root / "candidates.parquet"
    daily_path = root / "daily_summaries.parquet"
    base._write_parquet(candidate_path, candidates)
    base._write_parquet(daily_path, daily)
    files = {
        "candidates": common._file_record(candidate_path),
        "daily_summaries": common._file_record(daily_path),
    }
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed_adaptive_retrospective_current_cross_study",
        "study_id": STUDY_ID,
        "experiment_fingerprint": fingerprint,
        "study_sha256": common._sha256_file(frozen_path),
        "implementation_sha256": common._sha256_file(Path(__file__)),
        "selection_audit": selection_audit,
        "factor_audit": factor_audit,
        "market_field_audit": market_audit,
        "decrease_exit_audit": decrease_audit,
        "signal_counts": {
            name: int(mask.sum()) for name, mask in _signal_masks(candidates).items()
        },
        "summaries": summaries,
        "decision": {
            "adaptive_retrospective_only": True,
            "account_replay_allowed": False,
            "production_claim_allowed": False,
            "forward_confirmation_required": True,
        },
        "forbidden_2026_read_count": 0,
        "files": files,
    }
    common._write_json(summary_path, summary)
    return summary


def validate_summary(path: str | Path) -> dict[str, Any]:
    summary = _read_json(common._resolve_path(path))
    checks = {
        "schema": summary.get("schema") == SUMMARY_SCHEMA,
        "study": summary.get("study_id") == STUDY_ID,
        "completed": summary.get("status")
        == "completed_adaptive_retrospective_current_cross_study",
        "no_2026": int(summary.get("forbidden_2026_read_count", -1)) == 0,
        "not_production": summary.get("decision", {}).get("production_claim_allowed")
        is False,
        "files": all(
            common._record_valid(record, verify_hash=True)
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
