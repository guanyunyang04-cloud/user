"""Expanded-development D20 residual policy with a causal margin-data gate."""

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

from daily_research.path_policy import seq100_hierarchical_residual_policy as hierarchy
from daily_research.path_policy import seq100_stock_bad_tail as bad_tail
from daily_research.path_policy import seq100_stock_distribution as base

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_margin_residual_policy_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_margin_residual_policy_v1"
)
STUDY_ID = "seq100_margin_residual_policy_v1"
SUMMARY_SCHEMA = "seq100_margin_residual_policy_summary/1"
FORECAST_SCHEMA = "seq100_margin_residual_policy_forecast/1"
HORIZON = 20
EXPANDED_DEVELOPMENT_YEARS = tuple(range(2014, 2023))
NEW_DEVELOPMENT_YEARS = (2014, 2015, 2016)
REUSED_DEVELOPMENT_YEARS = hierarchy.DEVELOPMENT_YEARS
CONFIRMATION_YEARS = hierarchy.CONFIRMATION_YEARS
PRIMARY_POLICY = "margin_observed_residual_veto10_industry_cap4_top48"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    base._write_json(path, payload)


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
    if source.get("expected_label_fingerprint") != bad_tail.EXPECTED_LABEL_FINGERPRINT:
        raise ValueError("label_fingerprint_contract_mismatch")
    if int(source.get("expected_row_count", -1)) != base.EXPECTED_ROW_COUNT:
        raise ValueError("row_count_contract_mismatch")
    evaluation = dict(study["evaluation"])
    if (
        tuple(int(value) for value in evaluation["expanded_development_years"])
        != EXPANDED_DEVELOPMENT_YEARS
    ):
        raise ValueError("expanded_development_year_contract_mismatch")
    if (
        tuple(int(value) for value in evaluation["new_untouched_development_years"])
        != NEW_DEVELOPMENT_YEARS
    ):
        raise ValueError("new_development_year_contract_mismatch")
    if (
        tuple(int(value) for value in evaluation["reused_development_years"])
        != REUSED_DEVELOPMENT_YEARS
    ):
        raise ValueError("reused_development_year_contract_mismatch")
    if (
        tuple(int(value) for value in evaluation["retrospective_confirmation_years"])
        != CONFIRMATION_YEARS
    ):
        raise ValueError("confirmation_year_contract_mismatch")
    if str(study["policy"]["primary"]) != PRIMARY_POLICY:
        raise ValueError("primary_policy_contract_mismatch")
    if bool(study["decision_boundary"].get("account_replay_performed", True)):
        raise ValueError("study_must_start_without_account_replay")
    return study, study_path


def _source_contract(study: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(study["source"])
    input_path = base._resolve_path(source["input_manifest"])
    label_path = base._resolve_path(source["label_manifest"])
    ready_path = base._resolve_path(source["training_ready_manifest"])
    frozen_path = base._resolve_path(source["frozen_2017_2022_summary"])
    if base._sha256_file(ready_path) != source["training_ready_manifest_sha256"]:
        raise ValueError("training_ready_manifest_hash_mismatch")
    if base._sha256_file(frozen_path) != source["frozen_2017_2022_summary_sha256"]:
        raise ValueError("frozen_forecast_summary_hash_mismatch")
    ready = _read_json(ready_path)
    frozen = _read_json(frozen_path)
    if ready.get("status") != "completed" or ready.get("study_id") != (
        "seq100_quality_liquidity_training_ready"
    ):
        raise ValueError("training_ready_manifest_invalid")
    margin = dict(ready["blocks"]["margin_features"])
    if margin.get("status") != "completed" or margin.get("lagged") is not True:
        raise ValueError("causal_margin_block_invalid")
    return {
        "input_manifest_path": input_path,
        "label_manifest_path": label_path,
        "training_ready_manifest_path": ready_path,
        "training_ready_manifest": ready,
        "margin_partitions": margin["partitions"],
        "frozen_summary_path": frozen_path,
        "frozen_summary": frozen,
        "frozen_paths": hierarchy._source_forecast_paths(frozen),
    }


def _stock_training_study(study: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "evaluation": {
            "maximum_train_rows_per_date": int(
                study["evaluation"]["maximum_train_rows_per_date"]
            )
        },
        "models": {"frozen_stock_residual": dict(study["stock_models"])},
    }


def _forecast_fingerprint(study_path: Path, year: int) -> str:
    return _payload_hash(
        {
            "study_sha256": base._sha256_file(study_path),
            "year": int(year),
            "horizon": HORIZON,
            "input_fingerprint": base.EXPECTED_INPUT_FINGERPRINT,
            "label_fingerprint": bad_tail.EXPECTED_LABEL_FINGERPRINT,
        }
    )


def _load_cached_forecast(
    panel: base.StockPanel,
    *,
    root: Path,
    study_path: Path,
    year: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    summary_path = root / f"y{year}" / "summary.json"
    if not summary_path.is_file():
        return None
    summary = _read_json(summary_path)
    if summary.get("schema") != FORECAST_SCHEMA or summary.get(
        "fingerprint"
    ) != _forecast_fingerprint(study_path, year):
        raise ValueError(f"cached_forecast_fingerprint_mismatch:{year}")
    record = dict(summary["file"])
    if not base._record_valid(record, verify_hash=True):
        raise ValueError(f"cached_forecast_file_invalid:{year}")
    residual, loss = hierarchy._load_frozen_stock_forecasts(
        panel, year, Path(str(record["path"]))
    )
    return residual, loss, summary


def _train_and_cache_forecast(
    panel: base.StockPanel,
    *,
    study: Mapping[str, Any],
    study_path: Path,
    year: int,
    root: Path,
    force: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not force:
        cached = _load_cached_forecast(
            panel, root=root, study_path=study_path, year=year
        )
        if cached is not None:
            return cached
    year_root = root / f"y{year}"
    residual, loss, fit = hierarchy._train_stock_heads(
        panel,
        study=_stock_training_study(study),
        evaluation_year=year,
        model_root=year_root / "models",
    )
    rows = panel.rows_for_year(year)
    identity = panel.row_index.iloc[rows]
    frame = identity[["candidate_id", "date_idx", "symbol_idx", "trade_date"]].copy()
    frame["ranking_score__lightgbm_mean_residual"] = residual.astype(np.float32)
    frame["loss_probability__lightgbm_binary"] = loss.astype(np.float32)
    forecast_path = year_root / "candidate_forecasts.parquet"
    _write_parquet(forecast_path, frame)
    summary = {
        "schema": FORECAST_SCHEMA,
        "status": "completed",
        "study_id": STUDY_ID,
        "evaluation_year": int(year),
        "fingerprint": _forecast_fingerprint(study_path, year),
        "fit": fit,
        "file": base._file_record(forecast_path),
        "training_performed": True,
        "hyperparameter_selection_performed": False,
        "forbidden_2026_read_count": 0,
    }
    _write_json(year_root / "summary.json", summary)
    return residual, loss, summary


def _margin_observed_mask(
    panel: base.StockPanel,
    *,
    year: int,
    partition_record: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    partition_path = Path(str(partition_record.get("path", "")))
    if (
        not partition_path.is_file()
        or base._sha256_file(partition_path) != partition_record.get("sha256")
    ):
        raise ValueError(f"margin_partition_invalid:{year}")
    columns = [
        "candidate_id",
        "trade_date",
        "date_idx",
        "symbol_idx",
        "margin_detail_coverage_state",
        "margin_detail_feature_available_date",
    ]
    frame = pd.read_parquet(partition_path, columns=columns)
    rows = panel.rows_for_year(year)
    identity = panel.row_index.iloc[rows]
    for column in ("candidate_id", "date_idx", "symbol_idx"):
        if not np.array_equal(frame[column].to_numpy(), identity[column].to_numpy()):
            raise ValueError(f"margin_identity_mismatch:{year}:{column}")
    trade_date = frame["trade_date"].astype(str)
    available = frame["margin_detail_feature_available_date"].astype("string")
    future = available.notna() & (available.astype(str) > trade_date)
    if bool(future.any()) or bool(trade_date.str.startswith("2026-").any()):
        raise ValueError(f"margin_point_in_time_contract_failed:{year}")
    observed = frame["margin_detail_coverage_state"].eq("observed").to_numpy()
    return observed, {
        "year": int(year),
        "row_count": len(frame),
        "observed_count": int(observed.sum()),
        "observed_fraction": float(observed.mean()),
        "maximum_feature_available_date": str(available.dropna().max()),
        "partition_sha256": str(partition_record["sha256"]),
    }


def _select_industry_capped(
    *,
    residual_score: np.ndarray,
    bad_tail_score: np.ndarray,
    candidate_id: np.ndarray,
    industry_code: np.ndarray,
    margin_observed: np.ndarray,
    top_k: int,
    veto_fraction: float,
    industry_cap: int,
) -> np.ndarray:
    count = len(candidate_id)
    if not (
        len(residual_score)
        == len(bad_tail_score)
        == len(industry_code)
        == len(margin_observed)
        == count
    ):
        raise ValueError("selection_length_mismatch")
    veto_count = min(math.ceil(count * float(veto_fraction)), count)
    risk_order = hierarchy._rank_order(bad_tail_score, candidate_id)
    allowed = np.asarray(margin_observed, dtype=bool).copy()
    allowed[risk_order[:veto_count]] = False
    positions = np.flatnonzero(allowed)
    order = hierarchy._rank_order(
        np.asarray(residual_score)[positions], np.asarray(candidate_id)[positions]
    )
    ranked = positions[order]
    selected: list[int] = []
    counts: dict[int, int] = {}
    for position in ranked:
        code = int(industry_code[position])
        if counts.get(code, 0) >= int(industry_cap):
            continue
        selected.append(int(position))
        counts[code] = counts.get(code, 0) + 1
        if len(selected) == int(top_k):
            break
    return np.asarray(selected, dtype=np.int64)


def _evaluate_year(
    panel: base.StockPanel,
    *,
    year: int,
    residual_score: np.ndarray,
    bad_tail_score: np.ndarray,
    margin_observed: np.ndarray,
    study: Mapping[str, Any],
) -> pd.DataFrame:
    rows = panel.rows_for_year(year)
    dates = panel.date_idx[rows]
    ids = panel.row_index.iloc[rows]["candidate_id"].to_numpy(dtype=np.int64)
    industries = panel.industry_code[rows]
    gross_full = panel.target(f"executable_log_return_{HORIZON}")
    residual_full = panel.target(f"residual_log_return_{HORIZON}")
    gross_valid = panel.target_valid(f"executable_log_return_{HORIZON}")[rows]
    residual_valid = panel.target_valid(f"residual_log_return_{HORIZON}")[rows]
    gross = gross_full[rows]
    residual = residual_full[rows]
    policy = dict(study["policy"])
    top_k = int(policy["top_k"])
    cost = float(study["target"]["cost_proxy"])
    horizon_column = base.HORIZONS.index(HORIZON) + 1
    within_cutoff = (
        panel.flags[rows, horizon_column] & base.FLAG_OUTCOME_WITHIN_CUTOFF
    ) != 0
    boundaries = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1, len(rows)]
    records: list[dict[str, Any]] = []
    for left, right in pairwise(boundaries):
        left_i, right_i = int(left), int(right)
        local = slice(left_i, right_i)
        selected_local = _select_industry_capped(
            residual_score=np.asarray(residual_score)[local],
            bad_tail_score=np.asarray(bad_tail_score)[local],
            candidate_id=ids[local],
            industry_code=industries[local],
            margin_observed=np.asarray(margin_observed)[local],
            top_k=top_k,
            veto_fraction=float(policy["bad_tail_veto_fraction"]),
            industry_cap=int(policy["maximum_names_per_pit_industry"]),
        )
        selected = left_i + selected_local
        valid_gross = gross_valid[selected]
        valid_residual = residual_valid[selected]
        gross_simple = np.expm1(gross[selected][valid_gross])
        residual_simple = np.expm1(residual[selected][valid_residual])
        universe_valid = gross_valid[local]
        universe_simple = np.expm1(gross[local][universe_valid])
        selected_mean = (
            float(np.mean(gross_simple)) if len(gross_simple) else float("nan")
        )
        universe_mean = (
            float(np.mean(universe_simple)) if len(universe_simple) else float("nan")
        )
        records.append(
            {
                "date_idx": int(dates[left_i]),
                "trade_date": str(panel.trade_date[rows[left_i]]),
                "evaluation_year": int(year),
                "policy": PRIMARY_POLICY,
                "calendar_evaluable": bool(np.all(within_cutoff[local])),
                "candidate_count": right_i - left_i,
                "margin_observed_candidate_count": int(
                    np.asarray(margin_observed)[local].sum()
                ),
                "selected_count": len(selected),
                "gross_observed_count": int(valid_gross.sum()),
                "residual_observed_count": int(valid_residual.sum()),
                "observed_fraction": (
                    float(valid_gross.mean()) if len(selected) else 1.0
                ),
                "gross_net_stress": (
                    float(np.sum(gross_simple)) - cost * int(valid_gross.sum())
                )
                / top_k,
                "hedged_residual_net_stress": (
                    float(np.sum(residual_simple)) - cost * int(valid_residual.sum())
                )
                / top_k,
                "selected_mean_gross": selected_mean,
                "universe_mean_gross": universe_mean,
                "selected_excess_gross": (
                    selected_mean - universe_mean
                    if np.isfinite(selected_mean) and np.isfinite(universe_mean)
                    else float("nan")
                ),
            }
        )
    result = pd.DataFrame(records)
    if bool(result["trade_date"].str.startswith("2026-").any()):
        raise ValueError("forbidden_2026_evaluation_row")
    return result


def _summary(daily: pd.DataFrame, study: Mapping[str, Any]) -> dict[str, Any]:
    selected = daily.loc[daily["calendar_evaluable"]].copy()
    inference = dict(study["evaluation"]["inference"])
    stress = selected["gross_net_stress"].to_numpy(dtype=np.float64)
    excess = selected["selected_excess_gross"].to_numpy(dtype=np.float64)
    residual = selected["hedged_residual_net_stress"].to_numpy(dtype=np.float64)
    annual = (
        selected.groupby("evaluation_year", sort=True, as_index=False)
        .agg(
            gross_net_stress=("gross_net_stress", "mean"),
            hedged_residual_net_stress=("hedged_residual_net_stress", "mean"),
            selected_excess_gross=("selected_excess_gross", "mean"),
            mean_selected_count=("selected_count", "mean"),
            date_count=("date_idx", "size"),
        )
        .to_dict(orient="records")
    )
    return {
        "policy": PRIMARY_POLICY,
        "date_count": len(selected),
        "mean_selected_count": float(selected["selected_count"].mean()),
        "minimum_selected_count": int(selected["selected_count"].min()),
        "observed_outcome_fraction": float(
            selected["gross_observed_count"].sum()
            / max(selected["selected_count"].sum(), 1)
        ),
        "gross_net_stress": {
            **base._hac_mean(stress, lag=int(inference["hac_lag"])),
            "block": base._block_interval(
                stress,
                block_length=int(inference["moving_block_length"]),
                repetitions=int(inference["bootstrap_repetitions"]),
                seed=int(inference["seed"]),
            ),
        },
        "hedged_residual_net_stress": base._hac_mean(
            residual, lag=int(inference["hac_lag"])
        ),
        "selected_excess_gross": {
            **base._hac_mean(excess, lag=int(inference["hac_lag"])),
            "block": base._block_interval(
                excess,
                block_length=int(inference["moving_block_length"]),
                repetitions=int(inference["bootstrap_repetitions"]),
                seed=int(inference["seed"]) + 1,
            ),
        },
        "annual": annual,
    }


def _gate(summary: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    positive_years = sum(
        float(row["gross_net_stress"]) > 0 for row in summary["annual"]
    )
    checks = {
        "stress_net_hac_lcb_positive": float(summary["gross_net_stress"]["lcb_95"]) > 0,
        "stress_net_block_lcb_positive": float(
            summary["gross_net_stress"]["block"]["lcb_95"]
        )
        > 0,
        "minimum_positive_years": positive_years
        >= int(contract["minimum_positive_years"]),
        "minimum_mean_selected_count": float(summary["mean_selected_count"])
        >= float(contract.get("minimum_mean_selected_count", 0)),
        "minimum_observed_outcome_fraction": float(summary["observed_outcome_fraction"])
        >= float(contract["minimum_observed_outcome_fraction"]),
    }
    if contract.get("selected_excess_hac_lower_bound_strictly_positive"):
        checks["selected_excess_hac_lcb_positive"] = (
            float(summary["selected_excess_gross"]["lcb_95"]) > 0
        )
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "positive_year_count": positive_years,
        "required_positive_year_count": int(contract["minimum_positive_years"]),
    }


def _phase_complete(
    path: Path, *, study_sha256: str, years: Sequence[int]
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    summary = _read_json(path)
    if (
        summary.get("schema") != SUMMARY_SCHEMA
        or summary.get("study_sha256") != study_sha256
        or tuple(int(value) for value in summary.get("evaluation_years", ()))
        != tuple(int(value) for value in years)
    ):
        raise ValueError(f"existing_phase_contract_mismatch:{path.parent.name}")
    if not all(
        base._record_valid(record, verify_hash=True)
        for record in dict(summary.get("files", {}) or {}).values()
    ):
        raise ValueError(f"existing_phase_file_invalid:{path.parent.name}")
    return summary


def run_phase(
    *,
    panel: base.StockPanel,
    study: Mapping[str, Any],
    study_path: Path,
    source: Mapping[str, Any],
    output_root: Path,
    phase: str,
    force: bool = False,
) -> dict[str, Any]:
    if phase not in {"development", "confirmation"}:
        raise ValueError(f"unknown_phase:{phase}")
    years = EXPANDED_DEVELOPMENT_YEARS if phase == "development" else CONFIRMATION_YEARS
    phase_root = output_root / phase
    study_sha256 = base._sha256_file(study_path)
    summary_path = phase_root / "summary.json"
    if not force:
        current = _phase_complete(summary_path, study_sha256=study_sha256, years=years)
        if current is not None:
            return current
    daily_frames: list[pd.DataFrame] = []
    margin_audits: list[dict[str, Any]] = []
    forecast_records: list[dict[str, Any]] = []
    for year in years:
        if phase == "development" and year in REUSED_DEVELOPMENT_YEARS:
            residual, loss = hierarchy._load_frozen_stock_forecasts(
                panel, year, source["frozen_paths"][year]
            )
            forecast_record: dict[str, Any] = {
                "evaluation_year": year,
                "source_reused": True,
                "path": str(source["frozen_paths"][year]),
            }
        else:
            residual, loss, trained = _train_and_cache_forecast(
                panel,
                study=study,
                study_path=study_path,
                year=year,
                root=phase_root / "stock_forecasts",
                force=force,
            )
            forecast_record = {
                "evaluation_year": year,
                "source_reused": False,
                "summary": trained,
            }
        partition = dict(source["margin_partitions"][str(year)])
        margin_observed, margin_audit = _margin_observed_mask(
            panel, year=year, partition_record=partition
        )
        daily_frames.append(
            _evaluate_year(
                panel,
                year=year,
                residual_score=residual,
                bad_tail_score=loss,
                margin_observed=margin_observed,
                study=study,
            )
        )
        margin_audits.append(margin_audit)
        forecast_records.append(forecast_record)
    daily = pd.concat(daily_frames, ignore_index=True)
    policy_summary = _summary(daily, study)
    contract = dict(study["evaluation"][f"{phase}_gate"])
    gate = _gate(policy_summary, contract)
    daily_path = phase_root / "policy_daily.parquet"
    audit_path = phase_root / "margin_audit.parquet"
    forecasts_path = phase_root / "forecast_records.json"
    _write_parquet(daily_path, daily)
    _write_parquet(audit_path, pd.DataFrame(margin_audits))
    _write_json(forecasts_path, {"forecasts": forecast_records})
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "study_id": STUDY_ID,
        "phase": phase,
        "study_sha256": study_sha256,
        "phase_fingerprint": _payload_hash(
            {
                "study_sha256": study_sha256,
                "phase": phase,
                "years": list(years),
                "input_fingerprint": base.EXPECTED_INPUT_FINGERPRINT,
                "label_fingerprint": bad_tail.EXPECTED_LABEL_FINGERPRINT,
            }
        ),
        "evaluation_years": list(years),
        "primary_policy": PRIMARY_POLICY,
        "policy_summary": policy_summary,
        "gate": gate,
        "eligible_for_next_phase": bool(gate["passed"]),
        "new_development_forecasts_trained": (
            list(NEW_DEVELOPMENT_YEARS) if phase == "development" else []
        ),
        "retrospective_confirmation_consumed": phase == "confirmation",
        "candidate_selection_precedes_outcome_masks": True,
        "account_replay_performed": False,
        "portfolio_optimization_performed": False,
        "forbidden_2026_read_count": 0,
        "profit_claim_allowed": False,
        "files": {
            "policy_daily": base._file_record(daily_path),
            "margin_audit": base._file_record(audit_path),
            "forecast_records": base._file_record(forecasts_path),
        },
    }
    _write_json(summary_path, summary)
    return summary


def validate_summary(path: str | Path) -> dict[str, Any]:
    summary = _read_json(base._resolve_path(path))
    checks = {
        "schema": summary.get("schema") == SUMMARY_SCHEMA,
        "study": summary.get("study_id") == STUDY_ID,
        "selection_before_outcomes": summary.get(
            "candidate_selection_precedes_outcome_masks"
        )
        is True,
        "no_account": summary.get("account_replay_performed") is False,
        "no_2026": int(summary.get("forbidden_2026_read_count", -1)) == 0,
        "files": all(
            base._record_valid(record, verify_hash=True)
            for record in dict(summary.get("files", {}) or {}).values()
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"summary_validation_failed:{checks}")
    return {"status": "ok", "checks": checks}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--phase", choices=("development", "confirmation", "all"), default="all"
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    study, study_path = load_study(args.study)
    source = _source_contract(study)
    panel = base.load_panel(
        input_manifest_path=source["input_manifest_path"],
        label_manifest_path=source["label_manifest_path"],
    )
    if bool((panel.years == base.FORBIDDEN_YEAR).any()):
        raise ValueError("forbidden_2026_panel_row")
    output_root = base._resolve_path(args.output_root)
    development: dict[str, Any] | None = None
    if args.phase in {"development", "all"}:
        development = run_phase(
            panel=panel,
            study=study,
            study_path=study_path,
            source=source,
            output_root=output_root,
            phase="development",
            force=args.force,
        )
        print(json.dumps(development["gate"], ensure_ascii=False, indent=2))
    if args.phase in {"confirmation", "all"}:
        if development is None:
            development = _phase_complete(
                output_root / "development" / "summary.json",
                study_sha256=base._sha256_file(study_path),
                years=EXPANDED_DEVELOPMENT_YEARS,
            )
        if development is None or not bool(development["gate"]["passed"]):
            print("confirmation_not_authorized:expanded_development_gate_failed")
            return
        confirmation = run_phase(
            panel=panel,
            study=study,
            study_path=study_path,
            source=source,
            output_root=output_root,
            phase="confirmation",
            force=args.force,
        )
        print(json.dumps(confirmation["gate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
