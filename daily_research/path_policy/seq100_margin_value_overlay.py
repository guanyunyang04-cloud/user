"""Overlay a transparent PIT value score on the frozen D20 residual policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_causal_value_policy as value
from daily_research.path_policy import seq100_hierarchical_residual_policy as hierarchy
from daily_research.path_policy import seq100_margin_residual_policy as margin
from daily_research.path_policy import seq100_stock_distribution as base

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_margin_value_overlay_v1"
SUMMARY_SCHEMA = "seq100_margin_value_overlay_summary/1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_margin_value_overlay_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_margin_value_overlay_v1"
)
POLICIES = (
    "frozen_residual_baseline",
    "residual_with_value_median_gate",
    "equal_rank_residual_value_blend",
)
VALUE_FEATURES = (
    "signed_log_pe",
    "signed_log_pb",
    "log_total_market_value",
    "cashflow_free_cash_flow",
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
    source = dict(study["source"])
    if source.get("expected_input_fingerprint") != base.EXPECTED_INPUT_FINGERPRINT:
        raise ValueError("input_fingerprint_contract_mismatch")
    if int(source.get("expected_row_count", -1)) != base.EXPECTED_ROW_COUNT:
        raise ValueError("input_row_count_contract_mismatch")
    if int(source.get("forbidden_year", -1)) != base.FORBIDDEN_YEAR:
        raise ValueError("forbidden_year_contract_mismatch")
    if tuple(study["policies"].get("variants", ())) != POLICIES:
        raise ValueError("policy_contract_mismatch")
    excluded = " ".join(study["value"]["explicitly_excluded"]).lower()
    if "52_week_low" not in excluded or "distance_from_52_week_low" not in excluded:
        raise ValueError("low_price_reward_exclusion_missing")
    if bool(study["decision_boundary"].get("account_replay_performed", True)):
        raise ValueError("study_must_start_without_account_replay")
    return study, study_path


def _source_contract(
    study: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    source = dict(study["source"])
    for key in ("margin_study", "development_summary", "confirmation_summary"):
        path = base._resolve_path(source[key])
        expected = str(source[f"{key}_sha256"]).lower()
        if not path.is_file() or base._sha256_file(path).lower() != expected:
            raise ValueError(f"source_invalid:{key}")
    margin_study, margin_path = margin.load_study(source["margin_study"])
    margin_source = margin._source_contract(margin_study)
    return margin_study, margin_path, margin_source


def _load_forecasts(
    panel: base.StockPanel,
    *,
    year: int,
    margin_path: Path,
    margin_source: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if int(year) in margin.REUSED_DEVELOPMENT_YEARS:
        return hierarchy._load_frozen_stock_forecasts(
            panel, int(year), margin_source["frozen_paths"][int(year)]
        )
    phase = "development" if int(year) <= 2022 else "confirmation"
    cached = margin._load_cached_forecast(
        panel,
        root=margin.DEFAULT_OUTPUT_ROOT / phase / "stock_forecasts",
        study_path=margin_path,
        year=int(year),
    )
    if cached is None:
        raise ValueError(f"frozen_forecast_missing:{year}")
    residual, loss, _ = cached
    return residual, loss


def _value_score(
    panel: base.StockPanel,
    rows: np.ndarray,
    *,
    minimum_group_size: int,
) -> np.ndarray:
    positions = base._feature_positions(panel, VALUE_FEATURES)
    matrix = base._feature_matrix(panel, rows, positions).astype(np.float64)
    pe_log, pb_log, log_market_value, fcf = matrix.T
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        market_value = np.expm1(np.clip(log_market_value, 0.0, 50.0))
        fcf_yield = fcf / market_value
    result = np.full(len(rows), np.nan, dtype=np.float64)
    dates = panel.date_idx[rows]
    industries = panel.industry_code[rows]
    boundaries = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1, len(rows)]
    for left, right in pairwise(boundaries):
        local = slice(int(left), int(right))
        industry = industries[local]
        pe = pe_log[local]
        pb = pb_log[local]
        parts = (
            value._industry_percentile(
                np.where(pe > 0.0, -pe, np.nan),
                industry,
                minimum_group_size=minimum_group_size,
            ),
            value._industry_percentile(
                np.where(pb > 0.0, -pb, np.nan),
                industry,
                minimum_group_size=minimum_group_size,
            ),
            value._industry_percentile(
                fcf_yield[local],
                industry,
                minimum_group_size=minimum_group_size,
            ),
        )
        local_score = value._family_mean(parts, minimum_count=2)
        local_score[~((pe > 0.0) & (pb > 0.0))] = np.nan
        result[local] = local_score
    return result


def _variant_selection(
    *,
    policy: str,
    residual_score: np.ndarray,
    bad_tail_score: np.ndarray,
    value_score: np.ndarray,
    candidate_id: np.ndarray,
    industry_code: np.ndarray,
    margin_observed: np.ndarray,
    top_k: int,
    veto_fraction: float,
    industry_cap: int,
) -> np.ndarray:
    if policy == "frozen_residual_baseline":
        ranking = np.asarray(residual_score, dtype=np.float64)
        eligible = np.asarray(margin_observed, dtype=bool)
    elif policy == "residual_with_value_median_gate":
        ranking = np.asarray(residual_score, dtype=np.float64)
        eligible = (
            np.asarray(margin_observed, dtype=bool)
            & np.isfinite(value_score)
            & (np.asarray(value_score) >= 0.5)
        )
    elif policy == "equal_rank_residual_value_blend":
        residual_rank = value._rank_percentile(residual_score)
        ranking = value._family_mean(
            (residual_rank, np.asarray(value_score, dtype=np.float64)),
            minimum_count=2,
        )
        eligible = np.asarray(margin_observed, dtype=bool) & np.isfinite(ranking)
    else:
        raise ValueError(f"unknown_policy:{policy}")
    return margin._select_industry_capped(
        residual_score=ranking,
        bad_tail_score=bad_tail_score,
        candidate_id=candidate_id,
        industry_code=industry_code,
        margin_observed=eligible,
        top_k=top_k,
        veto_fraction=veto_fraction,
        industry_cap=industry_cap,
    )


def _evaluate_year(
    panel: base.StockPanel,
    *,
    year: int,
    residual_score: np.ndarray,
    bad_tail_score: np.ndarray,
    margin_observed: np.ndarray,
    value_score: np.ndarray,
    study: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = panel.rows_for_year(year)
    dates = panel.date_idx[rows]
    ids = panel.row_index.iloc[rows]["candidate_id"].to_numpy(dtype=np.int64)
    symbols = panel.row_index.iloc[rows]["symbol"].astype(str).to_numpy()
    industries = panel.industry_code[rows]
    gross_full = panel.target("executable_log_return_20")
    residual_full = panel.target("residual_log_return_20")
    gross_valid = panel.target_valid("executable_log_return_20")[rows]
    residual_valid = panel.target_valid("residual_log_return_20")[rows]
    gross = gross_full[rows]
    residual = residual_full[rows]
    target = dict(study["target"])
    policies = dict(study["policies"])
    top_k = int(policies["top_k"])
    cost = float(target["round_trip_cost_proxy"])
    within_cutoff = (
        panel.flags[rows, base.HORIZONS.index(20) + 1] & base.FLAG_OUTCOME_WITHIN_CUTOFF
    ) != 0
    boundaries = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1, len(rows)]
    daily_records: list[dict[str, Any]] = []
    selection_records: list[dict[str, Any]] = []
    for left, right in pairwise(boundaries):
        left_i, right_i = int(left), int(right)
        local = slice(left_i, right_i)
        universe_good = gross_valid[local]
        universe_simple = np.expm1(gross[local][universe_good])
        universe_mean = (
            float(universe_simple.mean()) if len(universe_simple) else np.nan
        )
        for policy in POLICIES:
            chosen_local = _variant_selection(
                policy=policy,
                residual_score=np.asarray(residual_score)[local],
                bad_tail_score=np.asarray(bad_tail_score)[local],
                value_score=np.asarray(value_score)[local],
                candidate_id=ids[local],
                industry_code=industries[local],
                margin_observed=np.asarray(margin_observed)[local],
                top_k=top_k,
                veto_fraction=float(policies["bad_tail_veto_fraction"]),
                industry_cap=int(policies["maximum_names_per_pit_industry"]),
            )
            chosen = left_i + chosen_local
            valid_gross = gross_valid[chosen]
            valid_residual = residual_valid[chosen]
            gross_simple = np.expm1(gross[chosen][valid_gross])
            residual_simple = np.expm1(residual[chosen][valid_residual])
            selected_mean = float(gross_simple.mean()) if len(gross_simple) else np.nan
            daily_records.append(
                {
                    "policy": policy,
                    "date_idx": int(dates[left_i]),
                    "trade_date": str(panel.trade_date[rows[left_i]]),
                    "evaluation_year": int(year),
                    "calendar_evaluable": bool(np.all(within_cutoff[local])),
                    "candidate_count": right_i - left_i,
                    "selected_count": len(chosen),
                    "observed_count": int(valid_gross.sum()),
                    "stress_net_return": (
                        float(gross_simple.sum()) - cost * int(valid_gross.sum())
                    )
                    / top_k,
                    "selected_mean_gross": selected_mean,
                    "universe_mean_gross": universe_mean,
                    "selected_excess_gross": (
                        selected_mean - universe_mean
                        if np.isfinite(selected_mean) and np.isfinite(universe_mean)
                        else np.nan
                    ),
                    "industry_residual_mean": (
                        float(residual_simple.mean())
                        if len(residual_simple)
                        else np.nan
                    ),
                }
            )
            for rank, position in enumerate(chosen, start=1):
                selection_records.append(
                    {
                        "policy": policy,
                        "date_idx": int(dates[left_i]),
                        "trade_date": str(panel.trade_date[rows[left_i]]),
                        "evaluation_year": int(year),
                        "selection_rank": rank,
                        "candidate_id": int(ids[position]),
                        "symbol": str(symbols[position]),
                        "industry_code": int(industries[position]),
                        "value_score": float(value_score[position]),
                        "frozen_residual_score": float(residual_score[position]),
                    }
                )
    return pd.DataFrame(daily_records), pd.DataFrame(selection_records)


def _period_name(year: int, periods: Mapping[str, Sequence[int]]) -> str:
    for name, years in periods.items():
        if int(year) in {int(value) for value in years}:
            return str(name)
    raise ValueError(f"year_not_in_period_contract:{year}")


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


def _summaries(daily: pd.DataFrame, study: Mapping[str, Any]) -> list[dict[str, Any]]:
    periods = dict(study["evaluation"]["periods"])
    inference = dict(study["evaluation"]["inference"])
    selected = daily.loc[daily["calendar_evaluable"]].copy()
    selected["period"] = selected["evaluation_year"].map(
        lambda year: _period_name(int(year), periods)
    )
    results: list[dict[str, Any]] = []
    for (period, policy), group in selected.groupby(["period", "policy"], sort=True):
        annual_frame = group.groupby("evaluation_year", sort=True, as_index=False).agg(
            stress_net_return=("stress_net_return", "mean"),
            selected_excess_gross=("selected_excess_gross", "mean"),
            industry_residual_mean=("industry_residual_mean", "mean"),
            mean_selected_count=("selected_count", "mean"),
            date_count=("date_idx", "size"),
        )
        arguments = {
            "lag": int(inference["hac_lag_open_days"]),
            "block_length": int(inference["moving_block_length_open_days"]),
            "repetitions": int(inference["bootstrap_repetitions"]),
        }
        results.append(
            {
                "period": period,
                "policy": policy,
                "date_count": len(group),
                "mean_selected_count": float(group["selected_count"].mean()),
                "minimum_selected_count": int(group["selected_count"].min()),
                "observed_fraction": float(
                    group["observed_count"].sum()
                    / max(group["selected_count"].sum(), 1)
                ),
                "stress_net": _inference(
                    group["stress_net_return"].to_numpy(dtype=float),
                    seed=int(inference["seed"]) + len(results),
                    **arguments,
                ),
                "selected_excess_gross": _inference(
                    group["selected_excess_gross"].to_numpy(dtype=float),
                    seed=int(inference["seed"]) + len(results) + 1000,
                    **arguments,
                ),
                "industry_residual": _inference(
                    group["industry_residual_mean"].to_numpy(dtype=float),
                    seed=int(inference["seed"]) + len(results) + 2000,
                    **arguments,
                ),
                "positive_year_count": int(
                    (annual_frame["stress_net_return"] > 0.0).sum()
                ),
                "year_count": len(annual_frame),
                "annual": annual_frame.to_dict(orient="records"),
            }
        )
    return results


def _paired_summaries(
    daily: pd.DataFrame, study: Mapping[str, Any]
) -> list[dict[str, Any]]:
    selected = daily.loc[daily["calendar_evaluable"]].copy()
    baseline = selected.loc[selected["policy"].eq(POLICIES[0])][
        ["date_idx", "evaluation_year", "stress_net_return"]
    ].rename(columns={"stress_net_return": "baseline_return"})
    challengers = selected.loc[~selected["policy"].eq(POLICIES[0])].merge(
        baseline,
        on=["date_idx", "evaluation_year"],
        how="inner",
        validate="many_to_one",
    )
    challengers["paired_delta"] = (
        challengers["stress_net_return"] - challengers["baseline_return"]
    )
    periods = dict(study["evaluation"]["periods"])
    challengers["period"] = challengers["evaluation_year"].map(
        lambda year: _period_name(int(year), periods)
    )
    inference = dict(study["evaluation"]["inference"])
    results: list[dict[str, Any]] = []
    for (period, policy), group in challengers.groupby(["period", "policy"], sort=True):
        results.append(
            {
                "period": period,
                "policy": policy,
                "paired_delta_vs_frozen_baseline": _inference(
                    group["paired_delta"].to_numpy(dtype=float),
                    lag=int(inference["hac_lag_open_days"]),
                    block_length=int(inference["moving_block_length_open_days"]),
                    repetitions=int(inference["bootstrap_repetitions"]),
                    seed=int(inference["seed"]) + len(results) + 3000,
                ),
            }
        )
    return results


def _decision(
    summaries: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    study: Mapping[str, Any],
) -> dict[str, Any]:
    primary = str(study["policies"]["primary_challenger"])
    summary_map = {(row["period"], row["policy"]): row for row in summaries}
    checks: list[dict[str, Any]] = []
    for row in paired:
        if row["policy"] != primary:
            continue
        absolute = summary_map[(row["period"], primary)]
        delta = row["paired_delta_vs_frozen_baseline"]
        checks.append(
            {
                "period": row["period"],
                "absolute_stress_hac_lcb_positive": float(
                    absolute["stress_net"]["lcb_95"]
                )
                > 0.0,
                "absolute_stress_block_lcb_positive": float(
                    absolute["stress_net"]["block"]["lcb_95"]
                )
                > 0.0,
                "paired_improvement_hac_lcb_positive": float(delta["lcb_95"]) > 0.0,
                "paired_improvement_block_lcb_positive": float(delta["block"]["lcb_95"])
                > 0.0,
            }
        )
    return {
        "primary_challenger": primary,
        "historical_overlay_passed": bool(
            checks
            and all(
                all(value for key, value in row.items() if key.endswith("positive"))
                for row in checks
            )
        ),
        "checks": checks,
        "account_replay_authorized": False,
        "reason": "overlay years are retrospectively consumed; a forward-only alpha claim is not allowed",
    }


def run_study(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    study, frozen_path = load_study(study_path)
    _margin_study, margin_path, margin_source = _source_contract(study)
    root = base._resolve_path(output_root)
    fingerprint = _payload_hash(
        {
            "study_sha256": base._sha256_file(frozen_path),
            "margin_study_sha256": base._sha256_file(margin_path),
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
        input_manifest_path=margin_source["input_manifest_path"],
        label_manifest_path=margin_source["label_manifest_path"],
    )
    periods = dict(study["evaluation"]["periods"])
    years = sorted({int(year) for values in periods.values() for year in values})
    daily_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []
    margin_audits: list[dict[str, Any]] = []
    for year in years:
        rows = panel.rows_for_year(year)
        residual, loss = _load_forecasts(
            panel,
            year=year,
            margin_path=margin_path,
            margin_source=margin_source,
        )
        observed, audit = margin._margin_observed_mask(
            panel,
            year=year,
            partition_record=margin_source["margin_partitions"][str(year)],
        )
        score = _value_score(
            panel,
            rows,
            minimum_group_size=int(study["value"]["minimum_industry_rank_group_size"]),
        )
        daily, selections = _evaluate_year(
            panel,
            year=year,
            residual_score=residual,
            bad_tail_score=loss,
            margin_observed=observed,
            value_score=score,
            study=study,
        )
        daily_frames.append(daily)
        selection_frames.append(selections)
        margin_audits.append(audit)
    daily = pd.concat(daily_frames, ignore_index=True)
    selections = pd.concat(selection_frames, ignore_index=True)
    audit_frame = pd.DataFrame(margin_audits)
    daily_path = root / "policy_daily.parquet"
    selections_path = root / "selections.parquet"
    audit_path = root / "margin_audit.parquet"
    _write_parquet(daily_path, daily)
    _write_parquet(selections_path, selections)
    _write_parquet(audit_path, audit_frame)
    summaries = _summaries(daily, study)
    paired = _paired_summaries(daily, study)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "study_id": STUDY_ID,
        "experiment_fingerprint": fingerprint,
        "study_sha256": base._sha256_file(frozen_path),
        "evaluation_years": years,
        "policy_summaries": summaries,
        "paired_summaries": paired,
        "decision": _decision(summaries, paired, study),
        "frozen_forecasts_reused_without_refit": True,
        "52_week_low_reward_used": False,
        "forbidden_2026_read_count": 0,
        "account_replay_performed": False,
        "profit_claim_allowed": False,
        "files": {
            "policy_daily": base._file_record(daily_path),
            "selections": base._file_record(selections_path),
            "margin_audit": base._file_record(audit_path),
        },
    }
    base._write_json(summary_path, summary)
    return summary


def validate_summary(path: str | Path) -> dict[str, Any]:
    summary = _read_json(base._resolve_path(path))
    checks = {
        "schema": summary.get("schema") == SUMMARY_SCHEMA,
        "study": summary.get("study_id") == STUDY_ID,
        "frozen": summary.get("frozen_forecasts_reused_without_refit") is True,
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
