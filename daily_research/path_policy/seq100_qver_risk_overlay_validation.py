"""Validate one frozen QVER risk-overlay challenger against the old core policy.

The old 30/30/25/10/5 score forms the exact industry-capped Top30 account
scan list before any outcome is read.  Three expanding logistic risk heads then
rerank only that fixed candidate band using labels whose legal D120 availability
strictly precedes each signal.  The study is retrospective and cannot create a
live or stable-profit claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from daily_research.path_policy import seq100_causal_value_policy as causal
from daily_research.path_policy import seq100_exact_value_growth_account as old_account
from daily_research.path_policy import (
    seq100_margin_residual_account_feasibility as feasibility,
)
from daily_research.path_policy import seq100_qver_confirmation_effect as qver
from daily_research.path_policy import seq100_stock_distribution as base
from daily_research.path_policy.seq100_exit_policy_audit import (
    CandidateCompleteAuditPack,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_qver_risk_overlay_validation_v1"
SUMMARY_SCHEMA = "seq100_qver_risk_overlay_validation_summary/1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT
    / "daily_research/studies/seq100_qver_risk_overlay_validation_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies"
    / "seq100_qver_risk_overlay_validation_v1"
)
HEAD_NAMES = ("revision_reversal", "multiple_compression", "early_spike_fade")
HORIZONS = (20, 60, 120)


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
    epistemic = dict(study["epistemic_contract"])
    if not bool(epistemic.get("candidate_band_is_frozen_before_outcome_reads")):
        raise ValueError("candidate_band_timing_contract_missing")
    if not bool(
        epistemic.get("risk_predictions_use_only_labels_available_before_each_signal")
    ):
        raise ValueError("risk_label_timing_contract_missing")
    if not bool(epistemic.get("parameter_weight_and_threshold_grid_forbidden")):
        raise ValueError("grid_prohibition_missing")
    if int(study["source"]["forbidden_year"]) != 2026:
        raise ValueError("forbidden_year_contract_mismatch")
    band = dict(study["candidate_band"])
    if (
        str(band["source_score"]) != "exact_full_score"
        or str(band["source_eligibility"]) != "eligible__exact_full_top10"
        or int(band["scan_count"]) != 30
        or int(band["selection_count"]) != 10
        or int(band["maximum_names_per_pit_industry"]) != 2
    ):
        raise ValueError("candidate_band_contract_mismatch")
    reranking = dict(study["reranking"])
    weights = {
        str(key): float(value) for key, value in reranking["head_weights"].items()
    }
    if set(weights) != set(HEAD_NAMES) or not math.isclose(
        sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("risk_head_weight_contract_mismatch")
    if bool(reranking.get("core_score_blending")):
        raise ValueError("core_score_blending_forbidden")
    if reranking.get("hard_risk_threshold") is not None:
        raise ValueError("hard_risk_threshold_forbidden")
    if bool(reranking.get("parameter_grid_performed")):
        raise ValueError("reranking_grid_forbidden")
    if not bool(study["model"].get("all_three_heads_required_for_reordering")):
        raise ValueError("partial_head_reranking_forbidden")
    if set(study["risk_heads"]) != set(HEAD_NAMES):
        raise ValueError("risk_head_set_mismatch")
    return study, study_path


def _verified_sources(
    study: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, Any], dict[str, Path]]:
    source = dict(study["source"])
    pairs = {
        "qver_study": "qver_study_sha256",
        "qver_summary": "qver_summary_sha256",
        "exact_account_summary": "exact_account_summary_sha256",
    }
    paths: dict[str, Path] = {}
    for key, hash_key in pairs.items():
        candidate = base._resolve_path(source[key])
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        if base._sha256_file(candidate).lower() != str(source[hash_key]).lower():
            raise ValueError(f"source_hash_mismatch:{key}")
        paths[key] = candidate
    qver_study, _ = qver.load_study(paths["qver_study"])
    qver_sources = qver._source_contract(qver_study)
    validation = qver.validate_summary(paths["qver_summary"])
    if not validation["passed"]:
        raise ValueError("qver_summary_validation_failed")
    old_summary = _read_json(paths["exact_account_summary"])
    expected_gross = float(study["account"]["matched_frozen_gross_fraction"])
    actual_gross = float(old_summary["risk_budget"]["frozen_target_gross_fraction"])
    if not math.isclose(expected_gross, actual_gross, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("matched_gross_fraction_source_mismatch")
    return paths, qver_study, qver_sources


def _prepare_features(
    panel: base.StockPanel,
    source_path: Path,
) -> pd.DataFrame:
    source = pd.read_parquet(source_path)
    frame = qver._attach_extra_features(panel, source)
    frame["framework_score"] = frame["exact_full_score"].to_numpy(dtype=float)
    frame["framework_score_100"] = 100.0 * frame["framework_score"]
    frame["framework_rating"] = "not_applicable"
    frame["eligible__framework"] = frame["eligible__exact_full_top10"].astype(bool)
    if int(frame["evaluation_year"].max()) >= 2026:
        raise ValueError("forbidden_2026_candidate")
    return frame


def _candidate_band(
    features: pd.DataFrame,
    study: Mapping[str, Any],
) -> pd.DataFrame:
    contract = dict(study["candidate_band"])
    records: list[pd.DataFrame] = []
    for _, local in features.groupby("date_idx", sort=True):
        chosen = causal._select_industry_capped(
            score=local[str(contract["source_score"])].to_numpy(dtype=np.float64),
            eligible=local[str(contract["source_eligibility"])].to_numpy(dtype=bool),
            candidate_id=local["candidate_id"].to_numpy(dtype=np.int64),
            industry_code=local["industry_code"].to_numpy(dtype=np.int64),
            top_k=int(contract["scan_count"]),
            industry_cap=int(contract["maximum_names_per_pit_industry"]),
        )
        if not len(chosen):
            continue
        band = local.iloc[chosen].copy()
        band["core_band_rank"] = np.arange(1, len(band) + 1, dtype=np.int16)
        band["selection_rank"] = band["core_band_rank"]
        records.append(band)
    if not records:
        raise ValueError("candidate_band_empty")
    result = pd.concat(records, ignore_index=True)
    cap = result.groupby(["date_idx", "industry_code"]).size().groupby("date_idx").max()
    if int(cap.max()) > int(contract["maximum_names_per_pit_industry"]):
        raise ValueError("candidate_band_industry_cap_breach")
    if bool(result["candidate_id"].duplicated().any()):
        raise ValueError("candidate_band_duplicate_candidate")
    return result.sort_values(
        ["date_idx", "core_band_rank"], kind="stable"
    ).reset_index(drop=True)


def _attach_outcomes(
    panel: base.StockPanel,
    features: pd.DataFrame,
    band: pd.DataFrame,
    qver_study: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = qver._candidate_returns(panel, features, qver_study)
    band_returns = returns.loc[returns["candidate_id"].isin(band["candidate_id"])]
    overlap = [
        column
        for column in band.columns
        if column in band_returns.columns and column != "candidate_id"
    ]
    result = band.merge(
        band_returns.drop(columns=overlap),
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    paths = qver._selected_paths(panel, band, qver_study)
    result = result.merge(paths, on="candidate_id", how="left", validate="one_to_one")
    result = qver._attach_future_snapshots(result, features)
    result["label_available_date_idx"] = result["legal_exit_date_idx_d120"].where(
        result["legal_exit_date_idx_d120"].ge(0), np.nan
    )

    future_revision = result[["future_revision_np_90", "future_revision_eps_90"]]
    revision_valid = result["future_snapshot_available"].astype(
        bool
    ) & future_revision.notna().any(axis=1)
    revision_mean = future_revision.mean(axis=1, skipna=True)
    result["label_revision_reversal"] = np.where(
        revision_valid, revision_mean.lt(0.0).astype(float), np.nan
    )

    multiple = result["implied_multiple_change_to_d120_snapshot"]
    result["label_multiple_compression"] = np.where(
        multiple.notna(), multiple.lt(-0.05).astype(float), np.nan
    )

    path_valid = (
        result["primary_shape"].notna() & result["legal_net_return_d120"].notna()
    )
    result["label_early_spike_fade"] = np.where(
        path_valid, result["primary_shape"].eq("early_spike_fade").astype(float), np.nan
    )
    return result, returns


def _numeric_features(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return (
        frame.loc[:, list(columns)]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )


def _pipeline(study: Mapping[str, Any]) -> Pipeline:
    model = dict(study["model"])
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", RobustScaler(quantile_range=(10.0, 90.0))),
            (
                "model",
                LogisticRegression(
                    C=float(model["c"]),
                    penalty=str(model["penalty"]),
                    solver=str(model["solver"]),
                    class_weight=model["class_weight"],
                    max_iter=int(model["maximum_iterations"]),
                    random_state=int(model["seed"]),
                ),
            ),
        ]
    )


def _training_rows(
    frame: pd.DataFrame,
    *,
    signal_date_idx: int,
    label_column: str,
) -> pd.DataFrame:
    available = pd.to_numeric(frame["label_available_date_idx"], errors="coerce")
    return frame.loc[available.lt(int(signal_date_idx)) & frame[label_column].notna()]


def _rerank_local(
    local: pd.DataFrame,
    *,
    head_weights: Mapping[str, float],
    all_heads_ready: bool,
) -> pd.DataFrame:
    result = local.copy()
    if not all_heads_ready:
        result["composite_risk_percentile"] = 0.5
        result["overlay_rank"] = result["core_band_rank"].astype(np.int16)
        result["reranking_active"] = False
        for head in HEAD_NAMES:
            result[f"risk_percentile_{head}"] = 0.5
        return result
    composite = np.zeros(len(result), dtype=np.float64)
    for head in HEAD_NAMES:
        percentile = result[f"probability_{head}"].rank(method="average", pct=True)
        result[f"risk_percentile_{head}"] = percentile.to_numpy(dtype=float)
        composite += float(head_weights[head]) * percentile.to_numpy(dtype=float)
    result["composite_risk_percentile"] = composite
    order = result.sort_values(
        ["composite_risk_percentile", "core_band_rank", "candidate_id"],
        ascending=[True, True, True],
        kind="stable",
    ).index
    rank = pd.Series(np.arange(1, len(result) + 1, dtype=np.int16), index=order)
    result["overlay_rank"] = rank.loc[result.index].to_numpy(dtype=np.int16)
    result["reranking_active"] = True
    return result


def _expanding_risk_predictions(
    labelled_band: pd.DataFrame,
    study: Mapping[str, Any],
) -> pd.DataFrame:
    model_contract = dict(study["model"])
    head_contracts = dict(study["risk_heads"])
    weights = {
        str(key): float(value)
        for key, value in study["reranking"]["head_weights"].items()
    }
    minimum_rows = int(model_contract["minimum_training_rows"])
    minimum_class = int(model_contract["minimum_class_count"])
    records: list[pd.DataFrame] = []
    source = labelled_band.sort_values(
        ["date_idx", "core_band_rank"], kind="stable"
    ).reset_index(drop=True)
    for date_idx, local in source.groupby("date_idx", sort=True):
        current = local.copy()
        readiness: dict[str, bool] = {}
        for head in HEAD_NAMES:
            label_column = f"label_{head}"
            features = tuple(str(value) for value in head_contracts[head]["features"])
            train = _training_rows(
                source, signal_date_idx=int(date_idx), label_column=label_column
            )
            y = train[label_column].astype(int)
            counts = y.value_counts()
            ready = bool(
                len(train) >= minimum_rows
                and len(counts) == 2
                and int(counts.min()) >= minimum_class
            )
            readiness[head] = ready
            current[f"model_ready_{head}"] = ready
            current[f"training_rows_{head}"] = len(train)
            current[f"training_positive_fraction_{head}"] = (
                float(y.mean()) if len(y) else math.nan
            )
            current[f"maximum_training_label_date_idx_{head}"] = (
                int(train["label_available_date_idx"].max()) if len(train) else -1
            )
            if not ready:
                current[f"probability_{head}"] = np.nan
                continue
            estimator = _pipeline(study)
            estimator.fit(_numeric_features(train, features), y)
            current[f"probability_{head}"] = estimator.predict_proba(
                _numeric_features(current, features)
            )[:, 1]
        current = _rerank_local(
            current,
            head_weights=weights,
            all_heads_ready=bool(all(readiness.values())),
        )
        records.append(current)
    result = pd.concat(records, ignore_index=True)
    for head in HEAD_NAMES:
        trained = result[f"model_ready_{head}"].astype(bool)
        if bool(
            (
                result.loc[trained, f"maximum_training_label_date_idx_{head}"]
                >= result.loc[trained, "date_idx"]
            ).any()
        ):
            raise ValueError(f"future_label_leakage:{head}")
    return result.sort_values(["date_idx", "overlay_rank"], kind="stable").reset_index(
        drop=True
    )


def _period_name(year: int, study: Mapping[str, Any]) -> str:
    for name, years in study["evaluation"]["periods"].items():
        if int(year) in {int(value) for value in years}:
            return str(name)
    raise ValueError(f"evaluation_year_unassigned:{year}")


def _risk_head_metrics(
    predictions: pd.DataFrame,
    study: Mapping[str, Any],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    period_masks: dict[str, pd.Series] = {
        "full_history": pd.Series(True, index=predictions.index)
    }
    for period, years in study["evaluation"]["periods"].items():
        period_masks[str(period)] = predictions["evaluation_year"].isin(
            [int(value) for value in years]
        )
    for head in HEAD_NAMES:
        probability = f"probability_{head}"
        label = f"label_{head}"
        for period, mask in period_masks.items():
            local = predictions.loc[mask, ["date_idx", probability, label]].dropna()
            y = local[label].astype(int)
            p = local[probability].to_numpy(dtype=float)
            auc = (
                float(roc_auc_score(y, p)) if len(y) and y.nunique() == 2 else math.nan
            )
            brier = float(brier_score_loss(y, p)) if len(y) else math.nan
            prevalence = float(y.mean()) if len(y) else math.nan
            null_brier = prevalence * (1.0 - prevalence) if len(y) else math.nan
            risk_rank = local.groupby("date_idx")[probability].rank(
                method="average", pct=True
            )
            top = y.loc[risk_rank.ge(0.8)]
            bottom = y.loc[risk_rank.le(0.2)]
            records.append(
                {
                    "head": head,
                    "period": period,
                    "observation_count": len(local),
                    "month_count": int(local["date_idx"].nunique()),
                    "event_fraction": prevalence,
                    "roc_auc": auc,
                    "brier_score": brier,
                    "null_brier_score": null_brier,
                    "brier_skill": (
                        float(1.0 - brier / null_brier)
                        if math.isfinite(null_brier) and null_brier > 0.0
                        else math.nan
                    ),
                    "highest_risk_quintile_event_fraction": (
                        float(top.mean()) if len(top) else math.nan
                    ),
                    "lowest_risk_quintile_event_fraction": (
                        float(bottom.mean()) if len(bottom) else math.nan
                    ),
                }
            )
    return pd.DataFrame(records)


def _monthly_cohorts(
    all_returns: pd.DataFrame,
    predictions: pd.DataFrame,
    study: Mapping[str, Any],
) -> pd.DataFrame:
    eligible = all_returns.loc[all_returns["eligible__framework"]].copy()
    baseline = predictions.loc[predictions["core_band_rank"].le(10)]
    challenger = predictions.loc[predictions["overlay_rank"].le(10)]
    records: list[dict[str, Any]] = []
    for date_idx, local in eligible.groupby("date_idx", sort=True):
        base_local = baseline.loc[baseline["date_idx"].eq(date_idx)]
        challenge_local = challenger.loc[challenger["date_idx"].eq(date_idx)]
        record: dict[str, Any] = {
            "date_idx": int(date_idx),
            "trade_date": str(local["trade_date"].iloc[0]),
            "evaluation_year": int(local["evaluation_year"].iloc[0]),
            "period": _period_name(int(local["evaluation_year"].iloc[0]), study),
            "eligible_count": len(local),
            "baseline_selected_count": len(base_local),
            "challenger_selected_count": len(challenge_local),
            "reranking_active": bool(challenge_local["reranking_active"].any()),
        }
        for horizon in HORIZONS:
            column = f"legal_net_return_d{horizon}"
            record[f"eligible_mean_d{horizon}"] = float(local[column].mean())
            record[f"baseline_mean_d{horizon}"] = float(base_local[column].mean())
            record[f"challenger_mean_d{horizon}"] = float(
                challenge_local[column].mean()
            )
            record[f"challenger_minus_baseline_d{horizon}"] = (
                record[f"challenger_mean_d{horizon}"]
                - record[f"baseline_mean_d{horizon}"]
            )
        records.append(record)
    return pd.DataFrame(records)


def _cohort_summaries(
    monthly: pd.DataFrame,
    study: Mapping[str, Any],
) -> list[dict[str, Any]]:
    periods: dict[str, pd.DataFrame] = {"full_history": monthly}
    for name in study["evaluation"]["periods"]:
        periods[str(name)] = monthly.loc[monthly["period"].eq(name)]
    records: list[dict[str, Any]] = []
    for period, local in periods.items():
        for horizon in HORIZONS:
            eligible = local[f"eligible_mean_d{horizon}"]
            baseline = local[f"baseline_mean_d{horizon}"]
            challenger = local[f"challenger_mean_d{horizon}"]
            difference = local[f"challenger_minus_baseline_d{horizon}"]
            for policy, values in (
                ("baseline_old_core", baseline),
                ("challenger_risk_overlay", challenger),
            ):
                excess = values - eligible
                annual = (
                    pd.DataFrame({"year": local["evaluation_year"], "value": values})
                    .groupby("year")["value"]
                    .mean()
                )
                records.append(
                    {
                        "period": period,
                        "horizon": horizon,
                        "policy": policy,
                        "selected_return": qver._inference(values.to_numpy(), study),
                        "eligible_excess": qver._inference(excess.to_numpy(), study),
                        "positive_year_count": int(annual.gt(0.0).sum()),
                        "year_count": len(annual),
                    }
                )
            records.append(
                {
                    "period": period,
                    "horizon": horizon,
                    "policy": "challenger_minus_baseline",
                    "paired_difference": qver._inference(difference.to_numpy(), study),
                    "active_month_count": int(local["reranking_active"].sum()),
                    "month_count": len(local),
                }
            )
    return records


def _selection_members(predictions: pd.DataFrame) -> pd.DataFrame:
    baseline = predictions.loc[predictions["core_band_rank"].le(10)].copy()
    baseline.insert(0, "policy", "baseline_old_core")
    challenger = predictions.loc[predictions["overlay_rank"].le(10)].copy()
    challenger.insert(0, "policy", "challenger_risk_overlay")
    return (
        pd.concat([baseline, challenger], ignore_index=True)
        .sort_values(["policy", "date_idx", "overlay_rank"], kind="stable")
        .reset_index(drop=True)
    )


def _selection_outcome_summary(selections: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for policy, local in selections.groupby("policy", sort=False):
        positive = local.loc[
            local["legal_net_return_d120"].gt(0.0), "legal_net_return_d120"
        ].sort_values(ascending=False)
        records.append(
            {
                "policy": str(policy),
                "decision_count": len(local),
                "reranking_active_fraction": float(local["reranking_active"].mean()),
                "mean_net_return_d60": float(local["legal_net_return_d60"].mean()),
                "mean_net_return_d120": float(local["legal_net_return_d120"].mean()),
                "early_spike_fade_fraction": float(
                    local["primary_shape"].eq("early_spike_fade").mean()
                ),
                "persistent_loss_fraction": float(
                    local["primary_shape"].eq("persistent_loss").mean()
                ),
                "revision_reversal_fraction": float(
                    local["label_revision_reversal"].mean()
                ),
                "multiple_compression_fraction": float(
                    local["label_multiple_compression"].mean()
                ),
                "top_10_positive_return_sum_share": (
                    float(positive.head(10).sum() / positive.sum())
                    if len(positive) and float(positive.sum()) > 0.0
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(records)


def _account_scored(
    predictions: pd.DataFrame,
    *,
    rank_column: str,
) -> pd.DataFrame:
    frame = predictions.copy()
    frame["eligible__framework"] = True
    frame["framework_score"] = -frame[rank_column].to_numpy(dtype=float)
    frame["framework_score_100"] = 100.0 * frame["framework_score"]
    frame["framework_rating"] = "not_applicable"
    return frame


def _winner_concentration(trades: pd.DataFrame) -> dict[str, Any]:
    positive = trades.loc[trades["net_pnl_cny"].gt(0.0), "net_pnl_cny"].sort_values(
        ascending=False
    )
    total = float(positive.sum())
    return {
        "positive_trade_count": len(positive),
        "positive_pnl_cny": total,
        "top_10_positive_pnl_share": (
            float(positive.head(10).sum() / total) if total > 0.0 else math.nan
        ),
        "top_1pct_positive_pnl_share": (
            float(positive.head(max(1, math.ceil(0.01 * len(positive)))).sum() / total)
            if total > 0.0
            else math.nan
        ),
    }


def _account_runs(
    panel: base.StockPanel,
    predictions: pd.DataFrame,
    study: Mapping[str, Any],
    qver_study: Mapping[str, Any],
    pack_manifest: Path,
) -> tuple[
    dict[str, Any],
    dict[str, tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]],
    dict[str, pd.DataFrame],
]:
    pack = CandidateCompleteAuditPack(pack_manifest)
    feasibility._validate_pack_alignment(panel, pack)
    baseline_scored = _account_scored(predictions, rank_column="core_band_rank")
    challenger_scored = _account_scored(predictions, rank_column="overlay_rank")
    baseline_book, baseline_candidates, baseline_audit = qver._build_account_book(
        baseline_scored, pack, qver_study
    )
    challenger_book, challenger_candidates, challenger_audit = qver._build_account_book(
        challenger_scored, pack, qver_study
    )
    adjust_factor, adjust_audit = feasibility._load_adjust_factor_panel(
        pack, study=qver_study
    )
    market = old_account._market(pack, adjust_factor=adjust_factor)
    signal_amount = feasibility._signal_amount_panel(panel, pack)
    dates = np.asarray(pack.date_values, dtype=str)
    cutoff_positions = np.flatnonzero(
        dates == str(study["source"]["maximum_account_mark_date"])
    )
    if len(cutoff_positions) != 1:
        raise ValueError("maximum_account_mark_date_missing")
    cutoff = int(cutoff_positions[0])
    years = tuple(int(value) for value in study["evaluation"]["years"])
    gross = float(study["account"]["matched_frozen_gross_fraction"])
    runs: dict[
        str,
        tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, list[dict[str, Any]]],
    ] = {}
    for policy, book in (
        ("baseline", baseline_book),
        ("challenger", challenger_book),
    ):
        runs[f"{policy}_full"] = qver._simulate_account(
            market=market,
            book=book,
            study=qver_study,
            signal_amount=signal_amount,
            terminal_recovery_date_idx=cutoff,
            target_gross_fraction=1.0,
            years=years,
        )
        runs[f"{policy}_matched"] = qver._simulate_account(
            market=market,
            book=book,
            study=qver_study,
            signal_amount=signal_amount,
            terminal_recovery_date_idx=cutoff,
            target_gross_fraction=gross,
            years=years,
        )
    summary = {
        "matched_frozen_gross_fraction": gross,
        "baseline_selection_audit": baseline_audit,
        "challenger_selection_audit": challenger_audit,
        "adjust_factor_audit": adjust_audit,
        "runs": {
            name: {
                "metric": run[0],
                "annual": run[3],
                "positive_year_count": int(
                    sum(float(row["net_return"]) > 0.0 for row in run[3])
                ),
                "year_count": len(run[3]),
                "winner_concentration": _winner_concentration(run[2]),
            }
            for name, run in runs.items()
        },
    }
    candidates = {
        "baseline_account_candidates": baseline_candidates,
        "challenger_account_candidates": challenger_candidates,
    }
    return summary, runs, candidates


def _acceptance_decision(
    *,
    risk_metrics: pd.DataFrame,
    cohort_summaries: Sequence[Mapping[str, Any]],
    selection_summary: pd.DataFrame,
    account_summary: Mapping[str, Any],
) -> dict[str, Any]:
    full_auc = risk_metrics.loc[risk_metrics["period"].eq("full_history")].set_index(
        "head"
    )["roc_auc"]
    paired = next(
        row
        for row in cohort_summaries
        if row["period"] == "full_history"
        and row["horizon"] == 120
        and row["policy"] == "challenger_minus_baseline"
    )
    selected = selection_summary.set_index("policy")
    baseline_account = account_summary["runs"]["baseline_matched"]
    challenger_account = account_summary["runs"]["challenger_matched"]
    baseline_metric = baseline_account["metric"]
    challenger_metric = challenger_account["metric"]
    checks = {
        "all_three_risk_head_full_history_auc_above_half": bool(
            set(full_auc.index) == set(HEAD_NAMES)
            and full_auc.notna().all()
            and full_auc.gt(0.5).all()
        ),
        "d120_challenger_minus_baseline_hac_lcb_nonnegative": bool(
            float(paired["paired_difference"]["lcb_95"]) >= 0.0
        ),
        "matched_account_annualized_log_growth_not_lower": bool(
            float(challenger_metric["annualized_log_growth"])
            >= float(baseline_metric["annualized_log_growth"])
        ),
        "matched_account_maximum_drawdown_not_worse": bool(
            float(challenger_metric["full_path_maximum_drawdown"])
            >= float(baseline_metric["full_path_maximum_drawdown"])
        ),
        "selected_early_spike_fade_fraction_not_higher": bool(
            float(selected.loc["challenger_risk_overlay", "early_spike_fade_fraction"])
            <= float(selected.loc["baseline_old_core", "early_spike_fade_fraction"])
        ),
        "account_top10_positive_pnl_share_not_higher": bool(
            float(
                challenger_account["winner_concentration"]["top_10_positive_pnl_share"]
            )
            <= float(
                baseline_account["winner_concentration"]["top_10_positive_pnl_share"]
            )
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "interpretation": (
            "historically_not_rejected"
            if all(checks.values())
            else "historically_rejected"
        ),
        "stable_profit_claim_allowed": False,
    }


def run_study(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    study, frozen_path = load_study(study_path)
    source_paths, qver_study, qver_sources = _verified_sources(study)
    root = base._resolve_path(output_root)
    implementation_path = Path(__file__)
    fingerprint = _payload_hash(
        {
            "study_sha256": base._sha256_file(frozen_path),
            "implementation_sha256": base._sha256_file(implementation_path),
            "source_hashes": {
                **{key: base._sha256_file(path) for key, path in source_paths.items()},
                **{
                    f"qver_{key}": base._sha256_file(path)
                    for key, path in qver_sources.items()
                    if path.is_file()
                },
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
        input_manifest_path=qver_sources["input_manifest"],
        label_manifest_path=qver_sources["label_manifest"],
    )
    if bool((panel.years == int(study["source"]["forbidden_year"])).any()):
        raise ValueError("forbidden_2026_panel_row")
    features = _prepare_features(panel, qver_sources["candidate_features"])

    # This exact band is materialized in memory before any outcome array is read.
    band = _candidate_band(features, study)
    band_candidate_ids = band["candidate_id"].to_numpy(dtype=np.int64).copy()

    labelled_band, all_returns = _attach_outcomes(panel, features, band, qver_study)
    if not np.array_equal(
        np.sort(band_candidate_ids),
        np.sort(labelled_band["candidate_id"].to_numpy(dtype=np.int64)),
    ):
        raise ValueError("candidate_band_changed_after_outcome_read")

    predictions = _expanding_risk_predictions(labelled_band, study)
    risk_metrics = _risk_head_metrics(predictions, study)
    monthly = _monthly_cohorts(all_returns, predictions, study)
    cohort_summaries = _cohort_summaries(monthly, study)
    selections = _selection_members(predictions)
    selection_summary = _selection_outcome_summary(selections)
    account_summary, account_runs, account_candidates = _account_runs(
        panel,
        predictions,
        study,
        qver_study,
        qver_sources["pack_manifest"],
    )

    old_summary = _read_json(source_paths["exact_account_summary"])
    reproduced = account_summary["runs"]["baseline_matched"]["metric"]
    if not math.isclose(
        float(reproduced["liquidated_ending_equity_cny"]),
        float(old_summary["risk_metric"]["liquidated_ending_equity_cny"]),
        rel_tol=0.0,
        abs_tol=0.01,
    ):
        raise ValueError("baseline_account_reproduction_mismatch")

    decision = _acceptance_decision(
        risk_metrics=risk_metrics,
        cohort_summaries=cohort_summaries,
        selection_summary=selection_summary,
        account_summary=account_summary,
    )

    outputs: dict[str, tuple[Path, pd.DataFrame]] = {
        "candidate_band_predictions": (
            root / "candidate_band_predictions.parquet",
            predictions,
        ),
        "selection_members": (root / "selection_members.parquet", selections),
        "selection_summary": (root / "selection_summary.parquet", selection_summary),
        "monthly_cohorts": (root / "monthly_cohorts.parquet", monthly),
        "risk_head_metrics": (root / "risk_head_metrics.parquet", risk_metrics),
        **{
            name: (root / f"{name}.parquet", frame)
            for name, frame in account_candidates.items()
        },
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

    full_risk = risk_metrics.loc[risk_metrics["period"].eq("full_history")]
    primary_paired = next(
        row
        for row in cohort_summaries
        if row["period"] == "full_history"
        and row["horizon"] == 120
        and row["policy"] == "challenger_minus_baseline"
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed_retrospective_single_challenger_validation",
        "study_id": STUDY_ID,
        "experiment_fingerprint": fingerprint,
        "study_sha256": base._sha256_file(frozen_path),
        "implementation_sha256": base._sha256_file(implementation_path),
        "candidate_band_precedes_outcome_reads": True,
        "candidate_band_count": len(band),
        "candidate_band_month_count": int(band["date_idx"].nunique()),
        "candidate_band_min_per_month": int(band.groupby("date_idx").size().min()),
        "candidate_band_max_per_month": int(band.groupby("date_idx").size().max()),
        "reranking_active_month_count": int(
            predictions.groupby("date_idx")["reranking_active"].any().sum()
        ),
        "risk_head_full_history": full_risk.to_dict(orient="records"),
        "cohort_summaries": cohort_summaries,
        "primary_d120_paired_difference": primary_paired,
        "selection_summary": selection_summary.to_dict(orient="records"),
        "account": account_summary,
        "baseline_account_reproduced": True,
        "acceptance": decision,
        "forbidden_2026_read_count": 0,
        "retrospective_only": True,
        "stable_profit_claim_allowed": False,
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
        == "completed_retrospective_single_challenger_validation",
        "band_before_outcomes": summary.get("candidate_band_precedes_outcome_reads")
        is True,
        "baseline_reproduced": summary.get("baseline_account_reproduced") is True,
        "no_2026": summary.get("forbidden_2026_read_count") == 0,
        "retrospective": summary.get("retrospective_only") is True,
        "no_stable_claim": summary.get("stable_profit_claim_allowed") is False,
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
