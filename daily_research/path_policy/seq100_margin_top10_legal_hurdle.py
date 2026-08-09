"""Legal-win and lower-tail hurdle challenge for financing Top10 candidates."""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_margin_top10_two_stage as parent
from daily_research.path_policy import seq100_path_label_learnability as learnability

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_margin_top10_legal_hurdle_v1"
SUMMARY_SCHEMA = "seq100_margin_top10_legal_hurdle_summary/1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_margin_top10_legal_hurdle_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT / "daily_research/output/path_policy/studies" / STUDY_ID
)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (
        (WORKSPACE_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    )


def load_study(path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    study_path = _resolve(path or DEFAULT_STUDY_PATH)
    study = json.loads(study_path.read_text(encoding="utf-8"))
    if study.get("study_id") != STUDY_ID:
        raise ValueError("study_id_mismatch")
    for key in (
        "parent_study",
        "prepared_manifest",
        "parent_summary",
        "candidate_oof_scores",
    ):
        source = _resolve(study["source"][key])
        if parent._sha256(source) != study["source"][f"{key}_sha256"]:
            raise ValueError(f"source_hash_mismatch:{key}")
    if int(study["source"]["forbidden_year"]) != 2026:
        raise ValueError("forbidden_year_mismatch")
    return study, study_path


def _parameters(study: Mapping[str, Any], target: str) -> dict[str, Any]:
    model = study["model"]
    result: dict[str, Any] = {
        "learning_rate": float(model["learning_rate"]),
        "num_leaves": int(model["num_leaves"]),
        "max_depth": int(model["max_depth"]),
        "min_child_samples": int(model["min_data_in_leaf"]),
        "reg_lambda": float(model["lambda_l2"]),
        "n_estimators": int(model["n_estimators"]),
        "colsample_bytree": float(model["feature_fraction"]),
        "subsample": 1.0,
        "n_jobs": int(model["num_threads"]),
        "random_state": int(study["validation"]["seed"]),
        "deterministic": True,
        "verbosity": -1,
    }
    if target == "legal_net_positive":
        result["objective"] = "binary"
    elif target == "legal_net_q20":
        result["objective"] = "quantile"
        result["alpha"] = float(model["quantile_alpha"])
    else:
        raise ValueError(f"unknown_target:{target}")
    return result


def _fit_oof(
    *,
    study: Mapping[str, Any],
    prepared: parent.PreparedData,
    candidates: pd.DataFrame,
    output_root: Path,
    folds: list[Mapping[str, Any]] | None = None,
    targets: tuple[str, ...] = ("legal_net_positive", "legal_net_q20"),
    model_prefix: str = "primary",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix, names = parent._candidate_matrix(
        prepared,
        candidates,
        variant="m2_direct_557_margin",
        base_scores=candidates[["candidate_id"]],
    )
    if matrix.shape[1] != int(study["features"]["feature_count"]):
        raise ValueError("feature_count_mismatch")
    feature_valid = np.isfinite(matrix).any(axis=1)
    date_idx = candidates["date_idx"].to_numpy(dtype=np.int32)
    exact_net = candidates["one_day_net_return"].fillna(0.0).to_numpy(dtype=np.float64)
    maximum_outcome_idx = int(prepared.manifest["labels"]["maximum_outcome_date_idx"])
    target_valid = date_idx + 2 <= maximum_outcome_idx
    predictions: list[pd.DataFrame] = []
    importance: list[pd.DataFrame] = []
    active_folds = list(folds or prepared.manifest["folds"])
    for fold in active_folds:
        fold_id = int(fold["fold"])
        train = (
            (date_idx <= int(fold["training_maximum_date_idx"]))
            & target_valid
            & feature_valid
        )
        evaluate = (
            (date_idx >= int(fold["validation_start_date_idx"]))
            & (date_idx <= int(fold["validation_end_date_idx"]))
            & target_valid
            & feature_valid
        )
        local = candidates.loc[evaluate, ["candidate_id", "date_idx"]].copy()
        for target in targets:
            if target == "legal_net_positive":
                values = (exact_net > 0.0).astype(np.int8)
                estimator: Any = lgb.LGBMClassifier(**_parameters(study, target))
            else:
                values = exact_net
                estimator = lgb.LGBMRegressor(**_parameters(study, target))
            estimator.fit(
                matrix[train],
                values[train],
                sample_weight=learnability.date_equal_weights(date_idx[train]),
                feature_name=list(names),
            )
            if target == "legal_net_positive":
                predicted = estimator.predict_proba(matrix[evaluate])[:, 1]
            else:
                predicted = estimator.predict(matrix[evaluate])
            local[target] = np.asarray(predicted, dtype=np.float32)
            model_path = (
                output_root / f"models/{model_prefix}__fold_{fold_id:04d}__{target}.txt"
            )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            estimator.booster_.save_model(str(model_path))
            importance.append(
                pd.DataFrame(
                    {
                        "fold": fold_id,
                        "target": target,
                        "feature": names,
                        "gain": estimator.booster_.feature_importance(
                            importance_type="gain"
                        ).astype(np.float64),
                    }
                )
            )
        local["fold"] = fold_id
        predictions.append(local)
    del matrix
    gc.collect()
    return (
        pd.concat(predictions, ignore_index=True),
        pd.concat(importance, ignore_index=True),
    )


def _history_folds(
    candidates: pd.DataFrame, study: Mapping[str, Any]
) -> list[dict[str, Any]]:
    date_idx = candidates["date_idx"].to_numpy(dtype=np.int32)
    years = candidates["evaluation_year"].to_numpy(dtype=np.int16)
    result: list[dict[str, Any]] = []
    for year in study["supplemental_history_extension"]["validation_years"]:
        validation = date_idx[years == int(year)]
        if not len(validation):
            raise ValueError(f"history_year_missing:{year}")
        start = int(validation.min())
        result.append(
            {
                "fold": int(year),
                "training_maximum_date_idx": start - 3,
                "validation_start_date_idx": start,
                "validation_end_date_idx": int(validation.max()),
            }
        )
    return result


def _within_date_rank(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("date_idx")[column].rank(pct=True, method="average")


def attach_scores(frame: pd.DataFrame) -> dict[str, tuple[str, str]]:
    expected = "candidate__m2_direct_557_margin__legal_gross_return"
    frame["score__parent_expected_net"] = frame[expected]
    frame["score__legal_win_probability"] = frame["legal_net_positive"]
    frame["score__legal_net_q20"] = frame["legal_net_q20"]
    frame["score__equal_rank_composite"] = (
        _within_date_rank(frame, expected)
        + _within_date_rank(frame, "legal_net_positive")
        + _within_date_rank(frame, "legal_net_q20")
    ) / 3.0
    return {
        "parent_expected_net": ("score__parent_expected_net", "legal"),
        "legal_win_probability": ("score__legal_win_probability", "direction"),
        "legal_net_q20": ("score__legal_net_q20", "legal"),
        "equal_rank_composite": ("score__equal_rank_composite", "composite"),
    }


def _select(
    frame: pd.DataFrame,
    *,
    score_name: str,
    score_column: str,
    top_k: int,
    no_trade: bool,
) -> pd.DataFrame:
    ordered = frame.loc[np.isfinite(frame[score_column])].sort_values(
        ["date_idx", score_column, "symbol"],
        ascending=[True, False, True],
        kind="stable",
    )
    selected = ordered.groupby("date_idx", sort=False).head(int(top_k)).copy()
    if not no_trade or selected.empty:
        return selected
    if score_name == "parent_expected_net":
        valid = selected[score_column] > 0.0
    elif score_name == "legal_win_probability":
        valid = selected[score_column] > 0.5
    elif score_name == "legal_net_q20":
        valid = selected[score_column] > 0.0
    elif score_name == "equal_rank_composite":
        valid = (selected["score__parent_expected_net"] > 0.0) & (
            selected["score__legal_win_probability"] > 0.5
        )
    else:
        raise ValueError(f"unknown_score:{score_name}")
    return selected.loc[valid].copy()


def _daily_rows(
    frame: pd.DataFrame,
    *,
    fold: int,
    score_name: str,
    score_column: str,
    top_k: int,
    no_trade: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = _select(
        frame,
        score_name=score_name,
        score_column=score_column,
        top_k=top_k,
        no_trade=no_trade,
    )
    baseline = frame.groupby(
        ["date_idx", "trade_date", "evaluation_year"], as_index=False
    ).agg(baseline_cash_net_return=("cash_net_return", "mean"))
    daily = (
        selected.groupby(["date_idx", "trade_date", "evaluation_year"], as_index=False)
        .agg(
            selected_count=("candidate_id", "size"),
            filled_count=("one_day_entry_filled", "sum"),
            daily_net_return=("cash_net_return", "mean"),
            next_close_up_fraction=("next_close_up_observed", "mean"),
            mean_score=(score_column, "mean"),
        )
        .merge(
            baseline,
            on=["date_idx", "trade_date", "evaluation_year"],
            how="left",
            validate="one_to_one",
        )
    )
    daily["paired_delta"] = (
        daily["daily_net_return"] - daily["baseline_cash_net_return"]
    )
    for target in (daily, selected):
        target["fold"] = int(fold)
        target["variant"] = "m2_legal_hurdle"
        target["score_type"] = score_name
        target["top_k"] = int(top_k)
        target["no_trade"] = bool(no_trade)
    selected["selection_score"] = selected[score_column]
    return daily, selected


def run(
    *,
    study_path: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    study, resolved_study_path = load_study(study_path)
    root = _resolve(output_root or study["decision_boundary"]["output_root"])
    _, _, prepared = parent.load_prepared(
        study_path=study["source"]["parent_study"],
        output_root=Path(study["source"]["prepared_manifest"]).parent.parent,
    )
    candidates = pd.read_parquet(prepared.manifest["candidate_panel"]["path"])
    parent_scores = pd.read_parquet(_resolve(study["source"]["candidate_oof_scores"]))
    predictions, importance = _fit_oof(
        study=study,
        prepared=prepared,
        candidates=candidates,
        output_root=root,
    )
    frame = parent_scores.merge(
        predictions,
        on=["candidate_id", "date_idx", "fold"],
        how="left",
        validate="one_to_one",
    )
    if frame["trade_date"].astype(str).str.startswith("2026").any():
        raise ValueError("forbidden_2026_row")
    mappings = attach_scores(frame)
    daily_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    for fold_id, local in frame.groupby("fold", sort=True):
        for score_name, (score_column, _) in mappings.items():
            for top_k in study["selection"]["daily_top_k"]:
                for no_trade in (False, True):
                    daily, selected = _daily_rows(
                        local,
                        fold=int(fold_id),
                        score_name=score_name,
                        score_column=score_column,
                        top_k=int(top_k),
                        no_trade=no_trade,
                    )
                    daily_frames.append(daily)
                    selected_frames.append(selected)
    daily = pd.concat(daily_frames, ignore_index=True)
    selected = pd.concat(selected_frames, ignore_index=True)
    summaries = parent._selection_summary(daily, selected, study)
    passed = [record for record in summaries if record["entry_gate_passed"]]

    history_predictions, history_importance = _fit_oof(
        study=study,
        prepared=prepared,
        candidates=candidates,
        output_root=root,
        folds=_history_folds(candidates, study),
        targets=("legal_net_positive",),
        model_prefix="history_extension",
    )
    history_years = {
        int(value)
        for value in study["supplemental_history_extension"]["validation_years"]
    }
    history_frame = candidates.loc[
        candidates["evaluation_year"].isin(history_years)
    ].merge(
        history_predictions,
        on=["candidate_id", "date_idx"],
        how="left",
        validate="one_to_one",
    )
    history_frame["cash_net_return"] = history_frame["one_day_net_return"].fillna(0.0)
    history_frame["next_close_observed"] = history_frame[
        "signal_close_return_d1"
    ].notna()
    history_frame["next_close_up_observed"] = (
        history_frame["signal_close_return_d1"] > 0.0
    )
    history_frame["score__legal_win_probability"] = history_frame["legal_net_positive"]
    history_daily_frames: list[pd.DataFrame] = []
    history_selected_frames: list[pd.DataFrame] = []
    for fold_id, local in history_frame.groupby("fold", sort=True):
        history_daily, history_selected = _daily_rows(
            local,
            fold=int(fold_id),
            score_name="legal_win_probability",
            score_column="score__legal_win_probability",
            top_k=3,
            no_trade=True,
        )
        history_daily_frames.append(history_daily)
        history_selected_frames.append(history_selected)
    history_daily = pd.concat(history_daily_frames, ignore_index=True)
    history_selected = pd.concat(history_selected_frames, ignore_index=True)
    history_summary = parent._selection_summary(history_daily, history_selected, study)[
        0
    ]
    primary_rule = (
        (daily["score_type"] == "legal_win_probability")
        & (daily["top_k"] == 3)
        & daily["no_trade"]
    )
    primary_selected_rule = (
        (selected["score_type"] == "legal_win_probability")
        & (selected["top_k"] == 3)
        & selected["no_trade"]
    )
    combined_daily = pd.concat(
        [history_daily, daily.loc[primary_rule]], ignore_index=True
    )
    combined_selected = pd.concat(
        [history_selected, selected.loc[primary_selected_rule]], ignore_index=True
    )
    combined_summary = parent._selection_summary(
        combined_daily, combined_selected, study
    )[0]
    importance = pd.concat([importance, history_importance], ignore_index=True)

    scores_path = root / "candidate_hurdle_oof_scores.parquet"
    daily_path = root / "daily_selection_metrics.parquet"
    selected_path = root / "selected_candidates.parquet"
    importance_path = root / "feature_importance.parquet"
    history_scores_path = root / "history_extension_oof_scores.parquet"
    history_daily_path = root / "history_extension_daily_metrics.parquet"
    history_selected_path = root / "history_extension_selected_candidates.parquet"
    parent._write_parquet(scores_path, frame)
    parent._write_parquet(daily_path, daily)
    parent._write_parquet(selected_path, selected)
    parent._write_parquet(importance_path, importance)
    parent._write_parquet(history_scores_path, history_frame)
    parent._write_parquet(history_daily_path, history_daily)
    parent._write_parquet(history_selected_path, history_selected)
    payload = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "completed_at": parent._now(),
        "study_id": STUDY_ID,
        "study_sha256": parent._sha256(resolved_study_path),
        "epistemic_status": "adaptive_retrospective_development_validation",
        "stable_profit_claim_allowed": False,
        "candidate_count": len(candidates),
        "validation_candidate_count": len(frame),
        "validation_date_count": int(frame["date_idx"].nunique()),
        "selection_summaries": summaries,
        "history_extension_summary": history_summary,
        "combined_2014_2025_summary": combined_summary,
        "passed_selection_count": len(passed),
        "passed_selections": passed,
        "account_replay_performed": False,
        "forbidden_2026_read_count": 0,
        "files": {
            "candidate_hurdle_oof_scores": parent._file_record(
                scores_path, row_count=len(frame)
            ),
            "daily_selection_metrics": parent._file_record(
                daily_path, row_count=len(daily)
            ),
            "selected_candidates": parent._file_record(
                selected_path, row_count=len(selected)
            ),
            "feature_importance": parent._file_record(
                importance_path, row_count=len(importance)
            ),
            "history_extension_oof_scores": parent._file_record(
                history_scores_path, row_count=len(history_frame)
            ),
            "history_extension_daily_metrics": parent._file_record(
                history_daily_path, row_count=len(history_daily)
            ),
            "history_extension_selected_candidates": parent._file_record(
                history_selected_path, row_count=len(history_selected)
            ),
        },
    }
    parent._write_json(root / "summary.json", payload)
    return payload


def validate(
    *,
    study_path: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    study, _ = load_study(study_path)
    root = _resolve(output_root or study["decision_boundary"]["output_root"])
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("schema") != SUMMARY_SCHEMA or summary.get("status") != "completed":
        raise ValueError("summary_contract_mismatch")
    if int(summary.get("forbidden_2026_read_count", -1)) != 0:
        raise ValueError("forbidden_2026_read")
    for record in summary["files"].values():
        if parent._sha256(Path(record["path"])) != record["sha256"]:
            raise ValueError("output_hash_mismatch")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "validate"))
    parser.add_argument("--study-path", default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    if args.command == "run":
        payload = run(study_path=args.study_path, output_root=args.output_root)
    else:
        payload = validate(study_path=args.study_path, output_root=args.output_root)
    print(
        json.dumps(payload, ensure_ascii=False, indent=2, default=parent._json_default)
    )


if __name__ == "__main__":
    main()
