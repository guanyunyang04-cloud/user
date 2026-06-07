"""Build prior-fit Baostock-only LightGBM ML signals for frontier diagnostics."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from traditional_quant_research.dataset_v2 import load_pit_manifest, load_quality_report, load_tradeable_panel
from traditional_quant_research.experiments.low_corr_candidate_frontier_audit import DEFAULT_FINAL_END_DATE, DEFAULT_HORIZON
from traditional_quant_research.research_panel import (
    FACTOR_SETS,
    add_cross_sectional_excess_return_labels,
    add_cross_sectional_zscores,
    build_factor_label_panel,
    factor_columns_for_set,
    normalize_factor_set,
    panel_summary,
)


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/frontier_ml_signal_rebuild")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/research_log/2026-06-04_frontier_ml_signal_rebuild.md")
DEFAULT_YEARS = tuple(range(2017, 2027))
DEFAULT_FACTOR_SET = "expanded"
DEFAULT_MAX_TRAIN_YEARS = 5
DEFAULT_MIN_TRAIN_YEARS = 1
TRAINING_MODES = ("baseline", "weak_weighted", "regime_weighted", "regime_heads")
DEFAULT_TRAINING_MODE = "baseline"
DEFAULT_WEAK_YEARS = (2017, 2018, 2022, 2023)
DEFAULT_WEAK_SAMPLE_WEIGHT = 2.0
DEFAULT_REGIME_SAMPLE_WEIGHT = 1.5
DEFAULT_MIN_REGIME_HEAD_ROWS = 100
DEFAULT_MODEL_N_JOBS = 4
DEFAULT_WRITE_FULL_PREDICTION_CSV = True
ML_SIGNAL_NAME_TEMPLATE = "ml_lgbm_xsec_excess_score_h{horizon}_prior_fit"
DEFAULT_MODEL_PARAMS: dict[str, Any] = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": DEFAULT_MODEL_N_JOBS,
}


def run_frontier_ml_signal_rebuild(
    *,
    root: str | None = None,
    years: Sequence[int] = DEFAULT_YEARS,
    final_end_date: str | None = DEFAULT_FINAL_END_DATE,
    horizon: int = DEFAULT_HORIZON,
    factor_set: str | None = DEFAULT_FACTOR_SET,
    max_train_years: int = DEFAULT_MAX_TRAIN_YEARS,
    min_train_years: int = DEFAULT_MIN_TRAIN_YEARS,
    training_mode: str = DEFAULT_TRAINING_MODE,
    weak_years: Sequence[int] = DEFAULT_WEAK_YEARS,
    weak_sample_weight: float = DEFAULT_WEAK_SAMPLE_WEIGHT,
    regime_sample_weight: float = DEFAULT_REGIME_SAMPLE_WEIGHT,
    min_regime_head_rows: int = DEFAULT_MIN_REGIME_HEAD_ROWS,
    model_n_jobs: int | None = DEFAULT_MODEL_N_JOBS,
    model_params: Mapping[str, Any] | None = None,
    write_prediction_shards: bool = True,
    write_full_prediction_csv: bool = DEFAULT_WRITE_FULL_PREDICTION_CSV,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: str | Path = DEFAULT_RESEARCH_LOG,
    model_class: Any | None = None,
) -> dict[str, Any]:
    """Train one prior-fit LightGBM model per eval year and write prediction artifacts."""

    selected_years = _parse_int_values(years, name="years")
    if not selected_years:
        raise ValueError("years must not be empty")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if max_train_years <= 0:
        raise ValueError("max_train_years must be positive")
    if min_train_years <= 0:
        raise ValueError("min_train_years must be positive")
    if max_train_years < min_train_years:
        raise ValueError("max_train_years must be >= min_train_years")
    selected_training_mode = normalize_training_mode(training_mode)
    selected_weak_years = _parse_int_values(weak_years, name="weak_years")
    if weak_sample_weight <= 0 or regime_sample_weight <= 0:
        raise ValueError("sample weights must be positive")
    if min_regime_head_rows <= 0:
        raise ValueError("min_regime_head_rows must be positive")
    if model_n_jobs == 0:
        raise ValueError("model_n_jobs must be non-zero or None")
    selected_factor_set = normalize_factor_set(factor_set)
    if selected_factor_set != "expanded":
        raise ValueError("frontier_ml_signal_rebuild v1 requires factor_set=expanded")

    run_id = f"frontier_ml_signal_rebuild_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    lgbm_class = model_class if model_class is not None else _load_lgbm_regressor()
    ml_signal_name = ML_SIGNAL_NAME_TEMPLATE.format(horizon=int(horizon))
    params = {**DEFAULT_MODEL_PARAMS, **dict(model_params or {})}
    if model_n_jobs is not None:
        params["n_jobs"] = int(model_n_jobs)
    if lgbm_class is None:
        summary = _dependency_missing_summary(
            run_id=run_id,
            run_dir=run_dir,
            years=selected_years,
            final_end_date=final_end_date,
            horizon=horizon,
            factor_set=selected_factor_set,
            ml_signal_name=ml_signal_name,
            model_params=params,
            training_mode=selected_training_mode,
            weak_years=selected_weak_years,
            weak_sample_weight=weak_sample_weight,
            regime_sample_weight=regime_sample_weight,
            min_regime_head_rows=min_regime_head_rows,
        )
        _write_artifacts(
            run_dir,
            summary,
            plan=pd.DataFrame(columns=_plan_columns()),
            predictions=pd.DataFrame(columns=_prediction_columns()),
            importance=pd.DataFrame(columns=_importance_columns()),
            training_audit=pd.DataFrame(columns=_training_audit_columns()),
            regime_audit=pd.DataFrame(columns=_regime_audit_columns()),
            markdown=render_ml_signal_markdown(summary, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()),
            write_full_prediction_csv=True,
            write_research_log=write_research_log,
            research_log_path=research_log_path,
        )
        return {**summary, "run_dir": str(run_dir), "research_log": str(research_log_path) if write_research_log else None}

    manifest = load_pit_manifest(root)
    quality = load_quality_report(root)
    panel_start = f"{min(selected_years) - int(max_train_years)}-01-01"
    panel_end = _panel_end_date(selected_years, final_end_date)
    raw_panel = load_tradeable_panel(
        root,
        start_date=panel_start,
        end_date=panel_end,
        include_metrics=True,
    )
    raw_panel_summary = panel_summary(raw_panel)
    factor_panel = build_ml_factor_panel(raw_panel, horizon=horizon, factor_set=selected_factor_set)
    feature_columns = [f"{column}_z" for column in factor_columns_for_set(selected_factor_set)]
    target_col = f"xsec_excess_ret_{int(horizon)}d"
    factor_panel = reduce_ml_factor_panel_memory(factor_panel, feature_columns=feature_columns, target_col=target_col)
    del raw_panel
    gc.collect()
    plan, predictions, importance, training_audit, regime_audit = build_ml_signal_artifacts(
        factor_panel,
        years=selected_years,
        final_end_date=final_end_date,
        horizon=horizon,
        feature_columns=feature_columns,
        target_col=target_col,
        model_class=lgbm_class,
        model_params=params,
        max_train_years=max_train_years,
        min_train_years=min_train_years,
        training_mode=selected_training_mode,
        weak_years=selected_weak_years,
        weak_sample_weight=weak_sample_weight,
        regime_sample_weight=regime_sample_weight,
        min_regime_head_rows=min_regime_head_rows,
        prediction_shard_dir=run_dir / "ml_signal_predictions_by_year" if write_prediction_shards else None,
        collect_predictions=bool(write_full_prediction_csv),
        ml_signal_name=ml_signal_name,
    )
    summary = summarize_ml_signal_rebuild(
        plan,
        predictions,
        importance,
        training_audit,
        run_id=run_id,
        run_dir=run_dir,
        manifest=manifest,
        quality=quality,
        raw_panel=raw_panel_summary,
        factor_panel=factor_panel,
        years=selected_years,
        final_end_date=final_end_date,
        horizon=horizon,
        factor_set=selected_factor_set,
        feature_columns=feature_columns,
        target_col=target_col,
        max_train_years=max_train_years,
        min_train_years=min_train_years,
        training_mode=selected_training_mode,
        weak_years=selected_weak_years,
        weak_sample_weight=weak_sample_weight,
        regime_sample_weight=regime_sample_weight,
        min_regime_head_rows=min_regime_head_rows,
        model_params=params,
        prediction_artifact_mode=_prediction_artifact_mode(
            write_prediction_shards=write_prediction_shards,
            write_full_prediction_csv=write_full_prediction_csv,
        ),
        prediction_shard_dir=run_dir / "ml_signal_predictions_by_year" if write_prediction_shards else None,
        write_full_prediction_csv=write_full_prediction_csv,
        ml_signal_name=ml_signal_name,
    )
    markdown = render_ml_signal_markdown(summary, plan, predictions, training_audit)
    _write_artifacts(
        run_dir,
        summary,
        plan=plan,
        predictions=predictions,
        importance=importance,
        training_audit=training_audit,
        regime_audit=regime_audit,
        markdown=markdown,
        write_full_prediction_csv=write_full_prediction_csv,
        write_research_log=write_research_log,
        research_log_path=research_log_path,
    )
    return {**summary, "run_dir": str(run_dir), "research_log": str(research_log_path) if write_research_log else None}


def build_ml_factor_panel(
    raw_panel: pd.DataFrame,
    *,
    horizon: int,
    factor_set: str | None = DEFAULT_FACTOR_SET,
) -> pd.DataFrame:
    """Build expanded z-score features and xsec excess labels from a tradeable panel."""

    if raw_panel.empty:
        return pd.DataFrame()
    selected_factor_set = normalize_factor_set(factor_set)
    factor_panel = build_factor_label_panel(raw_panel, horizons=tuple(sorted({1, 5, 20, int(horizon)})), factor_set=selected_factor_set)
    factor_panel = add_cross_sectional_excess_return_labels(factor_panel, horizons=(int(horizon),))
    raw_factor_columns = factor_columns_for_set(selected_factor_set)
    factor_panel = add_cross_sectional_zscores(factor_panel, raw_factor_columns)
    factor_panel = add_ml_market_regime_features(factor_panel)
    factor_panel["date"] = pd.to_datetime(factor_panel["date"])
    return factor_panel.sort_values(["date", "code"]).reset_index(drop=True)


def reduce_ml_factor_panel_memory(
    factor_panel: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_col: str,
) -> pd.DataFrame:
    """Keep only ML-required columns and downcast numeric model inputs."""

    if factor_panel.empty:
        return factor_panel
    keep_columns = [
        "date",
        "code",
        target_col,
        *feature_columns,
        "ml_market_ret_20d_mean",
        "ml_breadth_20d_positive_rate",
        "ml_market_volatility_20d_mean",
    ]
    output = factor_panel.loc[:, [column for column in keep_columns if column in factor_panel.columns]].copy()
    numeric_columns = [
        column
        for column in output.columns
        if column not in {"date", "code"} and pd.api.types.is_numeric_dtype(output[column])
    ]
    for column in numeric_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce").astype("float32")
    return output


def add_ml_market_regime_features(factor_panel: pd.DataFrame) -> pd.DataFrame:
    """Attach same-day broad market regime diagnostics used by ML v2 training."""

    if factor_panel.empty:
        return factor_panel.copy()
    required = {"date", "code", "ret_20d", "volatility_20d"}
    if missing := sorted(required - set(factor_panel.columns)):
        raise ValueError(f"factor panel missing ML regime columns: {missing}")
    output = factor_panel.copy()
    output["date"] = pd.to_datetime(output["date"])
    work = output.loc[:, ["date", "code", "ret_20d", "volatility_20d"]].copy()
    work["ret_20d"] = pd.to_numeric(work["ret_20d"], errors="coerce")
    work["volatility_20d"] = pd.to_numeric(work["volatility_20d"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for date, group in work.groupby("date", sort=True):
        ret20 = group["ret_20d"]
        vol20 = group["volatility_20d"]
        rows.append(
            {
                "date": pd.Timestamp(date),
                "ml_market_ret_20d_mean": _nanmean(ret20),
                "ml_breadth_20d_positive_rate": _positive_rate(ret20),
                "ml_market_volatility_20d_mean": _nanmean(vol20),
            }
        )
    regime = pd.DataFrame(rows)
    return output.merge(regime, on="date", how="left")


def build_ml_signal_artifacts(
    factor_panel: pd.DataFrame,
    *,
    years: Sequence[int],
    final_end_date: str | None,
    horizon: int,
    feature_columns: Sequence[str],
    target_col: str,
    model_class: Any,
    model_params: Mapping[str, Any],
    max_train_years: int,
    min_train_years: int,
    training_mode: str = DEFAULT_TRAINING_MODE,
    weak_years: Sequence[int] = DEFAULT_WEAK_YEARS,
    weak_sample_weight: float = DEFAULT_WEAK_SAMPLE_WEIGHT,
    regime_sample_weight: float = DEFAULT_REGIME_SAMPLE_WEIGHT,
    min_regime_head_rows: int = DEFAULT_MIN_REGIME_HEAD_ROWS,
    prediction_shard_dir: str | Path | None = None,
    collect_predictions: bool = True,
    ml_signal_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return plan, predictions, feature importance, and training audit frames."""

    selected_training_mode = normalize_training_mode(training_mode)
    selected_weak_years = _parse_int_values(weak_years, name="weak_years")
    plan_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    regime_audit_rows: list[dict[str, Any]] = []
    shard_dir = Path(prediction_shard_dir) if prediction_shard_dir is not None else None
    if shard_dir is not None:
        shard_dir.mkdir(parents=True, exist_ok=True)
    if factor_panel.empty:
        return (
            pd.DataFrame(columns=_plan_columns()),
            pd.DataFrame(columns=_prediction_columns()),
            pd.DataFrame(columns=_importance_columns()),
            pd.DataFrame(columns=_training_audit_columns()),
            pd.DataFrame(columns=_regime_audit_columns()),
        )

    panel = ensure_ml_regime_columns(factor_panel)
    panel["date"] = pd.to_datetime(panel["date"])
    trading_dates = sorted(panel["date"].dropna().drop_duplicates().tolist())
    feature_columns = tuple(feature_columns)
    required = {"date", "code", target_col, *feature_columns}
    if missing := sorted(required - set(panel.columns)):
        raise ValueError(f"factor panel missing required columns: {missing}")

    for eval_year in years:
        year_prediction_frames: list[pd.DataFrame] = []
        windows = _ml_year_windows(int(eval_year), final_end_date=final_end_date, max_train_years=max_train_years)
        label_cutoff = label_cutoff_date(trading_dates, windows["fit_end_date"], horizon=horizon)
        train_start = pd.Timestamp(windows["train_start_date"])
        eval_start = pd.Timestamp(windows["start_date"])
        eval_end = pd.Timestamp(windows["end_date"])
        if label_cutoff is None:
            status = "skipped/insufficient_label_calendar"
            train = pd.DataFrame()
        else:
            train = panel.loc[
                panel["date"].between(train_start, label_cutoff)
                & panel[list(feature_columns)].notna().all(axis=1)
                & pd.to_numeric(panel[target_col], errors="coerce").notna()
            ].copy()
            train_years = sorted(train["date"].dt.year.dropna().astype(int).unique().tolist()) if not train.empty else []
            status = "ready" if len(train_years) >= int(min_train_years) and not train.empty else "skipped/insufficient_training_data"

        train_years = sorted(train["date"].dt.year.dropna().astype(int).unique().tolist()) if not train.empty else []
        eval_frame = panel.loc[
            panel["date"].between(eval_start, eval_end)
            & panel[list(feature_columns)].notna().all(axis=1)
        ].copy()
        common = {
            "eval_year": int(eval_year),
            "horizon": int(horizon),
            "ml_signal_name": ml_signal_name,
            "training_mode": selected_training_mode,
            "train_start_date": windows["train_start_date"],
            "fit_end_date": windows["fit_end_date"],
            "label_cutoff_date": label_cutoff.strftime("%Y-%m-%d") if label_cutoff is not None else "",
            "eval_start_date": windows["start_date"],
            "eval_end_date": windows["end_date"],
            "train_years": ",".join(str(year) for year in train_years),
            "train_year_count": int(len(train_years)),
            "train_row_count": int(len(train)),
            "eval_feature_row_count": int(len(eval_frame)),
            "feature_count": int(len(feature_columns)),
            "features": ",".join(feature_columns),
            "target_col": target_col,
            "sample_weight_policy": sample_weight_policy_for_mode(selected_training_mode),
            "weak_years": ",".join(str(year) for year in selected_weak_years),
            "weak_sample_weight": float(weak_sample_weight),
            "regime_sample_weight": float(regime_sample_weight),
            "min_regime_head_rows": int(min_regime_head_rows),
            "fit_uses_eval_year": False,
            "evidence_grade": "diagnostic_ml_prior_fit",
            "status": status,
        }
        regime_thresholds = fit_regime_thresholds(train) if status == "ready" else empty_regime_thresholds()
        train_labeled = label_regime_frame(train, regime_thresholds) if status == "ready" else train.copy()
        eval_labeled = label_regime_frame(eval_frame, regime_thresholds) if status == "ready" else eval_frame.copy()
        if status == "ready" and selected_training_mode == "regime_heads":
            head_counts = train_labeled["ml_regime_label"].value_counts()
            ready_heads = [head for head in ("weak_regime", "normal_regime") if int(head_counts.get(head, 0)) >= int(min_regime_head_rows)]
            if len(ready_heads) < 2:
                status = "skipped/insufficient_regime_heads"
                common["status"] = status
        plan_rows.append({**common, **_threshold_plan_fields(regime_thresholds)})
        audit = {
            **common,
            **_threshold_plan_fields(regime_thresholds),
            "model_class": getattr(model_class, "__name__", str(model_class)),
            "prediction_row_count": 0,
            "target_mean": np.nan,
            "target_std": np.nan,
            "sample_weight_mean": np.nan,
            "sample_weight_max": np.nan,
            "train_weak_regime_rate": _regime_rate(train_labeled),
            "eval_weak_regime_rate": _regime_rate(eval_labeled),
        }
        regime_audit_rows.extend(
            build_regime_audit_rows(
                train_labeled,
                eval_labeled,
                eval_year=int(eval_year),
                horizon=horizon,
                ml_signal_name=ml_signal_name,
                training_mode=selected_training_mode,
                thresholds=regime_thresholds,
                status=status,
            )
        )
        if status != "ready":
            audit_rows.append(audit)
            _write_prediction_year_shard(
                shard_dir,
                eval_year=int(eval_year),
                predictions=pd.DataFrame(columns=_prediction_columns()),
            )
            gc.collect()
            continue

        x_train = train_labeled.loc[:, list(feature_columns)]
        y_train = pd.to_numeric(train_labeled[target_col], errors="coerce")
        sample_weight = build_sample_weight(
            train_labeled,
            mode=selected_training_mode,
            weak_years=selected_weak_years,
            weak_sample_weight=weak_sample_weight,
            regime_sample_weight=regime_sample_weight,
        )
        prediction_count = 0
        if selected_training_mode == "regime_heads":
            for head in ("weak_regime", "normal_regime"):
                head_train = train_labeled.loc[train_labeled["ml_regime_label"].eq(head)].copy()
                head_eval = eval_labeled.loc[eval_labeled["ml_regime_label"].eq(head)].copy()
                if len(head_train) < int(min_regime_head_rows):
                    continue
                model = model_class(**dict(model_params))
                head_x_train = head_train.loc[:, list(feature_columns)]
                head_y_train = pd.to_numeric(head_train[target_col], errors="coerce")
                fit_model(model, head_x_train, head_y_train, sample_weight=None)
                prediction_count += append_predictions_and_importance(
                    model,
                    head_eval,
                    feature_columns=feature_columns,
                    prediction_frames=year_prediction_frames,
                    importance_rows=importance_rows,
                    eval_year=int(eval_year),
                    horizon=horizon,
                    ml_signal_name=ml_signal_name,
                    training_mode=selected_training_mode,
                    model_head=head,
                    sample_weight_policy=sample_weight_policy_for_mode(selected_training_mode),
                )
        else:
            model = model_class(**dict(model_params))
            fit_model(model, x_train, y_train, sample_weight=sample_weight)
            prediction_count = append_predictions_and_importance(
                model,
                eval_labeled,
                feature_columns=feature_columns,
                prediction_frames=year_prediction_frames,
                importance_rows=importance_rows,
                eval_year=int(eval_year),
                horizon=horizon,
                ml_signal_name=ml_signal_name,
                training_mode=selected_training_mode,
                model_head="global",
                sample_weight_policy=sample_weight_policy_for_mode(selected_training_mode),
            )
        audit["prediction_row_count"] = int(prediction_count)
        audit["target_mean"] = float(y_train.mean())
        audit["target_std"] = float(y_train.std(ddof=0))
        audit["sample_weight_mean"] = float(sample_weight.mean()) if sample_weight is not None and len(sample_weight) else np.nan
        audit["sample_weight_max"] = float(sample_weight.max()) if sample_weight is not None and len(sample_weight) else np.nan
        audit_rows.append(audit)
        year_predictions = _concat_prediction_frames(year_prediction_frames)
        _write_prediction_year_shard(shard_dir, eval_year=int(eval_year), predictions=year_predictions)
        if collect_predictions and not year_predictions.empty:
            prediction_frames.append(year_predictions)
        gc.collect()

    predictions = _concat_prediction_frames(prediction_frames)
    return (
        pd.DataFrame(plan_rows, columns=_plan_columns()),
        predictions.loc[:, _prediction_columns()],
        pd.DataFrame(importance_rows, columns=_importance_columns()),
        pd.DataFrame(audit_rows, columns=_training_audit_columns()),
        pd.DataFrame(regime_audit_rows, columns=_regime_audit_columns()),
    )


def append_predictions_and_importance(
    model: Any,
    eval_frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    prediction_frames: list[pd.DataFrame],
    importance_rows: list[dict[str, Any]],
    eval_year: int,
    horizon: int,
    ml_signal_name: str,
    training_mode: str,
    model_head: str,
    sample_weight_policy: str,
) -> int:
    if not eval_frame.empty:
        scores = model.predict(eval_frame.loc[:, list(feature_columns)])
        predictions = eval_frame.loc[:, ["date", "code"]].copy()
        predictions.insert(0, "eval_year", int(eval_year))
        predictions["horizon"] = int(horizon)
        predictions["ml_signal_name"] = ml_signal_name
        predictions["score"] = np.asarray(scores, dtype=float)
        predictions["fit_uses_eval_year"] = False
        predictions["feature_count"] = int(len(feature_columns))
        predictions["training_mode"] = training_mode
        predictions["model_head"] = model_head
        predictions["regime_label"] = eval_frame.get("ml_regime_label", pd.Series("", index=eval_frame.index)).astype(str).to_numpy()
        predictions["sample_weight_policy"] = sample_weight_policy
        predictions["model_status"] = "ready"
        predictions["evidence_grade"] = "diagnostic_ml_prior_fit"
        prediction_frames.append(predictions.loc[:, _prediction_columns()])
        prediction_count = int(len(predictions))
    else:
        prediction_count = 0
    importances = getattr(model, "feature_importances_", np.zeros(len(feature_columns)))
    for feature, importance in zip(feature_columns, importances, strict=False):
        importance_rows.append(
            {
                "eval_year": int(eval_year),
                "horizon": int(horizon),
                "ml_signal_name": ml_signal_name,
                "training_mode": training_mode,
                "model_head": model_head,
                "feature": feature,
                "importance": float(importance),
                "fit_uses_eval_year": False,
                "evidence_grade": "diagnostic_ml_prior_fit",
            }
        )
    return prediction_count


def ensure_ml_regime_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep low-level tests and legacy callers working when regime fields are absent."""

    output = frame.copy()
    defaults = {
        "ml_market_ret_20d_mean": 0.0,
        "ml_breadth_20d_positive_rate": 0.5,
        "ml_market_volatility_20d_mean": 0.0,
    }
    for column, value in defaults.items():
        if column not in output.columns:
            output[column] = value
    return output


def fit_model(model: Any, x_train: pd.DataFrame, y_train: pd.Series, *, sample_weight: np.ndarray | None) -> Any:
    if sample_weight is None:
        return model.fit(x_train, y_train)
    try:
        parameters = inspect.signature(model.fit).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "sample_weight" in parameters:
        return model.fit(x_train, y_train, sample_weight=sample_weight)
    return model.fit(x_train, y_train)


def normalize_training_mode(value: str | None) -> str:
    mode = (value or DEFAULT_TRAINING_MODE).strip().lower()
    if mode not in TRAINING_MODES:
        raise ValueError(f"unsupported training_mode: {value}")
    return mode


def sample_weight_policy_for_mode(mode: str) -> str:
    if mode == "weak_weighted":
        return "fixed_weak_year_or_weak_regime_weight"
    if mode == "regime_weighted":
        return "same_day_market_regime_weight"
    if mode == "regime_heads":
        return "separate_regime_heads"
    return "none"


def fit_regime_thresholds(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return empty_regime_thresholds()
    return {
        "ret_20d_threshold": _median_or_nan(frame["ml_market_ret_20d_mean"]),
        "breadth_20d_threshold": _median_or_nan(frame["ml_breadth_20d_positive_rate"]),
        "volatility_20d_threshold": _median_or_nan(frame["ml_market_volatility_20d_mean"]),
    }


def empty_regime_thresholds() -> dict[str, float]:
    return {
        "ret_20d_threshold": np.nan,
        "breadth_20d_threshold": np.nan,
        "volatility_20d_threshold": np.nan,
    }


def label_regime_frame(frame: pd.DataFrame, thresholds: Mapping[str, float]) -> pd.DataFrame:
    output = frame.copy()
    if output.empty:
        output["ml_weak_regime"] = False
        output["ml_regime_label"] = ""
        return output
    ret_threshold = float(thresholds.get("ret_20d_threshold", np.nan))
    breadth_threshold = float(thresholds.get("breadth_20d_threshold", np.nan))
    volatility_threshold = float(thresholds.get("volatility_20d_threshold", np.nan))
    weak_ret = (
        pd.to_numeric(output["ml_market_ret_20d_mean"], errors="coerce").le(ret_threshold)
        if not np.isnan(ret_threshold)
        else pd.Series(False, index=output.index)
    )
    weak_breadth = (
        pd.to_numeric(output["ml_breadth_20d_positive_rate"], errors="coerce").le(breadth_threshold)
        if not np.isnan(breadth_threshold)
        else pd.Series(False, index=output.index)
    )
    high_vol = (
        pd.to_numeric(output["ml_market_volatility_20d_mean"], errors="coerce").ge(volatility_threshold)
        if not np.isnan(volatility_threshold)
        else pd.Series(False, index=output.index)
    )
    output["ml_weak_regime"] = ((weak_ret & weak_breadth) | (weak_breadth & high_vol)).fillna(False)
    output["ml_regime_label"] = np.where(output["ml_weak_regime"], "weak_regime", "normal_regime")
    return output


def build_sample_weight(
    train: pd.DataFrame,
    *,
    mode: str,
    weak_years: Sequence[int],
    weak_sample_weight: float,
    regime_sample_weight: float,
) -> np.ndarray | None:
    if mode == "baseline" or mode == "regime_heads":
        return None
    dates = pd.to_datetime(train["date"])
    weights = np.ones(len(train), dtype=float)
    if mode == "weak_weighted":
        weak_year_mask = dates.dt.year.isin([int(year) for year in weak_years]).to_numpy()
        weak_regime_mask = _truthy(train.get("ml_weak_regime", pd.Series(False, index=train.index))).to_numpy()
        weights[weak_year_mask | weak_regime_mask] = float(weak_sample_weight)
        return weights
    if mode == "regime_weighted":
        weak_regime_mask = _truthy(train.get("ml_weak_regime", pd.Series(False, index=train.index))).to_numpy()
        weights[weak_regime_mask] = float(regime_sample_weight)
        return weights
    raise ValueError(f"unsupported training mode: {mode}")


def build_regime_audit_rows(
    train: pd.DataFrame,
    eval_frame: pd.DataFrame,
    *,
    eval_year: int,
    horizon: int,
    ml_signal_name: str,
    training_mode: str,
    thresholds: Mapping[str, float],
    status: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = ("weak_regime", "normal_regime")
    for label in labels:
        train_subset = train.loc[train.get("ml_regime_label", pd.Series("", index=train.index)).astype(str).eq(label)]
        eval_subset = eval_frame.loc[eval_frame.get("ml_regime_label", pd.Series("", index=eval_frame.index)).astype(str).eq(label)]
        rows.append(
            {
                "eval_year": int(eval_year),
                "horizon": int(horizon),
                "ml_signal_name": ml_signal_name,
                "training_mode": training_mode,
                "regime_label": label,
                "train_row_count": int(len(train_subset)),
                "eval_row_count": int(len(eval_subset)),
                "train_date_count": int(train_subset["date"].nunique()) if "date" in train_subset.columns else 0,
                "eval_date_count": int(eval_subset["date"].nunique()) if "date" in eval_subset.columns else 0,
                **_threshold_plan_fields(thresholds),
                "fit_uses_eval_year": False,
                "status": status,
            }
        )
    return rows


def _threshold_plan_fields(thresholds: Mapping[str, float]) -> dict[str, float]:
    return {
        "regime_ret_20d_threshold": float(thresholds.get("ret_20d_threshold", np.nan)),
        "regime_breadth_20d_threshold": float(thresholds.get("breadth_20d_threshold", np.nan)),
        "regime_volatility_20d_threshold": float(thresholds.get("volatility_20d_threshold", np.nan)),
    }


def _regime_rate(frame: pd.DataFrame) -> float:
    if frame.empty or "ml_weak_regime" not in frame.columns:
        return np.nan
    return float(_truthy(frame["ml_weak_regime"]).mean())


def label_cutoff_date(trading_dates: Sequence[pd.Timestamp], fit_end_date: str, *, horizon: int) -> pd.Timestamp | None:
    """Return the last training date whose forward label is known by fit_end_date."""

    dates = [pd.Timestamp(date) for date in trading_dates if pd.Timestamp(date) <= pd.Timestamp(fit_end_date)]
    if len(dates) <= int(horizon):
        return None
    return pd.Timestamp(dates[-(int(horizon) + 1)])


def summarize_ml_signal_rebuild(
    plan: pd.DataFrame,
    predictions: pd.DataFrame,
    importance: pd.DataFrame,
    training_audit: pd.DataFrame,
    *,
    run_id: str,
    run_dir: Path,
    manifest: Mapping[str, Any],
    quality: Mapping[str, Any],
    raw_panel: pd.DataFrame | Mapping[str, Any],
    factor_panel: pd.DataFrame | Mapping[str, Any],
    years: Sequence[int],
    final_end_date: str | None,
    horizon: int,
    factor_set: str,
    feature_columns: Sequence[str],
    target_col: str,
    max_train_years: int,
    min_train_years: int,
    training_mode: str,
    weak_years: Sequence[int],
    weak_sample_weight: float,
    regime_sample_weight: float,
    min_regime_head_rows: int,
    model_params: Mapping[str, Any],
    prediction_artifact_mode: str,
    prediction_shard_dir: str | Path | None,
    write_full_prediction_csv: bool,
    ml_signal_name: str,
) -> dict[str, Any]:
    fit_uses = int(_truthy(plan.get("fit_uses_eval_year", pd.Series(dtype=bool))).sum()) if not plan.empty else 0
    ready_years = plan.loc[plan.get("status", pd.Series(dtype=str)).astype(str).eq("ready"), "eval_year"].tolist() if not plan.empty else []
    prediction_row_count = _prediction_row_count(predictions, training_audit)
    raw_panel_summary = dict(raw_panel) if isinstance(raw_panel, Mapping) else panel_summary(raw_panel)
    factor_panel_summary = dict(factor_panel) if isinstance(factor_panel, Mapping) else panel_summary(factor_panel)
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "north_star": "Baostock-only personal quant strategy research",
        "objective": "build a prior-fit LightGBM cross-sectional excess-return signal for existing personal gates",
        "snapshot_id": manifest.get("snapshot_id"),
        "years": [int(year) for year in years],
        "final_end_date": final_end_date,
        "horizon": int(horizon),
        "factor_set": factor_set,
        "target_col": target_col,
        "ml_signal_name": ml_signal_name,
        "model_family": "lightgbm.LGBMRegressor",
        "model_params": dict(model_params),
        "training_mode": training_mode,
        "sample_weight_policy": sample_weight_policy_for_mode(training_mode),
        "weak_years": [int(year) for year in weak_years],
        "weak_sample_weight": float(weak_sample_weight),
        "regime_sample_weight": float(regime_sample_weight),
        "min_regime_head_rows": int(min_regime_head_rows),
        "max_train_years": int(max_train_years),
        "min_train_years": int(min_train_years),
        "feature_columns": list(feature_columns),
        "feature_count": int(len(feature_columns)),
        "raw_panel": raw_panel_summary,
        "factor_panel": factor_panel_summary,
        "plan_rows": int(len(plan)),
        "ready_eval_years": [int(year) for year in ready_years],
        "ready_eval_year_count": int(len(ready_years)),
        "prediction_rows": prediction_row_count,
        "prediction_artifact_mode": prediction_artifact_mode,
        "prediction_shard_dir": str(prediction_shard_dir or ""),
        "write_full_prediction_csv": bool(write_full_prediction_csv),
        "feature_importance_rows": int(len(importance)),
        "training_audit_rows": int(len(training_audit)),
        "fit_uses_eval_year_count": fit_uses,
        "dependency_status": "available",
        "candidate_count": 0,
        "personal_backtest_candidate_count": 0,
        "personal_paper_candidate_count": 0,
        "strategy_candidate_count": 0,
        "decision": "diagnostic_ml_signal_ready" if prediction_row_count > 0 and fit_uses == 0 else "diagnostic_ml_signal_not_ready",
        "evidence_grade": "diagnostic_ml_prior_fit",
        "quality": {
            "failure_count": quality.get("failure_count"),
            "missing_bar_rows": quality.get("missing_bar_rows"),
            "st_rows": quality.get("st_rows"),
            "suspended_like_rows": quality.get("suspended_like_rows"),
        },
        "limitations": [
            "This run only creates a diagnostic ML signal; it never promotes a candidate by itself.",
            "Promotion requires the existing 2017-2026 formal personal protocol grid and personal candidate gate.",
            "The signal is Baostock-only and does not use true market-cap or float-cap fields.",
        ],
        "output_dir": str(run_dir),
    }


def render_ml_signal_markdown(
    summary: Mapping[str, Any],
    plan: pd.DataFrame,
    predictions: pd.DataFrame,
    training_audit: pd.DataFrame,
) -> str:
    lines = [
        "# Frontier ML Signal Rebuild",
        "",
        f"- run_id: `{summary.get('run_id', '')}`",
        f"- decision: `{summary.get('decision', '')}`",
        f"- ml_signal_name: `{summary.get('ml_signal_name', '')}`",
        f"- model_family: `{summary.get('model_family', '')}`",
        f"- training_mode: `{summary.get('training_mode', '')}`",
        f"- sample_weight_policy: `{summary.get('sample_weight_policy', '')}`",
        f"- years: `{summary.get('years')}`",
        f"- horizon: `{summary.get('horizon')}`",
        f"- factor_set: `{summary.get('factor_set')}`",
        f"- target_col: `{summary.get('target_col', '')}`",
        f"- feature_count: `{summary.get('feature_count', 0)}`",
        f"- ready_eval_year_count: `{summary.get('ready_eval_year_count', 0)}`",
        f"- prediction_rows: `{summary.get('prediction_rows', 0)}`",
        f"- fit_uses_eval_year_count: `{summary.get('fit_uses_eval_year_count', 0)}`",
        f"- candidate_count: `{summary.get('candidate_count', 0)}`",
        f"- Artifacts: `{summary.get('output_dir', '')}`",
        "",
        "## Plan",
        "",
        _markdown_table(plan),
        "",
        "## Training Audit",
        "",
        _markdown_table(training_audit),
        "",
        "## Prediction Sample",
        "",
        _markdown_table(predictions.head(20) if not predictions.empty else predictions),
        "",
        "## Interpretation",
        "",
        "This experiment creates a prior-fit Baostock-only ML signal for downstream formal gates. It is not a promotion artifact by itself.",
        "",
    ]
    return "\n".join(lines)


def _dependency_missing_summary(
    *,
    run_id: str,
    run_dir: Path,
    years: Sequence[int],
    final_end_date: str | None,
    horizon: int,
    factor_set: str,
    ml_signal_name: str,
    model_params: Mapping[str, Any],
    training_mode: str,
    weak_years: Sequence[int],
    weak_sample_weight: float,
    regime_sample_weight: float,
    min_regime_head_rows: int,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "north_star": "Baostock-only personal quant strategy research",
        "objective": "build a prior-fit LightGBM cross-sectional excess-return signal for existing personal gates",
        "snapshot_id": None,
        "years": [int(year) for year in years],
        "final_end_date": final_end_date,
        "horizon": int(horizon),
        "factor_set": factor_set,
        "target_col": f"xsec_excess_ret_{int(horizon)}d",
        "ml_signal_name": ml_signal_name,
        "model_family": "lightgbm.LGBMRegressor",
        "model_params": dict(model_params),
        "training_mode": training_mode,
        "sample_weight_policy": sample_weight_policy_for_mode(training_mode),
        "weak_years": [int(year) for year in weak_years],
        "weak_sample_weight": float(weak_sample_weight),
        "regime_sample_weight": float(regime_sample_weight),
        "min_regime_head_rows": int(min_regime_head_rows),
        "ready_eval_year_count": 0,
        "prediction_rows": 0,
        "fit_uses_eval_year_count": 0,
        "dependency_status": "missing",
        "candidate_count": 0,
        "personal_backtest_candidate_count": 0,
        "personal_paper_candidate_count": 0,
        "strategy_candidate_count": 0,
        "decision": "skipped/dependency_missing",
        "evidence_grade": "dependency_missing",
        "limitations": ["LightGBM is missing, so no ML predictions were created."],
        "output_dir": str(run_dir),
    }


def _write_artifacts(
    run_dir: Path,
    summary: Mapping[str, Any],
    *,
    plan: pd.DataFrame,
    predictions: pd.DataFrame,
    importance: pd.DataFrame,
    training_audit: pd.DataFrame,
    regime_audit: pd.DataFrame,
    markdown: str,
    write_full_prediction_csv: bool,
    write_research_log: bool,
    research_log_path: str | Path,
) -> None:
    plan.to_csv(run_dir / "ml_signal_plan.csv", index=False, encoding="utf-8-sig")
    prediction_output = predictions if write_full_prediction_csv else pd.DataFrame(columns=_prediction_columns())
    prediction_output.to_csv(run_dir / "ml_signal_predictions.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(run_dir / "ml_feature_importance.csv", index=False, encoding="utf-8-sig")
    training_audit.to_csv(run_dir / "ml_training_audit.csv", index=False, encoding="utf-8-sig")
    regime_audit.to_csv(run_dir / "ml_regime_training_audit.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        path = Path(research_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")


def _load_lgbm_regressor() -> Any | None:
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        return None
    return LGBMRegressor


def _ml_year_windows(year: int, *, final_end_date: str | None, max_train_years: int) -> dict[str, str]:
    if year <= 1900:
        raise ValueError("year must be after 1900")
    end_date = f"{year}-12-31"
    if final_end_date:
        final = pd.Timestamp(final_end_date)
        if year == int(final.year):
            end_date = final.strftime("%Y-%m-%d")
        elif year > int(final.year):
            raise ValueError(f"year {year} is after final_end_date {final_end_date}")
    return {
        "train_start_date": f"{year - int(max_train_years)}-01-01",
        "fit_end_date": f"{year - 1}-12-31",
        "start_date": f"{year}-01-01",
        "end_date": end_date,
    }


def _panel_end_date(years: Sequence[int], final_end_date: str | None) -> str:
    max_year = max(int(year) for year in years)
    if final_end_date and max_year == int(pd.Timestamp(final_end_date).year):
        return pd.Timestamp(final_end_date).strftime("%Y-%m-%d")
    return f"{max_year}-12-31"


def _parse_int_values(values: Sequence[int] | str, *, name: str) -> tuple[int, ...]:
    parts = values.split(",") if isinstance(values, str) else list(values)
    parsed = tuple(int(str(value).strip()) for value in parts if str(value).strip())
    if not parsed:
        raise ValueError(f"{name} must not be empty")
    return parsed


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _nanmean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.mean()) if not values.empty else np.nan


def _median_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.median()) if not values.empty else np.nan


def _positive_rate(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float((values > 0).mean()) if not values.empty else np.nan


def _plan_columns() -> list[str]:
    return [
        "eval_year",
        "horizon",
        "ml_signal_name",
        "training_mode",
        "train_start_date",
        "fit_end_date",
        "label_cutoff_date",
        "eval_start_date",
        "eval_end_date",
        "train_years",
        "train_year_count",
        "train_row_count",
        "eval_feature_row_count",
        "feature_count",
        "features",
        "target_col",
        "sample_weight_policy",
        "weak_years",
        "weak_sample_weight",
        "regime_sample_weight",
        "min_regime_head_rows",
        "regime_ret_20d_threshold",
        "regime_breadth_20d_threshold",
        "regime_volatility_20d_threshold",
        "fit_uses_eval_year",
        "evidence_grade",
        "status",
    ]


def _prediction_columns() -> list[str]:
    return [
        "eval_year",
        "date",
        "code",
        "horizon",
        "ml_signal_name",
        "score",
        "fit_uses_eval_year",
        "feature_count",
        "training_mode",
        "model_head",
        "regime_label",
        "sample_weight_policy",
        "model_status",
        "evidence_grade",
    ]


def _importance_columns() -> list[str]:
    return [
        "eval_year",
        "horizon",
        "ml_signal_name",
        "training_mode",
        "model_head",
        "feature",
        "importance",
        "fit_uses_eval_year",
        "evidence_grade",
    ]


def _training_audit_columns() -> list[str]:
    return [
        *_plan_columns(),
        "model_class",
        "prediction_row_count",
        "target_mean",
        "target_std",
        "sample_weight_mean",
        "sample_weight_max",
        "train_weak_regime_rate",
        "eval_weak_regime_rate",
    ]


def _regime_audit_columns() -> list[str]:
    return [
        "eval_year",
        "horizon",
        "ml_signal_name",
        "training_mode",
        "regime_label",
        "train_row_count",
        "eval_row_count",
        "train_date_count",
        "eval_date_count",
        "regime_ret_20d_threshold",
        "regime_breadth_20d_threshold",
        "regime_volatility_20d_threshold",
        "fit_uses_eval_year",
        "status",
    ]


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 40) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(_fmt)
    return view.to_markdown(index=False)


def _fmt(value: Any) -> str:
    if value is None:
        return "nan"
    try:
        if pd.isna(value):
            return "nan"
    except TypeError:
        pass
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--years", default=",".join(str(year) for year in DEFAULT_YEARS))
    parser.add_argument("--final-end-date", default=DEFAULT_FINAL_END_DATE)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--factor-set", choices=FACTOR_SETS, default=DEFAULT_FACTOR_SET)
    parser.add_argument("--max-train-years", type=int, default=DEFAULT_MAX_TRAIN_YEARS)
    parser.add_argument("--min-train-years", type=int, default=DEFAULT_MIN_TRAIN_YEARS)
    parser.add_argument("--training-mode", choices=TRAINING_MODES, default=DEFAULT_TRAINING_MODE)
    parser.add_argument("--weak-years", default=",".join(str(year) for year in DEFAULT_WEAK_YEARS))
    parser.add_argument("--weak-sample-weight", type=float, default=DEFAULT_WEAK_SAMPLE_WEIGHT)
    parser.add_argument("--regime-sample-weight", type=float, default=DEFAULT_REGIME_SAMPLE_WEIGHT)
    parser.add_argument("--min-regime-head-rows", type=int, default=DEFAULT_MIN_REGIME_HEAD_ROWS)
    parser.add_argument("--model-n-jobs", type=int, default=DEFAULT_MODEL_N_JOBS)
    parser.add_argument("--write-prediction-shards", dest="write_prediction_shards", action="store_true", default=True)
    parser.add_argument("--no-write-prediction-shards", dest="write_prediction_shards", action="store_false")
    parser.add_argument("--write-full-prediction-csv", dest="write_full_prediction_csv", action="store_true", default=DEFAULT_WRITE_FULL_PREDICTION_CSV)
    parser.add_argument("--no-write-full-prediction-csv", dest="write_full_prediction_csv", action="store_false")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", type=Path, default=DEFAULT_RESEARCH_LOG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_frontier_ml_signal_rebuild(
        root=args.root,
        years=_parse_int_values(args.years, name="years"),
        final_end_date=args.final_end_date,
        horizon=args.horizon,
        factor_set=args.factor_set,
        max_train_years=args.max_train_years,
        min_train_years=args.min_train_years,
        training_mode=args.training_mode,
        weak_years=_parse_int_values(args.weak_years, name="weak_years"),
        weak_sample_weight=args.weak_sample_weight,
        regime_sample_weight=args.regime_sample_weight,
        min_regime_head_rows=args.min_regime_head_rows,
        model_n_jobs=args.model_n_jobs,
        write_prediction_shards=args.write_prediction_shards,
        write_full_prediction_csv=args.write_full_prediction_csv,
        output_dir=args.output_dir,
        write_research_log=args.write_research_log,
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


def _concat_prediction_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=_prediction_columns())
    return pd.concat(frames, ignore_index=True).loc[:, _prediction_columns()]


def _write_prediction_year_shard(shard_dir: Path | None, *, eval_year: int, predictions: pd.DataFrame) -> None:
    if shard_dir is None:
        return
    output = predictions.loc[:, _prediction_columns()] if not predictions.empty else pd.DataFrame(columns=_prediction_columns())
    output.to_csv(shard_dir / f"eval_year={int(eval_year)}.csv", index=False, encoding="utf-8-sig")


def _prediction_row_count(predictions: pd.DataFrame, training_audit: pd.DataFrame) -> int:
    if not training_audit.empty and "prediction_row_count" in training_audit.columns:
        counts = pd.to_numeric(training_audit["prediction_row_count"], errors="coerce").fillna(0)
        return int(counts.sum())
    return int(len(predictions))


def _prediction_artifact_mode(*, write_prediction_shards: bool, write_full_prediction_csv: bool) -> str:
    if write_prediction_shards and write_full_prediction_csv:
        return "full_csv_and_year_shards"
    if write_prediction_shards:
        return "year_shards"
    if write_full_prediction_csv:
        return "full_csv"
    return "schema_only"


if __name__ == "__main__":
    main()
