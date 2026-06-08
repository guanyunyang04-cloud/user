"""Branch-specific event-factor model zoo for personal short-line research."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from traditional_quant_research.experiments.all_limitup_followup_research import (
    latest_run_dir,
    _json_ready,
    _markdown_table,
    _truthy_col,
)
from traditional_quant_research.experiments.all_limitup_ml_strategy_research import (
    DEFAULT_BIG_LOSS_THRESHOLD_PCT,
    DEFAULT_EVAL_YEARS,
    DEFAULT_FEE_BPS,
    DEFAULT_MAX_TRAIN_YEARS,
    DEFAULT_MIN_TRAIN_YEARS,
    DEFAULT_PROFILE,
    DEFAULT_RISK_PENALTY_PCT,
    find_latest_event_panel,
    infer_exit_dates,
    reduce_event_panel_memory,
    summarize_feature_importance,
    summarize_portfolio,
    summarize_portfolio_yearly,
)
from traditional_quant_research.experiments.generalized_strong_event_pool_research import (
    DEFAULT_OUTPUT_DIR as DEFAULT_GENERALIZED_OUTPUT_DIR,
    EVENT_FLAG_COLUMNS,
    EXTRA_EVENT_FEATURE_COLUMNS,
)
from traditional_quant_research.experiments.personal_short_event_fast_research import (
    apply_score_floor_and_prior_quantile,
    apply_simple_stress,
    parse_float_values,
    parse_int_values,
    select_positions,
)
from traditional_quant_research.experiments.short_open_known_factor_rebuild import (
    audit_buy_feature_columns,
    feature_columns_for_profile,
)
from traditional_quant_research.experiments.two_day_kama_atr_breakout_analysis import (
    DEFAULT_MIN_AVAILABLE_MEMORY_GB,
    _assert_memory_available,
    available_memory_gb,
)


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/personal_short_event_model_zoo_research")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/research_log/2026-06-07_personal_short_event_model_zoo_research.md")
DEFAULT_SCORE_FLOORS = (-np.inf, 0.0, 0.5, 1.0, 1.5, 2.0)
DEFAULT_SCORE_QUANTILES = (np.nan, 0.80, 0.90, 0.95)
DEFAULT_MAX_POSITIONS = (1, 2)
DEFAULT_MODELS = ("lgbm_deep", "xgb_deep", "extra_trees_deep", "ridge")
DEFAULT_MODEL_N_JOBS = 2
DEFAULT_MIN_TRAIN_ROWS = 400
DEFAULT_MIN_EVAL_ROWS = 20
DEFAULT_MIN_PERIODS_FOR_BEST = 20
MODEL_ZOO_VERSION = 1

TARGET_HELPER_COLUMNS = (
    "sell1_close_ret_pct",
    "sell1_max_high_pct",
    "sell1_min_low_pct",
)
SOURCE_HELPER_COLUMNS = (
    "pctChg",
    "amount",
    "volume",
    "close",
    "high",
    "low",
    "open",
    "limit_up_like",
)
BASE_ID_COLUMNS = (
    "date",
    "year",
    "code",
    "name_on_date",
    "industry",
    "entry_date",
    "entry_open",
    "entry_open_near_limit",
    "executable_entry",
    "open_below_kama_break_limitup",
    "close_cross_atr_upper",
    "entry_limit_up",
    "one_word_limit_like",
    "near_one_word_limit_like",
    "limit_up_run_ending_today",
    "board_stage",
)
GENERALIZED_DEFAULT_COLUMNS = ("primary_event_type", "event_strength_score", "event_flag_count", *EVENT_FLAG_COLUMNS)
RANK_BASE_COLUMNS = (
    "pctChg",
    "signal_amount_x20",
    "signal_vrat5",
    "signal_turn_x20",
    "signal_close_position",
    "signal_ret20_before_pct",
    "signal_ret60_before_pct",
    "volatility_20_pct",
    "market_limitup_rate",
    "industry_limitup_rate",
    "next_open_gap_pct",
)
EXPANDED_EVENT_FEATURE_COLUMNS = tuple(f"{column}_date_rank" for column in RANK_BASE_COLUMNS) + (
    "kama_atr_confirm",
    "volume_close_strength",
    "vrat_close_strength",
    "ret20_vol_ratio",
    "market_industry_heat",
    "open_gap_x_close_position",
    "event_heat_x_strength",
    "first_board_flag",
    "multi_board_flag",
    "nonlimit_flag",
)
PREDICTION_OUTPUT_COLUMNS = (
    "date",
    "year",
    "code",
    "name_on_date",
    "industry",
    "entry_date",
    "entry_open",
    "entry_open_near_limit",
    "executable_entry",
    "exit_date",
    "target_net_ret_pct",
    "target_raw_ret_pct",
    "sell1_max_high_pct",
    "sell1_min_low_pct",
    "open_below_kama_break_limitup",
    "close_cross_atr_upper",
    "limit_up_run_ending_today",
    "board_stage",
    "primary_event_type",
    "event_strength_score",
    "event_flag_count",
    *EVENT_FLAG_COLUMNS,
    "next_open_gap_pct",
    "signal_amount_x20",
    "signal_vrat5",
    "signal_close_position",
    "market_limitup_rate",
    "industry_limitup_rate",
)
STRESS_SCENARIOS = (
    {"scenario": "reported_30bps"},
    {"scenario": "fee_100bps", "extra_fee_bps": 70.0},
    {"scenario": "tail_x1_25", "tail_quantile": 0.10, "tail_loss_multiplier": 1.25},
    {"scenario": "combined_light", "extra_fee_bps": 70.0, "tail_quantile": 0.10, "tail_loss_multiplier": 1.25},
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    regressor_factory: Callable[[], Any]
    classifier_factory: Callable[[], Any]


def run_personal_short_event_model_zoo_research(
    *,
    limitup_event_file: str | Path | None = None,
    generalized_event_file: str | Path | None = None,
    generalized_run_dir: str | Path | None = None,
    profile: str = DEFAULT_PROFILE,
    target_window: int = 1,
    fee_bps: float = DEFAULT_FEE_BPS,
    eval_years: Sequence[int] = DEFAULT_EVAL_YEARS,
    max_train_years: int = DEFAULT_MAX_TRAIN_YEARS,
    min_train_years: int = DEFAULT_MIN_TRAIN_YEARS,
    min_train_rows: int = DEFAULT_MIN_TRAIN_ROWS,
    min_eval_rows: int = DEFAULT_MIN_EVAL_ROWS,
    big_loss_threshold_pct: float = DEFAULT_BIG_LOSS_THRESHOLD_PCT,
    risk_penalty_pct: float = DEFAULT_RISK_PENALTY_PCT,
    model_names: Sequence[str] = DEFAULT_MODELS,
    model_n_jobs: int = DEFAULT_MODEL_N_JOBS,
    score_floors: Sequence[float] = DEFAULT_SCORE_FLOORS,
    score_quantiles: Sequence[float] = DEFAULT_SCORE_QUANTILES,
    max_positions_values: Sequence[int] = DEFAULT_MAX_POSITIONS,
    min_periods_for_best: int = DEFAULT_MIN_PERIODS_FOR_BEST,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: str | Path = DEFAULT_RESEARCH_LOG,
    min_available_memory_gb: float = DEFAULT_MIN_AVAILABLE_MEMORY_GB,
) -> dict[str, Any]:
    selected_eval_years = parse_int_values(eval_years)
    selected_models = tuple(str(name).strip() for name in model_names if str(name).strip())
    if not selected_eval_years:
        raise ValueError("eval_years must not be empty")
    if not selected_models:
        raise ValueError("model_names must not be empty")
    if int(max_train_years) < int(min_train_years):
        raise ValueError("max_train_years must be >= min_train_years")
    if int(model_n_jobs) == 0:
        raise ValueError("model_n_jobs must be non-zero")

    run_id = f"personal_short_event_model_zoo_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = build_model_zoo_feature_columns(profile)
    target_col = f"sell{int(target_window)}_close_ret_pct"
    limitup_path = Path(limitup_event_file) if limitup_event_file is not None else find_latest_event_panel()
    generalized_path = resolve_generalized_event_file(generalized_event_file, generalized_run_dir)

    _assert_memory_available(min_available_memory_gb, context="before model-zoo event reads")
    limitup_events = read_model_zoo_event_panel(
        limitup_path,
        feature_columns=feature_columns,
        target_col=target_col,
        fee_bps=fee_bps,
        source_name="limitup",
    )
    _assert_memory_available(min_available_memory_gb, context="after limit-up model-zoo event read")
    generalized_events = read_model_zoo_event_panel(
        generalized_path,
        feature_columns=feature_columns,
        target_col=target_col,
        fee_bps=fee_bps,
        source_name="generalized",
    )
    _assert_memory_available(min_available_memory_gb, context="after generalized model-zoo event read")

    branch_events = build_branch_event_frame(limitup_events, generalized_events)
    del limitup_events, generalized_events
    gc.collect()
    branch_events = add_expanded_event_features(branch_events)
    branch_events = reduce_event_panel_memory(branch_events)
    _assert_memory_available(min_available_memory_gb, context="after branch/event-factor build")

    model_specs = build_model_specs(selected_models, model_n_jobs=int(model_n_jobs))
    predictions, training_audit, raw_importance, plan = build_model_zoo_predictions(
        branch_events,
        feature_columns=feature_columns,
        model_specs=model_specs,
        eval_years=selected_eval_years,
        max_train_years=int(max_train_years),
        min_train_years=int(min_train_years),
        min_train_rows=int(min_train_rows),
        min_eval_rows=int(min_eval_rows),
        big_loss_threshold_pct=float(big_loss_threshold_pct),
        risk_penalty_pct=float(risk_penalty_pct),
        min_available_memory_gb=float(min_available_memory_gb),
    )
    predictions = append_ensemble_predictions(predictions)
    _assert_memory_available(min_available_memory_gb, context="after model-zoo walk-forward predictions")

    event_pool_summary = summarize_branch_events(branch_events)
    feature_importance = summarize_model_feature_importance(raw_importance)
    strategy_summary, strategy_yearly, stress_summary, selected_trades = evaluate_model_zoo_predictions(
        predictions,
        score_floors=parse_float_values(score_floors),
        score_quantiles=parse_float_values(score_quantiles),
        max_positions_values=parse_int_values(max_positions_values),
    )
    best_rows = select_best_watchlist(strategy_summary, stress_summary, min_periods=int(min_periods_for_best))
    best_strategy_id = str(best_rows.iloc[0]["strategy_id"]) if not best_rows.empty else ""
    best_trades = (
        selected_trades.loc[selected_trades["strategy_id"].eq(best_strategy_id)].copy()
        if best_strategy_id and not selected_trades.empty
        else pd.DataFrame()
    )
    summary = build_summary(
        run_id=run_id,
        run_dir=run_dir,
        limitup_event_file=limitup_path,
        generalized_event_file=generalized_path,
        branch_events=branch_events,
        predictions=predictions,
        plan=plan,
        training_audit=training_audit,
        strategy_summary=strategy_summary,
        stress_summary=stress_summary,
        best_rows=best_rows,
        feature_columns=feature_columns,
        profile=profile,
        target_window=target_window,
        fee_bps=fee_bps,
        eval_years=selected_eval_years,
        max_train_years=max_train_years,
        min_train_years=min_train_years,
        min_train_rows=min_train_rows,
        min_eval_rows=min_eval_rows,
        big_loss_threshold_pct=big_loss_threshold_pct,
        risk_penalty_pct=risk_penalty_pct,
        model_names=selected_models,
        model_n_jobs=model_n_jobs,
        min_periods_for_best=min_periods_for_best,
        min_available_memory_gb=min_available_memory_gb,
    )
    markdown = render_markdown(
        summary,
        event_pool_summary=event_pool_summary,
        plan=plan,
        training_audit=training_audit,
        strategy_summary=strategy_summary,
        strategy_yearly=strategy_yearly,
        stress_summary=stress_summary,
        feature_importance=feature_importance,
        best_rows=best_rows,
        best_trades=best_trades,
    )

    event_pool_summary.to_csv(run_dir / "event_pool_summary.csv", index=False, encoding="utf-8-sig")
    plan.to_csv(run_dir / "walk_forward_plan.csv", index=False, encoding="utf-8-sig")
    training_audit.to_csv(run_dir / "model_training_audit.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(run_dir / "model_zoo_predictions.csv", index=False, encoding="utf-8-sig")
    raw_importance.to_csv(run_dir / "model_feature_importance_raw.csv", index=False, encoding="utf-8-sig")
    feature_importance.to_csv(run_dir / "model_feature_importance.csv", index=False, encoding="utf-8-sig")
    strategy_summary.to_csv(run_dir / "strategy_summary.csv", index=False, encoding="utf-8-sig")
    strategy_yearly.to_csv(run_dir / "strategy_yearly.csv", index=False, encoding="utf-8-sig")
    stress_summary.to_csv(run_dir / "stress_summary.csv", index=False, encoding="utf-8-sig")
    selected_trades.to_csv(run_dir / "selected_trades.csv", index=False, encoding="utf-8-sig")
    best_rows.to_csv(run_dir / "model_zoo_watchlist.csv", index=False, encoding="utf-8-sig")
    best_trades.to_csv(run_dir / "best_strategy_trades.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")

    log_path: Path | None = None
    if write_research_log:
        log_path = Path(research_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(markdown, encoding="utf-8")

    del branch_events, predictions
    gc.collect()
    _assert_memory_available(min_available_memory_gb, context="after model-zoo outputs")
    return {**summary, "run_dir": str(run_dir), "research_log": str(log_path) if log_path else None}


def build_model_zoo_feature_columns(profile: str) -> tuple[str, ...]:
    base = tuple(feature_columns_for_profile(profile))
    audit_buy_feature_columns(base)
    columns = tuple(dict.fromkeys((*base, *EXTRA_EVENT_FEATURE_COLUMNS, *SOURCE_HELPER_COLUMNS, *EXPANDED_EVENT_FEATURE_COLUMNS)))
    audit_model_zoo_feature_columns(columns)
    return columns


def audit_model_zoo_feature_columns(feature_columns: Sequence[str]) -> None:
    forbidden_prefixes = ("sell", "managed_", "future_", "label_", "target_")
    bad = [
        column
        for column in feature_columns
        if column in {"entry_limit_up", "entry_one_word_limit"}
        or any(str(column).startswith(prefix) for prefix in forbidden_prefixes)
    ]
    if bad:
        raise ValueError(f"model-zoo feature leakage risk: {bad}")


def resolve_generalized_event_file(
    generalized_event_file: str | Path | None,
    generalized_run_dir: str | Path | None,
) -> Path:
    if generalized_event_file is not None:
        path = Path(generalized_event_file)
    else:
        run_dir = Path(generalized_run_dir) if generalized_run_dir is not None else latest_run_dir(DEFAULT_GENERALIZED_OUTPUT_DIR)
        path = run_dir / "event_feature_panel.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_model_zoo_event_panel(
    event_file: str | Path,
    *,
    feature_columns: Sequence[str],
    target_col: str,
    fee_bps: float,
    source_name: str,
) -> pd.DataFrame:
    path = Path(event_file)
    if not path.exists():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0)
    if target_col not in header.columns:
        raise ValueError(f"{path} missing target column {target_col}")
    wanted = set(BASE_ID_COLUMNS).union(feature_columns).union(TARGET_HELPER_COLUMNS).union({target_col})
    existing = sorted(column for column in wanted if column in header.columns)
    events = pd.read_csv(path, usecols=existing, low_memory=False)
    events["date"] = pd.to_datetime(events["date"])
    events["entry_date"] = pd.to_datetime(events["entry_date"])
    events["year"] = pd.to_numeric(events["year"], errors="coerce").astype("Int64")
    events["target_raw_ret_pct"] = pd.to_numeric(events[target_col], errors="coerce")
    events["target_net_ret_pct"] = events["target_raw_ret_pct"] - float(fee_bps) / 100.0
    if "sell1_max_high_pct" in events.columns:
        events["sell1_max_high_pct"] = pd.to_numeric(events["sell1_max_high_pct"], errors="coerce") - float(fee_bps) / 100.0
    else:
        events["sell1_max_high_pct"] = np.nan
    if "sell1_min_low_pct" in events.columns:
        events["sell1_min_low_pct"] = pd.to_numeric(events["sell1_min_low_pct"], errors="coerce") - float(fee_bps) / 100.0
    else:
        events["sell1_min_low_pct"] = np.nan
    events["exit_date"] = infer_exit_dates(events)
    events["data_source"] = str(source_name)
    events = ensure_event_defaults(events, source_name=source_name)
    return reduce_event_panel_memory(events.sort_values(["date", "code"]).reset_index(drop=True))


def ensure_event_defaults(events: pd.DataFrame, *, source_name: str) -> pd.DataFrame:
    output = events.copy()
    index = output.index
    for column in BASE_ID_COLUMNS:
        if column not in output.columns:
            output[column] = default_column_value(column, index=index)
    source_is_limitup = str(source_name) == "limitup"
    if "primary_event_type" not in output.columns:
        output["primary_event_type"] = "limit_up_core" if source_is_limitup else "unknown"
    if "event_strength_score" not in output.columns:
        output["event_strength_score"] = 2.0 if source_is_limitup else 0.0
    if "event_flag_count" not in output.columns:
        output["event_flag_count"] = 1 if source_is_limitup else 0
    for column in EVENT_FLAG_COLUMNS:
        if column not in output.columns:
            output[column] = column == "event_limit_up_core" and source_is_limitup
    for column in EXPANDED_EVENT_FEATURE_COLUMNS:
        if column not in output.columns:
            output[column] = np.nan
    return output


def default_column_value(column: str, *, index: pd.Index) -> Any:
    if column in {"date", "entry_date"}:
        return pd.Series(pd.NaT, index=index)
    if column in {"year", "limit_up_run_ending_today"}:
        return pd.Series(np.nan, index=index)
    if column in {"code", "name_on_date", "industry", "board_stage"}:
        return pd.Series("", index=index)
    if column.startswith("entry_") or column in {
        "open_below_kama_break_limitup",
        "close_cross_atr_upper",
        "one_word_limit_like",
        "near_one_word_limit_like",
    }:
        return pd.Series(False, index=index)
    return pd.Series(np.nan, index=index)


def build_branch_event_frame(limitup_events: pd.DataFrame, generalized_events: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    limit_exec = executable_mask(limitup_events)
    first_board = pd.to_numeric(limitup_events.get("limit_up_run_ending_today"), errors="coerce").fillna(1).le(1)
    kama_atr = _truthy_col(limitup_events, "open_below_kama_break_limitup") & _truthy_col(limitup_events, "close_cross_atr_upper")
    frames.append(assign_branch(limitup_events.loc[limit_exec & first_board], "limitup_first_board_exec"))
    frames.append(assign_branch(limitup_events.loc[limit_exec & kama_atr], "limitup_kama_atr_exec"))

    gen_exec = executable_mask(generalized_events)
    nonlimit = ~_truthy_col(generalized_events, "event_limit_up_core")
    kama = _truthy_col(generalized_events, "event_kama_breakout")
    big_new_high = _truthy_col(generalized_events, "event_big_up") | _truthy_col(generalized_events, "event_new_high_breakout")
    combo = (
        pd.to_numeric(generalized_events.get("event_flag_count"), errors="coerce").fillna(0).ge(2)
        & (
            _truthy_col(generalized_events, "event_big_up")
            | _truthy_col(generalized_events, "event_kama_breakout")
            | _truthy_col(generalized_events, "event_new_high_breakout")
        )
    )
    base = gen_exec & nonlimit
    frames.append(assign_branch(generalized_events.loc[base & kama], "nonlimit_kama_breakout_exec"))
    frames.append(assign_branch(generalized_events.loc[base & big_new_high], "nonlimit_big_new_high_exec"))
    frames.append(assign_branch(generalized_events.loc[base & combo], "nonlimit_combo_exec"))

    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    output = pd.concat(non_empty, ignore_index=True, copy=False)
    output = output.dropna(subset=["date", "entry_date", "code", "target_net_ret_pct"])
    return output.sort_values(["branch_name", "date", "code"]).reset_index(drop=True)


def executable_mask(frame: pd.DataFrame) -> pd.Series:
    index = frame.index
    executable = _truthy_col(frame, "executable_entry") if "executable_entry" in frame.columns else pd.Series(True, index=index)
    near_open = _truthy_col(frame, "entry_open_near_limit") if "entry_open_near_limit" in frame.columns else pd.Series(False, index=index)
    return executable & ~near_open


def assign_branch(frame: pd.DataFrame, branch_name: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    output = frame.copy()
    output["branch_name"] = branch_name
    return output


def add_expanded_event_features(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    output = events.copy()
    for column in RANK_BASE_COLUMNS:
        values = numeric_series(output, column)
        output[f"{column}_date_rank"] = values.groupby(output["date"]).rank(method="average", pct=True)

    kama_break = _truthy_col(output, "open_below_kama_break_limitup").astype(float)
    atr_break = _truthy_col(output, "close_cross_atr_upper").astype(float)
    output["kama_atr_confirm"] = kama_break * atr_break
    output["volume_close_strength"] = numeric_series(output, "signal_amount_x20") * numeric_series(output, "signal_close_position")
    output["vrat_close_strength"] = numeric_series(output, "signal_vrat5") * numeric_series(output, "signal_close_position")
    output["ret20_vol_ratio"] = numeric_series(output, "signal_ret20_before_pct") / numeric_series(output, "volatility_20_pct").abs().replace(0, np.nan)
    output["market_industry_heat"] = numeric_series(output, "market_limitup_rate") * numeric_series(output, "industry_limitup_rate")
    output["open_gap_x_close_position"] = numeric_series(output, "next_open_gap_pct") * numeric_series(output, "signal_close_position")
    output["event_heat_x_strength"] = numeric_series(output, "event_strength_score") * output["market_industry_heat"]
    limit_run = pd.to_numeric(output.get("limit_up_run_ending_today"), errors="coerce")
    output["first_board_flag"] = limit_run.le(1).fillna(False).astype(float)
    output["multi_board_flag"] = limit_run.gt(1).fillna(False).astype(float)
    output["nonlimit_flag"] = (~_truthy_col(output, "event_limit_up_core")).astype(float)
    return output


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)


def build_model_specs(model_names: Sequence[str], *, model_n_jobs: int) -> tuple[ModelSpec, ...]:
    specs: list[ModelSpec] = []
    for raw_name in model_names:
        name = str(raw_name).strip().lower()
        if name == "lgbm":
            specs.append(build_lgbm_spec(model_n_jobs=model_n_jobs, deep=False))
        elif name == "lgbm_deep":
            specs.append(build_lgbm_spec(model_n_jobs=model_n_jobs, deep=True))
        elif name == "xgb":
            specs.append(build_xgb_spec(model_n_jobs=model_n_jobs, deep=False))
        elif name == "xgb_deep":
            specs.append(build_xgb_spec(model_n_jobs=model_n_jobs, deep=True))
        elif name in {"extra_trees", "extratrees", "et"}:
            specs.append(build_extra_trees_spec(model_n_jobs=model_n_jobs, deep=False))
        elif name in {"extra_trees_deep", "extratrees_deep", "et_deep"}:
            specs.append(build_extra_trees_spec(model_n_jobs=model_n_jobs, deep=True))
        elif name == "ridge":
            specs.append(build_ridge_spec())
        elif name == "mlp":
            specs.append(build_mlp_spec())
        else:
            raise ValueError(f"unknown model name: {raw_name}")
    return tuple(specs)


def build_lgbm_spec(*, model_n_jobs: int, deep: bool) -> ModelSpec:
    from lightgbm import LGBMClassifier, LGBMRegressor

    if deep:
        reg_params = {
            "n_estimators": 256,
            "learning_rate": 0.035,
            "num_leaves": 63,
            "max_depth": -1,
            "min_child_samples": 45,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.05,
            "reg_lambda": 0.8,
            "random_state": 42,
            "n_jobs": int(model_n_jobs),
            "verbosity": -1,
        }
    else:
        reg_params = {
            "n_estimators": 128,
            "learning_rate": 0.04,
            "num_leaves": 15,
            "min_child_samples": 90,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": 42,
            "n_jobs": int(model_n_jobs),
            "verbosity": -1,
        }
    clf_params = {**reg_params, "objective": "binary"}
    name = "lgbm_deep" if deep else "lgbm"
    return ModelSpec(name, lambda: LGBMRegressor(**reg_params), lambda: LGBMClassifier(**clf_params))


def build_xgb_spec(*, model_n_jobs: int, deep: bool) -> ModelSpec:
    from xgboost import XGBClassifier, XGBRegressor

    if deep:
        reg_params = {
            "n_estimators": 192,
            "max_depth": 5,
            "min_child_weight": 8,
            "learning_rate": 0.035,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.05,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": int(model_n_jobs),
            "verbosity": 0,
        }
    else:
        reg_params = {
            "n_estimators": 96,
            "max_depth": 3,
            "learning_rate": 0.04,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": int(model_n_jobs),
            "verbosity": 0,
        }
    clf_params = {**reg_params, "objective": "binary:logistic", "eval_metric": "logloss"}
    name = "xgb_deep" if deep else "xgb"
    return ModelSpec(name, lambda: XGBRegressor(**reg_params), lambda: XGBClassifier(**clf_params))


def build_extra_trees_spec(*, model_n_jobs: int, deep: bool) -> ModelSpec:
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

    if deep:
        reg_params = {
            "n_estimators": 160,
            "max_depth": 14,
            "min_samples_leaf": 18,
            "max_features": 0.75,
            "random_state": 42,
            "n_jobs": int(model_n_jobs),
        }
    else:
        reg_params = {
            "n_estimators": 96,
            "max_depth": 8,
            "min_samples_leaf": 40,
            "max_features": 0.75,
            "random_state": 42,
            "n_jobs": int(model_n_jobs),
        }
    clf_params = {**reg_params, "class_weight": "balanced_subsample"}
    name = "extra_trees_deep" if deep else "extra_trees"
    return ModelSpec(name, lambda: ExtraTreesRegressor(**reg_params), lambda: ExtraTreesClassifier(**clf_params))


def build_ridge_spec() -> ModelSpec:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return ModelSpec(
        "ridge",
        lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced", solver="lbfgs")),
    )


def build_mlp_spec() -> ModelSpec:
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return ModelSpec(
        "mlp",
        lambda: make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                alpha=0.001,
                batch_size=1024,
                learning_rate_init=0.001,
                max_iter=80,
                early_stopping=True,
                random_state=42,
            ),
        ),
        lambda: make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                alpha=0.001,
                batch_size=1024,
                learning_rate_init=0.001,
                max_iter=80,
                early_stopping=True,
                random_state=42,
            ),
        ),
    )


def build_model_zoo_predictions(
    events: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    model_specs: Sequence[ModelSpec],
    eval_years: Sequence[int],
    max_train_years: int,
    min_train_years: int,
    min_train_rows: int,
    min_eval_rows: int,
    big_loss_threshold_pct: float,
    risk_penalty_pct: float,
    min_available_memory_gb: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []
    target = pd.to_numeric(events["target_net_ret_pct"], errors="coerce")

    for branch_name in sorted(events["branch_name"].dropna().astype(str).unique().tolist()):
        branch_mask = events["branch_name"].eq(branch_name)
        branch_events = events.loc[branch_mask].copy()
        branch_target = target.loc[branch_mask]
        for eval_year in parse_int_values(eval_years):
            train_start_year = int(eval_year) - int(max_train_years)
            train_mask = (
                branch_events["year"].ge(train_start_year)
                & branch_events["year"].lt(int(eval_year))
                & branch_target.notna()
            )
            eval_mask = branch_events["year"].eq(int(eval_year)) & branch_target.notna()
            train_years = sorted(branch_events.loc[train_mask, "year"].dropna().astype(int).unique().tolist())
            train_count = int(train_mask.sum())
            eval_count = int(eval_mask.sum())
            status = (
                "ready"
                if len(train_years) >= int(min_train_years)
                and train_count >= int(min_train_rows)
                and eval_count >= int(min_eval_rows)
                else "skipped"
            )
            plan_row = {
                "branch_name": branch_name,
                "eval_year": int(eval_year),
                "train_start_year": int(train_start_year),
                "train_end_year": int(eval_year) - 1,
                "train_years": ",".join(str(year) for year in train_years),
                "train_year_count": int(len(train_years)),
                "train_row_count": train_count,
                "eval_row_count": eval_count,
                "status": status,
            }
            plan_rows.append(plan_row)
            if status != "ready":
                for spec in model_specs:
                    audit_rows.append({**plan_row, "model_name": spec.name, "model_status": status})
                continue

            _assert_memory_available(min_available_memory_gb, context=f"before {branch_name} {eval_year} encoding")
            train_x, eval_x, feature_names = encode_train_eval(
                branch_events.loc[train_mask],
                branch_events.loc[eval_mask],
                feature_columns=feature_columns,
            )
            train_y = branch_target.loc[train_mask].astype(float)
            eval_events = branch_events.loc[eval_mask].copy()
            for spec in model_specs:
                model_result = fit_predict_model(
                    train_x,
                    train_y,
                    eval_x,
                    spec=spec,
                    big_loss_threshold_pct=big_loss_threshold_pct,
                    risk_penalty_pct=risk_penalty_pct,
                )
                prediction = build_prediction_frame(
                    eval_events,
                    branch_name=branch_name,
                    model_name=spec.name,
                    eval_year=int(eval_year),
                    model_result=model_result,
                )
                prediction_frames.append(prediction)
                for feature, importance in zip(feature_names, model_result.get("feature_importance", [])):
                    importance_rows.append(
                        {
                            "branch_name": branch_name,
                            "eval_year": int(eval_year),
                            "model_name": spec.name,
                            "feature": feature,
                            "importance": float(importance),
                        }
                    )
                audit_rows.append(
                    {
                        **plan_row,
                        "model_name": spec.name,
                        "model_status": "ready",
                        "target_mean": float(train_y.mean()),
                        "target_std": float(train_y.std(ddof=0)),
                        "train_big_loss_rate": float(train_y.le(big_loss_threshold_pct).mean()),
                    }
                )
                gc.collect()
            del train_x, eval_x
            gc.collect()
            _assert_memory_available(min_available_memory_gb, context=f"after {branch_name} {eval_year} model fits")

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    audit = pd.DataFrame(audit_rows)
    importance = pd.DataFrame(importance_rows)
    plan = pd.DataFrame(plan_rows)
    return predictions, audit, importance, plan


def encode_train_eval(
    train: pd.DataFrame,
    eval_frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_features = normalize_feature_frame(train, feature_columns=feature_columns)
    eval_features = normalize_feature_frame(eval_frame, feature_columns=feature_columns)
    categorical = [
        column
        for column in train_features.columns
        if train_features[column].dtype == object or str(train_features[column].dtype) == "category"
    ]
    train_encoded = pd.get_dummies(train_features, columns=categorical, dummy_na=True, dtype=np.int8) if categorical else train_features
    eval_encoded = pd.get_dummies(eval_features, columns=categorical, dummy_na=True, dtype=np.int8) if categorical else eval_features
    eval_encoded = eval_encoded.reindex(columns=train_encoded.columns, fill_value=0)
    medians = train_encoded.median(numeric_only=True)
    train_encoded = train_encoded.fillna(medians).fillna(0.0)
    eval_encoded = eval_encoded.fillna(medians).fillna(0.0)
    for frame in (train_encoded, eval_encoded):
        for column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if pd.api.types.is_float_dtype(frame[column]):
                frame[column] = frame[column].astype("float32")
    return train_encoded, eval_encoded, train_encoded.columns.tolist()


def normalize_feature_frame(frame: pd.DataFrame, *, feature_columns: Sequence[str]) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for column in feature_columns:
        if column in frame.columns:
            values = frame[column]
        else:
            values = pd.Series(np.nan, index=frame.index)
        if pd.api.types.is_bool_dtype(values):
            columns[column] = values.fillna(False).astype("int8")
        elif values.dtype == object:
            lowered = values.astype(str).str.lower()
            if set(lowered.dropna().unique()).issubset({"true", "false", "nan", "none", ""}):
                columns[column] = lowered.eq("true").astype("int8")
            else:
                columns[column] = values.astype("object")
        elif str(values.dtype) == "category":
            columns[column] = values.astype("object")
        else:
            try:
                columns[column] = pd.to_numeric(values)
            except (TypeError, ValueError):
                columns[column] = values.astype("object")
    return pd.DataFrame(columns, index=frame.index)


def fit_predict_model(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    eval_x: pd.DataFrame,
    *,
    spec: ModelSpec,
    big_loss_threshold_pct: float,
    risk_penalty_pct: float,
) -> dict[str, Any]:
    reg_model = spec.regressor_factory()
    reg_model.fit(train_x, train_y)
    pred_ret = np.asarray(reg_model.predict(eval_x), dtype=float)

    train_loss = train_y.le(float(big_loss_threshold_pct)).astype(int)
    if train_loss.nunique() > 1:
        clf_model = spec.classifier_factory()
        clf_model.fit(train_x, train_loss)
        if hasattr(clf_model, "predict_proba"):
            big_loss_prob = np.asarray(clf_model.predict_proba(eval_x)[:, 1], dtype=float)
        else:
            big_loss_prob = np.asarray(clf_model.predict(eval_x), dtype=float)
    else:
        big_loss_prob = np.full(len(eval_x), float(train_loss.mean()), dtype=float)
    score = pred_ret - float(risk_penalty_pct) * big_loss_prob
    return {
        "predicted_ret_pct": pred_ret,
        "predicted_big_loss_prob": big_loss_prob,
        "ml_score": score,
        "feature_importance": extract_feature_importance(reg_model, len(train_x.columns)),
    }


def extract_feature_importance(model: Any, feature_count: int) -> np.ndarray:
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    if hasattr(model, "named_steps"):
        final_estimator = list(model.named_steps.values())[-1]
        coef = getattr(final_estimator, "coef_", None)
        if coef is not None:
            return np.abs(np.asarray(coef, dtype=float).reshape(-1))[:feature_count]
    return np.zeros(int(feature_count), dtype=float)


def build_prediction_frame(
    eval_events: pd.DataFrame,
    *,
    branch_name: str,
    model_name: str,
    eval_year: int,
    model_result: Mapping[str, Any],
) -> pd.DataFrame:
    columns = [column for column in PREDICTION_OUTPUT_COLUMNS if column in eval_events.columns]
    prediction = eval_events.loc[:, columns].copy()
    prediction["branch_name"] = branch_name
    prediction["model_name"] = model_name
    prediction["eval_year"] = int(eval_year)
    prediction["predicted_ret_pct"] = np.asarray(model_result["predicted_ret_pct"], dtype=float)
    prediction["predicted_big_loss_prob"] = np.asarray(model_result["predicted_big_loss_prob"], dtype=float)
    prediction["ml_score"] = np.asarray(model_result["ml_score"], dtype=float)
    prediction["model_score"] = prediction["ml_score"]
    return reduce_event_panel_memory(prediction)


def append_ensemble_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    base = predictions.copy()
    base["model_rank_score"] = (
        base.groupby(["branch_name", "eval_year", "model_name"], dropna=False)["ml_score"].rank(method="average", pct=True)
    )
    key_columns = ["branch_name", "eval_year", "date", "entry_date", "code"]
    first_columns = [
        column
        for column in predictions.columns
        if column not in {"model_name", "predicted_ret_pct", "predicted_big_loss_prob", "ml_score", "model_score"}
    ]
    grouped = base.groupby(key_columns, dropna=False)
    ensemble = grouped[first_columns].first().reset_index(drop=True)
    scores = grouped.agg(
        predicted_ret_pct=("predicted_ret_pct", "mean"),
        predicted_big_loss_prob=("predicted_big_loss_prob", "mean"),
        ml_score=("model_rank_score", "mean"),
    ).reset_index(drop=True)
    ensemble = pd.concat([ensemble, scores], axis=1)
    ensemble["model_name"] = "ensemble_rank_mean"
    ensemble["model_score"] = ensemble["ml_score"]
    return pd.concat([predictions, ensemble[predictions.columns]], ignore_index=True, copy=False)


def evaluate_model_zoo_predictions(
    predictions: pd.DataFrame,
    *,
    score_floors: Sequence[float],
    score_quantiles: Sequence[float],
    max_positions_values: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rows: list[dict[str, Any]] = []
    yearly_frames: list[pd.DataFrame] = []
    stress_rows: list[dict[str, Any]] = []
    selected_frames: list[pd.DataFrame] = []
    for (branch_name, model_name), subset in predictions.groupby(["branch_name", "model_name"], dropna=False):
        eligible_days = int(pd.to_datetime(subset["entry_date"]).nunique())
        for score_floor in score_floors:
            for score_quantile in score_quantiles:
                gated = apply_score_floor_and_prior_quantile(
                    subset,
                    score_col="model_score",
                    score_floor=float(score_floor),
                    score_quantile=float(score_quantile),
                )
                for max_positions in max_positions_values:
                    selected = select_positions(gated, score_col="model_score", max_positions=int(max_positions))
                    strategy_id = strategy_id_for(
                        str(branch_name),
                        str(model_name),
                        score_floor=float(score_floor),
                        score_quantile=float(score_quantile),
                        max_positions=int(max_positions),
                    )
                    summary = summarize_model_zoo_strategy(
                        selected,
                        strategy_id=strategy_id,
                        branch_name=str(branch_name),
                        model_name=str(model_name),
                        score_floor=float(score_floor),
                        score_quantile=float(score_quantile),
                        eligible_days=eligible_days,
                        max_positions=int(max_positions),
                    )
                    rows.append(summary)
                    for scenario in STRESS_SCENARIOS:
                        stressed = apply_simple_stress(selected, scenario)
                        stress_summary = summarize_portfolio(stressed, pool_name=str(branch_name), max_positions=int(max_positions))
                        stress_summary.update(
                            {
                                "strategy_id": strategy_id,
                                "branch_name": str(branch_name),
                                "model_name": str(model_name),
                                "score_floor": None if not np.isfinite(float(score_floor)) else float(score_floor),
                                "score_quantile": None if not np.isfinite(float(score_quantile)) else float(score_quantile),
                                "scenario": scenario["scenario"],
                            }
                        )
                        stress_rows.append(stress_summary)
                    if not selected.empty:
                        selected = selected.assign(
                            strategy_id=strategy_id,
                            branch_name=str(branch_name),
                            model_name=str(model_name),
                            score_floor=None if not np.isfinite(float(score_floor)) else float(score_floor),
                            score_quantile=None if not np.isfinite(float(score_quantile)) else float(score_quantile),
                            max_positions=int(max_positions),
                        )
                        selected_frames.append(selected)
                        yearly_frames.append(summarize_portfolio_yearly(selected, strategy_id=strategy_id))
    summary = pd.DataFrame(rows).sort_values(
        ["mean_period_net_ret_pct", "positive_year_rate", "min_year_period_ret_pct", "period_count"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    yearly = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    stress = pd.DataFrame(stress_rows).sort_values(
        ["scenario", "mean_period_net_ret_pct", "positive_year_rate", "period_count"],
        ascending=[True, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    selected_trades = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    return summary, yearly, stress, selected_trades


def summarize_model_zoo_strategy(
    selected: pd.DataFrame,
    *,
    strategy_id: str,
    branch_name: str,
    model_name: str,
    score_floor: float,
    score_quantile: float,
    eligible_days: int,
    max_positions: int,
) -> dict[str, Any]:
    summary = summarize_portfolio(selected, pool_name=branch_name, max_positions=int(max_positions))
    summary.update(
        {
            "strategy_id": strategy_id,
            "branch_name": branch_name,
            "model_name": model_name,
            "score_floor": None if not np.isfinite(float(score_floor)) else float(score_floor),
            "score_quantile": None if not np.isfinite(float(score_quantile)) else float(score_quantile),
            "eligible_entry_days": int(eligible_days),
            "selection_rate": float(summary["period_count"] / int(eligible_days)) if int(eligible_days) else np.nan,
            "mean_predicted_ret_pct": mean_or_nan(selected.get("predicted_ret_pct", pd.Series(dtype=float))),
            "mean_big_loss_prob": mean_or_nan(selected.get("predicted_big_loss_prob", pd.Series(dtype=float))),
            "mean_mfe_pct": mean_or_nan(selected.get("sell1_max_high_pct", pd.Series(dtype=float))),
            "mean_mae_pct": mean_or_nan(selected.get("sell1_min_low_pct", pd.Series(dtype=float))),
        }
    )
    return summary


def strategy_id_for(branch_name: str, model_name: str, *, score_floor: float, score_quantile: float, max_positions: int) -> str:
    floor_token = "none" if not np.isfinite(float(score_floor)) else float_token(score_floor)
    quantile_token = "none" if not np.isfinite(float(score_quantile)) else f"q{float_token(score_quantile)}"
    return f"{branch_name}__{model_name}__floor_{floor_token}__{quantile_token}__pos{int(max_positions)}"


def float_token(value: float) -> str:
    text = f"{float(value):.4g}"
    return text.replace("-", "m").replace(".", "p")


def mean_or_nan(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.mean()) if not clean.empty else np.nan


def select_best_watchlist(strategy_summary: pd.DataFrame, stress_summary: pd.DataFrame, *, min_periods: int) -> pd.DataFrame:
    if strategy_summary.empty:
        return pd.DataFrame()
    combined = stress_summary.loc[stress_summary["scenario"].eq("combined_light")].copy() if not stress_summary.empty else pd.DataFrame()
    combined = combined.rename(
        columns={
            "mean_period_net_ret_pct": "combined_light_mean_period_net_ret_pct",
            "positive_year_rate": "combined_light_positive_year_rate",
            "min_year_period_ret_pct": "combined_light_min_year_period_ret_pct",
        }
    )
    keep_cols = [
        "strategy_id",
        "combined_light_mean_period_net_ret_pct",
        "combined_light_positive_year_rate",
        "combined_light_min_year_period_ret_pct",
    ]
    merged = strategy_summary.merge(combined[[column for column in keep_cols if column in combined.columns]], on="strategy_id", how="left")
    practical = merged.loc[
        merged["period_count"].ge(int(min_periods))
        & merged["near_limit_open_rate"].le(0.01)
        & merged["executable_rate"].ge(0.95)
        & merged["mean_period_net_ret_pct"].gt(0)
    ].copy()
    practical = practical.sort_values(
        [
            "combined_light_mean_period_net_ret_pct",
            "mean_period_net_ret_pct",
            "positive_year_rate",
            "period_count",
        ],
        ascending=[False, False, False, False],
        na_position="last",
    )
    return practical.head(40).reset_index(drop=True)


def summarize_branch_events(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for branch_name, subset in events.groupby("branch_name", dropna=False):
        returns = pd.to_numeric(subset.get("target_net_ret_pct"), errors="coerce").dropna()
        rows.append(
            {
                "branch_name": str(branch_name),
                "event_count": int(len(subset)),
                "entry_day_count": int(pd.to_datetime(subset["entry_date"]).nunique()) if not subset.empty else 0,
                "year_count": int(subset["year"].dropna().astype(int).nunique()) if "year" in subset.columns else 0,
                "mean_net_ret_pct": float(returns.mean()) if not returns.empty else np.nan,
                "median_net_ret_pct": float(returns.median()) if not returns.empty else np.nan,
                "win_rate": float((returns > 0).mean()) if not returns.empty else np.nan,
                "big_loss_rate": float((returns <= -5.0).mean()) if not returns.empty else np.nan,
                "near_limit_open_rate": float(_truthy_col(subset, "entry_open_near_limit").mean()) if not subset.empty else np.nan,
                "executable_rate": float(_truthy_col(subset, "executable_entry").mean()) if not subset.empty else np.nan,
                "limit_up_share": float(_truthy_col(subset, "event_limit_up_core").mean()) if not subset.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("event_count", ascending=False).reset_index(drop=True)


def summarize_model_feature_importance(importance: pd.DataFrame) -> pd.DataFrame:
    if importance.empty:
        return pd.DataFrame(columns=["branch_name", "model_name", "feature", "mean_importance", "eval_year_count"])
    grouped = (
        importance.groupby(["branch_name", "model_name", "feature"], as_index=False)
        .agg(mean_importance=("importance", "mean"), eval_year_count=("eval_year", "nunique"))
        .sort_values(["branch_name", "model_name", "mean_importance"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    return grouped


def build_summary(
    *,
    run_id: str,
    run_dir: Path,
    limitup_event_file: Path,
    generalized_event_file: Path,
    branch_events: pd.DataFrame,
    predictions: pd.DataFrame,
    plan: pd.DataFrame,
    training_audit: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    stress_summary: pd.DataFrame,
    best_rows: pd.DataFrame,
    feature_columns: Sequence[str],
    profile: str,
    target_window: int,
    fee_bps: float,
    eval_years: Sequence[int],
    max_train_years: int,
    min_train_years: int,
    min_train_rows: int,
    min_eval_rows: int,
    big_loss_threshold_pct: float,
    risk_penalty_pct: float,
    model_names: Sequence[str],
    model_n_jobs: int,
    min_periods_for_best: int,
    min_available_memory_gb: float,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "model_zoo_version": MODEL_ZOO_VERSION,
        "limitup_event_file": str(limitup_event_file),
        "generalized_event_file": str(generalized_event_file),
        "profile": profile,
        "target_window": int(target_window),
        "fee_bps": float(fee_bps),
        "eval_years": list(parse_int_values(eval_years)),
        "max_train_years": int(max_train_years),
        "min_train_years": int(min_train_years),
        "min_train_rows": int(min_train_rows),
        "min_eval_rows": int(min_eval_rows),
        "big_loss_threshold_pct": float(big_loss_threshold_pct),
        "risk_penalty_pct": float(risk_penalty_pct),
        "model_names": list(model_names),
        "model_n_jobs": int(model_n_jobs),
        "branch_event_count": int(len(branch_events)),
        "branch_count": int(branch_events["branch_name"].nunique()) if not branch_events.empty else 0,
        "prediction_count": int(len(predictions)),
        "strategy_count": int(len(strategy_summary)),
        "watchlist_count": int(len(best_rows)),
        "feature_column_count": int(len(feature_columns)),
        "expanded_feature_count": int(len(EXPANDED_EVENT_FEATURE_COLUMNS)),
        "ready_plan_count": int(plan["status"].eq("ready").sum()) if not plan.empty and "status" in plan.columns else 0,
        "ready_model_fit_count": int(training_audit["model_status"].eq("ready").sum()) if not training_audit.empty else 0,
        "best_strategy": strategy_summary.iloc[0].to_dict() if not strategy_summary.empty else {},
        "best_watchlist_strategy": best_rows.iloc[0].to_dict() if not best_rows.empty else {},
        "best_combined_light_strategy": best_combined_light_row(stress_summary),
        "decision": "model_zoo_diagnostic_not_live_candidate",
        "min_periods_for_best": int(min_periods_for_best),
        "min_available_memory_gb": float(min_available_memory_gb),
        "available_memory_gb_at_summary": available_memory_gb(),
    }


def best_combined_light_row(stress_summary: pd.DataFrame) -> dict[str, Any]:
    if stress_summary.empty or "scenario" not in stress_summary.columns:
        return {}
    subset = stress_summary.loc[stress_summary["scenario"].eq("combined_light")].copy()
    if subset.empty:
        return {}
    subset = subset.sort_values(
        ["mean_period_net_ret_pct", "positive_year_rate", "period_count"],
        ascending=[False, False, False],
        na_position="last",
    )
    return subset.iloc[0].to_dict()


def render_markdown(
    summary: Mapping[str, Any],
    *,
    event_pool_summary: pd.DataFrame,
    plan: pd.DataFrame,
    training_audit: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    strategy_yearly: pd.DataFrame,
    stress_summary: pd.DataFrame,
    feature_importance: pd.DataFrame,
    best_rows: pd.DataFrame,
    best_trades: pd.DataFrame,
) -> str:
    best_id = str(summary.get("best_strategy", {}).get("strategy_id", ""))
    watchlist_id = str(summary.get("best_watchlist_strategy", {}).get("strategy_id", ""))
    combined_id = str(summary.get("best_combined_light_strategy", {}).get("strategy_id", ""))
    watchlist_yearly = (
        strategy_yearly.loc[strategy_yearly["strategy_id"].eq(watchlist_id)].copy()
        if not strategy_yearly.empty and watchlist_id
        else pd.DataFrame()
    )
    combined = stress_summary.loc[stress_summary["scenario"].eq("combined_light")].copy() if not stress_summary.empty else pd.DataFrame()
    ready_audit = training_audit.loc[training_audit["model_status"].eq("ready")].copy() if not training_audit.empty else pd.DataFrame()
    branch_model_fit = (
        ready_audit.groupby(["branch_name", "model_name"], as_index=False)
        .agg(fit_count=("eval_year", "nunique"), mean_train_rows=("train_row_count", "mean"), mean_eval_rows=("eval_row_count", "mean"))
        if not ready_audit.empty
        else pd.DataFrame()
    )
    lines = [
        "# Personal Short-Event Model Zoo Research",
        "",
        f"- run_id: `{summary.get('run_id')}`",
        f"- branch_event_count: `{summary.get('branch_event_count')}`",
        f"- prediction_count: `{summary.get('prediction_count')}`",
        f"- feature_column_count: `{summary.get('feature_column_count')}`",
        f"- model_names: `{','.join(summary.get('model_names', []))}`",
        f"- best_strategy_id: `{best_id}`",
        f"- best_watchlist_strategy_id: `{watchlist_id}`",
        f"- best_combined_light_strategy_id: `{combined_id}`",
        f"- decision: `{summary.get('decision')}`",
        "",
        "## Branch Pools",
        "",
        _markdown_table(event_pool_summary),
        "",
        "## Walk-Forward Plan",
        "",
        _markdown_table(plan.head(40)),
        "",
        "## Ready Model Fits",
        "",
        _markdown_table(branch_model_fit),
        "",
        "## Strategy Ranking",
        "",
        _markdown_table(
            strategy_summary.head(30),
            [
                "strategy_id",
                "branch_name",
                "model_name",
                "trade_count",
                "period_count",
                "selection_rate",
                "mean_period_net_ret_pct",
                "median_period_net_ret_pct",
                "period_win_rate",
                "positive_year_rate",
                "min_year_period_ret_pct",
                "p10_trade_net_ret_pct",
                "p90_trade_net_ret_pct",
                "mean_predicted_ret_pct",
                "mean_big_loss_prob",
            ],
        ),
        "",
        "## Watchlist",
        "",
        _markdown_table(
            best_rows.head(25),
            [
                "strategy_id",
                "branch_name",
                "model_name",
                "trade_count",
                "period_count",
                "selection_rate",
                "mean_period_net_ret_pct",
                "positive_year_rate",
                "combined_light_mean_period_net_ret_pct",
                "combined_light_positive_year_rate",
                "min_year_period_ret_pct",
            ],
        ),
        "",
        "## Combined-Light Stress",
        "",
        _markdown_table(
            combined.head(25),
            [
                "strategy_id",
                "branch_name",
                "model_name",
                "trade_count",
                "period_count",
                "mean_period_net_ret_pct",
                "positive_year_rate",
                "min_year_period_ret_pct",
                "near_limit_open_rate",
                "executable_rate",
            ],
        ),
        "",
        "## Watchlist Yearly",
        "",
        _markdown_table(watchlist_yearly),
        "",
        "## Top Feature Importance",
        "",
        _markdown_table(feature_importance.head(40)),
        "",
        "## Best Trades Sample",
        "",
        _markdown_table(
            best_trades.head(40),
            [
                "date",
                "entry_date",
                "code",
                "name_on_date",
                "branch_name",
                "model_name",
                "primary_event_type",
                "model_score",
                "predicted_ret_pct",
                "predicted_big_loss_prob",
                "target_net_ret_pct",
                "sell1_max_high_pct",
                "sell1_min_low_pct",
            ],
        ),
        "",
        "## Interpretation Boundary",
        "",
        "- This experiment deliberately expands factor count and model families, but keeps branch-specific walk-forward training.",
        "- The open-known profile is valid only after the D+1 open print is known.",
        "- Ensemble score is an equal-weight rank average of base-model scores inside each branch/year.",
        "- Results remain research diagnostics until five-minute data, executable fill modeling, and stronger slippage stress are added.",
    ]
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limitup-event-file", default=None)
    parser.add_argument("--generalized-event-file", default=None)
    parser.add_argument("--generalized-run-dir", default=None)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--target-window", type=int, default=1)
    parser.add_argument("--fee-bps", type=float, default=DEFAULT_FEE_BPS)
    parser.add_argument("--eval-years", default=",".join(str(year) for year in DEFAULT_EVAL_YEARS))
    parser.add_argument("--max-train-years", type=int, default=DEFAULT_MAX_TRAIN_YEARS)
    parser.add_argument("--min-train-years", type=int, default=DEFAULT_MIN_TRAIN_YEARS)
    parser.add_argument("--min-train-rows", type=int, default=DEFAULT_MIN_TRAIN_ROWS)
    parser.add_argument("--min-eval-rows", type=int, default=DEFAULT_MIN_EVAL_ROWS)
    parser.add_argument("--big-loss-threshold-pct", type=float, default=DEFAULT_BIG_LOSS_THRESHOLD_PCT)
    parser.add_argument("--risk-penalty-pct", type=float, default=DEFAULT_RISK_PENALTY_PCT)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--model-n-jobs", type=int, default=DEFAULT_MODEL_N_JOBS)
    parser.add_argument("--score-floors", default=",".join("none" if not np.isfinite(value) else str(value) for value in DEFAULT_SCORE_FLOORS))
    parser.add_argument("--score-quantiles", default=",".join("none" if not np.isfinite(value) else str(value) for value in DEFAULT_SCORE_QUANTILES))
    parser.add_argument("--max-positions", default=",".join(str(value) for value in DEFAULT_MAX_POSITIONS))
    parser.add_argument("--min-periods-for-best", type=int, default=DEFAULT_MIN_PERIODS_FOR_BEST)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", default=str(DEFAULT_RESEARCH_LOG))
    parser.add_argument("--min-available-memory-gb", type=float, default=DEFAULT_MIN_AVAILABLE_MEMORY_GB)
    return parser


def _parse_cli_float_values(text: str) -> tuple[float, ...]:
    values: list[float] = []
    for item in str(text).split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token in {"none", "nan"}:
            values.append(np.nan)
        elif token in {"-inf", "-infinity"}:
            values.append(-np.inf)
        else:
            values.append(float(token))
    return tuple(values)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_personal_short_event_model_zoo_research(
        limitup_event_file=args.limitup_event_file,
        generalized_event_file=args.generalized_event_file,
        generalized_run_dir=args.generalized_run_dir,
        profile=args.profile,
        target_window=args.target_window,
        fee_bps=args.fee_bps,
        eval_years=parse_int_values(args.eval_years),
        max_train_years=args.max_train_years,
        min_train_years=args.min_train_years,
        min_train_rows=args.min_train_rows,
        min_eval_rows=args.min_eval_rows,
        big_loss_threshold_pct=args.big_loss_threshold_pct,
        risk_penalty_pct=args.risk_penalty_pct,
        model_names=tuple(item.strip() for item in args.models.split(",") if item.strip()),
        model_n_jobs=args.model_n_jobs,
        score_floors=_parse_cli_float_values(args.score_floors),
        score_quantiles=_parse_cli_float_values(args.score_quantiles),
        max_positions_values=parse_int_values(args.max_positions),
        min_periods_for_best=args.min_periods_for_best,
        output_dir=args.output_dir,
        write_research_log=args.write_research_log,
        research_log_path=args.research_log_path,
        min_available_memory_gb=args.min_available_memory_gb,
    )


if __name__ == "__main__":
    main()
