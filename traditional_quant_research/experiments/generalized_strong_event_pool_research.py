"""Generalized strong-event pool research beyond pure limit-up events."""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from traditional_quant_research.dataset_v2 import load_pit_manifest, load_quality_report, load_tradeable_panel
from traditional_quant_research.experiments.all_limitup_ml_strategy_research import (
    DEFAULT_BIG_LOSS_THRESHOLD_PCT,
    DEFAULT_EVAL_YEARS,
    DEFAULT_FEE_BPS,
    DEFAULT_MAX_TRAIN_YEARS,
    DEFAULT_MIN_TRAIN_YEARS,
    DEFAULT_PROFILE,
    DEFAULT_RISK_PENALTY_PCT,
    ID_COLUMNS,
    build_walk_forward_plan,
    build_walk_forward_predictions,
    encode_feature_frame,
    infer_exit_dates,
    schedule_top_positions,
    summarize_feature_importance,
    summarize_portfolio,
    summarize_portfolio_yearly,
)
from traditional_quant_research.experiments.short_open_known_factor_rebuild import (
    DEFAULT_DATA_START_YEAR,
    DEFAULT_MIN_SIGNAL_AMOUNT,
    DEFAULT_SELL_WINDOWS,
    _add_window_return_metrics,
    _group_window_arrays,
    _parse_int_values,
    audit_buy_feature_columns,
    build_event_record,
    feature_columns_for_profile,
    prepare_short_factor_panel,
)
from traditional_quant_research.experiments.two_day_kama_atr_breakout_analysis import (
    DEFAULT_MIN_AVAILABLE_MEMORY_GB,
    _assert_memory_available,
    available_memory_gb,
)


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/generalized_strong_event_pool_research")
DEFAULT_CACHE_DIR = Path("traditional_quant_research/cache/generalized_strong_events")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/research_log/2026-06-07_generalized_strong_event_pool_research.md")
DEFAULT_YEARS = tuple(range(2017, 2027))
DEFAULT_MODEL_PARAMS: dict[str, Any] = {
    "n_estimators": 128,
    "learning_rate": 0.04,
    "num_leaves": 15,
    "min_child_samples": 90,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "random_state": 42,
    "n_jobs": 2,
    "verbosity": -1,
}
GENERALIZED_EVENT_CACHE_VERSION = 2
EVENT_FLAG_COLUMNS = (
    "event_limit_up_core",
    "event_near_limit",
    "event_big_up",
    "event_volume_atr_breakout",
    "event_new_high_breakout",
    "event_kama_breakout",
    "event_trend_accel",
)
EXTRA_EVENT_FEATURE_COLUMNS = (
    "primary_event_type",
    "event_strength_score",
    "event_flag_count",
) + EVENT_FLAG_COLUMNS
POOL_NAMES = (
    "all_strong_events",
    "executable_only",
    "limit_up_core",
    "non_limit_strong",
    "near_limit",
    "volume_atr_breakout",
    "new_high_breakout",
    "kama_breakout",
)


def run_generalized_strong_event_pool_research(
    *,
    root: str | None = None,
    years: Sequence[int] = DEFAULT_YEARS,
    eval_years: Sequence[int] = DEFAULT_EVAL_YEARS,
    profile: str = DEFAULT_PROFILE,
    sell_windows: Sequence[int] = DEFAULT_SELL_WINDOWS,
    target_window: int = 1,
    fee_bps: float = DEFAULT_FEE_BPS,
    min_signal_amount: float = DEFAULT_MIN_SIGNAL_AMOUNT,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    rebuild_cache: bool = False,
    warmup_years: int = 1,
    max_train_years: int = DEFAULT_MAX_TRAIN_YEARS,
    min_train_years: int = DEFAULT_MIN_TRAIN_YEARS,
    big_loss_threshold_pct: float = DEFAULT_BIG_LOSS_THRESHOLD_PCT,
    risk_penalty_pct: float = DEFAULT_RISK_PENALTY_PCT,
    model_params: Mapping[str, Any] | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: str | Path = DEFAULT_RESEARCH_LOG,
    min_available_memory_gb: float = DEFAULT_MIN_AVAILABLE_MEMORY_GB,
) -> dict[str, Any]:
    selected_years = _parse_int_values(years, "years")
    selected_eval_years = _parse_int_values(eval_years, "eval_years")
    sell_window_values = _parse_int_values(sell_windows, "sell_windows")
    if target_window not in sell_window_values:
        raise ValueError("target_window must be included in sell_windows")

    run_id = f"generalized_strong_event_pool_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_pit_manifest(root)
    quality = load_quality_report(root)
    feature_columns = tuple(feature_columns_for_profile(profile)) + EXTRA_EVENT_FEATURE_COLUMNS
    audit_buy_feature_columns(feature_columns)
    _assert_memory_available(min_available_memory_gb, context="before generalized event build")
    event_panel = load_or_build_generalized_event_panel(
        root=root,
        years=selected_years,
        cache_dir=Path(cache_dir),
        sell_windows=sell_window_values,
        min_signal_amount=min_signal_amount,
        warmup_years=warmup_years,
        rebuild=rebuild_cache,
        min_available_memory_gb=min_available_memory_gb,
    )
    target_col = f"sell{int(target_window)}_close_ret_pct"
    event_panel["target_raw_ret_pct"] = pd.to_numeric(event_panel[target_col], errors="coerce")
    event_panel["target_net_ret_pct"] = event_panel["target_raw_ret_pct"] - float(fee_bps) / 100.0
    event_panel["exit_date"] = infer_exit_dates(event_panel)
    event_panel = reduce_memory(event_panel)
    _assert_memory_available(min_available_memory_gb, context="after generalized event panel")

    encoded = encode_feature_frame(event_panel, feature_columns)
    plan = build_walk_forward_plan(
        event_panel,
        eval_years=selected_eval_years,
        max_train_years=max_train_years,
        min_train_years=min_train_years,
    )
    params = {**DEFAULT_MODEL_PARAMS, **dict(model_params or {})}
    predictions, importance, training_audit = build_walk_forward_predictions(
        event_panel,
        encoded,
        plan,
        target_col="target_net_ret_pct",
        big_loss_threshold_pct=big_loss_threshold_pct,
        risk_penalty_pct=risk_penalty_pct,
        model_params=params,
    )
    del encoded
    gc.collect()
    predictions = enrich_predictions_with_event_columns(predictions, event_panel)
    _assert_memory_available(min_available_memory_gb, context="after generalized ML fit")

    event_summary = summarize_event_pools(event_panel)
    event_type_summary = summarize_event_types(event_panel)
    portfolio, yearly, selected_trades = evaluate_prediction_pools(predictions)
    feature_importance = summarize_feature_importance(importance)
    best = portfolio.iloc[0].to_dict() if not portfolio.empty else {}
    best_trades = (
        selected_trades.loc[selected_trades["strategy_id"].eq(best.get("strategy_id", ""))].copy()
        if not selected_trades.empty and best
        else pd.DataFrame()
    )
    summary = build_summary(
        run_id=run_id,
        run_dir=run_dir,
        manifest=manifest,
        quality=quality,
        event_panel=event_panel,
        predictions=predictions,
        portfolio=portfolio,
        plan=plan,
        feature_columns=feature_columns,
        years=selected_years,
        eval_years=selected_eval_years,
        profile=profile,
        fee_bps=fee_bps,
        target_window=target_window,
        min_signal_amount=min_signal_amount,
        model_params=params,
        min_available_memory_gb=min_available_memory_gb,
    )
    markdown = render_markdown(
        summary,
        event_summary=event_summary,
        event_type_summary=event_type_summary,
        portfolio=portfolio,
        yearly=yearly,
        feature_importance=feature_importance,
        best_trades=best_trades,
    )

    event_summary.to_csv(run_dir / "event_pool_summary.csv", index=False, encoding="utf-8-sig")
    event_type_summary.to_csv(run_dir / "event_type_summary.csv", index=False, encoding="utf-8-sig")
    plan.to_csv(run_dir / "walk_forward_plan.csv", index=False, encoding="utf-8-sig")
    training_audit.to_csv(run_dir / "ml_training_audit.csv", index=False, encoding="utf-8-sig")
    feature_importance.to_csv(run_dir / "ml_feature_importance.csv", index=False, encoding="utf-8-sig")
    portfolio.to_csv(run_dir / "portfolio_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(run_dir / "portfolio_yearly.csv", index=False, encoding="utf-8-sig")
    selected_trades.to_csv(run_dir / "selected_trades.csv", index=False, encoding="utf-8-sig")
    best_trades.to_csv(run_dir / "best_strategy_trades.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(run_dir / "ml_predictions.csv", index=False, encoding="utf-8-sig")
    event_panel.to_csv(run_dir / "event_feature_panel.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    log_path: Path | None = None
    if write_research_log:
        log_path = Path(research_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(markdown, encoding="utf-8")

    del event_panel, predictions
    gc.collect()
    _assert_memory_available(min_available_memory_gb, context="after generalized outputs")
    return {**summary, "run_dir": str(run_dir), "research_log": str(log_path) if log_path else None}


def load_or_build_generalized_event_panel(
    *,
    root: str | None,
    years: Sequence[int],
    cache_dir: Path,
    sell_windows: Sequence[int],
    min_signal_amount: float,
    warmup_years: int,
    rebuild: bool,
    min_available_memory_gb: float,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for year in _parse_int_values(years, "years"):
        _assert_memory_available(min_available_memory_gb, context=f"before generalized event cache {year}")
        frames.append(
            load_or_build_year_generalized_event_cache(
                root=root,
                year=int(year),
                cache_dir=cache_dir,
                sell_windows=sell_windows,
                min_signal_amount=min_signal_amount,
                warmup_years=warmup_years,
                rebuild=rebuild,
            )
        )
        gc.collect()
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["date", "code"]).reset_index(drop=True)


def load_or_build_year_generalized_event_cache(
    *,
    root: str | None,
    year: int,
    cache_dir: Path,
    sell_windows: Sequence[int],
    min_signal_amount: float,
    warmup_years: int,
    rebuild: bool,
) -> pd.DataFrame:
    cache_path = cache_dir / f"strong_events_{int(year)}.csv"
    meta_path = cache_dir / f"strong_events_{int(year)}.meta.json"
    if cache_path.exists() and not rebuild and cache_meta_matches(
        meta_path,
        sell_windows=sell_windows,
        min_signal_amount=min_signal_amount,
        warmup_years=warmup_years,
    ):
        return pd.read_csv(cache_path, parse_dates=["date", "entry_date"], low_memory=False)
    start_year = max(DEFAULT_DATA_START_YEAR, int(year) - max(0, int(warmup_years)))
    raw_panel = load_tradeable_panel(
        root,
        start_date=f"{start_year}-01-01",
        end_date=f"{int(year)}-12-31",
        include_metrics=True,
        include_industry=True,
    )
    events = build_generalized_event_feature_panel(
        raw_panel,
        sell_windows=sell_windows,
        min_signal_amount=min_signal_amount,
    )
    events = events.loc[events["year"].eq(int(year))].sort_values(["date", "code"]).reset_index(drop=True)
    events.to_csv(cache_path, index=False, encoding="utf-8-sig")
    meta = {
        "cache_version": GENERALIZED_EVENT_CACHE_VERSION,
        "year": int(year),
        "start_year": int(start_year),
        "warmup_years": int(warmup_years),
        "sell_windows": list(_parse_int_values(sell_windows, "sell_windows")),
        "min_signal_amount": float(min_signal_amount),
        "rows": int(len(events)),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return events


def cache_meta_matches(
    meta_path: Path,
    *,
    sell_windows: Sequence[int],
    min_signal_amount: float,
    warmup_years: int,
) -> bool:
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        actual_windows = list(_parse_int_values(meta.get("sell_windows", ()), "sell_windows"))
        actual_amount = float(meta.get("min_signal_amount", np.nan))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    expected_windows = list(_parse_int_values(sell_windows, "sell_windows"))
    return (
        int(meta.get("cache_version", -1)) == GENERALIZED_EVENT_CACHE_VERSION
        and actual_windows == expected_windows
        and int(meta.get("warmup_years", -1)) == int(warmup_years)
        and abs(actual_amount - float(min_signal_amount)) < 1e-6
    )


def build_generalized_event_feature_panel(
    raw_panel: pd.DataFrame,
    *,
    sell_windows: Sequence[int] = DEFAULT_SELL_WINDOWS,
    min_signal_amount: float = DEFAULT_MIN_SIGNAL_AMOUNT,
) -> pd.DataFrame:
    if raw_panel.empty:
        return pd.DataFrame()
    sell_window_values = _parse_int_values(sell_windows, "sell_windows")
    panel = assign_strong_event_columns(prepare_short_factor_panel(raw_panel), min_signal_amount=min_signal_amount)
    rows: list[dict[str, Any]] = []
    for _, group in panel.groupby("code", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        date_idx_to_pos = {int(value): pos for pos, value in enumerate(group["date_index"].to_numpy())}
        arrays = _group_window_arrays(group)
        event_positions = np.flatnonzero(group["strong_event_like"].to_numpy(dtype=bool))
        for pos in event_positions:
            event_row = group.iloc[pos]
            event_idx = int(event_row["date_index"])
            entry_pos = date_idx_to_pos.get(event_idx + 1)
            if entry_pos is None:
                continue
            entry = group.iloc[entry_pos]
            record = build_event_record(event_row, entry, min_signal_amount=min_signal_amount)
            for column in EXTRA_EVENT_FEATURE_COLUMNS:
                value = event_row[column]
                if column.startswith("event_") and column not in {"event_strength_score", "event_flag_count"}:
                    value = bool(value)
                record[column] = value
            _add_window_return_metrics(
                record,
                arrays=arrays,
                date_idx_to_pos=date_idx_to_pos,
                event_idx=event_idx,
                entry_open=float(entry["open"]),
                sell_windows=sell_window_values,
            )
            if any(pd.notna(record.get(f"sell{window}_close_ret_pct")) for window in sell_window_values):
                rows.append(record)
    events = pd.DataFrame(rows)
    if events.empty:
        return events
    return events.sort_values(["date", "code"]).reset_index(drop=True)


def assign_strong_event_columns(panel: pd.DataFrame, *, min_signal_amount: float) -> pd.DataFrame:
    frame = panel.copy()
    liquid = pd.to_numeric(frame["amount"], errors="coerce").ge(float(min_signal_amount))
    close_position = pd.to_numeric(frame["signal_close_position"], errors="coerce").fillna(0.0)
    pct = pd.to_numeric(frame["pctChg"], errors="coerce")
    amount_x20 = pd.to_numeric(frame["signal_amount_x20"], errors="coerce")
    vrat5 = pd.to_numeric(frame["signal_vrat5"], errors="coerce")
    limit = frame["limit_up_like"].fillna(False).astype(bool)
    frame["event_limit_up_core"] = limit
    frame["event_near_limit"] = liquid & ~limit & pct.ge(7.0) & pct.lt(9.5) & close_position.ge(0.65)
    frame["event_big_up"] = liquid & ~limit & pct.ge(5.0) & pct.lt(7.0) & close_position.ge(0.70) & amount_x20.ge(1.50)
    frame["event_volume_atr_breakout"] = (
        liquid
        & ~limit
        & pct.ge(3.0)
        & close_position.ge(0.65)
        & amount_x20.ge(2.0)
        & (frame["close_cross_atr_upper"].fillna(False).astype(bool) | frame["high_cross_atr_upper"].fillna(False).astype(bool))
    )
    frame["event_new_high_breakout"] = (
        liquid
        & ~limit
        & pct.ge(3.0)
        & close_position.ge(0.65)
        & amount_x20.ge(1.20)
        & (frame["new_high_20"].fillna(False).astype(bool) | frame["new_high_60"].fillna(False).astype(bool))
    )
    frame["event_kama_breakout"] = (
        liquid
        & ~limit
        & pct.ge(3.0)
        & close_position.ge(0.65)
        & amount_x20.ge(1.20)
        & frame["open_below_kama_break_limitup"].fillna(False).astype(bool)
    )
    frame["event_trend_accel"] = (
        liquid
        & ~limit
        & pct.ge(4.0)
        & close_position.ge(0.70)
        & amount_x20.ge(1.50)
        & frame["ma_stack_bullish"].fillna(False).astype(bool)
        & frame["kama_slope_positive"].fillna(False).astype(bool)
    )
    frame["event_flag_count"] = frame.loc[:, list(EVENT_FLAG_COLUMNS)].sum(axis=1).astype("int16")
    conditions = [frame[column].fillna(False).astype(bool) for column in EVENT_FLAG_COLUMNS]
    labels = [
        "limit_up_core",
        "near_limit",
        "big_up",
        "volume_atr_breakout",
        "new_high_breakout",
        "kama_breakout",
        "trend_accel",
    ]
    frame["primary_event_type"] = np.select(conditions, labels, default="none")
    strength = (
        pct.fillna(0.0).clip(lower=0.0, upper=12.0)
        + close_position.clip(lower=0.0, upper=1.0) * 2.0
        + amount_x20.fillna(0.0).clip(lower=0.0, upper=5.0) * 0.55
        + vrat5.fillna(0.0).clip(lower=0.0, upper=5.0) * 0.35
        + frame["event_limit_up_core"].astype(float) * 2.0
        + frame["event_near_limit"].astype(float) * 1.0
        + frame["event_new_high_breakout"].astype(float) * 0.6
        + frame["event_volume_atr_breakout"].astype(float) * 0.6
    )
    frame["event_strength_score"] = strength.astype(float)
    frame["strong_event_like"] = frame["event_flag_count"].gt(0)
    return frame


def enrich_predictions_with_event_columns(predictions: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "date",
        "code",
        "primary_event_type",
        "event_strength_score",
        "event_flag_count",
        *EVENT_FLAG_COLUMNS,
    ]
    enriched = predictions.merge(events.loc[:, columns], on=["date", "code"], how="left")
    return reduce_memory(enriched)


def build_pool_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    index = frame.index
    executable = _truthy(frame.get("executable_entry", pd.Series(True, index=index)))
    near_open = _truthy(frame.get("entry_open_near_limit", pd.Series(False, index=index)))
    limit = _truthy(frame.get("event_limit_up_core", pd.Series(False, index=index)))
    return {
        "all_strong_events": pd.Series(True, index=index),
        "executable_only": executable & ~near_open,
        "limit_up_core": limit,
        "non_limit_strong": ~limit,
        "near_limit": _truthy(frame.get("event_near_limit", pd.Series(False, index=index))),
        "volume_atr_breakout": _truthy(frame.get("event_volume_atr_breakout", pd.Series(False, index=index))),
        "new_high_breakout": _truthy(frame.get("event_new_high_breakout", pd.Series(False, index=index))),
        "kama_breakout": _truthy(frame.get("event_kama_breakout", pd.Series(False, index=index))),
    }


def summarize_event_pools(events: pd.DataFrame) -> pd.DataFrame:
    masks = build_pool_masks(events)
    rows: list[dict[str, Any]] = []
    for name in POOL_NAMES:
        subset = events.loc[masks[name]].copy()
        returns = pd.to_numeric(subset.get("target_net_ret_pct"), errors="coerce").dropna()
        rows.append(
            {
                "pool_name": name,
                "event_count": int(len(subset)),
                "mean_net_ret_pct": _mean(returns),
                "median_net_ret_pct": _median(returns),
                "win_rate": _rate(returns > 0),
                "big_loss_rate": _rate(returns <= -5.0),
                "near_limit_open_rate": _rate(_truthy(subset.get("entry_open_near_limit", pd.Series(False, index=subset.index)))) if not subset.empty else np.nan,
                "executable_rate": _rate(_truthy(subset.get("executable_entry", pd.Series(True, index=subset.index)))) if not subset.empty else np.nan,
                "limit_up_share": _rate(_truthy(subset.get("event_limit_up_core", pd.Series(False, index=subset.index)))) if not subset.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_event_types(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event_type, subset in events.groupby("primary_event_type", dropna=False):
        returns = pd.to_numeric(subset.get("target_net_ret_pct"), errors="coerce").dropna()
        rows.append(
            {
                "primary_event_type": str(event_type),
                "event_count": int(len(subset)),
                "mean_net_ret_pct": _mean(returns),
                "median_net_ret_pct": _median(returns),
                "win_rate": _rate(returns > 0),
                "big_loss_rate": _rate(returns <= -5.0),
                "near_limit_open_rate": _rate(_truthy(subset.get("entry_open_near_limit", pd.Series(False, index=subset.index)))),
                "executable_rate": _rate(_truthy(subset.get("executable_entry", pd.Series(True, index=subset.index)))),
                "mean_event_strength_score": _mean(pd.to_numeric(subset.get("event_strength_score"), errors="coerce")),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_net_ret_pct", ascending=False, na_position="last").reset_index(drop=True)


def evaluate_prediction_pools(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    masks = build_pool_masks(predictions)
    summary_rows: list[dict[str, Any]] = []
    yearly_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    for pool_name in POOL_NAMES:
        mask = masks.get(pool_name, pd.Series(False, index=predictions.index))
        for max_positions in (1, 2):
            selected = schedule_top_positions(predictions.loc[mask].copy(), pool_name=pool_name, max_positions=max_positions)
            summary = summarize_portfolio(selected, pool_name=pool_name, max_positions=max_positions)
            strategy_id = f"{pool_name}__generalized_lgbm_rank__pos{int(max_positions)}"
            summary["strategy_id"] = strategy_id
            summary_rows.append(summary)
            if not selected.empty:
                selected = selected.assign(strategy_id=strategy_id, pool_name=pool_name, max_positions=int(max_positions))
                selected_frames.append(selected)
                yearly_frames.append(summarize_portfolio_yearly(selected, strategy_id=strategy_id))
    portfolio = pd.DataFrame(summary_rows).sort_values(
        ["mean_period_net_ret_pct", "positive_year_rate", "min_year_period_ret_pct", "period_count"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    yearly = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    selected = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    return portfolio, yearly, selected


def build_summary(
    *,
    run_id: str,
    run_dir: Path,
    manifest: Mapping[str, Any],
    quality: Mapping[str, Any],
    event_panel: pd.DataFrame,
    predictions: pd.DataFrame,
    portfolio: pd.DataFrame,
    plan: pd.DataFrame,
    feature_columns: Sequence[str],
    years: Sequence[int],
    eval_years: Sequence[int],
    profile: str,
    fee_bps: float,
    target_window: int,
    min_signal_amount: float,
    model_params: Mapping[str, Any],
    min_available_memory_gb: float,
) -> dict[str, Any]:
    practical = (
        portfolio.loc[
            portfolio["pool_name"].eq("executable_only")
            & portfolio["period_count"].ge(30)
            & portfolio["near_limit_open_rate"].le(0.01)
            & portfolio["executable_rate"].ge(0.95)
        ].copy()
        if not portfolio.empty
        else pd.DataFrame()
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "snapshot_id": manifest.get("snapshot_id") or quality.get("snapshot_id"),
        "profile": profile,
        "years": list(_parse_int_values(years, "years")),
        "eval_years": list(_parse_int_values(eval_years, "eval_years")),
        "target_window": int(target_window),
        "fee_bps": float(fee_bps),
        "min_signal_amount": float(min_signal_amount),
        "event_count": int(len(event_panel)),
        "prediction_count": int(len(predictions)),
        "feature_column_count": int(len(feature_columns)),
        "ready_eval_years": sorted(plan.loc[plan["status"].eq("ready"), "eval_year"].astype(int).tolist()) if not plan.empty else [],
        "model_family": "lightgbm.LGBMRegressor + LightGBM big-loss classifier",
        "model_params": dict(model_params),
        "best_strategy": portfolio.iloc[0].to_dict() if not portfolio.empty else {},
        "best_practical_strategy": practical.iloc[0].to_dict() if not practical.empty else {},
        "decision": "generalized_strong_event_pool_diagnostic",
        "strategy_candidate_count": 0,
        "min_available_memory_gb": float(min_available_memory_gb),
        "available_memory_gb_at_summary": available_memory_gb(),
    }


def render_markdown(
    summary: Mapping[str, Any],
    *,
    event_summary: pd.DataFrame,
    event_type_summary: pd.DataFrame,
    portfolio: pd.DataFrame,
    yearly: pd.DataFrame,
    feature_importance: pd.DataFrame,
    best_trades: pd.DataFrame,
) -> str:
    best_id = str(summary.get("best_strategy", {}).get("strategy_id", ""))
    practical_id = str(summary.get("best_practical_strategy", {}).get("strategy_id", ""))
    best_yearly = yearly.loc[yearly["strategy_id"].eq(best_id)].copy() if not yearly.empty and best_id else pd.DataFrame()
    practical_yearly = yearly.loc[yearly["strategy_id"].eq(practical_id)].copy() if not yearly.empty and practical_id else pd.DataFrame()
    lines = [
        "# Generalized Strong-Event Pool Research",
        "",
        f"- run_id: `{summary.get('run_id')}`",
        f"- event_count: `{summary.get('event_count')}`",
        f"- prediction_count: `{summary.get('prediction_count')}`",
        f"- profile: `{summary.get('profile')}`",
        f"- feature_column_count: `{summary.get('feature_column_count')}`",
        f"- best_strategy_id: `{best_id}`",
        f"- best_practical_strategy_id: `{practical_id}`",
        f"- decision: `{summary.get('decision')}`",
        f"- strategy_candidate_count: `{summary.get('strategy_candidate_count')}`",
        "",
        "## Event Pools",
        "",
        _markdown_table(event_summary),
        "",
        "## Event Types",
        "",
        _markdown_table(event_type_summary),
        "",
        "## Portfolio Ranking",
        "",
        _markdown_table(
            portfolio.head(20),
            [
                "strategy_id",
                "trade_count",
                "period_count",
                "mean_period_net_ret_pct",
                "median_period_net_ret_pct",
                "period_win_rate",
                "positive_year_rate",
                "min_year_period_ret_pct",
                "p10_trade_net_ret_pct",
                "p90_trade_net_ret_pct",
                "near_limit_open_rate",
                "executable_rate",
            ],
        ),
        "",
        "## Practical Yearly",
        "",
        _markdown_table(practical_yearly),
        "",
        "## Best Strategy Yearly",
        "",
        _markdown_table(best_yearly),
        "",
        "## Top Feature Importance",
        "",
        _markdown_table(feature_importance.head(30)),
        "",
        "## Best Trades Sample",
        "",
        _markdown_table(
            best_trades.head(30),
            [
                "date",
                "entry_date",
                "code",
                "name_on_date",
                "primary_event_type",
                "event_strength_score",
                "ml_score",
                "predicted_ret_pct",
                "predicted_big_loss_prob",
                "target_net_ret_pct",
                "entry_open_near_limit",
                "executable_entry",
            ],
        ),
        "",
        "## Interpretation Boundary",
        "",
        "- This is a diagnostic expansion from pure limit-up events to a broader strong-event pool.",
        "- Current event types are daily-bar approximations; touch-board, failed-board, and refill events require five-minute data.",
        "- The open-known profile includes next-open fields, so it is valid only after the Day+1 open print is known.",
        "- Strategy candidate count remains zero until execution and stability gates are strengthened.",
    ]
    return "\n".join(lines) + "\n"


def reduce_memory(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if column in {"date", "entry_date", "exit_date"}:
            continue
        if pd.api.types.is_bool_dtype(output[column]):
            output[column] = output[column].astype(bool)
        elif pd.api.types.is_integer_dtype(output[column]):
            output[column] = pd.to_numeric(output[column], downcast="integer")
        elif pd.api.types.is_float_dtype(output[column]):
            output[column] = pd.to_numeric(output[column], downcast="float")
    return output


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else np.nan


def _median(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.median()) if not values.empty else np.nan


def _rate(values: pd.Series) -> float:
    clean = pd.Series(values).dropna()
    return float(clean.mean()) if not clean.empty else np.nan


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    if frame is None or frame.empty:
        return "_No rows._"
    output = frame.copy()
    if columns is not None:
        output = output[[column for column in columns if column in output.columns]]
    for column in output.columns:
        if pd.api.types.is_numeric_dtype(output[column]):
            output[column] = output[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.6g}")
    return output.to_markdown(index=False)


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
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--years", default=",".join(str(year) for year in DEFAULT_YEARS))
    parser.add_argument("--eval-years", default=",".join(str(year) for year in DEFAULT_EVAL_YEARS))
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--sell-windows", default=",".join(str(window) for window in DEFAULT_SELL_WINDOWS))
    parser.add_argument("--target-window", type=int, default=1)
    parser.add_argument("--fee-bps", type=float, default=DEFAULT_FEE_BPS)
    parser.add_argument("--min-signal-amount", type=float, default=DEFAULT_MIN_SIGNAL_AMOUNT)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--warmup-years", type=int, default=1)
    parser.add_argument("--max-train-years", type=int, default=DEFAULT_MAX_TRAIN_YEARS)
    parser.add_argument("--min-train-years", type=int, default=DEFAULT_MIN_TRAIN_YEARS)
    parser.add_argument("--big-loss-threshold-pct", type=float, default=DEFAULT_BIG_LOSS_THRESHOLD_PCT)
    parser.add_argument("--risk-penalty-pct", type=float, default=DEFAULT_RISK_PENALTY_PCT)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", default=str(DEFAULT_RESEARCH_LOG))
    parser.add_argument("--min-available-memory-gb", type=float, default=DEFAULT_MIN_AVAILABLE_MEMORY_GB)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = run_generalized_strong_event_pool_research(
        root=args.root,
        years=_parse_int_values(args.years, "years"),
        eval_years=_parse_int_values(args.eval_years, "eval_years"),
        profile=args.profile,
        sell_windows=_parse_int_values(args.sell_windows, "sell_windows"),
        target_window=args.target_window,
        fee_bps=args.fee_bps,
        min_signal_amount=args.min_signal_amount,
        cache_dir=args.cache_dir,
        rebuild_cache=args.rebuild_cache,
        warmup_years=args.warmup_years,
        max_train_years=args.max_train_years,
        min_train_years=args.min_train_years,
        big_loss_threshold_pct=args.big_loss_threshold_pct,
        risk_penalty_pct=args.risk_penalty_pct,
        output_dir=args.output_dir,
        write_research_log=args.write_research_log,
        research_log_path=args.research_log_path,
        min_available_memory_gb=args.min_available_memory_gb,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
