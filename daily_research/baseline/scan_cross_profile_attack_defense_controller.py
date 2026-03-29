from __future__ import annotations

import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json

import numpy as np
import pandas as pd

from daily_research.baseline.advanced_ml_runtime import (
    build_prepared_bundle_with_cache,
    history_window_to_dict,
    load_raw_data_with_cache,
    resolve_history_window,
)
from daily_research.baseline.cli_utils import (
    load_stock_list_from_file,
    parse_csv_list,
    parse_horizon_weights,
    parse_int_tuple,
    parse_named_windows,
    parse_stock_list,
)
from daily_research.baseline.config import ResearchConfig
from daily_research.baseline.data_provider import load_industry_map_from_tq, load_style_map_from_tq, load_universe_from_tq
from daily_research.baseline.diagnose_advanced_ml_ensemble import (
    DEFAULT_FOCUS_STATE,
    _flatten_metrics,
    _load_pickle,
    _ml_score_cache_path,
    _save_pickle,
    _slice_metrics,
    _slice_metrics_by_quadrants,
)
from daily_research.baseline.ml_alpha import (
    EXPANDED_MARKET_FEATURE_PROFILE,
    LEGACY_MARKET_FEATURE_PROFILE,
    MLAplhaConfig,
    blend_scores,
    build_ml_feature_bundle,
    combine_per_horizon_ml_scores,
    normalize_market_feature_profile,
    rolling_ml_scores_multi_detail,
    select_market_feature_profile,
)
from daily_research.baseline.portfolio import build_target_weights
from daily_research.baseline.regime import apply_market_regime_filter
from daily_research.baseline.scan_advanced_ml_attack_defense_controller import (
    _build_dynamic_candidates,
    _parse_float_grid,
    _parse_weight_triplet,
    _parse_weight_triplet_grid,
    _score_for_balance,
)
from daily_research.execution.liquidity_universe import build_rolling_liquidity_membership
from daily_research.progress import StageProgress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Formally scan a cross-profile advanced_ml attack/defense controller."
    )
    parser.add_argument("--data-source", choices=["tq", "csv"], default="tq")
    parser.add_argument("--csv-folder", default=None)
    parser.add_argument("--stocks", default=None)
    parser.add_argument("--stocks-file", default=None)
    parser.add_argument("--start-date", default="20190101")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--universe-scope", default="all_a")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument(
        "--rolling-liquidity-pool",
        choices=["liquid300", "liquid500", "liquid800"],
        default="liquid500",
    )
    parser.add_argument("--pool-rebalance-days", type=int, default=21)
    parser.add_argument("--pool-adv-window", type=int, default=20)
    parser.add_argument("--holding-count", type=int, default=5)
    parser.add_argument("--rebalance-freq", default="1d")
    parser.add_argument("--enhanced-profile", default="up_low_breakout_v2")
    parser.add_argument("--focus-state", default=DEFAULT_FOCUS_STATE)
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--offense-market-feature-profile", default=LEGACY_MARKET_FEATURE_PROFILE)
    parser.add_argument("--defense-market-feature-profile", default=EXPANDED_MARKET_FEATURE_PROFILE)

    parser.add_argument("--min-adv20", type=float, default=50_000.0)
    parser.add_argument("--min-price", type=float, default=2.0)
    parser.add_argument("--max-price", type=float, default=300.0)
    parser.add_argument("--max-weight", type=float, default=0.25)
    parser.add_argument("--score-threshold", type=float, default=0.0)

    parser.add_argument("--no-market-regime-filter", action="store_true")
    parser.add_argument("--regime-ma-window", type=int, default=50)
    parser.add_argument("--regime-vol-window", type=int, default=20)
    parser.add_argument("--regime-max-annual-vol", type=float, default=0.32)
    parser.add_argument("--regime-quadrants", default="trend_up_low_vol,trend_up_high_vol")

    parser.add_argument("--no-style-cap", action="store_true")
    parser.add_argument("--max-style-weight", type=float, default=0.50)
    parser.add_argument("--industry-cap", action="store_true")
    parser.add_argument("--max-industry-weight", type=float, default=0.40)

    parser.add_argument("--ml-target-horizon", type=int, default=20)
    parser.add_argument("--ml-target-horizons", default="5,10,20")
    parser.add_argument("--ml-horizon-weights", default="5:0.2,10:0.3,20:0.5")
    parser.add_argument("--ml-train-window-days", type=int, default=504)
    parser.add_argument("--ml-retrain-every-days", type=int, default=21)
    parser.add_argument("--ml-min-train-dates", type=int, default=120)
    parser.add_argument("--ml-max-samples-per-day", type=int, default=600)
    parser.add_argument("--ml-max-train-rows", type=int, default=200000)
    parser.add_argument("--ml-random-seed", type=int, default=7)
    parser.add_argument("--ml-model-family", choices=["lgbm"], default="lgbm")
    parser.add_argument("--lgbm-n-estimators", type=int, default=520)

    parser.add_argument("--ensemble-ml-weight", type=float, default=0.70)
    parser.add_argument("--ensemble-none-weight", type=float, default=0.20)
    parser.add_argument("--ensemble-v2-weight", type=float, default=0.10)
    parser.add_argument("--offense-state-weights", default="ml:0.25,none:0.20,v2:0.55")
    parser.add_argument("--defense-state-weights", default="ml:0.25,none:0.25,v2:0.50")
    parser.add_argument(
        "--dynamic-offense-state-weights-grid",
        default="ml:0.25,none:0.20,v2:0.55;ml:0.25,none:0.22,v2:0.53",
    )
    parser.add_argument(
        "--dynamic-defense-state-weights-grid",
        default="ml:0.25,none:0.25,v2:0.50;ml:0.25,none:0.235,v2:0.515",
    )
    parser.add_argument("--offense-trend-gap-grid", default="0.024192")
    parser.add_argument("--offense-max-vol-grid", default="0.176128")
    parser.add_argument("--offense-benchmark-ret10-grid", default="0.003449,0.014717")
    parser.add_argument(
        "--windows",
        default="recent_full:20250307:20260327,weak_window_20250905_20260319:20250905:20260319",
    )
    parser.add_argument("--auto-trim-history", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def _safe_num(value: Any) -> str:
    if value is None or pd.isna(value):
        return "nan"
    return f"{float(value):.3f}"


def _safe_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "nan"
    return f"{float(value):.2%}"


def _zscore_cs(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)


def _subset_raw_df_dict_to_stocks(
    raw_df_dict: dict[str, pd.DataFrame],
    benchmark: str,
    stocks: list[str],
) -> dict[str, pd.DataFrame]:
    keep = [benchmark] + [stock for stock in stocks if stock != benchmark]
    out: dict[str, pd.DataFrame] = {}
    for field, frame in raw_df_dict.items():
        cols = [col for col in keep if col in frame.columns]
        out[field] = frame.reindex(columns=cols)
    return out


def _resolve_weak_window_name(windows: list[tuple[str, str, str]]) -> str:
    for name, _, _ in windows:
        if str(name).startswith("weak_window_"):
            return str(name)
    raise ValueError("windows must include one weak_window_* entry")


def _build_profile_ml_score_cache_path(
    prepared_cache_key: str,
    ml_cfg: MLAplhaConfig,
    market_feature_profile: str,
) -> Path:
    base_path = _ml_score_cache_path(prepared_cache_key, ml_cfg)
    return base_path.with_name(f"{base_path.stem}_{market_feature_profile}{base_path.suffix}")


def _load_profile_ml_score(
    *,
    profile: str,
    prepared_bundle: dict[str, Any],
    prepared_cache_key: str,
    ml_cfg: MLAplhaConfig,
    current_membership_mask: pd.DataFrame | None,
    use_cache: bool,
    refresh_cache: bool,
    training_logs_dir: Path,
) -> dict[str, Any]:
    cache_path = _build_profile_ml_score_cache_path(prepared_cache_key, ml_cfg, profile)
    cached_payload = None if refresh_cache else _load_pickle(cache_path)
    training_log = pd.DataFrame()
    per_horizon_scores = None

    if cached_payload is not None and isinstance(cached_payload, dict) and "per_horizon_scores" in cached_payload:
        training_log = cached_payload.get("training_log", pd.DataFrame())
        per_horizon_scores = cached_payload["per_horizon_scores"]
    else:
        per_horizon_scores = cached_payload

    if per_horizon_scores is None:
        selected_market_features = select_market_feature_profile(prepared_bundle["market_features"], profile)
        _, training_log, per_horizon_scores = rolling_ml_scores_multi_detail(
            feature_frames=prepared_bundle["feature_frames"],
            market_features=selected_market_features,
            close=prepared_bundle["factor_bundle"]["raw_inputs"]["Close"],
            benchmark_close=prepared_bundle["benchmark_close"],
            filter_mask=prepared_bundle["filter_mask"],
            regime_state=prepared_bundle["regime_state"],
            config=ml_cfg,
            open_df=prepared_bundle["factor_bundle"]["raw_inputs"]["Open"],
            benchmark_open=prepared_bundle["benchmark_open"],
        )
        if use_cache:
            _save_pickle(
                cache_path,
                {
                    "training_log": training_log,
                    "per_horizon_scores": per_horizon_scores,
                },
            )

    ml_score = combine_per_horizon_ml_scores(
        per_horizon_scores=per_horizon_scores,
        close=prepared_bundle["factor_bundle"]["raw_inputs"]["Close"],
        regime_state=prepared_bundle["regime_state"],
        config=ml_cfg,
    )
    if current_membership_mask is not None:
        ml_score = ml_score.where(current_membership_mask)
    if not training_log.empty:
        training_logs_dir.mkdir(parents=True, exist_ok=True)
        training_log.to_csv(training_logs_dir / f"{profile}_training_log.csv", index=False, encoding="utf-8-sig")

    return {
        "profile": profile,
        "cache_path": str(cache_path),
        "training_log": training_log,
        "per_horizon_scores": per_horizon_scores,
        "ml_score": ml_score,
    }


def _run_static_candidate_with_ml_score(
    *,
    label: str,
    profile_role: str,
    market_feature_profile: str,
    weights: dict[str, float],
    state_ensemble_weights: dict[str, dict[str, float]],
    cfg: ResearchConfig,
    ml_cfg: MLAplhaConfig,
    ml_score: pd.DataFrame,
    prepared_bundle: dict[str, Any],
    current_membership_mask: pd.DataFrame | None,
    industry_map: pd.Series | None,
    style_map: pd.DataFrame | None,
    windows: list[tuple[str, str, str]],
    focus_state: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    run_cfg = replace(cfg)
    run_ml_cfg = MLAplhaConfig(**asdict(ml_cfg))
    run_ml_cfg.ensemble_ml_weight = float(weights["ml"])
    run_ml_cfg.ensemble_none_weight = float(weights["none"])
    run_ml_cfg.ensemble_v2_weight = float(weights["v2"])
    run_ml_cfg.state_ensemble_weights = dict(state_ensemble_weights)

    final_score = blend_scores(
        ml_score,
        prepared_bundle["score_none"],
        prepared_bundle["score_v2"],
        run_ml_cfg,
        quadrant_series=prepared_bundle["regime_state"]["quadrant"],
    )
    if current_membership_mask is not None:
        final_score = final_score.where(current_membership_mask)

    target_weights = build_target_weights(
        final_score,
        run_cfg,
        industry_map=industry_map,
        style_map=style_map,
    )
    score_for_backtest = final_score.fillna(0.0)
    if run_cfg.enable_market_regime_filter:
        target_weights, score_for_backtest = apply_market_regime_filter(
            target_weights,
            score_for_backtest,
            prepared_bundle["regime_state"],
        )

    from daily_research.baseline.backtest import backtest

    equity_df, _, metrics = backtest(
        close=prepared_bundle["factor_bundle"]["raw_inputs"]["Close"],
        benchmark_close=prepared_bundle["benchmark_close"],
        open_df=prepared_bundle["factor_bundle"]["raw_inputs"]["Open"],
        benchmark_open=prepared_bundle["benchmark_open"],
        target_weights=target_weights,
        target_scores=score_for_backtest,
        config=run_cfg,
        regime_on=prepared_bundle["regime_state"]["regime_on"],
    )
    quadrant_series = prepared_bundle["regime_state"]["quadrant"].reindex(equity_df.index)

    row = {
        "label": label,
        "blend_kind": "shortlist_static",
        "profile_role": profile_role,
        "market_feature_profile": market_feature_profile,
        "offense_market_feature_profile": market_feature_profile if profile_role == "offense" else "",
        "defense_market_feature_profile": market_feature_profile if profile_role == "defense" else "",
        "ensemble_ml_weight": float(run_ml_cfg.ensemble_ml_weight),
        "ensemble_none_weight": float(run_ml_cfg.ensemble_none_weight),
        "ensemble_v2_weight": float(run_ml_cfg.ensemble_v2_weight),
        "state_ensemble_weights": json.dumps(run_ml_cfg.state_ensemble_weights or {}, ensure_ascii=False, sort_keys=True),
        "distance_to_default": 0.90,
        "focus_state": focus_state,
    }
    row.update(_flatten_metrics("full", metrics))
    row.update(
        _flatten_metrics(
            f"{focus_state}_full",
            _slice_metrics_by_quadrants(
                equity_df,
                quadrant_series,
                pd.Timestamp(equity_df.index.min()).strftime("%Y%m%d"),
                pd.Timestamp(equity_df.index.max()).strftime("%Y%m%d"),
                [focus_state],
            ),
        )
    )
    for window_name, start, end in windows:
        row.update(_flatten_metrics(window_name, _slice_metrics(equity_df, start, end)))
        row.update(
            _flatten_metrics(
                f"{focus_state}_{window_name}",
                _slice_metrics_by_quadrants(equity_df, quadrant_series, start, end, [focus_state]),
            )
        )
    return row, equity_df


def _build_weighted_score(
    ml_score: pd.DataFrame,
    score_none: pd.DataFrame,
    score_v2: pd.DataFrame,
    weights: dict[str, float],
    valid_mask: pd.DataFrame,
) -> pd.DataFrame:
    ml_z = _zscore_cs(ml_score)
    none_z = _zscore_cs(score_none)
    v2_z = _zscore_cs(score_v2)
    out = (
        ml_z.fillna(0.0) * float(weights["ml"])
        + none_z.fillna(0.0) * float(weights["none"])
        + v2_z.fillna(0.0) * float(weights["v2"])
    )
    return out.where(valid_mask)


def _build_cross_profile_dynamic_final_score(
    *,
    offense_ml_score: pd.DataFrame,
    defense_ml_score: pd.DataFrame,
    score_none: pd.DataFrame,
    score_v2: pd.DataFrame,
    regime_state: pd.DataFrame,
    benchmark_close: pd.Series,
    focus_state: str,
    base_weights: dict[str, float],
    offense_weights: dict[str, float],
    defense_weights: dict[str, float],
    offense_trend_gap_min: float,
    offense_max_annual_vol: float,
    offense_benchmark_ret10_min: float | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    valid_mask = (
        offense_ml_score.notna()
        | defense_ml_score.notna()
        | score_none.notna()
        | score_v2.notna()
    )

    base_score = _build_weighted_score(defense_ml_score, score_none, score_v2, base_weights, valid_mask)
    offense_score = _build_weighted_score(offense_ml_score, score_none, score_v2, offense_weights, valid_mask)
    defense_score = _build_weighted_score(defense_ml_score, score_none, score_v2, defense_weights, valid_mask)

    quadrant = regime_state["quadrant"].reindex(base_score.index)
    trend_gap = regime_state["benchmark_trend_gap"].reindex(base_score.index)
    annual_vol = regime_state["benchmark_annual_vol"].reindex(base_score.index)
    benchmark_ret_10d = benchmark_close.sort_index().pct_change(10).reindex(base_score.index)

    focus_mask = quadrant.eq(focus_state).fillna(False)
    offense_mask = focus_mask & trend_gap.ge(float(offense_trend_gap_min)).fillna(False)
    offense_mask = offense_mask & annual_vol.le(float(offense_max_annual_vol)).fillna(False)
    if offense_benchmark_ret10_min is not None and pd.notna(offense_benchmark_ret10_min):
        offense_mask = offense_mask & benchmark_ret_10d.ge(float(offense_benchmark_ret10_min)).fillna(False)

    final_score = base_score.copy()
    final_score.loc[focus_mask] = defense_score.loc[focus_mask]
    final_score.loc[offense_mask] = offense_score.loc[offense_mask]
    return final_score.where(valid_mask), focus_mask.astype(bool), offense_mask.astype(bool)


def _run_cross_profile_dynamic_candidate(
    *,
    candidate: dict[str, Any],
    offense_profile: str,
    defense_profile: str,
    offense_ml_score: pd.DataFrame,
    defense_ml_score: pd.DataFrame,
    cfg: ResearchConfig,
    prepared_bundle: dict[str, Any],
    current_membership_mask: pd.DataFrame | None,
    industry_map: pd.Series | None,
    style_map: pd.DataFrame | None,
    windows: list[tuple[str, str, str]],
    focus_state: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    run_cfg = replace(cfg)
    final_score, focus_mask, offense_mask = _build_cross_profile_dynamic_final_score(
        offense_ml_score=offense_ml_score,
        defense_ml_score=defense_ml_score,
        score_none=prepared_bundle["score_none"],
        score_v2=prepared_bundle["score_v2"],
        regime_state=prepared_bundle["regime_state"],
        benchmark_close=prepared_bundle["benchmark_close"],
        focus_state=focus_state,
        base_weights=candidate["weights"],
        offense_weights=candidate["offense_weights"],
        defense_weights=candidate["defense_weights"],
        offense_trend_gap_min=float(candidate["offense_trend_gap_min"]),
        offense_max_annual_vol=float(candidate["offense_max_annual_vol"]),
        offense_benchmark_ret10_min=(
            float(candidate["offense_benchmark_ret10_min"])
            if pd.notna(candidate.get("offense_benchmark_ret10_min", np.nan))
            else None
        ),
    )
    if current_membership_mask is not None:
        final_score = final_score.where(current_membership_mask)

    target_weights = build_target_weights(
        final_score,
        run_cfg,
        industry_map=industry_map,
        style_map=style_map,
    )
    score_for_backtest = final_score.fillna(0.0)
    if run_cfg.enable_market_regime_filter:
        target_weights, score_for_backtest = apply_market_regime_filter(
            target_weights,
            score_for_backtest,
            prepared_bundle["regime_state"],
        )

    from daily_research.baseline.backtest import backtest

    equity_df, _, metrics = backtest(
        close=prepared_bundle["factor_bundle"]["raw_inputs"]["Close"],
        benchmark_close=prepared_bundle["benchmark_close"],
        open_df=prepared_bundle["factor_bundle"]["raw_inputs"]["Open"],
        benchmark_open=prepared_bundle["benchmark_open"],
        target_weights=target_weights,
        target_scores=score_for_backtest,
        config=run_cfg,
        regime_on=prepared_bundle["regime_state"]["regime_on"],
    )

    quadrant_series = prepared_bundle["regime_state"]["quadrant"].reindex(equity_df.index)
    focus_mask = focus_mask.reindex(equity_df.index).fillna(False)
    offense_mask = offense_mask.reindex(equity_df.index).fillna(False)
    defense_mask = focus_mask & ~offense_mask

    row = {
        "label": str(candidate["label"]),
        "blend_kind": "dynamic_controller",
        "profile_role": "cross_profile",
        "market_feature_profile": f"{offense_profile}->{defense_profile}",
        "offense_market_feature_profile": offense_profile,
        "defense_market_feature_profile": defense_profile,
        "ensemble_ml_weight": float(candidate["weights"]["ml"]),
        "ensemble_none_weight": float(candidate["weights"]["none"]),
        "ensemble_v2_weight": float(candidate["weights"]["v2"]),
        "state_ensemble_weights": json.dumps({}, ensure_ascii=False, sort_keys=True),
        "distance_to_default": candidate.get("distance_to_default", np.nan),
        "focus_state": focus_state,
        "controller_kind": str(candidate["controller_kind"]),
        "offense_trend_gap_min": float(candidate["offense_trend_gap_min"]),
        "offense_max_annual_vol": float(candidate["offense_max_annual_vol"]),
        "offense_benchmark_ret10_min": (
            float(candidate["offense_benchmark_ret10_min"])
            if pd.notna(candidate.get("offense_benchmark_ret10_min", np.nan))
            else np.nan
        ),
        "offense_profile_label": str(candidate.get("offense_profile_label", "")),
        "defense_profile_label": str(candidate.get("defense_profile_label", "")),
        "offense_state_weights": json.dumps(candidate["offense_weights"], ensure_ascii=False, sort_keys=True),
        "defense_state_weights": json.dumps(candidate["defense_weights"], ensure_ascii=False, sort_keys=True),
        "focus_state_days": int(focus_mask.sum()),
        "offense_focus_state_days": int(offense_mask.sum()),
        "defense_focus_state_days": int(defense_mask.sum()),
        "offense_focus_state_ratio": float(offense_mask.mean()) if len(offense_mask) else np.nan,
        "offense_within_focus_ratio": float(offense_mask.sum() / focus_mask.sum()) if focus_mask.sum() > 0 else np.nan,
    }
    row.update(_flatten_metrics("full", metrics))
    row.update(
        _flatten_metrics(
            f"{focus_state}_full",
            _slice_metrics_by_quadrants(
                equity_df,
                quadrant_series,
                pd.Timestamp(equity_df.index.min()).strftime("%Y%m%d"),
                pd.Timestamp(equity_df.index.max()).strftime("%Y%m%d"),
                [focus_state],
            ),
        )
    )
    for window_name, start, end in windows:
        row.update(_flatten_metrics(window_name, _slice_metrics(equity_df, start, end)))
        row.update(
            _flatten_metrics(
                f"{focus_state}_{window_name}",
                _slice_metrics_by_quadrants(equity_df, quadrant_series, start, end, [focus_state]),
            )
        )
        window_focus_mask = focus_mask.loc[(focus_mask.index >= pd.Timestamp(start)) & (focus_mask.index <= pd.Timestamp(end))]
        window_offense_mask = offense_mask.loc[(offense_mask.index >= pd.Timestamp(start)) & (offense_mask.index <= pd.Timestamp(end))]
        row[f"{window_name}_offense_focus_state_ratio"] = float(window_offense_mask.mean()) if len(window_offense_mask) else np.nan
        row[f"{window_name}_offense_within_focus_ratio"] = (
            float(window_offense_mask.sum() / window_focus_mask.sum()) if window_focus_mask.sum() > 0 else np.nan
        )
    return row, equity_df


def _render_summary(
    *,
    latest_data_date: str,
    history_window: dict[str, Any],
    offense_profile: str,
    defense_profile: str,
    static_df: pd.DataFrame,
    dynamic_df: pd.DataFrame,
    best_full_df: pd.DataFrame,
    best_weak_df: pd.DataFrame,
    best_balance_df: pd.DataFrame,
    best_annual_df: pd.DataFrame,
    dominating_df: pd.DataFrame,
    weak_window_name: str,
    focus_state: str,
    trend_gap_grid: list[float],
    max_vol_grid: list[float],
    benchmark_ret10_grid: list[float],
) -> str:
    weak_metric = f"{weak_window_name}_excess_sharpe"
    focus_weak_metric = f"{focus_state}_{weak_window_name}_excess_sharpe"
    lines: list[str] = []
    lines.append("# Cross-Profile Advanced ML Attack / Defense Controller Scan")
    lines.append("")
    lines.append(f"- generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- latest_data_date: {latest_data_date}")
    lines.append(
        f"- history_window: {history_window['effective_start_date']} -> {history_window['end_date']} (mode={history_window['mode']})"
    )
    lines.append(f"- focus_state: {focus_state}")
    lines.append(f"- offense_profile: {offense_profile}")
    lines.append(f"- defense_profile: {defense_profile}")
    lines.append(f"- offense_trend_gap_grid: {trend_gap_grid}")
    lines.append(f"- offense_max_vol_grid: {max_vol_grid}")
    if benchmark_ret10_grid:
        lines.append(f"- offense_benchmark_ret10_grid: {benchmark_ret10_grid}")
    lines.append("")

    lines.append("## Static controls")
    for _, row in static_df.iterrows():
        lines.append(
            f"- {row['label']}: role={row['profile_role']}, profile={row['market_feature_profile']}, "
            f"full annual {_safe_pct(row['full_annual_return'])}, full excess Sharpe {_safe_num(row['full_excess_sharpe'])}, "
            f"{weak_window_name} excess Sharpe {_safe_num(row[weak_metric])}, "
            f"{focus_state}_{weak_window_name} excess Sharpe {_safe_num(row[focus_weak_metric])}"
        )
    lines.append("")

    if not dynamic_df.empty:
        lines.append("## Best dynamic controller by full_excess_sharpe")
        for _, row in best_full_df.iterrows():
            lines.append(
                f"- {row['label']}: full annual {_safe_pct(row['full_annual_return'])}, full excess Sharpe {_safe_num(row['full_excess_sharpe'])}, "
                f"{weak_window_name} excess Sharpe {_safe_num(row[weak_metric])}, "
                f"focus weak excess Sharpe {_safe_num(row[focus_weak_metric])}, "
                f"offense within focus {_safe_pct(row['offense_within_focus_ratio'])}"
            )
        lines.append("")

        lines.append(f"## Best dynamic controller by {weak_window_name}_excess_sharpe")
        for _, row in best_weak_df.iterrows():
            lines.append(
                f"- {row['label']}: full annual {_safe_pct(row['full_annual_return'])}, full excess Sharpe {_safe_num(row['full_excess_sharpe'])}, "
                f"{weak_window_name} excess Sharpe {_safe_num(row[weak_metric])}, "
                f"focus weak excess Sharpe {_safe_num(row[focus_weak_metric])}, "
                f"offense within focus {_safe_pct(row['offense_within_focus_ratio'])}"
            )
        lines.append("")

        lines.append("## Best dynamic controller by full_annual_return")
        for _, row in best_annual_df.iterrows():
            lines.append(
                f"- {row['label']}: full annual {_safe_pct(row['full_annual_return'])}, full excess annual {_safe_pct(row['full_excess_annual_return'])}, "
                f"full excess Sharpe {_safe_num(row['full_excess_sharpe'])}, "
                f"{weak_window_name} excess Sharpe {_safe_num(row[weak_metric])}"
            )
        lines.append("")

        lines.append("## Best dynamic controller by balance score")
        for _, row in best_balance_df.iterrows():
            lines.append(
                f"- {row['label']}: balance score {_safe_num(row['controller_balance_score'])}, "
                f"full annual {_safe_pct(row['full_annual_return'])}, full excess Sharpe {_safe_num(row['full_excess_sharpe'])}, "
                f"{weak_window_name} excess Sharpe {_safe_num(row[weak_metric])}, "
                f"focus weak excess Sharpe {_safe_num(row[focus_weak_metric])}"
            )
        lines.append("")

    lines.append("## Direct answer")
    if dominating_df.empty:
        lines.append("- No cross-profile dynamic controller currently dominates both static controls on full and weak-window excess Sharpe at the same time.")
        if not best_annual_df.empty:
            best_row = best_annual_df.iloc[0]
            lines.append(
                f"- Best current profit-oriented compromise is {best_row['label']}: "
                f"full annual {_safe_pct(best_row['full_annual_return'])}, full excess annual {_safe_pct(best_row['full_excess_annual_return'])}, "
                f"full excess Sharpe {_safe_num(best_row['full_excess_sharpe'])}."
            )
    else:
        winner = dominating_df.iloc[0]
        lines.append(
            f"- Cross-profile winner: {winner['label']} "
            f"(full annual {_safe_pct(winner['full_annual_return'])}, full excess Sharpe {_safe_num(winner['full_excess_sharpe'])}, "
            f"{weak_window_name} excess Sharpe {_safe_num(winner[weak_metric])})."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    stocks = parse_stock_list(args.stocks)
    file_stocks = load_stock_list_from_file(args.stocks_file)
    if stocks or file_stocks:
        stocks = list(dict.fromkeys(stocks + file_stocks))
    if args.rolling_liquidity_pool and stocks:
        raise ValueError("Use either fixed stocks or --rolling-liquidity-pool, not both.")

    offense_profile = normalize_market_feature_profile(args.offense_market_feature_profile)
    defense_profile = normalize_market_feature_profile(args.defense_market_feature_profile)

    cfg = ResearchConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        universe_scope=args.universe_scope,
        benchmark=args.benchmark,
        execution_mode="next_open",
        holding_count=args.holding_count,
        weighting_method="score",
        rebalance_freq=args.rebalance_freq,
        score_threshold=args.score_threshold,
        max_weight=args.max_weight,
        min_adv20=args.min_adv20,
        min_price=args.min_price,
        max_price=args.max_price,
        enable_market_regime_filter=not args.no_market_regime_filter,
        regime_ma_window=args.regime_ma_window,
        regime_vol_window=args.regime_vol_window,
        regime_max_annual_vol=args.regime_max_annual_vol,
        regime_allowed_quadrants=parse_csv_list(args.regime_quadrants),
        enable_style_cap=not args.no_style_cap,
        max_style_weight=args.max_style_weight,
        enable_industry_cap=args.industry_cap,
        max_industry_weight=args.max_industry_weight,
    )
    if stocks:
        cfg.universe = stocks

    base_weights = {
        "ml": float(args.ensemble_ml_weight),
        "none": float(args.ensemble_none_weight),
        "v2": float(args.ensemble_v2_weight),
    }
    offense_weights = _parse_weight_triplet(args.offense_state_weights)
    defense_weights = _parse_weight_triplet(args.defense_state_weights)
    dynamic_offense_weight_grid = _parse_weight_triplet_grid(args.dynamic_offense_state_weights_grid, offense_weights)
    dynamic_defense_weight_grid = _parse_weight_triplet_grid(args.dynamic_defense_state_weights_grid, defense_weights)
    trend_gap_grid = _parse_float_grid(args.offense_trend_gap_grid, [0.024192])
    max_vol_grid = _parse_float_grid(args.offense_max_vol_grid, [0.176128])
    benchmark_ret10_grid = _parse_float_grid(args.offense_benchmark_ret10_grid, [0.003449, 0.014717])
    windows = parse_named_windows(args.windows)
    weak_window_name = _resolve_weak_window_name(windows)
    focus_state = str(args.focus_state).strip().lower() or DEFAULT_FOCUS_STATE

    ml_cfg = MLAplhaConfig(
        target_horizon=args.ml_target_horizon,
        target_horizons=parse_int_tuple(args.ml_target_horizons, args.ml_target_horizon),
        target_horizon_weights=parse_horizon_weights(args.ml_horizon_weights),
        enhanced_profile=args.enhanced_profile,
        train_window_days=args.ml_train_window_days,
        retrain_every_days=args.ml_retrain_every_days,
        min_train_dates=args.ml_min_train_dates,
        max_samples_per_day=args.ml_max_samples_per_day,
        max_train_rows=args.ml_max_train_rows,
        random_seed=args.ml_random_seed,
        model_family=args.ml_model_family,
        lgbm_n_estimators=args.lgbm_n_estimators,
        ensemble_ml_weight=base_weights["ml"],
        ensemble_none_weight=base_weights["none"],
        ensemble_v2_weight=base_weights["v2"],
        train_regime_only=cfg.enable_market_regime_filter,
        execution_mode=cfg.execution_mode,
    )

    dynamic_candidates = _build_dynamic_candidates(
        focus_state=focus_state,
        base_weights=base_weights,
        offense_weight_grid=dynamic_offense_weight_grid,
        defense_weight_grid=dynamic_defense_weight_grid,
        trend_gap_grid=trend_gap_grid,
        max_vol_grid=max_vol_grid,
        benchmark_ret10_grid=benchmark_ret10_grid,
    )
    for candidate in dynamic_candidates:
        candidate["offense_market_feature_profile"] = offense_profile
        candidate["defense_market_feature_profile"] = defense_profile
        candidate["label"] = (
            f"{focus_state}_cross_{offense_profile}_off_{defense_profile}_def_"
            f"{candidate['label'].replace(f'{focus_state}_', '', 1)}"
        )

    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else Path("daily_research/output")
        / (
            args.experiment_tag.strip()
            or f"advanced_ml_cross_profile_attack_defense_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    )
    output_root.mkdir(parents=True, exist_ok=True)

    latest_data_date = ""
    with StageProgress(total=8, label="cross-profile controller") as progress:
        with progress.stage("load universe", args.universe_scope):
            if args.data_source == "tq" and not cfg.universe:
                cfg.universe = load_universe_from_tq(cfg.universe_scope)
            if not cfg.universe:
                raise ValueError("Universe is empty after loading.")

        with progress.stage("resolve history", args.start_date):
            history_window = resolve_history_window(
                cfg=cfg,
                ml_cfg=ml_cfg,
                requested_start_date=args.start_date,
                end_date=args.end_date,
                mode="train",
                auto_trim_history=args.auto_trim_history,
            )
            progress.log(
                f"history window: {history_window.effective_start_date} -> {history_window.end_date} "
                f"(required_trading_days={history_window.required_trading_days})"
            )

        with progress.stage("load market data", f"stocks={len(cfg.universe)}"):
            raw_df_dict, raw_cache_meta = load_raw_data_with_cache(
                data_source=args.data_source,
                csv_folder=args.csv_folder,
                universe=cfg.universe,
                benchmark=cfg.benchmark,
                history_window=history_window,
                use_cache=not args.no_cache,
                refresh_cache=args.refresh_cache,
                progress_desc="load raw bars",
                progress_position=1,
            )
            latest_data_date = history_window.end_date
            progress.log(f"raw cache: {'hit' if raw_cache_meta['cache_hit'] else 'build'} | {raw_cache_meta['cache_path']}")

        with progress.stage("build rolling pool", args.rolling_liquidity_pool or "fixed"):
            rolling_pool_artifact = None
            rolling_membership_mask = None
            prepared_raw_df_dict = raw_df_dict
            if args.rolling_liquidity_pool:
                raw_universe_df_dict = {key: frame.drop(columns=[cfg.benchmark], errors="ignore") for key, frame in raw_df_dict.items()}
                rolling_pool_artifact = build_rolling_liquidity_membership(
                    close_frame=raw_universe_df_dict["Close"],
                    amount_frame=raw_universe_df_dict["Amount"],
                    pool_name=args.rolling_liquidity_pool,
                    signal_start_date=cfg.start_date,
                    signal_end_date=args.end_date,
                    rebalance_every_days=args.pool_rebalance_days,
                    adv_window=args.pool_adv_window,
                    min_price=cfg.min_price,
                    max_price=cfg.max_price,
                )
                rolling_membership_mask = rolling_pool_artifact.membership_frame
                rolling_union = rolling_membership_mask.columns[rolling_membership_mask.any(axis=0)].tolist()
                if not rolling_union:
                    raise RuntimeError(f"Rolling {args.rolling_liquidity_pool} pool is empty for the requested window.")
                prepared_raw_df_dict = _subset_raw_df_dict_to_stocks(raw_df_dict, cfg.benchmark, rolling_union)
                progress.log(f"rolling pool union size: {len(rolling_union)}")

        with progress.stage("build prepared bundle", args.enhanced_profile):
            prepared_raw_cache_key = (
                raw_cache_meta["cache_key"]
                if rolling_pool_artifact is None
                else f"{raw_cache_meta['cache_key']}|{args.rolling_liquidity_pool}|{args.pool_rebalance_days}|{args.pool_adv_window}"
            )
            prepared_bundle, prepared_cache_meta = build_prepared_bundle_with_cache(
                raw_df_dict=prepared_raw_df_dict,
                raw_cache_key=prepared_raw_cache_key,
                cfg=cfg,
                enhanced_profile=args.enhanced_profile,
                use_cache=not args.no_cache,
                refresh_cache=args.refresh_cache,
            )
            progress.log(
                f"prepared cache: {'hit' if prepared_cache_meta['cache_hit'] else 'build'} | {prepared_cache_meta['cache_path']}"
            )
            current_membership_mask = None
            if rolling_membership_mask is not None:
                current_membership_mask = rolling_membership_mask.reindex(
                    index=prepared_bundle["df_dict"]["Close"].index,
                    columns=prepared_bundle["df_dict"]["Close"].columns,
                ).fillna(False)
                prepared_bundle["filter_mask"] = prepared_bundle["filter_mask"] & current_membership_mask
                prepared_bundle["score_none"] = prepared_bundle["score_none"].where(current_membership_mask)
                prepared_bundle["score_v2"] = prepared_bundle["score_v2"].where(current_membership_mask)
                feature_frames, market_features = build_ml_feature_bundle(
                    prepared_bundle["factor_bundle"],
                    prepared_bundle["regime_state"],
                    prepared_bundle["score_none"],
                    prepared_bundle["score_v2"],
                )
                prepared_bundle["feature_frames"] = feature_frames
                prepared_bundle["market_features"] = market_features

        with progress.stage("load exposure maps", "style/industry"):
            candidate_columns = [col for col in prepared_bundle["df_dict"]["Close"].columns if col != cfg.benchmark]
            industry_map = None
            style_map = None
            if cfg.enable_industry_cap and args.data_source == "tq":
                industry_map = load_industry_map_from_tq(candidate_columns)
            if cfg.enable_style_cap and args.data_source == "tq":
                style_map = load_style_map_from_tq(candidate_columns)

        with progress.stage("load profile ml scores", f"{offense_profile},{defense_profile}"):
            training_logs_dir = output_root / "training_logs"
            profile_scores: dict[str, dict[str, Any]] = {}
            for profile in dict.fromkeys([offense_profile, defense_profile]):
                profile_scores[profile] = _load_profile_ml_score(
                    profile=profile,
                    prepared_bundle=prepared_bundle,
                    prepared_cache_key=prepared_cache_meta["cache_key"],
                    ml_cfg=ml_cfg,
                    current_membership_mask=current_membership_mask,
                    use_cache=not args.no_cache,
                    refresh_cache=args.refresh_cache,
                    training_logs_dir=training_logs_dir,
                )
                progress.log(f"profile {profile}: ml cache {profile_scores[profile]['cache_path']}")

        with progress.stage("run comparisons", f"{2 + len(dynamic_candidates)} candidates"):
            rows: list[dict[str, Any]] = []
            row, _ = _run_static_candidate_with_ml_score(
                label=f"{defense_profile}__{focus_state}_ml25_none25_v250",
                profile_role="defense",
                market_feature_profile=defense_profile,
                weights=base_weights,
                state_ensemble_weights={focus_state: dict(defense_weights)},
                cfg=cfg,
                ml_cfg=ml_cfg,
                ml_score=profile_scores[defense_profile]["ml_score"],
                prepared_bundle=prepared_bundle,
                current_membership_mask=current_membership_mask,
                industry_map=industry_map,
                style_map=style_map,
                windows=windows,
                focus_state=focus_state,
            )
            rows.append(row)
            row, _ = _run_static_candidate_with_ml_score(
                label=f"{offense_profile}__{focus_state}_ml25_none20_v255",
                profile_role="offense",
                market_feature_profile=offense_profile,
                weights=base_weights,
                state_ensemble_weights={focus_state: dict(offense_weights)},
                cfg=cfg,
                ml_cfg=ml_cfg,
                ml_score=profile_scores[offense_profile]["ml_score"],
                prepared_bundle=prepared_bundle,
                current_membership_mask=current_membership_mask,
                industry_map=industry_map,
                style_map=style_map,
                windows=windows,
                focus_state=focus_state,
            )
            rows.append(row)

            for candidate in dynamic_candidates:
                row, _ = _run_cross_profile_dynamic_candidate(
                    candidate=candidate,
                    offense_profile=offense_profile,
                    defense_profile=defense_profile,
                    offense_ml_score=profile_scores[offense_profile]["ml_score"],
                    defense_ml_score=profile_scores[defense_profile]["ml_score"],
                    cfg=cfg,
                    prepared_bundle=prepared_bundle,
                    current_membership_mask=current_membership_mask,
                    industry_map=industry_map,
                    style_map=style_map,
                    windows=windows,
                    focus_state=focus_state,
                )
                rows.append(row)

    results_df = pd.DataFrame(rows)
    weak_metric = f"{weak_window_name}_excess_sharpe"
    focus_weak_metric = f"{focus_state}_{weak_window_name}_excess_sharpe"
    results_df["controller_balance_score"] = _score_for_balance(results_df, weak_metric, focus_weak_metric)
    results_df = results_df.sort_values(
        ["blend_kind", "controller_balance_score", "full_excess_sharpe"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    static_df = results_df.loc[results_df["blend_kind"].eq("shortlist_static")].copy().reset_index(drop=True)
    dynamic_df = results_df.loc[results_df["blend_kind"].eq("dynamic_controller")].copy().reset_index(drop=True)
    best_full_df = dynamic_df.sort_values(
        ["full_excess_sharpe", weak_metric, focus_weak_metric],
        ascending=[False, False, False],
    ).head(5).reset_index(drop=True)
    best_weak_df = dynamic_df.sort_values(
        [weak_metric, focus_weak_metric, "full_excess_sharpe"],
        ascending=[False, False, False],
    ).head(5).reset_index(drop=True)
    best_balance_df = dynamic_df.sort_values(
        ["controller_balance_score", "full_excess_sharpe", weak_metric],
        ascending=[False, False, False],
    ).head(5).reset_index(drop=True)
    best_annual_df = dynamic_df.sort_values(
        ["full_annual_return", "full_excess_annual_return", "full_excess_sharpe"],
        ascending=[False, False, False],
    ).head(5).reset_index(drop=True)

    static_full_max = float(static_df["full_excess_sharpe"].max()) if not static_df.empty else np.nan
    static_weak_max = float(static_df[weak_metric].max()) if not static_df.empty else np.nan
    static_focus_weak_max = float(static_df[focus_weak_metric].max()) if not static_df.empty else np.nan
    dominating_df = dynamic_df.loc[
        dynamic_df["full_excess_sharpe"].ge(static_full_max)
        & dynamic_df[weak_metric].ge(static_weak_max)
        & dynamic_df[focus_weak_metric].ge(static_focus_weak_max)
    ].sort_values(
        ["controller_balance_score", "full_excess_sharpe", weak_metric],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    summary_text = _render_summary(
        latest_data_date=latest_data_date,
        history_window=history_window_to_dict(history_window),
        offense_profile=offense_profile,
        defense_profile=defense_profile,
        static_df=static_df,
        dynamic_df=dynamic_df,
        best_full_df=best_full_df,
        best_weak_df=best_weak_df,
        best_balance_df=best_balance_df,
        best_annual_df=best_annual_df,
        dominating_df=dominating_df,
        weak_window_name=weak_window_name,
        focus_state=focus_state,
        trend_gap_grid=trend_gap_grid,
        max_vol_grid=max_vol_grid,
        benchmark_ret10_grid=benchmark_ret10_grid,
    )

    results_df.to_csv(output_root / "controller_scan_summary.csv", index=False, encoding="utf-8-sig")
    static_df.to_csv(output_root / "static_controls.csv", index=False, encoding="utf-8-sig")
    dynamic_df.to_csv(output_root / "dynamic_candidates.csv", index=False, encoding="utf-8-sig")
    best_full_df.to_csv(output_root / "best_dynamic_by_full_excess_sharpe.csv", index=False, encoding="utf-8-sig")
    best_weak_df.to_csv(output_root / "best_dynamic_by_weak_excess_sharpe.csv", index=False, encoding="utf-8-sig")
    best_balance_df.to_csv(output_root / "best_dynamic_by_balance_score.csv", index=False, encoding="utf-8-sig")
    best_annual_df.to_csv(output_root / "best_dynamic_by_full_annual_return.csv", index=False, encoding="utf-8-sig")
    dominating_df.to_csv(output_root / "dominating_dynamic_candidates.csv", index=False, encoding="utf-8-sig")
    (output_root / "summary.md").write_text(summary_text, encoding="utf-8")

    run_config = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "latest_data_date": latest_data_date,
        "history_window": history_window_to_dict(history_window),
        "focus_state": focus_state,
        "windows": [{"name": name, "start": start, "end": end} for name, start, end in windows],
        "offense_market_feature_profile": offense_profile,
        "defense_market_feature_profile": defense_profile,
        "rolling_liquidity_pool": args.rolling_liquidity_pool,
        "pool_rebalance_days": args.pool_rebalance_days,
        "pool_adv_window": args.pool_adv_window,
        "base_weights": base_weights,
        "offense_weights": offense_weights,
        "defense_weights": defense_weights,
        "dynamic_offense_weight_grid": dynamic_offense_weight_grid,
        "dynamic_defense_weight_grid": dynamic_defense_weight_grid,
        "offense_trend_gap_grid": trend_gap_grid,
        "offense_max_vol_grid": max_vol_grid,
        "offense_benchmark_ret10_grid": benchmark_ret10_grid,
        "static_labels": static_df["label"].tolist(),
        "dynamic_label_count": int(len(dynamic_df)),
        "raw_data_source": args.data_source,
        "benchmark": cfg.benchmark,
        "raw_cache": raw_cache_meta,
        "prepared_cache": prepared_cache_meta,
        "profile_ml_score_cache_paths": {
            profile: payload["cache_path"] for profile, payload in profile_scores.items()
        },
        "ml_config": {
            "train_window_days": int(ml_cfg.train_window_days),
            "retrain_every_days": int(ml_cfg.retrain_every_days),
            "lgbm_n_estimators": int(ml_cfg.lgbm_n_estimators),
            "target_horizons": list(ml_cfg.target_horizons),
            "target_horizon_weights": {str(k): float(v) for k, v in (ml_cfg.target_horizon_weights or {}).items()},
        },
        "dominance_reference": {
            "full_excess_sharpe": static_full_max,
            weak_metric: static_weak_max,
            focus_weak_metric: static_focus_weak_max,
        },
    }
    (output_root / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
