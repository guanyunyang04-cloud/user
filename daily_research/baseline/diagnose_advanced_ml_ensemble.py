from __future__ import annotations

import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
import math
import pickle

import numpy as np
import pandas as pd

from daily_research.baseline.advanced_ml_runtime import (
    build_prepared_bundle_with_cache,
    history_window_to_dict,
    load_raw_data_with_cache,
    resolve_history_window,
)
from daily_research.baseline.backtest import backtest
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
from daily_research.baseline.ml_alpha import (
    MLAplhaConfig,
    blend_scores,
    build_ml_feature_bundle,
    combine_per_horizon_ml_scores,
    rolling_ml_scores_multi_detail,
)
from daily_research.baseline.portfolio import build_target_weights
from daily_research.baseline.regime import apply_market_regime_filter
from daily_research.execution.liquidity_universe import build_rolling_liquidity_membership
from daily_research.progress import StageProgress, iter_progress


DEFAULT_WINDOWS = "recent_full:20250307:20260319,weak_window_20250905_20260319:20250905:20260319"
DEFAULT_FOCUS_STATE = "trend_up_low_vol"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ablation, state-only ensemble scan, and overlap diagnostics for the current advanced ML execution stack."
    )
    parser.add_argument("--data-source", choices=["tq", "csv"], default="tq")
    parser.add_argument("--csv-folder", default=None)
    parser.add_argument("--stocks", default=None)
    parser.add_argument("--stocks-file", default=None)
    parser.add_argument("--start-date", default="20210101")
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
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument("--output-dir", default="")

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
    parser.add_argument("--ml-model-family", choices=["histgb", "etr", "lgbm"], default="lgbm")
    parser.add_argument("--lgbm-n-estimators", type=int, default=260)
    parser.add_argument("--ensemble-ml-weight", type=float, default=0.70)
    parser.add_argument("--ensemble-none-weight", type=float, default=0.20)
    parser.add_argument("--ensemble-v2-weight", type=float, default=0.10)

    parser.add_argument("--focus-state", default=DEFAULT_FOCUS_STATE)
    parser.add_argument("--state-scan-step", type=float, default=0.10)
    parser.add_argument("--state-scan-topn", type=int, default=10)
    parser.add_argument("--state-ml-range", default="")
    parser.add_argument("--state-none-range", default="")
    parser.add_argument("--state-v2-range", default="")
    parser.add_argument("--ablation-weight-mode", choices=["preserve_ratio", "equal"], default="preserve_ratio")
    parser.add_argument("--top-n-overlap", type=int, default=5)
    parser.add_argument("--windows", default=DEFAULT_WINDOWS)
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-overlap", action="store_true")
    parser.add_argument("--auto-trim-history", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def _parse_float_range(raw: str | None) -> tuple[float, float] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    for sep in (":", ",", "~"):
        if sep in text:
            left_raw, right_raw = text.split(sep, 1)
            left = float(left_raw.strip())
            right = float(right_raw.strip())
            if left > right:
                left, right = right, left
            return left, right
    value = float(text)
    return value, value


def _subset_raw_df_dict_to_stocks(
    raw_df_dict: dict[str, pd.DataFrame],
    benchmark: str,
    stocks: list[str],
) -> dict[str, pd.DataFrame]:
    keep = [benchmark] + [stock for stock in stocks if stock != benchmark]
    keep_set = set(keep)
    out: dict[str, pd.DataFrame] = {}
    for field, frame in raw_df_dict.items():
        cols = [col for col in frame.columns if col in keep_set]
        out[field] = frame.reindex(columns=cols)
    return out


def _normalize_weight_map(raw: dict[str, float]) -> dict[str, float]:
    cleaned = {name: max(float(value), 0.0) for name, value in raw.items()}
    total = float(sum(cleaned.values()))
    if total <= 0:
        return {"ml": 1.0, "none": 0.0, "v2": 0.0}
    return {name: float(value) / total for name, value in cleaned.items()}


def _build_ablation_candidates(base_weights: dict[str, float], mode: str) -> list[dict[str, Any]]:
    combos = [
        ("ml_only", ("ml",)),
        ("none_only", ("none",)),
        ("v2_only", ("v2",)),
        ("ml_plus_none", ("ml", "none")),
        ("ml_plus_v2", ("ml", "v2")),
        ("ml_plus_none_plus_v2", ("ml", "none", "v2")),
    ]
    rows: list[dict[str, Any]] = []
    for label, active in combos:
        if mode == "equal":
            weight_map = {name: (1.0 / len(active) if name in active else 0.0) for name in ("ml", "none", "v2")}
        else:
            weight_map = {
                name: (base_weights.get(name, 0.0) if name in active else 0.0)
                for name in ("ml", "none", "v2")
            }
        rows.append(
            {
                "label": label,
                "blend_kind": "ablation",
                "weights": _normalize_weight_map(weight_map),
                "state_ensemble_weights": {},
            }
        )
    return rows


def _float_step_to_scale(step: float) -> int:
    rounded = round(1.0 / float(step))
    if not math.isclose(float(step) * rounded, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("state-scan-step must divide 1.0 exactly, e.g. 0.05, 0.10, 0.20")
    return int(rounded)


def _build_state_scan_candidates(
    *,
    focus_state: str,
    step: float,
    base_weights: dict[str, float],
    ml_range: tuple[float, float] | None = None,
    none_range: tuple[float, float] | None = None,
    v2_range: tuple[float, float] | None = None,
) -> list[dict[str, Any]]:
    scale = _float_step_to_scale(step)
    step_value = 1.0 / scale
    base = _normalize_weight_map(base_weights)
    rows: list[dict[str, Any]] = [
        {
            "label": "base_global",
            "blend_kind": "state_only",
            "weights": base,
            "state_ensemble_weights": {},
            "focus_state_weights": base,
            "distance_to_default": 0.0,
        }
    ]
    for ml_i in range(scale + 1):
        for none_i in range(scale - ml_i + 1):
            v2_i = scale - ml_i - none_i
            weights = {
                "ml": ml_i * step_value,
                "none": none_i * step_value,
                "v2": v2_i * step_value,
            }
            if ml_range is not None and not (float(ml_range[0]) <= weights["ml"] <= float(ml_range[1])):
                continue
            if none_range is not None and not (float(none_range[0]) <= weights["none"] <= float(none_range[1])):
                continue
            if v2_range is not None and not (float(v2_range[0]) <= weights["v2"] <= float(v2_range[1])):
                continue
            distance = abs(weights["ml"] - base["ml"]) + abs(weights["none"] - base["none"]) + abs(weights["v2"] - base["v2"])
            rows.append(
                {
                    "label": f"{focus_state}_ml{int(round(weights['ml'] * 100)):02d}_none{int(round(weights['none'] * 100)):02d}_v2{int(round(weights['v2'] * 100)):02d}",
                    "blend_kind": "state_only",
                    "weights": base,
                    "state_ensemble_weights": {focus_state: weights},
                    "focus_state_weights": weights,
                    "distance_to_default": float(distance),
                }
            )
    rows.sort(
        key=lambda item: (
            item["distance_to_default"],
            -item["focus_state_weights"]["v2"],
            -item["focus_state_weights"]["none"],
            -item["focus_state_weights"]["ml"],
            item["label"],
        )
    )
    return rows


def _resolve_weak_window_name(windows: list[tuple[str, str, str]]) -> str | None:
    for name, _, _ in windows:
        if str(name).startswith("weak_window_"):
            return str(name)
    for name, _, _ in windows:
        if str(name) == "latest_weak":
            return str(name)
    return None


def _annualized_return(equity: pd.Series) -> float:
    daily_ret = equity.pct_change().dropna()
    if daily_ret.empty:
        return 0.0
    return float(equity.iloc[-1] ** (252 / len(daily_ret)) - 1.0)


def _annualized_vol(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0.0


def _max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    return float(drawdown.min()) if not drawdown.empty else np.nan


def _compound_return(series: pd.Series) -> float:
    valid = series.dropna()
    if valid.empty:
        return np.nan
    return float((1.0 + valid).prod() - 1.0)


def _slice_metrics(equity_df: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    window = equity_df.loc[(equity_df.index >= pd.Timestamp(start)) & (equity_df.index <= pd.Timestamp(end))].copy()
    if window.empty:
        return {
            "start": pd.Timestamp(start).strftime("%Y-%m-%d"),
            "end": pd.Timestamp(end).strftime("%Y-%m-%d"),
            "days": 0,
            "total_return": np.nan,
            "annual_return": np.nan,
            "annual_vol": np.nan,
            "sharpe": np.nan,
            "benchmark_total_return": np.nan,
            "benchmark_annual_return": np.nan,
            "excess_total_return": np.nan,
            "excess_annual_return": np.nan,
            "excess_sharpe": np.nan,
            "excess_max_drawdown": np.nan,
            "max_drawdown": np.nan,
            "avg_holding_count": np.nan,
            "avg_turnover": np.nan,
            "hit_rate": np.nan,
            "regime_active_ratio": np.nan,
        }

    portfolio_equity = window["portfolio_equity"] / float(window["portfolio_equity"].iloc[0])
    benchmark_equity = window["benchmark_equity"] / float(window["benchmark_equity"].iloc[0])
    excess_equity = portfolio_equity / benchmark_equity.replace(0.0, np.nan)
    excess_equity = excess_equity.replace([np.inf, -np.inf], np.nan).ffill().dropna()

    portfolio_returns = window["portfolio_return"].dropna()
    excess_returns = window["excess_return"].dropna()
    portfolio_ann_ret = _annualized_return(portfolio_equity)
    benchmark_ann_ret = _annualized_return(benchmark_equity)
    excess_ann_ret = _annualized_return(excess_equity) if not excess_equity.empty else np.nan
    portfolio_ann_vol = _annualized_vol(portfolio_returns)
    excess_ann_vol = _annualized_vol(excess_returns)

    return {
        "start": pd.Timestamp(start).strftime("%Y-%m-%d"),
        "end": pd.Timestamp(end).strftime("%Y-%m-%d"),
        "days": int(len(window)),
        "total_return": float(portfolio_equity.iloc[-1] - 1.0),
        "annual_return": float(portfolio_ann_ret),
        "annual_vol": float(portfolio_ann_vol),
        "sharpe": float(portfolio_ann_ret / portfolio_ann_vol) if portfolio_ann_vol > 0 else np.nan,
        "benchmark_total_return": float(benchmark_equity.iloc[-1] - 1.0),
        "benchmark_annual_return": float(benchmark_ann_ret),
        "excess_total_return": float(excess_equity.iloc[-1] - 1.0) if not excess_equity.empty else np.nan,
        "excess_annual_return": float(excess_ann_ret) if pd.notna(excess_ann_ret) else np.nan,
        "excess_sharpe": float(excess_ann_ret / excess_ann_vol) if excess_ann_vol > 0 and pd.notna(excess_ann_ret) else np.nan,
        "excess_max_drawdown": float(_max_drawdown(excess_equity)) if not excess_equity.empty else np.nan,
        "max_drawdown": float(_max_drawdown(portfolio_equity)),
        "avg_holding_count": float(window["holding_count"].mean()),
        "avg_turnover": float(window["turnover"].mean()),
        "hit_rate": float((portfolio_returns > 0).mean()) if not portfolio_returns.empty else np.nan,
        "regime_active_ratio": float(window["regime_on"].mean()) if "regime_on" in window else np.nan,
    }


def _slice_metrics_by_quadrants(
    equity_df: pd.DataFrame,
    quadrant_series: pd.Series,
    start: str,
    end: str,
    quadrants: list[str],
) -> dict[str, Any]:
    window = equity_df.loc[(equity_df.index >= pd.Timestamp(start)) & (equity_df.index <= pd.Timestamp(end))].copy()
    if window.empty:
        return _slice_metrics(window, start, end)
    quadrant_series = quadrant_series.reindex(window.index)
    if quadrants:
        window = window.loc[quadrant_series.isin(quadrants)]
    return _slice_metrics(window, start, end)


def _flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _next_open_excess_by_signal_date(open_df: pd.DataFrame, benchmark_open: pd.Series) -> pd.DataFrame:
    benchmark_ret = benchmark_open.shift(-2).div(benchmark_open.shift(-1)).sub(1.0)
    stock_ret = open_df.shift(-2).div(open_df.shift(-1)).sub(1.0)
    return stock_ret.sub(benchmark_ret, axis=0)


def _rank_corr(x: pd.Series, y: pd.Series) -> float:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 2:
        return np.nan
    if pair.iloc[:, 0].nunique() <= 1 or pair.iloc[:, 1].nunique() <= 1:
        return np.nan
    xr = pair.iloc[:, 0].rank(method="average")
    yr = pair.iloc[:, 1].rank(method="average")
    return float(xr.corr(yr))


def _daily_topn_forward_excess(score_row: pd.Series, forward_row: pd.Series, top_n: int) -> float:
    aligned = pd.concat([score_row.rename("score"), forward_row.rename("forward")], axis=1).dropna()
    if aligned.empty:
        return np.nan
    picked = aligned.nlargest(min(int(top_n), len(aligned)), "score")
    if picked.empty:
        return np.nan
    return float(picked["forward"].mean())


def _safe_pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return f"{float(value):.2%}"


def _safe_num(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return f"{float(value):.3f}"


def _ml_score_cache_path(prepared_cache_key: str, ml_cfg: MLAplhaConfig) -> Path:
    payload = {
        "kind": "diagnose_shared_ml_scores",
        "score_engine_version": 2,
        "prepared_cache_key": str(prepared_cache_key),
        "ml_config": {
            "target_horizon": int(ml_cfg.target_horizon),
            "target_horizons": list(ml_cfg.target_horizons),
            "target_horizon_weights": {str(k): float(v) for k, v in (ml_cfg.target_horizon_weights or {}).items()},
            "train_window_days": int(ml_cfg.train_window_days),
            "retrain_every_days": int(ml_cfg.retrain_every_days),
            "min_train_dates": int(ml_cfg.min_train_dates),
            "max_samples_per_day": int(ml_cfg.max_samples_per_day),
            "max_train_rows": int(ml_cfg.max_train_rows),
            "random_seed": int(ml_cfg.random_seed),
            "model_family": str(ml_cfg.model_family),
            "lgbm_n_estimators": int(ml_cfg.lgbm_n_estimators),
            "train_regime_only": bool(ml_cfg.train_regime_only),
            "execution_mode": str(ml_cfg.execution_mode),
        },
    }
    cache_key = hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    cache_dir = Path("daily_research/cache/advanced_ml/ml_scores")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{cache_key}.pkl"


def _load_pickle(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def _save_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _build_overlap_daily(
    *,
    score_map: dict[str, pd.DataFrame],
    valid_mask: pd.DataFrame,
    quadrant_series: pd.Series,
    next_open_excess: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    pairs = [("ml", "none"), ("ml", "v2"), ("none", "v2")]
    rows: list[dict[str, Any]] = []
    for dt in iter_progress(valid_mask.index, total=len(valid_mask.index), desc="overlap diagnostics", unit="day", position=1):
        base_valid = valid_mask.loc[dt].fillna(False)
        if not bool(base_valid.any()):
            continue
        for left_name, right_name in pairs:
            left_row = score_map[left_name].loc[dt].where(base_valid)
            right_row = score_map[right_name].loc[dt].where(base_valid)
            aligned = pd.concat([left_row.rename("left"), right_row.rename("right")], axis=1)
            aligned = aligned.replace([np.inf, -np.inf], np.nan).dropna()
            if len(aligned) < max(2, int(top_n)):
                continue

            left_top = aligned.nlargest(int(top_n), "left").index.tolist()
            right_top = aligned.nlargest(int(top_n), "right").index.tolist()
            left_set = set(left_top)
            right_set = set(right_top)
            inter = left_set & right_set
            union = left_set | right_set
            forward_row = next_open_excess.loc[dt].where(base_valid)

            rows.append(
                {
                    "date": pd.Timestamp(dt),
                    "quadrant": str(quadrant_series.loc[dt]) if dt in quadrant_series.index else "",
                    "pair": f"{left_name}_vs_{right_name}",
                    "eligible_count": int(base_valid.sum()),
                    "valid_pair_count": int(len(aligned)),
                    "pearson": float(aligned["left"].corr(aligned["right"]))
                    if aligned["left"].nunique() > 1 and aligned["right"].nunique() > 1
                    else np.nan,
                    "spearman": _rank_corr(aligned["left"], aligned["right"]),
                    "top_overlap_count": int(len(inter)),
                    "top_overlap_ratio": float(len(inter) / float(top_n)),
                    "top_jaccard": float(len(inter) / len(union)) if union else np.nan,
                    "left_top5_next_open_excess": _daily_topn_forward_excess(left_row, forward_row, top_n),
                    "right_top5_next_open_excess": _daily_topn_forward_excess(right_row, forward_row, top_n),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "quadrant",
                "pair",
                "eligible_count",
                "valid_pair_count",
                "pearson",
                "spearman",
                "top_overlap_count",
                "top_overlap_ratio",
                "top_jaccard",
                "left_top5_next_open_excess",
                "right_top5_next_open_excess",
                "top5_next_open_gain_left_minus_right",
            ]
        )
    df["top5_next_open_gain_left_minus_right"] = df["left_top5_next_open_excess"] - df["right_top5_next_open_excess"]
    return df.sort_values(["date", "pair"]).reset_index(drop=True)


def _summarize_overlap_section(df: pd.DataFrame, section: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return rows
    for pair, pair_df in df.groupby("pair", sort=True):
        left_total = _compound_return(pair_df["left_top5_next_open_excess"])
        right_total = _compound_return(pair_df["right_top5_next_open_excess"])
        rows.append(
            {
                "section": section,
                "pair": pair,
                "days": int(len(pair_df)),
                "mean_pearson": float(pair_df["pearson"].mean()),
                "median_pearson": float(pair_df["pearson"].median()),
                "mean_spearman": float(pair_df["spearman"].mean()),
                "median_spearman": float(pair_df["spearman"].median()),
                "mean_top_overlap_count": float(pair_df["top_overlap_count"].mean()),
                "mean_top_overlap_ratio": float(pair_df["top_overlap_ratio"].mean()),
                "mean_top_jaccard": float(pair_df["top_jaccard"].mean()),
                "left_top5_mean_next_open_excess": float(pair_df["left_top5_next_open_excess"].mean()),
                "right_top5_mean_next_open_excess": float(pair_df["right_top5_next_open_excess"].mean()),
                "left_top5_total_next_open_excess": float(left_total) if pd.notna(left_total) else np.nan,
                "right_top5_total_next_open_excess": float(right_total) if pd.notna(right_total) else np.nan,
                "left_minus_right_mean_next_open_excess": float(pair_df["top5_next_open_gain_left_minus_right"].mean()),
                "left_minus_right_total_next_open_excess": float(left_total - right_total)
                if pd.notna(left_total) and pd.notna(right_total)
                else np.nan,
            }
        )
    return rows


def _run_candidate(
    *,
    candidate: dict[str, Any],
    cfg: ResearchConfig,
    ml_cfg: MLAplhaConfig,
    shared_per_horizon_scores: dict[int, pd.DataFrame],
    prepared_bundle: dict[str, Any],
    current_membership_mask: pd.DataFrame | None,
    industry_map: pd.Series | None,
    style_map: pd.DataFrame | None,
    windows: list[tuple[str, str, str]],
    focus_state: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    run_cfg = replace(cfg)
    run_ml_cfg = MLAplhaConfig(**asdict(ml_cfg))
    weights = candidate["weights"]
    run_ml_cfg.ensemble_ml_weight = float(weights["ml"])
    run_ml_cfg.ensemble_none_weight = float(weights["none"])
    run_ml_cfg.ensemble_v2_weight = float(weights["v2"])
    run_ml_cfg.state_ensemble_weights = dict(candidate.get("state_ensemble_weights") or {})

    ml_score = combine_per_horizon_ml_scores(
        per_horizon_scores=shared_per_horizon_scores,
        close=prepared_bundle["factor_bundle"]["raw_inputs"]["Close"],
        regime_state=prepared_bundle["regime_state"],
        config=run_ml_cfg,
    )
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
        "label": str(candidate["label"]),
        "blend_kind": str(candidate["blend_kind"]),
        "ensemble_ml_weight": float(run_ml_cfg.ensemble_ml_weight),
        "ensemble_none_weight": float(run_ml_cfg.ensemble_none_weight),
        "ensemble_v2_weight": float(run_ml_cfg.ensemble_v2_weight),
        "state_ensemble_weights": json.dumps(run_ml_cfg.state_ensemble_weights or {}, ensure_ascii=False, sort_keys=True),
        "distance_to_default": candidate.get("distance_to_default", np.nan),
        "focus_state": focus_state,
    }
    row.update(_flatten_metrics("full", metrics))
    full_focus_metrics = _slice_metrics_by_quadrants(
        equity_df,
        quadrant_series,
        pd.Timestamp(equity_df.index.min()).strftime("%Y%m%d"),
        pd.Timestamp(equity_df.index.max()).strftime("%Y%m%d"),
        [focus_state],
    )
    row.update(_flatten_metrics(f"{focus_state}_full", full_focus_metrics))
    for window_name, start, end in windows:
        row.update(_flatten_metrics(window_name, _slice_metrics(equity_df, start, end)))
        row.update(
            _flatten_metrics(
                f"{focus_state}_{window_name}",
                _slice_metrics_by_quadrants(equity_df, quadrant_series, start, end, [focus_state]),
            )
        )
    return row, equity_df


def _render_summary(
    *,
    latest_data_date: str,
    history_window: dict[str, Any],
    ablation_df: pd.DataFrame,
    state_scan_df: pd.DataFrame,
    overlap_summary_df: pd.DataFrame,
    weak_window_name: str | None,
    focus_state: str,
) -> str:
    lines: list[str] = []
    lines.append("# Advanced ML Ensemble Diagnostics")
    lines.append("")
    lines.append(f"- generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- latest_data_date: {latest_data_date}")
    lines.append(
        f"- history_window: {history_window['effective_start_date']} -> {history_window['end_date']} (mode={history_window['mode']})"
    )
    lines.append("")

    if not ablation_df.empty:
        lines.append("## Ablation")
        ablation_view = ablation_df.sort_values(["full_excess_sharpe", "full_excess_total_return"], ascending=[False, False]).reset_index(drop=True)
        for _, row in ablation_view.head(6).iterrows():
            lines.append(
                f"- {row['label']}: full excess Sharpe {_safe_num(row['full_excess_sharpe'])}, "
                f"full excess return {_safe_pct(row['full_excess_total_return'])}"
            )
        if weak_window_name and f"{weak_window_name}_excess_sharpe" in ablation_df.columns:
            lines.append("")
            lines.append(f"### {weak_window_name}")
            weak_view = ablation_df.sort_values(
                [f"{weak_window_name}_excess_sharpe", f"{weak_window_name}_excess_total_return"],
                ascending=[False, False],
            ).reset_index(drop=True)
            for _, row in weak_view.head(6).iterrows():
                lines.append(
                    f"- {row['label']}: weak excess Sharpe {_safe_num(row[f'{weak_window_name}_excess_sharpe'])}, "
                    f"weak excess return {_safe_pct(row[f'{weak_window_name}_excess_total_return'])}"
                )
        lines.append("")

    if not state_scan_df.empty:
        lines.append(f"## State-only scan ({focus_state})")
        rank_cols: list[str] = []
        if weak_window_name and f"{focus_state}_{weak_window_name}_excess_sharpe" in state_scan_df.columns:
            rank_cols.extend([f"{focus_state}_{weak_window_name}_excess_sharpe", f"{focus_state}_{weak_window_name}_excess_total_return"])
        rank_cols.extend(["full_excess_sharpe", "full_excess_total_return"])
        state_view = state_scan_df.sort_values(rank_cols, ascending=[False] * len(rank_cols)).reset_index(drop=True)
        for _, row in state_view.head(8).iterrows():
            state_weight_map = json.loads(row["state_ensemble_weights"] or "{}")
            focus_weights = state_weight_map.get(focus_state, {})
            lines.append(
                f"- {row['label']}: focus weights={focus_weights} | full excess Sharpe {_safe_num(row['full_excess_sharpe'])} "
                f"| focus weak excess Sharpe {_safe_num(row.get(f'{focus_state}_{weak_window_name}_excess_sharpe', np.nan)) if weak_window_name else 'nan'}"
            )
        lines.append("")

    if not overlap_summary_df.empty:
        lines.append("## Overlap")
        sections = ["overall"]
        if weak_window_name:
            sections.append(weak_window_name)
        sections.append(focus_state)
        if weak_window_name:
            sections.append(f"{focus_state}_{weak_window_name}")
        for section in sections:
            section_df = overlap_summary_df.loc[overlap_summary_df["section"] == section]
            if section_df.empty:
                continue
            lines.append(f"### {section}")
            for _, row in section_df.iterrows():
                lines.append(
                    f"- {row['pair']}: pearson {_safe_num(row['mean_pearson'])}, "
                    f"spearman {_safe_num(row['mean_spearman'])}, "
                    f"Top5 overlap {_safe_pct(row['mean_top_overlap_ratio'])}, "
                    f"left-right Top5 gain {_safe_pct(row['left_minus_right_total_next_open_excess'])}"
                )
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()

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

    stocks = parse_stock_list(args.stocks)
    file_stocks = load_stock_list_from_file(args.stocks_file)
    if stocks or file_stocks:
        stocks = list(dict.fromkeys(stocks + file_stocks))
    if stocks:
        cfg.universe = stocks

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
        ensemble_ml_weight=args.ensemble_ml_weight,
        ensemble_none_weight=args.ensemble_none_weight,
        ensemble_v2_weight=args.ensemble_v2_weight,
        train_regime_only=cfg.enable_market_regime_filter,
        execution_mode=cfg.execution_mode,
    )

    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else Path("daily_research/output")
        / (args.experiment_tag.strip() or f"advanced_ml_ensemble_diag_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    )
    output_root.mkdir(parents=True, exist_ok=True)

    windows = parse_named_windows(args.windows)
    weak_window_name = _resolve_weak_window_name(windows)
    focus_state = str(args.focus_state).strip().lower() or DEFAULT_FOCUS_STATE
    base_weights = {
        "ml": float(args.ensemble_ml_weight),
        "none": float(args.ensemble_none_weight),
        "v2": float(args.ensemble_v2_weight),
    }
    state_ml_range = _parse_float_range(args.state_ml_range)
    state_none_range = _parse_float_range(args.state_none_range)
    state_v2_range = _parse_float_range(args.state_v2_range)

    ablation_df = pd.DataFrame()
    state_scan_df = pd.DataFrame()
    overlap_summary_df = pd.DataFrame()

    with StageProgress(total=8, label="ensemble diagnostics") as progress:
        with progress.stage("prepare universe", f"source={args.data_source}"):
            if args.data_source == "tq":
                if args.rolling_liquidity_pool and not cfg.universe:
                    cfg.universe = load_universe_from_tq(cfg.universe_scope)
                elif not cfg.universe and cfg.universe_scope == "all_a":
                    cfg.universe = load_universe_from_tq(cfg.universe_scope)
                elif not cfg.universe:
                    raise ValueError("TQ mode without explicit stocks currently requires all_a universe scope.")
            elif not args.csv_folder:
                raise ValueError("CSV mode requires --csv-folder.")

        with progress.stage("resolve history window", args.start_date):
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

        with progress.stage("load market data", f"stocks={len(cfg.universe)} benchmark={cfg.benchmark}"):
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
            progress.log(
                f"raw cache: {'hit' if raw_cache_meta['cache_hit'] else 'build'} | {raw_cache_meta['cache_path']}"
            )

        with progress.stage("build rolling liquidity pool", args.rolling_liquidity_pool or "fixed universe"):
            rolling_pool_artifact = None
            rolling_membership_mask = None
            prepared_raw_df_dict = raw_df_dict
            if args.rolling_liquidity_pool:
                raw_universe_df_dict = {k: v.drop(columns=[cfg.benchmark], errors="ignore") for k, v in raw_df_dict.items()}
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
                progress.log(f"rolling pool union size: {len(rolling_union)} | rebalance count: {len(rolling_pool_artifact.schedule_df)}")

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
            progress.log(f"factor cache: {'hit' if prepared_cache_meta['cache_hit'] else 'build'} | {prepared_cache_meta['cache_path']}")
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

        with progress.stage("compute shared ml scores", ml_cfg.model_family):
            ml_score_cache_path = _ml_score_cache_path(prepared_cache_meta["cache_key"], ml_cfg)
            cached_ml_payload = None if args.refresh_cache else _load_pickle(ml_score_cache_path)
            if cached_ml_payload is not None:
                training_log = cached_ml_payload["training_log"]
                shared_per_horizon_scores = cached_ml_payload["per_horizon_scores"]
                progress.log(f"ml score cache: hit | {ml_score_cache_path}")
            else:
                _, training_log, shared_per_horizon_scores = rolling_ml_scores_multi_detail(
                    feature_frames=prepared_bundle["feature_frames"],
                    market_features=prepared_bundle["market_features"],
                    close=prepared_bundle["factor_bundle"]["raw_inputs"]["Close"],
                    benchmark_close=prepared_bundle["benchmark_close"],
                    open_df=prepared_bundle["factor_bundle"]["raw_inputs"]["Open"],
                    benchmark_open=prepared_bundle["benchmark_open"],
                    filter_mask=prepared_bundle["filter_mask"],
                    regime_state=prepared_bundle["regime_state"],
                    config=ml_cfg,
                )
                if not args.no_cache:
                    _save_pickle(
                        ml_score_cache_path,
                        {
                            "training_log": training_log,
                            "per_horizon_scores": shared_per_horizon_scores,
                        },
                    )
                    progress.log(f"ml score cache: build | {ml_score_cache_path}")
            training_log.to_csv(output_root / "shared_training_log.csv", index=False, encoding="utf-8-sig")

        with progress.stage("run backtest diagnostics", "ablation + state-only"):
            ablation_rows: list[dict[str, Any]] = []
            if not args.skip_ablation:
                ablation_runs_dir = output_root / "ablation_runs"
                ablation_runs_dir.mkdir(parents=True, exist_ok=True)
                ablation_candidates = _build_ablation_candidates(base_weights, args.ablation_weight_mode)
                for candidate in iter_progress(ablation_candidates, total=len(ablation_candidates), desc="ablation runs", unit="run", position=1):
                    row, equity_df = _run_candidate(
                        candidate=candidate,
                        cfg=cfg,
                        ml_cfg=ml_cfg,
                        shared_per_horizon_scores=shared_per_horizon_scores,
                        prepared_bundle=prepared_bundle,
                        current_membership_mask=current_membership_mask,
                        industry_map=industry_map,
                        style_map=style_map,
                        windows=windows,
                        focus_state=focus_state,
                    )
                    ablation_rows.append(row)
                    run_dir = ablation_runs_dir / str(candidate["label"])
                    run_dir.mkdir(parents=True, exist_ok=True)
                    equity_df.reset_index().rename(columns={"index": "date"}).to_csv(run_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
                    (run_dir / "metrics.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")

            state_candidates = _build_state_scan_candidates(
                focus_state=focus_state,
                step=args.state_scan_step,
                base_weights=base_weights,
                ml_range=state_ml_range,
                none_range=state_none_range,
                v2_range=state_v2_range,
            )
            state_rows: list[dict[str, Any]] = []
            for candidate in iter_progress(state_candidates, total=len(state_candidates), desc="state-only scan", unit="run", position=1):
                row, _ = _run_candidate(
                    candidate=candidate,
                    cfg=cfg,
                    ml_cfg=ml_cfg,
                    shared_per_horizon_scores=shared_per_horizon_scores,
                    prepared_bundle=prepared_bundle,
                    current_membership_mask=current_membership_mask,
                    industry_map=industry_map,
                    style_map=style_map,
                    windows=windows,
                    focus_state=focus_state,
                )
                state_rows.append(row)

            ablation_df = pd.DataFrame(ablation_rows)
            state_scan_df = pd.DataFrame(state_rows)
            if not state_scan_df.empty and weak_window_name:
                sort_cols = [
                    f"{focus_state}_{weak_window_name}_excess_sharpe",
                    f"{focus_state}_{weak_window_name}_excess_total_return",
                    "full_excess_sharpe",
                    "full_excess_total_return",
                ]
                state_scan_df = state_scan_df.sort_values(sort_cols, ascending=[False, False, False, False]).reset_index(drop=True)
            ablation_df.to_csv(output_root / "ablation_summary.csv", index=False, encoding="utf-8-sig")
            state_scan_df.to_csv(output_root / "state_only_scan_summary.csv", index=False, encoding="utf-8-sig")
            state_scan_df.head(max(int(args.state_scan_topn), 1)).to_csv(output_root / "state_only_scan_top.csv", index=False, encoding="utf-8-sig")

        with progress.stage("run overlap diagnostics", f"top{args.top_n_overlap}"):
            if args.skip_overlap:
                overlap_summary_df = pd.DataFrame()
            else:
                base_ml_score = combine_per_horizon_ml_scores(
                    per_horizon_scores=shared_per_horizon_scores,
                    close=prepared_bundle["factor_bundle"]["raw_inputs"]["Close"],
                    regime_state=prepared_bundle["regime_state"],
                    config=ml_cfg,
                )
                if current_membership_mask is not None:
                    base_ml_score = base_ml_score.where(current_membership_mask)
                overlap_daily_df = _build_overlap_daily(
                    score_map={"ml": base_ml_score, "none": prepared_bundle["score_none"], "v2": prepared_bundle["score_v2"]},
                    valid_mask=prepared_bundle["filter_mask"],
                    quadrant_series=prepared_bundle["regime_state"]["quadrant"],
                    next_open_excess=_next_open_excess_by_signal_date(
                        prepared_bundle["factor_bundle"]["raw_inputs"]["Open"],
                        prepared_bundle["benchmark_open"],
                    ),
                    top_n=args.top_n_overlap,
                )
                overlap_daily_df.to_csv(output_root / "overlap_daily.csv", index=False, encoding="utf-8-sig")

                overlap_summary_rows: list[dict[str, Any]] = []
                overlap_summary_rows.extend(_summarize_overlap_section(overlap_daily_df, "overall"))
                overlap_summary_rows.extend(_summarize_overlap_section(overlap_daily_df.loc[overlap_daily_df["quadrant"].eq(focus_state)], focus_state))
                if weak_window_name:
                    _, weak_start, weak_end = next(item for item in windows if item[0] == weak_window_name)
                    weak_mask = (overlap_daily_df["date"] >= pd.Timestamp(weak_start)) & (overlap_daily_df["date"] <= pd.Timestamp(weak_end))
                    overlap_summary_rows.extend(_summarize_overlap_section(overlap_daily_df.loc[weak_mask], weak_window_name))
                    overlap_summary_rows.extend(
                        _summarize_overlap_section(
                            overlap_daily_df.loc[weak_mask & overlap_daily_df["quadrant"].eq(focus_state)],
                            f"{focus_state}_{weak_window_name}",
                        )
                    )
                overlap_summary_df = pd.DataFrame(overlap_summary_rows)
                overlap_summary_df.to_csv(output_root / "overlap_summary.csv", index=False, encoding="utf-8-sig")

    latest_data_date = pd.Timestamp(prepared_bundle["factor_bundle"]["raw_inputs"]["Close"].dropna(how="all").index.max()).strftime("%Y-%m-%d")
    scan_meta = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "latest_data_date": latest_data_date,
        "history_window": history_window_to_dict(history_window),
        "raw_cache": raw_cache_meta,
        "prepared_cache": prepared_cache_meta,
        "rolling_liquidity_pool": args.rolling_liquidity_pool or "",
        "rolling_pool_rebalance_days": int(args.pool_rebalance_days),
        "rolling_pool_adv_window": int(args.pool_adv_window),
        "rolling_pool_union_size": 0 if rolling_membership_mask is None else int(rolling_membership_mask.columns[rolling_membership_mask.any(axis=0)].size),
        "rolling_pool_rebalance_count": 0 if rolling_pool_artifact is None else int(len(rolling_pool_artifact.schedule_df)),
        "base_config": {"research": asdict(cfg), "ml": asdict(ml_cfg)},
        "focus_state": focus_state,
        "ablation_weight_mode": args.ablation_weight_mode,
        "state_scan_step": float(args.state_scan_step),
        "state_ml_range": list(state_ml_range) if state_ml_range is not None else [],
        "state_none_range": list(state_none_range) if state_none_range is not None else [],
        "state_v2_range": list(state_v2_range) if state_v2_range is not None else [],
        "state_scan_topn": int(args.state_scan_topn),
        "top_n_overlap": int(args.top_n_overlap),
        "skip_ablation": bool(args.skip_ablation),
        "skip_overlap": bool(args.skip_overlap),
        "windows": [{"name": name, "start": start, "end": end} for name, start, end in windows],
    }
    (output_root / "scan_config.json").write_text(json.dumps(scan_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_text = _render_summary(
        latest_data_date=latest_data_date,
        history_window=history_window_to_dict(history_window),
        ablation_df=ablation_df,
        state_scan_df=state_scan_df.head(max(int(args.state_scan_topn), 1)),
        overlap_summary_df=overlap_summary_df,
        weak_window_name=weak_window_name,
        focus_state=focus_state,
    )
    (output_root / "summary.md").write_text(summary_text, encoding="utf-8")

    print(f"Output: {output_root}")
    print(summary_text)


if __name__ == "__main__":
    main()
