from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from daily_research.baseline.alpha import combine_scores_by_state
from daily_research.baseline.backtest import backtest
from daily_research.baseline.cli_utils import parse_csv_list
from daily_research.baseline.config import ResearchConfig
from daily_research.baseline.data_provider import load_daily_from_tq, load_style_map_from_tq, load_universe_from_tq, split_benchmark_from_universe
from daily_research.baseline.evaluation import compute_forward_returns
from daily_research.baseline.features import compute_factors
from daily_research.baseline.portfolio import build_target_weights
from daily_research.baseline.regime import apply_market_regime_filter, compute_market_regime_state
from daily_research.baseline.state_profiles import build_state_configs


def parse_args():
    parser = argparse.ArgumentParser(description="Walk-forward activate profiles using quadrant RankIC")
    parser.add_argument("--start-date", default="20210101")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--train-years", type=int, default=2)
    parser.add_argument(
        "--train-months",
        type=int,
        default=0,
        help="Optional rolling training window in months. If > 0, it overrides --train-years.",
    )
    parser.add_argument("--rebalance-freq", default="5d")
    parser.add_argument("--regime-max-annual-vol", type=float, default=0.32)
    parser.add_argument("--regime-quadrants", default="trend_up_low_vol,trend_up_high_vol")
    parser.add_argument("--style-cap", action="store_true")
    parser.add_argument("--max-style-weight", type=float, default=0.50)
    parser.add_argument("--profiles", default="none,up_low_breakout_v1,up_dual_v1")
    parser.add_argument(
        "--quadrant-candidates",
        default="",
        help="Optional per-quadrant candidate profiles, e.g. trend_up_low_vol=none,up_low_breakout_v2;trend_up_high_vol=none",
    )
    parser.add_argument("--ic-horizon", type=int, default=10, help="使用多少期前瞻收益计算 RankIC，默认 10")
    parser.add_argument(
        "--activation-frequency",
        default="year",
        choices=["year", "halfyear", "quarter"],
        help="How often to refresh profile activation choices during the test period",
    )
    parser.add_argument("--experiment-tag", default="")
    return parser.parse_args()


def _parse_quadrant_candidates(raw: str, default_profiles: List[str], quadrants: List[str]) -> Dict[str, List[str]]:
    mapping = {quadrant: list(default_profiles) for quadrant in quadrants}
    raw = str(raw or "").strip()
    if not raw:
        return mapping

    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        quadrant_name, profile_csv = chunk.split("=", 1)
        quadrant_name = quadrant_name.strip().lower()
        if quadrant_name not in mapping:
            continue
        profiles = parse_csv_list(profile_csv)
        if profiles:
            mapping[quadrant_name] = profiles
    return mapping


def _apply_rebalance_frequency(frame: pd.DataFrame, rebalance_freq: str) -> pd.DataFrame:
    rebalance_freq = str(rebalance_freq or "1d").lower().strip()
    if rebalance_freq in {"1d", "d", "daily"}:
        return frame.copy()
    step = int(rebalance_freq[:-1])
    return frame.loc[frame.index[::step]].reindex(frame.index).ffill().fillna(0.0)


def _build_profile_artifacts(
    factor_bundle: Dict[str, Dict[str, pd.DataFrame]],
    benchmark_close: pd.Series,
    cfg: ResearchConfig,
    profile_name: str,
    style_map: pd.DataFrame | None,
):
    regime_state = compute_market_regime_state(benchmark_close, cfg)
    state_configs = build_state_configs(cfg, profile_name)
    score, _, _ = combine_scores_by_state(
        factor_bundle,
        cfg,
        quadrant_series=regime_state["quadrant"],
        state_configs=state_configs,
    )
    target_weights = build_target_weights(score, cfg, style_map=style_map)
    target_weights = _apply_rebalance_frequency(target_weights, cfg.rebalance_freq)
    score_bt = _apply_rebalance_frequency(score.fillna(0.0), cfg.rebalance_freq)
    if cfg.enable_market_regime_filter:
        target_weights, score_bt = apply_market_regime_filter(target_weights, score_bt, regime_state)
    return {
        "score": score,
        "target_weights": target_weights,
        "target_scores": score_bt,
        "regime_state": regime_state,
    }


def _rank_corr(x: pd.Series, y: pd.Series) -> float:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 2:
        return np.nan
    if pair.iloc[:, 0].nunique() <= 1 or pair.iloc[:, 1].nunique() <= 1:
        return np.nan
    xr = pair.iloc[:, 0].rank(method="average")
    yr = pair.iloc[:, 1].rank(method="average")
    return float(xr.corr(yr))


def _quadrant_rank_ic(
    score: pd.DataFrame,
    forward_ret: pd.DataFrame,
    quadrant_series: pd.Series,
    quadrant_name: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> float:
    rows = []
    for dt in score.loc[start_dt:end_dt].index:
        if quadrant_series.get(dt) != quadrant_name:
            continue
        rows.append(_rank_corr(score.loc[dt], forward_ret.loc[dt]))
    valid = pd.Series(rows).dropna()
    return float(valid.mean()) if not valid.empty else float("-inf")


def _slice_backtest(
    close: pd.DataFrame,
    benchmark_close: pd.Series,
    target_weights: pd.DataFrame,
    target_scores: pd.DataFrame,
    regime_on: pd.Series,
    cfg: ResearchConfig,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
):
    return backtest(
        close=close.loc[start_dt:end_dt],
        benchmark_close=benchmark_close.loc[start_dt:end_dt],
        target_weights=target_weights.loc[start_dt:end_dt],
        target_scores=target_scores.loc[start_dt:end_dt],
        config=cfg,
        regime_on=regime_on.loc[start_dt:end_dt],
    )


def _year_span(index: pd.DatetimeIndex, year: int) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    mask = index.year == year
    if not bool(mask.any()):
        return None
    dates = index[mask]
    return dates.min(), dates.max()


def _build_test_windows(
    index: pd.DatetimeIndex,
    train_years: int,
    train_months: int,
    activation_frequency: str,
) -> List[Tuple[str, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp, List[int]]]:
    activation_frequency = str(activation_frequency or "year").lower().strip()
    if activation_frequency == "year":
        period_series = pd.Series(index.to_period("Y"), index=index)
    elif activation_frequency == "quarter":
        period_series = pd.Series(index.to_period("Q"), index=index)
    elif activation_frequency == "halfyear":
        labels = [f"{dt.year}H{1 if dt.month <= 6 else 2}" for dt in index]
        period_series = pd.Series(labels, index=index)
    else:
        raise ValueError(f"Unsupported activation frequency: {activation_frequency}")

    windows: List[Tuple[str, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp, List[int]]] = []
    for label in pd.unique(period_series):
        mask = period_series == label
        dates = index[mask]
        if len(dates) == 0:
            continue
        test_start = dates.min()
        test_end = dates.max()
        prior_dates = index[index < test_start]
        if len(prior_dates) == 0:
            continue
        train_end = prior_dates.max()
        effective_train_months = int(train_months) if int(train_months) > 0 else int(train_years) * 12
        train_start_target = test_start - pd.DateOffset(months=effective_train_months)
        train_slice = index[(index >= train_start_target) & (index < test_start)]
        if len(train_slice) == 0:
            continue
        if train_slice.min() > train_start_target + pd.DateOffset(days=31):
            continue
        train_year_list = sorted(set(train_slice.year))
        min_year_buckets = max(1, int(np.ceil(effective_train_months / 12.0)))
        if len(train_year_list) < min_year_buckets:
            continue
        windows.append((str(label), train_slice.min(), train_end, test_start, test_end, train_year_list))
    return windows


def _combine_equity_segments(segments: List[pd.DataFrame]) -> pd.DataFrame:
    running_portfolio = 1.0
    running_benchmark = 1.0
    rows = []
    for segment in segments:
        for dt, row in segment.iterrows():
            running_portfolio *= 1.0 + float(row["portfolio_return"])
            running_benchmark *= 1.0 + float(row["benchmark_return"])
            rows.append(
                {
                    "date": dt,
                    "portfolio_equity": running_portfolio,
                    "benchmark_equity": running_benchmark,
                    "excess_equity": running_portfolio / running_benchmark if running_benchmark > 0 else None,
                    "portfolio_return": float(row["portfolio_return"]),
                    "benchmark_return": float(row["benchmark_return"]),
                    "excess_return": float(row["excess_return"]),
                    "holding_count": float(row["holding_count"]),
                    "turnover": float(row["turnover"]),
                    "regime_on": bool(row["regime_on"]),
                }
            )
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def _stitch_selected_profile_outputs(
    profile_artifacts: Dict[str, dict],
    quadrant_selection: Dict[str, str],
    quadrant_series: pd.Series,
    index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    first = next(iter(profile_artifacts.values()))
    combined_weights = pd.DataFrame(0.0, index=index, columns=first["target_weights"].columns)
    combined_scores = pd.DataFrame(0.0, index=index, columns=first["target_scores"].columns)

    for dt in index:
        quadrant = quadrant_series.get(dt)
        selected_profile = quadrant_selection.get(quadrant, "none")
        artifacts = profile_artifacts[selected_profile]
        combined_weights.loc[dt] = artifacts["target_weights"].loc[dt]
        combined_scores.loc[dt] = artifacts["target_scores"].loc[dt]

    return combined_weights, combined_scores


def main():
    args = parse_args()
    profiles = parse_csv_list(args.profiles)
    allowed_quadrants = parse_csv_list(args.regime_quadrants)
    quadrant_candidates = _parse_quadrant_candidates(args.quadrant_candidates, profiles, allowed_quadrants)
    cfg = ResearchConfig(
        start_date=args.start_date,
        benchmark=args.benchmark,
        universe_scope="all_a",
        weighting_method="score",
        rebalance_freq=args.rebalance_freq,
        enable_market_regime_filter=True,
        regime_max_annual_vol=args.regime_max_annual_vol,
        regime_allowed_quadrants=allowed_quadrants,
        enable_style_cap=args.style_cap,
        max_style_weight=args.max_style_weight,
    )

    print("[1/6] 加载全A股票池...")
    universe = load_universe_from_tq("all_a")
    print(f"[2/6] 拉取日线数据，股票数: {len(universe)}")
    raw_df_dict = load_daily_from_tq(universe, args.start_date, benchmark=args.benchmark)
    df_dict, benchmark_close = split_benchmark_from_universe(raw_df_dict, args.benchmark)

    print("[3/6] 计算因子...")
    factor_bundle = compute_factors(df_dict)
    close = factor_bundle["raw_inputs"]["Close"]
    forward_returns = compute_forward_returns(close, horizons=[args.ic_horizon])[f"fwd_{args.ic_horizon}d"]

    style_map = None
    if cfg.enable_style_cap:
        print("[4/6] 加载风格映射...")
        style_map = load_style_map_from_tq(list(close.columns))

    print("[5/6] 预计算各 profile 的目标组合...")
    profile_artifacts = {
        profile: _build_profile_artifacts(factor_bundle, benchmark_close, cfg, profile, style_map)
        for profile in profiles
    }
    base_regime_state = next(iter(profile_artifacts.values()))["regime_state"]
    quadrant_series = base_regime_state["quadrant"]
    regime_on = base_regime_state["regime_on"]

    print("[6/6] 执行基于象限 RankIC 的滚动激活验证...")
    test_windows = _build_test_windows(
        close.index,
        args.train_years,
        args.train_months,
        args.activation_frequency,
    )
    selection_rows = []
    test_rows = []
    stitched_segments = []

    for window_label, train_start, train_end, test_start, test_end, train_window_years in test_windows:
        quadrant_selection: Dict[str, str] = {}
        for quadrant_name in allowed_quadrants:
            candidate_profiles = quadrant_candidates.get(quadrant_name, profiles)
            best_profile = None
            best_rank_ic = None
            for profile_name in candidate_profiles:
                artifacts = profile_artifacts[profile_name]
                rank_ic = _quadrant_rank_ic(
                    score=artifacts["score"],
                    forward_ret=forward_returns,
                    quadrant_series=quadrant_series,
                    quadrant_name=quadrant_name,
                    start_dt=train_start,
                    end_dt=train_end,
                )
                selection_rows.append(
                    {
                        "window_label": window_label,
                        "test_year": int(test_start.year),
                        "train_years": ",".join(map(str, train_window_years)),
                        "quadrant": quadrant_name,
                        "profile": profile_name,
                        "train_rank_ic": rank_ic,
                    }
                )
                if best_rank_ic is None or rank_ic > best_rank_ic:
                    best_rank_ic = rank_ic
                    best_profile = profile_name
            quadrant_selection[quadrant_name] = best_profile

        year_index = close.loc[test_start:test_end].index
        selected_weights, selected_scores = _stitch_selected_profile_outputs(
            profile_artifacts=profile_artifacts,
            quadrant_selection=quadrant_selection,
            quadrant_series=quadrant_series,
            index=year_index,
        )
        test_equity, _, test_metrics = _slice_backtest(
            close=close,
            benchmark_close=benchmark_close,
            target_weights=selected_weights,
            target_scores=selected_scores,
            regime_on=regime_on,
            cfg=cfg,
            start_dt=test_start,
            end_dt=test_end,
        )
        stitched_segments.append(test_equity)
        test_rows.append(
            {
                "window_label": window_label,
                "test_year": int(test_start.year),
                "train_years": ",".join(map(str, train_window_years)),
                **{f"{quadrant}_profile": profile for quadrant, profile in quadrant_selection.items()},
                **test_metrics,
            }
        )

    stitched_equity = _combine_equity_segments(stitched_segments)
    overall_metrics = {}
    if not stitched_equity.empty:
        portfolio_equity = stitched_equity["portfolio_equity"]
        benchmark_equity = stitched_equity["benchmark_equity"]
        excess_equity = stitched_equity["excess_equity"]
        portfolio_returns = stitched_equity["portfolio_return"]
        excess_returns = stitched_equity["excess_return"]

        def _ann_ret(series: pd.Series) -> float:
            ret = series.pct_change().dropna()
            if ret.empty:
                return 0.0
            return float(series.iloc[-1] ** (252 / len(ret)) - 1.0)

        def _ann_vol(series: pd.Series) -> float:
            return float(series.std() * np.sqrt(252)) if len(series) > 1 else 0.0

        def _mdd(series: pd.Series) -> float:
            peak = series.cummax()
            return float((series / peak - 1.0).min())

        portfolio_ann = _ann_ret(portfolio_equity)
        excess_ann = _ann_ret(excess_equity)
        portfolio_vol = _ann_vol(portfolio_returns)
        excess_vol = _ann_vol(excess_returns)
        overall_metrics = {
            "total_return": float(portfolio_equity.iloc[-1] - 1.0),
            "benchmark_total_return": float(benchmark_equity.iloc[-1] - 1.0),
            "excess_total_return": float(excess_equity.iloc[-1] - 1.0),
            "annual_return": float(portfolio_ann),
            "excess_annual_return": float(excess_ann),
            "sharpe": float(portfolio_ann / portfolio_vol) if portfolio_vol > 0 else 0.0,
            "excess_sharpe": float(excess_ann / excess_vol) if excess_vol > 0 else 0.0,
            "max_drawdown": float(_mdd(portfolio_equity)),
            "excess_max_drawdown": float(_mdd(excess_equity)),
            "avg_holding_count": float(stitched_equity["holding_count"].mean()),
            "avg_turnover": float(stitched_equity["turnover"].mean()),
            "regime_active_ratio": float(stitched_equity["regime_on"].mean()),
            "profiles": profiles,
            "quadrant_candidates": quadrant_candidates,
            "train_years": int(args.train_years),
            "train_months": int(args.train_months),
            "ic_horizon": int(args.ic_horizon),
            "activation_frequency": args.activation_frequency,
            "regime_quadrants": allowed_quadrants,
        }

    out_root = Path(__file__).resolve().parents[1] / "output"
    run_name = args.experiment_tag.strip() or datetime.now().strftime("quadrant_ic_activation_%Y%m%d_%H%M%S")
    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(selection_rows).to_csv(out_dir / "quadrant_profile_rankic.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(test_rows).to_csv(out_dir / "test_window_metrics.csv", index=False, encoding="utf-8-sig")
    if not stitched_equity.empty:
        stitched_equity.to_csv(out_dir / "activated_equity_curve.csv", encoding="utf-8-sig")
    with open(out_dir / "activation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, ensure_ascii=False, indent=2)

    print(f"输出目录: {out_dir}")
    if overall_metrics:
        print(json.dumps(overall_metrics, ensure_ascii=False, indent=2))
    if test_rows:
        print(pd.DataFrame(test_rows).to_string(index=False))


if __name__ == "__main__":
    main()
