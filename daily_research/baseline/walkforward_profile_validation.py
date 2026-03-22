from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from daily_research.baseline.alpha import combine_scores_by_state
from daily_research.baseline.backtest import backtest
from daily_research.baseline.config import ResearchConfig
from daily_research.baseline.data_provider import load_daily_from_tq, load_style_map_from_tq, load_universe_from_tq, split_benchmark_from_universe
from daily_research.baseline.features import compute_factors
from daily_research.baseline.portfolio import build_target_weights
from daily_research.baseline.regime import apply_market_regime_filter, compute_market_regime_state
from daily_research.baseline.state_profiles import build_state_configs


def parse_args():
    parser = argparse.ArgumentParser(description="Walk-forward validate state alpha profiles on yearly windows")
    parser.add_argument("--start-date", default="20210101")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--train-years", type=int, default=2)
    parser.add_argument("--rebalance-freq", default="5d")
    parser.add_argument("--regime-max-annual-vol", type=float, default=0.32)
    parser.add_argument("--regime-quadrants", default="trend_up_low_vol,trend_up_high_vol")
    parser.add_argument("--style-cap", action="store_true")
    parser.add_argument("--max-style-weight", type=float, default=0.50)
    parser.add_argument("--profiles", default="none,up_low_breakout_v1,up_dual_v1")
    parser.add_argument("--experiment-tag", default="")
    return parser.parse_args()


def _parse_csv_list(raw: str) -> List[str]:
    return [item.strip().lower() for item in str(raw).split(",") if item.strip()]


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
    close_slice = close.loc[start_dt:end_dt]
    benchmark_slice = benchmark_close.loc[start_dt:end_dt]
    weight_slice = target_weights.loc[start_dt:end_dt]
    score_slice = target_scores.loc[start_dt:end_dt]
    regime_slice = regime_on.loc[start_dt:end_dt]
    return backtest(
        close=close_slice,
        benchmark_close=benchmark_slice,
        target_weights=weight_slice,
        target_scores=score_slice,
        config=cfg,
        regime_on=regime_slice,
    )


def _year_span(index: pd.DatetimeIndex, year: int) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    mask = index.year == year
    if not bool(mask.any()):
        return None
    dates = index[mask]
    return dates.min(), dates.max()


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


def main():
    args = parse_args()
    profiles = _parse_csv_list(args.profiles)
    cfg = ResearchConfig(
        start_date=args.start_date,
        benchmark=args.benchmark,
        universe_scope="all_a",
        weighting_method="score",
        rebalance_freq=args.rebalance_freq,
        enable_market_regime_filter=True,
        regime_max_annual_vol=args.regime_max_annual_vol,
        regime_allowed_quadrants=_parse_csv_list(args.regime_quadrants),
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

    style_map = None
    if cfg.enable_style_cap:
        print("[4/6] 加载风格映射...")
        style_map = load_style_map_from_tq(list(close.columns))

    print("[5/6] 预计算各 profile 的目标组合...")
    profile_artifacts = {
        profile: _build_profile_artifacts(factor_bundle, benchmark_close, cfg, profile, style_map)
        for profile in profiles
    }

    years = sorted(set(close.index.year))
    test_years = years[args.train_years:]
    train_rows = []
    test_rows = []
    stitched_segments = []
    stitched_actions = []

    print("[6/6] 执行滚动样本外验证...")
    for test_year in test_years:
        train_window_years = [year for year in years if year < test_year][-args.train_years:]
        if len(train_window_years) < args.train_years:
            continue
        train_start, _ = _year_span(close.index, train_window_years[0])
        _, train_end = _year_span(close.index, train_window_years[-1])
        test_span = _year_span(close.index, test_year)
        if train_start is None or train_end is None or test_span is None:
            continue
        test_start, test_end = test_span

        best_profile = None
        best_score = None
        best_train_metrics = None

        for profile_name, artifacts in profile_artifacts.items():
            _, _, train_metrics = _slice_backtest(
                close=close,
                benchmark_close=benchmark_close,
                target_weights=artifacts["target_weights"],
                target_scores=artifacts["target_scores"],
                regime_on=artifacts["regime_state"]["regime_on"],
                cfg=cfg,
                start_dt=train_start,
                end_dt=train_end,
            )
            train_rows.append(
                {
                    "test_year": int(test_year),
                    "train_years": ",".join(map(str, train_window_years)),
                    "profile": profile_name,
                    **train_metrics,
                }
            )
            ranking_score = float(train_metrics.get("excess_sharpe", 0.0))
            if best_score is None or ranking_score > best_score:
                best_score = ranking_score
                best_profile = profile_name
                best_train_metrics = train_metrics

        selected = profile_artifacts[best_profile]
        test_equity, test_actions, test_metrics = _slice_backtest(
            close=close,
            benchmark_close=benchmark_close,
            target_weights=selected["target_weights"],
            target_scores=selected["target_scores"],
            regime_on=selected["regime_state"]["regime_on"],
            cfg=cfg,
            start_dt=test_start,
            end_dt=test_end,
        )
        stitched_segments.append(test_equity)
        if not test_actions.empty:
            test_actions = test_actions.copy()
            test_actions["test_year"] = int(test_year)
            test_actions["selected_profile"] = best_profile
            stitched_actions.append(test_actions)
        test_rows.append(
            {
                "test_year": int(test_year),
                "train_years": ",".join(map(str, train_window_years)),
                "selected_profile": best_profile,
                "train_excess_sharpe": float(best_train_metrics.get("excess_sharpe", 0.0)) if best_train_metrics else None,
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
            return float(series.std() * (252 ** 0.5)) if len(series) > 1 else 0.0

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
            "train_years": int(args.train_years),
        }

    out_root = Path(__file__).resolve().parents[1] / "output"
    run_name = args.experiment_tag.strip() or datetime.now().strftime("walkforward_%Y%m%d_%H%M%S")
    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train_rows).to_csv(out_dir / "train_profile_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(test_rows).to_csv(out_dir / "test_window_metrics.csv", index=False, encoding="utf-8-sig")
    if not stitched_equity.empty:
        stitched_equity.to_csv(out_dir / "walkforward_equity_curve.csv", encoding="utf-8-sig")
    if stitched_actions:
        pd.concat(stitched_actions, ignore_index=True).to_csv(out_dir / "walkforward_actions.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "walkforward_metrics.json", "w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, ensure_ascii=False, indent=2)

    print(f"输出目录: {out_dir}")
    if overall_metrics:
        print(json.dumps(overall_metrics, ensure_ascii=False, indent=2))
    if test_rows:
        print(pd.DataFrame(test_rows).to_string(index=False))


if __name__ == "__main__":
    main()
