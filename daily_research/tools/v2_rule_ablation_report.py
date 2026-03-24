from __future__ import annotations

import sys
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json

import numpy as np
import pandas as pd

from daily_research.baseline.alpha import combine_scores_by_state
from daily_research.baseline.backtest import backtest
from daily_research.baseline.config import ResearchConfig
from daily_research.baseline.data_provider import (
    get_latest_completed_trading_date,
    load_industry_map_from_tq,
    load_style_map_from_tq,
    load_universe_from_tq,
    split_benchmark_from_universe,
)
from daily_research.baseline.portfolio import build_target_weights
from daily_research.baseline.regime import apply_market_regime_filter, compute_market_regime_state
from daily_research.baseline.state_profiles import build_state_configs
from daily_research.baseline.advanced_ml_runtime import (
    history_window_to_dict,
    load_raw_data_with_cache,
    resolve_history_window,
)
from daily_research.baseline.ml_alpha import MLAplhaConfig
from daily_research.execution.liquidity_universe import build_rolling_liquidity_membership


DEFAULT_WINDOWS = "recent_full:20250307:20260319,latest_weak:20250905:20260319"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run v2 rule-layer ablation diagnostics under the current ma50 execution setting."
    )
    parser.add_argument("--data-source", choices=["tq"], default="tq")
    parser.add_argument(
        "--rolling-liquidity-pool",
        choices=["liquid300", "liquid500", "liquid800"],
        default="liquid500",
    )
    parser.add_argument("--pool-rebalance-days", type=int, default=21)
    parser.add_argument("--pool-adv-window", type=int, default=20)
    parser.add_argument("--start-date", default="20210101")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--holding-count", type=int, default=5)
    parser.add_argument("--rebalance-freq", default="5d")
    parser.add_argument("--max-weight", type=float, default=0.25)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--min-adv20", type=float, default=50_000.0)
    parser.add_argument("--min-price", type=float, default=2.0)
    parser.add_argument("--max-price", type=float, default=300.0)
    parser.add_argument("--regime-ma-window", type=int, default=50)
    parser.add_argument("--regime-vol-window", type=int, default=20)
    parser.add_argument("--regime-max-annual-vol", type=float, default=0.32)
    parser.add_argument("--regime-quadrants", default="trend_up_low_vol,trend_up_high_vol")
    parser.add_argument("--no-style-cap", action="store_true")
    parser.add_argument("--max-style-weight", type=float, default=0.50)
    parser.add_argument("--industry-cap", action="store_true")
    parser.add_argument("--max-industry-weight", type=float, default=0.40)
    parser.add_argument("--windows", default=DEFAULT_WINDOWS)
    parser.add_argument("--rankic-horizon", type=int, default=20)
    parser.add_argument("--enhanced-profile", default="up_low_breakout_v2")
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def _parse_csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _parse_named_windows(raw: str | None) -> list[tuple[str, str, str]]:
    if not raw:
        return []
    windows: list[tuple[str, str, str]] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid window spec: {chunk}")
        name, start, end = parts
        windows.append((name.strip(), start.strip(), end.strip()))
    return windows


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


def _annualized_return(equity: pd.Series) -> float:
    daily_ret = equity.pct_change().dropna()
    if daily_ret.empty:
        return 0.0
    return float(equity.iloc[-1] ** (252 / len(daily_ret)) - 1.0)


def _annualized_vol(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0.0


def _max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min())


def _slice_metrics(equity_df: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    window = equity_df.loc[(equity_df.index >= pd.Timestamp(start)) & (equity_df.index <= pd.Timestamp(end))].copy()
    if window.empty:
        return {
            "holdout_start": pd.Timestamp(start).strftime("%Y-%m-%d"),
            "holdout_end": pd.Timestamp(end).strftime("%Y-%m-%d"),
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
    excess_ann_ret = _annualized_return(excess_equity) if not excess_equity.empty else 0.0
    portfolio_ann_vol = _annualized_vol(portfolio_returns)
    excess_ann_vol = _annualized_vol(excess_returns)

    return {
        "holdout_start": pd.Timestamp(start).strftime("%Y-%m-%d"),
        "holdout_end": pd.Timestamp(end).strftime("%Y-%m-%d"),
        "total_return": float(portfolio_equity.iloc[-1] - 1.0),
        "annual_return": float(portfolio_ann_ret),
        "annual_vol": float(portfolio_ann_vol),
        "sharpe": float(portfolio_ann_ret / portfolio_ann_vol) if portfolio_ann_vol > 0 else 0.0,
        "benchmark_total_return": float(benchmark_equity.iloc[-1] - 1.0),
        "benchmark_annual_return": float(benchmark_ann_ret),
        "excess_total_return": float(excess_equity.iloc[-1] - 1.0) if not excess_equity.empty else 0.0,
        "excess_annual_return": float(excess_ann_ret),
        "excess_sharpe": float(excess_ann_ret / excess_ann_vol) if excess_ann_vol > 0 else 0.0,
        "excess_max_drawdown": float(_max_drawdown(excess_equity)) if not excess_equity.empty else 0.0,
        "max_drawdown": float(_max_drawdown(portfolio_equity)),
        "avg_holding_count": float(window["holding_count"].mean()),
        "avg_turnover": float(window["turnover"].mean()),
        "hit_rate": float((portfolio_returns > 0).mean()) if not portfolio_returns.empty else 0.0,
        "regime_active_ratio": float(window["regime_on"].mean()) if "regime_on" in window else np.nan,
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


def _safe_pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return f"{float(value):.2%}"


def _safe_num(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return f"{float(value):.3f}"


def _candidate_specs() -> list[dict[str, Any]]:
    return [
        {"label": "none", "kind": "baseline", "profile": "none"},
        {"label": "v2", "kind": "baseline", "profile": "v2"},
        {"label": "ablate_group_volume", "kind": "group", "profile": "v2", "zero_group": "volume"},
        {"label": "ablate_group_volatility", "kind": "group", "profile": "v2", "zero_group": "volatility"},
        {"label": "ablate_group_structure", "kind": "group", "profile": "v2", "zero_group": "structure"},
        {
            "label": "ablate_factor_volume_contraction",
            "kind": "factor",
            "profile": "v2",
            "zero_factor": "volume_contraction",
        },
        {
            "label": "ablate_factor_price_volume_divergence",
            "kind": "factor",
            "profile": "v2",
            "zero_factor": "price_volume_divergence",
        },
        {
            "label": "ablate_factor_volatility_20",
            "kind": "factor",
            "profile": "v2",
            "zero_factor": "volatility_20",
        },
        {
            "label": "ablate_factor_volatility_contraction",
            "kind": "factor",
            "profile": "v2",
            "zero_factor": "volatility_contraction",
        },
        {
            "label": "ablate_factor_close_strength",
            "kind": "factor",
            "profile": "v2",
            "zero_factor": "close_strength",
        },
        {
            "label": "ablate_factor_range_position_20",
            "kind": "factor",
            "profile": "v2",
            "zero_factor": "range_position_20",
        },
        {
            "label": "ablate_factor_drawdown_20",
            "kind": "factor",
            "profile": "v2",
            "zero_factor": "drawdown_20",
        },
    ]


def _build_v2_state_config(base_cfg: ResearchConfig, candidate: dict[str, Any]) -> ResearchConfig:
    v2_cfg = deepcopy(build_state_configs(base_cfg, "up_low_breakout_v2")["trend_up_low_vol"])
    zero_group = str(candidate.get("zero_group", "")).strip()
    zero_factor = str(candidate.get("zero_factor", "")).strip()
    if zero_group:
        v2_cfg.factor_group_weights = dict(v2_cfg.factor_group_weights)
        v2_cfg.factor_group_weights[zero_group] = 0.0
    if zero_factor:
        v2_cfg.factor_weights = dict(v2_cfg.factor_weights)
        v2_cfg.factor_weights[zero_factor] = 0.0
    return v2_cfg


def _candidate_state_configs(base_cfg: ResearchConfig, candidate: dict[str, Any]) -> dict[str, ResearchConfig]:
    profile = str(candidate.get("profile", "v2")).strip().lower()
    if profile == "none":
        return {}
    return {"trend_up_low_vol": _build_v2_state_config(base_cfg, candidate)}


def _compute_up_low_rankic(
    score: pd.DataFrame,
    open_df: pd.DataFrame,
    quadrant_series: pd.Series,
    filter_mask: pd.DataFrame,
    horizon: int,
) -> tuple[pd.Series, pd.DataFrame]:
    future_ret = open_df.shift(-int(horizon)).div(open_df).sub(1.0)
    score = score.align(future_ret, join="inner", axis=0)[0]
    future_ret = future_ret.reindex(score.index, columns=score.columns)
    filter_mask = filter_mask.reindex(score.index, columns=score.columns).fillna(False)
    quadrant_series = quadrant_series.reindex(score.index)

    rows: list[dict[str, Any]] = []
    for dt in score.index:
        if str(quadrant_series.loc[dt]) != "trend_up_low_vol":
            continue
        score_row = score.loc[dt].where(filter_mask.loc[dt])
        ret_row = future_ret.loc[dt].where(filter_mask.loc[dt])
        rank_ic = _rank_corr(score_row, ret_row)
        rows.append(
            {
                "date": dt,
                "quarter": str(pd.Timestamp(dt).to_period("Q")),
                "rank_ic": rank_ic,
                "stock_count": int(pd.concat([score_row, ret_row], axis=1).dropna().shape[0]),
            }
        )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.Series(dtype=float), detail
    series = detail.set_index("date")["rank_ic"].sort_index()
    return series, detail


def _rankic_summary(
    rankic_series: pd.Series,
    windows: list[tuple[str, str, str]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "full_rank_ic_mean": float(rankic_series.mean()) if not rankic_series.dropna().empty else np.nan,
        "full_rank_ic_std": float(rankic_series.std()) if len(rankic_series.dropna()) > 1 else np.nan,
        "full_rank_ic_ir": float(rankic_series.mean() / rankic_series.std() * np.sqrt(252))
        if len(rankic_series.dropna()) > 1 and float(rankic_series.std()) > 0
        else np.nan,
        "full_sample_days": int(rankic_series.notna().sum()),
    }
    for name, start, end in windows:
        window = rankic_series.loc[(rankic_series.index >= pd.Timestamp(start)) & (rankic_series.index <= pd.Timestamp(end))]
        summary[f"{name}_rank_ic_mean"] = float(window.mean()) if not window.dropna().empty else np.nan
        summary[f"{name}_sample_days"] = int(window.notna().sum())
    return summary


def _quarter_rankic_frame(label: str, detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame(columns=["label", "quarter", "rank_ic_mean", "sample_days"])
    return (
        detail_df.groupby("quarter", dropna=False)
        .agg(rank_ic_mean=("rank_ic", "mean"), sample_days=("rank_ic", lambda s: int(s.notna().sum())))
        .reset_index()
        .assign(label=label)
        [["label", "quarter", "rank_ic_mean", "sample_days"]]
    )


def _build_report(summary_df: pd.DataFrame) -> dict[str, Any]:
    baseline_v2 = summary_df.loc[summary_df["label"] == "v2"].iloc[0]
    factor_rows = summary_df.loc[summary_df["kind"] == "factor"].copy()
    group_rows = summary_df.loc[summary_df["kind"] == "group"].copy()
    for frame in (factor_rows, group_rows):
        frame["delta_full_excess_sharpe_vs_v2"] = frame["full_excess_sharpe"] - float(baseline_v2["full_excess_sharpe"])
        frame["delta_latest_weak_excess_return_vs_v2"] = (
            frame["latest_weak_excess_total_return"] - float(baseline_v2["latest_weak_excess_total_return"])
        )
        frame["delta_up_low_rankic_vs_v2"] = frame["full_rank_ic_mean"] - float(baseline_v2["full_rank_ic_mean"])

    def _rows(frame: pd.DataFrame, sort_cols: list[str], ascending: list[bool], top_n: int = 3) -> list[dict[str, Any]]:
        if frame.empty:
            return []
        ordered = frame.sort_values(sort_cols, ascending=ascending, na_position="last").head(top_n)
        return ordered[
            [
                "label",
                "full_excess_sharpe",
                "latest_weak_excess_total_return",
                "full_rank_ic_mean",
                "delta_full_excess_sharpe_vs_v2",
                "delta_latest_weak_excess_return_vs_v2",
                "delta_up_low_rankic_vs_v2",
            ]
        ].to_dict(orient="records")

    return {
        "baseline_v2": {
            "full_excess_sharpe": float(baseline_v2["full_excess_sharpe"]),
            "latest_weak_excess_total_return": float(baseline_v2["latest_weak_excess_total_return"]),
            "full_rank_ic_mean": float(baseline_v2["full_rank_ic_mean"]),
        },
        "factor_ablation_most_harmful_by_sharpe": _rows(
            factor_rows,
            ["delta_full_excess_sharpe_vs_v2", "delta_up_low_rankic_vs_v2"],
            [True, True],
        ),
        "factor_ablation_most_helpful_by_sharpe": _rows(
            factor_rows,
            ["delta_full_excess_sharpe_vs_v2", "delta_up_low_rankic_vs_v2"],
            [False, False],
        ),
        "factor_ablation_most_harmful_by_latest_weak": _rows(
            factor_rows,
            ["delta_latest_weak_excess_return_vs_v2", "delta_up_low_rankic_vs_v2"],
            [True, True],
        ),
        "group_ablation_summary": group_rows[
            [
                "label",
                "full_excess_sharpe",
                "latest_weak_excess_total_return",
                "full_rank_ic_mean",
                "delta_full_excess_sharpe_vs_v2",
                "delta_latest_weak_excess_return_vs_v2",
                "delta_up_low_rankic_vs_v2",
            ]
        ].to_dict(orient="records"),
    }


def _write_markdown_report(output_dir: Path, report: dict[str, Any]) -> None:
    baseline = report["baseline_v2"]

    def _render_rows(rows: list[dict[str, Any]]) -> list[str]:
        out: list[str] = []
        for row in rows:
            out.append(
                "- `{label}`: full_excess_sharpe={full_excess_sharpe}, latest_weak_excess_return={latest_weak_excess_total_return}, "
                "rank_ic={full_rank_ic_mean}, d_sharpe={delta_full_excess_sharpe_vs_v2}, d_latest_weak={delta_latest_weak_excess_return_vs_v2}, "
                "d_rank_ic={delta_up_low_rankic_vs_v2}".format(
                    label=row["label"],
                    full_excess_sharpe=_safe_num(row["full_excess_sharpe"]),
                    latest_weak_excess_total_return=_safe_pct(row["latest_weak_excess_total_return"]),
                    full_rank_ic_mean=_safe_num(row["full_rank_ic_mean"]),
                    delta_full_excess_sharpe_vs_v2=_safe_num(row["delta_full_excess_sharpe_vs_v2"]),
                    delta_latest_weak_excess_return_vs_v2=_safe_pct(row["delta_latest_weak_excess_return_vs_v2"]),
                    delta_up_low_rankic_vs_v2=_safe_num(row["delta_up_low_rankic_vs_v2"]),
                )
            )
        return out or ["- 无"]

    lines = [
        "# v2 Rule Ablation Report",
        "",
        "## Baseline",
        f"- `v2`: full_excess_sharpe={_safe_num(baseline['full_excess_sharpe'])}, latest_weak_excess_return={_safe_pct(baseline['latest_weak_excess_total_return'])}, rank_ic={_safe_num(baseline['full_rank_ic_mean'])}",
        "",
        "## Most Harmful Factor Ablations By Full Sharpe",
        *_render_rows(report["factor_ablation_most_harmful_by_sharpe"]),
        "",
        "## Most Helpful Factor Ablations By Full Sharpe",
        *_render_rows(report["factor_ablation_most_helpful_by_sharpe"]),
        "",
        "## Most Harmful Factor Ablations By Latest Weak Window",
        *_render_rows(report["factor_ablation_most_harmful_by_latest_weak"]),
        "",
        "## Group Ablation Summary",
        *_render_rows(report["group_ablation_summary"]),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    windows = _parse_named_windows(args.windows)
    latest_end = str(args.end_date or "").strip() or get_latest_completed_trading_date()

    cfg = ResearchConfig(
        start_date=args.start_date,
        end_date=latest_end,
        universe_scope="all_a",
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
        enable_market_regime_filter=True,
        regime_ma_window=args.regime_ma_window,
        regime_vol_window=args.regime_vol_window,
        regime_max_annual_vol=args.regime_max_annual_vol,
        regime_allowed_quadrants=_parse_csv_list(args.regime_quadrants),
        enable_style_cap=not args.no_style_cap,
        max_style_weight=args.max_style_weight,
        enable_industry_cap=args.industry_cap,
        max_industry_weight=args.max_industry_weight,
    )

    print("[1/7] loading all-A universe...")
    cfg.universe = load_universe_from_tq(cfg.universe_scope)

    history_window = resolve_history_window(
        cfg,
        MLAplhaConfig(),
        requested_start_date=args.start_date,
        end_date=latest_end,
        mode="train",
        auto_trim_history=False,
    )
    raw_df_dict, raw_cache_meta = load_raw_data_with_cache(
        data_source=args.data_source,
        csv_folder=None,
        universe=cfg.universe,
        benchmark=cfg.benchmark,
        history_window=history_window,
        use_cache=not args.no_cache,
        refresh_cache=args.refresh_cache,
    )

    print(
        f"[2/7] raw cache: {'hit' if raw_cache_meta['cache_hit'] else 'build'} | {raw_cache_meta['cache_path']}"
    )
    print(
        f"[3/7] building rolling {args.rolling_liquidity_pool} membership "
        f"(rebalance={args.pool_rebalance_days}d, adv_window={args.pool_adv_window})..."
    )
    raw_universe_df_dict, _ = split_benchmark_from_universe(raw_df_dict, cfg.benchmark)
    rolling_pool_artifact = build_rolling_liquidity_membership(
        close_frame=raw_universe_df_dict["Close"],
        amount_frame=raw_universe_df_dict["Amount"],
        pool_name=args.rolling_liquidity_pool,
        signal_start_date=cfg.start_date,
        signal_end_date=latest_end,
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

    print("[4/7] building shared factor bundle...")
    benchmark_open = prepared_raw_df_dict["Open"][cfg.benchmark].copy()
    df_dict, benchmark_close = split_benchmark_from_universe(prepared_raw_df_dict, cfg.benchmark)
    rolling_membership_mask = rolling_membership_mask.reindex(
        index=df_dict["Close"].index,
        columns=df_dict["Close"].columns,
    ).fillna(False)
    from daily_research.baseline.features import compute_factors

    factor_bundle = compute_factors(df_dict)
    regime_state = compute_market_regime_state(benchmark_close, cfg)

    print("[5/7] loading exposure maps...")
    candidate_columns = list(df_dict["Close"].columns)
    industry_map = load_industry_map_from_tq(candidate_columns) if cfg.enable_industry_cap else None
    style_map = load_style_map_from_tq(candidate_columns) if cfg.enable_style_cap else None

    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else Path("daily_research/output")
        / (args.experiment_tag.strip() or f"v2_rule_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    )
    output_root.mkdir(parents=True, exist_ok=True)
    rolling_pool_artifact.schedule_df.to_csv(output_root / "rolling_liquidity_schedule.csv", index=False, encoding="utf-8-sig")
    rolling_pool_artifact.summary_df.to_csv(output_root / "rolling_liquidity_summary.csv", index=False, encoding="utf-8-sig")
    scan_config = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "history_window": history_window_to_dict(history_window),
        "raw_cache": raw_cache_meta,
        "rolling_liquidity_pool": args.rolling_liquidity_pool,
        "rolling_pool_rebalance_days": int(args.pool_rebalance_days),
        "rolling_pool_adv_window": int(args.pool_adv_window),
        "base_config": asdict(cfg),
        "rankic_horizon": int(args.rankic_horizon),
        "windows": [{"name": name, "start": start, "end": end} for name, start, end in windows],
        "candidates": _candidate_specs(),
    }
    (output_root / "scan_config.json").write_text(json.dumps(scan_config, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[6/7] running candidate ablations...")
    summary_rows: list[dict[str, Any]] = []
    quarter_frames: list[pd.DataFrame] = []

    for candidate in _candidate_specs():
        label = str(candidate["label"])
        print(f"  - {label}")
        state_configs = _candidate_state_configs(cfg, candidate)
        score, _, filter_mask = combine_scores_by_state(
            factor_bundle,
            cfg,
            quadrant_series=regime_state["quadrant"],
            state_configs=state_configs,
        )
        filter_mask = filter_mask & rolling_membership_mask
        score = score.where(rolling_membership_mask)

        target_weights = build_target_weights(score, cfg, industry_map=industry_map, style_map=style_map)
        score_for_backtest = score.fillna(0.0)
        if cfg.enable_market_regime_filter:
            target_weights, score_for_backtest = apply_market_regime_filter(
                target_weights=target_weights,
                target_scores=score_for_backtest,
                regime_state=regime_state,
            )

        equity_df, action_df, metrics = backtest(
            close=factor_bundle["raw_inputs"]["Close"],
            benchmark_close=benchmark_close,
            open_df=factor_bundle["raw_inputs"]["Open"],
            benchmark_open=benchmark_open,
            target_weights=target_weights,
            target_scores=score_for_backtest,
            config=cfg,
            regime_on=regime_state["regime_on"],
        )
        rankic_series, rankic_detail = _compute_up_low_rankic(
            score=score,
            open_df=factor_bundle["raw_inputs"]["Open"],
            quadrant_series=regime_state["quadrant"],
            filter_mask=filter_mask,
            horizon=args.rankic_horizon,
        )
        rankic_metrics = _rankic_summary(rankic_series, windows)

        run_dir = output_root / label
        run_dir.mkdir(parents=True, exist_ok=True)
        equity_df.reset_index().to_csv(run_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
        action_df.to_csv(run_dir / "actions.csv", index=False, encoding="utf-8-sig")
        target_weights.to_csv(run_dir / "target_weights.csv", encoding="utf-8-sig")
        regime_state.to_csv(run_dir / "regime_state.csv", encoding="utf-8-sig")
        rankic_detail.to_csv(run_dir / "up_low_rankic_daily.csv", index=False, encoding="utf-8-sig")
        _quarter_rankic_frame(label, rankic_detail).to_csv(run_dir / "up_low_rankic_quarterly.csv", index=False, encoding="utf-8-sig")
        (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        for name, start, end in windows:
            window_metrics = _slice_metrics(equity_df, start, end)
            (run_dir / f"metrics_{name}.json").write_text(
                json.dumps(window_metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        row = {
            "label": label,
            "kind": candidate["kind"],
            "profile": candidate.get("profile", ""),
            "zero_group": candidate.get("zero_group", ""),
            "zero_factor": candidate.get("zero_factor", ""),
            "full_excess_total_return": metrics.get("excess_total_return"),
            "full_excess_sharpe": metrics.get("excess_sharpe"),
            "full_excess_max_drawdown": metrics.get("excess_max_drawdown"),
            "full_avg_turnover": metrics.get("avg_turnover"),
            **rankic_metrics,
        }
        for name, start, end in windows:
            window_metrics = _slice_metrics(equity_df, start, end)
            row[f"{name}_excess_total_return"] = window_metrics.get("excess_total_return")
            row[f"{name}_excess_sharpe"] = window_metrics.get("excess_sharpe")
            row[f"{name}_excess_max_drawdown"] = window_metrics.get("excess_max_drawdown")
            row[f"{name}_avg_turnover"] = window_metrics.get("avg_turnover")
        summary_rows.append(row)
        quarter_frames.append(_quarter_rankic_frame(label, rankic_detail))

    summary_df = pd.DataFrame(summary_rows)
    baseline_v2 = summary_df.loc[summary_df["label"] == "v2"].iloc[0]
    for col in ["full_excess_sharpe", "latest_weak_excess_total_return", "full_rank_ic_mean"]:
        summary_df[f"{col}_delta_vs_v2"] = summary_df[col] - float(baseline_v2[col])
    summary_df = summary_df.sort_values(
        ["full_excess_sharpe_delta_vs_v2", "latest_weak_excess_total_return_delta_vs_v2"],
        ascending=[True, True],
        na_position="last",
    ).reset_index(drop=True)
    summary_df.to_csv(output_root / "summary_rows.csv", index=False, encoding="utf-8-sig")
    if quarter_frames:
        pd.concat(quarter_frames, axis=0, ignore_index=True).to_csv(
            output_root / "quarter_rankic_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    report = _build_report(summary_df)
    (output_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown_report(output_root, report)

    print("[7/7] done.")
    print(f"Output: {output_root}")
    print(
        summary_df[
            [
                "label",
                "full_excess_sharpe",
                "latest_weak_excess_total_return",
                "full_rank_ic_mean",
                "full_excess_sharpe_delta_vs_v2",
                "latest_weak_excess_total_return_delta_vs_v2",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
