from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json

import pandas as pd

from daily_research.baseline.advanced_ml_runtime import HistoryWindow, load_raw_data_with_cache
from daily_research.baseline.alpha import build_filter_mask
from daily_research.baseline.backtest import backtest, summarize_backtest_by_month, summarize_monthly_diagnostics
from daily_research.baseline.external_target_weight_bridge import (
    apply_rebalance_frequency,
    apply_rebalance_schedule,
    build_target_weight_bridge,
    load_value_panel,
)
from daily_research.baseline.soft_state_sizing import apply_soft_state_sizing, resolve_soft_state_profile
from daily_research.baseline.cli_utils import load_stock_list_from_file, parse_csv_list, parse_stock_list
from daily_research.baseline.config import ResearchConfig
from daily_research.baseline.data_provider import (
    load_industry_map_from_tq,
    load_style_map_from_tq,
    split_benchmark_from_universe,
)
from daily_research.baseline.portfolio import build_target_weights
from daily_research.baseline.regime import apply_market_regime_filter, compute_market_regime_state


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest an external daily score or target-weight panel under the shared execution/backtest engine.")
    parser.add_argument("--score-panel-csv", default=None, help="Path to a daily score panel CSV.")
    parser.add_argument("--score-panel-format", choices=["auto", "long", "wide"], default="auto")
    parser.add_argument("--target-weight-panel-csv", default=None, help="Optional path to a daily target-weight panel CSV.")
    parser.add_argument("--target-weight-panel-format", choices=["auto", "long", "wide"], default="auto")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--stock-column", default="stock")
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--target-weight-column", default="target_weight")
    parser.add_argument("--target-weight-top-k", type=int, default=0, help="Optional top-k crop applied to direct target-weight rows. 0 keeps all names.")
    parser.add_argument("--target-weight-min-weight", type=float, default=0.0, help="Optional minimum weight threshold applied to direct target-weight rows before renormalization.")
    parser.add_argument("--target-weight-power", type=float, default=1.0, help="Optional power transform applied to positive direct target weights before renormalization.")
    parser.add_argument("--target-weight-full-invest", action="store_true", help="When using direct target weights, renormalize positive rows to 100%% gross even if the source leaves cash.")
    parser.add_argument("--data-source", choices=["tq", "csv"], default="tq")
    parser.add_argument("--csv-folder", default=None)
    parser.add_argument("--stocks", default=None)
    parser.add_argument("--stocks-file", default=None, help="Optional txt/csv file containing stock codes.")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--universe-scope", default="all_a")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--holding-count", type=int, default=5)
    parser.add_argument("--rebalance-freq", default="1d")
    parser.add_argument("--rebalance-offset", type=int, default=0, help="Optional rebalance phase offset in trading days for N-day rebalance schedules.")
    parser.add_argument("--rebalance-anchor-date", default="", help="Optional trading-date anchor for rebalance phase alignment, e.g. 2025-01-02.")
    parser.add_argument(
        "--rebalance-offset-mode",
        choices=["single", "all"],
        default="single",
        help="single uses one rebalance phase; all averages every offset sleeve into a phase-robust ensemble.",
    )
    parser.add_argument("--max-weight", type=float, default=0.25)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--min-adv20", type=float, default=50_000.0)
    parser.add_argument("--min-price", type=float, default=2.0)
    parser.add_argument("--max-price", type=float, default=300.0)
    parser.add_argument("--transaction-cost-bps", type=float, default=0.0, help="Per-turnover transaction fee/commission drag in basis points.")
    parser.add_argument("--slippage-bps", type=float, default=0.0, help="Per-turnover slippage drag in basis points.")
    parser.add_argument("--sell-tax-bps", type=float, default=0.0, help="Sell-side tax drag in basis points, applied to sell turnover only.")
    parser.add_argument(
        "--soft-state-profile",
        choices=["off", "quadrant_guard_v1", "trend_guard_v1", "market_state_guard_v1"],
        default="off",
        help="Optional soft state-conditioned gross exposure overlay applied after bridge construction.",
    )
    parser.add_argument(
        "--soft-state-selector",
        choices=["quadrant", "market_state", "trend_bucket", "vol_bucket"],
        default="",
        help="Optional selector override for soft-state sizing. Defaults to the profile's native selector.",
    )
    parser.add_argument(
        "--soft-state-gross-map",
        default="",
        help="Optional label:gross map override, e.g. trend_up_low_vol:1.0,trend_down_high_vol:0.55",
    )
    parser.add_argument("--no-market-regime-filter", action="store_true")
    parser.add_argument("--regime-ma-window", type=int, default=50)
    parser.add_argument("--regime-vol-window", type=int, default=20)
    parser.add_argument("--regime-max-annual-vol", type=float, default=0.32)
    parser.add_argument("--regime-trend-flat-band", type=float, default=0.01)
    parser.add_argument("--regime-vol-transition-band", type=float, default=0.10)
    parser.add_argument(
        "--regime-quadrants",
        default="trend_up_low_vol,trend_up_high_vol",
        help="Allowed regime labels for the market regime filter.",
    )
    parser.add_argument("--no-style-cap", action="store_true")
    parser.add_argument("--max-style-weight", type=float, default=0.50)
    parser.add_argument("--industry-cap", action="store_true")
    parser.add_argument("--max-industry-weight", type=float, default=0.40)
    parser.add_argument("--output-dir", default="daily_research/output")
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument("--candidate-label", default="")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def _panel_to_long(frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date", "stock", value_name])
    return (
        frame.copy()
        .rename_axis(index="date", columns="stock")
        .stack()
        .rename(value_name)
        .reset_index()
    )


def _build_latest_scores(score: pd.DataFrame, target_weights: pd.DataFrame, filter_mask: pd.DataFrame) -> pd.DataFrame:
    latest_dt = score.dropna(how="all").index.max()
    if pd.isna(latest_dt):
        return pd.DataFrame(columns=["date", "stock", "score", "target_weight", "filter_pass"])
    latest = pd.DataFrame(
        {
            "stock": score.columns,
            "score": score.loc[latest_dt].reindex(score.columns).values,
            "target_weight": target_weights.loc[latest_dt].reindex(score.columns).values,
            "filter_pass": filter_mask.loc[latest_dt].reindex(score.columns).fillna(False).astype(bool).values,
        }
    )
    latest.insert(0, "date", latest_dt)
    latest = latest.sort_values(["score", "target_weight"], ascending=[False, False], na_position="last")
    return latest.reset_index(drop=True)


def _load_score_panel(path: Path, *, panel_format: str, date_column: str, stock_column: str, score_column: str) -> pd.DataFrame:
    return load_value_panel(
        path,
        panel_format=panel_format,
        date_column=date_column,
        stock_column=stock_column,
        value_column=score_column,
        panel_label="score panel",
        value_name="score",
    )


def _load_target_weight_panel(
    path: Path,
    *,
    panel_format: str,
    date_column: str,
    stock_column: str,
    target_weight_column: str,
) -> pd.DataFrame:
    return load_value_panel(
        path,
        panel_format=panel_format,
        date_column=date_column,
        stock_column=stock_column,
        value_column=target_weight_column,
        panel_label="target-weight panel",
        value_name="target_weight",
    )

def _periodized_return_from_equity(equity: pd.Series, trading_days_per_period: float) -> float:
    daily_ret = equity.pct_change().dropna()
    if daily_ret.empty:
        return 0.0
    return float(equity.iloc[-1] ** (float(trading_days_per_period) / len(daily_ret)) - 1.0)


def _add_weeklyized_metrics(metrics: dict[str, float], equity_df: pd.DataFrame) -> None:
    portfolio_equity = equity_df["portfolio_equity"].astype(float)
    benchmark_equity = equity_df["benchmark_equity"].astype(float)
    excess_equity = equity_df["excess_equity"].replace([float("inf"), float("-inf")], pd.NA).ffill().dropna().astype(float)

    metrics["weeklyized_return"] = _periodized_return_from_equity(portfolio_equity, 5.0)
    metrics["benchmark_weeklyized_return"] = _periodized_return_from_equity(benchmark_equity, 5.0)
    metrics["excess_weeklyized_return"] = _periodized_return_from_equity(excess_equity, 5.0) if not excess_equity.empty else 0.0


def main():
    args = parse_args()
    score_panel_path = Path(args.score_panel_csv).expanduser() if args.score_panel_csv else None
    target_weight_panel_path = Path(args.target_weight_panel_csv).expanduser() if args.target_weight_panel_csv else None

    if score_panel_path is None and target_weight_panel_path is None:
        raise ValueError("Provide --score-panel-csv, --target-weight-panel-csv, or both.")
    if score_panel_path is not None and not score_panel_path.exists():
        raise FileNotFoundError(f"Score panel CSV not found: {score_panel_path}")
    if target_weight_panel_path is not None and not target_weight_panel_path.exists():
        raise FileNotFoundError(f"Target-weight panel CSV not found: {target_weight_panel_path}")

    score_panel = None
    if score_panel_path is not None:
        score_panel = _load_score_panel(
            score_panel_path,
            panel_format=args.score_panel_format,
            date_column=args.date_column,
            stock_column=args.stock_column,
            score_column=args.score_column,
        )
    target_weight_panel = None
    if target_weight_panel_path is not None:
        target_weight_panel = _load_target_weight_panel(
            target_weight_panel_path,
            panel_format=args.target_weight_panel_format,
            date_column=args.date_column,
            stock_column=args.stock_column,
            target_weight_column=args.target_weight_column,
        )

    input_panels = [panel for panel in (score_panel, target_weight_panel) if panel is not None]
    if not input_panels:
        raise RuntimeError("No usable input panel was loaded.")
    input_start = min(pd.Timestamp(panel.index.min()) for panel in input_panels)
    input_end = max(pd.Timestamp(panel.index.max()) for panel in input_panels)

    stocks = parse_stock_list(args.stocks)
    file_stocks = load_stock_list_from_file(args.stocks_file)
    if stocks or file_stocks:
        stocks = list(dict.fromkeys(stocks + file_stocks))
    elif args.data_source == "tq" and str(args.universe_scope or "").strip().lower() == "all_a":
        panel_stocks: list[str] = []
        for panel in input_panels:
            panel_stocks.extend(panel.columns)
        stocks = list(dict.fromkeys(panel_stocks))
    else:
        raise ValueError("Provide --stocks/--stocks-file for non-TQ mode or non-all_a universe.")

    cfg = ResearchConfig(
        start_date=args.start_date or input_start.strftime("%Y%m%d"),
        end_date=args.end_date or input_end.strftime("%Y%m%d"),
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
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        sell_tax_bps=args.sell_tax_bps,
        enable_market_regime_filter=not args.no_market_regime_filter,
        regime_ma_window=args.regime_ma_window,
        regime_vol_window=args.regime_vol_window,
        regime_max_annual_vol=args.regime_max_annual_vol,
        regime_trend_flat_band=args.regime_trend_flat_band,
        regime_vol_transition_band=args.regime_vol_transition_band,
        regime_allowed_quadrants=parse_csv_list(args.regime_quadrants),
        enable_style_cap=not args.no_style_cap,
        max_style_weight=args.max_style_weight,
        enable_industry_cap=args.industry_cap,
        max_industry_weight=args.max_industry_weight,
    )
    cfg.universe = stocks

    requested_start = args.start_date or input_start.strftime("%Y%m%d")
    requested_end = args.end_date or input_end.strftime("%Y%m%d")
    history_window = HistoryWindow(
        mode="external_score_panel",
        requested_start_date=requested_start,
        effective_start_date=requested_start,
        end_date=requested_end,
        required_trading_days=0,
    )

    raw_df_dict, raw_cache_meta = load_raw_data_with_cache(
        data_source=args.data_source,
        csv_folder=args.csv_folder,
        universe=cfg.universe,
        benchmark=cfg.benchmark,
        history_window=history_window,
        use_cache=not args.no_cache,
        refresh_cache=args.refresh_cache,
    )
    benchmark_symbol = str(args.benchmark).upper()
    benchmark_open = raw_df_dict["Open"][benchmark_symbol].copy()
    df_dict, benchmark_close = split_benchmark_from_universe(raw_df_dict, cfg.benchmark)

    eligible_stocks = [stock for stock in cfg.universe if stock in df_dict["Close"].columns]
    if not eligible_stocks:
        raise RuntimeError("No overlap between requested universe and loaded market data.")
    df_dict = {field: frame.reindex(columns=eligible_stocks) for field, frame in df_dict.items()}

    filter_mask = build_filter_mask({"raw_inputs": df_dict}, cfg)
    filter_mask = filter_mask.reindex(index=df_dict["Close"].index, columns=df_dict["Close"].columns).fillna(False)

    aligned_score = None
    if score_panel is not None:
        aligned_score = (
            score_panel.reindex(index=df_dict["Close"].index, columns=df_dict["Close"].columns)
            .sort_index()
        )
        aligned_score = aligned_score.loc[aligned_score.index.notna()]
        aligned_score = aligned_score.where(filter_mask.reindex_like(aligned_score).fillna(False))

    bridge_mode = "score_panel"
    bridge_meta: dict[str, object] = {
        "rebalance_freq": str(cfg.rebalance_freq),
        "rebalance_step": 1,
        "rebalance_offset_mode": "single",
        "rebalance_offset": 0,
        "rebalance_offsets": [0],
        "rebalance_sleeve_count": 1,
        "rebalance_anchor_date": "",
        "target_weight_top_k": int(args.target_weight_top_k),
        "target_weight_min_weight": float(args.target_weight_min_weight),
        "target_weight_power": float(args.target_weight_power),
        "target_weight_full_invest": bool(args.target_weight_full_invest),
    }
    if target_weight_panel is not None:
        raw_target_weights = (
            target_weight_panel.reindex(index=df_dict["Close"].index, columns=df_dict["Close"].columns)
            .sort_index()
        )
        raw_target_weights = raw_target_weights.loc[raw_target_weights.index.notna()]
        raw_target_weights = raw_target_weights.where(filter_mask, 0.0).fillna(0.0)
        target_weights, bridge_meta = build_target_weight_bridge(
            raw_target_weights,
            rebalance_freq=cfg.rebalance_freq,
            rebalance_offset=args.rebalance_offset,
            rebalance_offset_mode=args.rebalance_offset_mode,
            rebalance_anchor_date=args.rebalance_anchor_date,
            top_k=args.target_weight_top_k,
            min_weight=args.target_weight_min_weight,
            power=args.target_weight_power,
            full_invest=bool(args.target_weight_full_invest),
        )
        if aligned_score is not None:
            aligned_score_context, score_schedule_meta = apply_rebalance_schedule(
                aligned_score.fillna(0.0),
                rebalance_freq=cfg.rebalance_freq,
                rebalance_offset=args.rebalance_offset,
                rebalance_offset_mode=args.rebalance_offset_mode,
                rebalance_anchor_date=args.rebalance_anchor_date,
            )
            score_for_backtest = aligned_score_context
            bridge_mode = (
                "target_weight_panel_with_score_context"
                if str(score_schedule_meta["rebalance_offset_mode"]) == "single"
                else "target_weight_panel_with_score_context_ensemble"
            )
        else:
            score_for_backtest = target_weights.copy()
            bridge_mode = (
                "target_weight_panel"
                if str(bridge_meta["rebalance_offset_mode"]) == "single"
                else "target_weight_panel_ensemble"
            )
    else:
        if aligned_score is None:
            raise RuntimeError("Score panel is required when no target-weight panel is provided.")

        industry_map = None
        style_map = None
        if cfg.enable_industry_cap and args.data_source == "tq":
            industry_map = load_industry_map_from_tq(list(df_dict["Close"].columns))
        if cfg.enable_style_cap and args.data_source == "tq":
            style_map = load_style_map_from_tq(list(df_dict["Close"].columns))

        target_weights = build_target_weights(aligned_score, cfg, industry_map=industry_map, style_map=style_map)
        target_weights = apply_rebalance_frequency(
            target_weights,
            cfg.rebalance_freq,
            args.rebalance_offset,
            rebalance_anchor_date=args.rebalance_anchor_date,
        )
        score_for_backtest = apply_rebalance_frequency(
            aligned_score.fillna(0.0),
            cfg.rebalance_freq,
            args.rebalance_offset,
            rebalance_anchor_date=args.rebalance_anchor_date,
        )

    regime_state = compute_market_regime_state(benchmark_close, cfg)
    if cfg.enable_market_regime_filter:
        target_weights, score_for_backtest = apply_market_regime_filter(
            target_weights=target_weights,
            target_scores=score_for_backtest,
            regime_state=regime_state,
        )
    soft_state_profile = resolve_soft_state_profile(
        profile=args.soft_state_profile,
        selector=(args.soft_state_selector or None),
        gross_map_raw=(args.soft_state_gross_map or None),
    )
    target_weights, soft_state_scale, soft_state_meta = apply_soft_state_sizing(
        target_weights,
        regime_state,
        profile_meta=soft_state_profile,
    )

    common_dates = target_weights.index
    if len(common_dates) < 3:
        raise RuntimeError("External panel backtest has fewer than 3 valid trading dates after alignment/filtering.")

    equity_df, action_df, metrics = backtest(
        close=df_dict["Close"],
        benchmark_close=benchmark_close,
        target_weights=target_weights,
        target_scores=score_for_backtest,
        config=cfg,
        regime_on=regime_state["regime_on"] if regime_state is not None else None,
        open_df=df_dict["Open"],
        benchmark_open=benchmark_open,
    )
    _add_weeklyized_metrics(metrics, equity_df)
    monthly_backtest_summary = summarize_backtest_by_month(equity_df, action_df)
    monthly_backtest_diagnostics = summarize_monthly_diagnostics(
        monthly_backtest_summary,
        return_column="excess_return",
    )

    latest_scores = _build_latest_scores(score_for_backtest, target_weights, filter_mask.reindex_like(score_for_backtest).fillna(False))
    primary_input_path = target_weight_panel_path or score_panel_path
    metrics.update(
        {
            "framework": "external_target_weight_panel_backtest" if target_weight_panel_path is not None else "external_score_panel_backtest",
            "bridge_mode": bridge_mode,
            "candidate_label": str(args.candidate_label or primary_input_path.parent.name),
            "score_panel_csv": str(score_panel_path) if score_panel_path is not None else "",
            "score_panel_format": str(args.score_panel_format),
            "score_panel_rows": int(len(score_panel.index)) if score_panel is not None else 0,
            "score_panel_stocks": int(len(score_panel.columns)) if score_panel is not None else 0,
            "target_weight_panel_csv": str(target_weight_panel_path) if target_weight_panel_path is not None else "",
            "target_weight_panel_format": str(args.target_weight_panel_format),
            "target_weight_panel_rows": int(len(target_weight_panel.index)) if target_weight_panel is not None else 0,
            "target_weight_panel_stocks": int(len(target_weight_panel.columns)) if target_weight_panel is not None else 0,
            "target_weight_top_k": int(bridge_meta.get("target_weight_top_k", args.target_weight_top_k)),
            "target_weight_min_weight": float(bridge_meta.get("target_weight_min_weight", args.target_weight_min_weight)),
            "target_weight_power": float(bridge_meta.get("target_weight_power", args.target_weight_power)),
            "target_weight_full_invest": bool(bridge_meta.get("target_weight_full_invest", args.target_weight_full_invest)),
            "aligned_dates": int(len(score_for_backtest.index)),
            "aligned_stocks": int(len(score_for_backtest.columns)),
            "benchmark": cfg.benchmark,
            "execution_mode": cfg.execution_mode,
            "rebalance_freq": cfg.rebalance_freq,
            "rebalance_offset": bridge_meta.get("rebalance_offset"),
            "rebalance_offset_mode": str(bridge_meta.get("rebalance_offset_mode", args.rebalance_offset_mode)),
            "rebalance_offsets": bridge_meta.get("rebalance_offsets", [int(args.rebalance_offset)]),
            "rebalance_sleeve_count": int(bridge_meta.get("rebalance_sleeve_count", 1)),
            "rebalance_anchor_date": str(bridge_meta.get("rebalance_anchor_date", args.rebalance_anchor_date or "")),
             "holding_count_target": int(cfg.holding_count),
             "market_regime_filter": bool(cfg.enable_market_regime_filter),
             "transaction_cost_bps": float(cfg.transaction_cost_bps),
             "slippage_bps": float(cfg.slippage_bps),
             "sell_tax_bps": float(cfg.sell_tax_bps),
             "raw_cache": raw_cache_meta,
             "monthly_backtest_diagnostics": monthly_backtest_diagnostics,
         }
     )
    metrics.update(soft_state_meta)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = args.experiment_tag.strip() or f"external_score_backtest_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    equity_export = equity_df.reset_index().rename(columns={equity_df.index.name or "index": "date"})
    equity_export.to_csv(run_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
    action_df.to_csv(run_dir / "actions.csv", index=False, encoding="utf-8-sig")
    monthly_backtest_summary.to_csv(run_dir / "monthly_backtest_summary.csv", index=False, encoding="utf-8-sig")
    latest_scores.to_csv(run_dir / "latest_scores.csv", index=False, encoding="utf-8-sig")
    _panel_to_long(score_for_backtest, "score").to_csv(run_dir / "aligned_daily_score_panel.csv", index=False, encoding="utf-8-sig")
    _panel_to_long(target_weights, "target_weight").to_csv(run_dir / "aligned_daily_target_weight_panel.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"date": soft_state_scale.index, "soft_state_scale": soft_state_scale.values}).to_csv(
        run_dir / "soft_state_scale.csv",
        index=False,
        encoding="utf-8-sig",
    )
    regime_state.to_csv(run_dir / "regime_state.csv", encoding="utf-8-sig")
    with open(run_dir / "monthly_backtest_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(monthly_backtest_diagnostics, f, ensure_ascii=False, indent=2)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"candidate_label={metrics['candidate_label']}")
    print(f"output_dir={run_dir}")
    print(
        "core_metrics: "
        f"annual_return={metrics['annual_return']:.2%}, "
        f"weeklyized_return={metrics['weeklyized_return']:.2%}, "
        f"excess_annual_return={metrics['excess_annual_return']:.2%}, "
        f"excess_weeklyized_return={metrics['excess_weeklyized_return']:.2%}, "
        f"excess_sharpe={metrics['excess_sharpe']:.3f}, "
        f"max_drawdown={metrics['max_drawdown']:.2%}"
    )


if __name__ == "__main__":
    main()
