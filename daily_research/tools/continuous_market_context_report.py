from __future__ import annotations

import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.baseline.advanced_ml_runtime import load_raw_data_with_cache, resolve_history_window
from daily_research.baseline.config import ResearchConfig
from daily_research.baseline.data_provider import load_universe_from_tq, split_benchmark_from_universe
from daily_research.baseline.market_context import compute_continuous_market_context
from daily_research.baseline.ml_alpha import MLAplhaConfig
from daily_research.baseline.regime import compute_market_regime_state
from daily_research.execution.liquidity_universe import build_rolling_liquidity_membership


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = WORKSPACE_ROOT / "daily_research" / "output"


def latest_default_run() -> Path:
    matches = list(OUTPUT_ROOT.glob("advanced_ml_model_family_compare_*_formal_ma50_execution/lgbm"))
    if not matches:
        raise FileNotFoundError("No default ma50+lgbm formal run found.")
    return max(matches, key=lambda item: item.stat().st_mtime)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose continuous market-context features on top of the existing four-quadrant gate."
    )
    parser.add_argument("--run-dir", default="", help="Formal run dir to diagnose. Defaults to latest ma50+lgbm formal run.")
    parser.add_argument("--data-source", choices=["tq"], default="tq")
    parser.add_argument("--start-date", default="20210101")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--rolling-liquidity-pool", default="liquid500")
    parser.add_argument("--pool-rebalance-days", type=int, default=21)
    parser.add_argument("--pool-adv-window", type=int, default=20)
    parser.add_argument("--regime-ma-window", type=int, default=50)
    parser.add_argument("--regime-vol-window", type=int, default=20)
    parser.add_argument("--regime-max-annual-vol", type=float, default=0.32)
    parser.add_argument("--regime-quadrants", default="trend_up_low_vol,trend_up_high_vol")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output dir. Defaults to daily_research/output/continuous_market_context_<timestamp>",
    )
    return parser.parse_args()


def _parse_csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _safe_corr(x: pd.Series, y: pd.Series, method: str = "pearson") -> float:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 3:
        return float("nan")
    if pair.iloc[:, 0].nunique() <= 1 or pair.iloc[:, 1].nunique() <= 1:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method))


def _compound_return(series: pd.Series) -> float:
    valid = series.dropna()
    if valid.empty:
        return float("nan")
    return float((1.0 + valid).prod() - 1.0)


def _build_quantiles(series: pd.Series, quantiles: int = 5) -> pd.Series:
    valid = series.dropna()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    if len(valid) < quantiles:
        return out
    ranked = valid.rank(method="first")
    out.loc[valid.index] = pd.qcut(ranked, quantiles, labels=list(range(1, quantiles + 1))).astype(float)
    return out


def _forward_open_return(open_series: pd.Series, horizon: int) -> pd.Series:
    return open_series.shift(-(horizon + 1)).div(open_series.shift(-1)).sub(1.0)


def _load_run_frames(run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    equity = pd.read_csv(run_dir / "equity_curve.csv", parse_dates=["date", "signal_date", "execution_date"])
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    latest_weak = json.loads((run_dir / "metrics_latest_weak.json").read_text(encoding="utf-8"))
    return equity, metrics, latest_weak


def _summarize_quantiles(regime_on_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for quantile, g in regime_on_df.groupby("context_quantile", sort=True):
        rows.append(
            {
                "context_quantile": int(quantile),
                "count": int(len(g)),
                "avg_context_score": float(g["context_score"].mean()),
                "avg_benchmark_fwd_5d": float(g["benchmark_fwd_5d"].mean()),
                "avg_benchmark_fwd_20d": float(g["benchmark_fwd_20d"].mean()),
                "avg_portfolio_return_1d": float(g["portfolio_return"].mean()),
                "avg_excess_return_1d": float(g["excess_return"].mean()),
                "excess_hit_rate_1d": float((g["excess_return"] > 0).mean()),
                "wrong_open_rate_5d": float((g["benchmark_fwd_5d"] <= 0).mean()),
                "wrong_open_rate_20d": float((g["benchmark_fwd_20d"] <= 0).mean()),
                "avg_turnover": float(g["turnover"].mean()),
                "avg_holding_count": float(g["holding_count"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("context_quantile").reset_index(drop=True)


def _summarize_quadrants(regime_on_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for quadrant, g in regime_on_df.groupby("quadrant", sort=True):
        rows.append(
            {
                "quadrant": str(quadrant),
                "count": int(len(g)),
                "avg_context_score": float(g["context_score"].mean()),
                "avg_benchmark_fwd_20d": float(g["benchmark_fwd_20d"].mean()),
                "avg_excess_return_1d": float(g["excess_return"].mean()),
                "wrong_open_rate_20d": float((g["benchmark_fwd_20d"] <= 0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("quadrant").reset_index(drop=True)


def _weak_window_summary(regime_on_df: pd.DataFrame, latest_weak: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    weak_start = pd.Timestamp(str(latest_weak["holdout_start"]))
    weak_end = pd.Timestamp(str(latest_weak["holdout_end"]))
    tagged = regime_on_df.copy()
    tagged["is_latest_weak"] = (tagged.index >= weak_start) & (tagged.index <= weak_end)

    rows: list[dict[str, Any]] = []
    for label, g in (("latest_weak", tagged.loc[tagged["is_latest_weak"]]), ("other_regime_on", tagged.loc[~tagged["is_latest_weak"]])):
        rows.append(
            {
                "segment": label,
                "count": int(len(g)),
                "avg_context_score": float(g["context_score"].mean()) if not g.empty else np.nan,
                "median_context_score": float(g["context_score"].median()) if not g.empty else np.nan,
                "bottom_quintile_share": float((g["context_quantile"] == 1).mean()) if not g.empty else np.nan,
                "top_quintile_share": float((g["context_quantile"] == 5).mean()) if not g.empty else np.nan,
                "avg_benchmark_fwd_20d": float(g["benchmark_fwd_20d"].mean()) if not g.empty else np.nan,
                "avg_excess_return_1d": float(g["excess_return"].mean()) if not g.empty else np.nan,
                "compound_excess_return_1d": _compound_return(g["excess_return"]) if not g.empty else np.nan,
            }
        )
    summary_df = pd.DataFrame(rows)
    latest_row = summary_df.loc[summary_df["segment"] == "latest_weak"].iloc[0].to_dict()
    other_row = summary_df.loc[summary_df["segment"] == "other_regime_on"].iloc[0].to_dict()
    summary = {
        "holdout_start": str(latest_weak["holdout_start"]),
        "holdout_end": str(latest_weak["holdout_end"]),
        "latest_weak_avg_context_score": latest_row["avg_context_score"],
        "other_regime_on_avg_context_score": other_row["avg_context_score"],
        "latest_weak_bottom_quintile_share": latest_row["bottom_quintile_share"],
        "other_regime_on_bottom_quintile_share": other_row["bottom_quintile_share"],
        "latest_weak_compound_excess_return_1d": latest_row["compound_excess_return_1d"],
        "other_regime_on_compound_excess_return_1d": other_row["compound_excess_return_1d"],
    }
    return summary_df, summary


def _correlation_summary(regime_on_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in ["benchmark_fwd_5d", "benchmark_fwd_20d", "portfolio_return", "excess_return"]:
        rows.append(
            {
                "target": target,
                "pearson_corr": _safe_corr(regime_on_df["context_score"], regime_on_df[target], method="pearson"),
                "spearman_corr": _safe_corr(regime_on_df["context_score"], regime_on_df[target], method="spearman"),
            }
        )
    return pd.DataFrame(rows)


def _top_wrong_open_days(regime_on_df: pd.DataFrame) -> pd.DataFrame:
    out = regime_on_df.loc[regime_on_df["benchmark_fwd_20d"] <= 0].copy()
    out = out.sort_values(["context_score", "benchmark_fwd_20d", "benchmark_fwd_5d"], ascending=[True, True, True])
    cols = [
        "quadrant",
        "context_score",
        "benchmark_fwd_5d",
        "benchmark_fwd_20d",
        "portfolio_return",
        "excess_return",
        "turnover",
        "holding_count",
        "breadth_ma20",
        "breadth_ma60",
        "advance_decline_balance",
        "dispersion_20d",
        "liquidity_ratio_5_20",
    ]
    return out.loc[:, cols].head(50)


def _write_markdown(path: Path, report: dict[str, Any], quantile_df: pd.DataFrame, weak_window_df: pd.DataFrame) -> None:
    q1 = quantile_df.loc[quantile_df["context_quantile"] == 1].iloc[0].to_dict()
    q5 = quantile_df.loc[quantile_df["context_quantile"] == 5].iloc[0].to_dict()
    weak = weak_window_df.loc[weak_window_df["segment"] == "latest_weak"].iloc[0].to_dict()
    other = weak_window_df.loc[weak_window_df["segment"] == "other_regime_on"].iloc[0].to_dict()

    lines = [
        "# 连续状态诊断首轮报告",
        "",
        "## 本轮目标",
        "- 保留现有四象限门控不动。",
        "- 新增一层“宽度 + 分歧 + 流动性”的连续状态诊断。",
        "- 先回答它能否解释 `regime_on` 内的弱窗口与错误开仓，而不是直接改执行逻辑。",
        "",
        "## 核心结果",
        f"- 结论：{report['verdict']}",
        f"- Q1 低分位 vs Q5 高分位，`benchmark_fwd_20d` 均值：{q1['avg_benchmark_fwd_20d']:.2%} vs {q5['avg_benchmark_fwd_20d']:.2%}",
        f"- Q1 低分位 vs Q5 高分位，`excess_return_1d` 均值：{q1['avg_excess_return_1d']:.2%} vs {q5['avg_excess_return_1d']:.2%}",
        f"- Q1 低分位 vs Q5 高分位，`wrong_open_rate_20d`：{q1['wrong_open_rate_20d']:.2%} vs {q5['wrong_open_rate_20d']:.2%}",
        f"- 最新弱窗口 `{report['latest_weak_window']['holdout_start']} -> {report['latest_weak_window']['holdout_end']}` 的平均 `context_score`：{weak['avg_context_score']:.3f}，其余 `regime_on` 为 {other['avg_context_score']:.3f}",
        "",
        "## 当前判断",
        "- 若低分位持续对应更差的 20 日市场前景、更多错误开仓与更弱的 1 日超额，这条连续状态线就值得进入第二轮“只做软调节”的正式回测。",
        "- 若低高分位差异很弱，或只在单一窗口放大，则继续保留为诊断层，不晋级执行层。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else latest_default_run()
    if not run_dir.is_absolute():
        run_dir = (WORKSPACE_ROOT / run_dir).resolve()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / f"continuous_market_context_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not output_dir.is_absolute():
        output_dir = (WORKSPACE_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    equity, metrics, latest_weak = _load_run_frames(run_dir)
    observed_end_date = pd.to_datetime(equity["date"]).max().strftime("%Y%m%d")

    cfg = ResearchConfig(
        start_date=args.start_date,
        end_date=args.end_date or observed_end_date,
        universe_scope="all_a",
        benchmark=args.benchmark,
        execution_mode="next_open",
        weighting_method="score",
        rebalance_freq="1d",
        enable_market_regime_filter=True,
        regime_ma_window=args.regime_ma_window,
        regime_vol_window=args.regime_vol_window,
        regime_max_annual_vol=args.regime_max_annual_vol,
        regime_allowed_quadrants=_parse_csv_list(args.regime_quadrants),
    )
    ml_cfg = MLAplhaConfig(execution_mode="next_open")

    print("[1/5] loading all-A universe...")
    universe = load_universe_from_tq("all_a")

    print("[2/5] loading raw data with cache...")
    history_window = resolve_history_window(cfg, ml_cfg, cfg.start_date, cfg.end_date, mode="infer")
    raw_df_dict, raw_cache_meta = load_raw_data_with_cache(
        data_source=args.data_source,
        csv_folder=None,
        universe=universe,
        benchmark=cfg.benchmark,
        history_window=history_window,
        use_cache=True,
        refresh_cache=False,
    )

    print("[3/5] rebuilding rolling liquidity pool membership...")
    df_dict, benchmark_close = split_benchmark_from_universe(raw_df_dict, cfg.benchmark)
    benchmark_open = raw_df_dict["Open"][cfg.benchmark].copy()
    rolling_pool = build_rolling_liquidity_membership(
        close_frame=df_dict["Close"],
        amount_frame=df_dict["Amount"],
        pool_name=args.rolling_liquidity_pool,
        signal_start_date=cfg.start_date,
        signal_end_date=cfg.end_date,
        rebalance_every_days=args.pool_rebalance_days,
        adv_window=args.pool_adv_window,
        min_price=cfg.min_price,
        max_price=cfg.max_price,
    )

    print("[4/5] computing continuous market-context features...")
    regime_state = compute_market_regime_state(benchmark_close, cfg)
    context_df = compute_continuous_market_context(
        df_dict,
        regime_state=reindex_like(regime_state, rolling_pool.membership_frame.index),
        membership_mask=rolling_pool.membership_frame,
    )

    decision_df = equity.dropna(subset=["signal_date"]).copy()
    decision_df["signal_date"] = pd.to_datetime(decision_df["signal_date"])
    decision_df = decision_df.set_index("signal_date")
    decision_df.index.name = "date"

    benchmark_fwd_5d = _forward_open_return(benchmark_open, 5)
    benchmark_fwd_20d = _forward_open_return(benchmark_open, 20)

    diagnostic = context_df.join(
        decision_df[
            [
                "portfolio_return",
                "benchmark_return",
                "excess_return",
                "holding_count",
                "turnover",
                "execution_date",
                "regime_on",
            ]
        ],
        how="inner",
    )
    diagnostic["quadrant"] = regime_state["quadrant"].reindex(diagnostic.index)
    diagnostic["benchmark_fwd_5d"] = benchmark_fwd_5d.reindex(diagnostic.index)
    diagnostic["benchmark_fwd_20d"] = benchmark_fwd_20d.reindex(diagnostic.index)

    regime_on_df = diagnostic.loc[diagnostic["regime_on"].fillna(False)].copy()
    regime_on_df["context_quantile"] = _build_quantiles(regime_on_df["context_score"], quantiles=5)
    regime_on_df = regime_on_df.dropna(subset=["context_quantile"]).copy()
    regime_on_df["context_quantile"] = regime_on_df["context_quantile"].astype(int)

    quantile_df = _summarize_quantiles(regime_on_df)
    quadrant_df = _summarize_quadrants(regime_on_df)
    corr_df = _correlation_summary(regime_on_df)
    weak_window_df, weak_window_summary = _weak_window_summary(regime_on_df, latest_weak)
    wrong_open_df = _top_wrong_open_days(regime_on_df)

    q1 = quantile_df.loc[quantile_df["context_quantile"] == 1].iloc[0].to_dict()
    q5 = quantile_df.loc[quantile_df["context_quantile"] == 5].iloc[0].to_dict()
    verdict = "continuous_context_has_limited_signal"
    if (
        q5["avg_benchmark_fwd_20d"] > q1["avg_benchmark_fwd_20d"]
        and q1["wrong_open_rate_20d"] > q5["wrong_open_rate_20d"]
    ):
        verdict = "continuous_context_is_promising_for_risk_gating"
        if q5["avg_excess_return_1d"] > q1["avg_excess_return_1d"]:
            verdict = "continuous_context_has_promising_explanatory_power"

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir.relative_to(WORKSPACE_ROOT)),
        "raw_cache_meta": raw_cache_meta,
        "framework": "continuous_market_context_diagnostic",
        "baseline_execution_line": "advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open",
        "latest_weak_window": weak_window_summary,
        "quantile_spread": {
            "q1_avg_benchmark_fwd_20d": q1["avg_benchmark_fwd_20d"],
            "q5_avg_benchmark_fwd_20d": q5["avg_benchmark_fwd_20d"],
            "q1_avg_excess_return_1d": q1["avg_excess_return_1d"],
            "q5_avg_excess_return_1d": q5["avg_excess_return_1d"],
            "q1_wrong_open_rate_20d": q1["wrong_open_rate_20d"],
            "q5_wrong_open_rate_20d": q5["wrong_open_rate_20d"],
        },
        "correlations": corr_df.to_dict(orient="records"),
        "verdict": verdict,
    }

    print("[5/5] writing report artifacts...")
    context_df.to_csv(output_dir / "context_features.csv", encoding="utf-8-sig")
    diagnostic.to_csv(output_dir / "decision_context_diagnostic.csv", encoding="utf-8-sig")
    regime_on_df.to_csv(output_dir / "regime_on_context_diagnostic.csv", encoding="utf-8-sig")
    quantile_df.to_csv(output_dir / "regime_on_quantile_summary.csv", index=False, encoding="utf-8-sig")
    quadrant_df.to_csv(output_dir / "regime_on_quadrant_summary.csv", index=False, encoding="utf-8-sig")
    corr_df.to_csv(output_dir / "context_correlation_summary.csv", index=False, encoding="utf-8-sig")
    weak_window_df.to_csv(output_dir / "latest_weak_window_summary.csv", index=False, encoding="utf-8-sig")
    wrong_open_df.to_csv(output_dir / "top_wrong_open_days.csv", encoding="utf-8-sig")
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(output_dir / "report.md", report, quantile_df, weak_window_df)

    print(f"output: {output_dir}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def reindex_like(frame: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    out = frame.reindex(index)
    out.index = pd.to_datetime(out.index)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
