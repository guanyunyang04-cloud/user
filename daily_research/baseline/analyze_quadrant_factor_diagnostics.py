import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from daily_research.baseline.alpha import combine_scores
from daily_research.baseline.config import ResearchConfig
from daily_research.baseline.data_provider import load_daily_from_tq, load_universe_from_tq, split_benchmark_from_universe
from daily_research.baseline.evaluation import evaluate_factor_bundle
from daily_research.baseline.features import compute_factors
from daily_research.baseline.regime_analysis import classify_market_quadrants


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze factor effectiveness inside a market quadrant and year windows")
    parser.add_argument("--start-date", default="20210101")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--quadrant", default="trend_up_low_vol")
    parser.add_argument("--years", default="2024,2025,2026")
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument("--regime-ma-window", type=int, default=60)
    parser.add_argument("--regime-vol-window", type=int, default=20)
    parser.add_argument("--regime-max-annual-vol", type=float, default=0.32)
    return parser.parse_args()


def _parse_years(raw: str) -> list[int]:
    years = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        years.append(int(item))
    return years


def _subset_bundle(raw_factors: dict[str, pd.DataFrame], score: pd.DataFrame, close: pd.DataFrame, mask: pd.Series):
    mask = mask.reindex(close.index).astype("boolean").fillna(False).astype(bool)
    sub_factors = {
        name: frame.loc[mask]
        for name, frame in raw_factors.items()
    }
    sub_score = score.loc[mask]
    sub_close = close.loc[mask]
    return sub_factors, sub_score, sub_close


def main():
    args = parse_args()
    years = _parse_years(args.years)
    cfg = ResearchConfig(
        start_date=args.start_date,
        benchmark=args.benchmark,
        universe_scope="all_a",
        weighting_method="score",
        rebalance_freq="5d",
        regime_ma_window=args.regime_ma_window,
        regime_vol_window=args.regime_vol_window,
        regime_max_annual_vol=args.regime_max_annual_vol,
    )

    print("[1/5] 加载全A股票池...")
    universe = load_universe_from_tq("all_a")
    print(f"[2/5] 拉取日线数据，股票数: {len(universe)}")
    raw_df_dict = load_daily_from_tq(universe, args.start_date, benchmark=args.benchmark)
    df_dict, benchmark_close = split_benchmark_from_universe(raw_df_dict, args.benchmark)

    print("[3/5] 计算因子与综合分数...")
    factor_bundle = compute_factors(df_dict)
    score, _, _ = combine_scores(factor_bundle, cfg)

    print("[4/5] 识别市场状态象限...")
    quadrants = classify_market_quadrants(
        benchmark_close=benchmark_close,
        ma_window=args.regime_ma_window,
        vol_window=args.regime_vol_window,
        vol_threshold=args.regime_max_annual_vol,
    )

    output_rows = []
    all_ic = []
    all_quantile = []
    close = factor_bundle["raw_inputs"]["Close"]

    print("[5/5] 逐年输出因子诊断...")
    for year in years:
        mask = (quadrants.index.year == year) & (quadrants["quadrant"] == args.quadrant)
        sub_factors, sub_score, sub_close = _subset_bundle(
            raw_factors=factor_bundle["raw_factors"],
            score=score,
            close=close,
            mask=mask,
        )
        ic_df, quantile_df = evaluate_factor_bundle(
            raw_factors=sub_factors,
            score=sub_score,
            close=sub_close,
            quantiles=5,
        )
        ic_df.insert(0, "year", year)
        quantile_df.insert(0, "year", year)
        all_ic.append(ic_df)
        all_quantile.append(quantile_df)

        for horizon in ["fwd_5d", "fwd_10d", "fwd_20d"]:
            horizon_df = ic_df[ic_df["horizon"] == horizon].sort_values("rank_ic", ascending=False)
            top_row = horizon_df.iloc[0] if not horizon_df.empty else None
            score_row = horizon_df[horizon_df["factor"] == "composite_score"]
            output_rows.append(
                {
                    "year": year,
                    "quadrant": args.quadrant,
                    "sample_days": int(mask.sum()),
                    "horizon": horizon,
                    "best_factor": top_row["factor"] if top_row is not None else "",
                    "best_rank_ic": float(top_row["rank_ic"]) if top_row is not None else float("nan"),
                    "composite_rank_ic": float(score_row["rank_ic"].iloc[0]) if not score_row.empty else float("nan"),
                    "composite_ic": float(score_row["ic"].iloc[0]) if not score_row.empty else float("nan"),
                }
            )

    summary_df = pd.DataFrame(output_rows)
    ic_full = pd.concat(all_ic, ignore_index=True) if all_ic else pd.DataFrame()
    quantile_full = pd.concat(all_quantile, ignore_index=True) if all_quantile else pd.DataFrame()

    out_root = Path(__file__).resolve().parents[1] / "output"
    run_name = args.experiment_tag.strip() or datetime.now().strftime("quadrant_factor_diag_%Y%m%d_%H%M%S")
    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    quadrants.to_csv(out_dir / "quadrant_state.csv", encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "factor_diagnostic_summary.csv", index=False, encoding="utf-8-sig")
    ic_full.to_csv(out_dir / "factor_ic_by_year.csv", index=False, encoding="utf-8-sig")
    quantile_full.to_csv(out_dir / "factor_quantile_by_year.csv", index=False, encoding="utf-8-sig")

    print(f"输出目录: {out_dir}")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
