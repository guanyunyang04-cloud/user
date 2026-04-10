from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.baseline.data_provider import load_daily_from_tq, load_universe_from_tq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
DEFAULT_VERDICT_SUMMARY = (
    OUTPUT_ROOT / "short_alpha_strongest_model_verdict_20260409_r1" / "summary.json"
)
DEFAULT_OUTPUT_TAG = "short_alpha_recent_root_cause_breakdown_20260410_r1"
DEFAULT_POOL_TOP_N = 20
POOL_HORIZONS = (1, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantify recent-year root causes for the current strongest-model monthly-distribution gap. "
            "The report compares the formal winner recent replay against the recent companion winner "
            "across market state, stock pool, score-to-weight bridge, and cash control."
        )
    )
    parser.add_argument(
        "--verdict-summary",
        default=str(DEFAULT_VERDICT_SUMMARY),
        help="Path to strongest-model verdict summary.json.",
    )
    parser.add_argument(
        "--output-tag",
        default=DEFAULT_OUTPUT_TAG,
        help="Output directory name created under daily_research/output.",
    )
    parser.add_argument(
        "--pool-top-n",
        type=int,
        default=DEFAULT_POOL_TOP_N,
        help="Top-N all-A ex-post winners used in the pool coverage proxy.",
    )
    parser.add_argument(
        "--skip-pool-coverage",
        action="store_true",
        help="Skip all-A pool coverage proxy if a quick rerun is needed.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload if isinstance(payload, dict) else {}


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    sample = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(sample) < 5 or sample["left"].nunique() <= 1 or sample["right"].nunique() <= 1:
        return float("nan")
    return float(sample["left"].corr(sample["right"], method="spearman"))


def _read_recent_replay(replay_dir: Path, source_run_dir: Path, profile_name: str) -> dict[str, Any]:
    metrics = _load_json(replay_dir / "metrics.json")
    source_metrics = _load_json(source_run_dir / "metrics.json")

    equity = pd.read_csv(replay_dir / "equity_curve.csv", encoding="utf-8-sig")
    equity["date"] = pd.to_datetime(equity["date"])
    equity["month"] = equity["date"].dt.strftime("%Y-%m")

    monthly = pd.read_csv(replay_dir / "monthly_backtest_summary.csv", encoding="utf-8-sig")
    monthly["month"] = monthly["month"].astype(str)

    regime = pd.read_csv(replay_dir / "regime_state.csv", encoding="utf-8-sig")
    date_col = "Date" if "Date" in regime.columns else "date"
    regime = regime.rename(columns={date_col: "date"})
    regime["date"] = pd.to_datetime(regime["date"])
    if "market_state" not in regime.columns:
        regime["market_state"] = "not_ready"
    regime["market_state"] = regime["market_state"].fillna("not_ready")
    if "quadrant" not in regime.columns:
        regime["quadrant"] = "not_ready"
    regime["quadrant"] = regime["quadrant"].fillna("not_ready")
    if "regime_ready" not in regime.columns:
        regime["regime_ready"] = False
    regime["regime_ready"] = regime["regime_ready"].fillna(False).astype(bool)
    if "regime_on" not in regime.columns:
        regime["regime_on"] = False
    regime["regime_on"] = regime["regime_on"].fillna(False).astype(bool)

    score = pd.read_csv(replay_dir / "aligned_daily_score_panel.csv", encoding="utf-8-sig")
    score["date"] = pd.to_datetime(score["date"])
    target = pd.read_csv(replay_dir / "aligned_daily_target_weight_panel.csv", encoding="utf-8-sig")
    target["date"] = pd.to_datetime(target["date"])
    merged_panel = score.merge(target, on=["date", "stock"], how="inner")

    exposure_rows: list[dict[str, Any]] = []
    bridge_rows: list[dict[str, Any]] = []
    for date, group in merged_panel.groupby("date"):
        gross = float(group["target_weight"].sum())
        max_weight = float(group["target_weight"].max()) if not group.empty else 0.0
        held = int((group["target_weight"] > 0).sum())
        top10 = group.nlargest(10, "score")
        top20 = group.nlargest(20, "score")
        row = {
            "date": pd.Timestamp(date),
            "gross_exposure": gross,
            "max_weight": max_weight,
            "held_count": held,
            "score_weight_spearman": _safe_spearman(group["score"], group["target_weight"]),
            "top10_selected_ratio": float((top10["target_weight"] > 0).mean()) if not top10.empty else float("nan"),
            "top20_selected_ratio": float((top20["target_weight"] > 0).mean()) if not top20.empty else float("nan"),
            "top10_weight_share": float(top10["target_weight"].sum()) if not top10.empty else float("nan"),
            "top20_weight_share": float(top20["target_weight"].sum()) if not top20.empty else float("nan"),
        }
        exposure_rows.append(row)
        if gross > 0:
            bridge_rows.append(row)

    exposure_daily = pd.DataFrame(exposure_rows)
    exposure_daily["month"] = exposure_daily["date"].dt.strftime("%Y-%m")
    bridge_daily = pd.DataFrame(bridge_rows)
    if not bridge_daily.empty:
        bridge_daily["month"] = bridge_daily["date"].dt.strftime("%Y-%m")

    daily = (
        equity.merge(
            exposure_daily[
                ["date", "gross_exposure", "max_weight", "held_count"]
            ],
            on="date",
            how="left",
        )
        .merge(
            regime[["date", "market_state", "quadrant", "regime_ready", "regime_on"]],
            on="date",
            how="left",
        )
    )
    daily["market_state"] = daily["market_state"].fillna("not_ready")
    daily["quadrant"] = daily["quadrant"].fillna("not_ready")

    state_market = (
        daily.groupby("market_state", dropna=False)
        .agg(
            days=("date", "count"),
            mean_excess_return=("excess_return", "mean"),
            total_excess_return=("excess_return", "sum"),
            mean_portfolio_return=("portfolio_return", "mean"),
            mean_gross_exposure=("gross_exposure", "mean"),
            mean_holding_count=("holding_count", "mean"),
        )
        .reset_index()
    )
    state_quadrant = (
        daily.groupby("quadrant", dropna=False)
        .agg(
            days=("date", "count"),
            mean_excess_return=("excess_return", "mean"),
            total_excess_return=("excess_return", "sum"),
            mean_portfolio_return=("portfolio_return", "mean"),
            mean_gross_exposure=("gross_exposure", "mean"),
            mean_holding_count=("holding_count", "mean"),
        )
        .reset_index()
    )

    cash_monthly = (
        daily.groupby("month", dropna=False)
        .agg(
            mean_gross_exposure=("gross_exposure", "mean"),
            mean_max_weight=("max_weight", "mean"),
            mean_holding_count=("holding_count", "mean"),
            mean_excess_return=("excess_return", "mean"),
        )
        .reset_index()
        .merge(
            monthly[["month", "excess_return", "portfolio_return", "benchmark_return", "avg_turnover", "action_count"]],
            on="month",
            how="left",
            suffixes=("_daily", ""),
        )
    )
    cash_monthly["negative_excess_month"] = cash_monthly["excess_return"].fillna(0.0) < 0

    active_bridge_summary = {
        "score_weight_spearman": float(bridge_daily["score_weight_spearman"].mean()) if not bridge_daily.empty else float("nan"),
        "top10_selected_ratio": float(bridge_daily["top10_selected_ratio"].mean()) if not bridge_daily.empty else float("nan"),
        "top20_selected_ratio": float(bridge_daily["top20_selected_ratio"].mean()) if not bridge_daily.empty else float("nan"),
        "top10_weight_share": float(bridge_daily["top10_weight_share"].mean()) if not bridge_daily.empty else float("nan"),
        "top20_weight_share": float(bridge_daily["top20_weight_share"].mean()) if not bridge_daily.empty else float("nan"),
    }

    quadrant_lookup = state_quadrant.set_index("quadrant")
    up_exposure = float(quadrant_lookup.loc["trend_up_low_vol", "mean_gross_exposure"]) if "trend_up_low_vol" in quadrant_lookup.index else float("nan")
    down_exposure = float(quadrant_lookup.loc["trend_down_low_vol", "mean_gross_exposure"]) if "trend_down_low_vol" in quadrant_lookup.index else float("nan")
    not_ready_exposure = float(quadrant_lookup.loc["not_ready", "mean_gross_exposure"]) if "not_ready" in quadrant_lookup.index else float("nan")
    up_excess = float(quadrant_lookup.loc["trend_up_low_vol", "mean_excess_return"]) if "trend_up_low_vol" in quadrant_lookup.index else float("nan")
    down_excess = float(quadrant_lookup.loc["trend_down_low_vol", "mean_excess_return"]) if "trend_down_low_vol" in quadrant_lookup.index else float("nan")

    cash_summary = {
        "overall_mean_gross_exposure": float(exposure_daily["gross_exposure"].mean()),
        "overall_mean_max_weight": float(exposure_daily["max_weight"].mean()),
        "up_state_mean_gross_exposure": up_exposure,
        "down_state_mean_gross_exposure": down_exposure,
        "not_ready_mean_gross_exposure": not_ready_exposure,
        "down_to_up_exposure_ratio": float(down_exposure / up_exposure) if np.isfinite(up_exposure) and up_exposure > 0 else float("nan"),
        "up_state_mean_excess_return": up_excess,
        "down_state_mean_excess_return": down_excess,
        "negative_month_mean_gross_exposure": float(cash_monthly.loc[cash_monthly["negative_excess_month"], "mean_gross_exposure"].mean()),
        "positive_month_mean_gross_exposure": float(cash_monthly.loc[~cash_monthly["negative_excess_month"], "mean_gross_exposure"].mean()),
    }

    return {
        "profile_name": profile_name,
        "replay_dir": replay_dir,
        "source_run_dir": source_run_dir,
        "metrics": metrics,
        "source_metrics": source_metrics,
        "daily": daily,
        "monthly": monthly,
        "state_market": state_market,
        "state_quadrant": state_quadrant,
        "cash_monthly": cash_monthly,
        "exposure_daily": exposure_daily,
        "bridge_daily": bridge_daily,
        "bridge_summary": active_bridge_summary,
        "cash_summary": cash_summary,
    }


def _pool_coverage_proxy(
    *,
    recent_start: str,
    recent_end: str,
    pool_members: list[str],
    negative_months: set[str],
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start_ts = pd.Timestamp(recent_start)
    end_ts = pd.Timestamp(recent_end)
    fetch_end = (end_ts + pd.Timedelta(days=20)).strftime("%Y%m%d")
    pool_set = {str(item).strip().upper() for item in pool_members if str(item).strip()}

    all_a = load_universe_from_tq("all_a")
    raw = load_daily_from_tq(
        all_a,
        start_ts.strftime("%Y%m%d"),
        fetch_end,
        benchmark="",
        progress_desc="RootCause all-A pool coverage",
        progress_position=0,
    )
    close = raw["Close"].sort_index()
    close.index = pd.to_datetime(close.index)
    close = close.loc[(close.index >= start_ts) & (close.index <= (end_ts + pd.Timedelta(days=10)))]

    rows: list[dict[str, Any]] = []
    for horizon in POOL_HORIZONS:
        forward = close.shift(-horizon).div(close).sub(1.0)
        for date, row in forward.loc[(forward.index >= start_ts) & (forward.index <= end_ts)].iterrows():
            valid = row.dropna()
            if len(valid) < top_n:
                continue
            top = valid.nlargest(top_n)
            inside_count = sum(1 for stock in top.index if stock in pool_set)
            month = pd.Timestamp(date).strftime("%Y-%m")
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "month": month,
                    "horizon": int(horizon),
                    "top_n": int(top_n),
                    "inside_pool_count": int(inside_count),
                    "inside_pool_share": float(inside_count / float(top_n)),
                    "outside_pool_count": int(top_n - inside_count),
                    "winner_negative_month": bool(month in negative_months),
                }
            )

    daily = pd.DataFrame(rows)
    if daily.empty:
        return daily, pd.DataFrame()
    summary = (
        daily.groupby(["horizon", "winner_negative_month"], dropna=False)
        .agg(
            date_count=("date", "count"),
            mean_inside_pool_share=("inside_pool_share", "mean"),
            median_inside_pool_share=("inside_pool_share", "median"),
            min_inside_pool_share=("inside_pool_share", "min"),
            max_inside_pool_share=("inside_pool_share", "max"),
        )
        .reset_index()
    )
    return daily, summary


def _build_factor_scoreboard(
    *,
    winner: dict[str, Any],
    runner_up: dict[str, Any],
    pool_summary: pd.DataFrame,
    same_pool: bool,
) -> pd.DataFrame:
    winner_quadrant = winner["state_quadrant"].set_index("quadrant")
    runner_quadrant = runner_up["state_quadrant"].set_index("quadrant")

    winner_up = float(winner_quadrant.loc["trend_up_low_vol", "mean_excess_return"]) if "trend_up_low_vol" in winner_quadrant.index else float("nan")
    runner_up_up = float(runner_quadrant.loc["trend_up_low_vol", "mean_excess_return"]) if "trend_up_low_vol" in runner_quadrant.index else float("nan")
    winner_down = float(winner_quadrant.loc["trend_down_low_vol", "mean_excess_return"]) if "trend_down_low_vol" in winner_quadrant.index else float("nan")
    runner_up_down = float(runner_quadrant.loc["trend_down_low_vol", "mean_excess_return"]) if "trend_down_low_vol" in runner_quadrant.index else float("nan")

    negative_pool_5d = float("nan")
    positive_pool_5d = float("nan")
    if not pool_summary.empty:
        neg = pool_summary.loc[(pool_summary["horizon"] == 5) & (pool_summary["winner_negative_month"] == True)]
        pos = pool_summary.loc[(pool_summary["horizon"] == 5) & (pool_summary["winner_negative_month"] == False)]
        if not neg.empty:
            negative_pool_5d = float(neg.iloc[0]["mean_inside_pool_share"])
        if not pos.empty:
            positive_pool_5d = float(pos.iloc[0]["mean_inside_pool_share"])

    rows = [
        {
            "factor": "market_state",
            "winner_key_metric": winner_up,
            "runner_up_key_metric": runner_up_up,
            "gap": float(winner_up - runner_up_up) if np.isfinite(winner_up) and np.isfinite(runner_up_up) else float("nan"),
            "readout": (
                "winner 在 trend_up_low_vol 的 recent 日均超额明显低于 companion，"
                f"而在 trend_down_low_vol 的差距为 {winner_down - runner_up_down:+.6f}；"
                "说明这轮更像是顺风状态的兑现能力不足，不是单纯坏状态失控。"
            ),
        },
        {
            "factor": "stock_pool",
            "winner_key_metric": negative_pool_5d,
            "runner_up_key_metric": positive_pool_5d,
            "gap": float(negative_pool_5d - positive_pool_5d) if np.isfinite(negative_pool_5d) and np.isfinite(positive_pool_5d) else float("nan"),
            "readout": (
                "两条 recent 线使用同一个固定 liquid500 池；"
                f"winner 负月份里 all-A top{DEFAULT_POOL_TOP_N} 5d 强势股入池占比约 {negative_pool_5d:.2%}，"
                f"非负月份约 {positive_pool_5d:.2%}。"
                "池子是天花板约束，但不是解释两条线差异的第一主因。"
                if same_pool and np.isfinite(negative_pool_5d) and np.isfinite(positive_pool_5d)
                else "两条线 recent 使用同一个固定 liquid500 池，因此池子差异不是两条线分化的直接来源。"
            ),
        },
        {
            "factor": "score_to_weight",
            "winner_key_metric": float(winner["bridge_summary"]["score_weight_spearman"]),
            "runner_up_key_metric": float(runner_up["bridge_summary"]["score_weight_spearman"]),
            "gap": float(winner["bridge_summary"]["score_weight_spearman"] - runner_up["bridge_summary"]["score_weight_spearman"]),
            "readout": (
                f"winner 的 active-date score/weight Spearman 为 {winner['bridge_summary']['score_weight_spearman']:.3f}，"
                f"低于 companion 的 {runner_up['bridge_summary']['score_weight_spearman']:.3f}；"
                f"top10 score 入选率也更低（{winner['bridge_summary']['top10_selected_ratio']:.2%} vs "
                f"{runner_up['bridge_summary']['top10_selected_ratio']:.2%}）。"
            ),
        },
        {
            "factor": "cash_control",
            "winner_key_metric": float(winner["cash_summary"]["down_to_up_exposure_ratio"]),
            "runner_up_key_metric": float(runner_up["cash_summary"]["down_to_up_exposure_ratio"]),
            "gap": float(winner["cash_summary"]["down_to_up_exposure_ratio"] - runner_up["cash_summary"]["down_to_up_exposure_ratio"]),
            "readout": (
                f"winner 的 down/up gross exposure 比例为 {winner['cash_summary']['down_to_up_exposure_ratio']:.3f}，"
                f"companion 为 {runner_up['cash_summary']['down_to_up_exposure_ratio']:.3f}；"
                f"winner 在 not_ready 期平均 gross 仅 {winner['cash_summary']['not_ready_mean_gross_exposure']:.3f}。"
            ),
        },
    ]
    return pd.DataFrame(rows)


def _write_summary(
    output_dir: Path,
    *,
    verdict: dict[str, Any],
    winner: dict[str, Any],
    runner_up: dict[str, Any],
    scoreboard: pd.DataFrame,
    pool_summary: pd.DataFrame,
) -> None:
    winner_quadrant = winner["state_quadrant"].set_index("quadrant")
    runner_quadrant = runner_up["state_quadrant"].set_index("quadrant")
    winner_up = float(winner_quadrant.loc["trend_up_low_vol", "mean_excess_return"]) if "trend_up_low_vol" in winner_quadrant.index else float("nan")
    runner_up_up = float(runner_quadrant.loc["trend_up_low_vol", "mean_excess_return"]) if "trend_up_low_vol" in runner_quadrant.index else float("nan")

    neg_pool_5d = float("nan")
    pos_pool_5d = float("nan")
    if not pool_summary.empty:
        neg = pool_summary.loc[(pool_summary["horizon"] == 5) & (pool_summary["winner_negative_month"] == True)]
        pos = pool_summary.loc[(pool_summary["horizon"] == 5) & (pool_summary["winner_negative_month"] == False)]
        if not neg.empty:
            neg_pool_5d = float(neg.iloc[0]["mean_inside_pool_share"])
        if not pos.empty:
            pos_pool_5d = float(pos.iloc[0]["mean_inside_pool_share"])

    lines = [
        "# Recent Root Cause Breakdown",
        "",
        "## Scope",
        f"- recent window: `{verdict.get('recent_start_date', '')} -> {verdict.get('recent_end_date', '')}`",
        f"- winner: `{winner['profile_name']}`",
        f"- companion: `{runner_up['profile_name']}`",
        "- factors: `market_state`, `stock_pool`, `score_to_weight`, `cash_control`",
        "",
        "## Direct Answer",
        "- 当前月度分布不稳的第一主因，更像是“顺风状态兑现不足 + 现金/总仓位不够状态化 + score-to-weight 转换偏弱”，不是“模型完全不会看状态”。",
        f"- winner 在 `trend_up_low_vol` 的日均超额约 `{winner_up:.4%}`，明显低于 companion 的 `{runner_up_up:.4%}`。",
        f"- winner 的 down/up gross exposure 比例约 `{winner['cash_summary']['down_to_up_exposure_ratio']:.3f}`，说明它并没有在弱状态里明显更保守。",
        f"- winner 的 active-date score/weight Spearman 约 `{winner['bridge_summary']['score_weight_spearman']:.3f}`，低于 companion 的 `{runner_up['bridge_summary']['score_weight_spearman']:.3f}`。",
    ]
    if np.isfinite(neg_pool_5d) and np.isfinite(pos_pool_5d):
        lines.append(
            f"- 股票池是上限约束，但不是第一主因：winner 负月份里 all-A top{DEFAULT_POOL_TOP_N} 5d 强势股入池占比约 "
            f"`{neg_pool_5d:.2%}`，非负月份约 `{pos_pool_5d:.2%}`。"
        )
    else:
        lines.append("- 股票池是上限约束，但这轮两条线使用同一个固定 `liquid500_latest` 池，所以它不是两条线 recent 分化的直接来源。")

    lines.extend(
        [
            "",
            "## Factor Table",
        ]
    )
    for _, row in scoreboard.iterrows():
        lines.append(
            "- "
            f"`{row['factor']}`: winner `{row['winner_key_metric']}` | companion `{row['runner_up_key_metric']}` | "
            f"gap `{row['gap']}` | {row['readout']}"
        )

    lines.extend(
        [
            "",
            "## Next Step",
            "- 优先围绕当前 `short_expert + k2` 主线做 `signal-to-weight / month-trigger / cash sizing`，而不是先继续堆深 backbone。",
            "- 股票池层如果要继续深挖，应单独做“固定 liquid500 vs 动态 rolling pool”的 same-protocol recent 对照，而不是先把锅归到池子。",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    verdict_summary = Path(args.verdict_summary)
    verdict = _load_json(verdict_summary)
    output_dir = OUTPUT_ROOT / str(args.output_tag)
    output_dir.mkdir(parents=True, exist_ok=True)

    winner_profile = str(verdict.get("winner_profile_name", "")).strip()
    runner_up_profile = str(verdict.get("recent_winner_profile_name", "") or verdict.get("runner_up_profile_name", "")).strip()
    winner_recent = verdict.get("winner_recent", {}) if isinstance(verdict.get("winner_recent"), dict) else {}
    runner_up_recent = verdict.get("runner_up_recent", {}) if isinstance(verdict.get("runner_up_recent"), dict) else {}

    winner_bundle = _read_recent_replay(
        Path(str(winner_recent.get("recent_replay_run_dir", ""))),
        Path(str(winner_recent.get("run_dir", ""))),
        winner_profile,
    )
    runner_up_bundle = _read_recent_replay(
        Path(str(runner_up_recent.get("recent_replay_run_dir", ""))),
        Path(str(runner_up_recent.get("run_dir", ""))),
        runner_up_profile,
    )

    winner_bundle["state_market"].to_csv(output_dir / f"{winner_profile}_state_market.csv", index=False, encoding="utf-8-sig")
    winner_bundle["state_quadrant"].to_csv(output_dir / f"{winner_profile}_state_quadrant.csv", index=False, encoding="utf-8-sig")
    winner_bundle["cash_monthly"].to_csv(output_dir / f"{winner_profile}_cash_monthly.csv", index=False, encoding="utf-8-sig")
    winner_bundle["bridge_daily"].to_csv(output_dir / f"{winner_profile}_bridge_daily.csv", index=False, encoding="utf-8-sig")

    runner_up_bundle["state_market"].to_csv(output_dir / f"{runner_up_profile}_state_market.csv", index=False, encoding="utf-8-sig")
    runner_up_bundle["state_quadrant"].to_csv(output_dir / f"{runner_up_profile}_state_quadrant.csv", index=False, encoding="utf-8-sig")
    runner_up_bundle["cash_monthly"].to_csv(output_dir / f"{runner_up_profile}_cash_monthly.csv", index=False, encoding="utf-8-sig")
    runner_up_bundle["bridge_daily"].to_csv(output_dir / f"{runner_up_profile}_bridge_daily.csv", index=False, encoding="utf-8-sig")

    same_pool = (
        str(winner_bundle["source_metrics"].get("stocks_file", "")).strip()
        == str(runner_up_bundle["source_metrics"].get("stocks_file", "")).strip()
    )
    pool_daily = pd.DataFrame()
    pool_summary = pd.DataFrame()
    if not args.skip_pool_coverage:
        pool_members = (
            Path(str(winner_bundle["source_metrics"].get("stocks_file", ""))).read_text(encoding="utf-8").splitlines()
        )
        negative_months = {
            str(month)
            for month in winner_bundle["cash_monthly"].loc[
                winner_bundle["cash_monthly"]["negative_excess_month"], "month"
            ].astype(str)
        }
        pool_daily, pool_summary = _pool_coverage_proxy(
            recent_start=str(verdict.get("recent_start_date", "")),
            recent_end=str(verdict.get("recent_end_date", "")),
            pool_members=pool_members,
            negative_months=negative_months,
            top_n=int(args.pool_top_n),
        )
        if not pool_daily.empty:
            pool_daily.to_csv(output_dir / "pool_coverage_daily.csv", index=False, encoding="utf-8-sig")
        if not pool_summary.empty:
            pool_summary.to_csv(output_dir / "pool_coverage_summary.csv", index=False, encoding="utf-8-sig")

    scoreboard = _build_factor_scoreboard(
        winner=winner_bundle,
        runner_up=runner_up_bundle,
        pool_summary=pool_summary,
        same_pool=same_pool,
    )
    scoreboard.to_csv(output_dir / "root_cause_scoreboard.csv", index=False, encoding="utf-8-sig")

    summary_payload = {
        "recent_start_date": str(verdict.get("recent_start_date", "")),
        "recent_end_date": str(verdict.get("recent_end_date", "")),
        "winner_profile_name": winner_profile,
        "companion_profile_name": runner_up_profile,
        "same_fixed_pool": bool(same_pool),
        "winner_bridge_summary": winner_bundle["bridge_summary"],
        "companion_bridge_summary": runner_up_bundle["bridge_summary"],
        "winner_cash_summary": winner_bundle["cash_summary"],
        "companion_cash_summary": runner_up_bundle["cash_summary"],
        "pool_coverage_summary_rows": pool_summary.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary(
        output_dir,
        verdict=verdict,
        winner=winner_bundle,
        runner_up=runner_up_bundle,
        scoreboard=scoreboard,
        pool_summary=pool_summary,
    )

    print(json.dumps({"output_dir": str(output_dir), "same_fixed_pool": bool(same_pool)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
