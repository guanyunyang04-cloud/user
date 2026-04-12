from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.deep_alpha.experiment_guardrails import resolve_project_python_executable
from daily_research.baseline.advanced_ml_runtime import HistoryWindow, load_raw_data_with_cache
from daily_research.baseline.alpha import build_filter_mask
from daily_research.baseline.backtest import summarize_monthly_diagnostics
from daily_research.baseline.config import ResearchConfig
from daily_research.baseline.data_provider import split_benchmark_from_universe
from daily_research.baseline.external_target_weight_bridge import build_target_weight_bridge
from daily_research.baseline.portfolio import build_research_raw_target_weights


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
EXTERNAL_BACKTEST_SCRIPT = PROJECT_ROOT / "daily_research" / "baseline" / "backtest_external_score_panel.py"

DEFAULT_FORMAL_RUN_DIR = (
    OUTPUT_ROOT / "short_alpha_formal_head2head_20260409_recheck_r1" / "runs" / "state_liquidity_listwise_v1_20250318_20260331"
)
DEFAULT_RECENT_RUN_DIR = OUTPUT_ROOT / "deep_alpha_short_alpha_execalign_production_default"

BRIDGE_ANCHOR_DATE = "2025-01-02"
BENCHMARK = "000300.SH"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a monthly-attack signal-to-weight verdict with 30%%-month scoring, "
            "month-trigger challengers, and recent/live consistency checks."
        )
    )
    parser.add_argument("--formal-run-dir", default=str(DEFAULT_FORMAL_RUN_DIR))
    parser.add_argument("--recent-run-dir", default=str(DEFAULT_RECENT_RUN_DIR))
    parser.add_argument("--transaction-cost-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=7.0)
    parser.add_argument("--sell-tax-bps", type=float, default=10.0)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--root-tag", default="short_alpha_monthly_attack_signal_weight_verdict_20260409_r1")
    parser.add_argument("--python-executable", default=resolve_project_python_executable(sys.executable))
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_long_panel(path: Path, value_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame["date"])
    wide = (
        frame[["date", "stock", value_name]]
        .copy()
        .assign(stock=lambda x: x["stock"].astype(str).str.upper().str.strip())
        .pivot(index="date", columns="stock", values=value_name)
        .fillna(0.0)
        .sort_index()
    )
    wide = wide.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return wide


def _wide_to_long(frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date", "stock", value_name])
    long_frame = (
        frame.copy()
        .rename_axis(index="date", columns="stock")
        .stack()
        .rename(value_name)
        .reset_index()
    )
    long_frame["date"] = pd.to_datetime(long_frame["date"]).dt.strftime("%Y-%m-%d")
    long_frame = long_frame.loc[pd.to_numeric(long_frame[value_name], errors="coerce").fillna(0.0) > 0.0].copy()
    return long_frame.reset_index(drop=True)


def _build_score_weight_raw_panel(score_panel: pd.DataFrame) -> pd.DataFrame:
    if score_panel.empty:
        return pd.DataFrame()
    cfg = ResearchConfig(
        start_date=pd.Timestamp(score_panel.index.min()).strftime("%Y%m%d"),
        end_date=pd.Timestamp(score_panel.index.max()).strftime("%Y%m%d"),
        benchmark=BENCHMARK,
        execution_mode="next_open",
        holding_count=5,
        weighting_method="score",
        rebalance_freq="1d",
        score_threshold=0.0,
    )
    return build_research_raw_target_weights(score_panel, cfg).fillna(0.0)


def _build_trade_filter_mask(panel: pd.DataFrame, *, start_date: str, end_date: str) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    stocks = [str(stock).upper().strip() for stock in panel.columns if str(stock).strip()]
    cfg = ResearchConfig(
        start_date=start_date,
        end_date=end_date,
        universe_scope="all_a",
        benchmark=BENCHMARK,
        execution_mode="next_open",
        holding_count=5,
        weighting_method="score",
        rebalance_freq="1d",
        score_threshold=0.0,
    )
    cfg.universe = stocks
    history_window = HistoryWindow(
        mode="external_score_panel",
        requested_start_date=start_date,
        effective_start_date=start_date,
        end_date=end_date,
        required_trading_days=0,
    )
    raw_df_dict, _ = load_raw_data_with_cache(
        data_source="tq",
        csv_folder=None,
        universe=cfg.universe,
        benchmark=cfg.benchmark,
        history_window=history_window,
        use_cache=True,
        refresh_cache=False,
        progress_desc="读取 challenger 过滤基线",
    )
    df_dict, _ = split_benchmark_from_universe(raw_df_dict, cfg.benchmark)
    eligible_stocks = [stock for stock in cfg.universe if stock in df_dict["Close"].columns]
    if not eligible_stocks:
        return pd.DataFrame(False, index=panel.index, columns=panel.columns)
    df_dict = {field: frame.reindex(columns=eligible_stocks) for field, frame in df_dict.items()}
    filter_mask = build_filter_mask({"raw_inputs": df_dict}, cfg)
    filter_mask = filter_mask.reindex(index=df_dict["Close"].index, columns=df_dict["Close"].columns).fillna(False)
    return filter_mask.reindex(index=panel.index, columns=panel.columns).fillna(False)


def _apply_trade_filter(panel: pd.DataFrame, filter_mask: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return panel.copy()
    if filter_mask.empty:
        return panel.copy()
    aligned = panel.reindex(index=filter_mask.index, columns=filter_mask.columns).fillna(0.0)
    return aligned.where(filter_mask, 0.0).fillna(0.0)


def _build_month_features(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if panel.empty:
        return pd.DataFrame(columns=["month", "top1_mean", "top2_mean", "best_day_top1", "gross_mean", "breadth_mean"])
    for month, month_panel in panel.groupby(panel.index.to_period("M")):
        head = month_panel.iloc[: min(5, len(month_panel))].fillna(0.0)
        top1 = head.max(axis=1)
        top2 = head.apply(lambda row: row.sort_values(ascending=False).head(2).sum(), axis=1)
        gross = head.sum(axis=1)
        breadth = (head > 1e-12).sum(axis=1)
        rows.append(
            {
                "month": str(month),
                "month_start_date": str(pd.Timestamp(month_panel.index.min()).date()),
                "days_in_month": int(len(month_panel)),
                "top1_mean": float(top1.mean()),
                "top2_mean": float(top2.mean()),
                "best_day_top1": float(top1.max()),
                "gross_mean": float(gross.mean()),
                "breadth_mean": float(breadth.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def _derive_thresholds(formal_features: pd.DataFrame, k2_panel: pd.DataFrame) -> dict[str, float]:
    if formal_features.empty:
        return {
            "aggressive_top1": 0.0,
            "aggressive_top2": 0.0,
            "hard_top1": 0.0,
            "defensive_top1": 0.0,
            "defensive_top2": 0.0,
            "guard_cap": 1.0,
        }
    k2_max = k2_panel.max(axis=1) if not k2_panel.empty else pd.Series(dtype=float)
    return {
        "aggressive_top1": float(formal_features["top1_mean"].quantile(0.75)),
        "aggressive_top2": float(formal_features["top2_mean"].quantile(0.75)),
        "hard_top1": float(formal_features["top1_mean"].quantile(0.90)),
        "defensive_top1": float(formal_features["top1_mean"].quantile(0.25)),
        "defensive_top2": float(formal_features["top2_mean"].quantile(0.25)),
        "guard_cap": float(k2_max.quantile(0.90)) if not k2_max.empty else 1.0,
    }


def _apply_month_plan(features: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    if features.empty:
        return features.copy()
    work = features.copy()
    states: list[str] = []
    aggressive_modes: list[str] = []
    for row in work.to_dict("records"):
        top1_mean = float(row.get("top1_mean", 0.0) or 0.0)
        top2_mean = float(row.get("top2_mean", 0.0) or 0.0)
        if top1_mean >= float(thresholds["aggressive_top1"]) or top2_mean >= float(thresholds["aggressive_top2"]):
            states.append("aggressive")
            aggressive_modes.append("top1" if top1_mean >= float(thresholds["hard_top1"]) else "top2")
        elif top1_mean <= float(thresholds["defensive_top1"]) and top2_mean <= float(thresholds["defensive_top2"]):
            states.append("defensive")
            aggressive_modes.append("guard")
        else:
            states.append("base")
            aggressive_modes.append("k2")
    work["month_state"] = states
    work["aggressive_mode"] = aggressive_modes
    return work


def _bridge(
    panel: pd.DataFrame,
    *,
    rebalance_freq: str,
    rebalance_offset_mode: str,
    top_k: int,
    full_invest: bool = False,
) -> pd.DataFrame:
    bridged, _ = build_target_weight_bridge(
        panel,
        rebalance_freq=rebalance_freq,
        rebalance_offset_mode=rebalance_offset_mode,
        rebalance_anchor_date=BRIDGE_ANCHOR_DATE,
        top_k=top_k,
        min_weight=0.0,
        power=1.0,
        full_invest=full_invest,
    )
    return bridged.fillna(0.0)


def _sanitize_panel(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.fillna(0.0).clip(lower=0.0)
    row_sums = clean.sum(axis=1)
    overweight = row_sums > 1.0 + 1e-8
    if bool(overweight.any()):
        clean.loc[overweight] = clean.loc[overweight].div(row_sums.loc[overweight], axis=0).fillna(0.0)
    return clean


def _scale_panel(frame: pd.DataFrame, scale: float) -> pd.DataFrame:
    if abs(float(scale) - 1.0) <= 1e-12:
        return frame.copy()
    return _sanitize_panel(frame * float(scale))


def _build_dynamic_challenger_panels(
    raw_panel: pd.DataFrame,
    month_plan: pd.DataFrame,
    thresholds: dict[str, float],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    base_k2 = _bridge(raw_panel, rebalance_freq="5d", rebalance_offset_mode="all", top_k=2, full_invest=False)
    k2_full = _bridge(raw_panel, rebalance_freq="5d", rebalance_offset_mode="all", top_k=2, full_invest=True)
    k1_direct_full = _bridge(raw_panel, rebalance_freq="1d", rebalance_offset_mode="single", top_k=1, full_invest=True)
    k2_direct_full = _bridge(raw_panel, rebalance_freq="1d", rebalance_offset_mode="single", top_k=2, full_invest=True)
    k4_guard = _bridge(raw_panel, rebalance_freq="5d", rebalance_offset_mode="all", top_k=4, full_invest=False)

    plan_by_month = {
        str(row["month"]): {
            "month_state": str(row.get("month_state", "base")),
            "aggressive_mode": str(row.get("aggressive_mode", "k2")),
        }
        for row in month_plan.to_dict("records")
    }

    confidence_rows: list[pd.Series] = []
    aggressive_rows: list[pd.Series] = []
    guard_rows: list[pd.Series] = []
    guard_hit_count = 0

    for dt in base_k2.index:
        month_key = str(pd.Timestamp(dt).to_period("M"))
        plan = plan_by_month.get(month_key, {"month_state": "base", "aggressive_mode": "k2"})
        state = str(plan["month_state"])
        mode = str(plan["aggressive_mode"])

        if state == "aggressive":
            confidence_row = k2_full.loc[dt]
        elif state == "defensive":
            confidence_row = _scale_panel(base_k2.loc[[dt]], 0.85).iloc[0]
        else:
            confidence_row = base_k2.loc[dt]
        confidence_rows.append(confidence_row.rename(dt))

        if state == "aggressive" and mode == "top1":
            aggressive_row = k1_direct_full.loc[dt]
        elif state == "aggressive":
            aggressive_row = k2_direct_full.loc[dt]
        else:
            aggressive_row = base_k2.loc[dt]
        aggressive_rows.append(aggressive_row.rename(dt))

        if state == "aggressive":
            guard_row = k2_full.loc[dt]
        else:
            guard_row = base_k2.loc[dt]
        if float(guard_row.max()) > float(thresholds["guard_cap"]) or state == "defensive":
            guard_row = k4_guard.loc[dt]
            guard_hit_count += 1
        guard_rows.append(guard_row.rename(dt))

    variants = {
        "score_weight_k2_static": _sanitize_panel(base_k2),
        "score_weight_k2_confidence_scaled": _sanitize_panel(pd.DataFrame(confidence_rows)),
        "aggressive_month_top1_or_top2": _sanitize_panel(pd.DataFrame(aggressive_rows)),
        "score_weight_k2_concentration_guard": _sanitize_panel(pd.DataFrame(guard_rows)),
    }
    meta = {
        "guard_hit_count": int(guard_hit_count),
        "guard_cap": float(thresholds["guard_cap"]),
    }
    return variants, meta


def _run_external_backtest(
    *,
    python_executable: str,
    output_root: Path,
    variant_name: str,
    score_panel_csv: Path | None,
    target_weight_panel_csv: Path,
    start_date: str,
    end_date: str,
    transaction_cost_bps: float,
    slippage_bps: float,
    sell_tax_bps: float,
) -> Path:
    cmd = [
        str(python_executable),
        str(EXTERNAL_BACKTEST_SCRIPT),
        "--target-weight-panel-csv",
        str(target_weight_panel_csv),
        "--data-source",
        "tq",
        "--benchmark",
        BENCHMARK,
        "--start-date",
        str(start_date),
        "--end-date",
        str(end_date),
        "--universe-scope",
        "all_a",
        "--rebalance-freq",
        "1d",
        "--no-market-regime-filter",
        "--transaction-cost-bps",
        str(float(transaction_cost_bps)),
        "--slippage-bps",
        str(float(slippage_bps)),
        "--sell-tax-bps",
        str(float(sell_tax_bps)),
        "--output-dir",
        str(output_root),
        "--experiment-tag",
        variant_name,
        "--candidate-label",
        variant_name,
    ]
    if score_panel_csv is not None and score_panel_csv.exists():
        cmd.extend(["--score-panel-csv", str(score_panel_csv)])
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    return output_root / variant_name


def _collect_run_metrics(run_dir: Path) -> dict[str, Any]:
    metrics = _load_json(run_dir / "metrics.json")
    monthly = pd.read_csv(run_dir / "monthly_backtest_summary.csv")
    portfolio_diag = summarize_monthly_diagnostics(monthly, return_column="portfolio_return")
    excess_diag = summarize_monthly_diagnostics(monthly, return_column="excess_return")
    recent3 = monthly.tail(min(3, len(monthly))).copy()
    strong30_count = int((pd.to_numeric(monthly["portfolio_return"], errors="coerce").fillna(0.0) >= 0.30).sum())
    strong20_count = int((pd.to_numeric(monthly["portfolio_return"], errors="coerce").fillna(0.0) >= 0.20).sum())
    month_count = int(len(monthly))
    strong30_ratio = float(strong30_count / month_count) if month_count else 0.0
    strong20_ratio = float(strong20_count / month_count) if month_count else 0.0
    recent3_mean_return = (
        float(pd.to_numeric(recent3["portfolio_return"], errors="coerce").fillna(0.0).mean()) if not recent3.empty else 0.0
    )
    recent3_mean_excess = (
        float(pd.to_numeric(recent3["excess_return"], errors="coerce").fillna(0.0).mean()) if not recent3.empty else 0.0
    )
    worst_month = float(portfolio_diag.get("worst_monthly_return", 0.0) or 0.0)
    attack_score = float(
        1.60 * strong30_ratio
        + 0.80 * strong20_ratio
        + 0.45 * float(portfolio_diag.get("best_monthly_return", 0.0) or 0.0)
        + 0.35 * recent3_mean_return
        + 0.20 * float(portfolio_diag.get("mean_monthly_return", 0.0) or 0.0)
        + 0.15 * float(portfolio_diag.get("positive_month_ratio", 0.0) or 0.0)
        - 0.45 * max(-worst_month, 0.0)
        - 0.05 * max(float(metrics.get("avg_turnover", 0.0) or 0.0) - 0.35, 0.0)
    )
    return {
        "variant_name": run_dir.name,
        "run_dir": str(run_dir),
        "annual_return": float(metrics.get("annual_return", 0.0) or 0.0),
        "excess_annual_return": float(metrics.get("excess_annual_return", 0.0) or 0.0),
        "excess_sharpe": float(metrics.get("excess_sharpe", 0.0) or 0.0),
        "avg_turnover": float(metrics.get("avg_turnover", 0.0) or 0.0),
        "month_count": month_count,
        "strong_month_30_count": strong30_count,
        "strong_month_30_ratio": strong30_ratio,
        "strong_month_20_count": strong20_count,
        "strong_month_20_ratio": strong20_ratio,
        "best_monthly_return": float(portfolio_diag.get("best_monthly_return", 0.0) or 0.0),
        "worst_monthly_return": worst_month,
        "positive_month_ratio": float(portfolio_diag.get("positive_month_ratio", 0.0) or 0.0),
        "median_monthly_return": float(portfolio_diag.get("median_monthly_return", 0.0) or 0.0),
        "recent3_mean_return": recent3_mean_return,
        "recent3_mean_excess_return": recent3_mean_excess,
        "portfolio_issue_flags": "|".join(str(x) for x in portfolio_diag.get("issue_flags", []) if str(x)),
        "attack_score": attack_score,
        "excess_mean_monthly_return": float(excess_diag.get("mean_monthly_return", 0.0) or 0.0),
        "excess_median_monthly_return": float(excess_diag.get("median_monthly_return", 0.0) or 0.0),
    }


def _rank_scoreboard(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.sort_values(
        by=["attack_score", "strong_month_30_ratio", "recent3_mean_return", "annual_return"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def _pct(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except Exception:
        return "n/a"


def _num(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "n/a"


def _write_summary(
    output_dir: Path,
    *,
    formal_df: pd.DataFrame,
    recent_df: pd.DataFrame,
    thresholds: dict[str, float],
    formal_plan: pd.DataFrame,
    recent_plan: pd.DataFrame,
    dynamic_meta: dict[str, Any],
    recent_dynamic_meta: dict[str, Any],
) -> None:
    formal_winner = formal_df.iloc[0].to_dict() if not formal_df.empty else {}
    recent_winner = recent_df.iloc[0].to_dict() if not recent_df.empty else {}
    summary_payload = {
        "formal_winner": str(formal_winner.get("variant_name", "")),
        "recent_winner": str(recent_winner.get("variant_name", "")),
        "thresholds": thresholds,
        "formal_guard_hit_count": int(dynamic_meta.get("guard_hit_count", 0)),
        "recent_guard_hit_count": int(recent_dynamic_meta.get("guard_hit_count", 0)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    formal_state_counts = (
        formal_plan["month_state"].value_counts().sort_index().to_dict() if not formal_plan.empty and "month_state" in formal_plan.columns else {}
    )
    recent_state_counts = (
        recent_plan["month_state"].value_counts().sort_index().to_dict() if not recent_plan.empty and "month_state" in recent_plan.columns else {}
    )
    lines = [
        "# Monthly Attack Signal-to-Weight Verdict",
        "",
        "## Objective",
        "- 主目标：提高 `portfolio_return >= 30%` 的强月命中率。",
        "- 约束：不能用坏月明显恶化、连续弱月变长去换单次强月。",
        "- 这轮 formal 全窗使用 `daily_score_panel.csv` 做 signal-to-weight 研究；recent/live 只做当前 raw live panel 的一致性复核。",
        "",
        "## Thresholds",
        f"- aggressive_top1: `{_pct(thresholds.get('aggressive_top1', 0.0))}`",
        f"- aggressive_top2: `{_pct(thresholds.get('aggressive_top2', 0.0))}`",
        f"- hard_top1: `{_pct(thresholds.get('hard_top1', 0.0))}`",
        f"- defensive_top1: `{_pct(thresholds.get('defensive_top1', 0.0))}`",
        f"- defensive_top2: `{_pct(thresholds.get('defensive_top2', 0.0))}`",
        f"- concentration_guard_cap: `{_pct(thresholds.get('guard_cap', 0.0))}`",
        "",
        "## Formal Verdict",
        f"- winner: `{formal_winner.get('variant_name', 'n/a')}`",
        f"- winner attack_score: `{_num(formal_winner.get('attack_score'))}`",
        f"- winner strong_month_30_ratio: `{_pct(formal_winner.get('strong_month_30_ratio'))}`",
        f"- winner best_monthly_return: `{_pct(formal_winner.get('best_monthly_return'))}`",
        f"- winner worst_monthly_return: `{_pct(formal_winner.get('worst_monthly_return'))}`",
        f"- winner recent3_mean_return: `{_pct(formal_winner.get('recent3_mean_return'))}`",
        f"- formal month_state counts: `{formal_state_counts}`",
        f"- concentration_guard trigger count: `{int(dynamic_meta.get('guard_hit_count', 0))}`",
        "",
        "## Recent Gate",
        f"- winner: `{recent_winner.get('variant_name', 'n/a')}`",
        f"- winner attack_score: `{_num(recent_winner.get('attack_score'))}`",
        f"- winner strong_month_30_ratio: `{_pct(recent_winner.get('strong_month_30_ratio'))}`",
        f"- winner best_monthly_return: `{_pct(recent_winner.get('best_monthly_return'))}`",
        f"- winner worst_monthly_return: `{_pct(recent_winner.get('worst_monthly_return'))}`",
        f"- recent month_state counts: `{recent_state_counts}`",
        f"- recent concentration_guard trigger count: `{int(recent_dynamic_meta.get('guard_hit_count', 0))}`",
        "",
        "## Decision",
    ]
    if formal_winner.get("strong_month_30_count", 0) == 0:
        lines.append("- 没有任何 challenger 达到“稳定命中 30% 强月”的目标；当前答案仍是继续研究，而不是直接 live promotion。")
    if formal_winner.get("variant_name") == recent_winner.get("variant_name"):
        lines.append(f"- formal 与 recent 同时偏向 `{formal_winner.get('variant_name', 'n/a')}`，它是下一包优先深入的强月研究主线。")
    else:
        lines.append(
            f"- formal 更偏向 `{formal_winner.get('variant_name', 'n/a')}`，但 recent 更偏向 `{recent_winner.get('variant_name', 'n/a')}`；因此当前只适合保留为 research challenger，不直接替换 live `k2`。"
        )
    lines.append("- 下一步应优先继续做同一 winner 的阈值微调、月状态定义收敛和 recent/live 扩样，而不是扩大 challenger 数量。")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_root).expanduser() / str(args.root_tag)
    output_dir.mkdir(parents=True, exist_ok=True)

    formal_run_dir = Path(args.formal_run_dir).expanduser()
    recent_run_dir = Path(args.recent_run_dir).expanduser()

    formal_score_path = formal_run_dir / "daily_score_panel.csv"
    formal_current_target_path = formal_run_dir / "daily_target_weight_panel.csv"
    recent_score_path = recent_run_dir / "daily_live_score_panel.csv"
    recent_raw_target_path = recent_run_dir / "daily_live_target_weight_panel.csv"
    recent_current_target_path = recent_run_dir / "static_fallback_daily_live_target_weight_panel.csv"

    formal_score = _load_long_panel(formal_score_path, "score")
    formal_current_target = _load_long_panel(formal_current_target_path, "target_weight")
    recent_raw_target = _load_long_panel(recent_raw_target_path, "target_weight")

    if formal_score.empty:
        raise RuntimeError(f"Formal score panel is empty or missing: {formal_score_path}")
    if formal_current_target.empty:
        raise RuntimeError(f"Formal current target panel is empty or missing: {formal_current_target_path}")
    if recent_raw_target.empty:
        raise RuntimeError(f"Recent raw live target panel is empty or missing: {recent_raw_target_path}")

    formal_start = pd.Timestamp(formal_score.index.min()).strftime("%Y%m%d")
    formal_end = pd.Timestamp(formal_score.index.max()).strftime("%Y%m%d")
    recent_start = pd.Timestamp(recent_raw_target.index.min()).strftime("%Y%m%d")
    recent_end = pd.Timestamp(recent_raw_target.index.max()).strftime("%Y%m%d")

    formal_filter_mask = _build_trade_filter_mask(formal_score, start_date=formal_start, end_date=formal_end)
    recent_filter_mask = _build_trade_filter_mask(recent_raw_target, start_date=recent_start, end_date=recent_end)

    formal_current_target = _apply_trade_filter(formal_current_target, formal_filter_mask)
    recent_raw_target = _apply_trade_filter(recent_raw_target, recent_filter_mask)

    formal_raw_score_weights = _apply_trade_filter(_build_score_weight_raw_panel(formal_score), formal_filter_mask)
    formal_base_k2 = _bridge(formal_raw_score_weights, rebalance_freq="5d", rebalance_offset_mode="all", top_k=2)
    formal_features = _build_month_features(formal_raw_score_weights)
    thresholds = _derive_thresholds(formal_features, formal_base_k2)
    formal_plan = _apply_month_plan(formal_features, thresholds)
    formal_variants, formal_dynamic_meta = _build_dynamic_challenger_panels(formal_raw_score_weights, formal_plan, thresholds)

    formal_current_k1_bridge, _ = build_target_weight_bridge(
        formal_current_target,
        rebalance_freq="5d",
        rebalance_offset_mode="all",
        rebalance_anchor_date=BRIDGE_ANCHOR_DATE,
        top_k=1,
        min_weight=0.0,
        power=1.0,
        full_invest=False,
    )
    formal_current_k2_bridge, _ = build_target_weight_bridge(
        formal_current_target,
        rebalance_freq="5d",
        rebalance_offset_mode="all",
        rebalance_anchor_date=BRIDGE_ANCHOR_DATE,
        top_k=2,
        min_weight=0.0,
        power=1.0,
        full_invest=False,
    )
    formal_current_k3_bridge, _ = build_target_weight_bridge(
        formal_current_target,
        rebalance_freq="5d",
        rebalance_offset_mode="all",
        rebalance_anchor_date=BRIDGE_ANCHOR_DATE,
        top_k=3,
        min_weight=0.0,
        power=1.0,
        full_invest=False,
    )

    formal_panel_dir = output_dir / "formal_panels"
    formal_panel_dir.mkdir(parents=True, exist_ok=True)
    formal_variants = {
        "formal_current_equal_top5_direct": formal_current_target,
        "formal_current_equal_top5_k1_bridge": formal_current_k1_bridge.fillna(0.0),
        "formal_current_equal_top5_k2_bridge": formal_current_k2_bridge.fillna(0.0),
        "formal_current_equal_top5_k3_bridge": formal_current_k3_bridge.fillna(0.0),
        **formal_variants,
    }
    formal_plan.to_csv(output_dir / "formal_month_plan.csv", index=False, encoding="utf-8-sig")

    recent_features = _build_month_features(recent_raw_target)
    recent_plan = _apply_month_plan(recent_features, thresholds)
    recent_variants, recent_dynamic_meta = _build_dynamic_challenger_panels(recent_raw_target, recent_plan, thresholds)
    recent_plan.to_csv(output_dir / "recent_month_plan.csv", index=False, encoding="utf-8-sig")

    recent_panel_dir = output_dir / "recent_panels"
    recent_panel_dir.mkdir(parents=True, exist_ok=True)
    recent_variants = {
        "recent_current_live_k2_static": _load_long_panel(recent_current_target_path, "target_weight"),
        "recent_raw_k2_confidence_scaled": recent_variants["score_weight_k2_confidence_scaled"],
        "recent_aggressive_month_top1_or_top2": recent_variants["aggressive_month_top1_or_top2"],
        "recent_raw_k2_concentration_guard": recent_variants["score_weight_k2_concentration_guard"],
    }

    thresholds_path = output_dir / "month_trigger_thresholds.json"
    thresholds_path.write_text(json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8")

    formal_results: list[dict[str, Any]] = []
    for variant_name, panel in formal_variants.items():
        if panel.empty:
            continue
        target_csv = formal_current_target_path if variant_name == "formal_current_equal_top5_direct" else formal_panel_dir / f"{variant_name}.csv"
        if target_csv != formal_current_target_path:
            _wide_to_long(_sanitize_panel(panel), "target_weight").to_csv(target_csv, index=False, encoding="utf-8-sig")
        run_dir = _run_external_backtest(
            python_executable=args.python_executable,
            output_root=output_dir / "formal_runs",
            variant_name=variant_name,
            score_panel_csv=formal_score_path,
            target_weight_panel_csv=target_csv,
            start_date=formal_start,
            end_date=formal_end,
            transaction_cost_bps=args.transaction_cost_bps,
            slippage_bps=args.slippage_bps,
            sell_tax_bps=args.sell_tax_bps,
        )
        formal_results.append(_collect_run_metrics(run_dir))

    recent_results: list[dict[str, Any]] = []
    for variant_name, panel in recent_variants.items():
        if panel.empty:
            continue
        target_csv = recent_current_target_path if variant_name == "recent_current_live_k2_static" else recent_panel_dir / f"{variant_name}.csv"
        if target_csv != recent_current_target_path:
            _wide_to_long(_sanitize_panel(panel), "target_weight").to_csv(target_csv, index=False, encoding="utf-8-sig")
        run_dir = _run_external_backtest(
            python_executable=args.python_executable,
            output_root=output_dir / "recent_runs",
            variant_name=variant_name,
            score_panel_csv=recent_score_path if recent_score_path.exists() else None,
            target_weight_panel_csv=target_csv,
            start_date=recent_start,
            end_date=recent_end,
            transaction_cost_bps=args.transaction_cost_bps,
            slippage_bps=args.slippage_bps,
            sell_tax_bps=args.sell_tax_bps,
        )
        recent_results.append(_collect_run_metrics(run_dir))

    formal_df = _rank_scoreboard(pd.DataFrame(formal_results))
    recent_df = _rank_scoreboard(pd.DataFrame(recent_results))
    formal_df.to_csv(output_dir / "formal_attack_scoreboard.csv", index=False, encoding="utf-8-sig")
    recent_df.to_csv(output_dir / "recent_attack_scoreboard.csv", index=False, encoding="utf-8-sig")

    _write_summary(
        output_dir,
        formal_df=formal_df,
        recent_df=recent_df,
        thresholds=thresholds,
        formal_plan=formal_plan,
        recent_plan=recent_plan,
        dynamic_meta=formal_dynamic_meta,
        recent_dynamic_meta=recent_dynamic_meta,
    )

    print(f"output_dir={output_dir}")
    if not formal_df.empty:
        print(f"formal_winner={formal_df.iloc[0]['variant_name']}")
        print(f"formal_attack_score={formal_df.iloc[0]['attack_score']:.3f}")
    if not recent_df.empty:
        print(f"recent_winner={recent_df.iloc[0]['variant_name']}")
        print(f"recent_attack_score={recent_df.iloc[0]['attack_score']:.3f}")


if __name__ == "__main__":
    main()
