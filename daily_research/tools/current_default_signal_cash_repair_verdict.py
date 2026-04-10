from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.baseline.backtest import summarize_monthly_diagnostics
from daily_research.baseline.config import ResearchConfig
from daily_research.baseline.external_target_weight_bridge import build_target_weight_bridge
from daily_research.baseline.portfolio import build_research_raw_target_weights
from daily_research.baseline.soft_state_sizing import apply_soft_state_sizing, resolve_soft_state_profile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
EXTERNAL_BACKTEST_SCRIPT = PROJECT_ROOT / "daily_research" / "baseline" / "backtest_external_score_panel.py"
TRADE_PLAN_SCRIPT = PROJECT_ROOT / "daily_research" / "baseline" / "generate_daily_trade_plan.py"

DEFAULT_STRONGEST_SUMMARY = OUTPUT_ROOT / "short_alpha_strongest_model_verdict_20260409_r1" / "summary.json"
DEFAULT_ACTIVE_MANIFEST = OUTPUT_ROOT / "active_execution_strategy.json"
DEFAULT_ROOT_TAG = "short_alpha_current_default_signal_cash_repair_20260410_r1"

BENCHMARK = "000300.SH"
BRIDGE_ANCHOR_DATE = "2025-01-02"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair the current short_expert + k2 default chain by comparing a small set of "
            "signal-to-weight and month-trigger / cash-sizing challengers on the recent-year window."
        )
    )
    parser.add_argument("--strongest-summary", default=str(DEFAULT_STRONGEST_SUMMARY))
    parser.add_argument("--active-manifest", default=str(DEFAULT_ACTIVE_MANIFEST))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--root-tag", default=DEFAULT_ROOT_TAG)
    parser.add_argument("--transaction-cost-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=7.0)
    parser.add_argument("--sell-tax-bps", type=float, default=10.0)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--preview-cash", type=float, default=100000.0)
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
    return wide.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _load_regime_state(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame()
    if "date" in frame.columns:
        date_col = "date"
    elif "Date" in frame.columns:
        date_col = "Date"
    else:
        first_col = str(frame.columns[0])
        date_col = first_col
    frame[date_col] = pd.to_datetime(frame[date_col])
    return frame.set_index(date_col).sort_index()


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
    return long_frame.reset_index(drop=True)


def _sanitize_panel(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.fillna(0.0).clip(lower=0.0)
    row_sums = clean.sum(axis=1)
    overweight = row_sums > 1.0 + 1e-8
    if bool(overweight.any()):
        clean.loc[overweight] = clean.loc[overweight].div(row_sums.loc[overweight], axis=0).fillna(0.0)
    return clean.fillna(0.0)


def _rowwise_power(
    frame: pd.DataFrame,
    *,
    power: float,
    preserve_row_gross: bool,
    target_gross: float | None = None,
) -> pd.DataFrame:
    clean = _sanitize_panel(frame)
    rows: list[pd.Series] = []
    power = max(float(power or 1.0), 1e-8)
    for dt, row in clean.iterrows():
        weights = row.fillna(0.0).astype(float)
        positive = weights[weights > 0.0].copy()
        gross = float(weights.sum())
        if positive.empty or gross <= 0.0:
            rows.append(pd.Series(0.0, index=clean.columns, name=dt, dtype=float))
            continue
        positive = positive.pow(power)
        total = float(positive.sum())
        if total <= 0.0:
            rows.append(pd.Series(0.0, index=clean.columns, name=dt, dtype=float))
            continue
        desired_gross = float(target_gross) if target_gross is not None else (gross if preserve_row_gross else 1.0)
        desired_gross = min(max(desired_gross, 0.0), 1.0)
        positive = positive / total * desired_gross
        aligned = pd.Series(0.0, index=clean.columns, name=dt, dtype=float)
        aligned.loc[positive.index] = positive.values
        rows.append(aligned)
    return _sanitize_panel(pd.DataFrame(rows, index=clean.index, columns=clean.columns))


def _apply_soft_state(
    panel: pd.DataFrame,
    regime_state: pd.DataFrame,
    *,
    profile: str = "market_state_guard_v1",
    selector: str = "",
    gross_map_raw: str = "",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if panel.empty:
        return panel.copy(), {
            "soft_state_profile": "off",
            "soft_state_enabled": False,
            "soft_state_selector": selector or "market_state",
            "soft_state_gross_map": {},
        }
    profile_meta = resolve_soft_state_profile(
        profile=profile,
        selector=selector or None,
        gross_map_raw=gross_map_raw or None,
    )
    scaled, _scale, meta = apply_soft_state_sizing(
        panel,
        regime_state.reindex(panel.index),
        profile_meta=profile_meta,
        default_gross=1.0,
    )
    return _sanitize_panel(scaled), meta


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


def _bridge_k2_5d(raw_panel: pd.DataFrame) -> pd.DataFrame:
    bridged, _ = build_target_weight_bridge(
        raw_panel,
        rebalance_freq="5d",
        rebalance_offset_mode="all",
        rebalance_anchor_date=BRIDGE_ANCHOR_DATE,
        top_k=2,
        min_weight=0.0,
        power=1.0,
        full_invest=False,
    )
    return _sanitize_panel(bridged)


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    sample = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(sample) < 5 or sample["left"].nunique() <= 1 or sample["right"].nunique() <= 1:
        return float("nan")
    return float(sample["left"].corr(sample["right"], method="spearman"))


def _bridge_summary(score_panel: pd.DataFrame, target_panel: pd.DataFrame) -> dict[str, float]:
    aligned_score = score_panel.reindex(index=target_panel.index, columns=target_panel.columns).fillna(0.0)
    rows: list[dict[str, float]] = []
    for dt in target_panel.index:
        score_row = aligned_score.loc[dt]
        target_row = target_panel.loc[dt]
        active = target_row > 1e-12
        top10_names = list(score_row.sort_values(ascending=False).head(10).index)
        rows.append(
            {
                "score_weight_spearman": _safe_spearman(score_row, target_row),
                "top10_selected_ratio": float(active.reindex(top10_names).fillna(False).mean()) if top10_names else 0.0,
                "gross_exposure": float(target_row.sum()),
                "max_name_weight": float(target_row.max()),
            }
        )
    frame = pd.DataFrame(rows)
    return {
        "score_weight_spearman": float(frame["score_weight_spearman"].mean()) if not frame.empty else float("nan"),
        "top10_selected_ratio": float(frame["top10_selected_ratio"].mean()) if not frame.empty else float("nan"),
        "gross_exposure": float(frame["gross_exposure"].mean()) if not frame.empty else float("nan"),
        "max_name_weight": float(frame["max_name_weight"].mean()) if not frame.empty else float("nan"),
    }


def _monthly_robust_score(diag: dict[str, Any]) -> float:
    positive_ratio = float(diag.get("positive_month_ratio", 0.0) or 0.0)
    median_monthly_return = float(diag.get("median_monthly_return", 0.0) or 0.0)
    mean_monthly_return = float(diag.get("mean_monthly_return", 0.0) or 0.0)
    worst_monthly_return = float(diag.get("worst_monthly_return", 0.0) or 0.0)
    top3_positive_share = float(diag.get("top3_positive_month_share", 0.0) or 0.0)
    longest_negative_streak = int(diag.get("longest_negative_streak", 0) or 0)
    downside_penalty = max(-worst_monthly_return, 0.0)
    concentration_penalty = max(top3_positive_share - 0.60, 0.0)
    streak_penalty = max(longest_negative_streak - 2, 0)
    return float(
        mean_monthly_return
        + median_monthly_return
        + 0.05 * (positive_ratio - 0.50)
        - 0.35 * downside_penalty
        - 0.05 * concentration_penalty
        - 0.01 * float(streak_penalty)
    )


def _build_month_features(
    score_panel: pd.DataFrame,
    target_panel: pd.DataFrame,
    regime_state: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if score_panel.empty or target_panel.empty:
        return pd.DataFrame()
    market_state = regime_state.get("market_state", pd.Series(index=score_panel.index, dtype="object"))
    for month, month_score in score_panel.groupby(score_panel.index.to_period("M")):
        month_target = target_panel.loc[month_score.index]
        head_score = month_score.iloc[: min(5, len(month_score))].fillna(0.0)
        head_target = month_target.iloc[: min(5, len(month_target))].fillna(0.0)
        head_state = market_state.reindex(head_score.index)
        first_state = ""
        for item in head_state.tolist():
            if pd.notna(item) and str(item).strip():
                first_state = str(item).strip().lower()
                break
        top1_score = head_score.max(axis=1)
        top2_score = head_score.apply(lambda row: row.sort_values(ascending=False).head(2).sum(), axis=1)
        rows.append(
            {
                "month": str(month),
                "month_start_date": str(pd.Timestamp(month_score.index.min()).date()),
                "first_market_state": first_state,
                "top1_score_mean": float(top1_score.mean()),
                "top2_score_mean": float(top2_score.mean()),
                "gross_mean": float(head_target.sum(axis=1).mean()),
                "max_weight_mean": float(head_target.max(axis=1).mean()),
                "holding_count_mean": float((head_target > 1e-12).sum(axis=1).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def _derive_thresholds(features: pd.DataFrame) -> dict[str, float]:
    if features.empty:
        return {
            "aggressive_top1": 0.0,
            "aggressive_top2": 0.0,
            "defensive_top1": 0.0,
            "defensive_gross": 0.0,
        }
    return {
        "aggressive_top1": float(features["top1_score_mean"].quantile(0.75)),
        "aggressive_top2": float(features["top2_score_mean"].quantile(0.75)),
        "defensive_top1": float(features["top1_score_mean"].quantile(0.25)),
        "defensive_gross": float(features["gross_mean"].quantile(0.35)),
    }


def _apply_month_plan(features: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    if features.empty:
        return features.copy()
    work = features.copy()
    states: list[str] = []
    for row in work.to_dict("records"):
        market_state = str(row.get("first_market_state", "") or "").lower()
        up_state = market_state.startswith("trend_up")
        down_state = market_state.startswith("trend_down")
        top1_score_mean = float(row.get("top1_score_mean", 0.0) or 0.0)
        top2_score_mean = float(row.get("top2_score_mean", 0.0) or 0.0)
        gross_mean = float(row.get("gross_mean", 0.0) or 0.0)
        if up_state and (
            top1_score_mean >= float(thresholds["aggressive_top1"])
            or top2_score_mean >= float(thresholds["aggressive_top2"])
        ):
            states.append("aggressive")
        elif down_state or (
            top1_score_mean <= float(thresholds["defensive_top1"])
            and gross_mean <= float(thresholds["defensive_gross"])
        ):
            states.append("defensive")
        else:
            states.append("base")
    work["month_state"] = states
    return work


def _build_attack_defense_panel(
    base_target: pd.DataFrame,
    guarded_target: pd.DataFrame,
    month_plan: pd.DataFrame,
) -> pd.DataFrame:
    plan_by_month = {str(row["month"]): str(row.get("month_state", "base")) for row in month_plan.to_dict("records")}
    rows: list[pd.Series] = []
    for dt in base_target.index:
        month_key = str(pd.Timestamp(dt).to_period("M"))
        month_state = plan_by_month.get(month_key, "base")
        if month_state == "aggressive":
            row = _rowwise_power(base_target.loc[[dt]], power=1.25, preserve_row_gross=False, target_gross=1.0).iloc[0]
        elif month_state == "defensive":
            row = guarded_target.loc[dt]
        else:
            row = base_target.loc[dt]
        rows.append(row.rename(dt))
    return _sanitize_panel(pd.DataFrame(rows, index=base_target.index, columns=base_target.columns))


def _run_external_backtest(
    *,
    python_executable: str,
    output_root: Path,
    variant_name: str,
    score_panel_csv: Path,
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
        "--score-panel-csv",
        str(score_panel_csv),
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
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    return output_root / variant_name


def _collect_run_metrics(run_dir: Path, variant_name: str, bridge_summary: dict[str, float]) -> dict[str, Any]:
    metrics = _load_json(run_dir / "metrics.json")
    monthly = pd.read_csv(run_dir / "monthly_backtest_summary.csv")
    portfolio_diag = summarize_monthly_diagnostics(monthly, return_column="portfolio_return")
    excess_diag = summarize_monthly_diagnostics(monthly, return_column="excess_return")
    recent3 = monthly.tail(min(3, len(monthly))).copy()
    return {
        "variant_name": variant_name,
        "run_dir": str(run_dir),
        "annual_return": float(metrics.get("annual_return", 0.0) or 0.0),
        "excess_annual_return": float(metrics.get("excess_annual_return", 0.0) or 0.0),
        "excess_sharpe": float(metrics.get("excess_sharpe", 0.0) or 0.0),
        "avg_turnover": float(metrics.get("avg_turnover", 0.0) or 0.0),
        "positive_month_ratio": float(portfolio_diag.get("positive_month_ratio", 0.0) or 0.0),
        "median_monthly_return": float(portfolio_diag.get("median_monthly_return", 0.0) or 0.0),
        "mean_monthly_return": float(portfolio_diag.get("mean_monthly_return", 0.0) or 0.0),
        "worst_monthly_return": float(portfolio_diag.get("worst_monthly_return", 0.0) or 0.0),
        "top3_positive_month_share": float(portfolio_diag.get("top3_positive_month_share", 0.0) or 0.0),
        "longest_negative_streak": int(portfolio_diag.get("longest_negative_streak", 0) or 0),
        "recent3_mean_return": float(pd.to_numeric(recent3["portfolio_return"], errors="coerce").fillna(0.0).mean()) if not recent3.empty else 0.0,
        "recent3_mean_excess_return": float(pd.to_numeric(recent3["excess_return"], errors="coerce").fillna(0.0).mean()) if not recent3.empty else 0.0,
        "portfolio_issue_flags": "|".join(str(x) for x in portfolio_diag.get("issue_flags", []) if str(x)),
        "monthly_robust_score": _monthly_robust_score(portfolio_diag),
        "excess_mean_monthly_return": float(excess_diag.get("mean_monthly_return", 0.0) or 0.0),
        "excess_median_monthly_return": float(excess_diag.get("median_monthly_return", 0.0) or 0.0),
        "score_weight_spearman": float(bridge_summary.get("score_weight_spearman", float("nan"))),
        "top10_selected_ratio": float(bridge_summary.get("top10_selected_ratio", float("nan"))),
        "gross_exposure": float(bridge_summary.get("gross_exposure", float("nan"))),
        "max_name_weight": float(bridge_summary.get("max_name_weight", float("nan"))),
    }


def _rank_scoreboard(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.sort_values(
        by=["monthly_robust_score", "median_monthly_return", "excess_annual_return", "recent3_mean_return"],
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


def _variant_kind_summary(variant_name: str) -> str:
    if variant_name.startswith("companion_"):
        return "companion baseline"
    if "score_weight" in variant_name:
        return "score-derived bridge"
    if "attack_defense" in variant_name:
        return "month-trigger attack/defense"
    if "market_state_guard" in variant_name:
        return "cash-sizing guard"
    if "power125" in variant_name:
        return "signal-to-weight power repair"
    return "current target baseline"


def _write_summary(
    output_dir: Path,
    *,
    scoreboard: pd.DataFrame,
    thresholds: dict[str, float],
    month_plan: pd.DataFrame,
    repair_winner: dict[str, Any],
    live_preview_payload: dict[str, Any],
) -> None:
    winner = scoreboard.iloc[0].to_dict() if not scoreboard.empty else {}
    current_row = scoreboard.loc[scoreboard["variant_name"].eq("winner_current_target_static")].iloc[0].to_dict()
    companion_row = scoreboard.loc[scoreboard["variant_name"].eq("companion_current_target_static")].iloc[0].to_dict()
    month_state_counts = month_plan["month_state"].value_counts().sort_index().to_dict() if not month_plan.empty else {}
    summary_payload = {
        "overall_winner_variant": str(winner.get("variant_name", "")),
        "overall_winner_kind": _variant_kind_summary(str(winner.get("variant_name", ""))),
        "repair_winner_variant": str(repair_winner.get("variant_name", "")),
        "repair_winner_kind": _variant_kind_summary(str(repair_winner.get("variant_name", ""))),
        "thresholds": thresholds,
        "month_state_counts": month_state_counts,
        "live_preview": live_preview_payload,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Current Default Signal/Cash Repair Verdict",
        "",
        "## Objective",
        "- 在不推翻当前 `short_expert + k2` 默认链的前提下，先修 `signal-to-weight`，再修 `month-trigger / cash sizing`。",
        "- recent 窗口固定为最近一年 `12` 个月，winner 与 companion 都按同一协议回放。",
        "",
        "## Month Plan",
        f"- aggressive_top1 threshold: `{_num(thresholds.get('aggressive_top1'))}`",
        f"- aggressive_top2 threshold: `{_num(thresholds.get('aggressive_top2'))}`",
        f"- defensive_top1 threshold: `{_num(thresholds.get('defensive_top1'))}`",
        f"- defensive_gross threshold: `{_pct(thresholds.get('defensive_gross'))}`",
        f"- month_state counts: `{month_state_counts}`",
        "",
        "## Overall Winner",
        f"- overall winner: `{winner.get('variant_name', 'n/a')}` ({_variant_kind_summary(str(winner.get('variant_name', '')))}).",
        f"- overall winner monthly_robust_score: `{_num(winner.get('monthly_robust_score'))}`",
        f"- overall winner excess annual: `{_pct(winner.get('excess_annual_return'))}`",
        f"- overall winner excess Sharpe: `{_num(winner.get('excess_sharpe'))}`",
        "",
        "## Repair Winner",
        f"- repair winner: `{repair_winner.get('variant_name', 'n/a')}` ({_variant_kind_summary(str(repair_winner.get('variant_name', '')))}).",
        f"- repair winner monthly_robust_score: `{_num(repair_winner.get('monthly_robust_score'))}`",
        f"- repair winner excess annual: `{_pct(repair_winner.get('excess_annual_return'))}`",
        f"- repair winner excess Sharpe: `{_num(repair_winner.get('excess_sharpe'))}`",
        f"- repair winner positive month ratio: `{_pct(repair_winner.get('positive_month_ratio'))}`",
        f"- repair winner median monthly return: `{_pct(repair_winner.get('median_monthly_return'))}`",
        f"- repair winner worst month: `{_pct(repair_winner.get('worst_monthly_return'))}`",
        "",
        "## Baseline Compare",
        f"- current default monthly_robust_score: `{_num(current_row.get('monthly_robust_score'))}`",
        f"- companion monthly_robust_score: `{_num(companion_row.get('monthly_robust_score'))}`",
        f"- repair winner vs current default robust delta: `{_num(float(repair_winner.get('monthly_robust_score', 0.0) or 0.0) - float(current_row.get('monthly_robust_score', 0.0) or 0.0))}`",
        f"- repair winner vs companion robust delta: `{_num(float(repair_winner.get('monthly_robust_score', 0.0) or 0.0) - float(companion_row.get('monthly_robust_score', 0.0) or 0.0))}`",
        f"- repair winner score/weight Spearman: `{_num(repair_winner.get('score_weight_spearman'))}`",
        f"- current default score/weight Spearman: `{_num(current_row.get('score_weight_spearman'))}`",
        "",
        "## Decision",
    ]
    repair_winner_name = str(repair_winner.get("variant_name", ""))
    current_name = "winner_current_target_static"
    if repair_winner_name == current_name:
        lines.append("- 这轮 small challenger 没有推翻 current default；下一步应继续小范围细调，不直接切默认。")
    else:
        lines.append(f"- 这轮 current-default repair winner 已经从 current default 移到 `{repair_winner_name}`，说明第一包修补方向成立。")
        if float(repair_winner.get("monthly_robust_score", 0.0) or 0.0) > float(companion_row.get("monthly_robust_score", 0.0) or 0.0):
            lines.append("- 它在 recent 一年里也超过了 companion baseline，具备直接讨论默认替换的资格。")
        else:
            lines.append("- 它虽然修复了 current default，但还没有完全打赢 companion baseline，因此先保留为候选修补主线。")
    if str(winner.get("variant_name", "")).startswith("companion_"):
        lines.append("- 这说明 current default 的首要任务仍然是先缩小与 companion 的 recent 一年差距，再讨论更激进的 promotion。")
    if live_preview_payload.get("plan_path"):
        lines.append(f"- 已输出 winner 的当前 live 预览计划：`{live_preview_payload.get('plan_path')}`。")
    elif live_preview_payload.get("preview_target_weight_panel_csv"):
        lines.append(f"- 已输出 repair winner 的当前 live 预览权重面板：`{live_preview_payload.get('preview_target_weight_panel_csv')}`。")
        if live_preview_payload.get("preview_error"):
            lines.append("- 预览 trade plan 本轮未成功导出，但这不影响 recent verdict；错误信息已写入 `summary.json`。")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _generate_live_preview(
    *,
    output_dir: Path,
    variant_name: str,
    panel_builder: Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.DataFrame] | None,
    active_manifest: dict[str, Any],
    thresholds: dict[str, float],
    regime_state: pd.DataFrame,
    python_executable: str,
    preview_cash: float,
) -> dict[str, Any]:
    if panel_builder is None or not variant_name.startswith("winner_"):
        return {}
    production_root_raw = str(active_manifest.get("production_root", "")).strip()
    if not production_root_raw:
        return {}
    production_root = Path(production_root_raw)
    score_path = production_root / "execution_aligned_daily_live_score_panel.csv"
    target_path = production_root / "execution_aligned_daily_live_target_weight_panel.csv"
    if not score_path.exists() or not target_path.exists():
        return {}

    score_panel = _load_long_panel(score_path, "score")
    target_panel = _load_long_panel(target_path, "target_weight")
    if score_panel.empty or target_panel.empty:
        return {}

    aligned_regime_state = regime_state.reindex(score_panel.index)
    month_features = _build_month_features(score_panel, target_panel, aligned_regime_state)
    month_plan = _apply_month_plan(month_features, thresholds)
    panel = panel_builder(score_panel, target_panel, aligned_regime_state, month_plan)

    preview_dir = output_dir / "live_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_target_csv = preview_dir / f"{variant_name}_target_weight_panel.csv"
    _wide_to_long(panel, "target_weight").to_csv(preview_target_csv, index=False, encoding="utf-8-sig")

    trade_plan_dir = preview_dir / "trade_plan"
    trade_plan_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(python_executable),
        str(TRADE_PLAN_SCRIPT),
        "--external-score-csv",
        str(score_path),
        "--external-target-weight-csv",
        str(preview_target_csv),
        "--candidate-label",
        variant_name,
        "--cash",
        str(float(preview_cash)),
        "--output-dir",
        str(trade_plan_dir),
    ]
    preview_error = ""
    try:
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    except subprocess.CalledProcessError as exc:
        preview_error = str(exc)
    plan_path = trade_plan_dir / "latest_trade_plan.txt"
    return {
        "variant_name": variant_name,
        "preview_target_weight_panel_csv": str(preview_target_csv),
        "plan_path": str(plan_path) if plan_path.exists() else "",
        "preview_error": preview_error,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_root).expanduser() / str(args.root_tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    recent_runs_dir = output_dir / "recent_runs"
    recent_runs_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = output_dir / "variant_panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_json(Path(args.strongest_summary).expanduser())
    active_manifest = _load_json(Path(args.active_manifest).expanduser())
    winner_recent = summary.get("winner_recent", {})
    companion_recent = summary.get("runner_up_recent", {})
    winner_recent_dir = Path(str(winner_recent.get("recent_replay_run_dir", "")).strip())
    companion_recent_dir = Path(str(companion_recent.get("recent_replay_run_dir", "")).strip())
    if not winner_recent_dir.exists():
        raise FileNotFoundError(f"Winner recent replay dir not found: {winner_recent_dir}")
    if not companion_recent_dir.exists():
        raise FileNotFoundError(f"Companion recent replay dir not found: {companion_recent_dir}")

    recent_start = str(summary.get("recent_start_date", "")).strip()
    recent_end = str(summary.get("recent_end_date", "")).strip()
    if not recent_start or not recent_end:
        raise RuntimeError("Strongest-model summary is missing recent_start_date / recent_end_date.")

    winner_score_panel = _load_long_panel(winner_recent_dir / "aligned_daily_score_panel.csv", "score")
    winner_target_panel = _load_long_panel(winner_recent_dir / "aligned_daily_target_weight_panel.csv", "target_weight")
    companion_score_panel = _load_long_panel(companion_recent_dir / "aligned_daily_score_panel.csv", "score")
    companion_target_panel = _load_long_panel(companion_recent_dir / "aligned_daily_target_weight_panel.csv", "target_weight")
    winner_regime_state = _load_regime_state(winner_recent_dir / "regime_state.csv")

    if winner_score_panel.empty or winner_target_panel.empty:
        raise RuntimeError("Winner recent replay panels are empty.")
    if companion_target_panel.empty:
        raise RuntimeError("Companion recent replay target panel is empty.")

    month_features = _build_month_features(winner_score_panel, winner_target_panel, winner_regime_state)
    thresholds = _derive_thresholds(month_features)
    month_plan = _apply_month_plan(month_features, thresholds)
    month_plan.to_csv(output_dir / "month_plan.csv", index=False, encoding="utf-8-sig")
    (output_dir / "thresholds.json").write_text(json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8")

    winner_current_target_power125 = _rowwise_power(winner_target_panel, power=1.25, preserve_row_gross=True)
    winner_current_target_guard, winner_guard_meta = _apply_soft_state(
        winner_target_panel,
        winner_regime_state,
        profile="market_state_guard_v1",
    )
    winner_current_target_power125_guard, winner_power_guard_meta = _apply_soft_state(
        winner_current_target_power125,
        winner_regime_state,
        profile="market_state_guard_v1",
    )
    winner_current_target_attack_defense = _build_attack_defense_panel(
        winner_target_panel,
        winner_current_target_guard,
        month_plan,
    )
    winner_score_weight_k2 = _bridge_k2_5d(_build_score_weight_raw_panel(winner_score_panel))
    winner_score_weight_k2_guard, winner_score_guard_meta = _apply_soft_state(
        winner_score_weight_k2,
        winner_regime_state,
        profile="market_state_guard_v1",
    )

    variant_builders: dict[str, Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.DataFrame] | None] = {
        "winner_current_target_static": lambda score, target, regime, plan: _sanitize_panel(target),
        "winner_current_target_power125_cash_preserved": (
            lambda score, target, regime, plan: _rowwise_power(target, power=1.25, preserve_row_gross=True)
        ),
        "winner_current_target_market_state_guard_v1": (
            lambda score, target, regime, plan: _apply_soft_state(target, regime, profile="market_state_guard_v1")[0]
        ),
        "winner_current_target_power125_market_state_guard_v1": (
            lambda score, target, regime, plan: _apply_soft_state(
                _rowwise_power(target, power=1.25, preserve_row_gross=True),
                regime,
                profile="market_state_guard_v1",
            )[0]
        ),
        "winner_current_target_attack_defense_v1": (
            lambda score, target, regime, plan: _build_attack_defense_panel(
                target,
                _apply_soft_state(target, regime, profile="market_state_guard_v1")[0],
                plan,
            )
        ),
        "winner_score_weight_k2_static": (
            lambda score, target, regime, plan: _bridge_k2_5d(_build_score_weight_raw_panel(score))
        ),
        "winner_score_weight_k2_market_state_guard_v1": (
            lambda score, target, regime, plan: _apply_soft_state(
                _bridge_k2_5d(_build_score_weight_raw_panel(score)),
                regime,
                profile="market_state_guard_v1",
            )[0]
        ),
        "companion_current_target_static": None,
    }

    variant_panels: dict[str, pd.DataFrame] = {
        "winner_current_target_static": _sanitize_panel(winner_target_panel),
        "winner_current_target_power125_cash_preserved": winner_current_target_power125,
        "winner_current_target_market_state_guard_v1": winner_current_target_guard,
        "winner_current_target_power125_market_state_guard_v1": winner_current_target_power125_guard,
        "winner_current_target_attack_defense_v1": winner_current_target_attack_defense,
        "winner_score_weight_k2_static": winner_score_weight_k2,
        "winner_score_weight_k2_market_state_guard_v1": winner_score_weight_k2_guard,
        "companion_current_target_static": _sanitize_panel(companion_target_panel),
    }

    pd.DataFrame(
        [
            {"variant_name": "winner_current_target_market_state_guard_v1", "meta_json": json.dumps(winner_guard_meta, ensure_ascii=False)},
            {"variant_name": "winner_current_target_power125_market_state_guard_v1", "meta_json": json.dumps(winner_power_guard_meta, ensure_ascii=False)},
            {"variant_name": "winner_score_weight_k2_market_state_guard_v1", "meta_json": json.dumps(winner_score_guard_meta, ensure_ascii=False)},
        ]
    ).to_csv(output_dir / "variant_meta.csv", index=False, encoding="utf-8-sig")

    scoreboard_rows: list[dict[str, Any]] = []
    for variant_name, panel in variant_panels.items():
        panel_csv = panels_dir / f"{variant_name}.csv"
        _wide_to_long(panel, "target_weight").to_csv(panel_csv, index=False, encoding="utf-8-sig")
        score_panel_csv = winner_recent_dir / "aligned_daily_score_panel.csv"
        score_panel = winner_score_panel
        if variant_name.startswith("companion_"):
            score_panel_csv = companion_recent_dir / "aligned_daily_score_panel.csv"
            score_panel = companion_score_panel
        run_dir = _run_external_backtest(
            python_executable=str(args.python_executable),
            output_root=recent_runs_dir,
            variant_name=variant_name,
            score_panel_csv=score_panel_csv,
            target_weight_panel_csv=panel_csv,
            start_date=recent_start,
            end_date=recent_end,
            transaction_cost_bps=float(args.transaction_cost_bps),
            slippage_bps=float(args.slippage_bps),
            sell_tax_bps=float(args.sell_tax_bps),
        )
        scoreboard_rows.append(_collect_run_metrics(run_dir, variant_name, _bridge_summary(score_panel, panel)))

    scoreboard = _rank_scoreboard(pd.DataFrame(scoreboard_rows))
    scoreboard.to_csv(output_dir / "recent_scoreboard.csv", index=False, encoding="utf-8-sig")

    winner_variant_name = str(scoreboard.iloc[0]["variant_name"]) if not scoreboard.empty else ""
    repair_scoreboard = scoreboard.loc[scoreboard["variant_name"].astype(str).str.startswith("winner_")].copy()
    repair_winner = repair_scoreboard.iloc[0].to_dict() if not repair_scoreboard.empty else {}
    repair_winner_name = str(repair_winner.get("variant_name", ""))
    live_preview_payload = _generate_live_preview(
        output_dir=output_dir,
        variant_name=repair_winner_name,
        panel_builder=variant_builders.get(repair_winner_name),
        active_manifest=active_manifest,
        thresholds=thresholds,
        regime_state=winner_regime_state,
        python_executable=str(args.python_executable),
        preview_cash=float(args.preview_cash),
    )
    _write_summary(
        output_dir,
        scoreboard=scoreboard,
        thresholds=thresholds,
        month_plan=month_plan,
        repair_winner=repair_winner,
        live_preview_payload=live_preview_payload,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "overall_winner_variant": winner_variant_name,
                "repair_winner_variant": repair_winner_name,
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
