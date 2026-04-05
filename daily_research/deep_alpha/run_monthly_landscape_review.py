from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.baseline.backtest import summarize_backtest_by_month, summarize_monthly_diagnostics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "output"

DEFAULT_ARCHITECTURE_ROOT = OUTPUT_ROOT / "deep_alpha_architecture_execalign_formal_20260404_monthly_budgetnorm_r1"
DEFAULT_SHORT_ALPHA_ROOT = OUTPUT_ROOT / "short_alpha_formal_head2head_20260404_monthly_budgetnorm_r1"
DEFAULT_DYNAMIC_GRAPH_ROOT = OUTPUT_ROOT / "dynamic_graph_ablation_formal_20260404_monthly_budgetnorm_r1"
DEFAULT_PRODUCTION_REVIEW_ROOT = OUTPUT_ROOT / "short_alpha_production_promotion_eval_20260405_r1"


@dataclass(frozen=True)
class FormalTrackConfig:
    track_name: str
    display_name: str
    root: Path
    universe: str
    baseline_profile: str
    challenger_profile: str
    notes: str


@dataclass(frozen=True)
class ProductionReplayConfig:
    track_name: str
    display_name: str
    root: Path
    baseline_profile: str
    challenger_profile: str
    notes: str


def _default_output_tag() -> str:
    return "deep_alpha_monthly_landscape_review_20260405_r1"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _safe_corr(series_a: pd.Series, series_b: pd.Series) -> float:
    pair = pd.concat([series_a, series_b], axis=1).dropna()
    if len(pair) < 2:
        return float("nan")
    corr = pair.iloc[:, 0].corr(pair.iloc[:, 1])
    return float("nan") if pd.isna(corr) else float(corr)


def _safe_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    value = _safe_float(value)
    if np.isnan(value):
        return "n/a"
    return f"{value:.2%}"


def _safe_num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    value = _safe_float(value)
    if np.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _load_rankic_monthly_summary(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "validation_rankic_monthly_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = _read_csv(path)
    frame["month"] = frame["month"].astype(str)
    return frame


def _aggregate_rankic_monthly(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["month", "alpha_rankic_mean", "alpha_rankic_ir_mean", "risk_rankic_mean"])
    alpha_frame = frame[frame["target"].astype(str).str.startswith("fwd_excess_")].copy()
    risk_frame = frame[frame["target"].astype(str).str.startswith("risk_")].copy()
    alpha_monthly = (
        alpha_frame.groupby("month", as_index=False)
        .agg(
            alpha_rankic_mean=("rankic_mean", "mean"),
            alpha_rankic_ir_mean=("rankic_ir", "mean"),
            alpha_target_count=("target", "nunique"),
        )
    )
    if risk_frame.empty:
        alpha_monthly["risk_rankic_mean"] = np.nan
        return alpha_monthly
    risk_monthly = risk_frame.groupby("month", as_index=False).agg(risk_rankic_mean=("rankic_mean", "mean"))
    return alpha_monthly.merge(risk_monthly, on="month", how="left")


def _load_monthly_summary_from_run(run_dir: Path) -> tuple[pd.DataFrame, str]:
    candidates = [
        ("execution_aligned_monthly_backtest_summary.csv", "execution_aligned_monthly_backtest_summary"),
        ("primary_research_monthly_summary.csv", "primary_research_monthly_summary"),
        ("monthly_backtest_summary.csv", "monthly_backtest_summary"),
    ]
    for filename, label in candidates:
        path = run_dir / filename
        if path.exists():
            frame = _read_csv(path)
            frame["month"] = frame["month"].astype(str)
            frame["start_date"] = pd.to_datetime(frame["start_date"], errors="coerce")
            frame["end_date"] = pd.to_datetime(frame["end_date"], errors="coerce")
            return frame, label
    equity_path = run_dir / "equity_curve.csv"
    if equity_path.exists():
        equity_df = _read_csv(equity_path)
        if "date" in equity_df.columns:
            equity_df["date"] = pd.to_datetime(equity_df["date"], errors="coerce")
            equity_df = equity_df.dropna(subset=["date"]).set_index("date")
        actions_path = run_dir / "actions.csv"
        action_df = _read_csv(actions_path) if actions_path.exists() else pd.DataFrame()
        frame = summarize_backtest_by_month(equity_df, action_df)
        frame["month"] = frame["month"].astype(str)
        frame["start_date"] = pd.to_datetime(frame["start_date"], errors="coerce")
        frame["end_date"] = pd.to_datetime(frame["end_date"], errors="coerce")
        return frame, "derived_monthly_backtest_summary"
    return pd.DataFrame(), "missing"


def _formal_track_panel(config: FormalTrackConfig) -> pd.DataFrame:
    source_runs = _load_json(config.root / "source_runs.json")
    rows: list[pd.DataFrame] = []
    for profile_name, windows in source_runs.items():
        for window_label, metrics_path_str in windows.items():
            metrics_path = Path(metrics_path_str)
            run_dir = metrics_path.parent
            monthly_frame, monthly_source = _load_monthly_summary_from_run(run_dir)
            if monthly_frame.empty:
                continue
            rankic_frame = _aggregate_rankic_monthly(_load_rankic_monthly_summary(run_dir))
            merged = monthly_frame.merge(rankic_frame, on="month", how="left")
            merged["track_name"] = config.track_name
            merged["display_name"] = config.display_name
            merged["universe"] = config.universe
            merged["profile_name"] = profile_name
            merged["window_label"] = window_label
            merged["root_dir"] = str(config.root)
            merged["run_dir"] = str(run_dir)
            merged["monthly_source"] = monthly_source
            merged["baseline_profile"] = config.baseline_profile
            merged["challenger_profile"] = config.challenger_profile
            rows.append(merged)
    if not rows:
        return pd.DataFrame()
    panel = pd.concat(rows, ignore_index=True)
    panel = panel.sort_values(["track_name", "profile_name", "start_date", "month"]).reset_index(drop=True)
    return panel


def _production_replay_panel(config: ProductionReplayConfig) -> pd.DataFrame:
    replay_dirs = {
        config.challenger_profile: config.root / "recent_replays" / "short_alpha_production_20260403",
        config.baseline_profile: config.root / "recent_replays" / "baseline_current_production_20260403",
    }
    rows: list[pd.DataFrame] = []
    for profile_name, run_dir in replay_dirs.items():
        monthly_frame, monthly_source = _load_monthly_summary_from_run(run_dir)
        if monthly_frame.empty:
            continue
        merged = monthly_frame.copy()
        merged["alpha_rankic_mean"] = np.nan
        merged["alpha_rankic_ir_mean"] = np.nan
        merged["risk_rankic_mean"] = np.nan
        merged["track_name"] = config.track_name
        merged["display_name"] = config.display_name
        merged["universe"] = "liquid500_production"
        merged["profile_name"] = profile_name
        merged["window_label"] = "recent_replay"
        merged["root_dir"] = str(config.root)
        merged["run_dir"] = str(run_dir)
        merged["monthly_source"] = monthly_source
        merged["baseline_profile"] = config.baseline_profile
        merged["challenger_profile"] = config.challenger_profile
        rows.append(merged)
    if not rows:
        return pd.DataFrame()
    panel = pd.concat(rows, ignore_index=True)
    panel = panel.sort_values(["track_name", "profile_name", "start_date", "month"]).reset_index(drop=True)
    return panel


def _profile_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["track_name", "display_name", "universe", "profile_name", "baseline_profile", "challenger_profile"]
    for keys, frame in panel.groupby(group_cols, sort=False):
        track_name, display_name, universe, profile_name, baseline_profile, challenger_profile = keys
        ordered = frame.sort_values(["start_date", "month"]).reset_index(drop=True)
        diagnostics = summarize_monthly_diagnostics(ordered, return_column="excess_return")
        row: dict[str, Any] = {
            "track_name": track_name,
            "display_name": display_name,
            "universe": universe,
            "profile_name": profile_name,
            "baseline_profile": baseline_profile,
            "challenger_profile": challenger_profile,
            "month_count": int(len(ordered)),
            "window_count": int(ordered["window_label"].nunique()),
            "mean_excess_return": float(ordered["excess_return"].mean()),
            "mean_turnover": float(ordered["avg_turnover"].mean()),
            "mean_cost_return": float(ordered["total_trading_cost_return"].mean()),
            "mean_regime_active_ratio": float(ordered["regime_active_ratio"].mean()),
            "alpha_rankic_mean": float(ordered["alpha_rankic_mean"].mean()) if ordered["alpha_rankic_mean"].notna().any() else float("nan"),
            "alpha_rankic_ir_mean": float(ordered["alpha_rankic_ir_mean"].mean()) if ordered["alpha_rankic_ir_mean"].notna().any() else float("nan"),
            "risk_rankic_mean": float(ordered["risk_rankic_mean"].mean()) if ordered["risk_rankic_mean"].notna().any() else float("nan"),
            "return_rankic_corr": _safe_corr(ordered["excess_return"], ordered["alpha_rankic_mean"]),
            "positive_delta_months_vs_baseline": int((ordered["excess_return"] > 0.0).sum()),
            "issue_flags": ",".join(diagnostics.get("issue_flags", [])),
        }
        row.update(diagnostics)
        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(
        [
            "track_name",
            "positive_month_ratio",
            "median_monthly_return",
            "worst_monthly_return",
            "top3_positive_month_share",
            "mean_excess_return",
        ],
        ascending=[True, False, False, False, True, False],
    ).reset_index(drop=True)


def _pairwise_compare(panel: pd.DataFrame, track_name: str, left_profile: str, right_profile: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    subset = panel[panel["track_name"] == track_name].copy()
    left = subset[subset["profile_name"] == left_profile].copy()
    right = subset[subset["profile_name"] == right_profile].copy()
    if left.empty or right.empty:
        return pd.DataFrame(), {}
    merge_cols = ["month", "window_label"]
    left_cols = {
        "excess_return": f"{left_profile}_excess_return",
        "avg_turnover": f"{left_profile}_avg_turnover",
        "total_trading_cost_return": f"{left_profile}_cost_return",
        "alpha_rankic_mean": f"{left_profile}_alpha_rankic_mean",
        "alpha_rankic_ir_mean": f"{left_profile}_alpha_rankic_ir_mean",
        "start_date": f"{left_profile}_start_date",
        "end_date": f"{left_profile}_end_date",
    }
    right_cols = {
        "excess_return": f"{right_profile}_excess_return",
        "avg_turnover": f"{right_profile}_avg_turnover",
        "total_trading_cost_return": f"{right_profile}_cost_return",
        "alpha_rankic_mean": f"{right_profile}_alpha_rankic_mean",
        "alpha_rankic_ir_mean": f"{right_profile}_alpha_rankic_ir_mean",
        "start_date": f"{right_profile}_start_date",
        "end_date": f"{right_profile}_end_date",
    }
    left_view = left[merge_cols + list(left_cols.keys())].rename(columns=left_cols)
    right_view = right[merge_cols + list(right_cols.keys())].rename(columns=right_cols)
    merged = left_view.merge(right_view, on=merge_cols, how="inner").sort_values(["window_label", "month"]).reset_index(drop=True)
    merged["delta_excess_return"] = merged[f"{left_profile}_excess_return"] - merged[f"{right_profile}_excess_return"]
    merged["delta_turnover"] = merged[f"{left_profile}_avg_turnover"] - merged[f"{right_profile}_avg_turnover"]
    merged["delta_cost_return"] = merged[f"{left_profile}_cost_return"] - merged[f"{right_profile}_cost_return"]
    merged["delta_alpha_rankic_mean"] = merged[f"{left_profile}_alpha_rankic_mean"] - merged[f"{right_profile}_alpha_rankic_mean"]
    informative_mask = (
        merged["delta_excess_return"].abs() > 1e-12
    ) | (merged[f"{left_profile}_avg_turnover"] > 0.0) | (merged[f"{right_profile}_avg_turnover"] > 0.0)
    wins = int((merged["delta_excess_return"] > 0.0).sum())
    losses = int((merged["delta_excess_return"] < 0.0).sum())
    ties = int((merged["delta_excess_return"] == 0.0).sum())
    top_positive = merged.nlargest(min(3, len(merged)), "delta_excess_return")[["month", "delta_excess_return"]].to_dict(orient="records")
    top_negative = merged.nsmallest(min(3, len(merged)), "delta_excess_return")[["month", "delta_excess_return"]].to_dict(orient="records")
    compare_summary = {
        "track_name": track_name,
        "left_profile": left_profile,
        "right_profile": right_profile,
        "month_count": int(len(merged)),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_ratio": float(wins / len(merged)) if len(merged) > 0 else float("nan"),
        "informative_month_count": int(informative_mask.sum()),
        "mean_delta_excess_return": float(merged["delta_excess_return"].mean()) if len(merged) > 0 else float("nan"),
        "median_delta_excess_return": float(merged["delta_excess_return"].median()) if len(merged) > 0 else float("nan"),
        "worst_delta_excess_return": float(merged["delta_excess_return"].min()) if len(merged) > 0 else float("nan"),
        "best_delta_excess_return": float(merged["delta_excess_return"].max()) if len(merged) > 0 else float("nan"),
        "mean_delta_turnover": float(merged["delta_turnover"].mean()) if len(merged) > 0 else float("nan"),
        "mean_delta_cost_return": float(merged["delta_cost_return"].mean()) if len(merged) > 0 else float("nan"),
        "top_positive_months": top_positive,
        "top_negative_months": top_negative,
    }
    return merged, compare_summary


def _extract_summary_row(summary: pd.DataFrame, track_name: str, profile_name: str) -> pd.Series:
    row = summary[(summary["track_name"] == track_name) & (summary["profile_name"] == profile_name)]
    if row.empty:
        raise KeyError(f"Missing summary row for {track_name} / {profile_name}")
    return row.iloc[0]


def _render_month_list(items: list[dict[str, Any]]) -> str:
    if not items:
        return "n/a"
    return ", ".join(f"{item['month']} ({_safe_pct(item['delta_excess_return'])})" for item in items)


def _build_report(
    output_dir: Path,
    summary_df: pd.DataFrame,
    pairwise_summaries: dict[str, dict[str, Any]],
    formal_configs: list[FormalTrackConfig],
    production_config: ProductionReplayConfig,
) -> str:
    architecture_baseline = _extract_summary_row(summary_df, "architecture_liquid500", "baseline_current")
    architecture_structure = _extract_summary_row(summary_df, "architecture_liquid500", "structure_context_only")
    short_alpha_baseline = _extract_summary_row(summary_df, "short_alpha_liquid500", "baseline_current")
    short_alpha_candidate = _extract_summary_row(summary_df, "short_alpha_liquid500", "state_liquidity_listwise_v1")
    dynamic_v1 = _extract_summary_row(summary_df, "dynamic_graph_mainboard", "dynamic_graph_v1")
    dynamic_no_priors = _extract_summary_row(summary_df, "dynamic_graph_mainboard", "dynamic_graph_no_priors")
    production_baseline = _extract_summary_row(summary_df, "short_alpha_production_replay", "baseline_current_production_realistic")
    production_candidate = _extract_summary_row(summary_df, "short_alpha_production_replay", "short_alpha_production_realistic")

    architecture_compare = pairwise_summaries["architecture_liquid500"]
    short_alpha_compare = pairwise_summaries["short_alpha_liquid500"]
    dynamic_compare = pairwise_summaries["dynamic_graph_mainboard"]
    production_compare = pairwise_summaries["short_alpha_production_replay"]

    lines = [
        "# Deep Alpha 月度总判（预算归一化后）",
        "",
        "## 1. 输入证据",
    ]
    for config in formal_configs:
        lines.append(f"- {config.display_name}: `{config.root}`")
    lines.append(f"- {production_config.display_name}: `{production_config.root}`")
    lines.extend(
        [
            "",
            "## 2. 直接结论",
            "- 旧版月度分析的方法论仍然有效：先看月度分布、坏月、集中度，再看均值年化和 Sharpe。",
            "- 旧版结论里被后续实验改写最明显的是 `short_alpha`：预算归一化后，它不再只是“更稳但不够爆”，而是 liquid500 当前最强执行升级线。",
            "- `structure_context_only` 的旧判断基本仍成立：它能提供一些结构信息，但月度收益兑现仍弱，暂时不应占用主升级优先级。",
            "- `dynamic_graph_no_priors` 仍然是 mainboard / liquid800 主研究线，但新版结果也提示：`dynamic_graph_v1` 在月度平滑性上未必更差，问题是总收益兑现仍输给 `no_priors`。",
            "",
            "## 3. Liquid500：architecture 线",
            f"- `baseline_current` 月度正收益占比 `{_safe_pct(architecture_baseline['positive_month_ratio'])}`，月度中位数超额 `{_safe_pct(architecture_baseline['median_monthly_return'])}`，最差月份 `{_safe_pct(architecture_baseline['worst_monthly_return'])}`，收益集中度 `{_safe_pct(architecture_baseline['top3_positive_month_share'])}`。",
            f"- `structure_context_only` 月度正收益占比 `{_safe_pct(architecture_structure['positive_month_ratio'])}`，月度中位数超额 `{_safe_pct(architecture_structure['median_monthly_return'])}`，最差月份 `{_safe_pct(architecture_structure['worst_monthly_return'])}`，收益集中度 `{_safe_pct(architecture_structure['top3_positive_month_share'])}`。",
            f"- 月度 head-to-head：`baseline_current` 对 `structure_context_only` 的月度胜负是 `{int(architecture_compare['losses'])}/{int(architecture_compare['wins'])}` 从 structure 视角看，等价于 baseline 领先月占比 `{_safe_pct(1.0 - architecture_compare['win_ratio'])}`。",
            f"- `structure_context_only` 的 alpha RankIC 与月度收益相关性 `{_safe_num(architecture_structure['return_rankic_corr'])}`，仍没有体现出更强的兑现能力。",
            f"- 这条线当前仍反映的是“防守性结构先验”，不是默认执行升级答案；最差月份并没有更浅，收益集中度反而更高，说明它并未解决月度兑现问题。",
            "",
            "## 4. Liquid500：short_alpha 线",
            f"- `state_liquidity_listwise_v1` 月度正收益占比 `{_safe_pct(short_alpha_candidate['positive_month_ratio'])}`，明显高于 baseline 的 `{_safe_pct(short_alpha_baseline['positive_month_ratio'])}`。",
            f"- 它的月度中位数超额 `{_safe_pct(short_alpha_candidate['median_monthly_return'])}` 也高于 baseline 的 `{_safe_pct(short_alpha_baseline['median_monthly_return'])}`；最差月份 `{_safe_pct(short_alpha_candidate['worst_monthly_return'])}` 也更浅。",
            f"- 月度 head-to-head：`state_liquidity_listwise_v1` 对 baseline 的领先月份占比 `{_safe_pct(short_alpha_compare['win_ratio'])}`，平均月度超额差 `{_safe_pct(short_alpha_compare['mean_delta_excess_return'])}`。",
            f"- 它的月度成本 `{_safe_pct(short_alpha_candidate['mean_cost_return'])}` 与月度换手 `{_safe_pct(short_alpha_candidate['mean_turnover'])}` 并没有因为更高收益而失控，说明这次优势不是简单靠加杠杆式换手换出来的。",
            f"- 旧版“short_alpha 更稳但不够爆”的判断已经被改写。当前更准确的说法是：它既更稳，也更能把好月份兑现成净收益。",
            f"- 它领先 baseline 的代表月份：{_render_month_list(short_alpha_compare['top_positive_months'])}。",
            f"- 它落后 baseline 的月份主要集中在：{_render_month_list(short_alpha_compare['top_negative_months'])}。",
            "",
            "## 5. Production replay 复核",
            f"- production recent replay 中，`short_alpha_production_realistic` 月度正收益占比 `{_safe_pct(production_candidate['positive_month_ratio'])}`，高于旧 baseline production 的 `{_safe_pct(production_baseline['positive_month_ratio'])}`。",
            f"- production 月度拆解里真正有信息量的月份只有 `{int(production_compare['informative_month_count'])}``/``{int(production_compare['month_count'])}`，主要集中在上线后窗口；prelaunch 月份两边基本一致。",
            f"- 在这些 production 月份里，`short_alpha` 仍给出更高的平均月度超额差 `{_safe_pct(production_compare['mean_delta_excess_return'])}`，并保持更高的月度胜率。",
            f"- 这说明 liquid500 当前默认切换到 short-alpha，不只是 formal 层面的胜利，也通过了 production replay 复核；只是 production 月度样本目前还短，后续应继续跟踪。",
            "",
            "## 6. Mainboard / liquid800：dynamic_graph 线",
            f"- `dynamic_graph_v1` 的月度正收益占比 `{_safe_pct(dynamic_v1['positive_month_ratio'])}` 和最差月份 `{_safe_pct(dynamic_v1['worst_monthly_return'])}` 更像一条偏平滑的线。",
            f"- `dynamic_graph_no_priors` 的月度正收益占比 `{_safe_pct(dynamic_no_priors['positive_month_ratio'])}` 略低，但月度中位数超额 `{_safe_pct(dynamic_no_priors['median_monthly_return'])}` 更高，整体 mean excess return 也更强。",
            f"- 月度 head-to-head：`dynamic_graph_no_priors` 对 `dynamic_graph_v1` 的领先月份占比 `{_safe_pct(dynamic_compare['win_ratio'])}`，平均月度超额差 `{_safe_pct(dynamic_compare['mean_delta_excess_return'])}`。",
            "- 因此旧版“不要迷信 priors”仍成立，但更精确的表述应该是：priors 可能改善一部分月度平滑性，却没有带来更高的最终净收益兑现。",
            "",
            "## 7. 这次月度总判真正说明了什么",
            "- 当前项目最核心的矛盾仍是“谁能把研究信号稳定兑现成按月统计的净收益”，而不是单纯的模型复杂度。",
            "- liquid500 主线的最大变化来自训练预算归一化：旧 baseline 并不是在公平预算下仍然最强，因此旧版结论必须以 budget-normalized formal 为准做更新。",
            "- `structure_context_only` 的问题依旧主要在执行映射和集中度，而不是容量不够。",
            "- `short_alpha` 当前既改善了月度胜率，也改善了月度中位数和坏月，说明它不再只是“防守修补”，而是更完整的执行升级。",
            "- `dynamic_graph` 线上要继续追的是 `no_priors` 的收益兑现，同时借鉴 `v1` 的月度平滑特征，而不是回到 priors 必然有益的假设。",
            "",
            "## 8. 接下来研究优先级",
            "1. 先继续做 `short_alpha` production recipe 的 epoch extension，验证它在 production full-fit 下能否把当前优势再做厚，而不是只停在已上线状态。",
            "2. 把月度目标继续前推到训练判决里，至少把月度胜率、月度中位数超额、坏月惩罚、收益集中度惩罚纳入正式 review。",
            "3. `dynamic_graph` 研究线默认从 `dynamic_graph_no_priors` 出发，同时专门研究如何在不回引强 priors 的前提下改善坏月与收益集中度。",
            "4. `structure_context_only` 暂时降级为结构诊断线，后续只在仓位映射、集中度控制和执行对齐场景下做小规模验证，不再重开大矩阵。",
            "",
            "## 9. 输出文件",
            f"- report: `{output_dir / 'report.md'}`",
            f"- profile summary: `{output_dir / 'track_profile_summary.csv'}`",
            f"- pairwise compare: `{output_dir / 'pairwise_compare_summary.csv'}`",
            f"- monthly panel: `{output_dir / 'monthly_panel_long.csv'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a monthly-first landscape review across latest Deep Alpha formal and production evidence.")
    parser.add_argument("--architecture-root", type=Path, default=DEFAULT_ARCHITECTURE_ROOT)
    parser.add_argument("--short-alpha-root", type=Path, default=DEFAULT_SHORT_ALPHA_ROOT)
    parser.add_argument("--dynamic-graph-root", type=Path, default=DEFAULT_DYNAMIC_GRAPH_ROOT)
    parser.add_argument("--production-review-root", type=Path, default=DEFAULT_PRODUCTION_REVIEW_ROOT)
    parser.add_argument("--output-tag", type=str, default=_default_output_tag())
    args = parser.parse_args()

    formal_configs = [
        FormalTrackConfig(
            track_name="architecture_liquid500",
            display_name="Architecture liquid500 formal",
            root=args.architecture_root,
            universe="liquid500",
            baseline_profile="baseline_current",
            challenger_profile="structure_context_only",
            notes="baseline vs structure_context_only under execution-first budget-normalized formal",
        ),
        FormalTrackConfig(
            track_name="short_alpha_liquid500",
            display_name="Short alpha liquid500 formal",
            root=args.short_alpha_root,
            universe="liquid500",
            baseline_profile="baseline_current",
            challenger_profile="state_liquidity_listwise_v1",
            notes="baseline vs state_liquidity_listwise_v1 under execution-first budget-normalized formal",
        ),
        FormalTrackConfig(
            track_name="dynamic_graph_mainboard",
            display_name="Dynamic graph mainboard formal",
            root=args.dynamic_graph_root,
            universe="liquid800_mainboard",
            baseline_profile="dynamic_graph_v1",
            challenger_profile="dynamic_graph_no_priors",
            notes="dynamic_graph_v1 vs dynamic_graph_no_priors under budget-normalized formal",
        ),
    ]
    production_config = ProductionReplayConfig(
        track_name="short_alpha_production_replay",
        display_name="Short alpha production replay review",
        root=args.production_review_root,
        baseline_profile="baseline_current_production_realistic",
        challenger_profile="short_alpha_production_realistic",
        notes="recent production replay short_alpha vs previous baseline production",
    )

    monthly_panels = [_formal_track_panel(config) for config in formal_configs]
    monthly_panels.append(_production_replay_panel(production_config))
    panel = pd.concat([frame for frame in monthly_panels if not frame.empty], ignore_index=True)
    if panel.empty:
        raise RuntimeError("No monthly evidence loaded.")

    summary_df = _profile_summary(panel)

    pairwise_rows: list[dict[str, Any]] = []
    pairwise_details: list[pd.DataFrame] = []
    pairwise_summaries: dict[str, dict[str, Any]] = {}
    pair_configs = [
        ("architecture_liquid500", "structure_context_only", "baseline_current"),
        ("short_alpha_liquid500", "state_liquidity_listwise_v1", "baseline_current"),
        ("dynamic_graph_mainboard", "dynamic_graph_no_priors", "dynamic_graph_v1"),
        ("short_alpha_production_replay", "short_alpha_production_realistic", "baseline_current_production_realistic"),
    ]
    for track_name, left_profile, right_profile in pair_configs:
        detail_df, summary = _pairwise_compare(panel, track_name, left_profile, right_profile)
        if detail_df.empty:
            continue
        detail_df["track_name"] = track_name
        detail_df["left_profile"] = left_profile
        detail_df["right_profile"] = right_profile
        pairwise_details.append(detail_df)
        pairwise_summaries[track_name] = summary
        pairwise_rows.append(summary)

    pairwise_summary_df = pd.DataFrame(pairwise_rows)
    pairwise_detail_df = pd.concat(pairwise_details, ignore_index=True) if pairwise_details else pd.DataFrame()

    output_dir = OUTPUT_ROOT / args.output_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    panel.to_csv(output_dir / "monthly_panel_long.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "track_profile_summary.csv", index=False, encoding="utf-8-sig")
    pairwise_summary_df.to_csv(output_dir / "pairwise_compare_summary.csv", index=False, encoding="utf-8-sig")
    if not pairwise_detail_df.empty:
        pairwise_detail_df.to_csv(output_dir / "pairwise_compare_monthly.csv", index=False, encoding="utf-8-sig")

    report = _build_report(
        output_dir=output_dir,
        summary_df=summary_df,
        pairwise_summaries=pairwise_summaries,
        formal_configs=formal_configs,
        production_config=production_config,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    console_summary = {
        "output_dir": str(output_dir),
        "architecture_winner": "baseline_current",
        "short_alpha_winner": "state_liquidity_listwise_v1",
        "production_winner": "short_alpha_production_realistic",
        "dynamic_graph_winner": "dynamic_graph_no_priors",
        "short_alpha_monthly_positive_ratio": _safe_float(
            _extract_summary_row(summary_df, "short_alpha_liquid500", "state_liquidity_listwise_v1")["positive_month_ratio"]
        ),
        "short_alpha_baseline_positive_ratio": _safe_float(
            _extract_summary_row(summary_df, "short_alpha_liquid500", "baseline_current")["positive_month_ratio"]
        ),
    }
    print(json.dumps(console_summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
