from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
DEFAULT_FORMAL_ROOT = OUTPUT_ROOT / "short_alpha_formal_head2head_20260404_monthly_budgetnorm_r1"
DEFAULT_AUDIT_ROOTS = (
    OUTPUT_ROOT / "short_alpha_formal_execution_policy_audit_20230216_20240229_20260405_r1",
    OUTPUT_ROOT / "short_alpha_formal_execution_policy_audit_20240301_20250317_20260405_r1",
    OUTPUT_ROOT / "short_alpha_formal_execution_policy_audit_20260405_r1",
)
DEFAULT_CURRENT_PROFILE = "regoff_k1_5d_ensemble_native_anchor"


@dataclass(frozen=True)
class FormalRun:
    profile_name: str
    run_dir: Path
    window_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Review weak months for the liquid500 short_alpha main line and summarize "
            "which execution policies historically helped the most in those months."
        )
    )
    parser.add_argument("--formal-root", default=str(DEFAULT_FORMAL_ROOT))
    parser.add_argument(
        "--audit-roots",
        default=",".join(str(path) for path in DEFAULT_AUDIT_ROOTS),
        help="Comma-separated execution-policy audit roots.",
    )
    parser.add_argument("--candidate-profile", default="state_liquidity_listwise_v1")
    parser.add_argument("--baseline-profile", default="baseline_current")
    parser.add_argument("--current-policy", default=DEFAULT_CURRENT_PROFILE)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--root-tag", default="short_alpha_weak_month_review_20260405_r1")
    return parser.parse_args()


def _load_monthly_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["month"] = frame["month"].astype(str)
    frame["start_date"] = pd.to_datetime(frame["start_date"], errors="coerce")
    frame["end_date"] = pd.to_datetime(frame["end_date"], errors="coerce")
    return frame


def _load_formal_runs(formal_root: Path, profile_names: list[str]) -> list[FormalRun]:
    runs_root = formal_root / "runs"
    runs: list[FormalRun] = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        for profile_name in profile_names:
            prefix = f"{profile_name}_"
            if run_dir.name.startswith(prefix):
                runs.append(
                    FormalRun(
                        profile_name=profile_name,
                        run_dir=run_dir,
                        window_label=run_dir.name.removeprefix(prefix),
                    )
                )
                break
    return runs


def _load_profile_monthly_rows(formal_runs: list[FormalRun]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for item in formal_runs:
        monthly = _load_monthly_summary(item.run_dir / "execution_aligned_monthly_backtest_summary.csv")
        monthly.insert(0, "profile_name", item.profile_name)
        monthly.insert(1, "window_label", item.window_label)
        rows.append(monthly)
    if not rows:
        raise RuntimeError("No formal monthly summaries were found.")
    return pd.concat(rows, ignore_index=True)


def _parse_profile_name(profile_run_dir: Path) -> str:
    parts = profile_run_dir.name.split("_")
    if len(parts) >= 3 and all(token.isdigit() and len(token) == 8 for token in parts[-2:]):
        return "_".join(parts[:-2])
    if len(parts) < 3:
        return profile_run_dir.name
    return profile_run_dir.name


def _resolve_month_regimes(profile_run_dir: Path) -> dict[str, dict[str, Any]]:
    regime = pd.read_csv(profile_run_dir / "regime_state.csv")
    regime["Date"] = pd.to_datetime(regime["Date"], errors="coerce")
    regime = regime.dropna(subset=["Date"]).copy()
    regime["month"] = regime["Date"].dt.to_period("M").astype(str)
    rows: dict[str, dict[str, Any]] = {}
    for month, frame in regime.groupby("month", sort=True):
        first = frame.sort_values("Date").iloc[0]
        valid = frame["quadrant"].dropna()
        first_valid = str(valid.iloc[0]).strip() if not valid.empty and str(valid.iloc[0]).strip() else "not_ready"
        rows[str(month)] = {
            "month": str(month),
            "month_start_regime": first_valid,
            "month_start_market_state": str(first.get("market_state", "") or "").strip() or "unknown",
            "month_start_trend_bucket": str(first.get("trend_bucket", "") or "").strip() or "unknown",
            "month_start_vol_bucket": str(first.get("vol_bucket", "") or "").strip() or "unknown",
        }
    return rows


def _load_audit_monthly_rows(audit_root: Path) -> pd.DataFrame:
    profiles_root = audit_root / "profiles"
    rows: list[pd.DataFrame] = []
    regime_rows: dict[str, dict[str, Any]] = {}
    for profile_run_dir in sorted(profiles_root.iterdir()):
        if not profile_run_dir.is_dir():
            continue
        profile_name = _parse_profile_name(profile_run_dir)
        monthly = _load_monthly_summary(profile_run_dir / "monthly_backtest_summary.csv")
        monthly.insert(0, "audit_root", audit_root.name)
        monthly.insert(1, "profile_name", profile_name)
        rows.append(monthly)
        if not regime_rows:
            regime_rows = _resolve_month_regimes(profile_run_dir)
    if not rows:
        raise RuntimeError(f"No audit profile runs found under {audit_root}")
    frame = pd.concat(rows, ignore_index=True)
    regime_frame = pd.DataFrame(sorted(regime_rows.values(), key=lambda item: item["month"]))
    return frame.merge(regime_frame, on="month", how="left")


def _format_pct(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except Exception:
        return "n/a"


def main() -> None:
    args = parse_args()
    formal_root = Path(args.formal_root).resolve()
    output_dir = Path(args.output_root).resolve() / str(args.root_tag).strip()
    output_dir.mkdir(parents=True, exist_ok=True)

    formal_runs = _load_formal_runs(
        formal_root,
        [str(args.candidate_profile).strip(), str(args.baseline_profile).strip()],
    )
    monthly_df = _load_profile_monthly_rows(formal_runs)
    candidate_df = monthly_df.loc[monthly_df["profile_name"] == str(args.candidate_profile).strip()].copy()
    baseline_df = monthly_df.loc[monthly_df["profile_name"] == str(args.baseline_profile).strip()].copy()
    merged = candidate_df.merge(
        baseline_df[
            [
                "window_label",
                "month",
                "benchmark_return",
                "excess_return",
                "avg_turnover",
                "total_trading_cost_return",
            ]
        ],
        on=["window_label", "month"],
        suffixes=("_candidate", "_baseline"),
        how="inner",
    )
    merged["delta_excess_return"] = merged["excess_return_candidate"] - merged["excess_return_baseline"]
    merged["delta_turnover"] = merged["avg_turnover_candidate"] - merged["avg_turnover_baseline"]
    merged["delta_trading_cost_return"] = (
        merged["total_trading_cost_return_candidate"] - merged["total_trading_cost_return_baseline"]
    )
    merged["candidate_underperformed_baseline"] = merged["delta_excess_return"] < 0.0

    audit_rows = []
    for raw_path in str(args.audit_roots or "").split(","):
        token = raw_path.strip()
        if token:
            audit_rows.append(_load_audit_monthly_rows(Path(token).resolve()))
    if not audit_rows:
        raise RuntimeError("No execution-policy audit roots were provided.")
    audit_monthly = pd.concat(audit_rows, ignore_index=True)
    audit_monthly = audit_monthly.drop_duplicates(subset=["profile_name", "month", "start_date"], keep="last")

    current_policy = str(args.current_policy).strip()
    current_policy_df = audit_monthly.loc[audit_monthly["profile_name"] == current_policy].copy()
    current_policy_df = current_policy_df[
        [
            "month",
            "excess_return",
            "avg_turnover",
            "total_trading_cost_return",
            "month_start_regime",
            "month_start_market_state",
            "month_start_trend_bucket",
            "month_start_vol_bucket",
        ]
    ].rename(
        columns={
            "excess_return": "current_policy_excess_return",
            "avg_turnover": "current_policy_avg_turnover",
            "total_trading_cost_return": "current_policy_trading_cost_return",
        }
    )
    policy_ranks = audit_monthly.sort_values(["month", "excess_return", "avg_turnover"], ascending=[True, False, True]).copy()
    policy_ranks["policy_rank_within_month"] = policy_ranks.groupby("month").cumcount() + 1
    best_policy_by_month = policy_ranks.loc[policy_ranks["policy_rank_within_month"] == 1].copy()
    best_policy_by_month = best_policy_by_month[
        ["month", "profile_name", "excess_return", "avg_turnover", "total_trading_cost_return"]
    ].rename(
        columns={
            "profile_name": "best_policy_name",
            "excess_return": "best_policy_excess_return",
            "avg_turnover": "best_policy_avg_turnover",
            "total_trading_cost_return": "best_policy_trading_cost_return",
        }
    )

    review_df = (
        merged.merge(current_policy_df, on="month", how="left")
        .merge(best_policy_by_month, on="month", how="left")
        .sort_values(["window_label", "month"])
        .reset_index(drop=True)
    )
    review_df["best_policy_delta_vs_current"] = (
        review_df["best_policy_excess_return"] - review_df["current_policy_excess_return"]
    )
    review_df["best_policy_is_current"] = review_df["best_policy_name"] == current_policy
    review_df["candidate_gap_vs_best_policy"] = (
        review_df["best_policy_excess_return"] - review_df["excess_return_candidate"]
    )

    weak_df = review_df.loc[review_df["candidate_underperformed_baseline"]].copy()
    review_df.to_csv(output_dir / "all_month_review.csv", index=False, encoding="utf-8-sig")
    weak_df.to_csv(output_dir / "weak_month_review.csv", index=False, encoding="utf-8-sig")

    regime_summary = (
        weak_df.groupby("month_start_regime", dropna=False)
        .agg(
            weak_month_count=("month", "count"),
            avg_candidate_delta_vs_baseline=("delta_excess_return", "mean"),
            avg_best_policy_delta_vs_current=("best_policy_delta_vs_current", "mean"),
            avg_candidate_gap_vs_best_policy=("candidate_gap_vs_best_policy", "mean"),
        )
        .reset_index()
        .sort_values(["weak_month_count", "avg_candidate_gap_vs_best_policy"], ascending=[False, False])
    )
    regime_summary.to_csv(output_dir / "weak_month_regime_summary.csv", index=False, encoding="utf-8-sig")

    policy_help_summary = (
        weak_df.groupby("best_policy_name", dropna=False)
        .agg(
            weak_month_count=("month", "count"),
            avg_best_policy_excess_return=("best_policy_excess_return", "mean"),
            avg_current_policy_excess_return=("current_policy_excess_return", "mean"),
            avg_best_policy_delta_vs_current=("best_policy_delta_vs_current", "mean"),
        )
        .reset_index()
        .sort_values(["weak_month_count", "avg_best_policy_delta_vs_current"], ascending=[False, False])
    )
    policy_help_summary.to_csv(output_dir / "weak_month_policy_help_summary.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# Short Alpha Weak-Month Review",
        "",
        f"- candidate_profile: `{args.candidate_profile}`",
        f"- baseline_profile: `{args.baseline_profile}`",
        f"- current_policy: `{current_policy}`",
        f"- formal_month_count: `{len(review_df)}`",
        f"- weak_month_count: `{len(weak_df)}`",
        f"- weak_month_ratio: `{_format_pct(len(weak_df) / max(len(review_df), 1))}`",
        f"- average candidate delta vs baseline in weak months: `{_format_pct(weak_df['delta_excess_return'].mean())}`",
        f"- average best-policy lift vs current in weak months: `{_format_pct(weak_df['best_policy_delta_vs_current'].mean())}`",
        f"- average candidate gap vs best policy in weak months: `{_format_pct(weak_df['candidate_gap_vs_best_policy'].mean())}`",
        "",
        "## Weak Regime Summary",
    ]
    for _, row in regime_summary.head(5).iterrows():
        lines.append(
            f"- `{row['month_start_regime']}`: weak months `{int(row['weak_month_count'])}`, "
            f"candidate delta vs baseline `{_format_pct(row['avg_candidate_delta_vs_baseline'])}`, "
            f"best-policy lift `{_format_pct(row['avg_best_policy_delta_vs_current'])}`"
        )
    lines.extend(["", "## Best Policies On Weak Months"])
    for _, row in policy_help_summary.head(5).iterrows():
        lines.append(
            f"- `{row['best_policy_name']}`: weak-month wins `{int(row['weak_month_count'])}`, "
            f"lift vs current `{_format_pct(row['avg_best_policy_delta_vs_current'])}`"
        )
    lines.extend(["", "## Weak Month Details"])
    for _, row in weak_df.sort_values(["delta_excess_return", "month"]).iterrows():
        lines.append(
            f"- `{row['month']}` / `{row['window_label']}` / regime `{row['month_start_regime']}`: "
            f"benchmark `{_format_pct(row['benchmark_return_candidate'])}`, "
            f"candidate delta vs baseline `{_format_pct(row['delta_excess_return'])}`, "
            f"best policy `{row['best_policy_name']}` with lift `{_format_pct(row['best_policy_delta_vs_current'])}`"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
