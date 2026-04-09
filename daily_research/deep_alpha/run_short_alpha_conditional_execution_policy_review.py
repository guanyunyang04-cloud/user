from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
EXTERNAL_REPLAY_SCRIPT = PROJECT_ROOT / "daily_research" / "baseline" / "backtest_external_score_panel.py"
DEFAULT_AUDIT_ROOTS = (
    OUTPUT_ROOT / "short_alpha_formal_execution_policy_audit_20230216_20240229_20260405_r1",
    OUTPUT_ROOT / "short_alpha_formal_execution_policy_audit_20240301_20250317_20260405_r1",
    OUTPUT_ROOT / "short_alpha_formal_execution_policy_audit_20260405_r1",
)
DEFAULT_STATIC_PROFILE = "regoff_k2_5d_ensemble_native_anchor"


@dataclass(frozen=True)
class AuditWindow:
    window_key: str
    audit_root: Path
    profiles_root: Path
    start_date: str
    end_date: str
    month_regime_map: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a month-start-regime-conditioned execution policy from short_alpha "
            "formal execution-policy audits and evaluate it leave-window-out."
        )
    )
    parser.add_argument(
        "--audit-roots",
        default=",".join(str(path) for path in DEFAULT_AUDIT_ROOTS),
    )
    parser.add_argument("--static-profile", default=DEFAULT_STATIC_PROFILE)
    parser.add_argument("--min-regime-support", type=int, default=2)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--root-tag", default="short_alpha_conditional_execution_policy_review_20260405_r1")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_long_panel(path: Path, value_name: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["stock"] = raw["stock"].astype(str).str.upper().str.strip()
    raw[value_name] = pd.to_numeric(raw[value_name], errors="coerce")
    raw = raw.dropna(subset=["date", value_name])
    return (
        raw.sort_values(["date", "stock"])
        .drop_duplicates(subset=["date", "stock"], keep="last")
        .pivot(index="date", columns="stock", values=value_name)
        .sort_index()
        .fillna(0.0)
    )


def _parse_profile_name(profile_run_dir: Path) -> str:
    parts = profile_run_dir.name.split("_")
    if len(parts) >= 3 and all(token.isdigit() and len(token) == 8 for token in parts[-2:]):
        return "_".join(parts[:-2])
    if len(parts) < 3:
        return profile_run_dir.name
    return profile_run_dir.name


def _resolve_month_regime_map(profile_run_dir: Path) -> dict[str, str]:
    regime = pd.read_csv(profile_run_dir / "regime_state.csv")
    regime["Date"] = pd.to_datetime(regime["Date"], errors="coerce")
    regime = regime.dropna(subset=["Date"]).copy()
    regime["month"] = regime["Date"].dt.to_period("M").astype(str)
    out: dict[str, str] = {}
    for month, frame in regime.groupby("month", sort=True):
        values = frame["quadrant"].dropna()
        out[str(month)] = str(values.iloc[0]).strip() if not values.empty and str(values.iloc[0]).strip() else "not_ready"
    return out


def _discover_window(audit_root: Path) -> AuditWindow:
    profiles_root = audit_root / "profiles"
    first_profile_run = next(path for path in sorted(profiles_root.iterdir()) if path.is_dir())
    monthly = pd.read_csv(first_profile_run / "monthly_backtest_summary.csv")
    start_date = pd.to_datetime(monthly["start_date"], errors="coerce").min().strftime("%Y%m%d")
    end_date = pd.to_datetime(monthly["end_date"], errors="coerce").max().strftime("%Y%m%d")
    return AuditWindow(
        window_key=f"{start_date}_{end_date}",
        audit_root=audit_root,
        profiles_root=profiles_root,
        start_date=start_date,
        end_date=end_date,
        month_regime_map=_resolve_month_regime_map(first_profile_run),
    )


def _load_window_monthly(window: AuditWindow) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for profile_run_dir in sorted(window.profiles_root.iterdir()):
        if not profile_run_dir.is_dir():
            continue
        profile_name = _parse_profile_name(profile_run_dir)
        monthly = pd.read_csv(profile_run_dir / "monthly_backtest_summary.csv")
        monthly["month"] = monthly["month"].astype(str)
        monthly.insert(0, "window_key", window.window_key)
        monthly.insert(1, "profile_name", profile_name)
        monthly["month_start_regime"] = monthly["month"].map(window.month_regime_map).fillna("not_ready")
        rows.append(monthly)
    if not rows:
        raise RuntimeError(f"No profile monthly summaries found under {window.audit_root}")
    return pd.concat(rows, ignore_index=True)


def _resolve_profile_mapping(
    training_df: pd.DataFrame,
    *,
    static_profile: str,
    min_regime_support: int,
) -> tuple[dict[str, str], str, pd.DataFrame]:
    global_scores = (
        training_df.groupby("profile_name", dropna=False)["excess_return"]
        .mean()
        .sort_values(ascending=False)
        .reset_index(name="mean_excess_return")
    )
    global_best = str(global_scores.iloc[0]["profile_name"]) if not global_scores.empty else static_profile
    regime_rows: list[dict[str, Any]] = []
    mapping: dict[str, str] = {}
    for regime, frame in training_df.groupby("month_start_regime", dropna=False):
        score_table = (
            frame.groupby("profile_name", dropna=False)
            .agg(
                month_count=("month", "nunique"),
                mean_excess_return=("excess_return", "mean"),
                mean_turnover=("avg_turnover", "mean"),
            )
            .reset_index()
            .sort_values(["mean_excess_return", "mean_turnover"], ascending=[False, True])
        )
        support = int(frame["month"].nunique())
        chosen = str(score_table.iloc[0]["profile_name"]) if not score_table.empty and support >= int(min_regime_support) else global_best
        mapping[str(regime)] = chosen
        best_row = score_table.iloc[0] if not score_table.empty else None
        regime_rows.append(
            {
                "month_start_regime": str(regime),
                "month_support": support,
                "chosen_profile": chosen,
                "global_best_profile": global_best,
                "regime_best_profile": "" if best_row is None else str(best_row["profile_name"]),
                "regime_best_mean_excess_return": float("nan") if best_row is None else float(best_row["mean_excess_return"]),
            }
        )
    return mapping, global_best, pd.DataFrame(regime_rows).sort_values("month_start_regime").reset_index(drop=True)


def _build_hybrid_target_weight_panel(window: AuditWindow, month_choice_df: pd.DataFrame) -> pd.DataFrame:
    panels: dict[str, pd.DataFrame] = {}
    for profile_name in sorted(month_choice_df["selected_profile"].unique()):
        profile_dirs = [path for path in window.profiles_root.iterdir() if path.is_dir() and _parse_profile_name(path) == profile_name]
        if not profile_dirs:
            raise FileNotFoundError(f"Profile run not found for {profile_name} under {window.audit_root}")
        panels[profile_name] = _load_long_panel(profile_dirs[0] / "aligned_daily_target_weight_panel.csv", "target_weight")

    stitched: list[pd.DataFrame] = []
    for _, row in month_choice_df.iterrows():
        month = str(row["month"])
        period = pd.Period(month, freq="M")
        panel = panels[str(row["selected_profile"])]
        month_panel = panel.loc[panel.index.to_period("M") == period]
        if not month_panel.empty:
            stitched.append(month_panel)
    if not stitched:
        raise RuntimeError(f"No hybrid target weights were constructed for {window.window_key}")
    hybrid = pd.concat(stitched).sort_index()
    hybrid = hybrid.loc[~hybrid.index.duplicated(keep="last")]
    return hybrid.fillna(0.0)


def _run_replay(
    *,
    python_executable: str,
    hybrid_panel_path: Path,
    start_date: str,
    end_date: str,
    output_dir: Path,
    experiment_tag: str,
    candidate_label: str,
) -> Path:
    cmd = [
        str(python_executable),
        str(EXTERNAL_REPLAY_SCRIPT),
        "--target-weight-panel-csv",
        str(hybrid_panel_path),
        "--data-source",
        "tq",
        "--benchmark",
        "000300.SH",
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--rebalance-freq",
        "1d",
        "--rebalance-offset-mode",
        "single",
        "--transaction-cost-bps",
        "3",
        "--slippage-bps",
        "7",
        "--sell-tax-bps",
        "10",
        "--no-market-regime-filter",
        "--output-dir",
        str(output_dir),
        "--experiment-tag",
        experiment_tag,
        "--candidate-label",
        candidate_label,
    ]
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    return output_dir / experiment_tag


def _load_static_metrics(window: AuditWindow, static_profile: str) -> tuple[dict[str, Any], Path]:
    for path in window.profiles_root.iterdir():
        if path.is_dir() and _parse_profile_name(path) == static_profile:
            return _load_json(path / "metrics.json"), path
    raise FileNotFoundError(f"Static profile {static_profile} not found under {window.audit_root}")


def _format_pct(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except Exception:
        return "n/a"


def _format_num(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "n/a"


def main() -> None:
    args = parse_args()
    audit_roots = [Path(token.strip()).resolve() for token in str(args.audit_roots or "").split(",") if token.strip()]
    windows = [_discover_window(path) for path in audit_roots]
    if len(windows) < 2:
        raise RuntimeError("At least two audit windows are required for leave-window-out review.")

    output_dir = Path(args.output_root).resolve() / str(args.root_tag).strip()
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly_by_window = {window.window_key: _load_window_monthly(window) for window in windows}

    compare_rows: list[dict[str, Any]] = []
    month_choice_exports: list[pd.DataFrame] = []
    mapping_exports: list[pd.DataFrame] = []

    for window in windows:
        train_df = pd.concat([frame for key, frame in monthly_by_window.items() if key != window.window_key], ignore_index=True)
        mapping, global_best, mapping_df = _resolve_profile_mapping(
            train_df,
            static_profile=str(args.static_profile).strip(),
            min_regime_support=int(args.min_regime_support),
        )
        mapping_df.insert(0, "test_window_key", window.window_key)
        mapping_exports.append(mapping_df)

        test_months = monthly_by_window[window.window_key].copy()
        static_profile = str(args.static_profile).strip()
        static_months = test_months.loc[test_months["profile_name"] == static_profile, ["month", "month_start_regime"]].copy()
        static_months["selected_profile"] = static_months["month_start_regime"].map(mapping).fillna(global_best)
        static_months["global_best_profile"] = global_best
        static_months.insert(0, "test_window_key", window.window_key)
        month_choice_exports.append(static_months)

        hybrid_panel = _build_hybrid_target_weight_panel(window, static_months)
        hybrid_panel_path = output_dir / f"hybrid_target_weight_panel_{window.window_key}.csv"
        hybrid_panel.to_csv(hybrid_panel_path, index_label="date", encoding="utf-8-sig")
        replay_run_dir = _run_replay(
            python_executable=str(args.python_executable),
            hybrid_panel_path=hybrid_panel_path,
            start_date=window.start_date,
            end_date=window.end_date,
            output_dir=output_dir,
            experiment_tag=f"conditional_replays/{window.window_key}",
            candidate_label=f"conditional_policy_{window.window_key}",
        )
        conditional_metrics = _load_json(replay_run_dir / "metrics.json")
        static_metrics, static_run_dir = _load_static_metrics(window, static_profile)
        compare_rows.append(
            {
                "window_key": window.window_key,
                "train_global_best_profile": global_best,
                "conditional_excess_annual_return": float(conditional_metrics.get("excess_annual_return", 0.0) or 0.0),
                "conditional_excess_sharpe": float(conditional_metrics.get("excess_sharpe", 0.0) or 0.0),
                "conditional_avg_turnover": float(conditional_metrics.get("avg_turnover", 0.0) or 0.0),
                "static_profile": static_profile,
                "static_excess_annual_return": float(static_metrics.get("excess_annual_return", 0.0) or 0.0),
                "static_excess_sharpe": float(static_metrics.get("excess_sharpe", 0.0) or 0.0),
                "static_avg_turnover": float(static_metrics.get("avg_turnover", 0.0) or 0.0),
                "delta_excess_annual_return": float(conditional_metrics.get("excess_annual_return", 0.0) or 0.0) - float(static_metrics.get("excess_annual_return", 0.0) or 0.0),
                "delta_excess_sharpe": float(conditional_metrics.get("excess_sharpe", 0.0) or 0.0) - float(static_metrics.get("excess_sharpe", 0.0) or 0.0),
                "delta_avg_turnover": float(conditional_metrics.get("avg_turnover", 0.0) or 0.0) - float(static_metrics.get("avg_turnover", 0.0) or 0.0),
                "wins_excess_annual_return": bool(float(conditional_metrics.get("excess_annual_return", 0.0) or 0.0) > float(static_metrics.get("excess_annual_return", 0.0) or 0.0)),
                "wins_excess_sharpe": bool(float(conditional_metrics.get("excess_sharpe", 0.0) or 0.0) > float(static_metrics.get("excess_sharpe", 0.0) or 0.0)),
                "conditional_run_dir": str(replay_run_dir),
                "static_run_dir": str(static_run_dir),
            }
        )

    compare_df = pd.DataFrame(compare_rows).sort_values("window_key").reset_index(drop=True)
    compare_df.to_csv(output_dir / "window_compare.csv", index=False, encoding="utf-8-sig")
    pd.concat(month_choice_exports, ignore_index=True).to_csv(output_dir / "month_policy_choices.csv", index=False, encoding="utf-8-sig")
    pd.concat(mapping_exports, ignore_index=True).to_csv(output_dir / "regime_policy_mapping.csv", index=False, encoding="utf-8-sig")

    summary = {
        "static_profile": str(args.static_profile).strip(),
        "window_count": int(len(compare_df)),
        "conditional_mean_excess_annual_return": float(compare_df["conditional_excess_annual_return"].mean()),
        "conditional_mean_excess_sharpe": float(compare_df["conditional_excess_sharpe"].mean()),
        "static_mean_excess_annual_return": float(compare_df["static_excess_annual_return"].mean()),
        "static_mean_excess_sharpe": float(compare_df["static_excess_sharpe"].mean()),
        "delta_mean_excess_annual_return": float(compare_df["delta_excess_annual_return"].mean()),
        "delta_mean_excess_sharpe": float(compare_df["delta_excess_sharpe"].mean()),
        "wins_excess_annual_return": int(compare_df["wins_excess_annual_return"].sum()),
        "wins_excess_sharpe": int(compare_df["wins_excess_sharpe"].sum()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Short Alpha Conditional Execution Policy Review",
        "",
        f"- static_profile: `{summary['static_profile']}`",
        f"- window_count: `{summary['window_count']}`",
        f"- conditional mean excess annual: `{_format_pct(summary['conditional_mean_excess_annual_return'])}`",
        f"- conditional mean excess Sharpe: `{_format_num(summary['conditional_mean_excess_sharpe'])}`",
        f"- static mean excess annual: `{_format_pct(summary['static_mean_excess_annual_return'])}`",
        f"- static mean excess Sharpe: `{_format_num(summary['static_mean_excess_sharpe'])}`",
        f"- delta mean excess annual: `{_format_pct(summary['delta_mean_excess_annual_return'])}`",
        f"- delta mean excess Sharpe: `{_format_num(summary['delta_mean_excess_sharpe'])}`",
        f"- annual wins: `{summary['wins_excess_annual_return']}/{summary['window_count']}`",
        f"- Sharpe wins: `{summary['wins_excess_sharpe']}/{summary['window_count']}`",
        "",
        "## Window Results",
    ]
    for _, row in compare_df.iterrows():
        lines.append(
            f"- `{row['window_key']}`: conditional excess annual `{_format_pct(row['conditional_excess_annual_return'])}`, "
            f"static `{_format_pct(row['static_excess_annual_return'])}`, "
            f"delta `{_format_pct(row['delta_excess_annual_return'])}`; "
            f"conditional excess Sharpe `{_format_num(row['conditional_excess_sharpe'])}`, "
            f"static `{_format_num(row['static_excess_sharpe'])}`, "
            f"delta `{_format_num(row['delta_excess_sharpe'])}`"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
