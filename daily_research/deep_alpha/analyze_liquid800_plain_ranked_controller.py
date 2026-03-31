from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from daily_research.baseline.backtest import _annualized_return, _annualized_vol, _max_drawdown


WINDOW_RE = re.compile(r"(\d{8}_\d{8})$")


@dataclass(frozen=True)
class WindowRun:
    candidate: str
    window_label: str
    run_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a signal-state controller between liquid800 plain and ranked deep_alpha candidates."
    )
    parser.add_argument(
        "--root-dir",
        default="daily_research/output/deep_alpha_opportunity_liquid800_20260329_formal_r1",
        help="Root directory containing wf_* and ranked_wf_* run folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Defaults to <root-dir>/plain_ranked_controller_analysis.",
    )
    parser.add_argument(
        "--default-candidate",
        choices=["plain", "ranked"],
        default="plain",
        help="Fallback candidate when no trained state preference exists.",
    )
    return parser.parse_args()


def _extract_window_label(path: Path) -> str:
    match = WINDOW_RE.search(path.name)
    if not match:
        raise ValueError(f"Unable to parse window label from: {path}")
    return match.group(1)


def _discover_runs(root_dir: Path) -> list[WindowRun]:
    runs: list[WindowRun] = []
    for path in sorted(root_dir.iterdir()):
        if not path.is_dir():
            continue
        if path.name.startswith("wf_"):
            runs.append(WindowRun(candidate="plain", window_label=_extract_window_label(path), run_dir=path))
        elif path.name.startswith("ranked_wf_"):
            runs.append(WindowRun(candidate="ranked", window_label=_extract_window_label(path), run_dir=path))
    if not runs:
        raise RuntimeError(f"No plain/ranked window runs found under: {root_dir}")
    return runs


def _load_window_run(run: WindowRun) -> pd.DataFrame:
    equity = pd.read_csv(
        run.run_dir / "equity_curve.csv",
        encoding="utf-8-sig",
        parse_dates=["date", "signal_date", "execution_date"],
    )
    state = pd.read_csv(run.run_dir / "market_state_frame.csv", encoding="utf-8-sig")
    state_date_col = "Date" if "Date" in state.columns else "date"
    state[state_date_col] = pd.to_datetime(state[state_date_col])
    state = state.rename(columns={state_date_col: "signal_date"})
    merged = equity.merge(
        state[["signal_date", "state_name", "state_id"]],
        on="signal_date",
        how="left",
    )
    merged["candidate"] = run.candidate
    merged["window_label"] = run.window_label
    merged["state_name"] = merged["state_name"].fillna("unknown")
    merged["state_id"] = merged["state_id"].fillna(-1).astype(int)
    merged["soft_override_active"] = merged["soft_override_active"].where(
        merged["soft_override_active"].notna(),
        False,
    ).astype(bool)
    return merged


def _compute_return_metrics(
    stitched: pd.DataFrame,
    *,
    controller_label: str,
) -> dict[str, float | str]:
    work = stitched.copy()
    work = work.sort_values("date").reset_index(drop=True)
    if work.empty:
        raise ValueError("No stitched rows to score.")

    portfolio_equity = pd.Series(
        (1.0 + work["portfolio_return"].fillna(0.0)).cumprod().to_numpy(),
        index=work["date"],
    )
    benchmark_equity = pd.Series(
        (1.0 + work["benchmark_return"].fillna(0.0)).cumprod().to_numpy(),
        index=work["date"],
    )
    excess_equity = (portfolio_equity / benchmark_equity).replace([np.inf, -np.inf], np.nan).ffill().dropna()

    portfolio_ann_ret = _annualized_return(portfolio_equity)
    benchmark_ann_ret = _annualized_return(benchmark_equity)
    excess_ann_ret = _annualized_return(excess_equity) if not excess_equity.empty else 0.0

    portfolio_ann_vol = _annualized_vol(work["portfolio_return"].dropna())
    excess_ann_vol = _annualized_vol(work["excess_return"].dropna())

    ranked_ratio = float(work["chosen_candidate"].eq("ranked").mean()) if "chosen_candidate" in work.columns else np.nan
    return {
        "controller_label": controller_label,
        "total_return": float(portfolio_equity.iloc[-1] - 1.0),
        "annual_return": float(portfolio_ann_ret),
        "benchmark_total_return": float(benchmark_equity.iloc[-1] - 1.0),
        "benchmark_annual_return": float(benchmark_ann_ret),
        "excess_total_return": float(excess_equity.iloc[-1] - 1.0) if not excess_equity.empty else 0.0,
        "excess_annual_return": float(excess_ann_ret),
        "sharpe": float(portfolio_ann_ret / portfolio_ann_vol) if portfolio_ann_vol > 0 else 0.0,
        "excess_sharpe": float(excess_ann_ret / excess_ann_vol) if excess_ann_vol > 0 else 0.0,
        "max_drawdown": float(_max_drawdown(portfolio_equity)),
        "excess_max_drawdown": float(_max_drawdown(excess_equity)) if not excess_equity.empty else 0.0,
        "avg_turnover": float(work["turnover"].fillna(0.0).mean()),
        "avg_holding_count": float(work["holding_count"].fillna(0.0).mean()),
        "ranked_usage_ratio": ranked_ratio,
        "note": "approximate controller: stitches per-day candidate returns and ignores cross-candidate switching cost",
    }


def _summarize_candidate(rows: pd.DataFrame, candidate_label: str) -> dict[str, float | str]:
    rows = rows.copy()
    rows["chosen_candidate"] = candidate_label
    return _compute_return_metrics(rows, controller_label=candidate_label)


def _fit_state_preferences(train_rows: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        train_rows.groupby(["state_name", "candidate"], as_index=False)
        .agg(
            mean_excess_return=("excess_return", "mean"),
            mean_excess_sharpe_proxy=("excess_return", lambda s: float(s.mean() / s.std()) if len(s) > 1 and float(s.std()) > 0 else np.nan),
            mean_portfolio_return=("portfolio_return", "mean"),
            mean_turnover=("turnover", "mean"),
            date_count=("date", "nunique"),
        )
    )
    grouped["mean_excess_sharpe_proxy"] = grouped["mean_excess_sharpe_proxy"].fillna(-np.inf)
    ordered = grouped.sort_values(
        ["state_name", "mean_excess_return", "mean_excess_sharpe_proxy", "mean_portfolio_return"],
        ascending=[True, False, False, False],
    )
    winners = ordered.groupby("state_name", as_index=False).head(1).reset_index(drop=True)
    winners = winners.rename(columns={"candidate": "preferred_candidate"})
    return winners


def _apply_state_preferences(
    *,
    plain_rows: pd.DataFrame,
    ranked_rows: pd.DataFrame,
    preferences: pd.DataFrame,
    default_candidate: str,
) -> pd.DataFrame:
    base = plain_rows[
        [
            "date",
            "signal_date",
            "execution_date",
            "benchmark_return",
            "benchmark_equity",
            "state_name",
            "state_id",
            "window_label",
        ]
    ].copy()
    base = base.merge(
        plain_rows[["date", "portfolio_return", "excess_return", "turnover", "holding_count"]].rename(
            columns={
                "portfolio_return": "plain_portfolio_return",
                "excess_return": "plain_excess_return",
                "turnover": "plain_turnover",
                "holding_count": "plain_holding_count",
            }
        ),
        on="date",
        how="left",
    )
    base = base.merge(
        ranked_rows[["date", "portfolio_return", "excess_return", "turnover", "holding_count"]].rename(
            columns={
                "portfolio_return": "ranked_portfolio_return",
                "excess_return": "ranked_excess_return",
                "turnover": "ranked_turnover",
                "holding_count": "ranked_holding_count",
            }
        ),
        on="date",
        how="left",
    )
    pref_map = preferences[["state_name", "preferred_candidate"]].copy()
    base = base.merge(pref_map, on="state_name", how="left")
    base["chosen_candidate"] = base["preferred_candidate"].fillna(default_candidate)
    base["portfolio_return"] = np.where(
        base["chosen_candidate"].eq("ranked"),
        base["ranked_portfolio_return"],
        base["plain_portfolio_return"],
    )
    base["excess_return"] = np.where(
        base["chosen_candidate"].eq("ranked"),
        base["ranked_excess_return"],
        base["plain_excess_return"],
    )
    base["turnover"] = np.where(
        base["chosen_candidate"].eq("ranked"),
        base["ranked_turnover"],
        base["plain_turnover"],
    )
    base["holding_count"] = np.where(
        base["chosen_candidate"].eq("ranked"),
        base["ranked_holding_count"],
        base["plain_holding_count"],
    )
    return base.sort_values("date").reset_index(drop=True)


def _build_walkforward_controller(
    combined: pd.DataFrame,
    *,
    default_candidate: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    window_order = sorted(combined["window_label"].dropna().unique())
    preference_rows: list[pd.DataFrame] = []
    stitched_rows: list[pd.DataFrame] = []

    for idx, window_label in enumerate(window_order):
        eval_rows = combined.loc[combined["window_label"].eq(window_label)].copy()
        plain_eval = eval_rows.loc[eval_rows["candidate"].eq("plain")].copy()
        ranked_eval = eval_rows.loc[eval_rows["candidate"].eq("ranked")].copy()
        train_rows = combined.loc[combined["window_label"].isin(window_order[:idx])].copy()

        if train_rows.empty:
            preferences = pd.DataFrame(columns=["state_name", "preferred_candidate"])
        else:
            preferences = _fit_state_preferences(train_rows)
            preferences.insert(0, "fit_upto_window", window_order[idx - 1])
            preferences.insert(1, "eval_window", window_label)
            preference_rows.append(preferences.copy())

        controller_eval = _apply_state_preferences(
            plain_rows=plain_eval,
            ranked_rows=ranked_eval,
            preferences=preferences if not train_rows.empty else pd.DataFrame(columns=["state_name", "preferred_candidate"]),
            default_candidate=default_candidate,
        )
        controller_eval["controller_kind"] = "walkforward_state_controller"
        stitched_rows.append(controller_eval)

    pref_df = pd.concat(preference_rows, ignore_index=True) if preference_rows else pd.DataFrame(
        columns=["fit_upto_window", "eval_window", "state_name", "preferred_candidate"]
    )
    stitched_df = pd.concat(stitched_rows, ignore_index=True) if stitched_rows else pd.DataFrame()
    return pref_df, stitched_df


def _build_oracle_state_controller(
    combined: pd.DataFrame,
    *,
    default_candidate: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    preferences = _fit_state_preferences(combined)
    stitched_rows: list[pd.DataFrame] = []
    for window_label in sorted(combined["window_label"].dropna().unique()):
        eval_rows = combined.loc[combined["window_label"].eq(window_label)].copy()
        plain_eval = eval_rows.loc[eval_rows["candidate"].eq("plain")].copy()
        ranked_eval = eval_rows.loc[eval_rows["candidate"].eq("ranked")].copy()
        controller_eval = _apply_state_preferences(
            plain_rows=plain_eval,
            ranked_rows=ranked_eval,
            preferences=preferences,
            default_candidate=default_candidate,
        )
        controller_eval["controller_kind"] = "oracle_state_controller"
        stitched_rows.append(controller_eval)
    stitched_df = pd.concat(stitched_rows, ignore_index=True) if stitched_rows else pd.DataFrame()
    return preferences, stitched_df


def _build_window_oracle(combined: pd.DataFrame) -> pd.DataFrame:
    stitched: list[pd.DataFrame] = []
    for window_label in sorted(combined["window_label"].dropna().unique()):
        window_rows = combined.loc[combined["window_label"].eq(window_label)].copy()
        ranking = (
            window_rows.groupby("candidate", as_index=False)
            .agg(
                mean_excess_return=("excess_return", "mean"),
                mean_portfolio_return=("portfolio_return", "mean"),
            )
            .sort_values(["mean_excess_return", "mean_portfolio_return"], ascending=[False, False])
        )
        chosen = str(ranking.iloc[0]["candidate"])
        candidate_rows = window_rows.loc[window_rows["candidate"].eq(chosen)].copy()
        candidate_rows["chosen_candidate"] = chosen
        candidate_rows["controller_kind"] = "window_oracle"
        stitched.append(candidate_rows)
    return pd.concat(stitched, ignore_index=True) if stitched else pd.DataFrame()


def _write_summary_markdown(
    *,
    output_dir: Path,
    metrics_df: pd.DataFrame,
    state_advantage_df: pd.DataFrame,
    walkforward_pref_df: pd.DataFrame,
) -> None:
    lines = [
        "# Liquid800 Plain vs Ranked Controller Analysis",
        "",
        "All controller metrics here are approximate stitched-return diagnostics.",
        "They reuse per-day candidate returns and do not model cross-candidate switching cost.",
        "",
        "## Metrics",
        "",
        "| controller | annual_return | excess_annual_return | excess_sharpe | excess_max_drawdown | ranked_usage_ratio | avg_turnover |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in metrics_df.iterrows():
        lines.append(
            f"| `{row['controller_label']}` | "
            f"{row['annual_return']:.3f} | "
            f"{row['excess_annual_return']:.3f} | "
            f"{row['excess_sharpe']:.3f} | "
            f"{row['excess_max_drawdown']:.3f} | "
            f"{row['ranked_usage_ratio']:.3f} | "
            f"{row['avg_turnover']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## State Advantage",
            "",
            state_advantage_df.to_markdown(index=False) if not state_advantage_df.empty else "_no state rows_",
            "",
            "## Walkforward Fitted State Preferences",
            "",
            walkforward_pref_df.to_markdown(index=False) if not walkforward_pref_df.empty else "_only default fallback used_",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root_dir = (PROJECT_ROOT / args.root_dir).resolve() if not Path(args.root_dir).is_absolute() else Path(args.root_dir)
    output_dir = (
        (root_dir / "plain_ranked_controller_analysis").resolve()
        if not args.output_dir
        else ((PROJECT_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = _discover_runs(root_dir)
    run_frames = [_load_window_run(run) for run in runs]
    combined = pd.concat(run_frames, ignore_index=True).sort_values(["window_label", "candidate", "date"]).reset_index(drop=True)
    combined.to_csv(output_dir / "combined_daily_rows.csv", index=False, encoding="utf-8-sig")

    state_advantage = (
        combined.groupby(["state_name", "candidate"], as_index=False)
        .agg(
            mean_excess_return=("excess_return", "mean"),
            mean_portfolio_return=("portfolio_return", "mean"),
            mean_turnover=("turnover", "mean"),
            date_count=("date", "nunique"),
        )
        .pivot(index="state_name", columns="candidate", values=["mean_excess_return", "mean_portfolio_return", "mean_turnover", "date_count"])
    )
    state_advantage.columns = [f"{metric}_{candidate}" for metric, candidate in state_advantage.columns]
    state_advantage = state_advantage.reset_index()
    if "mean_excess_return_plain" in state_advantage.columns and "mean_excess_return_ranked" in state_advantage.columns:
        state_advantage["ranked_minus_plain_excess_return"] = (
            state_advantage["mean_excess_return_ranked"] - state_advantage["mean_excess_return_plain"]
        )
    state_advantage.to_csv(output_dir / "state_advantage_summary.csv", index=False, encoding="utf-8-sig")

    walkforward_pref_df, walkforward_rows = _build_walkforward_controller(
        combined,
        default_candidate=args.default_candidate,
    )
    walkforward_pref_df.to_csv(output_dir / "walkforward_state_preferences.csv", index=False, encoding="utf-8-sig")
    walkforward_rows.to_csv(output_dir / "walkforward_controller_rows.csv", index=False, encoding="utf-8-sig")

    oracle_pref_df, oracle_rows = _build_oracle_state_controller(
        combined,
        default_candidate=args.default_candidate,
    )
    oracle_pref_df.to_csv(output_dir / "oracle_state_preferences.csv", index=False, encoding="utf-8-sig")
    oracle_rows.to_csv(output_dir / "oracle_controller_rows.csv", index=False, encoding="utf-8-sig")

    window_oracle_rows = _build_window_oracle(combined)
    window_oracle_rows.to_csv(output_dir / "window_oracle_rows.csv", index=False, encoding="utf-8-sig")

    metrics_rows = [
        _summarize_candidate(combined.loc[combined["candidate"].eq("plain")].copy(), "plain"),
        _summarize_candidate(combined.loc[combined["candidate"].eq("ranked")].copy(), "ranked"),
        _compute_return_metrics(walkforward_rows, controller_label="walkforward_state_controller"),
        _compute_return_metrics(oracle_rows, controller_label="oracle_state_controller"),
        _compute_return_metrics(window_oracle_rows, controller_label="window_oracle"),
    ]
    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        ["excess_sharpe", "excess_annual_return", "annual_return"],
        ascending=[False, False, False],
    )
    metrics_df.to_csv(output_dir / "controller_metrics.csv", index=False, encoding="utf-8-sig")
    _write_summary_markdown(
        output_dir=output_dir,
        metrics_df=metrics_df,
        state_advantage_df=state_advantage,
        walkforward_pref_df=walkforward_pref_df,
    )

    diagnosis = {
        "root_dir": str(root_dir),
        "default_candidate": args.default_candidate,
        "best_controller_by_excess_sharpe": metrics_df.iloc[0].to_dict(),
        "approximation_warning": "controller metrics stitch per-day candidate returns and do not include cross-candidate switching cost",
    }
    (output_dir / "diagnosis.json").write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(diagnosis, ensure_ascii=False, indent=2))
    print(f"[ok] controller analysis written to: {output_dir}")


if __name__ == "__main__":
    main()
