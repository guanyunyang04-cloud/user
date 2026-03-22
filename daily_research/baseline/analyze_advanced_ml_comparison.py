from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two advanced ML output directories and generate attribution tables.")
    parser.add_argument("--base-dir", required=True, help="Reference strategy output directory")
    parser.add_argument("--candidate-dir", required=True, help="Candidate strategy output directory")
    parser.add_argument("--output-dir", default="", help="Optional output directory for comparison tables")
    return parser.parse_args()


def _load_run(run_dir: Path) -> dict:
    equity = pd.read_csv(run_dir / "equity_curve.csv", parse_dates=["date"])
    regime = pd.read_csv(run_dir / "regime_state.csv", parse_dates=["Date"]).rename(columns={"Date": "date"})
    actions = pd.read_csv(run_dir / "actions.csv", parse_dates=["date"])
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return {
        "equity": equity,
        "regime": regime,
        "actions": actions,
        "metrics": metrics,
    }


def _compound(series: pd.Series) -> float:
    series = series.dropna()
    if series.empty:
        return 0.0
    return float((1.0 + series).prod() - 1.0)


def _build_overall(base: dict, candidate: dict, base_name: str, cand_name: str) -> pd.DataFrame:
    keys = [
        "total_return",
        "annual_return",
        "sharpe",
        "max_drawdown",
        "excess_total_return",
        "excess_annual_return",
        "excess_sharpe",
        "excess_max_drawdown",
        "avg_holding_count",
        "avg_turnover",
        "hit_rate",
        "regime_active_ratio",
    ]
    rows = []
    for key in keys:
        b = base["metrics"].get(key)
        c = candidate["metrics"].get(key)
        rows.append(
            {
                "metric": key,
                base_name: b,
                cand_name: c,
                "candidate_minus_base": (c - b) if b is not None and c is not None else None,
            }
        )
    return pd.DataFrame(rows)


def _build_quarterly(run: dict, name: str) -> pd.DataFrame:
    df = run["equity"].copy()
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)
    out = (
        df.groupby("quarter")
        .apply(
            lambda g: pd.Series(
                {
                    f"{name}_portfolio_return": _compound(g["portfolio_return"]),
                    f"{name}_benchmark_return": _compound(g["benchmark_return"]),
                    f"{name}_excess_return": _compound(g["excess_return"]),
                    f"{name}_avg_holding_count": float(g["holding_count"].mean()),
                    f"{name}_avg_turnover": float(g["turnover"].mean()),
                    f"{name}_regime_on_ratio": float(g["regime_on"].mean()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    return out


def _build_quadrant(run: dict, name: str) -> pd.DataFrame:
    df = run["equity"].merge(run["regime"][["date", "quadrant", "regime_on"]], on="date", how="left", suffixes=("", "_r"))
    out = (
        df.groupby("quadrant")
        .apply(
            lambda g: pd.Series(
                {
                    f"{name}_days": int(len(g)),
                    f"{name}_portfolio_return": _compound(g["portfolio_return"]),
                    f"{name}_benchmark_return": _compound(g["benchmark_return"]),
                    f"{name}_excess_return": _compound(g["excess_return"]),
                    f"{name}_avg_holding_count": float(g["holding_count"].mean()),
                    f"{name}_avg_turnover": float(g["turnover"].mean()),
                    f"{name}_regime_on_ratio": float(g["regime_on"].mean()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    return out


def _build_action_summary(run: dict, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    actions = run["actions"].copy()
    action_counts = actions.groupby("action").size().reset_index(name=f"{name}_count")
    reason_counts = actions.groupby(["action", "reason"]).size().reset_index(name=f"{name}_count")
    return action_counts, reason_counts


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    cand_dir = Path(args.candidate_dir)
    if not base_dir.exists():
        raise FileNotFoundError(base_dir)
    if not cand_dir.exists():
        raise FileNotFoundError(cand_dir)

    base_name = base_dir.name
    cand_name = cand_dir.name
    output_dir = Path(args.output_dir) if args.output_dir else base_dir.parent / f"{base_name}_vs_{cand_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    base = _load_run(base_dir)
    cand = _load_run(cand_dir)

    overall = _build_overall(base, cand, base_name, cand_name)
    quarterly = _build_quarterly(base, base_name).merge(
        _build_quarterly(cand, cand_name), on="quarter", how="outer"
    )
    if f"{cand_name}_excess_return" in quarterly and f"{base_name}_excess_return" in quarterly:
        quarterly["candidate_minus_base_excess_return"] = (
            quarterly[f"{cand_name}_excess_return"] - quarterly[f"{base_name}_excess_return"]
        )

    quadrant = _build_quadrant(base, base_name).merge(
        _build_quadrant(cand, cand_name), on="quadrant", how="outer"
    )
    if f"{cand_name}_excess_return" in quadrant and f"{base_name}_excess_return" in quadrant:
        quadrant["candidate_minus_base_excess_return"] = (
            quadrant[f"{cand_name}_excess_return"] - quadrant[f"{base_name}_excess_return"]
        )

    action_base, reason_base = _build_action_summary(base, base_name)
    action_cand, reason_cand = _build_action_summary(cand, cand_name)
    action_summary = action_base.merge(action_cand, on="action", how="outer").fillna(0)
    reason_summary = reason_base.merge(reason_cand, on=["action", "reason"], how="outer").fillna(0)

    overall.to_csv(output_dir / "overall_comparison.csv", index=False, encoding="utf-8-sig")
    quarterly.to_csv(output_dir / "quarterly_comparison.csv", index=False, encoding="utf-8-sig")
    quadrant.to_csv(output_dir / "quadrant_comparison.csv", index=False, encoding="utf-8-sig")
    action_summary.to_csv(output_dir / "action_comparison.csv", index=False, encoding="utf-8-sig")
    reason_summary.to_csv(output_dir / "action_reason_comparison.csv", index=False, encoding="utf-8-sig")

    summary = {
        "base_dir": str(base_dir),
        "candidate_dir": str(cand_dir),
        "output_dir": str(output_dir),
        "best_by_excess_sharpe": base_name
        if float(base["metrics"].get("excess_sharpe", float("-inf"))) >= float(cand["metrics"].get("excess_sharpe", float("-inf")))
        else cand_name,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
