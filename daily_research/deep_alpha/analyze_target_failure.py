from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze deep_alpha first-layer target failure structure."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to a deep_alpha research output directory.",
    )
    parser.add_argument(
        "--output-subdir",
        default="target_failure_analysis",
        help="Name of analysis subdirectory created inside run-dir.",
    )
    return parser.parse_args()


def _cross_sectional_signal(pivot: pd.DataFrame, rank_blend: float) -> pd.DataFrame:
    z = pivot.sub(pivot.mean(axis=1), axis=0).div(
        pivot.std(axis=1).replace(0.0, np.nan), axis=0
    )
    rank = pivot.rank(axis=1, pct=True)
    rank_centered = (rank - 0.5) * 2.0
    return z.fillna(0.0) * (1.0 - rank_blend) + rank_centered.fillna(0.0) * rank_blend


def _safe_spearman(group: pd.DataFrame, left: str, right: str) -> float:
    sample = group[[left, right]].dropna()
    if sample[left].nunique() <= 1 or sample[right].nunique() <= 1:
        return np.nan
    return sample[left].corr(sample[right], method="spearman")


def _daily_rankcorr(df: pd.DataFrame, left: str, right: str) -> pd.Series:
    sample = df[["date", left, right]].copy()
    rows = {date: _safe_spearman(group, left, right) for date, group in sample.groupby("date")}
    return pd.Series(rows, dtype=float)


def _summarize_rankcorr(df: pd.DataFrame, left: str, right: str, label: str) -> dict[str, float | str]:
    daily = _daily_rankcorr(df, left, right)
    mean = float(daily.mean()) if not daily.empty else np.nan
    std = float(daily.std()) if not daily.empty else np.nan
    ir = float(mean / std) if std and np.isfinite(std) and std > 0 else np.nan
    return {
        "label": label,
        "left": left,
        "right": right,
        "rankcorr_mean": mean,
        "rankcorr_std": std,
        "rankcorr_ir": ir,
        "date_count": int(daily.notna().sum()),
    }


def _quantile_spread(df: pd.DataFrame, score_col: str) -> dict[str, float]:
    pieces: list[dict[str, float]] = []
    for _, group in df.groupby("date"):
        sample = group[[score_col, "true_fwd_excess_20", "true_risk_downside_20"]].dropna()
        if len(sample) < 10 or sample[score_col].nunique() < 5:
            continue
        ranks = sample[score_col].rank(method="first", pct=True)
        top = sample.loc[ranks >= 0.8]
        bot = sample.loc[ranks <= 0.2]
        if top.empty or bot.empty:
            continue
        pieces.append(
            {
                "top_true20": float(top["true_fwd_excess_20"].mean()),
                "bot_true20": float(bot["true_fwd_excess_20"].mean()),
                "spread20": float(top["true_fwd_excess_20"].mean() - bot["true_fwd_excess_20"].mean()),
                "top_downside": float(top["true_risk_downside_20"].mean()),
                "bot_downside": float(bot["true_risk_downside_20"].mean()),
                "spread_downside": float(top["true_risk_downside_20"].mean() - bot["true_risk_downside_20"].mean()),
            }
        )
    if not pieces:
        return {}
    summary = pd.DataFrame(pieces).mean(numeric_only=True).to_dict()
    return {k: float(v) for k, v in summary.items()}


def _build_manual_score(pred_df: pd.DataFrame, metrics: dict) -> pd.DataFrame:
    horizon_weights = {
        int(k): float(v) for k, v in metrics.get("score_horizon_weights", {}).items()
    }
    rank_blend = float(metrics.get("score_rank_blend", 0.35))
    downside_penalty = float(metrics.get("score_downside_penalty", 0.0))
    all_dates = pd.Index(sorted(pred_df["date"].unique()))
    all_stocks = sorted(pred_df["stock"].unique())

    frames: list[pd.DataFrame] = []
    for horizon, weight in sorted(horizon_weights.items()):
        col = f"pred_fwd_excess_{horizon}"
        if col not in pred_df.columns or weight <= 0:
            continue
        pivot = pred_df.pivot(index="date", columns="stock", values=col)
        frames.append(_cross_sectional_signal(pivot, rank_blend) * weight)
    score = sum(frame.reindex(index=all_dates, columns=all_stocks).fillna(0.0) for frame in frames)
    risk_signal_frame = pd.DataFrame(0.0, index=all_dates, columns=all_stocks)
    if downside_penalty > 0 and "pred_risk_downside_20" in pred_df.columns:
        downside = pred_df.pivot(index="date", columns="stock", values="pred_risk_downside_20")
        # Higher predicted downside risk should reduce the final score.
        risk_signal_frame = _cross_sectional_signal(downside, rank_blend).reindex(
            index=all_dates, columns=all_stocks
        ).fillna(0.0)
        score = score - downside_penalty * risk_signal_frame

    return_component = sum(
        frame.reindex(index=all_dates, columns=all_stocks).fillna(0.0) for frame in frames
    )
    score_long = score.stack(future_stack=True).rename("manual_score").reset_index()
    score_long.columns = ["date", "stock", "manual_score"]
    return_long = return_component.stack(future_stack=True).rename("return_component").reset_index()
    return_long.columns = ["date", "stock", "return_component"]
    risk_long = risk_signal_frame.stack(future_stack=True).rename("risk_component").reset_index()
    risk_long.columns = ["date", "stock", "risk_component"]

    out = pred_df.merge(score_long, on=["date", "stock"], how="left")
    out = out.merge(return_long, on=["date", "stock"], how="left")
    out = out.merge(risk_long, on=["date", "stock"], how="left")
    return out


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = run_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.read_csv(run_dir / "validation_predictions.csv")
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    state_df = pd.read_csv(run_dir / "market_state_frame.csv")
    date_col = "Date" if "Date" in state_df.columns else "date"
    state_df[date_col] = pd.to_datetime(state_df[date_col])
    state_df = state_df.rename(columns={date_col: "date"})

    with (run_dir / "metrics.json").open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)

    merged = pred_df.merge(state_df[["date", "state_name"]], on="date", how="left")
    merged["state_name"] = merged["state_name"].fillna("unknown")

    if metrics.get("score_head_method", "manual") == "manual":
        merged = _build_manual_score(merged, metrics)
    else:
        merged["manual_score"] = np.nan
        merged["return_component"] = np.nan
        merged["risk_component"] = np.nan

    target_cols = [
        "fwd_excess_5",
        "fwd_excess_10",
        "fwd_excess_20",
        "risk_downside_20",
    ]

    state_rows: list[dict[str, float | str]] = []
    for state_name, state_group in merged.groupby("state_name"):
        for target in target_cols:
            row = _summarize_rankcorr(
                state_group,
                f"pred_{target}",
                f"true_{target}",
                label=str(state_name),
            )
            row["target"] = target
            state_rows.append(row)
    state_rankic = pd.DataFrame(state_rows)
    state_rankic.to_csv(output_dir / "state_rankic_summary.csv", index=False, encoding="utf-8-sig")

    alignment_rows: list[dict[str, float | str]] = []
    for label, sample in [("overall", merged)] + list(merged.groupby("state_name")):
        for left, right in [
            ("manual_score", "true_fwd_excess_20"),
            ("manual_score", "true_risk_downside_20"),
            ("return_component", "manual_score"),
            ("risk_component", "manual_score"),
        ]:
            if left not in sample.columns:
                continue
            alignment_rows.append(_summarize_rankcorr(sample, left, right, str(label)))
    alignment_df = pd.DataFrame(alignment_rows)
    alignment_df.to_csv(output_dir / "score_alignment_summary.csv", index=False, encoding="utf-8-sig")

    quantile_rows: list[dict[str, float | str]] = []
    for label, sample in [("overall", merged)] + list(merged.groupby("state_name")):
        spread = _quantile_spread(sample, "manual_score")
        if not spread:
            continue
        spread["label"] = str(label)
        quantile_rows.append(spread)
    quantile_df = pd.DataFrame(quantile_rows)
    quantile_df.to_csv(output_dir / "score_quantile_spread.csv", index=False, encoding="utf-8-sig")

    diagnosis = {
        "run_dir": str(run_dir),
        "score_head_method": metrics.get("score_head_method", "manual"),
        "worst_return_target": None,
        "best_state_for_return": None,
        "worst_state_for_return": None,
        "risk_only_profile": None,
    }
    if not state_rankic.empty:
        return_only = state_rankic[state_rankic["target"].isin(["fwd_excess_5", "fwd_excess_10", "fwd_excess_20"])]
        overall_target = (
            return_only.groupby("target", as_index=False)["rankcorr_mean"].mean().sort_values("rankcorr_mean")
        )
        if not overall_target.empty:
            diagnosis["worst_return_target"] = overall_target.iloc[0].to_dict()
        target20 = state_rankic[state_rankic["target"] == "fwd_excess_20"].sort_values("rankcorr_mean")
        if not target20.empty:
            diagnosis["worst_state_for_return"] = target20.iloc[0].to_dict()
            diagnosis["best_state_for_return"] = target20.iloc[-1].to_dict()
    if not alignment_df.empty:
        overall = alignment_df[alignment_df["label"] == "overall"]
        score_ret = overall[
            (overall["left"] == "manual_score") & (overall["right"] == "true_fwd_excess_20")
        ]
        score_risk = overall[
            (overall["left"] == "manual_score") & (overall["right"] == "true_risk_downside_20")
        ]
        if not score_ret.empty and not score_risk.empty:
            diagnosis["risk_only_profile"] = {
                "score_vs_true20_rankcorr_mean": float(score_ret.iloc[0]["rankcorr_mean"]),
                "score_vs_downside_rankcorr_mean": float(score_risk.iloc[0]["rankcorr_mean"]),
            }

    with (output_dir / "diagnosis.json").open("w", encoding="utf-8") as fh:
        json.dump(diagnosis, fh, ensure_ascii=False, indent=2)

    print(f"[OK] Target failure analysis written to: {output_dir}")
    print(json.dumps(diagnosis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
