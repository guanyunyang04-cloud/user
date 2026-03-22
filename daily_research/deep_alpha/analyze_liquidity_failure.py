from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


RETURN_TARGETS = ["fwd_excess_5", "fwd_excess_10", "fwd_excess_20"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze deep_alpha return-learning quality by liquidity bucket and market state.")
    parser.add_argument("--run-dir", required=True, help="Path to a deep_alpha output directory.")
    parser.add_argument("--output-subdir", default="liquidity_failure_analysis")
    return parser.parse_args()


def _safe_rankcorr(group: pd.DataFrame, left: str, right: str) -> float:
    sample = group[[left, right]].dropna()
    if len(sample) < 5 or sample[left].nunique() <= 1 or sample[right].nunique() <= 1:
        return np.nan
    return float(sample[left].corr(sample[right], method="spearman"))


def _rankic_summary(df: pd.DataFrame, pred_col: str, true_col: str, label_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = df.groupby(label_cols + ["date"], dropna=False)
    daily_rows = []
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {name: value for name, value in zip(label_cols, keys)}
        row["date"] = pd.Timestamp(group["date"].iloc[0])
        row["rankic"] = _safe_rankcorr(group, pred_col, true_col)
        daily_rows.append(row)
    daily_df = pd.DataFrame(daily_rows)
    if daily_df.empty:
        return pd.DataFrame(columns=label_cols + ["rankic_mean", "rankic_std", "rankic_ir", "date_count"])
    summary = (
        daily_df.groupby(label_cols, dropna=False)["rankic"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "rankic_mean", "std": "rankic_std", "count": "date_count"})
    )
    summary["rankic_ir"] = summary["rankic_mean"] / summary["rankic_std"].replace(0.0, np.nan)
    return summary


def _top_bottom_spread(df: pd.DataFrame, score_col: str, true_col: str, label_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = df.groupby(label_cols + ["date"], dropna=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        sample = group[[score_col, true_col]].dropna()
        if len(sample) < 10 or sample[score_col].nunique() < 5:
            continue
        top_n = max(1, int(len(sample) * 0.2))
        spread = float(
            sample.nlargest(top_n, score_col)[true_col].mean()
            - sample.nsmallest(top_n, score_col)[true_col].mean()
        )
        row = {name: value for name, value in zip(label_cols, keys)}
        row["date"] = pd.Timestamp(group["date"].iloc[0])
        row["spread"] = spread
        rows.append(row)
    daily_df = pd.DataFrame(rows)
    if daily_df.empty:
        return pd.DataFrame(columns=label_cols + ["spread_mean", "spread_std", "date_count"])
    summary = (
        daily_df.groupby(label_cols, dropna=False)["spread"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "spread_mean", "std": "spread_std", "count": "date_count"})
    )
    return summary


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = run_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.read_csv(run_dir / "validation_predictions.csv", encoding="utf-8-sig")
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    if "liquidity_bucket" not in pred_df.columns:
        raise ValueError("validation_predictions.csv does not contain liquidity_bucket. Re-run the experiment with the updated deep_alpha pipeline.")
    pred_df["liquidity_bucket"] = pred_df["liquidity_bucket"].fillna(-1).astype(int)

    state_df = pd.read_csv(run_dir / "market_state_frame.csv", encoding="utf-8-sig")
    date_col = "Date" if "Date" in state_df.columns else "date"
    state_df[date_col] = pd.to_datetime(state_df[date_col])
    state_df = state_df.rename(columns={date_col: "date"})
    pred_df = pred_df.merge(state_df[["date", "state_name"]], on="date", how="left")
    pred_df["state_name"] = pred_df["state_name"].fillna("unknown")

    bucket_rows: list[pd.DataFrame] = []
    bucket_state_rows: list[pd.DataFrame] = []
    for target in RETURN_TARGETS:
        pred_col = f"pred_{target}"
        true_col = f"true_{target}"
        bucket_summary = _rankic_summary(pred_df, pred_col, true_col, ["liquidity_bucket"])
        bucket_summary["target"] = target
        bucket_rows.append(bucket_summary)
        bucket_state_summary = _rankic_summary(pred_df, pred_col, true_col, ["state_name", "liquidity_bucket"])
        bucket_state_summary["target"] = target
        bucket_state_rows.append(bucket_state_summary)

    bucket_rankic_df = pd.concat(bucket_rows, ignore_index=True) if bucket_rows else pd.DataFrame()
    bucket_state_rankic_df = pd.concat(bucket_state_rows, ignore_index=True) if bucket_state_rows else pd.DataFrame()
    bucket_rankic_df.to_csv(output_dir / "bucket_rankic_summary.csv", index=False, encoding="utf-8-sig")
    bucket_state_rankic_df.to_csv(output_dir / "bucket_state_rankic_summary.csv", index=False, encoding="utf-8-sig")

    spread_bucket_df = _top_bottom_spread(pred_df, "pred_fwd_excess_20", "true_fwd_excess_20", ["liquidity_bucket"])
    spread_bucket_state_df = _top_bottom_spread(pred_df, "pred_fwd_excess_20", "true_fwd_excess_20", ["state_name", "liquidity_bucket"])
    spread_bucket_df.to_csv(output_dir / "bucket_top_bottom_spread_20.csv", index=False, encoding="utf-8-sig")
    spread_bucket_state_df.to_csv(output_dir / "bucket_state_top_bottom_spread_20.csv", index=False, encoding="utf-8-sig")

    diagnosis: dict[str, object] = {"run_dir": str(run_dir)}
    if not bucket_rankic_df.empty:
        target20 = bucket_rankic_df[bucket_rankic_df["target"] == "fwd_excess_20"].sort_values("rankic_mean")
        if not target20.empty:
            diagnosis["worst_bucket_fwd20"] = target20.iloc[0].to_dict()
            diagnosis["best_bucket_fwd20"] = target20.iloc[-1].to_dict()
    if not bucket_state_rankic_df.empty:
        target20_state = bucket_state_rankic_df[bucket_state_rankic_df["target"] == "fwd_excess_20"].sort_values("rankic_mean")
        if not target20_state.empty:
            diagnosis["worst_bucket_state_fwd20"] = target20_state.iloc[0].to_dict()
            diagnosis["best_bucket_state_fwd20"] = target20_state.iloc[-1].to_dict()

    with (output_dir / "diagnosis.json").open("w", encoding="utf-8") as fh:
        json.dump(diagnosis, fh, ensure_ascii=False, indent=2)

    print(f"[OK] Liquidity failure analysis written to: {output_dir}")
    print(json.dumps(diagnosis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
