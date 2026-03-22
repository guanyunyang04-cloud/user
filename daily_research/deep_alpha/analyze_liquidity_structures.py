from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from daily_research.baseline.data_provider import load_daily_from_tq
from daily_research.deep_alpha.cache_utils import cache_key, get_cache_root, load_pickle, save_pickle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze which structures are strengthened in top-liquidity vs other groups.")
    parser.add_argument("--base-run-dir", required=True)
    parser.add_argument("--cond-run-dir", required=True)
    parser.add_argument("--output-subdir", default="liquidity_structure_analysis")
    return parser.parse_args()


def _safe_rankcorr(group: pd.DataFrame, left: str, right: str) -> float:
    sample = group[[left, right]].dropna()
    if len(sample) < 8 or sample[left].nunique() <= 1 or sample[right].nunique() <= 1:
        return np.nan
    return float(sample[left].corr(sample[right], method="spearman"))


def _top_bottom_spread(group: pd.DataFrame, score_col: str, true_col: str, top_frac: float = 0.2) -> float:
    sample = group[[score_col, true_col]].dropna()
    if len(sample) < 12 or sample[score_col].nunique() < 6:
        return np.nan
    top_n = max(1, int(np.ceil(len(sample) * float(top_frac))))
    return float(
        sample.nlargest(top_n, score_col)[true_col].mean()
        - sample.nsmallest(top_n, score_col)[true_col].mean()
    )


def _load_run_meta(run_dir: Path) -> tuple[dict, pd.DataFrame]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    pred_df = pd.read_csv(run_dir / "validation_predictions.csv", encoding="utf-8-sig")
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    return metrics, pred_df


def _load_raw_frames(run_dir: Path, pred_df: pd.DataFrame, metrics: dict) -> dict[str, pd.DataFrame]:
    model_artifact = torch.load(run_dir / "deep_alpha_model.pt", map_location="cpu", weights_only=False)
    cfg = model_artifact["config"]
    stocks = sorted(pred_df["stock"].dropna().astype(str).unique().tolist())
    end_date = pd.Timestamp(pred_df["date"].max()).strftime("%Y%m%d")
    raw_meta = {
        "version": 1,
        "data_source": "tq",
        "universe_scope": cfg.get("universe_scope", "custom"),
        "benchmark": cfg.get("benchmark", metrics.get("benchmark", "000300.SH")),
        "start_date": cfg.get("start_date", "20240101"),
        "end_date": end_date,
        "stocks_count": len(stocks),
        "stocks_hash": cache_key({"stocks": stocks}),
    }
    raw_key = cache_key(raw_meta)
    raw_path = get_cache_root() / "raw" / f"{raw_key}.pkl"
    cached = load_pickle(raw_path)
    if cached is not None:
        return cached
    raw = load_daily_from_tq(stocks, raw_meta["start_date"], raw_meta["end_date"], benchmark=raw_meta["benchmark"])
    save_pickle(raw_path, raw)
    return raw


def _build_structure_frame(raw_df_dict: dict[str, pd.DataFrame], benchmark: str) -> pd.DataFrame:
    close = raw_df_dict["Close"].drop(columns=[benchmark], errors="ignore").astype(float)
    open_df = raw_df_dict["Open"].drop(columns=[benchmark], errors="ignore").astype(float)
    high = raw_df_dict["High"].drop(columns=[benchmark], errors="ignore").astype(float)
    low = raw_df_dict["Low"].drop(columns=[benchmark], errors="ignore").astype(float)
    amount = raw_df_dict["Amount"].drop(columns=[benchmark], errors="ignore").astype(float)
    benchmark_close = raw_df_dict["Close"][benchmark].astype(float).reindex(close.index)

    ret_20 = close.pct_change(20, fill_method=None)
    rel_ret_5 = close.pct_change(5, fill_method=None).sub(benchmark_close.pct_change(5, fill_method=None), axis=0)
    ma_20_gap = close.div(close.rolling(20).mean()).sub(1.0)
    ma_60_gap = close.div(close.rolling(60).mean()).sub(1.0)
    amount_ratio_5_20 = amount.rolling(5).mean().div(amount.rolling(20).mean().replace(0, np.nan))
    prev_close = close.shift(1)
    range_pct = high.sub(low).div(prev_close.replace(0, np.nan))
    intraday_body = close.div(open_df.replace(0, np.nan)).sub(1.0)

    high_vol_cut = range_pct.quantile(0.8, axis=1)
    low_vol_cut = range_pct.quantile(0.5, axis=1)

    records: list[dict[str, object]] = []
    for dt in close.index:
        dt_ts = pd.Timestamp(dt)
        for stock in close.columns:
            r20 = ret_20.at[dt_ts, stock]
            rel5 = rel_ret_5.at[dt_ts, stock]
            g20 = ma_20_gap.at[dt_ts, stock]
            g60 = ma_60_gap.at[dt_ts, stock]
            ar = amount_ratio_5_20.at[dt_ts, stock]
            rng = range_pct.at[dt_ts, stock]
            body = intraday_body.at[dt_ts, stock]
            if not np.isfinite([r20, rel5, g20, g60, ar, rng, body]).all():
                continue
            if (r20 > 0.05) and (g20 > 0.03) and (ar > 1.0):
                structure = "trend_breakout"
            elif (g60 > 0.0) and (rel5 < -0.01) and (body > 0.0):
                structure = "pullback_rebound"
            elif (rng >= float(high_vol_cut.loc[dt_ts])) and (ar > 1.0):
                structure = "high_vol_expansion"
            elif (r20 > 0.03) and (g20 > 0.0) and (rng <= float(low_vol_cut.loc[dt_ts])):
                structure = "low_vol_trend"
            elif (g20 < 0.0) or (r20 < 0.0):
                structure = "weak_structure"
            else:
                structure = "neutral_mixed"
            records.append(
                {
                    "date": dt_ts,
                    "stock": stock,
                    "structure_label": structure,
                    "ret_20": float(r20),
                    "rel_ret_5": float(rel5),
                    "ma_20_gap": float(g20),
                    "ma_60_gap": float(g60),
                    "amount_ratio_5_20": float(ar),
                    "range_pct": float(rng),
                    "intraday_body": float(body),
                }
            )
    return pd.DataFrame(records)


def _summarize(pred_df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (liq_group, structure), group in pred_df.groupby(["liquidity_group", "structure_label"], dropna=False):
        daily_rankics = []
        daily_spreads = []
        for _dt, dt_group in group.groupby("date", dropna=False):
            daily_rankics.append(_safe_rankcorr(dt_group, score_col, "true_fwd_excess_20"))
            daily_spreads.append(_top_bottom_spread(dt_group, score_col, "true_fwd_excess_20"))
        rankic_mean = float(pd.Series(daily_rankics).dropna().mean()) if pd.Series(daily_rankics).dropna().size else np.nan
        spread_mean = float(pd.Series(daily_spreads).dropna().mean()) if pd.Series(daily_spreads).dropna().size else np.nan
        rows.append(
            {
                "liquidity_group": liq_group,
                "structure_label": structure,
                "rankic_mean": rankic_mean,
                "spread_mean": spread_mean,
                "sample_count": int(len(group)),
                "date_count": int(group["date"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    base_run_dir = Path(args.base_run_dir)
    cond_run_dir = Path(args.cond_run_dir)
    output_dir = cond_run_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    base_metrics, base_pred = _load_run_meta(base_run_dir)
    cond_metrics, cond_pred = _load_run_meta(cond_run_dir)
    benchmark = base_metrics.get("benchmark", "000300.SH")

    raw_df_dict = _load_raw_frames(base_run_dir, base_pred, base_metrics)
    structure_df = _build_structure_frame(raw_df_dict, benchmark=benchmark)

    for df in (base_pred, cond_pred):
        df["liquidity_group"] = np.where(df["liquidity_bucket"].fillna(-1).astype(int) >= 4, "top_liquidity", "other")
        df["score_20"] = df["pred_fwd_excess_20"]

    base_joined = base_pred.merge(structure_df, on=["date", "stock"], how="left")
    cond_joined = cond_pred.merge(structure_df, on=["date", "stock"], how="left")

    base_summary = _summarize(base_joined, "score_20").rename(
        columns={"rankic_mean": "base_rankic_mean", "spread_mean": "base_spread_mean"}
    )
    cond_summary = _summarize(cond_joined, "score_20").rename(
        columns={"rankic_mean": "cond_rankic_mean", "spread_mean": "cond_spread_mean"}
    )

    merged = base_summary.merge(
        cond_summary[["liquidity_group", "structure_label", "cond_rankic_mean", "cond_spread_mean"]],
        on=["liquidity_group", "structure_label"],
        how="outer",
    )
    merged["rankic_delta"] = merged["cond_rankic_mean"] - merged["base_rankic_mean"]
    merged["spread_delta"] = merged["cond_spread_mean"] - merged["base_spread_mean"]
    merged = merged.sort_values(["liquidity_group", "rankic_delta", "spread_delta"], ascending=[True, False, False])

    merged.to_csv(output_dir / "structure_delta_summary.csv", index=False, encoding="utf-8-sig")
    structure_df.to_csv(output_dir / "structure_snapshot.csv", index=False, encoding="utf-8-sig")

    diagnosis = {
        "base_run_dir": str(base_run_dir),
        "cond_run_dir": str(cond_run_dir),
        "top_liquidity_best_delta": (
            merged[merged["liquidity_group"] == "top_liquidity"].head(5).to_dict(orient="records")
            if not merged.empty
            else []
        ),
        "other_worst_delta": (
            merged[merged["liquidity_group"] == "other"].sort_values("rankic_delta").head(5).to_dict(orient="records")
            if not merged.empty
            else []
        ),
    }
    (output_dir / "diagnosis.json").write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] structure analysis written to: {output_dir}")
    print(json.dumps(diagnosis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
