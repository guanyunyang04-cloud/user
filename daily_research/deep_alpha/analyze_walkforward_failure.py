from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from daily_research.deep_alpha.analyze_target_failure import _build_manual_score, _safe_spearman
from daily_research.deep_alpha.sequence_dataset import STRUCTURE_ID_TO_LABEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a weak and strong deep_alpha walk-forward window."
    )
    parser.add_argument("--weak-run-dir", required=True)
    parser.add_argument("--strong-run-dir", required=True)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where comparison CSV/JSON outputs will be written.",
    )
    return parser.parse_args()


def _load_run(run_dir: Path) -> tuple[dict, pd.DataFrame]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    pred_df = pd.read_csv(run_dir / "validation_predictions.csv", encoding="utf-8-sig")
    pred_df["date"] = pd.to_datetime(pred_df["date"])

    state_df = pd.read_csv(run_dir / "market_state_frame.csv", encoding="utf-8-sig")
    date_col = "Date" if "Date" in state_df.columns else "date"
    state_df[date_col] = pd.to_datetime(state_df[date_col])
    state_df = state_df.rename(columns={date_col: "date"})

    merged = pred_df.merge(state_df[["date", "state_name"]], on="date", how="left")
    merged["state_name"] = merged["state_name"].fillna("unknown")
    merged["structure_label"] = (
        merged["structure_id"].map(STRUCTURE_ID_TO_LABEL).fillna("unknown")
    )
    merged = _build_manual_score(merged, metrics)
    return metrics, merged


def _top_bottom_spread(group: pd.DataFrame, score_col: str, true_col: str) -> float:
    sample = group[[score_col, true_col]].dropna()
    if len(sample) < 10 or sample[score_col].nunique() < 5:
        return np.nan
    top_n = max(1, int(np.ceil(len(sample) * 0.2)))
    return float(
        sample.nlargest(top_n, score_col)[true_col].mean()
        - sample.nsmallest(top_n, score_col)[true_col].mean()
    )


def _summarize(df: pd.DataFrame, by_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in df.groupby(by_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(by_cols, keys)}

        score_rankics = []
        pred20_rankics = []
        score_spreads = []
        pred20_spreads = []
        for _date, dt_group in group.groupby("date", dropna=False):
            score_rankics.append(_safe_spearman(dt_group, "manual_score", "true_fwd_excess_20"))
            pred20_rankics.append(
                _safe_spearman(dt_group, "pred_fwd_excess_20", "true_fwd_excess_20")
            )
            score_spreads.append(_top_bottom_spread(dt_group, "manual_score", "true_fwd_excess_20"))
            pred20_spreads.append(
                _top_bottom_spread(dt_group, "pred_fwd_excess_20", "true_fwd_excess_20")
            )

        score_rankic_s = pd.Series(score_rankics, dtype=float).dropna()
        pred20_rankic_s = pd.Series(pred20_rankics, dtype=float).dropna()
        score_spread_s = pd.Series(score_spreads, dtype=float).dropna()
        pred20_spread_s = pd.Series(pred20_spreads, dtype=float).dropna()

        row.update(
            {
                "score_true20_rankic": float(score_rankic_s.mean()) if not score_rankic_s.empty else np.nan,
                "pred20_true20_rankic": float(pred20_rankic_s.mean()) if not pred20_rankic_s.empty else np.nan,
                "score_true20_spread": float(score_spread_s.mean()) if not score_spread_s.empty else np.nan,
                "pred20_true20_spread": float(pred20_spread_s.mean()) if not pred20_spread_s.empty else np.nan,
                "date_count": int(group["date"].nunique()),
                "sample_count": int(len(group)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _attach_window(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "window_name", name)
    return out


def _merge_delta(
    weak_df: pd.DataFrame,
    strong_df: pd.DataFrame,
    by_cols: list[str],
) -> pd.DataFrame:
    weak = weak_df.rename(
        columns={
            "score_true20_rankic": "weak_score_true20_rankic",
            "pred20_true20_rankic": "weak_pred20_true20_rankic",
            "score_true20_spread": "weak_score_true20_spread",
            "pred20_true20_spread": "weak_pred20_true20_spread",
            "date_count": "weak_date_count",
            "sample_count": "weak_sample_count",
        }
    )
    strong = strong_df.rename(
        columns={
            "score_true20_rankic": "strong_score_true20_rankic",
            "pred20_true20_rankic": "strong_pred20_true20_rankic",
            "score_true20_spread": "strong_score_true20_spread",
            "pred20_true20_spread": "strong_pred20_true20_spread",
            "date_count": "strong_date_count",
            "sample_count": "strong_sample_count",
        }
    )
    merged = weak.merge(strong, on=by_cols, how="outer")
    merged["score_rankic_delta"] = (
        merged["strong_score_true20_rankic"] - merged["weak_score_true20_rankic"]
    )
    merged["score_spread_delta"] = (
        merged["strong_score_true20_spread"] - merged["weak_score_true20_spread"]
    )
    merged["pred20_rankic_delta"] = (
        merged["strong_pred20_true20_rankic"] - merged["weak_pred20_true20_rankic"]
    )
    merged["pred20_spread_delta"] = (
        merged["strong_pred20_true20_spread"] - merged["weak_pred20_true20_spread"]
    )
    return merged


def main() -> None:
    args = parse_args()
    weak_run_dir = Path(args.weak_run_dir)
    strong_run_dir = Path(args.strong_run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weak_metrics, weak_df = _load_run(weak_run_dir)
    strong_metrics, strong_df = _load_run(strong_run_dir)

    state_weak = _summarize(weak_df, ["state_name"])
    state_strong = _summarize(strong_df, ["state_name"])
    state_compare = _merge_delta(state_weak, state_strong, ["state_name"]).sort_values(
        "score_rankic_delta"
    )

    structure_weak = _summarize(weak_df, ["structure_label"])
    structure_strong = _summarize(strong_df, ["structure_label"])
    structure_compare = _merge_delta(
        structure_weak, structure_strong, ["structure_label"]
    ).sort_values("score_rankic_delta")

    bucket_weak = _summarize(weak_df, ["liquidity_bucket"])
    bucket_strong = _summarize(strong_df, ["liquidity_bucket"])
    bucket_compare = _merge_delta(
        bucket_weak, bucket_strong, ["liquidity_bucket"]
    ).sort_values("score_rankic_delta")

    state_structure_weak = _summarize(weak_df, ["state_name", "structure_label"])
    state_structure_strong = _summarize(strong_df, ["state_name", "structure_label"])
    state_structure_compare = _merge_delta(
        state_structure_weak, state_structure_strong, ["state_name", "structure_label"]
    ).sort_values("score_rankic_delta")

    _attach_window(state_weak, "weak").to_csv(
        output_dir / "weak_state_summary.csv", index=False, encoding="utf-8-sig"
    )
    _attach_window(state_strong, "strong").to_csv(
        output_dir / "strong_state_summary.csv", index=False, encoding="utf-8-sig"
    )
    state_compare.to_csv(
        output_dir / "state_compare_summary.csv", index=False, encoding="utf-8-sig"
    )

    _attach_window(structure_weak, "weak").to_csv(
        output_dir / "weak_structure_summary.csv", index=False, encoding="utf-8-sig"
    )
    _attach_window(structure_strong, "strong").to_csv(
        output_dir / "strong_structure_summary.csv", index=False, encoding="utf-8-sig"
    )
    structure_compare.to_csv(
        output_dir / "structure_compare_summary.csv", index=False, encoding="utf-8-sig"
    )

    bucket_compare.to_csv(
        output_dir / "liquidity_bucket_compare_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    state_structure_compare.to_csv(
        output_dir / "state_structure_compare_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    diagnosis = {
        "weak_window": {
            "run_dir": str(weak_run_dir),
            "valid_start": weak_metrics.get("valid_start"),
            "valid_end": weak_metrics.get("valid_end"),
        },
        "strong_window": {
            "run_dir": str(strong_run_dir),
            "valid_start": strong_metrics.get("valid_start"),
            "valid_end": strong_metrics.get("valid_end"),
        },
        "worst_state_delta": state_compare.head(5).to_dict(orient="records"),
        "worst_structure_delta": structure_compare.head(5).to_dict(orient="records"),
        "worst_state_structure_delta": state_structure_compare.head(10).to_dict(
            orient="records"
        ),
        "bucket_delta": bucket_compare.to_dict(orient="records"),
    }
    (output_dir / "diagnosis.json").write_text(
        json.dumps(diagnosis, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[OK] walk-forward failure analysis written to: {output_dir}")
    print(json.dumps(diagnosis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
