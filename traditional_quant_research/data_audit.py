"""Data and label audit helpers for v2 PIT research snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .dataset_v2 import load_daily_universe, load_pit_daily_bars, load_pit_manifest, load_quality_report, load_tradeable_panel
from .research_panel import panel_summary


DEFAULT_LABEL_HORIZONS = (1, 5, 20)
DEFAULT_EXTREME_ABS_RETURNS = (0.1, 0.2, 0.5)


@dataclass(frozen=True)
class V2DataLabelAudit:
    summary: dict[str, Any]
    universe_yearly: pd.DataFrame
    reject_reason_counts: pd.DataFrame
    bar_quality: pd.DataFrame
    label_summary: pd.DataFrame
    label_yearly: pd.DataFrame
    extreme_label_samples: pd.DataFrame


def audit_v2_data_labels(
    root: str | None = None,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    horizons: Sequence[int] = DEFAULT_LABEL_HORIZONS,
    extreme_abs_returns: Sequence[float] = DEFAULT_EXTREME_ABS_RETURNS,
    sample_size: int = 200,
) -> V2DataLabelAudit:
    """Audit v2 universe, OHLCV bars, and forward-return labels."""

    if any(horizon <= 0 for horizon in horizons):
        raise ValueError("horizons must be positive")
    if any(threshold <= 0 for threshold in extreme_abs_returns):
        raise ValueError("extreme_abs_returns thresholds must be positive")

    manifest = load_pit_manifest(root)
    quality_report = load_quality_report(root)
    universe = load_daily_universe(root, start_date=start_date, end_date=end_date)
    bars = load_pit_daily_bars(root, start_date=start_date, end_date=end_date)
    tradeable_panel = load_tradeable_panel(root, start_date=start_date, end_date=end_date)

    universe_yearly = audit_universe_by_year(universe)
    reject_reason_counts = audit_reject_reasons(universe)
    bar_quality = audit_bar_quality(bars)
    label_panel = add_forward_return_labels(tradeable_panel, horizons=horizons)
    label_summary = audit_label_summary(label_panel, horizons=horizons, extreme_abs_returns=extreme_abs_returns)
    label_yearly = audit_label_by_year(label_panel, horizons=horizons, extreme_abs_returns=extreme_abs_returns)
    extreme_label_samples = collect_extreme_label_samples(
        label_panel,
        horizons=horizons,
        threshold=max(extreme_abs_returns),
        sample_size=sample_size,
    )

    summary = {
        "snapshot_id": manifest.get("snapshot_id"),
        "snapshot_path": manifest.get("snapshot_path"),
        "start_date": start_date or manifest.get("dataset", {}).get("date_min"),
        "end_date": end_date or manifest.get("dataset", {}).get("date_max"),
        "manifest_dataset": manifest.get("dataset", {}),
        "manifest_quality": manifest.get("quality", {}),
        "quality_report": {
            "failure_count": quality_report.get("failure_count"),
            "missing_bar_rows": quality_report.get("missing_bar_rows"),
            "missing_basic_rows": quality_report.get("missing_basic_rows"),
            "st_rows": quality_report.get("st_rows"),
            "suspended_like_rows": quality_report.get("suspended_like_rows"),
        },
        "universe": panel_summary(universe),
        "bars": panel_summary(bars),
        "tradeable_panel": panel_summary(tradeable_panel),
        "bar_quality_flags": summarize_bar_quality_flags(bar_quality),
        "label_quality_flags": summarize_label_quality_flags(label_summary),
        "label_convention": "Signals at t use the next tradeable open as entry and the horizon-th next tradeable close as exit per code.",
    }
    return V2DataLabelAudit(
        summary=summary,
        universe_yearly=universe_yearly,
        reject_reason_counts=reject_reason_counts,
        bar_quality=bar_quality,
        label_summary=label_summary,
        label_yearly=label_yearly,
        extreme_label_samples=extreme_label_samples,
    )


def audit_universe_by_year(universe: pd.DataFrame) -> pd.DataFrame:
    if universe.empty:
        return pd.DataFrame(columns=["year", "rows", "dates", "securities", "tradeable_rows", "st_rows", "suspended_rows", "missing_bar_rows"])
    frame = universe.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["year"] = frame["date"].dt.year
    return (
        frame.groupby("year", sort=True)
        .agg(
            rows=("code", "size"),
            dates=("date", "nunique"),
            securities=("code", "nunique"),
            tradeable_rows=("is_tradeable", "sum"),
            st_rows=("is_st_on_date", "sum"),
            suspended_rows=("is_suspended_on_date", "sum"),
            missing_bar_rows=("has_bar", lambda values: int((~values.astype(bool)).sum())),
        )
        .reset_index()
    )


def audit_reject_reasons(universe: pd.DataFrame) -> pd.DataFrame:
    if universe.empty or "reject_reason" not in universe.columns:
        return pd.DataFrame(columns=["reject_reason", "rows"])
    reasons = universe["reject_reason"].fillna("").replace("", "ok")
    return reasons.value_counts(dropna=False).rename_axis("reject_reason").reset_index(name="rows")


def audit_bar_quality(bars: pd.DataFrame) -> pd.DataFrame:
    """Return one-row OHLCV quality counts for a bars table."""

    if bars.empty:
        return pd.DataFrame([{"rows": 0}])
    frame = bars.copy()
    numeric_columns = ["open", "high", "low", "close", "volume", "amount"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    checks = {
        "rows": len(frame),
        "dates": pd.to_datetime(frame["date"]).nunique() if "date" in frame.columns else 0,
        "securities": frame["code"].nunique() if "code" in frame.columns else 0,
    }
    for column in numeric_columns:
        checks[f"{column}_null_rows"] = int(frame[column].isna().sum())
    checks.update(
        {
            "nonpositive_open_rows": int((frame["open"] <= 0).sum()),
            "nonpositive_high_rows": int((frame["high"] <= 0).sum()),
            "nonpositive_low_rows": int((frame["low"] <= 0).sum()),
            "nonpositive_close_rows": int((frame["close"] <= 0).sum()),
            "negative_volume_rows": int((frame["volume"] < 0).sum()),
            "negative_amount_rows": int((frame["amount"] < 0).sum()),
            "zero_volume_rows": int((frame["volume"] == 0).sum()),
            "zero_amount_rows": int((frame["amount"] == 0).sum()),
            "high_below_low_rows": int((frame["high"] < frame["low"]).sum()),
            "high_below_open_or_close_rows": int((frame["high"] < frame[["open", "close"]].max(axis=1)).sum()),
            "low_above_open_or_close_rows": int((frame["low"] > frame[["open", "close"]].min(axis=1)).sum()),
        }
    )
    return pd.DataFrame([checks])


def add_forward_return_labels(panel: pd.DataFrame, *, horizons: Sequence[int] = DEFAULT_LABEL_HORIZONS) -> pd.DataFrame:
    """Add executable forward return labels to a tradeable panel."""

    if panel.empty:
        return panel.copy()
    missing = [column for column in ["date", "code", "open", "close"] if column not in panel.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    output = panel.copy()
    output["date"] = pd.to_datetime(output["date"])
    output = output.sort_values(["code", "date"]).reset_index(drop=True)
    output["open"] = pd.to_numeric(output["open"], errors="coerce")
    output["close"] = pd.to_numeric(output["close"], errors="coerce")
    grouped = output.groupby("code", sort=False)
    next_open = grouped["open"].shift(-1)
    for horizon in horizons:
        future_close = grouped["close"].shift(-horizon)
        output[f"fwd_ret_{horizon}d"] = (future_close / next_open - 1.0).replace([np.inf, -np.inf], np.nan)
    return output.sort_values(["date", "code"]).reset_index(drop=True)


def audit_label_summary(
    label_panel: pd.DataFrame,
    *,
    horizons: Sequence[int] = DEFAULT_LABEL_HORIZONS,
    extreme_abs_returns: Sequence[float] = DEFAULT_EXTREME_ABS_RETURNS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total_rows = int(len(label_panel))
    for horizon in horizons:
        column = f"fwd_ret_{horizon}d"
        if column not in label_panel.columns:
            raise ValueError(f"missing label column: {column}")
        values = pd.to_numeric(label_panel[column], errors="coerce")
        row: dict[str, Any] = {
            "label": column,
            "rows": total_rows,
            "available_rows": int(values.notna().sum()),
            "missing_rows": int(values.isna().sum()),
            "available_rate": float(values.notna().mean()) if total_rows else np.nan,
            "mean": float(values.mean()) if values.notna().any() else np.nan,
            "std": float(values.std(ddof=0)) if values.notna().any() else np.nan,
            "min": float(values.min()) if values.notna().any() else np.nan,
            "p01": float(values.quantile(0.01)) if values.notna().any() else np.nan,
            "p50": float(values.quantile(0.50)) if values.notna().any() else np.nan,
            "p99": float(values.quantile(0.99)) if values.notna().any() else np.nan,
            "max": float(values.max()) if values.notna().any() else np.nan,
        }
        for threshold in extreme_abs_returns:
            row[f"abs_gt_{threshold:g}_rows"] = int((values.abs() > threshold).sum())
            row[f"abs_gt_{threshold:g}_rate"] = float((values.abs() > threshold).mean()) if total_rows else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def audit_label_by_year(
    label_panel: pd.DataFrame,
    *,
    horizons: Sequence[int] = DEFAULT_LABEL_HORIZONS,
    extreme_abs_returns: Sequence[float] = DEFAULT_EXTREME_ABS_RETURNS,
) -> pd.DataFrame:
    if label_panel.empty:
        return pd.DataFrame()
    frame = label_panel.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["year"] = frame["date"].dt.year
    rows: list[dict[str, Any]] = []
    max_threshold = max(extreme_abs_returns)
    for year, group in frame.groupby("year", sort=True):
        for horizon in horizons:
            column = f"fwd_ret_{horizon}d"
            values = pd.to_numeric(group[column], errors="coerce")
            rows.append(
                {
                    "year": int(year),
                    "label": column,
                    "rows": int(len(group)),
                    "available_rows": int(values.notna().sum()),
                    "available_rate": float(values.notna().mean()) if len(group) else np.nan,
                    "mean": float(values.mean()) if values.notna().any() else np.nan,
                    "std": float(values.std(ddof=0)) if values.notna().any() else np.nan,
                    "p01": float(values.quantile(0.01)) if values.notna().any() else np.nan,
                    "p99": float(values.quantile(0.99)) if values.notna().any() else np.nan,
                    f"abs_gt_{max_threshold:g}_rows": int((values.abs() > max_threshold).sum()),
                }
            )
    return pd.DataFrame(rows)


def collect_extreme_label_samples(
    label_panel: pd.DataFrame,
    *,
    horizons: Sequence[int] = DEFAULT_LABEL_HORIZONS,
    threshold: float = 0.5,
    sample_size: int = 200,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    base_columns = [column for column in ["date", "code", "name_on_date", "open", "close", "volume", "amount"] if column in label_panel.columns]
    for horizon in horizons:
        column = f"fwd_ret_{horizon}d"
        if column not in label_panel.columns:
            continue
        frame = label_panel.loc[label_panel[column].abs() > threshold, [*base_columns, column]].copy()
        if frame.empty:
            continue
        frame["label"] = column
        frame["abs_return"] = frame[column].abs()
        frame = frame.sort_values("abs_return", ascending=False).head(sample_size)
        rows.append(frame)
    if not rows:
        return pd.DataFrame(columns=[*base_columns, "label", "abs_return"])
    return pd.concat(rows, ignore_index=True)


def summarize_bar_quality_flags(bar_quality: pd.DataFrame) -> dict[str, Any]:
    if bar_quality.empty:
        return {}
    row = bar_quality.iloc[0].to_dict()
    return {
        key: int(value)
        for key, value in row.items()
        if key.endswith("_rows") and pd.notna(value) and int(value) != 0
    }


def summarize_label_quality_flags(label_summary: pd.DataFrame) -> dict[str, Any]:
    flags: dict[str, Any] = {}
    for _, row in label_summary.iterrows():
        label = str(row["label"])
        flags[label] = {
            "available_rate": float(row["available_rate"]),
            "abs_gt_0.2_rows": int(row.get("abs_gt_0.2_rows", 0)),
            "abs_gt_0.5_rows": int(row.get("abs_gt_0.5_rows", 0)),
        }
    return flags
