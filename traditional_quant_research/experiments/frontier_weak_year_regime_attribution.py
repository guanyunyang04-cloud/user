"""Attribute frontier weak years to broad market regime diagnostics."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from traditional_quant_research.dataset_v2 import load_pit_manifest, load_tradeable_panel
from traditional_quant_research.experiments.frontier_promotion_gate import latest_run_dir
from traditional_quant_research.experiments.low_corr_regime_filter import build_market_regime_frame
from traditional_quant_research.research_panel import build_factor_label_panel


DEFAULT_FAILURE_ATTRIBUTION_ROOT = Path("traditional_quant_research/output/experiments/frontier_failure_attribution")
DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/frontier_weak_year_regime_attribution")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/research_log/2026-06-03_frontier_weak_year_regime_attribution.md")

REGIME_METRIC_COLUMNS = (
    "market_ret_1d_mean",
    "market_ret_5d_mean",
    "market_ret_20d_mean",
    "breadth_1d_positive_rate",
    "breadth_5d_positive_rate",
    "breadth_20d_positive_rate",
    "market_volatility_20d_mean",
    "market_amplitude_20d_mean",
)


def run_frontier_weak_year_regime_attribution(
    *,
    failure_run_dir: str | Path | None = None,
    root: str | Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: str | Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    """Compare market regime diagnostics in frontier weak years vs non-weak years."""

    source_dir = Path(failure_run_dir) if failure_run_dir is not None else latest_run_dir(DEFAULT_FAILURE_ATTRIBUTION_ROOT)
    yearly_failure = read_yearly_failure_attribution(source_dir)
    manifest = load_pit_manifest(root)
    effective_start, effective_end = resolve_effective_dates(
        yearly_failure,
        manifest,
        start_date=start_date,
        end_date=end_date,
    )

    run_id = f"frontier_weak_year_regime_attribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    panel = load_tradeable_panel(root, start_date=effective_start, end_date=effective_end)
    factor_panel = build_factor_label_panel(panel, horizons=(1, 5, 20))
    daily_regime = build_market_regime_frame(factor_panel)
    yearly_regime = summarize_yearly_market_regime(daily_regime)
    weak_profile = build_weak_year_regime_profile(yearly_failure, yearly_regime)
    signal_year_regime = join_signal_year_regime(yearly_failure, yearly_regime)
    weak_vs_positive = summarize_weak_vs_positive_regime(signal_year_regime)
    result = summarize_regime_attribution(
        yearly_failure,
        yearly_regime,
        weak_profile,
        weak_vs_positive,
        run_id=run_id,
        failure_run_dir=source_dir,
        manifest=manifest,
        start_date=effective_start,
        end_date=effective_end,
    )
    markdown = render_regime_attribution_markdown(result, weak_profile, weak_vs_positive)

    daily_regime.to_csv(run_dir / "market_regime_daily.csv", index=False, encoding="utf-8-sig")
    yearly_regime.to_csv(run_dir / "yearly_market_regime.csv", index=False, encoding="utf-8-sig")
    weak_profile.to_csv(run_dir / "weak_year_regime_profile.csv", index=False, encoding="utf-8-sig")
    signal_year_regime.to_csv(run_dir / "signal_year_regime_attribution.csv", index=False, encoding="utf-8-sig")
    weak_vs_positive.to_csv(run_dir / "weak_vs_positive_regime_summary.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        path = Path(research_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")

    return {**result, "run_dir": str(run_dir), "research_log": str(research_log_path) if write_research_log else None}


def read_yearly_failure_attribution(run_dir: str | Path) -> pd.DataFrame:
    """Read the failure-attribution table that marks weak signal-years."""

    path = Path(run_dir) / "yearly_failure_attribution.csv"
    if not path.exists():
        raise FileNotFoundError(f"yearly failure attribution not found: {path}")
    frame = pd.read_csv(path)
    required = {"eval_year", "signal", "annualized_return", "weak_year"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"yearly failure attribution missing required columns: {missing}")
    frame = frame.copy()
    frame["eval_year"] = pd.to_numeric(frame["eval_year"], errors="coerce").astype("Int64")
    frame["annualized_return"] = pd.to_numeric(frame["annualized_return"], errors="coerce")
    frame["weak_year"] = _to_bool(frame["weak_year"])
    if "exposure_penalty_strength" in frame.columns:
        frame["exposure_penalty_strength"] = pd.to_numeric(frame["exposure_penalty_strength"], errors="coerce")
    else:
        frame["exposure_penalty_strength"] = 0.0
    return frame.dropna(subset=["eval_year"]).reset_index(drop=True)


def resolve_effective_dates(
    yearly_failure: pd.DataFrame,
    manifest: Mapping[str, Any],
    *,
    start_date: str | None,
    end_date: str | None,
) -> tuple[str, str]:
    """Resolve the panel date span from requested dates, failure years, and manifest bounds."""

    if yearly_failure.empty:
        dataset = manifest.get("dataset", {})
        effective_start = start_date or dataset.get("date_min")
        effective_end = end_date or dataset.get("date_max")
    else:
        years = pd.to_numeric(yearly_failure["eval_year"], errors="coerce").dropna().astype(int)
        effective_start = start_date or f"{int(years.min())}-01-01"
        effective_end = end_date or f"{int(years.max())}-12-31"
        dataset = manifest.get("dataset", {})
        date_min = dataset.get("date_min")
        date_max = dataset.get("date_max")
        if date_min:
            effective_start = max(pd.Timestamp(effective_start), pd.Timestamp(date_min)).date().isoformat()
        if date_max:
            effective_end = min(pd.Timestamp(effective_end), pd.Timestamp(date_max)).date().isoformat()
    if not effective_start or not effective_end:
        raise ValueError("could not resolve effective start/end dates")
    return str(effective_start), str(effective_end)


def summarize_yearly_market_regime(daily_regime: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily market-regime diagnostics to one row per calendar year."""

    columns = [
        "eval_year",
        "trade_date_count",
        "security_count_mean",
        "security_count_min",
        "security_count_max",
        *REGIME_METRIC_COLUMNS,
    ]
    if daily_regime.empty:
        return pd.DataFrame(columns=columns)
    required = {"date", "security_count", *REGIME_METRIC_COLUMNS}
    if missing := sorted(required - set(daily_regime.columns)):
        raise ValueError(f"daily_regime missing required columns: {missing}")

    frame = daily_regime.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["eval_year"] = frame["date"].dt.year.astype(int)
    for column in ["security_count", *REGIME_METRIC_COLUMNS]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    rows: list[dict[str, Any]] = []
    for year, group in frame.groupby("eval_year", sort=True):
        row: dict[str, Any] = {
            "eval_year": int(year),
            "trade_date_count": int(group["date"].nunique()),
            "security_count_mean": float(group["security_count"].mean()),
            "security_count_min": int(group["security_count"].min()),
            "security_count_max": int(group["security_count"].max()),
        }
        for metric in REGIME_METRIC_COLUMNS:
            row[metric] = float(group[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def build_weak_year_regime_profile(yearly_failure: pd.DataFrame, yearly_regime: pd.DataFrame) -> pd.DataFrame:
    """Summarize weak-signal concentration by year and attach yearly regime values."""

    columns = [
        "eval_year",
        "signal_count",
        "weak_signal_count",
        "all_signals_weak",
        "mean_annualized_return",
        "min_annualized_return",
        "max_annualized_return",
        *yearly_regime_columns(),
    ]
    if yearly_failure.empty:
        return pd.DataFrame(columns=columns)
    frame = yearly_failure.copy()
    frame["weak_year"] = _to_bool(frame["weak_year"])
    frame["annualized_return"] = pd.to_numeric(frame["annualized_return"], errors="coerce")

    rows: list[dict[str, Any]] = []
    for year, group in frame.groupby("eval_year", sort=True):
        weak_count = int(group["weak_year"].sum())
        signal_count = int(group["signal"].nunique())
        rows.append(
            {
                "eval_year": int(year),
                "signal_count": signal_count,
                "weak_signal_count": weak_count,
                "all_signals_weak": bool(signal_count > 0 and weak_count == signal_count),
                "mean_annualized_return": float(group["annualized_return"].mean()),
                "min_annualized_return": float(group["annualized_return"].min()),
                "max_annualized_return": float(group["annualized_return"].max()),
            }
        )
    profile = pd.DataFrame(rows)
    if not yearly_regime.empty:
        profile = profile.merge(yearly_regime, on="eval_year", how="left")
    return profile.reindex(columns=columns).sort_values("eval_year").reset_index(drop=True)


def join_signal_year_regime(yearly_failure: pd.DataFrame, yearly_regime: pd.DataFrame) -> pd.DataFrame:
    """Attach yearly market-regime diagnostics to each signal-year failure row."""

    if yearly_failure.empty:
        return pd.DataFrame()
    joined = yearly_failure.merge(yearly_regime, on="eval_year", how="left")
    return joined.sort_values(["eval_year", "signal"]).reset_index(drop=True)


def summarize_weak_vs_positive_regime(
    signal_year_regime: pd.DataFrame,
    *,
    metrics: Sequence[str] = REGIME_METRIC_COLUMNS,
    include_pooled: bool = True,
) -> pd.DataFrame:
    """Compare yearly regime values for weak and non-weak signal-years."""

    columns = [
        "signal",
        "exposure_penalty_strength",
        "metric",
        "weak_year_count",
        "positive_year_count",
        "weak_mean",
        "positive_mean",
        "delta_weak_minus_positive",
        "abs_delta_weak_minus_positive",
        "diagnosis",
    ]
    if signal_year_regime.empty:
        return pd.DataFrame(columns=columns)

    frame = signal_year_regime.copy()
    frame["weak_year"] = _to_bool(frame["weak_year"])
    if "exposure_penalty_strength" not in frame.columns:
        frame["exposure_penalty_strength"] = 0.0
    frame["exposure_penalty_strength"] = pd.to_numeric(frame["exposure_penalty_strength"], errors="coerce").fillna(0.0)
    for metric in metrics:
        if metric in frame.columns:
            frame[metric] = pd.to_numeric(frame[metric], errors="coerce")

    rows: list[dict[str, Any]] = []
    groups: list[tuple[tuple[Any, ...], pd.DataFrame]] = list(frame.groupby(["signal", "exposure_penalty_strength"], sort=True))
    if include_pooled:
        groups.append((("__pooled__", 0.0), frame))
    for keys, group in groups:
        signal, strength = keys
        weak = group.loc[group["weak_year"]]
        positive = group.loc[~group["weak_year"]]
        for metric in metrics:
            if metric not in group.columns:
                continue
            weak_mean = _mean_or_nan(weak, metric)
            positive_mean = _mean_or_nan(positive, metric)
            delta = weak_mean - positive_mean
            rows.append(
                {
                    "signal": str(signal),
                    "exposure_penalty_strength": float(strength),
                    "metric": metric,
                    "weak_year_count": int(len(weak)),
                    "positive_year_count": int(len(positive)),
                    "weak_mean": weak_mean,
                    "positive_mean": positive_mean,
                    "delta_weak_minus_positive": delta,
                    "abs_delta_weak_minus_positive": abs(delta) if not np.isnan(delta) else np.nan,
                    "diagnosis": _metric_delta_diagnosis(metric, delta),
                }
            )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["signal", "abs_delta_weak_minus_positive"],
        ascending=[True, False],
    ).reset_index(drop=True)


def summarize_regime_attribution(
    yearly_failure: pd.DataFrame,
    yearly_regime: pd.DataFrame,
    weak_profile: pd.DataFrame,
    weak_vs_positive: pd.DataFrame,
    *,
    run_id: str,
    failure_run_dir: Path,
    manifest: Mapping[str, Any],
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    """Build a compact run summary."""

    if yearly_failure.empty:
        return {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "empty",
            "failure_run_dir": str(failure_run_dir),
            "candidate_count": 0,
            "decision": "no_failure_rows_to_attribute",
        }
    weak_signal_years = int(_to_bool(yearly_failure["weak_year"]).sum())
    common_weak_years = (
        weak_profile.loc[weak_profile["all_signals_weak"].fillna(False), "eval_year"].astype(int).astype(str).tolist()
        if not weak_profile.empty
        else []
    )
    pooled = weak_vs_positive.loc[weak_vs_positive["signal"] == "__pooled__"].copy()
    top_regime_deltas = []
    if not pooled.empty:
        pooled = pooled.sort_values("abs_delta_weak_minus_positive", ascending=False).head(5)
        top_regime_deltas = [
            {
                "metric": str(row["metric"]),
                "delta_weak_minus_positive": float(row["delta_weak_minus_positive"]),
                "diagnosis": str(row["diagnosis"]),
            }
            for row in pooled.to_dict("records")
        ]
    dataset = manifest.get("dataset", {})
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "ok",
        "failure_run_dir": str(failure_run_dir),
        "snapshot_id": manifest.get("snapshot_id", ""),
        "dataset_date_min": dataset.get("date_min"),
        "dataset_date_max": dataset.get("date_max"),
        "start_date": start_date,
        "end_date": end_date,
        "signal_count": int(yearly_failure["signal"].nunique()),
        "eval_year_count": int(yearly_failure["eval_year"].nunique()),
        "yearly_regime_count": int(len(yearly_regime)),
        "weak_signal_years": weak_signal_years,
        "common_weak_years": common_weak_years,
        "top_regime_deltas": top_regime_deltas,
        "candidate_count": 0,
        "decision": "keep_candidate_frontier_backtest_only_regime_rebuild_required",
        "limitations": [
            "This attribution reads existing failure rows and rebuilds only market-regime diagnostics; it does not rerun frontier backtests.",
            "Yearly regime deltas are descriptive and do not prove causality.",
            "The regime frame uses broad tradeable-universe price/volume diagnostics, not external macro or index data.",
        ],
    }


def render_regime_attribution_markdown(
    result: Mapping[str, Any],
    weak_profile: pd.DataFrame,
    weak_vs_positive: pd.DataFrame,
) -> str:
    """Render a research-log friendly summary."""

    pooled = weak_vs_positive.loc[weak_vs_positive["signal"] == "__pooled__"] if not weak_vs_positive.empty else pd.DataFrame()
    lines = [
        "# Frontier Weak-Year Regime Attribution",
        "",
        f"- run_id: `{result.get('run_id', '')}`",
        f"- decision: `{result.get('decision', '')}`",
        f"- candidate_count: `{result.get('candidate_count', 0)}`",
        f"- failure_run_dir: `{result.get('failure_run_dir', '')}`",
        f"- snapshot_id: `{result.get('snapshot_id', '')}`",
        f"- sample: `{result.get('start_date', '')}` to `{result.get('end_date', '')}`",
        f"- weak_signal_years: `{result.get('weak_signal_years', 0)}`",
        f"- common_weak_years: `{','.join(result.get('common_weak_years', []))}`",
        "",
        "## Weak-Year Profile",
        "",
        _markdown_table(weak_profile),
        "",
        "## Pooled Weak Vs Positive Regime",
        "",
        _markdown_table(pooled),
        "",
        "## Signal-Level Weak Vs Positive Regime",
        "",
        _markdown_table(weak_vs_positive),
        "",
        "## Interpretation",
        "",
        "This report checks whether the frontier's weak years line up with broad market-state diagnostics. "
        "It is a failure-attribution layer, not a new strategy or a promotion gate. Any regime rule derived from these deltas still needs fit/eval separation and a rerun through the full combined-constraint gate.",
        "",
    ]
    return "\n".join(lines)


def yearly_regime_columns() -> list[str]:
    return [
        "trade_date_count",
        "security_count_mean",
        "security_count_min",
        "security_count_max",
        *REGIME_METRIC_COLUMNS,
    ]


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    values = series.astype(str).str.strip().str.lower()
    return values.isin({"true", "1", "yes", "y"})


def _mean_or_nan(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return np.nan
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def _metric_delta_diagnosis(metric: str, delta: float) -> str:
    if np.isnan(delta):
        return "insufficient_regime_rows"
    if metric.startswith("market_ret") or metric.startswith("breadth"):
        if delta < 0:
            return "weak_years_lower_market_strength"
        if delta > 0:
            return "weak_years_higher_market_strength"
    if "volatility" in metric or "amplitude" in metric:
        if delta > 0:
            return "weak_years_higher_volatility_or_amplitude"
        if delta < 0:
            return "weak_years_lower_volatility_or_amplitude"
    return "weak_positive_regime_delta"


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(_fmt)
    return view.to_markdown(index=False)


def _fmt(value: Any) -> str:
    if value is None:
        return "nan"
    try:
        if pd.isna(value):
            return "nan"
    except TypeError:
        pass
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-run-dir", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", type=Path, default=DEFAULT_RESEARCH_LOG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_frontier_weak_year_regime_attribution(
        failure_run_dir=args.failure_run_dir,
        root=args.root,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        write_research_log=bool(args.write_research_log),
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
