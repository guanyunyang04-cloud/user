"""Run participation-based impact stress for current frontier signals."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from traditional_quant_research.experiments.low_corr_candidate_frontier_audit import (
    DEFAULT_BUFFER_MULTIPLIER,
    DEFAULT_FINAL_END_DATE,
    DEFAULT_HORIZON,
    DEFAULT_MAX_FACTOR_CORR,
    DEFAULT_REBALANCE_FREQUENCY,
    DEFAULT_TOP_N,
    DEFAULT_YEARS,
    LABEL_MODES,
    expand_selected_holdings,
    summarize_trade_liquidity,
)
from traditional_quant_research.experiments.low_corr_candidate_signal_comparison import (
    IC_WEIGHTED_SIGNAL,
    LOW_CORR_SIGNAL,
    ROLLING_IC_SIGNAL,
    build_candidate_protocol_signal_panel,
)
from traditional_quant_research.experiments.low_corr_regime_yearly_validation import year_windows
from traditional_quant_research.experiments.multifactor_baseline import summarize_horizon_trade_table
from traditional_quant_research.horizon_backtest import horizon_aligned_top_n_backtest


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/low_corr_frontier_impact_stress")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/brain/references/research_log/2026-06-02_low_corr_frontier_impact_stress.md")
DEFAULT_FRONTIER_SIGNALS = (
    ROLLING_IC_SIGNAL,
    IC_WEIGHTED_SIGNAL,
    LOW_CORR_SIGNAL,
)
DEFAULT_FEE_BPS_VALUES = (30.0,)
DEFAULT_CAPITAL_AMOUNTS = (10_000_000.0, 50_000_000.0, 100_000_000.0)
DEFAULT_IMPACT_BPS_PER_1PCT = (0.0, 1.0, 2.0, 5.0, 10.0)
DEFAULT_ROLLING_WINDOW = 252
DEFAULT_ROLLING_MIN_PERIODS = 60


def apply_fee_and_participation_impact(
    trades: pd.DataFrame,
    per_trade_liquidity: pd.DataFrame,
    *,
    fee_bps: float,
    capital_amount: float,
    impact_bps_per_1pct: float,
) -> pd.DataFrame:
    """Apply fixed fee plus participation-driven impact to an existing trade path."""

    output = trades.copy()
    if output.empty:
        return output
    required = {"trade_id", "gross_return", "turnover"}
    if missing := sorted(required - set(output.columns)):
        raise ValueError(f"trades missing required columns: {missing}")
    token = _capital_token(capital_amount)
    participation_col = f"participation_p95_{token}"
    required_liquidity = {"trade_id", participation_col}
    if missing := sorted(required_liquidity - set(per_trade_liquidity.columns)):
        raise ValueError(f"per_trade_liquidity missing required columns: {missing}")

    liquidity = per_trade_liquidity.loc[:, ["trade_id", participation_col]].copy()
    output = output.merge(liquidity, on="trade_id", how="left")
    fee_rate = float(fee_bps) / 10000.0
    impact_rate = pd.to_numeric(output[participation_col], errors="coerce").fillna(0.0) * float(impact_bps_per_1pct) / 100.0
    turnover = pd.to_numeric(output["turnover"], errors="coerce")
    output["fee_cost"] = turnover * fee_rate
    output["impact_rate"] = impact_rate
    output["impact_cost"] = turnover * impact_rate
    output["cost"] = output["fee_cost"] + output["impact_cost"]
    output["net_return"] = pd.to_numeric(output["gross_return"], errors="coerce") - output["cost"]
    output["fee_bps"] = float(fee_bps)
    output["capital_amount"] = float(capital_amount)
    output["impact_bps_per_1pct"] = float(impact_bps_per_1pct)
    return output


def run_low_corr_frontier_impact_stress(
    *,
    root: str | None = None,
    years: Sequence[int] = DEFAULT_YEARS,
    final_end_date: str | None = DEFAULT_FINAL_END_DATE,
    horizon: int = DEFAULT_HORIZON,
    label_mode: str = "raw",
    max_factor_corr: float = DEFAULT_MAX_FACTOR_CORR,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    rolling_min_periods: int = DEFAULT_ROLLING_MIN_PERIODS,
    signals: Sequence[str] = DEFAULT_FRONTIER_SIGNALS,
    top_n: int = DEFAULT_TOP_N,
    rebalance_frequency: str = DEFAULT_REBALANCE_FREQUENCY,
    buffer_multiplier: float = DEFAULT_BUFFER_MULTIPLIER,
    fee_bps_values: Sequence[float] = DEFAULT_FEE_BPS_VALUES,
    capital_amounts: Sequence[float] = DEFAULT_CAPITAL_AMOUNTS,
    impact_bps_per_1pct_values: Sequence[float] = DEFAULT_IMPACT_BPS_PER_1PCT,
    execution_constraints: bool = True,
    limit_threshold: float = 0.095,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    """Run participation impact stress for frontier signals."""

    if not years:
        raise ValueError("years must not be empty")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if label_mode not in LABEL_MODES:
        raise ValueError(f"unsupported label_mode: {label_mode}")
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if buffer_multiplier < 1.0:
        raise ValueError("buffer_multiplier must be at least 1.0")
    selected_signals = _normalize_signals(signals)
    if not fee_bps_values:
        raise ValueError("fee_bps_values must not be empty")
    if not capital_amounts:
        raise ValueError("capital_amounts must not be empty")
    if not impact_bps_per_1pct_values:
        raise ValueError("impact_bps_per_1pct_values must not be empty")

    run_id = f"low_corr_frontier_impact_stress_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    trade_frames: list[pd.DataFrame] = []
    liquidity_frames: list[pd.DataFrame] = []
    liquidity_summary_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []
    manifest: Mapping[str, Any] = {}
    quality: Mapping[str, Any] = {}

    for year in years:
        windows = year_windows(int(year), final_end_date=final_end_date)
        built = build_candidate_protocol_signal_panel(
            root=root,
            history_start_date=windows["history_start_date"],
            fit_start_date=windows["fit_start_date"],
            fit_end_date=windows["fit_end_date"],
            start_date=windows["start_date"],
            end_date=windows["end_date"],
            horizon=horizon,
            label_mode=label_mode,
            max_factor_corr=max_factor_corr,
            rolling_window=rolling_window,
            rolling_min_periods=rolling_min_periods,
        )
        manifest = built["manifest"]
        quality = built["quality"]
        evaluation_panel = built["evaluation_panel"]
        missing_signals = sorted(set(selected_signals) - set(built["available_signals"]))
        if missing_signals:
            raise ValueError(f"signals not available in panel: {missing_signals}")

        for signal in selected_signals:
            base_backtest = horizon_aligned_top_n_backtest(
                evaluation_panel,
                signal,
                horizon=horizon,
                top_n=top_n,
                fee_bps=0.0,
                rebalance_frequency=rebalance_frequency,
                buffer_multiplier=buffer_multiplier,
                execution_constraints=execution_constraints,
                limit_threshold=limit_threshold,
            )
            base_trades = base_backtest.trades.copy().reset_index(drop=True)
            if base_trades.empty:
                continue
            base_trades.insert(0, "trade_id", range(len(base_trades)))
            selected_holdings = expand_selected_holdings(
                evaluation_panel,
                base_trades,
                extra_columns=_holding_extra_columns(evaluation_panel, signal),
            )
            per_trade_liquidity, yearly_liquidity = summarize_trade_liquidity(
                selected_holdings,
                capital_amounts=capital_amounts,
            )
            if not per_trade_liquidity.empty:
                per_trade_liquidity.insert(0, "signal", signal)
                per_trade_liquidity.insert(0, "eval_year", int(year))
                liquidity_frames.append(per_trade_liquidity)
            if not yearly_liquidity.empty:
                yearly_liquidity.insert(0, "signal", signal)
                yearly_liquidity.insert(0, "eval_year", int(year))
                liquidity_summary_frames.append(yearly_liquidity)

            for fee_bps in fee_bps_values:
                for capital in capital_amounts:
                    for impact_bps in impact_bps_per_1pct_values:
                        trades = apply_fee_and_participation_impact(
                            base_trades,
                            per_trade_liquidity,
                            fee_bps=float(fee_bps),
                            capital_amount=float(capital),
                            impact_bps_per_1pct=float(impact_bps),
                        )
                        row = {
                            "eval_year": int(year),
                            "signal": signal,
                            "capital_amount": float(capital),
                            "impact_bps_per_1pct": float(impact_bps),
                        }
                        row.update(
                            summarize_horizon_trade_table(
                                trades,
                                horizon=horizon,
                                rebalance_frequency=rebalance_frequency,
                                top_n=top_n,
                                fee_bps=float(fee_bps),
                                buffer_multiplier=buffer_multiplier,
                                execution_constraints=execution_constraints,
                                limit_threshold=limit_threshold,
                            )
                        )
                        row.update(
                            {
                                "mean_fee_cost": float(trades["fee_cost"].mean()),
                                "mean_impact_cost": float(trades["impact_cost"].mean()),
                                "mean_total_cost": float(trades["cost"].mean()),
                                "mean_impact_rate": float(trades["impact_rate"].mean()),
                            }
                        )
                        summary_rows.append(row)
                        trades_out = trades.copy()
                        trades_out.insert(0, "signal", signal)
                        trades_out.insert(0, "eval_year", int(year))
                        trade_frames.append(trades_out)
        metadata_rows.append(
            {
                "eval_year": int(year),
                "history_start_date": windows["history_start_date"],
                "fit_start_date": windows["fit_start_date"],
                "fit_end_date": windows["fit_end_date"],
                "start_date": windows["start_date"],
                "end_date": windows["end_date"],
                "available_signals": json.dumps(list(selected_signals), ensure_ascii=False),
                "rolling_fallback_rate": float(built["rolling_fallback_rate"]),
                "evaluation_rows": int(len(evaluation_panel)),
                "evaluation_dates": int(evaluation_panel["date"].nunique()) if "date" in evaluation_panel.columns else 0,
                "evaluation_securities": int(evaluation_panel["code"].nunique()) if "code" in evaluation_panel.columns else 0,
            }
        )

    summary = pd.DataFrame(summary_rows)
    aggregate = summarize_impact_stress(summary)
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    liquidity = pd.concat(liquidity_frames, ignore_index=True) if liquidity_frames else pd.DataFrame()
    liquidity_summary = pd.concat(liquidity_summary_frames, ignore_index=True) if liquidity_summary_frames else pd.DataFrame()
    metadata = pd.DataFrame(metadata_rows)
    result = {
        "run_id": run_id,
        "snapshot_id": manifest.get("snapshot_id"),
        "years": [int(year) for year in years],
        "final_end_date": final_end_date,
        "horizon": int(horizon),
        "label_mode": label_mode,
        "rolling_window": int(rolling_window),
        "rolling_min_periods": int(rolling_min_periods),
        "signals": list(selected_signals),
        "top_n": int(top_n),
        "rebalance_frequency": rebalance_frequency,
        "buffer_multiplier": float(buffer_multiplier),
        "fee_bps_values": [float(value) for value in fee_bps_values],
        "capital_amounts": [float(value) for value in capital_amounts],
        "impact_bps_per_1pct_values": [float(value) for value in impact_bps_per_1pct_values],
        "execution_constraints": bool(execution_constraints),
        "limit_threshold": float(limit_threshold),
        "quality": {
            "failure_count": quality.get("failure_count"),
            "missing_bar_rows": quality.get("missing_bar_rows"),
            "st_rows": quality.get("st_rows"),
            "suspended_like_rows": quality.get("suspended_like_rows"),
        },
        "best_30bps_100m_rows": best_impact_rows(aggregate, fee_bps=30.0, capital_amount=100_000_000.0),
        "candidate_count": 0,
        "assessment": "impact stress evidence only; no strategy candidate is promoted by this experiment alone",
        "output_dir": str(run_dir),
    }

    summary.to_csv(run_dir / "impact_stress_summary.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(run_dir / "impact_stress_aggregate.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(run_dir / "impact_adjusted_trades.csv", index=False, encoding="utf-8-sig")
    liquidity.to_csv(run_dir / "impact_trade_liquidity.csv", index=False, encoding="utf-8-sig")
    liquidity_summary.to_csv(run_dir / "impact_liquidity_summary.csv", index=False, encoding="utf-8-sig")
    metadata.to_csv(run_dir / "impact_stress_meta.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_impact_stress_markdown(result, aggregate, liquidity_summary)
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        research_log_path.write_text(markdown, encoding="utf-8")
    return result


def summarize_impact_stress(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate impact stress rows by signal, fee, capital, and impact intensity."""

    if summary.empty:
        return pd.DataFrame()
    required = {
        "eval_year",
        "signal",
        "fee_bps",
        "capital_amount",
        "impact_bps_per_1pct",
        "annualized_return",
        "sharpe",
        "max_drawdown",
        "mean_turnover",
        "mean_impact_cost",
    }
    if missing := sorted(required - set(summary.columns)):
        raise ValueError(f"summary missing required columns: {missing}")
    work = summary.copy()
    no_impact = work.loc[
        pd.to_numeric(work["impact_bps_per_1pct"], errors="coerce") == 0.0,
        ["eval_year", "signal", "fee_bps", "capital_amount", "annualized_return"],
    ].rename(columns={"annualized_return": "annualized_return_no_impact"})
    work = work.merge(no_impact, on=["eval_year", "signal", "fee_bps", "capital_amount"], how="left")
    low_corr = work.loc[
        work["signal"] == LOW_CORR_SIGNAL,
        ["eval_year", "fee_bps", "capital_amount", "impact_bps_per_1pct", "annualized_return"],
    ].rename(columns={"annualized_return": "low_corr_annualized_return"})
    work = work.merge(
        low_corr,
        on=["eval_year", "fee_bps", "capital_amount", "impact_bps_per_1pct"],
        how="left",
    )
    work["impact_drag_vs_no_impact"] = pd.to_numeric(work["annualized_return"], errors="coerce") - pd.to_numeric(
        work["annualized_return_no_impact"],
        errors="coerce",
    )
    work["delta_annualized_return_vs_low_corr"] = pd.to_numeric(work["annualized_return"], errors="coerce") - pd.to_numeric(
        work["low_corr_annualized_return"],
        errors="coerce",
    )

    rows: list[dict[str, Any]] = []
    group_cols = ["signal", "fee_bps", "capital_amount", "impact_bps_per_1pct"]
    for keys, group in work.groupby(group_cols, sort=True):
        signal, fee_bps, capital, impact_bps = keys
        returns = pd.to_numeric(group["annualized_return"], errors="coerce")
        deltas = pd.to_numeric(group["delta_annualized_return_vs_low_corr"], errors="coerce")
        rows.append(
            {
                "signal": signal,
                "fee_bps": float(fee_bps),
                "capital_amount": float(capital),
                "impact_bps_per_1pct": float(impact_bps),
                "eval_year_count": int(group["eval_year"].nunique()),
                "mean_annualized_return": float(returns.mean()),
                "min_annualized_return": float(returns.min()),
                "positive_year_rate": float((returns > 0).mean()),
                "mean_sharpe": float(pd.to_numeric(group["sharpe"], errors="coerce").mean()),
                "worst_max_drawdown": float(pd.to_numeric(group["max_drawdown"], errors="coerce").min()),
                "mean_turnover": float(pd.to_numeric(group["mean_turnover"], errors="coerce").mean()),
                "mean_impact_cost": float(pd.to_numeric(group["mean_impact_cost"], errors="coerce").mean()),
                "mean_total_cost": float(pd.to_numeric(group["mean_total_cost"], errors="coerce").mean()),
                "mean_impact_drag_vs_no_impact": float(pd.to_numeric(group["impact_drag_vs_no_impact"], errors="coerce").mean()),
                "mean_delta_annualized_return_vs_low_corr": float(deltas.mean()) if not deltas.dropna().empty else np.nan,
                "positive_delta_year_rate_vs_low_corr": float((deltas > 0).mean()) if not deltas.dropna().empty else np.nan,
                "total_periods": int(pd.to_numeric(group["periods"], errors="coerce").sum()) if "periods" in group.columns else 0,
            }
        )
    output = pd.DataFrame(rows).sort_values(
        ["fee_bps", "capital_amount", "impact_bps_per_1pct", "mean_annualized_return"],
        ascending=[True, True, True, False],
    )
    if output.empty:
        return output
    output["rank_within_stress"] = output.groupby(["fee_bps", "capital_amount", "impact_bps_per_1pct"])["mean_annualized_return"].rank(
        method="first",
        ascending=False,
    ).astype(int)
    return output.reset_index(drop=True)


def best_impact_rows(
    aggregate: pd.DataFrame,
    *,
    fee_bps: float,
    capital_amount: float,
    top: int = 15,
) -> list[dict[str, Any]]:
    if aggregate.empty:
        return []
    rows = aggregate.loc[
        (pd.to_numeric(aggregate["fee_bps"], errors="coerce") == float(fee_bps))
        & (pd.to_numeric(aggregate["capital_amount"], errors="coerce") == float(capital_amount))
    ]
    if rows.empty:
        rows = aggregate
    return rows.sort_values(
        ["impact_bps_per_1pct", "mean_annualized_return", "positive_year_rate"],
        ascending=[True, False, False],
    ).head(top).to_dict("records")


def render_impact_stress_markdown(
    result: Mapping[str, Any],
    aggregate: pd.DataFrame,
    liquidity_summary: pd.DataFrame,
) -> str:
    best_rows = pd.DataFrame(result.get("best_30bps_100m_rows") or [])
    lines = [
        "# Low-Corr Frontier Impact Stress",
        "",
        "- Hypothesis: frontier signals should survive participation-based impact costs, not only fixed fee bps.",
        f"- Protocol: `horizon={result.get('horizon')}`, `{result.get('rebalance_frequency')}`, `top_n={result.get('top_n')}`, `buffer={result.get('buffer_multiplier')}`.",
        f"- Signals: `{result.get('signals')}`.",
        f"- Capital: `{result.get('capital_amounts')}`; impact bps per 1 pct participation `{result.get('impact_bps_per_1pct_values')}`.",
        "- Assessment: impact stress evidence only; candidate count remains `0` until all promotion gates pass.",
        f"- Artifacts: `{result.get('output_dir')}`",
        "",
        "## 30 bps / 100m Stress Rows",
        "",
        _markdown_table(best_rows),
        "",
        "## Aggregate",
        "",
        _markdown_table(aggregate),
        "",
        "## Liquidity Summary",
        "",
        _markdown_table(liquidity_summary),
        "",
    ]
    return "\n".join(lines)


def _holding_extra_columns(frame: pd.DataFrame, signal: str) -> list[str]:
    columns = [
        "amount",
        "volume",
        signal,
        "log_amount_mean_20d_z",
        "momentum_20d_z",
        "reversal_5d_z",
        "neg_volatility_20d_z",
        "neg_amplitude_20d_z",
    ]
    output: list[str] = []
    for column in columns:
        if column in frame.columns and column not in output:
            output.append(column)
    return output


def _capital_token(value: float) -> str:
    if value >= 1_000_000:
        return f"{int(round(value / 1_000_000))}m"
    return f"{int(round(value))}"


def _parse_years(spec: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in spec.split(",") if value.strip())
    if not values:
        raise ValueError("expected at least one year")
    return values


def _parse_float_tuple(spec: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in spec.split(",") if value.strip())
    if not values:
        raise ValueError("expected at least one numeric value")
    return values


def _parse_signal_tuple(spec: str) -> tuple[str, ...]:
    return _normalize_signals(spec.split(","))


def _normalize_signals(signals: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(signals, str):
        raw_values = signals.split(",")
    else:
        raw_values = signals
    values = tuple(value.strip() for value in raw_values if value and value.strip())
    if not values:
        raise ValueError("expected at least one signal")
    return values


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 40) -> str:
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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--years", default=",".join(str(year) for year in DEFAULT_YEARS))
    parser.add_argument("--final-end-date", default=DEFAULT_FINAL_END_DATE)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--label-mode", choices=LABEL_MODES, default="raw")
    parser.add_argument("--max-factor-corr", type=float, default=DEFAULT_MAX_FACTOR_CORR)
    parser.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW)
    parser.add_argument("--rolling-min-periods", type=int, default=DEFAULT_ROLLING_MIN_PERIODS)
    parser.add_argument("--signals", default=",".join(DEFAULT_FRONTIER_SIGNALS))
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--frequency", default=DEFAULT_REBALANCE_FREQUENCY)
    parser.add_argument("--buffer-multiplier", type=float, default=DEFAULT_BUFFER_MULTIPLIER)
    parser.add_argument("--fee-bps", default=",".join(str(value) for value in DEFAULT_FEE_BPS_VALUES))
    parser.add_argument("--capital-amounts", default=",".join(str(value) for value in DEFAULT_CAPITAL_AMOUNTS))
    parser.add_argument("--impact-bps-per-1pct", default=",".join(str(value) for value in DEFAULT_IMPACT_BPS_PER_1PCT))
    parser.add_argument("--execution-constraints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-threshold", type=float, default=0.095)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", type=Path, default=DEFAULT_RESEARCH_LOG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_low_corr_frontier_impact_stress(
        root=args.root,
        years=_parse_years(args.years),
        final_end_date=args.final_end_date,
        horizon=args.horizon,
        label_mode=args.label_mode,
        max_factor_corr=args.max_factor_corr,
        rolling_window=args.rolling_window,
        rolling_min_periods=args.rolling_min_periods,
        signals=_parse_signal_tuple(args.signals),
        top_n=args.top_n,
        rebalance_frequency=args.frequency,
        buffer_multiplier=args.buffer_multiplier,
        fee_bps_values=_parse_float_tuple(args.fee_bps),
        capital_amounts=_parse_float_tuple(args.capital_amounts),
        impact_bps_per_1pct_values=_parse_float_tuple(args.impact_bps_per_1pct),
        execution_constraints=args.execution_constraints,
        limit_threshold=args.limit_threshold,
        output_dir=args.output_dir,
        write_research_log=args.write_research_log,
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
