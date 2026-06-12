"""Create forward paper-tracking plan artifacts for personal frontier candidates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from traditional_quant_research.experiments.frontier_personal_paper_tracking_bootstrap import (
    BOOTSTRAPPED_TRACKING_STATUS,
    DEFAULT_OUTPUT_DIR as DEFAULT_BOOTSTRAP_OUTPUT_ROOT,
    PAPER_TRACKING_CANCELLED_MESSAGE,
    PERSONAL_PAPER_CANDIDATE_LEVEL,
)
from traditional_quant_research.experiments.frontier_promotion_gate import latest_run_dir


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/frontier_personal_paper_tracking_plan")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/brain/references/research_log/2026-06-04_frontier_personal_paper_tracking_plan.md")
DEFAULT_PLAN_PERIODS = 6

PLAN_STATUS_READY = "tracking_plan_scaffold_ready"
FUTURE_RECORD_STATUS = "future_observation_incomplete"
PERSONAL_BACKTEST_CANDIDATE_LEVEL = "personal_backtest_candidate"


def run_frontier_personal_paper_tracking_plan(
    *,
    bootstrap_run_dir: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    plan_periods: int = DEFAULT_PLAN_PERIODS,
    plan_start_date: str | None = None,
    write_research_log: bool = False,
    research_log_path: str | Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    """Build a future paper-tracking calendar and empty live log starter."""

    raise RuntimeError(PAPER_TRACKING_CANCELLED_MESSAGE)

    if plan_periods <= 0:
        raise ValueError("plan_periods must be positive")

    run_id = f"frontier_personal_paper_tracking_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    bootstrap_dir = Path(bootstrap_run_dir) if bootstrap_run_dir is not None else latest_run_dir(DEFAULT_BOOTSTRAP_OUTPUT_ROOT)
    candidates, protocol, bootstrap_summary = read_bootstrap_artifacts(bootstrap_dir)
    combined_summary = read_combined_summary_from_bootstrap(bootstrap_summary)
    historical_context = build_historical_context(candidates, bootstrap_summary)
    plan_calendar = build_paper_tracking_plan_calendar(
        candidates,
        protocol,
        bootstrap_summary,
        combined_summary,
        historical_context,
        plan_periods=plan_periods,
        plan_start_date=plan_start_date,
    )
    live_log_starter = build_paper_tracking_live_log_starter(plan_calendar)
    summary = summarize_paper_tracking_plan(
        plan_calendar,
        live_log_starter,
        historical_context,
        run_id=run_id,
        bootstrap_run_dir=bootstrap_dir,
        bootstrap_summary=bootstrap_summary,
        combined_summary=combined_summary,
        plan_periods=plan_periods,
        plan_start_date=plan_start_date,
    )
    markdown = render_paper_tracking_plan_markdown(summary, plan_calendar, historical_context)

    historical_context.to_csv(run_dir / "paper_tracking_historical_context.csv", index=False, encoding="utf-8-sig")
    plan_calendar.to_csv(run_dir / "paper_tracking_plan_calendar.csv", index=False, encoding="utf-8-sig")
    live_log_starter.to_csv(run_dir / "paper_tracking_live_log_starter.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        path = Path(research_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
    return {**summary, "run_dir": str(run_dir), "research_log": str(research_log_path) if write_research_log else None}


def read_bootstrap_artifacts(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    candidates_path = run_dir / "paper_tracking_candidates.csv"
    protocol_path = run_dir / "paper_tracking_protocol.csv"
    summary_path = run_dir / "summary.json"
    if not candidates_path.exists():
        raise FileNotFoundError(f"paper tracking candidates not found: {candidates_path}")
    if not protocol_path.exists():
        raise FileNotFoundError(f"paper tracking protocol not found: {protocol_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"paper tracking bootstrap summary not found: {summary_path}")
    return (
        pd.read_csv(candidates_path),
        pd.read_csv(protocol_path),
        json.loads(summary_path.read_text(encoding="utf-8-sig")),
    )


def read_combined_summary_from_bootstrap(bootstrap_summary: Mapping[str, Any]) -> dict[str, Any]:
    combined_dir = str(bootstrap_summary.get("combined_run_dir", "")).strip()
    if not combined_dir:
        return {}
    path = Path(combined_dir) / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_historical_context(candidates: pd.DataFrame, bootstrap_summary: Mapping[str, Any]) -> pd.DataFrame:
    """Attach the latest historical trade metadata when combined trades are available."""

    raise RuntimeError(PAPER_TRACKING_CANCELLED_MESSAGE)

    columns = _historical_context_columns()
    if candidates.empty:
        return pd.DataFrame(columns=columns)

    combined_dir = str(bootstrap_summary.get("combined_run_dir", "")).strip()
    trades_path = Path(combined_dir) / "combined_constraint_trades.csv" if combined_dir else Path()
    if not combined_dir or not trades_path.exists():
        return pd.DataFrame([_missing_historical_context(row, "missing_combined_trades") for row in candidates.to_dict("records")], columns=columns)

    trades = pd.read_csv(trades_path)
    if trades.empty:
        return pd.DataFrame([_missing_historical_context(row, "empty_combined_trades") for row in candidates.to_dict("records")], columns=columns)

    rows: list[dict[str, Any]] = []
    for candidate in candidates.to_dict("records"):
        candidate_trades = _filter_candidate_trades(trades, candidate)
        if candidate_trades.empty:
            rows.append(_missing_historical_context(candidate, "candidate_trade_not_found"))
            continue
        for column in ["signal_date", "entry_date", "exit_date"]:
            if column in candidate_trades.columns:
                candidate_trades[column] = pd.to_datetime(candidate_trades[column], errors="coerce")
        sort_cols = [column for column in ["signal_date", "entry_date", "exit_date"] if column in candidate_trades.columns]
        latest = candidate_trades.sort_values(sort_cols).tail(1).iloc[0].to_dict() if sort_cols else candidate_trades.tail(1).iloc[0].to_dict()
        rows.append(
            {
                "candidate_id": str(candidate.get("candidate_id", "")),
                "historical_context_status": "matched_combined_trades",
                "combined_trade_count": int(len(candidate_trades)),
                "last_eval_year": _optional_int(latest.get("eval_year")),
                "last_historical_signal_date": _date_str(latest.get("signal_date")),
                "last_historical_entry_date": _date_str(latest.get("entry_date")),
                "last_historical_exit_date": _date_str(latest.get("exit_date")),
                "last_historical_selected_count": _optional_int(latest.get("holdings")),
                "last_historical_requested_count": _optional_int(latest.get("requested_holdings")),
                "last_historical_blocked_entry_count": _optional_int(latest.get("blocked_entry_count")),
                "last_historical_entry_limit_up_count": _optional_int(latest.get("entry_limit_up_count")),
                "last_historical_exit_delayed_count": _optional_int(latest.get("exit_delayed_count")),
                "last_historical_exit_limit_down_count": _optional_int(latest.get("exit_limit_down_count")),
                "last_historical_turnover": _optional_float(latest.get("turnover")),
                "last_historical_net_return": _optional_float(latest.get("net_return")),
                "last_historical_capital_scale": _optional_float(latest.get("capital_scale")),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_paper_tracking_plan_calendar(
    candidates: pd.DataFrame,
    protocol: pd.DataFrame,
    bootstrap_summary: Mapping[str, Any],
    combined_summary: Mapping[str, Any],
    historical_context: pd.DataFrame,
    *,
    plan_periods: int = DEFAULT_PLAN_PERIODS,
    plan_start_date: str | None = None,
) -> pd.DataFrame:
    raise RuntimeError(PAPER_TRACKING_CANCELLED_MESSAGE)

    columns = _plan_calendar_columns()
    if candidates.empty:
        return pd.DataFrame(columns=columns)

    protocol_by_candidate = {
        str(row.get("candidate_id", "")): row
        for row in protocol.to_dict("records")
        if str(row.get("candidate_id", "")).strip()
    }
    context_by_candidate = {
        str(row.get("candidate_id", "")): row
        for row in historical_context.to_dict("records")
        if str(row.get("candidate_id", "")).strip()
    }
    anchor_date = _resolve_plan_anchor_date(plan_start_date, combined_summary, historical_context)

    rows: list[dict[str, Any]] = []
    for candidate in candidates.to_dict("records"):
        candidate_id = str(candidate.get("candidate_id", ""))
        protocol_row = protocol_by_candidate.get(candidate_id, {})
        context_row = context_by_candidate.get(candidate_id, {})
        rebalance_frequency = str(
            protocol_row.get("rebalance_frequency", candidate.get("rebalance_frequency", combined_summary.get("rebalance_frequency", "monthly")))
        )
        expected_signal_dates = _expected_signal_dates(anchor_date, rebalance_frequency, plan_periods)
        for idx, expected_signal_date in enumerate(expected_signal_dates, start=1):
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "observation_number": idx,
                    "tracking_plan_status": PLAN_STATUS_READY,
                    "record_status": FUTURE_RECORD_STATUS,
                    "evidence_scope": "future_paper_tracking_plan",
                    "current_level": str(candidate.get("current_level", PERSONAL_BACKTEST_CANDIDATE_LEVEL)),
                    "target_next_level": str(candidate.get("target_next_level", PERSONAL_PAPER_CANDIDATE_LEVEL)),
                    "personal_paper_candidate": False,
                    "strategy_candidate": False,
                    "bootstrap_run_id": str(bootstrap_summary.get("run_id", "")),
                    "combined_run_id": str(combined_summary.get("run_id", "")),
                    "snapshot_id": str(bootstrap_summary.get("snapshot_id", combined_summary.get("snapshot_id", ""))),
                    "last_available_data_date": _date_str(anchor_date),
                    "expected_signal_period_end": expected_signal_date.date().isoformat(),
                    "actual_signal_date": "",
                    "actual_rebalance_date": "",
                    "planned_entry_date": "",
                    "planned_exit_date": "",
                    "manual_fill_required": True,
                    "signal": str(candidate.get("signal", "")),
                    "constraint_variant": str(candidate.get("constraint_variant", "")),
                    "exposure_penalty_strength": _optional_float(candidate.get("exposure_penalty_strength")),
                    "rebalance_frequency": rebalance_frequency,
                    "horizon": _optional_int(protocol_row.get("horizon", candidate.get("horizon", combined_summary.get("horizon")))),
                    "top_n": _optional_int(protocol_row.get("top_n", candidate.get("top_n", combined_summary.get("top_n")))),
                    "buffer_multiplier": _optional_float(protocol_row.get("buffer_multiplier", candidate.get("buffer_multiplier", combined_summary.get("buffer_multiplier")))),
                    "fee_bps": _optional_float(protocol_row.get("fee_bps", candidate.get("fee_bps"))),
                    "impact_bps_per_1pct": _optional_float(protocol_row.get("impact_bps_per_1pct", candidate.get("impact_bps_per_1pct"))),
                    "personal_capital_amount": _optional_float(protocol_row.get("personal_capital_amount", candidate.get("personal_capital_amount"))),
                    "last_historical_signal_date": str(context_row.get("last_historical_signal_date", "")),
                    "last_historical_entry_date": str(context_row.get("last_historical_entry_date", "")),
                    "last_historical_exit_date": str(context_row.get("last_historical_exit_date", "")),
                    "last_historical_selected_count": _optional_int(context_row.get("last_historical_selected_count")),
                    "last_historical_blocked_entry_count": _optional_int(context_row.get("last_historical_blocked_entry_count")),
                    "notes": "Fill actual dates, execution counts, returns, drawdown, and notes after new Baostock data and paper execution are observed.",
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_paper_tracking_live_log_starter(plan_calendar: pd.DataFrame) -> pd.DataFrame:
    """Return empty rows that are compatible with the paper-tracking review input."""

    raise RuntimeError(PAPER_TRACKING_CANCELLED_MESSAGE)

    columns = _live_log_columns()
    if plan_calendar.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for row in plan_calendar.to_dict("records"):
        rows.append(
            {
                "record_date": "",
                "candidate_id": row.get("candidate_id", ""),
                "signal": row.get("signal", ""),
                "constraint_variant": row.get("constraint_variant", ""),
                "observation_number": row.get("observation_number", ""),
                "expected_signal_period_end": row.get("expected_signal_period_end", ""),
                "record_status": row.get("record_status", FUTURE_RECORD_STATUS),
                "signal_date": "",
                "rebalance_date": "",
                "planned_entry_date": "",
                "planned_exit_date": "",
                "target_top_n": row.get("top_n", ""),
                "selected_count": "",
                "blocked_entry_count": "",
                "entry_limit_up_count": "",
                "exit_delayed_count": "",
                "exit_limit_down_count": "",
                "gross_return": "",
                "net_return": "",
                "paper_account_return": "",
                "benchmark_return": "",
                "excess_return": "",
                "realized_turnover": "",
                "realized_fee_bps": "",
                "realized_impact_note": "",
                "max_drawdown_to_date": "",
                "execution_notes": "",
                "decision_notes": "",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summarize_paper_tracking_plan(
    plan_calendar: pd.DataFrame,
    live_log_starter: pd.DataFrame,
    historical_context: pd.DataFrame,
    *,
    run_id: str,
    bootstrap_run_dir: Path,
    bootstrap_summary: Mapping[str, Any],
    combined_summary: Mapping[str, Any],
    plan_periods: int,
    plan_start_date: str | None,
) -> dict[str, Any]:
    raise RuntimeError(PAPER_TRACKING_CANCELLED_MESSAGE)

    candidate_ids = sorted(plan_calendar["candidate_id"].astype(str).unique().tolist()) if not plan_calendar.empty else []
    matched_context = (
        int(historical_context["historical_context_status"].astype(str).eq("matched_combined_trades").sum())
        if "historical_context_status" in historical_context.columns
        else 0
    )
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "north_star": "Baostock-only personal quant strategy research",
        "bootstrap_run_dir": str(bootstrap_run_dir),
        "bootstrap_run_id": str(bootstrap_summary.get("run_id", "")),
        "combined_run_dir": str(bootstrap_summary.get("combined_run_dir", "")),
        "combined_run_id": str(combined_summary.get("run_id", "")),
        "snapshot_id": str(bootstrap_summary.get("snapshot_id", combined_summary.get("snapshot_id", ""))),
        "last_available_data_date": str(combined_summary.get("final_end_date", "")),
        "plan_periods": int(plan_periods),
        "plan_start_date": plan_start_date or "",
        "candidate_count": int(len(candidate_ids)),
        "plan_calendar_rows": int(len(plan_calendar)),
        "live_log_starter_rows": int(len(live_log_starter)),
        "historical_context_matched_count": matched_context,
        "personal_backtest_candidate_count": int(len(candidate_ids)),
        "personal_paper_candidate_count": 0,
        "strategy_candidate_count": 0,
        "decision": PLAN_STATUS_READY if candidate_ids else "no_personal_backtest_candidate_to_plan",
        "candidate_ids": candidate_ids,
        "limitations": [
            "This plan is an operational scaffold for future paper tracking and does not generate signals for future dates.",
            "Future rows are incomplete until actual Baostock data, execution counts, returns, and drawdown are recorded.",
            "This artifact cannot promote any row to personal_paper_candidate or strategy_candidate.",
        ],
    }


def render_paper_tracking_plan_markdown(
    summary: Mapping[str, Any],
    plan_calendar: pd.DataFrame,
    historical_context: pd.DataFrame,
) -> str:
    raise RuntimeError(PAPER_TRACKING_CANCELLED_MESSAGE)

    lines = [
        "# Frontier Personal Paper Tracking Plan",
        "",
        f"- run_id: `{summary.get('run_id', '')}`",
        f"- decision: `{summary.get('decision', '')}`",
        f"- candidate_count: `{summary.get('candidate_count', 0)}`",
        f"- plan_periods: `{summary.get('plan_periods', 0)}`",
        f"- plan_calendar_rows: `{summary.get('plan_calendar_rows', 0)}`",
        f"- live_log_starter_rows: `{summary.get('live_log_starter_rows', 0)}`",
        f"- historical_context_matched_count: `{summary.get('historical_context_matched_count', 0)}`",
        f"- personal_paper_candidate_count: `{summary.get('personal_paper_candidate_count', 0)}`",
        f"- strategy_candidate_count: `{summary.get('strategy_candidate_count', 0)}`",
        f"- bootstrap_run_id: `{summary.get('bootstrap_run_id', '')}`",
        f"- combined_run_id: `{summary.get('combined_run_id', '')}`",
        "",
        "## Plan Calendar",
        "",
        _markdown_table(plan_calendar),
        "",
        "## Historical Context",
        "",
        _markdown_table(historical_context),
        "",
        "## Interpretation",
        "",
        "This artifact turns bootstrapped personal backtest candidates into future paper-tracking rows. "
        "Rows remain incomplete until actual post-snapshot observations are filled. "
        "It deliberately keeps `personal_paper_candidate_count=0` and `strategy_candidate_count=0`.",
        "",
    ]
    return "\n".join(lines)


def _filter_candidate_trades(trades: pd.DataFrame, candidate: Mapping[str, Any]) -> pd.DataFrame:
    output = trades.copy()
    for column, candidate_key in [
        ("signal", "signal"),
        ("constraint_variant", "constraint_variant"),
    ]:
        if column in output.columns:
            output = output.loc[output[column].astype(str).eq(str(candidate.get(candidate_key, "")))]
    if "top_n" in output.columns and "top_n" in candidate:
        output = output.loc[pd.to_numeric(output["top_n"], errors="coerce").eq(_optional_float(candidate.get("top_n")))]
    if "exposure_penalty_strength" in output.columns and "exposure_penalty_strength" in candidate:
        candidate_strength = _optional_float(candidate.get("exposure_penalty_strength"))
        output = output.loc[np.isclose(pd.to_numeric(output["exposure_penalty_strength"], errors="coerce"), candidate_strength, equal_nan=False)]
    return output.copy()


def _missing_historical_context(candidate: Mapping[str, Any], status: str) -> dict[str, Any]:
    return {
        "candidate_id": str(candidate.get("candidate_id", "")),
        "historical_context_status": status,
        "combined_trade_count": 0,
        "last_eval_year": None,
        "last_historical_signal_date": "",
        "last_historical_entry_date": "",
        "last_historical_exit_date": "",
        "last_historical_selected_count": None,
        "last_historical_requested_count": None,
        "last_historical_blocked_entry_count": None,
        "last_historical_entry_limit_up_count": None,
        "last_historical_exit_delayed_count": None,
        "last_historical_exit_limit_down_count": None,
        "last_historical_turnover": np.nan,
        "last_historical_net_return": np.nan,
        "last_historical_capital_scale": np.nan,
    }


def _resolve_plan_anchor_date(
    plan_start_date: str | None,
    combined_summary: Mapping[str, Any],
    historical_context: pd.DataFrame,
) -> pd.Timestamp:
    if plan_start_date:
        return pd.Timestamp(plan_start_date)
    for key in ["final_end_date", "end_date"]:
        value = str(combined_summary.get(key, "")).strip()
        if value:
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.notna(parsed):
                return pd.Timestamp(parsed)
    if not historical_context.empty and "last_historical_signal_date" in historical_context.columns:
        dates = pd.to_datetime(historical_context["last_historical_signal_date"], errors="coerce").dropna()
        if not dates.empty:
            return pd.Timestamp(dates.max())
    return pd.Timestamp.today().normalize()


def _expected_signal_dates(anchor_date: pd.Timestamp, rebalance_frequency: str, periods: int) -> list[pd.Timestamp]:
    frequency = rebalance_frequency.strip().lower()
    start = pd.Timestamp(anchor_date).normalize()
    if frequency in {"monthly", "month", "m"}:
        first = start + pd.offsets.MonthEnd(0)
        if first <= start:
            first = start + pd.offsets.MonthEnd(1)
        return [pd.Timestamp(first + pd.offsets.MonthEnd(i)) for i in range(periods)]
    if frequency in {"weekly", "week", "w"}:
        dates = pd.date_range(start + pd.Timedelta(days=1), periods=periods, freq="W-FRI")
        return [pd.Timestamp(value) for value in dates]
    if frequency in {"quarterly", "quarter", "q"}:
        first = start + pd.offsets.QuarterEnd(0)
        if first <= start:
            first = start + pd.offsets.QuarterEnd(1)
        return [pd.Timestamp(first + pd.offsets.QuarterEnd(i)) for i in range(periods)]
    dates = pd.date_range(start + pd.Timedelta(days=1), periods=periods, freq="D")
    return [pd.Timestamp(value) for value in dates]


def _historical_context_columns() -> list[str]:
    return [
        "candidate_id",
        "historical_context_status",
        "combined_trade_count",
        "last_eval_year",
        "last_historical_signal_date",
        "last_historical_entry_date",
        "last_historical_exit_date",
        "last_historical_selected_count",
        "last_historical_requested_count",
        "last_historical_blocked_entry_count",
        "last_historical_entry_limit_up_count",
        "last_historical_exit_delayed_count",
        "last_historical_exit_limit_down_count",
        "last_historical_turnover",
        "last_historical_net_return",
        "last_historical_capital_scale",
    ]


def _plan_calendar_columns() -> list[str]:
    return [
        "candidate_id",
        "observation_number",
        "tracking_plan_status",
        "record_status",
        "evidence_scope",
        "current_level",
        "target_next_level",
        "personal_paper_candidate",
        "strategy_candidate",
        "bootstrap_run_id",
        "combined_run_id",
        "snapshot_id",
        "last_available_data_date",
        "expected_signal_period_end",
        "actual_signal_date",
        "actual_rebalance_date",
        "planned_entry_date",
        "planned_exit_date",
        "manual_fill_required",
        "signal",
        "constraint_variant",
        "exposure_penalty_strength",
        "rebalance_frequency",
        "horizon",
        "top_n",
        "buffer_multiplier",
        "fee_bps",
        "impact_bps_per_1pct",
        "personal_capital_amount",
        "last_historical_signal_date",
        "last_historical_entry_date",
        "last_historical_exit_date",
        "last_historical_selected_count",
        "last_historical_blocked_entry_count",
        "notes",
    ]


def _live_log_columns() -> list[str]:
    return [
        "record_date",
        "candidate_id",
        "signal",
        "constraint_variant",
        "observation_number",
        "expected_signal_period_end",
        "record_status",
        "signal_date",
        "rebalance_date",
        "planned_entry_date",
        "planned_exit_date",
        "target_top_n",
        "selected_count",
        "blocked_entry_count",
        "entry_limit_up_count",
        "exit_delayed_count",
        "exit_limit_down_count",
        "gross_return",
        "net_return",
        "paper_account_return",
        "benchmark_return",
        "excess_return",
        "realized_turnover",
        "realized_fee_bps",
        "realized_impact_note",
        "max_drawdown_to_date",
        "execution_notes",
        "decision_notes",
    ]


def _optional_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _optional_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _date_str(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return ""
    return pd.Timestamp(parsed).date().isoformat()


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
    parser.add_argument("--bootstrap-run-dir", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plan-periods", type=int, default=DEFAULT_PLAN_PERIODS)
    parser.add_argument("--plan-start-date", default=None)
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", type=Path, default=DEFAULT_RESEARCH_LOG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_frontier_personal_paper_tracking_plan(
        bootstrap_run_dir=args.bootstrap_run_dir,
        output_dir=args.output_dir,
        plan_periods=args.plan_periods,
        plan_start_date=args.plan_start_date,
        write_research_log=args.write_research_log,
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
