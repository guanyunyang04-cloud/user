"""Live-probe Tushare daily_basic as the v2.2 PIT size source candidate."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from traditional_quant_research.size_source import standardize_tushare_daily_basic_size


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/v2_tushare_size_probe")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/research_log/2026-06-03_v2_tushare_size_probe.md")
DEFAULT_SYMBOLS = ("600000.SH", "000001.SZ")
DEFAULT_TRADE_DATES = ("20260525", "20260601")

REQUIRED_FIELDS = (
    "ts_code",
    "trade_date",
    "total_mv",
    "circ_mv",
    "total_share",
    "float_share",
    "free_share",
)
OPTIONAL_FIELDS = (
    "close",
    "turnover_rate",
    "turnover_rate_f",
    "pe_ttm",
    "pb",
)
PROBE_FIELDS = REQUIRED_FIELDS + OPTIONAL_FIELDS
OUTPUT_COLUMNS = (
    "date",
    "code",
    "ts_code",
    "trade_date",
    "total_mv",
    "circ_mv",
    "total_share",
    "float_share",
    "free_share",
    "close",
    "turnover_rate",
    "turnover_rate_f",
    "pe_ttm",
    "pb",
    "source",
)
FAILURE_COLUMNS = ("code", "trade_date", "error_type", "message")


def normalize_tushare_symbols(symbols: str | Sequence[str]) -> list[str]:
    """Normalize project symbols into Tushare ts_code format."""

    if isinstance(symbols, str):
        raw_symbols = [item.strip() for item in symbols.split(",")]
    else:
        raw_symbols = [str(item).strip() for item in symbols]
    output: list[str] = []
    for symbol in raw_symbols:
        if not symbol:
            continue
        upper = symbol.upper()
        if upper.startswith("SH.") or upper.startswith("SZ."):
            market, code = upper.split(".", 1)
            upper = f"{code}.{market}"
        output.append(upper)
    return output


def normalize_trade_dates(trade_dates: str | Sequence[str]) -> list[str]:
    """Normalize date inputs to Tushare YYYYMMDD strings."""

    if isinstance(trade_dates, str):
        raw_dates = [item.strip() for item in trade_dates.split(",")]
    else:
        raw_dates = [str(item).strip() for item in trade_dates]
    output: list[str] = []
    for item in raw_dates:
        if not item:
            continue
        digits = item.replace("-", "")
        if len(digits) != 8 or not digits.isdigit():
            raise ValueError(f"Invalid trade date: {item!r}")
        output.append(digits)
    return output


def tushare_token(token: str | None = None, *, env: Mapping[str, str] | None = None) -> str:
    """Resolve an explicit or environment Tushare token."""

    if token:
        return token
    env_map = os.environ if env is None else env
    return str(env_map.get("TUSHARE_TOKEN") or env_map.get("TS_TOKEN") or "")


def run_v2_tushare_size_probe(
    *,
    symbols: str | Sequence[str] = DEFAULT_SYMBOLS,
    trade_dates: str | Sequence[str] = DEFAULT_TRADE_DATES,
    token: str | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: str | Path = DEFAULT_RESEARCH_LOG,
    tushare_module: Any | None = None,
) -> dict[str, Any]:
    """Run the Tushare live probe and write reproducible artifacts."""

    run_id = f"v2_tushare_size_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    symbols_list = normalize_tushare_symbols(symbols)
    trade_date_list = normalize_trade_dates(trade_dates)
    resolved_token = tushare_token(token)
    module, import_error = _load_tushare_module(tushare_module)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    status = "passed"
    skip_reason = ""

    if module is None:
        status = "skipped"
        skip_reason = "package_missing"
        failures.append({"code": "", "trade_date": "", "error_type": skip_reason, "message": import_error})
    elif not resolved_token:
        status = "skipped"
        skip_reason = "auth_missing"
        failures.append(
            {
                "code": "",
                "trade_date": "",
                "error_type": skip_reason,
                "message": "Set TUSHARE_TOKEN or TS_TOKEN, or pass --token, before live probing daily_basic.",
            }
        )
    else:
        rows, failures = _query_daily_basic(module, resolved_token, symbols_list, trade_date_list)

    probe = _normalize_probe_rows(rows)
    daily_size_probe = standardize_tushare_daily_basic_size(probe)
    failure_frame = pd.DataFrame(failures, columns=FAILURE_COLUMNS)
    summary = _summarize_probe(
        run_id=run_id,
        symbols=symbols_list,
        trade_dates=trade_date_list,
        probe=probe,
        failures=failure_frame,
        status=status,
        skip_reason=skip_reason,
        import_error=import_error,
        token_present=bool(resolved_token),
    )
    summary_md = render_summary_markdown(summary)

    probe.to_csv(run_dir / "tushare_daily_basic_probe.csv", index=False, encoding="utf-8-sig")
    daily_size_probe.to_csv(run_dir / "daily_size_probe.csv", index=False, encoding="utf-8-sig")
    daily_size_probe.to_parquet(run_dir / "daily_size_probe.parquet", index=False)
    failure_frame.to_csv(run_dir / "failures.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    if write_research_log:
        path = Path(research_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(summary_md, encoding="utf-8")

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": summary["status"],
        "skip_reason": summary["skip_reason"],
        "expected_pairs": summary["expected_pairs"],
        "returned_pairs": summary["returned_pairs"],
        "missing_pairs": summary["missing_pairs"],
        "failure_count": summary["failure_count"],
        "required_fields_present": summary["required_fields_present"],
        "v2_2_ready": summary["v2_2_ready"],
    }


def render_summary_markdown(summary: Mapping[str, Any]) -> str:
    """Render a concise research-log summary."""

    lines = [
        "# V2 Tushare Size Probe",
        "",
        "## Summary",
        "",
        f"- `run_id`: `{summary['run_id']}`",
        f"- `status`: `{summary['status']}`",
        f"- `skip_reason`: `{summary['skip_reason']}`",
        f"- `symbols`: `{','.join(summary['symbols'])}`",
        f"- `trade_dates`: `{','.join(summary['trade_dates'])}`",
        f"- `expected_pairs`: `{summary['expected_pairs']}`",
        f"- `returned_pairs`: `{summary['returned_pairs']}`",
        f"- `missing_pairs`: `{summary['missing_pairs']}`",
        f"- `failure_count`: `{summary['failure_count']}`",
        f"- `required_fields_present`: `{summary['required_fields_present']}`",
        f"- `v2_2_ready`: `{summary['v2_2_ready']}`",
        "",
        "## Artifacts",
        "",
        "- `tushare_daily_basic_probe.csv`: raw Tushare daily_basic-shaped probe rows.",
        "- `daily_size_probe.csv` / `daily_size_probe.parquet`: vendor-neutral v2 daily_size schema rows.",
        "- `failures.csv`: package, auth, empty-result or query failures.",
        "- `summary.json` / `summary.md`: structured and readable gate summary.",
        "",
        "## Field Non-Null Rates",
        "",
    ]
    for field, rate in summary["required_field_non_null_rates"].items():
        lines.append(f"- `{field}`: `{rate}`")
    lines.extend(
        [
            "",
            "## Unit Assumptions",
            "",
            "- `total_mv/circ_mv`: Tushare daily_basic documents market value fields in 10k CNY units.",
            "- `total_share/float_share/free_share`: Tushare daily_basic documents share-base fields in 10k share units.",
            "",
            "## Decision",
            "",
            str(summary["decision"]),
            "",
            "## Interpretation",
            "",
            "This probe is a v2.2 data gate. It does not promote a strategy candidate. "
            "The frontier remains `candidate-frontier/backtest_only` until a PIT size source is live-validated "
            "and integrated into the snapshot.",
            "",
        ]
    )
    return "\n".join(lines)


def _query_daily_basic(
    module: Any,
    token: str,
    symbols: Sequence[str],
    trade_dates: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if hasattr(module, "set_token"):
        module.set_token(token)
    pro = module.pro_api(token) if callable(getattr(module, "pro_api", None)) else module.pro_api()
    fields = ",".join(PROBE_FIELDS)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for trade_date in trade_dates:
        for code in symbols:
            try:
                frame = pro.daily_basic(ts_code=code, trade_date=trade_date, fields=fields)
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {"code": code, "trade_date": trade_date, "error_type": type(exc).__name__, "message": str(exc)}
                )
                continue
            if frame is None or frame.empty:
                failures.append(
                    {
                        "code": code,
                        "trade_date": trade_date,
                        "error_type": "empty_result",
                        "message": "daily_basic returned no rows for code/date.",
                    }
                )
                continue
            for row in frame.to_dict("records"):
                payload = dict(row)
                payload.setdefault("ts_code", code)
                payload.setdefault("trade_date", trade_date)
                rows.append(payload)
    return rows, failures


def _normalize_probe_rows(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    frame = pd.DataFrame(rows)
    for field in PROBE_FIELDS:
        if field not in frame.columns:
            frame[field] = pd.NA
    frame["code"] = frame["ts_code"].astype(str).str.upper()
    frame["date"] = frame["trade_date"].map(_format_trade_date)
    frame["source"] = "tushare.daily_basic"
    numeric_cols = [
        "total_mv",
        "circ_mv",
        "total_share",
        "float_share",
        "free_share",
        "close",
        "turnover_rate",
        "turnover_rate_f",
        "pe_ttm",
        "pb",
    ]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.loc[:, list(OUTPUT_COLUMNS)].sort_values(["date", "code"]).reset_index(drop=True)


def _summarize_probe(
    *,
    run_id: str,
    symbols: Sequence[str],
    trade_dates: Sequence[str],
    probe: pd.DataFrame,
    failures: pd.DataFrame,
    status: str,
    skip_reason: str,
    import_error: str,
    token_present: bool,
) -> dict[str, Any]:
    expected_pairs = len(symbols) * len(trade_dates)
    returned_pairs = int(probe[["code", "trade_date"]].drop_duplicates().shape[0]) if not probe.empty else 0
    missing_pairs = max(expected_pairs - returned_pairs, 0)
    required_fields_present = (not probe.empty) and all(field in probe.columns for field in REQUIRED_FIELDS)
    rates: dict[str, float | None] = {}
    for field in REQUIRED_FIELDS:
        if probe.empty or field not in probe.columns:
            rates[field] = None
        else:
            rates[field] = float(probe[field].notna().mean())
    if status != "skipped":
        required_non_null = all((value is not None and value == 1.0) for value in rates.values())
        if missing_pairs == 0 and int(failures.shape[0]) == 0 and required_fields_present and required_non_null:
            status = "passed"
        elif returned_pairs > 0:
            status = "partial"
        else:
            status = "failed"
    v2_2_ready = bool(status == "passed")
    decision = _decision(status=status, skip_reason=skip_reason, missing_pairs=missing_pairs, failure_count=int(failures.shape[0]))
    return {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "skip_reason": skip_reason,
        "token_present": token_present,
        "import_error": import_error,
        "symbols": list(symbols),
        "trade_dates": list(trade_dates),
        "expected_pairs": expected_pairs,
        "returned_pairs": returned_pairs,
        "missing_pairs": missing_pairs,
        "failure_count": int(failures.shape[0]),
        "required_fields": list(REQUIRED_FIELDS),
        "required_fields_present": bool(required_fields_present),
        "required_field_non_null_rates": rates,
        "unit_assumptions": {
            "total_mv": "10k CNY",
            "circ_mv": "10k CNY",
            "total_share": "10k shares",
            "float_share": "10k shares",
            "free_share": "10k shares",
        },
        "v2_2_ready": v2_2_ready,
        "decision": decision,
    }


def _decision(*, status: str, skip_reason: str, missing_pairs: int, failure_count: int) -> str:
    if status == "passed":
        return "Tushare daily_basic passed the small live probe and can move to broader coverage/unit validation."
    if skip_reason == "package_missing":
        return "Install Tushare before validating daily_basic as the v2.2 size source."
    if skip_reason == "auth_missing":
        return "Configure TUSHARE_TOKEN or TS_TOKEN before validating daily_basic as the v2.2 size source."
    if missing_pairs or failure_count:
        return "Do not promote Tushare yet; inspect missing pairs/failures and rerun the probe."
    return "Do not promote Tushare yet; required field coverage is incomplete."


def _load_tushare_module(module: Any | None) -> tuple[Any | None, str]:
    if module is not None:
        return module, ""
    try:
        return importlib.import_module("tushare"), ""
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _format_trade_date(value: Any) -> str:
    digits = str(value).replace("-", "")
    if len(digits) == 8 and digits.isdigit():
        return f"{digits[0:4]}-{digits[4:6]}-{digits[6:8]}"
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--trade-dates", default=",".join(DEFAULT_TRADE_DATES))
    parser.add_argument("--token", default=None, help="Optional Tushare token. Prefer TUSHARE_TOKEN/TS_TOKEN env vars.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", default=str(DEFAULT_RESEARCH_LOG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_v2_tushare_size_probe(
        symbols=args.symbols,
        trade_dates=args.trade_dates,
        token=args.token,
        output_dir=Path(args.output_dir),
        write_research_log=bool(args.write_research_log),
        research_log_path=Path(args.research_log_path),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
