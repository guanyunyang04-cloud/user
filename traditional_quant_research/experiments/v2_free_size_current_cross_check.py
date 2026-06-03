"""Cross-check reconstructed free-source daily_size against current quote sources."""

from __future__ import annotations

import argparse
import importlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from traditional_quant_research.dataset_v2 import load_pit_daily_bars, load_pit_manifest
from traditional_quant_research.experiments.v2_daily_size_audit import CURRENT_CROSS_CHECK_THRESHOLDS
from traditional_quant_research.size_source import (
    AKSHARE_CNINFO_RECONSTRUCTED_SOURCE,
    normalize_cninfo_share_change_events,
    normalize_project_symbol,
    reconstruct_daily_size_from_share_events,
    standardize_daily_size_frame,
)


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/v2_free_size_current_cross_check")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/research_log/2026-06-03_v2_free_size_current_cross_check.md")
DEFAULT_SYMBOLS = ("600000.SH", "000001.SZ", "000002.SZ", "601398.SH")

CROSS_CHECK_COLUMNS = [
    "source",
    "current_source",
    "field",
    "code",
    "date",
    "reconstructed_value",
    "current_value",
    "abs_relative_diff",
    "status",
    "message",
]
QUOTE_COLUMNS = ["current_source", "code", "current_total_market_cap", "current_float_market_cap", "status", "message"]


def run_v2_free_size_current_cross_check(
    *,
    root: str | Path | None = None,
    symbols: str | Sequence[str] = DEFAULT_SYMBOLS,
    as_of_date: str | None = None,
    reconstructed_size_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: str | Path = DEFAULT_RESEARCH_LOG,
    akshare_module: Any | None = None,
    efinance_module: Any | None = None,
) -> dict[str, Any]:
    """Build a diagnostic current-date cross-check artifact."""

    run_id = f"v2_free_size_current_cross_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    symbol_list = _normalize_symbols(symbols)
    resolved_date = _resolve_as_of_date(root=root, as_of_date=as_of_date)
    reconstructed = (
        _read_size_path(Path(reconstructed_size_path))
        if reconstructed_size_path is not None
        else _reconstruct_sample_size(root=root, symbols=symbol_list, as_of_date=resolved_date, akshare_module=akshare_module)
    )
    reconstructed = _filter_reconstructed(reconstructed, symbols=symbol_list, as_of_date=resolved_date)

    ak, ak_error = _load_module("akshare", akshare_module)
    ef, ef_error = _load_module("efinance", efinance_module)
    quote_frames = [
        _akshare_spot_quote(ak, ak_error),
        _akshare_individual_quotes(ak, ak_error, symbol_list),
        _efinance_quote(ef, ef_error, symbol_list),
    ]
    quotes = pd.concat(quote_frames, ignore_index=True) if quote_frames else pd.DataFrame(columns=QUOTE_COLUMNS)
    cross_check = build_current_cross_check(reconstructed, quotes, as_of_date=resolved_date)
    summary = summarize_current_cross_check(
        cross_check,
        quotes,
        reconstructed,
        run_id=run_id,
        symbols=symbol_list,
        as_of_date=resolved_date,
    )
    markdown = render_current_cross_check_markdown(summary, cross_check)

    cross_check.to_csv(run_dir / "daily_size_current_cross_check.csv", index=False, encoding="utf-8-sig")
    quotes.to_csv(run_dir / "current_quote_snapshot.csv", index=False, encoding="utf-8-sig")
    reconstructed.to_csv(run_dir / "reconstructed_daily_size_sample.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        path = Path(research_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
    return {**summary, "run_dir": str(run_dir), "research_log": str(research_log_path) if write_research_log else None}


def build_current_cross_check(
    reconstructed_size: pd.DataFrame,
    current_quotes: pd.DataFrame,
    *,
    as_of_date: str,
) -> pd.DataFrame:
    """Return long-form cross-check rows consumable by v2_daily_size_audit."""

    size = standardize_daily_size_frame(reconstructed_size)
    if not size.empty:
        size = size.loc[pd.to_datetime(size["date"]) == pd.Timestamp(as_of_date)].copy()
    quotes = current_quotes.copy() if current_quotes is not None else pd.DataFrame(columns=QUOTE_COLUMNS)
    for column in QUOTE_COLUMNS:
        if column not in quotes.columns:
            quotes[column] = pd.NA
    if size.empty:
        return pd.DataFrame(
            [
                _cross_row(
                    field=field,
                    code="",
                    date=as_of_date,
                    status="missing_reconstructed_size",
                    message="No reconstructed size rows are available for current cross-check.",
                )
                for field in CURRENT_CROSS_CHECK_THRESHOLDS
            ],
            columns=CROSS_CHECK_COLUMNS,
        )
    quote_values = quotes.loc[quotes["status"].astype(str) == "passed"].copy()
    if quote_values.empty:
        return pd.DataFrame(
            [
                _cross_row(
                    field=field,
                    code=str(row["code"]),
                    date=as_of_date,
                    reconstructed_value=float(row[field]) if pd.notna(row.get(field)) else np.nan,
                    status="current_cross_check_unavailable",
                    message=_quote_failure_message(quotes),
                )
                for row in size.to_dict("records")
                for field in CURRENT_CROSS_CHECK_THRESHOLDS
            ],
            columns=CROSS_CHECK_COLUMNS,
        )
    rows: list[dict[str, Any]] = []
    for size_row in size.to_dict("records"):
        code = str(size_row.get("code", ""))
        code_quotes = quote_values.loc[quote_values["code"].astype(str) == code]
        if code_quotes.empty:
            for field in CURRENT_CROSS_CHECK_THRESHOLDS:
                rows.append(
                    _cross_row(
                        field=field,
                        code=code,
                        date=as_of_date,
                        reconstructed_value=float(size_row.get(field, np.nan)),
                        status="missing_current_quote",
                        message="No current quote row matched this code.",
                    )
                )
            continue
        for quote_row in code_quotes.to_dict("records"):
            for field, current_column in [
                ("total_market_cap", "current_total_market_cap"),
                ("float_market_cap", "current_float_market_cap"),
            ]:
                reconstructed_value = pd.to_numeric(size_row.get(field, np.nan), errors="coerce")
                current_value = pd.to_numeric(quote_row.get(current_column, np.nan), errors="coerce")
                diff = _abs_relative_diff(reconstructed_value, current_value)
                status = "passed" if pd.notna(diff) else "missing_value"
                rows.append(
                    _cross_row(
                        current_source=str(quote_row.get("current_source", "")),
                        field=field,
                        code=code,
                        date=as_of_date,
                        reconstructed_value=float(reconstructed_value) if pd.notna(reconstructed_value) else np.nan,
                        current_value=float(current_value) if pd.notna(current_value) else np.nan,
                        abs_relative_diff=float(diff) if pd.notna(diff) else np.nan,
                        status=status,
                        message="" if status == "passed" else "Current or reconstructed value is missing/non-positive.",
                    )
                )
    return pd.DataFrame(rows, columns=CROSS_CHECK_COLUMNS)


def summarize_current_cross_check(
    cross_check: pd.DataFrame,
    quotes: pd.DataFrame,
    reconstructed: pd.DataFrame,
    *,
    run_id: str,
    symbols: Sequence[str],
    as_of_date: str,
) -> dict[str, Any]:
    metrics = _cross_check_metrics(cross_check)
    ok = bool(metrics and all(item["passed"] for item in metrics))
    unavailable = bool(cross_check.empty or cross_check["abs_relative_diff"].isna().all())
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "symbols": list(symbols),
        "as_of_date": as_of_date,
        "reconstructed_rows": int(len(reconstructed)),
        "quote_rows": int(len(quotes)),
        "cross_check_rows": int(len(cross_check)),
        "available_diff_rows": int(cross_check["abs_relative_diff"].notna().sum()) if "abs_relative_diff" in cross_check else 0,
        "field_metrics": metrics,
        "current_cross_check_ok": ok,
        "source_decision": "current_cross_check_passed" if ok else ("current_cross_check_unavailable" if unavailable else "current_cross_check_failed"),
        "evidence_grade": "diagnostic",
        "candidate_count": 0,
        "limitations": [
            "Current quote sources are cross-check inputs only and are never written into daily_size.",
            "Passing the sample cross-check does not prove full 2016-2026 coverage.",
            "Promotion still requires daily_size coverage/unit/source audit.",
        ],
    }


def render_current_cross_check_markdown(summary: Mapping[str, Any], cross_check: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# V2 Free Size Current Cross-Check",
            "",
            f"- run_id: `{summary.get('run_id', '')}`",
            f"- source_decision: `{summary.get('source_decision', '')}`",
            f"- evidence_grade: `{summary.get('evidence_grade', '')}`",
            f"- current_cross_check_ok: `{summary.get('current_cross_check_ok')}`",
            f"- available_diff_rows: `{summary.get('available_diff_rows', 0)}`",
            f"- candidate_count: `{summary.get('candidate_count', 0)}`",
            "",
            "## Cross-Check Rows",
            "",
            _markdown_table(cross_check),
            "",
            "## Interpretation",
            "",
            "This artifact compares diagnostic reconstructed size against free current quotes. "
            "It is a gate input, not a strategy promotion decision.",
            "",
        ]
    )


def _reconstruct_sample_size(
    *,
    root: str | Path | None,
    symbols: Sequence[str],
    as_of_date: str,
    akshare_module: Any | None,
) -> pd.DataFrame:
    try:
        bars = load_pit_daily_bars(root, start_date=as_of_date, end_date=as_of_date, symbols=symbols)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    if bars.empty:
        return pd.DataFrame()
    ak, _ = _load_module("akshare", akshare_module)
    if ak is None:
        return pd.DataFrame()
    event_frames: list[pd.DataFrame] = []
    for code in symbols:
        try:
            raw = ak.stock_share_change_cninfo(symbol=code.split(".", 1)[0], start_date="19000101", end_date=pd.Timestamp(as_of_date).strftime("%Y%m%d"))
        except Exception:  # noqa: BLE001
            continue
        events = normalize_cninfo_share_change_events(raw, code=code)
        if not events.empty:
            event_frames.append(events)
    share_events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    return reconstruct_daily_size_from_share_events(bars, share_events)


def _akshare_spot_quote(module: Any | None, import_error: str) -> pd.DataFrame:
    source = "akshare.stock_zh_a_spot_em"
    if module is None:
        return _quote_failure(source, "package_missing", import_error)
    try:
        frame = module.stock_zh_a_spot_em()
    except Exception as exc:  # noqa: BLE001
        return _quote_failure(source, type(exc).__name__, str(exc))
    return _normalize_quote_frame(
        frame,
        source=source,
        code_columns=("代码", "code", "股票代码"),
        total_columns=("总市值", "market_cap", "总市值-最新"),
        float_columns=("流通市值", "float_market_cap", "流通市值-最新"),
    )


def _akshare_individual_quotes(module: Any | None, import_error: str, symbols: Sequence[str]) -> pd.DataFrame:
    source = "akshare.stock_individual_info_em"
    if module is None:
        return _quote_failure(source, "package_missing", import_error)
    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    for code in symbols:
        try:
            frame = module.stock_individual_info_em(symbol=code.split(".", 1)[0])
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{code}:{type(exc).__name__}")
            continue
        frames.append(_normalize_individual_info(frame, source=source, code=code))
    valid = [frame for frame in frames if not frame.empty]
    if valid:
        return pd.concat(valid, ignore_index=True)
    return _quote_failure(source, "failed", ";".join(failures) if failures else "No individual quote rows.")


def _efinance_quote(module: Any | None, import_error: str, symbols: Sequence[str]) -> pd.DataFrame:
    source = "efinance.current_quote"
    if module is None:
        return _quote_failure(source, "package_missing", import_error)
    stock = getattr(module, "stock", None)
    getter = getattr(stock, "get_realtime_quotes", None)
    if getter is None:
        return _quote_failure(source, "endpoint_missing", "efinance.stock.get_realtime_quotes not found")
    frame, error = _call_efinance_getter(getter, symbols)
    if frame is None:
        return _quote_failure(source, "failed", error)
    return _normalize_quote_frame(
        frame,
        source=source,
        code_columns=("股票代码", "代码", "code"),
        total_columns=("总市值", "market_cap"),
        float_columns=("流通市值", "float_market_cap"),
    )


def _call_efinance_getter(getter: Any, symbols: Sequence[str]) -> tuple[pd.DataFrame | None, str]:
    code_list = [symbol.split(".", 1)[0] for symbol in symbols]
    attempts: list[tuple[Any, ...]] = [(code_list,), (",".join(code_list),), tuple()]
    errors: list[str] = []
    for args in attempts:
        try:
            frame = getter(*args)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        if frame is not None and not pd.DataFrame(frame).empty:
            return pd.DataFrame(frame), ""
    return None, "; ".join(errors) if errors else "No quote rows returned."


def _normalize_quote_frame(
    frame: pd.DataFrame | None,
    *,
    source: str,
    code_columns: Sequence[str],
    total_columns: Sequence[str],
    float_columns: Sequence[str],
) -> pd.DataFrame:
    if frame is None or pd.DataFrame(frame).empty:
        return _quote_failure(source, "empty_quote", "Quote endpoint returned no rows.")
    raw = pd.DataFrame(frame).copy()
    code_col = _first_column(raw, code_columns)
    total_col = _first_column(raw, total_columns)
    float_col = _first_column(raw, float_columns)
    if not code_col or not total_col or not float_col:
        return _quote_failure(
            source,
            "fields_missing",
            f"Missing required fields. columns={','.join(str(column) for column in raw.columns)}",
        )
    output = pd.DataFrame()
    output["current_source"] = source
    output["code"] = raw[code_col].map(normalize_project_symbol)
    output["current_total_market_cap"] = raw[total_col].map(_parse_money)
    output["current_float_market_cap"] = raw[float_col].map(_parse_money)
    output["status"] = "passed"
    output["message"] = ""
    return output.loc[output["code"] != "", QUOTE_COLUMNS].reset_index(drop=True)


def _normalize_individual_info(frame: pd.DataFrame | None, *, source: str, code: str) -> pd.DataFrame:
    if frame is None or pd.DataFrame(frame).empty:
        return _quote_failure(source, "empty_quote", f"No individual info for {code}.")
    raw = pd.DataFrame(frame)
    if {"item", "value"}.issubset(set(raw.columns)):
        lookup = {str(row["item"]): row["value"] for row in raw.to_dict("records")}
        total = _first_lookup(lookup, ("总市值", "market_cap"))
        floating = _first_lookup(lookup, ("流通市值", "float_market_cap"))
        return pd.DataFrame(
            [
                {
                    "current_source": source,
                    "code": normalize_project_symbol(code),
                    "current_total_market_cap": _parse_money(total),
                    "current_float_market_cap": _parse_money(floating),
                    "status": "passed" if total is not None and floating is not None else "fields_missing",
                    "message": "" if total is not None and floating is not None else "Missing total/float market cap in item/value frame.",
                }
            ],
            columns=QUOTE_COLUMNS,
        )
    normalized = _normalize_quote_frame(
        raw.assign(code=code),
        source=source,
        code_columns=("code", "代码", "股票代码"),
        total_columns=("总市值", "market_cap"),
        float_columns=("流通市值", "float_market_cap"),
    )
    normalized["code"] = normalize_project_symbol(code)
    return normalized


def _quote_failure(source: str, status: str, message: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "current_source": source,
                "code": "",
                "current_total_market_cap": np.nan,
                "current_float_market_cap": np.nan,
                "status": status,
                "message": message,
            }
        ],
        columns=QUOTE_COLUMNS,
    )


def _cross_check_metrics(cross_check: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if cross_check.empty:
        return rows
    for field, threshold in CURRENT_CROSS_CHECK_THRESHOLDS.items():
        values = pd.to_numeric(cross_check.loc[cross_check["field"].astype(str) == field, "abs_relative_diff"], errors="coerce").dropna()
        if values.empty:
            rows.append(
                {
                    "field": field,
                    "row_count": 0,
                    "median_abs_relative_diff": None,
                    "p95_abs_relative_diff": None,
                    "passed": False,
                }
            )
            continue
        median = float(values.median())
        p95 = float(values.quantile(0.95))
        rows.append(
            {
                "field": field,
                "row_count": int(len(values)),
                "median_abs_relative_diff": median,
                "p95_abs_relative_diff": p95,
                "passed": bool(median <= threshold["median"] and p95 <= threshold["p95"]),
            }
        )
    return rows


def _read_size_path(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _filter_reconstructed(frame: pd.DataFrame, *, symbols: Sequence[str], as_of_date: str) -> pd.DataFrame:
    size = standardize_daily_size_frame(frame)
    if size.empty:
        return size
    return size.loc[
        (pd.to_datetime(size["date"]) == pd.Timestamp(as_of_date))
        & size["code"].astype(str).isin(set(symbols))
    ].reset_index(drop=True)


def _resolve_as_of_date(*, root: str | Path | None, as_of_date: str | None) -> str:
    if as_of_date:
        return pd.Timestamp(as_of_date).strftime("%Y-%m-%d")
    manifest = load_pit_manifest(root)
    dataset = manifest.get("dataset", {}) if isinstance(manifest, dict) else {}
    date_max = dataset.get("date_max") or manifest.get("date_max")
    if not date_max:
        raise ValueError("as_of_date is required when manifest has no dataset.date_max")
    return pd.Timestamp(date_max).strftime("%Y-%m-%d")


def _cross_row(
    *,
    current_source: str = "",
    field: str,
    code: str,
    date: str,
    reconstructed_value: float = np.nan,
    current_value: float = np.nan,
    abs_relative_diff: float = np.nan,
    status: str,
    message: str,
) -> dict[str, Any]:
    return {
        "source": AKSHARE_CNINFO_RECONSTRUCTED_SOURCE,
        "current_source": current_source,
        "field": field,
        "code": code,
        "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
        "reconstructed_value": reconstructed_value,
        "current_value": current_value,
        "abs_relative_diff": abs_relative_diff,
        "status": status,
        "message": message,
    }


def _abs_relative_diff(reconstructed_value: Any, current_value: Any) -> float | None:
    reconstructed = pd.to_numeric(reconstructed_value, errors="coerce")
    current = pd.to_numeric(current_value, errors="coerce")
    if pd.isna(reconstructed) or pd.isna(current) or float(current) <= 0 or float(reconstructed) <= 0:
        return None
    return abs(float(reconstructed) - float(current)) / abs(float(current))


def _quote_failure_message(quotes: pd.DataFrame) -> str:
    if quotes.empty:
        return "No current quote endpoints returned rows."
    messages = [str(item) for item in quotes["message"].dropna().astype(str).tolist() if str(item)]
    return "; ".join(messages) or "No passed current quote rows."


def _parse_money(value: Any) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"nan", "none", "--", "-"}:
        return None
    multiplier = 1.0
    if "亿" in text:
        multiplier = 100_000_000.0
    elif "万" in text:
        multiplier = 10_000.0
    cleaned = (
        text.replace("亿元", "")
        .replace("万元", "")
        .replace("元", "")
        .replace("亿", "")
        .replace("万", "")
        .replace(" ", "")
    )
    numeric = pd.to_numeric(cleaned, errors="coerce")
    if pd.isna(numeric):
        return None
    return float(numeric) * multiplier


def _first_column(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    lowered = {str(column).strip().lower(): str(column) for column in frame.columns}
    for column in columns:
        if column in frame.columns:
            return str(column)
        key = str(column).strip().lower()
        if key in lowered:
            return lowered[key]
    return ""


def _first_lookup(lookup: Mapping[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key in lookup:
            return lookup[key]
    return None


def _normalize_symbols(symbols: str | Sequence[str]) -> list[str]:
    raw = [item.strip() for item in symbols.split(",")] if isinstance(symbols, str) else [str(item).strip() for item in symbols]
    return [normalize_project_symbol(item) for item in raw if normalize_project_symbol(item)]


def _load_module(name: str, module: Any | None) -> tuple[Any | None, str]:
    if module is not None:
        return module, ""
    try:
        return importlib.import_module(name), ""
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: f"{float(value):.6f}" if pd.notna(value) else "nan")
    return view.to_markdown(index=False)


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
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="Snapshot root or v2 root containing latest_manifest.json.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--as-of-date", default=None)
    parser.add_argument("--reconstructed-size-path", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", default=str(DEFAULT_RESEARCH_LOG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_v2_free_size_current_cross_check(
        root=args.root,
        symbols=args.symbols,
        as_of_date=args.as_of_date,
        reconstructed_size_path=args.reconstructed_size_path,
        output_dir=args.output_dir,
        write_research_log=bool(args.write_research_log),
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
