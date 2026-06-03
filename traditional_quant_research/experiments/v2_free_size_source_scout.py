"""Scout free/public size data sources before v2.2 size promotion."""

from __future__ import annotations

import argparse
import importlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from traditional_quant_research.size_source import (
    AKSHARE_CNINFO_RECONSTRUCTED_SOURCE,
    PROXY_AMOUNT_SOURCE,
    normalize_cninfo_share_change_events,
    normalize_project_symbol,
)


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/v2_free_size_source_scout")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/research_log/2026-06-03_v2_free_size_source_scout.md")
DEFAULT_SYMBOLS = ("600000.SH", "000001.SZ", "000002.SZ", "601398.SH")
DEFAULT_TRADE_DATES = ("20170103", "20180102", "20220104", "20230103", "20260601")

SOURCE_COLUMNS = (
    "source",
    "package",
    "endpoint",
    "status",
    "field_count",
    "fields_found",
    "historical_capability",
    "pit_timing_claim",
    "usable_for_daily_size",
    "promotion_eligible",
    "blocker",
)


def run_v2_free_size_source_scout(
    *,
    symbols: str | Sequence[str] = DEFAULT_SYMBOLS,
    trade_dates: str | Sequence[str] = DEFAULT_TRADE_DATES,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: str | Path = DEFAULT_RESEARCH_LOG,
    akshare_module: Any | None = None,
    efinance_module: Any | None = None,
) -> dict[str, Any]:
    """Probe free/public source shapes and write a conservative decision record."""

    run_id = f"v2_free_size_source_scout_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    symbol_list = _normalize_symbols(symbols)
    trade_date_list = _normalize_trade_dates(trade_dates)
    ak, ak_error = _load_module("akshare", akshare_module)
    ef, ef_error = _load_module("efinance", efinance_module)

    source_rows = [
        _probe_akshare_spot(ak, ak_error),
        _probe_akshare_individual_info(ak, ak_error, symbol_list[0] if symbol_list else "600000.SH"),
        _probe_akshare_cninfo_share_change(ak, ak_error, symbol_list),
        _probe_efinance_current_quote(ef, ef_error, symbol_list),
        _proxy_amount_row(),
    ]
    source_matrix = pd.DataFrame(source_rows, columns=SOURCE_COLUMNS)
    summary = summarize_free_size_scout(
        source_matrix,
        run_id=run_id,
        symbols=symbol_list,
        trade_dates=trade_date_list,
    )
    markdown = render_free_size_scout_markdown(summary, source_matrix)

    source_matrix.to_csv(run_dir / "free_size_source_matrix.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        path = Path(research_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")

    return {**summary, "run_dir": str(run_dir), "research_log": str(research_log_path) if write_research_log else None}


def summarize_free_size_scout(
    source_matrix: pd.DataFrame,
    *,
    run_id: str,
    symbols: Sequence[str],
    trade_dates: Sequence[str],
) -> dict[str, Any]:
    usable = source_matrix.loc[source_matrix["usable_for_daily_size"].eq(True)].copy()
    promotion_eligible = source_matrix.loc[source_matrix["promotion_eligible"].eq(True)].copy()
    free_reconstructed_ready = bool((usable["source"] == AKSHARE_CNINFO_RECONSTRUCTED_SOURCE).any())
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "symbols": list(symbols),
        "trade_dates": list(trade_dates),
        "free_source_count": int(len(source_matrix)),
        "usable_for_daily_size_count": int(len(usable)),
        "promotion_eligible_count": int(len(promotion_eligible)),
        "free_reconstructed_ready": free_reconstructed_ready,
        "decision": "free_source_scout_only",
        "recommended_size_source": (
            "akshare_cninfo_reconstructed_diagnostic_only"
            if free_reconstructed_ready
            else "proxy_amount_diagnostic_only"
        ),
        "candidate_count": 0,
        "limitations": [
            "Realtime public quote endpoints can cross-check current market cap but cannot prove historical PIT size.",
            "CNInfo share-change events are diagnostic reconstruction inputs and must not satisfy promotion size_gate.",
            "Proxy amount is diagnostic only and must not satisfy promotion gates.",
        ],
    }


def render_free_size_scout_markdown(summary: Mapping[str, Any], source_matrix: pd.DataFrame) -> str:
    lines = [
        "# V2 Free Size Source Scout",
        "",
        f"- run_id: `{summary.get('run_id', '')}`",
        f"- decision: `{summary.get('decision', '')}`",
        f"- recommended_size_source: `{summary.get('recommended_size_source', '')}`",
        f"- usable_for_daily_size_count: `{summary.get('usable_for_daily_size_count', 0)}`",
        f"- promotion_eligible_count: `{summary.get('promotion_eligible_count', 0)}`",
        f"- candidate_count: `{summary.get('candidate_count', 0)}`",
        "",
        "## Source Matrix",
        "",
        _markdown_table(source_matrix),
        "",
        "## Interpretation",
        "",
        "This scout checks free/public source shapes. It does not fetch a full PIT size table and does not promote a strategy candidate. "
        "A source can be usable for `daily_size` reconstruction while still being ineligible for promotion until coverage, unit and PIT timing audits pass.",
        "",
    ]
    return "\n".join(lines)


def _probe_akshare_spot(module: Any | None, import_error: str) -> dict[str, Any]:
    if module is None:
        return _source_row("akshare.stock_zh_a_spot_em", "akshare", "stock_zh_a_spot_em", "package_missing", [], "current_only", "none", False, False, import_error)
    try:
        frame = module.stock_zh_a_spot_em()
    except Exception as exc:  # noqa: BLE001
        return _source_row("akshare.stock_zh_a_spot_em", "akshare", "stock_zh_a_spot_em", "failed", [], "current_only", "none", False, False, str(exc))
    fields = _matching_fields(frame, ("总市值", "流通市值", "市值", "流通股本"))
    return _source_row(
        "akshare.stock_zh_a_spot_em",
        "akshare",
        "stock_zh_a_spot_em",
        "passed" if fields else "fields_missing",
        fields,
        "current_only",
        "not_pit",
        False,
        False,
        "Realtime quote endpoint; use only for current cross-check.",
    )


def _probe_akshare_individual_info(module: Any | None, import_error: str, symbol: str) -> dict[str, Any]:
    if module is None:
        return _source_row("akshare.stock_individual_info_em", "akshare", "stock_individual_info_em", "package_missing", [], "current_only", "none", False, False, import_error)
    try:
        frame = module.stock_individual_info_em(symbol=symbol.split(".", 1)[0])
    except Exception as exc:  # noqa: BLE001
        return _source_row("akshare.stock_individual_info_em", "akshare", "stock_individual_info_em", "failed", [], "current_only", "none", False, False, str(exc))
    fields = _matching_fields(frame, ("总市值", "流通市值", "总股本", "流通股"))
    return _source_row(
        "akshare.stock_individual_info_em",
        "akshare",
        "stock_individual_info_em",
        "passed" if fields else "fields_missing",
        fields,
        "current_only",
        "not_pit",
        False,
        False,
        "Single-stock current info; use only for spot cross-check.",
    )


def _probe_akshare_cninfo_share_change(module: Any | None, import_error: str, symbols: Sequence[str]) -> dict[str, Any]:
    if module is None:
        return _source_row(AKSHARE_CNINFO_RECONSTRUCTED_SOURCE, "akshare", "stock_share_change_cninfo", "package_missing", [], "historical_share_events", "unknown", False, False, import_error)
    event_frames: list[pd.DataFrame] = []
    failures: list[str] = []
    for symbol in symbols[:2]:
        try:
            raw = module.stock_share_change_cninfo(symbol=symbol.split(".", 1)[0], start_date="19000101", end_date="20260601")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{symbol}:{type(exc).__name__}")
            continue
        events = normalize_cninfo_share_change_events(raw, code=symbol)
        if not events.empty:
            event_frames.append(events)
    fields = ["date", "total_share", "float_share"] if event_frames else []
    return _source_row(
        AKSHARE_CNINFO_RECONSTRUCTED_SOURCE,
        "akshare",
        "stock_share_change_cninfo",
        "passed" if event_frames else "failed",
        fields,
        "historical_share_events",
        "effective_date_needs_audit",
        bool(event_frames),
        False,
        "Share events can reconstruct daily_size only after coverage and PIT timing audit."
        if event_frames
        else f"No parseable share events in probe. failures={';'.join(failures)}",
    )


def _probe_efinance_current_quote(module: Any | None, import_error: str, symbols: Sequence[str]) -> dict[str, Any]:
    if module is None:
        return _source_row("efinance.current_quote", "efinance", "stock.get_realtime_quotes", "package_missing", [], "current_only", "none", False, False, import_error)
    stock = getattr(module, "stock", None)
    getter = getattr(stock, "get_realtime_quotes", None)
    if getter is None:
        return _source_row("efinance.current_quote", "efinance", "stock.get_realtime_quotes", "endpoint_missing", [], "current_only", "none", False, False, "efinance.stock.get_realtime_quotes not found")
    frame, error = _call_realtime_quote_getter(getter, symbols)
    if frame is None:
        return _source_row("efinance.current_quote", "efinance", "stock.get_realtime_quotes", "failed", [], "current_only", "none", False, False, error)
    fields = _matching_fields(frame, ("总市值", "流通市值", "市值", "流通股本"))
    return _source_row("efinance.current_quote", "efinance", "stock.get_realtime_quotes", "passed" if fields else "fields_missing", fields, "current_only", "not_pit", False, False, "Realtime/current quote endpoint; use only for cross-check.")


def _call_realtime_quote_getter(getter: Any, symbols: Sequence[str]) -> tuple[pd.DataFrame | None, str]:
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
    return None, "; ".join(errors) if errors else "No realtime quote rows returned."


def _proxy_amount_row() -> dict[str, Any]:
    return _source_row(PROXY_AMOUNT_SOURCE, "local", "daily_bars.amount", "available", ["amount"], "historical_proxy", "local_pit_bars", True, False, "Diagnostic proxy only; cannot satisfy size_gate.")


def _source_row(
    source: str,
    package: str,
    endpoint: str,
    status: str,
    fields: Sequence[str],
    historical_capability: str,
    pit_timing_claim: str,
    usable_for_daily_size: bool,
    promotion_eligible: bool,
    blocker: str,
) -> dict[str, Any]:
    return {
        "source": source,
        "package": package,
        "endpoint": endpoint,
        "status": status,
        "field_count": int(len(fields)),
        "fields_found": ",".join(str(field) for field in fields),
        "historical_capability": historical_capability,
        "pit_timing_claim": pit_timing_claim,
        "usable_for_daily_size": bool(usable_for_daily_size),
        "promotion_eligible": bool(promotion_eligible),
        "blocker": blocker,
    }


def _matching_fields(frame: pd.DataFrame | None, needles: Sequence[str]) -> list[str]:
    if frame is None or frame.empty:
        return []
    fields = [str(column) for column in frame.columns]
    if {"item", "value"}.issubset(set(fields)):
        fields.extend(str(item) for item in frame["item"].dropna().astype(str).tolist())
    return sorted({field for field in fields for needle in needles if needle in field})


def _normalize_symbols(symbols: str | Sequence[str]) -> list[str]:
    raw = [item.strip() for item in symbols.split(",")] if isinstance(symbols, str) else [str(item).strip() for item in symbols]
    return [normalize_project_symbol(item) for item in raw if normalize_project_symbol(item)]


def _normalize_trade_dates(trade_dates: str | Sequence[str]) -> list[str]:
    raw = [item.strip() for item in trade_dates.split(",")] if isinstance(trade_dates, str) else [str(item).strip() for item in trade_dates]
    return [item.replace("-", "") for item in raw if item.strip()]


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
    return frame.head(max_rows).to_markdown(index=False)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--trade-dates", default=",".join(DEFAULT_TRADE_DATES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", default=str(DEFAULT_RESEARCH_LOG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_v2_free_size_source_scout(
        symbols=args.symbols,
        trade_dates=args.trade_dates,
        output_dir=args.output_dir,
        write_research_log=bool(args.write_research_log),
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
