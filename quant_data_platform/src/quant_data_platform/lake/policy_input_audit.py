from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from quant_data_platform.lake.catalog import ResearchDataLake
from quant_data_platform.lake.policy_input_loader import DEFAULT_POLICY_INPUT_LAKE_DATASET_ID

EXPECTED_MARKET_FIELDS = ("open", "high", "low", "close", "volume", "amount")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _date_text(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _date_set(values: Iterable[Any]) -> set[str]:
    if values is None:
        return set()
    dates = pd.to_datetime(pd.Index(values), errors="coerce")
    return {_date_text(item) for item in dates if pd.notna(item)}


def _date_bounds(dates: set[str]) -> dict[str, str]:
    if not dates:
        return {"start_date": "", "end_date": ""}
    ordered = sorted(dates)
    return {"start_date": ordered[0], "end_date": ordered[-1]}


def _parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return [str(name) for name in pq.ParquetFile(path).schema.names]
    except Exception:
        return [str(name) for name in pd.read_parquet(path, columns=[]).columns]


def _parquet_row_count(path: Path) -> int:
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return int(len(pd.read_parquet(path)))


def _read_date_column(path: Path, columns: Sequence[str]) -> tuple[str, set[str], int]:
    date_column = "trade_date" if "trade_date" in columns else "date" if "date" in columns else ""
    if not date_column:
        return "", set(), _parquet_row_count(path)
    frame = pd.read_parquet(path, columns=[date_column])
    return date_column, _date_set(frame[date_column]), int(len(frame))


def _normalize_windows(windows: Sequence[tuple[str, str, str] | Mapping[str, Any] | str] | None) -> list[dict[str, str]]:
    if not windows:
        return []
    normalized: list[dict[str, str]] = []
    for item in windows:
        if isinstance(item, str):
            parts = [part.strip() for part in item.split(":")]
            if len(parts) != 3 or not all(parts):
                raise ValueError(f"Invalid audit window {item!r}; expected name:start:end.")
            name, start_date, end_date = parts
        elif isinstance(item, Mapping):
            name = str(item.get("name", "") or item.get("window", "") or "").strip()
            start_date = str(item.get("start_date", "") or item.get("start", "") or "").strip()
            end_date = str(item.get("end_date", "") or item.get("end", "") or "").strip()
        else:
            name, start_date, end_date = [str(part).strip() for part in item]
        if not name or not start_date or not end_date:
            raise ValueError(f"Invalid audit window {item!r}; expected name/start/end.")
        normalized.append({"name": name, "start_date": _date_text(start_date), "end_date": _date_text(end_date)})
    return normalized


def _dates_in_window(dates: set[str], *, start_date: str, end_date: str) -> list[str]:
    start_text = _date_text(start_date)
    end_text = _date_text(end_date)
    return [item for item in sorted(dates) if start_text <= item <= end_text]


def _market_report(path: str) -> tuple[dict[str, Any], set[str], set[str], list[str]]:
    market_path = Path(path)
    if not market_path.exists():
        return {"path": str(market_path), "status": "missing"}, set(), set(), ["missing_market_data"]
    columns = _parquet_columns(market_path)
    selected_columns = [name for name in ("trade_date", "symbol", *EXPECTED_MARKET_FIELDS) if name in columns]
    market = pd.read_parquet(market_path, columns=selected_columns)
    if "trade_date" not in market.columns:
        return {"path": str(market_path), "status": "missing_trade_date", "columns": columns}, set(), set(), ["missing_market_dates"]
    market_dates = _date_set(market["trade_date"])
    symbols = {str(item).strip().upper() for item in market.get("symbol", pd.Series(dtype=str)).dropna().astype(str)}
    field_report: dict[str, Any] = {}
    missing_fields: list[str] = []
    for field in EXPECTED_MARKET_FIELDS:
        if field not in market.columns:
            missing_fields.append(field)
            field_report[field] = {"present": False, "non_null_cells": 0}
        else:
            field_report[field] = {"present": True, "non_null_cells": int(pd.to_numeric(market[field], errors="coerce").notna().sum())}
            if int(field_report[field]["non_null_cells"]) <= 0:
                missing_fields.append(field)
    blockers = ["missing_market_fields"] if missing_fields else []
    report = {
        "path": str(market_path),
        "status": "ok" if not blockers else "blocked",
        "row_count": int(len(market)),
        "date_count": int(len(market_dates)),
        "symbol_count": int(len(symbols)),
        "date_range": _date_bounds(market_dates),
        "fields": field_report,
        "missing_fields": missing_fields,
        "columns": columns,
    }
    return report, market_dates, symbols, blockers


def _benchmark_report(path: str, benchmark: str) -> tuple[dict[str, Any], set[str], set[str], list[str]]:
    benchmark_path = Path(path)
    if not benchmark_path.exists():
        return {"path": str(benchmark_path), "status": "missing"}, set(), set(), ["missing_benchmark"]
    frame = pd.read_parquet(benchmark_path)
    if "trade_date" not in frame.columns:
        return {"path": str(benchmark_path), "status": "missing_trade_date"}, set(), set(), ["missing_benchmark_dates"]
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    if "benchmark" in frame.columns and str(benchmark or "").strip():
        requested = str(benchmark or "").strip().upper()
        matched = frame.loc[frame["benchmark"].astype(str).str.upper() == requested].copy()
        if not matched.empty:
            frame = matched
    close_dates = _date_set(frame.loc[pd.to_numeric(frame.get("close", pd.Series(dtype=float)), errors="coerce").notna(), "trade_date"]) if "close" in frame.columns else set()
    open_dates = _date_set(frame.loc[pd.to_numeric(frame.get("open", pd.Series(dtype=float)), errors="coerce").notna(), "trade_date"]) if "open" in frame.columns else set()
    report = {
        "path": str(benchmark_path),
        "status": "ok" if close_dates else "blocked",
        "row_count": int(len(frame)),
        "benchmark": str(benchmark or ""),
        "has_open_column": bool("open" in frame.columns),
        "close_non_null_rows": int(len(close_dates)),
        "open_non_null_rows": int(len(open_dates)),
        "close_date_range": _date_bounds(close_dates),
        "open_date_range": _date_bounds(open_dates),
        "columns": [str(item) for item in frame.columns],
    }
    return report, close_dates, open_dates, [] if close_dates else ["missing_benchmark"]


def _membership_report(path: str, market_dates: set[str]) -> tuple[dict[str, Any], set[str], list[str]]:
    membership_path = Path(path)
    if not membership_path.exists():
        return {"path": str(membership_path), "status": "missing"}, set(), ["missing_membership"]
    frame = pd.read_parquet(membership_path)
    date_column = "trade_date" if "trade_date" in frame.columns else "date" if "date" in frame.columns else ""
    if date_column:
        dates = _date_set(frame[date_column])
        value_frame = frame.drop(columns=[date_column], errors="ignore")
    elif len(frame) == len(market_dates):
        dates = set(market_dates)
        value_frame = frame
    else:
        dates = set()
        value_frame = frame
    if value_frame.empty:
        true_rows = 0
    else:
        true_rows = int(value_frame.fillna(False).astype(bool).any(axis=1).sum())
    blockers = [] if dates and true_rows > 0 else ["missing_membership"]
    return (
        {
            "path": str(membership_path),
            "status": "ok" if not blockers else "blocked",
            "row_count": int(len(frame)),
            "date_count": int(len(dates)),
            "date_range": _date_bounds(dates),
            "true_rows": true_rows,
            "date_column": date_column,
        },
        dates,
        blockers,
    )


def _feature_report(feature_glob: str, market_dates: set[str], market_symbols: set[str]) -> tuple[dict[str, Any], list[str]]:
    feature_dir = Path(str(feature_glob)).parent if str(feature_glob).endswith("*.parquet") else Path(str(feature_glob))
    if not feature_dir.exists():
        return {"path": str(feature_dir), "status": "missing", "panel_count": 0, "mismatched_panels": []}, ["missing_feature_panels"]
    panels: list[dict[str, Any]] = []
    mismatched: list[dict[str, Any]] = []
    for path in sorted(feature_dir.glob("*.parquet")):
        columns = _parquet_columns(path)
        date_column, dates, row_count = _read_date_column(path, columns)
        symbol_columns = [item for item in columns if item not in {"date", "trade_date"}]
        missing_dates = sorted(market_dates - dates)[:20]
        extra_dates = sorted(dates - market_dates)[:20]
        symbol_count_matches = len(symbol_columns) == len(market_symbols)
        status = "ok" if not missing_dates and not extra_dates and symbol_count_matches else "mismatch"
        panel = {
            "feature_name": path.stem,
            "path": str(path),
            "status": status,
            "row_count": int(row_count),
            "date_column": date_column,
            "date_count": int(len(dates)),
            "date_range": _date_bounds(dates),
            "symbol_column_count": int(len(symbol_columns)),
            "missing_market_dates_sample": missing_dates,
            "extra_dates_sample": extra_dates,
        }
        panels.append(panel)
        if status != "ok":
            mismatched.append(panel)
    warnings = ["feature_panel_mismatch"] if mismatched else []
    return (
        {
            "path": str(feature_dir),
            "status": "ok" if not mismatched else "warning",
            "panel_count": int(len(panels)),
            "mismatched_panel_count": int(len(mismatched)),
            "mismatched_panels": mismatched,
            "panels": panels,
        },
        warnings,
    )


def _sample_feature_nan_scan(
    feature_glob: str,
    *,
    max_panels: int = 5,
    max_value_columns: int = 20,
    max_rows: int = 500,
) -> dict[str, Any]:
    feature_dir = Path(str(feature_glob)).parent if str(feature_glob).endswith("*.parquet") else Path(str(feature_glob))
    if not feature_dir.exists():
        return {
            "enabled": True,
            "status": "missing_feature_panels",
            "feature_nan_cells": 0,
            "feature_scanned_cells": 0,
            "feature_nan_ratio": 0.0,
            "panels": [],
        }
    panels: list[dict[str, Any]] = []
    total_nan_cells = 0
    total_cells = 0
    for path in sorted(feature_dir.glob("*.parquet"))[: int(max_panels)]:
        columns = _parquet_columns(path)
        date_columns = [name for name in ("trade_date", "date") if name in columns]
        value_columns = [name for name in columns if name not in {"trade_date", "date"}][: int(max_value_columns)]
        if not value_columns:
            panels.append({"feature_name": path.stem, "status": "no_value_columns", "nan_cells": 0, "scanned_cells": 0})
            continue
        frame = pd.read_parquet(path, columns=[*date_columns[:1], *value_columns])
        if int(max_rows) > 0 and len(frame) > int(max_rows):
            frame = frame.head(int(max_rows)).copy()
        values = frame[value_columns]
        scanned_cells = int(values.size)
        nan_cells = int(values.isna().sum().sum())
        total_cells += scanned_cells
        total_nan_cells += nan_cells
        panels.append(
            {
                "feature_name": path.stem,
                "status": "completed",
                "row_count": int(len(frame)),
                "value_column_count": int(len(value_columns)),
                "nan_cells": nan_cells,
                "scanned_cells": scanned_cells,
                "nan_ratio": float(nan_cells / scanned_cells) if scanned_cells else 0.0,
            }
        )
    return {
        "enabled": True,
        "status": "completed",
        "max_panels": int(max_panels),
        "max_value_columns": int(max_value_columns),
        "max_rows": int(max_rows),
        "scanned_panel_count": int(len(panels)),
        "feature_nan_cells": int(total_nan_cells),
        "feature_scanned_cells": int(total_cells),
        "feature_nan_ratio": float(total_nan_cells / total_cells) if total_cells else 0.0,
        "panels": panels,
    }


def _window_report(
    *,
    window: Mapping[str, str],
    market_dates: set[str],
    benchmark_close_dates: set[str],
    benchmark_open_dates: set[str],
    membership_dates: set[str],
    require_benchmark_open: bool,
) -> dict[str, Any]:
    start_date = str(window["start_date"])
    end_date = str(window["end_date"])
    window_market_dates = _dates_in_window(market_dates, start_date=start_date, end_date=end_date)
    market_set = set(window_market_dates)
    missing_close = sorted(market_set - benchmark_close_dates)
    missing_open = sorted(market_set - benchmark_open_dates) if require_benchmark_open else []
    missing_membership = sorted(market_set - membership_dates)
    blockers: list[str] = []
    if not window_market_dates:
        blockers.append("missing_market")
    if missing_close:
        blockers.append("missing_benchmark_close")
    if missing_open:
        blockers.append("missing_benchmark_open")
    if missing_membership:
        blockers.append("missing_membership")
    return {
        "name": str(window["name"]),
        "start_date": start_date,
        "end_date": end_date,
        "verdict": "blocked" if blockers else "usable",
        "market_trading_days": int(len(window_market_dates)),
        "blockers": blockers,
        "missing_benchmark_close_count": int(len(missing_close)),
        "missing_benchmark_close_dates": missing_close[:120],
        "missing_benchmark_open_count": int(len(missing_open)),
        "missing_benchmark_open_dates": missing_open[:120],
        "missing_membership_count": int(len(missing_membership)),
        "missing_membership_dates": missing_membership[:120],
    }


def audit_policy_input_bundle(
    *,
    lake: ResearchDataLake,
    dataset_id: str = DEFAULT_POLICY_INPUT_LAKE_DATASET_ID,
    windows: Sequence[tuple[str, str, str] | Mapping[str, Any] | str] | None = None,
    benchmark: str = "000300.SH",
    require_benchmark_open: bool = False,
    sample_nan_scan: bool = False,
) -> dict[str, Any]:
    metadata = lake.describe_dataset(str(dataset_id or DEFAULT_POLICY_INPUT_LAKE_DATASET_ID))
    paths = dict(metadata.get("content_paths", {}) or {})
    blockers: list[str] = []
    warnings: list[str] = []

    market, market_dates, market_symbols, market_blockers = _market_report(str(paths.get("bronze_market_data", "") or ""))
    blockers.extend(market_blockers)
    benchmark_report, benchmark_close_dates, benchmark_open_dates, benchmark_blockers = _benchmark_report(
        str(paths.get("silver_benchmark", "") or ""),
        str(benchmark or metadata.get("benchmark", "") or "000300.SH"),
    )
    blockers.extend(benchmark_blockers)
    membership, membership_dates, membership_blockers = _membership_report(str(paths.get("silver_membership", "") or ""), market_dates)
    blockers.extend(membership_blockers)
    features, feature_warnings = _feature_report(str(paths.get("silver_feature_panels", "") or paths.get("silver_feature_values", "") or ""), market_dates, market_symbols)
    warnings.extend(feature_warnings)

    normalized_windows = _normalize_windows(windows)
    window_reports = [
        _window_report(
            window=window,
            market_dates=market_dates,
            benchmark_close_dates=benchmark_close_dates,
            benchmark_open_dates=benchmark_open_dates,
            membership_dates=membership_dates,
            require_benchmark_open=bool(require_benchmark_open),
        )
        for window in normalized_windows
    ]
    for item in window_reports:
        blockers.extend(str(blocker) for blocker in item.get("blockers", []))

    if bool(require_benchmark_open) and not benchmark_report.get("has_open_column", False):
        blockers.append("missing_benchmark_open")

    unique_blockers = sorted(set(blockers))
    unique_warnings = sorted(set(warnings))
    verdict = "blocked" if unique_blockers else "usable_with_warnings" if unique_warnings else "usable"
    report = {
        "stage": "policy_input_bundle_audit",
        "dataset_id": str(metadata.get("dataset_id", dataset_id)),
        "dataset_kind": str(metadata.get("dataset_kind", "")),
        "verdict": verdict,
        "blockers": unique_blockers,
        "warnings": unique_warnings,
        "parameters": dict(metadata.get("parameters", {}) or {}),
        "row_counts": dict(metadata.get("row_counts", {}) or {}),
        "content_paths": paths,
        "market": market,
        "benchmark": benchmark_report,
        "membership": membership,
        "features": features,
        "windows": window_reports,
        "sample_nan_scan": _sample_feature_nan_scan(
            str(paths.get("silver_feature_panels", "") or paths.get("silver_feature_values", "") or "")
        )
        if bool(sample_nan_scan)
        else {"enabled": False, "status": "skipped"},
        "shadow_only": True,
        "promotion_allowed": False,
        "active_execution_strategy_expected_diff": "none",
    }
    return _json_ready(report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a policy input bundle without loading full feature matrices.")
    parser.add_argument("--dataset-id", default=DEFAULT_POLICY_INPUT_LAKE_DATASET_ID)
    parser.add_argument("--data-lake-root", default="")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument(
        "--windows",
        default="",
        help="Comma-separated windows in name:start:end form, e.g. path20_stage1:2018-01-01:2024-12-31.",
    )
    parser.add_argument("--require-benchmark-open", action="store_true")
    parser.add_argument("--sample-nan-scan", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    windows = [item.strip() for item in str(args.windows or "").split(",") if item.strip()]
    report = audit_policy_input_bundle(
        lake=ResearchDataLake(str(args.data_lake_root or "").strip() or None),
        dataset_id=str(args.dataset_id),
        windows=windows,
        benchmark=str(args.benchmark),
        require_benchmark_open=bool(args.require_benchmark_open),
        sample_nan_scan=bool(args.sample_nan_scan),
    )
    if str(args.output or "").strip():
        output_path = Path(str(args.output)).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    text = json.dumps(report, ensure_ascii=False, indent=2 if args.json else None)
    print(text)
    return 0 if str(report.get("verdict")) != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
