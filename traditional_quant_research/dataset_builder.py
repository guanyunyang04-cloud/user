"""Build the v1 Shanghai/Shenzhen mainboard daily research dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .data_source import DEFAULT_TQCENTER_PATH, TqDataSource, TqDataSourceConfig, TqDataSourceError
from .dataset import DEFAULT_SNAPSHOT_ROOT
from .universe import filter_sh_sz_mainboard_a_shares, is_active_common_stock_info


BIAS_STATEMENT = (
    "v1 uses the current mainboard non-ST universe from tqcenter stock metadata. "
    "It is suitable for diagnostic/backtest research on the current universe, "
    "but it is not a point-in-time survivorship-bias-free full-market dataset."
)
DAILY_FIELDS = ("open", "high", "low", "close", "volume", "amount", "forward_factor")
FIELD_ALIASES = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "Amount": "amount",
    "ForwardFactor": "forward_factor",
}


@dataclass(frozen=True)
class BuildConfig:
    output_root: Path = DEFAULT_SNAPSHOT_ROOT
    tqcenter_path: Path = DEFAULT_TQCENTER_PATH
    start_date: str = "19900101"
    end_date: str = "20260601"
    batch_size: int = 50
    symbols: tuple[str, ...] = ()
    command: str = "build"
    snapshot_id: str = ""


def _chunked(values: list[str], size: int) -> Iterable[list[str]]:
    if size <= 0:
        raise ValueError("batch_size must be positive")
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _market_from_code(code: str) -> str:
    return code.split(".", 1)[1]


def _board_from_code(code: str) -> str:
    return "sh_mainboard" if code.endswith(".SH") else "sz_mainboard"


def build_universe(source: TqDataSource, explicit_symbols: list[str] | None = None) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    raw_codes = explicit_symbols if explicit_symbols is not None else source.get_stock_list()
    mainboard_codes = filter_sh_sz_mainboard_a_shares(raw_codes)
    records: list[dict[str, Any]] = []
    selected: list[str] = []
    errors: list[dict[str, str]] = []

    for code in mainboard_codes:
        try:
            info = source.get_stock_info(code)
        except TqDataSourceError as exc:
            records.append(_universe_record(code, {}, selected=False, reject_reason="stock_info_error"))
            errors.append({"code": code, "error": str(exc)})
            continue
        is_selected = is_active_common_stock_info(info)
        reject_reason = "" if is_selected else _reject_reason(info)
        records.append(_universe_record(code, info, selected=is_selected, reject_reason=reject_reason))
        if is_selected:
            selected.append(code)

    frame = pd.DataFrame.from_records(records)
    summary = {
        "raw_count": len(raw_codes),
        "mainboard_code_count": len(mainboard_codes),
        "selected_count": len(selected),
        "rejected_count": len(records) - len(selected),
        "stock_info_error_count": len(errors),
        "stock_info_errors": errors,
    }
    return frame, selected, summary


def _universe_record(code: str, info: dict[str, Any], *, selected: bool, reject_reason: str) -> dict[str, Any]:
    return {
        "code": code,
        "name": str(info.get("Name", "") or ""),
        "market": _market_from_code(code),
        "board": _board_from_code(code),
        "list_date": str(info.get("J_start", "") or ""),
        "hs_stock_kind": str(info.get("HSStockKind", "") or ""),
        "selected": bool(selected),
        "reject_reason": reject_reason,
    }


def _reject_reason(info: dict[str, Any]) -> str:
    name = str(info.get("Name", "") or "").upper().replace(" ", "")
    if "ST" in name:
        return "st"
    for key in ("IsZS", "IsHKGP", "IsQH", "IsQQ"):
        if str(info.get(key, "") or "0") == "1":
            return key.lower()
    return "non_active_common_stock"


def _normalize_daily_payload(
    payload: dict[str, Any],
    codes: list[str],
    trading_dates: pd.DatetimeIndex | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    normalized: dict[str, pd.DataFrame] = {}
    missing_fields: list[str] = []
    for source_field, target_field in FIELD_ALIASES.items():
        value = payload.get(source_field)
        if value is None:
            missing_fields.append(source_field)
            continue
        frame = pd.DataFrame(value).copy()
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        if trading_dates is not None:
            frame = frame.reindex(index=trading_dates)
        frame = frame.reindex(columns=codes)
        frame = frame.apply(pd.to_numeric, errors="coerce")
        normalized[target_field] = frame

    if "close" not in normalized:
        index = trading_dates if trading_dates is not None else pd.DatetimeIndex([])
        status = _empty_status(index, codes)
        return pd.DataFrame(columns=["date", "code", *DAILY_FIELDS, "source"]), status, missing_fields

    index = trading_dates if trading_dates is not None else normalized["close"].index
    records: list[pd.DataFrame] = []
    for field in DAILY_FIELDS:
        frame = normalized.get(field)
        if frame is None:
            frame = pd.DataFrame(index=index, columns=codes, dtype="float64")
        else:
            frame = frame.reindex(index=index, columns=codes)
        stacked = frame.stack(future_stack=True).rename(field).reset_index()
        stacked.columns = ["date", "code", field]
        records.append(stacked)

    daily = records[0]
    for field_frame in records[1:]:
        daily = daily.merge(field_frame, on=["date", "code"], how="outer")
    daily["source"] = "tqcenter"
    daily = daily.sort_values(["code", "date"]).reset_index(drop=True)

    has_bar = daily[["open", "high", "low", "close"]].notna().any(axis=1)
    volume = pd.to_numeric(daily["volume"], errors="coerce")
    amount = pd.to_numeric(daily["amount"], errors="coerce")
    suspended = (~has_bar) | volume.isna() | amount.isna() | (volume <= 0) | (amount <= 0)
    status = daily[["date", "code"]].copy()
    status["has_bar"] = has_bar.astype(bool)
    status["is_suspended_like"] = suspended.astype(bool)
    status["is_tradeable"] = (~suspended).astype(bool)

    daily = daily.loc[has_bar].reset_index(drop=True)
    return daily, status.reset_index(drop=True), missing_fields


def _empty_status(index: pd.DatetimeIndex, codes: list[str]) -> pd.DataFrame:
    if len(index) == 0 or not codes:
        return pd.DataFrame(columns=["date", "code", "has_bar", "is_suspended_like", "is_tradeable"])
    grid = pd.MultiIndex.from_product([index, codes], names=["date", "code"]).to_frame(index=False)
    grid["has_bar"] = False
    grid["is_suspended_like"] = True
    grid["is_tradeable"] = False
    return grid


def _trading_date_index(source: TqDataSource, *, start_date: str, end_date: str) -> pd.DatetimeIndex:
    raw_dates = source.get_trading_dates(market="SH", start_time=start_date, end_time=end_date)
    dates = pd.to_datetime(raw_dates, format="%Y%m%d", errors="coerce").dropna()
    return pd.DatetimeIndex(dates).tz_localize(None).sort_values().unique()


def _fetch_daily_with_retry(
    source: TqDataSource,
    codes: list[str],
    *,
    start_date: str,
    end_date: str,
    trading_dates: pd.DatetimeIndex | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    try:
        payload = source.get_daily_bars(codes, start_time=start_date, end_time=end_date)
        daily, status, missing = _normalize_daily_payload(payload, codes, trading_dates)
        if daily.empty and len(codes) > 1:
            raise TqDataSourceError("empty batch payload")
        if missing:
            failures.append({"codes": codes, "kind": "missing_fields", "fields": missing})
        return daily, status, failures
    except Exception as exc:  # noqa: BLE001
        if len(codes) == 1:
            failures.append({"codes": codes, "kind": "fetch_error", "error": str(exc)})
            return pd.DataFrame(), pd.DataFrame(), failures

    daily_parts: list[pd.DataFrame] = []
    status_parts: list[pd.DataFrame] = []
    for code in codes:
        daily, status, single_failures = _fetch_daily_with_retry(
            source,
            [code],
            start_date=start_date,
            end_date=end_date,
            trading_dates=trading_dates,
        )
        failures.extend(single_failures)
        if not daily.empty:
            daily_parts.append(daily)
        if not status.empty:
            status_parts.append(status)
    return _concat(daily_parts), _concat(status_parts), failures


def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def build_snapshot(config: BuildConfig) -> dict[str, Any]:
    started_at = datetime.now().isoformat(timespec="seconds")
    snapshot_id = config.snapshot_id or f"tq_daily_mainboard_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    snapshot_root = config.output_root / snapshot_id
    snapshot_root.mkdir(parents=True, exist_ok=False)
    source_config = TqDataSourceConfig(tqcenter_path=config.tqcenter_path)
    failures: list[dict[str, Any]] = []

    with TqDataSource(source_config) as source:
        explicit = list(config.symbols) if config.symbols else None
        universe, selected, universe_summary = build_universe(source, explicit)
        trading_dates = _trading_date_index(source, start_date=config.start_date, end_date=config.end_date)
        daily_parts: list[pd.DataFrame] = []
        status_parts: list[pd.DataFrame] = []
        for batch in _chunked(selected, config.batch_size):
            daily, status, batch_failures = _fetch_daily_with_retry(
                source,
                batch,
                start_date=config.start_date,
                end_date=config.end_date,
                trading_dates=trading_dates,
            )
            failures.extend(batch_failures)
            if not daily.empty:
                daily_parts.append(daily)
            if not status.empty:
                status_parts.append(status)

        daily_bars = _concat(daily_parts)
        daily_status = _concat(status_parts)
        manifest = _manifest(
            config=config,
            snapshot_id=snapshot_id,
            snapshot_root=snapshot_root,
            source=source,
            started_at=started_at,
            universe_summary=universe_summary,
            daily_bars=daily_bars,
            daily_status=daily_status,
            failures=failures,
            trading_date_count=len(trading_dates),
        )

    _write_outputs(snapshot_root, config.output_root, universe, daily_bars, daily_status, manifest, failures)
    return manifest


def _manifest(
    *,
    config: BuildConfig,
    snapshot_id: str,
    snapshot_root: Path,
    source: TqDataSource,
    started_at: str,
    universe_summary: dict[str, Any],
    daily_bars: pd.DataFrame,
    daily_status: pd.DataFrame,
    failures: list[dict[str, Any]],
    trading_date_count: int,
) -> dict[str, Any]:
    date_min = str(daily_bars["date"].min().date()) if not daily_bars.empty else ""
    date_max = str(daily_bars["date"].max().date()) if not daily_bars.empty else ""
    return {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "snapshot_path": str(snapshot_root),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "started_at": started_at,
        "command": config.command,
        "source": {
            "tqcenter_path": str(config.tqcenter_path),
            "dll_path": source.dll_path,
            "run_id": source.run_id,
            "run_mode": source.run_mode,
            "cache_policy": "read_existing_tdx_cache_only",
        },
        "dataset": {
            "frequency": "1d",
            "start_date_requested": config.start_date,
            "end_date_requested": config.end_date,
            "date_min": date_min,
            "date_max": date_max,
            "fields": list(DAILY_FIELDS),
            "daily_bar_rows": int(len(daily_bars)),
            "daily_status_rows": int(len(daily_status)),
            "trading_date_count": int(trading_date_count),
            "batch_size": config.batch_size,
        },
        "universe": universe_summary,
        "quality": {
            "failure_count": len(failures),
            "tradeable_rows": int(daily_status["is_tradeable"].sum()) if not daily_status.empty else 0,
            "suspended_like_rows": int(daily_status["is_suspended_like"].sum()) if not daily_status.empty else 0,
        },
        "bias_statement": BIAS_STATEMENT,
    }


def _write_outputs(
    snapshot_root: Path,
    output_root: Path,
    universe: pd.DataFrame,
    daily_bars: pd.DataFrame,
    daily_status: pd.DataFrame,
    manifest: dict[str, Any],
    failures: list[dict[str, Any]],
) -> None:
    universe.to_csv(snapshot_root / "universe.csv", index=False, encoding="utf-8-sig")
    universe.to_parquet(snapshot_root / "universe.parquet", index=False)
    daily_bars.to_parquet(snapshot_root / "daily_bars.parquet", index=False)
    daily_status.to_parquet(snapshot_root / "daily_status.parquet", index=False)
    (snapshot_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    quality = {"failure_count": len(failures), "failures": failures}
    (snapshot_root / "quality_report.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    latest = {
        "snapshot_id": manifest["snapshot_id"],
        "snapshot_path": str(snapshot_root),
        "manifest_path": str(snapshot_root / "manifest.json"),
        "created_at": manifest["created_at"],
    }
    (output_root / "latest_manifest.json").write_text(json.dumps(latest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_symbols(raw: str) -> tuple[str, ...]:
    if not raw.strip():
        return ()
    return tuple(item.strip().upper() for item in raw.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build traditional quant daily research snapshots.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("dry-run", "build", "refresh", "rebuild"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--symbols", default="", help="Comma-separated explicit symbols for smoke builds.")
        cmd.add_argument("--output-root", default=str(DEFAULT_SNAPSHOT_ROOT))
        cmd.add_argument("--tqcenter-path", default=str(DEFAULT_TQCENTER_PATH))
        cmd.add_argument("--start-date", default="19900101")
        cmd.add_argument("--end-date", default="20260601")
        cmd.add_argument("--batch-size", type=int, default=50)
        cmd.add_argument("--snapshot-id", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    symbols = _parse_symbols(args.symbols)
    if args.command == "dry-run" and not symbols:
        symbols = ("600000.SH", "000001.SZ")
    config = BuildConfig(
        output_root=Path(args.output_root),
        tqcenter_path=Path(args.tqcenter_path),
        start_date=args.start_date,
        end_date=args.end_date,
        batch_size=args.batch_size,
        symbols=symbols,
        command=args.command,
        snapshot_id=args.snapshot_id,
    )
    manifest = build_snapshot(config)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
