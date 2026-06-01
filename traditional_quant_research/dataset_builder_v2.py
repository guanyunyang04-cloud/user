"""Build v2 point-in-time Shanghai/Shenzhen mainboard daily datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .baostock_source import BaostockSource, BaostockSourceError
from .dataset_v2 import DEFAULT_V2_SNAPSHOT_ROOT
from .universe import is_sh_sz_mainboard_a_share


PIT_BIAS_STATEMENT = (
    "v2 constructs the universe date by date from Baostock point-in-time stock lists, "
    "security master dates, daily tradestatus, and daily isST flags. It is intended "
    "to reduce survivorship bias versus v1, while remaining limited by Baostock field "
    "coverage and historical data quality."
)

BAOSTOCK_DAILY_FIELDS = (
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "tradestatus",
    "isST",
)


@dataclass(frozen=True)
class PitBuildConfig:
    output_root: Path = DEFAULT_V2_SNAPSHOT_ROOT
    start_date: str = "2016-01-01"
    end_date: str = "2026-06-01"
    symbols: tuple[str, ...] = ()
    max_symbols: int = 0
    command: str = "build"
    snapshot_id: str = ""


def _baostock_to_std_code(code: str) -> str:
    market, symbol = str(code).split(".", 1)
    suffix = {"sh": "SH", "sz": "SZ"}.get(market.lower(), market.upper())
    return f"{symbol}.{suffix}"


def _std_to_baostock_code(code: str) -> str:
    symbol, suffix = str(code).upper().split(".", 1)
    market = {"SH": "sh", "SZ": "sz"}[suffix]
    return f"{market}.{symbol}"


def _normalize_date(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return str(pd.to_datetime(text).date())


def _trade_dates(source: BaostockSource, start_date: str, end_date: str) -> pd.DataFrame:
    raw = source.query_trade_dates(start_date, end_date)
    if raw.empty:
        return pd.DataFrame(columns=["date", "is_trading_day"])
    frame = raw.rename(columns={"calendar_date": "date", "is_trading_day": "is_trading_day"}).copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["is_trading_day"] = frame["is_trading_day"].astype(str) == "1"
    return frame.loc[frame["is_trading_day"], ["date", "is_trading_day"]].reset_index(drop=True)


def _security_master(source: BaostockSource, codes: list[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for code in codes:
        raw = source.query_stock_basic(_std_to_baostock_code(code))
        if raw.empty:
            parts.append(pd.DataFrame([{"code": code, "basic_error": "empty_stock_basic"}]))
            continue
        item = raw.iloc[0].to_dict()
        parts.append(
            pd.DataFrame(
                [
                    {
                        "code": code,
                        "baostock_code": item.get("code", ""),
                        "name": item.get("code_name", ""),
                        "ipo_date": _normalize_date(item.get("ipoDate", "")),
                        "out_date": _normalize_date(item.get("outDate", "")),
                        "security_type": str(item.get("type", "") or ""),
                        "status": str(item.get("status", "") or ""),
                        "basic_error": "",
                    }
                ]
            )
        )
    if not parts:
        return pd.DataFrame(columns=["code", "baostock_code", "name", "ipo_date", "out_date", "security_type", "status", "basic_error"])
    return pd.concat(parts, ignore_index=True)


def _daily_stock_lists(source: BaostockSource, dates: pd.Series, explicit_symbols: set[str] | None) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for date in dates:
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        raw = source.query_all_stock(day=date_str)
        if raw.empty:
            continue
        frame = raw.copy()
        frame["date"] = pd.Timestamp(date)
        frame["code"] = frame["code"].map(_baostock_to_std_code)
        frame["name_on_date"] = frame["code_name"].astype(str)
        frame["query_all_trade_status"] = frame["tradeStatus"].astype(str)
        frame = frame[["date", "code", "name_on_date", "query_all_trade_status"]]
        frame = frame[frame["code"].map(is_sh_sz_mainboard_a_share)]
        if explicit_symbols is not None:
            frame = frame[frame["code"].isin(explicit_symbols)]
        parts.append(frame)
    if not parts:
        return pd.DataFrame(columns=["date", "code", "name_on_date", "query_all_trade_status"])
    return pd.concat(parts, ignore_index=True).drop_duplicates(["date", "code"])


def _daily_bars(source: BaostockSource, codes: list[str], start_date: str, end_date: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    parts: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    for code in codes:
        try:
            raw = source.query_daily_bars(_std_to_baostock_code(code), start_date, end_date)
        except BaostockSourceError as exc:
            failures.append({"code": code, "kind": "daily_fetch_error", "error": str(exc)})
            continue
        if raw.empty:
            failures.append({"code": code, "kind": "empty_daily_bars"})
            continue
        frame = raw.copy()
        frame["code"] = frame["code"].map(_baostock_to_std_code)
        frame["date"] = pd.to_datetime(frame["date"])
        for field in ("open", "high", "low", "close", "volume", "amount"):
            frame[field] = pd.to_numeric(frame[field], errors="coerce")
        frame["tradestatus"] = frame["tradestatus"].astype(str)
        frame["isST"] = frame["isST"].astype(str)
        frame["source"] = "baostock"
        parts.append(frame[[*BAOSTOCK_DAILY_FIELDS, "source"]])
    if not parts:
        return pd.DataFrame(columns=[*BAOSTOCK_DAILY_FIELDS, "source"]), failures
    return pd.concat(parts, ignore_index=True), failures


def build_daily_universe(
    stock_lists: pd.DataFrame,
    security_master: pd.DataFrame,
    daily_bars: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = stock_lists.merge(security_master, on="code", how="left")
    bar_flags = daily_bars[["date", "code", "open", "high", "low", "close", "volume", "amount", "tradestatus", "isST"]].copy()
    merged = merged.merge(bar_flags, on=["date", "code"], how="left")

    date_series = pd.to_datetime(merged["date"])
    ipo_dates = pd.to_datetime(merged["ipo_date"], errors="coerce")
    out_dates = pd.to_datetime(merged["out_date"], errors="coerce")
    has_bar = merged[["open", "high", "low", "close"]].notna().any(axis=1)
    volume = pd.to_numeric(merged["volume"], errors="coerce")
    amount = pd.to_numeric(merged["amount"], errors="coerce")
    is_stock_type = merged["security_type"].fillna("") == "1"
    is_active_status = merged["status"].fillna("") != "0"
    is_listed = (ipo_dates.isna() | (ipo_dates <= date_series)) & (out_dates.isna() | (date_series < out_dates))
    is_mainboard = merged["code"].map(is_sh_sz_mainboard_a_share)
    is_trade_status_ok = merged["tradestatus"].fillna(merged["query_all_trade_status"]).astype(str) == "1"
    is_st = merged["isST"].fillna("0").astype(str) == "1"
    is_suspended = (~is_trade_status_ok) | (~has_bar) | volume.isna() | amount.isna() | (volume <= 0) | (amount <= 0)

    merged["is_listed_on_date"] = is_listed.astype(bool)
    merged["is_mainboard"] = is_mainboard.astype(bool)
    merged["is_common_a_share"] = (is_stock_type & is_active_status).astype(bool)
    merged["is_st_on_date"] = is_st.astype(bool)
    merged["has_bar"] = has_bar.astype(bool)
    merged["is_suspended_on_date"] = is_suspended.astype(bool)
    merged["is_tradeable"] = (
        merged["is_listed_on_date"]
        & merged["is_mainboard"]
        & merged["is_common_a_share"]
        & (~merged["is_st_on_date"])
        & (~merged["is_suspended_on_date"])
    ).astype(bool)
    merged["reject_reason"] = _reject_reasons(merged)

    universe_cols = [
        "date",
        "code",
        "name_on_date",
        "ipo_date",
        "out_date",
        "security_type",
        "status",
        "is_listed_on_date",
        "is_mainboard",
        "is_common_a_share",
        "is_st_on_date",
        "is_suspended_on_date",
        "has_bar",
        "is_tradeable",
        "reject_reason",
    ]
    status = merged[["date", "code", "has_bar", "is_suspended_on_date", "is_tradeable"]].rename(
        columns={"is_suspended_on_date": "is_suspended_like"}
    )
    return merged[universe_cols].sort_values(["date", "code"]).reset_index(drop=True), status.sort_values(["date", "code"]).reset_index(drop=True)


def _reject_reasons(frame: pd.DataFrame) -> list[str]:
    reasons: list[str] = []
    for row in frame.itertuples(index=False):
        if not bool(row.is_listed_on_date):
            reasons.append("not_listed_on_date")
        elif not bool(row.is_mainboard):
            reasons.append("not_mainboard")
        elif not bool(row.is_common_a_share):
            reasons.append("not_common_a_share")
        elif bool(row.is_st_on_date):
            reasons.append("st_on_date")
        elif bool(row.is_suspended_on_date):
            reasons.append("suspended_on_date")
        else:
            reasons.append("")
    return reasons


def build_snapshot(config: PitBuildConfig) -> dict[str, Any]:
    started_at = datetime.now().isoformat(timespec="seconds")
    snapshot_id = config.snapshot_id or f"baostock_daily_mainboard_v2_pit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    snapshot_root = config.output_root / snapshot_id
    snapshot_root.mkdir(parents=True, exist_ok=False)

    with BaostockSource() as source:
        trade_dates = _trade_dates(source, config.start_date, config.end_date)
        explicit = set(config.symbols) if config.symbols else None
        stock_lists = _daily_stock_lists(source, trade_dates["date"], explicit)
        codes = sorted(stock_lists["code"].unique().tolist())
        if config.max_symbols > 0:
            codes = codes[: config.max_symbols]
            stock_lists = stock_lists[stock_lists["code"].isin(codes)].reset_index(drop=True)
        security_master = _security_master(source, codes)
        daily_bars, failures = _daily_bars(source, codes, config.start_date, config.end_date)
        daily_universe, daily_status = build_daily_universe(stock_lists, security_master, daily_bars)
        manifest = _manifest(
            config=config,
            snapshot_id=snapshot_id,
            snapshot_root=snapshot_root,
            source_version=source.version,
            started_at=started_at,
            trade_dates=trade_dates,
            security_master=security_master,
            daily_universe=daily_universe,
            daily_bars=daily_bars,
            daily_status=daily_status,
            failures=failures,
        )

    _write_outputs(snapshot_root, config.output_root, security_master, stock_lists, daily_universe, daily_bars, daily_status, manifest, failures)
    return manifest


def _manifest(
    *,
    config: PitBuildConfig,
    snapshot_id: str,
    snapshot_root: Path,
    source_version: str,
    started_at: str,
    trade_dates: pd.DataFrame,
    security_master: pd.DataFrame,
    daily_universe: pd.DataFrame,
    daily_bars: pd.DataFrame,
    daily_status: pd.DataFrame,
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "snapshot_id": snapshot_id,
        "snapshot_path": str(snapshot_root),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "started_at": started_at,
        "command": config.command,
        "source": {
            "name": "baostock",
            "version": source_version,
            "cache_policy": "remote_read_only_no_local_mutation",
        },
        "dataset": {
            "frequency": "1d",
            "start_date_requested": config.start_date,
            "end_date_requested": config.end_date,
            "date_min": str(daily_bars["date"].min().date()) if not daily_bars.empty else "",
            "date_max": str(daily_bars["date"].max().date()) if not daily_bars.empty else "",
            "trade_date_count": int(len(trade_dates)),
            "security_count": int(len(security_master)),
            "daily_universe_rows": int(len(daily_universe)),
            "daily_bar_rows": int(len(daily_bars)),
            "daily_status_rows": int(len(daily_status)),
            "fields": list(BAOSTOCK_DAILY_FIELDS),
        },
        "quality": {
            "failure_count": len(failures),
            "tradeable_rows": int(daily_status["is_tradeable"].sum()) if not daily_status.empty else 0,
            "suspended_like_rows": int(daily_status["is_suspended_like"].sum()) if not daily_status.empty else 0,
            "st_rows": int(daily_universe["is_st_on_date"].sum()) if not daily_universe.empty else 0,
        },
        "bias_statement": PIT_BIAS_STATEMENT,
    }


def _write_outputs(
    snapshot_root: Path,
    output_root: Path,
    security_master: pd.DataFrame,
    raw_daily_stock_lists: pd.DataFrame,
    daily_universe: pd.DataFrame,
    daily_bars: pd.DataFrame,
    daily_status: pd.DataFrame,
    manifest: dict[str, Any],
    failures: list[dict[str, Any]],
) -> None:
    security_master.to_parquet(snapshot_root / "security_master.parquet", index=False)
    raw_daily_stock_lists.to_parquet(snapshot_root / "raw_daily_stock_lists.parquet", index=False)
    daily_universe.to_parquet(snapshot_root / "daily_universe.parquet", index=False)
    daily_bars.to_parquet(snapshot_root / "daily_bars.parquet", index=False)
    daily_status.to_parquet(snapshot_root / "daily_status.parquet", index=False)
    (snapshot_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (snapshot_root / "quality_report.json").write_text(json.dumps({"failure_count": len(failures), "failures": failures}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
    parser = argparse.ArgumentParser(description="Build Baostock point-in-time daily research snapshots.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("dry-run", "build", "refresh", "rebuild"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--symbols", default="", help="Comma-separated symbols such as 600000.SH,000001.SZ.")
        cmd.add_argument("--max-symbols", type=int, default=0, help="Limit selected symbols after PIT universe discovery.")
        cmd.add_argument("--output-root", default=str(DEFAULT_V2_SNAPSHOT_ROOT))
        cmd.add_argument("--start-date", default="2016-01-01")
        cmd.add_argument("--end-date", default="2026-06-01")
        cmd.add_argument("--snapshot-id", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    symbols = _parse_symbols(args.symbols)
    if args.command == "dry-run" and not symbols and args.max_symbols <= 0:
        symbols = ("600000.SH", "000001.SZ")
    config = PitBuildConfig(
        output_root=Path(args.output_root),
        start_date=args.start_date,
        end_date=args.end_date,
        symbols=symbols,
        max_symbols=args.max_symbols,
        command=args.command,
        snapshot_id=args.snapshot_id,
    )
    manifest = build_snapshot(config)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
