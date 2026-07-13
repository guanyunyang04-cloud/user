from __future__ import annotations

import multiprocessing
import queue as queue_module
import shutil
import time
from hashlib import sha256
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from quant_data_platform.core.json_io import read_json
from quant_data_platform.domains.contracts import DataDomain, DatePartitionFetchRequest, DomainFetchRequest
from quant_data_platform.providers import (
    BAOSTOCK_BATCH_VERSION,
    BAOSTOCK_BATCH_WHEEL_SHA256,
    BAOSTOCK_BULK_PER_PAGE_COUNT,
    BaostockProvider,
    _baostock_bulk_query_to_frame,
    _fetch_baostock_stock_basic_frame_with_timeout,
    _quiet_baostock_call,
)
from quant_data_platform.qdp_v3.constants import BAOSTOCK_DAILY_FIELDS
from quant_data_platform.qdp_v3.identity import SecurityIdentityRegistry, board_for_symbol, normalize_symbol
from quant_data_platform.qdp_v3.manifest import atomic_write_json, sha256_file, utc_now
from quant_data_platform.qdp_v3.paths import ensure_qdp_v3_layout, qdp_v3_paths
from quant_data_platform.qdp_v3.storage import atomic_write_parquet, frame_content_sha256


GOLDEN_VERSION = "0.9.1"
GATE_FILENAME = "baostock_0_9_3_compatibility_gate.json"
GOLDEN_MANIFEST_FILENAME = "golden_0_9_1.json"
MANDATORY_SYMBOLS = ("300114.SZ", "302132.SZ", "600076.SH", "600000.SH")
EXACT_FIELDS = ("open", "high", "low", "close", "preclose", "volume", "amount", "adjustflag", "tradestatus", "pctChg", "isST")
VALUATION_FIELDS = ("turn", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM")


def compatibility_gate_path(workspace_root: str | Path | None = None) -> Path:
    return qdp_v3_paths(workspace_root).metadata / GATE_FILENAME


def golden_manifest_path(workspace_root: str | Path | None = None) -> Path:
    return qdp_v3_paths(workspace_root).compatibility / GOLDEN_MANIFEST_FILENAME


def assert_baostock_compatibility_gate(workspace_root: str | Path | None = None) -> dict[str, Any]:
    from quant_data_platform.core.json_io import read_json

    payload = read_json(compatibility_gate_path(workspace_root))
    if payload.get("status") != "passed" or payload.get("scope") != "full":
        raise RuntimeError("baostock_0_9_3_full_compatibility_gate_not_passed")
    if str(payload.get("golden_version", "")) != GOLDEN_VERSION or str(payload.get("runtime_version", "")) != BAOSTOCK_BATCH_VERSION:
        raise RuntimeError("baostock_compatibility_gate_version_mismatch")
    if str(payload.get("wheel_sha256", "")).lower() != BAOSTOCK_BATCH_WHEEL_SHA256:
        raise RuntimeError("baostock_compatibility_gate_wheel_hash_mismatch")
    return payload


def _symbol_query_worker(queue: Any, symbols: tuple[str, ...], trade_date: str) -> None:
    try:
        import baostock as bs  # type: ignore
        from quant_data_platform.providers import _baostock_query_to_frame, _to_baostock_code

        login = _quiet_baostock_call(bs.login)
        if str(getattr(login, "error_code", "1")) != "0":
            raise RuntimeError(f"baostock_login_failed:{getattr(login, 'error_msg', '')}")
        rows: list[pd.DataFrame] = []
        try:
            fields = ",".join(BAOSTOCK_DAILY_FIELDS)
            for symbol in symbols:
                last_error: Exception | None = None
                frame = pd.DataFrame()
                for attempt, delay in enumerate((0, 2, 5), start=1):
                    if delay:
                        time.sleep(delay)
                    try:
                        query = bs.query_history_k_data_plus(
                            _to_baostock_code(symbol),
                            fields,
                            start_date=str(trade_date),
                            end_date=str(trade_date),
                            frequency="d",
                            adjustflag="3",
                        )
                        frame = _baostock_query_to_frame(query, "baostock_compatibility_symbol_daily")
                        last_error = None
                        break
                    except Exception as exc:  # pragma: no cover - exercised against the live provider
                        last_error = exc
                if last_error is not None:
                    raise RuntimeError(
                        f"baostock_compatibility_symbol_failed:{symbol}:{type(last_error).__name__}:{last_error}"
                    ) from last_error
                if not frame.empty:
                    frame["requested_symbol"] = normalize_symbol(symbol)
                    rows.append(frame)
        finally:
            _quiet_baostock_call(bs.logout)
        queue.put({"status": "ok", "data": pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=[*BAOSTOCK_DAILY_FIELDS, "requested_symbol"])})
    except Exception as exc:
        queue.put({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})


def _query_symbols_for_date(symbols: Iterable[str], trade_date: str, *, timeout_seconds: int = 600) -> pd.DataFrame:
    normalized = tuple(sorted({normalize_symbol(item) for item in symbols if normalize_symbol(item)}))
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(target=_symbol_query_worker, args=(result_queue, normalized, str(trade_date)))
    process.start()
    process.join(max(1, int(timeout_seconds)))
    if process.is_alive():
        process.terminate()
        process.join(5)
        raise TimeoutError(f"baostock_compatibility_symbol_query_timeout:{trade_date}")
    try:
        payload = result_queue.get(timeout=5)
    except queue_module.Empty as exc:
        raise RuntimeError(f"baostock_compatibility_symbol_query_no_payload:{trade_date}:exit={process.exitcode}") from exc
    finally:
        result_queue.close()
    if payload.get("status") != "ok":
        raise RuntimeError(f"baostock_compatibility_symbol_query_failed:{payload.get('error_type')}:{payload.get('error')}")
    return payload["data"]


def _query_symbols_for_gate(
    symbols: Iterable[str],
    trade_date: str,
    *,
    workspace_root: str | Path | None,
    timeout_seconds: int = 60,
) -> pd.DataFrame:
    """Resumable 0.9.3 legacy-query proof with failure isolation.

    A stuck BaoStock socket must not discard dozens of already-proved
    symbols.  Fixed groups of eight are cached by exact symbol/date/version
    token.  If a group times out, it is recursively split until the failing
    symbol is isolated and reported.
    """

    paths = ensure_qdp_v3_layout(workspace_root)
    normalized = sorted({normalize_symbol(item) for item in symbols if normalize_symbol(item)})
    cache_root = paths.compatibility / ".gate_partial" / BAOSTOCK_BATCH_VERSION / str(trade_date)

    def fetch_group(group: list[str]) -> pd.DataFrame:
        token = sha256((BAOSTOCK_BATCH_VERSION + "|" + str(trade_date) + "|" + "|".join(group)).encode("utf-8")).hexdigest()
        path = cache_root / f"{token[:20]}.parquet"
        meta_path = cache_root / f"{token[:20]}.json"
        meta = read_json(meta_path) if meta_path.exists() else {}
        if (
            path.exists()
            and meta.get("token") == token
            and meta.get("symbols") == group
            and meta.get("parquet_sha256") == sha256_file(path)
        ):
            cached = pd.read_parquet(path, engine="pyarrow")
            if meta.get("content_sha256") == frame_content_sha256(cached):
                return cached
        try:
            frame = _query_symbols_for_date(group, trade_date, timeout_seconds=timeout_seconds)
        except Exception:
            if len(group) <= 1:
                raise
            middle = len(group) // 2
            frame = pd.concat([fetch_group(group[:middle]), fetch_group(group[middle:])], ignore_index=True)
        cache_root.mkdir(parents=True, exist_ok=True)
        atomic_write_parquet(path, frame)
        atomic_write_json(
            meta_path,
            {
                "token": token,
                "trade_date": str(trade_date),
                "symbols": group,
                "runtime_version": BAOSTOCK_BATCH_VERSION,
                "row_count": int(len(frame)),
                "content_sha256": frame_content_sha256(frame),
                "parquet_sha256": sha256_file(path),
                "captured_at": utc_now(),
            },
        )
        return frame

    frames = [fetch_group(normalized[index : index + 8]) for index in range(0, len(normalized), 8)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[*BAOSTOCK_DAILY_FIELDS, "requested_symbol"])


def _anchor_dates(calendar: pd.DataFrame, *, start_year: int, end_date: str) -> list[str]:
    frame = calendar.copy()
    frame["trade_date"] = frame["trade_date"].astype(str).str.slice(0, 10)
    opened = frame["is_open"].astype(str).str.lower().isin({"1", "true", "t", "yes"})
    frame = frame.loc[opened & frame["trade_date"].le(str(end_date))]
    frame["year"] = frame["trade_date"].str.slice(0, 4).astype(int)
    anchors: list[str] = []
    for year, group in frame.loc[frame["year"].ge(int(start_year))].groupby("year", sort=True):
        dates = sorted(set(group["trade_date"]))
        if dates:
            anchors.extend((dates[0], dates[-1]))
    return sorted(set(anchors))


def _select_symbols(universe: pd.DataFrame, *, trade_date: str, count: int) -> list[str]:
    symbols = universe.get("symbol", pd.Series(dtype=str)).map(normalize_symbol)
    data = pd.DataFrame({"symbol": symbols}).loc[symbols.ne("")].drop_duplicates("symbol")
    if data.empty:
        return list(MANDATORY_SYMBOLS)
    data["exchange"] = data["symbol"].str.rsplit(".", n=1).str[-1]
    data["board"] = data["symbol"].map(board_for_symbol)
    if "is_st" in universe.columns:
        data = data.merge(pd.DataFrame({"symbol": symbols, "is_st": universe["is_st"].astype(str)}), on="symbol", how="left")
    else:
        data["is_st"] = "unknown"
    data["rank"] = data["symbol"].map(lambda symbol: __import__("hashlib").sha256(f"{trade_date}|{symbol}".encode("utf-8")).hexdigest())
    buckets = {key: group.sort_values("rank")["symbol"].tolist() for key, group in data.groupby(["exchange", "board", "is_st"], dropna=False)}
    selected = [symbol for symbol in MANDATORY_SYMBOLS]
    positions = {key: 0 for key in buckets}
    while len(set(selected)) < int(count):
        progressed = False
        for key in sorted(buckets, key=str):
            position = positions[key]
            if position >= len(buckets[key]):
                continue
            selected.append(buckets[key][position])
            positions[key] += 1
            progressed = True
            if len(set(selected)) >= int(count):
                break
        if not progressed:
            break
    return sorted(set(selected))[: max(int(count), len(MANDATORY_SYMBOLS))]


def capture_baostock_0_9_1_golden(
    *,
    end_date: str,
    workspace_root: str | Path | None = None,
    start_year: int = 2010,
    sample_count: int = 64,
    smoke: bool = False,
) -> dict[str, Any]:
    runtime = importlib_metadata.version("baostock")
    if runtime != GOLDEN_VERSION:
        raise RuntimeError(f"golden_capture_requires_baostock_{GOLDEN_VERSION}:installed={runtime}")
    paths = ensure_qdp_v3_layout(workspace_root)
    provider = BaostockProvider()
    calendar = provider.fetch_domain(DomainFetchRequest(domain=DataDomain.TRADING_CALENDAR, start_date=f"{int(start_year)}-01-01", end_date=str(end_date), exchange="SSE")).data
    anchors = _anchor_dates(calendar, start_year=int(start_year), end_date=str(end_date))
    if smoke and len(anchors) > 3:
        anchors = [anchors[0], anchors[len(anchors) // 2], anchors[-1]]
    scope = "smoke" if smoke else "full"
    effective_sample_count = 8 if smoke else int(sample_count)
    fixture_dir = paths.compatibility / ("golden_0_9_1_smoke" if smoke else "golden_0_9_1")
    fixture_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = paths.compatibility / ".capture_partial" / scope
    partial_dir.mkdir(parents=True, exist_ok=True)
    progress_path = paths.compatibility / f"golden_0_9_1_{scope}_progress.json"
    progress: dict[str, Any] = {
        "status": "capturing",
        "scope": scope,
        "golden_version": runtime,
        "start_year": int(start_year),
        "end_date": str(end_date),
        "sample_count": effective_sample_count,
        "anchors": anchors,
        "anchor_progress": {},
        "updated_at": utc_now(),
    }
    if progress_path.exists():
        from quant_data_platform.core.json_io import read_json

        existing = read_json(progress_path)
        signature = (existing.get("scope"), existing.get("golden_version"), existing.get("start_year"), existing.get("end_date"), existing.get("sample_count"), existing.get("anchors"))
        expected = (scope, runtime, int(start_year), str(end_date), effective_sample_count, anchors)
        if signature == expected:
            progress = existing
    atomic_write_json(progress_path, progress)
    fixtures: list[dict[str, Any]] = []
    for trade_date in anchors:
        anchor_progress = dict((progress.get("anchor_progress", {}) or {}).get(trade_date, {}) or {})
        symbols = [normalize_symbol(item) for item in anchor_progress.get("symbols", []) if normalize_symbol(item)]
        if not symbols:
            universe = provider.fetch_domain(DomainFetchRequest(domain=DataDomain.UNIVERSE_SNAPSHOT, start_date=trade_date, end_date=trade_date)).data
            symbols = _select_symbols(universe, trade_date=trade_date, count=effective_sample_count)
            anchor_progress = {"symbols": symbols, "chunks": {}, "status": "capturing"}
            progress.setdefault("anchor_progress", {})[trade_date] = anchor_progress
            progress["updated_at"] = utc_now()
            atomic_write_json(progress_path, progress)

        chunk_frames: list[pd.DataFrame] = []
        chunks = [symbols[index : index + 8] for index in range(0, len(symbols), 8)]
        for chunk_index, chunk_symbols in enumerate(chunks):
            chunk_key = f"{chunk_index:03d}"
            chunk_token = sha256((trade_date + "|" + "|".join(chunk_symbols)).encode("utf-8")).hexdigest()
            chunk_path = partial_dir / trade_date / f"{chunk_key}-{chunk_token[:16]}.parquet"
            saved = dict((anchor_progress.get("chunks", {}) or {}).get(chunk_key, {}) or {})
            reusable = (
                saved.get("token") == chunk_token
                and saved.get("symbols") == chunk_symbols
                and chunk_path.exists()
                and saved.get("parquet_sha256") == sha256_file(chunk_path)
            )
            if reusable:
                chunk_frame = pd.read_parquet(chunk_path, engine="pyarrow")
                reusable = saved.get("content_sha256") == frame_content_sha256(chunk_frame)
            if not reusable:
                chunk_frame = _query_symbols_for_date(chunk_symbols, trade_date, timeout_seconds=300)
                chunk_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_parquet(chunk_path, chunk_frame)
                anchor_progress.setdefault("chunks", {})[chunk_key] = {
                    "token": chunk_token,
                    "symbols": chunk_symbols,
                    "row_count": int(len(chunk_frame)),
                    "content_sha256": frame_content_sha256(chunk_frame),
                    "parquet_sha256": sha256_file(chunk_path),
                    "path": str(chunk_path.relative_to(paths.root)).replace("\\", "/"),
                    "completed_at": utc_now(),
                }
                progress.setdefault("anchor_progress", {})[trade_date] = anchor_progress
                progress["updated_at"] = utc_now()
                atomic_write_json(progress_path, progress)
            chunk_frames.append(chunk_frame)
        frame = pd.concat(chunk_frames, ignore_index=True) if chunk_frames else pd.DataFrame(columns=[*BAOSTOCK_DAILY_FIELDS, "requested_symbol"])
        path = fixture_dir / f"{trade_date}.parquet"
        atomic_write_parquet(path, frame)
        fixture = {
            "trade_date": trade_date,
            "symbols": symbols,
            "row_count": int(len(frame)),
            "content_sha256": frame_content_sha256(frame),
            "parquet_sha256": sha256_file(path),
            "path": str(path.relative_to(paths.root)).replace("\\", "/"),
        }
        fixtures.append(fixture)
        anchor_progress["status"] = "captured"
        anchor_progress["fixture"] = fixture
        progress.setdefault("anchor_progress", {})[trade_date] = anchor_progress
        progress["updated_at"] = utc_now()
        atomic_write_json(progress_path, progress)
    payload = {
        "status": "captured",
        "scope": scope,
        "golden_version": runtime,
        "captured_at": utc_now(),
        "start_year": int(start_year),
        "end_date": str(end_date),
        "sample_count": effective_sample_count,
        "anchor_count": len(anchors),
        "fixtures": fixtures,
    }
    target = golden_manifest_path(workspace_root) if not smoke else paths.compatibility / "golden_0_9_1_smoke.json"
    atomic_write_json(target, payload)
    progress["status"] = "captured"
    progress["manifest_path"] = str(target.resolve())
    progress["updated_at"] = utc_now()
    atomic_write_json(progress_path, progress)
    shutil.rmtree(partial_dir, ignore_errors=True)
    return {**payload, "manifest_path": str(target.resolve())}


def _numeric_equal(left: Any, right: Any, *, relative_tolerance: float) -> bool:
    if str(left).strip() == "" and str(right).strip() == "":
        return True
    try:
        a = float(left)
        b = float(right)
    except (TypeError, ValueError):
        return str(left).strip() == str(right).strip()
    if not np.isfinite(a) or not np.isfinite(b):
        return False
    denominator = max(abs(a), abs(b), np.finfo(float).tiny)
    return abs(a - b) / denominator <= float(relative_tolerance)


def _compare_fixture(golden: pd.DataFrame, symbol_current: pd.DataFrame, batch: pd.DataFrame, *, trade_date: str, workspace_root: str | Path | None) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    all_symbols = set(golden.get("requested_symbol", pd.Series(dtype=str)).map(normalize_symbol)) | set(batch.get("code", pd.Series(dtype=str)).map(normalize_symbol))
    registry = SecurityIdentityRegistry.from_sources(provider_symbols=all_symbols, workspace_root=workspace_root)
    current_by_symbol = {symbol: group.iloc[0] for symbol, group in symbol_current.groupby("requested_symbol", sort=False)} if not symbol_current.empty else {}
    batch_copy = batch.copy()
    batch_copy["provider_symbol"] = batch_copy.get("code", pd.Series(dtype=str)).map(normalize_symbol)
    batch_copy["security_id"] = batch_copy["provider_symbol"].map(registry.security_id_for_provider_symbol)
    batch_by_security = {security_id: group.iloc[0] for security_id, group in batch_copy.groupby("security_id", sort=False) if security_id}
    for golden_row in golden.to_dict("records"):
        requested = normalize_symbol(golden_row.get("requested_symbol", ""))
        current_row = current_by_symbol.get(requested)
        if current_row is None:
            issues.append({"code": "current_symbol_query_row_missing", "trade_date": trade_date, "symbol": requested})
            continue
        security_id = registry.security_id_for_provider_symbol(requested)
        batch_row = batch_by_security.get(security_id)
        symbol_on_date = registry.symbol_for_date(security_id, trade_date) if security_id else ""
        compare_batch = not symbol_on_date or requested == symbol_on_date
        if batch_row is None and compare_batch:
            issues.append({"code": "batch_identity_row_missing", "trade_date": trade_date, "symbol": requested, "security_id": security_id})
        for field in EXACT_FIELDS:
            if not _numeric_equal(golden_row.get(field), current_row.get(field), relative_tolerance=0.0):
                issues.append({"code": "old_new_symbol_value_mismatch", "trade_date": trade_date, "symbol": requested, "field": field, "golden": golden_row.get(field), "current": current_row.get(field)})
            if compare_batch and batch_row is not None and not _numeric_equal(golden_row.get(field), batch_row.get(field), relative_tolerance=0.0):
                issues.append({"code": "golden_batch_value_mismatch", "trade_date": trade_date, "symbol": requested, "field": field, "golden": golden_row.get(field), "batch": batch_row.get(field)})
        for field in VALUATION_FIELDS:
            if not _numeric_equal(golden_row.get(field), current_row.get(field), relative_tolerance=1e-8):
                issues.append({"code": "old_new_valuation_mismatch", "trade_date": trade_date, "symbol": requested, "field": field})
            if compare_batch and batch_row is not None and not _numeric_equal(golden_row.get(field), batch_row.get(field), relative_tolerance=1e-8):
                issues.append({"code": "golden_batch_valuation_mismatch", "trade_date": trade_date, "symbol": requested, "field": field})
    return issues


class _BulkFixtureResult:
    def __init__(self, size: int) -> None:
        self.error_code = "0"
        self.error_msg = ""
        self.fields = ["value"]
        self.data = [[str(index)] for index in range(size)]
        self.per_page_count = BAOSTOCK_BULK_PER_PAGE_COUNT

    def next(self) -> bool:  # pragma: no cover - must never be called
        raise AssertionError("bulk_parser_called_next")


def _bulk_parser_regression() -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for size in (1999, 2000, 2001):
        frame = _baostock_bulk_query_to_frame(_BulkFixtureResult(size), "compatibility_bulk_fixture")
        if len(frame) != size:
            issues.append({"code": "bulk_parser_row_count_mismatch", "size": size, "actual": len(frame)})
    try:
        _baostock_bulk_query_to_frame(_BulkFixtureResult(20000), "compatibility_bulk_fixture")
    except RuntimeError as exc:
        if "potential_truncation" not in str(exc):
            issues.append({"code": "bulk_20000_wrong_failure", "error": str(exc)})
    else:
        issues.append({"code": "bulk_20000_not_blocked"})
    return issues


def _batch_daily_for_gate(
    provider: BaostockProvider,
    *,
    trade_date: str,
    workspace_root: str | Path | None,
) -> pd.DataFrame:
    paths = ensure_qdp_v3_layout(workspace_root)
    cache_root = paths.compatibility / ".gate_batch" / BAOSTOCK_BATCH_VERSION
    path = cache_root / f"{trade_date}.parquet"
    meta_path = cache_root / f"{trade_date}.json"
    meta = read_json(meta_path)
    if path.exists() and meta.get("parquet_sha256") == sha256_file(path):
        cached = pd.read_parquet(path, engine="pyarrow")
        if meta.get("content_sha256") == frame_content_sha256(cached):
            return cached
    result = provider.fetch_date_partition(
        DatePartitionFetchRequest(
            domain=DataDomain.MARKET_DAILY,
            trade_date=trade_date,
            universe_kind="all_a",
            fetch_mode="date_snapshot",
        )
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(path, result.raw_data)
    atomic_write_json(
        meta_path,
        {
            "trade_date": trade_date,
            "runtime_version": BAOSTOCK_BATCH_VERSION,
            "wheel_sha256": BAOSTOCK_BATCH_WHEEL_SHA256,
            "row_count": int(len(result.raw_data)),
            "content_sha256": frame_content_sha256(result.raw_data),
            "parquet_sha256": sha256_file(path),
            "coverage_report": result.coverage_report,
            "captured_at": utc_now(),
        },
    )
    return result.raw_data


def _stock_basic_for_gate(*, trade_date: str, workspace_root: str | Path | None) -> pd.DataFrame:
    paths = ensure_qdp_v3_layout(workspace_root)
    cache_root = paths.compatibility / ".gate_batch" / BAOSTOCK_BATCH_VERSION
    path = cache_root / f"stock_basic_{trade_date}.parquet"
    meta_path = cache_root / f"stock_basic_{trade_date}.json"
    meta = read_json(meta_path)
    if path.exists() and meta.get("parquet_sha256") == sha256_file(path):
        cached = pd.read_parquet(path, engine="pyarrow")
        if meta.get("content_sha256") == frame_content_sha256(cached):
            return cached
    frame = _fetch_baostock_stock_basic_frame_with_timeout(trade_date=trade_date, timeout_seconds=180)
    cache_root.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(path, frame)
    atomic_write_json(
        meta_path,
        {
            "trade_date": trade_date,
            "runtime_version": BAOSTOCK_BATCH_VERSION,
            "row_count": int(len(frame)),
            "content_sha256": frame_content_sha256(frame),
            "parquet_sha256": sha256_file(path),
            "captured_at": utc_now(),
        },
    )
    return frame


def run_baostock_0_9_3_compatibility_gate(
    *,
    workspace_root: str | Path | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    from quant_data_platform.core.json_io import read_json

    runtime = importlib_metadata.version("baostock")
    if runtime != BAOSTOCK_BATCH_VERSION:
        raise RuntimeError(f"compatibility_gate_requires_baostock_{BAOSTOCK_BATCH_VERSION}:installed={runtime}")
    paths = ensure_qdp_v3_layout(workspace_root)
    manifest_path = paths.compatibility / ("golden_0_9_1_smoke.json" if smoke else GOLDEN_MANIFEST_FILENAME)
    golden_manifest = read_json(manifest_path)
    if smoke:
        full_manifest = read_json(paths.compatibility / GOLDEN_MANIFEST_FILENAME)
        if full_manifest.get("status") == "captured" and full_manifest.get("scope") == "full":
            full_fixtures = list(full_manifest.get("fixtures", []) or [])
            selected = (
                [full_fixtures[0], full_fixtures[len(full_fixtures) // 2], full_fixtures[-1]]
                if len(full_fixtures) > 3
                else full_fixtures
            )
            golden_manifest = {**full_manifest, "scope": "smoke", "fixtures": selected, "anchor_count": len(selected)}
            manifest_path = paths.compatibility / GOLDEN_MANIFEST_FILENAME
    if golden_manifest.get("status") != "captured" or str(golden_manifest.get("golden_version", "")) != GOLDEN_VERSION:
        raise RuntimeError(f"valid_0_9_1_golden_manifest_missing:{manifest_path}")
    provider = BaostockProvider()
    issues = _bulk_parser_regression()
    fixture_results: list[dict[str, Any]] = []
    for fixture in list(golden_manifest.get("fixtures", []) or []):
        trade_date = str(fixture["trade_date"])
        path = paths.root / str(fixture["path"])
        if not path.exists() or sha256_file(path) != str(fixture.get("parquet_sha256", "")):
            issues.append({"code": "golden_fixture_missing_or_hash_mismatch", "trade_date": trade_date})
            continue
        golden = pd.read_parquet(path, engine="pyarrow")
        if frame_content_sha256(golden) != str(fixture.get("content_sha256", "")):
            issues.append({"code": "golden_fixture_content_hash_mismatch", "trade_date": trade_date})
            continue
        symbols = list(fixture.get("symbols", []) or [])
        current = _query_symbols_for_gate(symbols, trade_date, workspace_root=workspace_root)
        batch_raw = _batch_daily_for_gate(provider, trade_date=trade_date, workspace_root=workspace_root)
        before = len(issues)
        issues.extend(_compare_fixture(golden, current, batch_raw, trade_date=trade_date, workspace_root=workspace_root))
        fixture_results.append({"trade_date": trade_date, "symbol_count": len(symbols), "golden_row_count": len(golden), "batch_row_count": len(batch_raw), "issue_count": len(issues) - before})
    stock_basic = _stock_basic_for_gate(trade_date=str(golden_manifest.get("end_date", "")), workspace_root=workspace_root)
    if len(stock_basic) <= 2000:
        issues.append({"code": "ordinary_multipage_result_not_exercised", "row_count": int(len(stock_basic))})
    symbol_column = "symbol" if "symbol" in stock_basic.columns else "code" if "code" in stock_basic.columns else ""
    if not symbol_column or stock_basic[symbol_column].astype(str).duplicated().any():
        if not symbol_column:
            issues.append({"code": "ordinary_multipage_symbol_column_missing"})
        elif stock_basic[symbol_column].astype(str).duplicated().any():
            issues.append({"code": "ordinary_multipage_duplicate_rows", "duplicate_count": int(stock_basic[symbol_column].astype(str).duplicated().sum())})
    payload = {
        "status": "passed" if not issues else "failed",
        "scope": "smoke" if smoke else "full",
        "golden_version": GOLDEN_VERSION,
        "runtime_version": runtime,
        "wheel_sha256": BAOSTOCK_BATCH_WHEEL_SHA256,
        "checked_at": utc_now(),
        "anchor_count": len(fixture_results),
        "fixture_results": fixture_results,
        "ordinary_multipage_row_count": int(len(stock_basic)),
        "issue_count": len(issues),
        "issues": issues[:500],
    }
    target = compatibility_gate_path(workspace_root) if not smoke else paths.metadata / "baostock_0_9_3_compatibility_smoke.json"
    atomic_write_json(target, payload)
    return {**payload, "report_path": str(target.resolve())}
