"""Mootdx transport responsibilities."""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd


def _open_mootdx_client(
    client_factory: Any = None,
    *,
    excluded_servers: tuple[tuple[str, int], ...] = (),
) -> Any:
    if callable(client_factory):
        return client_factory()
    try:
        from mootdx.quotes import Quotes  # type: ignore
    except Exception as exc:
        raise RuntimeError("mootdx is not installed in the yolos environment") from exc
    option = {
        "multithread": False,
        "heartbeat": False,
        "bestip": False,
        "timeout": 2,
        "auto_retry": False,
        "raise_exception": True,
    }
    excluded = {(str(item[0]), int(item[1])) for item in excluded_servers}
    cached_server, cached_error = _mootdx_protocol_cache_snapshot(excluded_servers=excluded)
    if cached_error:
        raise RuntimeError(cached_error)
    if cached_server is not None:
        try:
            return Quotes.factory(market="std", server=cached_server, **option)
        except Exception:
            _clear_mootdx_protocol_cache()
    persisted = tuple(item for item in _mootdx_persisted_last_good_servers() if item not in excluded)
    if persisted:
        try:
            return _probe_mootdx_protocol_clients(
                Quotes,
                persisted,
                option=option,
                cache_failure=False,
            )
        except Exception:
            _clear_mootdx_protocol_cache()
    try:
        all_candidates = _mootdx_reachable_server_candidates(
            tuple(
                item for item in _mootdx_hq_server_candidates() if item not in excluded and item not in set(persisted)
            )
        )
        return _probe_mootdx_protocol_clients(Quotes, all_candidates[:24], option=option)
    except Exception as exc:
        _cache_mootdx_protocol_failure(str(exc))
        raise


def _mootdx_hq_server_candidates(limit: int = 256) -> tuple[tuple[str, int], ...]:
    raw_servers: list[Any] = list(_mootdx_persisted_last_good_servers())
    try:
        from tdxpy.constants import hq_hosts  # type: ignore

        raw_servers.extend(list(hq_hosts or []))
    except Exception:
        pass
    try:
        import mootdx.config as mootdx_config  # type: ignore

        raw_servers.extend(list(getattr(mootdx_config, "HQ_HOSTS", ()) or ()))
        raw_servers.extend(list(mootdx_config.get("SERVER.HQ") or []))
    except Exception:
        pass
    out: list[tuple[str, int]] = []
    for item in raw_servers:
        try:
            if isinstance(item, dict):
                host = item.get("host", item.get("ip", ""))
                port = item.get("port", 7709)
                out.append((str(host), int(port)))
            elif len(item) >= 3:
                out.append((str(item[1]), int(item[2])))
            elif len(item) >= 2:
                out.append((str(item[0]), int(item[1])))
        except Exception:
            continue
        if len(out) >= int(limit):
            break
    return tuple(dict.fromkeys(out))


_MOOTDX_REACHABLE_CACHE_LOCK = threading.Lock()


_MOOTDX_REACHABLE_CACHE: tuple[tuple[str, int], ...] = ()


_MOOTDX_REACHABLE_CACHE_AT = 0.0


_MOOTDX_PROTOCOL_CACHE_LOCK = threading.Lock()


_MOOTDX_PROTOCOL_SERVERS: tuple[tuple[str, int], ...] = ()


_MOOTDX_PROTOCOL_ERROR = ""


_MOOTDX_PROTOCOL_CACHE_AT = 0.0


def _mootdx_last_good_path() -> Path:
    configured = str(os.environ.get("QDP_MOOTDX_LAST_GOOD_PATH", "") or "").strip()
    return Path(configured).expanduser() if configured else Path.home() / ".mootdx" / "qdp_last_good_5m.json"


def _mootdx_persisted_last_good_servers() -> tuple[tuple[str, int], ...]:
    path = _mootdx_last_good_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return ()
    rows = payload.get("servers", []) if isinstance(payload, dict) else []
    servers: list[tuple[str, int]] = []
    for item in rows:
        try:
            if isinstance(item, dict):
                servers.append((str(item["host"]), int(item["port"])))
            else:
                servers.append((str(item[0]), int(item[1])))
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return tuple(dict.fromkeys(servers))[:3]


def _persist_mootdx_last_good_servers(servers: tuple[tuple[str, int], ...]) -> None:
    path = _mootdx_last_good_path()
    selected = tuple(dict.fromkeys((str(host), int(port)) for host, port in servers))[:3]
    if not selected:
        return
    payload = {
        "contract": "qdp_mootdx_5m_protocol_last_good_v1",
        "servers": [{"host": host, "port": port} for host, port in selected],
        "updated_at_epoch": int(time.time()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _mootdx_reachable_server_candidates(
    candidates: tuple[tuple[str, int], ...],
    *,
    timeout_seconds: float = 0.8,
    cache_seconds: float = 300.0,
) -> tuple[tuple[str, int], ...]:
    """Rank TCP-reachable TDX servers in parallel before opening mootdx.

    The bundled 0.11.7 server list contains many retired addresses. Testing
    them serially can add minutes before every fallback. This lightweight
    probe is only a reachability filter; the first real bars call still proves
    protocol health and failed protocol endpoints are excluded by the provider.
    """

    global _MOOTDX_REACHABLE_CACHE, _MOOTDX_REACHABLE_CACHE_AT
    now = time.monotonic()
    with _MOOTDX_REACHABLE_CACHE_LOCK:
        if _MOOTDX_REACHABLE_CACHE and now - _MOOTDX_REACHABLE_CACHE_AT < float(cache_seconds):
            cached = tuple(item for item in _MOOTDX_REACHABLE_CACHE if item in candidates)
            if cached:
                return cached

    def probe(server: tuple[str, int]) -> tuple[float, tuple[str, int]] | None:
        started = time.perf_counter()
        try:
            connection = socket.create_connection(server, timeout=max(0.2, float(timeout_seconds)))
        except OSError:
            return None
        try:
            return time.perf_counter() - started, server
        finally:
            connection.close()

    reachable: list[tuple[float, tuple[str, int]]] = []
    workers = min(32, max(1, len(candidates)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(probe, candidates):
            if result is not None:
                reachable.append(result)
    ranked = tuple(server for _, server in sorted(reachable))
    if not ranked:
        raise RuntimeError("mootdx_no_tcp_reachable_hq_server")
    with _MOOTDX_REACHABLE_CACHE_LOCK:
        _MOOTDX_REACHABLE_CACHE = ranked
        _MOOTDX_REACHABLE_CACHE_AT = time.monotonic()
    return ranked


def _mootdx_protocol_cache_snapshot(
    *,
    excluded_servers: set[tuple[str, int]],
    cache_seconds: float = 300.0,
) -> tuple[tuple[str, int] | None, str]:
    with _MOOTDX_PROTOCOL_CACHE_LOCK:
        fresh = time.monotonic() - _MOOTDX_PROTOCOL_CACHE_AT < float(cache_seconds)
        if not fresh:
            return None, ""
        if _MOOTDX_PROTOCOL_ERROR:
            return None, _MOOTDX_PROTOCOL_ERROR
        for server in _MOOTDX_PROTOCOL_SERVERS:
            if server not in excluded_servers:
                return server, ""
    return None, ""


def _clear_mootdx_protocol_cache() -> None:
    global _MOOTDX_PROTOCOL_SERVERS, _MOOTDX_PROTOCOL_ERROR, _MOOTDX_PROTOCOL_CACHE_AT
    with _MOOTDX_PROTOCOL_CACHE_LOCK:
        _MOOTDX_PROTOCOL_SERVERS = ()
        _MOOTDX_PROTOCOL_ERROR = ""
        _MOOTDX_PROTOCOL_CACHE_AT = 0.0


def _cache_mootdx_protocol_failure(message: str) -> None:
    global _MOOTDX_PROTOCOL_SERVERS, _MOOTDX_PROTOCOL_ERROR, _MOOTDX_PROTOCOL_CACHE_AT
    with _MOOTDX_PROTOCOL_CACHE_LOCK:
        _MOOTDX_PROTOCOL_SERVERS = ()
        _MOOTDX_PROTOCOL_ERROR = str(message or "mootdx_protocol_probe_failed")
        _MOOTDX_PROTOCOL_CACHE_AT = time.monotonic()


def _mootdx_complete_5m_probe(payload: Any) -> tuple[bool, str]:
    if payload is None or len(payload) == 0:
        return False, ""
    frame = payload.copy() if isinstance(payload, pd.DataFrame) else pd.DataFrame(payload)
    timestamp_source: Any = None
    for column in ("datetime", "trade_time", "date", "trade_date"):
        if column in frame.columns:
            timestamp_source = frame[column]
            break
    if timestamp_source is None and isinstance(frame.index, pd.DatetimeIndex):
        timestamp_source = pd.Series(frame.index, index=frame.index)
    if timestamp_source is None:
        return False, ""
    timestamps = pd.to_datetime(timestamp_source, errors="coerce")
    valid = timestamps.notna()
    if not valid.any():
        return False, ""
    work = pd.DataFrame({"timestamp": timestamps.loc[valid].to_numpy()})
    work["trade_date"] = work["timestamp"].dt.strftime("%Y-%m-%d")
    expected = {
        *pd.date_range("2000-01-01 09:35", "2000-01-01 11:30", freq="5min").strftime("%H:%M"),
        *pd.date_range("2000-01-01 13:05", "2000-01-01 15:00", freq="5min").strftime("%H:%M"),
    }
    for trade_date in sorted(set(work["trade_date"]), reverse=True):
        day = work.loc[work["trade_date"].eq(trade_date), "timestamp"]
        times = set(day.dt.strftime("%H:%M"))
        if len(day) == 48 and times == expected and max(times) == "15:00":
            return True, str(trade_date)
    return False, ""


def _mootdx_protocol_servers_snapshot() -> tuple[tuple[str, int], ...]:
    with _MOOTDX_PROTOCOL_CACHE_LOCK:
        return tuple(_MOOTDX_PROTOCOL_SERVERS)


def _probe_mootdx_protocol_clients(
    Quotes: Any,
    candidates: tuple[tuple[str, int], ...],
    *,
    option: dict[str, Any],
    cache_failure: bool = True,
) -> Any:
    """Select nodes that prove a complete recent 48-bar 5m trading day."""

    global _MOOTDX_PROTOCOL_SERVERS, _MOOTDX_PROTOCOL_ERROR, _MOOTDX_PROTOCOL_CACHE_AT
    if not candidates:
        error = "mootdx_no_unexcluded_protocol_candidate"
        if cache_failure:
            with _MOOTDX_PROTOCOL_CACHE_LOCK:
                _MOOTDX_PROTOCOL_SERVERS = ()
                _MOOTDX_PROTOCOL_ERROR = error
                _MOOTDX_PROTOCOL_CACHE_AT = time.monotonic()
        raise RuntimeError(error)

    def probe(server: tuple[str, int]) -> tuple[float, tuple[str, int], Any | None, str, str]:
        client = None
        started = time.perf_counter()
        try:
            client = Quotes.factory(market="std", server=server, **option)
            minute_payload = client.bars(symbol="600000", frequency=0, start=0, offset=96)
            complete, trade_date = _mootdx_complete_5m_probe(minute_payload)
            if not complete:
                raise RuntimeError("recent_complete_48_bar_5m_probe_missing")
            return time.perf_counter() - started, server, client, "", trade_date
        except Exception as exc:
            if client is not None:
                _close_mootdx_client(client)
            return time.perf_counter() - started, server, None, f"{type(exc).__name__}: {exc}", ""

    candidates = tuple(candidates[:24])
    workers = min(8, len(candidates))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(probe, candidates))
    successes = sorted((item for item in results if item[2] is not None), key=lambda item: item[0])
    if successes:
        _, _, selected_client, _, _ = successes[0]
        healthy_servers = tuple(item[1] for item in successes[:3])
        for _, _, client, _, _ in successes[1:]:
            _close_mootdx_client(client)
        with _MOOTDX_PROTOCOL_CACHE_LOCK:
            _MOOTDX_PROTOCOL_SERVERS = healthy_servers
            _MOOTDX_PROTOCOL_ERROR = ""
            _MOOTDX_PROTOCOL_CACHE_AT = time.monotonic()
        _persist_mootdx_last_good_servers(healthy_servers)
        return selected_client
    details = " | ".join(f"server={server}:{error}" for _, server, _, error, _ in results[-4:])
    error = "mootdx_no_protocol_healthy_hq_server:" + details
    if cache_failure:
        with _MOOTDX_PROTOCOL_CACHE_LOCK:
            _MOOTDX_PROTOCOL_SERVERS = ()
            _MOOTDX_PROTOCOL_ERROR = error
            _MOOTDX_PROTOCOL_CACHE_AT = time.monotonic()
    raise RuntimeError(error)


def _close_mootdx_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            return
