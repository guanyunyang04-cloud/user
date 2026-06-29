from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.qdp_v2.environment import runtime_environment
from quant_data_platform.qdp_v2.runtime import resolve_runtime_profile


def benchmark_providers(
    *,
    providers: list[str],
    runtime: str,
    workspace_root: str | Path | None = None,
    live: bool = False,
) -> dict[str, Any]:
    del workspace_root
    profile = resolve_runtime_profile(runtime)
    results: list[dict[str, Any]] = []
    for provider in providers:
        name = provider.strip().lower()
        if not name:
            continue
        start = time.perf_counter()
        if name == "baostock":
            result = _benchmark_baostock(live=live)
            result["configured_workers"] = profile.baostock_workers
            result["max_tasks_per_session"] = profile.baostock_max_tasks_per_session
        elif name == "mootdx":
            result = _benchmark_mootdx(live=live)
            result["configured_workers"] = profile.mootdx_workers
            result["page_size"] = profile.mootdx_page_size
        elif name == "cninfo":
            result = _benchmark_cninfo(live=live)
            result["configured_workers"] = profile.cninfo_workers
            result["timeout_seconds"] = profile.cninfo_timeout_seconds
        else:
            result = {"provider": name, "status": "unknown_provider"}
        result["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 2)
        results.append(result)
    return {
        "status": "ok",
        "runtime": asdict(profile),
        "runtime_environment": runtime_environment(),
        "live": bool(live),
        "results": results,
        "notes": [
            "benchmark is intentionally small; use provider run manifests for full throughput analysis",
            "qdp_v2 provider design expects persistent sessions instead of login/logout per task",
        ],
    }


def _benchmark_baostock(*, live: bool) -> dict[str, Any]:
    try:
        import baostock as bs  # type: ignore
    except Exception as exc:
        return {"provider": "baostock", "status": "unavailable", "error": str(exc)}
    if not live:
        return {"provider": "baostock", "status": "available", "session_model": "persistent_login_per_worker"}
    login = bs.login()
    try:
        if getattr(login, "error_code", "") not in {"0", 0, ""}:
            return {"provider": "baostock", "status": "login_failed", "error_code": getattr(login, "error_code", ""), "error_msg": getattr(login, "error_msg", "")}
        rs = bs.query_all_stock("2026-06-26")
        rows = 0
        while rs.next():
            rows += 1
        return {"provider": "baostock", "status": "ok", "sample": "query_all_stock", "rows": rows, "session_model": "persistent_login_per_worker"}
    finally:
        try:
            bs.logout()
        except Exception:
            pass


def _benchmark_mootdx(*, live: bool) -> dict[str, Any]:
    try:
        from mootdx.quotes import Quotes  # type: ignore
    except Exception as exc:
        return {"provider": "mootdx", "status": "unavailable", "error": str(exc)}
    if not live:
        return {"provider": "mootdx", "status": "available", "session_model": "persistent_client_per_worker"}
    client = Quotes.factory(market="std")
    try:
        data = client.bars(symbol="000001", frequency=9, offset=0, count=16)
        rows = int(len(data)) if data is not None else 0
        return {"provider": "mootdx", "status": "ok", "sample": "bars_000001_daily_16", "rows": rows, "session_model": "persistent_client_per_worker"}
    except Exception as exc:
        return {"provider": "mootdx", "status": "probe_failed", "error": str(exc), "session_model": "persistent_client_per_worker"}


def _benchmark_cninfo(*, live: bool) -> dict[str, Any]:
    try:
        import requests  # type: ignore
    except Exception as exc:
        return {"provider": "cninfo", "status": "unavailable", "error": str(exc)}
    if not live:
        return {"provider": "cninfo", "status": "available", "session_model": "requests.Session_pool"}
    session = requests.Session()
    try:
        response = session.get("https://www.cninfo.com.cn/new/index", timeout=10)
        return {"provider": "cninfo", "status": "ok" if response.status_code < 500 else "http_error", "status_code": response.status_code, "session_model": "requests.Session_pool"}
    except Exception as exc:
        return {"provider": "cninfo", "status": "probe_failed", "error": str(exc), "session_model": "requests.Session_pool"}
    finally:
        session.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp provider benchmark", description="Small provider availability/latency benchmark for qdp_v2.")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--provider", "--providers", default="mootdx,baostock,cninfo")
    parser.add_argument("--runtime", default="balanced", choices=("safe", "balanced", "fast"))
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    providers = [item.strip() for item in str(args.provider or "").split(",") if item.strip()]
    payload = benchmark_providers(providers=providers, runtime=str(args.runtime or "balanced"), workspace_root=str(args.workspace_root or "") or None, live=bool(args.live))
    if bool(args.json):
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    else:
        print(_format(payload))
    return 0 if payload.get("status") == "ok" else 2


def _format(payload: dict[str, Any]) -> str:
    lines = [f"status: {payload.get('status')}", f"live: {payload.get('live')}"]
    for item in list(payload.get("results", []) or []):
        lines.append(f"{item.get('provider')}: {item.get('status')} elapsed_ms={item.get('elapsed_ms')}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
