from __future__ import annotations

from pathlib import Path
from typing import Any

from daily_research.execution import app_runtime


def _date_compact(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        text = app_runtime.now_iso()[:10]
    return text[:10].replace("-", "")


def daily_runs_root(runtime_root: str | Path | None = None) -> Path:
    return Path(runtime_root or app_runtime.RUNTIME_ROOT) / "daily_runs"


def daily_run_dir(run_date: str, *, runtime_root: str | Path | None = None) -> Path:
    return daily_runs_root(runtime_root) / _date_compact(run_date)


def latest_daily_verdict(*, runtime_root: str | Path | None = None) -> dict[str, Any]:
    root = daily_runs_root(runtime_root)
    if not root.exists():
        return {}
    verdicts = [path for path in root.glob("*/verdict.json") if path.is_file()]
    if not verdicts:
        return {}
    latest = max(verdicts, key=lambda item: item.stat().st_mtime)
    payload = app_runtime.read_json_file(latest)
    if payload:
        payload.setdefault("evidence_paths", {})["verdict"] = str(latest.resolve())
    return payload


def daily_run_status(*, runtime_root: str | Path | None = None) -> dict[str, Any]:
    verdict = latest_daily_verdict(runtime_root=runtime_root)
    return {
        "status": str(verdict.get("status", "missing") if verdict else "missing"),
        "latest_run_date": str(verdict.get("run_date", "") or ""),
        "latest_verdict": verdict,
        "daily_runs_root": str(daily_runs_root(runtime_root).resolve()),
    }


def verdict_for_date(run_date: str, *, runtime_root: str | Path | None = None) -> dict[str, Any]:
    path = daily_run_dir(run_date, runtime_root=runtime_root) / "verdict.json"
    return app_runtime.read_json_file(path)
