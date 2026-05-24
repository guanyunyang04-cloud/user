from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_TIMEOUT_SECONDS = 7200
DEFAULT_STALE_AFTER_SECONDS = 3600


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return {"_load_error": str(exc)}
    return payload if isinstance(payload, dict) else {"_load_error": "progress payload is not a JSON object"}


def _number(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _tail(path: Path | None, line_count: int = 12) -> list[str]:
    if path is None or not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except Exception as exc:
        return [f"<tail_error: {exc}>"]
    return lines[-line_count:]


def _artifact_mtime(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    candidates = [item for item in path.rglob("*") if item.is_file()] if path.is_dir() else [path]
    if not candidates:
        return ""
    latest = max(item.stat().st_mtime for item in candidates)
    return datetime.fromtimestamp(latest, tz=timezone.utc).isoformat()


def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f"$p = Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue; if ($p) {{ 'alive' }}",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return "alive" in (result.stdout or "")


def _progress_fraction(progress: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    current = _number(progress, "current_step", "completed_steps", "step", "global_step")
    total = _number(progress, "total_steps", "step_count", "max_steps")
    if current is None or total is None:
        current = _number(progress, "current_epoch", "epoch", "completed_epochs")
        total = _number(progress, "total_epochs", "epochs", "max_epochs")
    if current is None or total is None or total <= 0:
        percent = _number(progress, "progress_percent")
        if percent is None:
            return None, None, None
        return percent / 100.0, None, None
    fraction = max(0.0, min(1.0, current / total))
    return fraction, current, total


def _started_at(progress: dict[str, Any], progress_path: Path | None, now: datetime) -> datetime | None:
    parsed = _parse_datetime(progress.get("started_at") or progress.get("start_time"))
    if parsed is not None:
        return parsed
    if progress_path is not None and progress_path.exists():
        return datetime.fromtimestamp(progress_path.stat().st_ctime, tz=timezone.utc)
    return None


def _updated_at(progress: dict[str, Any], progress_path: Path | None) -> datetime | None:
    parsed = _parse_datetime(progress.get("updated_at") or progress.get("last_update_at") or progress.get("timestamp"))
    if parsed is not None:
        return parsed
    if progress_path is not None and progress_path.exists():
        return datetime.fromtimestamp(progress_path.stat().st_mtime, tz=timezone.utc)
    return None


def build_template(*, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    timeout = int(timeout_seconds)
    return {
        "schema_version": 1,
        "poll_window_seconds": timeout,
        "eta_required": True,
        "required_wait_command": f"Wait-Process -Id <pid> -Timeout {timeout}",
        "powershell_template": "\n".join(
            [
                "$proc = Start-Process -FilePath <command> -ArgumentList <args> -PassThru -NoNewWindow",
                "$pid = $proc.Id",
                f"Wait-Process -Id <pid> -Timeout {timeout}",
                "C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.long_task_monitor status --pid $pid --progress <progress.json> --stdout <stdout.log> --stderr <stderr.log> --artifact-dir <artifact_dir> --json",
            ]
        ),
    }


def build_status(
    *,
    pid: int | None = None,
    progress_path: str | Path | None = None,
    stdout_path: str | Path | None = None,
    stderr_path: str | Path | None = None,
    artifact_dir: str | Path | None = None,
    stale_after_seconds: int = DEFAULT_STALE_AFTER_SECONDS,
) -> dict[str, Any]:
    progress_file = Path(progress_path) if progress_path else None
    stdout_file = Path(stdout_path) if stdout_path else None
    stderr_file = Path(stderr_path) if stderr_path else None
    artifact_path = Path(artifact_dir) if artifact_dir else None
    now = _utc_now()
    progress = _load_json(progress_file)
    started = _started_at(progress, progress_file, now)
    updated = _updated_at(progress, progress_file)
    elapsed = max(0.0, (now - started).total_seconds()) if started is not None else None
    fraction, current, total = _progress_fraction(progress)
    progress_percent = round(fraction * 100.0, 3) if fraction is not None else None

    estimated_remaining: float | None = None
    eta_at = ""
    eta_status = "progress_unavailable"
    if elapsed is not None and fraction is not None:
        stale_seconds = (now - updated).total_seconds() if updated is not None else 0.0
        if stale_seconds >= int(stale_after_seconds):
            eta_status = "stalled_or_waiting"
        elif current is not None and current <= 0:
            eta_status = "warming_up"
        elif fraction <= 0:
            eta_status = "warming_up"
        elif fraction >= 1.0:
            eta_status = "completed"
            estimated_remaining = 0.0
            eta_at = now.isoformat()
        else:
            estimated_remaining = max(0.0, elapsed * (1.0 - fraction) / fraction)
            eta_at = datetime.fromtimestamp(time.time() + estimated_remaining, tz=timezone.utc).isoformat()
            eta_status = "estimated"

    if eta_status == "stalled_or_waiting":
        decision = "inspect_logs_or_resources"
    elif _pid_alive(pid):
        decision = "continue_wait_process_window"
    elif eta_status == "completed":
        decision = "verify_artifacts"
    else:
        decision = "inspect_exit_or_artifacts"

    return {
        "schema_version": 1,
        "pid": pid,
        "pid_alive": _pid_alive(pid),
        "progress_path": str(progress_file or ""),
        "elapsed_seconds": round(elapsed, 3) if elapsed is not None else None,
        "estimated_remaining_seconds": round(estimated_remaining, 3) if estimated_remaining is not None else None,
        "eta_at": eta_at,
        "eta_status": eta_status,
        "progress_percent": progress_percent,
        "current_step": current,
        "total_steps": total,
        "stage": progress.get("stage", ""),
        "latest_metric": progress.get("latest_metric", {}),
        "last_log_lines": _tail(stdout_file),
        "last_error_lines": _tail(stderr_file),
        "artifact_mtime": _artifact_mtime(artifact_path),
        "updated_at": updated.isoformat() if updated is not None else "",
        "decision": decision,
    }


def _print_payload(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if "powershell_template" in payload:
        print(payload["powershell_template"])
        return
    print(
        "pid={pid} alive={pid_alive} elapsed={elapsed_seconds}s progress={progress_percent}% "
        "eta={estimated_remaining_seconds}s eta_status={eta_status} decision={decision}".format(**payload)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Brain long-task wait/status helper.")
    sub = parser.add_subparsers(dest="command", required=True)

    template = sub.add_parser("template", help="Print the required PowerShell long-task wait template.")
    template.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    template.add_argument("--json", action="store_true")

    status = sub.add_parser("status", help="Read PID, logs, progress, artifacts, and ETA.")
    status.add_argument("--pid", type=int, default=0)
    status.add_argument("--progress", default="")
    status.add_argument("--stdout", default="")
    status.add_argument("--stderr", default="")
    status.add_argument("--artifact-dir", default="")
    status.add_argument("--stale-after-seconds", type=int, default=DEFAULT_STALE_AFTER_SECONDS)
    status.add_argument("--json", action="store_true")

    wait_once = sub.add_parser("wait-once", help="Run one Wait-Process window, then report status.")
    wait_once.add_argument("--pid", type=int, required=True)
    wait_once.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    wait_once.add_argument("--progress", default="")
    wait_once.add_argument("--stdout", default="")
    wait_once.add_argument("--stderr", default="")
    wait_once.add_argument("--artifact-dir", default="")
    wait_once.add_argument("--stale-after-seconds", type=int, default=DEFAULT_STALE_AFTER_SECONDS)
    wait_once.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "template":
        _print_payload(build_template(timeout_seconds=args.timeout), as_json=bool(args.json))
        return 0
    if args.command == "wait-once":
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"Wait-Process -Id {int(args.pid)} -Timeout {int(args.timeout)} -ErrorAction SilentlyContinue",
            ],
            check=False,
        )
    payload = build_status(
        pid=int(getattr(args, "pid", 0) or 0),
        progress_path=str(getattr(args, "progress", "") or "") or None,
        stdout_path=str(getattr(args, "stdout", "") or "") or None,
        stderr_path=str(getattr(args, "stderr", "") or "") or None,
        artifact_dir=str(getattr(args, "artifact_dir", "") or "") or None,
        stale_after_seconds=int(getattr(args, "stale_after_seconds", DEFAULT_STALE_AFTER_SECONDS)),
    )
    _print_payload(payload, as_json=bool(getattr(args, "json", False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
