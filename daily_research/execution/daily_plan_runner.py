from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from daily_research.baseline.data_provider import get_latest_completed_trading_date
from daily_research.execution import app_service, data_readiness, scheduler_cli
from daily_research.execution.app_runtime import RUNTIME_ROOT, now_iso, read_json_file, write_json_file


SCHEMA_VERSION = 1
STAGES = (
    "preflight",
    "data_readiness",
    "data_refresh",
    "signal_refresh",
    "trade_plan",
    "paper_reconcile",
    "final_verdict",
)


def _date_compact(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        text = now_iso()[:10]
    return text[:10].replace("-", "")


def daily_runs_root(runtime_root: str | Path | None = None) -> Path:
    return Path(runtime_root or RUNTIME_ROOT) / "daily_runs"


def daily_run_dir(run_date: str, *, runtime_root: str | Path | None = None) -> Path:
    return daily_runs_root(runtime_root) / _date_compact(run_date)


def _write_daily_artifact(run_dir: Path, filename: str, payload: dict[str, Any]) -> str:
    path = run_dir / filename
    write_json_file(path, payload)
    return str(path.resolve())


def _merge_evidence(target: dict[str, str], payload: dict[str, Any]) -> None:
    for source in ("evidence_paths", "artifact_paths"):
        values = payload.get(source, {}) if isinstance(payload.get(source, {}), dict) else {}
        for key, value in values.items():
            text = str(value or "").strip()
            if text:
                target[str(key)] = text
    for key in ("refresh_manifest_path", "signal_refresh_manifest_path"):
        text = str(payload.get(key, "") or "").strip()
        if text:
            target[key.replace("_path", "")] = text


def _metadata(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("metadata"), dict):
        return dict(payload.get("metadata", {}) or {})
    return payload


def _is_success(payload: dict[str, Any]) -> bool:
    status = str(payload.get("status", "") or "").lower()
    exit_code = payload.get("exit_code", 0)
    metadata = _metadata(payload)
    business = str(metadata.get("business_status", "") or payload.get("business_status", "") or "").lower()
    return status in {"succeeded", "ok", "completed"} and int(exit_code or 0) == 0 and business not in {"blocked", "failed"}


def _blocked_verdict(
    *,
    run_date: str,
    target_trading_date: str,
    blocker_code: str,
    stage_results: dict[str, Any],
    run_dir: Path,
    evidence_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    paths = dict(evidence_paths or {})
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_date": run_date,
        "target_trading_date": target_trading_date,
        "status": "blocked",
        "blocker_code": str(blocker_code or "blocked"),
        "stage_results": stage_results,
        "dataset_id": "",
        "signal_panel_date": "",
        "trade_plan_run_dir": "",
        "paper_reconcile_status": "",
        "evidence_paths": paths,
        "created_at": now_iso(),
    }
    verdict_path = run_dir / "verdict.json"
    paths["verdict"] = str(verdict_path.resolve())
    _write_daily_artifact(run_dir, "verdict.json", payload)
    _write_daily_artifact(run_dir, "daily_run_manifest.json", {"schema_version": SCHEMA_VERSION, "stages": list(STAGES), "verdict": paths["verdict"]})
    return payload


def run_daily_plan(
    *,
    mode: str = "post-close",
    run_date: str = "",
    runtime_root: str | Path | None = None,
    scheduler_status_fn: Callable[[], dict[str, Any]] | None = None,
    latest_completed_trading_date_fn: Callable[[], str] | None = None,
    readiness_fn: Callable[..., dict[str, Any]] | None = None,
    task_runner: Callable[..., dict[str, Any]] | None = None,
    paper_reconcile_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    resolved_run_date = str(run_date or now_iso()[:10])[:10]
    run_dir = daily_run_dir(resolved_run_date, runtime_root=runtime_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    stage_results: dict[str, Any] = {}
    evidence_paths: dict[str, str] = {}

    scheduler_status_provider = scheduler_status_fn or scheduler_cli.scheduler_status
    latest_completed_provider = latest_completed_trading_date_fn or get_latest_completed_trading_date
    readiness_provider = readiness_fn or data_readiness.resolve_provider_ready_trading_date
    runner = task_runner or app_service.run_task_sync
    paper_reconcile = paper_reconcile_fn or app_service.paper_account_reconcile

    scheduler = scheduler_status_provider()
    target_trading_date = str(latest_completed_provider() or "")[:10]
    preflight = {
        "status": "ok",
        "mode": str(mode),
        "scheduler": scheduler,
        "target_trading_date": target_trading_date,
    }
    if str(mode) == "post-close":
        if not bool(scheduler.get("installed", False)):
            preflight["status"] = "blocked"
            preflight["blocker_code"] = "scheduler_not_installed"
        elif not bool(scheduler.get("enabled", False)):
            preflight["status"] = "blocked"
            preflight["blocker_code"] = "scheduler_disabled"
    stage_results["preflight"] = preflight
    if preflight["status"] == "blocked":
        return _blocked_verdict(
            run_date=resolved_run_date,
            target_trading_date=target_trading_date,
            blocker_code=str(preflight.get("blocker_code", "preflight_blocked")),
            stage_results=stage_results,
            run_dir=run_dir,
            evidence_paths=evidence_paths,
        )

    readiness = readiness_provider(candidate_date=target_trading_date)
    stage_results["data_readiness"] = readiness
    _merge_evidence(evidence_paths, readiness)
    if readiness.get("refresh_manifest_path"):
        evidence_paths["refresh_manifest"] = str(readiness.get("refresh_manifest_path", ""))
    if str(readiness.get("status", "")) != "ready":
        return _blocked_verdict(
            run_date=resolved_run_date,
            target_trading_date=target_trading_date,
            blocker_code=str(readiness.get("blocker_code", "data_not_ready")),
            stage_results=stage_results,
            run_dir=run_dir,
            evidence_paths=evidence_paths,
        )

    if str(mode) == "dry-run":
        stage_results["final_verdict"] = {"status": "blocked", "blocker_code": "dry_run_no_mutation"}
        return _blocked_verdict(
            run_date=resolved_run_date,
            target_trading_date=target_trading_date,
            blocker_code="dry_run_no_mutation",
            stage_results=stage_results,
            run_dir=run_dir,
            evidence_paths=evidence_paths,
        )

    refresh_args = [
        "--as-of-date",
        target_trading_date,
        "--universe",
        "all_a",
        "--provider-plan",
        app_service.FORMAL_DATA_PLATFORM_PROVIDER_PLAN,
        "--required-domains",
        ",".join(app_service.FORMAL_DATA_PLATFORM_REQUIRED_DOMAINS),
        "--domains",
        ",".join(app_service.FORMAL_DATA_PLATFORM_DOMAINS),
        "--json",
    ]
    data_refresh = runner(
        task_name="data-platform-refresh",
        passthrough_args=refresh_args,
        job_label=f"daily-plan:data-refresh:{target_trading_date}",
        force_unlock=False,
    )
    stage_results["data_refresh"] = data_refresh
    refresh_metadata = _metadata(data_refresh)
    _merge_evidence(evidence_paths, refresh_metadata)
    if not _is_success(data_refresh):
        return _blocked_verdict(
            run_date=resolved_run_date,
            target_trading_date=target_trading_date,
            blocker_code="data_refresh_failed",
            stage_results=stage_results,
            run_dir=run_dir,
            evidence_paths=evidence_paths,
        )
    dataset_id = str(
        refresh_metadata.get("refresh_dataset_id", "")
        or refresh_metadata.get("dataset_id", "")
        or refresh_metadata.get("active_dataset_id", "")
        or data_refresh.get("refresh_dataset_id", "")
        or ""
    )

    signal_refresh = runner(
        task_name="refresh-production-live-panels",
        passthrough_args=["--as-of-date", target_trading_date],
        job_label=f"daily-plan:signal-refresh:{target_trading_date}",
        force_unlock=False,
    )
    stage_results["signal_refresh"] = signal_refresh
    signal_metadata = _metadata(signal_refresh)
    _merge_evidence(evidence_paths, signal_metadata)
    if not _is_success(signal_refresh):
        return _blocked_verdict(
            run_date=resolved_run_date,
            target_trading_date=target_trading_date,
            blocker_code="signal_refresh_failed",
            stage_results=stage_results,
            run_dir=run_dir,
            evidence_paths=evidence_paths,
        )
    signal_panel_date = str(signal_metadata.get("panel_latest_date", "") or signal_metadata.get("signal_panel_latest_date", "") or "")

    trade_plan = runner(
        task_name="trade-plan",
        passthrough_args=[],
        job_label=f"daily-plan:trade-plan:{target_trading_date}",
        force_unlock=False,
    )
    stage_results["trade_plan"] = trade_plan
    trade_metadata = _metadata(trade_plan)
    _merge_evidence(evidence_paths, trade_metadata)
    if not _is_success(trade_plan):
        return _blocked_verdict(
            run_date=resolved_run_date,
            target_trading_date=target_trading_date,
            blocker_code="trade_plan_failed",
            stage_results=stage_results,
            run_dir=run_dir,
            evidence_paths=evidence_paths,
        )
    artifact_paths = trade_metadata.get("artifact_paths", {}) if isinstance(trade_metadata.get("artifact_paths", {}), dict) else {}
    trade_plan_run_dir = str(artifact_paths.get("run_dir", "") or trade_metadata.get("trade_plan_run_dir", "") or "")

    paper = paper_reconcile(as_of_date=target_trading_date)
    stage_results["paper_reconcile"] = paper
    _merge_evidence(evidence_paths, paper if isinstance(paper, dict) else {})
    paper_status = str((paper or {}).get("status", "") or "")
    if paper_status not in {"ok", "registered", "succeeded"}:
        return _blocked_verdict(
            run_date=resolved_run_date,
            target_trading_date=target_trading_date,
            blocker_code="paper_reconcile_failed",
            stage_results=stage_results,
            run_dir=run_dir,
            evidence_paths=evidence_paths,
        )

    verdict = {
        "schema_version": SCHEMA_VERSION,
        "run_date": resolved_run_date,
        "target_trading_date": target_trading_date,
        "status": "completed",
        "blocker_code": "",
        "stage_results": stage_results,
        "dataset_id": dataset_id,
        "signal_panel_date": signal_panel_date,
        "trade_plan_run_dir": trade_plan_run_dir,
        "paper_reconcile_status": paper_status,
        "evidence_paths": evidence_paths,
        "created_at": now_iso(),
    }
    stage_results["final_verdict"] = {"status": "completed"}
    verdict_path = run_dir / "verdict.json"
    evidence_paths["verdict"] = str(verdict_path.resolve())
    _write_daily_artifact(run_dir, "verdict.json", verdict)
    _write_daily_artifact(run_dir, "daily_run_manifest.json", {"schema_version": SCHEMA_VERSION, "stages": list(STAGES), "verdict": evidence_paths["verdict"]})
    return verdict


def latest_daily_verdict(*, runtime_root: str | Path | None = None) -> dict[str, Any]:
    root = daily_runs_root(runtime_root)
    if not root.exists():
        return {}
    verdicts = [path for path in root.glob("*/verdict.json") if path.is_file()]
    if not verdicts:
        return {}
    latest = max(verdicts, key=lambda item: item.stat().st_mtime)
    payload = read_json_file(latest)
    if payload:
        payload.setdefault("evidence_paths", {})["verdict"] = str(latest.resolve())
    return payload


def daily_run_status(*, runtime_root: str | Path | None = None) -> dict[str, Any]:
    verdict = latest_daily_verdict(runtime_root=runtime_root)
    scheduler = scheduler_cli.scheduler_status()
    return {
        "status": str(verdict.get("status", "missing") if verdict else "missing"),
        "latest_run_date": str(verdict.get("run_date", "") or ""),
        "latest_verdict": verdict,
        "scheduler": scheduler,
        "daily_runs_root": str(daily_runs_root(runtime_root).resolve()),
    }


def verdict_for_date(run_date: str, *, runtime_root: str | Path | None = None) -> dict[str, Any]:
    path = daily_run_dir(run_date, runtime_root=runtime_root) / "verdict.json"
    return read_json_file(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Daily Research post-close daily plan state machine.")
    parser.add_argument("--mode", choices=("post-close", "dry-run"), default="post-close")
    parser.add_argument("--run-date", default="")
    parser.add_argument("--runtime-root", default="")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_daily_plan(
        mode=args.mode,
        run_date=args.run_date,
        runtime_root=Path(args.runtime_root) if str(args.runtime_root or "").strip() else None,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.json else None))
    if args.mode == "dry-run":
        return 0
    return 0 if payload.get("status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
