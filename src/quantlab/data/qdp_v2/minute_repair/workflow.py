"""End-to-end orchestration for targeted historical one-minute repairs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.core.io import sha256_file, write_json
from quantlab.data.core.json_io import json_safe
from quantlab.data.core.paths import qdp_paths

from .candidate import (
    DEFAULT_BATCH_CALENDAR_DAYS,
    DEFAULT_BATCH_TRADING_DAYS,
    DEFAULT_PRIORITY_RELATIVE_ERROR,
    MinuteRepairError,
    download_target_batches,
    evaluate_target_set,
    load_cached_target_batches,
    load_local_target_minutes,
    load_open_trade_dates,
    load_priority_targets,
    load_provider_target_minutes,
    load_reusable_prior_batches,
    plan_target_batches,
    target_set_id,
    write_candidate_artifacts,
)
from .install import install_accepted_changes, resume_interrupted_install


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _field_counts(decisions: pd.DataFrame) -> dict[str, int]:
    counts = {field: 0 for field in ("open", "high", "low", "close")}
    for value in decisions.loc[decisions["status"].eq("accepted"), "repaired_fields"].astype(str):
        for field in filter(None, value.split(",")):
            counts[field] = counts.get(field, 0) + 1
    return counts


def _reason_counts(decisions: pd.DataFrame) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in decisions.loc[decisions["status"].eq("rejected"), "reason"].value_counts().items()
    }


def _artifact_hashes(paths: dict[str, str]) -> dict[str, str]:
    return {key: sha256_file(value) for key, value in paths.items() if Path(value).is_file()}


def _existing_artifact_paths(run_dir: Path) -> dict[str, str]:
    return {
        "targets": str(run_dir / "targets.parquet"),
        "batches": str(run_dir / "batches.json"),
        "decisions": str(run_dir / "candidate_decisions.parquet"),
        "changes": str(run_dir / "accepted_row_changes.parquet"),
    }


def _captured_raw_rows(run_dir: Path) -> int:
    import json

    payload = json.loads((run_dir / "batches.json").read_text(encoding="utf-8"))
    total = 0
    for item in list(payload.get("batches", []) or []):
        metadata_path = Path(str(item.get("metadata_path", "")))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        total += int(metadata.get("row_count", 0) or 0)
    return total


def _remaining_targets(
    targets: pd.DataFrame,
    batches: list[Any],
) -> pd.DataFrame:
    covered = {
        (str(batch.symbol), str(trade_date))
        for batch in batches
        for trade_date in batch.target_dates
    }
    keep = [
        (str(symbol), str(trade_date)) not in covered
        for symbol, trade_date in zip(targets["symbol"], targets["trade_date"], strict=True)
    ]
    return targets.loc[keep].copy()


def _result_payload(
    *,
    status: str,
    workspace: Path,
    run_dir: Path,
    minimum_relative_error: float,
    maximum_relative_error: float | None,
    selection_mode: str,
    max_calendar_days: int,
    max_trading_days: int,
    rpm: int,
    workers: int,
    targets: pd.DataFrame,
    batch_count: int,
    raw_response_rows: int,
    decisions: pd.DataFrame,
    changes: pd.DataFrame,
    artifacts: dict[str, str],
    installation: dict[str, Any],
    capture_breakdown: dict[str, int] | None = None,
) -> dict[str, Any]:
    accepted = decisions.loc[decisions["status"].eq("accepted")]
    return {
        "schema": "quantlab.targeted_minute_repair_result/v1",
        "status": status,
        "created_at": _utc_now(),
        "workspace": str(workspace),
        "run_dir": str(run_dir),
        "selection": {
            "id": run_dir.name,
            "mode": str(selection_mode),
            "policy": (
                "audited open-field anomalies above the configured open-relative-error threshold"
                if str(selection_mode) == "open"
                else (
                    "all audited daily-envelope overshoots, all audited anomalies above the configured "
                    "relative-error threshold, and previously probed cases"
                )
            ),
            "minimum_relative_error": float(minimum_relative_error),
            "maximum_relative_error": (
                None if maximum_relative_error is None else float(maximum_relative_error)
            ),
            "target_stock_days": int(len(targets)),
            "target_symbols": int(targets["symbol"].nunique()),
            "target_years": sorted({int(value) for value in targets["year"]}),
        },
        "download": {
            "batch_count": int(batch_count),
            "max_calendar_days": int(max_calendar_days),
            "max_trading_days": int(max_trading_days),
            "rpm": int(rpm),
            "workers": int(workers),
            "raw_response_rows": int(raw_response_rows),
            "raw_capture_directory": str(run_dir / "raw"),
            **dict(capture_breakdown or {}),
        },
        "evaluation": {
            "accepted_stock_days": int(len(accepted)),
            "rejected_stock_days": int(decisions["status"].eq("rejected").sum()),
            "changed_rows": int(len(changes)),
            "repaired_field_stock_days": _field_counts(decisions),
            "rejection_reasons": _reason_counts(decisions),
            "flow_policy": "keep existing QDP volume and amount; provider prices only",
        },
        "artifacts": artifacts,
        "artifact_sha256": _artifact_hashes(artifacts),
        "installation": installation,
    }


def run_targeted_minute_repair(
    *,
    workspace_root: str | Path | None = None,
    minimum_relative_error: float = DEFAULT_PRIORITY_RELATIVE_ERROR,
    maximum_relative_error: float | None = None,
    max_calendar_days: int = DEFAULT_BATCH_CALENDAR_DAYS,
    max_trading_days: int = DEFAULT_BATCH_TRADING_DAYS,
    rpm: int = 96,
    workers: int = 3,
    selection_mode: str = "priority",
    apply: bool = False,
) -> dict[str, Any]:
    """Download, evaluate, and optionally install the audited priority set."""

    workspace = qdp_paths(workspace_root).workspace_root
    targets = load_priority_targets(
        workspace,
        minimum_relative_error=float(minimum_relative_error),
        maximum_relative_error=maximum_relative_error,
        include_known_probes=True,
        selection_mode=selection_mode,
    )
    selection_id = target_set_id(
        targets,
        minimum_relative_error=float(minimum_relative_error),
        maximum_relative_error=maximum_relative_error,
        selection_mode=selection_mode,
    )
    run_dir = (
        qdp_paths(workspace).source_archives_dir
        / "tushare_compatible"
        / "minute_repair"
        / selection_id
    ).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = run_dir / "raw"
    cached = load_cached_target_batches(targets, raw_dir=raw_dir)
    pending = _remaining_targets(targets, cached)
    reused = load_reusable_prior_batches(
        pending,
        run_dir=run_dir,
        workspace_root=workspace,
    )
    pending = _remaining_targets(pending, reused)
    planned = plan_target_batches(
        pending,
        max_calendar_days=int(max_calendar_days),
        trading_dates=load_open_trade_dates(workspace),
        max_trading_days=int(max_trading_days),
    )
    downloaded = (
        download_target_batches(
            planned,
            raw_dir=raw_dir,
            workspace_root=workspace,
            rpm=int(rpm),
            workers=int(workers),
        )
        if planned
        else []
    )
    captured = [*cached, *reused, *downloaded]
    if not _remaining_targets(targets, captured).empty:
        raise MinuteRepairError("minute_repair_capture_coverage_incomplete")
    local = load_local_target_minutes(targets, workspace_root=workspace)
    provider, provider_sources = load_provider_target_minutes(captured)
    decisions, changes = evaluate_target_set(targets, local, provider, provider_sources)
    artifacts = write_candidate_artifacts(
        run_dir,
        targets=targets,
        batches=captured,
        decisions=decisions,
        changes=changes,
    )
    installation: dict[str, Any] = {}
    if apply:
        if changes.empty:
            status = "nothing_repairable"
        else:
            installation = install_accepted_changes(
                changes,
                decisions,
                run_dir=run_dir,
                workspace_root=workspace,
            )
            status = "applied"
    else:
        status = "would_apply" if not changes.empty else "nothing_repairable"

    result = _result_payload(
        status=status,
        workspace=workspace,
        run_dir=run_dir,
        minimum_relative_error=float(minimum_relative_error),
        maximum_relative_error=maximum_relative_error,
        selection_mode=selection_mode,
        max_calendar_days=int(max_calendar_days),
        max_trading_days=int(max_trading_days),
        rpm=int(rpm),
        workers=int(workers),
        targets=targets,
        batch_count=len(captured),
        raw_response_rows=sum(len(pd.read_parquet(item.raw_path, columns=["ts_code"])) for item in captured),
        decisions=decisions,
        changes=changes,
        artifacts=artifacts,
        installation=installation,
        capture_breakdown={
            "cached_current_run_batches": len(cached),
            "reused_prior_run_batches": len(reused),
            "new_provider_request_batches": len(downloaded),
        },
    )
    write_json(run_dir / "result.json", json_safe(result))
    return result


def resume_targeted_minute_repair(
    run_dir: str | Path,
    *,
    workspace_root: str | Path | None = None,
    minimum_relative_error: float = DEFAULT_PRIORITY_RELATIVE_ERROR,
    maximum_relative_error: float | None = None,
    max_calendar_days: int = DEFAULT_BATCH_CALENDAR_DAYS,
    max_trading_days: int = DEFAULT_BATCH_TRADING_DAYS,
    rpm: int = 96,
    workers: int = 3,
    selection_mode: str = "priority",
) -> dict[str, Any]:
    """Finish a run whose active shards were committed before audit completion."""

    workspace = qdp_paths(workspace_root).workspace_root
    output = Path(run_dir).resolve()
    artifacts = _existing_artifact_paths(output)
    for path in artifacts.values():
        if not Path(path).is_file():
            raise MinuteRepairError(f"minute_repair_resume_artifact_missing:{path}")
    targets = pd.read_parquet(artifacts["targets"])
    decisions = pd.read_parquet(artifacts["decisions"])
    changes = pd.read_parquet(artifacts["changes"])
    import json

    batch_payload = json.loads(Path(artifacts["batches"]).read_text(encoding="utf-8"))
    installation = resume_interrupted_install(
        changes,
        decisions,
        run_dir=output,
        workspace_root=workspace,
    )
    result = _result_payload(
        status="applied",
        workspace=workspace,
        run_dir=output,
        minimum_relative_error=float(minimum_relative_error),
        maximum_relative_error=maximum_relative_error,
        selection_mode=selection_mode,
        max_calendar_days=int(max_calendar_days),
        max_trading_days=int(max_trading_days),
        rpm=int(rpm),
        workers=int(workers),
        targets=targets,
        batch_count=len(list(batch_payload.get("batches", []) or [])),
        raw_response_rows=_captured_raw_rows(output),
        decisions=decisions,
        changes=changes,
        artifacts=artifacts,
        installation=installation,
    )
    write_json(output / "result.json", json_safe(result))
    return result


__all__ = ["MinuteRepairError", "resume_targeted_minute_repair", "run_targeted_minute_repair"]
