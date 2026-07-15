from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from quant_data_platform import cli as public_cli_module
from quant_data_platform.providers import _PROVIDER_CAPABILITIES
from quant_data_platform.qdp_v3 import cli as v3_cli
from quant_data_platform.qdp_v3.manifest import atomic_write_json
from quant_data_platform.qdp_v3.paths import ensure_qdp_v3_layout, qdp_v3_paths
from quant_data_platform.qdp_v3.constants import MANIFEST_VERSION, RAW_TRADING_CALENDAR, STRICT_RELEASE_DOMAINS
from quant_data_platform.qdp_v3.manifest import manifest_sha256
from quant_data_platform.qdp_v3.release import (
    candidate_path,
    validate_published_active,
    write_candidate,
)
from quant_data_platform.qdp_v3.storage import write_raw_partition
from quant_data_platform.qdp_v3.supervisor import requeue_stale_jobs, supervise_job
from quant_data_platform.qdp_v3.update import plan_update
from quant_data_platform.domains.contracts import DataDomain


def test_trusted_status_capture_only_schedules_core_suspend_facts() -> None:
    from quant_data_platform.qdp_v3.historical import REFERENCE_SPECS

    assert [spec.api_name for spec in REFERENCE_SPECS["status"]] == ["suspend_d"]


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    manifest = workspace / "brain" / "brain_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"schema_version": 1, "brain_type": "main"}), encoding="utf-8")
    return workspace


def test_public_cli_never_falls_through_to_v2_mutation_without_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[list[str]] = []

    def fake_v2(argv: list[str], _workspace_root: str) -> int:
        calls.append(argv)
        return 0

    monkeypatch.setattr(public_cli_module, "_run_qdp_v2", fake_v2)
    code = public_cli_module.main(["rebuild", "daily-panel", "--json"])

    assert code == 2
    assert calls == []
    assert json.loads(capsys.readouterr().out)["status"] == "archived"

    assert public_cli_module.main(["--generation", "v2", "rebuild", "daily-panel"]) == 0
    assert calls == [["rebuild", "daily-panel"]]


def test_v3_check_uses_active_candidate_when_candidate_is_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = _workspace(tmp_path)
    paths = ensure_qdp_v3_layout(workspace)
    datasets = {domain: f"{domain}__fixture" for domain in STRICT_RELEASE_DOMAINS}
    dataset_hashes = {domain: "a" * 64 for domain in STRICT_RELEASE_DOMAINS}
    candidate = write_candidate(
        datasets=datasets,
        dataset_manifest_sha256=dataset_hashes,
        quality_tiers={domain: "strict" for domain in STRICT_RELEASE_DOMAINS},
        blockers=[],
        coverage={},
        build={"fixture": "active_check"},
        raw_partitions=[],
        quarantine=[],
        workspace_root=workspace,
    )
    candidate_file = candidate_path(candidate.candidate_id, workspace_root=workspace)
    atomic_write_json(
        paths.active_manifest,
        {
            "manifest_version": MANIFEST_VERSION,
            "candidate_id": candidate.candidate_id,
            "candidate_manifest_sha256": manifest_sha256(candidate_file),
            "datasets": datasets,
            "dataset_manifest_sha256": dataset_hashes,
            "coverage": {},
            "published_at": "2026-07-15T00:00:00+00:00",
        },
    )
    calls: list[dict[str, object]] = []

    def fake_audit(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"status": "passed", "candidate_id": kwargs["candidate_id"], "mode": kwargs["mode"]}

    monkeypatch.setattr(v3_cli, "audit_candidate", fake_audit)
    code = v3_cli.dispatch(["check", "--workspace-root", str(workspace), "--semantic", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["target"] == "active"
    assert payload["candidate_id"] == candidate.candidate_id
    assert calls[0]["candidate_id"] == candidate.candidate_id
    assert calls[0]["mode"] == "semantic"


def test_invalid_active_placeholder_does_not_switch_default_reads(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = _workspace(tmp_path)
    paths = ensure_qdp_v3_layout(workspace)
    atomic_write_json(
        paths.active_manifest,
        {"candidate_id": "candidate__active", "datasets": {}, "coverage": {}},
    )

    validation = validate_published_active(workspace)
    assert validation["valid"] is False
    assert "active_manifest_version_invalid" in validation["blockers"]
    assert public_cli_module._default_read_generation(str(workspace)) == "v2"

    code = v3_cli.dispatch(["status", "--workspace-root", str(workspace), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "not_published"
    assert payload["active"] == {}
    assert payload["invalid_active"]["blockers"] == validation["blockers"]


def test_complete_current_active_switches_default_reads_to_v3(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    paths = ensure_qdp_v3_layout(workspace)
    datasets = {domain: f"{domain}__fixture" for domain in STRICT_RELEASE_DOMAINS}
    dataset_hashes = {domain: "a" * 64 for domain in STRICT_RELEASE_DOMAINS}
    candidate = write_candidate(
        datasets=datasets,
        dataset_manifest_sha256=dataset_hashes,
        quality_tiers={domain: "strict" for domain in STRICT_RELEASE_DOMAINS},
        blockers=[],
        coverage={"end_date": "2026-07-13"},
        build={"fixture": "published_active"},
        raw_partitions=[],
        quarantine=[],
        workspace_root=workspace,
    )
    candidate_file = candidate_path(candidate.candidate_id, workspace_root=workspace)
    atomic_write_json(
        paths.active_manifest,
        {
            "manifest_version": MANIFEST_VERSION,
            "candidate_id": candidate.candidate_id,
            "candidate_manifest_sha256": manifest_sha256(candidate_file),
            "datasets": datasets,
            "dataset_manifest_sha256": dataset_hashes,
            "quality_tiers": {domain: "strict" for domain in STRICT_RELEASE_DOMAINS},
            "coverage": {"end_date": "2026-07-13"},
            "published_at": "2026-07-15T00:00:00+00:00",
        },
    )

    assert validate_published_active(workspace)["valid"] is True
    assert public_cli_module._default_read_generation(str(workspace)) == "v3"


def test_trusted_bootstrap_plan_prioritizes_5m_and_skips_secondary_sources(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    plan = plan_update(
        as_of_date="2026-07-13",
        workspace_root=workspace,
        bootstrap=True,
        start_date="2010-01-01",
        historical_provider="tushare-proxy",
    )

    stages = {item["stage"]: item for item in plan["stages"]}
    assert stages["intraday_5m"]["provider"] == "tushare_proxy"
    assert stages["intraday_5m"]["priority"] == 1
    assert stages["quota_tail_reference"]["domains"] == ["status", "factor"]
    assert stages["quota_tail_reference"]["skip_domains"] == ["dividend", "financial"]
    assert not any("baostock" in item["stage"] for item in plan["stages"])
    assert "secondary_pit" not in stages


def test_incremental_plan_uses_full_10_and_60_day_refresh_windows(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    paths = ensure_qdp_v3_layout(workspace)
    datasets = {domain: f"{domain}__fixture" for domain in STRICT_RELEASE_DOMAINS}
    dataset_hashes = {domain: "a" * 64 for domain in STRICT_RELEASE_DOMAINS}
    candidate = write_candidate(
        datasets=datasets,
        dataset_manifest_sha256=dataset_hashes,
        quality_tiers={domain: "strict" for domain in STRICT_RELEASE_DOMAINS},
        blockers=[],
        coverage={"end_date": "2026-07-13"},
        build={"fixture": "incremental_plan"},
        raw_partitions=[],
        quarantine=[],
        workspace_root=workspace,
    )
    candidate_file = candidate_path(candidate.candidate_id, workspace_root=workspace)
    atomic_write_json(
        paths.active_manifest,
        {
            "manifest_version": MANIFEST_VERSION,
            "candidate_id": candidate.candidate_id,
            "candidate_manifest_sha256": manifest_sha256(candidate_file),
            "datasets": datasets,
            "dataset_manifest_sha256": dataset_hashes,
            "coverage": {"end_date": "2026-07-13"},
            "published_at": "2026-07-15T00:00:00+00:00",
        },
    )
    as_of_date = "2026-07-15"
    lookback_start = (pd.Timestamp(as_of_date) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    calendar_dates = pd.date_range(lookback_start, as_of_date, freq="D")
    calendar = pd.DataFrame(
        {
            "trade_date": calendar_dates.strftime("%Y-%m-%d"),
            "is_open": calendar_dates.dayofweek < 5,
            "exchange": "SSE/SZSE",
        }
    )
    write_raw_partition(
        raw_domain=RAW_TRADING_CALENDAR,
        partition_field="request_range",
        partition_value=f"{lookback_start}_{as_of_date}",
        frame=calendar,
        receipt={"quality_tier": "strict", "provider": "unit"},
        workspace_root=workspace,
    )

    plan = plan_update(as_of_date=as_of_date, workspace_root=workspace)

    stages = {item["stage"]: item for item in plan["stages"]}
    assert plan["start_date"] == lookback_start
    assert stages["baostock_incremental_daily"]["task_count"] == 10
    assert stages["baostock_incremental_factor_events"]["task_count"] == 60
    assert stages["baostock_incremental_daily"]["end_date"] == as_of_date
    assert stages["baostock_incremental_factor_events"]["end_date"] == as_of_date


def test_v3_provider_capabilities_have_no_1m_or_obsolete_router() -> None:
    assert "qdp_production_v2" not in _PROVIDER_CAPABILITIES
    assert DataDomain.MARKET_INTRADAY_1M not in _PROVIDER_CAPABILITIES["tushare_proxy"]["domains"]
    assert DataDomain.MARKET_INTRADAY_5M in _PROVIDER_CAPABILITIES["tushare_proxy"]["domains"]


def test_job_supervisor_writes_active_and_stopped_heartbeat(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    heartbeat_path = qdp_v3_paths(workspace).jobs / "heartbeats" / "unit_job.json"

    with supervise_job("unit_job", workspace_root=workspace, heartbeat_interval_seconds=0.01):
        heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
        assert heartbeat["active"] is True
        assert heartbeat["pid"] > 0

    stopped = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert stopped["active"] is False
    assert stopped["stopped_at"]


def test_atomic_json_write_retries_a_transient_windows_file_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "job.json"
    original_replace = Path.replace
    attempts = 0

    def flaky_replace(self: Path, destination: str | Path) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("simulated transient file lock")
        return original_replace(self, destination)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    atomic_write_json(target, {"status": "written"})

    assert attempts == 2
    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "written"}


def test_stale_running_job_is_requeued_only_without_os_lock(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    paths = ensure_qdp_v3_layout(workspace)
    stale_time = "2026-07-15T00:00:00+00:00"
    now = datetime(2026, 7, 15, 0, 10, tzinfo=timezone.utc)
    job_path = paths.jobs / "recoverable.json"
    atomic_write_json(
        job_path,
        {
            "job_id": "recoverable",
            "status": "running",
            "updated_at": stale_time,
            "tasks": {"a": {"task_id": "a", "status": "running", "updated_at": stale_time}},
        },
    )

    recovered = requeue_stale_jobs(workspace_root=workspace, now=now)

    assert recovered["recovered_count"] == 1
    state = json.loads(job_path.read_text(encoding="utf-8"))
    assert state["status"] == "interrupted_recoverable"
    assert state["tasks"]["a"]["status"] == "pending"

    locked_path = paths.jobs / "locked.json"
    with supervise_job("locked", workspace_root=workspace, heartbeat_interval_seconds=3600):
        atomic_write_json(
            locked_path,
            {"job_id": "locked", "status": "running", "updated_at": stale_time, "tasks": {}},
        )
        heartbeat_path = paths.jobs / "heartbeats" / "locked.json"
        atomic_write_json(
            heartbeat_path,
            {"job_id": "locked", "active": True, "heartbeat_at": stale_time},
        )
        result = requeue_stale_jobs(workspace_root=workspace, now=now)
        assert result["recovered_count"] == 0
        assert result["locked_jobs"] == ["locked"]
        assert json.loads(locked_path.read_text(encoding="utf-8"))["status"] == "running"
