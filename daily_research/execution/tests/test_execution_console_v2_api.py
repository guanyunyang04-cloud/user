from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest


FORMAL_REFRESH_DOMAINS = [
    "market_daily",
    "trading_calendar",
    "universe_snapshot",
    "security_status",
]


def test_trade_plan_summary_handles_missing_latest_artifact(tmp_path: Path) -> None:
    from daily_research.execution import app_service

    missing = tmp_path / "missing" / "latest_trade_plan.txt"
    payload = app_service.latest_trade_plan_summary(latest_txt_path=missing)

    assert payload["exists"] is False
    assert payload["status"] == "missing"
    assert payload["summary"] == {}
    assert payload["actions"] == []
    assert payload["holdings"] == []
    assert payload["watchlist"] == []
    assert payload["txt_preview"] == []


def test_trade_plan_summary_prefers_structured_latest_run(tmp_path: Path) -> None:
    from daily_research.execution import app_service

    output_dir = tmp_path / "execution" / "output"
    run_dir = output_dir / "20260523"
    run_dir.mkdir(parents=True)
    (output_dir / "latest_trade_plan.txt").write_text("plan text\nline2\n", encoding="utf-8")
    (run_dir / "daily_trade_plan.txt").write_text("run text\n", encoding="utf-8")
    (run_dir / "plan_summary.json").write_text(
        json.dumps(
            {
                "signal_date": "2026-05-22",
                "cash_input": 1000.0,
                "model_freshness": {
                    "trained_at": "2026-05-20T18:00:00",
                    "artifact_latest_data_date": "2026-05-20",
                    "trading_day_lag": 2,
                    "latest_completed_trading_date": "2026-05-22",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "actions_today.csv").write_text("stock,action,target_weight\n600000.SH,buy,0.1\n", encoding="utf-8-sig")
    (run_dir / "holdings_snapshot.csv").write_text("stock,current_weight,target_weight\n600000.SH,0,0.1\n", encoding="utf-8-sig")
    (run_dir / "watchlist.csv").write_text("stock,score\n600001.SH,0.7\n", encoding="utf-8-sig")

    payload = app_service.latest_trade_plan_summary(output_dir=output_dir)

    assert payload["status"] == "ok"
    assert payload["summary"]["signal_date"] == "2026-05-22"
    assert payload["actions"][0]["stock"] == "600000.SH"
    assert payload["holdings"][0]["stock"] == "600000.SH"
    assert payload["watchlist"][0]["stock"] == "600001.SH"
    assert payload["model_info"]["trading_day_lag"] == 2
    assert payload["txt_preview"] == ["plan text", "line2"]
    assert payload["artifact_paths"]["run_dir"].endswith("20260523")


def test_trade_plan_summary_normalizes_external_model_info(tmp_path: Path) -> None:
    from daily_research.execution import app_service

    output_dir = tmp_path / "execution" / "output"
    run_dir = output_dir / "20260523"
    run_dir.mkdir(parents=True)
    (output_dir / "latest_trade_plan.txt").write_text("plan text\n", encoding="utf-8")
    (run_dir / "plan_summary.json").write_text(
        json.dumps(
            {
                "signal_date": "2026-05-22",
                "production_model_train_end_date": "2026-04-03",
                "production_model_launch_cutoff_date": "2026-04-21",
                "production_model_status": "fresh",
                "production_model_trading_day_lag": 20,
                "candidate_target_weight_csv": "target.csv",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    payload = app_service.latest_trade_plan_summary(output_dir=output_dir)

    assert payload["model_info"]["train_end_date"] == "2026-04-03"
    assert payload["model_info"]["launch_cutoff_date"] == "2026-04-21"
    assert payload["model_info"]["latest_completed_trading_date"] == "2026-05-22"
    assert payload["model_info"]["trading_day_lag"] == 20
    assert payload["model_info"]["status"] == "fresh"


def test_models_payload_includes_core_model_roles() -> None:
    from daily_research.execution import app_service

    payload = app_service.models_summary()
    roles = {item["role"] for item in payload["models"]}

    assert payload["status"] == "ok"
    assert {"live", "production", "path_policy", "legacy"} <= roles
    assert "continuous_policy" not in roles
    assert not any(item["id"] == "continuous-policy-shadow" for item in payload["models"])
    assert any(item["is_live"] for item in payload["models"])


def test_data_sources_payload_includes_lake_and_platform() -> None:
    from daily_research.execution import app_service

    payload = app_service.data_sources_summary()

    assert payload["status"] == "ok"
    assert "lake_root" in payload
    assert "datasets" in payload
    assert "data_platform" in payload


def test_data_sources_payload_exposes_default_refresh_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    from daily_research.execution import app_service

    monkeypatch.setattr(app_service, "get_latest_completed_trading_date", lambda: "2026-05-22", raising=False)

    payload = app_service.data_sources_summary()

    assert payload["data_platform"]["latest_completed_trading_date"] == "2026-05-22"
    assert payload["data_platform"]["recommended_domains"] == FORMAL_REFRESH_DOMAINS
    assert payload["data_platform"]["default_refresh"] == {
        "as_of_date": "2026-05-22",
        "universe": "all_a",
        "domains": FORMAL_REFRESH_DOMAINS,
        "provider_plan": "baostock_only",
    }


def test_data_refresh_api_defaults_to_completed_date_and_formal_domains(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from daily_research.execution import web_server

    captured: dict[str, list[str]] = {}

    def fake_launch_task_async(**kwargs):
        captured["passthrough_args"] = list(kwargs["passthrough_args"])
        return {"job_id": "job1", "task_name": kwargs["task_name"], "status": "queued"}

    monkeypatch.setattr(web_server.app_service, "get_latest_completed_trading_date", lambda: "2026-05-22", raising=False)
    monkeypatch.setattr(web_server.app_service, "launch_task_async", fake_launch_task_async)

    client = TestClient(web_server.create_app())
    response = client.post("/api/data-sources/refresh", json={})

    assert response.status_code == 200
    assert captured["passthrough_args"] == [
        "--as-of-date",
        "2026-05-22",
        "--universe",
        "all_a",
        "--provider-plan",
        "baostock_only",
        "--domains",
        ",".join(FORMAL_REFRESH_DOMAINS),
    ]
    assert "--timeout-seconds" not in captured["passthrough_args"]


def test_data_refresh_api_ignores_timeout_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from daily_research.execution import web_server

    captured: dict[str, list[str]] = {}

    def fake_launch_task_async(**kwargs):
        captured["passthrough_args"] = list(kwargs["passthrough_args"])
        return {"job_id": "job1", "task_name": kwargs["task_name"], "status": "queued"}

    monkeypatch.setattr(web_server.app_service, "get_latest_completed_trading_date", lambda: "2026-05-22", raising=False)
    monkeypatch.setattr(web_server.app_service, "launch_task_async", fake_launch_task_async)

    client = TestClient(web_server.create_app())
    response = client.post("/api/data-sources/refresh", json={"advanced_args": "--timeout-seconds 1"})

    assert response.status_code == 200
    assert "--timeout-seconds" not in captured["passthrough_args"]
    assert "1" not in captured["passthrough_args"]


def test_latest_policy_input_dataset_prefers_newest_end_date_then_creation() -> None:
    from daily_research.execution import app_service

    frame = pd.DataFrame(
        [
            {
                "dataset_id": "policy_input_bundle__old_created_later",
                "dataset_kind": "policy_input_bundle",
                "end_date": "2026-05-21",
                "created_at": "2026-05-24T10:00:00",
                "source": "data_platform_refresh",
                "status": "stored",
            },
            {
                "dataset_id": "policy_input_bundle__latest",
                "dataset_kind": "policy_input_bundle",
                "end_date": "2026-05-22",
                "created_at": "2026-05-24T00:18:28",
                "source": "data_platform_refresh",
                "status": "stored",
            },
            {
                "dataset_id": "not_policy_bundle",
                "dataset_kind": "gold",
                "end_date": "2026-05-23",
                "created_at": "2026-05-24T11:00:00",
                "source": "research",
                "status": "stored",
            },
        ]
    )

    selected = app_service._select_latest_policy_input_lake_dataset(frame)

    assert selected["dataset_id"] == "policy_input_bundle__latest"
    assert selected["end_date"] == "2026-05-22"


def test_sync_active_manifest_to_latest_lake_dataset_writes_execution_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from daily_research.execution import app_service

    active_manifest = tmp_path / "active_execution_strategy.json"
    active_manifest.write_text(
        json.dumps(
            {
                "strategy_name": "active",
                "data_source": "lake",
                "lake_dataset_id": "policy_input_bundle__old",
                "source_market_dataset_id": "policy_input_bundle__old",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(app_service, "ACTIVE_MANIFEST_PATH", active_manifest)
    monkeypatch.setattr(
        app_service,
        "_latest_policy_input_lake_dataset",
        lambda lake_root=None: {
            "dataset_id": "policy_input_bundle__latest",
            "end_date": "2026-05-22",
            "created_at": "2026-05-24T00:18:28",
            "source": "data_platform_refresh",
            "status": "stored",
        },
    )
    monkeypatch.setattr(app_service, "append_event", lambda *args, **kwargs: None)

    result = app_service.sync_active_manifest_to_latest_lake_dataset(
        reason="data-platform-refresh",
        refresh_manifest_path=str(tmp_path / "refresh_manifest.json"),
    )
    payload = json.loads(active_manifest.read_text(encoding="utf-8"))

    assert result["updated"] is True
    assert result["dataset_id"] == "policy_input_bundle__latest"
    assert payload["data_source"] == "lake"
    assert payload["lake_dataset_id"] == "policy_input_bundle__latest"
    assert payload["source_market_dataset_id"] == "policy_input_bundle__latest"
    assert payload["lake_dataset_end_date"] == "2026-05-22"
    assert payload["latest_data_refresh_manifest"].endswith("refresh_manifest.json")


def test_data_refresh_job_success_syncs_active_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from daily_research.execution import app_service

    script = tmp_path / "refresh_success.py"
    script.write_text("print('refresh ok')\n", encoding="utf-8")

    class FakeSpec:
        description = "refresh data"

    def fake_create_job_record(**kwargs):
        job_dir = tmp_path / "job"
        job_dir.mkdir(exist_ok=True)

        class Paths:
            job_id = "refresh_job"
            stdout_path = job_dir / "stdout.log"
            stderr_path = job_dir / "stderr.log"
            metadata_path = job_dir / "metadata.json"

        return Paths()

    def fake_mark_job_finished(job_paths, **kwargs):
        return {
            "job_id": job_paths.job_id,
            "status": kwargs["status"],
            "exit_code": kwargs["exit_code"],
            "summary_note": kwargs["summary_note"],
        }

    captured: dict[str, object] = {}

    def fake_sync(**kwargs):
        captured.update(kwargs)
        return {"updated": True, "dataset_id": "policy_input_bundle__latest"}

    monkeypatch.setattr(app_service, "get_task_spec", lambda _: FakeSpec())
    monkeypatch.setattr(app_service, "build_task_command", lambda **_: [sys.executable, str(script)])
    monkeypatch.setattr(app_service, "create_job_record", fake_create_job_record)
    monkeypatch.setattr(app_service, "mark_job_started", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_service, "mark_job_finished", fake_mark_job_finished)
    monkeypatch.setattr(app_service, "update_job_metadata", lambda job_paths, **kwargs: {"active_manifest_update": kwargs["active_manifest_update"]})
    monkeypatch.setattr(app_service, "append_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_service, "sync_active_manifest_to_latest_lake_dataset", fake_sync)

    class FakeLock:
        payload = {}

        def __init__(self, **_: object) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

    monkeypatch.setattr(app_service, "ExecutionAppLock", FakeLock)

    result = app_service.run_task_sync(task_name="data-platform-refresh", passthrough_args=[], echo_output=False)

    assert result["status"] == "succeeded"
    assert captured["reason"] == "data-platform-refresh"
    assert "policy_input_bundle__latest" in result["metadata"]["summary_note"]
    assert result["metadata"]["active_manifest_update"]["dataset_id"] == "policy_input_bundle__latest"


def test_data_refresh_reconciles_successful_manifest_after_runner_output_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from daily_research.execution import app_service

    script = tmp_path / "refresh_manifest_then_output_error.py"
    refresh_manifest_path = tmp_path / "refresh_run" / "refresh_manifest.json"
    refresh_manifest_text = str(refresh_manifest_path).replace("\\", "/")
    script.write_text(
        "from pathlib import Path\n"
        "import json\n"
        f"manifest_path = Path({refresh_manifest_text!r})\n"
        "manifest_path.parent.mkdir(exist_ok=True)\n"
        "manifest_path.write_text(json.dumps({\n"
        "    'status': 'ok',\n"
        "    'refresh_run_id': 'refresh_ok',\n"
        "    'registered_market_dataset_id': 'policy_input_bundle__latest',\n"
        "    'policy_input_dataset_id': 'policy_input_bundle__latest',\n"
        "}, ensure_ascii=False), encoding='utf-8')\n"
        "print('business refresh completed')\n",
        encoding="utf-8",
    )

    class FakeSpec:
        description = "refresh data"

    def fake_create_job_record(**kwargs):
        job_dir = tmp_path / "job"
        job_dir.mkdir(exist_ok=True)

        class Paths:
            job_id = "refresh_job_reconciled"
            stdout_path = job_dir / "stdout.log"
            stderr_path = job_dir / "stderr.log"
            metadata_path = job_dir / "metadata.json"

        return Paths()

    finished: dict[str, object] = {}

    def fake_mark_job_finished(job_paths, **kwargs):
        finished.update(kwargs)
        return {
            "job_id": job_paths.job_id,
            "status": kwargs["status"],
            "exit_code": kwargs["exit_code"],
            "summary_note": kwargs["summary_note"],
            "business_status": kwargs.get("business_status", ""),
            "runner_status": kwargs.get("runner_status", ""),
            "runner_warnings": kwargs.get("runner_warnings", []),
            "artifact_status": kwargs.get("artifact_status", ""),
            "artifact_paths": kwargs.get("artifact_paths", {}),
        }

    captured: dict[str, object] = {}

    def fake_sync(**kwargs):
        captured.update(kwargs)
        return {"updated": True, "dataset_id": "policy_input_bundle__latest"}

    class BrokenFlush:
        def write(self, text: str) -> int:
            return len(text)

        def flush(self) -> None:
            raise OSError(22, "Invalid argument")

    monkeypatch.setattr(app_service, "get_task_spec", lambda _: FakeSpec())
    monkeypatch.setattr(app_service, "build_task_command", lambda **_: [sys.executable, str(script)])
    monkeypatch.setattr(app_service, "create_job_record", fake_create_job_record)
    monkeypatch.setattr(app_service, "mark_job_started", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_service, "mark_job_finished", fake_mark_job_finished)
    monkeypatch.setattr(app_service, "update_job_metadata", lambda job_paths, **kwargs: dict(kwargs))
    monkeypatch.setattr(app_service, "append_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_service, "sync_active_manifest_to_latest_lake_dataset", fake_sync)
    monkeypatch.setattr(app_service, "_latest_refresh_manifest_path", lambda: str(refresh_manifest_path.resolve()))
    monkeypatch.setattr(app_service.sys, "stdout", BrokenFlush())

    class FakeLock:
        payload = {}

        def __init__(self, **_: object) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

    monkeypatch.setattr(app_service, "ExecutionAppLock", FakeLock)

    result = app_service.run_task_sync(
        task_name="data-platform-refresh",
        passthrough_args=[],
        echo_output=True,
    )

    assert result["status"] == "succeeded"
    assert result["metadata"]["business_status"] == "ok"
    assert result["metadata"]["runner_status"] == "warning"
    assert result["metadata"]["runner_warnings"]
    assert result["metadata"]["artifact_status"] == "ok"
    assert result["metadata"]["artifact_paths"]["refresh_manifest"].endswith("refresh_manifest.json")
    assert captured["refresh_manifest_path"].endswith("refresh_manifest.json")
    assert finished["exit_code"] == 0


def test_data_refresh_post_process_persists_evidence_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from daily_research.execution import app_service

    refresh_manifest_path = tmp_path / "refresh_run" / "refresh_manifest.json"
    refresh_manifest_path.parent.mkdir()
    refresh_manifest_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "registered_market_dataset_id": "policy_input_bundle__latest",
                "policy_input_dataset_id": "policy_input_bundle__latest",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    class Paths:
        job_id = "refresh_job"
        metadata_path = tmp_path / "metadata.json"

    persisted: dict[str, object] = {}

    def fake_update(job_paths, **kwargs):
        persisted.update(kwargs)
        return dict(persisted)

    monkeypatch.setattr(app_service, "_latest_refresh_manifest_path", lambda: str(refresh_manifest_path.resolve()))
    monkeypatch.setattr(app_service, "update_job_metadata", fake_update)
    monkeypatch.setattr(
        app_service,
        "sync_active_manifest_to_latest_lake_dataset",
        lambda **kwargs: {"updated": True, "dataset_id": "policy_input_bundle__latest"},
    )

    _, metadata = app_service._post_process_successful_task(
        job_paths=Paths(),
        task_name="data-platform-refresh",
        summary_note="refresh data",
    )

    assert metadata["business_status"] == "ok"
    assert persisted["business_status"] == "ok"
    assert persisted["artifact_status"] == "ok"
    assert persisted["artifact_paths"]["refresh_manifest"].endswith("refresh_manifest.json")
    assert persisted["active_manifest_update"]["dataset_id"] == "policy_input_bundle__latest"


def test_blocked_refresh_manifest_does_not_sync_active_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from daily_research.execution import app_service

    refresh_manifest_path = tmp_path / "refresh_run" / "refresh_manifest.json"
    refresh_manifest_path.parent.mkdir()
    refresh_manifest_path.write_text(
        json.dumps(
            {
                "status": "blocked",
                "blockers": ["coverage_below_threshold"],
                "registered_market_dataset_id": "",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    class Paths:
        job_id = "refresh_blocked"
        metadata_path = tmp_path / "metadata.json"

    persisted: dict[str, object] = {}

    monkeypatch.setattr(app_service, "_latest_refresh_manifest_path", lambda: str(refresh_manifest_path.resolve()))
    monkeypatch.setattr(app_service, "update_job_metadata", lambda job_paths, **kwargs: persisted.update(kwargs) or dict(persisted))
    monkeypatch.setattr(
        app_service,
        "sync_active_manifest_to_latest_lake_dataset",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("blocked refresh must not sync active manifest")),
    )

    note, metadata = app_service._post_process_successful_task(
        job_paths=Paths(),
        task_name="data-platform-refresh",
        summary_note="refresh data",
    )

    assert "blocked" in note
    assert metadata["business_status"] == "blocked"
    assert "active_manifest_update" not in metadata
    assert persisted["artifact_status"] == "blocked"


def test_blocked_refresh_manifest_marks_job_blocked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from daily_research.execution import app_service

    script = tmp_path / "refresh_blocked.py"
    refresh_manifest_path = tmp_path / "refresh_run" / "refresh_manifest.json"
    refresh_manifest_text = str(refresh_manifest_path).replace("\\", "/")
    script.write_text(
        "from pathlib import Path\n"
        "import json\n"
        f"manifest_path = Path({refresh_manifest_text!r})\n"
        "manifest_path.parent.mkdir(exist_ok=True)\n"
        "manifest_path.write_text(json.dumps({\n"
        "    'status': 'blocked',\n"
        "    'blockers': ['coverage_below_threshold'],\n"
        "    'registered_market_dataset_id': '',\n"
        "}, ensure_ascii=False), encoding='utf-8')\n"
        "print('refresh blocked by coverage')\n",
        encoding="utf-8",
    )

    class FakeSpec:
        description = "refresh data"

    def fake_create_job_record(**kwargs):
        job_dir = tmp_path / "job"
        job_dir.mkdir(exist_ok=True)

        class Paths:
            job_id = "refresh_blocked_job"
            stdout_path = job_dir / "stdout.log"
            stderr_path = job_dir / "stderr.log"
            metadata_path = job_dir / "metadata.json"

        return Paths()

    monkeypatch.setattr(app_service, "get_task_spec", lambda _: FakeSpec())
    monkeypatch.setattr(app_service, "build_task_command", lambda **_: [sys.executable, str(script)])
    monkeypatch.setattr(app_service, "create_job_record", fake_create_job_record)
    monkeypatch.setattr(app_service, "mark_job_started", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_service, "append_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_service, "_latest_refresh_manifest_path", lambda: str(refresh_manifest_path.resolve()))
    monkeypatch.setattr(
        app_service,
        "sync_active_manifest_to_latest_lake_dataset",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("blocked refresh must not sync active manifest")),
    )

    class FakeLock:
        payload = {}

        def __init__(self, **_: object) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

    monkeypatch.setattr(app_service, "ExecutionAppLock", FakeLock)

    result = app_service.run_task_sync(task_name="data-platform-refresh", passthrough_args=[], echo_output=False)

    assert result["status"] == "blocked"
    assert result["exit_code"] == 0
    assert result["metadata"]["business_status"] == "blocked"
    assert result["metadata"]["runner_status"] == "ok"
    assert result["metadata"]["artifact_status"] == "blocked"


def test_missing_refresh_manifest_marks_job_failed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from daily_research.execution import app_service

    script = tmp_path / "refresh_without_manifest.py"
    script.write_text("print('refresh script exited without manifest')\n", encoding="utf-8")
    missing_manifest_path = tmp_path / "missing" / "refresh_manifest.json"

    class FakeSpec:
        description = "refresh data"

    def fake_create_job_record(**kwargs):
        job_dir = tmp_path / "job"
        job_dir.mkdir(exist_ok=True)

        class Paths:
            job_id = "refresh_missing_manifest"
            stdout_path = job_dir / "stdout.log"
            stderr_path = job_dir / "stderr.log"
            metadata_path = job_dir / "metadata.json"

        return Paths()

    monkeypatch.setattr(app_service, "get_task_spec", lambda _: FakeSpec())
    monkeypatch.setattr(app_service, "build_task_command", lambda **_: [sys.executable, str(script)])
    monkeypatch.setattr(app_service, "create_job_record", fake_create_job_record)
    monkeypatch.setattr(app_service, "mark_job_started", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_service, "append_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_service, "_latest_refresh_manifest_path", lambda: str(missing_manifest_path.resolve()))
    monkeypatch.setattr(
        app_service,
        "sync_active_manifest_to_latest_lake_dataset",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("missing refresh manifest must not sync active manifest")),
    )

    class FakeLock:
        payload = {}

        def __init__(self, **_: object) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

    monkeypatch.setattr(app_service, "ExecutionAppLock", FakeLock)

    result = app_service.run_task_sync(task_name="data-platform-refresh", passthrough_args=[], echo_output=False)

    assert result["status"] == "failed"
    assert result["exit_code"] == 0
    assert result["metadata"]["business_status"] == "missing"
    assert result["metadata"]["runner_status"] == "ok"
    assert result["metadata"]["artifact_status"] == "missing"


def test_tee_text_io_compresses_repeated_carriage_progress() -> None:
    from daily_research.execution.app_service import _TeeTextIO

    handle = StringIO()
    heartbeats: list[str] = []
    tee = _TeeTextIO(handle, heartbeat=heartbeats.append)

    tee.write("stage 1\r")
    tee.write("stage 1\r")
    tee.write("stage 2\r")
    tee.write("stage 2\r")
    tee.write("done\n")
    tee.flush()

    assert handle.getvalue().splitlines() == ["stage 1", "stage 2", "done"]
    assert heartbeats == ["stage 1", "stage 2", "done"]


def test_task_launch_strips_timeout_args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from daily_research.execution import app_service

    captured: dict[str, list[str]] = {}

    def fake_build_task_command(**kwargs):
        captured["passthrough_args"] = list(kwargs["passthrough_args"])
        return [sys.executable, str(tmp_path / "noop.py"), *captured["passthrough_args"]]

    def fake_create_job_record(**kwargs):
        job_dir = tmp_path / "job"
        job_dir.mkdir(exist_ok=True)

        class Paths:
            job_id = "job_without_timeout"
            stdout_path = job_dir / "stdout.log"
            stderr_path = job_dir / "stderr.log"
            metadata_path = job_dir / "metadata.json"

        return Paths()

    monkeypatch.setattr(app_service, "build_task_command", fake_build_task_command)
    monkeypatch.setattr(app_service, "create_job_record", fake_create_job_record)
    monkeypatch.setattr(app_service, "append_event", lambda *args, **kwargs: None)

    class FakeThread:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def start(self) -> None:
            return None

        def is_alive(self) -> bool:
            return False

    monkeypatch.setattr(app_service.threading, "Thread", FakeThread)

    app_service.launch_task_async(
        task_name="data-platform-refresh",
        passthrough_args=["--as-of-date", "2026-05-22", "--timeout-seconds", "1", "--universe", "all_a"],
    )

    assert captured["passthrough_args"] == ["--as-of-date", "2026-05-22", "--universe", "all_a"]


def test_run_task_sync_executes_in_current_process_without_popen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from daily_research.execution import app_service

    script = tmp_path / "inline_task.py"
    script.write_text("import sys\nprint('inline task argv=' + '|'.join(sys.argv[1:]))\n", encoding="utf-8")

    class FakeSpec:
        description = "inline task"

    def fake_create_job_record(**kwargs):
        job_dir = tmp_path / "job"
        job_dir.mkdir(exist_ok=True)

        class Paths:
            job_id = "inline_job"
            stdout_path = job_dir / "stdout.log"
            stderr_path = job_dir / "stderr.log"

        return Paths()

    monkeypatch.setattr(app_service, "get_task_spec", lambda _: FakeSpec())
    monkeypatch.setattr(app_service, "build_task_command", lambda **_: [sys.executable, str(script), "--flag", "x"])
    monkeypatch.setattr(app_service, "create_job_record", fake_create_job_record)
    monkeypatch.setattr(app_service, "mark_job_started", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_service, "mark_job_finished", lambda job_paths, **kwargs: {"status": kwargs["status"], "exit_code": kwargs["exit_code"]})
    monkeypatch.setattr(app_service, "append_event", lambda *args, **kwargs: None)

    class FakeLock:
        payload = {}

        def __init__(self, **_: object) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

    monkeypatch.setattr(app_service, "ExecutionAppLock", FakeLock)

    assert not hasattr(app_service, "subprocess")

    result = app_service.run_task_sync(task_name="inline", passthrough_args=["--flag", "x"], echo_output=False)

    assert result["status"] == "succeeded"
    assert "inline task argv=--flag|x" in (tmp_path / "job" / "stdout.log").read_text(encoding="utf-8")


def test_status_payload_omits_continuous_policy() -> None:
    from daily_research.execution import app_service

    payload = app_service.build_status_payload(history_limit=1)

    assert "continuous_policy" not in payload


def test_continuous_policy_api_is_removed_from_execution_console() -> None:
    from fastapi.testclient import TestClient

    from daily_research.execution import web_server

    client = TestClient(web_server.create_app())

    assert client.get("/api/continuous-policy").status_code == 404
    assert client.get("/api/models/continuous-policy-shadow").status_code == 404
    assert client.get("/continuous-policy").status_code == 404


def test_ui_paths_include_react_dist() -> None:
    from daily_research.execution.web_service import ui_paths

    payload = ui_paths()

    assert payload["react_dist"].name == "dist"
    assert payload["react_dist"].parent.name == "webapp"


def test_trade_plan_generate_preserves_windows_paths_with_spaces(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from daily_research.execution import web_server

    captured: dict[str, list[str]] = {}

    def fake_launch_task_async(**kwargs):
        captured["passthrough_args"] = list(kwargs["passthrough_args"])
        return {"job_id": "job1", "task_name": kwargs["task_name"], "status": "queued"}

    monkeypatch.setattr(web_server.app_service, "launch_task_async", fake_launch_task_async)

    client = TestClient(web_server.create_app())
    response = client.post(
        "/api/trade-plan/generate",
        json={
            "candidate_profile": "active_execution_strategy",
            "positions_file": "H:/quant project/current positions.csv",
            "cash": "12345.67",
        },
    )

    assert response.status_code == 200
    assert captured["passthrough_args"] == [
        "--candidate-profile",
        "active_execution_strategy",
        "--positions-file",
        "H:/quant project/current positions.csv",
        "--cash",
        "12345.67",
    ]


def test_model_train_rejects_non_production_models(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from daily_research.execution import web_server

    called = {"launch": False}

    def fake_launch_task_async(**kwargs):
        called["launch"] = True
        return {"job_id": "job1", "task_name": kwargs["task_name"], "status": "queued"}

    monkeypatch.setattr(web_server.app_service, "launch_task_async", fake_launch_task_async)

    client = TestClient(web_server.create_app())
    response = client.post("/api/models/path-policy-research/train", json={"dataset_mode": "latest"})

    assert response.status_code == 400
    assert called["launch"] is False


def test_production_model_train_preserves_explicit_window(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from daily_research.execution import web_server

    captured: dict[str, list[str]] = {}

    def fake_launch_task_async(**kwargs):
        captured["passthrough_args"] = list(kwargs["passthrough_args"])
        return {"job_id": "job1", "task_name": kwargs["task_name"], "status": "queued"}

    monkeypatch.setattr(web_server.app_service, "launch_task_async", fake_launch_task_async)

    client = TestClient(web_server.create_app())
    response = client.post(
        "/api/models/production-full-fit/train",
        json={"dataset_mode": "custom", "start_date": "20250101", "end_date": "20260522"},
    )

    assert response.status_code == 200
    assert captured["passthrough_args"] == ["--start-date", "20250101", "--end-date", "20260522"]


def test_react_build_is_primary_ui_when_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from daily_research.execution import web_server

    react_dist = tmp_path / "dist"
    react_dist.mkdir()
    (react_dist / "index.html").write_text('<div id="root">react shell</div>', encoding="utf-8")
    monkeypatch.setitem(web_server.UI_PATHS, "react_dist", react_dist)

    client = TestClient(web_server.create_app())

    assert "react shell" in client.get("/").text
    assert "react shell" in client.get("/models").text


def test_trade_plan_profile_defaults_do_not_auto_retrain(monkeypatch: pytest.MonkeyPatch) -> None:
    from daily_research.execution import research_candidate_profiles as profiles

    called = {"auto": False}

    def fake_auto(*args, **kwargs) -> None:
        called["auto"] = True

    monkeypatch.setattr(profiles, "_maybe_auto_retrain_production", fake_auto)
    monkeypatch.setattr(profiles, "_panel_is_fresh", lambda *args, **kwargs: True)

    profiles.apply_profile_defaults("active_execution_strategy", mode="trade_plan", ensure_live_panels=True)

    assert called["auto"] is False


def test_trade_plan_profile_defaults_do_not_refresh_live_panels_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from daily_research.execution import research_candidate_profiles as profiles

    called = {"refresh": False}

    def fake_refresh(*args, **kwargs) -> None:
        called["refresh"] = True
        raise AssertionError("trade-plan defaults must not refresh live panels implicitly")

    monkeypatch.setattr(profiles, "_ensure_live_panels", fake_refresh)

    profiles.apply_profile_defaults("active_execution_strategy", mode="trade_plan")

    assert called["refresh"] is False


def test_active_profile_uses_current_lake_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from daily_research.execution import research_candidate_profiles as profiles

    project_daily_research = tmp_path / "daily_research"
    production_dir = project_daily_research / "output" / "short_expert_policy_v5b_execalign_production_default"
    source_dir = project_daily_research / "output" / "short_alpha_policy_v5_family_formal_review_20260412_r1" / "runs" / "short_expert_policy_v5b"
    source_dir.mkdir(parents=True)
    production_dir.mkdir(parents=True)
    (source_dir / "execution_aligned_daily_live_target_weight_panel.csv").write_text(
        "date,stock,target_weight\n",
        encoding="utf-8",
    )
    (source_dir / "execution_aligned_daily_live_score_panel.csv").write_text(
        "date,stock,score\n",
        encoding="utf-8",
    )
    (production_dir / "execution_aligned_daily_live_target_weight_panel.csv").write_text(
        "date,stock,target_weight\n",
        encoding="utf-8",
    )
    (production_dir / "execution_aligned_daily_live_score_panel.csv").write_text(
        "date,stock,score\n",
        encoding="utf-8",
    )
    (production_dir / "production_retrain_manifest.json").write_text("{}", encoding="utf-8")
    (production_dir / "daily_live_score_reference.json").write_text("{}", encoding="utf-8")

    manifest = {
        "source_target_weight_panel_csv": str(source_dir / "execution_aligned_daily_live_target_weight_panel.csv"),
        "source_score_panel_csv": str(source_dir / "execution_aligned_daily_live_score_panel.csv"),
        "trade_plan_target_weight_panel_csv": str(
            production_dir / "execution_aligned_daily_live_target_weight_panel.csv"
        ),
        "trade_plan_score_panel_csv": str(production_dir / "execution_aligned_daily_live_score_panel.csv"),
        "production_manifest_json": str(production_dir / "production_retrain_manifest.json"),
        "score_reference_metadata_json": str(production_dir / "daily_live_score_reference.json"),
        "candidate_label": "active",
        "liquidity_pool_name": "liquid500",
        "data_source": "lake",
        "lake_dataset_id": "policy_input_bundle__x",
        "data_lake_root": str(project_daily_research / "output" / "research_data_lake"),
    }
    monkeypatch.setattr(profiles, "_DAILY_RESEARCH_ROOT", project_daily_research)
    monkeypatch.setattr(profiles, "load_strategy_manifest", lambda: manifest)

    profile = profiles.get_profile("active_execution_strategy")

    assert profile.trade_plan_target_weight_panel_csv == str(
        (production_dir / "execution_aligned_daily_live_target_weight_panel.csv").resolve()
    )
    assert profile.trade_plan_score_panel_csv == str(
        (production_dir / "execution_aligned_daily_live_score_panel.csv").resolve()
    )
    assert profile.trade_plan_model_manifest_json == str((production_dir / "production_retrain_manifest.json").resolve())
    assert profile.score_reference_metadata_json == str((production_dir / "daily_live_score_reference.json").resolve())
    assert profile.data_source == "lake"
    assert profile.lake_dataset_id == "policy_input_bundle__x"
    assert profile.data_lake_root == str(project_daily_research / "output" / "research_data_lake")


def test_model_freshness_lag_is_informational_not_blocking() -> None:
    from daily_research.baseline.generate_daily_trade_plan import _assess_model_freshness

    calendar = pd.DatetimeIndex(pd.to_datetime(["2026-05-18", "2026-05-19", "2026-05-20", "2026-05-21"]))
    payload = _assess_model_freshness(
        artifact_meta={"latest_data_date": "2026-05-18", "trained_at": "2026-05-18T20:00:00"},
        latest_signal_date=pd.Timestamp("2026-05-21"),
        trading_dates=calendar,
        warn_trading_days=1,
        max_trading_days=2,
    )

    assert payload["trading_day_lag"] == 3
    assert payload["status"] == "lagged"
    assert payload["status_text"] == "lagged"
    assert payload["should_block"] is False
    assert payload["warnings"] == []


def test_trade_plan_txt_renders_model_dates_as_information(tmp_path: Path) -> None:
    from daily_research.baseline.generate_daily_trade_plan import _write_trade_plan_txt

    path = tmp_path / "plan.txt"
    _write_trade_plan_txt(
        path,
        summary={
            "signal_date": "2026-05-22",
            "execution_date": "2026-05-25",
            "cash_input": 100000.0,
            "total_equity": 100000.0,
            "estimated_cash_after_plan": 100000.0,
            "current_position_count": 0,
            "blocked_buy_candidate_count": 0,
        },
        regime_state_row=pd.Series({"quadrant": "trend_up_low_vol", "regime_on": True}),
        action_df=pd.DataFrame(),
        hold_df=pd.DataFrame(),
        watch_df=pd.DataFrame(),
        model_info={
            "mode": "research_candidate_target_weight_csv",
            "market_regime_filter_enabled": False,
            "production_model_train_end_date": "2026-04-03",
            "production_model_launch_cutoff_date": "2026-04-21",
            "latest_completed_trading_date": "2026-05-22",
            "production_model_retrain_trading_day_lag": 20,
            "production_model_retrain_policy": "Retrain Every 21 Trading Days",
            "production_model_retrain_status": "fresh",
            "production_model_retrain_window": "2025-03-18 -> 2026-03-27",
            "freshness_status": "lagged",
            "freshness_label": "候选信号新鲜度",
            "trading_day_lag": 3,
        },
    )

    text = path.read_text(encoding="utf-8")

    assert "底层模型训练样本截止: 2026-04-03" in text
    assert "底层模型最近一次上线截止: 2026-04-21" in text
    assert "最新完成交易日: 2026-05-22" in text
    assert "底层模型距最新交易日: 20 个交易日" in text
    assert "候选信号距最新交易日: 3 个交易日" in text
    assert "lagged" not in text
    assert "fresh" not in text
    assert "候选信号间隔" not in text
    assert "重训策略" not in text
    assert "重训时效" not in text
    assert "模型新鲜度" not in text
    assert "候选信号新鲜度" not in text
    assert "提醒阈值" not in text
    assert "阻断" not in text
