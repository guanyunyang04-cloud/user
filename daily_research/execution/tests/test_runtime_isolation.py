from __future__ import annotations

from pathlib import Path


def test_runtime_root_can_be_retargeted_without_touching_real_execution_app(tmp_path: Path) -> None:
    from daily_research.execution import app_runtime, app_service

    real_runtime_root = app_runtime.PROJECT_ROOT / "output" / "execution_app"
    runtime_root = tmp_path / "isolated_runtime"

    app_runtime.configure_runtime_root(runtime_root)
    app_service.configure_runtime_root(runtime_root)

    job_paths = app_runtime.create_job_record(
        task_name="isolation_probe",
        command=["python", "-c", "print('ok')"],
        cwd=tmp_path,
        python_executable="python",
        passthrough_args=[],
    )

    assert runtime_root in job_paths.job_root.parents
    assert job_paths.metadata_path.exists()
    assert not (real_runtime_root / "jobs" / job_paths.job_id).exists()
    assert app_service.RUNTIME_ROOT == runtime_root
    assert app_service.PAPER_ACCOUNT_DB_PATH == runtime_root / "paper_account" / "paper_account.sqlite3"


def test_execution_tests_default_to_isolated_runtime_root() -> None:
    from daily_research.execution import app_runtime

    real_runtime_root = app_runtime.PROJECT_ROOT / "output" / "execution_app"

    assert app_runtime.RUNTIME_ROOT != real_runtime_root
    assert "pytest" in app_runtime.RUNTIME_ROOT.as_posix() or "tmp" in app_runtime.RUNTIME_ROOT.as_posix().lower()


def test_daily_verdict_follows_retargeted_runtime_root(tmp_path: Path) -> None:
    from daily_research.execution import app_runtime, daily_verdict

    runtime_root = tmp_path / "runtime"
    app_runtime.configure_runtime_root(runtime_root)

    verdict_dir = runtime_root / "daily_runs" / "20260527"
    verdict_dir.mkdir(parents=True)
    app_runtime.write_json_file(verdict_dir / "verdict.json", {"run_date": "2026-05-27", "status": "blocked"})

    payload = daily_verdict.daily_run_status()

    assert payload["latest_run_date"] == "2026-05-27"
    assert payload["status"] == "blocked"
    assert payload["daily_runs_root"] == str((runtime_root / "daily_runs").resolve())
