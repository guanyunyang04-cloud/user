from __future__ import annotations

import subprocess
import sys

from daily_research.execution.app_tasks import build_task_command, get_task_spec, list_core_frontend_task_specs, list_task_specs


def test_danger_tasks_are_marked_danger() -> None:
    assert get_task_spec("refresh-production-default").safety_level == "danger"
    assert get_task_spec("activate-single-mapping").safety_level == "danger"


def test_trade_plan_remains_safe_and_follows_active_strategy() -> None:
    spec = get_task_spec("trade-plan")

    assert spec.safety_level == "safe"
    assert "run_trade_plan.py" in spec.script_relative_path
    assert any("active_execution_strategy" in note for note in spec.launcher_notes)


def test_registered_task_scripts_exist() -> None:
    missing = [spec.name for spec in list_task_specs() if not spec.script_path.exists()]

    assert missing == []


def test_execution_task_registry_omits_continuous_policy_surface() -> None:
    task_names = {spec.name for spec in list_task_specs()}

    assert "continuous-policy-protocol" not in task_names
    assert "continuous-policy-train" not in task_names
    assert "continuous-policy-evaluate" not in task_names
    assert "continuous-policy-export" not in task_names


def test_default_frontend_task_surface_hides_dangerous_production_refresh() -> None:
    task_names = {spec.name for spec in list_core_frontend_task_specs()}

    assert "trade-plan" in task_names
    assert "refresh-production-default" not in task_names


def test_provider_health_check_runs_from_execution_task_command() -> None:
    command = build_task_command(
        task_name="provider-health-check",
        python_executable=sys.executable,
        passthrough_args=["--domains", "market_daily", "--symbols", "000001.SZ", "--json"],
    )

    result = subprocess.run(command, cwd=".", text=True, capture_output=True, timeout=120)

    assert result.returncode in {0, 2}
    assert "ModuleNotFoundError" not in result.stderr
    assert "provider_plan" in result.stdout
