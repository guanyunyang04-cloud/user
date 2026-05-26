from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_current_execution_docs_are_manual_only() -> None:
    docs = [
        PROJECT_ROOT / "daily_research" / "execution" / "使用教程.md",
        PROJECT_ROOT / "daily_research" / "brain" / "state_center.md",
        PROJECT_ROOT / "daily_research" / "brain" / "knowledge_center.md",
        PROJECT_ROOT / "daily_research" / "brain" / "operations_center.md",
    ]
    forbidden_terms = [
        "Windows Task Scheduler",
        "DailyResearchDailyPlan",
        "daily_plan_runner",
        "scheduler_cli",
        "daily-plan-runner",
        "/api/daily-run/run",
        "scheduler_not_installed",
        "scheduler_disabled",
        "自动盘后",
        "自动调度",
    ]

    offenders = {
        str(path.relative_to(PROJECT_ROOT)): [term for term in forbidden_terms if term in path.read_text(encoding="utf-8")]
        for path in docs
    }

    assert offenders == {str(path.relative_to(PROJECT_ROOT)): [] for path in docs}
