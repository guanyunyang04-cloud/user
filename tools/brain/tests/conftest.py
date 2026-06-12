from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SKILL_CACHE = ROOT / "brain" / "skills" / "workspace-brain" / "scripts" / "__pycache__"

SMOKE_FILES = {
    "test_capsule.py",
    "test_doc_guard.py",
    "test_platform.py",
    "test_project_commit.py",
    "test_rules.py",
    "test_selective_verification.py",
}

INTEGRATION_FILES = {
    "test_agent_run.py",
    "test_integrity_check.py",
    "test_long_task_monitor.py",
    "test_workflow_cli.py",
    "test_workspace_brain_runtime_init.py",
    "test_workspace_brain_skill_install.py",
}

SLOW_FILES = {
    "test_workspace_brain_skill_install.py",
}


sys.dont_write_bytecode = True


def pytest_configure(config) -> None:  # type: ignore[no-untyped-def]
    for marker in (
        "unit: fast deterministic unit tests",
        "integration: subprocess, CLI, filesystem, or service contract tests",
        "slow: tests that exceed the normal local loop or may be environment-sensitive",
        "research: model training, forecast/RL protocol, or research regression tests",
        "guard: doc guard, integrity, health, or brain audit checks",
        "external: tests requiring network, real providers, or production-like dependencies",
        "smoke: fast changed-surface smoke tests",
        "benchmark: performance threshold tests",
    ):
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        filename = Path(str(getattr(item, "path", getattr(item, "fspath", "")))).name
        if filename in SMOKE_FILES:
            item.add_marker(pytest.mark.smoke)
        if filename in INTEGRATION_FILES:
            item.add_marker(pytest.mark.integration)
        if filename in SLOW_FILES:
            item.add_marker(pytest.mark.slow)
        if filename.startswith("test_workspace_brain_") or filename in {"test_integrity_check.py", "test_doc_guard.py"}:
            item.add_marker(pytest.mark.guard)


def pytest_sessionfinish(session, exitstatus) -> None:  # type: ignore[no-untyped-def]
    if SKILL_CACHE.exists():
        shutil.rmtree(SKILL_CACHE)
