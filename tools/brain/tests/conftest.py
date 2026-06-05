from __future__ import annotations

import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SKILL_CACHE = ROOT / "brain" / "skills" / "workspace-brain" / "scripts" / "__pycache__"


sys.dont_write_bytecode = True


def pytest_configure(config) -> None:  # type: ignore[no-untyped-def]
    for marker in (
        "unit: fast deterministic unit tests",
        "integration: subprocess, CLI, filesystem, or service contract tests",
        "slow: tests that exceed the normal local loop or may be environment-sensitive",
        "research: model training, forecast/RL protocol, or research regression tests",
        "guard: doc guard, integrity, health, or brain audit checks",
        "external: tests requiring network, real providers, or production-like dependencies",
    ):
        config.addinivalue_line("markers", marker)


def pytest_sessionfinish(session, exitstatus) -> None:  # type: ignore[no-untyped-def]
    if SKILL_CACHE.exists():
        shutil.rmtree(SKILL_CACHE)
