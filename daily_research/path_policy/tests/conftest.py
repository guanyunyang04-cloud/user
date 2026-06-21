from __future__ import annotations

from pathlib import Path

import pytest


DATA_HEAVY_FILES = {
    "test_forecast_memmap_dataset.py",
}

RESEARCH_SLOW_FILES = {
    "test_forecast_training.py",
}

TRAINING_FILES = {
    "test_forecast_training.py",
}


def pytest_configure(config) -> None:  # type: ignore[no-untyped-def]
    for marker in (
        "unit: fast deterministic unit tests",
        "smoke: fast changed-surface smoke tests",
        "integration: subprocess, CLI, filesystem, or service contract tests",
        "slow: tests that exceed the normal local loop or may be environment-sensitive",
        "research: model training, forecast/RL protocol, or research regression tests",
        "data_heavy: tests that build or validate large datasets, memmaps, or lake artifacts",
        "training: tests that instantiate or exercise model training loops",
        "guard: doc guard, integrity, health, or brain audit checks",
        "external: tests requiring network, real providers, or production-like dependencies",
        "benchmark: performance threshold tests",
    ):
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items) -> None:  # type: ignore[no-untyped-def]
    for item in items:
        filename = Path(str(item.fspath)).name
        if filename in RESEARCH_SLOW_FILES:
            item.add_marker(pytest.mark.research)
            item.add_marker(pytest.mark.slow)
        if filename in DATA_HEAVY_FILES:
            item.add_marker(pytest.mark.data_heavy)
            item.add_marker(pytest.mark.slow)
        if filename in TRAINING_FILES:
            item.add_marker(pytest.mark.training)
