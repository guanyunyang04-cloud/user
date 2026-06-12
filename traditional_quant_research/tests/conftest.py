from __future__ import annotations

from pathlib import Path

import pytest


RESEARCH_NAME_TOKENS = (
    "audit",
    "breakout",
    "candidate",
    "diagnostics",
    "experiment",
    "factor_family",
    "factor_pruning",
    "first_loop_research",
    "frontier",
    "grid",
    "kama",
    "limitup",
    "low_corr",
    "metrics_exposure",
    "model_zoo",
    "personal",
    "probe",
    "promotion_gate",
    "rebuild",
    "regime",
    "scout",
    "short_event",
    "short_open",
    "strategy_research",
    "trial_ledger",
    "validation",
    "weak_year",
)

EXTERNAL_NAME_TOKENS = (
    "external",
    "free_size",
    "tushare",
    "cninfo",
)

DATA_HEAVY_NAME_TOKENS = (
    "dataset_builder",
    "daily_metrics",
    "daily_size",
    "industry_size",
)


def pytest_configure(config) -> None:  # type: ignore[no-untyped-def]
    for marker in (
        "unit: fast deterministic unit tests",
        "smoke: fast changed-surface smoke tests",
        "integration: subprocess, CLI, filesystem, or service contract tests",
        "slow: tests that exceed the normal local loop or may be environment-sensitive",
        "research: experiment, backtest, scout, audit, grid, or research-regression tests",
        "data_heavy: tests that build or validate large datasets or cached research panels",
        "external: tests requiring network, real providers, live probes, or third-party services",
        "benchmark: performance threshold tests",
    ):
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items) -> None:  # type: ignore[no-untyped-def]
    for item in items:
        filename = Path(str(item.fspath)).name.lower()
        stem = filename.removeprefix("test_").removesuffix(".py")
        if any(token in stem for token in RESEARCH_NAME_TOKENS):
            item.add_marker(pytest.mark.research)
        if any(token in stem for token in EXTERNAL_NAME_TOKENS):
            item.add_marker(pytest.mark.external)
        if any(token in stem for token in DATA_HEAVY_NAME_TOKENS):
            item.add_marker(pytest.mark.data_heavy)
