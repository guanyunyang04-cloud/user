from __future__ import annotations


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
