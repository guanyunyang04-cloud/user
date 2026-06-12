from __future__ import annotations

from pathlib import Path
import re

import pytest


SMOKE_FILES = {
    "data_platform/tests/test_domain_contract.py",
    "data_platform/tests/test_import_csv.py",
    "path_policy/tests/test_adapter.py",
    "path_policy/tests/test_labels.py",
    "tools/tests/test_doc_guard.py",
}

RESEARCH_PATH_PARTS = {
    "continuous_policy/tests",
}

RESEARCH_NAME_TOKENS = (
    "audit",
    "attribution",
    "backtest",
    "bad_month",
    "calibration",
    "candidate",
    "diagnostics",
    "discovery",
    "frontier",
    "horizon",
    "lineage",
    "local_risk",
    "local_state",
    "matrix",
    "protocol",
    "rebuild",
    "release_first",
    "reset_baseline",
    "scout",
    "score",
    "stage",
    "traditional_pit",
    "training",
)

DATA_HEAVY_PATH_PARTS = {
    "data_lake/tests",
    "deep_alpha",
}

DATA_HEAVY_NAME_TOKENS = (
    "canonical",
    "dataset",
    "gold_training",
    "import_external_quant_zip",
    "import_tdx_5m",
    "lake",
    "memmap",
)

EXTERNAL_NAME_TOKENS = (
    "baostock",
    "external",
    "provider_health",
    "tdx",
    "tq",
)

INTEGRATION_PATH_PARTS = {
    "baseline/tests",
    "data_platform/tests",
    "execution/tests",
}

TRAINING_NAME_TOKENS = (
    "forecast_training",
    "training_contract",
    "training_dataset",
    "training_runtime",
)

GUARD_NAME_TOKENS = (
    "active_manifest",
    "app_tasks_safety",
    "doc_guard",
    "runtime_isolation",
)


def _normalized_node_path(path: object) -> str:
    text = str(path).replace("\\", "/")
    marker = "/daily_research/"
    if marker in text:
        return text.split(marker, 1)[1].lower()
    return text.lower()


def _markexpr_requires_smoke(markexpr: str) -> bool:
    return re.search(r"(?<!not\s)\bsmoke\b", markexpr.lower()) is not None


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:
    node_path = _normalized_node_path(collection_path)
    markexpr = config.option.markexpr or ""
    if not _markexpr_requires_smoke(markexpr) or not node_path.endswith(".py"):
        return False
    return node_path not in SMOKE_FILES


def _existing_markers(item: pytest.Item) -> set[str]:
    return {marker.name for marker in item.iter_markers()}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        node_path = _normalized_node_path(getattr(item, "path", getattr(item, "fspath", "")))
        filename = Path(node_path).name
        stem = filename.removeprefix("test_").removesuffix(".py")
        markers: set[str] = set()

        if node_path in SMOKE_FILES:
            markers.add("smoke")
        if any(part in node_path for part in INTEGRATION_PATH_PARTS):
            markers.add("integration")
        if any(part in node_path for part in RESEARCH_PATH_PARTS) or any(token in stem for token in RESEARCH_NAME_TOKENS):
            markers.add("research")
        if any(part in node_path for part in DATA_HEAVY_PATH_PARTS) or any(token in stem for token in DATA_HEAVY_NAME_TOKENS):
            markers.add("data_heavy")
        if any(token in stem for token in EXTERNAL_NAME_TOKENS):
            markers.add("external")
        if any(token in stem for token in TRAINING_NAME_TOKENS):
            markers.update({"training", "slow"})
        if any(token in stem for token in GUARD_NAME_TOKENS):
            markers.add("guard")
        if "execution/tests" in node_path and node_path not in SMOKE_FILES:
            markers.add("slow")

        existing = _existing_markers(item)
        for marker_name in sorted(markers - existing):
            item.add_marker(getattr(pytest.mark, marker_name))
