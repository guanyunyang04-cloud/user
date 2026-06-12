from __future__ import annotations

from pathlib import Path
import re

import pytest


_BENCHMARK_FILES = {
    "test_search_performance.py",
}

_SMOKE_FILES = {
    "test_analysis_metadata.py",
    "test_config_manager.py",
    "test_config_registry.py",
    "test_stock_code_utils.py",
}

_INTEGRATION_NODE_TOKENS = (
    "TestSelectionSourceValidationIntegration",
)

_TOKEN_MARKERS = {
    "provider": (
        "akshare",
        "feishu",
        "hk_realtime",
        "searxng",
        "social_sentiment",
        "stooq",
        "tavily",
        "tickflow",
        "tushare",
        "yfinance",
    ),
    "llm": (
        "agent",
        "analyzer_news_prompt",
        "image_stock_extractor",
        "litellm",
        "llm",
        "multi_agent",
    ),
}


def _markexpr_requires_smoke(markexpr: str) -> bool:
    return re.search(r"(?<!not\s)\bsmoke\b", markexpr.lower()) is not None


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:
    filename = collection_path.name.lower()
    markexpr = config.option.markexpr or ""
    if not _markexpr_requires_smoke(markexpr) or not filename.startswith("test_"):
        return False
    return filename not in _SMOKE_FILES


def _item_filename(item: pytest.Item) -> str:
    item_path = getattr(item, "path", None)
    if item_path is None:
        item_path = getattr(item, "fspath", "")
    return Path(str(item_path)).name.lower()


def _existing_markers(item: pytest.Item) -> set[str]:
    return {marker.name for marker in item.iter_markers()}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        filename = _item_filename(item)
        nodeid = item.nodeid
        markers: set[str] = set()

        if any(token in nodeid for token in _INTEGRATION_NODE_TOKENS):
            markers.add("integration")
        elif filename in _SMOKE_FILES:
            markers.add("smoke")
        if filename in _BENCHMARK_FILES:
            markers.update({"benchmark", "slow"})

        for marker_name, tokens in _TOKEN_MARKERS.items():
            if any(token in filename for token in tokens):
                markers.add(marker_name)

        existing = _existing_markers(item)
        for marker_name in sorted(markers - existing):
            item.add_marker(getattr(pytest.mark, marker_name))
