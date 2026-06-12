from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "quant_data_platform" / "src"

SMOKE_FILES = {
    "test_contracts_and_profiles.py",
    "test_index_sidecar_loader.py",
    "test_registry.py",
}

DATA_HEAVY_FILES = {
    "test_sharded_memmap.py",
}

for path in (SRC, ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        filename = Path(str(getattr(item, "path", getattr(item, "fspath", "")))).name
        if filename in SMOKE_FILES:
            item.add_marker(pytest.mark.smoke)
        if filename in DATA_HEAVY_FILES:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.data_heavy)
