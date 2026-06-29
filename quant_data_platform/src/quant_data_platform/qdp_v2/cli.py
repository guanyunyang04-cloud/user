from __future__ import annotations

import importlib
from typing import Callable


CommandMain = Callable[[list[str] | None], int]


COMMAND_MODULES: dict[tuple[str, ...], str] = {
    ("migrate-v2",): "quant_data_platform.qdp_v2.migration",
    ("activate-v2",): "quant_data_platform.qdp_v2.activate",
    ("status",): "quant_data_platform.qdp_v2.status",
    ("dataset",): "quant_data_platform.qdp_v2.dataset",
    ("audit", "active"): "quant_data_platform.qdp_v2.audit",
    ("index",): "quant_data_platform.qdp_v2.index",
    ("lake", "gc"): "quant_data_platform.qdp_v2.gc",
    ("update",): "quant_data_platform.qdp_v2.update",
    ("clean",): "quant_data_platform.qdp_v2.cleaning",
    ("provider", "benchmark"): "quant_data_platform.qdp_v2.provider_benchmark",
}


def dispatch(argv: list[str]) -> int | None:
    raw = list(argv or [])
    for prefix, module_name in sorted(COMMAND_MODULES.items(), key=lambda item: len(item[0]), reverse=True):
        if tuple(raw[: len(prefix)]) == prefix:
            module = importlib.import_module(module_name)
            main = getattr(module, "main", None)
            if not callable(main):
                raise RuntimeError(f"{module_name} does not expose callable main(argv)")
            return int(main(raw[len(prefix) :]) or 0)
    return None


def main(argv: list[str] | None = None) -> int:
    result = dispatch(list(argv or []))
    if result is None:
        raise ValueError("unsupported_qdp_v2_command")
    return result
