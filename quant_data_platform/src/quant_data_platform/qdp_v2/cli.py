from __future__ import annotations

import importlib
from typing import Callable

from quant_data_platform.qdp_v2.environment import assert_yolos_environment


CommandMain = Callable[[list[str] | None], int]

HELP_TEXT = """usage: qdp [--workspace-root WORKSPACE_ROOT] <command> [options]

QDP local data base CLI.

commands:
  status                  Show active data base scope and table summary.
  list                    List active tables.
  describe <table>        Describe an active table or dataset id.
  check --quick           Fast manifest and contract check.
  check --full            Full data-quality audit.
  rebuild limit-intraday  Rebuild 1m-derived limit-board features.
  gc --dry-run            Show unreferenced data directories.
  update                  Update the active data base. Dry-run by default-friendly options are supported.

The active data base is manifest-first: parquet + dataset.json + active.json.
DuckDB indexes, memmaps, and archived repair tools are not part of active state.
"""


COMMAND_MODULES: dict[tuple[str, ...], str] = {
    ("status",): "quant_data_platform.qdp_v2.status",
    ("list",): "quant_data_platform.qdp_v2.dataset",
    ("describe",): "quant_data_platform.qdp_v2.dataset",
    ("check",): "quant_data_platform.qdp_v2.check",
    ("gc",): "quant_data_platform.qdp_v2.gc",
    ("update",): "quant_data_platform.qdp_v2.update",
    ("rebuild", "limit-intraday"): "quant_data_platform.qdp_v2.limit_intraday_features",
}

ENV_GUARDED_PREFIXES = {
    ("check",),
    ("rebuild", "limit-intraday"),
}

ARG_ALIASES: dict[tuple[str, ...], list[str]] = {
    ("list",): ["list"],
    ("describe",): ["describe"],
}


def dispatch(argv: list[str]) -> int | None:
    raw = list(argv or [])
    if not raw or raw in (["-h"], ["--help"]):
        print(HELP_TEXT)
        return 0
    for prefix, module_name in sorted(COMMAND_MODULES.items(), key=lambda item: len(item[0]), reverse=True):
        if tuple(raw[: len(prefix)]) == prefix:
            if _requires_yolos(prefix, raw[len(prefix) :]):
                assert_yolos_environment(command=" ".join(prefix))
            module = importlib.import_module(module_name)
            main = getattr(module, "main", None)
            if not callable(main):
                raise RuntimeError(f"{module_name} does not expose callable main(argv)")
            forwarded = list(ARG_ALIASES.get(prefix, [])) + raw[len(prefix) :]
            return int(main(forwarded) or 0)
    return None


def _requires_yolos(prefix: tuple[str, ...], args: list[str]) -> bool:
    if prefix in ENV_GUARDED_PREFIXES:
        return True
    if prefix == ("update",):
        return "--dry-run" not in args
    if prefix == ("gc",):
        return "--delete" in args
    return False


def main(argv: list[str] | None = None) -> int:
    result = dispatch(list(argv or []))
    if result is None:
        raise ValueError("unsupported_qdp_v2_command")
    return result
