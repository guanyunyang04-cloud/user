from __future__ import annotations

import importlib
from collections.abc import Callable

from quantlab.data.qdp_v2.environment import assert_yolos_environment

CommandMain = Callable[[list[str] | None], int]

HELP_TEXT = """usage: qdp [--workspace-root WORKSPACE_ROOT] <command> [options]

QDP active point-in-time research-data store.

commands:
  status                  Show table coverage and row counts.
  list                    List current tables.
  describe <table>        Describe a current table.
  check --quick|--full|--semantic
                          Validate physical or selected semantic contracts.
  update                  Add recent market data in place.
  iquant-parity           Compare a small local iQuant cache sample with QDP.
  minute-parity           Compare a selected minute ZIP extract with QDP/iQuant.
  compact                 Merge the 5-minute table into yearly files.
  repair                  Repair active shards with explicit CAS protection.
  gc                      Remove unreferenced files.

There is one active table per domain. Updates retain point-in-time lifecycle facts;
future ST or delisting state never removes earlier observations.
"""


COMMAND_MODULES: dict[tuple[str, ...], str] = {
    ("status",): "quantlab.data.qdp_v2.status",
    ("list",): "quantlab.data.qdp_v2.dataset",
    ("describe",): "quantlab.data.qdp_v2.dataset",
    ("check",): "quantlab.data.qdp_v2.check",
    ("gc",): "quantlab.data.qdp_v2.gc",
    ("update",): "quantlab.data.qdp_v2.update",
    ("iquant-parity",): "quantlab.data.iquant_parity",
    ("minute-parity",): "quantlab.data.minute_parity",
    ("compact",): "quantlab.data.qdp_v2.compact",
    ("repair",): "quantlab.data.qdp_v2.repair",
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
    for prefix, module_name in COMMAND_MODULES.items():
        if tuple(raw[: len(prefix)]) != prefix:
            continue
        if _requires_yolos(prefix, raw[len(prefix) :]):
            assert_yolos_environment(command=" ".join(prefix))
        module = importlib.import_module(module_name)
        main = getattr(module, "main", None)
        if not callable(main):
            raise TypeError(f"{module_name} does not expose callable main(argv)")
        global_args, tail = _split_workspace_option(raw[len(prefix) :])
        forwarded = global_args + list(ARG_ALIASES.get(prefix, [])) + tail
        return int(main(forwarded) or 0)
    return None


def _split_workspace_option(args: list[str]) -> tuple[list[str], list[str]]:
    global_args: list[str] = []
    tail: list[str] = []
    index = 0
    while index < len(args):
        item = args[index]
        if item == "--workspace-root":
            if index + 1 >= len(args):
                raise ValueError("missing_value:--workspace-root")
            global_args = [item, args[index + 1]]
            index += 2
            continue
        if item.startswith("--workspace-root="):
            global_args = [item]
        else:
            tail.append(item)
        index += 1
    return global_args, tail


def _requires_yolos(prefix: tuple[str, ...], args: list[str]) -> bool:
    if prefix == ("repair",):
        return True
    if prefix == ("check",):
        return True
    if prefix == ("update",):
        return "--dry-run" not in args
    if prefix == ("compact",):
        return "--dry-run" not in args
    if prefix == ("gc",):
        return "--delete" in args
    return False


def main(argv: list[str] | None = None) -> int:
    result = dispatch(list(argv or []))
    if result is None:
        raise ValueError("unsupported_qdp_command")
    return result
