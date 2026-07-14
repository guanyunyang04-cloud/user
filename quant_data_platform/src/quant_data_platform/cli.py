from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from typing import Any

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.progress import suppress_progress


ARCHIVE_NOTICE = (
    "This command belongs to the archived v1 lake/ingest/memmap/canonical workflow. "
    "Use the qdp_v2 commands shown by `qdp --help`; archival notes live in "
    "tools/archive_v1/README.md."
)

HELP_TEXT = """usage: qdp [--workspace-root ROOT] [--generation v2|v3] <command> [options]

QDP manifest-first data-base CLI.

v3 commands:
  ingest                         Fetch immutable provider partitions.
  build candidate               Build canonical datasets and a candidate.
  audit --candidate ID          Run quick/full/semantic release gates.
  diff --candidate ID           Compare a candidate with active.
  publish --candidate ID        Compare-and-swap publish an audited candidate.
  rollback                      Restore the latest rollback manifest.
  retire-v2-intraday            Delete the replaced v2 minute shards after v3 verification.
  freeze v2                     Hash and pin the legacy evidence base.
  update                         Run the v3 ingest/build/audit/publish DAG.
  gc                             Run transitive manifest-aware v3 GC.

read commands:
  status | list | describe | check
  These read v2 until v3 has an active manifest. Use --generation explicitly
  to select either generation.

Legacy v2 rebuild/verify commands remain available with --generation v2.
Training packs and memmaps belong to daily_research and are not QDP commands.
Use --json for stable machine-readable output.
"""

ARCHIVED_COMMANDS = {
    "init-registry",
    "audit",
    "build-bundle",
    "validate-memmap",
    "audit-sharded-feature-coverage",
    "build-sharded-memmap",
    "freeze-sharded-memmap",
    "plan-incremental-memmap",
    "compose-sharded-memmap",
    "build-training-pack",
    "build-regime-training-pack",
    "build-event-pack",
    "provider-eval",
    "provider-health",
    "refresh-daily",
    "daily-update",
    "import-csv",
    "baostock-backfill",
    "build-intraday-daily-features",
    "combine-domain-datasets",
    "combine-sharded-domain-datasets",
    "import-external-quant-zip",
    "normalize-intraday-1m-contract",
    "recover-external-quant-zip-import",
    "import-tdx-5m",
    "import-tdx-daily",
    "audit-gold-dataset",
    "build-canonical-policy-bundle",
    "build-gold-training-dataset",
    "build-pool-view",
    "build-research-database",
    "build-sector-board-view",
    "build-v2-status-sidecar",
    "canonical-audit",
    "lake-gc",
    "import-legacy-training-caches",
    "import-traditional-baostock-v2-snapshot",
    "import-traditional-pit-status-sidecar",
    "policy-input-audit",
    "v2-dataset-contract-audit",
    "cleanup",
}

ARCHIVED_ALIASES = {
    ("lake", "gc"): "gc",
    ("dataset", "list"): "list",
    ("dataset", "describe"): "describe",
    ("dataset", "validate"): "check --quick",
    ("audit", "active"): "check --quick",
    ("audit", "database"): "check --full",
    ("audit", "pk-deep"): "check --full",
    ("clean", "5m-from-1m"): "rebuild 5m",
    ("clean", "daily-market"): "rebuild daily-panel",
    ("clean", "valuation"): "rebuild valuation",
    ("derive", "limit-intraday"): "rebuild limit-intraday",
    ("index", "rebuild"): "archived",
    ("rebuild", "training-pack"): "archived",
}

V3_ONLY_PREFIXES = {
    ("ingest",),
    ("build", "candidate"),
    ("audit",),
    ("diff",),
    ("publish",),
    ("rollback",),
    ("retire-v2-intraday",),
    ("freeze", "v2"),
    ("compatibility",),
}

READ_COMMANDS = {"status", "list", "describe", "check"}


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in raw_argv
    if as_json:
        reconfigure = getattr(sys.stdout, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except (OSError, ValueError):
                pass
    try:
        with suppress_progress() if as_json else nullcontext():
            return _dispatch(raw_argv)
    except SystemExit:
        raise
    except Exception as exc:
        payload = {
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        if as_json:
            print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
        else:
            print(f"status: error\nerror_type: {payload['error_type']}\nmessage: {payload['message']}", file=sys.stderr)
        return 2


def _dispatch(raw_argv: list[str]) -> int:
    workspace = _extract_workspace_root(raw_argv)
    generation = _extract_generation(raw_argv)
    positional = _strip_global_options(raw_argv)
    if not positional or positional in (["-h"], ["--help"]):
        print(HELP_TEXT)
        return 0

    prefix = tuple(positional[:2])
    first = positional[0]
    if any(prefix[: len(candidate)] == candidate for candidate in V3_ONLY_PREFIXES):
        if generation == "v2":
            raise ValueError(f"command_requires_generation_v3:{' '.join(prefix)}")
        result = _run_qdp_v3(positional, workspace)
        if result is not None:
            return int(result)

    if first in {"update", "gc"} and generation != "v2":
        result = _run_qdp_v3(positional, workspace)
        if result is not None:
            return int(result)

    if first in READ_COMMANDS:
        selected = generation or _default_read_generation(workspace)
        runner = _run_qdp_v3 if selected == "v3" else _run_qdp_v2
        result = runner(positional, workspace)
        if result is not None:
            return int(result)

    if generation == "v3":
        result = _run_qdp_v3(positional, workspace)
        if result is not None:
            return int(result)
    else:
        result = _run_qdp_v2(positional, workspace)
        if result is not None:
            return int(result)

    if first == "refresh-daily" and any(item in {"-h", "--help"} for item in positional[1:]):
        # Keep the historical script-help contract discoverable while actual
        # v3 updates move to `qdp update`.
        from quant_data_platform.ingest.refresh_daily import main as refresh_daily_main

        return int(refresh_daily_main(positional[1:]) or 0)

    archived = _archived_command(raw_argv)
    if archived:
        _print_archived(archived, as_json="--json" in raw_argv)
        return 0 if any(item in {"-h", "--help"} for item in raw_argv) else 2
    _print_archived(" ".join(positional) or "unknown", as_json="--json" in raw_argv)
    return 2


def _run_qdp_v2(positional: list[str], workspace: str) -> int | None:
    first = positional[0] if positional else "--help"
    if first in {"-h", "--help", "status", "list", "describe", "check", "rebuild", "verify", "gc", "update"}:
        from quant_data_platform.qdp_v2.cli import dispatch

        result = dispatch(_with_workspace(positional or ["--help"], workspace))
        if result is not None:
            return result
    for prefix, replacement in ARCHIVED_ALIASES.items():
        if tuple(positional[: len(prefix)]) == prefix and replacement not in {"archived"}:
            from quant_data_platform.qdp_v2.cli import dispatch

            return dispatch(_with_workspace(replacement.split() + positional[len(prefix) :], workspace))
    return None


def _maybe_run_qdp_v2(raw_argv: list[str]) -> int | None:
    """Backward-compatible internal hook for legacy v2 callers/tests.

    Public routing now selects generations explicitly, but downstream code
    that imported this helper still receives the original v2-only behavior.
    """

    workspace = _extract_workspace_root(raw_argv)
    positional = _strip_global_options(raw_argv) or ["--help"]
    return _run_qdp_v2(positional, workspace)


def _run_qdp_v3(positional: list[str], workspace: str) -> int | None:
    from quant_data_platform.qdp_v3.cli import dispatch

    return dispatch(_with_workspace(positional, workspace))


def _default_read_generation(workspace: str) -> str:
    from quant_data_platform.qdp_v3.paths import qdp_v3_paths

    return "v3" if qdp_v3_paths(workspace or None).active_manifest.exists() else "v2"


def _archived_command(raw_argv: list[str]) -> str:
    positional = _strip_global_options(raw_argv)
    if not positional:
        return ""
    for prefix in sorted(ARCHIVED_ALIASES, key=len, reverse=True):
        if tuple(positional[: len(prefix)]) == prefix:
            return " ".join(positional[: len(prefix)])
    if positional[0] in ARCHIVED_COMMANDS:
        return positional[0]
    return ""


def _print_archived(command: str, *, as_json: bool) -> None:
    payload: dict[str, Any] = {
        "status": "archived",
        "command": command,
        "message": ARCHIVE_NOTICE,
        "current_commands": ["status", "list", "describe", "check", "rebuild", "verify", "gc", "update"],
        "archive_notes": "tools/archive_v1/README.md",
    }
    if as_json:
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
        return
    print(f"status: {payload['status']}")
    print(f"command: {payload['command']}")
    print(f"message: {payload['message']}")
    print("current_commands: status, list, describe, check, rebuild, verify, gc, update")


def _extract_workspace_root(raw_argv: list[str]) -> str:
    for index, item in enumerate(raw_argv):
        if item == "--workspace-root" and index + 1 < len(raw_argv):
            return str(raw_argv[index + 1])
        if item.startswith("--workspace-root="):
            return item.split("=", 1)[1]
    return ""


def _extract_generation(raw_argv: list[str]) -> str:
    for index, item in enumerate(raw_argv):
        if item == "--generation" and index + 1 < len(raw_argv):
            value = str(raw_argv[index + 1]).lower()
            break
        if item.startswith("--generation="):
            value = item.split("=", 1)[1].lower()
            break
    else:
        return ""
    if value not in {"v2", "v3"}:
        raise ValueError(f"unsupported_generation:{value}")
    return value


def _with_workspace(argv: list[str], workspace: str) -> list[str]:
    if not workspace or argv[:1] in (["-h"], ["--help"]):
        return argv
    return list(argv) + ["--workspace-root", workspace]


def _strip_global_options(raw_argv: list[str]) -> list[str]:
    cleaned: list[str] = []
    skip_next = False
    for item in raw_argv:
        if skip_next:
            skip_next = False
            continue
        if item in {"--workspace-root", "--generation"}:
            skip_next = True
            continue
        if item.startswith("--workspace-root=") or item.startswith("--generation="):
            continue
        cleaned.append(item)
    return cleaned


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
