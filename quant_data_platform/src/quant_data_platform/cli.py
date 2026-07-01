from __future__ import annotations

import json
import sys
from typing import Any

from quant_data_platform.core.json_io import json_safe


ARCHIVE_NOTICE = (
    "This command belongs to the archived v1 lake/ingest/memmap/canonical workflow. "
    "Use the qdp_v2 commands shown by `qdp --help`; archival notes live in "
    "tools/archive_v1/README.md."
)

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
}


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    v2_result = _maybe_run_qdp_v2(raw_argv)
    if v2_result is not None:
        return int(v2_result)
    archived = _archived_command(raw_argv)
    if archived:
        _print_archived(archived, as_json="--json" in raw_argv)
        return 2
    _print_archived(" ".join(_strip_global_options(raw_argv)) or "unknown", as_json="--json" in raw_argv)
    return 2


def _maybe_run_qdp_v2(raw_argv: list[str]) -> int | None:
    workspace = _extract_workspace_root(raw_argv)
    positional = _strip_global_options(raw_argv)
    if not positional:
        positional = ["--help"]
    first = positional[0]
    if first in {"-h", "--help", "status", "list", "describe", "check", "rebuild", "gc", "update"}:
        from quant_data_platform.qdp_v2.cli import dispatch

        return dispatch(_with_workspace(positional, workspace))
    for prefix, replacement in ARCHIVED_ALIASES.items():
        if tuple(positional[: len(prefix)]) == prefix and replacement not in {"archived"}:
            from quant_data_platform.qdp_v2.cli import dispatch

            return dispatch(_with_workspace(replacement.split() + positional[len(prefix) :], workspace))
    return None


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
        "current_commands": ["status", "list", "describe", "check", "rebuild", "gc", "update"],
        "archive_notes": "tools/archive_v1/README.md",
    }
    if as_json:
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
        return
    print(f"status: {payload['status']}")
    print(f"command: {payload['command']}")
    print(f"message: {payload['message']}")
    print("current_commands: status, list, describe, check, rebuild, gc, update")


def _extract_workspace_root(raw_argv: list[str]) -> str:
    for index, item in enumerate(raw_argv):
        if item == "--workspace-root" and index + 1 < len(raw_argv):
            return str(raw_argv[index + 1])
        if item.startswith("--workspace-root="):
            return item.split("=", 1)[1]
    return ""


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
        if item == "--workspace-root":
            skip_next = True
            continue
        if item.startswith("--workspace-root="):
            continue
        cleaned.append(item)
    return cleaned


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
