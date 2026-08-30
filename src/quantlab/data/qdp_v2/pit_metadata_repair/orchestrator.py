"""Pit Metadata Repair: orchestrator responsibilities."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from quantlab.core.io import json_safe

from .config import (
    SUPPORTED_DOMAINS,
)
from .context import (
    _workspace,
)
from .identity import (
    _prepare_list_date,
)
from .industry import (
    _prepare_industry,
)
from .names import (
    _prepare_universe_name_pit,
)


def run_repair(
    *,
    workspace_root: str | Path | None = None,
    domains: Sequence[str] = SUPPORTED_DOMAINS,
    apply: bool = False,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    selected = tuple(str(item) for item in domains)
    unknown = sorted(set(selected).difference(SUPPORTED_DOMAINS))
    if unknown:
        raise ValueError(f"unknown_pit_metadata_domains:{','.join(unknown)}")
    results: dict[str, Any] = {}
    if "universe-list-date" in selected:
        results["universe-list-date"] = _prepare_list_date(workspace, apply=apply)
    if "universe-name-pit" in selected:
        results["universe-name-pit"] = _prepare_universe_name_pit(workspace, apply=apply)
    if "industry-unknown" in selected:
        results["industry-unknown"] = _prepare_industry(workspace, apply=apply)
    return json_safe({"status": "applied" if apply else "planned", "domains": results})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qdp pit-metadata-repair")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--domains", default=",".join(SUPPORTED_DOMAINS))
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.plan) == bool(args.apply):
        parser.error("exactly one of --plan or --apply is required")
    domains = tuple(item.strip() for item in str(args.domains).split(",") if item.strip())
    payload = run_repair(
        workspace_root=str(args.workspace_root or "") or None,
        domains=domains,
        apply=bool(args.apply),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0
