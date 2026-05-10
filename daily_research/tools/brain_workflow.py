from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.tools.brain_platform import (
    build_workflow_state,
    build_writeback_plan,
    check_brain_health,
    print_json,
    resolve_bootstrap,
    write_workflow_output,
)


def _maybe_write(kind: str, payload: dict[str, Any], enabled: bool) -> dict[str, Any]:
    if enabled:
        payload = dict(payload)
        payload["output_path"] = write_workflow_output(kind, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Machine-readable brain/workflow state helper.")
    sub = parser.add_subparsers(dest="command", required=True)

    handoff = sub.add_parser("handoff", help="Resolve a main/child brain handoff capsule.")
    handoff.add_argument("--child", default="daily_research")
    handoff.add_argument("--json", action="store_true")
    handoff.add_argument("--write-output", action="store_true")

    health = sub.add_parser("health", help="Run read-only brain and environment health checks.")
    health.add_argument("--json", action="store_true")
    health.add_argument("--write-output", action="store_true")

    status = sub.add_parser("status", help="Build workflow status from registry and latest artifacts.")
    status.add_argument("--workflow", required=True)
    status.add_argument("--study-tag", default="")
    status.add_argument("--json", action="store_true")
    status.add_argument("--write-output", action="store_true")

    preflight = sub.add_parser("preflight", help="Build read-only preflight state for a workflow.")
    preflight.add_argument("--workflow", required=True)
    preflight.add_argument("--study-tag", default="")
    preflight.add_argument("--json", action="store_true")
    preflight.add_argument("--write-output", action="store_true")

    writeback = sub.add_parser("writeback-plan", help="Generate a routed brain writeback plan.")
    writeback.add_argument("--source", default="latest")
    writeback.add_argument("--json", action="store_true")
    writeback.add_argument("--write-output", action="store_true")
    writeback.add_argument("--apply-brain-writeback", action="store_true")
    return parser


def build_payload(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    if args.command == "handoff":
        return "handoff", resolve_bootstrap(args.child).to_dict()
    if args.command == "health":
        return "health", check_brain_health().to_dict()
    if args.command in {"status", "preflight"}:
        payload = build_workflow_state(args.workflow, study_tag=str(getattr(args, "study_tag", "") or "")).to_dict()
        if args.command == "preflight":
            payload = dict(payload)
            payload["preflight_only"] = True
        return args.command, payload
    if args.command == "writeback-plan":
        return "writeback_plan", build_writeback_plan(args.source, apply_brain_writeback=args.apply_brain_writeback)
    raise ValueError(f"Unsupported command: {args.command}")


def main() -> int:
    args = build_parser().parse_args()
    kind, payload = build_payload(args)
    payload = _maybe_write(kind, payload, bool(getattr(args, "write_output", False)))
    print_json(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
