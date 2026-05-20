from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.tools.brain_platform import (
    audit_brain_system,
    build_workflow_guide,
    build_workflow_state,
    build_writeback_plan,
    check_brain_health,
    print_json,
    resolve_bootstrap,
    select_workflow_for_task,
    write_workflow_output,
)
from daily_research.tools.brain_capsule import build_task_capsule
from daily_research.tools.brain_evidence_registry import (
    build_evidence_registry,
    query_evidence_registry,
    write_evidence_registry,
)
from daily_research.tools.frontier_scanner import build_frontier_report
from daily_research.tools.selective_verification import build_verification_plan


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
    preflight.add_argument("--task", default="")
    preflight.add_argument("--json", action="store_true")
    preflight.add_argument("--write-output", action="store_true")

    capsule = sub.add_parser("capsule", help="Build a task-scoped daily_research brain capsule.")
    capsule.add_argument("--child", default="daily_research")
    capsule.add_argument("--task", default="")
    capsule.add_argument("--workflow", default="brain_handoff")
    capsule.add_argument("--study-tag", default="")
    capsule.add_argument("--json", action="store_true")
    capsule.add_argument("--write-output", action="store_true")

    frontier = sub.add_parser("current-frontier", help="Scan latest output studies against brain references and registry.")
    frontier.add_argument("--json", action="store_true")
    frontier.add_argument("--write-output", action="store_true")

    evidence = sub.add_parser("evidence-index", help="Build or read the brain evidence registry.")
    evidence.add_argument("--rebuild", action="store_true")
    evidence.add_argument("--json", action="store_true")
    evidence.add_argument("--write-output", action="store_true")

    query = sub.add_parser("query", help="Query the brain evidence registry without inferring new conclusions.")
    query.add_argument("--q", required=True)
    query.add_argument("--json", action="store_true")
    query.add_argument("--write-output", action="store_true")

    guide = sub.add_parser("workflow-guide", help="Return a workflow checklist and guard guide.")
    guide.add_argument("--workflow", required=True)
    guide.add_argument("--json", action="store_true")
    guide.add_argument("--write-output", action="store_true")

    selector = sub.add_parser("select-workflow", help="Select a brain workflow from task intent.")
    selector.add_argument("--task", required=True)
    selector.add_argument("--json", action="store_true")
    selector.add_argument("--write-output", action="store_true")

    audit = sub.add_parser("audit-brain", help="Audit all workspace brain roots, language policy, workflows, and guards.")
    audit.add_argument("--scope", default="all", choices=("all", "attached"))
    audit.add_argument("--json", action="store_true")
    audit.add_argument("--write-output", action="store_true")

    writeback = sub.add_parser("writeback-plan", help="Generate a routed brain writeback plan.")
    writeback.add_argument("--source", default="latest")
    writeback.add_argument("--json", action="store_true")
    writeback.add_argument("--write-output", action="store_true")
    writeback.add_argument("--apply-brain-writeback", action="store_true")

    verify = sub.add_parser("verify-plan", help="Recommend change-aware verification commands without running them.")
    verify.add_argument("--base", default="")
    verify.add_argument("--paths", nargs="*", default=None)
    verify.add_argument("--json", action="store_true")
    verify.add_argument("--write-output", action="store_true")
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
            payload["task"] = str(getattr(args, "task", "") or "")
        return args.command, payload
    if args.command == "capsule":
        return "capsule", build_task_capsule(
            child=str(getattr(args, "child", "") or "daily_research"),
            task=str(getattr(args, "task", "") or ""),
            workflow=str(getattr(args, "workflow", "") or "brain_handoff"),
            study_tag=str(getattr(args, "study_tag", "") or ""),
        )
    if args.command == "current-frontier":
        return "current_frontier", build_frontier_report()
    if args.command == "evidence-index":
        if bool(getattr(args, "rebuild", False)):
            output_path = write_evidence_registry()
            payload = build_evidence_registry()
            payload["registry_path"] = output_path.as_posix()
            return "evidence_index", payload
        return "evidence_index", build_evidence_registry()
    if args.command == "query":
        return "query", query_evidence_registry(str(getattr(args, "q", "") or ""))
    if args.command == "workflow-guide":
        return "workflow_guide", build_workflow_guide(str(getattr(args, "workflow", "") or ""))
    if args.command == "select-workflow":
        return "select_workflow", select_workflow_for_task(str(getattr(args, "task", "") or ""))
    if args.command == "audit-brain":
        return "audit_brain", audit_brain_system(scope=str(getattr(args, "scope", "") or "all"))
    if args.command == "writeback-plan":
        return "writeback_plan", build_writeback_plan(args.source, apply_brain_writeback=args.apply_brain_writeback)
    if args.command == "verify-plan":
        return "verify_plan", build_verification_plan(
            paths=list(args.paths) if getattr(args, "paths", None) is not None else None,
            base=str(getattr(args, "base", "") or "") or None,
        )
    raise ValueError(f"Unsupported command: {args.command}")


def main() -> int:
    args = build_parser().parse_args()
    kind, payload = build_payload(args)
    payload = _maybe_write(kind, payload, bool(getattr(args, "write_output", False)))
    print_json(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
