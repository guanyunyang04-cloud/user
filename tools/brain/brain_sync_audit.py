from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


HOT_PATHS = (
    "brain/identity_layer.md",
    "brain/state_center.md",
    "brain/knowledge_center.md",
    "brain/master_brain.md",
    "brain/operations_center.md",
    "brain/governance_layer.md",
    "brain/skills/workspace-brain/SKILL.md",
    "quant_data_platform/brain/identity_layer.md",
    "quant_data_platform/brain/state_center.md",
    "quant_data_platform/brain/knowledge_center.md",
    "quant_data_platform/brain/operations_center.md",
    "quant_data_platform/brain/governance_layer.md",
    "quant_data_platform/brain/brain_architecture.md",
    "daily_research/brain/state_center.md",
    "daily_research/brain/knowledge_center.md",
    "daily_research/brain/operations_center.md",
    "daily_research/brain/governance_layer.md",
)

STALE_PATTERNS = {
    "canonical_data_v1": "QDP v2 active data base is manifest-first, not canonical_data_v1.",
    "policy bundle": "Policy bundles are archived v1 workflow terms in hot paths.",
    "policy_bundle": "Policy bundles are archived v1 workflow terms in hot paths.",
    "memmap registry": "Memmap registry is not QDP active data-base state.",
    "active_sharded_memmap": "Active sharded memmap is a downstream artifact, not QDP active data-base state.",
    "ResearchDataLake": "Catalog-first lake is archived in current QDP hot paths.",
    "build-sharded-memmap": "Old memmap build command is archived.",
    "validate-memmap": "Old memmap validation command is archived.",
    "provider-eval": "Old provider-eval command is archived from public QDP CLI.",
    "qdp clean": "Use qdp rebuild for long-lived rebuildable tables.",
    "migrate-v2": "Migration is not a current public QDP workflow.",
    "activate-v2": "Activation is internal; public active state is managed by QDP workflows.",
    "brain-burden-audit": "Removed command; use brain-structure-audit.",
}

ALLOWED_BY_PATH = {
    "brain/skills/workspace-brain/SKILL.md": {"memmap registry", "active_sharded_memmap"},
}


def audit(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(workspace_root or ".").resolve()
    findings: list[dict[str, Any]] = []
    scanned: list[str] = []
    for rel in HOT_PATHS:
        path = root / rel
        if not path.exists():
            findings.append({"severity": "warning", "path": rel, "pattern": "missing_hot_path", "message": "Hot-path brain file is missing."})
            continue
        scanned.append(rel)
        text = path.read_text(encoding="utf-8", errors="replace")
        allowed = ALLOWED_BY_PATH.get(rel, set())
        for pattern, message in STALE_PATTERNS.items():
            if pattern in allowed:
                continue
            if pattern.lower() in text.lower():
                findings.append({"severity": "error", "path": rel, "pattern": pattern, "message": message})
    errors = [item for item in findings if item.get("severity") == "error"]
    warnings = [item for item in findings if item.get("severity") == "warning"]
    return {
        "status": "ok" if not errors else "needs_sync",
        "workspace_root": str(root),
        "scanned_count": len(scanned),
        "finding_count": len(findings),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "findings": findings,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit brain hot paths for stale project-truth terms.")
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = audit(workspace_root=args.workspace_root)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"status: {payload['status']}")
        print(f"scanned_count: {payload['scanned_count']}")
        print(f"finding_count: {payload['finding_count']}")
        for item in payload["findings"]:
            print(f"{item['severity']}: {item['path']}: {item['pattern']} - {item['message']}")
    return 0 if payload["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
