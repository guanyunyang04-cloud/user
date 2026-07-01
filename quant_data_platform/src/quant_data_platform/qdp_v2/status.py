from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.qdp_v2.manifest import (
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
)


def v2_active_exists(workspace_root: str | Path | None = None) -> bool:
    root = qdp_v2_root(workspace_root)
    return (root / "active" / "active.json").exists()


def status_payload(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    root = qdp_v2_root(workspace_root)
    active = read_active_manifest(root)
    if not active:
        return {
            "status": "not_initialized",
            "qdp_v2_root": str(root.resolve()),
            "message": "qdp_v2 active manifest not found.",
        }
    datasets: dict[str, Any] = {}
    missing: list[dict[str, str]] = []
    for _, domain, dataset_id in _active_dataset_refs(active):
        manifest_path = dataset_manifest_for_id(root, dataset_id, domain)
        if manifest_path is None:
            missing.append({"domain": domain, "dataset_id": dataset_id})
            continue
        manifest = read_dataset_manifest(manifest_path)
        shard_count = len(manifest.shards)
        existing_shards = 0
        for shard in manifest.shards:
            if resolve_manifest_path(shard.path, root=root).exists():
                existing_shards += 1
        datasets[domain] = {
            "dataset_id": manifest.dataset_id,
            "domain": manifest.domain,
            "layer": manifest.layer,
            "frequency": manifest.frequency,
            "contract_version": manifest.contract_version,
            "start_date": manifest.start_date,
            "end_date": manifest.end_date,
            "row_count": manifest.row_count,
            "shard_count": shard_count,
            "existing_shards": existing_shards,
        }
    return {
        "status": "ok" if not missing else "missing_dataset_manifest",
        "qdp_v2_root": str(root.resolve()),
        "active_as_of_date": str(active.get("active_as_of_date", "") or ""),
        "scope": dict(active.get("scope", {}) or {}),
        "active_manifest": str((root / "active" / "active.json").resolve()),
        "dataset_count": len(datasets),
        "missing": missing,
        "datasets": datasets,
    }


def print_status(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
        return
    print(f"status: {payload.get('status', '')}")
    print(f"qdp_v2_root: {payload.get('qdp_v2_root', '')}")
    if payload.get("active_as_of_date"):
        print(f"active_as_of_date: {payload.get('active_as_of_date')}")
    print(f"dataset_count: {payload.get('dataset_count', 0)}")
    for key, item in dict(payload.get("datasets", {}) or {}).items():
        print(
            f"{key}: {item.get('dataset_id')} "
            f"{item.get('start_date', '')}..{item.get('end_date', '')} "
            f"rows={item.get('row_count', 0)} shards={item.get('existing_shards', 0)}/{item.get('shard_count', 0)}"
        )
    missing = list(payload.get("missing", []) or [])
    if missing:
        print(f"missing: {json.dumps(json_safe(missing), ensure_ascii=False)}")


def _active_dataset_refs(active: dict[str, Any]) -> list[tuple[str, str, str]]:
    mapping = active_dataset_map(active)
    return [("datasets", str(domain), str(dataset_id)) for domain, dataset_id in sorted(mapping.items()) if str(dataset_id or "").strip()]


def active_dataset_map(active: dict[str, Any] | Any) -> dict[str, str]:
    payload = dict(active or {})
    if isinstance(payload.get("datasets"), dict):
        return {str(domain): str(dataset_id) for domain, dataset_id in dict(payload.get("datasets", {}) or {}).items() if str(dataset_id or "").strip()}
    refs: list[tuple[str, str, str]] = []
    for section in ("raw", "derived", "research_panels"):
        mapping = dict(payload.get(section, {}) or {})
        for domain, dataset_id in sorted(mapping.items()):
            text = str(dataset_id or "").strip()
            if text:
                refs.append((section, str(domain), text))
    return {domain: dataset_id for _, domain, dataset_id in refs}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp status", description="Show qdp_v2 active data base status.")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = status_payload(workspace_root=str(args.workspace_root or "") or None)
    print_status(payload, as_json=bool(args.json))
    return 0 if str(payload.get("status", "")) in {"ok", "not_initialized"} else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
