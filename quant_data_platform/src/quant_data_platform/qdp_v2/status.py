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
)


def v2_active_exists(workspace_root: str | Path | None = None) -> bool:
    root = qdp_v2_root(workspace_root)
    return (root / "active" / "active.json").exists()


def status_payload(*, workspace_root: str | Path | None = None, verify_files: bool = False) -> dict[str, Any]:
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
    missing_shards: list[dict[str, Any]] = []
    for _, domain, dataset_id in _active_dataset_refs(active):
        manifest_path = dataset_manifest_for_id(root, dataset_id, domain)
        if manifest_path is None:
            missing.append({"domain": domain, "dataset_id": dataset_id})
            continue
        manifest = read_dataset_manifest(manifest_path)
        shard_count = len(manifest.shards)
        existing_shards: int | None = None
        if verify_files:
            existing_shards = sum(
                1 for shard in manifest.shards if _manifest_path(shard.path, root).is_file()
            )
            if existing_shards != shard_count:
                missing_shards.append(
                    {
                        "domain": domain,
                        "dataset_id": dataset_id,
                        "missing_shard_count": shard_count - existing_shards,
                    }
                )
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
            "file_verification": "verified" if verify_files else "not_requested",
            "checked_through": str(
                dict(manifest.source or {}).get("checked_through", "")
            ),
            "source_contract": str(
                dict(manifest.source or {}).get("source_contract", "")
            ),
            "missing_daily_keys": dict(manifest.source or {}).get(
                "missing_daily_keys"
            ),
            "future_source_dates": dict(manifest.quality or {}).get(
                "future_source_dates"
            ),
            "secondary_validation_status": str(
                dict(manifest.source or {}).get(
                    "secondary_validation_status", ""
                )
            ),
            "secondary_validation_at": str(
                dict(manifest.source or {}).get(
                    "secondary_validation_at", ""
                )
            ),
            "secondary_compared_count": dict(manifest.source or {}).get(
                "secondary_compared_count"
            ),
            "secondary_material_mismatch_count": dict(
                manifest.source or {}
            ).get("secondary_material_mismatch_count"),
        }
    return {
        "status": "missing_dataset_manifest" if missing else ("missing_shards" if missing_shards else "ok"),
        "qdp_v2_root": str(root.resolve()),
        "active_as_of_date": str(active.get("active_as_of_date", "") or ""),
        "scope": dict(active.get("scope", {}) or {}),
        "active_manifest": str((root / "active" / "active.json").resolve()),
        "dataset_count": len(datasets),
        "missing": missing,
        "missing_shards": missing_shards,
        "datasets": datasets,
    }


def _manifest_path(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


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
            f"rows={item.get('row_count', 0)} shards="
            f"{item.get('existing_shards') if item.get('existing_shards') is not None else 'not_verified'}/{item.get('shard_count', 0)}"
        )
        if item.get("checked_through"):
            print(
                "  auxiliary: "
                f"checked_through={item.get('checked_through')} "
                f"missing_daily_keys={item.get('missing_daily_keys')} "
                f"future_source_dates={item.get('future_source_dates')} "
                f"secondary={item.get('secondary_validation_status')} "
                f"compared={item.get('secondary_compared_count')} "
                f"mismatches={item.get('secondary_material_mismatch_count')} "
                f"validated_at={item.get('secondary_validation_at')}"
            )
    missing = list(payload.get("missing", []) or [])
    if missing:
        print(f"missing: {json.dumps(json_safe(missing), ensure_ascii=False)}")


def _active_dataset_refs(active: dict[str, Any]) -> list[tuple[str, str, str]]:
    mapping = active_dataset_map(active)
    return [("datasets", str(domain), str(dataset_id)) for domain, dataset_id in sorted(mapping.items()) if str(dataset_id or "").strip()]


def active_dataset_map(active: dict[str, Any] | Any) -> dict[str, str]:
    payload = dict(active or {})
    return {str(domain): str(dataset_id) for domain, dataset_id in dict(payload.get("datasets", {}) or {}).items() if str(dataset_id or "").strip()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp status", description="Show qdp_v2 active data base status.")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--verify-files", action="store_true", help="Verify every active shard path; omitted status is metadata-only.")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = status_payload(workspace_root=str(args.workspace_root or "") or None, verify_files=bool(args.verify_files))
    print_status(payload, as_json=bool(args.json))
    return 0 if str(payload.get("status", "")) in {"ok", "not_initialized"} else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
