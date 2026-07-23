from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from daily_research.model_registry import (
    DEFAULT_REGISTRY,
    WORKSPACE_ROOT,
    bundles,
    load_registry,
    resolve_bundle,
    verify_registry,
)


RECORD_INDEX = WORKSPACE_ROOT / "daily_research/research_records/seq100/index.json"
STUDY_ROOT = WORKSPACE_ROOT / "daily_research/studies"
QDP_ACTIVE = WORKSPACE_ROOT / "quant_data_platform/data/qdp_v2/active/active.json"
BASE_PACK = WORKSPACE_ROOT / "daily_research/data/research_store/seq100_pit_l35v2_v1/pack/manifest.json"
ACTIVE_STUDIES = {
    "signal-close-path-value-2x2": STUDY_ROOT / "signal_close_path_value_2x2_v1.json",
    "pit-oof-trees": STUDY_ROOT / "pit_l35v2_oof_trees_v1.json",
    "l35v2-batch1024": STUDY_ROOT / "l35v2_batch1024.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def status() -> dict[str, Any]:
    registry = load_registry(DEFAULT_REGISTRY)
    qdp = _read_json(QDP_ACTIVE) if QDP_ACTIVE.is_file() else {}
    studies = {
        name: _read_json(path) if path.is_file() else {"status": "missing"}
        for name, path in ACTIVE_STUDIES.items()
    }
    return {
        "status": "ok",
        "qdp_active_as_of_date": qdp.get("active_as_of_date"),
        "base_pack_present": BASE_PACK.is_file(),
        "selected_research_baseline": registry["selected_research_baseline"],
        "registered_model_bundles": len(list(registry["bundles"])),
        "active_studies": {
            name: {
                "study_id": value.get("study_id"),
                "status": value.get("status"),
            }
            for name, value in studies.items()
        },
        "research_record_index": str(RECORD_INDEX),
    }


def verify() -> dict[str, Any]:
    model_check = verify_registry(DEFAULT_REGISTRY)
    errors = list(model_check["errors"])
    if not QDP_ACTIVE.is_file():
        errors.append(f"missing_qdp_active:{QDP_ACTIVE}")
    if not BASE_PACK.is_file():
        errors.append(f"missing_base_pack:{BASE_PACK}")
    if not RECORD_INDEX.is_file():
        errors.append(f"missing_record_index:{RECORD_INDEX}")
        records: dict[str, Any] = {}
    else:
        records = _read_json(RECORD_INDEX)
        for row in list(records.get("records", []) or []):
            artifact = (WORKSPACE_ROOT / str(row.get("path", ""))).resolve()
            if not artifact.is_file():
                errors.append(f"missing_research_record:{row.get('id')}:{artifact}")
    for name, path in ACTIVE_STUDIES.items():
        if not path.is_file():
            errors.append(f"missing_active_study:{name}:{path}")
            continue
        study = _read_json(path)
        if isinstance(study.get("contract"), dict):
            declared = str(study.get("contract_sha256", "") or "")
            computed = _canonical_json_sha256(study["contract"])
            if not declared or declared != computed:
                errors.append(f"active_study_contract_hash_mismatch:{name}")
    return {
        "status": "ok" if not errors else "blocked",
        "model_registry": model_check,
        "research_record_count": len(list(records.get("records", []) or [])),
        "active_study_count": len(ACTIVE_STUDIES),
        "errors": errors,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compact Seq100 research entrypoint.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="Show data, model registry, and active studies.")
    commands.add_parser("verify", help="Verify stable models and compact research records.")
    model = commands.add_parser("model", help="Resolve one registered checkpoint bundle.")
    model.add_argument("model_id")
    model.add_argument("year", type=int)
    study = commands.add_parser("study", help="Show one active study contract.")
    study.add_argument("study_id", choices=tuple(ACTIVE_STUDIES))
    commands.add_parser("models", help="List registered model bundles.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "status":
        result: Any = status()
    elif args.command == "verify":
        result = verify()
    elif args.command == "model":
        result = resolve_bundle(args.model_id, args.year)
    elif args.command == "study":
        result = _read_json(ACTIVE_STUDIES[args.study_id])
    else:
        result = bundles(load_registry(DEFAULT_REGISTRY))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not isinstance(result, dict) or result.get("status") != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
