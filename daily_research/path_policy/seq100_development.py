from __future__ import annotations

import argparse
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


STUDY_ROOT = WORKSPACE_ROOT / "daily_research/studies"
QDP_ACTIVE = WORKSPACE_ROOT / "quant_data_platform/data/qdp_v2/active/active.json"
BASE_PACK = WORKSPACE_ROOT / "daily_research/data/research_store/seq100_pit_l35v2_v1/pack/manifest.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _study_configs() -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(STUDY_ROOT.glob("*.json")):
        payload = _read_json(path)
        study_id = str(payload.get("study_id", path.stem))
        result[study_id] = path
    return result


def status() -> dict[str, Any]:
    registry = load_registry(DEFAULT_REGISTRY)
    qdp = _read_json(QDP_ACTIVE) if QDP_ACTIVE.is_file() else {}
    studies = {name: _read_json(path) for name, path in _study_configs().items()}
    return {
        "status": "ok",
        "qdp_active_as_of_date": qdp.get("active_as_of_date"),
        "base_pack_present": BASE_PACK.is_file(),
        "selected_research_baseline": registry["selected_research_baseline"],
        "registered_model_bundles": len(list(registry["bundles"])),
        "study_configs": {
            name: {
                "study_id": value.get("study_id"),
                "status": value.get("status"),
            }
            for name, value in studies.items()
        },
    }


def verify() -> dict[str, Any]:
    model_check = verify_registry(DEFAULT_REGISTRY)
    errors = list(model_check["errors"])
    if not QDP_ACTIVE.is_file():
        errors.append(f"missing_qdp_active:{QDP_ACTIVE}")
    if not BASE_PACK.is_file():
        errors.append(f"missing_base_pack:{BASE_PACK}")
    configs = _study_configs()
    return {
        "status": "ok" if not errors else "blocked",
        "model_registry": model_check,
        "study_config_count": len(configs),
        "errors": errors,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compact Seq100 research entrypoint.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="Show data, model registry, and study configs.")
    commands.add_parser("verify", help="Check required data and model files.")
    model = commands.add_parser("model", help="Resolve one registered checkpoint bundle.")
    model.add_argument("model_id")
    model.add_argument("year", type=int)
    study = commands.add_parser("study", help="Show one scientific study config.")
    study.add_argument("study_id")
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
        configs = _study_configs()
        if args.study_id not in configs:
            raise KeyError(f"unknown study config: {args.study_id}")
        result = _read_json(configs[args.study_id])
    else:
        result = bundles(load_registry(DEFAULT_REGISTRY))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not isinstance(result, dict) or result.get("status") != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
