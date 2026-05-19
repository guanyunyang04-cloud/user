from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.data_lake.catalog import ResearchDataLake
from daily_research.data_lake.sector_board_views import (
    DEFAULT_BOARD_SOURCE_PATH,
    DEFAULT_INDUSTRY_SOURCE_PATH,
    SNAPSHOT_SEMANTICS,
    SectorBoardViewSpec,
    build_sector_board_view_from_policy_bundle,
)

ACTIVE_ARTIFACT = "daily_research/output/active_execution_strategy.json"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an auditable sector/board metadata view from an existing policy input bundle.")
    parser.add_argument("--data-lake-root", default="")
    parser.add_argument("--source-market-dataset-id", required=True)
    parser.add_argument("--industry-source-path", default=DEFAULT_INDUSTRY_SOURCE_PATH)
    parser.add_argument("--board-source-path", default=DEFAULT_BOARD_SOURCE_PATH)
    parser.add_argument("--as-of-date", default="")
    parser.add_argument("--view-kind", default=SNAPSHOT_SEMANTICS)
    parser.add_argument("--view-name", default="sector_board_latest_static")
    parser.add_argument("--refresh", action="store_true")
    return parser


def _active_artifact_has_diff() -> bool:
    result = subprocess.run(
        ["git", "diff", "--", ACTIVE_ARTIFACT],
        check=False,
        capture_output=True,
        text=True,
    )
    return bool(str(result.stdout or "").strip() or str(result.stderr or "").strip())


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if _active_artifact_has_diff():
        raise ValueError(f"active_artifact_diff_blocker: {ACTIVE_ARTIFACT} has uncommitted diff.")
    lake = ResearchDataLake(str(args.data_lake_root or "").strip() or None)
    spec = SectorBoardViewSpec(
        source_market_dataset_id=str(args.source_market_dataset_id).strip(),
        industry_source_path=str(args.industry_source_path or "").strip(),
        board_source_path=str(args.board_source_path or "").strip(),
        as_of_date=str(args.as_of_date or "").strip(),
        view_kind=str(args.view_kind or SNAPSHOT_SEMANTICS).strip(),
        view_name=str(args.view_name or "sector_board_latest_static").strip(),
        snapshot_semantics=SNAPSHOT_SEMANTICS,
    )
    record = build_sector_board_view_from_policy_bundle(lake=lake, spec=spec, reuse=not bool(args.refresh))
    metadata = lake.describe_dataset(record.dataset_id)
    source_cache = dict(metadata.get("source_cache", {}) or {})
    manifest = {
        "status": "ok",
        "dataset_id": record.dataset_id,
        "dataset_kind": record.dataset_kind,
        "fingerprint": record.fingerprint,
        "view_kind": str(metadata.get("parameters", {}).get("view_kind", "") or ""),
        "view_name": str(metadata.get("parameters", {}).get("view_name", "") or ""),
        "snapshot_semantics": str(metadata.get("parameters", {}).get("snapshot_semantics", "") or SNAPSHOT_SEMANTICS),
        "source_market_dataset_id": spec.source_market_dataset_id,
        "row_counts": record.row_counts,
        "content_paths": record.content_paths,
        "industry_coverage": source_cache.get("industry_coverage", {}),
        "board_coverage": source_cache.get("board_coverage", {}),
    }
    lake.write_catalog_manifest()
    print(json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2))
    return manifest


if __name__ == "__main__":
    main()
