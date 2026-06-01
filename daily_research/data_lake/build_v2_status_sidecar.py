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
from daily_research.data_lake.v2_status_sidecar import build_v2_status_sidecar


ACTIVE_ARTIFACT = "daily_research/output/active_execution_strategy.json"
DEFAULT_DATASET_ID = "policy_input_bundle__45e3d8c059ba718426a9f887"
DEFAULT_RUN_TAG = "v2_status_sidecar_20260601_01"


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
    return str(value) if isinstance(value, Path) else value


def _active_artifact_has_diff() -> bool:
    result = subprocess.run(["git", "diff", "--", ACTIVE_ARTIFACT], check=False, capture_output=True, text=True)
    return bool(str(result.stdout or "").strip() or str(result.stderr or "").strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build v2 PIT status sidecar for daily_research data lake.")
    parser.add_argument("--data-lake-root", default="")
    parser.add_argument("--source-market-dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--universe-snapshot-dataset-id", default="")
    parser.add_argument("--security-status-dataset-id", default="")
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--run-tag", default=DEFAULT_RUN_TAG)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if _active_artifact_has_diff():
        raise ValueError(f"active_artifact_diff_blocker: {ACTIVE_ARTIFACT} has uncommitted diff.")
    lake = ResearchDataLake(str(args.data_lake_root or "").strip() or None)
    record, _frame, summary = build_v2_status_sidecar(
        lake=lake,
        source_market_dataset_id=str(args.source_market_dataset_id).strip(),
        start_date=str(args.start_date or "").strip(),
        end_date=str(args.end_date or "").strip(),
        universe_snapshot_dataset_id=str(args.universe_snapshot_dataset_id or "").strip(),
        security_status_dataset_id=str(args.security_status_dataset_id or "").strip(),
        reuse=not bool(args.refresh),
    )
    payload = {
        "status": "ok",
        "run_tag": str(args.run_tag),
        "dataset_id": record.dataset_id,
        "dataset_kind": record.dataset_kind,
        "fingerprint": record.fingerprint,
        "content_paths": record.content_paths,
        "row_counts": record.row_counts,
        "summary": summary,
    }
    lake.write_catalog_manifest()
    print(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    main()
