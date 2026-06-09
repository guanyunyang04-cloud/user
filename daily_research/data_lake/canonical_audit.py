from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from daily_research.data_lake.canonical import (
    DEFAULT_CANONICAL_START_DATE,
    DEFAULT_EXTERNAL_QUANT_DATA_ROOT,
    DEFAULT_PATH_POLICY_ROOT,
    build_lake_inventory,
    write_inventory_report,
)
from daily_research.data_lake.catalog import DEFAULT_DATA_LAKE_ROOT, ResearchDataLake


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit canonical data lake, external zips, and forecast memmaps.")
    parser.add_argument("--lake-root", default=str(DEFAULT_DATA_LAKE_ROOT))
    parser.add_argument("--external-data-root", default=str(DEFAULT_EXTERNAL_QUANT_DATA_ROOT))
    parser.add_argument("--path-policy-root", default=str(DEFAULT_PATH_POLICY_ROOT))
    parser.add_argument("--start-date", default=DEFAULT_CANONICAL_START_DATE)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--json", action="store_true", help="Print compact JSON summary to stdout.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lake = ResearchDataLake(args.lake_root)
    output_dir = Path(args.output_dir) if args.output_dir else lake.root / "canonical" / "audits" / datetime.now().strftime("%Y%m%d")
    inventory = build_lake_inventory(
        lake,
        external_data_root=args.external_data_root,
        path_policy_root=args.path_policy_root,
        start_date=args.start_date,
    )
    report_path = write_inventory_report(inventory, output_dir=output_dir)
    summary: dict[str, Any] = {
        "status": "ok",
        "report_path": str(report_path.resolve()),
        "dataset_count": int(inventory.get("dataset_count", 0) or 0),
        "duplicate_group_count": int(len(inventory.get("duplicate_dataset_groups", []) or [])),
        "empty_dir_count": int(len(inventory.get("filesystem", {}).get("empty_dirs", []) or [])),
        "orphan_fingerprint_dir_count": int(len(inventory.get("filesystem", {}).get("orphan_fingerprint_dirs", []) or [])),
        "memmap_manifest_count": int(inventory.get("memmaps", {}).get("manifest_count", 0) or 0),
        "forecast_dat_bytes": int(inventory.get("memmaps", {}).get("forecast_dat_bytes", 0) or 0),
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    else:
        print(f"canonical_audit_report={report_path.resolve()}")
        print(f"dataset_count={summary['dataset_count']}")
        print(f"duplicate_group_count={summary['duplicate_group_count']}")
        print(f"empty_dir_count={summary['empty_dir_count']}")
        print(f"orphan_fingerprint_dir_count={summary['orphan_fingerprint_dir_count']}")
        print(f"memmap_manifest_count={summary['memmap_manifest_count']}")
        print(f"forecast_dat_bytes={summary['forecast_dat_bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
