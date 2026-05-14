from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.continuous_policy.training_dataset_cache import TRAINING_DATASET_CACHE_ROOT
from daily_research.data_lake import ResearchDataLake


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, dict) else {}


def _default_label_summary(sample_frame: pd.DataFrame, daily_frame: pd.DataFrame, *, zone: str) -> dict[str, Any]:
    return {
        "zone": zone,
        "sample_rows": int(len(sample_frame)),
        "daily_rows": int(len(daily_frame)),
        "observed_label_rows": int(len(sample_frame)),
        "unobserved_label_rows": 0,
        "observed_label_row_ratio": 1.0 if len(sample_frame) else 0.0,
        "daily_observed_rows": int(len(daily_frame)),
        "daily_unobserved_rows": 0,
        "is_training_safe": bool(zone == "strict_train" and len(sample_frame) > 0),
        "imported_from_legacy_pickle_cache": True,
    }


def import_legacy_training_dataset_caches(
    *,
    cache_root: str | Path | None = None,
    lake_root: str | Path | None = None,
    zone: str = "strict_train",
) -> list[dict[str, Any]]:
    root = Path(cache_root) if cache_root is not None else TRAINING_DATASET_CACHE_ROOT
    lake = ResearchDataLake(lake_root)
    imported: list[dict[str, Any]] = []
    if not root.exists():
        return imported
    for metadata_path in sorted(root.glob("*/metadata.json")):
        cache_dir = metadata_path.parent
        try:
            metadata = _read_json(metadata_path)
            spec = dict(metadata.get("spec", {}) or {})
            sample_frame = pd.read_pickle(cache_dir / "sample_frame.pkl")
            daily_frame = pd.read_pickle(cache_dir / "daily_frame.pkl")
            teacher_summary = _read_json(cache_dir / "teacher_summary.json")
        except Exception as exc:
            imported.append(
                {
                    "cache_dir": str(cache_dir.resolve()),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        label_summary = _default_label_summary(sample_frame, daily_frame, zone=zone)
        record = lake.save_training_dataset(
            spec=spec,
            sample_frame=sample_frame,
            daily_frame=daily_frame,
            teacher_summary=teacher_summary,
            zone=zone,
            label_completeness_summary=label_summary,
            source_cache={"legacy_cache_metadata": metadata},
            reuse=True,
        )
        imported.append(
            {
                "cache_dir": str(cache_dir.resolve()),
                "dataset_id": record.dataset_id,
                "zone": record.zone,
                "status": record.status,
                "sample_rows": int(len(record.sample_frame)),
                "daily_rows": int(len(record.daily_frame)),
            }
        )
    return imported


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import legacy pickle training dataset caches into the data lake.")
    parser.add_argument("--cache-root", default="")
    parser.add_argument("--data-lake-root", default="")
    parser.add_argument("--zone", default="strict_train")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    imported = import_legacy_training_dataset_caches(
        cache_root=str(args.cache_root or "").strip() or None,
        lake_root=str(args.data_lake_root or "").strip() or None,
        zone=str(args.zone or "strict_train"),
    )
    lake = ResearchDataLake(str(args.data_lake_root or "").strip() or None)
    manifest_path = lake.write_catalog_manifest()
    print(
        json.dumps(
            {"status": "ok", "imported": imported, "manifest_path": str(manifest_path.resolve())},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
