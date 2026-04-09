from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.execution.output_root_resolver import STATIC_PRODUCTION_ROOT
from daily_research.execution.update_default_candidate_production import (
    _resolve_default_static_execution_profile,
    _sync_production_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh production_root static fallback artifacts and manifest from an existing "
            "production run without retraining the model."
        )
    )
    parser.add_argument("--production-root", default=str(STATIC_PRODUCTION_ROOT))
    parser.add_argument("--production-run-dir", default="", help="Optional existing production run dir. Defaults to manifest active_production_run_dir.")
    parser.add_argument("--source-run-dir", default="", help="Optional formal source run dir. Defaults to manifest source_formal_run_dir.")
    parser.add_argument("--static-fallback-profile", default="", help="Optional static fallback execution profile override.")
    return parser.parse_args()


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    args = parse_args()
    production_root = Path(args.production_root).resolve()
    manifest_path = production_root / "production_retrain_manifest.json"
    manifest = _load_manifest(manifest_path)
    if not manifest:
        raise FileNotFoundError(f"Production manifest not found or empty: {manifest_path}")

    run_dir = Path(
        str(args.production_run_dir or manifest.get("active_production_run_dir", "")).strip()
    ).resolve()
    source_run_dir = Path(
        str(args.source_run_dir or manifest.get("source_formal_run_dir", "")).strip()
    ).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Production run dir does not exist: {run_dir}")
    if not source_run_dir.exists():
        raise FileNotFoundError(f"Source formal run dir does not exist: {source_run_dir}")

    latest_completed_date = str(manifest.get("launch_cutoff_date", "")).strip()
    latest_trainable_date = str(manifest.get("train_end_date", "")).strip()
    internal_monitor_start_date = str(manifest.get("internal_monitor_start_date", "")).strip()
    internal_monitor_days = int(manifest.get("internal_monitor_days", 0) or 0)
    train_start_date = str(manifest.get("train_start_date", "")).strip() or "20210101"
    static_fallback_profile = str(
        args.static_fallback_profile or _resolve_default_static_execution_profile()
    ).strip()
    if not latest_completed_date or not latest_trainable_date or not internal_monitor_start_date or internal_monitor_days <= 0:
        raise RuntimeError(
            "Production manifest is missing required training/launch metadata for a safe production-root refresh."
        )

    _sync_production_root(
        run_dir=run_dir,
        production_root=production_root,
        source_run_dir=source_run_dir,
        latest_completed_date=latest_completed_date,
        latest_trainable_date=latest_trainable_date,
        internal_monitor_start_date=internal_monitor_start_date,
        internal_monitor_days=internal_monitor_days,
        train_start_date=train_start_date,
        static_fallback_profile=static_fallback_profile,
        strategy_manifest_path=None,
        activate_strategy=False,
    )
    print(f"production_root={production_root}")
    print(f"production_run_dir={run_dir}")
    print(f"source_run_dir={source_run_dir}")
    print(f"static_fallback_profile={static_fallback_profile}")


if __name__ == "__main__":
    main()
