from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from daily_research.path_policy.forecast_dataset import load_forecast_memmap_dataset


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a forecast memmap manifest before reusing it for training.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expect-source-market-dataset-id", default="")
    parser.add_argument("--expect-pool-view-id", default="")
    parser.add_argument("--expect-pool-view-kind", default="")
    parser.add_argument("--expect-pool-view-name", default="")
    parser.add_argument("--min-universe-size", type=int, default=0)
    parser.add_argument("--min-train-rows", type=int, default=0)
    parser.add_argument("--require-roles", default="train,validation,test")
    parser.add_argument("--json-output", default="")
    return parser


def _required_roles(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def _load_failure_blocker(message: str) -> str:
    text = str(message or "").lower()
    if "missing" in text and ("forecast_" in text or "feature_store" in text or "memmap" in text):
        return "missing_memmap_file"
    if "invalid" in text and "size" in text:
        return "memmap_file_size_mismatch"
    if "sample_count" in text:
        return "sample_count_mismatch"
    if "feature_store_shape" in text:
        return "feature_store_shape_mismatch"
    return "load_failed"


def validate_forecast_memmap_manifest(
    *,
    manifest: str | Path,
    expect_source_market_dataset_id: str = "",
    expect_pool_view_id: str = "",
    expect_pool_view_kind: str = "",
    expect_pool_view_name: str = "",
    min_universe_size: int = 0,
    min_train_rows: int = 0,
    require_roles: list[str] | tuple[str, ...] | str = ("train", "validation", "test"),
) -> dict[str, Any]:
    blockers: list[str] = []
    manifest_path = Path(manifest)
    try:
        dataset = load_forecast_memmap_dataset(manifest_path)
    except Exception as exc:
        blocker = _load_failure_blocker(str(exc))
        return {
            "status": "blocked",
            "blockers": [blocker],
            "error_type": type(exc).__name__,
            "error": str(exc),
            "manifest_path": str(manifest_path.resolve()),
        }

    payload = dict(dataset.manifest)
    role_counts = {str(key): int(value) for key, value in dict(payload.get("sample_count_by_role", {}) or {}).items()}
    roles = _required_roles(require_roles if isinstance(require_roles, str) else ",".join(require_roles))
    for role in roles:
        if int(role_counts.get(role, 0)) <= 0:
            blockers.append(f"missing_role_{role}")

    if str(expect_source_market_dataset_id or "").strip():
        actual = str(payload.get("source_market_dataset_id", "") or "")
        if actual != str(expect_source_market_dataset_id).strip():
            blockers.append("source_market_dataset_mismatch")
    if str(expect_pool_view_id or "").strip():
        actual = str(payload.get("source_pool_view_id", "") or "")
        if actual != str(expect_pool_view_id).strip():
            blockers.append("source_pool_view_mismatch")
    if str(expect_pool_view_kind or "").strip():
        actual = str(payload.get("source_pool_view_kind", "") or "")
        if actual != str(expect_pool_view_kind).strip():
            blockers.append("source_pool_view_kind_mismatch")
    if str(expect_pool_view_name or "").strip():
        actual = str(payload.get("source_pool_view_name", "") or "")
        if actual != str(expect_pool_view_name).strip():
            blockers.append("source_pool_view_name_mismatch")

    universe_size = int(len(dataset.stock_values)) if int(len(dataset.stock_values)) > 0 else int(dataset.feature_store_shape[1])
    if int(min_universe_size or 0) > 0 and universe_size < int(min_universe_size):
        blockers.append("universe_size_below_minimum")
    if int(min_train_rows or 0) > 0 and int(role_counts.get("train", 0)) < int(min_train_rows):
        blockers.append("train_rows_below_minimum")

    sample_count = int(payload.get("sample_count", dataset.row_count) or dataset.row_count)
    if sample_count != int(dataset.row_count):
        blockers.append("sample_count_mismatch")
    if int(dataset.feature_store_shape[2]) != int(dataset.input_dim):
        blockers.append("feature_store_shape_mismatch")

    return {
        "status": "ok" if not blockers else "blocked",
        "blockers": blockers,
        "manifest_path": str(manifest_path.resolve()),
        "sample_count": int(dataset.row_count),
        "sample_count_by_role": role_counts,
        "universe_size": universe_size,
        "feature_count": int(dataset.input_dim),
        "feature_store_shape": [int(item) for item in dataset.feature_store_shape],
        "source_market_dataset_id": str(payload.get("source_market_dataset_id", "") or ""),
        "source_pool_view_id": str(payload.get("source_pool_view_id", "") or ""),
        "source_pool_view_kind": str(payload.get("source_pool_view_kind", "") or ""),
        "source_pool_view_name": str(payload.get("source_pool_view_name", "") or ""),
        "source_sector_board_view_id": str(payload.get("source_sector_board_view_id", "") or ""),
        "dataset_mode": str(payload.get("dataset_mode", "") or ""),
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    report = validate_forecast_memmap_manifest(
        manifest=args.manifest,
        expect_source_market_dataset_id=str(args.expect_source_market_dataset_id or "").strip(),
        expect_pool_view_id=str(args.expect_pool_view_id or "").strip(),
        expect_pool_view_kind=str(args.expect_pool_view_kind or "").strip(),
        expect_pool_view_name=str(args.expect_pool_view_name or "").strip(),
        min_universe_size=int(args.min_universe_size),
        min_train_rows=int(args.min_train_rows),
        require_roles=_required_roles(args.require_roles),
    )
    if str(args.json_output or "").strip():
        output_path = Path(str(args.json_output).strip())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_json_safe(report), ensure_ascii=False, indent=2))
    if str(report.get("status", "")) != "ok":
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    main()
