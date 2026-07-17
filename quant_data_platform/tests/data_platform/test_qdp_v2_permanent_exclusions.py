from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_data_platform.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    atomic_write_json,
    qdp_v2_root,
    read_dataset_manifest,
    write_active_manifest,
    write_dataset_manifest,
)
from quant_data_platform.qdp_v2.permanent_exclusions import (
    POLICY_ID,
    empty_registry,
    filter_excluded_candidates,
    merge_registry_entries,
    purge_permanent_exclusions,
    registry_path,
)


def _entry(
    *,
    security_id: str = "QDP-CN-SSE-600001",
    symbol: str = "600001.SH",
    reason: str = "st",
) -> dict[str, object]:
    return {
        "security_id": security_id,
        "current_symbol": symbol,
        "symbols": [symbol],
        "exclusion_date": "2026-07-16",
        "trigger_name": "ST测试",
        "reason": reason,
    }


def test_registry_is_one_way_when_security_later_recovers() -> None:
    registry, added = merge_registry_entries(empty_registry(), [_entry()])
    recovered, recovered_added = merge_registry_entries(registry, [])

    assert len(added) == 1
    assert recovered_added == []
    assert recovered["exclusion_count"] == 1
    assert recovered["exclusions"][0]["reason"] == "st"


def test_new_st_is_added_and_stable_identity_or_symbol_is_filtered() -> None:
    registry, _ = merge_registry_entries(empty_registry(), [_entry()])
    updated, added = merge_registry_entries(
        registry,
        [
            _entry(
                security_id="QDP-CN-SZSE-000001",
                symbol="000001.SZ",
            )
        ],
    )
    frame = pd.DataFrame(
        {
            "security_id": [
                "QDP-CN-SSE-600001",
                "QDP-CN-SZSE-000001",
                "QDP-CN-SSE-600002",
            ],
            "symbol": ["699999.SH", "000001.SZ", "600002.SH"],
        }
    )

    filtered = filter_excluded_candidates(
        frame,
        domain="symbol_history",
        registry=updated,
    )

    assert len(added) == 1
    assert filtered["symbol"].tolist() == ["600002.SH"]


def test_recovered_security_with_changed_code_is_filtered_by_normalized_name() -> None:
    registry, _ = merge_registry_entries(empty_registry(), [_entry()])
    provider_snapshot = pd.DataFrame(
        {
            "symbol": ["605999.SH", "600002.SH"],
            "name": ["测试", "正常股份"],
        }
    )

    filtered = filter_excluded_candidates(
        provider_snapshot,
        domain="universe_snapshot",
        registry=registry,
    )

    assert filtered["symbol"].tolist() == ["600002.SH"]


def _write_daily_store(workspace: Path) -> tuple[Path, str]:
    root = qdp_v2_root(workspace)
    dataset_id = "market_daily_raw__unit"
    shard = root / "datasets" / "market_daily_raw" / dataset_id / "shards" / "part.parquet"
    shard.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "symbol": ["600001.SH", "600002.SH"],
            "trade_date": ["2026-07-16", "2026-07-16"],
            "open": [10.0, 20.0],
            "high": [10.5, 20.5],
            "low": [9.5, 19.5],
            "close": [10.0, 20.0],
            "volume": [100.0, 200.0],
            "amount": [1000.0, 2000.0],
            "source": ["unit", "unit"],
            "adjusted_flag": ["none", "none"],
        }
    )
    frame.to_parquet(shard, index=False)
    schema = [
        {"name": str(field.name), "type": str(field.type)}
        for field in __import__("pyarrow.parquet", fromlist=["read_schema"]).read_schema(shard)
    ]
    write_dataset_manifest(
        root,
        DatasetManifest(
            dataset_id=dataset_id,
            domain="market_daily_raw",
            layer="raw",
            frequency="1d",
            contract_version="unit",
            primary_key=["trade_date", "symbol"],
            start_date="2026-07-16",
            end_date="2026-07-16",
            row_count=2,
            schema_hash="unit",
            schema=schema,
            shards=[
                ShardManifestEntry(
                    path=shard.relative_to(root).as_posix(),
                    row_count=2,
                    start_date="2026-07-16",
                    end_date="2026-07-16",
                )
            ],
            source={"provider": "unit"},
            quality={},
        ),
    )
    write_active_manifest(
        root,
        {
            "version": 2,
            "active_as_of_date": "2026-07-16",
            "scope": {"start_date": "2026-07-16", "end_date": "2026-07-16"},
            "datasets": {"market_daily_raw": dataset_id},
        },
    )
    return root, dataset_id


def test_in_place_purge_keeps_dataset_id_and_removes_rows(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    root, dataset_id = _write_daily_store(workspace)
    registry, _ = merge_registry_entries(empty_registry(), [_entry()])
    atomic_write_json(registry_path(workspace_root=workspace), registry)

    result = purge_permanent_exclusions(
        workspace_root=workspace,
        domains=("market_daily_raw",),
    )

    manifest = read_dataset_manifest(
        root / "datasets" / "market_daily_raw" / dataset_id / "dataset.json"
    )
    remaining = pd.read_parquet(root / manifest.shards[0].path)
    assert result["status"] == "completed"
    assert result["removed_rows"] == 1
    assert manifest.dataset_id == dataset_id
    assert manifest.row_count == 1
    assert remaining["symbol"].tolist() == ["600002.SH"]
    assert POLICY_ID in str(manifest.notes)
