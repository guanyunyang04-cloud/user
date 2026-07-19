from __future__ import annotations

from pathlib import Path

from quant_data_platform.qdp_v2.manifest import (
    atomic_write_json,
    qdp_snapshot_payload,
    qdp_snapshot_sha256,
)


def _write_snapshot(root: Path, *, reverse: bool = False) -> None:
    rows = [
        ("market_daily_raw", "daily_v1", 10),
        ("security_status", "status_v1", 4),
    ]
    if reverse:
        rows.reverse()
    for domain, dataset_id, row_count in rows:
        atomic_write_json(
            root / "datasets" / domain / dataset_id / "dataset.json",
            {
                "dataset_id": dataset_id,
                "domain": domain,
                "row_count": row_count,
                "schema_hash": f"schema-{domain}",
                "shards": [],
            },
        )
    atomic_write_json(
        root / "active" / "active.json",
        {
            "version": 2,
            "active_as_of_date": "2026-07-16",
            "datasets": {domain: dataset_id for domain, dataset_id, _ in rows},
        },
    )


def test_qdp_snapshot_digest_is_order_independent_and_manifest_sensitive(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_snapshot(first)
    _write_snapshot(second, reverse=True)

    first_payload = qdp_snapshot_payload(first)
    assert first_payload["active_as_of_date"] == "2026-07-16"
    assert list(first_payload["datasets"]) == ["market_daily_raw", "security_status"]
    assert qdp_snapshot_sha256(first) == qdp_snapshot_sha256(second)

    status_path = second / "datasets" / "security_status" / "status_v1" / "dataset.json"
    changed = {
        "dataset_id": "status_v1",
        "domain": "security_status",
        "row_count": 5,
        "schema_hash": "schema-security_status",
        "shards": [],
    }
    atomic_write_json(status_path, changed)
    assert qdp_snapshot_sha256(first) != qdp_snapshot_sha256(second)
