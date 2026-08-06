from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from quant_data_platform.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    write_active_manifest,
    write_dataset_manifest,
)
from quant_data_platform.qdp_v2.pit_metadata_repair import (
    PitMetadataRepairError,
    run_repair,
)
from quant_data_platform.qdp_v2.status import active_dataset_map


def _install_domain(
    workspace: Path,
    *,
    domain: str,
    dataset_id: str,
    frame: pd.DataFrame,
    primary_key: list[str],
) -> None:
    root = qdp_v2_root(workspace)
    shard = root / "datasets" / domain / dataset_id / "shards" / "part.parquet"
    shard.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(shard, index=False)
    dates = frame["trade_date"].astype(str)
    write_dataset_manifest(
        root,
        DatasetManifest(
            dataset_id=dataset_id,
            domain=domain,
            layer="raw",
            frequency="1d",
            contract_version="unit",
            primary_key=primary_key,
            start_date=str(dates.min()),
            end_date=str(dates.max()),
            row_count=len(frame),
            shards=[
                ShardManifestEntry(
                    path=str(shard.relative_to(root)).replace("\\", "/"),
                    row_count=len(frame),
                    start_date=str(dates.min()),
                    end_date=str(dates.max()),
                    file_size=shard.stat().st_size,
                )
            ],
            source={"provider": "unit"},
            quality={},
        ),
    )


def _workspace(tmp_path: Path, *, broken_chain: bool = False) -> Path:
    workspace = tmp_path / "workspace"
    universe_id = "universe_snapshot__unit"
    name_id = "name_change__unit"
    _install_domain(
        workspace,
        domain="universe_snapshot",
        dataset_id=universe_id,
        primary_key=["trade_date", "symbol"],
        frame=pd.DataFrame(
            {
                "symbol": ["000001.SZ"] * 4 + ["000002.SZ"] * 2,
                "trade_date": [
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-04",
                    "2020-01-01",
                    "2020-01-02",
                ],
                "name": ["LATEST"] * 4 + ["UNCHANGED"] * 2,
                "exchange": ["SZSE"] * 6,
            }
        ),
    )
    _install_domain(
        workspace,
        domain="name_change",
        dataset_id=name_id,
        primary_key=["symbol", "trade_date", "change_type"],
        frame=pd.DataFrame(
            {
                "symbol": ["000001.SZ", "000001.SZ"],
                "trade_date": ["2020-01-02", "2020-01-04"],
                "old_name": ["OLD", "WRONG" if broken_chain else "NEW"],
                "new_name": ["NEW", "LATEST"],
                "change_type": ["short_name", "short_name"],
                "source": ["unit", "unit"],
            }
        ),
    )
    write_active_manifest(
        qdp_v2_root(workspace),
        {
            "version": 2,
            "active_as_of_date": "2020-01-04",
            "datasets": {
                "universe_snapshot": universe_id,
                "name_change": name_id,
            },
        },
    )
    return workspace


def _active_frame(workspace: Path) -> pd.DataFrame:
    root = qdp_v2_root(workspace)
    datasets = active_dataset_map(read_active_manifest(root))
    manifest_path = dataset_manifest_for_id(
        root, datasets["universe_snapshot"], "universe_snapshot"
    )
    assert manifest_path is not None
    manifest = read_dataset_manifest(manifest_path)
    paths = [resolve_manifest_path(item.path, root=root) for item in manifest.shards]
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def test_universe_name_pit_repair_preserves_dataset_and_untargeted_rows(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    before_active = read_active_manifest(qdp_v2_root(workspace))
    before = _active_frame(workspace)

    plan = run_repair(
        workspace_root=workspace,
        domains=("universe-name-pit",),
        apply=False,
    )["domains"]["universe-name-pit"]

    assert plan["changed_row_count"] == 3
    assert plan["changed_symbol_count"] == 1
    pd.testing.assert_frame_equal(_active_frame(workspace), before)

    applied = run_repair(
        workspace_root=workspace,
        domains=("universe-name-pit",),
        apply=True,
    )["domains"]["universe-name-pit"]
    after_active = read_active_manifest(qdp_v2_root(workspace))
    result = _active_frame(workspace).sort_values(["symbol", "trade_date"])

    assert applied["status"] == "applied"
    assert applied["post_commit_mismatch_rows"] == 0
    assert active_dataset_map(after_active) == active_dataset_map(before_active)
    assert result.loc[result["symbol"].eq("000001.SZ"), "name"].tolist() == [
        "OLD",
        "NEW",
        "NEW",
        "LATEST",
    ]
    assert result.loc[result["symbol"].eq("000002.SZ"), "name"].tolist() == [
        "UNCHANGED",
        "UNCHANGED",
    ]
    assert result["exchange"].eq("SZSE").all()


def test_universe_name_pit_repair_rejects_broken_event_chain(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path, broken_chain=True)

    with pytest.raises(PitMetadataRepairError, match="event_contract_failed"):
        run_repair(
            workspace_root=workspace,
            domains=("universe-name-pit",),
            apply=False,
        )
