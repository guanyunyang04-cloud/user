from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from quantlab.data.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    _manifest_schema_from_arrow,
    qdp_v2_root,
    read_dataset_manifest,
    resolve_manifest_path,
    write_active_manifest,
    write_dataset_manifest,
)
from quantlab.data.qdp_v2.repair import (
    QdpV2RepairError,
    append_active_shard,
    bulk_append_active_shards_from_parquet,
    mutate_active_shards_from_parquet,
    patch_active_cells,
    replace_active_table_from_parquet,
    resolve_active_domain,
)
from quantlab.data.qdp_v2.repair import mutation as repair_mutation

DOMAIN = "market_daily_raw"
DATASET_ID = "market_daily_raw__repair_fixture"


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    (workspace / "data" / "qdp").mkdir(parents=True)
    return workspace


def _frame(trade_date: str, symbol: str, close: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [trade_date],
            "symbol": [symbol],
            "open": [close - 0.1],
            "high": [close + 0.2],
            "low": [close - 0.2],
            "close": [close],
            "volume": [100.0],
            "amount": [1000.0],
        }
    )


def _prepared(workspace: Path, name: str, frame: pd.DataFrame) -> Path:
    path = workspace / "prepared" / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, engine="pyarrow")
    return path


def _install_active(workspace: Path) -> list[Path]:
    root = qdp_v2_root(workspace)
    frames = [
        _frame("2026-01-05", "000001.SZ", 10.0),
        _frame("2026-01-06", "000002.SZ", 20.0),
    ]
    paths: list[Path] = []
    entries: list[ShardManifestEntry] = []
    for index, frame in enumerate(frames):
        path = root / "datasets" / DOMAIN / DATASET_ID / "shards" / f"part_{index}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False, engine="pyarrow")
        paths.append(path)
        entries.append(
            ShardManifestEntry(
                path=path.relative_to(root).as_posix(),
                row_count=len(frame),
                start_date=str(frame["trade_date"].min()),
                end_date=str(frame["trade_date"].max()),
                file_size=path.stat().st_size,
            )
        )
    schema = _manifest_schema_from_arrow(pq.read_schema(paths[0]))
    write_dataset_manifest(
        root,
        DatasetManifest(
            dataset_id=DATASET_ID,
            domain=DOMAIN,
            layer="raw",
            frequency="1d",
            contract_version="qdp_v2_market_daily_raw_v1",
            primary_key=["trade_date", "symbol"],
            start_date="2026-01-05",
            end_date="2026-01-06",
            row_count=2,
            shards=entries,
            source={"provider": "unit"},
            quality={"primary_key_unique": True},
            schema=schema,
        ),
    )
    write_active_manifest(
        root,
        {
            "version": 2,
            "active_as_of_date": "2026-01-06",
            "datasets": {DOMAIN: DATASET_ID},
        },
    )
    return paths


def _rows(workspace: Path) -> pd.DataFrame:
    context = resolve_active_domain(DOMAIN, workspace_root=workspace)
    return pd.concat(
        [pd.read_parquet(path) for path in context.shard_paths],
        ignore_index=True,
    ).sort_values(["trade_date", "symbol"])


def _patch_request(workspace: Path, changes: list[dict[str, object]]) -> Path:
    path = workspace / "requests" / "patch.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "domain": DOMAIN,
                "changes": changes,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_append_and_bulk_append_update_readable_manifest(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _install_active(workspace)

    result = append_active_shard(
        DOMAIN,
        _frame("2026-01-07", "000003.SZ", 30.0),
        workspace_root=workspace,
    )
    more = _prepared(
        workspace,
        "more",
        pd.concat(
            [
                _frame("2026-01-08", "000004.SZ", 40.0),
                _frame("2026-01-09", "000005.SZ", 50.0),
            ],
            ignore_index=True,
        ),
    )
    bulk = bulk_append_active_shards_from_parquet(
        DOMAIN,
        [more],
        workspace_root=workspace,
    )

    context = resolve_active_domain(DOMAIN, workspace_root=workspace)
    assert result["status"] == "appended"
    assert bulk["status"] == "appended"
    assert context.manifest.row_count == 5
    assert len(context.shard_paths) == 4
    assert _rows(workspace)["symbol"].tolist() == [
        "000001.SZ",
        "000002.SZ",
        "000003.SZ",
        "000004.SZ",
        "000005.SZ",
    ]


def test_append_rejects_primary_key_overlap_without_changing_manifest(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _install_active(workspace)
    context = resolve_active_domain(DOMAIN, workspace_root=workspace)
    before = context.manifest_path.read_bytes()

    with pytest.raises(QdpV2RepairError, match="primary_key_overlap"):
        append_active_shard(
            DOMAIN,
            _frame("2026-01-05", "000001.SZ", 99.0),
            workspace_root=workspace,
        )

    assert context.manifest_path.read_bytes() == before
    assert resolve_active_domain(DOMAIN, workspace_root=workspace).manifest.row_count == 2


def test_mutate_commits_manifest_before_removing_old_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    old_paths = _install_active(workspace)
    replacement = _prepared(
        workspace, "replacement", _frame("2026-01-05", "000001.SZ", 11.0)
    )
    appended = _prepared(
        workspace, "appended", _frame("2026-01-07", "000003.SZ", 30.0)
    )
    events: list[str] = []
    original_commit = repair_mutation._commit_manifest
    original_remove = repair_mutation._remove_old_shards

    def record_commit(context, manifest):
        original_commit(context, manifest)
        events.append("manifest_committed")

    def record_remove(context, paths):
        events.append("old_shards_removed")
        return original_remove(context, paths)

    monkeypatch.setattr(repair_mutation, "_commit_manifest", record_commit)
    monkeypatch.setattr(repair_mutation, "_remove_old_shards", record_remove)

    result = mutate_active_shards_from_parquet(
        DOMAIN,
        replacements=[(old_paths[0], replacement)],
        removals=[old_paths[1]],
        appends=[appended],
        workspace_root=workspace,
    )

    assert events == ["manifest_committed", "old_shards_removed"]
    assert result["status"] == "mutated"
    assert not old_paths[0].exists()
    assert not old_paths[1].exists()
    assert _rows(workspace)[["symbol", "close"]].to_dict("records") == [
        {"symbol": "000001.SZ", "close": 11.0},
        {"symbol": "000003.SZ", "close": 30.0},
    ]


def test_failed_commit_keeps_old_data_and_retry_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    old_paths = _install_active(workspace)
    before = resolve_active_domain(DOMAIN, workspace_root=workspace).manifest_path.read_bytes()
    original_commit = repair_mutation._commit_manifest

    def fail_commit(*_args, **_kwargs):
        raise OSError("simulated interruption")

    monkeypatch.setattr(repair_mutation, "_commit_manifest", fail_commit)
    with pytest.raises(OSError, match="simulated interruption"):
        append_active_shard(
            DOMAIN,
            _frame("2026-01-07", "000003.SZ", 30.0),
            workspace_root=workspace,
        )

    context = resolve_active_domain(DOMAIN, workspace_root=workspace)
    assert context.manifest_path.read_bytes() == before
    assert all(path.is_file() for path in old_paths)
    monkeypatch.setattr(repair_mutation, "_commit_manifest", original_commit)

    result = append_active_shard(
        DOMAIN,
        _frame("2026-01-07", "000003.SZ", 30.0),
        workspace_root=workspace,
    )
    assert result["status"] == "appended"
    assert resolve_active_domain(DOMAIN, workspace_root=workspace).manifest.row_count == 3


def test_replace_table_allows_explicit_schema_and_primary_key_change(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    old_paths = _install_active(workspace)
    prepared = _prepared(
        workspace,
        "dimension",
        pd.DataFrame({"symbol": ["000001.SZ", "000002.SZ"], "name": ["A", "B"]}),
    )

    result = replace_active_table_from_parquet(
        DOMAIN,
        prepared,
        workspace_root=workspace,
        primary_key=["symbol"],
        contract_version="unit_dimension_v1",
    )

    context = resolve_active_domain(DOMAIN, workspace_root=workspace)
    assert result["status"] == "replaced"
    assert context.manifest.primary_key == ["symbol"]
    assert context.manifest.contract_version == "unit_dimension_v1"
    assert [item["name"] for item in context.manifest.schema] == ["symbol", "name"]
    assert pd.read_parquet(context.shard_paths[0]).to_dict("records") == [
        {"symbol": "000001.SZ", "name": "A"},
        {"symbol": "000002.SZ", "name": "B"},
    ]
    assert all(not path.exists() for path in old_paths)


def test_patch_checks_expected_value_and_changes_only_requested_cell(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _install_active(workspace)
    change = {
        "key": {"trade_date": "2026-01-05", "symbol": "000001.SZ"},
        "column": "close",
        "expected": 10.0,
        "value": 10.5,
    }
    request = _patch_request(workspace, [change])
    context = resolve_active_domain(DOMAIN, workspace_root=workspace)
    before_manifest = context.manifest_path.read_bytes()
    before_shards = [path.read_bytes() for path in context.shard_paths]

    dry_run = patch_active_cells(request, workspace_root=workspace, apply=False)
    assert dry_run["status"] == "would_patch"
    assert context.manifest_path.read_bytes() == before_manifest
    assert [path.read_bytes() for path in context.shard_paths] == before_shards

    wrong = _patch_request(workspace, [{**change, "expected": 12.0}])
    with pytest.raises(QdpV2RepairError, match="expected_mismatch"):
        patch_active_cells(wrong, workspace_root=workspace, apply=True)

    request = _patch_request(workspace, [change])
    result = patch_active_cells(request, workspace_root=workspace, apply=True)
    rows = _rows(workspace)
    assert result["status"] == "patched"
    assert rows.loc[rows["symbol"] == "000001.SZ", "close"].item() == 10.5
    assert rows.loc[rows["symbol"] == "000002.SZ", "close"].item() == 20.0


def test_manifest_paths_remain_readable_after_updates(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _install_active(workspace)
    append_active_shard(
        DOMAIN,
        _frame("2026-01-07", "000003.SZ", 30.0),
        workspace_root=workspace,
    )
    context = resolve_active_domain(DOMAIN, workspace_root=workspace)
    manifest = read_dataset_manifest(context.manifest_path)

    for entry in manifest.shards:
        path = resolve_manifest_path(entry.path, root=context.root)
        assert path.is_file()
        assert pq.ParquetFile(path).metadata.num_rows == entry.row_count
