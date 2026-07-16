from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from quant_data_platform.core.json_io import read_json
from quant_data_platform.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    qdp_v2_root,
    read_dataset_manifest,
    resolve_manifest_path,
    write_active_manifest,
    write_dataset_manifest,
)
from quant_data_platform.qdp_v2.repair import (
    QdpV2RepairError,
    append_active_shard,
    bulk_append_active_shards_from_parquet,
    mutate_active_shards_from_parquet,
    repair_log_path,
    replace_active_shard_from_parquet,
    replace_active_shards,
    replace_active_shards_from_parquet,
    resolve_active_domain,
)


DOMAIN = "market_daily_raw"
DATASET_ID = "market_daily_raw__active_repair_fixture"
SCHEMA = [
    {"name": "trade_date", "type": "object"},
    {"name": "symbol", "type": "object"},
    {"name": "open", "type": "float64"},
    {"name": "high", "type": "float64"},
    {"name": "low", "type": "float64"},
    {"name": "close", "type": "float64"},
    {"name": "volume", "type": "float64"},
    {"name": "amount", "type": "float64"},
]


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    brain_manifest = workspace / "brain" / "brain_manifest.json"
    brain_manifest.parent.mkdir(parents=True, exist_ok=True)
    brain_manifest.write_text(
        json.dumps({"schema_version": 1, "brain_type": "main"}),
        encoding="utf-8",
    )
    return workspace


def _frame(
    trade_date: str,
    symbol: str,
    *,
    close: float,
) -> pd.DataFrame:
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_active(
    workspace: Path,
) -> tuple[Path, Path, list[Path]]:
    root = qdp_v2_root(workspace)
    frames = [
        _frame("2026-01-05", "000001.SZ", close=10.0),
        _frame("2026-01-06", "000002.SZ", close=20.0),
    ]
    shard_entries: list[ShardManifestEntry] = []
    shard_paths: list[Path] = []
    for index, frame in enumerate(frames):
        shard = (
            root
            / "datasets"
            / DOMAIN
            / DATASET_ID
            / "shards"
            / f"part_{index:02d}.parquet"
        )
        shard.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(shard, index=False, engine="pyarrow")
        shard_paths.append(shard)
        shard_entries.append(
            ShardManifestEntry(
                path=shard.relative_to(root).as_posix(),
                row_count=len(frame),
                start_date=str(frame["trade_date"].min()),
                end_date=str(frame["trade_date"].max()),
                file_size=shard.stat().st_size,
                schema_hash="unit-schema",
                content_key=f"fixture-{index}",
            )
        )
    manifest = DatasetManifest(
        dataset_id=DATASET_ID,
        domain=DOMAIN,
        layer="raw",
        frequency="1d",
        contract_version="qdp_v2_market_daily_raw_v1",
        primary_key=["trade_date", "symbol"],
        start_date="2026-01-05",
        end_date="2026-01-06",
        row_count=2,
        schema_hash="unit-schema",
        shards=shard_entries,
        source={"provider": "unit"},
        quality={"primary_key_unique": True},
        schema=SCHEMA,
    )
    manifest_path = write_dataset_manifest(root, manifest)
    active_path = write_active_manifest(
        root,
        {
            "version": 2,
            "active_as_of_date": "2026-01-06",
            "datasets": {DOMAIN: DATASET_ID},
        },
    )
    return active_path, manifest_path, shard_paths


def _log_records(workspace: Path) -> list[dict[str, object]]:
    path = repair_log_path(workspace)
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _prepared(workspace: Path, name: str, frame: pd.DataFrame) -> Path:
    path = workspace / "prepared" / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, engine="pyarrow")
    return path


def test_resolve_active_domain_uses_existing_active_dataset(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    active_path, manifest_path, shard_paths = _install_active(workspace)

    resolved = resolve_active_domain(DOMAIN, workspace_root=workspace)

    assert resolved.domain == DOMAIN
    assert resolved.dataset_id == DATASET_ID
    assert resolved.active_path == active_path.resolve()
    assert resolved.manifest_path == manifest_path.resolve()
    assert resolved.shard_paths == tuple(path.resolve() for path in shard_paths)
    assert resolved.active_sha256 == _sha256(active_path)
    assert resolved.manifest_sha256 == _sha256(manifest_path)


def test_replace_updates_only_selected_shard_in_place(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    active_path, manifest_path, shard_paths = _install_active(workspace)
    root = qdp_v2_root(workspace)
    active_before = active_path.read_bytes()
    retained_path = shard_paths[1]
    retained_hash = _sha256(retained_path)
    old_path = shard_paths[0]
    replacement = _frame("2026-01-05", "000001.SZ", close=11.0)

    result = replace_active_shards(
        DOMAIN,
        {old_path.resolve(): replacement},
        [old_path.relative_to(root).as_posix()],
        "correct one known bad close",
        workspace_root=workspace,
    )

    assert result["status"] == "replaced"
    assert result["dataset_id"] == DATASET_ID
    assert active_path.read_bytes() == active_before
    assert not old_path.exists()
    new_path = Path(result["new_shard_paths"][0])
    assert new_path.exists()
    assert new_path.parent == old_path.parent
    assert pd.read_parquet(new_path)["close"].tolist() == [11.0]
    assert retained_path.exists()
    assert _sha256(retained_path) == retained_hash

    updated = read_dataset_manifest(manifest_path)
    assert updated.dataset_id == DATASET_ID
    assert updated.row_count == 2
    assert len(updated.shards) == 2
    assert updated.shards[1].path == shard_paths[1].relative_to(root).as_posix()
    assert resolve_manifest_path(updated.shards[0].path, root=root) == new_path
    assert read_json(active_path)["datasets"][DOMAIN] == DATASET_ID
    records = _log_records(workspace)
    assert len(records) == 1
    assert records[0]["action"] == "replace_active_shards"
    assert records[0]["dataset_id"] == DATASET_ID


def test_append_is_content_idempotent(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    active_path, manifest_path, _ = _install_active(workspace)
    active_before = active_path.read_bytes()
    addition = _frame("2026-01-07", "000003.SZ", close=30.0)

    first = append_active_shard(
        DOMAIN,
        addition,
        "fill one missing day",
        workspace_root=workspace,
    )
    second = append_active_shard(
        DOMAIN,
        addition.copy(),
        "same repair resumed",
        workspace_root=workspace,
    )

    assert first["status"] == "appended"
    assert second["status"] == "reused"
    assert first["shard_path"] == second["shard_path"]
    assert active_path.read_bytes() == active_before
    updated = read_dataset_manifest(manifest_path)
    assert updated.dataset_id == DATASET_ID
    assert updated.row_count == 3
    assert len(updated.shards) == 3
    assert len(_log_records(workspace)) == 1


def test_commit_failure_preserves_manifest_and_old_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    active_path, manifest_path, shard_paths = _install_active(workspace)
    active_before = active_path.read_bytes()
    manifest_before = manifest_path.read_bytes()
    old_hash = _sha256(shard_paths[0])

    import quant_data_platform.qdp_v2.repair as repair

    def fail_before_commit(*args: object, **kwargs: object) -> None:
        raise RuntimeError("unit commit failure")

    monkeypatch.setattr(repair, "_commit_manifest", fail_before_commit)
    with pytest.raises(RuntimeError, match="unit commit failure"):
        replace_active_shards(
            DOMAIN,
            _frame("2026-01-05", "000001.SZ", close=11.0),
            [shard_paths[0]],
            "failure injection",
            workspace_root=workspace,
        )

    assert active_path.read_bytes() == active_before
    assert manifest_path.read_bytes() == manifest_before
    assert shard_paths[0].exists()
    assert _sha256(shard_paths[0]) == old_hash
    assert not list(shard_paths[0].parent.glob("repair_replace_*.parquet"))
    assert _log_records(workspace) == []


@pytest.mark.parametrize("failure", ["schema", "duplicate", "cross_shard"])
def test_invalid_repairs_are_rejected_without_commit(
    tmp_path: Path,
    failure: str,
) -> None:
    workspace = _workspace(tmp_path)
    active_path, manifest_path, shard_paths = _install_active(workspace)
    active_before = active_path.read_bytes()
    manifest_before = manifest_path.read_bytes()

    if failure == "schema":
        invalid = _frame("2026-01-07", "000003.SZ", close=30.0).drop(
            columns=["amount"]
        )
        message = "schema_columns_mismatch"
    elif failure == "duplicate":
        row = _frame("2026-01-07", "000003.SZ", close=30.0)
        invalid = pd.concat([row, row], ignore_index=True)
        message = "primary_key_duplicate"
    else:
        invalid = _frame("2026-01-06", "000002.SZ", close=21.0)
        message = "retained_shard_primary_key_overlap"

    with pytest.raises(QdpV2RepairError, match=message):
        append_active_shard(
            DOMAIN,
            invalid,
            f"reject {failure}",
            workspace_root=workspace,
        )

    assert active_path.read_bytes() == active_before
    assert manifest_path.read_bytes() == manifest_before
    assert all(path.exists() for path in shard_paths)
    assert not list(shard_paths[0].parent.glob("repair_append_*.parquet"))
    assert _log_records(workspace) == []


def test_single_parquet_path_replace_does_not_load_pandas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    active_path, manifest_path, shard_paths = _install_active(workspace)
    active_before = active_path.read_bytes()
    retained_hash = _sha256(shard_paths[1])
    prepared = _prepared(
        workspace,
        "replacement",
        _frame("2026-01-05", "000001.SZ", close=12.0),
    )

    def pandas_read_forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("prepared Parquet must not be loaded into pandas")

    monkeypatch.setattr(pd, "read_parquet", pandas_read_forbidden)
    result = replace_active_shard_from_parquet(
        DOMAIN,
        prepared,
        shard_paths[0],
        "replace from prepared parquet",
        workspace_root=workspace,
    )

    new_path = Path(result["new_shard_paths"][0])
    assert result["status"] == "replaced"
    assert result["dataset_id"] == DATASET_ID
    assert active_path.read_bytes() == active_before
    assert manifest_path.exists()
    assert not shard_paths[0].exists()
    assert new_path.parent == shard_paths[0].parent
    assert _sha256(new_path) == _sha256(prepared)
    assert _sha256(shard_paths[1]) == retained_hash
    assert prepared.exists()
    assert repair_log_path(workspace).parent == (
        workspace / "quant_data_platform" / "data" / "qdp_runtime"
    ).resolve()


def test_batch_parquet_path_replace_commits_all_selected_shards(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    active_path, manifest_path, shard_paths = _install_active(workspace)
    active_before = active_path.read_bytes()
    first = _prepared(
        workspace,
        "replace_first",
        _frame("2026-01-05", "000001.SZ", close=13.0),
    )
    second = _prepared(
        workspace,
        "replace_second",
        _frame("2026-01-06", "000002.SZ", close=23.0),
    )

    result = replace_active_shards_from_parquet(
        DOMAIN,
        [(shard_paths[0], first), (shard_paths[1], second)],
        "replace two prepared shards",
        workspace_root=workspace,
    )

    assert result["status"] == "replaced"
    assert result["replacement_count"] == 2
    assert result["row_count"] == 2
    assert active_path.read_bytes() == active_before
    assert all(not path.exists() for path in shard_paths)
    new_paths = [Path(item) for item in result["new_shard_paths"]]
    assert [_sha256(path) for path in new_paths] == [
        _sha256(first),
        _sha256(second),
    ]
    updated = read_dataset_manifest(manifest_path)
    assert updated.dataset_id == DATASET_ID
    assert updated.row_count == 2
    assert len(updated.shards) == 2


def test_bulk_append_validates_primary_keys_once_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    active_path, manifest_path, _ = _install_active(workspace)
    active_before = active_path.read_bytes()
    first = _prepared(
        workspace,
        "append_first",
        _frame("2026-01-07", "000003.SZ", close=30.0),
    )
    second = _prepared(
        workspace,
        "append_second",
        _frame("2026-01-08", "000004.SZ", close=40.0),
    )
    import quant_data_platform.qdp_v2.repair as repair

    original_validate = repair._validate_parquet_batch_primary_keys
    validation_calls = 0

    def counted_validation(*args: object, **kwargs: object) -> None:
        nonlocal validation_calls
        validation_calls += 1
        original_validate(*args, **kwargs)

    monkeypatch.setattr(
        repair,
        "_validate_parquet_batch_primary_keys",
        counted_validation,
    )
    first_result = bulk_append_active_shards_from_parquet(
        DOMAIN,
        [first, second],
        "append two prepared shards",
        workspace_root=workspace,
    )
    second_result = bulk_append_active_shards_from_parquet(
        DOMAIN,
        [first, second],
        "resume same prepared shards",
        workspace_root=workspace,
    )

    assert first_result["status"] == "appended"
    assert first_result["appended_count"] == 2
    assert first_result["reused_count"] == 0
    assert second_result["status"] == "reused"
    assert second_result["appended_count"] == 0
    assert second_result["reused_count"] == 2
    assert validation_calls == 1
    assert active_path.read_bytes() == active_before
    updated = read_dataset_manifest(manifest_path)
    assert updated.dataset_id == DATASET_ID
    assert updated.row_count == 4
    assert len(updated.shards) == 4
    assert (qdp_v2_root(workspace) / "tmp").is_dir()
    assert len(_log_records(workspace)) == 1


@pytest.mark.parametrize("failure", ["between_new", "against_active"])
def test_bulk_append_rejects_primary_key_overlap_and_cleans_targets(
    tmp_path: Path,
    failure: str,
) -> None:
    workspace = _workspace(tmp_path)
    active_path, manifest_path, shard_paths = _install_active(workspace)
    active_before = active_path.read_bytes()
    manifest_before = manifest_path.read_bytes()
    if failure == "between_new":
        paths = [
            _prepared(
                workspace,
                "duplicate_a",
                _frame("2026-01-07", "000003.SZ", close=30.0),
            ),
            _prepared(
                workspace,
                "duplicate_b",
                _frame("2026-01-07", "000003.SZ", close=31.0),
            ),
        ]
        message = "new_shards_primary_key_overlap"
    else:
        paths = [
            _prepared(
                workspace,
                "active_overlap",
                _frame("2026-01-06", "000002.SZ", close=21.0),
            )
        ]
        message = "retained_shard_primary_key_overlap"

    with pytest.raises(QdpV2RepairError, match=message):
        bulk_append_active_shards_from_parquet(
            DOMAIN,
            paths,
            f"reject {failure}",
            workspace_root=workspace,
        )

    assert active_path.read_bytes() == active_before
    assert manifest_path.read_bytes() == manifest_before
    assert all(path.exists() for path in shard_paths)
    assert not list(shard_paths[0].parent.glob("repair_append_*.parquet"))
    assert all(path.exists() for path in paths)
    assert _log_records(workspace) == []


def test_prepared_parquet_commit_failure_rolls_back_new_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    active_path, manifest_path, shard_paths = _install_active(workspace)
    active_before = active_path.read_bytes()
    manifest_before = manifest_path.read_bytes()
    old_hash = _sha256(shard_paths[0])
    prepared = _prepared(
        workspace,
        "commit_failure",
        _frame("2026-01-05", "000001.SZ", close=15.0),
    )
    import quant_data_platform.qdp_v2.repair as repair

    def fail_before_commit(*args: object, **kwargs: object) -> None:
        raise RuntimeError("prepared commit failure")

    monkeypatch.setattr(repair, "_commit_manifest", fail_before_commit)
    with pytest.raises(RuntimeError, match="prepared commit failure"):
        replace_active_shard_from_parquet(
            DOMAIN,
            prepared,
            shard_paths[0],
            "prepared failure injection",
            workspace_root=workspace,
        )

    assert active_path.read_bytes() == active_before
    assert manifest_path.read_bytes() == manifest_before
    assert shard_paths[0].exists()
    assert _sha256(shard_paths[0]) == old_hash
    assert not list(shard_paths[0].parent.glob("repair_replace_*.parquet"))
    assert prepared.exists()
    assert _log_records(workspace) == []


def test_bulk_append_commit_failure_rolls_back_every_new_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    active_path, manifest_path, shard_paths = _install_active(workspace)
    active_before = active_path.read_bytes()
    manifest_before = manifest_path.read_bytes()
    prepared = [
        _prepared(
            workspace,
            "bulk_failure_a",
            _frame("2026-01-07", "000003.SZ", close=30.0),
        ),
        _prepared(
            workspace,
            "bulk_failure_b",
            _frame("2026-01-08", "000004.SZ", close=40.0),
        ),
    ]
    import quant_data_platform.qdp_v2.repair as repair

    def fail_before_commit(*args: object, **kwargs: object) -> None:
        raise RuntimeError("bulk commit failure")

    monkeypatch.setattr(repair, "_commit_manifest", fail_before_commit)
    with pytest.raises(RuntimeError, match="bulk commit failure"):
        bulk_append_active_shards_from_parquet(
            DOMAIN,
            prepared,
            "bulk failure injection",
            workspace_root=workspace,
        )

    assert active_path.read_bytes() == active_before
    assert manifest_path.read_bytes() == manifest_before
    assert all(path.exists() for path in shard_paths)
    assert not list(shard_paths[0].parent.glob("repair_append_*.parquet"))
    assert all(path.exists() for path in prepared)
    assert _log_records(workspace) == []


def test_atomic_parquet_mutation_replaces_removes_and_appends_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    active_path, manifest_path, shard_paths = _install_active(workspace)
    active_before = active_path.read_bytes()
    replacement = _prepared(
        workspace,
        "atomic_replacement",
        _frame("2026-01-05", "000001.SZ", close=16.0),
    )
    addition = _prepared(
        workspace,
        "atomic_append",
        _frame("2026-01-07", "000003.SZ", close=30.0),
    )

    import quant_data_platform.qdp_v2.repair as repair

    original_commit = repair._commit_manifest
    commit_calls = 0

    def counted_commit(*args: object, **kwargs: object) -> None:
        nonlocal commit_calls
        commit_calls += 1
        # Superseded shards must still exist at the sole manifest CAS point.
        assert all(path.exists() for path in shard_paths)
        original_commit(*args, **kwargs)

    monkeypatch.setattr(repair, "_commit_manifest", counted_commit)
    result = mutate_active_shards_from_parquet(
        DOMAIN,
        replacements=[(shard_paths[0], replacement)],
        removals=[shard_paths[1]],
        appends=[addition],
        reason="one atomic overlay",
        workspace_root=workspace,
    )

    assert result["status"] == "mutated"
    assert result["dataset_id"] == DATASET_ID
    assert result["replacement_count"] == 1
    assert result["removal_count"] == 1
    assert result["appended_count"] == 1
    assert result["manifest_row_count"] == 2
    assert commit_calls == 1
    assert active_path.read_bytes() == active_before
    assert all(not path.exists() for path in shard_paths)
    updated = read_dataset_manifest(manifest_path)
    assert updated.dataset_id == DATASET_ID
    assert updated.row_count == 2
    assert len(updated.shards) == 2
    frames = [
        pd.read_parquet(resolve_manifest_path(item.path, root=qdp_v2_root(workspace)))
        for item in updated.shards
    ]
    combined = pd.concat(frames, ignore_index=True).sort_values("trade_date")
    assert combined["close"].tolist() == [16.0, 30.0]

    manifest_after = manifest_path.read_bytes()
    replay = mutate_active_shards_from_parquet(
        DOMAIN,
        replacements=[(shard_paths[0], replacement)],
        removals=[shard_paths[1]],
        appends=[addition],
        reason="replay one atomic overlay",
        workspace_root=workspace,
    )
    assert replay["status"] == "reused"
    assert replay["mutation_id"] == result["mutation_id"]
    assert commit_calls == 1
    assert manifest_path.read_bytes() == manifest_after
    assert active_path.read_bytes() == active_before
    assert len(_log_records(workspace)) == 1


def test_atomic_parquet_mutation_commit_failure_keeps_manifest_and_old_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    active_path, manifest_path, shard_paths = _install_active(workspace)
    active_before = active_path.read_bytes()
    manifest_before = manifest_path.read_bytes()
    replacement = _prepared(
        workspace,
        "atomic_commit_failure",
        _frame("2026-01-05", "000001.SZ", close=17.0),
    )

    import quant_data_platform.qdp_v2.repair as repair

    def fail_before_commit(*args: object, **kwargs: object) -> None:
        assert all(path.exists() for path in shard_paths)
        raise RuntimeError("atomic mutation commit failure")

    monkeypatch.setattr(repair, "_commit_manifest", fail_before_commit)
    with pytest.raises(RuntimeError, match="atomic mutation commit failure"):
        mutate_active_shards_from_parquet(
            DOMAIN,
            replacements=[(shard_paths[0], replacement)],
            removals=[shard_paths[1]],
            reason="failure before atomic CAS",
            workspace_root=workspace,
        )

    assert active_path.read_bytes() == active_before
    assert manifest_path.read_bytes() == manifest_before
    assert all(path.exists() for path in shard_paths)
    assert not list(shard_paths[0].parent.glob("repair_mutate_*.parquet"))
    assert _log_records(workspace) == []


def test_atomic_parquet_mutation_rejects_target_hash_collision(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    active_path, manifest_path, shard_paths = _install_active(workspace)
    active_before = active_path.read_bytes()
    manifest_before = manifest_path.read_bytes()
    addition = _prepared(
        workspace,
        "atomic_hash_collision",
        _frame("2026-01-07", "000003.SZ", close=30.0),
    )
    digest = _sha256(addition)
    collision = (
        shard_paths[0].parent
        / f"repair_mutate_append_{digest[:24]}.parquet"
    )
    collision.write_bytes(b"not the prepared parquet")

    with pytest.raises(QdpV2RepairError, match="existing_target_hash_mismatch"):
        mutate_active_shards_from_parquet(
            DOMAIN,
            appends=[addition],
            reason="reject content address collision",
            workspace_root=workspace,
        )

    assert active_path.read_bytes() == active_before
    assert manifest_path.read_bytes() == manifest_before
    assert all(path.exists() for path in shard_paths)
    assert collision.read_bytes() == b"not the prepared parquet"
    assert _log_records(workspace) == []
