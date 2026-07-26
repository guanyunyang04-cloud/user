from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from quant_data_platform.qdp_v2 import repair as repair_module
from quant_data_platform.qdp_v2.check import run_check
from quant_data_platform.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    _manifest_schema_from_arrow,
    qdp_v2_root,
    read_dataset_manifest,
    resolve_manifest_path,
    schema_hash,
    write_active_manifest,
    write_dataset_manifest,
)
from quant_data_platform.qdp_v2.repair import (
    QdpV2RepairError,
    append_active_shard,
    bulk_append_active_shards_from_parquet,
    mutate_active_shards_from_parquet,
    replace_active_table_from_parquet,
    repair_log_path,
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


def _patch_request(
    workspace: Path,
    manifest_sha256: str,
    changes: list[dict[str, object]],
) -> Path:
    path = workspace / "requests" / "patch.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "domain": DOMAIN,
                "expected_manifest_sha256": manifest_sha256,
                "changes": changes,
            }
        ),
        encoding="utf-8",
    )
    return path


def _workspace_snapshot(workspace: Path) -> dict[str, bytes | None]:
    return {
        path.relative_to(workspace).as_posix(): path.read_bytes() if path.is_file() else None
        for path in sorted(workspace.rglob("*"))
    }


def test_mutate_supports_dimension_without_date_column(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    root = qdp_v2_root(workspace)
    domain = "security_identity"
    dataset_id = "security_identity__dimension_repair_fixture"
    frame = pd.DataFrame(
        {
            "security_id": ["QDP-1", "QDP-2"],
            "current_symbol": ["000001.SZ", "000002.SZ"],
        }
    )
    shard = root / "datasets" / domain / dataset_id / "shards" / "part.parquet"
    shard.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(shard, index=False, engine="pyarrow")
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        domain=domain,
        layer="canonical",
        frequency="static",
        contract_version="unit",
        primary_key=["security_id"],
        start_date="",
        end_date="",
        row_count=2,
        schema_hash="unit-schema",
        shards=[
            ShardManifestEntry(
                path=shard.relative_to(root).as_posix(),
                row_count=2,
                start_date="",
                end_date="",
                file_size=shard.stat().st_size,
                schema_hash="unit-schema",
                content_key="fixture",
            )
        ],
        source={"provider": "unit"},
        quality={"primary_key_unique": True},
        schema=[
            {"name": "security_id", "type": "object"},
            {"name": "current_symbol", "type": "object"},
        ],
    )
    write_dataset_manifest(root, manifest)
    write_active_manifest(
        root,
        {
            "version": 2,
            "active_as_of_date": "2026-01-06",
            "datasets": {domain: dataset_id},
        },
    )
    prepared = _prepared(workspace, "identity", frame.iloc[[0]].copy())

    result = mutate_active_shards_from_parquet(
        domain,
        replacements=[(shard, prepared)],
        reason="unit dimension repair",
        workspace_root=workspace,
    )

    repaired = resolve_active_domain(domain, workspace_root=workspace)
    assert result["status"] == "mutated"
    assert repaired.manifest.row_count == 1
    assert repaired.manifest.start_date == ""
    assert repaired.manifest.end_date == ""


def test_append_supports_dimension_without_date_column(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    root = qdp_v2_root(workspace)
    domain = "security_identity"
    dataset_id = "security_identity__dimension_append_fixture"
    frame = pd.DataFrame(
        {
            "security_id": ["QDP-1"],
            "current_symbol": ["000001.SZ"],
        }
    )
    shard = root / "datasets" / domain / dataset_id / "shards" / "part.parquet"
    shard.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(shard, index=False, engine="pyarrow")
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        domain=domain,
        layer="canonical",
        frequency="static",
        contract_version="unit",
        primary_key=["security_id"],
        start_date="",
        end_date="",
        row_count=1,
        schema_hash="unit-schema",
        shards=[
            ShardManifestEntry(
                path=shard.relative_to(root).as_posix(),
                row_count=1,
                start_date="",
                end_date="",
                file_size=shard.stat().st_size,
                schema_hash="unit-schema",
                content_key="fixture",
            )
        ],
        source={"provider": "unit"},
        quality={"primary_key_unique": True},
        schema=[
            {"name": "security_id", "type": "object"},
            {"name": "current_symbol", "type": "object"},
        ],
    )
    write_dataset_manifest(root, manifest)
    active_path = write_active_manifest(
        root,
        {
            "version": 2,
            "active_as_of_date": "2026-01-06",
            "datasets": {domain: dataset_id},
        },
    )
    active_before = active_path.read_bytes()

    result = append_active_shard(
        domain,
        pd.DataFrame(
            {
                "security_id": ["QDP-2"],
                "current_symbol": ["000002.SZ"],
            }
        ),
        "append one identity",
        workspace_root=workspace,
    )

    repaired = resolve_active_domain(domain, workspace_root=workspace)
    assert result["status"] == "appended"
    assert repaired.manifest.row_count == 2
    assert repaired.manifest.start_date == ""
    assert repaired.manifest.end_date == ""
    assert repaired.manifest.shards[-1].start_date == ""
    assert repaired.manifest.shards[-1].end_date == ""
    assert active_path.read_bytes() == active_before


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


def test_append_to_composite_installs_under_current_dataset(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _, _, original_paths = _install_active(workspace)
    root = qdp_v2_root(workspace)
    composite_id = "market_daily_raw__composite_fixture"
    original = resolve_active_domain(DOMAIN, workspace_root=workspace).manifest
    composite = replace(original, dataset_id=composite_id)
    write_dataset_manifest(root, composite)
    write_active_manifest(
        root,
        {
            "version": 2,
            "active_as_of_date": "2026-01-06",
            "datasets": {DOMAIN: composite_id},
        },
    )

    result = append_active_shard(
        DOMAIN,
        _frame("2026-01-07", "000003.SZ", close=30.0),
        "append to composite",
        workspace_root=workspace,
    )

    appended = Path(result["shard_path"])
    assert result["status"] == "appended"
    assert appended.parent == (
        root / "datasets" / DOMAIN / composite_id / "shards"
    ).resolve()
    assert all(path.exists() for path in original_paths)


def test_mutate_composite_replacement_is_copy_on_write(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _, _, original_paths = _install_active(workspace)
    root = qdp_v2_root(workspace)
    composite_id = "market_daily_raw__composite_mutation_fixture"
    original = resolve_active_domain(DOMAIN, workspace_root=workspace).manifest
    write_dataset_manifest(root, replace(original, dataset_id=composite_id))
    write_active_manifest(
        root,
        {
            "version": 2,
            "active_as_of_date": "2026-01-06",
            "datasets": {DOMAIN: composite_id},
        },
    )
    prepared = _prepared(
        workspace,
        "composite_replacement",
        _frame("2026-01-05", "000001.SZ", close=11.0),
    )

    result = mutate_active_shards_from_parquet(
        DOMAIN,
        replacements=[(original_paths[0], prepared)],
        reason="copy-on-write composite replacement",
        workspace_root=workspace,
    )

    replacement_path = Path(result["new_shard_paths"][0])
    assert result["status"] == "mutated"
    assert replacement_path.parent == (
        root / "datasets" / DOMAIN / composite_id / "shards"
    ).resolve()
    assert original_paths[0].exists()
    assert str(original_paths[0].resolve()) in result["retained_old_shard_paths"]
    current = resolve_active_domain(DOMAIN, workspace_root=workspace)
    assert original_paths[0].resolve() not in current.shard_paths
    assert replacement_path.resolve() in current.shard_paths


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


def test_atomic_replace_retries_transient_windows_file_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.tmp"
    target = tmp_path / "target.parquet"
    source.write_bytes(b"parquet payload")
    original_replace = Path.replace
    attempts = 0

    def flaky_replace(path: Path, replacement: Path) -> Path:
        nonlocal attempts
        if path == source and attempts < 2:
            attempts += 1
            raise PermissionError("transient file lock")
        return original_replace(path, replacement)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    repair_module._replace_file_with_retry(source, target, timeout_seconds=1.0)

    assert attempts == 2
    assert target.read_bytes() == b"parquet payload"


def test_patch_dry_run_does_not_change_workspace(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _, manifest_path, _ = _install_active(workspace)
    digest = _sha256(manifest_path)
    request = _patch_request(
        workspace,
        digest,
        [
            {
                "key": {"trade_date": "2026-01-05", "symbol": "000001.SZ"},
                "column": "close",
                "expected": 10.0,
                "value": 10.5,
            }
        ],
    )
    before = _workspace_snapshot(workspace)

    result = repair_module.patch_active_cells(
        request,
        reason="preview one correction",
        expected_manifest_sha256=digest,
        workspace_root=workspace,
    )

    assert result["status"] == "would_patch"
    assert result["row_count_touched"] == 1
    assert result["schema_hash"] == "unit-schema"
    assert _workspace_snapshot(workspace) == before


def test_patch_applies_exact_cell_and_records_receipt(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _, manifest_path, _ = _install_active(workspace)
    digest = _sha256(manifest_path)
    request = _patch_request(
        workspace,
        digest,
        [
            {
                "key": {"trade_date": "2026-01-05", "symbol": "000001.SZ"},
                "column": "close",
                "expected": 10.0,
                "value": 10.5,
            }
        ],
    )

    result = repair_module.patch_active_cells(
        request,
        reason="correct one close",
        expected_manifest_sha256=digest,
        workspace_root=workspace,
        apply=True,
    )

    current = resolve_active_domain(DOMAIN, workspace_root=workspace)
    combined = pd.concat(
        [pd.read_parquet(path) for path in current.shard_paths], ignore_index=True
    )
    value = combined.loc[combined["symbol"] == "000001.SZ", "close"].item()
    receipt = current.manifest.source["repair_shard_mutations"][result["mutation_id"]]
    assert value == 10.5
    assert receipt == {
        "reason": "correct one close",
        "previous_manifest_sha256": digest,
        "affected_date_range": {"start": "2026-01-05", "end": "2026-01-05"},
        "changed_columns": ["close"],
        "old_schema_hash": "unit-schema",
        "new_schema_hash": "unit-schema",
        "utc_timestamp": receipt["utc_timestamp"],
        "old_shard_sha256": receipt["old_shard_sha256"],
    }
    assert receipt["utc_timestamp"].endswith("+00:00")
    assert len(receipt["old_shard_sha256"]) == 1


def test_patch_rejects_wrong_expected_value(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _, manifest_path, _ = _install_active(workspace)
    digest = _sha256(manifest_path)
    request = _patch_request(
        workspace,
        digest,
        [
            {
                "key": {"trade_date": "2026-01-05", "symbol": "000001.SZ"},
                "column": "close",
                "expected": 99.0,
                "value": 10.5,
            }
        ],
    )

    with pytest.raises(QdpV2RepairError, match="expected_mismatch.*expected=99.0:actual=10.0"):
        repair_module.patch_active_cells(
            request,
            reason="reject stale expected value",
            expected_manifest_sha256=digest,
            workspace_root=workspace,
        )


@pytest.mark.parametrize("match_count", [0, 2])
def test_patch_requires_exactly_one_matching_row(
    tmp_path: Path,
    match_count: int,
) -> None:
    workspace = _workspace(tmp_path)
    _, manifest_path, shard_paths = _install_active(workspace)
    if match_count == 2:
        duplicate = _frame("2026-01-05", "000001.SZ", close=20.0)
        duplicate.to_parquet(shard_paths[1], index=False, engine="pyarrow")
        key = {"trade_date": "2026-01-05", "symbol": "000001.SZ"}
    else:
        key = {"trade_date": "2026-01-09", "symbol": "999999.SZ"}
    digest = _sha256(manifest_path)
    request = _patch_request(
        workspace,
        digest,
        [{"key": key, "column": "close", "expected": 10.0, "value": 10.5}],
    )

    with pytest.raises(QdpV2RepairError, match=f"matches={match_count}"):
        repair_module.patch_active_cells(
            request,
            reason="enforce unique patch target",
            expected_manifest_sha256=digest,
            workspace_root=workspace,
        )


def test_patch_forbids_primary_key_edit(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _, manifest_path, _ = _install_active(workspace)
    digest = _sha256(manifest_path)
    request = _patch_request(
        workspace,
        digest,
        [
            {
                "key": {"trade_date": "2026-01-05", "symbol": "000001.SZ"},
                "column": "trade_date",
                "expected": "2026-01-05",
                "value": "2026-01-06",
            }
        ],
    )

    with pytest.raises(QdpV2RepairError, match="primary_key_edit_forbidden"):
        repair_module.patch_active_cells(
            request,
            reason="reject key edit",
            expected_manifest_sha256=digest,
            workspace_root=workspace,
        )


def test_patch_cli_refuses_stale_manifest_cas(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _, manifest_path, _ = _install_active(workspace)
    digest = _sha256(manifest_path)
    request = _patch_request(workspace, digest, [])
    payload = json.loads(request.read_text(encoding="utf-8"))
    payload["changes"] = [
        {
            "key": {"trade_date": "2026-01-05", "symbol": "000001.SZ"},
            "column": "close",
            "expected": 10.0,
            "value": 10.5,
        }
    ]
    request.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(repair_module.QdpV2RepairConflictError, match="expected_manifest_sha256_mismatch"):
        repair_module.main(
            [
                "patch",
                "--request",
                str(request),
                "--reason",
                "reject stale manifest",
                "--expected-manifest-sha256",
                "0" * 64,
                "--workspace-root",
                str(workspace),
            ]
        )


def test_mutation_replay_preserves_receipt_bytes(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _, manifest_path, shard_paths = _install_active(workspace)
    replacement = _prepared(
        workspace,
        "receipt_replacement",
        _frame("2026-01-05", "000001.SZ", close=12.0),
    )
    first = mutate_active_shards_from_parquet(
        DOMAIN,
        replacements=[(shard_paths[0], replacement)],
        reason="receipt replay",
        workspace_root=workspace,
        changed_columns=["close"],
    )
    manifest_after = manifest_path.read_bytes()
    receipt_before = resolve_active_domain(
        DOMAIN, workspace_root=workspace
    ).manifest.source["repair_shard_mutations"][first["mutation_id"]]

    current_digest = _sha256(manifest_path)
    assert repair_module.main(
        [
            "mutate",
            "--domain",
            DOMAIN,
            "--replace",
            f"{shard_paths[0]}={replacement}",
            "--reason",
            "different replay reason",
            "--expected-manifest-sha256",
            current_digest,
            "--expected-mutation-id",
            first["mutation_id"],
            "--workspace-root",
            str(workspace),
            "--apply",
        ]
    ) == 0
    receipt_after = resolve_active_domain(
        DOMAIN, workspace_root=workspace
    ).manifest.source["repair_shard_mutations"][first["mutation_id"]]

    assert manifest_path.read_bytes() == manifest_after
    assert receipt_after == receipt_before
    assert len(resolve_active_domain(DOMAIN, workspace_root=workspace).manifest.source["repair_shard_mutations"]) == 1


def test_replace_table_schema_change_requires_cli_contract_guards(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _, manifest_path, _ = _install_active(workspace)
    digest = _sha256(manifest_path)
    changed = _frame("2026-01-05", "000001.SZ", close=10.0).assign(extra=["x"])
    prepared = _prepared(workspace, "schema_change", changed)
    new_schema = _manifest_schema_from_arrow(pq.read_schema(prepared))
    new_hash = schema_hash(new_schema)
    base = [
        "replace-table",
        "--domain",
        DOMAIN,
        "--input",
        str(prepared),
        "--reason",
        "add one declared field",
        "--expected-manifest-sha256",
        digest,
        "--workspace-root",
        str(workspace),
        "--apply",
    ]

    with pytest.raises(ValueError, match="schema_hash_invalid"):
        repair_module.main(base)
    with pytest.raises(ValueError, match="contract_version_required"):
        repair_module.main([*base, "--expected-new-schema-hash", new_hash])
    assert repair_module.main(
        [
            *base,
            "--expected-new-schema-hash",
            new_hash,
            "--contract-version",
            "qdp_v2_market_daily_raw_v2",
        ]
    ) == 0
    assert read_dataset_manifest(manifest_path).schema_hash == new_hash


@pytest.mark.parametrize("drift", ["type", "order"])
def test_quick_check_detects_manifest_footer_schema_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    workspace = _workspace(tmp_path)
    _, manifest_path, shard_paths = _install_active(workspace)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = _manifest_schema_from_arrow(pq.read_schema(shard_paths[0]))
    if drift == "type":
        declared[0]["type"] = "DOUBLE"
    else:
        declared = [declared[1], declared[0], *declared[2:]]
    payload["schema"] = declared
    payload["schema_hash"] = schema_hash(declared)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    result = run_check(workspace_root=workspace, full=False)
    errors = result["active"]["errors"]

    # One drift yields one diagnostic: a reordered declaration is reported as a
    # column-order mismatch, and a type-only difference as a footer mismatch.
    if drift == "order":
        assert any("schema_column_order_mismatch" in item for item in errors)
        assert not any("footer_schema_mismatch" in item for item in errors)
    else:
        assert any("footer_schema_mismatch" in item for item in errors)
        assert not any("schema_column_order_mismatch" in item for item in errors)


def test_replace_active_table_accepts_single_and_multiple_paths(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _, _, _ = _install_active(workspace)
    single = _prepared(
        workspace,
        "single_replace",
        _frame("2026-01-07", "000003.SZ", close=30.0),
    )

    first = replace_active_table_from_parquet(
        DOMAIN,
        single,
        reason="single path remains valid",
        workspace_root=workspace,
    )
    left = _prepared(
        workspace,
        "multi_left",
        _frame("2026-01-08", "000004.SZ", close=40.0),
    )
    right = _prepared(
        workspace,
        "multi_right",
        _frame("2026-01-09", "000005.SZ", close=50.0),
    )
    second = replace_active_table_from_parquet(
        DOMAIN,
        [left, right],
        reason="preserve two prepared partitions",
        workspace_root=workspace,
    )

    current = resolve_active_domain(DOMAIN, workspace_root=workspace)
    assert first["status"] == "replaced"
    assert len(first["new_shard_paths"]) == 1
    assert second["status"] == "replaced"
    assert len(second["new_shard_paths"]) == 2
    assert len(current.manifest.shards) == 2
    assert current.manifest.row_count == 2
