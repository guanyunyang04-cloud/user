from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import quant_data_platform.qdp_v3.compaction as compaction_module
import quant_data_platform.qdp_v3.storage as storage_module
from quant_data_platform.qdp_v3.bundles import BundleSizingPolicy, stable_raw_bucket
from quant_data_platform.qdp_v3.compaction import compact_raw_domain
from quant_data_platform.qdp_v3.audit import _audit_raw_references
from quant_data_platform.qdp_v3.build import _provider_evidence
from quant_data_platform.qdp_v3.constants import (
    RAW_TRADING_CALENDAR,
    RAW_TUSHARE_PROXY_SUSPEND,
    RAW_TUSHARE_PROXY_STOCK_BASIC,
    RAW_TUSHARE_PROXY_TRADE_CALENDAR,
)
from quant_data_platform.qdp_v3.historical import (
    ingest_tushare_proxy_reference,
    proxy_symbol_inventory,
)
from quant_data_platform.qdp_v3.ingest import ingest_trading_calendar
from quant_data_platform.qdp_v3.manifest import sha256_file
from quant_data_platform.qdp_v3.paths import ensure_qdp_v3_layout, qdp_v3_paths
from quant_data_platform.qdp_v3.runtime_index import RuntimeIndex
from quant_data_platform.qdp_v3.storage import (
    get_raw_partition,
    get_raw_partition_version,
    iter_raw_partitions,
    read_raw_partition,
    read_raw_receipt,
    write_empty_raw_partition,
    write_raw_partition,
)


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    manifest = workspace / "brain" / "brain_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"schema_version": 1, "brain_type": "main"}), encoding="utf-8")
    return workspace


def _symbols_in_different_buckets() -> tuple[str, str]:
    first = "QDP-CN-SSE-000001"
    for number in range(2, 10_000):
        candidate = f"QDP-CN-SSE-{number:06d}"
        if stable_raw_bucket(candidate) != stable_raw_bucket(first):
            return first, candidate
    raise AssertionError("failed to find deterministic bucket fixture")


def test_qdp_v3_paths_split_data_and_runtime_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = tmp_path / "persistent-data"
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("QDP_DATA_ROOT", str(data_root))
    monkeypatch.setenv("QDP_RUNTIME_ROOT", str(runtime_root))

    paths = ensure_qdp_v3_layout(_workspace(tmp_path))

    assert paths.data_root == data_root.resolve()
    assert paths.root == data_root.resolve() / "qdp_v3"
    assert paths.raw_bundles == paths.root / "raw_bundles"
    assert paths.runtime_root == runtime_root.resolve()
    assert paths.jobs == runtime_root.resolve() / "jobs"
    assert paths.staging == runtime_root.resolve() / "jobs" / "staging"
    assert paths.runtime_index == runtime_root.resolve() / "runtime.sqlite3"


def test_runtime_index_tables_and_secret_rejection(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    index = RuntimeIndex(workspace_root=workspace)
    index.upsert_job(
        job_id="job-1",
        domain="market_intraday_5m",
        provider="tushare_proxy",
        status="running",
        payload={"start_date": "2020-01-01"},
    )
    index.record_page(
        job_id="job-1",
        task_key="600000.SH",
        page_number=0,
        request_range={"start_at": "2020-01-01", "end_at": "2020-12-31"},
        row_count=8000,
        response_sha256="a" * 64,
        status="complete",
    )
    index.record_correction_applied(
        correction_id="fix-1",
        domain="symbol_history",
        record_key={"security_id": "QDP-1"},
        field="symbol",
        old_value="300114.SZ",
        new_value="302132.SZ",
    )

    with sqlite3.connect(index.path) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
    assert {"raw_partitions", "raw_receipts", "jobs", "pages", "corrections_applied"}.issubset(tables)
    assert index.table_frame("jobs").loc[0, "status"] == "running"
    assert index.table_frame("pages").loc[0, "row_count"] == 8000
    assert index.table_frame("corrections_applied").loc[0, "correction_id"] == "fix-1"

    with pytest.raises(ValueError, match="runtime_index_secret_key_rejected"):
        index.upsert_job(
            job_id="unsafe",
            domain="x",
            provider="x",
            status="pending",
            payload={"token": "must-not-be-persisted"},
        )


def test_compactor_bundles_by_year_and_bucket_without_deleting_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    raw_domain = "compact_unit_intraday_5m_raw"
    first_symbol, second_symbol = _symbols_in_different_buckets()
    first = pd.DataFrame(
        {
            "security_id": [first_symbol, second_symbol, first_symbol, second_symbol],
            "trade_date": ["2020-01-02", "2020-01-02", "2021-01-04", "2021-01-04"],
            "close": [10.0, 20.0, 11.0, 21.0],
        }
    )
    second = pd.DataFrame(
        {
            "security_id": [first_symbol] * 20,
            "trade_date": ["2020-01-03"] * 20,
            "close": [12.0 + index / 100 for index in range(20)],
        }
    )
    first_ref, _ = write_raw_partition(
        raw_domain=raw_domain,
        partition_field="query_date",
        partition_value="2020-01-02",
        frame=first,
        receipt={"quality_tier": "strict", "provider": "tushare_proxy", "response_sha256": "1" * 64},
        workspace_root=workspace,
    )
    second_ref, _ = write_raw_partition(
        raw_domain=raw_domain,
        partition_field="query_date",
        partition_value="2020-01-03",
        frame=second,
        receipt={"quality_tier": "strict", "provider": "tushare_proxy", "response_sha256": "2" * 64},
        workspace_root=workspace,
    )
    empty_ref, _ = write_raw_partition(
        raw_domain=raw_domain,
        partition_field="query_date",
        partition_value="2022-01-03",
        frame=first.iloc[0:0].copy(),
        receipt={"quality_tier": "strict", "provider": "tushare_proxy", "response_sha256": "3" * 64},
        workspace_root=workspace,
    )
    source_paths = {ref.payload_path for ref in (first_ref, second_ref, empty_ref)}
    index = RuntimeIndex(workspace_root=workspace)
    export_dir = qdp_v3_paths(workspace).metadata / "raw-index"

    result = compact_raw_domain(
        raw_domain,
        workspace_root=workspace,
        runtime_index=index,
        sizing=BundleSizingPolicy(target_part_bytes=256, max_part_bytes=2 * 1024 * 1024, estimated_compression_ratio=1.0),
        export_index_dir=export_dir,
    )

    assert result.source_partition_count == 3
    assert result.source_row_count == 24
    assert result.indexed_row_count == 24
    assert result.empty_partition_count == 1
    assert result.created_part_count == len(result.parts)
    assert len(result.parts) >= 3
    assert all(part.path.stat().st_size <= 2 * 1024 * 1024 for part in result.parts)
    assert all("year=" in part.relative_path and "security_bucket=" in part.relative_path for part in result.parts)
    assert {part.year for part in result.parts} == {"2020", "2021"}
    assert {part.security_bucket for part in result.parts}.issuperset(
        {stable_raw_bucket(first_symbol), stable_raw_bucket(second_symbol)}
    )


    raw_index = index.table_frame("raw_partitions")
    empty_rows = raw_index.loc[raw_index["status"].eq("empty_success")]
    assert len(empty_rows) == 1
    assert empty_rows.iloc[0]["bundle_path"] == ""
    assert int(empty_rows.iloc[0]["row_group"]) == -1
    assert not any("2022" in part.relative_path for part in result.parts)
    assert {first_ref.content_sha256, second_ref.content_sha256, empty_ref.content_sha256}.issubset(
        set(raw_index["source_content_sha256"])
    )
    assert result.raw_index_export is not None
    assert result.raw_index_export.sha256 == sha256_file(result.raw_index_export.parquet_path)
    assert result.raw_index_export.hash_path.read_text(encoding="ascii").startswith(
        result.raw_index_export.sha256
    )
    assert result.raw_index_export.receipts_sha256 == sha256_file(
        result.raw_index_export.receipts_path
    )
    assert result.raw_index_export.receipt_count == 3
    receipts = index.table_frame("raw_receipts")
    assert len(receipts) == 3
    assert set(receipts["source_content_sha256"]) == {
        first_ref.content_sha256,
        second_ref.content_sha256,
        empty_ref.content_sha256,
    }

    second_result = compact_raw_domain(
        raw_domain,
        workspace_root=workspace,
        runtime_index=index,
        sizing=BundleSizingPolicy(target_part_bytes=256, max_part_bytes=2 * 1024 * 1024, estimated_compression_ratio=1.0),
    )
    assert second_result.created_part_count == 0
    assert second_result.reused_part_count == len(result.parts)
    assert {part.sha256 for part in second_result.parts} == {part.sha256 for part in result.parts}
    assert len(index.table_frame("raw_partitions")) == len(raw_index)
    assert all(path.exists() for path in source_paths)

    with pytest.raises(ValueError, match="requires_yes"):
        from quant_data_platform.qdp_v3.compaction import compact_raw_partitions

        compact_raw_partitions(
            (first_ref,),
            raw_domain=raw_domain,
            workspace_root=workspace,
            runtime_index=index,
            delete_sources=True,
        )

    monkeypatch.setattr(
        compaction_module,
        "read_raw_partition",
        lambda *_args, **_kwargs: pytest.fail(
            "verified delete-source path must not decode legacy frames again"
        ),
    )
    deleted = compact_raw_domain(
        raw_domain,
        workspace_root=workspace,
        runtime_index=index,
        sizing=BundleSizingPolicy(
            target_part_bytes=256,
            max_part_bytes=2 * 1024 * 1024,
            estimated_compression_ratio=1.0,
        ),
        export_index_dir=export_dir,
        delete_sources=True,
        yes=True,
    )
    assert deleted.deleted_source_partition_count == 3
    assert all(not path.exists() for path in source_paths)
    assert all(part.path.exists() for part in deleted.parts)

    bundled_refs = iter_raw_partitions(raw_domain, workspace_root=workspace)
    assert len(bundled_refs) == 3
    assert {ref.storage_kind for ref in bundled_refs} == {"bundle"}
    bundled_by_value = {ref.partition_value: ref for ref in bundled_refs}
    pd.testing.assert_frame_equal(
        read_raw_partition(bundled_by_value["2020-01-02"]),
        first.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        read_raw_partition(bundled_by_value["2020-01-03"]),
        second.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        read_raw_partition(bundled_by_value["2022-01-03"]),
        first.iloc[0:0].reset_index(drop=True),
    )
    direct = get_raw_partition(
        raw_domain,
        partition_field="query_date",
        partition_value="2020-01-02",
        workspace_root=workspace,
    )
    assert direct is not None
    assert read_raw_receipt(direct)["content_sha256"] == first_ref.content_sha256
    pd.testing.assert_frame_equal(read_raw_partition(direct), first.reset_index(drop=True))

    # The immutable exported index is sufficient when the runtime SQLite DB is lost.
    runtime_path = index.path
    runtime_path.unlink()
    runtime_path.with_name(f"{runtime_path.name}-wal").unlink(missing_ok=True)
    runtime_path.with_name(f"{runtime_path.name}-shm").unlink(missing_ok=True)
    fallback_refs = iter_raw_partitions(raw_domain, workspace_root=workspace)
    assert len(fallback_refs) == 3
    fallback_by_value = {ref.partition_value: ref for ref in fallback_refs}
    pd.testing.assert_frame_equal(
        read_raw_partition(fallback_by_value["2020-01-02"]),
        first.reset_index(drop=True),
    )

    tampered_path = fallback_by_value["2020-01-02"].bundle_segments[0].bundle_path
    original_bytes = tampered_path.read_bytes()
    try:
        tampered_path.write_bytes(original_bytes + b"tampered")
        with pytest.raises(RuntimeError, match="raw_bundle_hash_mismatch"):
            read_raw_partition(fallback_by_value["2020-01-02"])
    finally:
        tampered_path.write_bytes(original_bytes)


def test_failed_compaction_restores_last_export_and_removes_new_parts(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    paths = ensure_qdp_v3_layout(workspace)
    export_dir = paths.metadata / "raw_index"
    write_raw_partition(
        raw_domain="safe_export_raw",
        partition_field="query_date",
        partition_value="2020-01-01",
        frame=pd.DataFrame({"trade_date": ["2020-01-01"], "value": [1]}),
        receipt={"provider": "unit", "quality_tier": "strict"},
        workspace_root=workspace,
    )
    compact_raw_domain(
        "safe_export_raw",
        workspace_root=workspace,
        export_index_dir=export_dir,
    )

    write_raw_partition(
        raw_domain="failed_compaction_raw",
        partition_field="query_date",
        partition_value="2020-01-01",
        frame=pd.DataFrame({"trade_date": ["2020-01-01"], "value": [1]}),
        receipt={"provider": "unit", "quality_tier": "strict"},
        workspace_root=workspace,
    )
    bad_ref, _ = write_raw_partition(
        raw_domain="failed_compaction_raw",
        partition_field="query_date",
        partition_value="2020-01-02",
        frame=pd.DataFrame({"trade_date": ["2020-01-02"], "value": [2]}),
        receipt={"provider": "unit", "quality_tier": "strict"},
        workspace_root=workspace,
    )
    pd.DataFrame({"trade_date": ["2020-01-02"], "value": [999]}).to_parquet(
        bad_ref.payload_path,
        index=False,
    )

    with pytest.raises(RuntimeError, match=r"raw_(?:parquet|content)_hash_mismatch"):
        compact_raw_domain(
            "failed_compaction_raw",
            workspace_root=workspace,
            export_index_dir=export_dir,
            sizing=BundleSizingPolicy(target_part_bytes=1, max_part_bytes=1024 * 1024),
        )

    index = RuntimeIndex(workspace_root=workspace)
    assert index.table_frame("raw_partitions").loc[
        lambda frame: frame["domain"].eq("failed_compaction_raw")
    ].empty
    assert index.table_frame("raw_receipts").loc[
        lambda frame: frame["domain"].eq("failed_compaction_raw")
    ].empty
    failed_bundle_root = paths.raw_bundles / "failed_compaction_raw"
    assert not list(failed_bundle_root.rglob("*.parquet"))
    safe = iter_raw_partitions("safe_export_raw", workspace_root=workspace)
    assert len(safe) == 1
    assert read_raw_partition(safe[0]).loc[0, "value"] == 1


def test_compactor_keeps_reference_history_in_one_undated_bundle(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    frame = pd.DataFrame(
        {
            "exchange": ["SSE", "SSE", "SSE"],
            "cal_date": ["2010-01-04", "2020-01-02", "2026-07-13"],
            "is_open": [1, 1, 1],
        }
    )
    write_raw_partition(
        raw_domain=RAW_TUSHARE_PROXY_TRADE_CALENDAR,
        partition_field="request_range",
        partition_value="2010-01-01_2026-07-13",
        frame=frame,
        receipt={"quality_tier": "strict", "provider": "tushare_proxy"},
        workspace_root=workspace,
    )

    result = compact_raw_domain(
        RAW_TUSHARE_PROXY_TRADE_CALENDAR,
        workspace_root=workspace,
        export_index_dir=qdp_v3_paths(workspace).metadata / "raw-index",
    )

    assert result.source_partition_count == 1
    assert len(result.parts) == 1
    assert result.parts[0].year == "undated"
    assert result.parts[0].security_bucket == 0


def test_stock_basic_revisions_share_one_undated_bundle(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    first = pd.DataFrame(
        {
            "ts_code": ["600000.SH", "000001.SZ"],
            "list_date": ["19991110", "19910403"],
        }
    )
    second = pd.concat(
        [
            first,
            pd.DataFrame({"ts_code": ["600001.SH"], "list_date": ["19991201"]}),
        ],
        ignore_index=True,
    )
    first_ref, _ = write_raw_partition(
        raw_domain=RAW_TUSHARE_PROXY_STOCK_BASIC,
        partition_field="as_of_date",
        partition_value="2026-07-13",
        frame=first,
        receipt={"quality_tier": "strict", "provider": "tushare_proxy"},
        workspace_root=workspace,
    )
    second_ref, _ = write_raw_partition(
        raw_domain=RAW_TUSHARE_PROXY_STOCK_BASIC,
        partition_field="as_of_date",
        partition_value="2026-07-13",
        frame=second,
        receipt={"quality_tier": "strict", "provider": "tushare_proxy"},
        workspace_root=workspace,
    )

    result = compact_raw_domain(
        RAW_TUSHARE_PROXY_STOCK_BASIC,
        workspace_root=workspace,
        export_index_dir=qdp_v3_paths(workspace).metadata / "raw-index",
    )

    assert result.source_partition_count == 2
    assert result.source_row_count == len(first) + len(second)
    assert len(result.parts) == 1
    assert result.parts[0].year == "undated"
    assert result.parts[0].security_bucket == 0
    assert result.parts[0].row_group_count == 2
    assert {first_ref.content_sha256, second_ref.content_sha256}.issubset(
        set(RuntimeIndex(workspace_root=workspace).table_frame("raw_receipts")["source_content_sha256"])
    )


def test_compactor_uses_year_only_layout_for_low_frequency_domains(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    raw_domain = "compact_unit_daily_raw"
    first_symbol, second_symbol = _symbols_in_different_buckets()
    frame = pd.DataFrame(
        {
            "security_id": [first_symbol, second_symbol, first_symbol, second_symbol],
            "trade_date": ["2020-01-02", "2020-01-02", "2021-01-04", "2021-01-04"],
            "close": [10.0, 20.0, 11.0, 21.0],
        }
    )
    write_raw_partition(
        raw_domain=raw_domain,
        partition_field="query_date",
        partition_value="2020-01-02",
        frame=frame,
        receipt={"quality_tier": "strict", "provider": "tushare_proxy"},
        workspace_root=workspace,
    )

    result = compact_raw_domain(
        raw_domain,
        workspace_root=workspace,
        export_index_dir=qdp_v3_paths(workspace).metadata / "raw-index",
    )

    assert len(result.parts) == 2
    assert {part.year for part in result.parts} == {"2020", "2021"}
    assert {part.security_bucket for part in result.parts} == {0}


def test_bundle_catalog_indexes_mapping_rows_once_for_many_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    raw_domain = "mapping_index_unit_daily_raw"
    partition_values = [f"2020-01-{day:02d}" for day in range(2, 10)]
    original_ref = None
    for ordinal, partition_value in enumerate(partition_values):
        ref, _ = write_raw_partition(
            raw_domain=raw_domain,
            partition_field="query_date",
            partition_value=partition_value,
            frame=pd.DataFrame(
                {"trade_date": [partition_value], "value": [ordinal]}
            ),
            receipt={"quality_tier": "strict", "provider": "unit"},
            workspace_root=workspace,
        )
        if ordinal == 0:
            original_ref = ref
    revised_ref, _ = write_raw_partition(
        raw_domain=raw_domain,
        partition_field="query_date",
        partition_value=partition_values[0],
        frame=pd.DataFrame(
            {"trade_date": [partition_values[0]], "value": [99]}
        ),
        receipt={"quality_tier": "strict", "provider": "unit"},
        workspace_root=workspace,
    )
    empty_value = "2020-01-20"
    write_raw_partition(
        raw_domain=raw_domain,
        partition_field="query_date",
        partition_value=empty_value,
        frame=pd.DataFrame(
            {
                "trade_date": pd.Series(dtype="object"),
                "value": pd.Series(dtype="int64"),
            }
        ),
        receipt={"quality_tier": "strict", "provider": "unit"},
        workspace_root=workspace,
    )
    compact_raw_domain(
        raw_domain,
        workspace_root=workspace,
        export_index_dir=qdp_v3_paths(workspace).metadata / "raw-index",
        delete_sources=True,
        yes=True,
    )

    original_indexer = storage_module._index_bundle_partition_mappings
    indexed_row_counts: list[int] = []

    def counted_indexer(frame: pd.DataFrame) -> dict[tuple[str, str], tuple[int, ...]]:
        indexed_row_counts.append(len(frame))
        return original_indexer(frame)

    monkeypatch.setattr(
        storage_module,
        "_index_bundle_partition_mappings",
        counted_indexer,
    )
    refs = iter_raw_partitions(raw_domain, workspace_root=workspace)

    assert indexed_row_counts == [10]
    assert len(refs) == 9
    refs_by_value = {ref.partition_value: ref for ref in refs}
    assert refs_by_value[partition_values[0]].content_sha256 == revised_ref.content_sha256
    assert read_raw_partition(refs_by_value[partition_values[0]])["value"].tolist() == [99]
    assert read_raw_partition(refs_by_value[empty_value]).empty

    indexed_row_counts.clear()
    assert original_ref is not None
    exact_original = get_raw_partition_version(
        raw_domain,
        partition_field="query_date",
        partition_value=partition_values[0],
        content_sha256=original_ref.content_sha256,
        workspace_root=workspace,
    )
    assert indexed_row_counts == [2]
    assert exact_original is not None
    assert read_raw_partition(exact_original)["value"].tolist() == [0]


def test_compact_cli_requires_explicit_yes_for_source_deletion(
    tmp_path: Path,
) -> None:
    from quant_data_platform.qdp_v3.cli import dispatch

    workspace = _workspace(tmp_path)
    with pytest.raises(ValueError, match="raw_compaction_source_deletion_requires_yes"):
        dispatch(
            [
                "compact",
                "--workspace-root",
                str(workspace),
                "--raw-domain",
                "unit_raw",
                "--delete-source",
            ]
        )


def test_exact_candidate_raw_revision_survives_legacy_deletion_and_index_fallback(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    first = pd.DataFrame({"trade_date": ["2020-01-02"], "value": [1]})
    second = pd.DataFrame({"trade_date": ["2020-01-02"], "value": [2]})
    first_ref, _ = write_raw_partition(
        raw_domain="revision_unit_raw",
        partition_field="query_date",
        partition_value="2020-01-02",
        frame=first,
        receipt={"quality_tier": "strict", "provider": "unit", "endpoint": "first"},
        workspace_root=workspace,
    )
    candidate_item = first_ref.to_dict(root=qdp_v3_paths(workspace).root)
    second_ref, _ = write_raw_partition(
        raw_domain="revision_unit_raw",
        partition_field="query_date",
        partition_value="2020-01-02",
        frame=second,
        receipt={"quality_tier": "strict", "provider": "unit", "endpoint": "second"},
        workspace_root=workspace,
    )
    export_dir = qdp_v3_paths(workspace).metadata / "raw-index"
    compact_raw_domain(
        "revision_unit_raw",
        workspace_root=workspace,
        export_index_dir=export_dir,
        delete_sources=True,
        yes=True,
    )

    # Prove the immutable export alone can resolve an older, non-latest version.
    runtime_path = qdp_v3_paths(workspace).runtime_index
    runtime_path.unlink()
    runtime_path.with_name(f"{runtime_path.name}-wal").unlink(missing_ok=True)
    runtime_path.with_name(f"{runtime_path.name}-shm").unlink(missing_ok=True)
    exact = get_raw_partition_version(
        "revision_unit_raw",
        partition_field="query_date",
        partition_value="2020-01-02",
        content_sha256=first_ref.content_sha256,
        workspace_root=workspace,
    )
    latest = get_raw_partition(
        "revision_unit_raw",
        partition_field="query_date",
        partition_value="2020-01-02",
        workspace_root=workspace,
    )
    assert exact is not None and exact.storage_kind == "bundle"
    assert latest is not None and latest.content_sha256 == second_ref.content_sha256
    pd.testing.assert_frame_equal(read_raw_partition(exact), first)
    pd.testing.assert_frame_equal(read_raw_partition(latest), second)

    findings, metrics = _audit_raw_references(
        SimpleNamespace(raw_partitions=[candidate_item]),
        root=qdp_v3_paths(workspace).root,
        workspace_root=workspace,
        full=True,
    )
    assert findings == []
    assert metrics["raw_partition_checked_count"] == 1


def test_bundle_backed_callers_reuse_calendar_inventory_and_provider_receipt(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    calendar = pd.DataFrame(
        {
            "trade_date": ["2020-01-02", "2020-01-03"],
            "is_open": [True, True],
            "exchange": ["SSE", "SSE"],
        }
    )
    write_raw_partition(
        raw_domain=RAW_TRADING_CALENDAR,
        partition_field="request_range",
        partition_value="2020-01-01_2020-01-03",
        frame=calendar,
        receipt={
            "quality_tier": "strict",
            "provider": "baostock",
            "endpoint": "query_trade_dates",
        },
        workspace_root=workspace,
    )
    stock_basic = pd.DataFrame(
        {
            "ts_code": ["600000.SH", "000001.SZ"],
            "list_date": ["19991110", "19910403"],
            "delist_date": ["", ""],
        }
    )
    write_raw_partition(
        raw_domain=RAW_TUSHARE_PROXY_STOCK_BASIC,
        partition_field="as_of_date",
        partition_value="2020-01-03",
        frame=stock_basic,
        receipt={
            "quality_tier": "strict",
            "provider": "tushare_proxy",
            "endpoint": "stock_basic",
        },
        workspace_root=workspace,
    )
    export_dir = qdp_v3_paths(workspace).metadata / "raw-index"
    for domain in (RAW_TRADING_CALENDAR, RAW_TUSHARE_PROXY_STOCK_BASIC):
        compact_raw_domain(
            domain,
            workspace_root=workspace,
            export_index_dir=export_dir,
            delete_sources=True,
            yes=True,
        )

    reused_calendar, calendar_ref = ingest_trading_calendar(
        start_date="2020-01-01",
        end_date="2020-01-03",
        workspace_root=workspace,
    )
    pd.testing.assert_frame_equal(reused_calendar, calendar)
    assert calendar_ref.storage_kind == "bundle"
    evidence = _provider_evidence([calendar_ref])
    assert len(evidence) == 1
    assert evidence[0].provider == "baostock"
    assert evidence[0].endpoint == "query_trade_dates"

    symbols, lifecycle = proxy_symbol_inventory(
        workspace_root=workspace,
        mainboard_only=True,
    )
    assert symbols == ["000001.SZ", "600000.SH"]
    assert lifecycle["600000.SH"] == ("1999-11-10", "")


def test_incremental_raw_write_reuses_bundle_and_preserves_revision_chain(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    first = pd.DataFrame({"trade_date": ["2020-01-02"], "value": [1]})
    first_ref, _ = write_raw_partition(
        raw_domain="incremental_unit_raw",
        partition_field="query_date",
        partition_value="2020-01-02",
        frame=first,
        receipt={"quality_tier": "strict", "provider": "unit"},
        workspace_root=workspace,
    )
    export_dir = qdp_v3_paths(workspace).metadata / "raw-index"
    compact_raw_domain(
        "incremental_unit_raw",
        workspace_root=workspace,
        export_index_dir=export_dir,
        delete_sources=True,
        yes=True,
    )

    reused, created = write_raw_partition(
        raw_domain="incremental_unit_raw",
        partition_field="query_date",
        partition_value="2020-01-02",
        frame=first,
        receipt={"quality_tier": "strict", "provider": "unit"},
        workspace_root=workspace,
    )
    assert not created
    assert reused.storage_kind == "bundle"
    assert reused.content_sha256 == first_ref.content_sha256

    second = pd.DataFrame({"trade_date": ["2020-01-02"], "value": [2]})
    second_ref, created = write_raw_partition(
        raw_domain="incremental_unit_raw",
        partition_field="query_date",
        partition_value="2020-01-02",
        frame=second,
        receipt={"quality_tier": "strict", "provider": "unit"},
        workspace_root=workspace,
    )
    assert created
    assert second_ref.storage_kind == "legacy"
    assert second_ref.revision_of == first_ref.content_sha256

    compact_raw_domain(
        "incremental_unit_raw",
        workspace_root=workspace,
        export_index_dir=export_dir,
        delete_sources=True,
        yes=True,
    )
    latest = get_raw_partition(
        "incremental_unit_raw",
        partition_field="query_date",
        partition_value="2020-01-02",
        workspace_root=workspace,
    )
    assert latest is not None and latest.storage_kind == "bundle"
    assert latest.content_sha256 == second_ref.content_sha256
    assert read_raw_receipt(latest)["revision_of"] == first_ref.content_sha256
    pd.testing.assert_frame_equal(read_raw_partition(latest), second)


def test_fresh_runtime_index_restores_export_before_compacting_another_domain(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    paths = qdp_v3_paths(workspace)
    export_dir = paths.metadata / "raw-index"
    first = pd.DataFrame({"trade_date": ["2020-01-02"], "value": [1]})
    second = pd.DataFrame({"trade_date": ["2021-01-04"], "value": [2]})
    write_raw_partition(
        raw_domain="recovery_domain_a",
        partition_field="query_date",
        partition_value="2020-01-02",
        frame=first,
        receipt={"quality_tier": "strict", "provider": "unit"},
        workspace_root=workspace,
    )
    compact_raw_domain(
        "recovery_domain_a",
        workspace_root=workspace,
        export_index_dir=export_dir,
        delete_sources=True,
        yes=True,
    )

    paths.runtime_index.unlink()
    paths.runtime_index.with_name(f"{paths.runtime_index.name}-wal").unlink(missing_ok=True)
    paths.runtime_index.with_name(f"{paths.runtime_index.name}-shm").unlink(missing_ok=True)
    write_raw_partition(
        raw_domain="recovery_domain_b",
        partition_field="query_date",
        partition_value="2021-01-04",
        frame=second,
        receipt={"quality_tier": "strict", "provider": "unit"},
        workspace_root=workspace,
    )
    compact_raw_domain(
        "recovery_domain_b",
        workspace_root=workspace,
        export_index_dir=export_dir,
        delete_sources=True,
        yes=True,
    )

    rebuilt = RuntimeIndex(workspace_root=workspace)
    assert set(rebuilt.table_frame("raw_partitions")["domain"]) == {
        "recovery_domain_a",
        "recovery_domain_b",
    }
    assert set(rebuilt.table_frame("raw_receipts")["domain"]) == {
        "recovery_domain_a",
        "recovery_domain_b",
    }
    pd.testing.assert_frame_equal(
        read_raw_partition(iter_raw_partitions("recovery_domain_a", workspace_root=workspace)[0]),
        first,
    )
    pd.testing.assert_frame_equal(
        read_raw_partition(iter_raw_partitions("recovery_domain_b", workspace_root=workspace)[0]),
        second,
    )

    exported = pd.read_parquet(export_dir / "raw_index.parquet", engine="pyarrow")
    assert set(exported["domain"]) == {"recovery_domain_a", "recovery_domain_b"}
    paths.runtime_index.unlink()
    paths.runtime_index.with_name(f"{paths.runtime_index.name}-wal").unlink(missing_ok=True)
    paths.runtime_index.with_name(f"{paths.runtime_index.name}-shm").unlink(missing_ok=True)
    assert iter_raw_partitions("recovery_domain_a", workspace_root=workspace)[0].storage_kind == "bundle"
    assert iter_raw_partitions("recovery_domain_b", workspace_root=workspace)[0].storage_kind == "bundle"


def test_fresh_runtime_index_rejects_tampered_export(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    paths = qdp_v3_paths(workspace)
    export_dir = paths.metadata / "raw-index"
    write_raw_partition(
        raw_domain="tampered_export_raw",
        partition_field="query_date",
        partition_value="2020-01-02",
        frame=pd.DataFrame({"trade_date": ["2020-01-02"], "value": [1]}),
        receipt={"quality_tier": "strict", "provider": "unit"},
        workspace_root=workspace,
    )
    compact_raw_domain(
        "tampered_export_raw",
        workspace_root=workspace,
        export_index_dir=export_dir,
    )
    paths.runtime_index.unlink()
    paths.runtime_index.with_name(f"{paths.runtime_index.name}-wal").unlink(missing_ok=True)
    paths.runtime_index.with_name(f"{paths.runtime_index.name}-shm").unlink(missing_ok=True)
    index_path = export_dir / "raw_index.parquet"
    index_path.write_bytes(index_path.read_bytes() + b"tampered")

    with pytest.raises(RuntimeError, match="runtime_index_export_hash_mismatch"):
        RuntimeIndex(workspace_root=workspace)


def test_index_only_empty_raw_survives_restart_audit_export_and_compaction(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    paths = qdp_v3_paths(workspace)
    empty = pd.DataFrame(
        {
            "ts_code": pd.Series(dtype="object"),
            "trade_date": pd.Series(dtype="object"),
            "value": pd.Series(dtype="float64"),
        }
    )
    first, created = write_empty_raw_partition(
        raw_domain="index_only_empty_raw",
        partition_field="trade_date",
        partition_value="2020-01-02",
        frame=empty,
        receipt={
            "quality_tier": "strict",
            "provider": "unit",
            "request": {"trade_date": "20200102"},
            "response_sha256": "a" * 64,
        },
        workspace_root=workspace,
    )
    assert created
    assert first.storage_kind == "bundle"
    assert not (paths.raw / "index_only_empty_raw").exists()
    pd.testing.assert_frame_equal(read_raw_partition(first), empty)

    repeated, created = write_empty_raw_partition(
        raw_domain="index_only_empty_raw",
        partition_field="trade_date",
        partition_value="2020-01-02",
        frame=empty,
        receipt={
            "quality_tier": "strict",
            "provider": "unit",
            "request": {"trade_date": "20200102"},
            "response_sha256": "a" * 64,
        },
        workspace_root=workspace,
    )
    assert not created
    assert repeated.content_sha256 == first.content_sha256
    assert len(RuntimeIndex(workspace_root=workspace).table_frame("raw_receipts")) == 1

    revised_empty = empty.assign(extra=pd.Series(dtype="int64"))
    revised, created = write_empty_raw_partition(
        raw_domain="index_only_empty_raw",
        partition_field="trade_date",
        partition_value="2020-01-02",
        frame=revised_empty,
        receipt={
            "quality_tier": "strict",
            "provider": "unit",
            "request": {"trade_date": "20200102", "schema": "v2"},
            "response_sha256": "d" * 64,
        },
        workspace_root=workspace,
    )
    assert created
    assert revised.revision_of == first.content_sha256
    assert get_raw_partition(
        "index_only_empty_raw",
        partition_field="trade_date",
        partition_value="2020-01-02",
        workspace_root=workspace,
    ).content_sha256 == revised.content_sha256
    exact_first = get_raw_partition_version(
        "index_only_empty_raw",
        partition_field="trade_date",
        partition_value="2020-01-02",
        content_sha256=first.content_sha256,
        workspace_root=workspace,
    )
    assert exact_first is not None
    pd.testing.assert_frame_equal(read_raw_partition(exact_first), empty)
    assert len(RuntimeIndex(workspace_root=workspace).table_frame("raw_receipts")) == 2

    candidate_item = first.to_dict(root=paths.root)
    findings, _ = _audit_raw_references(
        SimpleNamespace(raw_partitions=[candidate_item]),
        root=paths.root,
        workspace_root=workspace,
        full=True,
    )
    assert findings == []

    export_dir = paths.metadata / "raw_index"
    exported = RuntimeIndex(workspace_root=workspace).export_raw_index(export_dir)
    compacted = compact_raw_domain(
        "index_only_empty_raw",
        workspace_root=workspace,
        export_index_dir=export_dir,
    )
    assert compacted.source_partition_count == 0
    assert compacted.raw_index_export is not None
    assert compacted.raw_index_export.receipt_count == 2
    assert exported.receipt_count == 2

    paths.runtime_index.unlink()
    paths.runtime_index.with_name(f"{paths.runtime_index.name}-wal").unlink(missing_ok=True)
    paths.runtime_index.with_name(f"{paths.runtime_index.name}-shm").unlink(missing_ok=True)
    restored = get_raw_partition(
        "index_only_empty_raw",
        partition_field="trade_date",
        partition_value="2020-01-02",
        workspace_root=workspace,
    )
    assert restored is not None and restored.storage_kind == "bundle"
    pd.testing.assert_frame_equal(read_raw_partition(restored), revised_empty)
    findings, _ = _audit_raw_references(
        SimpleNamespace(raw_partitions=[candidate_item]),
        root=paths.root,
        workspace_root=workspace,
        full=True,
    )
    assert findings == []


def test_tushare_reference_empty_uses_index_only_and_exports_for_resume(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    paths = qdp_v3_paths(workspace)

    class EmptyReferenceClient:
        def __init__(self, *, fail_on_fetch: bool = False) -> None:
            self.fail_on_fetch = fail_on_fetch
            self.fetch_count = 0
            self.config = SimpleNamespace(
                token="unit-secret",
                public_metadata=lambda: {
                    "provider": "tushare_proxy",
                    "endpoint": "https://unit.test/api",
                    "token_sha256": "b" * 64,
                },
            )

        def fetch_frame(self, *, api_name: str, params: dict[str, object], fields: object) -> object:
            self.fetch_count += 1
            if self.fail_on_fetch:
                raise AssertionError("completed empty task must resume without a request")
            columns = list(fields) if not isinstance(fields, str) else fields.split(",")
            return SimpleNamespace(
                frame=pd.DataFrame(columns=columns),
                request_metadata_without_token={
                    "api_name": api_name,
                    "params": params,
                    "fields": ",".join(columns),
                },
                response_sha256="c" * 64,
                elapsed_seconds=0.01,
                attempts=1,
            )

        def operational_metrics(self) -> dict[str, object]:
            return {"request_count": self.fetch_count, "error_rate": 0.0}

        def close_thread_session(self) -> None:
            return None

    first_client = EmptyReferenceClient()
    first_result = ingest_tushare_proxy_reference(
        domain="status",
        start_date="2020-01-02",
        end_date="2020-01-02",
        trade_dates=("2020-01-02",),
        workspace_root=workspace,
        client=first_client,
        max_workers=1,
    )
    assert first_result["status"] == "completed"
    assert first_client.fetch_count == 1
    assert first_result["raw_index_export"]["receipt_count"] == 1
    assert not (paths.raw / RAW_TUSHARE_PROXY_SUSPEND).exists()
    refs = iter_raw_partitions(RAW_TUSHARE_PROXY_SUSPEND, workspace_root=workspace)
    assert len(refs) == 1 and refs[0].row_count == 0
    assert read_raw_partition(refs[0]).empty

    paths.runtime_index.unlink()
    paths.runtime_index.with_name(f"{paths.runtime_index.name}-wal").unlink(missing_ok=True)
    paths.runtime_index.with_name(f"{paths.runtime_index.name}-shm").unlink(missing_ok=True)
    resumed_client = EmptyReferenceClient(fail_on_fetch=True)
    resumed = ingest_tushare_proxy_reference(
        domain="status",
        start_date="2020-01-02",
        end_date="2020-01-02",
        trade_dates=("2020-01-02",),
        workspace_root=workspace,
        client=resumed_client,
        max_workers=1,
    )
    assert resumed["status"] == "completed"
    assert resumed_client.fetch_count == 0
    assert len(iter_raw_partitions(RAW_TUSHARE_PROXY_SUSPEND, workspace_root=workspace)) == 1
