from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quantlab.data.qdp_v2.runtime_archive import (
    RuntimeArchiveError,
    restore_archive,
    seal_completed_workflow,
    seal_workflows,
    verify_archive,
)

WORKFLOW = "research_report_rc_backfill_v1"


def _expanded_fixture(workspace: Path) -> Path:
    root = (
        workspace
        / "data"
        / "qdp"
        / "qdp_runtime"
        / WORKFLOW
        / "raw"
        / "moneyflow"
        / "year=2020"
        / "date=2020-01-02"
    )
    root.mkdir(parents=True)
    pd.DataFrame({"symbol": ["000001.SZ"], "value": [1.0]}).to_parquet(
        root / "offset=000000000.parquet", index=False
    )
    (root / "offset=000000000.json").write_text(
        json.dumps(
            {
                "status": "observed",
                "offset": 0,
                "params": {"trade_date": "20200102"},
                "attempt_count": 2,
                "empty_confirmation_count": 0,
                "completed_at": "2020-01-03T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    workflow_root = root.parents[3]
    (workflow_root / "state.json").write_text(
        json.dumps(
            {
                "status": "applied",
                "token": "fixture-secret",
                "days": {"2020-01-02": {"status": "observed"}},
            }
        ),
        encoding="utf-8",
    )
    return root


def test_runtime_archive_seal_verify_restore_and_explicit_delete(
    tmp_path: Path,
) -> None:
    source = _expanded_fixture(tmp_path)

    sealed = seal_workflows(
        workspace_root=tmp_path,
        workflows=(WORKFLOW,),
    )

    assert sealed["status"] == "sealed"
    assert sealed["source_file_count"] == 2
    assert sealed["sample_restore"]["status"] == "restored_and_hash_verified"
    manifest = json.loads(Path(sealed["manifests"][0]).read_text(encoding="utf-8"))
    snapshot = json.loads(
        Path(manifest["state_snapshot"]["path"]).read_text(encoding="utf-8")
    )
    ledger = pd.read_parquet(manifest["ledger_path"])
    metadata = ledger.loc[ledger["suffix"].eq(".json")].iloc[0]
    assert snapshot["token"] == "<redacted>"
    assert snapshot["status"] == "applied"
    assert metadata["attempt_count"] == 2
    assert metadata["empty_confirmation_count"] == 0
    assert metadata["completed_at"] == "2020-01-03T00:00:00Z"
    restored = restore_archive(
        manifest["archive_path"],
        manifest["ledger_path"],
        target_root=tmp_path / "manual_restore",
    )
    assert restored["restored_file_count"] == 2
    assert list((tmp_path / "manual_restore").rglob("*.parquet"))
    with pytest.raises(RuntimeArchiveError, match="requires_yes"):
        seal_workflows(
            workspace_root=tmp_path,
            workflows=(WORKFLOW,),
            delete_expanded=True,
        )

    deleted = seal_workflows(
        workspace_root=tmp_path,
        workflows=(WORKFLOW,),
        delete_expanded=True,
        yes=True,
    )

    assert deleted["status"] == "sealed_and_deleted"
    assert not any(source.rglob("*"))


def test_runtime_archive_corruption_is_rejected(tmp_path: Path) -> None:
    _expanded_fixture(tmp_path)
    sealed = seal_workflows(
        workspace_root=tmp_path,
        workflows=(WORKFLOW,),
    )
    manifest = json.loads(Path(sealed["manifests"][0]).read_text(encoding="utf-8"))
    archive = Path(manifest["archive_path"])
    payload = bytearray(archive.read_bytes())
    payload[len(payload) // 2] ^= 0xFF
    archive.write_bytes(payload)

    with pytest.raises(RuntimeArchiveError):
        verify_archive(archive, pd.read_parquet(manifest["ledger_path"]))


def test_runtime_archive_auto_seal_and_different_existing_unit_is_rejected(
    tmp_path: Path,
) -> None:
    source = _expanded_fixture(tmp_path)
    sealed = seal_completed_workflow(
        WORKFLOW,
        workspace_root=tmp_path,
    )
    assert sealed["status"] == "sealed_and_deleted"
    assert not any(source.rglob("*"))

    source.mkdir(parents=True, exist_ok=True)
    (source / "new.json").write_text('{"status":"observed"}', encoding="utf-8")
    with pytest.raises(RuntimeArchiveError, match="restore_before_reseal"):
        seal_completed_workflow(
            WORKFLOW,
            workspace_root=tmp_path,
        )
