from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from quant_data_platform.qdp_v2.audit import audit_active
from quant_data_platform.qdp_v2.completion import rebuild_pit_signal_universe
from quant_data_platform.qdp_v2.manifest import qdp_v2_root, read_active_manifest, read_dataset_manifest, write_active_manifest


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    manifest = workspace / "brain" / "brain_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"schema_version": 1, "brain_type": "main"}), encoding="utf-8")
    root = qdp_v2_root(workspace)
    write_active_manifest(
        root,
        {
            "version": 2,
            "active_as_of_date": "2020-01-03",
            "scope": {"start_date": "2020-01-02", "end_date": "2020-01-03"},
            "datasets": {},
        },
    )
    return workspace


def _write_source(snapshot: Path) -> None:
    rows = [
        # A future-looking name must not disqualify an otherwise eligible historical row.
        ("2020-01-02", "000001.SZ", "平安银行", "", True, True, True, False, False, True, True),
        ("2020-01-03", "000001.SZ", "平安银行", "", True, True, True, False, False, True, True),
        ("2020-01-02", "600001.SH", "退市示例", "2020-01-04", True, True, True, False, False, True, True),
        ("2020-01-03", "600001.SH", "退市示例", "2020-01-04", True, True, True, True, False, True, False),
        # ChiNext is rejected even if an upstream flag is accidentally permissive.
        ("2020-01-02", "300001.SZ", "特锐德", "", True, True, True, False, False, True, False),
        ("2020-01-03", "300001.SZ", "特锐德", "", True, True, True, False, False, True, False),
        # Suspension keeps the stock in the research universe but removes it from the signal universe.
        ("2020-01-02", "600002.SH", "齐鲁石化", "", True, True, True, False, True, True, False),
        ("2020-01-03", "600002.SH", "齐鲁石化", "", True, True, True, False, False, True, True),
        # A security that delists later remains eligible before its out date and need not exist afterward.
        ("2020-01-02", "600003.SH", "历史退", "2020-01-03", True, True, True, False, False, True, True),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "date",
            "code",
            "name_on_date",
            "out_date",
            "is_listed_on_date",
            "is_mainboard",
            "is_common_a_share",
            "is_st_on_date",
            "is_suspended_on_date",
            "has_bar",
            "is_tradeable",
        ],
    )
    snapshot.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(":memory:") as con:
        con.register("source_frame", frame)
        con.execute("copy source_frame to ? (format parquet)", [str(snapshot / "daily_universe.parquet")])
    (snapshot / "manifest.json").write_text(
        json.dumps({"schema_version": 2, "snapshot_id": "unit_pit_snapshot"}),
        encoding="utf-8",
    )


def _read_dataset(root: Path, domain: str, dataset_id: str) -> pd.DataFrame:
    manifest = read_dataset_manifest(root / "datasets" / domain / dataset_id / "dataset.json")
    path = root / manifest.shards[0].path
    return pd.read_parquet(path)


def test_rebuild_pit_signal_universe_is_date_local_and_auditable(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    snapshot = workspace / "source" / "pit_snapshot"
    _write_source(snapshot)

    result = rebuild_pit_signal_universe(
        workspace_root=workspace,
        runtime="safe",
        duckdb_memory_limit="1GB",
        threads=1,
        snapshot_root=snapshot,
        activate=True,
    )

    assert result["status"] == "ok"
    assert result["blockers"] == []
    assert result["stats"]["row_count"] == 9
    assert result["stats"]["source_tradeable_mismatch_rows"] == 0
    assert result["stats"]["historical_eligible_not_latest_symbols"] == 2
    assert result["stats"]["source_name_contains_delist_rows_ignored"] == 3

    root = qdp_v2_root(workspace)
    active = read_active_manifest(root)
    universe_id = active["datasets"]["pit_signal_universe"]
    daily_id = active["datasets"]["pit_signal_universe_daily"]
    assert active["research_scopes"]["pit_mainboard_non_st_v1"]["current_survivor_filter_used"] is False

    universe = _read_dataset(root, "pit_signal_universe", universe_id)
    assert "name_on_date" not in universe.columns
    assert "out_date" not in universe.columns
    indexed = universe.set_index(["trade_date", "symbol"])
    assert bool(indexed.loc[("2020-01-02", "600001.SH"), "eligible_for_signal"])
    assert not bool(indexed.loc[("2020-01-03", "600001.SH"), "eligible_for_signal"])
    assert bool(indexed.loc[("2020-01-02", "600003.SH"), "eligible_for_signal"])
    assert bool(indexed.loc[("2020-01-02", "600002.SH"), "eligible_for_research"])
    assert not bool(indexed.loc[("2020-01-02", "600002.SH"), "eligible_for_signal"])
    assert not bool(indexed.loc[("2020-01-02", "300001.SZ"), "eligible_for_research"])

    daily = _read_dataset(root, "pit_signal_universe_daily", daily_id).set_index("trade_date")
    assert int(daily.loc["2020-01-02", "signal_eligible_symbols"]) == 3
    assert int(daily.loc["2020-01-03", "signal_eligible_symbols"]) == 2
    assert daily["signal_membership_hash"].str.len().eq(32).all()
    assert daily["signal_membership_hash"].nunique() == 2

    audit = audit_active(workspace_root=workspace)
    assert audit["status"] == "ok"
    assert audit["errors"] == []
