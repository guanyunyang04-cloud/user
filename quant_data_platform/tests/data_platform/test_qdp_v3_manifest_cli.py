from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quant_data_platform.cli import main as public_cli
from quant_data_platform.qdp_v3.datasets import read_dataset_frame, write_dataset, write_partitioned_dataset
from quant_data_platform.qdp_v3.gc import collect_garbage
from quant_data_platform.qdp_v3.freeze import freeze_v2, validate_v2_freeze_proof
from quant_data_platform.qdp_v3.ingest import _ingest_job_lock, resolve_trade_dates
from quant_data_platform.qdp_v3.manifest import dataset_manifest_for_id, read_dataset_manifest
from quant_data_platform.qdp_v3.paths import qdp_v3_paths
from quant_data_platform.qdp_v3.quality import report_for
from quant_data_platform.qdp_v3.release import _active_manifest_lock, diff_candidate, write_candidate
from quant_data_platform.qdp_v3.storage import write_raw_partition
from quant_data_platform.qdp_v3.constants import (
    DOMAIN_ADJUST_FACTOR_DAILY,
    DOMAIN_ADJUST_FACTOR_EVENT,
    DOMAIN_VALUATION_DAILY,
    RAW_ADJUST_FACTOR_EVENT,
    RAW_CORPORATE_ACTION_XDXR,
    RAW_DAILY_ASTOCK,
    RAW_TRADING_CALENDAR,
    STRICT_RELEASE_DOMAINS,
)


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    path = workspace / "brain" / "brain_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "brain_type": "main"}), encoding="utf-8")
    return workspace


def test_trade_date_resolution_reuses_covering_calendar_partition(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    calendar = pd.DataFrame(
        {
            "trade_date": ["2012-01-03", "2012-01-04", "2012-01-05", "2012-01-06"],
            "is_open": [False, True, True, False],
            "exchange": ["SSE"] * 4,
        }
    )
    write_raw_partition(
        raw_domain=RAW_TRADING_CALENDAR,
        partition_field="request_range",
        partition_value="2010-01-01_2020-12-31",
        frame=calendar,
        receipt={"quality_tier": "strict", "provider": "unit"},
        workspace_root=workspace,
    )

    dates = resolve_trade_dates(
        start_date="2012-01-04",
        end_date="2012-01-05",
        workspace_root=workspace,
    )

    assert dates == ["2012-01-04", "2012-01-05"]


def test_release_requires_strict_valuation_and_factor_domains() -> None:
    required = {DOMAIN_VALUATION_DAILY, DOMAIN_ADJUST_FACTOR_EVENT, DOMAIN_ADJUST_FACTOR_DAILY}
    assert required.issubset(set(STRICT_RELEASE_DOMAINS))


def _write_v2_lineage_gap_fixture(workspace: Path) -> tuple[str, str]:
    from quant_data_platform.qdp_v2.manifest import (
        DatasetManifest,
        ShardManifestEntry,
        qdp_v2_root,
        write_active_manifest,
        write_dataset_manifest,
    )

    root = qdp_v2_root(workspace)
    leaf_id = "market_daily_raw__leaf"
    missing_id = "market_daily_raw__missing_ancestor"
    shard_path = root / "datasets" / "market_daily_raw" / leaf_id / "shards" / "part.parquet"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_bytes(b"immutable-leaf-evidence")
    missing_source_path = root / "datasets" / "market_daily_raw" / missing_id / "shards" / "source.parquet"
    write_dataset_manifest(
        root,
        DatasetManifest(
            dataset_id=leaf_id,
            domain="market_daily_raw",
            layer="canonical_raw_fact",
            frequency="1d",
            contract_version="qdp_v2",
            primary_key=["symbol", "trade_date"],
            start_date="2026-01-05",
            end_date="2026-01-05",
            row_count=1,
            schema_hash="schema",
            shards=[ShardManifestEntry(path=str(shard_path), source_path=str(missing_source_path))],
            source={"source_dataset_id": missing_id},
            quality={},
        ),
    )
    write_active_manifest(root, {"datasets": {"market_daily_raw": leaf_id}})
    return leaf_id, missing_id


def test_v2_freeze_gap_requires_explicit_authorization_and_hashes_extant_bytes(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _, missing_id = _write_v2_lineage_gap_fixture(workspace)

    strict = freeze_v2(workspace_root=workspace, hash_shards=True, expected_active_sha256="")
    assert strict["status"] == "blocked_missing_lineage"
    assert strict["proof_validation"]["acceptable"] is False

    accepted = freeze_v2(
        workspace_root=workspace,
        hash_shards=True,
        expected_active_sha256="",
        accept_missing_lineage=True,
        authorization_note="Unit-test operator explicitly accepts the irrecoverable ancestor gap.",
    )
    assert accepted["status"] == "complete_with_authorized_lineage_gap"
    assert accepted["extant_shards_fully_hashed"] is True
    assert accepted["proof_validation"] == {
        "status": "accepted",
        "acceptable": True,
        "proof_kind": "authorized_lineage_gap",
        "issue_count": 0,
        "issues": [],
        "missing_dataset_count": 1,
    }
    freeze_payload = json.loads(
        (qdp_v3_paths(workspace).metadata / "v2_freeze_20260713.json").read_text(encoding="utf-8")
    )
    assert freeze_payload["missing_dataset_ids"] == [missing_id]
    assert validate_v2_freeze_proof(freeze_payload)["acceptable"] is True
    freeze_payload["authorized_lineage_gap_contract"]["authorization_note"] = "tampered"
    assert "authorized_gap_contract_hash_mismatch" in validate_v2_freeze_proof(freeze_payload)["issues"]


def test_v2_freeze_gap_cli_requires_authorization_note(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = _workspace(tmp_path)
    _write_v2_lineage_gap_fixture(workspace)

    code = public_cli(
        [
            "--generation",
            "v3",
            "freeze",
            "v2",
            "--workspace-root",
            str(workspace),
            "--expect-active-sha",
            "",
            "--accept-missing-lineage",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["message"] == "qdp_v2_missing_lineage_authorization_note_required"


def test_update_plan_summarizes_dates_instead_of_serializing_every_task(tmp_path: Path) -> None:
    from quant_data_platform.qdp_v3.update import _date_task_summary

    dates = [f"2026-01-{day:02d}" for day in range(1, 32)]
    summary = _date_task_summary(dates)

    assert summary["task_count"] == 31
    assert summary["start_date"] == "2026-01-01"
    assert summary["end_date"] == "2026-01-31"
    assert summary["sample_dates"] == dates[:5] + dates[-5:]
    assert "dates" not in summary


def test_candidate_diff_uses_v2_active_before_first_v3_publish(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    workspace = _workspace(tmp_path)
    v2_active = workspace / "quant_data_platform" / "data" / "qdp_v2" / "active" / "active.json"
    v2_active.parent.mkdir(parents=True, exist_ok=True)
    v2_active.write_text(
        json.dumps({"datasets": {"market_daily_raw": "market_daily_raw__legacy"}}),
        encoding="utf-8",
    )
    candidate = write_candidate(
        datasets={"market_daily_raw": "market_daily_raw__v3"},
        dataset_manifest_sha256={},
        quality_tiers={"market_daily_raw": "strict"},
        blockers=[],
        coverage={},
        build={"test": True},
        raw_partitions=[],
        quarantine=[],
        workspace_root=workspace,
    )

    result = diff_candidate(candidate.candidate_id, workspace_root=workspace, against="active")

    assert result["against"] == "v2_active"
    assert result["active_generation"] == "v2"
    assert result["changes"][0]["active_dataset_id"] == "market_daily_raw__legacy"
    code = public_cli(
        [
            "--generation",
            "v3",
            "diff",
            "--candidate",
            candidate.candidate_id,
            "--against",
            "active",
            "--workspace-root",
            str(workspace),
            "--json",
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["against"] == "v2_active"


def test_active_manifest_lock_rejects_concurrent_local_writer(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    with _active_manifest_lock(workspace):
        with pytest.raises(RuntimeError, match="active_manifest_lock_held"):
            with _active_manifest_lock(workspace):
                raise AssertionError("a second writer must not enter the critical section")
        with pytest.raises(RuntimeError, match="active_manifest_lock_held"):
            collect_garbage(workspace_root=workspace, apply=True, yes=True)


def test_ingest_job_lock_rejects_second_writer_for_same_job(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    with _ingest_job_lock("annual_daily_2012", workspace_root=workspace):
        with pytest.raises(RuntimeError, match="ingest_job_lock_held:annual_daily_2012"):
            with _ingest_job_lock("annual_daily_2012", workspace_root=workspace):
                raise AssertionError("a second writer must not enter the ingest job")
        with _ingest_job_lock("annual_factor_2012", workspace_root=workspace):
            pass


def test_stream_dataset_is_content_addressed_and_bounded(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    root = qdp_v3_paths(workspace).root
    columns = {
        "security_id": ["S1", "S1"],
        "trade_date": ["2026-01-05", "2026-01-05"],
        "bar_end": ["09:35", "09:40"],
        "value": [1.0, 2.0],
    }
    first = pd.DataFrame(columns)
    second = pd.DataFrame({**columns, "trade_date": ["2026-02-05", "2026-02-05"], "value": [3.0, 4.0]})

    manifest = write_partitioned_dataset(
        root=root,
        domain="unit_stream",
        partition_frames=iter([("S1_202601", "202601", first), ("S1_202602", "202602", second)]),
        layer="canonical",
        frequency="5m",
        primary_key=["security_id", "trade_date", "bar_end"],
        quality_report=report_for("unit_stream", []),
        partitioning="natural_month",
    )
    repeated = write_partitioned_dataset(
        root=root,
        domain="unit_stream",
        partition_frames=iter([("S1_202601", "202601", first), ("S1_202602", "202602", second)]),
        layer="canonical",
        frequency="5m",
        primary_key=["security_id", "trade_date", "bar_end"],
        quality_report=report_for("unit_stream", []),
        partitioning="natural_month",
    )

    assert manifest.dataset_id == repeated.dataset_id
    assert manifest.row_count == 4
    assert len(manifest.shards) == 2
    assert len(read_dataset_frame(root, manifest)) == 4


def test_stream_dataset_rejects_schema_drift(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    root = qdp_v3_paths(workspace).root
    one = pd.DataFrame({"id": [1], "trade_date": ["2026-01-01"]})
    two = pd.DataFrame({"id": [2], "trade_date": ["2026-02-01"], "extra": [1]})
    with pytest.raises(ValueError, match="stream_dataset_schema_mismatch"):
        write_partitioned_dataset(
            root=root,
            domain="unit_bad_stream",
            partition_frames=iter([("one", "202601", one), ("two", "202602", two)]),
            layer="canonical",
            frequency="1d",
            primary_key=["id"],
            quality_report=report_for("unit_bad_stream", []),
            partitioning="month",
        )


def test_dataset_identity_includes_quality_coverage_and_partition_contract(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    root = qdp_v3_paths(workspace).root
    frame = pd.DataFrame({"id": [1], "trade_date": ["2026-01-05"]})

    first = write_dataset(
        root=root,
        domain="unit_semantic_identity",
        frame=frame,
        layer="canonical",
        frequency="1d",
        primary_key=["id"],
        quality_report=report_for("unit_semantic_identity", []),
        coverage={"proved": 1},
        date_column="trade_date",
        partitioning="year",
    )
    changed_coverage = write_dataset(
        root=root,
        domain="unit_semantic_identity",
        frame=frame,
        layer="canonical",
        frequency="1d",
        primary_key=["id"],
        quality_report=report_for("unit_semantic_identity", []),
        coverage={"proved": 0},
        date_column="trade_date",
        partitioning="year",
    )
    changed_partitioning = write_dataset(
        root=root,
        domain="unit_semantic_identity",
        frame=frame,
        layer="canonical",
        frequency="1d",
        primary_key=["id"],
        quality_report=report_for("unit_semantic_identity", []),
        coverage={"proved": 1},
        date_column="trade_date",
        partitioning="month",
    )

    assert len({first.dataset_id, changed_coverage.dataset_id, changed_partitioning.dataset_id}) == 3


def test_gc_keeps_candidate_inputs_transitively(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    paths = qdp_v3_paths(workspace)
    source = write_dataset(
        root=paths.root,
        domain="source",
        frame=pd.DataFrame({"id": [1]}),
        layer="raw",
        frequency="event",
        primary_key=["id"],
        quality_report=report_for("source", []),
        date_column="",
        partitioning="single",
    )
    source_path = dataset_manifest_for_id(paths.root, source.dataset_id, source.domain)
    assert source_path is not None
    from quant_data_platform.qdp_v3.datasets import dataset_input_ref
    from quant_data_platform.qdp_v3.manifest import manifest_sha256
    from quant_data_platform.qdp_v3.release import write_candidate

    derived = write_dataset(
        root=paths.root,
        domain="derived",
        frame=pd.DataFrame({"id": [1]}),
        layer="canonical",
        frequency="event",
        primary_key=["id"],
        quality_report=report_for("derived", []),
        inputs=[dataset_input_ref(paths.root, source)],
        date_column="",
        partitioning="single",
    )
    derived_path = dataset_manifest_for_id(paths.root, derived.dataset_id, derived.domain)
    assert derived_path is not None
    write_candidate(
        datasets={"derived": derived.dataset_id},
        dataset_manifest_sha256={"derived": manifest_sha256(derived_path)},
        quality_tiers={"derived": "strict"},
        blockers=[],
        coverage={},
        build={"unit": True},
        raw_partitions=[],
        quarantine=[],
        workspace_root=workspace,
    )

    result = collect_garbage(workspace_root=workspace, min_age_days=0)

    assert result["reachable_dataset_count"] == 2
    assert result["eligible_count"] == 0


def test_candidate_graph_validation_walks_transitive_inputs(tmp_path: Path) -> None:
    from quant_data_platform.qdp_v3.datasets import dataset_input_ref
    from quant_data_platform.qdp_v3.manifest import manifest_sha256
    from quant_data_platform.qdp_v3.release import validate_candidate_graph, write_candidate

    workspace = _workspace(tmp_path)
    paths = qdp_v3_paths(workspace)
    source = write_dataset(
        root=paths.root,
        domain="source",
        frame=pd.DataFrame({"id": [1]}),
        layer="raw",
        frequency="event",
        primary_key=["id"],
        quality_report=report_for("source", []),
        date_column="",
        partitioning="single",
    )
    middle = write_dataset(
        root=paths.root,
        domain="middle",
        frame=pd.DataFrame({"id": [1]}),
        layer="derived",
        frequency="event",
        primary_key=["id"],
        quality_report=report_for("middle", []),
        inputs=[dataset_input_ref(paths.root, source)],
        date_column="",
        partitioning="single",
    )
    top = write_dataset(
        root=paths.root,
        domain="top",
        frame=pd.DataFrame({"id": [1]}),
        layer="canonical",
        frequency="event",
        primary_key=["id"],
        quality_report=report_for("top", []),
        inputs=[dataset_input_ref(paths.root, middle)],
        date_column="",
        partitioning="single",
    )
    top_path = dataset_manifest_for_id(paths.root, top.dataset_id, top.domain)
    source_path = dataset_manifest_for_id(paths.root, source.dataset_id, source.domain)
    assert top_path is not None and source_path is not None
    candidate = write_candidate(
        datasets={"top": top.dataset_id},
        dataset_manifest_sha256={"top": manifest_sha256(top_path)},
        quality_tiers={"top": "strict"},
        blockers=[],
        coverage={},
        build={"unit": "recursive_graph"},
        raw_partitions=[],
        quarantine=[],
        workspace_root=workspace,
    )
    source_path.unlink()

    findings = validate_candidate_graph(candidate, workspace_root=workspace)

    assert any(item["code"] == "dataset_input_missing" and item["dataset_id"] == source.dataset_id for item in findings)


def test_public_cli_routes_v3_and_archives_training_pack(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    workspace = _workspace(tmp_path)
    code = public_cli(["--generation", "v3", "status", "--workspace-root", str(workspace), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["generation"] == "v3"
    assert payload["status"] == "not_published"

    code = public_cli(["rebuild", "training-pack", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["status"] == "archived"


def test_public_cli_json_errors_are_stable(capsys: pytest.CaptureFixture[str]) -> None:
    code = public_cli(["--generation", "future", "status", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload == {
        "status": "error",
        "error_type": "ValueError",
        "message": "unsupported_generation:future",
    }


def test_public_cli_json_suppresses_progress_chatter(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from quant_data_platform import cli as public_cli_module
    from quant_data_platform.progress import progress_write

    def fake_dispatch(_argv: list[str]) -> int:
        progress_write("provider progress must not pollute JSON")
        print('{"status":"ok"}')
        return 0

    monkeypatch.setattr(public_cli_module, "_dispatch", fake_dispatch)

    assert public_cli_module.main(["status", "--json"]) == 0
    assert capsys.readouterr().out == '{"status":"ok"}\n'


def test_small_candidate_build_uses_streamed_year_partitions(tmp_path: Path) -> None:
    from quant_data_platform.qdp_v3.build import build_candidate

    workspace = _workspace(tmp_path)
    daily = pd.DataFrame(
        {
            "date": ["2016-01-04"],
            "code": ["sz.302132"],
            "open": ["10"],
            "high": ["11"],
            "low": ["9"],
            "close": ["10.5"],
            "preclose": ["10"],
            "volume": ["100"],
            "amount": ["1000"],
            "adjustflag": ["3"],
            "turn": ["1"],
            "tradestatus": ["1"],
            "pctChg": ["5"],
            "peTTM": ["10"],
            "pbMRQ": ["2"],
            "psTTM": ["3"],
            "pcfNcfTTM": ["4"],
            "isST": ["0"],
        }
    )
    write_raw_partition(raw_domain=RAW_DAILY_ASTOCK, partition_value="2016-01-04", frame=daily, receipt={"quality_tier": "strict"}, workspace_root=workspace)
    write_raw_partition(
        raw_domain=RAW_ADJUST_FACTOR_EVENT,
        partition_value="2016-01-04",
        frame=pd.DataFrame(
            {
                "code": ["sz.302132"],
                "dividOperateDate": ["2016-01-04"],
                "foreAdjustFactor": ["1.0"],
                "backAdjustFactor": ["1.0"],
                "adjustFacto": ["1.0"],
            }
        ),
        receipt={"quality_tier": "strict"},
        workspace_root=workspace,
    )
    xdxr = pd.DataFrame(
        {
            "year": [2016, 2016],
            "month": [1, 1],
            "day": [4, 4],
            "category": [1, 5],
            "name": ["除权除息", "股本变化"],
            "fenhong": [1.0, None],
            "peigujia": [0.0, None],
            "songzhuangu": [0.0, None],
            "peigu": [0.0, None],
            "panqianliutong": [None, 100.0],
            "panhouliutong": [None, 120.0],
            "qianzongguben": [None, 200.0],
            "houzongguben": [None, 220.0],
            "provider_symbol": ["302132.SZ", "302132.SZ"],
        }
    )
    write_raw_partition(
        raw_domain=RAW_CORPORATE_ACTION_XDXR,
        partition_field="provider_symbol",
        partition_value="302132.SZ",
        frame=xdxr,
        receipt={"quality_tier": "provisional", "provider": "mootdx_online", "endpoint": "xdxr/get_xdxr_info", "package_version": "0.11.7"},
        workspace_root=workspace,
    )
    calendar = pd.DataFrame({"trade_date": ["2016-01-04"], "is_open": [True], "exchange": ["SSE/SZSE"], "source": ["unit"]})
    write_raw_partition(raw_domain=RAW_TRADING_CALENDAR, partition_field="request_range", partition_value="2016-01-01_2016-01-04", frame=calendar, receipt={"quality_tier": "strict"}, workspace_root=workspace)
    identity_config = Path(__file__).resolve().parents[2] / "configs" / "qdp_v3_symbol_history.json"

    candidate = build_candidate(
        workspace_root=workspace,
        start_date="2016-01-04",
        end_date="2016-01-04",
        identity_config=identity_config,
        require_factor_dual_path=False,
    )

    market_id = candidate.datasets["market_daily_raw"]
    market_path = dataset_manifest_for_id(qdp_v3_paths(workspace).root, market_id, "market_daily_raw")
    assert market_path is not None
    payload = json.loads(market_path.read_text(encoding="utf-8"))
    assert payload["shards"][0]["partition"]["kind"] == "natural_year"
    assert payload["shards"][0]["partition"]["value"] == "2016"
    assert candidate.datasets["eligible_signal_D"]
    assert candidate.datasets["tradable_open_D1"]
    assert candidate.datasets["corporate_actions"]
    assert candidate.datasets["share_capital_event"]
    assert candidate.datasets["share_capital_daily"]
    blocker_codes = {str(item.get("code", "")) for item in candidate.blockers}
    assert "factor_identity_conflicts_quarantined" not in blocker_codes
    assert "factor_event_not_verified_or_arbitrated" in blocker_codes
    capital_path = dataset_manifest_for_id(qdp_v3_paths(workspace).root, candidate.datasets["share_capital_daily"], "share_capital_daily")
    assert capital_path is not None
    capital = read_dataset_frame(qdp_v3_paths(workspace).root, read_dataset_manifest(capital_path))
    assert capital.loc[0, "total_share"] == pytest.approx(2_200_000.0)
    from quant_data_platform.qdp_v3.audit import audit_candidate

    audit = audit_candidate(candidate_id=candidate.candidate_id, mode="semantic", workspace_root=workspace, write_report=False)
    assert audit["status"] == "failed"
    assert audit["blocker_count"] > 0
    audit_codes = {str(item.get("code", "")) for item in audit["findings"]}
    assert "strict_5m_dataset_missing" not in audit_codes
