from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

import quant_data_platform.qdp_v2.direct_repair as direct_module
from quant_data_platform.qdp_v2.direct_repair import (
    main,
    plan_direct_repair,
    run_direct_repair,
)
from quant_data_platform.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    qdp_v2_root,
    read_dataset_manifest,
    resolve_manifest_path,
    write_dataset_manifest,
)
from quant_data_platform.qdp_v3.constants import (
    RAW_TUSHARE_PROXY_ADJ_FACTOR,
    RAW_TUSHARE_PROXY_DAILY,
    RAW_TUSHARE_PROXY_STOCK_BASIC,
    RAW_TUSHARE_PROXY_TRADE_CALENDAR,
)
from quant_data_platform.qdp_v3.storage import RawPartitionRef, write_raw_partition


def _workspace(tmp_path: Path, monkeypatch) -> Path:
    workspace = tmp_path / "workspace"
    brain = workspace / "brain" / "brain_manifest.json"
    brain.parent.mkdir(parents=True)
    brain.write_text(json.dumps({"schema_version": 1, "brain_type": "main"}), encoding="utf-8")
    runtime = workspace / "quant_data_platform" / "data" / "qdp_runtime"
    monkeypatch.setenv("QDP_RUNTIME_ROOT", str(runtime))
    config = workspace / "quant_data_platform" / "configs" / "qdp_v3_symbol_history.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps({"identities": [], "symbol_history": []}), encoding="utf-8")
    return workspace


def _write_domain(
    root: Path,
    domain: str,
    dataset_id: str,
    frame: pd.DataFrame,
    contract: str,
    *,
    primary_key: list[str] | None = None,
) -> None:
    path = root / "datasets" / domain / dataset_id / "shards" / "part.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    date_column = "trade_date" if "trade_date" in frame.columns else ""
    start = str(frame[date_column].min()) if date_column and not frame.empty else ""
    end = str(frame[date_column].max()) if date_column and not frame.empty else ""
    write_dataset_manifest(
        root,
        DatasetManifest(
            dataset_id=dataset_id,
            domain=domain,
            layer="raw",
            frequency="1d",
            contract_version=contract,
            primary_key=primary_key or ["trade_date", "symbol"],
            start_date=start,
            end_date=end,
            row_count=len(frame),
            schema_hash="fixture",
            shards=[
                ShardManifestEntry(
                    path=path.relative_to(root).as_posix(),
                    row_count=len(frame),
                    start_date=start,
                    end_date=end,
                )
            ],
            source={"provider": "fixture"},
            quality={"path_refs_exist": True, "primary_key_unique": True},
        ),
    )


def _write_raw(
    workspace: Path,
    *,
    domain: str,
    field: str,
    value: str,
    frame: pd.DataFrame,
) -> RawPartitionRef:
    ref, _ = write_raw_partition(
        raw_domain=domain,
        partition_field=field,
        partition_value=value,
        frame=frame,
        receipt={"provider": "tushare_proxy", "quality_tier": "strict"},
        workspace_root=workspace,
    )
    return ref


def _daily(symbol: str, date: str, close: float) -> dict[str, object]:
    return {
        "ts_code": symbol,
        "trade_date": date.replace("-", ""),
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "pre_close": close - 0.1,
        "change": 0.1,
        "pct_chg": 1.0,
        "vol": 10.0,
        "amount": 20.0,
    }


def _fixture(tmp_path: Path, monkeypatch) -> dict[str, object]:
    workspace = _workspace(tmp_path, monkeypatch)
    root = qdp_v2_root(workspace)
    old_daily = pd.DataFrame(
        [
            {
                "symbol": "600000.SH",
                "trade_date": "2010-01-04",
                "open": 999.0,
                "high": 1_000.0,
                "low": 998.0,
                "close": 999.0,
                "volume": 1.0,
                "amount": 2.0,
                "source": "old_keep_me",
                "adjusted_flag": "none",
            },
            {
                "symbol": "600076.SH",
                "trade_date": "2024-01-02",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.0,
                "volume": 1.0,
                "amount": 2.0,
                "source": "old_keep_me",
                "adjusted_flag": "none",
            },
        ]
    )
    calendar = pd.DataFrame(
        {
            "trade_date": ["2010-01-04", "2024-01-02"],
            "is_open": [True, True],
            "exchange": ["SSE"] * 2,
            "source": ["fixture"] * 2,
        }
    )
    universe = pd.DataFrame(
        {
            "symbol": ["600000.SH", "600076.SH"],
            "trade_date": ["2010-01-04", "2024-01-02"],
            "name": ["old", "old"],
            "exchange": ["SH", "SH"],
            "board": ["main", "main"],
            "list_status": ["L", "L"],
            "list_date": ["", ""],
            "delist_date": ["", ""],
            "source": ["old", "old"],
        }
    )
    old_status = pd.DataFrame(
        {
            "symbol": ["600000.SH", "600076.SH"],
            "trade_date": ["2010-01-04", "2024-01-02"],
            "is_st": [True, False],
            "is_suspended": [False, False],
            "is_delisted": [False, False],
            "status_reason": ["old_wrong_but_preserved", "old"],
            "source": ["old_keep_me", "old_keep_me"],
        }
    )
    old_factor = pd.DataFrame(
        {
            "symbol": old_daily["symbol"],
            "trade_date": old_daily["trade_date"],
            "fore_adjust_factor": [99.0, 88.0],
            "back_adjust_factor": [99.0, 88.0],
            "adjust_factor": [99.0, 88.0],
            "factor_provider": ["bad_old", "bad_old"],
            "factor_semantics": ["bad", "bad"],
            "source": ["bad_old", "bad_old"],
            "factor_source_date": old_daily["trade_date"],
            "ffill_days": [0, 0],
        }
    )
    _write_domain(root, "market_daily_raw", "market_daily_raw__old", old_daily, "qdp_v2_market_daily_raw_v1")
    _write_domain(
        root,
        "trading_calendar",
        "trading_calendar__old",
        calendar,
        "qdp_v2_trading_calendar_v1",
        primary_key=["trade_date"],
    )
    _write_domain(root, "universe_snapshot", "universe_snapshot__old", universe, "qdp_v2_universe_snapshot_v1")
    _write_domain(root, "security_status", "security_status__old", old_status, "qdp_v2_security_status_v1")
    _write_domain(root, "adjust_factor", "adjust_factor__old", old_factor, "qdp_v2_adjust_factor_standard_v2")
    active = {
        "version": 2,
        "active_as_of_date": "2024-01-03",
        "datasets": {
            "market_daily_raw": "market_daily_raw__old",
            "trading_calendar": "trading_calendar__old",
            "universe_snapshot": "universe_snapshot__old",
            "security_status": "security_status__old",
            "adjust_factor": "adjust_factor__old",
        },
    }
    active_path = root / "active" / "active.json"
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active_path.write_text(json.dumps(active, sort_keys=True) + "\n", encoding="utf-8")

    stock_ref = _write_raw(
        workspace,
        domain=RAW_TUSHARE_PROXY_STOCK_BASIC,
        field="as_of_date",
        value="2026-07-13",
        frame=pd.DataFrame(
            [
                {"ts_code": "600000.SH", "name": "浦发银行", "list_date": "19991110", "delist_date": ""},
                {"ts_code": "600076.SH", "name": "康欣新材", "list_date": "19970526", "delist_date": ""},
            ]
        ),
    )
    calendar_ref = _write_raw(
        workspace,
        domain=RAW_TUSHARE_PROXY_TRADE_CALENDAR,
        field="request_range",
        value="2010-01-01_2024-01-03",
        frame=pd.DataFrame(
            {
                "exchange": ["SSE"] * 5,
                "cal_date": [
                    "20100101",
                    "20100104",
                    "20100105",
                    "20240102",
                    "20240103",
                ],
                "is_open": [0, 1, 1, 1, 1],
                "pretrade_date": ["", "20091231", "20100104", "20231229", "20240102"],
            }
        ),
    )
    daily_refs = [
        _write_raw(
            workspace,
            domain=RAW_TUSHARE_PROXY_DAILY,
            field="trade_date",
            value="2010",
            frame=pd.DataFrame(
                [
                    _daily("600000.SH", "2010-01-04", 10.0),
                    _daily("600000.SH", "2010-01-05", 11.0),
                ]
            ),
        ),
        _write_raw(
            workspace,
            domain=RAW_TUSHARE_PROXY_DAILY,
            field="trade_date",
            value="2024",
            frame=pd.DataFrame(
                [
                    _daily("600076.SH", "2024-01-02", 10.0),
                    _daily("600076.SH", "2024-01-03", 10.5),
                ]
            ),
        ),
    ]
    factor_refs = [
        _write_raw(
            workspace,
            domain=RAW_TUSHARE_PROXY_ADJ_FACTOR,
            field="provider_symbol",
            value="600000.SH",
            frame=pd.DataFrame(
                {
                    "ts_code": ["600000.SH", "600000.SH"],
                    "trade_date": ["20100104", "20100105"],
                    "adj_factor": [10.0, 20.0],
                }
            ),
        ),
        _write_raw(
            workspace,
            domain=RAW_TUSHARE_PROXY_ADJ_FACTOR,
            field="provider_symbol",
            value="600076.SH",
            frame=pd.DataFrame(
                {
                    "ts_code": ["600076.SH"] * 4,
                    "trade_date": ["20100104", "20231229", "20240102", "20240103"],
                    # Trusted Tushare has no 600076 corporate-action jump in 2024.
                    "adj_factor": [10.0, 20.0, 20.0, 20.0],
                }
            ),
        ),
    ]
    return {
        "workspace": workspace,
        "root": root,
        "active_path": active_path,
        "active_sha": hashlib.sha256(active_path.read_bytes()).hexdigest(),
        "calendar_refs": [calendar_ref],
        "stock_refs": [stock_ref],
        "daily_refs": daily_refs,
        "factor_refs": factor_refs,
    }


def _args(fixture: dict[str, object]) -> dict[str, object]:
    return {
        "workspace_root": fixture["workspace"],
        "calendar_refs": fixture["calendar_refs"],
        "stock_basic_refs": fixture["stock_refs"],
        "daily_refs": fixture["daily_refs"],
        "factor_refs": fixture["factor_refs"],
        "suspend_refs": [],
        "namechange_refs": [],
    }


def _read_candidate(result: dict[str, object], domain: str) -> pd.DataFrame:
    manifest = read_dataset_manifest(dict(result["manifest_paths"])[domain])
    root = Path(str(result["qdp_v2_root"]))
    return pd.concat(
        [pd.read_parquet(resolve_manifest_path(item.path, root=root)) for item in manifest.shards],
        ignore_index=True,
    )


def test_dry_run_is_deterministic_and_does_not_create_candidate(tmp_path: Path, monkeypatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    root = Path(fixture["root"])
    dataset_dirs_before = sorted(str(item) for item in (root / "datasets").glob("*/*"))

    first = plan_direct_repair(**_args(fixture))
    second = run_direct_repair(**_args(fixture))

    assert first["plan_id"] == second["plan_id"]
    assert second["status"] == "dry_run"
    assert second["domains"]["trading_calendar"]["missing_rows"] == 3
    assert second["domains"]["market_daily_raw"]["missing_rows"] == 2
    assert second["domains"]["security_status"]["missing_rows"] == 6
    assert (
        second["domains"]["security_status"]["output_rows"]
        == second["domains"]["universe_snapshot"]["output_rows"]
        == 8
    )
    assert sorted(str(item) for item in (root / "datasets").glob("*/*")) == dataset_dirs_before
    assert not (root / "candidates").exists()
    assert hashlib.sha256(Path(fixture["active_path"]).read_bytes()).hexdigest() == fixture["active_sha"]


def test_apply_repairs_same_dataset_ids_and_deletes_prepared_files(tmp_path: Path, monkeypatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    factor_output = Path(fixture["root"]) / "explicit" / "factor.parquet"
    configured_spill: list[Path] = []
    configure_duckdb = direct_module._configure_duckdb

    def capture_duckdb_configuration(connection, *, temp_directory):
        configured_spill.append(Path(temp_directory).resolve())
        return configure_duckdb(connection, temp_directory=temp_directory)

    monkeypatch.setattr(direct_module, "_configure_duckdb", capture_duckdb_configuration)

    result = run_direct_repair(
        **_args(fixture),
        apply=True,
        factor_output_path=factor_output,
    )

    assert result["status"] == "applied"
    assert result["active_manifest_unchanged"] is True
    assert result["dataset_ids"]["market_daily_raw"] == "market_daily_raw__old"
    assert result["dataset_ids"]["trading_calendar"] == "trading_calendar__old"
    assert result["dataset_ids"]["adjust_factor"] == "adjust_factor__old"
    assert hashlib.sha256(Path(fixture["active_path"]).read_bytes()).hexdigest() == fixture["active_sha"]
    runtime_root = Path(fixture["root"]).parent / "qdp_runtime" / "direct_repair"
    assert configured_spill
    assert all(path.is_relative_to(runtime_root) for path in configured_spill)

    calendar = _read_candidate(result, "trading_calendar")
    assert calendar["trade_date"].tolist() == [
        "2010-01-04",
        "2024-01-02",
        "2010-01-01",
        "2010-01-05",
        "2024-01-03",
    ]
    old_calendar = calendar.loc[calendar["trade_date"].eq("2010-01-04")].iloc[0]
    assert old_calendar["source"] == "fixture"
    added_closed = calendar.loc[calendar["trade_date"].eq("2010-01-01")].iloc[0]
    assert not bool(added_closed["is_open"])
    assert added_closed["source"] == "tushare_proxy.trade_cal"

    daily = _read_candidate(result, "market_daily_raw")
    kept = daily.loc[(daily["symbol"] == "600000.SH") & (daily["trade_date"] == "2010-01-04")].iloc[0]
    assert kept["close"] == 999.0
    assert kept["source"] == "old_keep_me"
    assert set(daily["trade_date"]) == {"2010-01-04", "2010-01-05", "2024-01-02", "2024-01-03"}

    universe = _read_candidate(result, "universe_snapshot")
    assert len(universe) == 8
    assert set(universe["symbol"]) == {"600000.SH", "600076.SH"}

    status = _read_candidate(result, "security_status")
    old = status.loc[(status["symbol"] == "600000.SH") & (status["trade_date"] == "2010-01-04")].iloc[0]
    assert bool(old["is_st"])
    assert old["status_reason"] == "old_wrong_but_preserved"
    assert len(status) == 8
    assert set(zip(status["trade_date"], status["symbol"])) == set(
        zip(universe["trade_date"], universe["symbol"])
    )
    no_bar = status.loc[
        status["symbol"].eq("600076.SH")
        & status["trade_date"].eq("2010-01-05")
    ].iloc[0]
    assert not bool(no_bar["is_suspended"])
    assert no_bar["status_reason"] == "missing_or_invalid_bar"

    factor_manifest = read_dataset_manifest(dict(result["manifest_paths"])["adjust_factor"])
    assert len(factor_manifest.shards) == 1
    assert not factor_output.exists()
    installed_factor = Path(result["commit_results"]["adjust_factor"]["new_shard_paths"][0])
    assert pq.ParquetFile(installed_factor).metadata.num_rows == len(daily)
    factor = pd.read_parquet(installed_factor)
    assert set(zip(factor["trade_date"], factor["symbol"])) == set(zip(daily["trade_date"], daily["symbol"]))
    assert factor["factor_provider"].eq("tushare_proxy").all()
    polluted = factor.loc[
        factor["symbol"].eq("600076.SH") & factor["trade_date"].str.startswith("2024"),
        "adjust_factor",
    ]
    assert polluted.nunique() == 1
    assert polluted.iloc[0] == 2.0


def test_full_scan_duckdb_configuration_is_bounded_and_uses_requested_spill(
    tmp_path: Path,
) -> None:
    statements: list[str] = []

    class FakeConnection:
        def execute(self, statement: str):
            statements.append(statement)
            return self

    spill = tmp_path / "repository_runtime" / "duckdb_spill"
    direct_module._configure_duckdb(
        FakeConnection(),
        temp_directory=spill,
        memory_sampler=lambda: 8 * 1024**3,
        total_memory_sampler=lambda: 16 * 1024**3,
    )

    assert spill.is_dir()
    memory_statements = [item for item in statements if "memory_limit=" in item]
    assert memory_statements == ["SET memory_limit='16642998272B'"]
    assert int(memory_statements[0].split("'")[1][:-1]) > 2 * 1024**3
    assert any("preserve_insertion_order=false" in item for item in statements)
    temp_statements = [item for item in statements if "temp_directory=" in item]
    assert len(temp_statements) == 1
    assert str(spill.resolve()) in temp_statements[0]


def test_full_scan_validator_checks_memory_before_and_after_queries(tmp_path: Path) -> None:
    path = tmp_path / "keys.parquet"
    pd.DataFrame(
        {"trade_date": ["2024-01-02", "2024-01-03"], "symbol": ["600000.SH"] * 2}
    ).to_parquet(path, index=False)

    class CountingGuard:
        def __init__(self) -> None:
            self.calls = 0

        def checkpoint(self) -> int:
            self.calls += 1
            return 8 * 1024**3

    guard = CountingGuard()
    direct_module._validate_primary_key(
        [path],
        expected_rows=2,
        work_dir=tmp_path / "validator_work",
        memory_guard=guard,
    )

    assert guard.calls >= 4
    assert (tmp_path / "validator_work" / "spill").is_dir()


def test_universe_status_validator_requires_bidirectional_key_equality(tmp_path: Path) -> None:
    universe = tmp_path / "universe.parquet"
    status = tmp_path / "status.parquet"
    pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-03"],
            "symbol": ["600000.SH", "600000.SH"],
        }
    ).to_parquet(universe, index=False)
    pd.DataFrame(
        {"trade_date": ["2024-01-02"], "symbol": ["600000.SH"]}
    ).to_parquet(status, index=False)
    guard = direct_module._MemoryGuard(floor_bytes=0)

    with pytest.raises(
        direct_module.DirectRepairError,
        match="key_alignment_mismatch:universe_snapshot_only",
    ):
        direct_module._validate_bidirectional_key_alignment(
            [universe],
            [status],
            left_label="universe_snapshot",
            right_label="security_status",
            work_dir=tmp_path / "alignment_work",
            memory_guard=guard,
        )

    pd.read_parquet(universe).to_parquet(status, index=False)
    direct_module._validate_bidirectional_key_alignment(
        [universe],
        [status],
        left_label="universe_snapshot",
        right_label="security_status",
        work_dir=tmp_path / "alignment_work_equal",
        memory_guard=guard,
    )


@pytest.mark.parametrize(
    ("factor_mode", "expected_error"),
    [
        ("missing", "tushare_factor_security_missing:600000.SH"),
        ("late", "tushare_factor_first_date_not_covered:symbol=600000.SH"),
    ],
)
def test_factor_source_must_exist_and_start_no_later_than_final_daily(
    tmp_path: Path,
    monkeypatch,
    factor_mode: str,
    expected_error: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    factor_refs = list(fixture["factor_refs"])[1:]
    if factor_mode == "late":
        factor_refs.insert(
            0,
            _write_raw(
                Path(fixture["workspace"]),
                domain=RAW_TUSHARE_PROXY_ADJ_FACTOR,
                field="provider_symbol",
                value="600000.SH",
                frame=pd.DataFrame(
                    {
                        "ts_code": ["600000.SH"],
                        "trade_date": ["20100105"],
                        "adj_factor": [10.0],
                    }
                ),
            ),
        )
    args = _args(fixture)
    args["factor_refs"] = factor_refs
    root = Path(fixture["root"])
    manifests_before = {
        path: path.read_bytes()
        for path in root.joinpath("datasets").glob("*/*/dataset.json")
    }

    with pytest.raises(direct_module.DirectRepairError, match=expected_error):
        run_direct_repair(**args, apply=True)

    assert all(path.read_bytes() == payload for path, payload in manifests_before.items())
    assert hashlib.sha256(Path(fixture["active_path"]).read_bytes()).hexdigest() == fixture["active_sha"]


def test_low_memory_must_persist_for_five_seconds_before_pause(tmp_path: Path, monkeypatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    now = [0.0]

    def clock() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    result = run_direct_repair(
        **_args(fixture),
        apply=True,
        memory_sampler=lambda: 100 * 1024**2,
        clock=clock,
        sleeper=sleep,
    )

    assert result["status"] == "paused_resource_guard"
    assert now[0] >= 5.0
    assert hashlib.sha256(Path(fixture["active_path"]).read_bytes()).hexdigest() == fixture["active_sha"]
    assert not (Path(fixture["root"]) / "candidates").exists()


def test_commit_failure_keeps_completed_domains_and_resume_finishes(tmp_path: Path, monkeypatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    root = Path(fixture["root"])
    manifest_paths = {
        domain: root / "datasets" / domain / dataset_id / "dataset.json"
        for domain, dataset_id in {
            "trading_calendar": "trading_calendar__old",
            "market_daily_raw": "market_daily_raw__old",
            "universe_snapshot": "universe_snapshot__old",
            "security_status": "security_status__old",
            "adjust_factor": "adjust_factor__old",
        }.items()
    }
    manifest_bytes = {domain: path.read_bytes() for domain, path in manifest_paths.items()}
    shard_paths = {
        domain: sorted(str(item) for item in path.parent.joinpath("shards").glob("*.parquet"))
        for domain, path in manifest_paths.items()
    }

    import quant_data_platform.qdp_v2.direct_repair as direct_module

    original_factor_replace = direct_module.replace_active_shard_from_parquet

    def fail_factor_replace(*args, **kwargs):
        raise RuntimeError("injected_factor_commit_failure")

    monkeypatch.setattr(
        "quant_data_platform.qdp_v2.direct_repair.replace_active_shard_from_parquet",
        fail_factor_replace,
    )
    with pytest.raises(RuntimeError, match="injected_factor_commit_failure"):
        run_direct_repair(**_args(fixture), apply=True)

    assert hashlib.sha256(Path(fixture["active_path"]).read_bytes()).hexdigest() == fixture["active_sha"]
    for domain in (
        "trading_calendar",
        "market_daily_raw",
        "universe_snapshot",
        "security_status",
    ):
        path = manifest_paths[domain]
        assert path.read_bytes() != manifest_bytes[domain]
        assert len(list(path.parent.joinpath("shards").glob("*.parquet"))) > len(shard_paths[domain])
    assert manifest_paths["adjust_factor"].read_bytes() == manifest_bytes["adjust_factor"]
    assert sorted(str(item) for item in manifest_paths["adjust_factor"].parent.joinpath("shards").glob("*.parquet")) == shard_paths["adjust_factor"]

    states = list(
        (Path(fixture["root"]).parent / "qdp_runtime" / "direct_repair").glob("*/state.json")
    )
    partial = json.loads(max(states, key=lambda item: item.stat().st_mtime_ns).read_text(encoding="utf-8"))
    assert partial["status"] == "partial_applied_recoverable"
    assert partial["completed_domains"] == [
        "trading_calendar",
        "market_daily_raw",
        "universe_snapshot",
        "security_status",
    ]
    assert partial["prepared_files_deleted"] is True

    monkeypatch.setattr(direct_module, "replace_active_shard_from_parquet", original_factor_replace)
    resumed = run_direct_repair(**_args(fixture), apply=True)
    assert resumed["status"] == "applied"
    assert resumed["domains"]["trading_calendar"]["missing_rows"] == 0
    assert resumed["domains"]["market_daily_raw"]["missing_rows"] == 0
    assert resumed["domains"]["universe_snapshot"]["missing_rows"] == 0
    assert resumed["domains"]["security_status"]["missing_rows"] == 0
    assert manifest_paths["adjust_factor"].read_bytes() != manifest_bytes["adjust_factor"]
    assert hashlib.sha256(Path(fixture["active_path"]).read_bytes()).hexdigest() == fixture["active_sha"]


def test_cli_defaults_to_dry_run(monkeypatch, capsys) -> None:
    calls: list[bool] = []

    def fake_run_direct_repair(**kwargs):
        calls.append(bool(kwargs["apply"]))
        return {"status": "dry_run", "run_id": "unit", "active_manifest_unchanged": True}

    monkeypatch.setattr(
        "quant_data_platform.qdp_v2.direct_repair.run_direct_repair",
        fake_run_direct_repair,
    )
    assert main([]) == 0
    assert calls == [False]
    assert "status: dry_run" in capsys.readouterr().out
