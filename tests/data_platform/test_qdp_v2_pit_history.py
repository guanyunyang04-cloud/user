from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab.data.qdp_v2 import update as update_module
from quantlab.data.qdp_v2.active import resolve_active_domain
from quantlab.data.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_dataset_manifest,
    write_active_manifest,
    write_dataset_manifest,
)
from quantlab.data.qdp_v2.pit_history.config import (
    _factor_rows,
    _historical_names,
    _name_implies_st,
    _normalize_eastmoney_history,
    _normalize_sina_factors,
)
from quantlab.data.qdp_v2.pit_history.context import PitHistoryContext
from quantlab.data.qdp_v2.pit_history.download import _historical_st_status
from quantlab.data.qdp_v2.pit_history.lifecycle_audit import audit_symbol_lifecycle_effectivity
from quantlab.data.qdp_v2.pit_history.orchestrator import normalize_symbol_lifecycle_effectivity
from quantlab.data.qdp_v2.pit_history.prepare import _prepare_symbol_parts


def _lifecycle_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    (workspace / "data" / "qdp").mkdir(parents=True)
    return workspace


def _install_lifecycle_domain(
    workspace: Path,
    domain: str,
    frame: pd.DataFrame,
    *,
    primary_key: list[str],
    frequency: str = "1d",
) -> str:
    root = qdp_v2_root(workspace)
    dataset_id = f"{domain}__lifecycle_fixture"
    shard = root / "datasets" / domain / dataset_id / "shards" / "part.parquet"
    shard.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(shard, index=False, engine="pyarrow")
    date_column = "trade_date" if "trade_date" in frame.columns else ""
    start_date = str(frame[date_column].min()) if date_column else ""
    end_date = str(frame[date_column].max()) if date_column else ""
    schema = [
        {"name": str(column), "type": str(frame[column].dtype)}
        for column in frame.columns
    ]
    write_dataset_manifest(
        root,
        DatasetManifest(
            dataset_id=dataset_id,
            domain=domain,
            layer="canonical",
            frequency=frequency,
            contract_version=f"unit_{domain}_v1",
            primary_key=primary_key,
            start_date=start_date,
            end_date=end_date,
            row_count=len(frame),
            shards=[
                ShardManifestEntry(
                    path=shard.relative_to(root).as_posix(),
                    row_count=len(frame),
                    start_date=start_date,
                    end_date=end_date,
                    file_size=shard.stat().st_size,
                )
            ],
            source={"provider": "unit"},
            quality={"primary_key_unique": True},
            schema=schema,
        ),
    )
    return dataset_id


def test_factor_rows_are_past_only_and_normalized_to_first_observation() -> None:
    history = pd.DataFrame(
        {
            "symbol": ["000001.SZ"] * 4,
            "trade_date": [
                "2020-01-02",
                "2020-01-03",
                "2020-01-06",
                "2020-01-07",
            ],
        }
    )
    events = pd.DataFrame(
        {
            "trade_date": ["2019-01-01", "2020-01-06"],
            "back_adjust_factor": [2.0, 3.0],
        }
    )

    result = _factor_rows(history, events)

    assert result["adjust_factor"].tolist() == [1.0, 1.0, 1.5, 1.5]
    assert result["factor_source_date"].tolist() == [
        "2019-01-01",
        "2019-01-01",
        "2020-01-06",
        "2020-01-06",
    ]
    assert (
        pd.to_datetime(result["factor_source_date"])
        <= pd.to_datetime(result["trade_date"])
    ).all()


def test_historical_names_use_only_effective_intervals() -> None:
    dates = pd.Series(["2020-01-02", "2020-02-03", "2020-03-02"])
    intervals = pd.DataFrame(
        {
            "name": ["旧名", "新名"],
            "start_date": ["2010-01-01", "2020-02-01"],
            "end_date": ["2020-01-31", ""],
        }
    )

    assert _historical_names(dates, intervals, fallback="未知").tolist() == [
        "旧名",
        "新名",
        "新名",
    ]


def test_sse_st_status_uses_dated_transitions_before_archive() -> None:
    history = pd.DataFrame(
        {
            "trade_date": [
                "2010-02-26",
                "2010-03-01",
                "2011-10-31",
                "2011-11-01",
                "2012-01-04",
            ],
            "isST": ["", "", "", "", "0"],
        }
    )
    names = pd.Series(["公司"] * len(history))

    result = _historical_st_status(history, symbol="600077.SH", names=names)

    assert result.tolist() == [False, True, True, False, False]


def test_historical_st_name_parser_uses_exchange_prefixes_only() -> None:
    names = pd.Series(
        [
            "ST one",
            "*ST two",
            "SST three",
            "S*ST four",
            "S ST five",
            "GST six",
            "G*ST seven",
            "BEST technology",
            pd.NA,
        ]
    )

    result = _name_implies_st(names)

    assert result.iloc[:7].tolist() == [True] * 7
    assert not bool(result.iloc[7])
    assert pd.isna(result.iloc[8])


def test_historical_st_status_does_not_coerce_unknown_to_false() -> None:
    history = pd.DataFrame({"trade_date": ["2011-01-04"], "isST": [""]})

    result = _historical_st_status(
        history,
        symbol="000001.SZ",
        names=pd.Series([pd.NA]),
    )

    assert str(result.dtype) == "boolean"
    assert pd.isna(result.iloc[0])


def test_eastmoney_history_converts_lots_to_shares() -> None:
    raw = pd.DataFrame(
        {
            "日期": ["2020-01-02"],
            "开盘": [10.0],
            "最高": [10.5],
            "最低": [9.8],
            "收盘": [10.2],
            "成交量": [1234],
            "成交额": [1_250_000.0],
            "换手率": [1.5],
            "涨跌幅": [2.0],
        }
    )

    result = _normalize_eastmoney_history(raw, symbol="600001.SH")

    assert result.loc[0, "volume"] == 123_400.0
    assert result.loc[0, "tradestatus"] == "1"
    assert result.loc[0, "history_source"] == "akshare_eastmoney_unadjusted_history"


def test_sina_factor_provenance_is_explicit() -> None:
    raw = pd.DataFrame(
        {"date": ["1900-01-01", "2020-01-06"], "hfq_factor": [1.0, 1.5]}
    )

    result = _normalize_sina_factors(raw, symbol="600001.SH")

    assert result["factor_provider"].unique().tolist() == ["sina_via_akshare"]
    assert result["source"].unique().tolist() == ["akshare_sina_hfq_factor_event"]


def test_prepare_symbol_parts_keeps_suspension_and_infers_float_shares(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    for directory in (
        "history_parts",
        "factor_parts",
        "share_event_parts",
        "name_interval_parts",
    ):
        (runtime / directory).mkdir(parents=True, exist_ok=True)
    symbol_file = "000033_SZ.parquet"
    history = pd.DataFrame(
        {
            "trade_date": [
                "2020-01-02",
                "2020-01-03",
                "2020-01-06",
                "2020-01-07",
            ],
            "symbol": ["000033.SZ"] * 4,
            "provider_code": ["sz.000033"] * 4,
            "open": [10.0, 10.0, 9.0, 8.0],
            "high": [10.5, 10.0, 9.5, 8.5],
            "low": [9.5, 10.0, 8.5, 7.5],
            "close": [10.0, 10.0, 9.0, 8.0],
            "preclose": [9.8, 10.0, 10.0, 9.0],
            "volume": [1_000_000.0, np.nan, 900_000.0, 800_000.0],
            "amount": [10_000_000.0, np.nan, 8_100_000.0, 6_400_000.0],
            "adjustflag": ["3"] * 4,
            "turn": [1.0, np.nan, 1.0, 1.0],
            "tradestatus": ["1", "0", "1", ""],
            "pctChg": [2.0, 0.0, -10.0, -11.11],
            "peTTM": [10.0, 10.0, 9.0, 8.0],
            "pbMRQ": [1.0, 1.0, 0.9, 0.8],
            "psTTM": [1.0, 1.0, 0.9, 0.8],
            "pcfNcfTTM": [5.0, 5.0, 4.5, 4.0],
            "isST": ["0", "1", "1", "0"],
        }
    )
    factors = pd.DataFrame(
        {
            "symbol": ["000033.SZ"],
            "trade_date": ["2019-01-01"],
            "fore_adjust_factor": [1.0],
            "back_adjust_factor": [1.0],
            "adjust_factor": [1.0],
        }
    )
    history.to_parquet(runtime / "history_parts" / symbol_file, index=False)
    factors.to_parquet(runtime / "factor_parts" / symbol_file, index=False)
    pd.DataFrame(
        columns=[
            "symbol",
            "variation_date",
            "source_date",
            "total_share",
            "float_share",
            "source",
        ]
    ).to_parquet(runtime / "share_event_parts" / symbol_file, index=False)
    pd.DataFrame(
        columns=[
            "symbol",
            "name",
            "start_date",
            "end_date",
            "announcement_date",
            "change_reason",
            "source",
        ]
    ).to_parquet(runtime / "name_interval_parts" / symbol_file, index=False)
    context = PitHistoryContext(
        workspace=tmp_path,
        root=tmp_path / "qdp_v2",
        runtime=runtime,
        start_date="2020-01-01",
        end_date="2020-12-31",
    )

    _prepare_symbol_parts(
        context,
        row={
            "symbol": "000033.SZ",
            "name": "新都退",
            "list_date": "1994-01-03",
            "delist_date": "",
        },
    )

    daily = pd.read_parquet(
        runtime / "domain_parts" / "market_daily_raw" / symbol_file
    )
    status = pd.read_parquet(
        runtime / "domain_parts" / "security_status" / symbol_file
    )
    shares = pd.read_parquet(
        runtime / "domain_parts" / "share_capital" / symbol_file
    )
    assert daily["trade_date"].tolist() == [
        "2020-01-02",
        "2020-01-06",
        "2020-01-07",
    ]
    assert bool(status.loc[1, "is_suspended"])
    assert bool(status.loc[1, "is_st"])
    assert pd.isna(status.loc[3, "is_suspended"])
    assert status.loc[3, "status_reason"] == "status_unknown"
    assert shares["float_share"].tolist() == [
        100_000_000.0,
        90_000_000.0,
        80_000_000.0,
    ]
    assert shares["float_share_source_date"].tolist() == [
        "2020-01-02",
        "2020-01-06",
        "2020-01-07",
    ]

    (runtime / "domain_parts" / "done" / "000033_SZ.json").unlink()
    _prepare_symbol_parts(
        context,
        row={
            "symbol": "000033.SZ",
            "name": "新都退",
            "list_date": "1994-01-03",
            "delist_date": "",
        },
        identity_already_known=True,
    )
    assert pd.read_parquet(
        runtime / "domain_parts" / "security_identity" / symbol_file
    ).empty
    assert pd.read_parquet(
        runtime / "domain_parts" / "symbol_history" / symbol_file
    ).empty


def test_symbol_lifecycle_normalization_is_complete_deterministic_and_idempotent(
    tmp_path: Path,
) -> None:
    workspace = _lifecycle_workspace(tmp_path)
    root = qdp_v2_root(workspace)
    security_id = "QDP-SECURITY-1"
    identity = pd.DataFrame(
        {
            "security_id": [security_id],
            "current_symbol": ["000003.SZ"],
            "list_date": ["2010-01-01"],
        }
    )
    history = pd.DataFrame(
        {
            "security_id": [security_id] * 3,
            "symbol": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "effective_from": ["2010-01-01", "2020-01-01", "2022-01-01"],
            "effective_to": ["2019-12-31", "2021-12-31", "9999-12-31"],
            "name_on_date": ["Old Co", "Mid Co", "New Co"],
        }
    )
    daily = pd.DataFrame(
        {
            "trade_date": [
                "2019-06-03",
                "2019-06-03",
                "2020-06-01",
                "2023-06-01",
                "2023-06-01",
            ],
            "symbol": [
                "000003.SZ",
                "000001.SZ",
                "000003.SZ",
                "000001.SZ",
                "000003.SZ",
            ],
            "open": [30.0, 10.0, 31.0, 12.0, 33.0],
            "high": [30.0, 10.0, 31.0, 12.0, 33.0],
            "low": [30.0, 10.0, 31.0, 12.0, 33.0],
            "close": [30.0, 10.0, 31.0, 12.0, 33.0],
            "volume": [100.0] * 5,
            "amount": [1000.0] * 5,
            "source": ["unit"] * 5,
        }
    )
    universe = pd.DataFrame(
        {
            "trade_date": ["2019-06-03", "2020-06-01", "2023-06-01"],
            "symbol": ["000003.SZ", "000003.SZ", "000001.SZ"],
            "name": ["New Co", "New Co", "Old Co"],
            "list_date": ["2022-01-01", "2022-01-01", "2010-01-01"],
            "delist_date": ["", "", "2020-01-01"],
            "source": ["unit"] * 3,
        }
    )
    intraday = pd.DataFrame(
        {
            "trade_date": ["2019-06-03"],
            "symbol": ["000003.SZ"],
            "bar_time": ["093500000"],
            "close": [30.0],
        }
    )
    datasets = {
        "security_identity": _install_lifecycle_domain(
            workspace,
            "security_identity",
            identity,
            primary_key=["security_id"],
            frequency="static",
        ),
        "symbol_history": _install_lifecycle_domain(
            workspace,
            "symbol_history",
            history,
            primary_key=["security_id", "symbol", "effective_from"],
            frequency="event",
        ),
        "market_daily_raw": _install_lifecycle_domain(
            workspace,
            "market_daily_raw",
            daily,
            primary_key=["trade_date", "symbol"],
        ),
        "universe_snapshot": _install_lifecycle_domain(
            workspace,
            "universe_snapshot",
            universe,
            primary_key=["trade_date", "symbol"],
        ),
        "market_intraday_5m": _install_lifecycle_domain(
            workspace,
            "market_intraday_5m",
            intraday,
            primary_key=["trade_date", "symbol", "bar_time"],
            frequency="5m",
        ),
    }
    write_active_manifest(
        root,
        {
            "version": 2,
            "active_as_of_date": "2023-06-01",
            "datasets": datasets,
        },
    )
    domains = ("market_daily_raw", "universe_snapshot")
    contexts_before = {
        domain: resolve_active_domain(domain, workspace_root=workspace)
        for domain in (*domains, "market_intraday_5m")
    }
    files_before = {
        domain: {
            "manifest": context.manifest_path.read_bytes(),
            "shards": [path.read_bytes() for path in context.shard_paths],
        }
        for domain, context in contexts_before.items()
    }

    audit_before = audit_symbol_lifecycle_effectivity(
        workspace_root=workspace,
        domains=domains,
    )
    assert audit_before["status"] == "needs_repair"
    assert audit_before["multi_symbol_identity_count"] == 1
    assert audit_before["transition_count"] == 2

    dry_run = normalize_symbol_lifecycle_effectivity(
        workspace_root=workspace,
        domains=domains,
        apply=False,
    )

    assert dry_run["status"] == "planned"
    assert dry_run["intraday_5m_changed"] is False
    assert dry_run["domains"]["market_daily_raw"]["source_row_count"] == 5
    assert dry_run["domains"]["market_daily_raw"]["canonical_row_count"] == 3
    assert dry_run["domains"]["market_daily_raw"]["deduplicated_row_count"] == 2
    assert dry_run["domains"]["market_daily_raw"]["projected_manifest_row_count"] == 3
    for domain, before in files_before.items():
        context = resolve_active_domain(domain, workspace_root=workspace)
        assert context.manifest_path.read_bytes() == before["manifest"]
        assert [path.read_bytes() for path in context.shard_paths] == before["shards"]

    result = normalize_symbol_lifecycle_effectivity(
        workspace_root=workspace,
        domains=domains,
        apply=True,
    )

    assert result["status"] == "normalized"
    assert result["after"]["status"] == "ok"
    assert result["intraday_5m_changed"] is False
    daily_context = resolve_active_domain("market_daily_raw", workspace_root=workspace)
    normalized_daily = pd.concat(
        [pd.read_parquet(path) for path in daily_context.shard_paths],
        ignore_index=True,
    ).sort_values(["trade_date", "symbol"])
    assert normalized_daily[["trade_date", "symbol", "close"]].to_dict("records") == [
        {"trade_date": "2019-06-03", "symbol": "000001.SZ", "close": 10.0},
        {"trade_date": "2020-06-01", "symbol": "000002.SZ", "close": 31.0},
        {"trade_date": "2023-06-01", "symbol": "000003.SZ", "close": 33.0},
    ]
    universe_context = resolve_active_domain(
        "universe_snapshot", workspace_root=workspace
    )
    normalized_universe = pd.concat(
        [pd.read_parquet(path) for path in universe_context.shard_paths],
        ignore_index=True,
    ).sort_values(["trade_date", "symbol"])
    assert normalized_universe[
        ["trade_date", "symbol", "name", "list_date", "delist_date"]
    ].to_dict("records") == [
        {
            "trade_date": "2019-06-03",
            "symbol": "000001.SZ",
            "name": "Old Co",
            "list_date": "2010-01-01",
            "delist_date": "2020-01-01",
        },
        {
            "trade_date": "2020-06-01",
            "symbol": "000002.SZ",
            "name": "Mid Co",
            "list_date": "2010-01-01",
            "delist_date": "2022-01-01",
        },
        {
            "trade_date": "2023-06-01",
            "symbol": "000003.SZ",
            "name": "New Co",
            "list_date": "2010-01-01",
            "delist_date": "",
        },
    ]
    intraday_context = resolve_active_domain(
        "market_intraday_5m", workspace_root=workspace
    )
    assert intraday_context.manifest_path.read_bytes() == files_before[
        "market_intraday_5m"
    ]["manifest"]
    assert [path.read_bytes() for path in intraday_context.shard_paths] == files_before[
        "market_intraday_5m"
    ]["shards"]

    normalized_manifests = {
        domain: resolve_active_domain(
            domain, workspace_root=workspace
        ).manifest_path.read_bytes()
        for domain in domains
    }
    replay = normalize_symbol_lifecycle_effectivity(
        workspace_root=workspace,
        domains=domains,
        apply=True,
    )
    assert replay["status"] == "already_normalized"
    assert normalized_manifests == {
        domain: resolve_active_domain(
            domain, workspace_root=workspace
        ).manifest_path.read_bytes()
        for domain in domains
    }


def test_update_cli_lifecycle_dry_run_and_mode_exclusion(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[bool] = []

    def fake_normalize(**kwargs: object) -> dict[str, object]:
        calls.append(bool(kwargs["apply"]))
        return {"status": "planned", "domains": {}}

    monkeypatch.setattr(
        update_module,
        "normalize_symbol_lifecycle_effectivity",
        fake_normalize,
    )

    assert update_module.main(
        ["--normalize-symbol-lifecycle", "--dry-run", "--json"]
    ) == 0
    assert calls == [False]
    assert json.loads(capsys.readouterr().out)["status"] == "planned"
    with pytest.raises(ValueError, match="mutually_exclusive"):
        update_module.main(
            ["--normalize-symbol-lifecycle", "--restore-pit-history"]
        )


def test_lifecycle_normalization_projects_heterogeneous_shards_to_manifest_schema(
    tmp_path: Path,
) -> None:
    workspace = _lifecycle_workspace(tmp_path)
    root = qdp_v2_root(workspace)
    security_id = "QDP-SECURITY-HETEROGENEOUS"
    identity_id = _install_lifecycle_domain(
        workspace,
        "security_identity",
        pd.DataFrame(
            {
                "security_id": [security_id],
                "current_symbol": ["000002.SZ"],
                "list_date": ["2010-01-01"],
            }
        ),
        primary_key=["security_id"],
        frequency="static",
    )
    history_id = _install_lifecycle_domain(
        workspace,
        "symbol_history",
        pd.DataFrame(
            {
                "security_id": [security_id, security_id],
                "symbol": ["000001.SZ", "000002.SZ"],
                "effective_from": ["2010-01-01", "2020-01-01"],
                "effective_to": ["2019-12-31", "9999-12-31"],
                "name_on_date": ["Old Co", "New Co"],
            }
        ),
        primary_key=["security_id", "symbol", "effective_from"],
        frequency="event",
    )
    columns = {
        "trade_date": ["2019-01-02"],
        "symbol": ["000002.SZ"],
        "open": [10.0],
        "high": [10.0],
        "low": [10.0],
        "close": [10.0],
        "volume": [100.0],
        "amount": [1000.0],
        "source": ["unit"],
    }
    daily_id = _install_lifecycle_domain(
        workspace,
        "market_daily_raw",
        pd.DataFrame(columns),
        primary_key=["trade_date", "symbol"],
    )
    manifest_path = dataset_manifest_for_id(root, daily_id, "market_daily_raw")
    assert manifest_path is not None
    manifest = read_dataset_manifest(manifest_path)
    extra = pd.DataFrame(
        {
            "trade_date": ["2019-01-03", "2019-01-03"],
            "symbol": ["000002.SZ", "999999.SZ"],
            "semantic_version": ["restore_v5", "restore_v5"],
            "open": [11.0, 20.0],
            "high": [11.0, 20.0],
            "low": [11.0, 20.0],
            "close": [11.0, 20.0],
            "volume": [100.0, 100.0],
            "amount": [1000.0, 2000.0],
            "source": ["unit", "unit"],
        }
    )
    extra_path = manifest_path.parent / "shards" / "pit_restore.parquet"
    extra.to_parquet(extra_path, index=False, engine="pyarrow")
    write_dataset_manifest(
        root,
        replace(
            manifest,
            row_count=3,
            end_date="2019-01-03",
            shards=[
                *manifest.shards,
                ShardManifestEntry(
                    path=extra_path.relative_to(root).as_posix(),
                    row_count=2,
                    start_date="2019-01-03",
                    end_date="2019-01-03",
                    file_size=extra_path.stat().st_size,
                ),
            ],
        ),
    )
    write_active_manifest(
        root,
        {
            "version": 2,
            "active_as_of_date": "2019-01-03",
            "datasets": {
                "security_identity": identity_id,
                "symbol_history": history_id,
                "market_daily_raw": daily_id,
            },
        },
    )

    dry_run = normalize_symbol_lifecycle_effectivity(
        workspace_root=workspace,
        domains=("market_daily_raw",),
        apply=False,
    )
    assert dry_run["status"] == "planned"
    assert dry_run["domains"]["market_daily_raw"][
        "projected_manifest_row_count"
    ] == 3

    result = normalize_symbol_lifecycle_effectivity(
        workspace_root=workspace,
        domains=("market_daily_raw",),
        apply=True,
    )
    assert result["status"] == "normalized"
    context = resolve_active_domain("market_daily_raw", workspace_root=workspace)
    normalized = pd.concat(
        [pd.read_parquet(path) for path in context.shard_paths],
        ignore_index=True,
    ).sort_values(["trade_date", "symbol"])
    assert list(normalized.columns) == list(columns)
    assert normalized[["trade_date", "symbol", "close"]].to_dict("records") == [
        {"trade_date": "2019-01-02", "symbol": "000001.SZ", "close": 10.0},
        {"trade_date": "2019-01-03", "symbol": "000001.SZ", "close": 11.0},
        {"trade_date": "2019-01-03", "symbol": "999999.SZ", "close": 20.0},
    ]


def test_lifecycle_normalization_aligns_remapped_factor_scale_from_overlap(
    tmp_path: Path,
) -> None:
    workspace = _lifecycle_workspace(tmp_path)
    root = qdp_v2_root(workspace)
    security_id = "QDP-SECURITY-FACTOR"
    identity_id = _install_lifecycle_domain(
        workspace,
        "security_identity",
        pd.DataFrame(
            {
                "security_id": [security_id],
                "current_symbol": ["000002.SZ"],
                "list_date": ["2010-01-01"],
            }
        ),
        primary_key=["security_id"],
        frequency="static",
    )
    history_id = _install_lifecycle_domain(
        workspace,
        "symbol_history",
        pd.DataFrame(
            {
                "security_id": [security_id, security_id],
                "symbol": ["000001.SZ", "000002.SZ"],
                "effective_from": ["2010-01-01", "2020-01-01"],
                "effective_to": ["2019-12-31", "9999-12-31"],
                "name_on_date": ["Old Co", "New Co"],
            }
        ),
        primary_key=["security_id", "symbol", "effective_from"],
        frequency="event",
    )
    factor_id = _install_lifecycle_domain(
        workspace,
        "adjust_factor",
        pd.DataFrame(
            {
                "symbol": [
                    "000002.SZ",
                    "000001.SZ",
                    "000002.SZ",
                    "000002.SZ",
                ],
                "trade_date": [
                    "2019-01-02",
                    "2019-01-02",
                    "2019-01-03",
                    "2020-01-02",
                ],
                "fore_adjust_factor": [1.0, 2.0, 1.1, 2.3],
                "back_adjust_factor": [1.0, 2.0, 1.1, 2.3],
                "adjust_factor": [1.0, 2.0, 1.1, 2.3],
                "source": ["restore", "official", "restore", "official"],
            }
        ),
        primary_key=["trade_date", "symbol"],
    )
    write_active_manifest(
        root,
        {
            "version": 2,
            "active_as_of_date": "2020-01-02",
            "datasets": {
                "security_identity": identity_id,
                "symbol_history": history_id,
                "adjust_factor": factor_id,
            },
        },
    )

    result = normalize_symbol_lifecycle_effectivity(
        workspace_root=workspace,
        domains=("adjust_factor",),
        apply=True,
    )

    assert result["status"] == "normalized"
    context = resolve_active_domain("adjust_factor", workspace_root=workspace)
    normalized = pd.concat(
        [pd.read_parquet(path) for path in context.shard_paths],
        ignore_index=True,
    ).sort_values(["trade_date", "symbol"])
    assert normalized[["trade_date", "symbol", "adjust_factor"]].to_dict(
        "records"
    ) == [
        {
            "trade_date": "2019-01-02",
            "symbol": "000001.SZ",
            "adjust_factor": 2.0,
        },
        {
            "trade_date": "2019-01-03",
            "symbol": "000001.SZ",
            "adjust_factor": 2.2,
        },
        {
            "trade_date": "2020-01-02",
            "symbol": "000002.SZ",
            "adjust_factor": 2.3,
        },
    ]
    assert normalized.loc[
        normalized["trade_date"].eq("2019-01-03"), "source"
    ].item().endswith("+lifecycle_factor_scale_overlap_v1")


def test_lifecycle_normalization_stitches_existing_restored_factor_segment(
    tmp_path: Path,
) -> None:
    workspace = _lifecycle_workspace(tmp_path)
    root = qdp_v2_root(workspace)
    security_id = "QDP-SECURITY-FACTOR-STITCH"
    identity_id = _install_lifecycle_domain(
        workspace,
        "security_identity",
        pd.DataFrame(
            {
                "security_id": [security_id],
                "current_symbol": ["000002.SZ"],
                "list_date": ["2010-01-01"],
            }
        ),
        primary_key=["security_id"],
        frequency="static",
    )
    history_id = _install_lifecycle_domain(
        workspace,
        "symbol_history",
        pd.DataFrame(
            {
                "security_id": [security_id, security_id],
                "symbol": ["000001.SZ", "000002.SZ"],
                "effective_from": ["2010-01-01", "2020-01-01"],
                "effective_to": ["2019-12-31", "9999-12-31"],
                "name_on_date": ["Old Co", "New Co"],
            }
        ),
        primary_key=["security_id", "symbol", "effective_from"],
        frequency="event",
    )
    factor_id = _install_lifecycle_domain(
        workspace,
        "adjust_factor",
        pd.DataFrame(
            {
                "symbol": [
                    "000001.SZ",
                    "000001.SZ",
                    "000001.SZ",
                    "000002.SZ",
                ],
                "trade_date": [
                    "2011-12-30",
                    "2012-01-04",
                    "2019-12-30",
                    "2020-01-02",
                ],
                "fore_adjust_factor": [2.0, 1.0, 1.5, 3.0],
                "back_adjust_factor": [2.0, 1.0, 1.5, 3.0],
                "adjust_factor": [2.0, 1.0, 1.5, 3.0],
                "source": [
                    "official",
                    "akshare+pit_history_restore",
                    "akshare+pit_history_restore",
                    "official",
                ],
            }
        ),
        primary_key=["trade_date", "symbol"],
    )
    write_active_manifest(
        root,
        {
            "version": 2,
            "active_as_of_date": "2020-01-02",
            "datasets": {
                "security_identity": identity_id,
                "symbol_history": history_id,
                "adjust_factor": factor_id,
            },
        },
    )

    dry_run = normalize_symbol_lifecycle_effectivity(
        workspace_root=workspace,
        domains=("adjust_factor",),
        apply=False,
    )
    assert dry_run["status"] == "planned"
    assert dry_run["factor_scale"]["plans"][0]["scale"] == 2.0

    result = normalize_symbol_lifecycle_effectivity(
        workspace_root=workspace,
        domains=("adjust_factor",),
        apply=True,
    )
    assert result["status"] == "normalized"
    assert result["factor_scale"]["status"] == "mutated"
    context = resolve_active_domain("adjust_factor", workspace_root=workspace)
    normalized = pd.concat(
        [pd.read_parquet(path) for path in context.shard_paths],
        ignore_index=True,
    ).sort_values(["trade_date", "symbol"])
    assert normalized["adjust_factor"].tolist() == [2.0, 2.0, 3.0, 3.0]
    assert normalized.loc[
        normalized["trade_date"].isin(["2012-01-04", "2019-12-30"]),
        "source",
    ].str.endswith("+lifecycle_factor_scale_stitch_v1").all()

    replay = normalize_symbol_lifecycle_effectivity(
        workspace_root=workspace,
        domains=("adjust_factor",),
        apply=True,
    )
    assert replay["status"] == "already_normalized"
    assert replay["factor_scale"]["status"] == "already_aligned"
