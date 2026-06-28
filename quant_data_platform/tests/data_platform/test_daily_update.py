from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import quant_data_platform.ingest.daily_update as daily_update_module
from quant_data_platform.core.paths import qdp_paths
from quant_data_platform.core.registry import load_root_manifest, write_root_manifest_update
from quant_data_platform.domains.contracts import DataDomain, DomainFetchRequest, ProviderResult
from quant_data_platform.ingest.daily_update import (
    DEFAULT_DAILY_UPDATE_DOMAINS,
    DailyUpdateConfig,
    _duplicate_key_report,
    _read_dataset_frame,
    build_parser,
    run_daily_update,
)
from quant_data_platform.lake.canonical import write_canonical_manifest
from quant_data_platform.lake.catalog import ResearchDataLake
from quant_data_platform.provider_manager import InMemoryDomainProvider


def _make_workspace(tmp_path: Path):
    brain = tmp_path / "brain"
    brain.mkdir(parents=True)
    (brain / "brain_manifest.json").write_text(json.dumps({"brain_type": "main"}), encoding="utf-8")
    return qdp_paths(tmp_path)


def _market_frame(dates: list[str]) -> pd.DataFrame:
    rows = []
    values = {
        "000001.SZ": (10.0, 10.5, 9.8, 10.2, 1000, 10200),
        "600000.SH": (20.0, 20.5, 19.8, 20.2, 2000, 40400),
        "000300.SH": (4000.0, 4010.0, 3990.0, 4005.0, 3000, 12015000),
    }
    for trade_date in dates:
        for symbol, (open_, high, low, close, volume, amount) in values.items():
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "amount": amount,
                    "source": "unit",
                    "adjusted_flag": "none",
                }
            )
    return pd.DataFrame(rows)


def _calendar_frame(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": dates,
            "is_open": [True] * len(dates),
            "exchange": ["SSE"] * len(dates),
            "source": ["unit"] * len(dates),
        }
    )


def _universe_frame(dates: list[str]) -> pd.DataFrame:
    rows = []
    for trade_date in dates:
        for symbol, name in {"000001.SZ": "平安银行", "600000.SH": "浦发银行"}.items():
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "name": name,
                    "exchange": "SH" if symbol.endswith(".SH") else "SZ",
                    "board": "main",
                    "list_status": "L",
                    "list_date": "1991-04-03",
                    "delist_date": "",
                    "source": "unit",
                }
            )
    return pd.DataFrame(rows)


def _status_frame(dates: list[str]) -> pd.DataFrame:
    rows = []
    for trade_date in dates:
        for symbol in ["000001.SZ", "600000.SH"]:
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "is_st": False,
                    "is_suspended": False,
                    "is_delisted": False,
                    "status_reason": "",
                    "source": "unit",
                }
            )
    return pd.DataFrame(rows)


def _industry_frame(dates: list[str]) -> pd.DataFrame:
    rows = []
    for trade_date in dates:
        rows.extend(
            [
                {
                    "symbol": "000001.SZ",
                    "trade_date": trade_date,
                    "industry": "bank",
                    "concept_tags": "finance;large_cap",
                    "source": "unit",
                },
                {
                    "symbol": "600000.SH",
                    "trade_date": trade_date,
                    "industry": "bank",
                    "concept_tags": "finance",
                    "source": "unit",
                },
            ]
        )
    return pd.DataFrame(rows)


def _index_constituents_frame(dates: list[str]) -> pd.DataFrame:
    rows = []
    for trade_date in dates:
        rows.extend(
            [
                {
                    "index_symbol": "000300.SH",
                    "symbol": "000001.SZ",
                    "trade_date": trade_date,
                    "index_name": "沪深300",
                    "source": "unit",
                },
                {
                    "index_symbol": "000016.SH",
                    "symbol": "000001.SZ",
                    "trade_date": trade_date,
                    "index_name": "上证50",
                    "source": "unit",
                },
            ]
        )
    return pd.DataFrame(rows)


def _save_active_bundle(lake: ResearchDataLake, paths, dates: list[str]) -> dict[str, str]:
    calendar = lake.save_domain_dataset(
        domain=DataDomain.TRADING_CALENDAR,
        frame=_calendar_frame(dates),
        spec={"dataset": "data_platform_trading_calendar", "source": "unit", "start_date": dates[0], "end_date": dates[-1]},
        source="unit",
        reuse=False,
    )
    universe = lake.save_domain_dataset(
        domain=DataDomain.UNIVERSE_SNAPSHOT,
        frame=_universe_frame(dates),
        spec={"dataset": "data_platform_universe_snapshot", "source": "unit", "start_date": dates[0], "end_date": dates[-1]},
        source="unit",
        reuse=False,
    )
    status = lake.save_domain_dataset(
        domain=DataDomain.SECURITY_STATUS,
        frame=_status_frame(dates),
        spec={"dataset": "data_platform_security_status", "source": "unit", "start_date": dates[0], "end_date": dates[-1]},
        source="unit",
        reuse=False,
    )
    market = _market_frame(dates)
    market["trade_date"] = pd.to_datetime(market["trade_date"])
    stocks = market.loc[market["symbol"] != "000300.SH"].copy()
    benchmark = market.loc[market["symbol"] == "000300.SH"].sort_values("trade_date").copy()
    market_frames = {
        name: stocks.pivot_table(index="trade_date", columns="symbol", values=field, aggfunc="last").sort_index()
        for name, field in {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Amount": "amount",
        }.items()
    }
    sidecars = {
        DataDomain.TRADING_CALENDAR: calendar.dataset_id,
        DataDomain.UNIVERSE_SNAPSHOT: universe.dataset_id,
        DataDomain.SECURITY_STATUS: status.dataset_id,
    }
    bundle = lake.save_market_data_bundle(
        spec={
            "dataset": "policy_input_bundle",
            "source": "unit_active",
            "start_date": dates[0],
            "end_date": dates[-1],
            "benchmark": "000300.SH",
            "sidecar_dataset_ids": sidecars,
        },
        market_frames=market_frames,
        benchmark_close=pd.Series(benchmark["close"].to_numpy(dtype=float), index=benchmark["trade_date"], name="000300.SH"),
        benchmark_open=pd.Series(benchmark["open"].to_numpy(dtype=float), index=benchmark["trade_date"], name="000300.SH"),
        membership_frame=market_frames["Close"].notna().astype(bool),
        feature_frames={},
        source="unit_active",
        reuse=False,
    )
    manifest_path = write_canonical_manifest(
        lake,
        dataset_id=bundle.dataset_id,
        start_date=dates[0],
        end_date=dates[-1],
        sidecar_dataset_ids=sidecars,
        notes="unit active",
    )
    write_root_manifest_update(
        {
            "status": "unit_active",
            "canonical_dataset_id": bundle.dataset_id,
            "canonical_manifest": str(manifest_path),
            "canonical_start_date": dates[0],
            "canonical_component_dataset_ids": {DataDomain.MARKET_DAILY: bundle.dataset_id, **sidecars},
            "canonical_bundle_sidecar_dataset_ids": sidecars,
        },
        paths=paths,
    )
    return {"bundle": bundle.dataset_id, **sidecars}


def _save_industry_dataset(lake: ResearchDataLake, dates: list[str]) -> str:
    record = lake.save_domain_dataset(
        domain=DataDomain.INDUSTRY_CONCEPT,
        frame=_industry_frame(dates),
        spec={
            "dataset": "data_platform_industry_concept",
            "source": "unit",
            "start_date": dates[0],
            "end_date": dates[-1],
        },
        source="unit",
        reuse=False,
    )
    return record.dataset_id


def _attach_active_sidecar(paths, domain: str, dataset_id: str) -> None:
    root = load_root_manifest(paths)
    sidecars = dict(root.get("canonical_bundle_sidecar_dataset_ids", {}) or {})
    sidecars[domain] = dataset_id
    components = dict(root.get("canonical_component_dataset_ids", {}) or {})
    components[domain] = dataset_id
    write_root_manifest_update(
        {
            "canonical_bundle_sidecar_dataset_ids": sidecars,
            "canonical_component_dataset_ids": components,
        },
        paths=paths,
    )


def _raw_5m_frame(dates: list[str]) -> pd.DataFrame:
    rows = []
    bar_times = [
        *pd.date_range("09:35", "11:30", freq="5min").strftime("%H%M%S000").tolist(),
        *pd.date_range("13:05", "15:00", freq="5min").strftime("%H%M%S000").tolist(),
    ]
    for trade_date in dates:
        for symbol in ["000001.SZ", "600000.SH"]:
            for idx, bar_time in enumerate(bar_times, start=1):
                close = 10.0 + idx / 100.0
                rows.append(
                    {
                        "symbol": symbol,
                        "trade_date": trade_date,
                        "bar_time": bar_time,
                        "open": close - 0.01,
                        "high": close + 0.02,
                        "low": close - 0.02,
                        "close": close,
                        "volume": 1000 + idx,
                        "amount": (1000 + idx) * close,
                        "source": "unit_raw_5m",
                        "adjusted_flag": "none",
                    }
                )
    return pd.DataFrame(rows)


def _raw_1m_frame(dates: list[str], *, symbols: list[str] | None = None) -> pd.DataFrame:
    rows = []
    bar_times = [
        *pd.date_range("09:31", "11:30", freq="1min").strftime("%H%M%S000").tolist(),
        *pd.date_range("13:01", "15:00", freq="1min").strftime("%H%M%S000").tolist(),
    ]
    for trade_date in dates:
        for symbol in symbols or ["000001.SZ", "600000.SH"]:
            for idx, bar_time in enumerate(bar_times, start=1):
                close = 10.0 + idx / 100.0
                rows.append(
                    {
                        "symbol": symbol,
                        "trade_date": trade_date,
                        "bar_time": bar_time,
                        "open": close - 0.01,
                        "high": close + 0.02,
                        "low": close - 0.02,
                        "close": close,
                        "volume": 1000 + idx,
                        "amount": (1000 + idx) * close,
                        "source": "unit_raw_1m",
                        "adjusted_flag": "none",
                    }
                )
    return pd.DataFrame(rows)


def _attach_active_raw_1m_sidecar(lake: ResearchDataLake, paths, dates: list[str]) -> str:
    spec = {
        "dataset": "data_platform_market_intraday_1m",
        "source": "unit_raw_1m",
        "start_date": dates[0],
        "end_date": dates[-1],
        "sharded": True,
    }
    identity = lake.build_domain_dataset_identity(domain=DataDomain.MARKET_INTRADAY_1M, spec=spec)
    shard_dir = Path(identity["dataset_dir"]) / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / "raw_1m_active.parquet"
    frame = _raw_1m_frame(dates)
    frame.to_parquet(shard_path, index=False)
    record = lake.save_sharded_domain_dataset(
        domain=DataDomain.MARKET_INTRADAY_1M,
        spec=spec,
        shard_records=[
            {
                "status": "stored",
                "chunk_id": "raw_1m_active",
                "task_kind": "unit_raw_1m",
                "start_date": dates[0],
                "end_date": dates[-1],
                "symbol_count": 2,
                "row_count": int(len(frame)),
                "path": str(shard_path.resolve()),
                "error_count": 0,
                "error_report": [],
            }
        ],
        source="unit_raw_1m",
        reuse=False,
    )
    root = load_root_manifest(paths)
    sidecars = dict(root.get("canonical_bundle_sidecar_dataset_ids", {}) or {})
    sidecars[DataDomain.MARKET_INTRADAY_1M] = record.dataset_id
    components = dict(root.get("canonical_component_dataset_ids", {}) or {})
    components[DataDomain.MARKET_INTRADAY_1M] = record.dataset_id
    write_root_manifest_update(
        {
            "canonical_bundle_sidecar_dataset_ids": sidecars,
            "canonical_component_dataset_ids": components,
        },
        paths=paths,
    )
    return record.dataset_id


def _save_raw_1m_dataset(lake: ResearchDataLake, dates: list[str], *, start_date: str, end_date: str, tag: str) -> str:
    spec = {
        "dataset": "data_platform_market_intraday_1m",
        "source": tag,
        "start_date": start_date,
        "end_date": end_date,
        "sharded": True,
    }
    identity = lake.build_domain_dataset_identity(domain=DataDomain.MARKET_INTRADAY_1M, spec=spec)
    shard_dir = Path(identity["dataset_dir"]) / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"{tag}.parquet"
    frame = _raw_1m_frame(dates)
    frame.to_parquet(shard_path, index=False)
    record = lake.save_sharded_domain_dataset(
        domain=DataDomain.MARKET_INTRADAY_1M,
        spec=spec,
        shard_records=[
            {
                "status": "stored",
                "chunk_id": tag,
                "task_kind": tag,
                "start_date": dates[0],
                "end_date": dates[-1],
                "symbol_count": 2,
                "row_count": int(len(frame)),
                "path": str(shard_path.resolve()),
                "error_count": 0,
                "error_report": [],
            }
        ],
        source=tag,
        reuse=False,
    )
    return record.dataset_id


def _save_intraday_feature_dataset(lake: ResearchDataLake, dates: list[str], *, start_date: str, end_date: str, tag: str) -> str:
    spec = {
        "dataset": "data_platform_intraday_daily_features",
        "source": tag,
        "start_date": start_date,
        "end_date": end_date,
        "sharded": True,
    }
    identity = lake.build_domain_dataset_identity(domain=DataDomain.INTRADAY_DAILY_FEATURES, spec=spec)
    shard_dir = Path(identity["dataset_dir"]) / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"{tag}.parquet"
    frame = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "trade_date": trade_date,
                "bar_count": 48,
                "source": tag,
                "adjusted_flag": "none",
            }
            for trade_date in dates
            for symbol in ("000001.SZ", "600000.SH")
        ]
    )
    frame.to_parquet(shard_path, index=False)
    record = lake.save_sharded_domain_dataset(
        domain=DataDomain.INTRADAY_DAILY_FEATURES,
        spec=spec,
        shard_records=[
            {
                "status": "stored",
                "chunk_id": tag,
                "task_kind": tag,
                "start_date": dates[0],
                "end_date": dates[-1],
                "symbol_count": 2,
                "row_count": int(len(frame)),
                "path": str(shard_path.resolve()),
                "error_count": 0,
                "error_report": [],
            }
        ],
        source=tag,
        reuse=False,
    )
    return record.dataset_id


def _attach_active_raw_5m_sidecar(lake: ResearchDataLake, paths, dates: list[str]) -> str:
    spec = {
        "dataset": "data_platform_market_intraday_5m",
        "source": "unit_raw_5m",
        "start_date": dates[0],
        "end_date": dates[-1],
        "sharded": True,
    }
    identity = lake.build_domain_dataset_identity(domain=DataDomain.MARKET_INTRADAY_5M, spec=spec)
    shard_dir = Path(identity["dataset_dir"]) / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / "raw_5m_tail.parquet"
    frame = _raw_5m_frame(dates)
    frame.to_parquet(shard_path, index=False)
    record = lake.save_sharded_domain_dataset(
        domain=DataDomain.MARKET_INTRADAY_5M,
        spec=spec,
        shard_records=[
            {
                "status": "stored",
                "chunk_id": "raw_5m_tail",
                "task_kind": "unit_raw_5m",
                "start_date": dates[0],
                "end_date": dates[-1],
                "symbol_count": 2,
                "row_count": int(len(frame)),
                "path": str(shard_path.resolve()),
                "error_count": 0,
                "error_report": [],
            }
        ],
        source="unit_raw_5m",
        reuse=False,
    )
    root = load_root_manifest(paths)
    sidecars = dict(root.get("canonical_bundle_sidecar_dataset_ids", {}) or {})
    sidecars[DataDomain.MARKET_INTRADAY_5M] = record.dataset_id
    components = dict(root.get("canonical_component_dataset_ids", {}) or {})
    components[DataDomain.MARKET_INTRADAY_5M] = record.dataset_id
    write_root_manifest_update(
        {
            "canonical_bundle_sidecar_dataset_ids": sidecars,
            "canonical_component_dataset_ids": components,
        },
        paths=paths,
    )
    return record.dataset_id


class DailyBackfillProvider:
    name = "baostock"

    def fetch_domain(self, request: DomainFetchRequest) -> ProviderResult:
        request = request.normalized()
        dates = pd.bdate_range(request.start_date, request.end_date).strftime("%Y-%m-%d").tolist()
        if request.domain == DataDomain.TRADING_CALENDAR:
            frame = _calendar_frame(dates)
        elif request.domain == DataDomain.UNIVERSE_SNAPSHOT:
            frame = _universe_frame([request.end_date])
        elif request.domain == DataDomain.SECURITY_STATUS:
            frame = _status_frame([request.end_date])
        else:
            frame = pd.DataFrame()
        return ProviderResult(provider=self.name, data=frame)


class IntradayBackfillProvider(DailyBackfillProvider):
    name = "qdp_production_v1"

    def fetch_domain(self, request: DomainFetchRequest) -> ProviderResult:
        request = request.normalized()
        if request.domain == DataDomain.MARKET_INTRADAY_5M:
            dates = pd.bdate_range(request.start_date, request.end_date).strftime("%Y-%m-%d").tolist()
            frame = _raw_5m_frame(dates)
            if request.symbols:
                frame = frame.loc[frame["symbol"].isin(request.symbols)].copy()
            return ProviderResult(provider=self.name, data=frame)
        return super().fetch_domain(request)


class Intraday1mBackfillProvider(DailyBackfillProvider):
    name = "qdp_production_v1"

    def __init__(self) -> None:
        self.requests: list[DomainFetchRequest] = []

    def fetch_domain(self, request: DomainFetchRequest) -> ProviderResult:
        request = request.normalized()
        self.requests.append(request)
        if request.domain == DataDomain.MARKET_INTRADAY_1M:
            dates = pd.bdate_range(request.start_date, request.end_date).strftime("%Y-%m-%d").tolist()
            frame = _raw_1m_frame(dates, symbols=list(request.symbols or ["000001.SZ", "600000.SH"]))
            return ProviderResult(provider=self.name, data=frame)
        return super().fetch_domain(request)


class CalendarTailBackfillProvider:
    name = "baostock"

    def __init__(self) -> None:
        self.snapshot_dates: dict[str, list[str]] = {
            DataDomain.UNIVERSE_SNAPSHOT: [],
            DataDomain.SECURITY_STATUS: [],
        }

    def fetch_domain(self, request: DomainFetchRequest) -> ProviderResult:
        request = request.normalized()
        if request.domain == DataDomain.TRADING_CALENDAR:
            frame = _calendar_frame(["2026-01-07", "2026-01-09"])
        elif request.domain == DataDomain.UNIVERSE_SNAPSHOT:
            self.snapshot_dates[DataDomain.UNIVERSE_SNAPSHOT].append(request.end_date)
            frame = _universe_frame([request.end_date])
        elif request.domain == DataDomain.SECURITY_STATUS:
            self.snapshot_dates[DataDomain.SECURITY_STATUS].append(request.end_date)
            frame = _status_frame([request.end_date])
        else:
            frame = pd.DataFrame()
        return ProviderResult(provider=self.name, data=frame)


class FailingBackfillProvider:
    name = "baostock"

    def fetch_domain(self, request: DomainFetchRequest) -> ProviderResult:
        raise AssertionError(f"daily_update should not call provider backfill for {request.domain}")


def test_daily_update_parser_accepts_production_flags() -> None:
    args = build_parser().parse_args(
        [
            "--as-of-date",
            "2026-06-26",
            "--readiness-mode",
            "strict",
            "--build-policy-bundle",
            "--build-v2-status-sidecar",
            "--build-memmap",
            "--activate",
        ]
    )

    assert args.as_of_date == "2026-06-26"
    assert args.build_policy_bundle is True
    assert args.activate is True
    assert args.intraday_features_mode == "auto"
    assert args.task_workers == 2
    assert args.snapshot_workers == 2
    assert args.adjust_factor_workers == 2


def test_daily_update_dry_run_identifies_required_pit_tail_gap(tmp_path: Path) -> None:
    paths = _make_workspace(tmp_path)
    lake = ResearchDataLake(paths.lake_root)
    _save_active_bundle(lake, paths, ["2026-01-05", "2026-01-06"])

    result = run_daily_update(
        DailyUpdateConfig(
            workspace_root=paths.workspace_root,
            run_id="unit_daily_plan",
            as_of_date="2026-01-08",
            domains=(DataDomain.TRADING_CALENDAR, DataDomain.UNIVERSE_SNAPSHOT, DataDomain.SECURITY_STATUS),
            dry_run=True,
        ),
        paths=paths,
    )

    assert result.status == "planned"
    assert result.domain_plans[DataDomain.UNIVERSE_SNAPSHOT].gap_start_date == "2026-01-07"
    assert result.domain_plans[DataDomain.UNIVERSE_SNAPSHOT].planned_task_count == 2
    assert result.domain_plans[DataDomain.SECURITY_STATUS].planned_task_count == 2
    assert result.domain_plans[DataDomain.TRADING_CALENDAR].planned_task_count == 1


def test_daily_update_fallback_prefers_full_history_intraday_dataset_over_tail_only(tmp_path: Path) -> None:
    paths = _make_workspace(tmp_path)
    lake = ResearchDataLake(paths.lake_root)
    _save_active_bundle(lake, paths, ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"])
    full_history_id = _save_raw_1m_dataset(
        lake,
        ["2025-12-31"],
        start_date="2010-01-01",
        end_date="2025-12-31",
        tag="unit_full_history_1m",
    )
    _save_raw_1m_dataset(
        lake,
        ["2026-01-06"],
        start_date="2026-01-01",
        end_date="2026-03-27",
        tag="unit_tail_only_1m",
    )

    result = run_daily_update(
        DailyUpdateConfig(
            workspace_root=paths.workspace_root,
            run_id="unit_1m_fallback_prefers_full_history",
            as_of_date="2026-01-08",
            start_date="2026-01-05",
            domains=(DataDomain.MARKET_INTRADAY_1M,),
            external_intraday_1m_mode="skip",
            dry_run=True,
        ),
        paths=paths,
    )

    plan = result.domain_plans[DataDomain.MARKET_INTRADAY_1M]
    assert plan.inherited_dataset_id == full_history_id
    assert plan.gap_start_date == "2026-01-01"


def test_daily_update_fallback_treats_first_trading_day_feature_dataset_as_full_history(tmp_path: Path) -> None:
    paths = _make_workspace(tmp_path)
    lake = ResearchDataLake(paths.lake_root)
    _save_active_bundle(lake, paths, ["2026-06-10", "2026-06-26"])
    _save_intraday_feature_dataset(
        lake,
        ["2010-01-04", "2026-06-10"],
        start_date="2010-01-01",
        end_date="2026-06-10",
        tag="unit_old_feature_metadata_starts_calendar_day",
    )
    _save_intraday_feature_dataset(
        lake,
        ["2026-06-11", "2026-06-26"],
        start_date="2026-06-11",
        end_date="2026-06-26",
        tag="unit_tail_feature_only",
    )
    full_history_id = _save_intraday_feature_dataset(
        lake,
        ["2010-01-04", "2026-06-26"],
        start_date="2010-01-04",
        end_date="2026-06-26",
        tag="unit_new_feature_first_trading_day",
    )

    result = run_daily_update(
        DailyUpdateConfig(
            workspace_root=paths.workspace_root,
            run_id="unit_feature_first_trading_day_full_history",
            as_of_date="2026-06-26",
            domains=(DataDomain.INTRADAY_DAILY_FEATURES,),
            intraday_features_mode="skip",
            dry_run=True,
        ),
        paths=paths,
    )

    plan = result.domain_plans[DataDomain.INTRADAY_DAILY_FEATURES]
    assert plan.inherited_dataset_id == full_history_id
    assert plan.needs_backfill is False


def test_daily_update_recomputes_pit_dates_after_calendar_tail(tmp_path: Path) -> None:
    paths = _make_workspace(tmp_path)
    lake = ResearchDataLake(paths.lake_root)
    _save_active_bundle(lake, paths, ["2026-01-05"])
    calendar = lake.save_domain_dataset(
        domain=DataDomain.TRADING_CALENDAR,
        frame=_calendar_frame(["2026-01-05", "2026-01-06"]),
        spec={"dataset": "data_platform_trading_calendar", "source": "unit", "start_date": "2026-01-05", "end_date": "2026-01-06"},
        source="unit",
        reuse=False,
    )
    root = load_root_manifest(paths)
    sidecars = dict(root.get("canonical_bundle_sidecar_dataset_ids", {}) or {})
    sidecars[DataDomain.TRADING_CALENDAR] = calendar.dataset_id
    components = dict(root.get("canonical_component_dataset_ids", {}) or {})
    components[DataDomain.TRADING_CALENDAR] = calendar.dataset_id
    write_root_manifest_update(
        {
            "canonical_bundle_sidecar_dataset_ids": sidecars,
            "canonical_component_dataset_ids": components,
        },
        paths=paths,
    )
    provider = CalendarTailBackfillProvider()

    result = run_daily_update(
        DailyUpdateConfig(
            workspace_root=paths.workspace_root,
            run_id="unit_calendar_tail_replan",
            as_of_date="2026-01-09",
            start_date="2026-01-05",
            domains=(DataDomain.TRADING_CALENDAR, DataDomain.UNIVERSE_SNAPSHOT, DataDomain.SECURITY_STATUS),
            required_pit_domains=(DataDomain.TRADING_CALENDAR, DataDomain.UNIVERSE_SNAPSHOT, DataDomain.SECURITY_STATUS),
            task_workers=1,
            snapshot_workers=1,
            max_duplicate_check_rows=10000,
        ),
        paths=paths,
        backfill_provider=provider,
    )

    expected_tail = ("2026-01-06", "2026-01-07", "2026-01-09")
    assert result.status == "ok"
    assert result.blockers == []
    assert result.domain_plans[DataDomain.UNIVERSE_SNAPSHOT].expected_trade_dates == expected_tail
    assert result.domain_plans[DataDomain.SECURITY_STATUS].expected_trade_dates == expected_tail
    assert provider.snapshot_dates[DataDomain.UNIVERSE_SNAPSHOT] == list(expected_tail)
    assert provider.snapshot_dates[DataDomain.SECURITY_STATUS] == list(expected_tail)
    assert result.after_coverage[DataDomain.UNIVERSE_SNAPSHOT]["trade_date_count"] == 4
    assert result.after_coverage[DataDomain.SECURITY_STATUS]["trade_date_count"] == 4


def test_daily_update_blocks_intraday_features_when_raw_5m_tail_is_missing(tmp_path: Path) -> None:
    paths = _make_workspace(tmp_path)
    lake = ResearchDataLake(paths.lake_root)
    _save_active_bundle(lake, paths, ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"])

    result = run_daily_update(
        DailyUpdateConfig(
            workspace_root=paths.workspace_root,
            run_id="unit_intraday_skip",
            as_of_date="2026-01-08",
            start_date="2026-01-07",
            domains=(DataDomain.INTRADAY_DAILY_FEATURES,),
        ),
        paths=paths,
        backfill_provider=FailingBackfillProvider(),
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.status == "blocked"
    assert DataDomain.INTRADAY_DAILY_FEATURES not in result.backfill_dataset_ids
    assert result.backfill_results[DataDomain.INTRADAY_DAILY_FEATURES]["action"] == "skip"
    assert result.lagging_domains[DataDomain.INTRADAY_DAILY_FEATURES]["reason"] == "market_intraday_5m_dataset_missing"
    assert f"required_component_missing:{DataDomain.INTRADAY_DAILY_FEATURES}" in result.blockers
    assert manifest["backfill_results"][DataDomain.INTRADAY_DAILY_FEATURES]["action"] == "skip"


def test_daily_update_auto_derives_intraday_features_from_existing_raw_5m_tail(tmp_path: Path) -> None:
    paths = _make_workspace(tmp_path)
    lake = ResearchDataLake(paths.lake_root)
    _save_active_bundle(lake, paths, ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"])
    raw_dataset_id = _attach_active_raw_5m_sidecar(lake, paths, ["2026-01-07", "2026-01-08"])

    result = run_daily_update(
        DailyUpdateConfig(
            workspace_root=paths.workspace_root,
            run_id="unit_intraday_derive",
            as_of_date="2026-01-08",
            start_date="2026-01-07",
            domains=(DataDomain.INTRADAY_DAILY_FEATURES,),
        ),
        paths=paths,
        backfill_provider=FailingBackfillProvider(),
    )
    dataset_id = result.backfill_dataset_ids[DataDomain.INTRADAY_DAILY_FEATURES]
    derived = _read_dataset_frame(lake=lake, dataset_id=dataset_id)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.status == "ok"
    assert result.backfill_results[DataDomain.INTRADAY_DAILY_FEATURES]["action"] == "derive"
    assert result.backfill_results[DataDomain.INTRADAY_DAILY_FEATURES]["source_dataset_id"] == raw_dataset_id
    assert not derived.empty
    assert sorted(derived["trade_date"].unique().tolist()) == ["2026-01-07", "2026-01-08"]
    assert manifest["backfill_results"][DataDomain.INTRADAY_DAILY_FEATURES]["timing"]["chunk_count"] == 1


def test_daily_update_backfills_raw_5m_before_deriving_intraday_features(tmp_path: Path) -> None:
    paths = _make_workspace(tmp_path)
    lake = ResearchDataLake(paths.lake_root)
    _save_active_bundle(lake, paths, ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"])
    _attach_active_raw_5m_sidecar(lake, paths, ["2026-01-05", "2026-01-06"])

    result = run_daily_update(
        DailyUpdateConfig(
            workspace_root=paths.workspace_root,
            run_id="unit_intraday_5m_then_features",
            as_of_date="2026-01-08",
            start_date="2026-01-05",
            domains=(DataDomain.MARKET_INTRADAY_5M, DataDomain.INTRADAY_DAILY_FEATURES),
            chunk_size_symbols=10,
            chunk_size_months=1,
            task_workers=1,
            max_duplicate_check_rows=10000,
        ),
        paths=paths,
        backfill_provider=IntradayBackfillProvider(),
    )
    raw_dataset_id = result.backfill_dataset_ids[DataDomain.MARKET_INTRADAY_5M]
    feature_dataset_id = result.backfill_dataset_ids[DataDomain.INTRADAY_DAILY_FEATURES]
    derived = _read_dataset_frame(lake=lake, dataset_id=feature_dataset_id)

    assert result.status == "ok"
    assert result.blockers == []
    assert result.backfill_results[DataDomain.INTRADAY_DAILY_FEATURES]["action"] == "derive"
    assert result.backfill_results[DataDomain.INTRADAY_DAILY_FEATURES]["source_dataset_id"] == raw_dataset_id
    assert sorted(derived["trade_date"].unique().tolist()) == ["2026-01-07", "2026-01-08"]


def test_daily_update_imports_external_1m_then_fetches_mootdx_tail(tmp_path: Path) -> None:
    paths = _make_workspace(tmp_path)
    lake = ResearchDataLake(paths.lake_root)
    _save_active_bundle(lake, paths, ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"])
    _attach_active_raw_1m_sidecar(lake, paths, ["2026-01-05"])
    source_root = tmp_path / "量化数据"
    csv_dir = source_root / "2026" / "1分钟"
    csv_dir.mkdir(parents=True)
    csv_template = "\n".join(
        [
            "日期,开盘,最高,最低,收盘,成交量(股),成交额(元)",
            "2026-01-06 09:30:00,10.0,10.2,9.9,10.1,100,1010",
            "2026-01-06 09:31:00,10.1,10.3,10.0,10.2,200,2040",
        ]
    )
    (csv_dir / "sz000001.csv").write_text(csv_template, encoding="utf-8")
    (csv_dir / "sh600000.csv").write_text(csv_template, encoding="utf-8")
    provider = Intraday1mBackfillProvider()

    result = run_daily_update(
        DailyUpdateConfig(
            workspace_root=paths.workspace_root,
            run_id="unit_external_1m_then_mootdx",
            as_of_date="2026-01-08",
            start_date="2026-01-05",
            domains=(DataDomain.MARKET_INTRADAY_1M,),
            external_intraday_source_root=source_root,
            include_unpacked_external_csv=True,
            external_intraday_shard_batch_members=8,
            chunk_size_symbols=10,
            chunk_size_months=1,
            task_workers=1,
            max_duplicate_check_rows=10000,
        ),
        paths=paths,
        backfill_provider=provider,
    )

    tail_result = result.backfill_results[DataDomain.MARKET_INTRADAY_1M]
    external_step = tail_result["pre_provider_steps"]["external_intraday_1m"]
    effective = _read_dataset_frame(lake=lake, dataset_id=result.effective_sidecar_dataset_ids[DataDomain.MARKET_INTRADAY_1M])

    assert result.status == "ok"
    assert result.blockers == []
    assert external_step["end_date"] == "2026-01-06"
    assert external_step["remaining_provider_start_date"] == "2026-01-07"
    assert provider.requests[0].start_date == "2026-01-07"
    assert provider.requests[0].end_date == "2026-01-08"
    assert result.backfill_dataset_ids[DataDomain.MARKET_INTRADAY_1M] == tail_result["coalesced_tail"]["dataset_id"]
    assert len(tail_result["coalesced_tail"]["source_dataset_ids"]) == 2
    assert result.after_coverage[DataDomain.MARKET_INTRADAY_1M]["end_date"] == "2026-01-08"
    assert "093000000" in set(effective["bar_time"].astype(str))
    assert sorted(effective["trade_date"].unique().tolist()) == ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]


def test_daily_update_strict_activation_combines_inherited_pit_with_tail(tmp_path: Path) -> None:
    paths = _make_workspace(tmp_path)
    lake = ResearchDataLake(paths.lake_root)
    active_ids = _save_active_bundle(lake, paths, ["2026-01-05", "2026-01-06"])
    refresh_provider = InMemoryDomainProvider(
        "akshare_eastmoney",
        payloads={
            DataDomain.TRADING_CALENDAR: _calendar_frame(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]),
            DataDomain.UNIVERSE_SNAPSHOT: _universe_frame(["2026-01-08"]),
            DataDomain.MARKET_DAILY: _market_frame(["2026-01-07", "2026-01-08"]),
        },
    )

    result = run_daily_update(
        DailyUpdateConfig(
            workspace_root=paths.workspace_root,
            run_id="unit_daily_activate",
            as_of_date="2026-01-08",
            start_date="2026-01-05",
            domains=(DataDomain.MARKET_DAILY, DataDomain.TRADING_CALENDAR, DataDomain.UNIVERSE_SNAPSHOT, DataDomain.SECURITY_STATUS),
            required_pit_domains=(DataDomain.TRADING_CALENDAR, DataDomain.UNIVERSE_SNAPSHOT, DataDomain.SECURITY_STATUS),
            build_policy_bundle=True,
            activate=True,
            universe="all_a",
            max_duplicate_check_rows=10000,
        ),
        paths=paths,
        refresh_providers=[refresh_provider],
        backfill_provider=DailyBackfillProvider(),
    )
    root = load_root_manifest(paths)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.status == "activated"
    assert result.blockers == []
    assert result.canonical_dataset_id
    assert root["canonical_dataset_id"] == result.canonical_dataset_id
    assert root["canonical_component_dataset_ids"][DataDomain.MARKET_DAILY] == result.canonical_dataset_id
    assert "benchmark_daily" not in root["canonical_component_dataset_ids"]
    assert "benchmark_daily" not in root["canonical_bundle_sidecar_dataset_ids"]
    assert result.effective_sidecar_dataset_ids[DataDomain.UNIVERSE_SNAPSHOT] != active_ids[DataDomain.UNIVERSE_SNAPSHOT]
    assert result.effective_sidecar_dataset_ids[DataDomain.SECURITY_STATUS] != active_ids[DataDomain.SECURITY_STATUS]
    assert result.after_coverage[DataDomain.UNIVERSE_SNAPSHOT]["end_date"] == "2026-01-08"
    assert result.after_coverage[DataDomain.SECURITY_STATUS]["trade_date_count"] == 4
    assert manifest["combined_sidecar_dataset_ids"][DataDomain.UNIVERSE_SNAPSHOT] == result.effective_sidecar_dataset_ids[DataDomain.UNIVERSE_SNAPSHOT]


def test_daily_update_default_domains_match_canonical_feature_policy() -> None:
    assert DataDomain.INDUSTRY_CONCEPT in DEFAULT_DAILY_UPDATE_DOMAINS
    assert DataDomain.INDEX_CONSTITUENTS in DEFAULT_DAILY_UPDATE_DOMAINS
    assert "benchmark_daily" not in DEFAULT_DAILY_UPDATE_DOMAINS


def test_duplicate_report_uses_index_constituents_natural_key(tmp_path: Path) -> None:
    paths = _make_workspace(tmp_path)
    lake = ResearchDataLake(paths.lake_root)
    record = lake.save_domain_dataset(
        domain=DataDomain.INDEX_CONSTITUENTS,
        frame=_index_constituents_frame(["2026-01-05"]),
        spec={
            "dataset": "data_platform_index_constituents",
            "source": "unit",
            "start_date": "2026-01-05",
            "end_date": "2026-01-05",
        },
        source="unit",
        reuse=False,
    )

    report = _duplicate_key_report(lake=lake, dataset_id=record.dataset_id, max_rows=10000)

    assert report["key_columns"] == ["trade_date", "index_symbol", "symbol"]
    assert report["duplicate_count"] == 0


def test_daily_update_carry_forwards_industry_concept_when_provider_fails(tmp_path: Path) -> None:
    paths = _make_workspace(tmp_path)
    lake = ResearchDataLake(paths.lake_root)
    _save_active_bundle(lake, paths, ["2026-01-05", "2026-01-06"])
    industry_id = _save_industry_dataset(lake, ["2026-01-05", "2026-01-06"])
    _attach_active_sidecar(paths, DataDomain.INDUSTRY_CONCEPT, industry_id)

    result = run_daily_update(
        DailyUpdateConfig(
            workspace_root=paths.workspace_root,
            run_id="unit_industry_carry_forward",
            as_of_date="2026-01-08",
            start_date="2026-01-05",
            domains=(DataDomain.INDUSTRY_CONCEPT,),
            required_pit_domains=(DataDomain.INDUSTRY_CONCEPT,),
            max_duplicate_check_rows=10000,
        ),
        paths=paths,
        backfill_provider=FailingBackfillProvider(),
    )
    action = result.backfill_results[DataDomain.INDUSTRY_CONCEPT]
    effective = _read_dataset_frame(lake=lake, dataset_id=result.effective_sidecar_dataset_ids[DataDomain.INDUSTRY_CONCEPT])
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.status == "ok"
    assert result.blockers == []
    assert action["action"] == "pit_carry_forward"
    assert action["source_dataset_id"] == industry_id
    assert action["source_trade_date"] == "2026-01-06"
    assert action["trade_dates"] == ["2026-01-07", "2026-01-08"]
    assert result.after_coverage[DataDomain.INDUSTRY_CONCEPT]["end_date"] == "2026-01-08"
    assert sorted(effective["trade_date"].unique().tolist()) == ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
    assert "pit_carry_forward" in set(effective["source"].astype(str))
    assert manifest["backfill_results"][DataDomain.INDUSTRY_CONCEPT]["action"] == "pit_carry_forward"
    assert manifest["backfill_results"][DataDomain.INDUSTRY_CONCEPT]["provider_attempt"]["error_counts"][DataDomain.INDUSTRY_CONCEPT] == 2


def test_daily_update_does_not_carry_forward_core_pit_when_provider_fails(tmp_path: Path) -> None:
    paths = _make_workspace(tmp_path)
    lake = ResearchDataLake(paths.lake_root)
    _save_active_bundle(lake, paths, ["2026-01-05", "2026-01-06"])

    result = run_daily_update(
        DailyUpdateConfig(
            workspace_root=paths.workspace_root,
            run_id="unit_core_pit_no_carry_forward",
            as_of_date="2026-01-08",
            start_date="2026-01-05",
            domains=(DataDomain.SECURITY_STATUS,),
            required_pit_domains=(DataDomain.SECURITY_STATUS,),
            max_duplicate_check_rows=10000,
        ),
        paths=paths,
        backfill_provider=FailingBackfillProvider(),
    )

    assert result.status == "blocked"
    assert DataDomain.SECURITY_STATUS not in result.backfill_dataset_ids
    assert f"backfill_errors:{DataDomain.SECURITY_STATUS}:2" in result.blockers
    assert any(item.startswith(f"required_pit_lagging:{DataDomain.SECURITY_STATUS}:") for item in result.blockers)


def test_daily_update_memmap_failure_activates_canonical_and_keeps_memmap_stale(tmp_path: Path, monkeypatch) -> None:
    paths = _make_workspace(tmp_path)
    lake = ResearchDataLake(paths.lake_root)
    _save_active_bundle(lake, paths, ["2026-01-05", "2026-01-06"])
    active_root = load_root_manifest(paths)
    refresh_provider = InMemoryDomainProvider(
        "akshare_eastmoney",
        payloads={
            DataDomain.TRADING_CALENDAR: _calendar_frame(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]),
            DataDomain.UNIVERSE_SNAPSHOT: _universe_frame(["2026-01-08"]),
            DataDomain.MARKET_DAILY: _market_frame(["2026-01-07", "2026-01-08"]),
        },
    )

    def fail_memmap(**_kwargs):
        raise RuntimeError("synthetic_memmap_failure")

    monkeypatch.setattr(daily_update_module, "_build_and_validate_memmap", fail_memmap)

    result = run_daily_update(
        DailyUpdateConfig(
            workspace_root=paths.workspace_root,
            run_id="unit_memmap_failure_canonical_activates",
            as_of_date="2026-01-08",
            start_date="2026-01-05",
            domains=(DataDomain.MARKET_DAILY, DataDomain.TRADING_CALENDAR, DataDomain.UNIVERSE_SNAPSHOT, DataDomain.SECURITY_STATUS),
            required_pit_domains=(DataDomain.TRADING_CALENDAR, DataDomain.UNIVERSE_SNAPSHOT, DataDomain.SECURITY_STATUS),
            build_policy_bundle=True,
            build_memmap=True,
            activate=True,
            universe="all_a",
            max_duplicate_check_rows=10000,
        ),
        paths=paths,
        refresh_providers=[refresh_provider],
        backfill_provider=DailyBackfillProvider(),
    )
    root = load_root_manifest(paths)

    assert result.status == "activated_memmap_failed"
    assert result.canonical_activation_status == "activated"
    assert result.memmap_activation_status == "failed"
    assert any(item.startswith("memmap_build_failed:RuntimeError") for item in result.memmap_blockers)
    assert result.blockers == []
    assert root["canonical_dataset_id"] == result.canonical_dataset_id
    assert root["canonical_dataset_id"] != active_root["canonical_dataset_id"]
