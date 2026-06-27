from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_data_platform.core.paths import qdp_paths
from quant_data_platform.core.registry import load_root_manifest, write_root_manifest_update
from quant_data_platform.domains.contracts import DataDomain, DomainFetchRequest, ProviderResult
from quant_data_platform.ingest.daily_update import DailyUpdateConfig, build_parser, run_daily_update
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
    assert result.effective_sidecar_dataset_ids[DataDomain.UNIVERSE_SNAPSHOT] != active_ids[DataDomain.UNIVERSE_SNAPSHOT]
    assert result.effective_sidecar_dataset_ids[DataDomain.SECURITY_STATUS] != active_ids[DataDomain.SECURITY_STATUS]
    assert result.after_coverage[DataDomain.UNIVERSE_SNAPSHOT]["end_date"] == "2026-01-08"
    assert result.after_coverage[DataDomain.SECURITY_STATUS]["trade_date_count"] == 4
    assert manifest["combined_sidecar_dataset_ids"][DataDomain.UNIVERSE_SNAPSHOT] == result.effective_sidecar_dataset_ids[DataDomain.UNIVERSE_SNAPSHOT]
