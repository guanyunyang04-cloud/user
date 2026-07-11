from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from quant_data_platform.cli import _maybe_run_qdp_v2
from quant_data_platform.qdp_v2.pit_market_substrate import (
    OVERRIDE_DOMAINS,
    build_pit_market_substrate,
    verify_dataset_view,
)


def _snapshot(
    root: Path,
    *,
    created_at: str,
    universe: list[dict[str, object]],
    bars: list[dict[str, object]],
) -> Path:
    root.mkdir(parents=True)
    pd.DataFrame(universe).to_parquet(root / "daily_universe.parquet", index=False)
    pd.DataFrame(bars).to_parquet(root / "daily_bars.parquet", index=False)
    (root / "manifest.json").write_text(
        json.dumps({"schema_version": 2, "snapshot_id": root.name, "created_at": created_at}),
        encoding="utf-8",
    )
    return root


def _universe_row(
    date: str,
    code: str,
    *,
    out_date: str = "",
    is_st: bool = False,
    suspended: bool = False,
    has_bar: bool = True,
) -> dict[str, object]:
    return {
        "date": pd.Timestamp(date),
        "code": code,
        "name_on_date": code,
        "ipo_date": "2000-01-01",
        "out_date": out_date,
        "is_listed_on_date": True,
        "is_mainboard": True,
        "is_common_a_share": True,
        "is_st_on_date": is_st,
        "is_suspended_on_date": suspended,
        "has_bar": has_bar,
        "is_tradeable": has_bar and not suspended and not is_st,
        "reject_reason": "suspended" if suspended else ("missing_bar" if not has_bar else ""),
    }


def _bar(date: str, code: str, close: float, *, open_price: float | None = None) -> dict[str, object]:
    open_value = close if open_price is None else open_price
    return {
        "date": pd.Timestamp(date),
        "code": code,
        "open": open_value,
        "high": max(open_value, close),
        "low": min(open_value, close),
        "close": close,
        "volume": 1000.0,
        "amount": close * 1000.0,
        "tradestatus": "1",
        "isST": "0",
        "source": "fixture",
    }


def _read_domain(qdp_root: Path, view: dict[str, object], domain: str) -> pd.DataFrame:
    dataset_id = dict(view["overrides"])[domain]
    manifest = json.loads(
        (qdp_root / "datasets" / domain / dataset_id / "dataset.json").read_text(encoding="utf-8")
    )
    return pd.concat([pd.read_parquet(qdp_root / item["path"]) for item in manifest["shards"]], ignore_index=True)


def test_build_multi_snapshot_pit_substrate_and_preserve_active_bytes(tmp_path: Path) -> None:
    qdp_root = tmp_path / "qdp_v2"
    active_path = qdp_root / "active" / "active.json"
    active_path.parent.mkdir(parents=True)
    active_payload = {
        "version": 2,
        "updated_at": "fixture",
        "datasets": {"trading_calendar": "trading_calendar__fixture"},
    }
    active_path.write_text(json.dumps(active_payload, indent=1) + "\n", encoding="utf-8")
    active_sha = hashlib.sha256(active_path.read_bytes()).hexdigest()

    early = _snapshot(
        tmp_path / "snap_early",
        created_at="2026-01-01T00:00:00",
        universe=[
            _universe_row("2020-01-02", "000001.SZ"),
            _universe_row("2020-01-03", "000001.SZ"),
            _universe_row("2020-01-06", "000001.SZ"),
            _universe_row("2020-01-02", "000999.SZ", out_date="2020-01-06"),
            _universe_row("2020-01-03", "000999.SZ", out_date="2020-01-06"),
            _universe_row("2020-01-06", "000999.SZ", out_date="2020-01-06", has_bar=False),
            _universe_row("2020-01-03", "600001.SH", suspended=True, has_bar=False),
        ],
        bars=[
            _bar("2020-01-02", "000001.SZ", 10.00),
            _bar("2020-01-03", "000001.SZ", 10.50),
            _bar("2020-01-06", "000001.SZ", 8.79),
            _bar("2020-01-02", "000999.SZ", 5.00),
            _bar("2020-01-03", "000999.SZ", 5.20),
        ],
    )
    late = _snapshot(
        tmp_path / "snap_late",
        created_at="2026-02-01T00:00:00",
        universe=[
            _universe_row("2020-01-03", "000001.SZ"),
            _universe_row("2020-01-06", "000001.SZ"),
        ],
        bars=[
            # The newer snapshot deterministically wins this overlap.
            _bar("2020-01-03", "000001.SZ", 10.00),
            _bar("2020-01-06", "000001.SZ", 8.79),
        ],
    )
    factors = tmp_path / "factor_events.parquet"
    pd.DataFrame(
        [
            {"symbol": "000001.SZ", "trade_date": "2020-01-02", "back_adjust_factor": 2.0},
            {"symbol": "000001.SZ", "trade_date": "2020-01-06", "back_adjust_factor": 2.5},
            {"symbol": "000999.SZ", "trade_date": "2020-01-02", "back_adjust_factor": 3.0},
        ]
    ).to_parquet(factors, index=False)

    result = build_pit_market_substrate(
        snapshot_roots=[early, late],
        factor_events_path=factors,
        qdp_root=qdp_root,
        reuse_active_factor=False,
        view_name="fixture_pit",
        threads=1,
    )

    assert result["status"] == "ok"
    assert result["snapshot_count"] == 2
    assert hashlib.sha256(active_path.read_bytes()).hexdigest() == active_sha
    view_path = Path(result["view_path"])
    view = json.loads(view_path.read_text(encoding="utf-8"))
    assert set(view["overrides"]) == set(OVERRIDE_DOMAINS)
    assert view["datasets"]["trading_calendar"] == "trading_calendar__fixture"
    assert verify_dataset_view(view_path=view_path, qdp_root=qdp_root)["status"] == "ok"

    market = _read_domain(qdp_root, view, "market_daily_raw")
    assert not market.duplicated(["trade_date", "symbol"]).any()
    overlap = market.loc[(market["trade_date"] == "2020-01-03") & (market["symbol"] == "000001.SZ")]
    assert overlap.iloc[0]["close"] == 10.00
    # Historical delisted security remains before its date-local out_date.
    assert set(market.loc[market["symbol"] == "000999.SZ", "trade_date"]) == {"2020-01-02", "2020-01-03"}

    status = _read_domain(qdp_root, view, "security_status")
    suspended = status.loc[(status["trade_date"] == "2020-01-03") & (status["symbol"] == "600001.SH")].iloc[0]
    assert bool(suspended["is_suspended"])
    assert not bool(suspended["has_valid_market_bar"])
    delisted = status.loc[(status["trade_date"] == "2020-01-06") & (status["symbol"] == "000999.SZ")].iloc[0]
    assert bool(delisted["is_delisted"])

    limits = _read_domain(qdp_root, view, "limit_status")
    one_tick_below = limits.loc[(limits["trade_date"] == "2020-01-06") & (limits["symbol"] == "000001.SZ")].iloc[0]
    assert one_tick_below["prior_valid_adjusted_close"] == 20.00
    assert one_tick_below["reference_raw_today"] == 8.00
    assert one_tick_below["up_limit"] == 8.80
    assert not bool(one_tick_below["is_limit_up"])

    factor = _read_domain(qdp_root, view, "adjust_factor")
    assert len(factor) == len(market)
    assert factor["back_adjust_factor"].gt(0).all()
    ffilled = factor.loc[(factor["trade_date"] == "2020-01-03") & (factor["symbol"] == "000001.SZ")].iloc[0]
    assert ffilled["back_adjust_factor"] == 2.0
    assert ffilled["factor_source_date"] == "2020-01-02"

    pit = _read_domain(qdp_root, view, "pit_signal_universe")
    assert not pit.duplicated(["trade_date", "symbol"]).any()
    assert {"status_valid", "is_st", "is_suspended", "is_delisted", "has_bar", "eligible_for_signal"}.issubset(pit.columns)
    assert not bool(pit.loc[(pit["trade_date"] == "2020-01-03") & (pit["symbol"] == "600001.SH"), "eligible_for_signal"].iloc[0])
    eligible = pit.loc[pit["eligible_for_signal"], ["trade_date", "symbol"]]
    assert len(eligible.merge(market[["trade_date", "symbol"]], how="left", indicator=True).query("_merge != 'both'")) == 0


def test_factor_gate_blocks_missing_symbol_day_without_touching_active(tmp_path: Path) -> None:
    qdp_root = tmp_path / "qdp_v2"
    active_path = qdp_root / "active" / "active.json"
    active_path.parent.mkdir(parents=True)
    active_path.write_text('{"version":2,"datasets":{}}\n', encoding="utf-8")
    before = active_path.read_bytes()
    snapshot = _snapshot(
        tmp_path / "snapshot",
        created_at="2026-01-01T00:00:00",
        universe=[_universe_row("2020-01-02", "000001.SZ"), _universe_row("2020-01-02", "000002.SZ")],
        bars=[_bar("2020-01-02", "000001.SZ", 10.0), _bar("2020-01-02", "000002.SZ", 20.0)],
    )
    factors = tmp_path / "factor_events.parquet"
    pd.DataFrame(
        [{"symbol": "000001.SZ", "trade_date": "2020-01-02", "back_adjust_factor": 2.0}]
    ).to_parquet(factors, index=False)

    try:
        build_pit_market_substrate(
            snapshot_roots=[snapshot],
            factor_events_path=factors,
            qdp_root=qdp_root,
            reuse_active_factor=False,
            threads=1,
        )
    except ValueError as exc:
        assert "missing_factor_rows" in str(exc)
    else:
        raise AssertionError("missing factor coverage must block the substrate build")
    assert active_path.read_bytes() == before
    assert not list((qdp_root / "views").glob("*.json")) if (qdp_root / "views").exists() else True


def test_gate0_blocks_research_eligible_universe_row_without_market_bar(tmp_path: Path) -> None:
    qdp_root = tmp_path / "qdp_v2"
    active_path = qdp_root / "active" / "active.json"
    active_path.parent.mkdir(parents=True)
    active_path.write_text('{"version":2,"datasets":{}}\n', encoding="utf-8")
    before = active_path.read_bytes()
    snapshot = _snapshot(
        tmp_path / "snapshot",
        created_at="2026-01-01T00:00:00",
        universe=[
            _universe_row("2020-01-02", "000001.SZ"),
            _universe_row("2020-01-03", "000001.SZ"),
        ],
        bars=[_bar("2020-01-03", "000001.SZ", 10.0)],
    )
    factors = tmp_path / "factor_events.parquet"
    pd.DataFrame(
        [{"symbol": "000001.SZ", "trade_date": "2020-01-03", "back_adjust_factor": 2.0}]
    ).to_parquet(factors, index=False)

    try:
        build_pit_market_substrate(
            snapshot_roots=[snapshot],
            factor_events_path=factors,
            qdp_root=qdp_root,
            reuse_active_factor=False,
            threads=1,
        )
    except ValueError as exc:
        assert "pit_gate0_research_non_suspended_without_market" in str(exc)
        assert "pit_gate0_market_start_coverage" in str(exc)
    else:
        raise AssertionError("missing market bars for research-eligible rows must block the view")
    assert active_path.read_bytes() == before


def test_public_cli_dispatches_pit_view_commands(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_dispatch(argv: list[str]) -> int:
        calls.append(argv)
        return 0

    monkeypatch.setattr("quant_data_platform.qdp_v2.cli.dispatch", fake_dispatch)
    assert _maybe_run_qdp_v2(["rebuild", "pit-market-substrate", "--help"]) == 0
    assert _maybe_run_qdp_v2(["verify", "pit-market-view", "--help"]) == 0
    assert calls == [
        ["rebuild", "pit-market-substrate", "--help"],
        ["verify", "pit-market-view", "--help"],
    ]


def test_recovery_cache_plus_independent_market_manifest(tmp_path: Path) -> None:
    qdp_root = tmp_path / "qdp_v2"
    active_path = qdp_root / "active" / "active.json"
    active_path.parent.mkdir(parents=True)
    active_path.write_text('{"version":2,"datasets":{}}\n', encoding="utf-8")
    recovery = tmp_path / "recovery"
    stock_lists = recovery / "cache" / "daily_stock_lists"
    stock_lists.mkdir(parents=True)
    pd.DataFrame(
        [
            {"date": "2012-01-04", "code": "600005.SH", "name_on_date": "武钢股份", "query_all_trade_status": "1"},
            {"date": "2012-01-05", "code": "600005.SH", "name_on_date": "武钢股份", "query_all_trade_status": "1"},
            {"date": "2017-02-14", "code": "600005.SH", "name_on_date": "武钢股份", "query_all_trade_status": "0"},
            {"date": "2012-01-04", "code": "000002.SZ", "name_on_date": "*ST测试", "query_all_trade_status": "1"},
        ]
    ).to_parquet(stock_lists / "year=2012.parquet", index=False)
    pd.DataFrame(
        [
            {"code": "600005.SH", "ipo_date": "1999-08-03", "out_date": "2017-02-14", "security_type": "1"},
            {"code": "000002.SZ", "ipo_date": "1991-01-29", "out_date": "", "security_type": "1"},
        ]
    ).to_parquet(recovery / "cache" / "security_master.parquet", index=False)
    bars_path = tmp_path / "market_daily.parquet"
    pd.DataFrame(
        [
            {"code": "600005.SH", "date": "2012-01-04", "open": 3.0, "high": 3.1, "low": 2.9, "close": 3.0, "volume": 100.0, "amount": 300.0},
            {"code": "600005.SH", "date": "2012-01-05", "open": 3.0, "high": 3.2, "low": 3.0, "close": 3.1, "volume": 100.0, "amount": 310.0},
            {"code": "000002.SZ", "date": "2012-01-04", "open": 5.0, "high": 5.1, "low": 4.9, "close": 5.0, "volume": 100.0, "amount": 500.0},
        ]
    ).to_parquet(bars_path, index=False)
    shard_manifest = tmp_path / "market_shard_manifest.json"
    shard_manifest.write_text(
        json.dumps({"domain": "market_daily", "shards": [{"path": str(bars_path)}]}),
        encoding="utf-8",
    )
    factors = tmp_path / "factors.parquet"
    pd.DataFrame(
        [
            {"symbol": "600005.SH", "trade_date": "2012-01-04", "back_adjust_factor": 1.5},
            {"symbol": "000002.SZ", "trade_date": "2012-01-04", "back_adjust_factor": 1.0},
        ]
    ).to_parquet(factors, index=False)

    result = build_pit_market_substrate(
        snapshot_roots=[recovery],
        market_bar_sources=[shard_manifest],
        factor_events_path=factors,
        qdp_root=qdp_root,
        reuse_active_factor=False,
        threads=1,
    )
    view = json.loads(Path(result["view_path"]).read_text(encoding="utf-8"))
    market = _read_domain(qdp_root, view, "market_daily_raw")
    pit = _read_domain(qdp_root, view, "pit_signal_universe")
    status = _read_domain(qdp_root, view, "security_status")
    assert len(market) == 3
    # Future out_date does not remove the valid 2012 history.
    assert set(market.loc[market["symbol"] == "600005.SH", "trade_date"]) == {"2012-01-04", "2012-01-05"}
    st = pit.loc[pit["symbol"] == "000002.SZ"].iloc[0]
    assert bool(st["is_st"])
    assert not bool(st["eligible_for_signal"])
    assert len(status) == len(pit)
    delisted = status.loc[(status["symbol"] == "600005.SH") & (status["trade_date"] == "2017-02-14")].iloc[0]
    assert bool(delisted["is_delisted"])
