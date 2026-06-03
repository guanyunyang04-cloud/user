from __future__ import annotations

from pathlib import Path

import pandas as pd

from traditional_quant_research.experiments import v2_industry_size_source_audit as audit


def _write_fixture_snapshot(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "manifest.json").write_text('{"snapshot_id":"fixture-v2"}', encoding="utf-8")
    pd.DataFrame(
        [
            {
                "code": "600000.SH",
                "baostock_code": "sh.600000",
                "name": "PF Bank",
                "ipo_date": "1999-11-10",
                "out_date": "",
                "security_type": "1",
                "status": "1",
            }
        ]
    ).to_parquet(root / "security_master.parquet", index=False)
    pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-02"),
                "code": "600000.SH",
                "is_st_on_date": False,
                "is_suspended_on_date": False,
                "is_tradeable": True,
            }
        ]
    ).to_parquet(root / "daily_universe.parquet", index=False)
    pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-02"),
                "code": "600000.SH",
                "open": 10.0,
                "close": 10.1,
                "volume": 1000.0,
                "amount": 10000.0,
                "tradestatus": "1",
                "isST": "0",
            }
        ]
    ).to_parquet(root / "daily_bars.parquet", index=False)
    pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-02"),
                "code": "600000.SH",
                "has_bar": True,
                "is_suspended_like": False,
                "is_tradeable": True,
            }
        ]
    ).to_parquet(root / "daily_status.parquet", index=False)
    pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-02"),
                "code": "600000.SH",
                "turn": 0.85,
                "pctChg": 1.2,
                "peTTM": 6.5,
                "pbMRQ": 0.7,
                "psTTM": 2.1,
                "pcfNcfTTM": 4.2,
            }
        ]
    ).to_parquet(root / "daily_metrics.parquet", index=False)


def test_snapshot_audit_detects_missing_true_industry_and_market_cap(tmp_path: Path) -> None:
    snapshot_root = tmp_path / "snapshot"
    _write_fixture_snapshot(snapshot_root)

    result = audit.run_v2_industry_size_source_audit(
        root=snapshot_root,
        output_dir=tmp_path / "output",
        probe_baostock=False,
    )

    field_summary = result["field_summary"]
    assert field_summary["true_industry_in_snapshot"] is False
    assert field_summary["true_market_cap_in_snapshot"] is False
    assert field_summary["turnover_in_snapshot"] is True
    assert field_summary["size_liquidity_proxy_in_snapshot"] is True
    assert field_summary["pit_status_fields_in_snapshot"] is True
    assert result["candidate_count"] == 0

    run_dir = Path(result["run_dir"])
    schema = pd.read_csv(run_dir / "snapshot_schema.csv")
    source_audit = pd.read_csv(run_dir / "source_field_audit.csv")
    assert "tradeable_panel" in set(schema["table"])
    assert "daily_metrics" in set(schema["table"])
    assert source_audit.loc[source_audit["domain"] == "true_industry", "available"].eq(False).all()
    assert source_audit.loc[source_audit["field_or_method"] == "turn", "available"].iloc[0]
    assert source_audit.loc[source_audit["field_or_method"] == "amount", "available"].iloc[0]
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary.md").exists()


def test_build_source_field_audit_detects_available_true_fields() -> None:
    schema = pd.DataFrame(
        [
            {"table": "security_master", "field": "industry", "dtype": "string", "exists": True, "path": "x"},
            {"table": "daily_bars", "field": "float_market_cap", "dtype": "double", "exists": True, "path": "x"},
            {"table": "security_master", "field": "totalShare", "dtype": "double", "exists": True, "path": "x"},
            {"table": "daily_bars", "field": "amount", "dtype": "double", "exists": True, "path": "x"},
        ]
    )

    source_audit = audit.build_source_field_audit(schema)
    summary = audit.summarize_audit(
        snapshot_root=Path("fixture"),
        manifest={"snapshot_id": "schema-fixture"},
        schema=schema,
        audit=source_audit,
        run_id="run-fixture",
    )

    assert summary["field_summary"]["true_industry_in_snapshot"] is True
    assert summary["field_summary"]["true_market_cap_in_snapshot"] is True
    assert summary["field_summary"]["share_base_in_snapshot"] is True
    assert summary["field_summary"]["size_liquidity_proxy_in_snapshot"] is True


def test_build_source_field_audit_includes_history_field_probe() -> None:
    schema = pd.DataFrame(
        [
            {"table": "daily_bars", "field": "amount", "dtype": "double", "exists": True, "path": "x"},
        ]
    )
    history_probe = pd.DataFrame(
        [
            {"field_or_method": "turn", "available": True, "evidence": "rows=6", "notes": ""},
            {"field_or_method": "turnover", "available": False, "evidence": "", "notes": "invalid field"},
            {"field_or_method": "peTTM", "available": True, "evidence": "rows=6", "notes": ""},
            {"field_or_method": "totalShare", "available": False, "evidence": "", "notes": "invalid field"},
        ]
    )

    source_audit = audit.build_source_field_audit(schema, history_field_probe=history_probe)
    summary = audit.summarize_audit(
        snapshot_root=Path("fixture"),
        manifest={"snapshot_id": "schema-fixture"},
        schema=schema,
        audit=source_audit,
        run_id="run-fixture",
    )

    assert summary["field_summary"]["baostock_history_turn_available"] is True
    assert summary["field_summary"]["baostock_history_valuation_fields_available"] is True
    assert summary["field_summary"]["baostock_history_share_base_fields_available"] is False
    assert "turn" in set(source_audit.loc[source_audit["domain"] == "baostock_history_field", "field_or_method"])


def test_build_source_field_audit_includes_stock_basic_probe() -> None:
    schema = pd.DataFrame(
        [
            {"table": "daily_bars", "field": "amount", "dtype": "double", "exists": True, "path": "x"},
        ]
    )
    stock_basic_probe = pd.DataFrame(
        [
            {"field_or_method": "totalShare", "available": True, "evidence": "fields=totalShare", "notes": ""},
            {"field_or_method": "market_cap", "available": False, "evidence": "", "notes": "field not returned"},
        ]
    )

    source_audit = audit.build_source_field_audit(schema, stock_basic_field_probe=stock_basic_probe)
    summary = audit.summarize_audit(
        snapshot_root=Path("fixture"),
        manifest={"snapshot_id": "schema-fixture"},
        schema=schema,
        audit=source_audit,
        run_id="run-fixture",
    )

    assert summary["field_summary"]["baostock_stock_basic_share_base_fields_available"] is True
    assert summary["field_summary"]["baostock_stock_basic_market_cap_fields_available"] is False
    assert "totalShare" in set(
        source_audit.loc[source_audit["domain"] == "baostock_stock_basic_field", "field_or_method"]
    )


def test_run_audit_writes_history_field_probe_when_enabled(tmp_path: Path, monkeypatch) -> None:
    snapshot_root = tmp_path / "snapshot"
    _write_fixture_snapshot(snapshot_root)

    def fake_probe(**kwargs):
        return pd.DataFrame(
            [
                {"field_or_method": "turn", "available": True, "evidence": "rows=6", "notes": ""},
                {"field_or_method": "turnover", "available": False, "evidence": "", "notes": "invalid field"},
            ]
        )

    monkeypatch.setattr(audit, "probe_baostock_history_fields", fake_probe)

    result = audit.run_v2_industry_size_source_audit(
        root=snapshot_root,
        output_dir=tmp_path / "output",
        probe_baostock=False,
        live_history_probe=True,
    )

    run_dir = Path(result["run_dir"])
    assert (run_dir / "baostock_history_field_probe.csv").exists()
    assert result["field_summary"]["baostock_history_turn_available"] is True


def test_run_audit_writes_stock_basic_probe_when_enabled(tmp_path: Path, monkeypatch) -> None:
    snapshot_root = tmp_path / "snapshot"
    _write_fixture_snapshot(snapshot_root)

    def fake_probe(**kwargs):
        return pd.DataFrame(
            [
                {"field_or_method": "totalShare", "available": False, "evidence": "", "notes": "field not returned"},
            ]
        )

    monkeypatch.setattr(audit, "probe_baostock_stock_basic_fields", fake_probe)

    result = audit.run_v2_industry_size_source_audit(
        root=snapshot_root,
        output_dir=tmp_path / "output",
        probe_baostock=False,
        live_stock_basic_probe=True,
    )

    run_dir = Path(result["run_dir"])
    assert (run_dir / "baostock_stock_basic_field_probe.csv").exists()
    assert result["field_summary"]["baostock_stock_basic_share_base_fields_available"] is False


def test_probe_baostock_client_handles_missing_methods() -> None:
    class EmptyClient:
        __version__ = "fixture"

    capabilities = audit.probe_baostock_client(client=EmptyClient())

    industry = capabilities.loc[capabilities["field_or_method"] == "query_stock_industry"].iloc[0]
    assert bool(industry["available"]) is False
    assert "method not found" in industry["notes"]


def test_probe_baostock_client_detects_industry_method() -> None:
    class FakeClient:
        __version__ = "fixture"

        def query_stock_industry(self) -> None:
            return None

        def query_history_k_data_plus(self) -> None:
            return None

    capabilities = audit.probe_baostock_client(client=FakeClient())

    industry = capabilities.loc[capabilities["field_or_method"] == "query_stock_industry"].iloc[0]
    history = capabilities.loc[capabilities["field_or_method"] == "query_history_k_data_plus"].iloc[0]
    assert bool(industry["available"]) is True
    assert bool(history["available"]) is True
