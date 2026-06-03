from __future__ import annotations

from pathlib import Path

import pandas as pd

from traditional_quant_research.dataset_v2 import DAILY_SIZE_COLUMNS
from traditional_quant_research.experiments import v2_tushare_size_probe as probe


class _FakePro:
    def __init__(self, rows_by_pair=None, fail_pairs=None):
        self.rows_by_pair = rows_by_pair or {}
        self.fail_pairs = fail_pairs or set()
        self.calls = []

    def daily_basic(self, *, ts_code: str, trade_date: str, fields: str):
        self.calls.append((ts_code, trade_date, fields))
        if (ts_code, trade_date) in self.fail_pairs:
            raise RuntimeError("fixture failure")
        row = self.rows_by_pair.get((ts_code, trade_date))
        if row is None:
            return pd.DataFrame()
        return pd.DataFrame([row])


class _FakeTushare:
    def __init__(self, pro):
        self.pro = pro
        self.token = ""

    def set_token(self, token: str) -> None:
        self.token = token

    def pro_api(self, token: str | None = None):
        if token:
            self.token = token
        return self.pro


def _row(code: str, trade_date: str) -> dict[str, object]:
    return {
        "ts_code": code,
        "trade_date": trade_date,
        "total_mv": 100000.0,
        "circ_mv": 80000.0,
        "total_share": 10000.0,
        "float_share": 8000.0,
        "free_share": 7000.0,
        "close": 10.0,
        "turnover_rate": 1.2,
        "turnover_rate_f": 1.5,
        "pe_ttm": 8.0,
        "pb": 0.8,
    }


def test_normalize_symbols_and_trade_dates() -> None:
    assert probe.normalize_tushare_symbols("600000.sh,sz.000001, 000002.SZ") == [
        "600000.SH",
        "000001.SZ",
        "000002.SZ",
    ]
    assert probe.normalize_trade_dates("2026-05-25,20260601") == ["20260525", "20260601"]


def test_missing_package_skips_and_writes_artifacts(tmp_path: Path, monkeypatch) -> None:
    real_import = probe.importlib.import_module

    def fake_import(name: str):
        if name == "tushare":
            raise ModuleNotFoundError("No module named 'tushare'")
        return real_import(name)

    monkeypatch.setattr(probe.importlib, "import_module", fake_import)

    result = probe.run_v2_tushare_size_probe(
        output_dir=tmp_path / "output",
        token="",
        tushare_module=None,
        write_research_log=True,
        research_log_path=tmp_path / "probe.md",
    )

    run_dir = Path(result["run_dir"])
    assert result["status"] == "skipped"
    assert result["skip_reason"] == "package_missing"
    assert result["required_fields_present"] is False
    assert result["v2_2_ready"] is False
    assert (run_dir / "tushare_daily_basic_probe.csv").exists()
    assert (run_dir / "daily_size_probe.csv").exists()
    assert (run_dir / "daily_size_probe.parquet").exists()
    assert (run_dir / "failures.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (tmp_path / "probe.md").exists()


def test_missing_token_skips_after_package_available(tmp_path: Path) -> None:
    fake = _FakeTushare(_FakePro())

    result = probe.run_v2_tushare_size_probe(
        output_dir=tmp_path / "output",
        token="",
        tushare_module=fake,
    )

    assert result["status"] == "skipped"
    assert result["skip_reason"] == "auth_missing"
    assert result["failure_count"] == 1
    assert fake.token == ""


def test_successful_probe_returns_passed_and_required_fields(tmp_path: Path) -> None:
    rows = {}
    for code in ["600000.SH", "000001.SZ"]:
        for trade_date in ["20260525", "20260601"]:
            rows[(code, trade_date)] = _row(code, trade_date)
    fake_pro = _FakePro(rows)
    fake = _FakeTushare(fake_pro)

    result = probe.run_v2_tushare_size_probe(
        symbols="600000.SH,000001.SZ",
        trade_dates="2026-05-25,2026-06-01",
        token="fixture-token",
        output_dir=tmp_path / "output",
        tushare_module=fake,
    )

    run_dir = Path(result["run_dir"])
    assert result["status"] == "passed"
    assert result["expected_pairs"] == 4
    assert result["returned_pairs"] == 4
    assert result["missing_pairs"] == 0
    assert result["required_fields_present"] is True
    assert result["v2_2_ready"] is True

    output = pd.read_csv(run_dir / "tushare_daily_basic_probe.csv")
    daily_size = pd.read_parquet(run_dir / "daily_size_probe.parquet")
    assert set(["total_mv", "circ_mv", "total_share", "float_share", "free_share"]).issubset(output.columns)
    assert list(daily_size.columns) == DAILY_SIZE_COLUMNS
    assert daily_size["total_market_cap"].notna().all()
    assert daily_size["float_market_cap"].notna().all()
    assert daily_size["market_cap_unit"].unique().tolist() == ["10k CNY"]
    assert daily_size["share_unit"].unique().tolist() == ["10k shares"]
    assert output["date"].tolist()[0] == "2026-05-25"
    assert fake_pro.calls[0][2].startswith("ts_code,trade_date,total_mv")


def test_partial_probe_records_failures_without_crashing(tmp_path: Path) -> None:
    rows = {("600000.SH", "20260601"): _row("600000.SH", "20260601")}
    fake = _FakeTushare(_FakePro(rows_by_pair=rows, fail_pairs={("000001.SZ", "20260601")}))

    result = probe.run_v2_tushare_size_probe(
        symbols=["600000.SH", "000001.SZ"],
        trade_dates=["20260601"],
        token="fixture-token",
        output_dir=tmp_path / "output",
        tushare_module=fake,
    )

    run_dir = Path(result["run_dir"])
    failures = pd.read_csv(run_dir / "failures.csv")
    assert result["status"] == "partial"
    assert result["returned_pairs"] == 1
    assert result["missing_pairs"] == 1
    assert result["failure_count"] == 1
    assert failures["error_type"].tolist() == ["RuntimeError"]
