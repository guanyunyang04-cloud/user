from __future__ import annotations

from pathlib import Path

import pandas as pd

from traditional_quant_research.experiments import v2_free_size_source_scout as scout


class FakeAkshare:
    @staticmethod
    def stock_zh_a_spot_em() -> pd.DataFrame:
        return pd.DataFrame([{"代码": "600000", "总市值": 1000.0, "流通市值": 800.0}])

    @staticmethod
    def stock_individual_info_em(symbol: str) -> pd.DataFrame:
        return pd.DataFrame([{"item": "总市值", "value": 1000.0}, {"item": "流通市值", "value": 800.0}])

    @staticmethod
    def stock_share_change_cninfo(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        return pd.DataFrame([{"变动日期": "2026-01-01", "总股本": "1万股", "流通股": "8000股"}])


class FakeEfinanceStock:
    @staticmethod
    def get_realtime_quotes(symbols) -> pd.DataFrame:
        return pd.DataFrame([{"股票代码": "600000", "总市值": 1000.0, "流通市值": 800.0}])


class FakeEfinance:
    stock = FakeEfinanceStock()


def test_free_size_source_scout_marks_reconstructed_source_usable_but_not_promoted(tmp_path: Path) -> None:
    result = scout.run_v2_free_size_source_scout(
        symbols=["600000.SH", "000001.SZ"],
        output_dir=tmp_path / "output",
        akshare_module=FakeAkshare,
        efinance_module=FakeEfinance,
    )

    assert result["usable_for_daily_size_count"] >= 2
    assert result["promotion_eligible_count"] == 0
    assert result["recommended_size_source"] == "akshare_cninfo_reconstructed_diagnostic_only"
    assert result["candidate_count"] == 0


def test_summarize_free_size_scout_keeps_proxy_diagnostic_only() -> None:
    matrix = pd.DataFrame(
        [
            {
                "source": "proxy.amount",
                "package": "local",
                "endpoint": "daily_bars.amount",
                "status": "available",
                "field_count": 1,
                "fields_found": "amount",
                "historical_capability": "historical_proxy",
                "pit_timing_claim": "local_pit_bars",
                "usable_for_daily_size": True,
                "promotion_eligible": False,
                "blocker": "diagnostic only",
            }
        ]
    )

    summary = scout.summarize_free_size_scout(matrix, run_id="fixture", symbols=["600000.SH"], trade_dates=["20260601"])

    assert summary["usable_for_daily_size_count"] == 1
    assert summary["promotion_eligible_count"] == 0
    assert summary["recommended_size_source"] == "proxy_amount_diagnostic_only"


class FallbackEfinanceStock:
    @staticmethod
    def get_realtime_quotes(*args) -> pd.DataFrame:
        if args:
            raise ValueError("code-list unsupported")
        return pd.DataFrame([{"股票代码": "600000", "总市值": 1000.0, "流通市值": 800.0}])


class FallbackEfinance:
    stock = FallbackEfinanceStock()


def test_efinance_probe_falls_back_to_default_quote_snapshot() -> None:
    row = scout._probe_efinance_current_quote(FallbackEfinance, "", ["600000.SH"])

    assert row["status"] == "passed"
    assert "总市值" in row["fields_found"]
