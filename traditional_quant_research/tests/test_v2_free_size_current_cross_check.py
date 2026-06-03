from __future__ import annotations

from pathlib import Path

import pandas as pd

from traditional_quant_research.experiments import v2_free_size_current_cross_check as cross


def _reconstructed() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-06-01"),
                "code": "600000.SH",
                "total_market_cap": 1000.0,
                "float_market_cap": 800.0,
                "total_share": 100.0,
                "float_share": 80.0,
                "free_share": 70.0,
                "market_cap_unit": "CNY",
                "share_unit": "shares",
                "source": "akshare.cninfo_reconstructed",
                "source_trade_date": "20260601",
            }
        ]
    )


def test_build_current_cross_check_outputs_audit_consumable_long_rows() -> None:
    quotes = pd.DataFrame(
        [
            {
                "current_source": "fixture.current",
                "code": "600000.SH",
                "current_total_market_cap": 1000.0,
                "current_float_market_cap": 790.0,
                "status": "passed",
                "message": "",
            }
        ]
    )

    frame = cross.build_current_cross_check(_reconstructed(), quotes, as_of_date="2026-06-01")

    by_field = frame.set_index("field")
    assert frame["source"].unique().tolist() == ["akshare.cninfo_reconstructed"]
    assert by_field.loc["total_market_cap", "abs_relative_diff"] == 0.0
    assert by_field.loc["float_market_cap", "status"] == "passed"


def test_build_current_cross_check_records_unavailable_current_quotes() -> None:
    quotes = pd.DataFrame(
        [
            {
                "current_source": "fixture.current",
                "code": "",
                "current_total_market_cap": None,
                "current_float_market_cap": None,
                "status": "failed",
                "message": "network unavailable",
            }
        ]
    )

    frame = cross.build_current_cross_check(_reconstructed(), quotes, as_of_date="2026-06-01")

    assert frame["status"].unique().tolist() == ["current_cross_check_unavailable"]
    assert frame["abs_relative_diff"].isna().all()
    assert "network unavailable" in frame["message"].iloc[0]


class FakeAkshare:
    @staticmethod
    def stock_zh_a_spot_em() -> pd.DataFrame:
        return pd.DataFrame([{"代码": "600000", "总市值": 1000.0, "流通市值": 800.0}])

    @staticmethod
    def stock_individual_info_em(symbol: str) -> pd.DataFrame:
        return pd.DataFrame([{"item": "总市值", "value": 1000.0}, {"item": "流通市值", "value": 800.0}])


class FakeEfinanceStock:
    @staticmethod
    def get_realtime_quotes(symbols) -> pd.DataFrame:
        return pd.DataFrame([{"股票代码": "600000", "总市值": 1000.0, "流通市值": 800.0}])


class FakeEfinance:
    stock = FakeEfinanceStock()


def test_run_current_cross_check_writes_artifacts_from_reconstructed_path(tmp_path: Path) -> None:
    reconstructed_path = tmp_path / "reconstructed.csv"
    _reconstructed().to_csv(reconstructed_path, index=False)

    result = cross.run_v2_free_size_current_cross_check(
        symbols=["600000.SH"],
        as_of_date="2026-06-01",
        reconstructed_size_path=reconstructed_path,
        output_dir=tmp_path / "output",
        akshare_module=FakeAkshare,
        efinance_module=FakeEfinance,
    )

    run_dir = Path(result["run_dir"])
    assert result["current_cross_check_ok"] is True
    assert result["available_diff_rows"] >= 2
    assert result["candidate_count"] == 0
    assert (run_dir / "daily_size_current_cross_check.csv").exists()
    assert (run_dir / "current_quote_snapshot.csv").exists()
    assert (run_dir / "summary.json").exists()
