from __future__ import annotations

from pathlib import Path

import pandas as pd

from traditional_quant_research.experiments import v2_cninfo_size_event_audit as audit


class FakeAkshare:
    @staticmethod
    def stock_share_change_cninfo(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        if symbol == "000001":
            raise RuntimeError("rate limited")
        return pd.DataFrame(
            [
                {"变动日期": "2020-01-01", "公告日期": "2020-01-02", "总股本": "1万股", "已流通股份": "8000股"},
                {"变动日期": "2021-01-01", "公告日期": "2021-01-02", "总股本": "1.2万股", "已流通股份": "9000股"},
            ]
        )


def test_cninfo_symbol_audit_marks_publication_date_and_usable_events() -> None:
    row, events, raw_fields = audit.audit_cninfo_symbol(
        "600000.SH",
        akshare_module=FakeAkshare,
        import_error="",
        start_date="20160101",
        end_date="20261231",
    )

    assert row["status"] == "passed"
    assert row["pit_semantics"] == "publication_date_available"
    assert row["usable_for_reconstruction"] is True
    assert row["promotion_evidence_ready"] is False
    assert events["total_share"].tolist() == [10000.0, 12000.0]
    assert {item["matched_role"] for item in raw_fields} >= {"date", "publish_date", "total_share", "float_share"}


def test_cninfo_size_event_audit_writes_structured_failures(tmp_path: Path) -> None:
    result = audit.run_v2_cninfo_size_event_audit(
        symbols=["600000.SH", "000001.SZ"],
        years=[2020, 2021],
        output_dir=tmp_path / "output",
        akshare_module=FakeAkshare,
    )

    run_dir = Path(result["run_dir"])
    symbol_audit = pd.read_csv(run_dir / "cninfo_symbol_event_audit.csv")

    assert result["usable_for_reconstruction_count"] == 1
    assert result["promotion_evidence_ready_count"] == 0
    assert result["source_decision"] == "diagnostic_ready_for_cross_check"
    assert symbol_audit.set_index("code").loc["000001.SZ", "failure_type"] == "RuntimeError"
    assert (run_dir / "cninfo_share_events.csv").exists()
    assert (run_dir / "summary.md").exists()


def test_cninfo_symbol_audit_handles_missing_package() -> None:
    row, events, raw_fields = audit.audit_cninfo_symbol(
        "600000.SH",
        akshare_module=None,
        import_error="no module",
        start_date="20160101",
        end_date="20261231",
    )

    assert row["status"] == "package_missing"
    assert row["failure_type"] == "package_missing"
    assert events.empty
    assert raw_fields == []
