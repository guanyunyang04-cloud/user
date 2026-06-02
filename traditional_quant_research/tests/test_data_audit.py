from __future__ import annotations

import pandas as pd

from traditional_quant_research.data_audit import (
    add_forward_return_labels,
    audit_bar_quality,
    audit_label_by_year,
    audit_label_summary,
    audit_reject_reasons,
    audit_universe_by_year,
    collect_extreme_label_samples,
)


def test_audit_bar_quality_flags_ohlcv_anomalies() -> None:
    bars = pd.DataFrame(
        [
            {"date": "2026-01-02", "code": "600000.SH", "open": 1, "high": 2, "low": 0.8, "close": 1.5, "volume": 100, "amount": 1000},
            {"date": "2026-01-05", "code": "600000.SH", "open": 0, "high": 1, "low": 2, "close": 1.2, "volume": -1, "amount": 0},
        ]
    )

    result = audit_bar_quality(bars).iloc[0].to_dict()

    assert result["rows"] == 2
    assert result["nonpositive_open_rows"] == 1
    assert result["negative_volume_rows"] == 1
    assert result["zero_amount_rows"] == 1
    assert result["high_below_low_rows"] == 1


def test_forward_labels_and_extreme_samples_are_audited() -> None:
    panel = pd.DataFrame(
        [
            {"date": "2026-01-02", "code": "600000.SH", "open": 10.0, "close": 10.0, "volume": 100, "amount": 1000},
            {"date": "2026-01-05", "code": "600000.SH", "open": 10.0, "close": 20.0, "volume": 100, "amount": 2000},
            {"date": "2026-01-06", "code": "600000.SH", "open": 20.0, "close": 21.0, "volume": 100, "amount": 2100},
            {"date": "2026-01-02", "code": "000001.SZ", "open": 5.0, "close": 5.0, "volume": 100, "amount": 500},
            {"date": "2026-01-05", "code": "000001.SZ", "open": 5.0, "close": 5.1, "volume": 100, "amount": 510},
            {"date": "2026-01-06", "code": "000001.SZ", "open": 5.1, "close": 5.2, "volume": 100, "amount": 520},
        ]
    )

    labels = add_forward_return_labels(panel, horizons=(1, 2))
    summary = audit_label_summary(labels, horizons=(1, 2), extreme_abs_returns=(0.5,))
    yearly = audit_label_by_year(labels, horizons=(1,), extreme_abs_returns=(0.5,))
    samples = collect_extreme_label_samples(labels, horizons=(1,), threshold=0.5)

    first = labels.loc[(labels["code"] == "600000.SH") & (labels["date"] == pd.Timestamp("2026-01-02"))].iloc[0]
    assert first["fwd_ret_1d"] == 1.0
    assert summary.loc[summary["label"] == "fwd_ret_1d", "abs_gt_0.5_rows"].iloc[0] == 1
    assert yearly["available_rows"].iloc[0] == 4
    assert samples["code"].tolist() == ["600000.SH"]


def test_universe_yearly_and_reject_reason_counts() -> None:
    universe = pd.DataFrame(
        [
            {"date": "2026-01-02", "code": "600000.SH", "is_tradeable": True, "is_st_on_date": False, "is_suspended_on_date": False, "has_bar": True, "reject_reason": ""},
            {"date": "2026-01-02", "code": "000001.SZ", "is_tradeable": False, "is_st_on_date": True, "is_suspended_on_date": False, "has_bar": True, "reject_reason": "st_on_date"},
            {"date": "2026-01-05", "code": "600000.SH", "is_tradeable": False, "is_st_on_date": False, "is_suspended_on_date": True, "has_bar": False, "reject_reason": "missing_bar"},
        ]
    )

    yearly = audit_universe_by_year(universe)
    reasons = audit_reject_reasons(universe)

    assert yearly.loc[0, "rows"] == 3
    assert yearly.loc[0, "tradeable_rows"] == 1
    assert yearly.loc[0, "st_rows"] == 1
    assert yearly.loc[0, "suspended_rows"] == 1
    assert yearly.loc[0, "missing_bar_rows"] == 1
    assert set(reasons["reject_reason"]) == {"ok", "st_on_date", "missing_bar"}
