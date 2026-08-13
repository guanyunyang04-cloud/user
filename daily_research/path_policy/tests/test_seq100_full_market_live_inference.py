from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from daily_research.path_policy import seq100_full_market_live_inference as live


def test_frozen_core_contract_is_314_plus_243() -> None:
    contract = live._load_contract()
    assert len(contract.feature_names) == 557
    assert len(contract.core_feature_names) == 314
    assert len(contract.masked_feature_names) == 243
    assert contract.feature_names[:314] == contract.core_feature_names
    assert contract.feature_names[314:] == contract.masked_feature_names
    assert len(live.daily_feature_names()) == 105


def test_last_daily_features_preserve_lag_and_window_semantics() -> None:
    days = 61
    symbols = 3
    raw = np.zeros((days, symbols, 13), dtype=np.float32)
    close = np.arange(10.0, 10.0 + days, dtype=np.float32)[:, None]
    close = close * np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    raw[:, :, 0] = close * 0.99
    raw[:, :, 1] = close * 1.01
    raw[:, :, 2] = close * 0.98
    raw[:, :, 3] = close
    raw[:, :, 4] = 1_000.0
    raw[:, :, 5] = 10_000.0
    turnover = np.ones((days, symbols, 2), dtype=np.float32)
    features = live._last_daily_features(raw, turnover)
    assert tuple(features) == live.daily_feature_names()
    expected_1d = close[-1] / close[-2] - 1.0
    expected_60d = close[-1] / close[0] - 1.0
    np.testing.assert_allclose(features["close_ret_1d"], expected_1d, rtol=1e-6)
    np.testing.assert_allclose(features["return_60d"], expected_60d, rtol=1e-6)
    np.testing.assert_allclose(features["amount_ratio_60d"], 1.0, rtol=1e-6)
    np.testing.assert_allclose(features["volume_ratio_60d"], 1.0, rtol=1e-6)


def test_signal_date_cannot_exceed_active_snapshot() -> None:
    dates = pd.bdate_range(end="2026-07-21", periods=80).strftime("%Y-%m-%d").tolist()
    pack = {"date_values": dates}
    assert live.resolve_signal_date(
        pack, active_as_of_date="2026-07-21", requested=None
    ) == ("2026-07-21", 79)
    with pytest.raises(live.LiveInferenceError, match="after_active"):
        live.resolve_signal_date(
            pack, active_as_of_date="2026-07-21", requested="2026-07-22"
        )


def test_minute_feature_sql_includes_first_bar_open_to_close_return() -> None:
    sql = live._minute_feature_sql(signal_date="2025-12-31", intraday_scan="bars")
    assert "WHEN bar_no=1 AND open>0 AND close>0 THEN abs(ln(close/open))" in sql
    assert "first(bar_no ORDER BY high DESC NULLS LAST,bar_no ASC)" in sql
    assert "first(bar_no ORDER BY low ASC NULLS LAST,bar_no ASC)" in sql


def test_dual_gate_requires_all_four_conditions() -> None:
    assert live.dual_market_gate(
        ridge_prediction=0.01,
        linear_probability=0.51,
        neural_return_prediction=0.001,
        neural_probability=0.51,
    )
    for field in (
        "ridge_prediction",
        "linear_probability",
        "neural_return_prediction",
        "neural_probability",
    ):
        values = {
            "ridge_prediction": 0.01,
            "linear_probability": 0.51,
            "neural_return_prediction": 0.001,
            "neural_probability": 0.51,
        }
        values[field] = 0.0
        assert not live.dual_market_gate(**values)


def test_requested_orders_are_top10_without_real_authority_or_substitution() -> None:
    scores = pd.DataFrame(
        {
            "signal_date": "2026-07-21",
            "symbol": [f"{index:06d}.SZ" for index in range(12)],
            "name": [f"stock-{index}" for index in range(12)],
            "score_rank": np.arange(1, 13),
            "ensemble_score": np.linspace(1.0, 0.0, 12),
            "market_gate_active": True,
        }
    )
    orders = live.requested_orders(scores)
    assert len(orders) == 10
    assert orders["symbol"].tolist() == scores.head(10)["symbol"].tolist()
    assert not orders["rank_substitution"].any()
    assert not orders["real_order_authority"].any()
    scores["market_gate_active"] = False
    assert live.requested_orders(scores).empty


def test_membership_sql_uses_strict_publication_time_and_no_outcome_table() -> None:
    required = (
        "market_daily_raw",
        "market_intraday_5m",
        "financial_quarterly",
        "universe_snapshot",
        "security_status",
        "valuation",
        "share_capital",
        "industry_concept",
        "trading_calendar",
    )
    sql = live._quality_membership_sql(
        signal_date="2025-12-31", scans={name: name for name in required}
    )
    compact = "".join(sql.split())
    assert "c.trade_date>f.publish_date" in compact
    assert "future_" not in sql.lower()
    assert "entry_" not in sql.lower()
    assert "label_" not in sql.lower()
