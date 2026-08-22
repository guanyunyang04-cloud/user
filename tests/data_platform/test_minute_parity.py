from __future__ import annotations

import pandas as pd
import pytest

from quantlab.data.minute_parity import compare_bar_frames


def _frame(symbol: str, close: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [symbol],
            "trade_date": ["2025-01-02"],
            "bar_time": ["093500000"],
            "open": [10.0],
            "high": [10.1],
            "low": [9.9],
            "close": [close],
            "volume": [1000.0],
            "amount": [10000.0],
        }
    )


def test_compare_bar_frames_reports_keys_and_numeric_error() -> None:
    left = pd.concat([_frame("600000.SH", 10.0), _frame("600001.SH", 11.0)], ignore_index=True)
    right = pd.concat([_frame("600000.SH", 10.1), _frame("600002.SH", 12.0)], ignore_index=True)
    result = compare_bar_frames(left, right, left_name="local", right_name="source")
    assert result["key_counts"] == {"both": 1, "local_only": 1, "source_only": 1}
    assert result["metrics"]["open"]["exact_rate"] == 1.0
    assert result["metrics"]["close"]["exact_rate"] == 0.0
    assert result["metrics"]["close"]["maximum_absolute_error"] == pytest.approx(0.1)
