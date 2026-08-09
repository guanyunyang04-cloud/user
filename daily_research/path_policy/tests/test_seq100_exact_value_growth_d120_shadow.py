from __future__ import annotations

import pandas as pd

from daily_research.path_policy import seq100_exact_value_growth_d120_shadow as shadow


def test_parallel_d120_shadow_preserves_original_prefill_orders(tmp_path) -> None:
    summary = shadow.initialize_shadow(output_root=tmp_path)
    assert summary["planned_exit_open_days_from_signal"] == 120
    assert summary["realized_fill_count"] == 0
    assert summary["original_d60_arm_modified"] is False
    orders = pd.read_parquet(summary["files"]["frozen_orders"]["path"])
    assert len(orders) == 10
    assert orders["shadow_arm"].eq("parallel_d120").all()
    assert orders["entry_price_raw"].isna().all()
