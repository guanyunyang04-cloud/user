from __future__ import annotations

import pandas as pd

from daily_research.path_policy import seq100_exact_value_growth_shadow as shadow


def test_shadow_contract_freezes_ten_pending_candidates_without_outcomes() -> None:
    study, _ = shadow.load_study()

    assert len(study["candidates"]) == 10
    assert [row["rank"] for row in study["candidates"]] == list(range(1, 11))
    assert study["freeze"]["signal_close_date"] == "2026-08-07"
    assert study["freeze"]["observation_status"] == "pending_forward_fill"
    assert study["decision_boundary"]["realized_fill_count"] == 0
    assert study["decision_boundary"]["realized_exit_count"] == 0


def test_initialized_shadow_has_no_realized_values() -> None:
    summary = shadow.initialize_shadow()
    orders = pd.read_parquet(summary["files"]["frozen_orders"]["path"])

    assert summary["realized_performance_available"] is False
    assert orders["status"].eq("pending_forward_fill").all()
    assert orders["entry_price_raw"].isna().all()
    assert orders["realized_net_return"].isna().all()


def test_shadow_summary_validates() -> None:
    shadow.initialize_shadow()

    result = shadow.validate_summary(shadow.DEFAULT_OUTPUT_ROOT / "summary.json")

    assert result["status"] == "ok"
    assert all(result["checks"].values())
