from __future__ import annotations

import pandas as pd
import pytest

from daily_research.path_policy import seq100_strict_chan_outcome_screen as screen
from daily_research.path_policy import seq100_strict_chan_stratified_audit as sample


def test_expanded_sample_contract_has_140_outcome_blind_cases() -> None:
    path = screen.WORKSPACE_ROOT / (
        "daily_research/studies/seq100_strict_chan_outcome_sample_v1.json"
    )
    spec = sample.load_stratified_spec(path)
    assert spec["study_id"] == sample.OUTCOME_SAMPLE_STUDY_ID
    assert sum(int(item["samples"]) for item in spec["selection"]["strata"]) == 140
    assert spec["boundaries"]["return_test_performed"] is False


def test_strict_gate_requires_development_and_validation() -> None:
    study = screen._load_study()
    common = {
        "point_type": 3,
        "horizon_sessions": 5,
        "event_count": 100,
        "case_count": 30,
        "case_mean_net_base": 0.02,
        "case_mean_net_stress": 0.01,
        "case_mean_excess_base": 0.01,
        "bootstrap_case_excess_low": 0.001,
    }
    aggregate = pd.DataFrame(
        [
            {"evaluation_split": "development", **common},
            {"evaluation_split": "validation", **common},
        ]
    )
    result = screen._gate(aggregate, study)
    assert bool(result.iloc[0]["strict_gate_passed"])

    failed = screen._gate(aggregate.iloc[:1], study)
    assert not bool(failed.iloc[0]["strict_gate_passed"])
    assert "validation:missing" in failed.iloc[0]["failure_reasons"]


def test_baseline_uses_adjusted_return_and_raw_cost_price() -> None:
    study = screen._load_study()
    case = {
        "case_id": "case",
        "symbol": "000001.SZ",
        "stratum_id": "2020_2021_sz",
        "stratum_years": [2020, 2021],
        "focal_date": "2020-01-02",
        "display_start_date": "2020-01-02",
        "display_end_date": "2020-01-02",
    }
    daily = pd.DataFrame(
        {
            "trade_date": ["2020-01-02", "2020-01-03"],
            "open": [5.0, 5.0],
            "close": [5.0, 5.5],
            "raw_open": [100.0, 100.0],
            "raw_close": [100.0, 110.0],
        }
    )
    rows = screen._baseline_rows(
        case=case,
        daily=daily,
        horizons=[1],
        costs=study["execution"]["costs"],
    )
    assert len(rows) == 1
    assert rows[0]["raw_return"] == pytest.approx(0.1)
    assert 0 < rows[0]["net_return_base"] < 0.1
