from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from daily_research.path_policy import seq100_causal_path_structure as study_module
from daily_research.path_policy import (
    seq100_causal_path_structure_validate as validator,
)


def _study() -> dict:
    return study_module.load_study()


def _path(close: np.ndarray) -> dict[str, np.ndarray]:
    close = np.asarray(close, dtype=float)
    return {
        "adj_high": close * 1.01,
        "adj_low": close * 0.99,
        "adj_close": close,
        "amount": np.linspace(1.0e8, 2.0e8, len(close)),
        "date_indices": np.arange(len(close), dtype=np.int64),
    }


def test_load_study_rejects_orthodox_truth_claim() -> None:
    study = _study()
    broken = copy.deepcopy(study)
    broken["study_id"] = "wrong"
    path = study_module.WORKSPACE_ROOT / "tmp_invalid_causal_path_study.json"
    try:
        study_module._write_json(path, broken)
        with pytest.raises(ValueError, match="study_id_mismatch"):
            study_module.load_study(path)
    finally:
        path.unlink(missing_ok=True)


def test_causal_volatility_excludes_current_return() -> None:
    values = np.zeros(14, dtype=float)
    values[-1] = 0.20
    volatility = study_module._causal_volatility(
        values,
        window=20,
        minimum_observations=10,
        cost_log_return=0.006,
    )
    assert volatility[-1] == pytest.approx(0.006)
    extended = study_module._causal_volatility(
        np.append(values, 0.20),
        window=20,
        minimum_observations=10,
        cost_log_return=0.006,
    )
    assert extended[-1] > 0.006


def test_directional_confirmation_is_written_on_confirmation_date() -> None:
    values = np.asarray([100.0, 102.0, 104.0, 103.5, 101.5, 100.0])
    parsed = study_module.parse_causal_path(**_path(values), study=_study())
    peak_events = np.flatnonzero(parsed["dc_small_event"] == -1)
    assert len(peak_events) >= 1
    confirmation = int(peak_events[0])
    assert confirmation > 2
    assert parsed["dc_small_event"][2] == 0
    assert parsed["dc_small_mode"][confirmation] == -1


def test_confirmed_fractal_waits_until_right_bar_is_locked() -> None:
    study = _study()
    high = np.asarray([9.0, 11.0, 10.0, 9.5])
    low = np.asarray([8.0, 10.0, 9.0, 8.5])
    close = (high + low) / 2.0
    parsed = study_module.parse_causal_path(
        adj_high=high,
        adj_low=low,
        adj_close=close,
        amount=np.full(4, 1.0e8),
        date_indices=np.arange(4),
        study=study,
    )
    assert parsed["chan_provisional_fractal"][2] == -1
    assert parsed["chan_confirmed_fractal_event"][2] == 0
    assert parsed["chan_confirmed_fractal_event"][3] == -1


def test_parser_is_prefix_invariant_on_mixed_path() -> None:
    random = np.random.default_rng(20260807)
    returns = random.normal(0.0005, 0.025, 90)
    close = 20.0 * np.exp(np.cumsum(returns))
    spread = random.uniform(0.005, 0.035, len(close))
    frame = pd.DataFrame(
        {
            "adj_high": close * np.exp(spread),
            "adj_low": close * np.exp(-spread),
            "adj_close": close,
            "amount": np.exp(random.normal(18.0, 0.8, len(close))),
            "date_idx": np.arange(100, 100 + len(close)),
        }
    )
    result = validator.verify_prefix_invariance(
        frame,
        _study(),
        cutoffs=[1, 3, 4, 9, 20, 47, 90],
        tolerance=1.0e-7,
    )
    assert result["cutoffs"] == 7
    assert result["maximum_float_difference"] == 0.0


def test_same_date_risk_matching_uses_absent_pattern_controls() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["2020-01-02"] * 4 + ["2020-01-03"] * 2,
            "date_idx": [1] * 4 + [2] * 2,
            "risk_match_cell_code": [10, 10, 20, 20, 10, 10],
            "actual_buy_advantage_vs_cash": [0.10, -0.02, 0.04, -0.06, 0.08, -0.04],
            "actual_positive": [True, False, True, False, True, False],
            "actual_upside_component": [0.10, 0.0, 0.04, 0.0, 0.08, 0.0],
            "actual_downside_component": [0.0, 0.02, 0.0, 0.06, 0.0, 0.04],
            "pattern_nested_up_pullback": [True, False, True, False, True, False],
        }
    )
    result = study_module._daily_pattern_diagnostics(
        frame,
        pattern="nested_up_pullback",
        cost_scenario="base",
        evaluation_year=2020,
    )
    first = result[result["date_idx"].eq(1)].iloc[0]
    assert first["candidate_action_value"] == pytest.approx(0.07)
    assert first["matched_action_value_difference"] == pytest.approx(0.11)
    assert first["matched_failure_positive_rows"] == 0


def test_feature_contract_has_provisional_and_confirmed_states() -> None:
    columns = set(study_module.FEATURE_COLUMNS)
    assert "chan_provisional_fractal" in columns
    assert "chan_confirmed_fractal_event" in columns
    assert "dc_small_mode" in columns
    assert "dc_large_exhaustion" in columns
    assert "pattern_zone_reentry_after_up_breakout" in columns
