from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from daily_research.path_policy import (
    seq100_margin_top10_adaptive_entry as study_module,
)


def _adaptive_study() -> dict[str, object]:
    return {
        "adaptive_line": {
            "efficiency_lookback": 10,
            "fast_n": 2,
            "slow_n": 30,
            "smooth_n": 2,
        }
    }


def test_frozen_study_loads() -> None:
    study, path = study_module.load_study()

    assert path.is_file()
    assert study["study_id"] == study_module.STUDY_ID
    assert study["epistemic_contract"]["production_claim_allowed"] is False


def test_adaptive_line_is_prefix_invariant() -> None:
    close = np.linspace(10.0, 18.0, 40, dtype=np.float32).reshape(-1, 1)
    close[13, 0] -= 1.2
    close[21, 0] += 0.8
    factor = np.ones_like(close)

    full = study_module._adaptive_line(
        close,
        factor,
        symbol_indices=[0],
        maximum_date_idx=39,
        study=_adaptive_study(),
    )
    prefix = study_module._adaptive_line(
        close[:25],
        factor[:25],
        symbol_indices=[0],
        maximum_date_idx=24,
        study=_adaptive_study(),
    )

    np.testing.assert_allclose(full[:25, 0], prefix[:, 0], rtol=0.0, atol=0.0)


def test_cross_support_matches_user_formula_timing() -> None:
    pack = SimpleNamespace(
        exit_close_raw=np.asarray([[10.0], [10.0], [9.0], [11.0], [12.0]]),
        entry_open_raw=np.asarray([[10.0], [10.0], [9.5], [10.0], [11.5]]),
    )
    factor = np.ones((5, 1), dtype=np.float32)
    high = np.asarray([[10.2], [10.2], [9.8], [11.3], [12.5]])
    low = np.asarray([[9.8], [9.8], [8.8], [9.9], [11.0]])
    adp = np.asarray([[10.0], [10.0], [10.0], [10.0], [10.5]])
    candidates = pd.DataFrame(
        {
            "candidate_id": [1],
            "symbol_idx": [0],
            "date_idx": [4],
            "margin_increment": [100.0],
            "prior_margin_balance": [1_000.0],
            "financing_buy": [120.0],
            "financing_repayment": [20.0],
        }
    )

    result = study_module._attach_technical(
        candidates,
        pack=pack,
        factor=factor,
        high_raw=high,
        low_raw=low,
        adp=adp,
    )

    assert bool(result.loc[0, "cross_support"])
    assert not bool(result.loc[0, "down_close"])
    assert bool(result.loc[0, "user_union_close"])
    assert result.loc[0, "low_to_adp"] > 0.0


def test_first_decrease_exit_uses_next_session_and_times_out() -> None:
    candidates = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "600000.SH"],
            "date_idx": [10, 20],
        }
    )
    timeline = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ", "600000.SH"],
            "available_date_idx": [11, 12, 21],
            "source_date_idx": [10, 11, 20],
            "margin_increment": [5.0, -2.0, 1.0],
        }
    )

    requests, audit = study_module._first_decrease_request(
        candidates, timeline, maximum_request_day=60
    )

    assert requests.tolist() == [13, 80]
    assert audit["decrease_detected_count"] == 1
    assert audit["timeout_count"] == 1
    assert audit["detection_offsets"].tolist() == [2, -1]
