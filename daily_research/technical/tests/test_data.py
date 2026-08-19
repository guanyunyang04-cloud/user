from __future__ import annotations

import numpy as np
import pandas as pd

from daily_research.path_policy.seq100_full_market_multitask_forecast import (
    _date_group_sizes as old_date_group_sizes,
)
from daily_research.path_policy.seq100_full_market_multitask_forecast import (
    _date_relevance_labels as old_date_relevance_labels,
)
from daily_research.path_policy.seq100_full_market_multitask_forecast import (
    _equal_date_weights as old_equal_date_weights,
)
from daily_research.path_policy.seq100_full_market_multitask_forecast import (
    build_forward_folds as old_build_forward_folds,
)
from daily_research.technical.data import (
    build_forward_folds,
    date_group_sizes,
    date_relevance_labels,
    equal_date_weights,
    load_data,
)


def test_feature_sets_are_exact_prefixes_of_current_183() -> None:
    data = load_data()
    assert data.feature_count(158) == 158
    assert data.feature_count(183) == 183
    assert data.feature_families[:104] == ("daily_price_volume_technical",) * 104
    assert data.feature_families[104:158] == ("market_state",) * 54
    assert data.feature_families[158:] == ("same_day_5m",) * 25


def test_forward_folds_match_legacy_contract() -> None:
    dates = np.repeat(np.arange(200, dtype=np.int32), 3)
    unique_labels = pd.bdate_range("2019-01-02", periods=200).strftime("%Y-%m-%d")
    labels = np.repeat(np.asarray(unique_labels), 3)
    arguments = {
        "date_idx": dates,
        "trade_date": labels,
        "validation_start_date": str(unique_labels[60]),
        "validation_end_date": str(labels[-1]),
        "fold_count": 5,
        "purge_days": 10,
    }
    assert build_forward_folds(**arguments) == old_build_forward_folds(**arguments)


def test_ranking_helpers_match_legacy_contract() -> None:
    dates = np.repeat(np.arange(4, dtype=np.int32), 15)
    values = np.linspace(-0.2, 0.3, len(dates), dtype=np.float32)
    valid = np.ones(len(dates), dtype=bool)
    valid[[2, 19, 44]] = False
    np.testing.assert_array_equal(date_group_sizes(dates), old_date_group_sizes(dates))
    np.testing.assert_allclose(
        equal_date_weights(dates, valid), old_equal_date_weights(dates, valid)
    )
    np.testing.assert_array_equal(
        date_relevance_labels(dates, values, valid),
        old_date_relevance_labels(dates, values, valid),
    )
