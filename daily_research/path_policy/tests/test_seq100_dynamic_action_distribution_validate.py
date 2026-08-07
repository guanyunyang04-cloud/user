from __future__ import annotations

import numpy as np

from daily_research.path_policy import (
    seq100_dynamic_action_distribution_validate as validate,
)


def test_safe_offsets_handles_missing_keys() -> None:
    keys = np.asarray([1, 3, 5, 7], dtype=np.int64)
    wanted = np.asarray([1, 2, 7, 9], dtype=np.int64)
    offsets, found = validate._safe_offsets(keys, wanted)
    assert np.array_equal(offsets, [0, 1, 3, 4])
    assert np.array_equal(found, [True, False, True, False])
