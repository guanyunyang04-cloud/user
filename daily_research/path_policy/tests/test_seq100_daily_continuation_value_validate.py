from __future__ import annotations

import numpy as np

from daily_research.path_policy import (
    seq100_daily_continuation_value_validate as validate,
)


def test_independent_sale_respects_cutoff() -> None:
    class Pack:
        date_values = np.asarray(["2025-12-30", "2025-12-31", "2026-01-02"])
        exit_sellable = np.ones((3, 1), dtype=bool)
        exit_close_raw = np.asarray([[10.0], [11.0], [12.0]], dtype=np.float32)

    pack = Pack()
    date_idx, price = validate._independent_sale(pack, 0, 0, 2, 80, 1)
    assert date_idx == -1
    assert np.isnan(price)


def test_validation_schema_is_stable() -> None:
    assert validate.VALIDATION_SCHEMA.endswith("/1")
