from __future__ import annotations

import math

from quantlab.data.qdp_v2.share_capital_float_share_invariant_repair import (
    _capped_float_share,
)
from quantlab.data.qdp_v2.share_capital_total_share_repair import (
    TOTAL_SHARE_UNIT_MULTIPLIER,
    _market_value,
    _repair_value,
)


def test_total_share_repair_only_fills_null_and_converts_wan_shares() -> None:
    assert TOTAL_SHARE_UNIT_MULTIPLIER == 10_000.0
    assert _repair_value(None, 12.5) == 125_000.0
    assert _repair_value(float("nan"), "12.5") == 125_000.0
    assert _repair_value(123_456.0, 12.5) == 123_456.0


def test_total_share_repair_keeps_unavailable_values_unknown() -> None:
    assert _repair_value(None, None) is None
    assert _repair_value(None, "") is None
    assert _repair_value(None, 0) is None
    assert _repair_value(None, -1) is None


def test_market_value_uses_unadjusted_close_times_share_count() -> None:
    assert math.isclose(_market_value(12.36, 1_045_909_322), 12_927_439_219.92)
    assert _market_value(None, 100) is None
    assert _market_value(10, None) is None


def test_turnover_inferred_float_share_is_capped_by_total_share() -> None:
    assert _capped_float_share(100.0, 100.001) == 100.0
    assert _capped_float_share(100.0, 80.0) == 80.0
