from __future__ import annotations

from daily_research.path_policy import seq100_daily_action_value_validate as validate


def test_validation_schema_is_stable() -> None:
    assert validate.VALIDATION_SCHEMA == "seq100_daily_action_value_validation/1"
