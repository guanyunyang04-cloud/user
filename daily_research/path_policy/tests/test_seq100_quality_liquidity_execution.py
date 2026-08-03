from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from daily_research.path_policy import seq100_quality_liquidity_execution as execution
from daily_research.path_policy import seq100_v4_economic_realizability as economic


def test_study_and_task_inventory_are_frozen() -> None:
    study = execution.load_study()
    specs = execution.task_specs()

    assert len(specs) == 72
    assert len({spec.task_id for spec in specs}) == 72
    assert study["model_contract"]["feature_count"] == 557
    assert study["model_contract"]["all_557_inputs_used_by_every_head"] is True
    assert study["period"]["signal_years"] == [2023, 2024, 2025]
    assert study["period"]["forbidden_year"] == 2026


def test_study_rejects_a_changed_feature_contract(tmp_path) -> None:
    study = copy.deepcopy(execution.load_study())
    study["model_contract"]["feature_count"] = 556
    path = tmp_path / "study.json"
    path.write_text(json.dumps(study), encoding="utf-8")

    with pytest.raises(ValueError, match="557"):
        execution.load_study(path)


def test_daily_prediction_ranks_preserve_head_semantics() -> None:
    ranks = execution._daily_prediction_ranks(
        mfe10=np.asarray([2.0, 1.0, 3.0]),
        mfe20=np.asarray([3.0, 2.0, 1.0]),
        risk10=np.asarray([-0.30, -0.10, -0.20]),
        risk20=np.asarray([-0.20, -0.30, -0.10]),
        state=np.asarray(
            [
                [0.80, 0.10, 0.10],
                [0.10, 0.20, 0.70],
                [0.30, 0.40, 0.30],
            ]
        ),
    )

    assert ranks.shape == (3, 7)
    assert int(np.argmax(ranks[:, 0])) == 2
    assert int(np.argmax(ranks[:, 1])) == 0
    assert int(np.argmax(ranks[:, 5])) == 1
    assert int(np.argmax(ranks[:, 6])) == 2
    assert int(np.argmin(ranks[:, 2])) == 1
    assert int(np.argmax(ranks[:, 4])) == 1


class _FrozenBook:
    def trading_allowed(self, day: int) -> bool:
        return False


def test_trading_cutoff_returns_no_orders_before_book_access() -> None:
    spec = economic.TaskSpec(
        family="mfe10_primary",
        exposure_mode="target_full",
        slot_count=6,
        buffer_multiplier=0.0,
        cost_scenario="base",
    )

    pending = economic.plan_orders(book=_FrozenBook(), spec=spec, day=10, positions={})

    assert pending.signal_day == 10
    assert pending.sells == ()
    assert pending.unpaired_buys == ()


def test_stable_region_requires_adjacent_slot_counts() -> None:
    rows = []
    for family in execution.FAMILIES:
        for buffer in execution.BUFFER_MULTIPLIERS:
            for slot_count in execution.SLOT_COUNTS:
                rows.append(
                    {
                        "family": family,
                        "buffer_multiplier": buffer,
                        "slot_count": slot_count,
                        "economic_pass": family == execution.FAMILIES[0]
                        and buffer == 0.0
                        and slot_count in {6, 12},
                    }
                )
    regions = execution._stable_regions(pd.DataFrame(rows))

    accepted = regions[regions["robust_region"].astype(bool)]
    assert len(accepted) == 1
    assert accepted.iloc[0]["family"] == execution.FAMILIES[0]
    assert json.loads(accepted.iloc[0]["adjacent_passing_pairs"]) == [[6, 12]]
