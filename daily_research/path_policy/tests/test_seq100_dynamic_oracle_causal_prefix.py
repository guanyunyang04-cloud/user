from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_dynamic_oracle as oracle
from daily_research.path_policy import seq100_dynamic_oracle_causal_prefix as prefix
from daily_research.path_policy import seq100_dynamic_oracle_resolution as resolution
from daily_research.path_policy.tests.test_seq100_dynamic_oracle_resolution import (
    _market,
)


def test_study_contract_keeps_final_rows_out_of_causal_outputs() -> None:
    study = prefix.load_study()
    assert study["experience"]["fixed_holding_horizon_used"] is False
    assert study["experience"]["forced_terminal_resolution_is_learnable"] is False
    assert (
        study["experience"]["final_oracle_row_labels_allowed_in_causal_outputs"]
        is False
    )
    assert study["boundaries"]["training_performed"] is False


def test_truncated_market_preserves_absolute_indices() -> None:
    market = _market()
    truncated = prefix.truncate_market(market, terminal_date_idx=3)
    assert truncated.start_idx == market.start_idx
    assert truncated.end_idx == 3
    assert truncated.day_count == 4
    assert truncated.quality_mask.shape == (4, market.symbol_count)
    assert truncated.mark_close.shape == (4, market.symbol_count)
    assert truncated.quality_rows == 4 * market.symbol_count


def test_prefix_version_reconstructs_oracle_action_values(tmp_path: Path) -> None:
    market = _market()
    solution = oracle.solve_relaxed_oracle(
        market=market,
        study=oracle.load_study(),
        cost_scenario="base",
        output_root=tmp_path,
        write_outputs=True,
    )
    graph = resolution.build_cash_policy_graph_from_arrays(
        market=market,
        cash_action_symbol=solution.cash_action_symbol,
        cash_log_value=solution.cash_log,
        holding_exit_policy=solution.hold_exit_policy,
    )
    version = prefix.extract_prefix_version(
        market=market,
        solution=solution,
        graph=graph,
        signal_year=2023,
        as_of_year=2023,
        version_role="signal_year_end",
    )
    labels = pd.concat(
        [pd.read_parquet(record["path"]) for record in solution.label_files],
        ignore_index=True,
    )
    expected = labels.loc[
        labels["buy_action_valid"].astype(bool),
        ["date_idx", "symbol_idx", "buy_advantage_vs_cash"],
    ]
    joined = version.merge(
        expected,
        on=["date_idx", "symbol_idx"],
        validate="one_to_one",
        suffixes=("_prefix", "_oracle"),
    )
    assert len(joined) == len(expected) == len(version)
    assert np.allclose(
        joined["buy_advantage_vs_cash_prefix"],
        joined["buy_advantage_vs_cash_oracle"],
        atol=1.0e-8,
        rtol=0.0,
    )
    decomposition = (
        version["buy_trade_log_return_to_first_cash"].astype(float)
        + version["cash_value_difference_first_cash_vs_wait"].astype(float)
    )
    assert np.allclose(
        decomposition,
        version["buy_advantage_vs_cash"].astype(float),
        atol=5.0e-8,
        rtol=0.0,
    )


def _version_frame(
    *, role: str, as_of_year: int, advantages: tuple[float, float], natural: tuple[bool, bool]
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cost_scenario": ["base", "base"],
            "version_role": [role, role],
            "signal_year": [2023, 2023],
            "as_of_year": [as_of_year, as_of_year],
            "as_of_date": [f"{as_of_year}-12-31", f"{as_of_year}-12-31"],
            "as_of_date_idx": [as_of_year, as_of_year],
            "trade_date": ["2023-01-03", "2023-01-03"],
            "date_idx": [10, 10],
            "symbol_idx": [1, 2],
            "prefix_oracle_cash_buy_selected": [True, False],
            "buy_advantage_vs_cash": list(advantages),
            "buy_first_cash_date_idx": [12, 12],
            "cash_resolution_date_idx": [13, 13],
            "cash_resolution_delay": [3, 3],
            "cash_resolution_is_terminal": [not value for value in natural],
            "sessions_after_resolution": [1 if value else 0 for value in natural],
            "naturally_resolved": list(natural),
            "buy_trade_log_return_to_first_cash": [0.05, -0.05],
            "cash_value_difference_first_cash_vs_wait": [0.05, -0.05],
        }
    )


def test_first_experience_and_updates_are_timestamped() -> None:
    first_version = _version_frame(
        role="signal_year_end",
        as_of_year=2023,
        advantages=(0.10, -0.10),
        natural=(True, False),
    )
    next_version = _version_frame(
        role="next_year_end",
        as_of_year=2024,
        advantages=(0.20, -0.20),
        natural=(True, True),
    )
    first, updates = prefix.build_first_experience_and_updates(
        signal_year_end=first_version,
        next_year_end=next_version,
        tolerance=0.0002,
    )
    assert len(first) == 2
    roles = dict(zip(first["symbol_idx"], first["first_observed_version_role"]))
    assert roles == {1: "signal_year_end", 2: "next_year_end"}
    assert bool(first["naturally_resolved"].all())
    assert len(updates) == 1
    assert int(updates.iloc[0]["symbol_idx"]) == 1
    assert np.isclose(float(updates.iloc[0]["update_minus_previous"]), 0.10)
    assert not bool(updates.iloc[0]["stable_within_tolerance"])
