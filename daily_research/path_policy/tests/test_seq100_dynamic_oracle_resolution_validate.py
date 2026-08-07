from __future__ import annotations

import numpy as np

from daily_research.path_policy import seq100_dynamic_oracle as oracle
from daily_research.path_policy import seq100_dynamic_oracle_resolution as resolution
from daily_research.path_policy import (
    seq100_dynamic_oracle_resolution_validate as validate,
)
from daily_research.path_policy.tests.test_seq100_dynamic_oracle_resolution import (
    _market,
)


def test_independent_trace_matches_cash_graph(tmp_path) -> None:
    market = _market()
    oracle.solve_relaxed_oracle(
        market=market,
        study=oracle.load_study(),
        cost_scenario="base",
        output_root=tmp_path,
        write_outputs=True,
    )
    scenario_root = tmp_path / "relaxed_oracle" / "cost=base"
    record = {
        "cash_action_symbol": {
            "path": str(scenario_root / "cash_action_symbol.npy"),
            "sha256": resolution.replay.sha256(
                scenario_root / "cash_action_symbol.npy"
            ),
        },
        "cash_log_value": {
            "path": str(scenario_root / "cash_log_value.npy"),
            "sha256": resolution.replay.sha256(scenario_root / "cash_log_value.npy"),
        },
        "holding_exit_policy": {
            "path": str(scenario_root / "holding_exit_policy.npy"),
            "sha256": resolution.replay.sha256(
                scenario_root / "holding_exit_policy.npy"
            ),
        },
    }
    graph = resolution.build_cash_policy_graph(
        market=market, oracle_record=record
    )
    cash_action = np.load(scenario_root / "cash_action_symbol.npy")
    hold_exit = np.load(scenario_root / "holding_exit_policy.npy")
    for signal_local in range(market.day_count - 2):
        for symbol_idx in range(market.symbol_count):
            first, meeting = validate._trace_resolution(
                market=market,
                cash_action_symbol=cash_action,
                hold_exit_policy=hold_exit,
                signal_local=signal_local,
                symbol_idx=symbol_idx,
            )
            if first < 0:
                continue
            expected = graph.first_common_cash(
                np.asarray([first]), np.asarray([signal_local + 1])
            )[0]
            assert int(expected) == meeting


def test_validator_rejects_missing_manifest(tmp_path) -> None:
    try:
        validate.validate(output_root=tmp_path)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing manifest should be rejected")
