from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from daily_research.path_policy import seq100_dynamic_oracle as oracle
from daily_research.path_policy import seq100_dynamic_oracle_causal_prefix as prefix
from daily_research.path_policy import (
    seq100_dynamic_oracle_causal_prefix_validate as validate,
)
from daily_research.path_policy import seq100_dynamic_oracle_resolution as resolution
from daily_research.path_policy.tests.test_seq100_dynamic_oracle_resolution import (
    _market,
)


def test_independent_version_trace_matches_prefix_policy(tmp_path: Path) -> None:
    market = _market()
    solution = oracle.solve_relaxed_oracle(
        market=market,
        study=oracle.load_study(),
        cost_scenario="base",
        output_root=tmp_path,
        write_outputs=False,
    )
    graph = resolution.build_cash_policy_graph_from_arrays(
        market=market,
        cash_action_symbol=solution.cash_action_symbol,
        cash_log_value=solution.cash_log,
        holding_exit_policy=solution.hold_exit_policy,
    )
    frame = prefix.extract_prefix_version(
        market=market,
        solution=solution,
        graph=graph,
        signal_year=2023,
        as_of_year=2023,
        version_role="signal_year_end",
    )
    blocking: defaultdict[str, int] = defaultdict(int)
    traced = validate._validate_version_frame(
        frame=frame,
        record={
            "cost_scenario": "base",
            "signal_year": 2023,
            "as_of_year": 2023,
        },
        market=market,
        cash_log=solution.cash_log,
        cash_action=solution.cash_action_symbol,
        hold_exit=solution.hold_exit_policy,
        trace_rows=len(frame),
        blocking=blocking,
    )
    assert traced == len(frame)
    assert not blocking


def test_validator_rejects_missing_manifest(tmp_path: Path) -> None:
    try:
        validate.validate(output_root=tmp_path)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing manifest should be rejected")
