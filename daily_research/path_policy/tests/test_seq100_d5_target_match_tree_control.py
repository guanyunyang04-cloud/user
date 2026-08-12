from __future__ import annotations

from daily_research.path_policy.seq100_d5_target_match_tree_control import (
    robust_pareto_comparison,
)


def _result(
    *, total: float, drawdown: float, years: int, hac: float, cap10: float
) -> dict:
    return {
        "primary": {
            "total_net_return": total,
            "maximum_drawdown": drawdown,
            "positive_year_count": years,
            "daily_hac20_net_return": {"lower": hac},
        },
        "winner_cap_10pct": {"total_net_return": cap10},
    }


def test_robust_pareto_requires_noninferiority_on_every_metric() -> None:
    control = _result(total=0.5, drawdown=-0.10, years=5, hac=0.0, cap10=0.05)
    mixed = _result(total=0.6, drawdown=-0.20, years=5, hac=0.01, cap10=0.06)
    comparison = robust_pareto_comparison(matched_d5=mixed, d10_control=control)
    assert not comparison["matched_d5_robust_pareto_dominates"]
    assert not comparison["metrics"]["maximum_drawdown"]["matched_d5_noninferior"]


def test_robust_pareto_accepts_strict_improvement_without_regression() -> None:
    control = _result(total=0.5, drawdown=-0.10, years=5, hac=0.0, cap10=0.05)
    better = _result(total=0.6, drawdown=-0.09, years=5, hac=0.01, cap10=0.05)
    comparison = robust_pareto_comparison(matched_d5=better, d10_control=control)
    assert comparison["matched_d5_robust_pareto_dominates"]
