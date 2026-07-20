from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from daily_research.path_policy import seq100_development
from daily_research.path_policy import seq100_structured_training_window as window


def test_cli_surface_and_long_task_default() -> None:
    parser = seq100_development._parser()
    commands = {
        action.dest: set(action.choices or {})
        for action in parser._actions
        if action.dest == "command"
    }["command"]
    assert {
        "prepare-structured-training-window",
        "run-structured-training-window",
        "evaluate-structured-training-window",
        "status-structured-training-window",
        "summarize-structured-training-window",
        "verify-structured-training-window",
    }.issubset(commands)
    args = parser.parse_args(["run-structured-training-window"])
    assert args.max_tasks == 1


@pytest.mark.parametrize(
    ("years", "expected_date", "expected_rows"),
    [
        (12, "2012-08-30", 6_273_740),
        (8, "2016-08-30", 4_859_679),
        (5, "2019-08-30", 3_354_874),
        (3, "2021-08-30", 2_096_298),
    ],
)
def test_registered_2025_window_boundaries(
    years: int, expected_date: str, expected_rows: int
) -> None:
    view = window._read_json(window._expanding_view(2025))
    samples = pd.read_parquet(Path(str(view["sample_index_path"])))
    train = samples[samples["split"].astype(str).eq("train")]
    start = window.bounded_start_date(
        safe_train_signal_end=str(view["development_walkforward"]["safe_train_signal_end"]),
        window_years=years,
        legal_signal_dates=sorted(train["trade_date"].astype(str).unique()),
    )
    assert start == expected_date
    assert int(train["trade_date"].astype(str).ge(start).sum()) == expected_rows
    assert int(view["development_walkforward"]["purged_signal_date_count"]) == 80
    assert int(view["candidate_count"]) == 724_125
    assert int(view["sample_count_by_split"]["development"]) == 723_807


def test_shortlist_is_top_two_plus_five_percent_and_keeps_expanding() -> None:
    result = window.shortlist_from_scores(
        {
            "expanding": 0.70,
            "window12y": 1.00,
            "window8y": 0.96,
            "window5y": 0.94,
            "window3y": 0.50,
        }
    )
    assert result["champion_model_id"] == "window12y"
    assert result["shortlisted_short_model_ids"] == ["window12y", "window8y"]
    assert "expanding" in result["selected_model_ids"]


def _capital_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_id, offset in (("expanding", 0.0), ("window8y", 0.2)):
        for top_k, slots in window.CAPITAL_CONFIGS:
            for policy_name, policy_kind, day, score in (
                ("model_plan", "model_plan", None, 0.20),
                ("rolling_path", "rolling", None, 0.30),
                ("fixed_d7", "fixed", 7, 0.40),
                ("fixed_d9", "fixed", 9, 0.90),
            ):
                for cost in window.COST_SCENARIOS:
                    rows.append(
                        {
                            "model_id": model_id,
                            "top_k": top_k,
                            "slot_count": slots,
                            "policy_name": policy_name,
                            "policy_kind": policy_kind,
                            "fixed_day": day,
                            "cost_scenario": cost,
                            "annualized_log_growth": score + offset,
                            "signal_period_total_return": score,
                            "signal_period_maximum_drawdown": -0.2,
                            "closed_trade_count": 10,
                            "winning_trade_rate": 0.6,
                            "mean_occupied_sessions": 7.0,
                            "mean_signal_capital_utilization": 0.8,
                            "skipped_no_slot_signal_count": 3,
                            "net_pnl_per_deployed_capital_session": 0.001,
                        }
                    )
    return pd.DataFrame(rows)


def test_screen_selection_uses_three_configs_stress_and_registered_fallback() -> None:
    result = window.select_capital_strategies(
        _capital_rows(), allowed_policy_names=window.SCREEN_SELECTION_POLICIES
    )
    assert result["champion_model_id"] == "window8y"
    assert result["scores"] == pytest.approx({"expanding": 0.4, "window8y": 0.6})
    assert len(result["selected_strategies"]) == 6
    assert {row["policy_name"] for row in result["selected_strategies"]} == {
        "fixed_d7"
    }


def test_confirmation_can_jointly_select_any_fixed_day() -> None:
    result = window.select_capital_strategies(
        _capital_rows(), allowed_policy_names=None
    )
    assert {row["policy_name"] for row in result["selected_strategies"]} == {
        "fixed_d9"
    }


def test_epoch_confirmation_reasons_cover_margin_year_and_loss_conflicts() -> None:
    reasons = window.epoch_confirmation_reasons(
        scores={"expanding": 1.00, "window8y": 1.03, "window5y": 1.02},
        champion_id="window8y",
        yearly_returns={
            "expanding": {2023: 0.2, 2024: 0.2, 2025: 0.2},
            "window8y": {2023: 0.1, 2024: 0.3, 2025: 0.1},
        },
        cohort_losses={
            "expanding": {2023: 0.10, 2024: 0.10, 2025: 0.10},
            "window8y": {2023: 0.10, 2024: 0.13, 2025: 0.10},
        },
    )
    assert "champion_lead_over_expanding_below_5pct" in reasons
    assert "champion_lead_over_runner_up_below_5pct" in reasons
    assert "champion_beats_expanding_in_fewer_than_two_years" in reasons
    assert "epoch1_development_price_loss_exceeds_expanding_by_25pct" in reasons


def test_expanding_champion_does_not_compare_against_itself() -> None:
    reasons = window.epoch_confirmation_reasons(
        scores={"expanding": 1.05, "window8y": 0.85},
        champion_id="expanding",
        strongest_short_id="window8y",
        yearly_returns={
            "expanding": {2023: 0.3, 2024: 0.3, 2025: 0.3},
            "window8y": {2023: 0.2, 2024: 0.4, 2025: 0.2},
        },
        cohort_losses={
            "expanding": {2023: 0.10, 2024: 0.10, 2025: 0.10},
            "window8y": {2023: 0.11, 2024: 0.11, 2025: 0.11},
        },
    )
    assert reasons == []


@pytest.mark.parametrize(
    ("exit_code", "memory", "valid", "expected"),
    [
        (0, "completed", True, "task_completed"),
        (0, "completed", False, "invalid_terminal_artifacts"),
        (1, "failed", False, "task_failed"),
        (1, "killed_low_available_memory", False, "low_memory_terminated"),
    ],
)
def test_monitor_terminal_classification(
    exit_code: int, memory: str, valid: bool, expected: str
) -> None:
    assert (
        window.classify_monitor_terminal(
            exit_code=exit_code,
            memory_guard_status=memory,
            artifacts_valid=valid,
        )
        == expected
    )


def test_capital_grid_has_all_curves_and_own_exits() -> None:
    jobs = window._capital_jobs("test", ["expanding", "window8y"])
    assert len(jobs) == 2 * 3 * (59 + 2) * 2
    assert {job.policy_name for job in jobs if job.policy_kind == "fixed"} == {
        f"fixed_d{day}" for day in range(2, 61)
    }
    assert {job.policy_name for job in jobs if job.policy_kind != "fixed"} == {
        "model_plan",
        "rolling_path",
    }


def test_source_declares_no_stage3_or_provider_work() -> None:
    contract = window._semantic_contract()
    protected = contract["protected_boundaries"]
    assert protected["provider_calls"] == 0
    assert protected["stage3_started"] is False
    assert protected["qdp_read_only"] is True
    assert protected["old_artifacts_mutated"] is False
