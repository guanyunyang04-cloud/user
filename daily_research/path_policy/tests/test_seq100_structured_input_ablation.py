from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from daily_research.path_policy import qdp_v2_sequence_path_training as training
from daily_research.path_policy import seq100_structured_input_ablation as ablation
from daily_research.path_policy.seq100_development import _parser


def test_cli_surface_and_protected_boundaries() -> None:
    prepare = _parser().parse_args(["prepare-structured-input-ablation"])
    run = _parser().parse_args(
        ["run-structured-input-ablation", "--max-tasks", "2"]
    )
    status = _parser().parse_args(["status-structured-input-ablation"])
    review = _parser().parse_args(
        ["review-structured-input-stage1-capital"]
    )
    review_stage = _parser().parse_args(
        ["review-structured-input-stage-capital", "--stage", "2"]
    )
    summarize = _parser().parse_args(["summarize-structured-input-ablation"])
    verify = _parser().parse_args(["verify-structured-input-ablation"])
    assert prepare.command == "prepare-structured-input-ablation"
    assert run.max_tasks == 2
    assert status.command == "status-structured-input-ablation"
    assert review.command == "review-structured-input-stage1-capital"
    assert review_stage.stage == 2
    assert summarize.command == "summarize-structured-input-ablation"
    assert verify.command == "verify-structured-input-ablation"
    protected = ablation._semantic_contract(512)["protected_boundaries"]
    assert protected == {
        "update_qdp": False,
        "call_provider": False,
        "mutate_base_pack": False,
        "mutate_old_checkpoint": False,
        "change_active_execution": False,
        "evaluate_2026": False,
        "create_final_model": False,
    }


@pytest.mark.parametrize(
    ("variant", "expected_profile", "expected_order", "expected_dim"),
    [
        (
            ablation.InputVariant(100),
            "daily_only",
            ["daily_raw", "daily_state"],
            32,
        ),
        (
            ablation.InputVariant(100, turnover=True),
            "daily_only_turnover",
            ["daily_raw", "daily_state", "turnover"],
            35,
        ),
        (
            ablation.InputVariant(100, intraday=True),
            "daily_only_intraday",
            ["daily_raw", "daily_state", "intraday_micro"],
            40,
        ),
        (
            ablation.InputVariant(180, turnover=True, intraday=True),
            "daily_only_turnover_intraday",
            ["daily_raw", "daily_state", "turnover", "intraday_micro"],
            43,
        ),
    ],
)
def test_registered_input_shapes(
    variant: ablation.InputVariant,
    expected_profile: str,
    expected_order: list[str],
    expected_dim: int,
) -> None:
    assert variant.input_channel_profile == expected_profile
    assert training._input_channel_order(expected_profile) == expected_order
    assert variant.input_dim == expected_dim


def test_intraday_flat_day_and_amount_hhi() -> None:
    prices = np.full(48, 10.0)
    volume = np.arange(1, 49, dtype=np.float64)
    amount = volume * prices
    features, valid, close_position = ablation.derive_intraday_features(
        open_values=prices,
        high_values=prices,
        low_values=prices,
        close_values=prices,
        volume_values=volume,
        amount_values=amount,
    )
    weights = amount / amount.sum()
    assert valid is True
    assert features[0] == pytest.approx(0.0)
    assert features[1] == pytest.approx(0.0)
    assert features[2] == pytest.approx(0.0)
    assert features[3] == pytest.approx(0.0)
    assert features[4] == pytest.approx(0.5)
    assert features[5] == pytest.approx(0.0)
    assert features[6] == pytest.approx(np.square(weights).sum())
    assert close_position == pytest.approx(0.5)


def test_intraday_high_before_low_and_registered_efficiency_formula() -> None:
    open_values = np.full(48, 10.0)
    close_values = np.linspace(10.0, 11.0, 48)
    high_values = np.maximum(open_values, close_values) + 0.1
    low_values = np.minimum(open_values, close_values) - 0.1
    high_values[5] = 13.0
    low_values[10] = 7.0
    volume = np.ones(48)
    amount = close_values.copy()
    features, valid, _position = ablation.derive_intraday_features(
        open_values=open_values,
        high_values=high_values,
        low_values=low_values,
        close_values=close_values,
        volume_values=volume,
        amount_values=amount,
    )
    first_move = abs(np.log(close_values[0] / open_values[0]))
    later = np.abs(np.log(close_values[1:] / close_values[:-1])).sum()
    expected = np.log(close_values[-1] / open_values[0]) / (
        first_move + later + 1.0e-12
    )
    assert valid is True
    assert features[0] == pytest.approx(np.clip(expected, -1.0, 1.0))
    assert features[3] == pytest.approx(10 / 47)
    assert features[4] == pytest.approx(1.0)


def test_intraday_invalid_for_missing_bar_zero_flow_or_illegal_ohlc() -> None:
    prices = np.full(48, 10.0)
    cases = [
        dict(
            open_values=prices[:-1],
            high_values=prices[:-1],
            low_values=prices[:-1],
            close_values=prices[:-1],
            volume_values=np.ones(47),
            amount_values=np.ones(47),
        ),
        dict(
            open_values=prices,
            high_values=prices,
            low_values=prices,
            close_values=prices,
            volume_values=np.zeros(48),
            amount_values=np.zeros(48),
        ),
        dict(
            open_values=prices,
            high_values=np.r_[9.0, prices[1:]],
            low_values=prices,
            close_values=prices,
            volume_values=np.ones(48),
            amount_values=np.ones(48) * 10,
        ),
    ]
    for case in cases:
        features, valid, close_position = ablation.derive_intraday_features(**case)
        assert valid is False
        assert np.isnan(features).all()
        assert np.isnan(close_position)


def test_long_fold_filters_only_early_training_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source.parquet"
    frame = pd.DataFrame(
        {
            "split": ["train", "train", "train", "development"],
            "date_idx": [178, 179, 180, 300],
            "trade_date": [
                "2010-09-28",
                "2010-09-29",
                "2010-09-30",
                "2023-01-03",
            ],
            "symbol_idx": [0, 0, 0, 0],
        }
    )
    frame.to_parquet(source_path, index=False)
    monkeypatch.setattr(ablation, "OVERLAY_ROOT", tmp_path / "overlay")
    monkeypatch.setitem(ablation.EXPECTED_LONG_TRAIN_ROWS, 2023, 2)
    target, audit = ablation._long_sample_index(
        {"sample_index_path": str(source_path)}, 2023
    )
    observed = pd.read_parquet(target)
    assert observed["trade_date"].tolist() == [
        "2010-09-29",
        "2010-09-30",
        "2023-01-03",
    ]
    assert audit["train_row_count"] == 2
    assert audit["first_train_signal"] == "2010-09-29"


def _evaluation(
    *,
    descriptor: dict[str, object],
    top3: list[float],
    regret: list[float],
    close_mae: list[float],
    concentration: list[float] | None = None,
) -> dict[str, object]:
    concentration = concentration or [0.4, 0.4, 0.4]
    yearly = []
    for year, alpha, exit_regret, path_mae, share in zip(
        ablation.DEVELOPMENT_YEARS,
        top3,
        regret,
        close_mae,
        concentration,
        strict=True,
    ):
        yearly.append(
            {
                "development_year": year,
                "metrics": {
                    "top3_base_alpha": alpha,
                    "exit_regret": exit_regret,
                    "path_close_mae": path_mae,
                    "candidate_score_coverage": 1.0,
                    "top3_legal_exit_max_day_share": share,
                    "ohlc_geometry_violation_count": 0,
                },
            }
        )
    return {
        "descriptor": descriptor,
        "yearly": yearly,
        "equal_year": {
            "top3_base_alpha": float(np.mean(top3)),
            "worst_year_top3_base_alpha": float(min(top3)),
            "exit_regret": float(np.mean(regret)),
            "maximum_top3_exit_day_share": float(max(concentration)),
            "geometry_violation_count": 0,
        },
        "fairness_by_year": {str(year): {"hash": "same"} for year in ablation.DEVELOPMENT_YEARS},
    }


def test_cohort_diagnostics_never_select_or_reject_strategy() -> None:
    incumbent_descriptor = ablation._initial_incumbent()
    challenger_descriptor = {
        "source": "ablation_run",
        "origin_stage": 1,
        "variant": ablation.InputVariant(180).to_dict(),
    }
    incumbent = _evaluation(
        descriptor=incumbent_descriptor,
        top3=[0.01, 0.02, 0.03],
        regret=[0.20, 0.20, 0.20],
        close_mae=[0.10, 0.10, 0.10],
    )
    challenger = _evaluation(
        descriptor=challenger_descriptor,
        top3=[0.011, 0.021, 0.031],
        regret=[0.19, 0.19, 0.19],
        close_mae=[0.09, 0.10, 0.11],
    )
    favorable = ablation.stage_cohort_diagnostics(
        stage=1, incumbent=incumbent, challenger=challenger
    )
    assert "challenger_passed" not in favorable
    assert "selected_incumbent" not in favorable
    assert favorable["cohort_diagnostics"]["selection_authority"] is False
    assert favorable["decision_scope"] == (
        "cohort_diagnostics_pending_capital_review"
    )
    assert favorable["capital_efficiency_review"]["status"] == "pending"
    challenger["yearly"][0]["metrics"]["top3_base_alpha"] = -0.001
    unfavorable = ablation.stage_cohort_diagnostics(
        stage=1, incumbent=incumbent, challenger=challenger
    )
    assert "challenger_passed" not in unfavorable
    assert "selected_incumbent" not in unfavorable
    assert unfavorable["cohort_diagnostics"]["selection_authority"] is False


def test_stage2_waits_for_capital_review() -> None:
    branch = ablation._initial_incumbent()
    decisions = {
        "stages": [
            {
                "stage": 1,
                "capital_efficiency_review": {"status": "pending"},
            }
        ]
    }
    with pytest.raises(ValueError, match="finite-capital"):
        ablation._incumbent_before_stage(decisions, 2)
    decisions["stages"][0]["capital_efficiency_review"] = {
        "status": "completed",
        "training_branch_descriptor": branch,
    }
    assert ablation._incumbent_before_stage(decisions, 2) == branch


def test_capital_speed_model_supersedes_legacy_component_review() -> None:
    legacy = ablation._initial_incumbent()
    selected = {
        "source": "ablation_run",
        "origin_stage": 1,
        "variant": ablation.InputVariant(180).to_dict(),
    }
    decisions = {
        "stages": [
            {
                "stage": 1,
                "capital_speed_review": {
                    "status": "completed",
                    "selected_model_descriptor": selected,
                },
                "capital_efficiency_review": {
                    "status": "completed",
                    "training_branch_descriptor": legacy,
                },
            }
        ]
    }
    assert ablation._incumbent_before_stage(decisions, 2) == selected


def test_stage3_challenger_extends_one_complete_model(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = {
        "source": "ablation_run",
        "origin_stage": 2,
        "variant": ablation.InputVariant(180, turnover=True).to_dict(),
    }
    monkeypatch.setattr(ablation, "STUDY_ROOT", Path("missing-study-root"))
    decisions = {
        "stages": [
            {
                "stage": 2,
                "challenger": selected,
                "capital_efficiency_review": {
                    "status": "completed",
                    "training_branch_descriptor": selected,
                },
            }
        ]
    }
    challenger = ablation._challenger_for_stage(decisions, 3)
    assert challenger["variant"] == ablation.InputVariant(
        180, turnover=True, intraday=True
    ).to_dict()


def test_hybrid_uses_short_ranking_and_long_exit_forecast() -> None:
    from daily_research.path_policy import seq100_finite_capital_backtest as finite

    symbols = np.arange(4, dtype=np.int32)
    ranking = finite.ForecastBook("ranking")
    ranking.add_day(
        date_idx=10,
        symbol_idx=symbols,
        score=np.asarray([4.0, 3.0, 2.0, 1.0]),
        planned_day=np.asarray([40, 40, 40, 40], dtype=np.int16),
    )
    exits = finite.ForecastBook("exits")
    exits.add_day(
        date_idx=10,
        symbol_idx=symbols,
        score=np.asarray([0.1, -0.2, 5.0, 4.0]),
        planned_day=np.asarray([12, 13, 14, 15], dtype=np.int16),
    )
    hybrid = ablation._hybrid_forecast_book(
        ranking, exits, profile_name="hybrid"
    )
    assert hybrid.days[10].top3_symbol_idx == (0, 1, 2)
    assert hybrid.lookup(10, 0) == pytest.approx((0.1, 12))
    assert hybrid.lookup(10, 1) == pytest.approx((-0.2, 13))


def test_primary_capital_selection_uses_unit_time_account_profit() -> None:
    rows = []
    multiples = {
        "rank100_exit100": 1.5,
        "rank180_exit180": 1.8,
        "rank100_exit180": 2.0,
    }
    for configuration, multiple in multiples.items():
        for slots in ablation.CAPITAL_REVIEW_SLOTS:
            rows.append(
                {
                    "comparison_layer": "full_model_path_exit",
                    "configuration": configuration,
                    "policy": "rolling_path",
                    "cost_scenario": "base",
                    "slot_count": slots,
                    "liquidated_ending_equity_cny": 1_000_000.0 * multiple,
                    "starting_cash_cny": 1_000_000.0,
                    "signal_period_cagr_trading_days": multiple - 1.0,
                    "signal_period_total_return": multiple - 1.0,
                    "equity_path_session_count": 807,
                }
            )
    selected = ablation._stage1_primary_account_selection(pd.DataFrame(rows))
    assert selected["selected_configuration"] == "rank100_exit180"
    assert selected["objective_is_unit_time_profit"] is True
    assert selected["common_calendar_horizon_sessions"] == 807
    assert selected["cohort_metrics_used_as_selection_gate"] is False
    assert selected["selected_configuration_by_slot"] == {
        str(slots): "rank100_exit180"
        for slots in ablation.CAPITAL_REVIEW_SLOTS
    }


def test_primary_capital_selection_rejects_mixed_calendar_horizons() -> None:
    rows = []
    configurations = (
        "rank100_exit100",
        "rank180_exit180",
        "rank100_exit180",
    )
    for configuration_index, configuration in enumerate(configurations):
        for slots in ablation.CAPITAL_REVIEW_SLOTS:
            rows.append(
                {
                    "comparison_layer": "full_model_path_exit",
                    "configuration": configuration,
                    "policy": "rolling_path",
                    "cost_scenario": "base",
                    "slot_count": slots,
                    "liquidated_ending_equity_cny": 1_500_000.0,
                    "starting_cash_cny": 1_000_000.0,
                    "signal_period_cagr_trading_days": 0.5,
                    "signal_period_total_return": 0.5,
                    "equity_path_session_count": 807 + configuration_index,
                }
            )
    with pytest.raises(ValueError, match="common positive calendar horizon"):
        ablation._stage1_primary_account_selection(pd.DataFrame(rows))


def test_legacy_cohort_gate_migrates_to_diagnostics_only(tmp_path: Path) -> None:
    branch = ablation._initial_incumbent()
    payload = {
        "schema_version": 1,
        "artifact_type": "seq100_structured_input_ablation_stage_decisions",
        "study_id": ablation.STUDY_ID,
        "stages": [
            {
                "stage": 1,
                "incumbent": branch,
                "challenger": branch,
                "gates": {"top3_alpha_higher": False},
                "challenger_passed": False,
                "selected_incumbent": branch,
                "cohort_reference_preferred": branch,
                "overall_model_rejection": False,
                "capital_efficiency_review": {
                    "status": "completed",
                    "training_branch_descriptor": branch,
                    "selected_strategy": {
                        "ranking_descriptor": branch,
                        "exit_descriptor": branch,
                    },
                },
            }
        ],
    }
    ablation._write_json(tmp_path / "stage_decisions.json", payload)
    migrated = ablation._load_stage_decisions(tmp_path)
    row = migrated["stages"][0]
    assert migrated["schema_version"] == 2
    assert "gates" not in row
    assert "challenger_passed" not in row
    assert "selected_incumbent" not in row
    assert row["decision_scope"] == "continuous_account_profit_primary"
    assert row["cohort_diagnostics"]["selection_authority"] is False


def test_noncircular_moving_block_is_deterministic_and_does_not_wrap() -> None:
    groups = [np.arange(100, dtype=np.float64), np.arange(100, dtype=np.float64) + 1]
    first = ablation.noncircular_grouped_moving_block_interval(
        groups, block_length=60, replications=200, seed=7
    )
    second = ablation.noncircular_grouped_moving_block_interval(
        groups, block_length=60, replications=200, seed=7
    )
    assert first == second
    assert first["method"].startswith("noncircular")
    assert first["formal_gate"] is False
    with pytest.raises(ValueError, match="shorter"):
        ablation.noncircular_grouped_moving_block_interval(
            [np.arange(59)], block_length=60, replications=10
        )


@pytest.mark.parametrize(
    ("exit_code", "memory_status", "valid", "expected"),
    [
        (0, "completed", True, "task_completed"),
        (1, "failed", False, "task_failed"),
        (137, "killed_low_available_memory", False, "low_memory_terminated"),
        (0, "completed", False, "invalid_terminal_artifacts"),
    ],
)
def test_monitor_terminal_classification(
    exit_code: int, memory_status: str, valid: bool, expected: str
) -> None:
    assert (
        ablation.classify_monitor_terminal(
            exit_code=exit_code,
            memory_guard_status=memory_status,
            artifacts_valid=valid,
        )
        == expected
    )


def test_monitor_event_survives_closed_console(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def closed_console(*_args: object, **_kwargs: object) -> None:
        raise OSError(22, "Invalid argument")

    monkeypatch.setattr("builtins.print", closed_console)
    ablation._append_monitor_event(
        tmp_path,
        {
            "event": "progress",
            "status": "running",
            "message": "still training",
        },
    )

    monitor = ablation._read_json(tmp_path / "monitor.json")
    assert monitor["event"] == "progress"
    assert monitor["message"] == "still training"
    events = (tmp_path / "monitor_events.jsonl").read_text(encoding="utf-8")
    assert '"event": "progress"' in events
