from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from daily_research.path_policy.seq100_qcurve import (
    QCURVE_ENTRY_HORIZONS,
    QCurveCostContract,
    QCurveModel,
    allocate_dynamic_qcurve_portfolio,
    build_open_execution_masks,
    build_qcurve_targets,
    full_day_topk_rank_loss,
    qcurve_distribution_loss,
)


CONTRACT = QCurveCostContract(
    lot_size=100,
    commission_bps=3.0,
    minimum_commission_cny=5.0,
    transfer_fee_bps=0.1,
    slippage_bps=7.0,
    stress_slippage_multiplier=2.0,
    stamp_tax_schedule=(("1900-01-01", 10.0), ("2023-08-28", 5.0)),
    terminal_recovery_fraction=0.0,
)


def test_open_masks_use_strict_tick_limits() -> None:
    open_raw = np.asarray([10.00, 11.00, 9.00, 10.00])
    up = np.asarray([11.00, 11.00, 11.00, 11.00])
    down = np.asarray([9.00, 9.00, 9.00, 9.00])
    observed = np.ones(4, dtype=bool)
    valid = np.ones(4, dtype=bool)
    suspended = np.asarray([False, False, False, True])
    delisted = np.zeros(4, dtype=bool)

    buyable, sellable = build_open_execution_masks(
        open_raw,
        up,
        down,
        observed=observed,
        status_valid=valid,
        suspended=suspended,
        delisted=delisted,
    )

    assert buyable.tolist() == [True, False, True, False]
    assert sellable.tolist() == [True, True, False, False]


def test_qcurve_targets_enforce_t_plus_one_deferral_and_unfilled_cash() -> None:
    days = 80
    opens = np.vstack(
        [
            np.linspace(10.0, 18.0, days),
            np.linspace(10.0, 18.0, days),
        ]
    )
    sellable = np.ones_like(opens, dtype=bool)
    sellable[:, 1] = False
    dates = np.tile(
        np.arange(np.datetime64("2024-01-02"), np.datetime64("2024-01-02") + days),
        (2, 1),
    )

    result = build_qcurve_targets(
        signal_close_raw=np.asarray([9.5, 9.5]),
        future_open_raw=opens,
        future_open_sellable=sellable,
        future_trade_dates=dates,
        entry_filled=np.asarray([True, False]),
        contract=CONTRACT,
    )

    assert result["enter_net_log_return"].shape == (2, 59)
    assert result["hold_net_log_return"].shape == (2, 60)
    assert result["enter_horizons"].tolist() == list(QCURVE_ENTRY_HORIZONS)
    assert result["enter_resolved_exit_day"][0, 0] == 3
    assert np.allclose(result["enter_net_log_return"][1], 0.0)
    assert np.isfinite(result["hold_net_log_return"]).all()


def test_each_horizon_has_its_own_twenty_day_exit_tail() -> None:
    days = 80
    opens = np.linspace(10.0, 18.0, days, dtype=np.float64)[None, :]
    sellable = np.zeros_like(opens, dtype=bool)
    sellable[:, 30] = True
    dates = np.arange(
        np.datetime64("2024-01-02"),
        np.datetime64("2024-01-02") + days,
    )[None, :]
    result = build_qcurve_targets(
        signal_close_raw=np.asarray([9.5]),
        future_open_raw=opens,
        future_open_sellable=sellable,
        future_trade_dates=dates,
        entry_filled=np.asarray([True]),
        contract=CONTRACT,
    )

    assert result["enter_resolved_exit_day"][0, 0] == 22  # h2 + 20-day retry tail
    assert result["enter_exit_resolved"][0, 0] == 0
    assert result["enter_resolved_exit_day"][0, 8] == 30  # h10 terminal date
    assert result["enter_exit_resolved"][0, 8] == 0
    assert result["enter_resolved_exit_day"][0, 18] == 31  # h20 can reach the sellable day
    assert result["enter_exit_resolved"][0, 18] == 1
    assert result["hold_resolved_exit_day"][0, 0] == 21  # h1 + 20-day retry tail


def test_qcurve_model_orders_quantiles_and_loss_is_finite() -> None:
    model = QCurveModel(input_dim=8, hidden_dim=16, layers=1, dropout=0.0)
    outputs = model(torch.randn(4, 20, 8))

    assert outputs["enter_mean"].shape == (4, 59)
    assert outputs["hold_mean"].shape == (4, 60)
    assert torch.all(outputs["enter_q20"] <= outputs["enter_q50"])
    assert torch.all(outputs["enter_q50"] <= outputs["enter_q80"])
    assert torch.all(outputs["path_upper_spread"] >= 0.0)
    assert torch.all(outputs["path_lower_spread"] >= 0.0)

    loss, parts = qcurve_distribution_loss(
        outputs,
        enter_target=torch.randn(4, 59) * 0.02,
        hold_target=torch.randn(4, 60) * 0.02,
    )
    assert torch.isfinite(loss)
    assert set(parts) == {
        "mean_loss",
        "quantile_loss",
        "positive_loss",
        "soft_exit_loss",
        "distribution_loss",
    }
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_full_day_rank_uses_all_names_and_rewards_correct_top() -> None:
    target = torch.tensor([[3.0], [2.0], [1.0], [0.0], [-1.0]])
    good = full_day_topk_rank_loss(target.clone(), target, top_count=2)
    bad = full_day_topk_rank_loss(-target, target, top_count=2)
    assert good < bad


def _curve(value: float, length: int) -> np.ndarray:
    return np.full(length, value, dtype=np.float32)


def test_dynamic_allocator_caps_three_names_and_preserves_cash() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "is_held": [False, False, False, False],
            "current_weight": [0.0] * 4,
            "sellable_next_open": [True] * 4,
            "enter_mean_curve": [_curve(v, 59) for v in (0.40, 0.20, 0.10, 0.05)],
            "enter_q20_curve": [_curve(v, 59) for v in (0.20, 0.10, 0.05, -0.01)],
            "hold_mean_curve": [_curve(0.0, 60) for _ in range(4)],
            "hold_q20_curve": [_curve(0.0, 60) for _ in range(4)],
        }
    )

    result, diagnostics = allocate_dynamic_qcurve_portfolio(frame)

    assert result["selected"].sum() == 3
    assert result["target_weight"].max() <= 0.60 + 1.0e-12
    assert result.loc[result["symbol"].eq("A"), "target_weight"].iloc[0] == pytest.approx(4.0 / 7.0)
    assert result.loc[result["symbol"].eq("D"), "target_weight"].iloc[0] == 0.0
    assert diagnostics["gross_exposure"] == pytest.approx(1.0)


def test_dynamic_allocator_locks_unsellable_position() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["LOCKED", "A", "B", "C"],
            "is_held": [True, False, False, False],
            "current_weight": [0.55, 0.0, 0.0, 0.0],
            "sellable_next_open": [False, True, True, True],
            "enter_mean_curve": [_curve(v, 59) for v in (0.0, 0.30, 0.20, 0.10)],
            "enter_q20_curve": [_curve(v, 59) for v in (-0.01, 0.20, 0.10, 0.05)],
            "hold_mean_curve": [_curve(-0.01, 60) for _ in range(4)],
            "hold_q20_curve": [_curve(-0.02, 60) for _ in range(4)],
        }
    )

    result, diagnostics = allocate_dynamic_qcurve_portfolio(frame)

    assert result.loc[result["symbol"].eq("LOCKED"), "target_weight"].iloc[0] == pytest.approx(0.55)
    assert result["selected"].sum() <= 3
    assert result["target_weight"].sum() <= 1.0 + 1.0e-12
    assert diagnostics["locked_count"] == 1.0
