from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from daily_research.path_policy import seq100_multi_horizon_distribution as study


def test_frozen_study_keeps_d5_d10_primary_and_d2_d3_deferred() -> None:
    contract = study.load_study()

    assert contract["targets"]["primary_horizons"] == [5, 10]
    assert contract["targets"]["deferred_auxiliary_horizons"] == [2, 3]
    assert contract["features"]["variant"] == "price_path_core_183"
    assert contract["validation"]["D2_D3_training_allowed_in_v1"] is False
    assert contract["validation"]["horizon_choice_per_stock_allowed"] is False


def test_variant_contract_isolates_point_control_and_distribution_challenger() -> None:
    contract = study.load_study()

    assert study._variant_contract(contract, "single_d10_point") == ((10,), False)
    assert study._variant_contract(contract, "multi_d5_d10_distribution") == (
        (5, 10),
        True,
    )


def test_monotone_quantile_parameterization_cannot_cross() -> None:
    lower = torch.tensor([0.2, -1.0, 0.0])
    deltas = torch.tensor([[0.0, 0.0], [-10.0, 3.0], [5.0, -5.0]])

    result = study.monotone_quantiles(lower, deltas)

    assert result.shape == (3, 3)
    assert torch.all(result[:, 0] <= result[:, 1])
    assert torch.all(result[:, 1] <= result[:, 2])


def test_pinball_loss_is_zero_for_exact_prediction() -> None:
    target = torch.tensor([1.0, -2.0])
    prediction = target[:, None].repeat(1, 3)
    quantiles = torch.tensor([0.1, 0.5, 0.9])

    assert study.pinball_loss(prediction, target, quantiles) == 0.0


def test_model_outputs_full_slate_heads_with_ordered_quantiles() -> None:
    model = study.MultiHorizonSequenceModel(horizons=(5, 10), distribution_heads=True)
    static = torch.randn(5, 183)
    missing = torch.zeros_like(static)
    sequence = torch.randn(5, 16, 32)
    market = torch.randn(2, 54)

    output = model(static, missing, sequence, market, (2, 3))

    assert set(output.point) == {5, 10}
    assert set(output.quantiles) == {5, 10}
    assert output.entry_logit.shape == (5,)
    assert output.market_return.shape == (2,)
    for horizon in (5, 10):
        assert output.point[horizon].shape == (5,)
        assert output.exit_logit[horizon].shape == (5,)
        assert output.quantiles[horizon].shape == (5, 3)
        assert torch.all(
            output.quantiles[horizon][:, 0] <= output.quantiles[horizon][:, 1]
        )
        assert torch.all(
            output.quantiles[horizon][:, 1] <= output.quantiles[horizon][:, 2]
        )


def test_distribution_loss_respects_masks_and_backpropagates() -> None:
    model = study.MultiHorizonSequenceModel(horizons=(5, 10), distribution_heads=True)
    static = torch.randn(5, 183)
    missing = torch.zeros_like(static)
    sequence = torch.randn(5, 16, 32)
    market = torch.randn(2, 54)
    output = model(static, missing, sequence, market, (2, 3))
    batch = study.MultiHorizonBatch(
        static=static,
        sequence=sequence,
        market=market,
        returns=torch.tensor(
            [[0.1, 0.2], [0.3, 0.1], [-0.2, -0.1], [0.0, 0.2], [0.1, 0.4]]
        ),
        return_valid=torch.tensor(
            [[True, True], [True, True], [True, True], [False, True], [True, True]]
        ),
        entry_fill=torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0]),
        entry_valid=torch.tensor([True, True, True, True, True]),
        exit_block=torch.tensor(
            [[0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]
        ),
        exit_valid=torch.tensor(
            [[True, True], [True, True], [False, False], [True, True], [True, True]]
        ),
        group_sizes=(2, 3),
        row_positions=np.arange(5),
        date_indices=(10, 11),
    )
    weights = study.load_study()["model"]["loss_weights"]

    loss, components = study.payoff_distribution_loss(
        output,
        batch,
        horizons=(5, 10),
        distribution_heads=True,
        weights=weights,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert components["quantile_pinball_loss"] > 0.0
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_date_equal_binary_metrics_do_not_weight_large_dates_more() -> None:
    dates = np.asarray([1, 1, 1, 2])
    actual = np.asarray([1.0, 1.0, 1.0, 0.0])
    probability = np.asarray([0.0, 0.0, 0.0, 0.0])

    metrics = study._date_equal_binary_metrics(
        dates, actual, probability, np.ones(4, dtype=bool)
    )

    assert metrics["date_count"] == 2
    assert metrics["date_equal_brier"] == 0.5


def test_task_roots_keep_variants_and_folds_isolated() -> None:
    root = Path("output")

    assert (
        study._task_root(root, variant="single_d10_point", fold=1)
        == root / "models/single_d10_point/fold_1"
    )
    assert (
        study._task_root(root, variant="multi_d5_d10_distribution", fold=1)
        == root / "models/multi_d5_d10_distribution/fold_1"
    )
