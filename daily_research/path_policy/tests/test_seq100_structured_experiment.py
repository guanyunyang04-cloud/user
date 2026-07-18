from __future__ import annotations

import numpy as np
import torch

from daily_research.path_policy import qdp_v2_sequence_path_training as training
from daily_research.path_policy import seq100_structured_experiment as experiment


def test_past20_positive_median_is_strictly_asof_and_ignores_zero_days() -> None:
    values = np.asarray([0.0, *range(1, 21), 0.0, 1000.0], dtype=np.float32)
    result = experiment._past20_positive_median(values)

    assert np.isnan(result[:20]).all()
    assert result[20] == np.median(np.arange(1, 21, dtype=np.float32))
    assert result[21] == result[20]
    assert result[22] == np.median(np.asarray([*range(2, 21), 1000], dtype=np.float32))


def test_structured_contract_keeps_activity_out_of_checkpoint_selector() -> None:
    payload = experiment.contract()["contract"]
    assert payload["legal_exit_contract"] == {
        "earliest_legal_exit_day": 2,
        "exit_argmax_domain": [2, 60],
        "tie_break": "earliest_legal_day",
    }
    assert payload["early_stopping"]["metric"] == "development_price_total_loss"
    assert "turnover" in payload["early_stopping"]["excludes"]
    structured = payload["profiles"]["structured_joint_turnover"]
    assert structured["model_type"] == "gru_structured_joint_turnover"
    assert structured["path_loss_weight"] == 0.35
    assert structured["geometry_loss_weight"] == 0.10
    assert structured["utility_curve_loss_weight"] == 0.05


def test_development_price_selector_is_invariant_to_auxiliary_losses() -> None:
    base = {
        "path_loss": 1.0,
        "summary_loss": 2.0,
        "value_loss": 3.0,
        "rank_loss": 4.0,
        "geometry_loss": 5.0,
        "utility_curve_loss": 6.0,
        "turnover_level_loss": 7.0,
        "turnover_delta_loss": 8.0,
    }
    changed = dict(base)
    changed.update(
        geometry_loss=5000.0,
        utility_curve_loss=6000.0,
        turnover_level_loss=7000.0,
        turnover_delta_loss=8000.0,
    )
    assert training._development_price_total_loss(base) == training._development_price_total_loss(
        changed
    )


def test_future_gru_carries_state_across_horizons_without_future_teacher_forcing() -> None:
    model = training.SequencePathModel(
        input_dim=32,
        hidden_dim=16,
        layers=2,
        forward_days=8,
        summary_dim=12,
        dropout=0.0,
        model_type="gru_structured_joint_turnover",
    )
    with torch.no_grad():
        model.horizon_embedding.weight.zero_()
    output = model(torch.randn(2, 100, 32))["future_path"]
    assert model.future_decoder.num_layers == 2
    assert not torch.allclose(output[:, 0], output[:, -1])

