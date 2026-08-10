from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from daily_research.path_policy import seq100_full_market_sequence_challenger as study


def test_date_blocks_preserve_contiguous_row_contract() -> None:
    frame = pd.DataFrame(
        {
            "date_idx": [10, 10, 11, 11, 11],
            "trade_date": ["2020-01-02"] * 2 + ["2020-01-03"] * 3,
        }
    )

    blocks = study._date_blocks(frame)

    assert (blocks[10].start, blocks[10].stop) == (0, 2)
    assert (blocks[11].start, blocks[11].stop) == (2, 5)
    assert blocks[11].trade_date == "2020-01-03"


def test_cross_sectional_standardize_is_date_local_and_keeps_missing_mask() -> None:
    values = torch.tensor(
        [
            [1.0, float("nan")],
            [3.0, 4.0],
            [100.0, 8.0],
            [104.0, 12.0],
        ]
    )

    normalized, missing = study.cross_sectional_standardize(values, (2, 2))

    assert torch.allclose(normalized[:2, 0], torch.tensor([-1.0, 1.0]))
    assert torch.allclose(normalized[2:, 0], torch.tensor([-1.0, 1.0]))
    assert normalized[0, 1] == 0.0
    assert missing[0, 1] == 1.0
    assert missing[1:, 1].sum() == 0.0


def test_source_availability_mask_is_copy_on_write() -> None:
    source = np.arange(12, dtype=np.float32).reshape(3, 4)

    masked = study.mask_static_columns(source, (1, 3))

    assert np.isfinite(source).all()
    assert np.isnan(masked[:, [1, 3]]).all()
    assert np.array_equal(masked[:, [0, 2]], source[:, [0, 2]])
    with pytest.raises(study.SequenceChallengerError, match="out_of_range"):
        study.mask_static_columns(source, (4,))


def test_static_override_is_row_aligned_then_masked() -> None:
    source = np.arange(20, dtype=np.float32).reshape(5, 4)
    repaired = np.arange(10, dtype=np.float32).reshape(5, 2) + 100.0
    override = study.StaticFeatureOverride(
        values=repaired,
        columns=np.asarray([1, 3], dtype=np.int32),
        feature_names=("a", "b"),
        manifest={"fingerprint": "test"},
    )

    result = study.apply_static_feature_contract(
        source[[4, 1]],
        row_positions=np.asarray([4, 1]),
        masked_columns=(3,),
        override=override,
    )

    np.testing.assert_array_equal(result[:, 0], source[[4, 1], 0])
    np.testing.assert_array_equal(result[:, 1], repaired[[4, 1], 0])
    assert np.isnan(result[:, 3]).all()
    assert np.isfinite(source).all()


def test_vectorized_rank_matches_frozen_average_percentile_definition() -> None:
    values = np.asarray(
        [[1.0, np.nan, 7.0], [2.0, 5.0, 7.0], [2.0, np.nan, 7.0]],
        dtype=np.float32,
    )

    ranked = study._rank_percentile_columns(values)

    np.testing.assert_allclose(ranked[:, 0], [0.0, 0.75, 0.75])
    np.testing.assert_allclose(ranked[:, 1], [np.nan, 1.0, np.nan], equal_nan=True)
    np.testing.assert_allclose(ranked[:, 2], [0.5, 0.5, 0.5])


def test_final_refit_uses_deterministic_middle_oof_setting() -> None:
    assert study.frozen_median([37, 5, 4, 3, 29]) == 5
    assert study.frozen_median([2, 1, 3, 4, 2]) == 2
    with pytest.raises(study.SequenceChallengerError, match="support_empty"):
        study.frozen_median([])


def test_final_bundle_cli_is_explicit() -> None:
    args = study._build_parser().parse_args(["--freeze-final-bundle"])
    core = study._build_parser().parse_args(["--core-availability-stress"])
    exact = study._build_parser().parse_args(["--exact-core-availability-stress"])
    repaired = study._build_parser().parse_args(
        ["--repaired-rank-core-availability-stress"]
    )

    assert args.freeze_final_bundle is True
    assert args.availability_stress is False
    assert core.core_availability_stress is True
    assert core.freeze_final_bundle is False
    assert exact.exact_core_availability_stress is True
    assert exact.core_availability_stress is False
    assert repaired.repaired_rank_core_availability_stress is True
    assert repaired.exact_core_availability_stress is False


def test_frozen_candidate_keeps_adaptive_boundary_and_no_2026_outcome() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "studies/seq100_full_market_dual_gate_d3_candidate_v1.json"
    )
    candidate = json.loads(path.read_text(encoding="utf-8"))

    assert candidate["frozen_policy"]["tree_stock_head"].startswith(
        "exact_net_return_d5_rank"
    )
    assert candidate["frozen_policy"]["exit"].startswith("D3 close")
    assert candidate["historical_evidence"]["candidate_gate_passed"] is True
    assert candidate["historical_evidence"]["small_gain_robustness_passed"] is False
    assert candidate["epistemic_contract"]["no_2026_outcome_was_read"] is True
    assert candidate["epistemic_contract"]["stable_profit_claim_allowed"] is False


def test_payoff_model_and_multitask_loss_have_finite_gradients() -> None:
    generator = torch.Generator().manual_seed(7)
    model = study.PayoffSequenceModel(
        static_dim=5,
        sequence_dim=3,
        market_dim=2,
        hidden_dim=8,
        dropout=0.0,
    )
    static = torch.randn(7, 5, generator=generator)
    static[0, 0] = float("nan")
    normalized, missing = study.cross_sectional_standardize(static, (3, 4))
    sequence = torch.randn(7, 4, 3, generator=generator)
    market = torch.randn(2, 2, generator=generator)
    target = torch.tensor([0.01, -0.01, 0.02, -0.02, 0.00, 0.03, 0.01])

    output = model(normalized, missing, sequence, market, (3, 4))
    loss, components = study.payoff_loss(*output, target, (3, 4))
    loss.backward()

    assert [tuple(value.shape) for value in output] == [(7,), (2,), (2,)]
    assert torch.isfinite(loss)
    assert all(np.isfinite(value) for value in components.values())
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
