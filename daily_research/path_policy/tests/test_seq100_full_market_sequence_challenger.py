from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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


def test_lookback_artifacts_preserve_baseline_and_isolate_challengers() -> None:
    root = Path("output")

    assert study._lookback_artifact_root(
        root, "evaluation", lookback=8
    ) == root / "evaluation"
    assert study._lookback_artifact_root(
        root, "evaluation", lookback=16
    ) == root / "evaluation" / "lookback_16"


def test_sequence_training_modes_use_isolated_task_roots() -> None:
    root = Path("output")

    nested = study._sequence_task_root(
        root,
        lookback=16,
        fold=2,
        training_mode="causal_nested",
    )
    outer = study._sequence_task_root(
        root,
        lookback=16,
        fold=2,
        training_mode="outer_early_stop",
    )
    seeded = study._sequence_task_root(
        root,
        lookback=16,
        fold=2,
        training_mode="outer_early_stop",
        seed_offset=101,
    )

    assert nested == root / "sequence_challenger/lookback_16/fold_2"
    assert outer == root / "sequence_challenger/outer_early_stop/lookback_16/fold_2"
    assert seeded == (
        root
        / "sequence_challenger/outer_early_stop/lookback_16/seed_offset_101/fold_2"
    )
    assert nested != outer != seeded

    horizon_10 = study._sequence_task_root(
        root,
        lookback=16,
        fold=2,
        training_mode="outer_early_stop",
        horizon=10,
    )
    assert horizon_10 == (
        root
        / "sequence_challenger/horizon_10/outer_early_stop/lookback_16/fold_2"
    )
    assert horizon_10 != outer


def test_ensemble_protocols_use_isolated_artifact_roots() -> None:
    root = Path("output")

    baseline = study._ensemble_protocol_root(
        root,
        "evaluation",
        lookback=16,
        training_mode="outer_early_stop",
        tree_training_mode="outer_early_stop",
        seed_offsets=(0,),
    )
    seeded = study._ensemble_protocol_root(
        root,
        "evaluation",
        lookback=16,
        training_mode="outer_early_stop",
        tree_training_mode="outer_early_stop",
        seed_offsets=(0, 101, 202),
    )

    assert baseline == (
        root
        / "evaluation/lookback_16/sequence_outer_early_stop__tree_outer_early_stop"
    )
    assert seeded == (
        root
        / "evaluation/lookback_16/sequence_outer_early_stop__tree_outer_early_stop__seeds_0_101_202"
    )
    assert baseline != seeded


def test_payoff_ensemble_artifacts_are_horizon_isolated() -> None:
    root = Path("output")

    d5 = study._payoff_ensemble_root(
        root,
        "payoff_ensemble_evaluation",
        lookback=16,
        horizon=5,
        training_mode="outer_early_stop",
        tree_training_mode="outer_early_stop",
        seed_offsets=(0,),
    )
    d10 = study._payoff_ensemble_root(
        root,
        "payoff_ensemble_evaluation",
        lookback=16,
        horizon=10,
        training_mode="outer_early_stop",
        tree_training_mode="outer_early_stop",
        seed_offsets=(0,),
    )
    conservative = study._payoff_ensemble_root(
        root,
        "payoff_ensemble_evaluation",
        lookback=16,
        horizon=10,
        training_mode="outer_early_stop",
        tree_training_mode="outer_early_stop",
        seed_offsets=(0,),
        fusion_method="minimum",
    )
    multi_tree = study._payoff_ensemble_root(
        root,
        "payoff_ensemble_evaluation",
        lookback=16,
        horizon=10,
        training_mode="outer_early_stop",
        tree_training_mode="outer_early_stop",
        seed_offsets=(0,),
        tree_profiles=("strong_127", "qlib_capacity_210"),
    )
    core_tree = study._payoff_ensemble_root(
        root,
        "payoff_ensemble_evaluation",
        lookback=16,
        horizon=10,
        training_mode="outer_early_stop",
        tree_training_mode="outer_early_stop",
        seed_offsets=(0,),
        tree_profiles=("strong_127", "qlib_capacity_210"),
        tree_feature_variant="price_path_core_183",
    )

    assert d5 == (
        root
        / "payoff_ensemble_evaluation/horizon_5/lookback_16"
        / "sequence_outer_early_stop__tree_outer_early_stop"
    )
    assert d10 == (
        root
        / "payoff_ensemble_evaluation/horizon_10/lookback_16"
        / "sequence_outer_early_stop__tree_outer_early_stop"
    )
    assert conservative == (
        root
        / "payoff_ensemble_evaluation/horizon_10/lookback_16"
        / "sequence_outer_early_stop__tree_outer_early_stop__fusion_minimum"
    )
    assert multi_tree == (
        root
        / "payoff_ensemble_evaluation/horizon_10/lookback_16"
        / "sequence_outer_early_stop__tree_outer_early_stop"
        "__tree_profiles_strong_127_qlib_capacity_210"
    )
    assert core_tree == (
        root
        / "payoff_ensemble_evaluation/horizon_10/lookback_16"
        / "sequence_outer_early_stop__tree_outer_early_stop"
        "__tree_profiles_strong_127_qlib_capacity_210"
        "__tree_features_price_path_core_183"
    )
    assert d5 != d10
    assert conservative != d10
    assert multi_tree != d10
    assert core_tree != multi_tree


def test_audited_manifest_cache_requires_explicit_zero_forbidden_reads() -> None:
    manifest = {"status": "completed", "fingerprint": "expected"}

    assert not study._audited_manifest_is_current(manifest, "expected")
    manifest["forbidden_2026_read_count"] = 1
    assert not study._audited_manifest_is_current(manifest, "expected")
    manifest["forbidden_2026_read_count"] = 0
    assert study._audited_manifest_is_current(manifest, "expected")
    assert not study._audited_manifest_is_current(manifest, "different")


def test_exact_payoff_account_specs_match_tree_stress_contract() -> None:
    specs = study._exact_payoff_account_specs(
        variant="sequence_payoff", horizon=10
    )

    assert len(specs) == 15
    assert {item["variant"] for item in specs} == {"sequence_payoff"}
    assert {item["cohort_equity_fraction"] for item in specs} == {0.1}
    assert {item["planned_fill_day"] for item in specs} == {10}
    assert {
        int(item["top_k"])
        for item in specs
        if item.get("maximum_credited_gross_return") is None
    } == {1, 3, 5, 10}
    assert sum(
        item.get("maximum_credited_gross_return") is None for item in specs
    ) == 9
    assert {
        item.get("maximum_credited_gross_return")
        for item in specs
        if item.get("maximum_credited_gross_return") is not None
    } == {0.05, 0.10}
    assert sum(
        not item.get("allow_overlapping_same_symbol", True) for item in specs
    ) == 3


def test_payoff_ensemble_prediction_uses_date_local_component_ranks() -> None:
    row_index = pd.DataFrame(
        {
            "candidate_id": [1, 2, 3, 4],
            "date_idx": [10, 10, 11, 11],
        }
    )
    exact_valid = np.ones((4, 2), dtype=np.uint8)
    exact_valid[3, :] = 0
    sources = SimpleNamespace(
        context=SimpleNamespace(row_index=row_index),
        exact_valid=exact_valid,
        base_column=0,
        stress_column=1,
    )

    prediction, components = study._payoff_ensemble_prediction(
        sources=sources,
        validation_positions=np.arange(4, dtype=np.int64),
        sequence_prediction=np.asarray([1.0, 2.0, 100.0, 200.0]),
        tree_prediction=np.asarray([2.0, 1.0, 100.0, 200.0]),
    )

    np.testing.assert_allclose(prediction.stock_score, [0.75, 0.75, 0.5, 1.0])
    np.testing.assert_array_equal(prediction.rows, np.arange(4, dtype=np.int64))
    np.testing.assert_allclose(
        components["sequence_rank"].to_numpy(), [0.5, 1.0, 0.5, 1.0]
    )
    assert prediction.market_return == {10: 0.0, 11: 0.0}
    assert prediction.market_probability == {10: 0.5, 11: 0.5}

    conservative, _ = study._payoff_ensemble_prediction(
        sources=sources,
        validation_positions=np.arange(4, dtype=np.int64),
        sequence_prediction=np.asarray([1.0, 2.0, 100.0, 200.0]),
        tree_prediction=np.asarray([2.0, 1.0, 100.0, 200.0]),
        fusion_method="minimum",
    )
    np.testing.assert_allclose(conservative.stock_score, [0.5, 0.5, 0.5, 1.0])

    multi_tree, components = study._payoff_ensemble_prediction(
        sources=sources,
        validation_positions=np.arange(4, dtype=np.int64),
        sequence_prediction=np.asarray([1.0, 2.0, 100.0, 200.0]),
        tree_prediction=np.asarray([2.0, 1.0, 100.0, 200.0]),
        additional_tree_predictions={
            "qlib_capacity_210": np.asarray([1.0, 2.0, 200.0, 100.0])
        },
    )
    np.testing.assert_allclose(
        multi_tree.stock_score,
        [2.0 / 3.0, 5.0 / 6.0, 2.0 / 3.0, 5.0 / 6.0],
    )
    assert "tree_rank__qlib_capacity_210" in components


def test_sequence_parser_defaults_to_direct_outer_early_stop() -> None:
    args = study._build_parser().parse_args([])
    multi_tree = study._build_parser().parse_args(
        ["--payoff-tree-profiles", "strong_127,qlib_capacity_210"]
    )
    core_tree = study._build_parser().parse_args(
        ["--payoff-tree-feature-variant", "price_path_core_183"]
    )

    assert args.training_mode == "outer_early_stop"
    assert args.seed_offset == 0
    assert args.maximum_epochs == study.DEFAULT_MAX_EPOCHS
    assert args.patience == study.DEFAULT_PATIENCE
    assert args.horizon == 5
    assert args.payoff_fusion_method == "mean"
    assert args.payoff_tree_feature_variant == "all_causal_557"
    assert core_tree.payoff_tree_feature_variant == "price_path_core_183"
    assert args.payoff_tree_profiles == ""
    assert multi_tree.payoff_tree_profiles == "strong_127,qlib_capacity_210"
    assert (
        study._build_parser().parse_args(["--horizon", "10"]).horizon == 10
    )
    assert (
        study._build_parser()
        .parse_args(["--payoff-fusion-method", "minimum"])
        .payoff_fusion_method
        == "minimum"
    )


def test_final_bundle_cli_is_explicit() -> None:
    args = study._build_parser().parse_args(["--freeze-final-bundle"])
    core = study._build_parser().parse_args(["--core-availability-stress"])
    exact = study._build_parser().parse_args(["--exact-core-availability-stress"])
    repaired = study._build_parser().parse_args(
        ["--repaired-rank-core-availability-stress"]
    )
    corrected_full = study._build_parser().parse_args(
        ["--corrected-rank-full-core-availability-stress"]
    )
    rebuildable = study._build_parser().parse_args(
        ["--rebuildable-current-core-availability-stress"]
    )
    robustness = study._build_parser().parse_args(["--rebuildable-core-robustness"])
    payoff_evaluation = study._build_parser().parse_args(
        ["--evaluate-payoff-ensemble", "--horizon", "10"]
    )
    payoff_replay = study._build_parser().parse_args(
        ["--replay-payoff-ensemble", "--horizon", "10"]
    )

    assert args.freeze_final_bundle is True
    assert args.availability_stress is False
    assert core.core_availability_stress is True
    assert core.freeze_final_bundle is False
    assert exact.exact_core_availability_stress is True
    assert exact.core_availability_stress is False
    assert repaired.repaired_rank_core_availability_stress is True
    assert repaired.exact_core_availability_stress is False
    assert corrected_full.corrected_rank_full_core_availability_stress is True
    assert rebuildable.rebuildable_current_core_availability_stress is True
    assert robustness.rebuildable_core_robustness is True
    assert payoff_evaluation.evaluate_payoff_ensemble is True
    assert payoff_evaluation.horizon == 10
    assert payoff_replay.replay_payoff_ensemble is True
    assert payoff_replay.horizon == 10


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
