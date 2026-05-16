from __future__ import annotations

import torch

from daily_research.path_policy.rl_episode import (
    EPISODE_REWARD_COLUMNS,
    build_path20_market_episode,
    episode_array_manifest,
    episode_policy_rollout_loss,
    episode_to_arrays,
    fit_episode_normalization,
    load_episode_arrays,
    market_episode_from_long_frame,
    projection_parity_diagnostics,
    project_target_weights_torch,
    save_episode_arrays,
    stable_episode_manifest_hash,
)
from daily_research.path_policy.rl_models import (
    DecisionTransformerTargetWeightPolicy,
    SequencePolicyConfig,
    SequenceTargetWeightPolicy,
)
from daily_research.path_policy.tests.fixtures import make_prepared_policy_inputs


def test_episode_dataset_keeps_reward_columns_out_of_model_inputs() -> None:
    prepared = make_prepared_policy_inputs(days=35, stocks=("AAA", "BBB", "CCC"))
    episode = build_path20_market_episode(
        prepared,
        start_date="20240102",
        end_date="20240209",
        lake_dataset_id="policy_input_bundle__fixture",
        sequence_length=4,
        min_trading_days=2,
    )

    model_inputs = set(episode.manifest["model_input_feature_columns"])
    assert episode.manifest["artifact_type"] == "market_episode"
    assert episode.manifest["leakage_guard"]["status"] == "passed"
    assert not model_inputs.intersection(EPISODE_REWARD_COLUMNS)
    assert not any(column.startswith(("future_", "oracle_path20_", "path_q", "path_mu_")) for column in model_inputs)


def test_episode_normalization_uses_train_episode_only() -> None:
    train = build_path20_market_episode(
        make_prepared_policy_inputs(days=30, stocks=("AAA", "BBB")),
        start_date="20240102",
        end_date="20240209",
        lake_dataset_id="policy_input_bundle__fixture_train",
        sequence_length=4,
        min_trading_days=2,
    )
    validation = build_path20_market_episode(
        make_prepared_policy_inputs(days=30, stocks=("AAA", "BBB")),
        start_date="20240102",
        end_date="20240209",
        lake_dataset_id="policy_input_bundle__fixture_validation",
        sequence_length=4,
        min_trading_days=2,
    )
    train_only = fit_episode_normalization([train])
    with_validation = fit_episode_normalization([train, validation])

    assert train_only["scope"] == "train_years"
    assert train_only["mean"] == fit_episode_normalization([train])["mean"]
    assert with_validation["feature_columns"] == train_only["feature_columns"]


def test_torch_projection_enforces_contract_and_writes_diagnostics() -> None:
    raw = torch.tensor([0.50, 0.30, -0.10, 0.20])
    current = torch.tensor([0.05, 0.05, 0.10, 0.0])
    mask = torch.tensor([True, True, False, True])

    result = project_target_weights_torch(
        raw,
        current,
        mask,
        max_position_weight=0.20,
        max_gross_exposure=0.50,
        max_positions=3,
        turnover_budget=0.40,
    )
    target = result["projected_target_weight"]
    diagnostics = result["diagnostics"]

    assert float(target.min()) >= 0.0
    assert float(target.max()) <= 0.20 + 1.0e-6
    assert float(target.sum()) <= 0.50 + 1.0e-6
    assert abs(float(target[2]) - 0.10) < 1.0e-6
    assert float((target - current).abs().sum()) <= 0.40 + 1.0e-6
    assert "projection_l1_distance" in diagnostics
    assert float(diagnostics["masked_count"]) >= 1.0


def test_episode_npz_arrays_match_csv_episode_frame(tmp_path) -> None:
    prepared = make_prepared_policy_inputs(days=12, stocks=("AAA", "BBB"))
    episode = build_path20_market_episode(
        prepared,
        start_date="20240102",
        end_date="20240131",
        lake_dataset_id="policy_input_bundle__fixture",
        sequence_length=3,
        min_trading_days=2,
    )
    arrays = episode_to_arrays(episode)
    arrays_path = save_episode_arrays(tmp_path / "episode_arrays.npz", arrays)
    loaded = load_episode_arrays(arrays_path)
    restored = market_episode_from_long_frame(episode.to_long_frame(), episode.manifest)
    restored_arrays = episode_to_arrays(restored)
    manifest = {**episode.manifest, "manifest_hash": stable_episode_manifest_hash(episode.manifest)}
    array_manifest = episode_array_manifest(arrays, manifest=manifest, arrays_npz_path=arrays_path)

    assert array_manifest["artifact_type"] == "market_episode_arrays"
    assert loaded["state"].shape == arrays["state"].shape
    assert (loaded["dates"].astype(str) == arrays["dates"].astype(str)).all()
    assert (loaded["stocks"].astype(str) == arrays["stocks"].astype(str)).all()
    assert torch.tensor(loaded["state"]).shape == torch.tensor(restored_arrays["state"]).shape
    assert (loaded["mask"] == restored_arrays["mask"]).all()


def test_projection_parity_fixture_gap_is_below_threshold() -> None:
    raw = torch.tensor([0.50, 0.30, -0.10, 0.20])
    current = torch.tensor([0.05, 0.05, 0.10, 0.0])
    mask = torch.tensor([True, True, False, True])

    summary = projection_parity_diagnostics(
        raw,
        current,
        mask,
        stocks=["AAA", "BBB", "CCC", "DDD"],
        max_position_weight=0.20,
        max_gross_exposure=0.50,
        max_positions=3,
        turnover_budget=0.40,
        max_l1_gap=0.02,
    )

    assert summary["status"] == "passed"
    assert summary["projection_mismatch_warning"] is False
    assert summary["target_l1_gap"] <= 0.02


def test_episode_rollout_uses_projected_weights_and_rolls_current_weight() -> None:
    prepared = make_prepared_policy_inputs(days=35, stocks=("AAA", "BBB", "CCC"))
    episode = build_path20_market_episode(
        prepared,
        start_date="20240102",
        end_date="20240209",
        lake_dataset_id="policy_input_bundle__fixture",
        sequence_length=4,
        min_trading_days=2,
    )
    normalization = fit_episode_normalization([episode])
    config = SequencePolicyConfig(
        stock_feature_dim=len(episode.feature_columns) + 1,
        portfolio_feature_dim=8,
        hidden_dim=16,
        dropout=0.0,
        max_position_weight=0.20,
    )
    model = SequenceTargetWeightPolicy(config)

    result = episode_policy_rollout_loss(
        model,
        episode,
        sequence_length=4,
        normalization=normalization,
        model_family="sequence_gru",
        transaction_cost_bps=3.0,
        slippage_bps=7.0,
        sell_tax_bps=10.0,
        max_position_weight=0.20,
        max_gross_exposure=0.50,
        max_positions=2,
        turnover_budget=0.40,
    )

    assert result["status"] == "completed"
    assert torch.isfinite(result["loss"])
    assert result["used_projected_weights_for_loss"] is True
    assert result["rolled_current_weight"] is True
    assert result["diagnostics"]["avg_projected_turnover"] > 0.0
    assert result["rollout_grad_mode"] == "detached"
    assert result["context_coverage"]["previous_weight_nonzero_rate"] > 0.0


def test_episode_rollout_truncated_mode_reports_chunk_contract() -> None:
    prepared = make_prepared_policy_inputs(days=35, stocks=("AAA", "BBB", "CCC"))
    episode = build_path20_market_episode(
        prepared,
        start_date="20240102",
        end_date="20240209",
        lake_dataset_id="policy_input_bundle__fixture",
        sequence_length=4,
        min_trading_days=2,
    )
    normalization = fit_episode_normalization([episode])
    config = SequencePolicyConfig(
        stock_feature_dim=len(episode.feature_columns) + 1,
        portfolio_feature_dim=8,
        hidden_dim=16,
        dropout=0.0,
        max_position_weight=0.20,
    )
    model = SequenceTargetWeightPolicy(config)

    result = episode_policy_rollout_loss(
        model,
        episode,
        sequence_length=4,
        normalization=normalization,
        model_family="sequence_gru",
        transaction_cost_bps=3.0,
        slippage_bps=7.0,
        sell_tax_bps=10.0,
        max_position_weight=0.20,
        max_gross_exposure=0.50,
        max_positions=2,
        turnover_budget=0.40,
        rollout_grad_mode="truncated",
        rollout_chunk_days=5,
    )

    assert result["status"] == "completed"
    assert result["rollout_grad_mode"] == "truncated"
    assert result["rollout_chunk_days"] == 5
    assert torch.isfinite(result["loss"])


def test_decision_transformer_receives_nonzero_previous_rollout_context() -> None:
    prepared = make_prepared_policy_inputs(days=35, stocks=("AAA", "BBB", "CCC"))
    episode = build_path20_market_episode(
        prepared,
        start_date="20240102",
        end_date="20240209",
        lake_dataset_id="policy_input_bundle__fixture",
        sequence_length=4,
        min_trading_days=2,
    )
    normalization = fit_episode_normalization([episode])
    config = SequencePolicyConfig(
        stock_feature_dim=len(episode.feature_columns) + 1,
        portfolio_feature_dim=8,
        hidden_dim=16,
        dropout=0.0,
        max_position_weight=0.20,
    )
    model = DecisionTransformerTargetWeightPolicy(config, num_layers=1, num_heads=2)

    result = episode_policy_rollout_loss(
        model,
        episode,
        sequence_length=4,
        normalization=normalization,
        model_family="decision_transformer",
        transaction_cost_bps=3.0,
        slippage_bps=7.0,
        sell_tax_bps=10.0,
        max_position_weight=0.20,
        max_gross_exposure=0.50,
        max_positions=2,
        turnover_budget=0.40,
    )

    assert result["status"] == "completed"
    assert result["decision_transformer_previous_context_nonzero"] is True
    assert result["context_coverage"]["previous_weight_nonzero_rate"] > 0.0
    assert "previous_reward_nonzero_rate" in result["context_coverage"]
