from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DeepAlphaConfig:
    start_date: str = "20210101"
    end_date: str = ""
    benchmark: str = "000300.SH"
    universe_scope: str = "all_a"
    lookback_window: int = 120
    prediction_horizons: tuple[int, ...] = (5, 10, 20)
    train_end_date: str = ""
    valid_start_date: str = ""
    valid_days: int = 252
    train_eval_window_days: int = 126
    batch_size: int = 256
    num_workers: int = 0
    pin_memory: bool = True
    hidden_dim: int = 96
    encoder_family: str = "gru"
    patch_len: int = 5
    pretrained_encoder_path: str = ""
    return_head_mode: str = "shared"
    context_dim: int = 16
    state_context: bool = False
    liquidity_context: bool = False
    structure_context: bool = False
    aux_structure_task: bool = False
    aux_structure_loss_weight: float = 0.10
    aux_structure_label_smoothing: float = 0.05
    structure_prototype_task: bool = False
    structure_prototype_loss_weight: float = 0.05
    structure_prototype_temperature: float = 0.20
    transformer_heads: int = 4
    transformer_layers: int = 2
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 8
    min_epochs: int = 4
    early_stop_patience: int = 2
    lr_plateau_patience: int = 1
    lr_plateau_factor: float = 0.5
    min_improvement: float = 1e-4
    use_amp: bool = True
    safe_runtime_profile: bool = True
    grad_clip: float = 1.0
    ranking_loss_weight: float = 0.0
    listwise_loss_weight: float = 0.0
    listwise_temperature: float = 0.35
    max_rank_pairs_per_group: int = 2048
    return_loss_mode: str = "top_bottom_bce"
    return_target_transform: str = "raw"
    return_top_frac: float = 0.2
    return_bottom_frac: float = 0.2
    liquidity_conditioning_mode: str = "none"
    top_liquidity_return_loss_weight: float = 1.0
    other_liquidity_return_loss_weight: float = 1.0
    top_liquidity_rank_loss_weight: float = 1.0
    other_liquidity_rank_loss_weight: float = 1.0
    top_liquidity_sample_weight: float = 1.0
    other_liquidity_sample_weight: float = 1.0
    structure_conditioning_mode: str = "none"
    top_attack_structure_names: tuple[str, ...] = ("trend_breakout", "high_vol_expansion")
    other_protect_structure_names: tuple[str, ...] = ("neutral_mixed", "pullback_rebound", "low_vol_trend")
    top_attack_rank_weight: float = 1.0
    other_protect_rank_weight: float = 1.0
    target_state_names: tuple[str, ...] = ("trend_up_low_vol",)
    target_state_attack_structure_names: tuple[str, ...] = (
        "neutral_mixed",
        "pullback_rebound",
        "trend_breakout",
    )
    target_state_protect_structure_names: tuple[str, ...] = ("low_vol_trend",)
    target_state_rank_weight: float = 1.20
    target_state_protect_rank_weight: float = 1.05
    target_loss_weights: dict[str, float] = field(
        default_factory=lambda: {
            "fwd_excess_5": 0.2,
            "fwd_excess_10": 0.3,
            "fwd_excess_20": 0.5,
            "risk_downside_20": 0.35,
        }
    )
    score_rank_blend: float = 0.35
    score_downside_penalty: float = 0.25
    score_risk_mode: str = "subtract"
    score_risk_gate_threshold: float = 0.35
    liquidity_layer: bool = False
    liquidity_bucket_count: int = 5
    random_seed: int = 7
    market_state_count: int = 4
    holding_count: int = 5
    rebalance_freq: str = "5d"
    max_weight: float = 0.25
    min_adv20: float = 50_000.0
    min_price: float = 2.0
    max_price: float = 300.0
    score_horizon_weights: dict[int, float] = field(default_factory=lambda: {5: 0.2, 10: 0.3, 20: 0.5})
