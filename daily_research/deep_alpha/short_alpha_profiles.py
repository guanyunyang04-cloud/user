from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShortAlphaProfile:
    name: str
    description: str
    state_context: bool = False
    liquidity_context: bool = False
    structure_context: bool = False
    ranking_loss_weight: float = 0.0
    listwise_loss_weight: float = 0.0
    listwise_temperature: float = 0.35
    prediction_horizons: str = "5,10,20"
    task_loss_weights: str = "5:0.2,10:0.3,20:0.5,downside:0.35"
    score_horizon_weights: str = "5:0.2,10:0.3,20:0.5"
    short_alpha_features: bool = False
    breakout_event_horizon: int = 5
    breakout_event_threshold: float = 0.08
    breakout_event_pullback_limit: float = 0.03
    breakout_event_loss_weight: float = 0.0
    clean_breakout_event_loss_weight: float = 0.0
    score_head_method: str = ""
    adaptive_task_weights: bool = False
    research_objective_mode: str = ""
    checkpoint_selection_objective: str = ""
    score_downside_penalty: float | None = None
    structure_conditioning_mode: str = ""
    target_state_names: str = ""
    target_state_attack_structures: str = ""
    target_state_protect_structures: str = ""
    target_state_rank_weight: float | None = None
    target_state_protect_rank_weight: float | None = None


PROFILE_REGISTRY: dict[str, ShortAlphaProfile] = {
    "baseline_current": ShortAlphaProfile(
        name="baseline_current",
        description="Current dynamic_graph_v1 liquid500 recent-formal baseline.",
    ),
    "state_context_v1": ShortAlphaProfile(
        name="state_context_v1",
        description="Turn on learned market-state context without changing targets or inputs.",
        state_context=True,
    ),
    "state_liquidity_listwise_v1": ShortAlphaProfile(
        name="state_liquidity_listwise_v1",
        description="Joint structure-capacity probe: state + liquidity context with light ranking/listwise losses.",
        state_context=True,
        liquidity_context=True,
        ranking_loss_weight=0.02,
        listwise_loss_weight=0.05,
        listwise_temperature=0.35,
    ),
    "short_target_v1": ShortAlphaProfile(
        name="short_target_v1",
        description="Short-line target probe: 1/2/3/5-day excess targets plus breakout-event auxiliary labels.",
        prediction_horizons="1,2,3,5",
        task_loss_weights="1:0.30,2:0.25,3:0.25,5:0.20,downside:0.30",
        score_horizon_weights="1:0.35,2:0.25,3:0.20,5:0.20",
        breakout_event_horizon=5,
        breakout_event_threshold=0.08,
        breakout_event_pullback_limit=0.03,
        breakout_event_loss_weight=0.20,
        clean_breakout_event_loss_weight=0.20,
    ),
    "short_input_v1": ShortAlphaProfile(
        name="short_input_v1",
        description="Short-line input probe: add breakout/compression/energy candidate features while keeping current targets.",
        short_alpha_features=True,
    ),
    "short_combo_v1": ShortAlphaProfile(
        name="short_combo_v1",
        description="Combined short-line probe: short targets plus new breakout/compression/energy inputs.",
        prediction_horizons="1,2,3,5",
        task_loss_weights="1:0.30,2:0.25,3:0.25,5:0.20,downside:0.30",
        score_horizon_weights="1:0.35,2:0.25,3:0.20,5:0.20",
        short_alpha_features=True,
        breakout_event_horizon=5,
        breakout_event_threshold=0.08,
        breakout_event_pullback_limit=0.03,
        breakout_event_loss_weight=0.20,
        clean_breakout_event_loss_weight=0.20,
    ),
    "short_expert_v1": ShortAlphaProfile(
        name="short_expert_v1",
        description=(
            "Short-line expert candidate: shorter horizons + breakout auxiliaries + "
            "expanded short-alpha inputs + state/liquidity context + learned score head."
        ),
        state_context=True,
        liquidity_context=True,
        ranking_loss_weight=0.03,
        listwise_loss_weight=0.06,
        listwise_temperature=0.30,
        prediction_horizons="1,3,5,10",
        task_loss_weights="1:0.30,3:0.30,5:0.25,10:0.15,downside:0.35",
        score_horizon_weights="1:0.35,3:0.30,5:0.20,10:0.15",
        short_alpha_features=True,
        breakout_event_horizon=5,
        breakout_event_threshold=0.08,
        breakout_event_pullback_limit=0.03,
        breakout_event_loss_weight=0.15,
        clean_breakout_event_loss_weight=0.20,
        score_head_method="ridge",
        adaptive_task_weights=True,
    ),
    "short_expert_monthly_v1": ShortAlphaProfile(
        name="short_expert_monthly_v1",
        description=(
            "Monthly-first short-line expert candidate: short_expert_v1 plus "
            "monthly robust checkpoint selection."
        ),
        state_context=True,
        liquidity_context=True,
        ranking_loss_weight=0.03,
        listwise_loss_weight=0.06,
        listwise_temperature=0.30,
        prediction_horizons="1,3,5,10",
        task_loss_weights="1:0.30,3:0.30,5:0.25,10:0.15,downside:0.35",
        score_horizon_weights="1:0.35,3:0.30,5:0.20,10:0.15",
        short_alpha_features=True,
        breakout_event_horizon=5,
        breakout_event_threshold=0.08,
        breakout_event_pullback_limit=0.03,
        breakout_event_loss_weight=0.15,
        clean_breakout_event_loss_weight=0.20,
        score_head_method="ridge",
        adaptive_task_weights=True,
        research_objective_mode="execution_first",
        checkpoint_selection_objective="primary_monthly_robust_score",
    ),
    "short_expert_feature_only_v1": ShortAlphaProfile(
        name="short_expert_feature_only_v1",
        description=(
            "Current-code feature-only ablation: rerun the short expert monthly base under "
            "the refreshed first-week/breadth/failure feature pack without the v2 penalty bundle."
        ),
        state_context=True,
        liquidity_context=True,
        ranking_loss_weight=0.03,
        listwise_loss_weight=0.06,
        listwise_temperature=0.30,
        prediction_horizons="1,3,5,10",
        task_loss_weights="1:0.30,3:0.30,5:0.25,10:0.15,downside:0.35",
        score_horizon_weights="1:0.35,3:0.30,5:0.20,10:0.15",
        short_alpha_features=True,
        breakout_event_horizon=5,
        breakout_event_threshold=0.08,
        breakout_event_pullback_limit=0.03,
        breakout_event_loss_weight=0.15,
        clean_breakout_event_loss_weight=0.20,
        score_head_method="ridge",
        adaptive_task_weights=True,
    ),
    "short_expert_feature_only_monthly_v1": ShortAlphaProfile(
        name="short_expert_feature_only_monthly_v1",
        description=(
            "Monthly-first current-code feature-only ablation: fresh rerun of the short expert "
            "base with the refreshed feature pack and without state-targeted penalty extensions."
        ),
        state_context=True,
        liquidity_context=True,
        ranking_loss_weight=0.03,
        listwise_loss_weight=0.06,
        listwise_temperature=0.30,
        prediction_horizons="1,3,5,10",
        task_loss_weights="1:0.30,3:0.30,5:0.25,10:0.15,downside:0.35",
        score_horizon_weights="1:0.35,3:0.30,5:0.20,10:0.15",
        short_alpha_features=True,
        breakout_event_horizon=5,
        breakout_event_threshold=0.08,
        breakout_event_pullback_limit=0.03,
        breakout_event_loss_weight=0.15,
        clean_breakout_event_loss_weight=0.20,
        score_head_method="ridge",
        adaptive_task_weights=True,
        research_objective_mode="execution_first",
        checkpoint_selection_objective="primary_monthly_robust_score",
    ),
    "short_expert_penalty_only_v1": ShortAlphaProfile(
        name="short_expert_penalty_only_v1",
        description=(
            "Current-code penalty-only ablation: add the state-targeted downside/rank penalty "
            "bundle on top of the refreshed feature-only short expert base."
        ),
        state_context=True,
        liquidity_context=True,
        ranking_loss_weight=0.03,
        listwise_loss_weight=0.06,
        listwise_temperature=0.30,
        prediction_horizons="1,3,5,10",
        task_loss_weights="1:0.30,3:0.30,5:0.25,10:0.15,downside:0.35",
        score_horizon_weights="1:0.35,3:0.30,5:0.20,10:0.15",
        short_alpha_features=True,
        breakout_event_horizon=5,
        breakout_event_threshold=0.08,
        breakout_event_pullback_limit=0.03,
        breakout_event_loss_weight=0.15,
        clean_breakout_event_loss_weight=0.20,
        score_head_method="ridge",
        adaptive_task_weights=True,
        score_downside_penalty=0.35,
        structure_conditioning_mode="state_targeted_rank",
        target_state_names="trend_down_low_vol,trend_up_low_vol",
        target_state_attack_structures="neutral_mixed,pullback_rebound,trend_breakout,high_vol_expansion",
        target_state_protect_structures="low_vol_trend",
        target_state_rank_weight=1.35,
        target_state_protect_rank_weight=1.10,
    ),
    "short_expert_penalty_only_monthly_v1": ShortAlphaProfile(
        name="short_expert_penalty_only_monthly_v1",
        description=(
            "Monthly-first current-code penalty-only ablation: feature-only base plus "
            "state-targeted downside/rank penalties without the full v2 bundle."
        ),
        state_context=True,
        liquidity_context=True,
        ranking_loss_weight=0.03,
        listwise_loss_weight=0.06,
        listwise_temperature=0.30,
        prediction_horizons="1,3,5,10",
        task_loss_weights="1:0.30,3:0.30,5:0.25,10:0.15,downside:0.35",
        score_horizon_weights="1:0.35,3:0.30,5:0.20,10:0.15",
        short_alpha_features=True,
        breakout_event_horizon=5,
        breakout_event_threshold=0.08,
        breakout_event_pullback_limit=0.03,
        breakout_event_loss_weight=0.15,
        clean_breakout_event_loss_weight=0.20,
        score_head_method="ridge",
        adaptive_task_weights=True,
        research_objective_mode="execution_first",
        checkpoint_selection_objective="primary_monthly_robust_score",
        score_downside_penalty=0.35,
        structure_conditioning_mode="state_targeted_rank",
        target_state_names="trend_down_low_vol,trend_up_low_vol",
        target_state_attack_structures="neutral_mixed,pullback_rebound,trend_breakout,high_vol_expansion",
        target_state_protect_structures="low_vol_trend",
        target_state_rank_weight=1.35,
        target_state_protect_rank_weight=1.10,
    ),
    "short_expert_scorehead_v1": ShortAlphaProfile(
        name="short_expert_scorehead_v1",
        description=(
            "Short-line expert output-head probe: short_expert_v1 with a learned "
            "selection/confidence/sizing score head."
        ),
        state_context=True,
        liquidity_context=True,
        ranking_loss_weight=0.03,
        listwise_loss_weight=0.06,
        listwise_temperature=0.30,
        prediction_horizons="1,3,5,10",
        task_loss_weights="1:0.30,3:0.30,5:0.25,10:0.15,downside:0.35",
        score_horizon_weights="1:0.35,3:0.30,5:0.20,10:0.15",
        short_alpha_features=True,
        breakout_event_horizon=5,
        breakout_event_threshold=0.08,
        breakout_event_pullback_limit=0.03,
        breakout_event_loss_weight=0.15,
        clean_breakout_event_loss_weight=0.20,
        score_head_method="short_expert",
        adaptive_task_weights=True,
    ),
    "short_expert_scorehead_monthly_v1": ShortAlphaProfile(
        name="short_expert_scorehead_monthly_v1",
        description=(
            "Monthly-first short-line output-head probe: short_expert_scorehead_v1 "
            "with monthly robust checkpoint selection."
        ),
        state_context=True,
        liquidity_context=True,
        ranking_loss_weight=0.03,
        listwise_loss_weight=0.06,
        listwise_temperature=0.30,
        prediction_horizons="1,3,5,10",
        task_loss_weights="1:0.30,3:0.30,5:0.25,10:0.15,downside:0.35",
        score_horizon_weights="1:0.35,3:0.30,5:0.20,10:0.15",
        short_alpha_features=True,
        breakout_event_horizon=5,
        breakout_event_threshold=0.08,
        breakout_event_pullback_limit=0.03,
        breakout_event_loss_weight=0.15,
        clean_breakout_event_loss_weight=0.20,
        score_head_method="short_expert",
        adaptive_task_weights=True,
        research_objective_mode="execution_first",
        checkpoint_selection_objective="primary_monthly_robust_score",
    ),
    "short_expert_v2": ShortAlphaProfile(
        name="short_expert_v2",
        description=(
            "State-targeted short-line expert v2: stronger downside-aware monthly recipe "
            "with richer first-week/breadth/failure inputs."
        ),
        state_context=True,
        liquidity_context=True,
        structure_context=True,
        ranking_loss_weight=0.04,
        listwise_loss_weight=0.08,
        listwise_temperature=0.28,
        prediction_horizons="1,3,5,10",
        task_loss_weights="1:0.28,3:0.30,5:0.22,10:0.10,downside:0.55",
        score_horizon_weights="1:0.38,3:0.32,5:0.20,10:0.10",
        short_alpha_features=True,
        breakout_event_horizon=5,
        breakout_event_threshold=0.08,
        breakout_event_pullback_limit=0.03,
        breakout_event_loss_weight=0.20,
        clean_breakout_event_loss_weight=0.25,
        score_head_method="ridge",
        adaptive_task_weights=True,
        score_downside_penalty=0.35,
        structure_conditioning_mode="state_targeted_rank",
        target_state_names="trend_down_low_vol,trend_up_low_vol",
        target_state_attack_structures="neutral_mixed,pullback_rebound,trend_breakout,high_vol_expansion",
        target_state_protect_structures="low_vol_trend",
        target_state_rank_weight=1.35,
        target_state_protect_rank_weight=1.10,
    ),
    "short_expert_monthly_v2": ShortAlphaProfile(
        name="short_expert_monthly_v2",
        description=(
            "Monthly-first short-line expert v2: short_expert_v2 plus monthly robust "
            "checkpoint selection."
        ),
        state_context=True,
        liquidity_context=True,
        structure_context=True,
        ranking_loss_weight=0.04,
        listwise_loss_weight=0.08,
        listwise_temperature=0.28,
        prediction_horizons="1,3,5,10",
        task_loss_weights="1:0.28,3:0.30,5:0.22,10:0.10,downside:0.55",
        score_horizon_weights="1:0.38,3:0.32,5:0.20,10:0.10",
        short_alpha_features=True,
        breakout_event_horizon=5,
        breakout_event_threshold=0.08,
        breakout_event_pullback_limit=0.03,
        breakout_event_loss_weight=0.20,
        clean_breakout_event_loss_weight=0.25,
        score_head_method="ridge",
        adaptive_task_weights=True,
        research_objective_mode="execution_first",
        checkpoint_selection_objective="primary_monthly_robust_score",
        score_downside_penalty=0.35,
        structure_conditioning_mode="state_targeted_rank",
        target_state_names="trend_down_low_vol,trend_up_low_vol",
        target_state_attack_structures="neutral_mixed,pullback_rebound,trend_breakout,high_vol_expansion",
        target_state_protect_structures="low_vol_trend",
        target_state_rank_weight=1.35,
        target_state_protect_rank_weight=1.10,
    ),
}

PROFILE_ALIASES: dict[str, str] = {
    "default": "baseline_current",
    "baseline": "baseline_current",
    "state": "state_context_v1",
    "capacity": "state_liquidity_listwise_v1",
    "target": "short_target_v1",
    "input": "short_input_v1",
    "combo": "short_combo_v1",
    "expert": "short_expert_v1",
    "expert_monthly": "short_expert_monthly_v1",
    "expert_feature_only": "short_expert_feature_only_v1",
    "expert_feature_only_monthly": "short_expert_feature_only_monthly_v1",
    "expert_penalty_only": "short_expert_penalty_only_v1",
    "expert_penalty_only_monthly": "short_expert_penalty_only_monthly_v1",
    "expert_head": "short_expert_scorehead_v1",
    "expert_head_monthly": "short_expert_scorehead_monthly_v1",
    "expert_v2": "short_expert_v2",
    "expert_monthly_v2": "short_expert_monthly_v2",
}

DEFAULT_SHORT_ALPHA_PROFILE = "baseline_current"


def resolve_profile_name(name: str) -> str:
    normalized = str(name or "").strip()
    if not normalized:
        raise KeyError("Empty short-alpha profile name.")
    return PROFILE_ALIASES.get(normalized, normalized)


def get_profile(name: str) -> ShortAlphaProfile:
    resolved = resolve_profile_name(name)
    if resolved not in PROFILE_REGISTRY:
        available = ", ".join(sorted(PROFILE_REGISTRY))
        aliases = ", ".join(f"{alias}->{target}" for alias, target in sorted(PROFILE_ALIASES.items()))
        raise KeyError(f"Unknown short-alpha profile: {name}. Available: {available}. Aliases: {aliases}")
    return PROFILE_REGISTRY[resolved]


def build_profile_cli_args(profile: ShortAlphaProfile, *, include_objective_overrides: bool = True) -> list[str]:
    args = [
        "--prediction-horizons",
        profile.prediction_horizons,
        "--task-loss-weights",
        profile.task_loss_weights,
        "--score-horizon-weights",
        profile.score_horizon_weights,
        "--ranking-loss-weight",
        str(profile.ranking_loss_weight),
        "--listwise-loss-weight",
        str(profile.listwise_loss_weight),
        "--listwise-temperature",
        str(profile.listwise_temperature),
        "--breakout-event-horizon",
        str(profile.breakout_event_horizon),
        "--breakout-event-threshold",
        str(profile.breakout_event_threshold),
        "--breakout-event-pullback-limit",
        str(profile.breakout_event_pullback_limit),
        "--breakout-event-loss-weight",
        str(profile.breakout_event_loss_weight),
        "--clean-breakout-event-loss-weight",
        str(profile.clean_breakout_event_loss_weight),
    ]
    if profile.score_head_method:
        args.extend(["--score-head-method", profile.score_head_method])
    if include_objective_overrides and profile.research_objective_mode:
        args.extend(["--research-objective-mode", profile.research_objective_mode])
    if include_objective_overrides and profile.checkpoint_selection_objective:
        args.extend(["--checkpoint-selection-objective", profile.checkpoint_selection_objective])
    if profile.score_downside_penalty is not None:
        args.extend(["--score-downside-penalty", str(profile.score_downside_penalty)])
    if profile.structure_conditioning_mode:
        args.extend(["--structure-conditioning-mode", profile.structure_conditioning_mode])
    if profile.target_state_names:
        args.extend(["--target-state-names", profile.target_state_names])
    if profile.target_state_attack_structures:
        args.extend(["--target-state-attack-structures", profile.target_state_attack_structures])
    if profile.target_state_protect_structures:
        args.extend(["--target-state-protect-structures", profile.target_state_protect_structures])
    if profile.target_state_rank_weight is not None:
        args.extend(["--target-state-rank-weight", str(profile.target_state_rank_weight)])
    if profile.target_state_protect_rank_weight is not None:
        args.extend(["--target-state-protect-rank-weight", str(profile.target_state_protect_rank_weight)])
    if profile.adaptive_task_weights:
        args.append("--adaptive-task-weights")
    if profile.state_context:
        args.append("--state-context")
    if profile.liquidity_context:
        args.append("--liquidity-context")
    if profile.structure_context:
        args.append("--structure-context")
    if profile.short_alpha_features:
        args.append("--short-alpha-features")
    return args


def list_profile_lines() -> list[str]:
    lines = [f"default={DEFAULT_SHORT_ALPHA_PROFILE}", "profiles:"]
    for key in sorted(PROFILE_REGISTRY):
        profile = PROFILE_REGISTRY[key]
        lines.append(f"- {profile.name}: {profile.description}")
    alias_text = ", ".join(f"{alias}->{target}" for alias, target in sorted(PROFILE_ALIASES.items()))
    lines.append(f"aliases: {alias_text}")
    return lines
