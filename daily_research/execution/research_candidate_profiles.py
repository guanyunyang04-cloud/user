from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from daily_research.execution.entrypoint_utils import inject_default_arg, inject_flag_arg


@dataclass(frozen=True)
class ResearchCandidateProfile:
    name: str
    description: str
    target_weight_panel_csv: str
    score_panel_csv: str
    candidate_label: str
    data_source: str = "tq"
    benchmark: str = "000300.SH"
    backtest_start_date: str = "20250101"
    trade_plan_start_date: str = "20210101"
    rebalance_freq: str = "10d"
    rebalance_offset_mode: str = "all"
    rebalance_anchor_date: str = "2025-01-02"
    target_weight_top_k: int = 0
    target_weight_min_weight: float = 0.0
    target_weight_power: float = 1.0
    target_weight_full_invest: bool = False
    use_market_regime_filter: bool = True
    soft_state_profile: str = ""


_DAILY_RESEARCH_ROOT = Path(__file__).resolve().parents[1]
_DYNAMIC_GRAPH_FORMAL_ROOT = _DAILY_RESEARCH_ROOT / "output" / "deep_alpha_liquid500_dynamic_graph_bridge_20260401_formal_r1"
_EXECALIGN_AUTO_R4_ROOT = _DAILY_RESEARCH_ROOT / "output" / "deep_alpha_liquid500_dynamic_graph_v1_execalign_auto_20260401_formal_r4"


def _source(name: str) -> str:
    return str((_DYNAMIC_GRAPH_FORMAL_ROOT / name).resolve())


PROFILE_REGISTRY: dict[str, ResearchCandidateProfile] = {
    "regoff_k2_10d_ensemble_native_anchor": ResearchCandidateProfile(
        name="regoff_k2_10d_ensemble_native_anchor",
        description="Current default execution candidate: 10d anchored all-offset ensemble, top-k 2, regime filter off.",
        target_weight_panel_csv=_source("daily_target_weight_panel.csv"),
        score_panel_csv=_source("daily_score_panel.csv"),
        candidate_label="dynamic_graph_regoff_k2_10d_ensemble_native_anchor",
        target_weight_top_k=2,
        use_market_regime_filter=False,
    ),
    "regon_k1_10d_ensemble_native_anchor": ResearchCandidateProfile(
        name="regon_k1_10d_ensemble_native_anchor",
        description="More aggressive upside comparator: 10d anchored all-offset ensemble, top-k 1, regime filter on.",
        target_weight_panel_csv=_source("daily_target_weight_panel.csv"),
        score_panel_csv=_source("daily_score_panel.csv"),
        candidate_label="dynamic_graph_regon_k1_10d_ensemble_native_anchor",
        target_weight_top_k=1,
        use_market_regime_filter=True,
    ),
    "regoff_k2_10d_market_state_guard_v1": ResearchCandidateProfile(
        name="regoff_k2_10d_market_state_guard_v1",
        description="Risk-shaping comparator: regoff_k2 bridge plus soft market_state guard v1.",
        target_weight_panel_csv=_source("daily_target_weight_panel.csv"),
        score_panel_csv=_source("daily_score_panel.csv"),
        candidate_label="dynamic_graph_regoff_k2_10d_market_state_guard_v1",
        target_weight_top_k=2,
        use_market_regime_filter=False,
        soft_state_profile="market_state_guard_v1",
    ),
    "target_weight_1d_baseline": ResearchCandidateProfile(
        name="target_weight_1d_baseline",
        description="Direct 1d target-weight bridge baseline for translation-loss comparison.",
        target_weight_panel_csv=_source("daily_target_weight_panel.csv"),
        score_panel_csv=_source("daily_score_panel.csv"),
        candidate_label="dynamic_graph_v1_execbridge_weightpanel_baseline",
        rebalance_freq="1d",
        rebalance_offset_mode="single",
        rebalance_anchor_date="",
        target_weight_top_k=0,
        use_market_regime_filter=False,
    ),
    "execalign_auto_r4_topk2_1d_regoff": ResearchCandidateProfile(
        name="execalign_auto_r4_topk2_1d_regoff",
        description="Execution-objective auto-aligned candidate from the 2026-04-01 robust-composite + 252d train_eval formal run.",
        target_weight_panel_csv=str((_EXECALIGN_AUTO_R4_ROOT / "execution_aligned_daily_target_weight_panel.csv").resolve()),
        score_panel_csv=str((_EXECALIGN_AUTO_R4_ROOT / "execution_aligned_daily_score_panel.csv").resolve()),
        candidate_label="dynamic_graph_execalign_auto_r4_topk2_1d_regoff",
        backtest_start_date="20250318",
        trade_plan_start_date="20210101",
        rebalance_freq="1d",
        rebalance_offset_mode="single",
        rebalance_anchor_date="",
        target_weight_top_k=0,
        use_market_regime_filter=False,
    ),
}

PROFILE_ALIASES: dict[str, str] = {
    "default": "regoff_k2_10d_ensemble_native_anchor",
    "aggressive": "regon_k1_10d_ensemble_native_anchor",
    "soft_guard": "regoff_k2_10d_market_state_guard_v1",
    "baseline": "target_weight_1d_baseline",
    "robust_auto": "execalign_auto_r4_topk2_1d_regoff",
}

DEFAULT_EXECUTION_CANDIDATE_PROFILE = "regoff_k2_10d_ensemble_native_anchor"
UPSIDE_EXECUTION_CANDIDATE_PROFILE = "regon_k1_10d_ensemble_native_anchor"


def resolve_profile_name(name: str) -> str:
    normalized = str(name or "").strip()
    if not normalized:
        raise KeyError("Empty candidate profile name.")
    return PROFILE_ALIASES.get(normalized, normalized)


def get_profile(name: str) -> ResearchCandidateProfile:
    resolved = resolve_profile_name(name)
    if resolved not in PROFILE_REGISTRY:
        available = ", ".join(sorted(PROFILE_REGISTRY))
        aliases = ", ".join(f"{alias}->{target}" for alias, target in sorted(PROFILE_ALIASES.items()))
        raise KeyError(f"Unknown candidate profile: {name}. Available: {available}. Aliases: {aliases}")
    return PROFILE_REGISTRY[resolved]


def list_profile_lines() -> list[str]:
    lines = [f"default={DEFAULT_EXECUTION_CANDIDATE_PROFILE}", f"upside={UPSIDE_EXECUTION_CANDIDATE_PROFILE}", "profiles:"]
    for key in sorted(PROFILE_REGISTRY):
        profile = PROFILE_REGISTRY[key]
        lines.append(f"- {profile.name}: {profile.description}")
    if PROFILE_ALIASES:
        alias_text = ", ".join(f"{alias}->{target}" for alias, target in sorted(PROFILE_ALIASES.items()))
        lines.append(f"aliases: {alias_text}")
    return lines


def apply_profile_defaults(profile_name: str, *, mode: str) -> ResearchCandidateProfile:
    profile = get_profile(profile_name)
    if mode == "backtest":
        inject_default_arg("--data-source", profile.data_source)
        inject_default_arg("--benchmark", profile.benchmark)
        inject_default_arg("--start-date", profile.backtest_start_date)
        inject_default_arg("--target-weight-panel-csv", profile.target_weight_panel_csv)
        inject_default_arg("--score-panel-csv", profile.score_panel_csv)
    elif mode == "trade_plan":
        inject_default_arg("--data-source", profile.data_source)
        inject_default_arg("--benchmark", profile.benchmark)
        inject_default_arg("--start-date", profile.trade_plan_start_date)
        inject_default_arg("--external-target-weight-csv", profile.target_weight_panel_csv)
        inject_default_arg("--external-score-csv", profile.score_panel_csv)
    else:
        raise ValueError(f"Unsupported profile application mode: {mode}")
    inject_default_arg("--candidate-label", profile.candidate_label)
    inject_default_arg("--rebalance-freq", profile.rebalance_freq)
    inject_default_arg("--rebalance-offset-mode", profile.rebalance_offset_mode)
    if profile.rebalance_anchor_date:
        inject_default_arg("--rebalance-anchor-date", profile.rebalance_anchor_date)
    inject_default_arg("--target-weight-top-k", str(profile.target_weight_top_k))
    inject_default_arg("--target-weight-min-weight", str(profile.target_weight_min_weight))
    inject_default_arg("--target-weight-power", str(profile.target_weight_power))
    if profile.target_weight_full_invest:
        inject_flag_arg("--target-weight-full-invest")
    if not profile.use_market_regime_filter:
        inject_flag_arg("--no-market-regime-filter")
    if profile.soft_state_profile:
        inject_default_arg("--soft-state-profile", profile.soft_state_profile)
    return profile
