from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from daily_research.baseline.data_provider import get_latest_completed_trading_date
from daily_research.execution.entrypoint_utils import inject_default_arg, inject_flag_arg


@dataclass(frozen=True)
class ResearchCandidateProfile:
    name: str
    description: str
    target_weight_panel_csv: str
    score_panel_csv: str
    candidate_label: str
    trade_plan_target_weight_panel_csv: str = ""
    trade_plan_score_panel_csv: str = ""
    trade_plan_candidate_label: str = ""
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
    refresh_run_dir: str = ""
    trade_plan_refresh_run_dir: str = ""
    trade_plan_model_manifest_json: str = ""


_DAILY_RESEARCH_ROOT = Path(__file__).resolve().parents[1]
_DYNAMIC_GRAPH_FORMAL_ROOT = _DAILY_RESEARCH_ROOT / "output" / "deep_alpha_liquid500_dynamic_graph_bridge_20260401_formal_r1"
_DYNAMIC_GRAPH_PRODUCTION_ROOT = _DAILY_RESEARCH_ROOT / "output" / "deep_alpha_liquid500_dynamic_graph_bridge_production_default"
_EXECALIGN_AUTO_R4_ROOT = _DAILY_RESEARCH_ROOT / "output" / "deep_alpha_liquid500_dynamic_graph_v1_execalign_auto_20260401_formal_r4"


def _source(name: str) -> str:
    return str((_DYNAMIC_GRAPH_FORMAL_ROOT / name).resolve())


PROFILE_REGISTRY: dict[str, ResearchCandidateProfile] = {
    "regoff_k2_10d_ensemble_native_anchor": ResearchCandidateProfile(
        name="regoff_k2_10d_ensemble_native_anchor",
        description="Current daily default execution strategy: formal winner frozen for research evidence, production full-fit model for daily trade plan; 10d anchored all-offset ensemble, top-k 2, regime filter off.",
        target_weight_panel_csv=_source("daily_live_target_weight_panel.csv"),
        score_panel_csv=_source("daily_live_score_panel.csv"),
        candidate_label="dynamic_graph_regoff_k2_10d_ensemble_native_anchor",
        trade_plan_target_weight_panel_csv=str((_DYNAMIC_GRAPH_PRODUCTION_ROOT / "daily_live_target_weight_panel.csv").resolve()),
        trade_plan_score_panel_csv=str((_DYNAMIC_GRAPH_PRODUCTION_ROOT / "daily_live_score_panel.csv").resolve()),
        trade_plan_candidate_label="dynamic_graph_regoff_k2_10d_ensemble_native_anchor_production_fullfit",
        target_weight_top_k=2,
        use_market_regime_filter=False,
        refresh_run_dir=str(_DYNAMIC_GRAPH_FORMAL_ROOT.resolve()),
        trade_plan_refresh_run_dir=str(_DYNAMIC_GRAPH_PRODUCTION_ROOT.resolve()),
        trade_plan_model_manifest_json=str((_DYNAMIC_GRAPH_PRODUCTION_ROOT / "production_retrain_manifest.json").resolve()),
    ),
    "regon_k1_10d_ensemble_native_anchor": ResearchCandidateProfile(
        name="regon_k1_10d_ensemble_native_anchor",
        description="More aggressive upside comparator: 10d anchored all-offset ensemble, top-k 1, regime filter on.",
        target_weight_panel_csv=_source("daily_live_target_weight_panel.csv"),
        score_panel_csv=_source("daily_live_score_panel.csv"),
        candidate_label="dynamic_graph_regon_k1_10d_ensemble_native_anchor",
        target_weight_top_k=1,
        use_market_regime_filter=True,
        refresh_run_dir=str(_DYNAMIC_GRAPH_FORMAL_ROOT.resolve()),
    ),
    "regoff_k2_10d_market_state_guard_v1": ResearchCandidateProfile(
        name="regoff_k2_10d_market_state_guard_v1",
        description="Risk-shaping comparator: regoff_k2 bridge plus soft market_state guard v1.",
        target_weight_panel_csv=_source("daily_live_target_weight_panel.csv"),
        score_panel_csv=_source("daily_live_score_panel.csv"),
        candidate_label="dynamic_graph_regoff_k2_10d_market_state_guard_v1",
        target_weight_top_k=2,
        use_market_regime_filter=False,
        soft_state_profile="market_state_guard_v1",
        refresh_run_dir=str(_DYNAMIC_GRAPH_FORMAL_ROOT.resolve()),
    ),
    "target_weight_1d_baseline": ResearchCandidateProfile(
        name="target_weight_1d_baseline",
        description="Direct 1d target-weight bridge baseline for translation-loss comparison.",
        target_weight_panel_csv=_source("daily_live_target_weight_panel.csv"),
        score_panel_csv=_source("daily_live_score_panel.csv"),
        candidate_label="dynamic_graph_v1_execbridge_weightpanel_baseline",
        rebalance_freq="1d",
        rebalance_offset_mode="single",
        rebalance_anchor_date="",
        target_weight_top_k=0,
        use_market_regime_filter=False,
        refresh_run_dir=str(_DYNAMIC_GRAPH_FORMAL_ROOT.resolve()),
    ),
    "execalign_auto_r4_topk2_1d_regoff": ResearchCandidateProfile(
        name="execalign_auto_r4_topk2_1d_regoff",
        description="High-turnover no-cost comparator from the 2026-04-01 robust-composite + 252d train_eval formal run; failed realistic-cost review on 2026-04-01.",
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
        refresh_run_dir=str(_EXECALIGN_AUTO_R4_ROOT.resolve()),
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


def _read_panel_latest_date(path_str: str) -> pd.Timestamp | None:
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, usecols=["date"])
    except Exception:
        return None
    if frame.empty or "date" not in frame.columns:
        return None
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return pd.Timestamp(dates.max())


def _trade_plan_target_weight_path(profile: ResearchCandidateProfile) -> str:
    return profile.trade_plan_target_weight_panel_csv or profile.target_weight_panel_csv


def _trade_plan_score_path(profile: ResearchCandidateProfile) -> str:
    return profile.trade_plan_score_panel_csv or profile.score_panel_csv


def _trade_plan_candidate_label(profile: ResearchCandidateProfile) -> str:
    return profile.trade_plan_candidate_label or profile.candidate_label


def _refresh_run_dir_for_mode(profile: ResearchCandidateProfile, mode: str) -> str:
    if mode == "trade_plan" and profile.trade_plan_refresh_run_dir:
        return profile.trade_plan_refresh_run_dir
    return profile.refresh_run_dir


def _target_weight_path_for_mode(profile: ResearchCandidateProfile, mode: str) -> str:
    if mode == "trade_plan":
        return _trade_plan_target_weight_path(profile)
    return profile.target_weight_panel_csv


def _score_path_for_mode(profile: ResearchCandidateProfile, mode: str) -> str:
    if mode == "trade_plan":
        return _trade_plan_score_path(profile)
    return profile.score_panel_csv


def _candidate_label_for_mode(profile: ResearchCandidateProfile, mode: str) -> str:
    if mode == "trade_plan":
        return _trade_plan_candidate_label(profile)
    return profile.candidate_label


def _ensure_live_panels(profile: ResearchCandidateProfile, *, mode: str) -> None:
    refresh_run_dir = _refresh_run_dir_for_mode(profile, mode)
    if not refresh_run_dir:
        return
    latest_completed = pd.Timestamp(get_latest_completed_trading_date())
    latest_panel_date = _read_panel_latest_date(_target_weight_path_for_mode(profile, mode))
    if latest_panel_date is not None and latest_panel_date >= latest_completed:
        return
    from daily_research.deep_alpha.export_live_panels_from_run import refresh_live_panels_for_run

    refresh_live_panels_for_run(
        Path(refresh_run_dir),
        latest_end_date=latest_completed.strftime("%Y%m%d"),
    )


def apply_profile_defaults(profile_name: str, *, mode: str) -> ResearchCandidateProfile:
    profile = get_profile(profile_name)
    _ensure_live_panels(profile, mode=mode)
    if mode == "backtest":
        inject_default_arg("--data-source", profile.data_source)
        inject_default_arg("--benchmark", profile.benchmark)
        inject_default_arg("--start-date", profile.backtest_start_date)
        inject_default_arg("--target-weight-panel-csv", _target_weight_path_for_mode(profile, mode))
        inject_default_arg("--score-panel-csv", _score_path_for_mode(profile, mode))
    elif mode == "trade_plan":
        inject_default_arg("--data-source", profile.data_source)
        inject_default_arg("--benchmark", profile.benchmark)
        inject_default_arg("--start-date", profile.trade_plan_start_date)
        inject_default_arg("--external-target-weight-csv", _target_weight_path_for_mode(profile, mode))
        inject_default_arg("--external-score-csv", _score_path_for_mode(profile, mode))
        if profile.trade_plan_model_manifest_json:
            inject_default_arg("--external-model-manifest", profile.trade_plan_model_manifest_json)
    else:
        raise ValueError(f"Unsupported profile application mode: {mode}")
    inject_default_arg("--candidate-label", _candidate_label_for_mode(profile, mode))
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
