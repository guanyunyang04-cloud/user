from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from daily_research.baseline.data_provider import get_latest_completed_trading_date
from daily_research.execution.entrypoint_utils import inject_default_arg, inject_flag_arg
from daily_research.execution.strategy_manifest import load_strategy_manifest


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
_UPDATE_DEFAULT_PRODUCTION_SCRIPT = (_DAILY_RESEARCH_ROOT / "execution" / "update_default_candidate_production.py").resolve()


def _source(name: str) -> str:
    return str((_DYNAMIC_GRAPH_FORMAL_ROOT / name).resolve())


STATIC_PROFILE_REGISTRY: dict[str, ResearchCandidateProfile] = {
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

ACTIVE_EXECUTION_CANDIDATE_PROFILE = "active_execution_strategy"
LEGACY_DEFAULT_EXECUTION_CANDIDATE_PROFILE = "regoff_k2_10d_ensemble_native_anchor"

PROFILE_ALIASES: dict[str, str] = {
    "aggressive": "regon_k1_10d_ensemble_native_anchor",
    "soft_guard": "regoff_k2_10d_market_state_guard_v1",
    "baseline": "target_weight_1d_baseline",
    "robust_auto": "execalign_auto_r4_topk2_1d_regoff",
}

DEFAULT_EXECUTION_CANDIDATE_PROFILE = ACTIVE_EXECUTION_CANDIDATE_PROFILE
UPSIDE_EXECUTION_CANDIDATE_PROFILE = "regon_k1_10d_ensemble_native_anchor"


def _build_active_execution_profile() -> ResearchCandidateProfile | None:
    manifest = load_strategy_manifest()
    if not manifest:
        return None
    source_target_weight = str(manifest.get("source_target_weight_panel_csv", "")).strip()
    source_score = str(manifest.get("source_score_panel_csv", "")).strip()
    trade_plan_target_weight = str(manifest.get("trade_plan_target_weight_panel_csv", "")).strip()
    trade_plan_score = str(manifest.get("trade_plan_score_panel_csv", "")).strip()
    if not source_target_weight or not source_score or not trade_plan_target_weight or not trade_plan_score:
        return None
    description = (
        "Current active execution strategy promoted from the primary research winner; "
        "default daily execution should follow this manifest instead of a hard-coded profile."
    )
    strategy_name = str(manifest.get("strategy_name", "")).strip()
    panel_mode = str(manifest.get("panel_mode", "")).strip()
    if strategy_name:
        description = f"{description} strategy={strategy_name}."
    if panel_mode:
        description = f"{description} panel_mode={panel_mode}."
    return ResearchCandidateProfile(
        name=ACTIVE_EXECUTION_CANDIDATE_PROFILE,
        description=description,
        target_weight_panel_csv=source_target_weight,
        score_panel_csv=source_score,
        candidate_label=str(manifest.get("candidate_label", "")).strip() or "active_execution_strategy",
        trade_plan_target_weight_panel_csv=trade_plan_target_weight,
        trade_plan_score_panel_csv=trade_plan_score,
        trade_plan_candidate_label=str(manifest.get("trade_plan_candidate_label", "")).strip()
        or str(manifest.get("candidate_label", "")).strip()
        or "active_execution_strategy",
        data_source=str(manifest.get("data_source", "tq") or "tq"),
        benchmark=str(manifest.get("benchmark", "000300.SH") or "000300.SH"),
        backtest_start_date=str(manifest.get("backtest_start_date", "20210101") or "20210101"),
        trade_plan_start_date=str(manifest.get("trade_plan_start_date", "20210101") or "20210101"),
        rebalance_freq=str(manifest.get("rebalance_freq", "1d") or "1d"),
        rebalance_offset_mode=str(manifest.get("rebalance_offset_mode", "single") or "single"),
        rebalance_anchor_date=str(manifest.get("rebalance_anchor_date", "") or ""),
        target_weight_top_k=int(manifest.get("target_weight_top_k", 0) or 0),
        target_weight_min_weight=float(manifest.get("target_weight_min_weight", 0.0) or 0.0),
        target_weight_power=float(manifest.get("target_weight_power", 1.0) or 1.0),
        target_weight_full_invest=bool(manifest.get("target_weight_full_invest", False)),
        use_market_regime_filter=bool(manifest.get("use_market_regime_filter", False)),
        soft_state_profile=str(manifest.get("soft_state_profile", "")).strip(),
        refresh_run_dir=str(manifest.get("source_refresh_run_dir", "")).strip()
        or str(manifest.get("source_run_dir", "")).strip(),
        trade_plan_refresh_run_dir=str(manifest.get("trade_plan_refresh_run_dir", "")).strip()
        or str(manifest.get("production_root", "")).strip(),
        trade_plan_model_manifest_json=str(manifest.get("production_manifest_json", "")).strip(),
    )


def _current_default_profile_name() -> str:
    active_profile = _build_active_execution_profile()
    if active_profile is not None:
        return active_profile.name
    return LEGACY_DEFAULT_EXECUTION_CANDIDATE_PROFILE


def _profile_registry() -> dict[str, ResearchCandidateProfile]:
    registry = dict(STATIC_PROFILE_REGISTRY)
    active_profile = _build_active_execution_profile()
    if active_profile is not None:
        registry[active_profile.name] = active_profile
    return registry


def resolve_profile_name(name: str) -> str:
    normalized = str(name or "").strip()
    if not normalized:
        raise KeyError("Empty candidate profile name.")
    if normalized == "default":
        return _current_default_profile_name()
    return PROFILE_ALIASES.get(normalized, normalized)


def get_profile(name: str) -> ResearchCandidateProfile:
    resolved = resolve_profile_name(name)
    registry = _profile_registry()
    if resolved not in registry:
        available = ", ".join(sorted(registry))
        aliases = ", ".join(f"{alias}->{target}" for alias, target in sorted(PROFILE_ALIASES.items()))
        raise KeyError(f"Unknown candidate profile: {name}. Available: {available}. Aliases: {aliases}")
    return registry[resolved]


def list_profile_lines() -> list[str]:
    registry = _profile_registry()
    lines = [f"default={_current_default_profile_name()}", f"upside={UPSIDE_EXECUTION_CANDIDATE_PROFILE}", "profiles:"]
    for key in sorted(registry):
        profile = registry[key]
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


def _load_json_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_timestamp(raw: object) -> pd.Timestamp | None:
    if raw in {None, ""}:
        return None
    try:
        return pd.Timestamp(raw)
    except Exception:
        return None


def _coerce_positive_int(raw: object) -> int:
    try:
        value = int(raw)
    except Exception:
        return 0
    return value if value > 0 else 0


def _estimate_trading_day_lag(start_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    if end_ts <= start_ts:
        return 0
    return max(len(pd.bdate_range(start=start_ts, end=end_ts)) - 1, 0)


def _build_auto_retrain_plan(
    *,
    manifest_path: Path,
    latest_completed: pd.Timestamp,
) -> dict[str, Any]:
    manifest = _load_json_payload(manifest_path)
    policy = manifest.get("retrain_frequency_policy") if isinstance(manifest.get("retrain_frequency_policy"), dict) else {}
    launch_cutoff_date = _safe_timestamp(manifest.get("launch_cutoff_date"))
    created_at = _safe_timestamp(manifest.get("created_at"))
    if launch_cutoff_date is None and created_at is not None:
        launch_cutoff_date = pd.Timestamp(created_at).normalize()
    preferred_cadence = str(policy.get("auto_retrain_mode") or policy.get("preferred_cadence") or "").strip()
    auto_retrain_enabled = bool(policy.get("auto_retrain_enabled", False))
    fallback_trading_days = (
        _coerce_positive_int(policy.get("auto_retrain_fallback_trading_days"))
        or _coerce_positive_int(policy.get("warn_after_trading_days"))
    )
    plan: dict[str, Any] = {
        "enabled": auto_retrain_enabled,
        "manifest": manifest,
        "policy": policy,
        "launch_cutoff_date": launch_cutoff_date,
        "preferred_cadence": preferred_cadence,
        "fallback_trading_days": fallback_trading_days,
        "estimated_trading_day_lag": None,
        "should_retrain": False,
        "reason": "",
    }
    if not auto_retrain_enabled or launch_cutoff_date is None or latest_completed <= launch_cutoff_date:
        return plan

    lag = _estimate_trading_day_lag(launch_cutoff_date, latest_completed)
    plan["estimated_trading_day_lag"] = int(lag)

    if preferred_cadence == "monthly_calendar":
        if latest_completed.to_period("M") > launch_cutoff_date.to_period("M"):
            plan["should_retrain"] = True
            plan["reason"] = (
                f"launch_cutoff_date={launch_cutoff_date.date()} 已跨到新自然月 "
                f"{latest_completed.strftime('%Y-%m')}，按 Retrain Monthly 自动重训。"
            )
        return plan

    if fallback_trading_days > 0 and lag >= fallback_trading_days:
        plan["should_retrain"] = True
        plan["reason"] = (
            f"launch_cutoff_date={launch_cutoff_date.date()} 相对最新完成交易日 "
            f"{latest_completed.date()} 估算已滞后 {lag} 个交易日，达到自动重训阈值 {fallback_trading_days}。"
        )
    return plan


def _maybe_auto_retrain_production(profile: ResearchCandidateProfile, *, mode: str) -> None:
    if mode != "trade_plan" or not profile.trade_plan_model_manifest_json:
        return
    manifest_path = Path(profile.trade_plan_model_manifest_json)
    latest_completed = pd.Timestamp(get_latest_completed_trading_date())
    plan = _build_auto_retrain_plan(manifest_path=manifest_path, latest_completed=latest_completed)
    if not plan.get("enabled"):
        return
    if not plan.get("should_retrain"):
        print(
            "production_retrain_status=monthly_auto_ready"
            f" launch_cutoff_date={plan.get('launch_cutoff_date').date() if plan.get('launch_cutoff_date') is not None else ''}"
            f" latest_completed_date={latest_completed.date()}"
        )
        return

    manifest = plan.get("manifest") if isinstance(plan.get("manifest"), dict) else {}
    cmd = [sys.executable, str(_UPDATE_DEFAULT_PRODUCTION_SCRIPT), "--end-date", latest_completed.strftime("%Y%m%d")]
    source_run_dir = str(manifest.get("source_formal_run_dir", "")).strip()
    production_root = str(manifest.get("production_root", "")).strip()
    if source_run_dir:
        cmd.extend(["--source-run-dir", source_run_dir])
    if production_root:
        cmd.extend(["--production-root", production_root])

    print("production_retrain_status=monthly_auto_triggered")
    print(f"production_retrain_reason={plan.get('reason', '')}")
    subprocess.run(
        cmd,
        check=True,
        cwd=str(_DAILY_RESEARCH_ROOT.parent),
    )


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
    _maybe_auto_retrain_production(profile, mode=mode)
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
