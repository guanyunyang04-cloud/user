from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daily_research.deep_alpha.execution_alignment import resolve_profile
from daily_research.deep_alpha.research_objective import resolve_primary_panel_mode


DEFAULT_ACTIVE_EXECUTION_STRATEGY_MANIFEST = Path(
    "daily_research/output/active_execution_strategy.json"
).resolve()


def normalize_liquidity_pool_name(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return ""
    if text.startswith("liquid") and text[6:].isdigit():
        return text
    if text.isdigit():
        return f"liquid{int(text)}"
    return ""


def liquidity_pool_size_from_name(pool_name: str) -> int:
    normalized = normalize_liquidity_pool_name(pool_name)
    if normalized.startswith("liquid") and normalized[6:].isdigit():
        return int(normalized[6:])
    return 0


def resolve_manifest_liquidity_pool_name(payload: dict[str, Any] | None) -> str:
    manifest = payload if isinstance(payload, dict) else {}
    explicit = normalize_liquidity_pool_name(manifest.get("liquidity_pool_name", ""))
    if explicit:
        return explicit
    for key in ("rolling_liquidity_pool", "liquidity_pool"):
        resolved = normalize_liquidity_pool_name(manifest.get(key, ""))
        if resolved:
            return resolved
    explicit_size = liquidity_pool_size_from_name(str(manifest.get("liquidity_pool_size", "") or ""))
    if explicit_size > 0:
        return f"liquid{explicit_size}"
    return ""


def infer_liquidity_pool_name(*metrics_payloads: dict[str, Any] | None) -> str:
    for payload in metrics_payloads:
        resolved = resolve_manifest_liquidity_pool_name(payload)
        if resolved:
            return resolved
    return ""


def load_strategy_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = (path or DEFAULT_ACTIVE_EXECUTION_STRATEGY_MANIFEST).resolve()
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_strategy_manifest(payload: dict[str, Any], path: Path | None = None) -> Path:
    manifest_path = (path or DEFAULT_ACTIVE_EXECUTION_STRATEGY_MANIFEST).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def resolve_panel_filenames(panel_mode: str) -> tuple[str, str]:
    normalized = str(panel_mode or "raw").strip().lower()
    if normalized == "execution_aligned":
        return (
            "execution_aligned_daily_live_target_weight_panel.csv",
            "execution_aligned_daily_live_score_panel.csv",
        )
    return (
        "daily_live_target_weight_panel.csv",
        "daily_live_score_panel.csv",
    )


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _build_effective_live_metadata(
    *,
    production_root: Path,
    resolved_panel_mode: str,
    execution_policy_label: str,
    execution_profile_spec: dict[str, Any],
    bridge_meta: dict[str, Any],
    has_production_manifest: bool,
) -> dict[str, Any]:
    score_reference_path = production_root / "daily_live_score_reference.json"
    static_fallback_meta = _load_optional_json(
        production_root / "static_fallback_daily_live_target_weight_meta.json"
    )
    effective_bridge_meta = dict(bridge_meta)
    if not effective_bridge_meta and isinstance(static_fallback_meta.get("bridge_meta"), dict):
        effective_bridge_meta = dict(static_fallback_meta.get("bridge_meta", {}))

    profile_description = str(
        execution_profile_spec.get("description", "")
        or execution_profile_spec.get("profile_description", "")
        or static_fallback_meta.get("static_fallback_profile_description", "")
        or ""
    ).strip()

    if resolved_panel_mode == "execution_aligned":
        if has_production_manifest:
            weight_generation_note = (
                "Current live weights come from the promoted production full-fit "
                "execution-aligned live target-weight panel; the displayed score "
                "is the signal-day model composite decision score before final "
                "execution sizing/bridging, so it does not need to be monotonic "
                "with final weight."
            )
            score_note = (
                "Displayed score is the model composite decision score before final "
                "execution sizing/bridging from the promoted production full-fit live panel."
            )
        else:
            weight_generation_note = (
                "Current live weights come from the selected active "
                "execution-aligned live target-weight panel; the displayed score "
                "is the signal-day model composite decision score before final "
                "execution sizing/bridging, so it does not need to be monotonic "
                "with final weight."
            )
            score_note = (
                "Displayed score is the model composite decision score before final "
                "execution sizing/bridging from the selected active live panel."
            )
        live_mode = "execution_aligned_live"
    else:
        weight_generation_note = str(
            static_fallback_meta.get("target_weight_cap_note", "")
            or "Current live weights come from the promoted production live panel."
        ).strip()
        score_note = "Displayed score is the pre-weight score associated with the current live target-weight panel."
        live_mode = "raw_live_panel"

    return {
        "score_panel_role": "execution_preweight_score_panel",
        "score_reference_metadata_json": str(score_reference_path.resolve()) if score_reference_path.exists() else "",
        "effective_live_target_weight_mode": live_mode,
        "effective_live_execution_profile": str(execution_policy_label or "").strip(),
        "effective_live_execution_profile_description": profile_description,
        "effective_live_execution_bridge_meta": effective_bridge_meta,
        "effective_live_score_note": score_note,
        "effective_live_weight_generation_note": weight_generation_note,
    }


def build_active_strategy_manifest(
    *,
    source_run_dir: Path,
    production_root: Path,
    source_metrics: dict[str, Any] | None = None,
    strategy_metrics: dict[str, Any],
    panel_mode: str = "auto",
    strategy_name: str = "",
    promoted_at: str = "",
) -> dict[str, Any]:
    source_metrics = source_metrics if isinstance(source_metrics, dict) else {}
    resolved_panel_mode = (
        resolve_primary_panel_mode(strategy_metrics)
        if str(panel_mode or "auto").strip().lower() == "auto"
        else str(panel_mode).strip().lower()
    )
    target_weight_name, score_name = resolve_panel_filenames(resolved_panel_mode)
    execution_profile = str(strategy_metrics.get("execution_alignment_profile", "") or "").strip()
    execution_profile_spec = (
        strategy_metrics.get("execution_alignment_selected_profile_spec")
        if isinstance(strategy_metrics.get("execution_alignment_selected_profile_spec"), dict)
        else {}
    )
    if not execution_profile_spec and isinstance(source_metrics.get("execution_alignment_selected_profile_spec"), dict):
        execution_profile_spec = dict(source_metrics.get("execution_alignment_selected_profile_spec") or {})
    if not execution_profile_spec and execution_profile:
        execution_profile_spec = dict(resolve_profile(name=execution_profile).__dict__)
    execution_policy_label = str(
        execution_profile
        or execution_profile_spec.get("name", "")
        or strategy_metrics.get("execution_policy_label", "")
        or ""
    ).strip()
    liquidity_pool_name = infer_liquidity_pool_name(strategy_metrics, source_metrics)
    liquidity_pool_size = liquidity_pool_size_from_name(liquidity_pool_name)
    candidate_label = str(strategy_metrics.get("experiment_tag", "") or source_run_dir.name).strip()
    if execution_policy_label:
        candidate_label = f"{candidate_label}__{execution_policy_label}__active"
    elif candidate_label:
        candidate_label = f"{candidate_label}__active"

    if resolved_panel_mode == "execution_aligned":
        rebalance_freq = "1d"
        rebalance_offset_mode = "single"
        rebalance_anchor_date = ""
        target_weight_top_k = 0
        target_weight_min_weight = 0.0
        target_weight_power = 1.0
        target_weight_full_invest = False
        use_market_regime_filter = False
    else:
        bridge_meta = strategy_metrics.get("execution_alignment_selected_bridge_meta")
        bridge_meta = bridge_meta if isinstance(bridge_meta, dict) else {}
        rebalance_freq = str(bridge_meta.get("rebalance_freq", "10d") or "10d")
        rebalance_offset_mode = str(bridge_meta.get("rebalance_offset_mode", "all") or "all")
        rebalance_anchor_date = str(bridge_meta.get("rebalance_anchor_date", "2025-01-02") or "2025-01-02")
        target_weight_top_k = int(bridge_meta.get("target_weight_top_k", 2) or 2)
        target_weight_min_weight = float(bridge_meta.get("target_weight_min_weight", 0.0) or 0.0)
        target_weight_power = float(bridge_meta.get("target_weight_power", 1.0) or 1.0)
        target_weight_full_invest = bool(bridge_meta.get("target_weight_full_invest", False))
        use_market_regime_filter = bool(bridge_meta.get("market_regime_filter", False))
    bridge_meta = strategy_metrics.get("execution_alignment_selected_bridge_meta")
    bridge_meta = bridge_meta if isinstance(bridge_meta, dict) else {}
    transaction_cost_bps = float(
        strategy_metrics.get("transaction_cost_bps", bridge_meta.get("transaction_cost_bps", 3.0)) or 3.0
    )
    slippage_bps = float(
        strategy_metrics.get("slippage_bps", bridge_meta.get("slippage_bps", 7.0)) or 7.0
    )
    sell_tax_bps = float(
        strategy_metrics.get("sell_tax_bps", bridge_meta.get("sell_tax_bps", 10.0)) or 10.0
    )

    source_run_dir = source_run_dir.resolve()
    production_root = production_root.resolve()
    source_panel_root = source_run_dir
    if not all((source_run_dir / name).exists() for name in (target_weight_name, score_name)):
        source_panel_root = production_root
    source_panel_origin = "formal_source" if source_panel_root == source_run_dir else "production_fallback"
    source_panel_metrics = source_metrics if source_panel_origin == "formal_source" else strategy_metrics
    raw_target_weight_name, raw_score_name = resolve_panel_filenames("raw")
    research_panel_root = production_root
    if not all((production_root / name).exists() for name in (raw_target_weight_name, raw_score_name)):
        research_panel_root = source_run_dir
    production_manifest_json = production_root / "production_retrain_manifest.json"
    has_production_manifest = production_manifest_json.exists()
    effective_live_metadata = _build_effective_live_metadata(
        production_root=production_root,
        resolved_panel_mode=resolved_panel_mode,
        execution_policy_label=execution_policy_label,
        execution_profile_spec=execution_profile_spec if isinstance(execution_profile_spec, dict) else {},
        bridge_meta=bridge_meta,
        has_production_manifest=has_production_manifest,
    )
    return {
        "strategy_name": str(strategy_name or source_run_dir.name),
        "candidate_label": candidate_label,
        "source_run_dir": str(source_run_dir),
        "source_formal_run_dir": str(source_run_dir),
        "source_refresh_run_dir": str(source_panel_root),
        "source_panel_origin": source_panel_origin,
        "production_root": str(production_root),
        "trade_plan_refresh_run_dir": str(production_root),
        "production_manifest_json": str(production_manifest_json.resolve()) if has_production_manifest else "",
        "panel_mode": resolved_panel_mode,
        "source_target_weight_panel_csv": str((source_panel_root / target_weight_name).resolve()),
        "source_score_panel_csv": str((source_panel_root / score_name).resolve()),
        "trade_plan_target_weight_panel_csv": str((production_root / target_weight_name).resolve()),
        "trade_plan_score_panel_csv": str((production_root / score_name).resolve()),
        "research_candidate_target_weight_panel_csv": str((research_panel_root / raw_target_weight_name).resolve()),
        "research_candidate_score_panel_csv": str((research_panel_root / raw_score_name).resolve()),
        "trade_plan_candidate_label": candidate_label,
        "target_weight_semantics": "research_raw_target_weight",
        "target_weight_cap_mode": "follow_research_raw_no_global_cap",
        "target_weight_cap_note": (
            "Execution follows the research target-weight panel directly. "
            "Do not impose the generic max_weight cap on the external target-weight path."
        ),
        "benchmark": str(source_panel_metrics.get("benchmark", "000300.SH") or "000300.SH"),
        "data_source": "tq",
        "liquidity_pool_name": liquidity_pool_name,
        "liquidity_pool_size": int(liquidity_pool_size),
        "liquidity_pool": str(strategy_metrics.get("liquidity_pool", "") or source_metrics.get("liquidity_pool", "") or ""),
        "rolling_liquidity_pool": str(strategy_metrics.get("rolling_liquidity_pool", "") or source_metrics.get("rolling_liquidity_pool", "") or ""),
        "rolling_pool_rebalance_days": int(strategy_metrics.get("rolling_pool_rebalance_days", source_metrics.get("rolling_pool_rebalance_days", 0)) or 0),
        "rolling_pool_adv_window": int(strategy_metrics.get("rolling_pool_adv_window", source_metrics.get("rolling_pool_adv_window", 0)) or 0),
        "backtest_start_date": str(source_panel_metrics.get("valid_start", "20210101") or "20210101").replace("-", ""),
        "trade_plan_start_date": str(strategy_metrics.get("start_date", "20210101") or "20210101"),
        "transaction_cost_bps": float(transaction_cost_bps),
        "slippage_bps": float(slippage_bps),
        "sell_tax_bps": float(sell_tax_bps),
        "rebalance_freq": rebalance_freq,
        "rebalance_offset_mode": rebalance_offset_mode,
        "rebalance_anchor_date": rebalance_anchor_date,
        "target_weight_top_k": int(target_weight_top_k),
        "target_weight_min_weight": float(target_weight_min_weight),
        "target_weight_power": float(target_weight_power),
        "target_weight_full_invest": bool(target_weight_full_invest),
        "use_market_regime_filter": bool(use_market_regime_filter),
        "execution_alignment_mode": str(strategy_metrics.get("execution_alignment_mode", "") or ""),
        "execution_alignment_objective": str(strategy_metrics.get("execution_alignment_objective", "") or ""),
        "execution_alignment_profile": execution_profile,
        "score_panel_role": str(effective_live_metadata.get("score_panel_role", "") or ""),
        "score_reference_metadata_json": str(effective_live_metadata.get("score_reference_metadata_json", "") or ""),
        "execution_policy_label": execution_policy_label,
        "execution_alignment_selected_profile_spec": execution_profile_spec,
        "execution_alignment_selected_bridge_meta": (
            strategy_metrics.get("execution_alignment_selected_bridge_meta")
            if isinstance(strategy_metrics.get("execution_alignment_selected_bridge_meta"), dict)
            else (
                source_metrics.get("execution_alignment_selected_bridge_meta")
                if isinstance(source_metrics.get("execution_alignment_selected_bridge_meta"), dict)
                else {}
            )
        ),
        "effective_live_target_weight_mode": str(
            effective_live_metadata.get("effective_live_target_weight_mode", "") or ""
        ),
        "effective_live_execution_profile": str(
            effective_live_metadata.get("effective_live_execution_profile", "") or ""
        ),
        "effective_live_execution_profile_description": str(
            effective_live_metadata.get("effective_live_execution_profile_description", "") or ""
        ),
        "effective_live_execution_bridge_meta": (
            effective_live_metadata.get("effective_live_execution_bridge_meta")
            if isinstance(effective_live_metadata.get("effective_live_execution_bridge_meta"), dict)
            else {}
        ),
        "effective_live_score_note": str(effective_live_metadata.get("effective_live_score_note", "") or ""),
        "effective_live_weight_generation_note": str(
            effective_live_metadata.get("effective_live_weight_generation_note", "") or ""
        ),
        "primary_research_backtest_label": str(strategy_metrics.get("primary_research_backtest_label", "") or ""),
        "research_objective_mode": str(strategy_metrics.get("research_objective_mode", "") or ""),
        "research_time_unit": str(strategy_metrics.get("research_time_unit", "") or ""),
        "valid_months": int(strategy_metrics.get("valid_months", 0) or 0),
        "train_eval_window_months": int(strategy_metrics.get("train_eval_window_months", 0) or 0),
        "adaptive_task_window_months": int(strategy_metrics.get("adaptive_task_window_months", 0) or 0),
        "promoted_at": str(promoted_at or ""),
    }
