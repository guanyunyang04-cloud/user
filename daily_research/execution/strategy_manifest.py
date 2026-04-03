from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daily_research.deep_alpha.research_objective import resolve_primary_panel_mode


DEFAULT_ACTIVE_EXECUTION_STRATEGY_MANIFEST = Path(
    "daily_research/output/active_execution_strategy.json"
).resolve()


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


def build_active_strategy_manifest(
    *,
    source_run_dir: Path,
    production_root: Path,
    strategy_metrics: dict[str, Any],
    panel_mode: str = "auto",
    strategy_name: str = "",
    promoted_at: str = "",
) -> dict[str, Any]:
    resolved_panel_mode = (
        resolve_primary_panel_mode(strategy_metrics)
        if str(panel_mode or "auto").strip().lower() == "auto"
        else str(panel_mode).strip().lower()
    )
    target_weight_name, score_name = resolve_panel_filenames(resolved_panel_mode)
    execution_profile = str(strategy_metrics.get("execution_alignment_profile", "") or "").strip()
    candidate_label = str(strategy_metrics.get("experiment_tag", "") or source_run_dir.name).strip()
    if resolved_panel_mode == "execution_aligned" and execution_profile:
        candidate_label = f"{candidate_label}__{execution_profile}__active"
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

    source_run_dir = source_run_dir.resolve()
    production_root = production_root.resolve()
    source_panel_root = source_run_dir
    if not all((source_run_dir / name).exists() for name in (target_weight_name, score_name)):
        source_panel_root = production_root
    source_panel_origin = "formal_source" if source_panel_root == source_run_dir else "production_fallback"
    production_manifest_json = production_root / "production_retrain_manifest.json"
    return {
        "strategy_name": str(strategy_name or source_run_dir.name),
        "candidate_label": candidate_label,
        "source_run_dir": str(source_run_dir),
        "source_formal_run_dir": str(source_run_dir),
        "source_refresh_run_dir": str(source_panel_root),
        "source_panel_origin": source_panel_origin,
        "production_root": str(production_root),
        "trade_plan_refresh_run_dir": str(production_root),
        "production_manifest_json": str(production_manifest_json.resolve()),
        "panel_mode": resolved_panel_mode,
        "source_target_weight_panel_csv": str((source_panel_root / target_weight_name).resolve()),
        "source_score_panel_csv": str((source_panel_root / score_name).resolve()),
        "trade_plan_target_weight_panel_csv": str((production_root / target_weight_name).resolve()),
        "trade_plan_score_panel_csv": str((production_root / score_name).resolve()),
        "trade_plan_candidate_label": candidate_label,
        "benchmark": str(strategy_metrics.get("benchmark", "000300.SH") or "000300.SH"),
        "data_source": "tq",
        "backtest_start_date": str(strategy_metrics.get("valid_start", "20210101") or "20210101").replace("-", ""),
        "trade_plan_start_date": str(strategy_metrics.get("start_date", "20210101") or "20210101"),
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
        "primary_research_backtest_label": str(strategy_metrics.get("primary_research_backtest_label", "") or ""),
        "research_objective_mode": str(strategy_metrics.get("research_objective_mode", "") or ""),
        "promoted_at": str(promoted_at or ""),
    }
