from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.execution.output_root_resolver import (
    OUTPUT_ROOT,
    resolve_single_mapping_pipeline_root,
)
from daily_research.execution.strategy_manifest import load_strategy_manifest, write_strategy_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ACTIVE_MANIFEST = OUTPUT_ROOT / "active_execution_strategy.json"
DEFAULT_TRANSACTION_COST_BPS = 3.0
DEFAULT_SLIPPAGE_BPS = 7.0
DEFAULT_SELL_TAX_BPS = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote the short-alpha single-mapping execution candidate into the active execution strategy manifest."
    )
    parser.add_argument("--pipeline-root", default="")
    parser.add_argument("--active-manifest", default=str(DEFAULT_ACTIVE_MANIFEST))
    parser.add_argument("--python-executable", default=sys.executable)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline_root = resolve_single_mapping_pipeline_root(args.pipeline_root)
    active_manifest_path = Path(args.active_manifest).resolve()
    if not pipeline_root.exists():
        raise FileNotFoundError(f"Pipeline root does not exist: {pipeline_root}")

    live_panel = pipeline_root / "daily_live_target_weight_panel.csv"
    if not live_panel.exists():
        raise FileNotFoundError(f"Candidate live target-weight panel missing: {live_panel}")
    live_score_panel = pipeline_root / "daily_live_score_panel.csv"
    if not live_score_panel.exists():
        raise FileNotFoundError(f"Candidate live score panel missing: {live_score_panel}")
    live_score_reference = pipeline_root / "daily_live_score_reference.json"

    base_manifest = load_strategy_manifest(active_manifest_path)
    if not base_manifest:
        raise RuntimeError(f"Base active manifest is empty or missing: {active_manifest_path}")
    production_manifest_path = Path(str(base_manifest.get("production_manifest_json", "")).strip()).resolve() if str(base_manifest.get("production_manifest_json", "")).strip() else Path()
    production_manifest: dict[str, object] = {}
    if production_manifest_path and production_manifest_path.exists():
        payload = json.loads(production_manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            production_manifest = payload

    live_monitor = json.loads((pipeline_root / "live_trigger_monitor.json").read_text(encoding="utf-8"))
    mapping_label = str(live_monitor.get("mapping_label", "")).strip() or "single_mapping_candidate"
    summary_payload = json.loads((pipeline_root / "formal_trigger_tradeoff_summary.json").read_text(encoding="utf-8"))
    score_reference_payload: dict[str, object] = {}
    if live_score_reference.exists():
        payload = json.loads(live_score_reference.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            score_reference_payload = payload
    resolved_score_panel_role = str(score_reference_payload.get("role", "")).strip() or "execution_preweight_score_panel"
    effective_live_bridge_meta = (
        dict(score_reference_payload.get("effective_execution_bridge_meta", {}))
        if isinstance(score_reference_payload.get("effective_execution_bridge_meta"), dict)
        else {}
    )

    new_payload = dict(base_manifest)
    candidate_label = "state_liquidity_listwise_v1__single_mapping_candidate_active"
    transaction_cost_bps = float(base_manifest.get("transaction_cost_bps", DEFAULT_TRANSACTION_COST_BPS) or DEFAULT_TRANSACTION_COST_BPS)
    slippage_bps = float(base_manifest.get("slippage_bps", DEFAULT_SLIPPAGE_BPS) or DEFAULT_SLIPPAGE_BPS)
    sell_tax_bps = float(base_manifest.get("sell_tax_bps", DEFAULT_SELL_TAX_BPS) or DEFAULT_SELL_TAX_BPS)
    new_payload.update(
        {
            "strategy_name": "state_liquidity_listwise_v1_execfirst_single_mapping_candidate_active",
            "candidate_label": candidate_label,
            "trade_plan_candidate_label": candidate_label,
            "source_run_dir": str(production_manifest.get("source_formal_run_dir", base_manifest.get("source_run_dir", ""))),
            "source_formal_run_dir": str(production_manifest.get("source_formal_run_dir", base_manifest.get("source_formal_run_dir", ""))),
            "source_panel_origin": "execution_single_mapping_candidate_pipeline",
            "source_refresh_run_dir": str(pipeline_root),
            "production_root": str(pipeline_root),
            "trade_plan_refresh_run_dir": str(pipeline_root),
            "source_target_weight_panel_csv": str(live_panel),
            "source_score_panel_csv": str(live_score_panel),
            "trade_plan_target_weight_panel_csv": str(live_panel),
            "trade_plan_score_panel_csv": str(live_score_panel),
            "target_weight_semantics": "research_raw_target_weight",
            "target_weight_cap_mode": "follow_research_raw_no_global_cap",
            "target_weight_cap_note": (
                "Execution follows the research target-weight panel directly. "
                "Do not impose the generic max_weight cap on the external target-weight path."
            ),
            "transaction_cost_bps": float(transaction_cost_bps),
            "slippage_bps": float(slippage_bps),
            "sell_tax_bps": float(sell_tax_bps),
            "rebalance_freq": "1d",
            "rebalance_offset_mode": "single",
            "rebalance_anchor_date": "",
            "target_weight_top_k": 0,
            "target_weight_min_weight": 0.0,
            "target_weight_power": 1.0,
            "target_weight_full_invest": False,
            "use_market_regime_filter": False,
            "execution_alignment_mode": "custom_mapping_panel",
            "execution_alignment_objective": "single_mapping_candidate_pipeline",
            "execution_alignment_profile": "",
            "score_panel_role": resolved_score_panel_role,
            "score_reference_metadata_json": str(live_score_reference),
            "execution_policy_label": mapping_label,
            "execution_alignment_selected_profile_spec": {
                "name": "single_mapping_candidate_pipeline",
                "mapping_label": mapping_label,
                "mapping": summary_payload.get("mapping", {}),
                "trigger_mode": summary_payload.get("trigger_mode", ""),
            },
            "execution_alignment_selected_bridge_meta": {
                "mapping_label": mapping_label,
                "mapping": summary_payload.get("mapping", {}),
                "trigger_mode": summary_payload.get("trigger_mode", ""),
                "month_count": summary_payload.get("month_count", 0),
                "triggered_month_count": summary_payload.get("triggered_month_count", 0),
                "trigger_coverage_ratio": summary_payload.get("trigger_coverage_ratio", 0.0),
            },
            "effective_live_target_weight_mode": str(live_monitor.get("live_target_weight_mode", "")),
            "effective_live_candidate_active": bool(live_monitor.get("candidate_active_now", False)),
            "effective_live_month": str(live_monitor.get("latest_month", "")),
            "effective_live_trigger_key": str(live_monitor.get("latest_trigger_key", "")),
            "effective_live_execution_profile": str(
                score_reference_payload.get("effective_execution_profile", "")
                or live_monitor.get("effective_execution_profile", "")
                or live_monitor.get("latest_selected_profile", "")
            ),
            "effective_live_execution_profile_description": str(
                score_reference_payload.get("effective_execution_profile_description", "")
                or live_monitor.get("effective_execution_profile_description", "")
            ),
            "effective_live_execution_bridge_meta": effective_live_bridge_meta,
            "effective_live_score_note": str(score_reference_payload.get("note", "")),
            "effective_live_weight_generation_note": str(score_reference_payload.get("weight_generation_note", "")),
            "trade_plan_refresh_command": [
                str(args.python_executable),
                str((PROJECT_ROOT / "daily_research" / "execution" / "run_short_alpha_execution_single_mapping_candidate_pipeline.py").resolve()),
                "--live-only",
                "--python-executable",
                str(args.python_executable),
                "--output-root",
                str(OUTPUT_ROOT),
                "--root-tag",
                pipeline_root.name,
            ],
            "promoted_at": datetime.now().isoformat(),
        }
    )
    write_strategy_manifest(new_payload, path=active_manifest_path)
    written_manifest = load_strategy_manifest(active_manifest_path)
    if str(written_manifest.get("score_panel_role", "")).strip() != resolved_score_panel_role:
        raise RuntimeError(
            "Active execution manifest score_panel_role did not persist after activation: "
            f"expected={resolved_score_panel_role} got={written_manifest.get('score_panel_role', '')}"
        )
    print(f"active_execution_strategy_manifest={active_manifest_path}")
    print(f"active_execution_strategy_name={new_payload.get('strategy_name', '')}")


if __name__ == "__main__":
    main()
