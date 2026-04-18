from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.continuous_policy.label_builder import LABEL_CONFIGS, build_future_path_metrics
from daily_research.continuous_policy.model import load_artifact
from daily_research.continuous_policy.pipeline_utils import (
    BUDGET_OBJECTIVE_CHOICES,
    DEFAULT_BUDGET_OBJECTIVE,
    evaluate_reference_panel,
    load_target_weight_panel,
    run_policy_rollout,
)
from daily_research.continuous_policy.portfolio_simulator import (
    BUDGET_CALIBRATION_CHOICES,
    BUDGET_SEMANTICS_CHOICES,
    DEFAULT_BUDGET_CALIBRATION,
    DEFAULT_BUDGET_SEMANTICS,
    DEFAULT_EXECUTION_SEMANTICS,
    EXECUTION_SEMANTICS_CHOICES,
)
from daily_research.continuous_policy.runtime import (
    EVALUATIONS_ROOT,
    now_iso,
    resolve_latest_model_artifact,
    timestamp_tag,
    update_latest_summary,
    write_json,
)
from daily_research.continuous_policy.state_builder import DEFAULT_ALPHA_PRIOR_SOURCE, prepare_policy_inputs, resolve_active_policy_defaults
from daily_research.execution.strategy_manifest import load_strategy_manifest


def _parse_reference_panel_specs(values: list[str] | None) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for raw_value in values or []:
        text = str(raw_value or "").strip()
        if not text:
            continue
        label = ""
        path_text = text
        if "=" in text:
            label, path_text = text.split("=", 1)
        path = Path(path_text.strip()).expanduser().resolve()
        specs.append(
            {
                "label": str(label or path.stem or "reference").strip(),
                "panel_path": str(path),
            }
        )
    return specs


def build_parser() -> argparse.ArgumentParser:
    defaults = resolve_active_policy_defaults()
    parser = argparse.ArgumentParser(description="Evaluate the continuous portfolio policy shadow stack.")
    parser.add_argument("--model-path", default="")
    parser.add_argument(
        "--pool-name",
        default=defaults["pool_name"] or "liquid500",
        help="Rolling liquidity pool name, or `all_a` / `learned_all_a` to evaluate whole-A learned selection.",
    )
    parser.add_argument("--start-date", default=defaults["start_date"] or "20250318")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--benchmark", default=defaults["benchmark"] or "000300.SH")
    parser.add_argument("--data-source", default="tq", choices=("tq", "csv"))
    parser.add_argument("--csv-folder", default="")
    parser.add_argument("--pool-rebalance-days", type=int, default=21)
    parser.add_argument("--pool-adv-window", type=int, default=20)
    parser.add_argument("--max-universe-size", type=int, default=0)
    parser.add_argument("--transaction-cost-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=7.0)
    parser.add_argument("--sell-tax-bps", type=float, default=10.0)
    parser.add_argument(
        "--execution-semantics",
        default=DEFAULT_EXECUTION_SEMANTICS,
        choices=EXECUTION_SEMANTICS_CHOICES,
        help="Use semantic_preserving_v1 to keep lifecycle intent separate from weight-change orders.",
    )
    parser.add_argument(
        "--budget-semantics",
        default=DEFAULT_BUDGET_SEMANTICS,
        choices=BUDGET_SEMANTICS_CHOICES,
        help="How candidate budget is applied during rollout.",
    )
    parser.add_argument(
        "--budget-calibration",
        default=DEFAULT_BUDGET_CALIBRATION,
        choices=BUDGET_CALIBRATION_CHOICES,
        help="Optional portfolio-level gross/candidate/turnover calibration.",
    )
    parser.add_argument(
        "--budget-objective",
        default="",
        choices=("", *BUDGET_OBJECTIVE_CHOICES),
        help="Teacher-oracle budget objective override. Empty uses the model artifact setting.",
    )
    parser.add_argument("--alpha-prior-source", default="", help="Empty uses the model artifact alpha-prior source.")
    parser.add_argument("--alpha-prior-score-panel", default="")
    parser.add_argument("--alpha-prior-target-weight-panel", default="")
    parser.add_argument(
        "--label-preset",
        default="",
        choices=("", *tuple(sorted(LABEL_CONFIGS))),
        help="Optional teacher lifecycle label preset override. Defaults to the preset recorded in the model artifact.",
    )
    parser.add_argument(
        "--reference-panel",
        action="append",
        default=[],
        help="Repeatable extra reference panel in the form `label=path/to/panel.csv`.",
    )
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--tag", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact_path = resolve_latest_model_artifact(args.model_path)
    artifact = load_artifact(artifact_path)
    label_preset = str(
        args.label_preset
        or artifact.train_summary.get("label_preset")
        or artifact.train_summary.get("teacher_summary", {}).get("label_preset")
        or "balanced_v2"
    ).strip()
    alpha_prior_source = str(
        args.alpha_prior_source
        or artifact.train_summary.get("alpha_prior_source")
        or artifact.train_summary.get("prepared_summary", {}).get("alpha_prior_summary", {}).get("source")
        or DEFAULT_ALPHA_PRIOR_SOURCE
    ).strip()
    alpha_prior_score_panel = str(
        args.alpha_prior_score_panel
        or artifact.train_summary.get("alpha_prior_score_panel")
        or artifact.train_summary.get("prepared_summary", {}).get("alpha_prior_summary", {}).get("score_panel_csv")
        or ""
    ).strip()
    alpha_prior_target_weight_panel = str(
        args.alpha_prior_target_weight_panel
        or artifact.train_summary.get("alpha_prior_target_weight_panel")
        or artifact.train_summary.get("prepared_summary", {}).get("alpha_prior_summary", {}).get("target_weight_panel_csv")
        or ""
    ).strip()
    budget_objective = str(
        args.budget_objective
        or artifact.train_summary.get("budget_objective")
        or artifact.train_summary.get("teacher_summary", {}).get("budget_objective")
        or DEFAULT_BUDGET_OBJECTIVE
    ).strip()
    run_tag = str(args.tag or timestamp_tag("evaluation"))
    run_root = EVALUATIONS_ROOT / run_tag
    run_root.mkdir(parents=True, exist_ok=True)

    prepared = prepare_policy_inputs(
        pool_name=args.pool_name,
        start_date=args.start_date,
        end_date=args.end_date,
        benchmark=args.benchmark,
        data_source=args.data_source,
        csv_folder=args.csv_folder,
        max_universe_size=args.max_universe_size,
        pool_rebalance_days=args.pool_rebalance_days,
        pool_adv_window=args.pool_adv_window,
        alpha_prior_source=alpha_prior_source,
        alpha_prior_score_panel=alpha_prior_score_panel,
        alpha_prior_target_weight_panel=alpha_prior_target_weight_panel,
        refresh_cache=args.refresh_cache,
        progress_desc="continuous policy evaluate",
    )
    future_metrics = build_future_path_metrics(prepared)
    model_rollout = run_policy_rollout(
        prepared=prepared,
        artifact=artifact,
        start_date=args.start_date,
        end_date=args.end_date,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        sell_tax_bps=args.sell_tax_bps,
        source_label="continuous_policy",
        execution_semantics=args.execution_semantics,
        budget_semantics=args.budget_semantics,
        budget_calibration=args.budget_calibration,
        budget_objective=budget_objective,
    )
    teacher_rollout = run_policy_rollout(
        prepared=prepared,
        artifact=None,
        future_metrics=future_metrics,
        start_date=args.start_date,
        end_date=args.end_date,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        sell_tax_bps=args.sell_tax_bps,
        source_label="teacher_oracle",
        label_preset=label_preset,
        execution_semantics=args.execution_semantics,
        budget_semantics=args.budget_semantics,
        budget_calibration=args.budget_calibration,
        budget_objective=budget_objective,
    )

    manifest = load_strategy_manifest()
    reference_payload: dict[str, object] = {}
    reference_payloads: list[dict[str, object]] = []
    reference_panel_path = str(manifest.get("trade_plan_target_weight_panel_csv", "") or manifest.get("source_target_weight_panel_csv", "") or "").strip()
    if reference_panel_path:
        try:
            panel_frame = load_target_weight_panel(reference_panel_path)
            reference_result = evaluate_reference_panel(
                prepared=prepared,
                panel_frame=panel_frame,
                start_date=args.start_date,
                end_date=args.end_date,
                transaction_cost_bps=args.transaction_cost_bps,
                slippage_bps=args.slippage_bps,
                sell_tax_bps=args.sell_tax_bps,
            )
            reference_payload = {
                "label": "active_manifest_reference",
                "panel_path": reference_panel_path,
                "metrics": reference_result["metrics"],
            }
        except Exception as exc:
            reference_payload = {"label": "active_manifest_reference", "panel_path": reference_panel_path, "error": str(exc)}
        reference_payloads.append(reference_payload)

    for extra_spec in _parse_reference_panel_specs(args.reference_panel):
        try:
            panel_frame = load_target_weight_panel(extra_spec["panel_path"])
            reference_result = evaluate_reference_panel(
                prepared=prepared,
                panel_frame=panel_frame,
                start_date=args.start_date,
                end_date=args.end_date,
                transaction_cost_bps=args.transaction_cost_bps,
                slippage_bps=args.slippage_bps,
                sell_tax_bps=args.sell_tax_bps,
            )
            reference_payloads.append(
                {
                    "label": extra_spec["label"],
                    "panel_path": extra_spec["panel_path"],
                    "metrics": reference_result["metrics"],
                }
            )
        except Exception as exc:
            reference_payloads.append(
                {
                    "label": extra_spec["label"],
                    "panel_path": extra_spec["panel_path"],
                    "error": str(exc),
                }
            )

    model_rollout["action_panel"].to_csv(run_root / "daily_action_panel.csv", index=False, encoding="utf-8-sig")
    model_rollout["action_outcomes"].to_csv(run_root / "daily_action_outcomes.csv", index=False, encoding="utf-8-sig")
    model_rollout["turnover_frame"].to_csv(run_root / "daily_turnover.csv", index=False, encoding="utf-8-sig")
    model_rollout["position_history"].to_csv(run_root / "daily_position_history.csv", index=False, encoding="utf-8-sig")
    model_rollout["returns"].rename("daily_return").to_csv(run_root / "daily_returns.csv", encoding="utf-8-sig")

    summary_payload = {
        "run_tag": run_tag,
        "evaluated_at": now_iso(),
        "model_artifact_path": str(artifact_path.resolve()),
        "trainer_backend": str(
            artifact.train_summary.get("trainer_backend", "")
            or getattr(artifact, "training_contract", {}).get("trainer_backend", "")
            or ""
        ),
        "decoder_profile": str(artifact.train_summary.get("decoder_profile", "") or "default_v2"),
        "training_contract": dict(artifact.train_summary.get("training_contract", {}) or getattr(artifact, "training_contract", {}) or {}),
        "training_diagnostics": dict(artifact.train_summary.get("training_diagnostics", {}) or getattr(artifact, "training_diagnostics", {}) or {}),
        "daily_head_layout": str(artifact.train_summary.get("daily_head_layout", "") or ""),
        "pool_name": prepared.pool_name,
        "benchmark": prepared.benchmark,
        "start_date": args.start_date,
        "end_date": args.end_date or prepared.end_date,
        "label_preset": label_preset,
        "execution_semantics": str(args.execution_semantics),
        "budget_semantics": str(args.budget_semantics),
        "budget_calibration": str(args.budget_calibration),
        "budget_objective": budget_objective,
        "alpha_prior_source": alpha_prior_source,
        "alpha_prior_score_panel": alpha_prior_score_panel,
        "alpha_prior_target_weight_panel": alpha_prior_target_weight_panel,
        "prepared_summary": prepared.to_summary(),
        "continuous_policy_metrics": model_rollout["metrics"],
        "continuity_metrics": model_rollout["continuity_metrics"],
        "teacher_oracle_metrics": teacher_rollout["metrics"],
        "active_manifest_reference": reference_payload,
        "reference_panels": reference_payloads,
        "action_panel_csv": str((run_root / "daily_action_panel.csv").resolve()),
        "action_outcomes_csv": str((run_root / "daily_action_outcomes.csv").resolve()),
        "turnover_csv": str((run_root / "daily_turnover.csv").resolve()),
        "position_history_csv": str((run_root / "daily_position_history.csv").resolve()),
        "returns_csv": str((run_root / "daily_returns.csv").resolve()),
    }
    write_json(run_root / "evaluation_summary.json", summary_payload)
    update_latest_summary("evaluation", summary_payload)
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
