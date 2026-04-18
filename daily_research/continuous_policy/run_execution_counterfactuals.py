from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.continuous_policy.model import load_artifact
from daily_research.continuous_policy.pipeline_utils import run_policy_rollout
from daily_research.continuous_policy.portfolio_simulator import (
    BUDGET_CALIBRATION_CHOICES,
    BUDGET_SEMANTICS_CHOICES,
    DEFAULT_BUDGET_CALIBRATION,
    DEFAULT_BUDGET_SEMANTICS,
    DEFAULT_EXECUTION_SEMANTICS,
    EXECUTION_SEMANTICS_CHOICES,
)
from daily_research.continuous_policy.runtime import (
    CONTINUOUS_POLICY_ROOT,
    now_iso,
    resolve_latest_model_artifact,
    timestamp_tag,
    write_json,
)
from daily_research.continuous_policy.state_builder import (
    DEFAULT_ALPHA_PRIOR_SOURCE,
    prepare_policy_inputs,
    resolve_active_policy_defaults,
)


def _artifact_alpha_prior_args(args: argparse.Namespace, artifact: Any) -> dict[str, str]:
    train_summary = dict(getattr(artifact, "train_summary", {}) or {})
    prepared_summary = dict(train_summary.get("prepared_summary", {}) or {})
    alpha_summary = dict(prepared_summary.get("alpha_prior_summary", {}) or {})
    return {
        "alpha_prior_source": str(
            args.alpha_prior_source
            or train_summary.get("alpha_prior_source")
            or alpha_summary.get("source")
            or DEFAULT_ALPHA_PRIOR_SOURCE
        ).strip(),
        "alpha_prior_score_panel": str(
            args.alpha_prior_score_panel
            or train_summary.get("alpha_prior_score_panel")
            or alpha_summary.get("score_panel_csv")
            or ""
        ).strip(),
        "alpha_prior_target_weight_panel": str(
            args.alpha_prior_target_weight_panel
            or train_summary.get("alpha_prior_target_weight_panel")
            or alpha_summary.get("target_weight_panel_csv")
            or ""
        ).strip(),
    }


def _parse_variant(raw_value: str) -> dict[str, str]:
    text = str(raw_value or "").strip()
    if not text:
        raise ValueError("Empty counterfactual variant.")
    if "=" in text:
        label, spec = text.split("=", 1)
    else:
        label, spec = "", text
    parts = [part.strip() for part in spec.split(":")]
    if len(parts) == 2:
        execution_semantics = DEFAULT_EXECUTION_SEMANTICS
        budget_semantics, budget_calibration = parts
    elif len(parts) == 3:
        execution_semantics, budget_semantics, budget_calibration = parts
    else:
        raise ValueError(
            "Variant must be `label=budget_semantics:budget_calibration` "
            "or `label=execution_semantics:budget_semantics:budget_calibration`."
        )
    label = label or "__".join([execution_semantics, budget_semantics, budget_calibration])
    return {
        "label": label,
        "execution_semantics": execution_semantics,
        "budget_semantics": budget_semantics,
        "budget_calibration": budget_calibration,
    }


def _default_variants(args: argparse.Namespace) -> list[dict[str, str]]:
    return [
        {
            "label": "requested_default",
            "execution_semantics": str(args.execution_semantics),
            "budget_semantics": str(args.budget_semantics),
            "budget_calibration": str(args.budget_calibration),
        },
        {
            "label": "legacy_total_candidate__none",
            "execution_semantics": DEFAULT_EXECUTION_SEMANTICS,
            "budget_semantics": "legacy_total_candidate",
            "budget_calibration": "none",
        },
        {
            "label": "action_budget_split_v1__none",
            "execution_semantics": DEFAULT_EXECUTION_SEMANTICS,
            "budget_semantics": "action_budget_split_v1",
            "budget_calibration": "none",
        },
        {
            "label": "action_budget_split_v1__cash_exit_guard_v1",
            "execution_semantics": DEFAULT_EXECUTION_SEMANTICS,
            "budget_semantics": "action_budget_split_v1",
            "budget_calibration": "cash_exit_guard_v1",
        },
    ]


def _metric_row(label: str, variant: dict[str, str], rollout: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(rollout.get("metrics", {}) or {})
    continuity = dict(rollout.get("continuity_metrics", {}) or {})
    keys = (
        "total_return",
        "annual_return",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "avg_turnover",
        "avg_gross_exposure",
        "avg_holding_count",
        "avg_semantic_conflict_rate",
        "avg_order_translation_conflict_rate",
        "cash_timing_quality_1d",
        "reduce_success_rate_5d",
        "exit_timeliness_rate_5d",
        "trend_capture_rate_10d",
        "missed_main_leg_rate_10d",
        "immediate_reversal_rate_3d",
    )
    merged = {**metrics, **continuity}
    return {
        "variant": label,
        "execution_semantics": variant["execution_semantics"],
        "budget_semantics": variant["budget_semantics"],
        "budget_calibration": variant["budget_calibration"],
        **{key: merged.get(key, 0.0) for key in keys},
    }


def build_parser() -> argparse.ArgumentParser:
    defaults = resolve_active_policy_defaults()
    parser = argparse.ArgumentParser(description="Run execution-layer counterfactual rollouts for a fixed continuous-policy model.")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--pool-name", default=defaults["pool_name"] or "liquid500")
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
    parser.add_argument("--execution-semantics", default=DEFAULT_EXECUTION_SEMANTICS, choices=EXECUTION_SEMANTICS_CHOICES)
    parser.add_argument("--budget-semantics", default=DEFAULT_BUDGET_SEMANTICS, choices=BUDGET_SEMANTICS_CHOICES)
    parser.add_argument("--budget-calibration", default=DEFAULT_BUDGET_CALIBRATION, choices=BUDGET_CALIBRATION_CHOICES)
    parser.add_argument("--alpha-prior-source", default="")
    parser.add_argument("--alpha-prior-score-panel", default="")
    parser.add_argument("--alpha-prior-target-weight-panel", default="")
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Repeatable `label=budget_semantics:budget_calibration` or `label=execution_semantics:budget_semantics:budget_calibration`.",
    )
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--tag", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact_path = resolve_latest_model_artifact(args.model_path)
    artifact = load_artifact(artifact_path)
    alpha_args = _artifact_alpha_prior_args(args, artifact)
    run_tag = str(args.tag or timestamp_tag("execution_counterfactuals"))
    run_root = CONTINUOUS_POLICY_ROOT / "analysis" / "counterfactuals" / run_tag
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
        refresh_cache=args.refresh_cache,
        progress_desc="continuous policy execution counterfactuals",
        **alpha_args,
    )
    variants = [_parse_variant(item) for item in args.variant] if args.variant else _default_variants(args)
    seen: set[tuple[str, str, str]] = set()
    deduped_variants: list[dict[str, str]] = []
    for variant in variants:
        key = (variant["execution_semantics"], variant["budget_semantics"], variant["budget_calibration"])
        if key in seen:
            continue
        seen.add(key)
        deduped_variants.append(variant)

    rows: list[dict[str, Any]] = []
    variant_payloads: list[dict[str, Any]] = []
    for variant in deduped_variants:
        rollout = run_policy_rollout(
            prepared=prepared,
            artifact=artifact,
            start_date=args.start_date,
            end_date=args.end_date,
            transaction_cost_bps=args.transaction_cost_bps,
            slippage_bps=args.slippage_bps,
            sell_tax_bps=args.sell_tax_bps,
            source_label=f"counterfactual_{variant['label']}",
            execution_semantics=variant["execution_semantics"],
            budget_semantics=variant["budget_semantics"],
            budget_calibration=variant["budget_calibration"],
        )
        variant_dir = run_root / str(variant["label"])
        variant_dir.mkdir(parents=True, exist_ok=True)
        rollout["action_panel"].to_csv(variant_dir / "daily_action_panel.csv", index=False, encoding="utf-8-sig")
        rollout["action_outcomes"].to_csv(variant_dir / "daily_action_outcomes.csv", index=False, encoding="utf-8-sig")
        rollout["turnover_frame"].to_csv(variant_dir / "daily_turnover.csv", index=False, encoding="utf-8-sig")
        rollout["returns"].rename("daily_return").to_csv(variant_dir / "daily_returns.csv", encoding="utf-8-sig")
        rows.append(_metric_row(str(variant["label"]), variant, rollout))
        variant_payloads.append(
            {
                **variant,
                "metrics": rollout["metrics"],
                "continuity_metrics": rollout["continuity_metrics"],
                "artifact_dir": str(variant_dir.resolve()),
            }
        )

    ranking = pd.DataFrame(rows).sort_values(["annual_return", "sharpe"], ascending=[False, False]).reset_index(drop=True)
    ranking_path = run_root / "counterfactual_ranking.csv"
    ranking.to_csv(ranking_path, index=False, encoding="utf-8-sig")
    summary = {
        "run_tag": run_tag,
        "generated_at": now_iso(),
        "model_artifact_path": str(artifact_path.resolve()),
        "pool_name": prepared.pool_name,
        "benchmark": prepared.benchmark,
        "start_date": args.start_date,
        "end_date": args.end_date or prepared.end_date,
        "alpha_prior": alpha_args,
        "prepared_summary": prepared.to_summary(),
        "ranking_csv": str(ranking_path.resolve()),
        "variants": variant_payloads,
    }
    write_json(run_root / "counterfactual_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
