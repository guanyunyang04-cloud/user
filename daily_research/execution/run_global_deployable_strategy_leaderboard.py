from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.deep_alpha.execution_alignment import resolve_profile
from daily_research.execution.strategy_manifest import (
    DEFAULT_ACTIVE_EXECUTION_STRATEGY_MANIFEST,
    build_active_strategy_manifest,
    write_strategy_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"

DEFAULT_SHORT_ALPHA_REVIEW_ROOT = OUTPUT_ROOT / "short_alpha_execution_policy_formal_review_20260405_r1"
DEFAULT_SHORT_ALPHA_SOURCE_RUN = OUTPUT_ROOT / "short_alpha_formal_head2head_20260404_monthly_budgetnorm_r1" / "runs" / "state_liquidity_listwise_v1_20250318_20260331"
DEFAULT_SHORT_ALPHA_PRODUCTION_ROOT = OUTPUT_ROOT / "deep_alpha_short_alpha_execalign_production_default"

DEFAULT_DYNAMIC_GRAPH_REVIEW_ROOT = OUTPUT_ROOT / "dynamic_graph_no_priors_execution_policy_formal_review_20260406_r1"
DEFAULT_DYNAMIC_GRAPH_SOURCE_RUN = OUTPUT_ROOT / "dynamic_graph_ablation_formal_20260404_monthly_budgetnorm_r1" / "runs" / "dynamic_graph_no_priors_20250318_20260331"


@dataclass(frozen=True)
class CandidateReview:
    candidate_name: str
    strategy_name: str
    universe: str
    review_root: Path
    source_run_dir: Path
    production_root: Path
    panel_mode: str = "raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank current cross-universe deployable strategy lines under a single "
            "same-cost, non-capacity-adjusted leaderboard. By default this is a "
            "read-only refresh; use --activate-winner only when you explicitly "
            "want to overwrite the active execution manifest."
        )
    )
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument("--strategy-manifest-path", default=str(DEFAULT_ACTIVE_EXECUTION_STRATEGY_MANIFEST))
    parser.add_argument("--activate-winner", dest="activate_winner", action="store_true")
    parser.add_argument("--no-activate-winner", dest="activate_winner", action="store_false")
    parser.set_defaults(activate_winner=False)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _resolve_candidate_reviews() -> list[CandidateReview]:
    return [
        CandidateReview(
            candidate_name="state_liquidity_listwise_v1",
            strategy_name="state_liquidity_listwise_v1_execfirst_profitmax_global_winner",
            universe="liquid500",
            review_root=DEFAULT_SHORT_ALPHA_REVIEW_ROOT,
            source_run_dir=DEFAULT_SHORT_ALPHA_SOURCE_RUN,
            production_root=DEFAULT_SHORT_ALPHA_PRODUCTION_ROOT,
            panel_mode="raw",
        ),
        CandidateReview(
            candidate_name="dynamic_graph_no_priors",
            strategy_name="dynamic_graph_no_priors_execfirst_profitmax_global_winner",
            universe="liquid800",
            review_root=DEFAULT_DYNAMIC_GRAPH_REVIEW_ROOT,
            source_run_dir=DEFAULT_DYNAMIC_GRAPH_SOURCE_RUN,
            production_root=DEFAULT_DYNAMIC_GRAPH_SOURCE_RUN,
            panel_mode="raw",
        ),
    ]


def _review_payload(review_root: Path) -> dict[str, Any]:
    payload = _load_json(review_root / "leaderboard.json")
    if payload:
        return payload
    summary_path = review_root / "profile_mean_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Formal execution-policy review not found: {review_root}")
    summary_df = pd.read_csv(summary_path)
    if summary_df.empty:
        raise RuntimeError(f"Formal execution-policy review is empty: {summary_path}")
    row = summary_df.iloc[0].to_dict()
    return {
        "best_profile": str(row.get("profile_name", "")),
        "best_mean_excess_annual_return": float(row.get("mean_excess_annual_return", 0.0) or 0.0),
        "best_mean_excess_sharpe": float(row.get("mean_excess_sharpe", 0.0) or 0.0),
        "window_win_count": int(row.get("window_win_count", 0) or 0),
    }


def _rank_tuple(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(row.get("best_mean_excess_annual_return", float("-inf"))),
        float(row.get("best_mean_excess_sharpe", float("-inf"))),
        float(row.get("window_win_count", float("-inf"))),
    )


def _build_candidate_row(candidate: CandidateReview) -> dict[str, Any]:
    review_payload = _review_payload(candidate.review_root)
    return {
        "candidate_name": candidate.candidate_name,
        "strategy_name": candidate.strategy_name,
        "universe": candidate.universe,
        "review_root": str(candidate.review_root.resolve()),
        "source_run_dir": str(candidate.source_run_dir.resolve()),
        "production_root": str(candidate.production_root.resolve()),
        "panel_mode": candidate.panel_mode,
        "best_profile": str(review_payload.get("best_profile", "")),
        "best_mean_excess_annual_return": float(review_payload.get("best_mean_excess_annual_return", 0.0) or 0.0),
        "best_mean_excess_sharpe": float(review_payload.get("best_mean_excess_sharpe", 0.0) or 0.0),
        "window_win_count": int(review_payload.get("window_win_count", 0) or 0),
        "leaderboard_scope": "global_deployable_non_capacity_adjusted_v1",
        "leaderboard_metric": "mean_excess_annual_return",
    }


def _load_strategy_metrics(source_run_dir: Path, production_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source_metrics = _load_json(source_run_dir / "metrics.json")
    production_metrics = _load_json(production_root / "metrics.json")
    if not source_metrics:
        raise FileNotFoundError(f"Source metrics missing: {source_run_dir / 'metrics.json'}")
    if not production_metrics:
        production_metrics = dict(source_metrics)
    return source_metrics, production_metrics


def _activate_winner(
    *,
    winner_row: dict[str, Any],
    strategy_manifest_path: Path,
    leaderboard_rows: list[dict[str, Any]],
) -> Path:
    source_run_dir = Path(str(winner_row.get("source_run_dir", ""))).resolve()
    production_root = Path(str(winner_row.get("production_root", ""))).resolve()
    source_metrics, strategy_metrics = _load_strategy_metrics(source_run_dir, production_root)
    strategy_metrics = dict(strategy_metrics)
    best_profile_name = str(winner_row.get("best_profile", "")).strip()
    best_profile = resolve_profile(name=best_profile_name)
    transaction_cost_bps = float(
        strategy_metrics.get(
            "execution_alignment_transaction_cost_bps",
            source_metrics.get("execution_alignment_transaction_cost_bps", 3.0),
        )
        or 3.0
    )
    slippage_bps = float(
        strategy_metrics.get(
            "execution_alignment_slippage_bps",
            source_metrics.get("execution_alignment_slippage_bps", 7.0),
        )
        or 7.0
    )
    sell_tax_bps = float(
        strategy_metrics.get(
            "execution_alignment_sell_tax_bps",
            source_metrics.get("execution_alignment_sell_tax_bps", 10.0),
        )
        or 10.0
    )
    strategy_metrics["execution_alignment_profile"] = best_profile_name
    strategy_metrics["execution_policy_label"] = best_profile_name
    strategy_metrics["execution_alignment_selected_profile_spec"] = {
        "name": best_profile.name,
        "description": best_profile.description,
        "rebalance_freq": best_profile.rebalance_freq,
        "rebalance_offset_mode": best_profile.rebalance_offset_mode,
        "rebalance_anchor_date": best_profile.rebalance_anchor_date,
        "target_weight_top_k": int(best_profile.target_weight_top_k),
        "target_weight_min_weight": float(best_profile.target_weight_min_weight),
        "target_weight_power": float(best_profile.target_weight_power),
        "target_weight_full_invest": bool(best_profile.target_weight_full_invest),
        "use_market_regime_filter": bool(best_profile.use_market_regime_filter),
    }
    strategy_metrics["execution_alignment_selected_bridge_meta"] = {
        "profile_name": best_profile.name,
        "profile_description": best_profile.description,
        "rebalance_freq": best_profile.rebalance_freq,
        "rebalance_offset_mode": best_profile.rebalance_offset_mode,
        "rebalance_anchor_date": best_profile.rebalance_anchor_date,
        "target_weight_top_k": int(best_profile.target_weight_top_k),
        "target_weight_min_weight": float(best_profile.target_weight_min_weight),
        "target_weight_power": float(best_profile.target_weight_power),
        "target_weight_full_invest": bool(best_profile.target_weight_full_invest),
        "market_regime_filter": bool(best_profile.use_market_regime_filter),
        "transaction_cost_bps": transaction_cost_bps,
        "slippage_bps": slippage_bps,
        "sell_tax_bps": sell_tax_bps,
    }
    payload = build_active_strategy_manifest(
        source_run_dir=source_run_dir,
        production_root=production_root,
        source_metrics=source_metrics,
        strategy_metrics=strategy_metrics,
        panel_mode=str(winner_row.get("panel_mode", "raw") or "raw"),
        strategy_name=str(winner_row.get("strategy_name", "")).strip(),
        promoted_at=datetime.now().isoformat(),
    )
    payload["leaderboard_scope"] = str(winner_row.get("leaderboard_scope", "") or "")
    payload["leaderboard_metric"] = str(winner_row.get("leaderboard_metric", "") or "")
    payload["global_deployable_winner"] = True
    payload["global_deployable_rank"] = 1
    payload["global_deployable_candidate_count"] = int(len(leaderboard_rows))
    payload["global_deployable_summary_rows"] = [
        {
            "candidate_name": str(row.get("candidate_name", "")),
            "strategy_name": str(row.get("strategy_name", "")),
            "universe": str(row.get("universe", "")),
            "best_profile": str(row.get("best_profile", "")),
            "best_mean_excess_annual_return": float(row.get("best_mean_excess_annual_return", 0.0) or 0.0),
            "best_mean_excess_sharpe": float(row.get("best_mean_excess_sharpe", 0.0) or 0.0),
        }
        for row in leaderboard_rows
    ]
    return write_strategy_manifest(payload, path=strategy_manifest_path)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    root_tag = args.experiment_tag.strip() or f"global_deployable_strategy_leaderboard_{datetime.now():%Y%m%d_r1}"
    leaderboard_root = output_root / root_tag
    leaderboard_root.mkdir(parents=True, exist_ok=True)
    strategy_manifest_path = Path(args.strategy_manifest_path).resolve()

    candidate_rows = [_build_candidate_row(candidate) for candidate in _resolve_candidate_reviews()]
    candidate_rows.sort(key=_rank_tuple, reverse=True)
    for idx, row in enumerate(candidate_rows, start=1):
        row["global_rank"] = int(idx)

    summary_df = pd.DataFrame(candidate_rows)
    summary_df.to_csv(leaderboard_root / "global_candidate_summary.csv", index=False, encoding="utf-8-sig")
    winner_row = dict(candidate_rows[0])

    active_manifest_path = ""
    if args.activate_winner:
        active_manifest_path = str(_activate_winner(
            winner_row=winner_row,
            strategy_manifest_path=strategy_manifest_path,
            leaderboard_rows=candidate_rows,
        ))

    summary_payload = {
        "leaderboard_scope": "global_deployable_non_capacity_adjusted_v1",
        "leaderboard_metric": "mean_excess_annual_return",
        "candidate_count": int(len(candidate_rows)),
        "winner": winner_row,
        "active_manifest_path": active_manifest_path,
        "leaderboard_root": str(leaderboard_root),
    }
    (leaderboard_root / "leaderboard.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Global Deployable Strategy Leaderboard",
        "",
        "- scope: `global_deployable_non_capacity_adjusted_v1`",
        "- note: `liquid500` and `liquid800` are ranked under the same current cost engine and execution policy search, but still without explicit capacity penalties.",
        f"- winner: `{winner_row['strategy_name']}`",
        f"- winner_universe: `{winner_row['universe']}`",
        f"- winner_profile: `{winner_row['best_profile']}`",
        f"- winner_mean_excess_annual_return: `{float(winner_row['best_mean_excess_annual_return']):.2%}`",
        f"- winner_mean_excess_sharpe: `{float(winner_row['best_mean_excess_sharpe']):.3f}`",
        f"- active_manifest_path: `{active_manifest_path or 'not_updated'}`",
        "",
        "## Candidate Ranking",
        "",
    ]
    for row in candidate_rows:
        lines.append(
            f"- rank {int(row['global_rank'])}: `{row['strategy_name']}` / `{row['universe']}` / "
            f"`{row['best_profile']}` / excess annual `{float(row['best_mean_excess_annual_return']):.2%}` / "
            f"excess Sharpe `{float(row['best_mean_excess_sharpe']):.3f}`"
        )
    (leaderboard_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary_payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
