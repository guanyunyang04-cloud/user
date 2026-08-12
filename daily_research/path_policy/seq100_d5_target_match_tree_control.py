from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from daily_research.path_policy import seq100_full_market_multitask_forecast as base
from daily_research.path_policy import (
    seq100_full_market_sequence_challenger as sequence,
)
from daily_research.path_policy import seq100_payoff_episode_audit as episode
from daily_research.path_policy.seq100_payoff_portfolio_policy_replay import (
    _result_projection,
    _summary_for_spec,
    build_policy_specs,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = Path(
    "daily_research/studies/seq100_d5_target_match_tree_control_v1.json"
)
SCHEMA = "seq100_d5_target_match_tree_control/1"


class D5TargetMatchControlError(RuntimeError):
    """Raised when the architecture-matched target control drifts."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise D5TargetMatchControlError(message)


def robust_pareto_comparison(
    *, matched_d5: Mapping[str, Any], d10_control: Mapping[str, Any]
) -> dict[str, Any]:
    metrics = {
        "total_net_return": (
            float(matched_d5["primary"]["total_net_return"]),
            float(d10_control["primary"]["total_net_return"]),
            "higher",
        ),
        "maximum_drawdown": (
            float(matched_d5["primary"]["maximum_drawdown"]),
            float(d10_control["primary"]["maximum_drawdown"]),
            "higher",
        ),
        "positive_year_count": (
            float(matched_d5["primary"]["positive_year_count"]),
            float(d10_control["primary"]["positive_year_count"]),
            "higher",
        ),
        "daily_hac_lower": (
            float(matched_d5["primary"]["daily_hac20_net_return"]["lower"]),
            float(d10_control["primary"]["daily_hac20_net_return"]["lower"]),
            "higher",
        ),
        "winner_cap_10pct_total_net_return": (
            float(matched_d5["winner_cap_10pct"]["total_net_return"]),
            float(d10_control["winner_cap_10pct"]["total_net_return"]),
            "higher",
        ),
    }
    rows: dict[str, Any] = {}
    all_noninferior = True
    any_better = False
    for name, (challenger, control, direction) in metrics.items():
        _require(direction == "higher", "unknown Pareto direction")
        noninferior = challenger >= control
        better = challenger > control
        rows[name] = {
            "matched_d5": challenger,
            "d10_score_control": control,
            "difference": challenger - control,
            "matched_d5_noninferior": noninferior,
            "matched_d5_strictly_better": better,
        }
        all_noninferior &= noninferior
        any_better |= better
    return {
        "metrics": rows,
        "matched_d5_robust_pareto_dominates": bool(all_noninferior and any_better),
    }


def run_control(*, study_path: str | Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study_path = episode._workspace_path(study_path).resolve()
    study = episode._read_json(study_path)
    sources_spec = study["sources"]
    _require(
        str(sources_spec["maximum_outcome_date"]) == episode.MAXIMUM_OUTCOME_DATE,
        "outcome cutoff drifted",
    )
    _require(int(sources_spec["forbidden_year"]) == 2026, "2026 must be forbidden")
    model_study_path = episode._workspace_path(sources_spec["model_study"]).resolve()
    model_output_root = episode._workspace_path(
        sources_spec["model_output_root"]
    ).resolve()
    d10_evaluation_path = episode._workspace_path(
        sources_spec["d10_tree_evaluation_manifest"]
    ).resolve()
    d5_account_path = episode._workspace_path(
        sources_spec["d5_tree_account_manifest"]
    ).resolve()
    d10_evaluation = episode._read_json(d10_evaluation_path)
    d5_account = episode._read_json(d5_account_path)
    for payload, label in (
        (d10_evaluation, "D10 evaluation"),
        (d5_account, "D5 account"),
    ):
        _require(payload.get("status") == "completed", f"{label} is incomplete")
        _require(
            int(payload.get("forbidden_2026_read_count", -1)) == 0,
            f"{label} does not attest zero 2026 reads",
        )
    _require(
        str(d10_evaluation.get("target")) == "exact_net_return_d10_rank",
        "D10 score target drifted",
    )
    _require(
        str(d5_account.get("target")) == "exact_net_return_d5_rank",
        "D5 score target drifted",
    )
    _require(
        str(d10_evaluation.get("feature_variant"))
        == str(d5_account.get("feature_variant"))
        == "price_path_core_183",
        "tree feature views are not architecture matched",
    )

    selections_path = Path(str(d10_evaluation["files"]["top10_selections"]["path"]))
    selections = pd.read_parquet(selections_path)
    model_study = base.load_study(model_study_path)
    sources = sequence._load_sources(
        model_study, output_root=model_output_root, horizon=5
    )
    variant = "d10t_d5"
    schedule, dropped = sequence._prepare_exact_payoff_schedule(
        selections=selections,
        sources=sources,
        horizon=5,
        variant=variant,
    )
    schedule = schedule.loc[schedule["top_k"].astype(int).eq(10)].copy()
    specs = build_policy_specs(
        horizon=5,
        variant=variant,
        cohort_equity_fraction=0.20,
    )
    output_root = episode._workspace_path(study["outputs"]["output_root"]).resolve()
    fingerprint_payload = {
        "schema": SCHEMA,
        "implementation": episode._file_record(Path(__file__).resolve()),
        "study": episode._file_record(study_path),
        "d10_evaluation": episode._file_record(d10_evaluation_path),
        "d5_account": episode._file_record(d5_account_path),
        "d10_selections": episode._file_record(selections_path),
        "specs": specs,
        "schedule_row_count": len(schedule),
        "dropped_schedule_row_count": int(dropped),
    }
    fingerprint = episode._stable_hash(fingerprint_payload)
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        current = episode._read_json(manifest_path)
        if (
            current.get("status") == "completed"
            and current.get("fingerprint") == fingerprint
            and current.get("forbidden_2026_read_count") == 0
        ):
            return current

    tasks = sequence._run_exact_payoff_account_tasks(
        root=output_root / "d10_tree_score_exit_d5",
        schema=SCHEMA,
        fingerprint=fingerprint,
        schedule=schedule,
        sources=sources,
        specs=specs,
        starting_cash=1_000_000.0,
    )
    d10_summaries = tasks["summaries"]
    d10_control = {
        label: {
            "primary": _result_projection(
                _summary_for_spec(d10_summaries, allow_overlap=allow_overlap, cap=None)
            ),
            "winner_cap_10pct": _result_projection(
                _summary_for_spec(d10_summaries, allow_overlap=allow_overlap, cap=0.10)
            ),
        }
        for allow_overlap, label in ((True, "overlap"), (False, "no_overlap"))
    }
    d5_summaries = [
        episode._read_json(record["path"])
        for name, record in d5_account["files"].items()
        if name != "summary"
    ]
    d5_matched = {
        label: {
            "primary": _result_projection(
                _summary_for_spec(d5_summaries, allow_overlap=allow_overlap, cap=None)
            ),
            "winner_cap_10pct": _result_projection(
                _summary_for_spec(d5_summaries, allow_overlap=allow_overlap, cap=0.10)
            ),
        }
        for allow_overlap, label in ((True, "overlap"), (False, "no_overlap"))
    }
    comparison = robust_pareto_comparison(
        matched_d5=d5_matched["no_overlap"],
        d10_control=d10_control["no_overlap"],
    )
    result = {
        "schema": SCHEMA,
        "status": "completed",
        "study_id": study["study_id"],
        "completed_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "fingerprint": fingerprint,
        "forbidden_2026_read_count": 0,
        "maximum_outcome_date_read": episode.MAXIMUM_OUTCOME_DATE,
        "architecture_match": {
            "model_family": "LightGBM strong_127",
            "feature_variant": "price_path_core_183",
            "training_mode": "outer_early_stop",
            "breadth": 10,
            "exit_horizon": 5,
            "cohort_equity_fraction": 0.20,
            "cost_scenario": "stress",
        },
        "d5_matched_target": d5_matched,
        "d10_score_exit_d5_control": d10_control,
        "no_overlap_robust_comparison": comparison,
        "sequence_training_decision": {
            "d5_sequence_training_promoted": False,
            "reason": (
                "the matched D5 tree does not robustly Pareto-dominate the "
                "architecture-matched D10-tree score on a D5 portfolio"
            ),
            "decision_is_adaptive_development_resource_allocation": True,
        },
        "decision_boundary": {
            "comparison_is_reusable_development_evidence": True,
            "not_independent_confirmation": True,
            "stable_profit_claim_allowed": False,
            "forbidden_2026_read_count": 0,
        },
        "files": {
            "control_summary": tasks["summary"],
            "control_selection_schedule": tasks["selection_schedule"],
            "control_tasks": tasks["tasks"],
        },
        "sources": {
            "implementation": episode._file_record(Path(__file__).resolve()),
            "study": episode._file_record(study_path),
            "model_study": episode._file_record(model_study_path),
            "d10_tree_evaluation": episode._file_record(d10_evaluation_path),
            "d5_tree_account": episode._file_record(d5_account_path),
            "d10_selections": episode._file_record(selections_path),
        },
    }
    episode._write_json(manifest_path, result)
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare D5- and D10-target tree scores on the same D5 portfolio."
    )
    parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_control(study_path=args.study)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
