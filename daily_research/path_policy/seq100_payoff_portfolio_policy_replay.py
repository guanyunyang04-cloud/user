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

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = Path(
    "daily_research/studies/seq100_payoff_portfolio_policy_replay_v1.json"
)
SCHEMA = "seq100_payoff_portfolio_policy_replay/1"
HORIZONS = (2, 3, 5, 10)
CAPITAL_POLICIES = {
    "fixed_daily_cohort_10pct": lambda _horizon: 0.10,
    "horizon_normalized_full_investment": lambda horizon: 1.0 / float(horizon),
}
CAPITAL_POLICY_SLUGS = {
    "fixed_daily_cohort_10pct": "fdc10",
    "horizon_normalized_full_investment": "hnfi",
}


class PortfolioPolicyReplayError(RuntimeError):
    """Raised when the frozen portfolio replay contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PortfolioPolicyReplayError(message)


def build_policy_specs(
    *, horizon: int, variant: str, cohort_equity_fraction: float
) -> tuple[dict[str, Any], ...]:
    _require(int(horizon) in HORIZONS, "unsupported portfolio replay horizon")
    _require(
        0.0 < float(cohort_equity_fraction) <= 0.5,
        "invalid cohort equity fraction",
    )
    specs: list[dict[str, Any]] = []
    for allow_overlap in (True, False):
        for cap in (None, 0.10):
            spec: dict[str, Any] = {
                "variant": str(variant),
                "exit_policy": "planned_close",
                "gate": "always",
                "top_k": 10,
                "cost_scenario": "stress",
                "slippage_multiplier": float(base.EXACT_NET_SCENARIOS["stress"]),
                "cohort_equity_fraction": float(cohort_equity_fraction),
                "planned_fill_day": int(horizon),
                "allow_overlapping_same_symbol": bool(allow_overlap),
            }
            if cap is not None:
                spec["maximum_credited_gross_return"] = float(cap)
            specs.append(spec)
    return tuple(specs)


def common_model_rows(schedules: Mapping[int, pd.DataFrame]) -> set[int]:
    _require(set(schedules) == set(HORIZONS), "all frozen horizons are required")
    row_sets: list[set[int]] = []
    for horizon in HORIZONS:
        frame = schedules[horizon]
        _require(not frame.empty, f"empty schedule for D{horizon}")
        _require(
            bool(frame["top_k"].astype(int).eq(10).all()),
            "common-row schedule must contain only Top10 rows",
        )
        _require(
            not bool(frame["model_row_position"].duplicated().any()),
            f"duplicate model row in D{horizon} schedule",
        )
        row_sets.append(set(frame["model_row_position"].astype(int)))
    return set.intersection(*row_sets)


def _summary_for_spec(
    summaries: Sequence[Mapping[str, Any]],
    *,
    allow_overlap: bool,
    cap: float | None,
) -> Mapping[str, Any]:
    matches = [
        item
        for item in summaries
        if int(item["spec"]["top_k"]) == 10
        and str(item["spec"]["cost_scenario"]) == "stress"
        and bool(item["spec"].get("allow_overlapping_same_symbol", True))
        == bool(allow_overlap)
        and item["spec"].get("maximum_credited_gross_return") == cap
    ]
    _require(len(matches) == 1, "portfolio replay result is not unique")
    return matches[0]


def _episode_diagnostic(*, root: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    task_id = str(result["task_id"])
    trades_path = root / "tasks" / task_id / "trades.parquet"
    trades = pd.read_parquet(trades_path)
    assigned = episode.assign_episodes(trades, gap_days=0)
    episodes = episode.summarize_episodes(assigned, definition="overlap")
    return {
        "trade_count": len(trades),
        "episode_count": len(episodes),
        "maximum_trades_per_episode": int(episodes["trade_count"].max()),
        "trade_concentration": episode.concentration_metrics(trades["pnl"]),
        "episode_concentration": episode.concentration_metrics(episodes["pnl"]),
        "files": {"trades": episode._file_record(trades_path, row_count=len(trades))},
    }


def _result_projection(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_id": result["task_id"],
        "spec": result["spec"],
        "starting_cash": result["starting_cash"],
        "ending_equity": result["ending_equity"],
        "total_net_return": result["total_net_return"],
        "maximum_drawdown": result["maximum_drawdown"],
        "annualized_daily_sharpe": result["annualized_daily_sharpe"],
        "trade_count": result["trade_count"],
        "positive_trade_fraction": result["positive_trade_fraction"],
        "mean_trade_net_return": result["mean_trade_net_return"],
        "positive_year_count": result["positive_year_count"],
        "negative_year_count": result["negative_year_count"],
        "worst_year_return": result["worst_year_return"],
        "minimum_cash": result["minimum_cash"],
        "maximum_position_count": result["maximum_position_count"],
        "maximum_unique_symbol_count": result["maximum_unique_symbol_count"],
        "maximum_same_symbol_open_cohorts": result["maximum_same_symbol_open_cohorts"],
        "same_symbol_overlap_order_count": result["same_symbol_overlap_order_count"],
        "daily_hac20_net_return": result["daily_hac20_net_return"],
        "annual": result["annual"],
        "fold_account_metrics": result["fold_account_metrics"],
    }


def run_replay(*, study_path: str | Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study_path = episode._workspace_path(study_path).resolve()
    study = episode._read_json(study_path)
    source_spec = study["sources"]
    _require(
        str(source_spec["maximum_outcome_date"]) == episode.MAXIMUM_OUTCOME_DATE,
        "portfolio replay outcome cutoff drifted",
    )
    _require(int(source_spec["forbidden_year"]) == 2026, "2026 must be forbidden")
    model_study_path = episode._workspace_path(source_spec["model_study"]).resolve()
    model_output_root = episode._workspace_path(
        source_spec["model_output_root"]
    ).resolve()
    evaluation_manifest_path = episode._workspace_path(
        source_spec["d10_evaluation_manifest"]
    ).resolve()
    evaluation = episode._read_json(evaluation_manifest_path)
    _require(evaluation.get("status") == "completed", "D10 evaluation is incomplete")
    _require(int(evaluation.get("horizon", -1)) == 10, "source score is not D10")
    _require(
        int(evaluation.get("forbidden_2026_read_count", -1)) == 0,
        "source D10 evaluation does not attest zero 2026 reads",
    )
    selection_path = Path(str(evaluation["files"]["top10_selections"]["path"]))
    selections = pd.read_parquet(selection_path)
    _require(
        bool(selections["selection_rank"].between(1, 10).all()),
        "source selection schedule is not Top10",
    )
    _require(
        not bool(selections["model_row_position"].duplicated().any()),
        "source selection rows are not unique",
    )

    model_study = base.load_study(model_study_path)
    sources_by_horizon: dict[int, sequence.SequenceSources] = {}
    schedules: dict[int, pd.DataFrame] = {}
    dropped_by_horizon: dict[str, int] = {}
    for horizon in HORIZONS:
        sources = sequence._load_sources(
            model_study, output_root=model_output_root, horizon=horizon
        )
        schedule, dropped = sequence._prepare_exact_payoff_schedule(
            selections=selections,
            sources=sources,
            horizon=horizon,
            variant=f"frozen_d10_score__legal_exit_d{horizon}",
        )
        schedule = schedule.loc[schedule["top_k"].astype(int).eq(10)].copy()
        sources_by_horizon[horizon] = sources
        schedules[horizon] = schedule
        dropped_by_horizon[str(horizon)] = int(dropped)

    common_rows = common_model_rows(schedules)
    _require(bool(common_rows), "no common exact-outcome rows across horizons")
    for horizon in HORIZONS:
        schedules[horizon] = (
            schedules[horizon]
            .loc[schedules[horizon]["model_row_position"].astype(int).isin(common_rows)]
            .copy()
        )
        _require(
            len(schedules[horizon]) == len(common_rows),
            f"common schedule alignment failed for D{horizon}",
        )

    output_root = episode._workspace_path(study["outputs"]["output_root"]).resolve()
    fingerprint_payload = {
        "schema": SCHEMA,
        "implementation": episode._file_record(Path(__file__).resolve()),
        "episode_implementation": episode._file_record(
            Path(episode.__file__).resolve()
        ),
        "study": episode._file_record(study_path),
        "model_study": episode._file_record(model_study_path),
        "evaluation_manifest": episode._file_record(evaluation_manifest_path),
        "selection_schedule": episode._file_record(selection_path),
        "horizons": HORIZONS,
        "capital_policies": tuple(CAPITAL_POLICIES),
        "common_model_row_count": len(common_rows),
        "maximum_outcome_date": episode.MAXIMUM_OUTCOME_DATE,
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

    policy_results: dict[str, Any] = {}
    files: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for policy_name, fraction_for_horizon in CAPITAL_POLICIES.items():
        policy_slug = CAPITAL_POLICY_SLUGS[policy_name]
        horizon_results: dict[str, Any] = {}
        for horizon in HORIZONS:
            cohort_fraction = float(fraction_for_horizon(horizon))
            variant = f"d10s_d{horizon}_{policy_slug}"
            schedule = schedules[horizon].copy()
            schedule["variant"] = variant
            specs = build_policy_specs(
                horizon=horizon,
                variant=variant,
                cohort_equity_fraction=cohort_fraction,
            )
            task_root = output_root / policy_slug / f"d{horizon}"
            task_fingerprint = episode._stable_hash(
                {
                    "parent": fingerprint,
                    "policy": policy_name,
                    "horizon": horizon,
                    "cohort_equity_fraction": cohort_fraction,
                    "specs": specs,
                }
            )
            tasks = sequence._run_exact_payoff_account_tasks(
                root=task_root,
                schema=SCHEMA,
                fingerprint=task_fingerprint,
                schedule=schedule,
                sources=sources_by_horizon[horizon],
                specs=specs,
                starting_cash=1_000_000.0,
            )
            summaries = tasks["summaries"]
            projected: dict[str, Any] = {}
            for allow_overlap, label in ((True, "overlap"), (False, "no_overlap")):
                primary = _summary_for_spec(
                    summaries, allow_overlap=allow_overlap, cap=None
                )
                cap10 = _summary_for_spec(
                    summaries, allow_overlap=allow_overlap, cap=0.10
                )
                projected[label] = {
                    "primary": _result_projection(primary),
                    "winner_cap_10pct": _result_projection(cap10),
                    "episode_diagnostic": _episode_diagnostic(
                        root=task_root, result=primary
                    ),
                }
                summary_rows.append(
                    {
                        "capital_policy": policy_name,
                        "horizon": int(horizon),
                        "allow_overlapping_same_symbol": bool(allow_overlap),
                        "cohort_equity_fraction": cohort_fraction,
                        "total_net_return": float(primary["total_net_return"]),
                        "maximum_drawdown": float(primary["maximum_drawdown"]),
                        "positive_year_count": int(primary["positive_year_count"]),
                        "worst_year_return": float(primary["worst_year_return"]),
                        "daily_hac_lower": float(
                            primary["daily_hac20_net_return"]["lower"]
                        ),
                        "cap10_total_net_return": float(cap10["total_net_return"]),
                        "cap10_maximum_drawdown": float(cap10["maximum_drawdown"]),
                        "trade_count": int(primary["trade_count"]),
                    }
                )
            horizon_results[str(horizon)] = {
                "cohort_equity_fraction": cohort_fraction,
                "common_selection_row_count": len(schedule),
                "scenarios": projected,
                "files": {
                    "summary": tasks["summary"],
                    "selection_schedule": tasks["selection_schedule"],
                    "tasks": tasks["tasks"],
                },
            }
        policy_results[policy_name] = horizon_results

    summary = pd.DataFrame(summary_rows).sort_values(
        ["capital_policy", "allow_overlapping_same_symbol", "horizon"],
        kind="mergesort",
    )
    summary_path = episode._write_parquet(output_root / "summary.parquet", summary)
    files["summary"] = episode._file_record(summary_path, row_count=len(summary))
    result = {
        "schema": SCHEMA,
        "status": "completed",
        "study_id": study["study_id"],
        "completed_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "fingerprint": fingerprint,
        "forbidden_2026_read_count": 0,
        "maximum_outcome_date_read": episode.MAXIMUM_OUTCOME_DATE,
        "source_model_score": {
            "target": evaluation["target"],
            "lookback": evaluation["lookback"],
            "horizon": evaluation["horizon"],
            "fusion_method": evaluation["fusion_method"],
            "tree_feature_variant": evaluation["tree_feature_variant"],
            "score_retrained_for_exit_horizon": False,
        },
        "common_selection_contract": {
            "row_count": len(common_rows),
            "same_stock_date_rank_rows_across_all_horizons": True,
            "dropped_before_common_intersection_by_horizon": dropped_by_horizon,
            "no_lower_rank_substitution": True,
        },
        "policy_results": policy_results,
        "decision_boundary": {
            "replay_role": "portfolio_policy_diagnostic_not_new_model_evidence",
            "same_frozen_d10_oof_score_for_every_policy": True,
            "stress_costs_only": True,
            "top_k_fixed_at_10": True,
            "winner_cap_is_pressure_test_not_executable_exit_rule": True,
            "fixed_daily_cohort_policy_isolates_order_sizing": True,
            "horizon_normalized_policy_approximately_equalizes_gross_exposure": True,
            "stable_profit_claim_allowed": False,
            "forbidden_2026_read_count": 0,
        },
        "files": files,
        "sources": {
            "implementation": episode._file_record(Path(__file__).resolve()),
            "episode_implementation": episode._file_record(
                Path(episode.__file__).resolve()
            ),
            "study": episode._file_record(study_path),
            "model_study": episode._file_record(model_study_path),
            "evaluation_manifest": episode._file_record(evaluation_manifest_path),
            "selection_schedule": episode._file_record(selection_path),
        },
    }
    episode._write_json(manifest_path, result)
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay one frozen model score under independent portfolio policies."
    )
    parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_replay(study_path=args.study)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
