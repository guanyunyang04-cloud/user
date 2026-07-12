"""Current seq100 development facade.

This module intentionally exposes only the four operational steps used by the
2022-2025 candidate-complete development workflow.  The larger
``seq100_research_generation`` module remains the immutable compatibility
engine for historical registries and retired fixed-OOS experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from daily_research.path_policy import seq100_research_generation as generation


WORKFLOW_ID = "seq100_candidate_complete_development"
OPERATIONAL_COMMANDS = ("register", "run", "select", "freeze")
CURRENT_SOURCE_VIEW = generation.DEFAULT_DEVELOPMENT_SOURCE_VIEW
CURRENT_PROFILES = generation.DEFAULT_DEVELOPMENT_PROFILES
MEMORY_GUARD_MIN_AVAILABLE_GIB = 1.0


def current_contract() -> dict[str, Any]:
    """Return the small, current semantic surface without historical branches."""

    return {
        "schema_version": 1,
        "workflow_id": WORKFLOW_ID,
        "status": "development_closed_without_winner",
        "operational_commands": list(OPERATIONAL_COMMANDS),
        "source_pack": CURRENT_SOURCE_VIEW.as_posix(),
        "development_years": list(generation.DEVELOPMENT_YEARS),
        "seed": generation.DEVELOPMENT_SEED,
        "training": {
            "sample_policy": "all_eligible_rows",
            "minimum_complete_epochs": 1,
            "maximum_epochs": 10,
            "early_stopping_metric": "development_total_loss",
            "early_stopping_patience": 2,
            "restore_best_checkpoint": True,
        },
        "profile_roles": {
            "baseline": "control_only_not_champion",
            "hard_st": "rejected_candidate",
            "champion": None,
        },
        "metric_layers": {
            "checkpoint_selection": ["development_total_loss"],
            "candidate_eligibility": [
                "top1_3_5_10_cost_adjusted_realized_plan_alpha",
                "top3_stress_alpha",
                "positive_top3_development_years",
                "execution_and_plan_coverage",
            ],
            "diagnosis_only": [
                "opportunity_alpha",
                "oracle_regret",
                "daily_rank_ic",
                "path_mae",
                "entry_fill_rate",
            ],
        },
        "value_term": {
            "canonical_name": "predicted_path_opportunity_score_v2",
            "historical_field_alias": "path_trade_value_v2",
            "warning": "opportunity score is not executable return or live PnL",
        },
        "winner": None,
        "freeze_allowed": False,
        "next_decision": "audit_exit_policy_gap_then_add_one_execution_aligned_soft_exit_candidate",
        "blocked_branch": "hard_st_global_tail_until_soft_exit_candidate_passes",
        "memory_guard_min_available_gib": MEMORY_GUARD_MIN_AVAILABLE_GIB,
        "compatibility_engine": "daily_research.path_policy.seq100_research_generation",
        "historical_fixed_oos_is_current": False,
        "active_execution_changed": False,
        "qdp_active_changed": False,
    }


def _profiles(value: str) -> tuple[str, ...]:
    profiles = tuple(dict.fromkeys(item.strip() for item in str(value).split(",") if item.strip()))
    if not profiles:
        raise argparse.ArgumentTypeError("at least one development profile is required")
    unsupported = [item for item in profiles if item not in CURRENT_PROFILES]
    if unsupported:
        allowed = ", ".join(CURRENT_PROFILES)
        raise argparse.ArgumentTypeError(
            f"current facade accepts only {allowed}; use the compatibility engine for historical profiles"
        )
    return profiles


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Current candidate-complete seq100 development workflow. "
            "Historical screen/confirmation/outer-OOS commands live in the compatibility engine."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    contract = sub.add_parser("contract", help="Print the current slim semantic contract.")
    contract.add_argument("--compact", action="store_true")

    register = sub.add_parser("register", help="Freeze one full-data 2022-2025 development matrix.")
    register.add_argument("--root", type=Path, required=True)
    register.add_argument("--store-root", type=Path, required=True)
    register.add_argument("--source-view", type=Path, default=CURRENT_SOURCE_VIEW)
    register.add_argument(
        "--profiles",
        type=_profiles,
        required=True,
        help="Explicit comma-separated candidate set; there is no implicit default/champion profile.",
    )
    register.add_argument("--additional-provenance-path", type=Path, action="append", default=[])

    run = sub.add_parser("run", help="Run or resume registered jobs; launch this command through memory_guard.")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--registry", type=Path, required=True)
    run.add_argument("--max-jobs", type=int, default=0)

    select = sub.add_parser("select", help="Apply the frozen eligibility and ranking policy.")
    select.add_argument("--registry", type=Path, required=True)
    select.add_argument("--ledger", type=Path, required=True)

    freeze = sub.add_parser("freeze", help="Freeze only the policy winner; winner=null remains blocked.")
    freeze.add_argument("--root", type=Path, required=True)
    freeze.add_argument("--registry", type=Path, required=True)
    freeze.add_argument("--ledger", type=Path, required=True)
    freeze.add_argument("--champion", choices=CURRENT_PROFILES, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "contract":
        result = current_contract()
        indent = None if bool(args.compact) else 2
    elif args.command == "register":
        result = generation.initialize_development_registry(
            root=args.root,
            source_view=args.source_view,
            store_root=args.store_root,
            profiles=tuple(args.profiles),
            additional_provenance_paths=tuple(args.additional_provenance_path),
        )
        indent = 2
    elif args.command == "run":
        result = generation.run_development_registry(
            root=args.root,
            registry_path=args.registry,
            max_jobs=int(args.max_jobs),
        )
        indent = 2
    elif args.command == "select":
        result = generation.select_development_profiles(
            registry_path=args.registry,
            ledger_path=args.ledger,
        )
        indent = 2
    else:
        result = generation.freeze_development_champion(
            root=args.root,
            registry_path=args.registry,
            ledger_path=args.ledger,
            champion=args.champion,
        )
        indent = 2
    print(json.dumps(result, ensure_ascii=False, indent=indent, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
