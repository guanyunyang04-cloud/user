from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.execution.entrypoint_utils import (
    bootstrap_execution_paths,
    consume_flag_arg,
    consume_option_arg,
    ensure_default_pool_argument,
    ensure_execution_strategy_defaults,
    ensure_external_target_weight_universe_argument,
    ensure_text_file_from_example,
    has_arg,
    inject_default_arg,
    is_help_request,
    missing_runtime_dependency_error,
)


def main():
    exec_dir = bootstrap_execution_paths(__file__)
    positions_file = exec_dir / "current_positions.csv"
    example_file = exec_dir / "current_positions.example.csv"
    output_dir = exec_dir / "output" / "research_candidates"
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_help_request():
        print("wrapper_options: --candidate-profile <name> | --list-candidate-profiles")
        return

    try:
        from daily_research.execution.research_candidate_profiles import (
            DEFAULT_EXECUTION_CANDIDATE_PROFILE,
            apply_profile_defaults,
            list_profile_lines,
        )
    except ModuleNotFoundError as exc:
        raise missing_runtime_dependency_error(
            exc,
            command_hint="python daily_research/execution/run_research_candidate_trade_plan.py",
        ) from exc

    if consume_flag_arg("--list-candidate-profiles"):
        print("\n".join(list_profile_lines()))
        return
    candidate_profile = consume_option_arg("--candidate-profile")

    ensure_text_file_from_example(
        positions_file,
        example_file,
        "stock,shares,cost_price\n",
    )

    inject_default_arg("--positions-file", str(positions_file))
    inject_default_arg("--output-dir", str(output_dir))
    inject_default_arg("--external-score-column", "model_decision_score")
    inject_default_arg("--external-watch-score-column", "model_decision_score")
    inject_default_arg("--external-target-weight-column", "target_weight")

    if not candidate_profile and not has_arg("--external-score-csv") and not has_arg("--external-target-weight-csv") and not is_help_request():
        candidate_profile = DEFAULT_EXECUTION_CANDIDATE_PROFILE
    pool_name_hint = ""
    if candidate_profile:
        resolved = apply_profile_defaults(
            candidate_profile,
            mode="trade_plan",
            ensure_live_panels=False,
        )
        pool_name_hint = str(resolved.liquidity_pool_name or "").strip()
        if not is_help_request():
            print(f"candidate_profile={resolved.name}")

    if not ensure_external_target_weight_universe_argument():
        ensure_default_pool_argument(pool_name=pool_name_hint)
    ensure_execution_strategy_defaults()

    if not has_arg("--external-score-csv") and not has_arg("--external-target-weight-csv") and not is_help_request():
        raise ValueError(
            "Missing --external-score-csv or --external-target-weight-csv. "
            "Pass --candidate-profile, --list-candidate-profiles, or explicit CSV paths."
        )

    try:
        from daily_research.baseline.generate_daily_trade_plan import main as generate_daily_trade_plan_main
    except ModuleNotFoundError as exc:
        raise missing_runtime_dependency_error(
            exc,
            command_hint="python daily_research/execution/run_research_candidate_trade_plan.py",
        ) from exc

    generate_daily_trade_plan_main()


if __name__ == "__main__":
    main()
