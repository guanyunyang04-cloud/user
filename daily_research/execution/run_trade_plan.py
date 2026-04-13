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
    output_dir = exec_dir / "output"
    model_dir = exec_dir / "models"
    model_artifact = model_dir / "latest_ml_model.joblib"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    if is_help_request():
        print("wrapper_options: --candidate-profile <name> | --list-candidate-profiles | --legacy-ml")
        return

    try:
        from daily_research.execution.research_candidate_profiles import (
            DEFAULT_EXECUTION_CANDIDATE_PROFILE,
            apply_profile_defaults,
            get_profile,
            list_profile_lines,
        )
    except ModuleNotFoundError as exc:
        raise missing_runtime_dependency_error(
            exc,
            command_hint="python daily_research/execution/run_trade_plan.py",
        ) from exc

    if consume_flag_arg("--list-candidate-profiles"):
        print("\n".join(list_profile_lines()))
        return
    candidate_profile = consume_option_arg("--candidate-profile")
    legacy_ml = consume_flag_arg("--legacy-ml")
    explicit_model_artifact = has_arg("--model-artifact")
    explicit_external_candidate = has_arg("--external-score-csv") or has_arg("--external-target-weight-csv")

    if legacy_ml and candidate_profile:
        raise ValueError("--legacy-ml cannot be combined with --candidate-profile.")
    if legacy_ml and explicit_external_candidate:
        raise ValueError("--legacy-ml cannot be combined with --external-score-csv or --external-target-weight-csv.")

    ensure_text_file_from_example(
        positions_file,
        example_file,
        "stock,shares,cost_price\n",
    )

    inject_default_arg("--positions-file", str(positions_file))
    inject_default_arg("--output-dir", str(output_dir))
    inject_default_arg("--external-score-column", "latest_score")
    inject_default_arg("--external-target-weight-column", "target_weight")

    if (
        not legacy_ml
        and not explicit_model_artifact
        and not explicit_external_candidate
        and not candidate_profile
        and not is_help_request()
    ):
        candidate_profile = DEFAULT_EXECUTION_CANDIDATE_PROFILE

    pool_name_hint = ""
    if candidate_profile:
        try:
            pool_name_hint = str(get_profile(candidate_profile).liquidity_pool_name or "").strip()
        except Exception:
            pool_name_hint = ""

    ensure_default_pool_argument(pool_name=pool_name_hint)
    ensure_execution_strategy_defaults()

    if candidate_profile:
        resolved = apply_profile_defaults(
            candidate_profile,
            mode="trade_plan",
            ensure_live_panels=not is_help_request(),
        )
        if not is_help_request():
            print("execution_mode=research_candidate_default")
            print(f"candidate_profile={resolved.name}")
    else:
        inject_default_arg("--model-artifact", str(model_artifact))
        if legacy_ml or explicit_model_artifact:
            print("execution_mode=model_artifact")

    try:
        from daily_research.baseline.generate_daily_trade_plan import main as generate_daily_trade_plan_main
    except ModuleNotFoundError as exc:
        raise missing_runtime_dependency_error(
            exc,
            command_hint="python daily_research/execution/run_trade_plan.py",
        ) from exc

    generate_daily_trade_plan_main()


if __name__ == "__main__":
    main()
