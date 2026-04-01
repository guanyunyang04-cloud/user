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
    has_arg,
    inject_default_arg,
    is_help_request,
)
from daily_research.execution.research_candidate_profiles import (
    DEFAULT_EXECUTION_CANDIDATE_PROFILE,
    apply_profile_defaults,
    list_profile_lines,
)


def main():
    exec_dir = bootstrap_execution_paths(__file__)
    output_dir = exec_dir.parent / "output"

    if consume_flag_arg("--list-candidate-profiles"):
        print("\n".join(list_profile_lines()))
        return
    candidate_profile = consume_option_arg("--candidate-profile")

    inject_default_arg("--output-dir", str(output_dir))
    ensure_default_pool_argument()
    inject_default_arg("--regime-ma-window", "50")

    if not candidate_profile and not has_arg("--score-panel-csv") and not has_arg("--target-weight-panel-csv") and not is_help_request():
        candidate_profile = DEFAULT_EXECUTION_CANDIDATE_PROFILE
    if candidate_profile:
        resolved = apply_profile_defaults(candidate_profile, mode="backtest")
        print(f"candidate_profile={resolved.name}")

    if not has_arg("--score-panel-csv") and not has_arg("--target-weight-panel-csv") and not is_help_request():
        raise ValueError(
            "Missing --score-panel-csv or --target-weight-panel-csv. "
            "Pass --candidate-profile, --list-candidate-profiles, or explicit CSV paths."
        )

    from daily_research.baseline.backtest_external_score_panel import main as backtest_external_score_panel_main

    backtest_external_score_panel_main()


if __name__ == "__main__":
    main()
