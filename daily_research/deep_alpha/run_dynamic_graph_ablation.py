from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.deep_alpha.dynamic_graph_profiles import (
    DEFAULT_DYNAMIC_GRAPH_PROFILE,
    get_profile,
    list_profile_lines,
)
from daily_research.execution.entrypoint_utils import has_arg, inject_default_arg, inject_flag_arg


def _parse_wrapper_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile", default=DEFAULT_DYNAMIC_GRAPH_PROFILE)
    parser.add_argument("--list-profiles", action="store_true")
    return parser.parse_known_args(sys.argv[1:])


def main() -> None:
    wrapper_args, passthrough = _parse_wrapper_args()
    if wrapper_args.list_profiles:
        print("\n".join(list_profile_lines()))
        return

    sys.argv = [sys.argv[0], *passthrough]
    profile = get_profile(wrapper_args.profile)

    inject_default_arg("--data-source", "tq")
    inject_default_arg("--rolling-liquidity-pool", "liquid800")
    inject_default_arg("--start-date", "20220101")
    inject_default_arg("--benchmark", "000300.SH")
    inject_default_arg("--encoder-family", "patch_transformer")
    inject_default_arg("--score-head-method", "manual")
    inject_default_arg("--valid-days", "252")
    inject_flag_arg("--safe-runtime-profile")

    if profile.dynamic_graph_layer:
        inject_flag_arg("--dynamic-graph-layer")
        inject_default_arg("--dynamic-graph-top-k", str(profile.top_k))
        inject_default_arg("--dynamic-graph-temperature", str(profile.temperature))
        inject_default_arg("--dynamic-graph-industry-boost", str(profile.industry_boost))
        inject_default_arg("--dynamic-graph-style-boost", str(profile.style_boost))

    if not has_arg("--experiment-tag"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        inject_default_arg("--experiment-tag", f"deep_alpha_liquid800_{profile.name}_{timestamp}")

    print(f"dynamic_graph_profile={profile.name}")

    from daily_research.deep_alpha.run_deep_alpha_research import main as run_deep_alpha_research_main

    run_deep_alpha_research_main()


if __name__ == "__main__":
    main()
