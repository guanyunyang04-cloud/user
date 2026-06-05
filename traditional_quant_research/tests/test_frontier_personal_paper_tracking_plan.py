from __future__ import annotations

import pandas as pd
import pytest

from traditional_quant_research.experiments.frontier_personal_paper_tracking_plan import (
    build_historical_context,
    build_paper_tracking_live_log_starter,
    build_paper_tracking_plan_calendar,
    render_paper_tracking_plan_markdown,
    run_frontier_personal_paper_tracking_plan,
    summarize_paper_tracking_plan,
)


CANCELLED_PATTERN = "paper tracking is cancelled"


def test_run_plan_is_cancelled(tmp_path) -> None:
    with pytest.raises(RuntimeError, match=CANCELLED_PATTERN):
        run_frontier_personal_paper_tracking_plan(output_dir=tmp_path / "out")


def test_plan_helpers_are_cancelled() -> None:
    with pytest.raises(RuntimeError, match=CANCELLED_PATTERN):
        build_historical_context(pd.DataFrame(), {})
    with pytest.raises(RuntimeError, match=CANCELLED_PATTERN):
        build_paper_tracking_plan_calendar(pd.DataFrame(), pd.DataFrame(), {}, {}, pd.DataFrame())
    with pytest.raises(RuntimeError, match=CANCELLED_PATTERN):
        build_paper_tracking_live_log_starter(pd.DataFrame())
    with pytest.raises(RuntimeError, match=CANCELLED_PATTERN):
        summarize_paper_tracking_plan(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            run_id="plan",
            bootstrap_run_dir=tmp_path_or_dot(),
            bootstrap_summary={},
            combined_summary={},
            plan_periods=6,
            plan_start_date=None,
        )
    with pytest.raises(RuntimeError, match=CANCELLED_PATTERN):
        render_paper_tracking_plan_markdown({}, pd.DataFrame(), pd.DataFrame())


def tmp_path_or_dot():
    return "."
