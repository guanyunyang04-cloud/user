from __future__ import annotations

import pandas as pd
import pytest

from traditional_quant_research.experiments.frontier_personal_paper_tracking_bootstrap import (
    build_paper_tracking_candidates,
    build_paper_tracking_log_template,
    build_paper_tracking_protocol,
    build_paper_tracking_review_rules,
    run_frontier_personal_paper_tracking_bootstrap,
)


CANCELLED_PATTERN = "paper tracking is cancelled"


def test_run_bootstrap_is_cancelled(tmp_path) -> None:
    with pytest.raises(RuntimeError, match=CANCELLED_PATTERN):
        run_frontier_personal_paper_tracking_bootstrap(output_dir=tmp_path / "out")


def test_bootstrap_candidate_builder_is_cancelled() -> None:
    with pytest.raises(RuntimeError, match=CANCELLED_PATTERN):
        build_paper_tracking_candidates(pd.DataFrame(), {}, {})


def test_bootstrap_protocol_builder_is_cancelled() -> None:
    with pytest.raises(RuntimeError, match=CANCELLED_PATTERN):
        build_paper_tracking_protocol(
            pd.DataFrame(),
            {},
            min_tracking_periods=6,
            min_tracking_days=120,
            max_paper_drawdown=-0.20,
            max_single_period_loss=-0.12,
            min_paper_excess_return=-0.02,
        )


def test_bootstrap_log_and_review_rule_builders_are_cancelled() -> None:
    with pytest.raises(RuntimeError, match=CANCELLED_PATTERN):
        build_paper_tracking_log_template(pd.DataFrame(), {})
    with pytest.raises(RuntimeError, match=CANCELLED_PATTERN):
        build_paper_tracking_review_rules(
            min_tracking_periods=6,
            min_tracking_days=120,
            max_paper_drawdown=-0.20,
            max_single_period_loss=-0.12,
            min_paper_excess_return=-0.02,
        )
