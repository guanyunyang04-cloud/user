from __future__ import annotations

import pandas as pd
import pytest

from traditional_quant_research.experiments.frontier_personal_paper_tracking_review import (
    evaluate_paper_tracking_log,
    run_frontier_personal_paper_tracking_review,
)


CANCELLED_PATTERN = "paper tracking review is cancelled"


def test_run_review_is_cancelled(tmp_path) -> None:
    with pytest.raises(RuntimeError, match=CANCELLED_PATTERN):
        run_frontier_personal_paper_tracking_review(output_dir=tmp_path / "out")


def test_review_evaluator_is_cancelled() -> None:
    with pytest.raises(RuntimeError, match=CANCELLED_PATTERN):
        evaluate_paper_tracking_log(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
