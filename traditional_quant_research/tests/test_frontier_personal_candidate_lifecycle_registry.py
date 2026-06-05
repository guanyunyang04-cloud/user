from __future__ import annotations

import pandas as pd
import pytest

from traditional_quant_research.experiments.frontier_personal_candidate_lifecycle_registry import (
    build_lifecycle_registry,
    run_frontier_personal_candidate_lifecycle_registry,
)


CANCELLED_PATTERN = "paper tracking is cancelled"


def test_run_lifecycle_registry_is_cancelled(tmp_path) -> None:
    with pytest.raises(RuntimeError, match=CANCELLED_PATTERN):
        run_frontier_personal_candidate_lifecycle_registry(output_dir=tmp_path / "out")


def test_lifecycle_registry_builder_is_cancelled() -> None:
    with pytest.raises(RuntimeError, match=CANCELLED_PATTERN):
        build_lifecycle_registry(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
