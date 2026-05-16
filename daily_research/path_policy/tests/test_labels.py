from __future__ import annotations

import pandas as pd

from daily_research.path_policy.labels import build_path20_dataset_frame, build_path20_labels, path20_label_frame_for_date
from daily_research.path_policy.tests.fixtures import make_prepared_policy_inputs


def test_path20_next_open_label_alignment_has_no_same_day_leakage() -> None:
    prepared = make_prepared_policy_inputs(days=30, stocks=("AAA", "BBB"))
    labels = build_path20_labels(prepared, execution_mode="next_open")
    signal_dt = prepared.open_.index[0]
    frame = path20_label_frame_for_date(labels, signal_dt).set_index("stock")

    expected_1d = prepared.open_.loc[prepared.open_.index[2], "AAA"] / prepared.open_.loc[prepared.open_.index[1], "AAA"] - 1.0
    expected_20d = prepared.open_.loc[prepared.open_.index[21], "AAA"] / prepared.open_.loc[prepared.open_.index[1], "AAA"] - 1.0
    bench_1d = prepared.benchmark_open.loc[prepared.open_.index[2]] / prepared.benchmark_open.loc[prepared.open_.index[1]] - 1.0

    assert frame.loc["AAA", "future_return_1d"] == expected_1d
    assert frame.loc["AAA", "future_cum_return_20d"] == expected_20d
    assert frame.loc["AAA", "future_excess_return_1d"] == expected_1d - bench_1d
    assert "2024-01-02" == pd.Timestamp(signal_dt).strftime("%Y-%m-%d")


def test_build_path20_dataset_frame_includes_state_features_and_manifest() -> None:
    prepared = make_prepared_policy_inputs(days=34, stocks=("AAA", "BBB", "CCC"))
    dataset, manifest = build_path20_dataset_frame(
        prepared,
        start_date="20240102",
        end_date="20240119",
        execution_mode="next_open",
        include_state_features=True,
    )

    assert manifest["status"] == "completed"
    assert manifest["row_count"] == len(dataset)
    assert manifest["include_state_features"] is True
    assert not dataset.empty
    assert {"future_return_1d", "future_cum_excess_return_20d", "score_blend", "current_weight"}.issubset(dataset.columns)
