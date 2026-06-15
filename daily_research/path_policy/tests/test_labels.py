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


def test_path20_basic_v2_labels_cover_all_horizons_and_execution_flags() -> None:
    prepared = make_prepared_policy_inputs(days=30, stocks=("AAA", "BBB", "CCC"))
    prepared.metadata_frames["industry_map"] = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "industry": ["tech", "tech", "bank"],
        }
    )
    signal_dt = prepared.open_.index[0]
    entry_dt = prepared.open_.index[1]
    prepared.membership_frame.loc[entry_dt, "BBB"] = False
    prepared.open_.loc[entry_dt, "AAA"] = prepared.close.loc[signal_dt, "AAA"] * 1.10

    labels = build_path20_labels(prepared, execution_mode="next_open", horizon=5, cumulative_horizons=(1, 3, 5))

    bench_1d = prepared.benchmark_open.loc[prepared.open_.index[2]] / prepared.benchmark_open.loc[entry_dt] - 1.0
    expected_3d_excess = sum(labels.daily_excess_return[step].loc[signal_dt, "AAA"] for step in (1, 2, 3))

    assert labels.metadata["label_schema_version"] == 2
    assert labels.benchmark_daily_return[1].loc[signal_dt] == bench_1d
    assert labels.cumulative_excess_return_1to20[3].loc[signal_dt, "AAA"] == expected_3d_excess
    assert labels.forward_rank_1to20[5].loc[signal_dt].notna().all()
    assert labels.industry_rank_by_horizon[3].loc[signal_dt, "CCC"] == 1.0
    assert labels.entry_tradeable.loc[signal_dt, "AAA"] == 1.0
    assert labels.entry_limit_up_buy_blocked.loc[signal_dt, "AAA"] == 1.0
    assert labels.entry_tradeable.loc[signal_dt, "BBB"] == 0.0
    assert labels.entry_suspended_or_no_open.loc[signal_dt, "BBB"] == 1.0
    assert labels.forward_tradeable_ratio_by_horizon[3].loc[signal_dt, "BBB"] < 1.0


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
