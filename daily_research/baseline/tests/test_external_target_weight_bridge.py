from __future__ import annotations

import pandas as pd
import pytest

from daily_research.baseline.external_target_weight_bridge import (
    build_target_weight_bridge,
    sanitize_target_weights,
)


def test_raw_target_weight_bridge_preserves_sub_one_gross_without_full_invest() -> None:
    frame = pd.DataFrame(
        {
            "AAA": [0.10, 0.15],
            "BBB": [0.20, 0.05],
            "CCC": [0.00, 0.00],
        },
        index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
    )

    bridged, meta = build_target_weight_bridge(
        frame,
        rebalance_freq="1d",
        top_k=0,
        min_weight=0.0,
        power=1.0,
        full_invest=False,
    )

    pd.testing.assert_frame_equal(bridged, frame.astype(float))
    assert bridged.sum(axis=1).tolist() == pytest.approx([0.30, 0.20])
    assert meta["target_weight_full_invest"] is False


def test_sanitize_target_weights_clips_negative_and_normalizes_overweight_rows() -> None:
    frame = pd.DataFrame(
        {
            "AAA": [0.8, -0.1],
            "BBB": [0.6, 0.4],
            "CCC": [0.0, 0.0],
        },
        index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
    )

    clean = sanitize_target_weights(frame)

    assert clean.loc[pd.Timestamp("2026-01-02"), "AAA"] == 0.8 / 1.4
    assert clean.loc[pd.Timestamp("2026-01-02"), "BBB"] == 0.6 / 1.4
    assert clean.loc[pd.Timestamp("2026-01-05"), "AAA"] == 0.0
    assert clean.loc[pd.Timestamp("2026-01-05")].sum() == 0.4


def test_rebalance_schedule_single_offset_metadata() -> None:
    frame = pd.DataFrame(
        {"AAA": [0.1, 0.2, 0.3, 0.4], "BBB": [0.0, 0.0, 0.0, 0.0]},
        index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]),
    )

    bridged, meta = build_target_weight_bridge(
        frame,
        rebalance_freq="2d",
        rebalance_offset=1,
        rebalance_offset_mode="single",
        top_k=0,
        min_weight=0.0,
        power=1.0,
        full_invest=False,
    )

    assert meta["rebalance_offset_mode"] == "single"
    assert meta["rebalance_offsets"] == [1]
    assert bridged["AAA"].tolist() == [0.0, 0.2, 0.2, 0.4]


def test_rebalance_schedule_all_offsets_metadata() -> None:
    frame = pd.DataFrame(
        {"AAA": [0.1, 0.2, 0.3, 0.4], "BBB": [0.0, 0.0, 0.0, 0.0]},
        index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]),
    )

    bridged, meta = build_target_weight_bridge(
        frame,
        rebalance_freq="2d",
        rebalance_offset_mode="all",
        top_k=0,
        min_weight=0.0,
        power=1.0,
        full_invest=False,
    )

    assert meta["rebalance_offset_mode"] == "all"
    assert meta["rebalance_offsets"] == [0, 1]
    assert meta["rebalance_sleeve_count"] == 2
    assert bridged["AAA"].tolist() == pytest.approx([0.05, 0.15, 0.25, 0.35])
