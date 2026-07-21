from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from daily_research.path_policy import seq100_structured_180x35_2026 as experiment
from daily_research.path_policy import qdp_v2_sequence_path_training as training


ROOT = Path(r"H:/quant_project")


def test_registered_2026_windows_and_d7_boundaries() -> None:
    manifest = json.loads(
        (ROOT / "daily_research/data/research_store/seq100_2026_overlay_v1/manifest.json")
        .read_text(encoding="utf-8")
    )
    windows = experiment.signal_window_indices(manifest["date_values"])
    assert len(windows["full_label_dates"]) == 48
    assert len(windows["strict_dates"]) == 101
    assert len(windows["extended_dates"]) == 121
    assert windows["extended_dates"][-1] == "2026-07-07"
    assert windows["strict_dates"][-1] == "2026-06-08"


def test_account_strategy_ids_are_fixed_and_non_hybrid() -> None:
    assert [spec.strategy_id for spec in experiment.ACCOUNT_SPECS] == [
        "top1_slots1_fixed_d7",
        "top3_slots3_fixed_d7",
    ]
    assert all(spec.top_k <= spec.slots for spec in experiment.ACCOUNT_SPECS)


def test_evaluation_scope_contains_only_the_preselected_model() -> None:
    assert experiment.MODEL_ORDER == (experiment.MODEL_2026_180,)
    jobs = experiment._evaluation_jobs()
    assert len(jobs) == 10
    assert all(args[0] == experiment.MODEL_2026_180 for _, args in jobs)


def test_selected_signals_use_stable_score_then_symbol_tie_break() -> None:
    frame = pd.DataFrame(
        {
            "date_idx": [10, 10, 10],
            "trade_date": ["2026-01-05"] * 3,
            "symbol_idx": [2, 1, 3],
            "score": [0.5, 0.5, 0.4],
            "entry_filled": [True, True, True],
            "symbol": ["B", "A", "C"],
        }
    )
    selected = experiment._selected_signals(frame, top_k=1)
    assert selected[10][0]["symbol_idx"] == 1


def test_extended_structured_score_uses_the_training_path_value_semantics() -> None:
    path = torch.zeros((2, 60, 4), dtype=torch.float32)
    path[0, :, 3] = torch.linspace(0.0, 0.2, 60)
    path[0, :, 1] = path[0, :, 3]
    path[1, :, 3] = torch.linspace(0.0, -0.1, 60)
    path[1, :, 2] = path[1, :, 3]
    actual = experiment._structured_ranking_score(
        {"future_path": path},
        price_anchor="entry_open",
        forward_days=60,
    )
    summary = training._derive_path_summary_numpy(
        path.numpy(),
        price_anchor="entry_open",
    )
    columns = training.derived_path_summary_columns(60, path_dim=4)
    expected = summary[:, columns.index(training.value_column_for_path(60, path_dim=4))]
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_json_safe_maps_nonfinite_metrics_to_null() -> None:
    actual = experiment._json_safe(
        {"finite": np.float64(1.25), "missing": float("nan"), "nested": [float("inf")]}
    )
    assert actual == {"finite": 1.25, "missing": None, "nested": [None]}


def test_extended_entry_mask_matches_the_frozen_full_label_candidates() -> None:
    overlay = json.loads((experiment.OVERLAY_ROOT / "manifest.json").read_text(encoding="utf-8"))
    entry_meta = overlay["execution_masks"]["entry_filled_extended"]
    entry = np.memmap(
        entry_meta["path"],
        dtype="bool",
        mode="r",
        shape=tuple(entry_meta["shape"]),
    )
    frozen_view = json.loads(experiment.FROZEN_2026_VIEW.read_text(encoding="utf-8"))
    candidates = pd.read_parquet(
        frozen_view["candidate_index_path"],
        columns=["trade_date", "symbol_idx", "entry_filled"],
    )
    dates = experiment.signal_window_indices(
        json.loads(experiment.FROZEN_2026_OVERLAY.read_text(encoding="utf-8"))["date_values"]
    )["full_label_dates"]
    mismatch = 0
    for date_idx, trade_date in enumerate(dates):
        rows = candidates[candidates["trade_date"].astype(str).eq(trade_date)]
        mismatch += int(
            np.count_nonzero(
                entry[date_idx, rows["symbol_idx"].to_numpy(dtype=np.int64)]
                != rows["entry_filled"].to_numpy(dtype=bool)
            )
        )
    assert mismatch == 0
    assert float(overlay["candidate_audit"]["native_180_entry_fill_rate"]) > 0.99
