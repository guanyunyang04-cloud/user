from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_v4_pit_stock_pool_audit as audit


def test_config_keeps_pit_pool_boundaries_and_forbids_intraday_membership() -> None:
    config = audit._load_config(audit.DEFAULT_STUDY_PATH)

    assert tuple(config["period"]["years"]) == (2023, 2024, 2025)
    assert config["period"]["maximum_consumed_date"] == "2025-12-31"
    assert config["period"]["forbidden_year"] == 2026
    assert config["pool_definition"]["minute_availability_is_membership_filter"] is False


def test_rank01_is_finite_date_local_and_tie_stable() -> None:
    values = np.asarray([3.0, np.nan, 1.0, 1.0], dtype=np.float64)

    result = audit._rank01(values)

    np.testing.assert_allclose(result[[0, 2, 3]], [1.0, 0.25, 0.25])
    assert np.isnan(result[1])
    assert np.all((result[np.isfinite(result)] >= 0.0) & (result[np.isfinite(result)] <= 1.0))


def test_quality_gate_uses_asof_financial_publication_and_does_not_read_minute_flag(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2023-01-03", "2023-01-03"]),
            "symbol": ["A", "B"],
            "industry": ["Tech", "Tech"],
            "list_status": ["L", "L"],
            "list_date": ["2020-01-01", "2020-01-01"],
            "is_st": [False, False],
            "is_suspended": [False, False],
            "is_delisted": [False, False],
            "circ_mv": [2.0e8, 1.0e8],
            "valid_amount_20": [20, 20],
            "amount_median_20": [2.0e7, 1.0e7],
            "roe_avg": [0.2, 0.1],
            "net_profit_margin": [0.2, 0.1],
            "net_profit_yoy": [0.2, 0.1],
            "revenue_yoy": [0.2, 0.1],
            "cash_flow_ps": [2.0, 1.0],
            "debt_to_asset": [0.2, 0.4],
            "report_date": ["2022-09-30", "2022-09-30"],
            # Both records are deliberately represented as already selected
            # rows; the SQL layer is responsible for the strict as-of filter.
            "publish_date": ["2022-10-31", "2022-10-31"],
        }
    )
    frame = pd.concat([frame.iloc[[0]]] * 20, ignore_index=True)
    frame["symbol"] = [f"S{idx:03d}" for idx in range(len(frame))]
    frame["day"] = 0
    frame["symbol_idx"] = np.arange(len(frame), dtype=np.int32)

    class _Book:
        day_count = 1
        symbol_count = len(frame)

    mask = audit._quality_mask(_Book(), frame, tmp_path)

    assert mask.shape == (1, len(frame))
    assert mask.dtype == bool
    assert mask[0].all()


def test_pool_book_builder_preserves_candidate_subset_and_reranks_inside_pool(tmp_path) -> None:
    class _Base:
        day_count = 1
        symbol_count = 4
        rank_panel = np.asarray(
            [
                [
                    [0.9, 0.8, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.7, 0.6, 0.5],
                    [0.8, 0.7, 0.3, 0.2, 0.5, 0.4, 0.7, 0.7, 0.6, 0.5, 0.4],
                    [0.7, 0.6, 0.4, 0.1, 0.6, 0.3, 0.8, 0.6, 0.5, 0.4, 0.3],
                    [0.6, 0.5, 0.5, 0.0, 0.7, 0.2, 0.9, 0.5, 0.4, 0.3, 0.2],
                ]
            ],
            dtype=np.float32,
        )
        signal_date_idx = np.asarray([0], dtype=np.int32)
        candidate_counts = np.asarray([4], dtype=np.int32)
        symbol_values = np.asarray(["A", "B", "C", "D"])
        manifest: ClassVar[dict] = {
            "files": {
                "signal_date_idx": {},
                "next_open_buyable": {},
                "next_open_sellable": {},
            }
        }
        terminal_recovery_fraction = 0.0

        def date_text(self, day: int) -> str:
            return "2023-01-03"

    base = _Base()
    mask = np.asarray([[True, False, True, False]], dtype=bool)
    manifest = audit._build_book(
        base,
        pool_name="test",
        rank_mode="filter_then_pool_rerank",
        pool_mask=mask,
        output_root=tmp_path,
    )

    assert manifest["candidate_row_count"] == 2
    saved_mask = np.load(tmp_path / "signal_books/test__filter_then_pool_rerank/pool_mask.npy")
    assert saved_mask.tolist() == mask.tolist()
    panel = np.load(tmp_path / "signal_books/test__filter_then_pool_rerank/rank_panel.npy")
    assert np.isnan(panel[0, 1]).all()
    assert panel[0, 0, 0] == 1.0
    assert panel[0, 2, 0] == 0.0
