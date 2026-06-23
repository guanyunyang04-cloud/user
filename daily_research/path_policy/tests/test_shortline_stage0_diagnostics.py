from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from daily_research.path_policy.shortline_stage0_diagnostics import (
    add_shortline_mechanism_scores,
    build_shortline_stage0_diagnostics_from_frame,
)


def _base_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for date in ("2025-01-02", "2025-01-03"):
        for stock, strength, ret1, future1, blocked in [
            ("AAA", 5.0, 0.05, 0.050, 1.0),
            ("BBB", 4.0, 0.03, 0.040, 0.0),
            ("CCC", 3.0, -0.01, 0.020, 0.0),
            ("DDD", 2.0, -0.02, -0.010, 0.0),
            ("EEE", 1.0, -0.03, -0.020, 0.0),
        ]:
            rows.append(
                {
                    "role": "validation",
                    "date": date,
                    "stock": stock,
                    "raw_limit_up_like_1d": strength,
                    "raw_amount_ratio_5_20": strength,
                    "raw_volume_ratio_5_20": strength,
                    "intraday_close_position": strength,
                    "intraday_last_30m_ret": strength,
                    "intraday_close_to_vwap": strength,
                    "intraday_price_above_vwap_share": strength,
                    "intraday_cum_vwap_slope": strength,
                    "intraday_close_pressure_30m": strength,
                    "industry_ret_5_excess": strength,
                    "industry_positive_share_5": strength,
                    "stock_ret_5_minus_industry": strength,
                    "ret_1d": ret1,
                    "raw_close_from_prev_close_1d": ret1,
                    "ret_20d": strength,
                    "cs_rank_ret_20d": strength,
                    "stock_ret_20_minus_industry": strength,
                    "intraday_low_to_close_ret": strength,
                    "industry_rank_ret_5": strength,
                    "industry_rank_turn": strength,
                    "industry_member_count_log": strength,
                    "turn_ratio_5_20": strength,
                    "entry_tradeable": 1.0,
                    "entry_limit_up_buy_blocked": blocked,
                    "entry_suspended_or_no_open": 0.0,
                    "future_cum_return_1d": future1,
                    "future_cum_excess_return_1d": future1 - 0.005,
                    "future_rank_1d": 0.9 if future1 > 0 else 0.2,
                    "future_cum_return_2d": future1 + 0.01,
                    "future_cum_excess_return_2d": future1 + 0.005,
                    "future_rank_2d": 0.9 if future1 > 0 else 0.2,
                }
            )
    return rows


def test_shortline_scores_rank_strong_continuation() -> None:
    frame = pd.DataFrame(_base_rows())

    scored = add_shortline_mechanism_scores(frame)

    one_day = scored[scored["date"] == "2025-01-02"].sort_values("score_strong_continuation", ascending=False)
    assert one_day.iloc[0]["stock"] == "AAA"
    assert one_day.iloc[-1]["stock"] == "EEE"
    assert scored["score_combined_shortline"].between(0.0, 1.0).all()


def test_stage0_diagnostics_filters_blocked_entry_and_applies_tplus1_cost(tmp_path: Path) -> None:
    frame = pd.DataFrame(_base_rows())

    report = build_shortline_stage0_diagnostics_from_frame(
        frame=frame,
        output_root=tmp_path / "stage0",
        top_ks=(1,),
        label_horizons=(1,),
        score_columns=("score_strong_continuation",),
        round_trip_cost_bps=20.0,
        filter_entry=True,
    )

    assert report["status"] == "completed"
    summary = pd.read_csv(report["outputs"]["summary_csv"])
    row = summary.iloc[0]
    assert row["exit_label"] == "D+2_open"
    assert row["top_k"] == 1
    assert row["gross_abs_mean"] == pytest.approx(0.040)
    assert row["net_abs_mean"] == pytest.approx(0.038)
    assert row["candidate_blocked_rate_mean"] == pytest.approx(1.0)
    assert Path(report["outputs"]["daily_csv"]).exists()
    assert Path(report["outputs"]["monthly_csv"]).exists()
    assert Path(report["outputs"]["report_json"]).exists()


def test_stage0_diagnostics_keeps_multiple_executable_open_exits(tmp_path: Path) -> None:
    frame = pd.DataFrame(_base_rows())

    report = build_shortline_stage0_diagnostics_from_frame(
        frame=frame,
        output_root=tmp_path / "stage0_multi",
        top_ks=(1, 3),
        label_horizons=(1, 2),
        score_columns=("score_strong_continuation",),
        round_trip_cost_bps=0.0,
        filter_entry=True,
    )

    summary = pd.read_csv(report["outputs"]["summary_csv"])
    assert set(summary["exit_label"].astype(str)) == {"D+2_open", "D+3_open"}
    assert set(summary["top_k"].astype(int)) == {1, 3}
    saved = pd.read_csv(report["outputs"]["daily_csv"])
    assert not saved.empty
