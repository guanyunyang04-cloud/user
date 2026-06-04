from __future__ import annotations

from pathlib import Path

import pandas as pd

from traditional_quant_research.experiments.short_limitup_strategy_search import (
    conservative_stop_target_return,
    materialize_strategy_events,
    run_short_limitup_strategy_search,
    search_limitup_strategies,
)


def _events() -> pd.DataFrame:
    rows = []
    for idx in range(8):
        year = 2025 if idx < 4 else 2026
        rows.append(
            {
                "date": f"{year}-01-{idx % 4 + 2:02d}",
                "year": year,
                "code": f"A{idx}",
                "next_gap_bucket": "gap_0_to_3" if idx % 2 == 0 else "gap_ge_6",
                "entry_access": "normal",
                "board_stage": "third_board" if idx >= 4 else "first_board",
                "one_word_limit_like": idx >= 4,
                "near_one_word_limit_like": idx >= 4,
                "entry_limit_up": idx % 2 == 0 or idx >= 4,
                "entry_one_word_limit": False,
                "entry_day_close_ret_pct": 9.9 if idx % 2 == 0 else -3.0,
                "first_sell_open_ret_pct": 2.0 if idx % 2 == 0 else -4.0,
                "first_sell_close_ret_pct": 6.0 if idx % 2 == 0 else -5.0,
                "sell1_close_ret_pct": 6.0 if idx % 2 == 0 else -5.0,
                "sell1_max_high_pct": 12.0 if idx % 2 == 0 else 3.0,
                "sell1_min_low_pct": -2.0 if idx % 2 == 0 else -8.0,
                "sell3_close_ret_pct": 18.0 if idx % 2 == 0 else -7.0,
                "sell3_max_high_pct": 25.0 if idx % 2 == 0 else 4.0,
                "sell3_min_low_pct": -3.0 if idx % 2 == 0 else -9.0,
            }
        )
    return pd.DataFrame(rows)


def test_conservative_stop_target_counts_stop_first_when_both_hit() -> None:
    returns = conservative_stop_target_return(
        pd.Series([3.0, 4.0, -1.0]),
        pd.Series([20.0, 12.0, 2.0]),
        pd.Series([-6.0, -1.0, -8.0]),
        stop_loss=5,
        target=10,
    )

    assert returns.tolist() == [-5.0, 10.0, -5.0]


def test_search_limitup_strategies_respects_t1_confirmation_filters() -> None:
    grid, yearly = search_limitup_strategies(
        _events(),
        sell_windows=(1, 3),
        stop_losses=(5,),
        targets=(10,),
    )

    row = grid.loc[grid["setup_name"].eq("normal_entry_t1_limit_up") & grid["sell_window"].eq(1)].iloc[0]
    assert row["n"] == 6
    assert row["mean_ret_pct"] > 0
    assert row["positive_year_rate"] == 1.0
    assert set(yearly["year"]) == {2025, 2026}


def test_materialize_strategy_events_adds_managed_return() -> None:
    events = _events()
    strategy = {
        "setup_name": "normal_entry_t1_limit_up",
        "sell_window": 1,
        "stop_loss": 5.0,
        "target": 10.0,
    }

    selected = materialize_strategy_events(events, strategy)

    assert "managed_return_pct" in selected.columns
    assert selected["entry_limit_up"].all()
    assert len(selected) == 6


def test_run_short_limitup_strategy_search_writes_artifacts(tmp_path: Path) -> None:
    event_file = tmp_path / "events.csv"
    _events().to_csv(event_file, index=False)

    result = run_short_limitup_strategy_search(
        event_file=event_file,
        output_dir=tmp_path / "out",
        sell_windows=(1, 3),
        stop_losses=(5,),
        targets=(10,),
        min_trades=2,
        write_research_log=True,
        research_log_path=tmp_path / "short.md",
    )

    run_dir = Path(result["run_dir"])
    assert result["decision"] == "shortline_backtest_grid_ready"
    assert result["ranked_rows"] > 0
    assert (run_dir / "strategy_grid.csv").exists()
    assert (run_dir / "strategy_grid_ranked.csv").exists()
    assert (run_dir / "strategy_yearly.csv").exists()
    assert (run_dir / "summary.md").exists()
    assert (tmp_path / "short.md").exists()
