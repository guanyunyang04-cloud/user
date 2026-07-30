from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from daily_research.path_policy import (
    seq100_v4_economic_realizability as economic,
)
from daily_research.path_policy.seq100_candidate_execution import ExecutionCosts


def _costs() -> ExecutionCosts:
    return ExecutionCosts(
        lot_size=100,
        commission_bps=3.0,
        minimum_commission_cny=5.0,
        transfer_fee_bps=0.1,
        slippage_bps=7.0,
        stress_slippage_multiplier=2.0,
        stamp_tax_schedule=(
            ("1900-01-01", 10.0),
            ("2023-08-28", 5.0),
        ),
    )


def test_preregistered_surface_and_forbidden_year_are_frozen() -> None:
    study = economic.load_study()
    specs = economic.task_specs()

    assert len(specs) == 672
    assert len({spec.task_id for spec in specs}) == 672
    assert study["period"]["signal_years"] == [2023, 2024, 2025]
    assert study["period"]["forbidden_year"] == 2026
    assert study["period"]["uses_2020_2022_policy_outcomes"] is False
    assert tuple(study["policies"]["formal_families"]) == economic.FORMAL_FAMILIES
    assert tuple(study["account"]["slot_counts"]) == economic.SLOT_COUNTS


def test_noise_and_open_execution_masks_are_deterministic() -> None:
    symbols = np.asarray([9, 2, 7, 3], dtype=np.int32)
    first = economic._noise_rank(date_idx=42, symbol_idx=symbols, seed=701)
    second = economic._noise_rank(date_idx=42, symbol_idx=symbols, seed=701)
    changed = economic._noise_rank(date_idx=43, symbol_idx=symbols, seed=701)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, changed)
    assert np.array_equal(np.sort(first), np.linspace(0.0, 1.0, 4, dtype=np.float32))

    sellable = economic.derive_open_sellable(
        raw_open=np.asarray([10.0, 9.0, 8.0, 8.0, math.nan]),
        raw_down_limit=np.asarray([9.0, 9.0, 7.0, 7.0, 1.0]),
        status_valid=np.asarray([True, True, True, False, True]),
        suspended=np.asarray([False, False, True, False, False]),
        delisted=np.asarray([False, False, False, False, False]),
    )
    assert sellable.tolist() == [True, False, False, False, False]


def test_total_return_ledger_uses_adjusted_ratio_and_historical_tax() -> None:
    costs = _costs()
    position, buy = economic._buy_position(
        available_cash=1_000_000.0,
        allocated_cash=1_000_000.0,
        symbol_idx=0,
        signal_date_idx=0,
        execution_date_idx=1,
        raw_open=10.0,
        adjusted_open=20.0,
        costs=costs,
        slippage_multiplier=1.0,
    )
    assert position is not None
    assert position.shares % 100 == 0
    assert buy["commission"] >= 5.0

    gross_value = economic._position_value(position, 30.0)
    assert math.isclose(
        gross_value,
        position.gross_entry_notional * 1.5,
        rel_tol=0.0,
        abs_tol=1.0e-8,
    )
    _, before = economic._sell_position(
        position=position,
        adjusted_open=30.0,
        trade_date="2023-08-25",
        costs=costs,
        slippage_multiplier=1.0,
    )
    _, after = economic._sell_position(
        position=position,
        adjusted_open=30.0,
        trade_date="2023-08-28",
        costs=costs,
        slippage_multiplier=1.0,
    )
    assert math.isclose(
        before["stamp_tax"],
        after["stamp_tax"] * 2.0,
        rel_tol=1.0e-12,
    )


@dataclass
class _PlanningBook:
    study: dict
    rank_panel: np.ndarray
    orders: np.ndarray

    def selector_column(self, family: str) -> int:
        return economic.SignalBook.selector_column(self, family)

    def order_column(self, family: str) -> int:
        return economic.SignalBook.order_column(self, family)

    def symbols_for_day(self, family: str, day: int) -> np.ndarray:
        return self.orders[self.order_column(family), day]

    def rank(self, day: int, symbol_idx: int, column: int) -> float:
        return float(self.rank_panel[day, symbol_idx, column])

    def candidate_count(self, day: int) -> int:
        return int(self.rank_panel.shape[1])


def _planning_book(symbol_count: int = 100) -> _PlanningBook:
    study = economic.load_study()
    ranks = np.zeros((1, symbol_count, len(economic.RANK_COLUMNS)), dtype=np.float32)
    base = np.linspace(0.0, 1.0, symbol_count, dtype=np.float32)
    for column in range(ranks.shape[2]):
        ranks[0, :, column] = base
    orders = np.empty((4, 1, symbol_count), dtype=np.int32)
    descending = np.arange(symbol_count - 1, -1, -1, dtype=np.int32)
    orders[:, 0, :] = descending
    return _PlanningBook(study=study, rank_panel=ranks, orders=orders)


def _position(symbol_idx: int) -> economic.Position:
    return economic.Position(
        symbol_idx=symbol_idx,
        shares=100,
        entry_signal_date_idx=0,
        entry_date_idx=1,
        entry_raw_open=10.0,
        entry_adjusted_open=10.0,
        gross_entry_notional=1_000.0,
        net_cash_outflow=1_010.0,
        gross_cash_outflow=1_000.0,
        last_adjusted_price=10.0,
    )


def test_strong_cash_hysteresis_and_auxiliary_veto_semantics() -> None:
    book = _planning_book()
    spec = economic.TaskSpec(
        family="mfe10_primary",
        exposure_mode="strong_candidate_cash",
        slot_count=1,
        buffer_multiplier=0.0,
        cost_scenario="base",
    )
    # The incumbent is below 0.95 and is forcibly exited; the 1.00 candidate
    # is paired even though no fixed holding period exists.
    book.rank_panel[0, 98, 0] = 0.94
    orders = economic.plan_orders(
        book=book,
        spec=spec,
        day=0,
        positions={98: _position(98)},
    )
    assert len(orders.sells) == 1
    assert orders.sells[0].reason == "below_retention_threshold"
    assert orders.sells[0].paired_buy_symbol_idx == 99

    veto_spec = economic.TaskSpec(
        family="dual_mfe_state_veto",
        exposure_mode="target_full",
        slot_count=1,
        buffer_multiplier=2.0,
        cost_scenario="base",
    )
    # A poor state score never causes an exit by itself.  The incumbent is
    # already the best opportunity, so the state veto has no sell authority.
    book.rank_panel[0, 99, 2] = 1.0
    veto_orders = economic.plan_orders(
        book=book,
        spec=veto_spec,
        day=0,
        positions={99: _position(99)},
    )
    assert veto_orders.sells == ()


class _SimulationBook:
    def __init__(self) -> None:
        self.study = economic.load_study()
        self.signal_date_idx = np.arange(6, dtype=np.int32)
        self.date_values = np.asarray(
            [
                "2023-01-03",
                "2023-12-29",
                "2024-01-02",
                "2024-12-31",
                "2025-01-02",
                "2025-12-31",
            ],
            dtype=str,
        )
        self.symbol_values = np.asarray(["000001.SZ", "000002.SZ", "000003.SZ"])
        self.rank_panel = np.zeros((6, 3, len(economic.RANK_COLUMNS)), dtype=np.float32)
        leaders = (0, 1, 1, 1, 2, 2)
        self.orders = np.empty((4, 6, 3), dtype=np.int32)
        for day, leader in enumerate(leaders):
            order = [leader, *[value for value in range(3) if value != leader]]
            scores = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
            scores[leader] = 1.0
            self.rank_panel[day, :, :] = scores[:, None]
            self.orders[:, day, :] = np.asarray(order, dtype=np.int32)
        self.candidate_counts = np.full(6, 3, dtype=np.int32)
        self.next_buyable = np.ones((6, 3), dtype=bool)
        self.next_sellable = np.ones((6, 3), dtype=bool)
        self.next_sellable[1, 0] = False
        self.next_buyable[2, 1] = False
        self.raw_open = np.full((6, 3), 10.0, dtype=np.float32)
        self.adjusted_open = np.full((6, 3), 10.0, dtype=np.float32)
        self.adjusted_close = np.full((6, 3), 10.0, dtype=np.float32)
        self.amount = np.full((6, 3), 1.0e9, dtype=np.float32)
        self.status_valid = np.ones((6, 3), dtype=bool)
        self.has_bar = np.ones((6, 3), dtype=bool)
        self.is_delisted = np.zeros((6, 3), dtype=bool)
        self.benchmark_returns = np.zeros(6, dtype=np.float64)
        self.costs = _costs()
        self.terminal_recovery_fraction = 0.0
        self.manifest = {"files": {"rank_panel": {"sha256": "synthetic"}}}

    @property
    def day_count(self) -> int:
        return len(self.signal_date_idx)

    def date_text(self, day: int) -> str:
        return str(self.date_values[day])

    def selector_column(self, family: str) -> int:
        return economic.SignalBook.selector_column(self, family)

    def order_column(self, family: str) -> int:
        return economic.SignalBook.order_column(self, family)

    def symbols_for_day(self, family: str, day: int) -> np.ndarray:
        return self.orders[self.order_column(family), day]

    def rank(self, day: int, symbol_idx: int, column: int) -> float:
        return float(self.rank_panel[day, symbol_idx, column])

    def candidate_count(self, day: int) -> int:
        return 3

    def trailing_amount(self, *, signal_date_idx: int, symbol_idx: int) -> float:
        return 1.0e9


def test_account_replay_obeys_failed_order_and_continuity_semantics() -> None:
    book = _SimulationBook()
    spec = economic.TaskSpec(
        family="mfe10_primary",
        exposure_mode="target_full",
        slot_count=1,
        buffer_multiplier=0.0,
        cost_scenario="base",
    )
    result, equity, attempts, monthly = economic.simulate_task(book=book, spec=spec)

    failed_sell_date = "2024-01-02"
    assert not (
        attempts["execution_date"].astype(str).eq(failed_sell_date)
        & attempts["side"].astype(str).eq("buy")
    ).any()
    failed_buy_date = "2024-12-31"
    buys_on_failed_date = attempts[
        attempts["execution_date"].astype(str).eq(failed_buy_date)
        & attempts["side"].astype(str).eq("buy")
    ]
    assert buys_on_failed_date["status"].tolist() == ["failed"]
    assert buys_on_failed_date["symbol_idx"].tolist() == [1]

    delayed_fill = attempts[
        attempts["side"].astype(str).eq("sell")
        & attempts["status"].astype(str).eq("filled")
        & attempts["symbol_idx"].astype(int).eq(0)
    ]
    assert delayed_fill["sell_delay_days"].astype(int).tolist() == [1]
    assert result["metrics"]["delayed_sell_count"] == 1
    assert result["metrics"]["terminal_recovery_count"] == 0
    assert result["metrics"]["maximum_conservation_error"] < 1.0e-6
    assert "rolling_63d_net_return" in equity
    assert "rolling_63d_relative_excess_return" in equity
    assert set(pd.to_datetime(equity["trade_date"]).dt.year) == {2023, 2024, 2025}
    assert len(monthly) == 6
    assert result["period"]["annual_reset"] is False
    assert result["period"]["forbidden_2026_row_count"] == 0


def test_terminal_lifecycle_uses_pack_recovery_without_2026_price() -> None:
    book = _SimulationBook()
    # Keep symbol 1 through the final date, then mark the PIT lifecycle absent.
    book.orders[:, 4, :] = np.asarray([1, 2, 0], dtype=np.int32)
    book.rank_panel[4, :, :] = np.asarray([0.1, 1.0, 0.2])[:, None]
    book.status_valid[-1, 1] = False
    book.has_bar[-1, 1] = False
    book.adjusted_close[-1, 1] = np.nan
    spec = economic.TaskSpec(
        family="mfe10_primary",
        exposure_mode="target_full",
        slot_count=1,
        buffer_multiplier=0.0,
        cost_scenario="base",
    )
    result, equity, attempts, _ = economic.simulate_task(book=book, spec=spec)

    assert result["metrics"]["terminal_recovery_count"] == 1
    assert result["period"]["forbidden_2026_row_count"] == 0
    assert int(equity["position_count"].iloc[-1]) == 0
    terminal = attempts[attempts["status"].astype(str).eq("recovery")]
    assert terminal["reason"].tolist() == ["pack_terminal_lifecycle_recovery"]
    assert np.isfinite(equity["net_equity"].to_numpy(dtype=float)).all()


def test_statistics_and_contiguous_rectangle_decision() -> None:
    assert np.allclose(
        economic.benjamini_hochberg([0.01, 0.04, 0.03]),
        np.asarray([0.03, 0.04, 0.04]),
    )
    frame = pd.DataFrame(
        [
            {
                "slot_count": slot,
                "buffer_multiplier": buffer,
                "passed": slot in {1, 3, 6} and buffer in {0.0, 0.5},
            }
            for slot in economic.SLOT_COUNTS
            for buffer in economic.BUFFER_MULTIPLIERS
        ]
    )
    rectangles = economic.robust_rectangles(
        frame,
        flag="passed",
        slot_width=3,
        buffer_width=2,
    )
    assert rectangles == [
        {
            "slot_counts": [1, 3, 6],
            "buffer_multipliers": [0.0, 0.5],
            "cell_count": 6,
        }
    ]

    rng = np.random.default_rng(7)
    positive = rng.normal(0.001, 0.0001, size=300)
    hac = economic.hac_mean_test(positive, maximum_lag=20)
    bootstrap = economic.moving_block_bootstrap_lower(
        positive,
        block_length=20,
        repetitions=200,
        seed=7,
        lower_quantile=0.10,
    )
    assert hac["one_sided_p_value"] < 0.05
    assert bootstrap["lower_bound"] > 0.0


def test_reporting_tables_preserve_year_and_surface_dimensions() -> None:
    results = [
        {
            "task": {
                "family": "mfe10_primary",
                "exposure_mode": "target_full",
                "slot_count": 3,
                "buffer_multiplier": 0.5,
                "cost_scenario": "base",
            },
            "task_id": "synthetic",
            "annual": [
                {
                    "year": year,
                    "net_return": 0.01,
                    "benchmark_return": 0.005,
                    "relative_excess_return": 0.004975,
                    "ending_equity": 1_010_000.0,
                }
                for year in economic.YEARS
            ],
        }
    ]
    annual = economic._annual_metric_table(results)
    assert annual["year"].tolist() == [2023, 2024, 2025]

    metric_rows = []
    for family in economic.FAMILIES:
        for exposure in economic.EXPOSURE_MODES:
            for slots in economic.SLOT_COUNTS:
                for buffer in economic.BUFFER_MULTIPLIERS:
                    for cost in economic.COST_SCENARIOS:
                        metric_rows.append(
                            {
                                "family": family,
                                "exposure_mode": exposure,
                                "slot_count": slots,
                                "buffer_multiplier": buffer,
                                "cost_scenario": cost,
                                "task_id": f"{family}-{exposure}-{slots}-{buffer}-{cost}",
                                "terminal_cost_accrued_return": 0.01,
                                "terminal_cost_accrued_relative_excess_return": 0.0,
                                "gross_cumulative_return": 0.02,
                                "maximum_drawdown": -0.1,
                                "turnover_to_starting_cash": 2.0,
                                "average_cash_fraction": 0.1,
                                "terminal_recovery_count": 0,
                            }
                        )
    slices = economic._surface_slice_table(pd.DataFrame(metric_rows))
    assert set(slices["slice_name"]) == {
        "family_exposure_cost",
        "slot_exposure_cost",
        "buffer_exposure_cost",
        "cost_exposure",
    }
    assert int(slices["task_count"].sum()) > 0
