from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from quantlab.research.minute_v2.builder import (
    _checkpoint_specs_compatible,
    _protect_existing_manifest,
)
from quantlab.research.minute_v2.contracts import (
    DAILY_WINDOWS,
    EXPECTED_DECISION_BARS,
    MODEL_FEATURE_COLUMNS,
    MinuteV2Config,
    MinuteV2Error,
    is_decision_bar,
)
from quantlab.research.minute_v2.features import build_feature_frame
from quantlab.research.minute_v2.labels import build_label_frame
from quantlab.research.minute_v2.mining import mine_formula_features
from quantlab.research.minute_v2.models import (
    evaluate_scores,
    fit_ridge,
    fit_ridge_chunks,
    rule_score,
)
from quantlab.research.minute_v2.replay import EventReplayConfig, replay_events
from quantlab.research.minute_v2.sampling import (
    EVENT_CONTEXT_COLUMNS,
    EVENT_FLAG_COLUMNS,
    audit_candidate_recall,
    build_event_frame,
)
from quantlab.research.minute_v2.source import stock_day_query
from quantlab.research.minute_v2.training import (
    _collect_cross_section_sample,
    _evaluate_score_file,
    _load_top_scored,
    _write_scored_period,
)


def _times() -> list[str]:
    morning = pd.date_range("2000-01-01 09:31", "2000-01-01 11:30", freq="min")
    afternoon = pd.date_range("2000-01-01 13:01", "2000-01-01 15:00", freq="min")
    return [*morning.strftime("%H%M00000"), *afternoon.strftime("%H%M00000")]


def test_checkpoint_reuses_parts_when_only_resource_controls_change() -> None:
    current = {
        "schema": "quantlab.minute_v2_day_parts/3",
        "source_dataset_ids": {"minute": "stable"},
        "year": 2022,
        "month": 6,
        "keep_base": True,
        "extended_end": "2022-07-08",
        "config": {
            "processing_days_per_chunk": 1,
            "duckdb_threads": 2,
            "memory_floor_gib": 4.0,
            "duckdb_memory_limit_gib": 1.0,
            "candidate_background_percent": 1,
        },
    }
    requested = {
        **current,
        "config": {
            **current["config"],
            "duckdb_threads": 4,
            "memory_floor_gib": 1.0,
            "duckdb_memory_limit_gib": 2.0,
        },
    }
    assert _checkpoint_specs_compatible(current, requested)
    requested["config"]["candidate_background_percent"] = 2
    assert not _checkpoint_specs_compatible(current, requested)


def test_builder_refuses_old_month_manifest_without_force(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"schema":"quantlab.minute_v2_month/1","status":"ok"}',
        encoding="utf-8",
    )
    with pytest.raises(MinuteV2Error, match="requires_force"):
        _protect_existing_manifest(manifest, force=False)
    _protect_existing_manifest(manifest, force=True)


def _bars(
    *,
    dates: tuple[str, ...] = ("2022-06-01",),
    symbols: tuple[str, ...] = ("600000.SH", "000001.SZ"),
) -> pd.DataFrame:
    rows = []
    for date_index, trade_date in enumerate(dates):
        for symbol_index, symbol in enumerate(symbols):
            for minute_index, bar_time in enumerate(_times()):
                price = 10.0 + symbol_index + date_index * 0.1 + minute_index * 0.001
                volume = 1000.0 + minute_index + symbol_index * 10
                rows.append(
                    {
                        "symbol": symbol,
                        "trade_date": trade_date,
                        "bar_time": bar_time,
                        "open": price,
                        "high": price + 0.01,
                        "low": price - 0.01,
                        "close": price + 0.005,
                        "volume": volume,
                        "amount": volume * (price + 0.002),
                    }
                )
    return pd.DataFrame(rows)


def _stock_days(
    *,
    trade_date: str = "2022-06-01",
    symbols: tuple[str, ...] = ("600000.SH", "000001.SZ"),
) -> pd.DataFrame:
    rows = []
    for symbol_index, symbol in enumerate(symbols):
        row = {
                "symbol": symbol,
                "trade_date": trade_date,
                "industry_name": f"industry_{symbol_index}",
                "adjust_factor": 1.0,
                "previous_adjust_factor": 1.0,
                "previous_close": 9.9 + symbol_index,
                "auction_price": 10.0 + symbol_index,
                "auction_amount": 1_000_000.0,
                "previous_return_1d": 0.01,
                "previous_amount_20d": 200_000_000.0,
                "history_120d_available": True,
                "history_240d_available": True,
                "previous_total_share": 2_000_000_000.0,
                "previous_float_share": 1_000_000_000.0,
                "previous_total_mv": 20_000_000.0,
                "previous_circ_mv": 10_000_000.0,
                "previous_turnover_rate": 1.2,
                "previous_pe": 12.0,
                "previous_pb": 1.1,
                "corporate_action_today": False,
                "cash_dividend_per_10": 0.0,
                "bonus_share_per_10": 0.0,
                "transfer_share_per_10": 0.0,
                "daily_liquidity_rank": 0.8,
                "exclude_open": False,
                "exclude_high": symbol_index == 1,
                "exclude_low": False,
                "exclude_close": False,
            }
        for window in DAILY_WINDOWS:
            row[f"previous_return_{window}d"] = window / 10_000.0
            row[f"previous_close_to_sma_{window}d"] = window / 20_000.0
            row[f"previous_volatility_{window}d"] = 0.01 + window / 100_000.0
            row[f"previous_amount_ratio_{window}d"] = window / 50_000.0
        rows.append(row)
    return pd.DataFrame(rows)


def _stock_day_source_frames() -> tuple[dict[str, pd.DataFrame], list[str]]:
    dates = pd.bdate_range("2022-01-03", periods=62).strftime("%Y-%m-%d").tolist()
    symbol = "600000.SH"
    daily = pd.DataFrame(
        {
            "symbol": symbol,
            "trade_date": dates,
            "close": np.linspace(10.0, 10.61, len(dates)),
            "amount": 100_000_000.0 + np.arange(len(dates)) * 100_000.0,
        }
    )
    factors = pd.DataFrame(
        {"symbol": symbol, "trade_date": dates, "adjust_factor": 1.0}
    )
    capital = pd.DataFrame(
        {
            "symbol": symbol,
            "trade_date": dates,
            "total_share": 2_000_000_000.0,
            "float_share": 1_000_000_000.0,
        }
    )
    valuation = pd.DataFrame(
        {
            "symbol": symbol,
            "trade_date": dates,
            "total_mv": 20_000_000.0 + np.arange(len(dates)),
            "circ_mv": 10_000_000.0 + np.arange(len(dates)),
            "turnover_rate": 1.0,
            "pe": 12.0,
            "pb": 1.2,
        }
    )
    targets = dates[59:61]
    frames = {
        "trading_calendar": pd.DataFrame(
            {"trade_date": dates, "is_open": True, "exchange": "SSE"}
        ),
        "adjust_factor": factors,
        "daily_raw": daily,
        "share_capital": capital,
        "valuation": valuation,
        "universe_snapshot": pd.DataFrame(
            {"symbol": symbol, "trade_date": targets, "board": "main"}
        ),
        "security_status": pd.DataFrame(
            {
                "symbol": symbol,
                "trade_date": targets,
                "is_st": False,
                "is_suspended": False,
                "is_delisted": False,
            }
        ),
        "industry_concept": pd.DataFrame(
            {
                "symbol": symbol,
                "trade_date": targets,
                "industry_name": "bank",
                "industry": "bank",
            }
        ),
        "opening_auction": pd.DataFrame(
            {
                "symbol": symbol,
                "trade_date": targets,
                "volume": 1000.0,
                "amount": 10_000.0,
            }
        ),
        "corporate_actions": pd.DataFrame(
            {
                "symbol": [symbol],
                "trade_date": [targets[1]],
                "announcement_date": [targets[0]],
                "ex_date": [targets[1]],
                "cash_dividend_per_10": [1.0],
                "bonus_share_per_10": [0.0],
                "transfer_share_per_10": [0.0],
            }
        ),
        "minute_feature_exclusions": pd.DataFrame(
            {
                "symbol": pd.Series(dtype="string"),
                "trade_date": pd.Series(dtype="string"),
                "exclude_open": pd.Series(dtype=bool),
                "exclude_high": pd.Series(dtype=bool),
                "exclude_low": pd.Series(dtype=bool),
                "exclude_close": pd.Series(dtype=bool),
            }
        ),
        "session_feature_exclusions": pd.DataFrame(
            {
                "symbol": pd.Series(dtype="string"),
                "trade_date": pd.Series(dtype="string"),
                "exclusion_reason": pd.Series(dtype="string"),
            }
        ),
    }
    return frames, dates


def _run_stock_day_fixture(frames: dict[str, pd.DataFrame], dates: list[str]) -> pd.DataFrame:
    con = duckdb.connect(":memory:")
    try:
        for name, frame in frames.items():
            con.register(name, frame)
        return con.execute(
            stock_day_query(
                start_date=dates[59],
                end_date=dates[60],
                config=MinuteV2Config(),
            )
        ).fetchdf()
    finally:
        con.close()


def test_stock_pool_requires_sixty_prior_days_and_lags_daily_state() -> None:
    frames, dates = _stock_day_source_frames()
    original = _run_stock_day_fixture(frames, dates)
    assert original["trade_date"].astype(str).tolist() == [dates[60]]
    assert not bool(original.iloc[0]["history_120d_available"])
    assert bool(original.iloc[0]["corporate_action_today"])
    mutated = {name: frame.copy() for name, frame in frames.items()}
    current = mutated["daily_raw"]["trade_date"].eq(dates[60])
    mutated["daily_raw"].loc[current, ["close", "amount"]] *= 100.0
    mutated["valuation"].loc[
        mutated["valuation"]["trade_date"].eq(dates[60]),
        ["total_mv", "circ_mv", "turnover_rate", "pe", "pb"],
    ] *= 100.0
    after = _run_stock_day_fixture(mutated, dates)
    assert_frame_equal(original, after, check_exact=True)


def test_decision_grid_excludes_lunch_and_closing_auction() -> None:
    selected = [value for value in _times() if is_decision_bar(value)]
    assert len(selected) == EXPECTED_DECISION_BARS
    assert "112900000" in selected
    assert "113000000" not in selected
    assert "145500000" in selected
    assert "145600000" not in selected


def test_features_are_causal_and_field_masks_are_specific() -> None:
    dates = ("2022-05-30", "2022-05-31", "2022-06-01")
    bars = _bars(dates=dates)
    stock_days = _stock_days()
    original = build_feature_frame(bars, stock_days)
    mutated_bars = bars.copy()
    future = mutated_bars["trade_date"].eq("2022-06-01") & mutated_bars["bar_time"].gt(
        "100000000"
    )
    mutated_bars.loc[future, ["open", "high", "low", "close"]] *= 8.0
    mutated_bars.loc[future, ["volume", "amount"]] *= 5.0
    mutated = build_feature_frame(mutated_bars, stock_days)
    columns = ["symbol", "trade_date", "bar_time", *MODEL_FEATURE_COLUMNS]
    before = original.loc[original["bar_time"] <= "100000000", columns].reset_index(drop=True)
    after = mutated.loc[mutated["bar_time"] <= "100000000", columns].reset_index(drop=True)
    assert_frame_equal(before, after, check_exact=True)
    masked = original.loc[original["symbol"] == "000001.SZ"]
    assert masked["cumulative_range"].isna().all()
    assert masked["breakout_20m"].isna().all()
    assert masked["return_5m"].notna().sum() > 0
    assert masked["rolling_range_20m"].isna().all()
    first = original.loc[original["symbol"] == "600000.SH"].set_index("bar_time")
    assert bool(first.loc["093100000", "crossed_overnight_from_previous_bar"])
    assert np.isfinite(first.loc["093100000", "return_240m"])
    assert first.loc["093100000", "m60_close_to_sma_5bar"] == first.loc[
        "102900000", "m60_close_to_sma_5bar"
    ]
    assert first.loc["102900000", "m60_close_to_sma_5bar"] != first.loc[
        "103000000", "m60_close_to_sma_5bar"
    ]
    assert np.isfinite(first.loc["093100000", "return_120m"])
    assert np.isfinite(first.loc["130100000", "return_120m"])
    assert np.isnan(masked.set_index("bar_time").loc["103000000", "m60_range_5bar"])

    prior_mutated = bars.copy()
    prior_tail = prior_mutated["trade_date"].eq("2022-05-31") & prior_mutated[
        "bar_time"
    ].ge("145000000")
    prior_mutated.loc[prior_tail, ["open", "high", "low", "close"]] *= 1.02
    changed = build_feature_frame(prior_mutated, stock_days)
    original_open = original.loc[
        (original["symbol"] == "600000.SH") & (original["bar_time"] == "093100000"),
        "moving_average_deviation_240m",
    ].iloc[0]
    changed_open = changed.loc[
        (changed["symbol"] == "600000.SH") & (changed["bar_time"] == "093100000"),
        "moving_average_deviation_240m",
    ].iloc[0]
    assert original_open != changed_open


def test_candidate_gate_is_deterministic_and_keeps_every_decision_minute() -> None:
    features = build_feature_frame(_bars(), _stock_days())
    first = build_event_frame(features)
    second = build_event_frame(features)
    assert_frame_equal(first, second)
    assert 0 < len(first) <= len(features)
    assert first.groupby(["trade_date", "bar_time"]).ngroups == EXPECTED_DECISION_BARS
    assert first.groupby(["trade_date", "bar_time"]).size().ge(1).all()
    assert first["candidate_selected"].all()
    assert first["event_mask"].ge(0).all()
    assert not any(column.startswith(("label_", "entry_", "actual_exit")) for column in first.columns)
    assert set(EVENT_CONTEXT_COLUMNS).issubset(first.columns)
    assert set(EVENT_FLAG_COLUMNS).issubset(first.columns)
    assert not set(MODEL_FEATURE_COLUMNS).difference(EVENT_CONTEXT_COLUMNS).intersection(first.columns)


def test_candidate_recall_audit_uses_keys_and_reports_top_k_hits() -> None:
    base = pd.DataFrame(
        [
            {"symbol": "A", "trade_date": "2022-06-01", "bar_time": "093100000"},
            {"symbol": "B", "trade_date": "2022-06-01", "bar_time": "093100000"},
            {"symbol": "C", "trade_date": "2022-06-01", "bar_time": "093100000"},
            {"symbol": "A", "trade_date": "2022-06-01", "bar_time": "093200000"},
            {"symbol": "B", "trade_date": "2022-06-01", "bar_time": "093200000"},
            {"symbol": "C", "trade_date": "2022-06-01", "bar_time": "093200000"},
        ]
    )
    candidates = base.loc[
        ((base["symbol"] == "B") & (base["bar_time"] == "093100000"))
        | ((base["symbol"] == "B") & (base["bar_time"] == "093200000"))
    ].copy()
    outcomes = base.copy()
    outcomes["label_return_5m"] = [0.01, 0.03, 0.02, 0.03, 0.01, 0.02]
    result = audit_candidate_recall(base, candidates, outcomes)
    assert result["group_count"] == 2
    assert result["groups_without_candidates"] == 0
    assert result["top_k_hits"]["top_1"] == 1
    assert result["top_k_recall"]["top_1"] == 0.5
    assert result["top_k_recall"]["top_3"] == 1.0


def test_labels_use_next_bar_and_next_market_day() -> None:
    bars = _bars(dates=("2022-06-01", "2022-06-02"), symbols=("600000.SH",))
    features = build_feature_frame(
        bars.loc[bars["trade_date"] == "2022-06-01"],
        _stock_days(symbols=("600000.SH",)),
    )
    events = features.loc[features["bar_time"] == "093500000"].copy()
    label_stock_days = pd.DataFrame(
        [
            {
                "symbol": "600000.SH",
                "trade_date": "2022-06-02",
                "high": 10.4,
                "low": 10.0,
                "close": 10.3,
                "adjust_factor": 1.0,
                "previous_close": 10.2,
                "previous_adjust_factor": 1.0,
                "is_suspended": False,
                "is_delisted": False,
                "exclude_high": False,
                "exclude_low": False,
                "exclude_close": False,
                "corporate_action_count": 0,
            }
        ]
    )
    calendar = pd.DataFrame(
        [
            {"trade_date": "2022-06-01", "calendar_index": 1, "next_trade_date": "2022-06-02"},
            {"trade_date": "2022-06-02", "calendar_index": 2, "next_trade_date": None},
        ]
    )
    labels = build_label_frame(
        events,
        bars.loc[bars["trade_date"] == "2022-06-01"],
        bars,
        label_stock_days,
        calendar,
    )
    row = labels.iloc[0]
    assert row["entry_bar_time"] == "093600000"
    assert row["planned_exit_date"] == "2022-06-02"
    assert row["actual_exit_date"] == "2022-06-02"
    assert bool(row["entry_executable"])
    assert bool(row["label_observed"])
    assert np.isfinite(row["label_net_return"])
    assert row["label_net_return"] < row["label_gross_return"]
    assert bool(row["label_5m_observed"])
    assert bool(row["label_1d_observed"])


def test_minute_labels_are_near_close_safe_and_masks_are_field_specific() -> None:
    bars = _bars(dates=("2022-06-01", "2022-06-02"), symbols=("600000.SH",))
    stock_days = _stock_days(symbols=("600000.SH",))
    stock_days.loc[:, "exclude_high"] = True
    features = build_feature_frame(bars, stock_days)
    events = features.loc[
        features["bar_time"].isin(["093500000", "112900000", "145500000"])
    ].copy()
    empty_daily = pd.DataFrame(
        columns=[
            "symbol",
            "trade_date",
            "high",
            "low",
            "close",
            "adjust_factor",
            "previous_close",
            "previous_adjust_factor",
            "is_suspended",
            "is_delisted",
            "exclude_high",
            "exclude_low",
            "exclude_close",
            "corporate_action_count",
        ]
    )
    calendar = pd.DataFrame(
        [{"trade_date": "2022-06-01", "calendar_index": 1, "next_trade_date": None}]
    )
    calendar = pd.DataFrame(
        [
            {"trade_date": "2022-06-01", "calendar_index": 1, "next_trade_date": "2022-06-02"},
            {"trade_date": "2022-06-02", "calendar_index": 2, "next_trade_date": None},
        ]
    )
    labels = build_label_frame(
        events,
        bars.loc[bars["trade_date"] == "2022-06-01"],
        bars,
        empty_daily,
        calendar,
    )
    early = labels.loc[labels["bar_time"] == "093500000"].iloc[0]
    lunch = labels.loc[labels["bar_time"] == "112900000"].iloc[0]
    late = labels.loc[labels["bar_time"] == "145500000"].iloc[0]
    assert bool(early["entry_executable"])
    assert np.isfinite(early["label_return_5m"])
    assert np.isnan(early["label_mfe_5m"])
    assert np.isfinite(early["label_mae_5m"])
    assert lunch["label_end_bar_time_5m"] == "130500000"
    assert bool(lunch["label_5m_crossed_lunch"])
    assert lunch["label_5m_invalid_reason"] == ""
    assert bool(late["label_5m_observed"])
    assert bool(late["label_5m_crossed_overnight"])
    assert late["label_end_date_5m"] == "2022-06-02"
    assert late["label_end_bar_time_5m"] == "093500000"
    assert bool(late["label_session_5m_observed"])
    assert not bool(late["label_session_15m_observed"])
    assert late["label_session_15m_invalid_reason"] == "same_session_window_incomplete"


def test_daily_labels_cross_holidays_and_adjust_for_corporate_actions() -> None:
    calendar_days = [
        "2022-06-01",
        "2022-06-02",
        "2022-06-06",
        "2022-06-07",
        "2022-06-08",
        "2022-06-09",
        "2022-06-10",
        "2022-06-13",
        "2022-06-14",
        "2022-06-15",
        "2022-06-16",
    ]
    target = _bars(symbols=("600000.SH",))
    extended = _bars(dates=("2022-06-01", "2022-06-02"), symbols=("600000.SH",))
    features = build_feature_frame(target, _stock_days(symbols=("600000.SH",)))
    events = features.loc[features["bar_time"] == "093500000"].copy()
    entry_price = float(
        target.loc[target["bar_time"] == "093600000", "amount"].iloc[0]
        / target.loc[target["bar_time"] == "093600000", "volume"].iloc[0]
    )
    daily_rows = []
    for index, trade_date in enumerate(calendar_days[1:], start=1):
        factor = 2.0
        adjusted_close = entry_price if index == 1 else entry_price * (1.0 + index / 100.0)
        raw_close = adjusted_close / factor
        daily_rows.append(
            {
                "symbol": "600000.SH",
                "trade_date": trade_date,
                "high": raw_close * 1.01,
                "low": raw_close * 0.99,
                "close": raw_close,
                "adjust_factor": factor,
                "previous_close": 10.0 if index == 1 else raw_close,
                "previous_adjust_factor": 1.0 if index == 1 else factor,
                "is_suspended": False,
                "is_delisted": False,
                "exclude_high": False,
                "exclude_low": False,
                "exclude_close": False,
                "corporate_action_count": 1 if index == 1 else 0,
            }
        )
    calendar = pd.DataFrame(
        [
            {
                "trade_date": value,
                "calendar_index": index + 1,
                "next_trade_date": calendar_days[index + 1]
                if index + 1 < len(calendar_days)
                else None,
            }
            for index, value in enumerate(calendar_days)
        ]
    )
    labels = build_label_frame(
        events,
        target,
        extended,
        pd.DataFrame(daily_rows),
        calendar,
    )
    row = labels.iloc[0]
    assert row["label_end_date_3d"] == "2022-06-07"
    assert abs(float(row["label_return_1d"])) < 1.0e-12
    assert int(row["label_action_count_1d"]) == 1


def test_replay_does_not_replace_an_unfilled_top_rank() -> None:
    frame = pd.DataFrame(
        [
            {
                "symbol": "600000.SH",
                "trade_date": "2022-06-01",
                "bar_time": "093500000",
                "score": 2.0,
                "planned_exit_date": "2022-06-02",
                "entry_bar_time": "093600000",
                "entry_price": 10.0,
                "entry_amount": 10_000_000.0,
                "entry_executable": False,
                "actual_exit_date": None,
                "exit_amount": np.nan,
                "label_net_return": np.nan,
                "label_observed": False,
            },
            {
                "symbol": "000001.SZ",
                "trade_date": "2022-06-01",
                "bar_time": "093500000",
                "score": 1.0,
                "planned_exit_date": "2022-06-02",
                "entry_bar_time": "093600000",
                "entry_price": 11.0,
                "entry_amount": 10_000_000.0,
                "entry_executable": True,
                "actual_exit_date": "2022-06-02",
                "exit_amount": 10_000_000.0,
                "label_net_return": 0.02,
                "label_observed": True,
            },
        ]
    )
    result, _, trades = replay_events(
        frame,
        replay_config=EventReplayConfig(top_k_per_minute=1),
    )
    assert result["selected_event_count"] == 1
    assert result["unfilled_selection_count"] == 1
    assert trades.empty


def test_rule_ridge_and_formula_mining_have_small_deterministic_contracts() -> None:
    rng = np.random.default_rng(7)
    rows = []
    for day in range(20):
        period = "train" if day < 14 else "validation"
        for minute in ("093500000", "100000000"):
            for symbol_index in range(10):
                signal = rng.normal()
                row = {
                    "symbol": f"{600000 + symbol_index}.SH",
                    "trade_date": f"2022-06-{day + 1:02d}",
                    "bar_time": minute,
                    "period": period,
                    "return_5m": signal,
                    "return_1m": signal * 0.2,
                    "label_net_return": signal * 0.01 + rng.normal(scale=0.001),
                }
                for feature in MODEL_FEATURE_COLUMNS:
                    row.setdefault(feature, rng.normal())
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame["score"] = rule_score(frame)
    assert frame["score"].notna().all()
    ridge = fit_ridge(
        frame.loc[frame["period"] == "train"],
        feature_names=("return_1m", "return_5m"),
        alpha=1.0,
    )
    prediction = ridge.predict(frame.loc[frame["period"] == "validation"])
    assert np.corrcoef(prediction, frame.loc[frame["period"] == "validation", "label_net_return"])[0, 1] > 0.8
    mining = mine_formula_features(
        frame.loc[frame["period"] == "train"],
        frame.loc[frame["period"] == "validation"],
        seed_features=("return_1m", "return_5m"),
        maximum_candidates=12,
        maximum_selected=3,
        minimum_coverage=0.9,
    )
    assert mining["candidate_count"] == 9
    assert mining["selected_count"] > 0


def test_streamed_ridge_matches_batch_fit_when_medians_are_fully_observed() -> None:
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        {
            "label_net_return": rng.normal(size=300),
            "f1": rng.normal(size=300),
            "f2": rng.normal(size=300),
        }
    )
    batch = fit_ridge(
        frame,
        target="label_net_return",
        feature_names=("f1", "f2"),
        alpha=1.0,
    )
    streamed, metadata = fit_ridge_chunks(
        lambda: [frame.iloc[:100], frame.iloc[100:200], frame.iloc[200:]],
        target="label_net_return",
        feature_names=("f1", "f2"),
        alpha=1.0,
        median_sample_size=1000,
    )
    assert metadata["row_count"] == len(frame)
    assert metadata["second_pass_rows"] == len(frame)
    assert np.allclose(batch.coefficients, streamed.coefficients)
    assert np.allclose(batch.means, streamed.means)
    assert np.allclose(batch.scales, streamed.scales)
    assert streamed.intercept == pytest.approx(batch.intercept)


def test_cross_section_sample_cap_is_global_across_input_chunks() -> None:
    chunks = [
        pd.DataFrame(
            {
                "symbol": [f"{part}{index}" for index in range(10)],
                "trade_date": ["2022-06-01"] * 10,
                "bar_time": ["093100000"] * 10,
                "label_net_return": np.arange(10, dtype=float),
            }
        )
        for part in ("A", "B", "C", "D")
    ]
    sample, metadata = _collect_cross_section_sample(
        lambda: chunks,
        rows_per_group=2,
        maximum_rows=100,
    )
    assert metadata["sample_rows"] == 2
    assert sample.groupby(["trade_date", "bar_time"]).size().tolist() == [2]


def test_streamed_score_file_matches_in_memory_metrics(tmp_path) -> None:
    rows = []
    for minute_index, bar_time in enumerate(("093100000", "093200000")):
        for symbol_index in range(6):
            rows.append(
                {
                    "symbol": f"{symbol_index:06d}.SH",
                    "trade_date": "2022-06-01",
                    "bar_time": bar_time,
                    "score": float(symbol_index + minute_index * 0.1),
                    "label_net_return": float(0.01 * (5 - symbol_index) + minute_index * 0.001),
                    "planned_exit_date": "2022-06-02",
                    "entry_bar_time": "093600000",
                    "entry_price": 10.0,
                    "entry_amount": 100_000.0,
                    "entry_executable": True,
                    "actual_exit_date": "2022-06-02",
                    "exit_amount": 100_000.0,
                    "label_observed": True,
                }
            )
    frame = pd.DataFrame(rows)
    path = tmp_path / "scores.parquet"
    metadata = _write_scored_period(
        lambda: [frame.iloc[:5], frame.iloc[5:]],
        lambda value: value["score"],
        path,
    )
    assert metadata["row_count"] == len(frame)
    streamed = _evaluate_score_file(path, top_k=3)
    expected = evaluate_scores(frame, score="score", target="label_net_return", top_k=3)
    for key in (
        "row_count",
        "group_count",
        "rank_ic_mean",
        "rank_ic_positive_fraction",
        "universe_row_mean_net_return",
        "universe_group_mean_net_return",
        "top_k_mean_net_return",
        "top_k_mean_excess_over_group_mean",
        "top_k_positive_excess_group_fraction",
        "daily_top_k_mean_net_return",
        "positive_day_fraction",
    ):
        assert streamed[key] == pytest.approx(expected[key])
    top = _load_top_scored(path, top_k=3)
    assert len(top) == 6
