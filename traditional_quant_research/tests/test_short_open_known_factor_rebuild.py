from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from traditional_quant_research.experiments import short_open_known_factor_rebuild as mod


def _manual_kama(close: pd.Series) -> pd.Series:
    d1 = (close - close.shift(10)).abs()
    v1 = (close - close.shift(1)).abs().rolling(10, min_periods=10).sum()
    er = (d1 / v1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cs = er * (2.0 / 3.0 - 2.0 / 31.0) + 2.0 / 31.0
    cq = (cs * cs).clip(lower=0.0, upper=1.0)
    kbas = []
    previous = np.nan
    for price, alpha in zip(close.to_numpy(), cq.to_numpy()):
        if np.isnan(previous):
            previous = price
        else:
            previous = alpha * price + (1.0 - alpha) * previous
        kbas.append(previous)
    return pd.Series(kbas, index=close.index).ewm(span=2, adjust=False).mean()


def _raw_panel() -> pd.DataFrame:
    dates = list(pd.bdate_range("2025-01-02", periods=80)) + list(pd.bdate_range("2026-01-02", periods=40))
    rows = []
    for code, base in [("A", 10.0), ("B", 20.0)]:
        previous_close = base
        for index, date in enumerate(dates):
            is_limit_day = index in {30, 65, 92, 112}
            if is_limit_day:
                open_price = previous_close * 0.99
                close = previous_close * 1.10
                high = close
                low = open_price * 0.99
            else:
                open_price = previous_close * (1.0 + 0.001 * ((index % 3) - 1))
                close = open_price * (1.0 + 0.002 * ((index % 5) - 2))
                high = max(open_price, close) * 1.01
                low = min(open_price, close) * 0.99
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "name_on_date": code,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": 100000 + index * 1000,
                    "amount": (100000 + index * 1000) * close,
                    "turn": 3.0 + index % 7,
                    "pctChg": (close / previous_close - 1.0) * 100.0,
                }
            )
            previous_close = close
    return pd.DataFrame(rows)


def _event_rows() -> pd.DataFrame:
    rows = []
    for year in [2025, 2026]:
        for index in range(4):
            rows.append(
                {
                    "date": pd.Timestamp(f"{year}-01-{index + 2:02d}"),
                    "entry_date": pd.Timestamp(f"{year}-01-{index + 3:02d}"),
                    "year": year,
                    "code": f"{year}_{index}",
                    "next_gap_bucket": "gap_0_to_3",
                    "next_open_gap_pct": 1.0,
                    "entry_open_near_limit": False,
                    "entry_limit_up": index % 2 == 0,
                    "one_word_limit_like": False,
                    "near_one_word_limit_like": False,
                    "board_stage": "first_board",
                    "bias_kama_pct": 5.0,
                    "kama_slope_pct": 1.0,
                    "kama_bias_0_3": False,
                    "kama_bias_3_8": True,
                    "open_below_kama_break_limitup": True,
                    "close_cross_atr_upper": True,
                    "kama_slope_positive": True,
                    "signal_ret20_before_pct": 10.0,
                    "signal_vrat5": 1.5,
                    "signal_close_position": 0.9,
                    "limit_up_run_ending_today": 1,
                    "sell1_close_ret_pct": 8.0 if index != 3 else -3.0,
                    "sell1_max_high_pct": 12.0 if index != 3 else 2.0,
                    "sell1_min_low_pct": -2.0 if index != 3 else -4.0,
                    "sell3_close_ret_pct": 12.0,
                    "sell3_max_high_pct": 18.0,
                    "sell3_min_low_pct": -2.0,
                }
            )
    return pd.DataFrame(rows)


def test_kama_formula_matches_user_definition() -> None:
    close = pd.Series([10, 10.1, 10.3, 10.2, 10.5, 10.8, 10.7, 11.0, 11.2, 11.5, 11.4, 11.8], dtype=float)

    actual = mod.compute_kama_frame(close)["kama"]
    expected = _manual_kama(close)

    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_buy_feature_audit_blocks_future_and_t1_close_fields() -> None:
    for profile in ["preopen_submit", "open_print_filter", "executable_only"]:
        mod.audit_buy_feature_columns(mod.feature_columns_for_profile(profile))

    with pytest.raises(ValueError, match="entry_limit_up"):
        mod.audit_buy_feature_columns(["kama_slope_pct", "entry_limit_up"])
    with pytest.raises(ValueError, match="sell1"):
        mod.audit_buy_feature_columns(["kama_slope_pct", "sell1_close_ret_pct"])


def test_event_feature_panel_uses_t1_open_and_t2_sell_window() -> None:
    events = mod.build_event_feature_panel(_raw_panel(), sell_windows=(1, 3))

    assert not events.empty
    row = events.iloc[0]
    assert row["entry_date"] > row["date"]
    assert "entry_day_close_ret_pct" in events.columns
    assert "sell1_close_ret_pct" in events.columns
    assert pd.notna(row["sell1_close_ret_pct"])
    assert row["limit_up_like"] is True or bool(row["limit_up_like"]) is True


def test_event_feature_panel_adds_v2_ma_executable_and_managed_exit_fields() -> None:
    events = mod.build_event_feature_panel(_raw_panel(), sell_windows=(1, 3), min_signal_amount=1.0)

    assert {"ma5_bias_pct", "ma60_slope_pct", "ma_compression_5_60_pct", "executable_entry"}.issubset(events.columns)
    assert "managed_kama_break_sell1_close_ret_pct" in events.columns
    assert "managed_ma5_break_sell3_holding_days" in events.columns
    assert events["executable_entry"].fillna(False).any()


def test_prior_fit_uses_prior_years_only_and_emits_oos_trades() -> None:
    plan, trades = mod.build_prior_fit_oos(
        _event_rows(),
        years=(2026,),
        profile="preopen_submit",
        sell_windows=(1,),
        stop_losses=(5,),
        targets=(10,),
        fee_bps=30,
        min_train_trades=1,
    )

    row = plan.iloc[0]
    assert row["fit_years"] == "2025"
    assert bool(row["fit_uses_eval_year"]) is False
    assert row["selection_status"] == "selected"
    assert set(trades["eval_year"]) == {2026}
    assert trades["fit_uses_eval_year"].eq(False).all()
    assert row["exit_rule"] in mod.EXIT_RULES
    assert trades["exit_rule"].isin(mod.EXIT_RULES).all()


def test_managed_exit_rule_uses_managed_path_columns() -> None:
    events = _event_rows()
    events["managed_ma5_break_sell1_close_ret_pct"] = 30.0
    events["managed_ma5_break_sell1_max_high_pct"] = 2.0
    events["managed_ma5_break_sell1_min_low_pct"] = -8.0
    events["managed_ma5_break_sell1_holding_days"] = 1.0

    trades = mod.materialize_rule_trades(
        events,
        rule_name="all_limitup",
        sell_window=1,
        exit_rule="ma5_break",
        stop_loss=5,
        target=10,
        fee_bps=30,
        diagnostic_only=False,
        requires_open_known=False,
    )

    assert not trades.empty
    assert trades["exit_rule"].eq("ma5_break").all()
    assert trades["gross_ret_pct"].eq(-5.0).all()
    assert trades["net_ret_pct"].eq(-5.3).all()
    assert trades["holding_days"].eq(1.0).all()


def test_executable_only_profile_excludes_one_word_near_limit_and_low_liquidity() -> None:
    events = _event_rows()
    events["executable_entry"] = [True, False, False, True, True, False, True, False]
    events["entry_open_near_limit"] = [False, True, False, False, False, False, False, True]
    events["one_word_limit_like"] = [False, False, True, False, False, False, False, False]
    events["near_one_word_limit_like"] = [False, False, False, False, False, True, False, False]

    mask = mod.evaluate_rule(events, "executable_all")
    grid = mod.build_rule_grid(
        events,
        profile="executable_only",
        sell_windows=(1,),
        stop_losses=(5,),
        targets=(10,),
        fee_bps=30,
    )

    assert mask.tolist() == events["executable_entry"].tolist()
    formal = grid.loc[grid["rule_name"].eq("executable_all")]
    assert not formal.empty
    assert formal["executable_trade_rate"].eq(1.0).all()
    assert formal["near_limit_trade_rate"].eq(0.0).all()


def test_diagnostic_one_word_path_cannot_be_formal_candidate() -> None:
    trades = pd.DataFrame(
        [
            {
                "eval_year": year,
                "net_ret_pct": 5.0,
                "diagnostic_only": True,
                "requires_open_known": False,
            }
            for year in mod.DEFAULT_YEARS
        ]
    )
    yearly = mod.summarize_oos_yearly(trades)
    plan = pd.DataFrame({"fit_uses_eval_year": [False]})

    candidate = mod.build_candidate_strategy_summary(
        trades,
        yearly,
        plan,
        profile="preopen_submit",
        expected_years=mod.DEFAULT_YEARS,
        min_trades=1,
    ).iloc[0]

    assert candidate["evidence_grade"] == "shortline_diagnostic_only"
    assert "diagnostic_or_unfilled_path" in candidate["failed_gates"]


def test_open_print_filter_is_marked_open_known_diagnostic() -> None:
    trades = pd.DataFrame(
        [
            {
                "eval_year": year,
                "net_ret_pct": 5.0,
                "diagnostic_only": False,
                "requires_open_known": True,
            }
            for year in mod.DEFAULT_YEARS
        ]
    )
    yearly = mod.summarize_oos_yearly(trades)
    plan = pd.DataFrame({"fit_uses_eval_year": [False]})

    candidate = mod.build_candidate_strategy_summary(
        trades,
        yearly,
        plan,
        profile="open_print_filter",
        expected_years=mod.DEFAULT_YEARS,
        min_trades=1,
    ).iloc[0]

    assert candidate["evidence_grade"] == "open_known_diagnostic"
    assert "open_known_diagnostic" in candidate["failed_gates"]


def test_executable_only_is_marked_open_known_diagnostic_even_when_clean() -> None:
    trades = pd.DataFrame(
        [
            {
                "eval_year": year,
                "net_ret_pct": 5.0,
                "diagnostic_only": False,
                "requires_open_known": False,
                "executable_entry": True,
                "entry_open_near_limit": False,
                "one_word_limit_like": False,
                "near_one_word_limit_like": False,
            }
            for year in mod.DEFAULT_YEARS
        ]
    )
    yearly = mod.summarize_oos_yearly(trades)
    plan = pd.DataFrame({"fit_uses_eval_year": [False]})

    candidate = mod.build_candidate_strategy_summary(
        trades,
        yearly,
        plan,
        profile="executable_only",
        expected_years=mod.DEFAULT_YEARS,
        min_trades=1,
    ).iloc[0]

    assert candidate["evidence_grade"] == "open_known_diagnostic"
    assert "open_known_diagnostic" in candidate["failed_gates"]


def test_portfolio_topk_selects_highest_scored_trade_per_date() -> None:
    trades = pd.DataFrame(
        [
            {"date": "2026-01-02", "code": "A", "trade_score": 2.0, "net_ret_pct": 10.0},
            {"date": "2026-01-02", "code": "B", "trade_score": 1.0, "net_ret_pct": -5.0},
            {"date": "2026-01-03", "code": "C", "trade_score": 3.0, "net_ret_pct": 4.0},
        ]
    )

    summary = mod.build_portfolio_topk_summary(trades, top_k_values=(1, 2))

    top1 = summary.loc[summary["top_k"].eq(1)].iloc[0]
    top2 = summary.loc[summary["top_k"].eq(2)].iloc[0]
    assert top1["mean_period_net_ret_pct"] == pytest.approx(7.0)
    assert top2["mean_period_net_ret_pct"] == pytest.approx(3.25)
    assert "max_drawdown_pct" in summary.columns
    assert "longest_losing_streak" in summary.columns


def test_preopen_score_does_not_use_t1_open_gap_penalty() -> None:
    trades = pd.DataFrame(
        [
            {"kama_slope_pct": 1.0, "signal_close_position": 0.9, "bias_kama_pct": 5.0, "limit_up_run_ending_today": 1, "next_open_gap_pct": 1.0},
            {"kama_slope_pct": 1.0, "signal_close_position": 0.9, "bias_kama_pct": 5.0, "limit_up_run_ending_today": 1, "next_open_gap_pct": 9.0},
        ]
    )
    preopen_score = mod.score_trades(trades)
    open_known_score = mod.score_trades(trades.assign(requires_open_known=True))

    assert preopen_score.iloc[0] == pytest.approx(preopen_score.iloc[1])
    assert open_known_score.iloc[0] > open_known_score.iloc[1]


def test_run_short_open_known_factor_rebuild_writes_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "load_pit_manifest", lambda root=None: {"snapshot_id": "fixture"})
    monkeypatch.setattr(mod, "load_quality_report", lambda root=None: {"snapshot_id": "fixture"})
    monkeypatch.setattr(mod, "load_tradeable_panel", lambda *args, **kwargs: _raw_panel())

    result = mod.run_short_open_known_factor_rebuild(
        years=(2026,),
        profile="preopen_submit",
        output_dir=tmp_path / "out",
        sell_windows=(1,),
        stop_losses=(5,),
        targets=(10,),
        top_k_values=(1, 3),
        fee_bps=30,
        min_train_trades=1,
        min_trades=1,
        write_research_log=True,
        research_log_path=tmp_path / "short_open.md",
    )

    run_dir = Path(result["run_dir"])
    assert result["feature_leakage_audit"] == "passed"
    assert result["min_signal_amount"] == mod.DEFAULT_MIN_SIGNAL_AMOUNT
    assert (run_dir / "event_feature_panel.csv").exists()
    assert (run_dir / "event_feature_panel_v2.csv").exists()
    assert (run_dir / "factor_diagnostics.csv").exists()
    assert (run_dir / "factor_bucket_diagnostics_v2.csv").exists()
    assert (run_dir / "factor_interaction_diagnostics_v2.csv").exists()
    assert (run_dir / "executable_rule_grid_v2.csv").exists()
    assert (run_dir / "prior_fit_rule_plan.csv").exists()
    assert (run_dir / "prior_fit_rule_plan_v2.csv").exists()
    assert (run_dir / "prior_fit_oos_trades.csv").exists()
    assert (run_dir / "prior_fit_oos_trades_v2.csv").exists()
    assert (run_dir / "sell_rule_diagnostics_v2.csv").exists()
    assert (run_dir / "portfolio_topk_summary.csv").exists()
    assert (run_dir / "portfolio_topk_summary_v2.csv").exists()
    assert (run_dir / "candidate_shortline_summary_v2.csv").exists()
    assert (tmp_path / "short_open.md").exists()


def test_stable_cache_builds_and_reuses_year_event_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_loader(*args: object, **kwargs: object) -> pd.DataFrame:
        calls.append((str(kwargs["start_date"]), str(kwargs["end_date"])))
        return _raw_panel()

    monkeypatch.setattr(mod, "load_tradeable_panel", fake_loader)
    cache_dir = tmp_path / "cache"

    first = mod.load_or_build_event_panel(
        root=None,
        selected_years=(2026,),
        event_start_year=2025,
        sell_windows=(1,),
        min_signal_amount=1.0,
        stable_cache=True,
        event_cache_dir=cache_dir,
        event_cache_warmup_years=1,
        rebuild_event_cache=False,
    )
    second = mod.load_or_build_event_panel(
        root=None,
        selected_years=(2026,),
        event_start_year=2025,
        sell_windows=(1,),
        min_signal_amount=1.0,
        stable_cache=True,
        event_cache_dir=cache_dir,
        event_cache_warmup_years=1,
        rebuild_event_cache=False,
    )

    assert not first.empty
    assert not second.empty
    assert calls == [("2024-01-01", "2025-12-31"), ("2025-01-01", "2026-12-31")]
    assert (cache_dir / "event_features_2025.csv").exists()
    assert (cache_dir / "event_features_2026.csv").exists()
    assert (cache_dir / "event_features_2026.meta.json").exists()


def test_event_panel_write_mode_can_limit_or_skip_large_outputs(tmp_path: Path) -> None:
    events = _event_rows()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    mod.write_event_panel_outputs(events, run_dir=run_dir, selected_years=(2026,), mode="selected_years")
    selected = pd.read_csv(run_dir / "event_feature_panel_v2.csv")
    assert set(selected["year"]) == {2026}

    skip_dir = tmp_path / "skip"
    skip_dir.mkdir()
    mod.write_event_panel_outputs(events, run_dir=skip_dir, selected_years=(2026,), mode="none")
    assert not (skip_dir / "event_feature_panel_v2.csv").exists()


def test_build_event_cache_only_writes_cache_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "load_pit_manifest", lambda root=None: {"snapshot_id": "fixture"})
    monkeypatch.setattr(mod, "load_quality_report", lambda root=None: {"snapshot_id": "fixture"})
    monkeypatch.setattr(mod, "load_tradeable_panel", lambda *args, **kwargs: _raw_panel())

    result = mod.run_short_open_known_factor_rebuild(
        years=(2026,),
        profile="preopen_submit",
        output_dir=tmp_path / "out",
        sell_windows=(1,),
        top_k_values=(1,),
        fee_bps=30,
        min_signal_amount=1.0,
        stable_cache=True,
        event_cache_dir=tmp_path / "cache",
        build_event_cache_only=True,
        write_research_log=True,
        research_log_path=tmp_path / "cache.md",
    )

    run_dir = Path(result["run_dir"])
    assert result["decision"] == "event_cache_ready"
    assert result["event_cache_only"] is True
    assert (run_dir / "event_cache_summary.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert not (run_dir / "candidate_shortline_summary_v2.csv").exists()
    assert (tmp_path / "cache" / "event_features_2026.csv").exists()
    assert (tmp_path / "cache.md").exists()
