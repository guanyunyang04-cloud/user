from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _write_snapshot(path: Path) -> None:
    path.write_text(
        "record_type,stock,shares,cost_price,available_cash\n"
        "account,,,,100000\n"
        "position,600000.SH,100,10.5,\n",
        encoding="utf-8",
    )


def _write_trade_plan_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plan_summary.json").write_text(
        json.dumps(
            {
                "signal_date": "2026-05-22",
                "execution_date": "2026-05-25",
                "candidate_label": "short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active",
                "transaction_cost_bps": 3.0,
                "slippage_bps": 7.0,
                "sell_tax_bps": 10.0,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "actions_today.csv").write_text(
        "stock,action,shares,price,est_value,reason,current_weight,target_weight\n"
        "002866.SZ,买入,1500,22.0,33000.0,进入目标组合,0,0.3333333333\n"
        "600000.SH,卖出,100,11.0,1100.0,调出目标组合,0.1,0\n",
        encoding="utf-8-sig",
    )


def test_paper_ledger_initializes_from_current_positions_once(tmp_path: Path) -> None:
    from daily_research.execution import paper_trading

    db_path = tmp_path / "paper_account.sqlite3"
    snapshot = tmp_path / "current_positions.csv"
    _write_snapshot(snapshot)

    first = paper_trading.ensure_ledger(db_path=db_path, snapshot_path=snapshot)
    second = paper_trading.ensure_ledger(db_path=db_path, snapshot_path=snapshot)
    account = paper_trading.account_summary(db_path=db_path)

    assert first["initialized_from_snapshot"] is True
    assert second["initialized_from_snapshot"] is False
    assert account["available_cash"] == 100000.0
    assert account["positions"][0]["stock"] == "600000.SH"
    assert account["positions"][0]["shares"] == 100


def test_register_trade_plan_is_idempotent(tmp_path: Path) -> None:
    from daily_research.execution import paper_trading

    db_path = tmp_path / "paper_account.sqlite3"
    paper_trading.ensure_ledger(db_path=db_path)
    run_dir = tmp_path / "execution" / "output" / "20260522"
    _write_trade_plan_run(run_dir)

    first = paper_trading.register_trade_plan_run(db_path=db_path, run_dir=run_dir)
    second = paper_trading.register_trade_plan_run(db_path=db_path, run_dir=run_dir)
    account = paper_trading.account_summary(db_path=db_path)

    assert first["status"] == "registered"
    assert second["status"] == "already_registered"
    assert first["batch_id"] == second["batch_id"]
    assert account["pending_order_count"] == 2


def test_apply_pending_orders_uses_execution_open_and_keeps_missing_price_pending(tmp_path: Path) -> None:
    from daily_research.execution import paper_trading

    db_path = tmp_path / "paper_account.sqlite3"
    snapshot = tmp_path / "current_positions.csv"
    _write_snapshot(snapshot)
    paper_trading.ensure_ledger(db_path=db_path, snapshot_path=snapshot)
    run_dir = tmp_path / "execution" / "output" / "20260522"
    _write_trade_plan_run(run_dir)
    paper_trading.register_trade_plan_run(db_path=db_path, run_dir=run_dir)

    result = paper_trading.apply_pending_orders(
        db_path=db_path,
        price_lookup={
            "2026-05-25": {
                "002866.SZ": {"open": 22.0, "close": 22.5},
                "600000.SH": {"open": 11.0, "close": 11.2},
            }
        },
        execution_date="2026-05-25",
    )
    account = paper_trading.account_summary(db_path=db_path)

    assert result["filled_order_count"] == 2
    assert result["pending_order_count"] == 0
    assert account["pending_order_count"] == 0
    assert account["positions_by_stock"]["002866.SZ"]["shares"] == 1500
    assert "600000.SH" not in account["positions_by_stock"]
    assert account["available_cash"] < 100000.0


def test_missing_execution_open_keeps_order_pending(tmp_path: Path) -> None:
    from daily_research.execution import paper_trading

    db_path = tmp_path / "paper_account.sqlite3"
    paper_trading.ensure_ledger(db_path=db_path)
    run_dir = tmp_path / "execution" / "output" / "20260522"
    _write_trade_plan_run(run_dir)
    paper_trading.register_trade_plan_run(db_path=db_path, run_dir=run_dir)

    result = paper_trading.apply_pending_orders(
        db_path=db_path,
        price_lookup={"2026-05-25": {"600000.SH": {"open": 11.0}}},
        execution_date="2026-05-25",
    )

    assert result["filled_order_count"] == 0
    assert result["pending_order_count"] == 1
    assert result["blocked_order_count"] == 1
    assert "missing_execution_open" in result["blockers"]


def test_cash_flows_do_not_distort_twr(tmp_path: Path) -> None:
    from daily_research.execution import paper_trading

    db_path = tmp_path / "paper_account.sqlite3"
    paper_trading.ensure_ledger(db_path=db_path)
    paper_trading.record_cash_flow(db_path=db_path, flow_type="deposit", amount=100000, reason="initial")
    first = paper_trading.write_daily_equity(
        db_path=db_path,
        price_lookup={"2026-05-22": {}},
        as_of_date="2026-05-22",
    )
    paper_trading.record_cash_flow(db_path=db_path, flow_type="deposit", amount=50000, reason="top up")
    second = paper_trading.write_daily_equity(
        db_path=db_path,
        price_lookup={"2026-05-25": {}},
        as_of_date="2026-05-25",
    )
    performance = paper_trading.performance_summary(db_path=db_path, start_date="2026-05-22", end_date="2026-05-25")

    assert first["total_equity"] == 100000.0
    assert second["total_equity"] == 150000.0
    assert second["daily_return"] == 0.0
    assert performance["total_return"] == 0.0
    assert performance["net_cash_flow"] == 50000.0


def test_export_snapshot_csv_matches_ledger_state(tmp_path: Path) -> None:
    from daily_research.execution import paper_trading

    db_path = tmp_path / "paper_account.sqlite3"
    snapshot = tmp_path / "current_positions.csv"
    _write_snapshot(snapshot)
    paper_trading.ensure_ledger(db_path=db_path, snapshot_path=snapshot)
    export_path = tmp_path / "exported_positions.csv"

    paper_trading.export_account_snapshot(db_path=db_path, path=export_path)

    text = export_path.read_text(encoding="utf-8")
    assert "account,,,,100000" in text
    assert "position,600000.SH,100,10.5," in text
