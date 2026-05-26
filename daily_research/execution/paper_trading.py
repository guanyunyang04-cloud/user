from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = "1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_number(value: float | int | None) -> str:
    if value is None:
        return ""
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.8f}".rstrip("0").rstrip(".")


@contextmanager
def _connect(db_path: Path | str) -> Iterator[sqlite3.Connection]:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS cash_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            flow_date TEXT NOT NULL,
            flow_type TEXT NOT NULL,
            amount REAL NOT NULL,
            reason TEXT NOT NULL DEFAULT '',
            external_flow INTEGER NOT NULL DEFAULT 1,
            ref TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS positions (
            stock TEXT PRIMARY KEY,
            shares INTEGER NOT NULL,
            cost_price REAL NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS order_batches (
            batch_id TEXT PRIMARY KEY,
            run_dir TEXT NOT NULL,
            signal_date TEXT NOT NULL,
            execution_date TEXT NOT NULL,
            candidate_label TEXT NOT NULL,
            actions_hash TEXT NOT NULL,
            status TEXT NOT NULL,
            registered_at TEXT NOT NULL,
            summary_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT NOT NULL,
            stock TEXT NOT NULL,
            side TEXT NOT NULL,
            action TEXT NOT NULL,
            requested_shares INTEGER NOT NULL,
            remaining_shares INTEGER NOT NULL,
            reference_price REAL NOT NULL DEFAULT 0,
            target_weight REAL NOT NULL DEFAULT 0,
            status TEXT NOT NULL,
            reason TEXT NOT NULL DEFAULT '',
            execution_date TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(batch_id) REFERENCES order_batches(batch_id)
        );

        CREATE TABLE IF NOT EXISTS fills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            batch_id TEXT NOT NULL,
            stock TEXT NOT NULL,
            side TEXT NOT NULL,
            shares INTEGER NOT NULL,
            price REAL NOT NULL,
            gross_amount REAL NOT NULL,
            cost_amount REAL NOT NULL,
            tax_amount REAL NOT NULL,
            net_cash_delta REAL NOT NULL,
            execution_date TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(order_id) REFERENCES orders(id)
        );

        CREATE TABLE IF NOT EXISTS daily_equity (
            as_of_date TEXT PRIMARY KEY,
            cash REAL NOT NULL,
            market_value REAL NOT NULL,
            total_equity REAL NOT NULL,
            daily_return REAL NOT NULL,
            net_external_flow_since_previous REAL NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
        (SCHEMA_VERSION,),
    )


def _insert_audit(conn: sqlite3.Connection, event_type: str, payload: dict[str, Any]) -> None:
    conn.execute(
        "INSERT INTO audit_events(created_at, event_type, payload_json) VALUES(?, ?, ?)",
        (_now_iso(), event_type, json.dumps(payload, ensure_ascii=False, sort_keys=True)),
    )


def _cash_balance(conn: sqlite3.Connection) -> float:
    row = conn.execute("SELECT COALESCE(SUM(amount), 0) AS cash FROM cash_ledger").fetchone()
    return float(row["cash"] if row else 0.0)


def _record_cash(
    conn: sqlite3.Connection,
    *,
    amount: float,
    flow_type: str,
    reason: str,
    external_flow: bool,
    ref: str = "",
    flow_date: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO cash_ledger(created_at, flow_date, flow_type, amount, reason, external_flow, ref)
        VALUES(?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _now_iso(),
            flow_date or datetime.now().date().isoformat(),
            flow_type,
            float(amount),
            reason,
            1 if external_flow else 0,
            ref,
        ),
    )


def _import_snapshot(conn: sqlite3.Connection, snapshot_path: Path) -> bool:
    if not snapshot_path.exists():
        return False
    imported = conn.execute("SELECT value FROM meta WHERE key='snapshot_imported'").fetchone()
    if imported and imported["value"] == "true":
        return False

    with snapshot_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            record_type = (row.get("record_type") or "").strip().lower()
            if record_type == "account":
                raw_cash = (row.get("available_cash") or "").strip()
                if raw_cash:
                    _record_cash(
                        conn,
                        amount=float(raw_cash),
                        flow_type="initial_snapshot",
                        reason=f"Imported from {snapshot_path.name}",
                        external_flow=True,
                        ref=str(snapshot_path),
                    )
            elif record_type == "position":
                stock = (row.get("stock") or "").strip()
                if not stock:
                    continue
                shares = int(float(row.get("shares") or 0))
                cost_price = float(row.get("cost_price") or 0)
                if shares > 0:
                    conn.execute(
                        """
                        INSERT INTO positions(stock, shares, cost_price, updated_at)
                        VALUES(?, ?, ?, ?)
                        ON CONFLICT(stock) DO UPDATE SET
                            shares=excluded.shares,
                            cost_price=excluded.cost_price,
                            updated_at=excluded.updated_at
                        """,
                        (stock, shares, cost_price, _now_iso()),
                    )

    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('snapshot_imported', 'true')")
    _insert_audit(conn, "snapshot_imported", {"snapshot_path": str(snapshot_path)})
    return True


def ensure_ledger(db_path: Path | str, snapshot_path: Path | str | None = None) -> dict[str, Any]:
    with _connect(db_path) as conn:
        _init_schema(conn)
        initialized = False
        if snapshot_path is not None:
            initialized = _import_snapshot(conn, Path(snapshot_path))
        else:
            conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('snapshot_imported', 'true')"
            )
        return {"db_path": str(Path(db_path)), "initialized_from_snapshot": initialized}


def account_summary(db_path: Path | str) -> dict[str, Any]:
    ensure_ledger(db_path)
    with _connect(db_path) as conn:
        positions = [
            {
                "stock": row["stock"],
                "shares": int(row["shares"]),
                "cost_price": float(row["cost_price"]),
            }
            for row in conn.execute(
                "SELECT stock, shares, cost_price FROM positions WHERE shares > 0 ORDER BY stock"
            )
        ]
        pending_order_count = int(
            conn.execute(
                "SELECT COUNT(*) AS n FROM orders WHERE status IN ('pending', 'partial')"
            ).fetchone()["n"]
        )
        filled_order_count = int(
            conn.execute("SELECT COUNT(*) AS n FROM orders WHERE status='filled'").fetchone()["n"]
        )
        blocked_order_count = int(
            conn.execute("SELECT COUNT(*) AS n FROM orders WHERE status='blocked'").fetchone()["n"]
        )
        recent_fills = [
            dict(row)
            for row in conn.execute(
                """
                SELECT stock, side, shares, price, gross_amount, cost_amount, tax_amount,
                       net_cash_delta, execution_date, created_at
                FROM fills
                ORDER BY id DESC
                LIMIT 20
                """
            )
        ]
        pending_orders = [
            dict(row)
            for row in conn.execute(
                """
                SELECT id, batch_id, stock, side, action, requested_shares, remaining_shares,
                       reference_price, target_weight, status, reason, execution_date
                FROM orders
                WHERE status IN ('pending', 'partial', 'blocked')
                ORDER BY id
                LIMIT 100
                """
            )
        ]
        recent_cash_flows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT flow_date, flow_type, amount, reason, external_flow, ref, created_at
                FROM cash_ledger
                ORDER BY id DESC
                LIMIT 40
                """
            )
        ]
        latest_equity = conn.execute(
            """
            SELECT as_of_date, cash, market_value, total_equity, daily_return
            FROM daily_equity
            ORDER BY as_of_date DESC
            LIMIT 1
            """
        ).fetchone()
        return {
            "db_path": str(Path(db_path)),
            "available_cash": _cash_balance(conn),
            "positions": positions,
            "positions_by_stock": {item["stock"]: item for item in positions},
            "pending_order_count": pending_order_count,
            "filled_order_count": filled_order_count,
            "blocked_order_count": blocked_order_count,
            "pending_orders": pending_orders,
            "recent_fills": recent_fills,
            "recent_cash_flows": recent_cash_flows,
            "latest_equity": dict(latest_equity) if latest_equity else {},
        }


def _actions_hash(actions_path: Path) -> str:
    return hashlib.sha256(actions_path.read_bytes()).hexdigest()


def _batch_id(summary: dict[str, Any], actions_hash: str) -> str:
    raw = "|".join(
        [
            str(summary.get("signal_date") or ""),
            str(summary.get("execution_date") or ""),
            str(summary.get("candidate_label") or ""),
            actions_hash,
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _classify_side(action: str) -> str:
    if "买" in action or "加" in action:
        return "buy"
    if "卖" in action or "减" in action or "清" in action:
        return "sell"
    return "unknown"


def _normalized_shares(side: str, raw_shares: Any) -> int:
    shares = int(float(raw_shares or 0))
    if side == "buy":
        shares = (shares // 100) * 100
    return max(0, shares)


def register_trade_plan_run(db_path: Path | str, run_dir: Path | str) -> dict[str, Any]:
    ensure_ledger(db_path)
    run = Path(run_dir)
    summary_path = run / "plan_summary.json"
    actions_path = run / "actions_today.csv"
    if not summary_path.exists() or not actions_path.exists():
        return {
            "status": "missing_artifacts",
            "run_dir": str(run),
            "missing": [
                str(path)
                for path in (summary_path, actions_path)
                if not path.exists()
            ],
        }

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    action_hash = _actions_hash(actions_path)
    batch_id = _batch_id(summary, action_hash)
    now = _now_iso()

    with _connect(db_path) as conn:
        existing = conn.execute(
            "SELECT batch_id FROM order_batches WHERE batch_id=?", (batch_id,)
        ).fetchone()
        if existing:
            return {"status": "already_registered", "batch_id": batch_id, "run_dir": str(run)}

        signal_date = str(summary.get("signal_date") or "")
        execution_date = str(summary.get("execution_date") or "")
        candidate_label = str(summary.get("candidate_label") or "")
        conn.execute(
            """
            INSERT INTO order_batches(
                batch_id, run_dir, signal_date, execution_date, candidate_label,
                actions_hash, status, registered_at, summary_json
            )
            VALUES(?, ?, ?, ?, ?, ?, 'registered', ?, ?)
            """,
            (
                batch_id,
                str(run),
                signal_date,
                execution_date,
                candidate_label,
                action_hash,
                now,
                json.dumps(summary, ensure_ascii=False, sort_keys=True),
            ),
        )

        created_orders = 0
        with actions_path.open("r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                stock = (row.get("stock") or "").strip()
                action = (row.get("action") or "").strip()
                side = _classify_side(action)
                shares = _normalized_shares(side, row.get("shares"))
                if not stock or side == "unknown" or shares <= 0:
                    continue
                conn.execute(
                    """
                    INSERT INTO orders(
                        batch_id, stock, side, action, requested_shares, remaining_shares,
                        reference_price, target_weight, status, reason, execution_date,
                        created_at, updated_at
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, 'pending', '', ?, ?, ?)
                    """,
                    (
                        batch_id,
                        stock,
                        side,
                        action,
                        shares,
                        shares,
                        float(row.get("price") or 0),
                        float(row.get("target_weight") or 0),
                        execution_date,
                        now,
                        now,
                    ),
                )
                created_orders += 1

        _insert_audit(
            conn,
            "order_batch_registered",
            {
                "batch_id": batch_id,
                "run_dir": str(run),
                "created_orders": created_orders,
                "signal_date": signal_date,
                "execution_date": execution_date,
            },
        )
        return {
            "status": "registered",
            "batch_id": batch_id,
            "run_dir": str(run),
            "created_order_count": created_orders,
        }


def trade_plan_batch_status(db_path: Path | str, run_dir: Path | str) -> dict[str, Any]:
    ensure_ledger(db_path)
    run = Path(run_dir)
    summary_path = run / "plan_summary.json"
    actions_path = run / "actions_today.csv"
    if not summary_path.exists() or not actions_path.exists():
        return {"status": "missing_artifacts", "run_dir": str(run)}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    action_hash = _actions_hash(actions_path)
    batch_id = _batch_id(summary, action_hash)
    with _connect(db_path) as conn:
        batch = conn.execute(
            "SELECT batch_id, signal_date, execution_date, status FROM order_batches WHERE batch_id=?",
            (batch_id,),
        ).fetchone()
        if not batch:
            return {
                "status": "unregistered",
                "batch_id": batch_id,
                "run_dir": str(run),
                "signal_date": str(summary.get("signal_date") or ""),
                "execution_date": str(summary.get("execution_date") or ""),
            }
        counts = {
            row["status"]: int(row["n"])
            for row in conn.execute(
                "SELECT status, COUNT(*) AS n FROM orders WHERE batch_id=? GROUP BY status",
                (batch_id,),
            )
        }
        return {
            "status": "registered",
            "batch_id": batch_id,
            "run_dir": str(run),
            "signal_date": batch["signal_date"],
            "execution_date": batch["execution_date"],
            "order_status_counts": counts,
            "pending_order_count": int(counts.get("pending", 0) + counts.get("partial", 0)),
            "filled_order_count": int(counts.get("filled", 0)),
            "blocked_order_count": int(counts.get("blocked", 0)),
        }


def _lookup_open(price_lookup: dict[str, Any], execution_date: str, stock: str) -> float | None:
    by_date = price_lookup.get(execution_date) or {}
    row = by_date.get(stock) or {}
    value = row.get("open")
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number


def _get_position(conn: sqlite3.Connection, stock: str) -> tuple[int, float]:
    row = conn.execute(
        "SELECT shares, cost_price FROM positions WHERE stock=?", (stock,)
    ).fetchone()
    if not row:
        return 0, 0.0
    return int(row["shares"]), float(row["cost_price"])


def _set_position(conn: sqlite3.Connection, stock: str, shares: int, cost_price: float) -> None:
    if shares <= 0:
        conn.execute("DELETE FROM positions WHERE stock=?", (stock,))
        return
    conn.execute(
        """
        INSERT INTO positions(stock, shares, cost_price, updated_at)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(stock) DO UPDATE SET
            shares=excluded.shares,
            cost_price=excluded.cost_price,
            updated_at=excluded.updated_at
        """,
        (stock, int(shares), float(cost_price), _now_iso()),
    )


def _load_batch_costs(conn: sqlite3.Connection, batch_id: str) -> dict[str, float]:
    row = conn.execute(
        "SELECT summary_json FROM order_batches WHERE batch_id=?", (batch_id,)
    ).fetchone()
    if not row:
        return {"transaction": 3.0, "slippage": 7.0, "sell_tax": 10.0}
    summary = json.loads(row["summary_json"])
    return {
        "transaction": float(summary.get("transaction_cost_bps") or 3.0),
        "slippage": float(summary.get("slippage_bps") or 7.0),
        "sell_tax": float(summary.get("sell_tax_bps") or 10.0),
    }


def apply_pending_orders(
    db_path: Path | str,
    price_lookup: dict[str, Any],
    execution_date: str,
) -> dict[str, Any]:
    ensure_ledger(db_path)
    blockers: set[str] = set()
    filled = 0
    blocked = 0

    with _connect(db_path) as conn:
        orders = conn.execute(
            """
            SELECT id, batch_id, stock, side, remaining_shares, execution_date
            FROM orders
            WHERE status IN ('pending', 'partial', 'blocked')
              AND execution_date=?
            ORDER BY id
            """,
            (execution_date,),
        ).fetchall()
        for order in orders:
            order_id = int(order["id"])
            stock = str(order["stock"])
            side = str(order["side"])
            remaining = int(order["remaining_shares"])
            open_price = _lookup_open(price_lookup, execution_date, stock)
            if open_price is None:
                blockers.add("missing_execution_open")
                blocked += 1
                conn.execute(
                    "UPDATE orders SET status='pending', reason=?, updated_at=? WHERE id=?",
                    ("missing_execution_open", _now_iso(), order_id),
                )
                continue

            costs = _load_batch_costs(conn, str(order["batch_id"]))
            transaction_rate = costs["transaction"] / 10000.0
            slippage_rate = costs["slippage"] / 10000.0
            sell_tax_rate = costs["sell_tax"] / 10000.0

            if side == "sell":
                held_shares, held_cost = _get_position(conn, stock)
                fill_shares = min(remaining, held_shares)
                if fill_shares <= 0:
                    blockers.add("no_position_to_sell")
                    blocked += 1
                    conn.execute(
                        "UPDATE orders SET status='blocked', reason=?, updated_at=? WHERE id=?",
                        ("no_position_to_sell", _now_iso(), order_id),
                    )
                    continue
                fill_price = open_price * (1 - slippage_rate)
                gross = fill_price * fill_shares
                cost = gross * transaction_rate
                tax = gross * sell_tax_rate
                net_cash_delta = gross - cost - tax
                _record_cash(
                    conn,
                    amount=net_cash_delta,
                    flow_type="sell_fill",
                    reason=f"paper sell {stock}",
                    external_flow=False,
                    ref=str(order_id),
                    flow_date=execution_date,
                )
                _set_position(conn, stock, held_shares - fill_shares, held_cost)
            else:
                lot_shares = (remaining // 100) * 100
                if lot_shares <= 0:
                    blockers.add("invalid_lot_size")
                    blocked += 1
                    conn.execute(
                        "UPDATE orders SET status='blocked', reason=?, updated_at=? WHERE id=?",
                        ("invalid_lot_size", _now_iso(), order_id),
                    )
                    continue
                fill_price = open_price * (1 + slippage_rate)
                gross = fill_price * lot_shares
                cost = gross * transaction_rate
                cash_needed = gross + cost
                cash = _cash_balance(conn)
                fill_shares = lot_shares
                if cash_needed > cash:
                    affordable_lots = int(cash // ((fill_price * 100) * (1 + transaction_rate)))
                    fill_shares = affordable_lots * 100
                    if fill_shares <= 0:
                        blockers.add("insufficient_cash")
                        blocked += 1
                        conn.execute(
                            "UPDATE orders SET status='blocked', reason=?, updated_at=? WHERE id=?",
                            ("insufficient_cash", _now_iso(), order_id),
                        )
                        continue
                    blockers.add("partial_due_to_cash")
                    blockers.add("insufficient_cash_remainder")
                gross = fill_price * fill_shares
                cost = gross * transaction_rate
                tax = 0.0
                net_cash_delta = -(gross + cost)
                held_shares, held_cost = _get_position(conn, stock)
                new_shares = held_shares + fill_shares
                new_cost = (
                    ((held_shares * held_cost) + (fill_shares * fill_price)) / new_shares
                    if new_shares > 0
                    else 0.0
                )
                _record_cash(
                    conn,
                    amount=net_cash_delta,
                    flow_type="buy_fill",
                    reason=f"paper buy {stock}",
                    external_flow=False,
                    ref=str(order_id),
                    flow_date=execution_date,
                )
                _set_position(conn, stock, new_shares, new_cost)

            remaining_after = remaining - fill_shares
            status = "filled" if remaining_after <= 0 else "partial"
            reason = ""
            if side == "buy" and remaining_after > 0 and "partial_due_to_cash" in blockers:
                status = "blocked"
                reason = "insufficient_cash_remainder"
            conn.execute(
                """
                INSERT INTO fills(
                    order_id, batch_id, stock, side, shares, price, gross_amount,
                    cost_amount, tax_amount, net_cash_delta, execution_date, created_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order_id,
                    order["batch_id"],
                    stock,
                    side,
                    fill_shares,
                    fill_price,
                    gross,
                    cost,
                    tax,
                    net_cash_delta,
                    execution_date,
                    _now_iso(),
                ),
            )
            conn.execute(
                "UPDATE orders SET status=?, remaining_shares=?, reason=?, updated_at=? WHERE id=?",
                (status, max(0, remaining_after), reason, _now_iso(), order_id),
            )
            filled += 1

        pending = int(
            conn.execute(
                "SELECT COUNT(*) AS n FROM orders WHERE status IN ('pending', 'partial')"
            ).fetchone()["n"]
        )
        blocked_total = int(
            conn.execute("SELECT COUNT(*) AS n FROM orders WHERE status='blocked'").fetchone()["n"]
        )
        _insert_audit(
            conn,
            "orders_applied",
            {
                "execution_date": execution_date,
                "filled_order_count": filled,
                "pending_order_count": pending,
                "blocked_order_count": blocked_total,
                "blockers": sorted(blockers),
            },
        )
        return {
            "status": "ok",
            "execution_date": execution_date,
            "filled_order_count": filled,
            "pending_order_count": pending,
            "blocked_order_count": blocked_total,
            "blockers": sorted(blockers),
        }


def record_cash_flow(
    db_path: Path | str,
    flow_type: str,
    amount: float,
    reason: str = "",
) -> dict[str, Any]:
    ensure_ledger(db_path)
    normalized = flow_type.strip().lower()
    if normalized in {"withdraw", "withdrawal", "提现"}:
        signed_amount = -abs(float(amount))
        normalized = "withdrawal"
    elif normalized in {"deposit", "充值"}:
        signed_amount = abs(float(amount))
        normalized = "deposit"
    else:
        signed_amount = float(amount)
    with _connect(db_path) as conn:
        _record_cash(
            conn,
            amount=signed_amount,
            flow_type=normalized,
            reason=reason,
            external_flow=True,
        )
        _insert_audit(
            conn,
            "cash_flow_recorded",
            {"flow_type": normalized, "amount": signed_amount, "reason": reason},
        )
        return {"status": "ok", "available_cash": _cash_balance(conn)}


def replace_account_state(
    db_path: Path | str,
    *,
    available_cash: float | int | str | None,
    positions: list[dict[str, Any]] | None,
    reason: str = "manual snapshot correction",
) -> dict[str, Any]:
    ensure_ledger(db_path)
    with _connect(db_path) as conn:
        if available_cash not in {None, ""}:
            target_cash = float(available_cash)
            current_cash = _cash_balance(conn)
            delta = target_cash - current_cash
            if abs(delta) > 1e-9:
                _record_cash(
                    conn,
                    amount=delta,
                    flow_type="manual_cash_correction",
                    reason=reason,
                    external_flow=True,
                )
        conn.execute("DELETE FROM positions")
        for position in positions or []:
            stock = str(position.get("stock", "") or "").strip().upper()
            if not stock:
                continue
            shares = int(float(position.get("shares") or 0))
            cost_price = float(position.get("cost_price") or 0)
            if shares <= 0:
                continue
            _set_position(conn, stock, shares, cost_price)
        _insert_audit(
            conn,
            "account_state_replaced",
            {
                "available_cash": available_cash,
                "position_count": len([item for item in (positions or []) if str(item.get("stock", "") or "").strip()]),
                "reason": reason,
            },
        )
    return account_summary(db_path)


def record_manual_adjustment(
    db_path: Path | str,
    *,
    adjustment_type: str,
    stock: str = "",
    shares: float | int | str | None = None,
    cost_price: float | int | str | None = None,
    amount: float | int | str | None = None,
    reason: str = "",
) -> dict[str, Any]:
    ensure_ledger(db_path)
    kind = str(adjustment_type or "").strip().lower()
    with _connect(db_path) as conn:
        if kind in {"cash", "cash_adjustment", "现金"}:
            delta = float(amount or 0)
            _record_cash(
                conn,
                amount=delta,
                flow_type="manual_cash_adjustment",
                reason=reason,
                external_flow=True,
            )
        elif kind in {"position", "position_adjustment", "持仓"}:
            symbol = str(stock or "").strip().upper()
            if not symbol:
                raise ValueError("manual position adjustment requires stock")
            target_shares = int(float(shares or 0))
            target_cost = float(cost_price or 0)
            _set_position(conn, symbol, target_shares, target_cost)
        else:
            raise ValueError(f"unsupported manual adjustment type: {adjustment_type}")
        _insert_audit(
            conn,
            "manual_adjustment",
            {
                "adjustment_type": kind,
                "stock": stock,
                "shares": shares,
                "cost_price": cost_price,
                "amount": amount,
                "reason": reason,
            },
        )
    return account_summary(db_path)


def _market_value(conn: sqlite3.Connection, price_lookup: dict[str, Any], as_of_date: str) -> float:
    by_date = price_lookup.get(as_of_date) or {}
    total = 0.0
    for row in conn.execute("SELECT stock, shares, cost_price FROM positions WHERE shares > 0"):
        stock = row["stock"]
        quote = by_date.get(stock) or {}
        close = quote.get("close")
        try:
            price = float(close) if close not in (None, "") else float(row["cost_price"])
        except (TypeError, ValueError):
            price = float(row["cost_price"])
        total += int(row["shares"]) * price
    return total


def _external_flow_between(conn: sqlite3.Connection, start_exclusive: str | None, end_inclusive: str) -> float:
    if start_exclusive:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(amount), 0) AS flow
            FROM cash_ledger
            WHERE external_flow=1 AND flow_date > ? AND flow_date <= ?
            """,
            (start_exclusive, end_inclusive),
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(amount), 0) AS flow
            FROM cash_ledger
            WHERE external_flow=1 AND flow_date <= ?
            """,
            (end_inclusive,),
        ).fetchone()
    return float(row["flow"] if row else 0.0)


def _external_flow_created_between(
    conn: sqlite3.Connection,
    start_created_exclusive: str | None,
    end_created_inclusive: str | None,
) -> float:
    if start_created_exclusive and end_created_inclusive:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(amount), 0) AS flow
            FROM cash_ledger
            WHERE external_flow=1 AND created_at > ? AND created_at <= ?
            """,
            (start_created_exclusive, end_created_inclusive),
        ).fetchone()
    elif start_created_exclusive:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(amount), 0) AS flow
            FROM cash_ledger
            WHERE external_flow=1 AND created_at > ?
            """,
            (start_created_exclusive,),
        ).fetchone()
    elif end_created_inclusive:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(amount), 0) AS flow
            FROM cash_ledger
            WHERE external_flow=1 AND created_at <= ?
            """,
            (end_created_inclusive,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) AS flow FROM cash_ledger WHERE external_flow=1"
        ).fetchone()
    return float(row["flow"] if row else 0.0)


def write_daily_equity(
    db_path: Path | str,
    price_lookup: dict[str, Any],
    as_of_date: str,
) -> dict[str, Any]:
    ensure_ledger(db_path)
    with _connect(db_path) as conn:
        cash = _cash_balance(conn)
        market_value = _market_value(conn, price_lookup, as_of_date)
        total_equity = cash + market_value
        previous = conn.execute(
            """
            SELECT as_of_date, total_equity, created_at
            FROM daily_equity
            WHERE as_of_date < ?
            ORDER BY as_of_date DESC
            LIMIT 1
            """,
            (as_of_date,),
        ).fetchone()
        if previous:
            net_flow = _external_flow_created_between(conn, previous["created_at"], _now_iso())
            prev_equity = float(previous["total_equity"])
            daily_return = (
                (total_equity - prev_equity - net_flow) / prev_equity if prev_equity else 0.0
            )
        else:
            net_flow = _external_flow_between(conn, None, as_of_date)
            daily_return = 0.0

        conn.execute(
            """
            INSERT INTO daily_equity(
                as_of_date, cash, market_value, total_equity, daily_return,
                net_external_flow_since_previous, created_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(as_of_date) DO UPDATE SET
                cash=excluded.cash,
                market_value=excluded.market_value,
                total_equity=excluded.total_equity,
                daily_return=excluded.daily_return,
                net_external_flow_since_previous=excluded.net_external_flow_since_previous,
                created_at=excluded.created_at
            """,
            (
                as_of_date,
                cash,
                market_value,
                total_equity,
                daily_return,
                net_flow,
                _now_iso(),
            ),
        )
        return {
            "status": "ok",
            "as_of_date": as_of_date,
            "cash": cash,
            "market_value": market_value,
            "total_equity": total_equity,
            "daily_return": daily_return,
            "net_external_flow_since_previous": net_flow,
        }


def performance_summary(db_path: Path | str, start_date: str, end_date: str) -> dict[str, Any]:
    ensure_ledger(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT as_of_date, cash, market_value, total_equity, daily_return, created_at
            FROM daily_equity
            WHERE as_of_date >= ? AND as_of_date <= ?
            ORDER BY as_of_date
            """,
            (start_date, end_date),
        ).fetchall()
        total_return = 1.0
        peak = None
        max_drawdown = 0.0
        curve: list[dict[str, Any]] = []
        for row in rows:
            daily_return = float(row["daily_return"])
            total_return *= 1.0 + daily_return
            equity = float(row["total_equity"])
            peak = equity if peak is None else max(peak, equity)
            drawdown = (equity / peak - 1.0) if peak else 0.0
            max_drawdown = min(max_drawdown, drawdown)
            curve.append(
                {
                    "date": row["as_of_date"],
                    "cash": float(row["cash"]),
                    "market_value": float(row["market_value"]),
                    "total_equity": equity,
                    "daily_return": daily_return,
                    "drawdown": drawdown,
                }
            )
        net_flow = 0.0
        if rows:
            net_flow = _external_flow_created_between(
                conn,
                rows[0]["created_at"],
                rows[-1]["created_at"],
            )
        return {
            "status": "ok",
            "start_date": start_date,
            "end_date": end_date,
            "total_return": total_return - 1.0,
            "max_drawdown": max_drawdown,
            "net_cash_flow": net_flow,
            "points": curve,
        }


def export_account_snapshot(db_path: Path | str, path: Path | str) -> dict[str, Any]:
    ensure_ledger(db_path)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn, target.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["record_type", "stock", "shares", "cost_price", "available_cash"])
        writer.writerow(["account", "", "", "", _format_number(_cash_balance(conn))])
        for row in conn.execute(
            "SELECT stock, shares, cost_price FROM positions WHERE shares > 0 ORDER BY stock"
        ):
            writer.writerow(
                [
                    "position",
                    row["stock"],
                    _format_number(row["shares"]),
                    _format_number(row["cost_price"]),
                    "",
                ]
            )
    return {"status": "ok", "path": str(target)}
