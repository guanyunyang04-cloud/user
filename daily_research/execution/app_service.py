from __future__ import annotations

import csv
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import pandas as pd

from daily_research.baseline.data_provider import get_latest_completed_trading_date
from daily_research.data_platform.providers import (
    FORMAL_FREE_V3_OPTIONAL_DOMAINS,
    FORMAL_FREE_V3_REQUIRED_DOMAINS,
    provider_capability_matrix,
)
from daily_research.data_platform.provider_health import ProviderHealthConfig, run_provider_health
from daily_research.deep_alpha.experiment_guardrails import resolve_project_python_executable
from daily_research.execution import paper_trading
from daily_research.execution import production_signal
from daily_research.execution import freeze_guard
from daily_research.execution.app_runtime import (
    EVENTS_PATH,
    JOBS_ROOT,
    LOCK_PATH,
    PROJECT_ROOT,
    RUNTIME_ROOT,
    STATE_PATH,
    WORKSPACE_ROOT,
    ExecutionAppLock,
    ExecutionAppLockError,
    append_event,
    build_job_paths,
    clear_lock_file,
    create_job_record,
    ensure_runtime_layout,
    list_recent_job_metadata,
    load_job_metadata,
    load_runtime_state,
    mark_job_finished,
    mark_job_heartbeat,
    mark_job_started,
    now_iso,
    read_file_increment,
    read_json_file,
    tail_file,
    update_job_progress,
    update_job_metadata,
    write_json_file,
    write_runtime_state,
)
from daily_research.execution.app_tasks import (
    build_task_command,
    get_task_spec,
    list_core_frontend_task_specs,
    list_task_specs,
    serialize_task_spec,
)


EXECUTION_DIR = Path(__file__).resolve().parent
ACTIVE_MANIFEST_PATH = PROJECT_ROOT / "output" / "active_execution_strategy.json"
POSITIONS_PATH = EXECUTION_DIR / "current_positions.csv"
POSITIONS_EXAMPLE_PATH = EXECUTION_DIR / "current_positions.example.csv"
PAPER_ACCOUNT_ROOT = RUNTIME_ROOT / "paper_account"
PAPER_ACCOUNT_DB_PATH = PAPER_ACCOUNT_ROOT / "paper_account.sqlite3"
ENVIRONMENT_PATH = PROJECT_ROOT / "environment.yml"
LATEST_TRADE_PLAN_PATH = EXECUTION_DIR / "output" / "latest_trade_plan.txt"
ACCOUNT_FILE_HEADERS = ("record_type", "stock", "shares", "cost_price", "available_cash")
FORMAL_DATA_PLATFORM_DOMAINS: tuple[str, ...] = (
    *FORMAL_FREE_V3_REQUIRED_DOMAINS,
    *FORMAL_FREE_V3_OPTIONAL_DOMAINS,
)
FORMAL_DATA_PLATFORM_REQUIRED_DOMAINS: tuple[str, ...] = FORMAL_FREE_V3_REQUIRED_DOMAINS
FORMAL_DATA_PLATFORM_PROVIDER_PLAN = "formal_free_v3"
_ACTIVE_JOB_THREADS: dict[str, threading.Thread] = {}
_ACTIVE_JOB_THREADS_LOCK = threading.Lock()
_RUNNER_WARNING_LIMIT = 20
RUNNER_TIMEOUT_EXIT_CODE = 124


def configure_runtime_root(runtime_root: str | Path) -> Path:
    from daily_research.execution import app_runtime

    global RUNTIME_ROOT, JOBS_ROOT, STATE_PATH, EVENTS_PATH, LOCK_PATH, PAPER_ACCOUNT_ROOT, PAPER_ACCOUNT_DB_PATH
    resolved = app_runtime.configure_runtime_root(runtime_root)
    RUNTIME_ROOT = resolved
    JOBS_ROOT = resolved / "jobs"
    STATE_PATH = resolved / "runtime_state.json"
    EVENTS_PATH = resolved / "events.jsonl"
    LOCK_PATH = resolved / "execution_app.lock"
    PAPER_ACCOUNT_ROOT = resolved / "paper_account"
    PAPER_ACCOUNT_DB_PATH = PAPER_ACCOUNT_ROOT / "paper_account.sqlite3"
    return resolved


def sanitize_passthrough_args(values: list[str] | None) -> list[str]:
    items = [str(item) for item in (values or [])]
    if items[:1] == ["--"]:
        items = items[1:]
    return strip_execution_timeout_args(items)


def _read_json(path: Path) -> dict[str, Any]:
    return read_json_file(path)


def _resolve_existing_project_path(raw_path: Any) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        return Path("")
    path = Path(text).expanduser()
    candidates = [path]
    normalized = text.replace("\\", "/")
    marker = "/daily_research/"
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        candidates.append(PROJECT_ROOT / suffix)
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if resolved.exists():
            return resolved
    try:
        return path.resolve()
    except Exception:
        return path


def _file_mtime_text(path: Path) -> str:
    if not path.exists():
        return ""
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")


def _normalize_date_text(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        import pandas as pd

        parsed = pd.to_datetime(raw, errors="coerce")
        if pd.isna(parsed):
            return raw
        return parsed.strftime("%Y-%m-%d")
    except Exception:
        return raw[:10]


def _date_covers(left: Any, right: Any) -> bool:
    left_text = _normalize_date_text(left)
    right_text = _normalize_date_text(right)
    if not left_text or not right_text:
        return False
    try:
        import pandas as pd

        return bool(pd.Timestamp(left_text) >= pd.Timestamp(right_text))
    except Exception:
        return left_text >= right_text


def _clean_csv_cell(value: Any) -> str:
    return str(value or "").strip()


def _normalize_position_record(row: dict[str, Any], *, row_index: int) -> dict[str, Any]:
    stock = str(row.get("stock", "") or "").strip().upper()
    shares_text = str(row.get("shares", "") or "").strip()
    cost_text = str(row.get("cost_price", "") or "").strip()
    if not stock and not shares_text and not cost_text:
        return {}
    if not stock:
        raise ValueError(f"第 {row_index} 行缺少证券代码。")
    try:
        shares_number = float(shares_text)
    except ValueError as exc:
        raise ValueError(f"第 {row_index} 行持仓数量不是有效数字：{shares_text}") from exc
    if shares_number <= 0 or int(shares_number) != shares_number:
        raise ValueError(f"第 {row_index} 行持仓数量必须是正整数：{shares_text}")
    try:
        cost_price = float(cost_text or 0.0)
    except ValueError as exc:
        raise ValueError(f"第 {row_index} 行成本价不是有效数字：{cost_text}") from exc
    if cost_price < 0:
        raise ValueError(f"第 {row_index} 行成本价不能为负数：{cost_text}")
    return {"stock": stock, "shares": int(shares_number), "cost_price": float(cost_price)}


def _format_decimal(value: float | int | None) -> str:
    if value is None:
        return ""
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _write_csv_atomic(path: Path, *, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ACCOUNT_FILE_HEADERS))
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def _load_account_snapshot_csv(path: Path) -> dict[str, Any]:
    snapshot = {
        "path": str(path.resolve()),
        "exists": path.exists(),
        "source": "missing",
        "headers_ok": False,
        "available_cash": None,
        "positions": [],
        "position_count": 0,
        "total_shares": 0,
        "tickers": [],
        "last_modified_at": _file_mtime_text(path),
    }
    if not path.exists():
        return snapshot

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        raw_fieldnames = [str(name or "") for name in (reader.fieldnames or [])]
        normalized_fieldnames = {_clean_csv_cell(name).lower(): name for name in raw_fieldnames if _clean_csv_cell(name)}
        rows: list[dict[str, str]] = []
        for raw_row in reader:
            normalized_row: dict[str, str] = {}
            for normalized_name, original_name in normalized_fieldnames.items():
                normalized_row[normalized_name] = _clean_csv_cell(raw_row.get(original_name, ""))
            rows.append(normalized_row)

    if not normalized_fieldnames:
        return snapshot

    required_position_fields = {"stock", "shares"}
    positions: list[dict[str, Any]] = []
    available_cash: float | None = None
    has_record_type = "record_type" in normalized_fieldnames
    if has_record_type:
        snapshot["source"] = "account_snapshot"
        snapshot["headers_ok"] = {"record_type", "available_cash"} <= set(normalized_fieldnames) or required_position_fields <= set(
            normalized_fieldnames
        )
        for row in rows:
            record_type = str(row.get("record_type", "") or "").strip().lower()
            if record_type == "account":
                cash_text = str(row.get("available_cash", "") or row.get("cash", "") or "").strip()
                if cash_text:
                    try:
                        available_cash = float(cash_text)
                    except ValueError as exc:
                        raise ValueError(f"账户快照中的 available_cash 不是有效数字：{cash_text}") from exc
                continue
            if record_type not in {"position", "holding", "hold", ""}:
                continue
            normalized = _normalize_position_record(row, row_index=len(positions) + 1)
            if normalized:
                positions.append(normalized)
    else:
        snapshot["source"] = "positions_only"
        snapshot["headers_ok"] = required_position_fields <= set(normalized_fieldnames)
        for row in rows:
            normalized = _normalize_position_record(row, row_index=len(positions) + 1)
            if normalized:
                positions.append(normalized)

    snapshot["available_cash"] = available_cash
    snapshot["positions"] = positions
    snapshot["position_count"] = len(positions)
    snapshot["total_shares"] = sum(int(item["shares"]) for item in positions)
    snapshot["tickers"] = [str(item["stock"]) for item in positions[:20]]
    return snapshot


def _ensure_paper_account() -> dict[str, Any]:
    result = paper_trading.ensure_ledger(db_path=PAPER_ACCOUNT_DB_PATH, snapshot_path=POSITIONS_PATH)
    paper_trading.export_account_snapshot(db_path=PAPER_ACCOUNT_DB_PATH, path=POSITIONS_PATH)
    return result


def _paper_summary_to_account_snapshot(summary: dict[str, Any], *, path: Path) -> dict[str, Any]:
    positions = [
        {
            "stock": str(item.get("stock", "") or ""),
            "shares": int(float(item.get("shares", 0) or 0)),
            "cost_price": float(item.get("cost_price", 0) or 0),
        }
        for item in list(summary.get("positions", []) if isinstance(summary.get("positions"), list) else [])
    ]
    return {
        "status": "ok",
        "path": str(path.resolve()),
        "db_path": str(PAPER_ACCOUNT_DB_PATH.resolve()),
        "exists": path.exists(),
        "source": "paper_ledger",
        "headers_ok": True,
        "available_cash": float(summary.get("available_cash", 0) or 0),
        "positions": positions,
        "positions_by_stock": summary.get("positions_by_stock", {}),
        "position_count": len(positions),
        "total_shares": sum(int(item["shares"]) for item in positions),
        "tickers": [str(item["stock"]) for item in positions[:20]],
        "last_modified_at": _file_mtime_text(path),
        "pending_order_count": int(summary.get("pending_order_count", 0) or 0),
        "filled_order_count": int(summary.get("filled_order_count", 0) or 0),
        "blocked_order_count": int(summary.get("blocked_order_count", 0) or 0),
        "pending_orders": list(summary.get("pending_orders", []) if isinstance(summary.get("pending_orders"), list) else []),
        "recent_fills": list(summary.get("recent_fills", []) if isinstance(summary.get("recent_fills"), list) else []),
        "recent_cash_flows": list(summary.get("recent_cash_flows", []) if isinstance(summary.get("recent_cash_flows"), list) else []),
        "latest_equity": dict(summary.get("latest_equity", {}) if isinstance(summary.get("latest_equity"), dict) else {}),
    }


def paper_account_summary() -> dict[str, Any]:
    init = _ensure_paper_account()
    summary = paper_trading.account_summary(db_path=PAPER_ACCOUNT_DB_PATH)
    return {
        **_paper_summary_to_account_snapshot(summary, path=POSITIONS_PATH),
        "initialized_from_snapshot": bool(init.get("initialized_from_snapshot")),
        "ledger": summary,
    }


def load_account_snapshot(path: Path | None = None) -> dict[str, Any]:
    resolved_path = path or POSITIONS_PATH
    if Path(resolved_path) != POSITIONS_PATH:
        return _load_account_snapshot_csv(Path(resolved_path))
    return paper_account_summary()


def positions_summary(path: Path | None = None) -> dict[str, Any]:
    snapshot = load_account_snapshot(path)
    return {
        "path": snapshot["path"],
        "exists": snapshot["exists"],
        "row_count": snapshot["position_count"],
        "tickers": snapshot["tickers"][:12],
        "headers_ok": snapshot["headers_ok"],
        "available_cash": snapshot["available_cash"],
        "last_modified_at": snapshot["last_modified_at"],
        "source": snapshot["source"],
    }


def active_manifest_summary() -> dict[str, Any]:
    payload = _read_json(ACTIVE_MANIFEST_PATH)
    if not payload:
        return {
            "path": str(ACTIVE_MANIFEST_PATH.resolve()),
            "exists": ACTIVE_MANIFEST_PATH.exists(),
            "candidate_label": "",
            "strategy_name": "",
            "liquidity_pool_name": "",
            "source_target_weight_panel_csv": "",
            "trade_plan_target_weight_panel_csv": "",
            "trade_plan_score_panel_csv": "",
            "production_root": "",
            "production_manifest_json": "",
            "production_anchor_source_run_dir": "",
            "production_anchor_sync_manifest": "",
            "execution_policy_label": "",
            "effective_live_target_weight_mode": "",
            "effective_live_execution_profile": "",
            "data_source": "",
            "lake_dataset_id": "",
            "source_market_dataset_id": "",
            "lake_dataset_end_date": "",
        }
    return {
        "path": str(ACTIVE_MANIFEST_PATH.resolve()),
        "exists": True,
        "candidate_label": str(payload.get("candidate_label", "") or ""),
        "strategy_name": str(payload.get("strategy_name", "") or ""),
        "liquidity_pool_name": str(
            payload.get("liquidity_pool_name", "")
            or payload.get("rolling_liquidity_pool", "")
            or payload.get("liquidity_pool", "")
            or ""
        ),
        "source_target_weight_panel_csv": str(payload.get("source_target_weight_panel_csv", "") or ""),
        "trade_plan_target_weight_panel_csv": str(payload.get("trade_plan_target_weight_panel_csv", "") or ""),
        "trade_plan_score_panel_csv": str(payload.get("trade_plan_score_panel_csv", "") or ""),
        "production_root": str(payload.get("production_root", "") or payload.get("trade_plan_refresh_run_dir", "") or ""),
        "production_manifest_json": str(payload.get("production_manifest_json", "") or ""),
        "production_anchor_source_run_dir": str(payload.get("production_anchor_source_run_dir", "") or ""),
        "production_anchor_sync_manifest": str(payload.get("production_anchor_sync_manifest", "") or ""),
        "execution_policy_label": str(payload.get("execution_policy_label", "") or payload.get("execution_alignment_profile", "") or ""),
        "effective_live_target_weight_mode": str(payload.get("effective_live_target_weight_mode", "") or ""),
        "effective_live_execution_profile": str(payload.get("effective_live_execution_profile", "") or ""),
        "data_source": str(payload.get("data_source", "") or ""),
        "lake_dataset_id": str(payload.get("lake_dataset_id", "") or ""),
        "source_market_dataset_id": str(payload.get("source_market_dataset_id", "") or ""),
        "lake_dataset_end_date": str(payload.get("lake_dataset_end_date", "") or ""),
    }


def _latest_trade_plan_run_dir(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    candidates = [
        path
        for path in output_dir.iterdir()
        if path.is_dir()
        and any((path / name).exists() for name in ("plan_summary.json", "actions_today.csv", "daily_trade_plan.txt"))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _normalize_trade_plan_model_info(summary: dict[str, Any]) -> dict[str, Any]:
    freshness = summary.get("model_freshness") if isinstance(summary.get("model_freshness"), dict) else {}
    model_info = dict(freshness)
    if summary.get("production_model_train_end_date"):
        model_info.setdefault("train_end_date", str(summary.get("production_model_train_end_date", "")))
        model_info.setdefault("artifact_latest_data_date", str(summary.get("production_model_train_end_date", "")))
    if summary.get("production_model_launch_cutoff_date"):
        model_info.setdefault("launch_cutoff_date", str(summary.get("production_model_launch_cutoff_date", "")))
    if summary.get("signal_date"):
        model_info.setdefault("signal_date", str(summary.get("signal_date", "")))
        model_info.setdefault("latest_completed_trading_date", str(summary.get("signal_date", "")))
    for key in ("source_signal_date", "execution_signal_date", "signal_panel_status"):
        if summary.get(key) not in {None, ""}:
            model_info.setdefault(key, str(summary.get(key, "")))
    trading_day_lag = (
        summary.get("production_model_trading_day_lag")
        if summary.get("production_model_trading_day_lag") is not None
        else summary.get("production_model_retrain_trading_day_lag")
    )
    if trading_day_lag is not None:
        model_info.setdefault("trading_day_lag", trading_day_lag)
    status = str(summary.get("production_model_status", "") or summary.get("production_model_retrain_status", "") or "")
    if status:
        model_info.setdefault("status", status)
        model_info.setdefault("status_text", status)
    return model_info


def _trade_plan_diagnostics(
    *,
    summary: dict[str, Any],
    actions: list[dict[str, str]],
    holdings: list[dict[str, str]],
    watchlist: list[dict[str, str]],
    txt_preview: list[str],
) -> dict[str, Any]:
    def pick_number(key: str) -> int | float | str:
        value = summary.get(key)
        if value in {None, ""}:
            return 0
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        return int(numeric) if numeric.is_integer() else numeric

    joined_text = "\n".join(str(line) for line in txt_preview)
    score_context_status = str(summary.get("score_context_status", "") or "").strip()
    if not score_context_status:
        lowered = joined_text.lower()
        if "external score context fallback" in lowered and "no usable rows" in lowered:
            score_context_status = "fallback_no_signal_rows"
        elif joined_text:
            score_context_status = "ok"
        else:
            score_context_status = "unknown"

    regime_state = str(summary.get("regime_state", "") or summary.get("market_regime_state", "") or "").strip()
    if not regime_state:
        for line in txt_preview:
            text_line = str(line)
            if "市场状态" in text_line and ":" in text_line:
                regime_state = text_line.split(":", 1)[1].strip()
                break

    market_filter_text = str(summary.get("market_filter_text", "") or summary.get("market_filter_status", "") or "").strip()
    if not market_filter_text:
        for line in txt_preview:
            text_line = str(line)
            if "市场过滤" in text_line and ":" in text_line:
                market_filter_text = text_line.split(":", 1)[1].strip()
                break
    if not market_filter_text and summary.get("market_regime_filter_enabled") is False:
        market_filter_text = "不拦截"

    reason = str(summary.get("empty_plan_reason", "") or "").strip()
    if not actions and not reason:
        parts: list[str] = []
        if not holdings and int(float(summary.get("current_position_count", 0) or 0)) == 0:
            parts.append("无当前持仓")
        if int(float(summary.get("target_position_count", 0) or 0)) == 0:
            parts.append("目标仓位数为 0")
        if int(float(summary.get("actionable_target_position_count", 0) or 0)) == 0:
            parts.append("无可执行目标")
        if score_context_status == "fallback_no_signal_rows" or "no usable rows" in joined_text.lower():
            parts.append("score panel 对信号日无可用行")
        blocked = int(float(summary.get("blocked_buy_candidate_count", 0) or 0))
        if blocked:
            parts.append(f"{blocked} 个买入候选被过滤")
        reason = "；".join(dict.fromkeys(parts)) or "当前计划无交易动作"

    return {
        "target_position_count": pick_number("target_position_count"),
        "actionable_target_position_count": pick_number("actionable_target_position_count"),
        "candidate_total_rows": pick_number("candidate_total_rows"),
        "candidate_usable_rows": pick_number("candidate_usable_rows"),
        "candidate_dropped_rows": pick_number("candidate_dropped_rows"),
        "blocked_buy_candidate_count": pick_number("blocked_buy_candidate_count"),
        "score_context_status": score_context_status,
        "empty_plan_reason": reason,
        "regime_state": regime_state,
        "market_filter_text": market_filter_text,
        "source_signal_date": str(summary.get("source_signal_date", "") or ""),
        "execution_signal_date": str(summary.get("execution_signal_date", "") or summary.get("signal_date", "") or ""),
        "signal_panel_status": str(summary.get("signal_panel_status", "") or ""),
        "action_count": len(actions),
        "holding_count": len(holdings),
        "watchlist_count": len(watchlist),
    }


def _paper_trade_plan_status(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {"status": "missing_trade_plan_run"}
    try:
        paper_trading.ensure_ledger(db_path=PAPER_ACCOUNT_DB_PATH, snapshot_path=POSITIONS_PATH)
        return paper_trading.trade_plan_batch_status(db_path=PAPER_ACCOUNT_DB_PATH, run_dir=run_dir)
    except Exception as exc:
        return {"status": "warning", "error": str(exc), "run_dir": str(run_dir)}


def latest_trade_plan_summary(
    *,
    max_lines: int = 80,
    output_dir: Path | str | None = None,
    latest_txt_path: Path | str | None = None,
) -> dict[str, Any]:
    if output_dir is not None:
        resolved_output_dir = Path(output_dir)
    elif latest_txt_path is not None:
        resolved_output_dir = Path(latest_txt_path).parent
    else:
        resolved_output_dir = EXECUTION_DIR / "output"
    resolved_latest_txt = Path(latest_txt_path) if latest_txt_path is not None else resolved_output_dir / "latest_trade_plan.txt"
    run_dir = _latest_trade_plan_run_dir(resolved_output_dir)
    summary_path = run_dir / "plan_summary.json" if run_dir is not None else None
    actions_path = run_dir / "actions_today.csv" if run_dir is not None else None
    holdings_path = run_dir / "holdings_snapshot.csv" if run_dir is not None else None
    watchlist_path = run_dir / "watchlist.csv" if run_dir is not None else None
    txt_path = resolved_latest_txt if resolved_latest_txt.exists() else (run_dir / "daily_trade_plan.txt" if run_dir is not None else resolved_latest_txt)
    lines = tail_file(txt_path, lines=max_lines) if txt_path.exists() else []
    summary = _read_json(summary_path) if summary_path is not None and summary_path.exists() else {}
    model_info = _normalize_trade_plan_model_info(summary)
    actions = _read_csv_preview(actions_path, limit=200) if actions_path and actions_path.exists() else []
    holdings = _read_csv_preview(holdings_path, limit=300) if holdings_path and holdings_path.exists() else []
    watchlist = _read_csv_preview(watchlist_path, limit=200) if watchlist_path and watchlist_path.exists() else []
    diagnostics = _trade_plan_diagnostics(
        summary=summary,
        actions=actions,
        holdings=holdings,
        watchlist=watchlist,
        txt_preview=lines,
    )
    paper_status = _paper_trade_plan_status(run_dir)
    return {
        "path": str(txt_path.resolve()),
        "exists": txt_path.exists(),
        "status": "ok" if txt_path.exists() or run_dir is not None else "missing",
        "summary": summary,
        "actions": actions,
        "holdings": holdings,
        "watchlist": watchlist,
        "model_info": model_info,
        "diagnostics": diagnostics,
        "paper_trading": paper_status,
        "txt_preview": lines,
        "preview_lines": lines,
        "line_count": len(lines),
        "artifact_paths": {
            "output_dir": str(resolved_output_dir.resolve()),
            "run_dir": str(run_dir.resolve()) if run_dir is not None else "",
            "summary_json": str(summary_path.resolve()) if summary_path is not None else "",
            "actions_csv": str(actions_path.resolve()) if actions_path is not None else "",
            "holdings_csv": str(holdings_path.resolve()) if holdings_path is not None else "",
            "watchlist_csv": str(watchlist_path.resolve()) if watchlist_path is not None else "",
            "txt": str(txt_path.resolve()),
        },
    }


def _read_csv_preview(path: Path, *, limit: int = 12) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({str(key): str(value or "") for key, value in row.items()})
            if len(rows) >= max(int(limit), 0):
                break
    return rows


def _model_card(
    *,
    model_id: str,
    role: str,
    name: str,
    status: str,
    usage: str,
    is_live: bool = False,
    is_shadow: bool = False,
    is_legacy: bool = False,
    data_source: str = "",
    dataset_id: str = "",
    trained_at: str = "",
    train_start_date: str = "",
    train_end_date: str = "",
    signal_date: str = "",
    artifact_path: str = "",
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": model_id,
        "role": role,
        "name": name,
        "status": status,
        "usage": usage,
        "is_live": bool(is_live),
        "is_shadow": bool(is_shadow),
        "is_legacy": bool(is_legacy),
        "data_source": data_source,
        "dataset_id": dataset_id,
        "trained_at": trained_at,
        "train_start_date": train_start_date,
        "train_end_date": train_end_date,
        "signal_date": signal_date,
        "artifact_path": artifact_path,
        "detail": detail or {},
    }


def _safe_nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return ""
        current = current.get(key)
    return current


def models_summary() -> dict[str, Any]:
    active = active_manifest_summary()
    active_payload = _read_json(ACTIVE_MANIFEST_PATH)
    production_manifest_path = _resolve_existing_project_path(active_payload.get("production_manifest_json", ""))
    production_manifest = _read_json(production_manifest_path) if production_manifest_path and production_manifest_path.exists() else {}
    path_studies_root = PROJECT_ROOT / "output" / "path_policy" / "studies"
    path_summary_path = Path("")
    if path_studies_root.exists():
        summaries = sorted(path_studies_root.glob("*/study_summary.json"), key=lambda item: item.stat().st_mtime, reverse=True)
        path_summary_path = summaries[0] if summaries else Path("")
    path_summary = _read_json(path_summary_path) if path_summary_path and path_summary_path.exists() else {}
    models = [
        _model_card(
            model_id="active-live",
            role="live",
            name=active.get("strategy_name") or active.get("candidate_label") or "active execution",
            status="active",
            usage="当前交易计划默认执行链路",
            is_live=True,
            data_source=str(active_payload.get("data_source", "")),
            trained_at=str(active_payload.get("promoted_at", "")),
            train_end_date=str(production_manifest.get("train_end_date", "")),
            signal_date=str(active_payload.get("effective_live_target_weight_mode", "")),
            artifact_path=str(active.get("path", "")),
            detail={"manifest": active_payload, "summary": active},
        ),
        _model_card(
            model_id="production-full-fit",
            role="production",
            name=str(production_manifest.get("strategy_name") or active.get("candidate_label") or "production full-fit"),
            status="manual",
            usage="显式 production retrain 后供交易计划读取",
            data_source=str(active_payload.get("data_source", "")),
            trained_at=str(production_manifest.get("created_at", "")),
            train_start_date=str(production_manifest.get("train_start_date", "")),
            train_end_date=str(production_manifest.get("train_end_date", "")),
            artifact_path=str(production_manifest.get("active_production_run_dir", "")),
            detail={"manifest_path": str(production_manifest_path), "manifest": production_manifest},
        ),
        _model_card(
            model_id="path-policy-research",
            role="path_policy",
            name=str(path_summary.get("study_tag") or path_summary.get("tag") or (path_summary_path.parent.name if path_summary_path else "path policy research")),
            status=str(path_summary.get("status") or path_summary.get("evidence_verdict") or "research"),
            usage="多 horizon 交易效用排序研究",
            is_shadow=True,
            data_source="lake",
            dataset_id=", ".join(str(item) for item in path_summary.get("dataset_ids", []) if str(item)) if isinstance(path_summary.get("dataset_ids"), list) else "",
            trained_at=str(path_summary.get("completed_at") or path_summary.get("created_at") or ""),
            artifact_path=str(path_summary_path),
            detail=path_summary,
        ),
        _model_card(
            model_id="legacy-ml",
            role="legacy",
            name="legacy ML trade model",
            status="legacy",
            usage="旧版 ML 兼容入口，默认 UI 隐藏",
            is_legacy=True,
            data_source="legacy",
            artifact_path=str((EXECUTION_DIR / "models" / "latest_ml_model.joblib").resolve()),
        ),
    ]
    return {"status": "ok", "models": models, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}


def model_detail(model_id: str) -> dict[str, Any]:
    payload = models_summary()
    for item in payload["models"]:
        if item["id"] == model_id:
            return {"status": "ok", "model": item}
    raise KeyError(f"Unknown model id: {model_id}")


def data_sources_summary(*, dataset_limit: int = 60) -> dict[str, Any]:
    from daily_research.data_lake import DEFAULT_DATA_LAKE_ROOT, ResearchDataLake

    lake_root = DEFAULT_DATA_LAKE_ROOT
    datasets: list[dict[str, Any]] = []
    catalog_status = "missing"
    try:
        lake = ResearchDataLake(lake_root)
        frame = lake.list_datasets()
        catalog_status = "ok"
        if not frame.empty:
            frame = frame.sort_values("created_at", ascending=False).head(int(dataset_limit))
            datasets = [
                {
                    "dataset_id": str(row.get("dataset_id", "")),
                    "dataset_kind": str(row.get("dataset_kind", "")),
                    "domain": str(row.get("domain", "")),
                    "zone": str(row.get("zone", "")),
                    "status": str(row.get("status", "")),
                    "start_date": str(row.get("start_date", "")),
                    "end_date": str(row.get("end_date", "")),
                    "created_at": str(row.get("created_at", "")),
                }
                for _, row in frame.iterrows()
            ]
    except Exception as exc:
        catalog_status = f"unavailable:{exc}"
    platform_runs = PROJECT_ROOT / "output" / "research_data_lake" / "data_platform" / "runs"
    latest_refresh = ""
    if platform_runs.exists():
        runs = sorted(platform_runs.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True)
        latest_refresh = str(runs[0].resolve()) if runs else ""
    latest_manifest_path = _latest_refresh_manifest_path()
    latest_manifest = _read_json(Path(latest_manifest_path)) if latest_manifest_path else {}
    try:
        latest_policy_input = _latest_policy_input_lake_dataset(lake_root=lake_root)
    except Exception:
        latest_policy_input = {}
    active = active_manifest_summary()
    active_dataset_id = str(active.get("lake_dataset_id", "") or active.get("source_market_dataset_id", "") or "")
    latest_policy_input_dataset_id = str(latest_policy_input.get("dataset_id", "") or "")
    latest_policy_input_end_date = _normalize_date_text(latest_policy_input.get("end_date", ""))
    active_dataset_end_date = _normalize_date_text(active.get("lake_dataset_end_date", "") or latest_policy_input_end_date)
    if not active_dataset_id or not latest_policy_input_dataset_id:
        dataset_sync_status = "unknown"
    elif active_dataset_id == latest_policy_input_dataset_id:
        dataset_sync_status = "synced"
    else:
        dataset_sync_status = "stale"
    try:
        latest_completed = str(get_latest_completed_trading_date())
    except Exception:
        latest_completed = ""
    latest_completed_date = _normalize_date_text(latest_completed)
    is_synced = dataset_sync_status == "synced"
    is_current_dataset_complete = bool(active_dataset_id and latest_completed_date and _date_covers(active_dataset_end_date, latest_completed_date))
    is_current_dataset_latest = bool(is_synced and latest_policy_input_dataset_id and latest_completed_date and _date_covers(latest_policy_input_end_date, latest_completed_date))
    latest_manifest_status = str(latest_manifest.get("status", "") or "")
    if is_current_dataset_latest and is_current_dataset_complete:
        current_dataset_status = "latest_complete"
        next_refresh_action = "skip"
        refresh_explanation = f"当前 active dataset 已覆盖最新完成交易日 {latest_completed_date}。"
    elif not active_dataset_id:
        current_dataset_status = "missing_active_dataset"
        next_refresh_action = "refresh"
        refresh_explanation = "当前 active manifest 未接入 policy_input_bundle，需要刷新并同步数据集。"
    elif not is_synced:
        current_dataset_status = "stale_active_dataset"
        next_refresh_action = "refresh"
        refresh_explanation = "当前 active dataset 不是最新 policy_input_bundle，需要同步到最新完整数据。"
    elif not is_current_dataset_complete:
        current_dataset_status = "incomplete"
        next_refresh_action = "refresh"
        refresh_explanation = f"当前 active dataset 只覆盖到 {active_dataset_end_date or '未知'}，需要补齐到 {latest_completed_date or '最新完成交易日'}。"
    else:
        current_dataset_status = "unknown"
        next_refresh_action = "refresh"
        refresh_explanation = "无法确认当前数据集是否完整，建议执行刷新。"
    signal_summary = production_signal.signal_panel_summary(active, latest_completed_date=latest_completed_date)
    anchor_summary = production_signal.audit_production_anchor()
    signal_panel_status = str(signal_summary.get("status", "unknown") or "unknown")
    next_signal_action = str(signal_summary.get("next_signal_action", "refresh") or "refresh")
    if next_refresh_action == "skip" and next_signal_action == "refresh":
        refresh_explanation = (
            f"{refresh_explanation} 但 production signal panels 只覆盖到 "
            f"{signal_summary.get('latest_date', '') or '未知'}，需要刷新生产信号。"
        )
    recommended_domains = list(FORMAL_DATA_PLATFORM_DOMAINS)
    state = _read_json(STATE_PATH) or load_runtime_state()
    provider_health = dict(state.get("last_provider_health", {}) or {})
    current_dataset_health = {
        "status": current_dataset_status,
        "is_latest": is_current_dataset_latest,
        "is_complete": is_current_dataset_complete,
        "active_dataset_id": active_dataset_id,
        "active_dataset_end_date": active_dataset_end_date,
        "latest_policy_input_dataset_id": latest_policy_input_dataset_id,
        "latest_policy_input_dataset_end_date": latest_policy_input_end_date,
    }
    signal_panel_health = {
        "status": signal_panel_status,
        "latest_date": str(signal_summary.get("latest_date", "") or ""),
        "target_panel_latest_date": str(signal_summary.get("target_panel_latest_date", "") or ""),
        "score_panel_latest_date": str(signal_summary.get("score_panel_latest_date", "") or ""),
        "required_date": str(signal_summary.get("required_date", "") or latest_completed_date),
        "next_signal_action": next_signal_action,
    }
    return {
        "status": "ok",
        "lake_root": str((PROJECT_ROOT / "output" / "research_data_lake").resolve()),
        "formal_provider_plan": FORMAL_DATA_PLATFORM_PROVIDER_PLAN,
        "domain_matrix": provider_capability_matrix(FORMAL_DATA_PLATFORM_PROVIDER_PLAN),
        "provider_health": provider_health,
        "current_dataset_health": current_dataset_health,
        "signal_panel_health": signal_panel_health,
        "catalog_status": catalog_status,
        "datasets": datasets,
        "active_dataset_id": active_dataset_id,
        "latest_policy_input_dataset_id": latest_policy_input_dataset_id,
        "latest_policy_input_dataset_end_date": latest_policy_input_end_date,
        "active_dataset_end_date": active_dataset_end_date,
        "dataset_sync_status": dataset_sync_status,
        "current_dataset_status": current_dataset_status,
        "is_current_dataset_latest": is_current_dataset_latest,
        "is_current_dataset_complete": is_current_dataset_complete,
        "next_refresh_action": next_refresh_action,
        "signal_panel_status": signal_panel_status,
        "signal_panel_latest_date": str(signal_summary.get("latest_date", "") or ""),
        "signal_panel_target_latest_date": str(signal_summary.get("target_panel_latest_date", "") or ""),
        "signal_panel_score_latest_date": str(signal_summary.get("score_panel_latest_date", "") or ""),
        "next_signal_action": next_signal_action,
        "production_anchor_status": str(anchor_summary.get("status", "unknown") or "unknown"),
        "production_anchor": anchor_summary,
        "signal_panels": signal_summary,
        "refresh_explanation": refresh_explanation,
        "data_platform": {
            "runs_root": str(platform_runs.resolve()),
            "latest_refresh_run": latest_refresh,
            "latest_refresh_manifest_path": latest_manifest_path,
            "latest_refresh_manifest_status": latest_manifest_status,
            "latest_refresh_registered_dataset_id": str(
                latest_manifest.get("registered_market_dataset_id", "")
                or latest_manifest.get("policy_input_dataset_id", "")
                or ""
            ),
            "latest_refresh_blockers": list(latest_manifest.get("blockers", []) if isinstance(latest_manifest.get("blockers"), list) else []),
            "provider_plan": FORMAL_DATA_PLATFORM_PROVIDER_PLAN,
            "latest_completed_trading_date": latest_completed,
            "recommended_domains": recommended_domains,
            "default_refresh": {
                "as_of_date": latest_completed,
                "universe": "all_a",
                "domains": recommended_domains,
                "required_domains": list(FORMAL_DATA_PLATFORM_REQUIRED_DOMAINS),
                "provider_plan": FORMAL_DATA_PLATFORM_PROVIDER_PLAN,
            },
        },
    }


def data_refresh_skip_payload(*, data_sources: dict[str, Any], as_of_date: str = "") -> dict[str, Any]:
    dataset_id = str(data_sources.get("active_dataset_id", "") or data_sources.get("latest_policy_input_dataset_id", "") or "")
    latest_completed = _normalize_date_text(
        as_of_date
        or _safe_nested(data_sources, "data_platform", "latest_completed_trading_date")
        or _safe_nested(data_sources, "data_platform", "default_refresh", "as_of_date")
    )
    explanation = str(data_sources.get("refresh_explanation", "") or "")
    job_id = f"data-platform-refresh_skipped_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    summary_note = explanation or f"当前 active dataset 已覆盖最新完成交易日 {latest_completed}。"
    try:
        paper_result = paper_account_reconcile(as_of_date=latest_completed)
    except Exception as exc:
        paper_result = {"status": "warning", "as_of_date": latest_completed, "error": str(exc)}
    return {
        "job_id": job_id,
        "task_name": "data-platform-refresh",
        "status": "skipped",
        "business_status": "synced",
        "runner_status": "ok",
        "artifact_status": "not_applicable",
        "dataset_id": dataset_id,
        "active_dataset_id": dataset_id,
        "latest_completed_trading_date": latest_completed,
        "summary_note": summary_note,
        "refresh_explanation": explanation,
        "paper_trading": paper_result,
        "metadata": {
            "job_id": job_id,
            "task_name": "data-platform-refresh",
            "status": "skipped",
            "business_status": "synced",
            "runner_status": "ok",
            "artifact_status": "not_applicable",
            "active_dataset_id": dataset_id,
            "latest_completed_trading_date": latest_completed,
            "summary_note": summary_note,
            "paper_trading": paper_result,
        },
    }


def provider_health_summary(
    *,
    provider_plan: str = FORMAL_DATA_PLATFORM_PROVIDER_PLAN,
    as_of_date: str = "",
    domains: list[str] | tuple[str, ...] | None = None,
    symbols: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    resolved_as_of = _normalize_date_text(as_of_date or get_latest_completed_trading_date())
    payload = run_provider_health(
        ProviderHealthConfig(
            provider_plan=str(provider_plan or FORMAL_DATA_PLATFORM_PROVIDER_PLAN),
            as_of_date=resolved_as_of,
            domains=tuple(domains or FORMAL_DATA_PLATFORM_REQUIRED_DOMAINS),
            symbols=tuple(symbols or ("000001.SZ", "600000.SH", "000300.SH")),
        )
    )
    state = _read_json(STATE_PATH) or load_runtime_state()
    state["last_provider_health"] = payload
    write_json_file(STATE_PATH, state)
    return payload


def _has_active_runtime_job() -> bool:
    lock_payload = _read_json(LOCK_PATH)
    if lock_payload:
        if _recover_stale_running_lock(lock_payload):
            lock_payload = _read_json(LOCK_PATH)
            if not lock_payload:
                return bool(list_active_thread_job_ids())
        if not _lock_payload_refers_to_finished_job(lock_payload):
            return True
        if not _clear_runtime_lock_payload(lock_payload):
            return True
    return bool(list_active_thread_job_ids())


def _lock_payload_refers_to_finished_job(payload: dict[str, Any]) -> bool:
    job_id = str(payload.get("job_id", "") or "").strip() if isinstance(payload, dict) else ""
    if not job_id:
        return False
    metadata = _read_json(JOBS_ROOT / job_id / "metadata.json")
    status = str(metadata.get("status", "") or "").strip().lower()
    return status in {"succeeded", "failed", "blocked", "abandoned", "skipped"}


def _parse_iso_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return None


def _job_runtime_age_seconds(metadata: dict[str, Any], payload: dict[str, Any]) -> float:
    started_at = _parse_iso_datetime(metadata.get("started_at")) or _parse_iso_datetime(payload.get("acquired_at"))
    if started_at is None:
        return 0.0
    now = datetime.now(started_at.tzinfo) if started_at.tzinfo is not None else datetime.now()
    return max((now - started_at).total_seconds(), 0.0)


def _pid_is_running(pid: Any) -> bool:
    try:
        clean_pid = int(pid or 0)
    except Exception:
        return False
    if clean_pid <= 0:
        return False
    if clean_pid == os.getpid():
        return True
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, clean_pid)
            if not handle:
                return False
            exit_code = ctypes.c_ulong()
            try:
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return False
                return int(exit_code.value) == 259
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            try:
                result = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {clean_pid}", "/FO", "CSV", "/NH"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                return f'"{clean_pid}"' in result.stdout or f",{clean_pid}," in result.stdout
            except Exception:
                return False
    try:
        os.kill(clean_pid, 0)
        return True
    except OSError:
        return False


def _recover_missing_subprocess_worker(payload: dict[str, Any], metadata: dict[str, Any] | None = None) -> bool:
    job_id = str(payload.get("job_id", "") or "").strip() if isinstance(payload, dict) else ""
    if not job_id:
        return False
    metadata_path = JOBS_ROOT / job_id / "metadata.json"
    loaded_metadata = metadata if isinstance(metadata, dict) and metadata else _read_json(metadata_path)
    if str(loaded_metadata.get("status", "") or "").strip().lower() not in {"queued", "running"}:
        return False
    if str(loaded_metadata.get("async_runner_mode", "") or "").strip().lower() == "detached_worker":
        launcher_pid = int(loaded_metadata.get("worker_launcher_pid", 0) or payload.get("worker_launcher_pid", 0) or 0)
        if launcher_pid > 0 and _pid_is_running(launcher_pid):
            return False
    if str(loaded_metadata.get("runner_mode", "") or "").strip().lower() != "subprocess":
        return False
    worker_pid = int(loaded_metadata.get("worker_pid", 0) or payload.get("worker_pid", 0) or 0)
    if worker_pid <= 0 or _pid_is_running(worker_pid):
        return False
    task_name = str(loaded_metadata.get("task_name", "") or payload.get("task_name", "") or "")
    warnings = list(loaded_metadata.get("runner_warnings", []) if isinstance(loaded_metadata.get("runner_warnings"), list) else [])
    _append_limited_warning(
        warnings,
        f"subprocess_worker_missing: worker_pid={worker_pid} server_pid={loaded_metadata.get('server_pid', payload.get('server_pid', ''))}",
    )
    job_paths = type("JobPathsRef", (), {"job_id": job_id, "metadata_path": metadata_path})()
    mark_job_finished(
        job_paths,
        status="blocked",
        exit_code=3,
        summary_note=f"Subprocess worker pid={worker_pid} is no longer running; job was marked blocked and the runtime lock was released.",
        business_status="blocked",
        runner_status="blocked",
        artifact_status=str(loaded_metadata.get("artifact_status", "") or ""),
        artifact_paths=loaded_metadata.get("artifact_paths", {}) if isinstance(loaded_metadata.get("artifact_paths", {}), dict) else {},
        evidence_paths=loaded_metadata.get("evidence_paths", {}) if isinstance(loaded_metadata.get("evidence_paths", {}), dict) else {},
        runner_warnings=warnings,
    )
    _clear_runtime_lock_payload({"job_id": job_id})
    append_event("subprocess_worker_missing_recovered", job_id=job_id, task_name=task_name, worker_pid=worker_pid)
    return True


def _recover_missing_detached_worker(payload: dict[str, Any], metadata: dict[str, Any] | None = None) -> bool:
    job_id = str(payload.get("job_id", "") or "").strip() if isinstance(payload, dict) else ""
    if not job_id:
        return False
    metadata_path = JOBS_ROOT / job_id / "metadata.json"
    loaded_metadata = metadata if isinstance(metadata, dict) and metadata else _read_json(metadata_path)
    if str(loaded_metadata.get("status", "") or "").strip().lower() not in {"queued", "running"}:
        return False
    if str(loaded_metadata.get("async_runner_mode", "") or "").strip().lower() != "detached_worker":
        return False
    launcher_pid = int(loaded_metadata.get("worker_launcher_pid", 0) or payload.get("worker_launcher_pid", 0) or 0)
    task_worker_pid = int(loaded_metadata.get("worker_pid", 0) or payload.get("worker_pid", 0) or 0)
    if launcher_pid > 0 and _pid_is_running(launcher_pid):
        return False
    if task_worker_pid > 0 and _pid_is_running(task_worker_pid):
        return False
    task_name = str(loaded_metadata.get("task_name", "") or payload.get("task_name", "") or "")
    warnings = list(loaded_metadata.get("runner_warnings", []) if isinstance(loaded_metadata.get("runner_warnings"), list) else [])
    _append_limited_warning(
        warnings,
        f"detached_worker_missing: worker_launcher_pid={launcher_pid} worker_pid={task_worker_pid}",
    )
    job_paths = type("JobPathsRef", (), {"job_id": job_id, "metadata_path": metadata_path})()
    mark_job_finished(
        job_paths,
        status="blocked",
        exit_code=3,
        summary_note=(
            f"Detached execution worker pid={launcher_pid or 'unknown'} is no longer running; "
            "job was marked blocked and the runtime lock was released."
        ),
        business_status="blocked",
        runner_status="blocked",
        artifact_status=str(loaded_metadata.get("artifact_status", "") or ""),
        artifact_paths=loaded_metadata.get("artifact_paths", {}) if isinstance(loaded_metadata.get("artifact_paths", {}), dict) else {},
        evidence_paths=loaded_metadata.get("evidence_paths", {}) if isinstance(loaded_metadata.get("evidence_paths", {}), dict) else {},
        runner_warnings=warnings,
    )
    _clear_runtime_lock_payload({"job_id": job_id})
    append_event("detached_worker_missing_recovered", job_id=job_id, task_name=task_name, worker_launcher_pid=launcher_pid)
    return True


def _recover_stale_running_lock(payload: dict[str, Any]) -> bool:
    job_id = str(payload.get("job_id", "") or "").strip() if isinstance(payload, dict) else ""
    if not job_id:
        return False
    metadata_path = JOBS_ROOT / job_id / "metadata.json"
    metadata = _read_json(metadata_path)
    if str(metadata.get("status", "") or "").strip().lower() not in {"queued", "running"}:
        return False
    if _recover_missing_detached_worker(payload, metadata):
        return True
    if _recover_missing_subprocess_worker(payload, metadata):
        return True
    if str(metadata.get("runner_mode", "") or "").strip():
        return False
    if int(metadata.get("worker_pid", 0) or payload.get("worker_pid", 0) or 0) > 0:
        return False
    task_name = str(metadata.get("task_name", "") or payload.get("task_name", "") or "")
    try:
        timeout_seconds = int(getattr(get_task_spec(task_name), "timeout_seconds", 0) or 0)
    except Exception:
        timeout_seconds = 0
    if timeout_seconds <= 0:
        return False
    age_seconds = _job_runtime_age_seconds(metadata, payload)
    grace_seconds = max(300, int(timeout_seconds * 0.1))
    if age_seconds < timeout_seconds + grace_seconds:
        return False
    job_paths = type("JobPathsRef", (), {"job_id": job_id, "metadata_path": metadata_path})()
    warnings = list(metadata.get("runner_warnings", []) if isinstance(metadata.get("runner_warnings"), list) else [])
    _append_limited_warning(
        warnings,
        f"stale_inprocess_runner_recovered: exceeded {timeout_seconds} seconds without worker_pid",
    )
    mark_job_finished(
        job_paths,
        status="blocked",
        exit_code=RUNNER_TIMEOUT_EXIT_CODE,
        summary_note=f"旧版 in-process runner 超过 {timeout_seconds} 秒且没有 worker_pid，已标记为 blocked 并释放 execution app 锁。",
        business_status="blocked",
        runner_status="blocked",
        artifact_status=str(metadata.get("artifact_status", "") or ""),
        artifact_paths=metadata.get("artifact_paths", {}) if isinstance(metadata.get("artifact_paths", {}), dict) else {},
        evidence_paths=metadata.get("evidence_paths", {}) if isinstance(metadata.get("evidence_paths", {}), dict) else {},
        runner_warnings=warnings,
    )
    _clear_runtime_lock_payload(payload)
    append_event("stale_inprocess_runner_recovered", job_id=job_id, task_name=task_name, age_seconds=age_seconds)
    return True


def _clear_runtime_lock_payload(payload: dict[str, Any]) -> bool:
    job_id = str(payload.get("job_id", "") or "").strip() if isinstance(payload, dict) else ""
    try:
        if LOCK_PATH.exists():
            current = _read_json(LOCK_PATH)
            if not job_id or str(current.get("job_id", "") or "") == job_id:
                LOCK_PATH.unlink()
        state = _read_json(STATE_PATH) or load_runtime_state()
        if not job_id or str(state.get("lock", {}).get("job_id", "") or "") == job_id:
            state["lock"] = {}
        if not job_id or str(state.get("current_job", {}).get("job_id", "") or "") == job_id:
            state["current_job"] = {}
        write_json_file(STATE_PATH, state)
        append_event("stale_lock_cleared", job_id=job_id)
        return True
    except Exception:
        return False


def _patch_runtime_worker_payload(*, job_id: str, worker_pid: int) -> None:
    clean_job_id = str(job_id or "").strip()
    pid = int(worker_pid or 0)
    if not clean_job_id or pid <= 0:
        return
    server_pid = int(os.getpid())
    if LOCK_PATH.exists():
        lock_payload = _read_json(LOCK_PATH)
        if str(lock_payload.get("job_id", "") or "") == clean_job_id:
            lock_payload.update(
                {
                    "pid": pid,
                    "server_pid": server_pid,
                    "worker_pid": pid,
                    "worker_started_at": now_iso(),
                }
            )
            write_json_file(LOCK_PATH, lock_payload)
    state = _read_json(STATE_PATH) or load_runtime_state()
    changed = False
    lock_state = dict(state.get("lock", {}) or {}) if isinstance(state.get("lock", {}), dict) else {}
    if str(lock_state.get("job_id", "") or "") == clean_job_id:
        lock_state.update({"pid": pid, "server_pid": server_pid, "worker_pid": pid})
        state["lock"] = lock_state
        changed = True
    current_job = dict(state.get("current_job", {}) or {}) if isinstance(state.get("current_job", {}), dict) else {}
    if str(current_job.get("job_id", "") or "") == clean_job_id:
        current_job.update({"worker_pid": pid, "server_pid": server_pid})
        state["current_job"] = current_job
        changed = True
    if changed:
        write_json_file(STATE_PATH, state)


def _signal_refresh_metadata(*, as_of_date: str = "") -> dict[str, Any]:
    try:
        result = production_signal.refresh_production_live_panels(as_of_date=as_of_date)
    except Exception as exc:
        return {
            "business_status": "signal_failed",
            "runner_status": "ok",
            "artifact_status": "failed",
            "signal_refresh_error": str(exc),
            "artifact_paths": {},
            "evidence_paths": {},
        }
    metadata = {
        "business_status": "ok",
        "runner_status": "ok",
        "artifact_status": "ok",
        "signal_refresh": result.get("signal_refresh", result),
        "signal_refresh_manifest_path": str(result.get("signal_refresh_manifest_path", "") or ""),
        "panel_latest_date": str(result.get("panel_latest_date", "") or ""),
        "signal_panel_latest_date": str(result.get("signal_panel_latest_date", "") or ""),
        "artifact_paths": result.get("artifact_paths", {}) if isinstance(result.get("artifact_paths", {}), dict) else {},
        "evidence_paths": result.get("evidence_paths", {}) if isinstance(result.get("evidence_paths", {}), dict) else {},
    }
    return metadata


def _signal_refresh_artifact_status() -> dict[str, Any]:
    active = active_manifest_summary()
    production_root = Path(str(active.get("production_root", "") or ""))
    manifest_path = production_root / "live_panel_refresh_manifest.json" if str(production_root) else Path("")
    manifest = _read_json(manifest_path) if manifest_path and manifest_path.exists() else {}
    status = str(manifest.get("status", "") or "").lower()
    panel_status = manifest.get("panel_status", {}) if isinstance(manifest.get("panel_status"), dict) else {}
    artifact_paths = {
        "live_panel_refresh_manifest": str(manifest_path.resolve()) if manifest_path and manifest_path.exists() else "",
        "target_panel": str(panel_status.get("target_panel", "") or active.get("trade_plan_target_weight_panel_csv", "") or ""),
        "score_panel": str(panel_status.get("score_panel", "") or active.get("trade_plan_score_panel_csv", "") or ""),
        "production_anchor_sync_manifest": str(
            _safe_nested(manifest, "anchor_sync", "manifest_path")
            or active.get("production_anchor_sync_manifest", "")
            or ""
        ),
    }
    return {
        "business_status": "ok" if status == "ok" else (status or "missing"),
        "runner_status": "ok",
        "artifact_status": status or "missing",
        "signal_refresh": manifest,
        "signal_refresh_manifest_path": artifact_paths["live_panel_refresh_manifest"],
        "panel_latest_date": str(panel_status.get("latest_date", "") or ""),
        "signal_panel_latest_date": str(panel_status.get("latest_date", "") or ""),
        "artifact_paths": artifact_paths,
        "evidence_paths": artifact_paths,
    }


def _safe_paper_reconcile(*, as_of_date: str = "") -> dict[str, Any]:
    try:
        return paper_account_reconcile(as_of_date=as_of_date)
    except Exception as exc:
        return {"status": "warning", "error": str(exc), "as_of_date": str(as_of_date or "")}


def _select_latest_policy_input_lake_dataset(frame: Any) -> dict[str, Any]:
    import pandas as pd

    if frame is None or getattr(frame, "empty", True):
        return {}
    filtered = frame.copy()
    if "dataset_kind" in filtered.columns:
        filtered = filtered.loc[filtered["dataset_kind"].astype(str).eq("policy_input_bundle")].copy()
    if "dataset_id" in filtered.columns:
        filtered = filtered.loc[filtered["dataset_id"].astype(str).str.startswith("policy_input_bundle__")].copy()
    if "status" in filtered.columns:
        status = filtered["status"].astype(str).str.lower()
        filtered = filtered.loc[~status.isin({"blocked", "failed", "error"})].copy()
    if filtered.empty:
        return {}
    filtered["_end_date_sort"] = pd.to_datetime(filtered.get("end_date", ""), errors="coerce")
    filtered["_created_at_sort"] = pd.to_datetime(filtered.get("created_at", ""), errors="coerce")
    filtered = filtered.sort_values(["_end_date_sort", "_created_at_sort", "dataset_id"], ascending=[False, False, False])
    row = filtered.iloc[0].to_dict()
    return {
        "dataset_id": str(row.get("dataset_id", "") or ""),
        "end_date": str(row.get("end_date", "") or ""),
        "start_date": str(row.get("start_date", "") or ""),
        "created_at": str(row.get("created_at", "") or ""),
        "source": str(row.get("source", "") or ""),
        "status": str(row.get("status", "") or ""),
    }


def _latest_policy_input_lake_dataset(lake_root: Path | str | None = None) -> dict[str, Any]:
    from daily_research.data_lake import DEFAULT_DATA_LAKE_ROOT, ResearchDataLake

    lake = ResearchDataLake(lake_root or DEFAULT_DATA_LAKE_ROOT)
    frame = lake.list_datasets(dataset_kind="policy_input_bundle")
    return _select_latest_policy_input_lake_dataset(frame)


def sync_active_manifest_to_latest_lake_dataset(
    *,
    reason: str = "",
    refresh_manifest_path: str = "",
    lake_root: Path | str | None = None,
) -> dict[str, Any]:
    freeze_guard.assert_write_action_allowed("active_manifest_write")
    payload = _read_json(ACTIVE_MANIFEST_PATH)
    if not payload:
        raise FileNotFoundError(f"active execution strategy manifest 缺失：{ACTIVE_MANIFEST_PATH}")
    latest = _latest_policy_input_lake_dataset(lake_root=lake_root)
    dataset_id = str(latest.get("dataset_id", "") or "").strip()
    if not dataset_id:
        raise ValueError("未找到可用的 policy_input_bundle lake dataset。")
    previous_dataset_id = str(payload.get("lake_dataset_id", "") or payload.get("source_market_dataset_id", "") or "").strip()
    payload["data_source"] = "lake"
    payload["lake_dataset_id"] = dataset_id
    payload["source_market_dataset_id"] = dataset_id
    payload["data_lake_root"] = str((Path(lake_root) if lake_root is not None else PROJECT_ROOT / "output" / "research_data_lake").resolve())
    if latest.get("end_date"):
        payload["lake_dataset_end_date"] = str(latest.get("end_date", ""))
    if latest.get("start_date"):
        payload["lake_dataset_start_date"] = str(latest.get("start_date", ""))
    if latest.get("created_at"):
        payload["lake_dataset_created_at"] = str(latest.get("created_at", ""))
    payload["latest_data_refresh_synced_at"] = now_iso()
    payload["latest_data_refresh_sync_reason"] = str(reason or "manual")
    if refresh_manifest_path:
        payload["latest_data_refresh_manifest"] = str(Path(refresh_manifest_path).resolve())
    temp_path = ACTIVE_MANIFEST_PATH.with_suffix(ACTIVE_MANIFEST_PATH.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(ACTIVE_MANIFEST_PATH)
    result = {
        "status": "ok",
        "updated": dataset_id != previous_dataset_id,
        "previous_dataset_id": previous_dataset_id,
        "dataset_id": dataset_id,
        "end_date": str(latest.get("end_date", "") or ""),
        "manifest_path": str(ACTIVE_MANIFEST_PATH.resolve()),
        "refresh_manifest_path": str(Path(refresh_manifest_path).resolve()) if refresh_manifest_path else "",
    }
    append_event("active_manifest_lake_dataset_synced", reason=str(reason or ""), **result)
    return result


def _latest_refresh_manifest_path() -> str:
    runs_root = PROJECT_ROOT / "output" / "research_data_lake" / "data_platform" / "runs"
    if not runs_root.exists():
        return ""
    candidates = [path / "refresh_manifest.json" for path in runs_root.iterdir() if path.is_dir() and (path / "refresh_manifest.json").exists()]
    if not candidates:
        return ""
    latest = max(candidates, key=lambda item: item.stat().st_mtime)
    return str(latest.resolve())


def _arg_value(args: list[str] | tuple[str, ...], name: str) -> str:
    values = [str(item) for item in (args or [])]
    for idx, item in enumerate(values):
        if item == name and idx + 1 < len(values):
            return str(values[idx + 1])
        prefix = f"{name}="
        if item.startswith(prefix):
            return str(item[len(prefix) :])
    return ""


def _refresh_manifest_path_from_args(args: list[str] | tuple[str, ...]) -> str:
    data_lake_root = _arg_value(args, "--data-lake-root")
    run_id = _arg_value(args, "--run-id")
    if not data_lake_root or not run_id:
        return ""
    root = Path(data_lake_root)
    if not root.is_absolute():
        root = WORKSPACE_ROOT / root
    return str((root / "data_platform" / "runs" / run_id / "refresh_manifest.json").resolve())


def _run_uses_custom_data_lake(args: list[str] | tuple[str, ...]) -> bool:
    root_text = _arg_value(args, "--data-lake-root")
    if not root_text:
        return False
    root = Path(root_text)
    if not root.is_absolute():
        root = WORKSPACE_ROOT / root
    try:
        return root.resolve() != (PROJECT_ROOT / "output" / "research_data_lake").resolve()
    except Exception:
        return True


def _data_refresh_artifact_status(refresh_manifest_path: str = "", *, allow_latest_fallback: bool = True) -> dict[str, Any]:
    manifest_path = str(refresh_manifest_path or (_latest_refresh_manifest_path() if allow_latest_fallback else "") or "").strip()
    manifest_file = Path(manifest_path) if manifest_path else Path("")
    manifest = _read_json(manifest_file) if manifest_path and manifest_file.exists() else {}
    status = str(manifest.get("status", "") or "").strip().lower()
    dataset_id = str(
        manifest.get("registered_market_dataset_id", "")
        or manifest.get("policy_input_dataset_id", "")
        or manifest.get("registered_dataset_id", "")
        or ""
    ).strip()
    artifact_status = status or ("missing" if not manifest_path or not manifest_file.exists() else "unknown")
    business_status = "ok" if status in {"ok", "skipped"} and (dataset_id or status == "skipped") else artifact_status
    return {
        "business_status": business_status,
        "artifact_status": artifact_status,
        "artifact_paths": {"refresh_manifest": str(Path(manifest_path).resolve()) if manifest_path else ""},
        "evidence_paths": {"refresh_manifest": str(Path(manifest_path).resolve()) if manifest_path else ""},
        "refresh_manifest": manifest,
        "refresh_dataset_id": dataset_id,
        "refresh_manifest_path": str(Path(manifest_path).resolve()) if manifest_path else "",
    }


def _post_process_successful_task(
    *,
    job_paths: Any,
    task_name: str,
    summary_note: str,
    passthrough_args: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, dict[str, Any]]:
    if task_name == "refresh-production-live-panels":
        metadata = _signal_refresh_artifact_status()
        paper_result = _safe_paper_reconcile(
            as_of_date=str(metadata.get("panel_latest_date", "") or metadata.get("signal_panel_latest_date", "") or "")
        )
        metadata["paper_trading"] = paper_result
        status = str(metadata.get("business_status", ""))
        note = f"{summary_note} signal_refresh_status={status} paper_reconcile_status={paper_result.get('status', '')}"
        update_job_metadata(job_paths, **metadata)
        return note, metadata
    if task_name == "trade-plan":
        paper_result = paper_account_register_latest_plan()
        metadata = {
            "business_status": "ok",
            "runner_status": "ok",
            "artifact_status": "ok",
            "paper_trading": paper_result,
        }
        note = f"{summary_note} paper_order_status={paper_result.get('status', '')}"
        update_job_metadata(job_paths, **metadata)
        return note, metadata
    if task_name != "data-platform-refresh":
        return summary_note, {"business_status": "ok", "runner_status": "ok", "artifact_status": "not_applicable"}
    clean_args = [str(item) for item in (passthrough_args or [])]
    refresh_manifest_path = _refresh_manifest_path_from_args(clean_args)
    custom_data_lake = _run_uses_custom_data_lake(clean_args)
    allow_latest_fallback = not refresh_manifest_path and not custom_data_lake
    if allow_latest_fallback:
        refresh_manifest_path = _latest_refresh_manifest_path()
    artifact_metadata = _data_refresh_artifact_status(
        refresh_manifest_path,
        allow_latest_fallback=allow_latest_fallback,
    )
    if artifact_metadata.get("business_status") != "ok":
        status = str(artifact_metadata.get("business_status", "") or artifact_metadata.get("artifact_status", "unknown"))
        note = f"{summary_note} refresh manifest business_status={status}; active manifest unchanged"
        update_job_metadata(job_paths, **artifact_metadata)
        return note, artifact_metadata
    if custom_data_lake:
        artifact_metadata["active_manifest_update"] = {
            "status": "skipped",
            "reason": "custom_data_lake_root",
            "refresh_manifest_path": str(artifact_metadata.get("refresh_manifest_path", "") or ""),
        }
        artifact_metadata["business_status"] = "ok"
        artifact_metadata["runner_status"] = "ok"
        note = f"{summary_note} custom data lake refresh completed; active manifest unchanged"
        update_job_metadata(job_paths, **artifact_metadata)
        return note, artifact_metadata
    update = sync_active_manifest_to_latest_lake_dataset(
        reason=task_name,
        refresh_manifest_path=refresh_manifest_path,
    )
    note = f"{summary_note} active manifest lake_dataset_id={update.get('dataset_id', '')}"
    signal_metadata = _signal_refresh_metadata(as_of_date=str(update.get("end_date", "") or ""))
    paper_result = _safe_paper_reconcile(as_of_date=str(update.get("end_date", "") or ""))
    metadata = {
        **artifact_metadata,
        "active_manifest_update": update,
        "data_refresh": artifact_metadata,
        "signal_refresh_result": signal_metadata,
        "paper_trading": paper_result,
        "business_status": "ok" if signal_metadata.get("business_status") == "ok" else "signal_failed",
        "runner_status": signal_metadata.get("runner_status", "ok"),
        "artifact_status": "ok" if signal_metadata.get("artifact_status") == "ok" else str(signal_metadata.get("artifact_status", "")),
        "artifact_paths": {
            **(artifact_metadata.get("artifact_paths", {}) if isinstance(artifact_metadata.get("artifact_paths", {}), dict) else {}),
            **(signal_metadata.get("artifact_paths", {}) if isinstance(signal_metadata.get("artifact_paths", {}), dict) else {}),
        },
        "evidence_paths": {
            **(artifact_metadata.get("evidence_paths", {}) if isinstance(artifact_metadata.get("evidence_paths", {}), dict) else {}),
            **(signal_metadata.get("evidence_paths", {}) if isinstance(signal_metadata.get("evidence_paths", {}), dict) else {}),
        },
        "signal_refresh_manifest_path": str(signal_metadata.get("signal_refresh_manifest_path", "") or ""),
        "panel_latest_date": str(signal_metadata.get("panel_latest_date", "") or ""),
    }
    update_job_metadata(job_paths, **metadata)
    return note, metadata


def save_account_snapshot(
    *,
    available_cash: float | int | str | None,
    positions: list[dict[str, Any]] | None,
    path: Path | None = None,
) -> dict[str, Any]:
    freeze_guard.assert_write_action_allowed("save_account_snapshot")
    resolved_path = path or POSITIONS_PATH
    cash_value: float | None = None
    if available_cash not in {None, ""}:
        try:
            cash_value = float(available_cash)
        except ValueError as exc:
            raise ValueError(f"可用现金不是有效数字：{available_cash}") from exc
        if cash_value < 0:
            raise ValueError("可用现金不能为负数。")

    normalized_positions = []
    for row_index, position in enumerate(positions or [], start=1):
        normalized = _normalize_position_record(position if isinstance(position, dict) else {}, row_index=row_index)
        if normalized:
            normalized_positions.append(normalized)

    rows = [
        {
            "record_type": "account",
            "stock": "",
            "shares": "",
            "cost_price": "",
            "available_cash": _format_decimal(cash_value),
        }
    ]
    rows.extend(
        {
            "record_type": "position",
            "stock": str(position["stock"]),
            "shares": str(int(position["shares"])),
            "cost_price": _format_decimal(float(position["cost_price"])),
            "available_cash": "",
        }
        for position in normalized_positions
    )
    _write_csv_atomic(resolved_path, rows=rows)
    if resolved_path == POSITIONS_PATH:
        paper_trading.ensure_ledger(db_path=PAPER_ACCOUNT_DB_PATH, snapshot_path=resolved_path)
        paper_trading.replace_account_state(
            db_path=PAPER_ACCOUNT_DB_PATH,
            available_cash=cash_value,
            positions=normalized_positions,
            reason="legacy /api/account save",
        )
        paper_trading.export_account_snapshot(db_path=PAPER_ACCOUNT_DB_PATH, path=resolved_path)
    append_event(
        "account_snapshot_saved",
        path=str(resolved_path.resolve()),
        position_count=len(normalized_positions),
        available_cash=cash_value,
    )
    return load_account_snapshot(resolved_path)


def reset_account_snapshot_from_example(
    *,
    target_path: Path | None = None,
    example_path: Path = POSITIONS_EXAMPLE_PATH,
) -> dict[str, Any]:
    resolved_target = target_path or POSITIONS_PATH
    if not example_path.exists():
        raise FileNotFoundError(f"未找到示例账户文件：{example_path}")
    example_snapshot = _load_account_snapshot_csv(example_path)
    restored = save_account_snapshot(
        available_cash=example_snapshot["available_cash"],
        positions=example_snapshot["positions"],
        path=resolved_target,
    )
    append_event("account_snapshot_reset_from_example", target_path=str(resolved_target.resolve()), example_path=str(example_path.resolve()))
    return restored


def paper_account_cash_flow(*, flow_type: str, amount: float | int | str, reason: str = "") -> dict[str, Any]:
    freeze_guard.assert_write_action_allowed("paper_account_cash_flow")
    _ensure_paper_account()
    payload = paper_trading.record_cash_flow(
        db_path=PAPER_ACCOUNT_DB_PATH,
        flow_type=str(flow_type or ""),
        amount=float(amount),
        reason=str(reason or ""),
    )
    paper_trading.export_account_snapshot(db_path=PAPER_ACCOUNT_DB_PATH, path=POSITIONS_PATH)
    append_event("paper_account_cash_flow", flow_type=str(flow_type or ""), amount=float(amount), reason=str(reason or ""))
    return {**paper_account_summary(), "cash_flow": payload}


def paper_account_manual_adjustment(
    *,
    adjustment_type: str,
    stock: str = "",
    shares: float | int | str | None = None,
    cost_price: float | int | str | None = None,
    amount: float | int | str | None = None,
    reason: str = "",
) -> dict[str, Any]:
    freeze_guard.assert_write_action_allowed("paper_account_manual_adjustment")
    _ensure_paper_account()
    payload = paper_trading.record_manual_adjustment(
        db_path=PAPER_ACCOUNT_DB_PATH,
        adjustment_type=adjustment_type,
        stock=stock,
        shares=shares,
        cost_price=cost_price,
        amount=amount,
        reason=reason,
    )
    paper_trading.export_account_snapshot(db_path=PAPER_ACCOUNT_DB_PATH, path=POSITIONS_PATH)
    append_event("paper_account_manual_adjustment", adjustment_type=adjustment_type, stock=stock, reason=reason)
    return {**_paper_summary_to_account_snapshot(payload, path=POSITIONS_PATH), "manual_adjustment_status": "ok"}


def _paper_price_lookup_for_date(as_of_date: str) -> dict[str, Any]:
    active_payload = _read_json(ACTIVE_MANIFEST_PATH)
    dataset_id = str(active_payload.get("lake_dataset_id", "") or active_payload.get("source_market_dataset_id", "") or "").strip()
    if not dataset_id:
        return {as_of_date: {}}
    lake_root = Path(str(active_payload.get("data_lake_root", "") or PROJECT_ROOT / "output" / "research_data_lake"))
    try:
        from daily_research.data_lake import ResearchDataLake, load_policy_inputs_from_lake

        lake = ResearchDataLake(lake_root)
        prepared = load_policy_inputs_from_lake(
            lake=lake,
            dataset_id=dataset_id,
            start_date=as_of_date,
            end_date=as_of_date,
            min_trading_days=1,
        )
        open_frame = prepared.open_.copy()
        close_frame = prepared.close.copy()
        ts = pd.Timestamp(as_of_date)
        if ts not in open_frame.index:
            return {as_of_date: {}}
        lookup: dict[str, dict[str, float]] = {}
        open_row = open_frame.loc[ts]
        close_row = close_frame.loc[ts] if ts in close_frame.index else pd.Series(dtype=float)
        for stock in open_row.index:
            open_value = pd.to_numeric(pd.Series([open_row.get(stock)]), errors="coerce").iloc[0]
            close_value = pd.to_numeric(pd.Series([close_row.get(stock)]), errors="coerce").iloc[0] if stock in close_row.index else float("nan")
            item: dict[str, float] = {}
            if pd.notna(open_value):
                item["open"] = float(open_value)
            if pd.notna(close_value):
                item["close"] = float(close_value)
            if item:
                lookup[str(stock)] = item
        return {as_of_date: lookup}
    except Exception as exc:
        append_event("paper_account_price_lookup_failed", as_of_date=as_of_date, error=str(exc))
        return {as_of_date: {}}


def paper_account_register_latest_plan() -> dict[str, Any]:
    _ensure_paper_account()
    plan = latest_trade_plan_summary(max_lines=40)
    run_dir = str(_safe_nested(plan, "artifact_paths", "run_dir") or "")
    if not run_dir:
        return {"status": "missing_trade_plan_run", "trade_plan": plan}
    result = paper_trading.register_trade_plan_run(db_path=PAPER_ACCOUNT_DB_PATH, run_dir=run_dir)
    paper_trading.export_account_snapshot(db_path=PAPER_ACCOUNT_DB_PATH, path=POSITIONS_PATH)
    append_event("paper_account_trade_plan_registered", **{key: value for key, value in result.items() if key != "trade_plan"})
    return {**result, "trade_plan": plan}


def paper_account_apply_latest_plan(*, execution_date: str = "") -> dict[str, Any]:
    freeze_guard.assert_write_action_allowed("paper_account_apply_latest_plan")
    _ensure_paper_account()
    registration = paper_account_register_latest_plan()
    plan_summary = dict(_safe_nested(registration, "trade_plan", "summary") or {})
    resolved_execution_date = _normalize_date_text(execution_date or plan_summary.get("execution_date", ""))
    if not resolved_execution_date:
        return {"status": "missing_execution_date", "registration": registration, "account": paper_account_summary()}
    price_lookup = _paper_price_lookup_for_date(resolved_execution_date)
    apply_result = paper_trading.apply_pending_orders(
        db_path=PAPER_ACCOUNT_DB_PATH,
        price_lookup=price_lookup,
        execution_date=resolved_execution_date,
    )
    equity_result = paper_trading.write_daily_equity(
        db_path=PAPER_ACCOUNT_DB_PATH,
        price_lookup=price_lookup,
        as_of_date=resolved_execution_date,
    )
    paper_trading.export_account_snapshot(db_path=PAPER_ACCOUNT_DB_PATH, path=POSITIONS_PATH)
    result = {
        "status": "ok",
        "execution_date": resolved_execution_date,
        "registration": registration,
        "apply_result": apply_result,
        "equity_result": equity_result,
        "account": paper_account_summary(),
    }
    append_event("paper_account_latest_plan_applied", execution_date=resolved_execution_date, filled_order_count=apply_result.get("filled_order_count", 0))
    return result


def paper_account_reconcile(*, as_of_date: str = "") -> dict[str, Any]:
    _ensure_paper_account()
    resolved_date = _normalize_date_text(as_of_date or get_latest_completed_trading_date())
    price_lookup = _paper_price_lookup_for_date(resolved_date)
    apply_result = paper_trading.apply_pending_orders(
        db_path=PAPER_ACCOUNT_DB_PATH,
        price_lookup=price_lookup,
        execution_date=resolved_date,
    )
    equity_result = paper_trading.write_daily_equity(
        db_path=PAPER_ACCOUNT_DB_PATH,
        price_lookup=price_lookup,
        as_of_date=resolved_date,
    )
    paper_trading.export_account_snapshot(db_path=PAPER_ACCOUNT_DB_PATH, path=POSITIONS_PATH)
    return {"status": "ok", "as_of_date": resolved_date, "apply_result": apply_result, "equity_result": equity_result}


def paper_account_performance(*, start_date: str, end_date: str) -> dict[str, Any]:
    _ensure_paper_account()
    return paper_trading.performance_summary(
        db_path=PAPER_ACCOUNT_DB_PATH,
        start_date=_normalize_date_text(start_date),
        end_date=_normalize_date_text(end_date),
    )


def build_status_payload(*, history_limit: int = 8) -> dict[str, Any]:
    ensure_runtime_layout()
    _mark_abandoned_jobs()
    lock_payload = _read_json(LOCK_PATH)
    if lock_payload and _recover_stale_running_lock(lock_payload):
        lock_payload = _read_json(LOCK_PATH)
    if lock_payload and _lock_payload_refers_to_finished_job(lock_payload):
        _clear_runtime_lock_payload(lock_payload)
        lock_payload = _read_json(LOCK_PATH)
    state = _read_json(STATE_PATH) or load_runtime_state()
    recent_jobs = list_recent_job_metadata(limit=history_limit)
    warnings: list[str] = []
    yolos_python = resolve_project_python_executable(sys.executable)
    if not Path(yolos_python).exists():
        warnings.append(f"解析出的 yolos Python 不存在：{yolos_python}")
    if not ACTIVE_MANIFEST_PATH.exists():
        warnings.append("active execution strategy manifest 缺失")
    if not POSITIONS_PATH.exists():
        warnings.append("current_positions.csv 缺失")
    if lock_payload:
        warnings.append(
            "execution app 当前被锁定："
            f"job_id={lock_payload.get('job_id', '')} task={lock_payload.get('task_name', '')}"
        )
    try:
        paper_account = paper_account_summary()
    except Exception as exc:
        paper_account = {"status": "warning", "error": str(exc)}
    return {
        "runtime_root": str(RUNTIME_ROOT.resolve()),
        "state_path": str(STATE_PATH.resolve()),
        "events_path": str(EVENTS_PATH.resolve()),
        "jobs_root": str(JOBS_ROOT.resolve()),
        "yolos_python": yolos_python,
        "active_thread_job_ids": list_active_thread_job_ids(),
        "lock": lock_payload,
        "current_job": state.get("current_job", {}),
        "recent_jobs": recent_jobs,
        "active_manifest": active_manifest_summary(),
        "current_positions": positions_summary(),
        "paper_account": paper_account,
        "latest_trade_plan": latest_trade_plan_summary(max_lines=24),
        "warnings": warnings,
        "updated_at": state.get("updated_at", ""),
    }


def _mark_abandoned_jobs(limit: int = 50) -> None:
    lock_payload = _read_json(LOCK_PATH)
    locked_job_id = str(lock_payload.get("job_id", "") or "") if lock_payload else ""
    active_thread_ids = set(list_active_thread_job_ids())
    for metadata in list_recent_job_metadata(limit=limit):
        status = str(metadata.get("status", "") or "")
        job_id = str(metadata.get("job_id", "") or "")
        if status not in {"queued", "running"}:
            continue
        if job_id and (job_id == locked_job_id or job_id in active_thread_ids):
            continue
        job_paths = type("JobPathsRef", (), {"metadata_path": JOBS_ROOT / job_id / "metadata.json"})()
        update_job_metadata(
            job_paths,
            status="abandoned",
            completed_at=now_iso(),
            exit_code=3,
            summary_note="旧任务未持有运行时锁或活跃线程，已标记为 abandoned。",
        )


def _reconcile_running_job_metadata(job_id: str) -> dict[str, Any]:
    metadata = _read_json(JOBS_ROOT / str(job_id) / "metadata.json")
    if not metadata:
        return {}
    status = str(metadata.get("status", "") or "").strip().lower()
    if status not in {"queued", "running"}:
        return metadata
    lock_payload = _read_json(LOCK_PATH)
    if lock_payload and str(lock_payload.get("job_id", "") or "") == str(job_id):
        if _recover_stale_running_lock(lock_payload):
            return _read_json(JOBS_ROOT / str(job_id) / "metadata.json")
    if _recover_missing_detached_worker({"job_id": str(job_id), **(lock_payload if isinstance(lock_payload, dict) else {})}, metadata):
        return _read_json(JOBS_ROOT / str(job_id) / "metadata.json")
    if _recover_missing_subprocess_worker({"job_id": str(job_id), **(lock_payload if isinstance(lock_payload, dict) else {})}, metadata):
        return _read_json(JOBS_ROOT / str(job_id) / "metadata.json")
    return metadata


def build_doctor_payload() -> dict[str, Any]:
    ensure_runtime_layout()
    yolos_python = resolve_project_python_executable(sys.executable)
    checks: list[dict[str, Any]] = []

    def add_check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

    add_check("environment_source", ENVIRONMENT_PATH.exists(), f"应存在：{ENVIRONMENT_PATH.resolve()}")
    add_check("yolos_python", Path(yolos_python).exists(), yolos_python)
    add_check("active_manifest", ACTIVE_MANIFEST_PATH.exists() and bool(_read_json(ACTIVE_MANIFEST_PATH)), str(ACTIVE_MANIFEST_PATH.resolve()))
    positions = positions_summary()
    add_check("current_positions", positions["exists"] and positions["headers_ok"], f"{positions['path']} 行数={positions['row_count']}")
    add_check("positions_example", POSITIONS_EXAMPLE_PATH.exists(), str(POSITIONS_EXAMPLE_PATH.resolve()))
    missing_scripts = [spec.name for spec in list_task_specs() if not spec.script_path.exists()]
    add_check("task_registry_scripts", not missing_scripts, "全部任务脚本存在" if not missing_scripts else ", ".join(missing_scripts))
    add_check("runtime_layout", RUNTIME_ROOT.exists() and JOBS_ROOT.exists(), str(RUNTIME_ROOT.resolve()))
    lock_payload = _read_json(LOCK_PATH)
    if lock_payload and _lock_payload_refers_to_finished_job(lock_payload):
        _clear_runtime_lock_payload(lock_payload)
        lock_payload = _read_json(LOCK_PATH)
    add_check("runtime_lock", not bool(lock_payload), "未锁定" if not lock_payload else json.dumps(lock_payload, ensure_ascii=False))
    add_check("web_dependency_fastapi", importlib.util.find_spec("fastapi") is not None, "fastapi")
    add_check("web_dependency_uvicorn", importlib.util.find_spec("uvicorn") is not None, "uvicorn")
    add_check("latest_trade_plan_artifact", LATEST_TRADE_PLAN_PATH.exists(), str(LATEST_TRADE_PLAN_PATH.resolve()))
    overall_ok = all(bool(item["ok"]) for item in checks)
    return {"status": "ok" if overall_ok else "degraded", "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": checks}


def list_tasks_payload(*, core_only: bool = False) -> list[dict[str, Any]]:
    specs = list_core_frontend_task_specs() if core_only else list_task_specs()
    return [serialize_task_spec(spec) for spec in specs]


def list_jobs_payload(*, limit: int = 20) -> list[dict[str, Any]]:
    jobs = list_recent_job_metadata(limit=limit)
    for metadata in jobs:
        job_id = str(metadata.get("job_id", "") or "")
        if job_id and str(metadata.get("status", "") or "").strip().lower() in {"queued", "running"}:
            _reconcile_running_job_metadata(job_id)
    return list_recent_job_metadata(limit=limit)


def build_job_detail_payload(job_id: str, *, lines: int = 80) -> dict[str, Any]:
    metadata = _reconcile_running_job_metadata(str(job_id))
    if not metadata:
        metadata = load_job_metadata(str(job_id))
    if not metadata:
        raise FileNotFoundError(f"未找到作业元数据：{job_id}")
    stdout_lines = tail_file(Path(str(metadata.get("stdout_log", ""))), lines=lines)
    stderr_lines = tail_file(Path(str(metadata.get("stderr_log", ""))), lines=lines)
    status = str(metadata.get("status", "") or "")
    artifact_paths = metadata.get("artifact_paths", {}) if isinstance(metadata.get("artifact_paths", {}), dict) else {}
    evidence_paths = metadata.get("evidence_paths", {}) if isinstance(metadata.get("evidence_paths", {}), dict) else artifact_paths
    return {
        "job_id": str(metadata.get("job_id", "")),
        "task_name": str(metadata.get("task_name", "")),
        "status": status,
        "business_status": str(metadata.get("business_status", "") or ""),
        "runner_status": str(metadata.get("runner_status", "") or ""),
        "artifact_status": str(metadata.get("artifact_status", "") or ""),
        "artifact_paths": artifact_paths,
        "evidence_paths": evidence_paths,
        "runner_warnings": list(metadata.get("runner_warnings", []) if isinstance(metadata.get("runner_warnings"), list) else []),
        "metadata": metadata,
        "stdout_tail": stdout_lines,
        "stderr_tail": stderr_lines,
        "can_resume": status in {"failed", "blocked", "succeeded"},
    }


def iter_job_stream_events(job_id: str, *, poll_interval_seconds: float = 0.4, max_idle_seconds: float = 2.0):
    clean_job_id = str(job_id or "").strip()
    if not clean_job_id:
        yield {"event": "error", "job_id": "", "status": "error", "timestamp": now_iso(), "message": "job_id is required"}
        return
    stdout_offset = 0
    stderr_offset = 0
    stdout_line_no = 0
    stderr_line_no = 0
    idle_started_at = time.monotonic()
    sent_done = False

    while True:
        metadata = _reconcile_running_job_metadata(clean_job_id)
        if not metadata:
            metadata = load_job_metadata(clean_job_id)
        if not metadata:
            yield {
                "event": "error",
                "job_id": clean_job_id,
                "status": "missing",
                "timestamp": now_iso(),
                "message": f"未找到作业元数据：{clean_job_id}",
            }
            return

        status = str(metadata.get("status", "") or "")
        base = {"job_id": clean_job_id, "status": status, "timestamp": now_iso()}
        yield {
            "event": "snapshot",
            **base,
            "metadata": metadata,
            "progress": metadata.get("progress", {}) if isinstance(metadata.get("progress", {}), dict) else {},
        }

        stdout_delta = read_file_increment(Path(str(metadata.get("stdout_log", ""))), offset=stdout_offset)
        stdout_offset = int(stdout_delta.get("offset", stdout_offset) or 0)
        if stdout_delta.get("truncated"):
            stdout_line_no = 0
        for line in stdout_delta.get("lines", []):
            stdout_line_no += 1
            yield {"event": "stdout", **base, "stream": "stdout", "line": str(line), "line_no": stdout_line_no}

        stderr_delta = read_file_increment(Path(str(metadata.get("stderr_log", ""))), offset=stderr_offset)
        stderr_offset = int(stderr_delta.get("offset", stderr_offset) or 0)
        if stderr_delta.get("truncated"):
            stderr_line_no = 0
        for line in stderr_delta.get("lines", []):
            stderr_line_no += 1
            yield {"event": "stderr", **base, "stream": "stderr", "line": str(line), "line_no": stderr_line_no}

        progress = metadata.get("progress", {}) if isinstance(metadata.get("progress", {}), dict) else {}
        if progress:
            yield {"event": "progress", **base, "progress": progress, "metadata": metadata}

        if str(status).lower() not in {"queued", "running", "launch_pending"}:
            yield {"event": "done", **base, "metadata": metadata, "progress": progress}
            sent_done = True
            return
        if time.monotonic() - idle_started_at >= max_idle_seconds:
            idle_started_at = time.monotonic()
        if sent_done:
            return
        time.sleep(max(float(poll_interval_seconds), 0.1))


def resolve_resume_metadata(job_id: str = "") -> dict[str, Any]:
    requested_job_id = str(job_id or "").strip()
    if requested_job_id:
        metadata = load_job_metadata(requested_job_id)
        if not metadata:
            raise FileNotFoundError(f"未找到作业元数据：{requested_job_id}")
        return metadata
    for metadata in list_recent_job_metadata(limit=20):
        if str(metadata.get("status", "")) in {"failed", "blocked"}:
            return metadata
    raise FileNotFoundError("没有找到可恢复的失败或阻塞作业。")


def _run_existing_job(
    *,
    job_paths,
    task_name: str,
    command: list[str],
    python_executable: str,
    passthrough_args: list[str],
    job_label: str,
    resumed_from_job_id: str = "",
    force_unlock: bool = False,
    echo_output: bool = True,
) -> dict[str, Any]:
    spec = get_task_spec(task_name)
    exit_code = 1
    status = "failed"
    summary_note = ""
    post_process_metadata: dict[str, Any] = {}
    runner_warnings: list[str] = []
    try:
        with ExecutionAppLock(job_id=job_paths.job_id, task_name=task_name, force=force_unlock) as lock:
            mark_job_started(job_paths, lock_payload=lock.payload)
            with job_paths.stdout_path.open("w", encoding="utf-8") as stdout_handle:
                with job_paths.stderr_path.open("w", encoding="utf-8") as stderr_handle:
                    exit_code = _run_command_in_subprocess(
                        command=command,
                        stdout_handle=stdout_handle,
                        stderr_handle=stderr_handle,
                        console_stdout=sys.stdout if echo_output else None,
                        console_stderr=sys.stderr if echo_output else None,
                        job_paths=job_paths,
                        runner_warnings=runner_warnings,
                        timeout_seconds=int(getattr(spec, "timeout_seconds", 0) or 0),
                    )
            status = "succeeded" if exit_code == 0 else "failed"
            summary_note = spec.description
            if exit_code == RUNNER_TIMEOUT_EXIT_CODE:
                status = "blocked"
                summary_note = f"{spec.description} timed out after {int(getattr(spec, 'timeout_seconds', 0) or 0)} seconds"
            elif status == "succeeded":
                summary_note, post_process_metadata = _post_process_successful_task(
                    job_paths=job_paths,
                    task_name=task_name,
                    summary_note=summary_note,
                    passthrough_args=passthrough_args,
                )
                business_status = str(post_process_metadata.get("business_status", "") or "").lower()
                if business_status == "blocked":
                    status = "blocked"
                elif business_status in {"failed", "missing", "unknown"} or business_status.endswith("_failed"):
                    status = "failed"
            elif task_name == "data-platform-refresh":
                refresh_manifest_path = _refresh_manifest_path_from_args(passthrough_args)
                allow_latest_fallback = False
                if not refresh_manifest_path:
                    latest_candidate = _latest_refresh_manifest_path()
                    latest_metadata = _data_refresh_artifact_status(
                        latest_candidate,
                        allow_latest_fallback=False,
                    )
                    if latest_metadata.get("business_status") == "blocked":
                        refresh_manifest_path = latest_candidate
                artifact_metadata = _data_refresh_artifact_status(
                    refresh_manifest_path,
                    allow_latest_fallback=allow_latest_fallback,
                )
                if artifact_metadata.get("business_status") == "blocked":
                    status = "blocked"
                    summary_note = f"{spec.description} refresh manifest business_status=blocked"
                    post_process_metadata = artifact_metadata
                elif artifact_metadata.get("business_status") == "ok":
                    try:
                        if _run_uses_custom_data_lake(passthrough_args):
                            artifact_metadata["active_manifest_update"] = {
                                "status": "skipped",
                                "reason": "custom_data_lake_root",
                            }
                        else:
                            refresh_manifest_path = str(artifact_metadata.get("refresh_manifest_path", "") or "")
                            update = sync_active_manifest_to_latest_lake_dataset(
                                reason=task_name,
                                refresh_manifest_path=refresh_manifest_path,
                            )
                            artifact_metadata["active_manifest_update"] = update
                        update_job_metadata(job_paths, **artifact_metadata)
                    except Exception as exc:
                        runner_warnings.append(f"active_manifest_sync_after_reconcile_failed: {exc}")
                    status = "succeeded"
                    exit_code = 0
                    summary_note = f"{spec.description} reconciled from refresh manifest"
                    post_process_metadata = artifact_metadata
    except ExecutionAppLockError as exc:
        summary_note = str(exc)
        exit_code = 2
        status = "blocked"
    except Exception as exc:
        summary_note = str(exc)
        exit_code = 1
        status = "failed"
    finally:
        if runner_warnings:
            runner_status = "warning"
        elif int(exit_code) == 0 or str(post_process_metadata.get("business_status", "") or "").lower() == "blocked":
            runner_status = "ok"
        elif status == "blocked":
            runner_status = "blocked"
        else:
            runner_status = "failed"
        metadata_runner_status = str(post_process_metadata.get("runner_status", runner_status))
        if runner_warnings and metadata_runner_status == "ok":
            metadata_runner_status = "warning"
        metadata = mark_job_finished(
            job_paths,
            status=status,
            exit_code=exit_code,
            summary_note=summary_note,
            business_status=str(post_process_metadata.get("business_status", "ok" if status == "succeeded" else status)),
            runner_status=metadata_runner_status,
            artifact_status=str(post_process_metadata.get("artifact_status", "")),
            artifact_paths=post_process_metadata.get("artifact_paths", {}) if isinstance(post_process_metadata.get("artifact_paths", {}), dict) else {},
            evidence_paths=post_process_metadata.get("evidence_paths", {}) if isinstance(post_process_metadata.get("evidence_paths", {}), dict) else {},
            runner_warnings=runner_warnings,
        )
        if post_process_metadata:
            metadata.update(post_process_metadata)
            metadata["runner_warnings"] = runner_warnings
            metadata["runner_status"] = metadata_runner_status
        with _ACTIVE_JOB_THREADS_LOCK:
            _ACTIVE_JOB_THREADS.pop(job_paths.job_id, None)
    return {
        "job_id": job_paths.job_id,
        "task_name": task_name,
        "status": status,
        "exit_code": exit_code,
        "stdout_log": str(job_paths.stdout_path.resolve()),
        "stderr_log": str(job_paths.stderr_path.resolve()),
        "metadata": metadata,
        "job_label": job_label,
        "resumed_from_job_id": resumed_from_job_id,
        "passthrough_args": passthrough_args,
        "python_executable": python_executable,
    }


class _TeeTextIO:
    def __init__(self, *handles: TextIO | None, heartbeat: Any | None = None, runner_warnings: list[str] | None = None) -> None:
        self._handles = [handle for handle in handles if handle is not None]
        self._heartbeat = heartbeat
        self._buffer = ""
        self._runner_warnings = runner_warnings if runner_warnings is not None else []
        self._last_heartbeat_line = ""
        self._last_written_progress_line = ""

    def _record_warning(self, message: str) -> None:
        if len(self._runner_warnings) >= _RUNNER_WARNING_LIMIT:
            return
        if message not in self._runner_warnings:
            self._runner_warnings.append(message)

    def write(self, text: str) -> int:
        value = str(text)
        write_value = value
        if "\r" in value and "\n" not in value:
            clean_value = value.rstrip("\r").strip()
            if clean_value and clean_value == self._last_written_progress_line:
                write_value = ""
            else:
                self._last_written_progress_line = clean_value
        for handle in self._handles:
            if not write_value:
                continue
            try:
                handle.write(write_value)
                handle.flush()
            except OSError as exc:
                self._record_warning(f"output_sink_error: {exc}")
                continue
        self._buffer += value
        while "\n" in self._buffer or "\r" in self._buffer:
            newline_pos = self._buffer.find("\n")
            carriage_pos = self._buffer.find("\r")
            positions = [pos for pos in (newline_pos, carriage_pos) if pos >= 0]
            split_pos = min(positions)
            line = self._buffer[:split_pos]
            self._buffer = self._buffer[split_pos + 1 :]
            clean_line = line.strip()
            if self._heartbeat is not None and clean_line and clean_line != self._last_heartbeat_line:
                try:
                    self._heartbeat(clean_line)
                    self._last_heartbeat_line = clean_line
                except Exception:
                    pass
        return len(value)

    def flush(self) -> None:
        for handle in self._handles:
            try:
                handle.flush()
            except OSError as exc:
                self._record_warning(f"output_flush_error: {exc}")
                continue


def _append_limited_warning(runner_warnings: list[str] | None, message: str) -> None:
    if runner_warnings is None:
        return
    clean = str(message or "").strip()
    if not clean:
        return
    if len(runner_warnings) >= _RUNNER_WARNING_LIMIT:
        return
    if clean not in runner_warnings:
        runner_warnings.append(clean)


def _safe_update_job_metadata(job_paths: Any, runner_warnings: list[str] | None = None, **patch: Any) -> dict[str, Any]:
    try:
        return update_job_metadata(job_paths, **patch)
    except Exception as exc:
        _append_limited_warning(runner_warnings, f"metadata_update_error: {exc}")
        return dict(patch)


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    try:
        if process.poll() is not None:
            return
    except Exception:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(int(process.pid)), "/T", "/F"],
                capture_output=True,
                text=True,
                check=False,
            )
            return
        except Exception:
            pass
    try:
        process.terminate()
        process.wait(timeout=5)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def _start_new_process_group_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"preexec_fn": os.setsid}


def _python_unbuffered_command(command: list[str]) -> list[str]:
    clean_command = [str(item) for item in command]
    if not clean_command:
        return clean_command
    executable = Path(clean_command[0]).name.lower()
    if "python" not in executable:
        return clean_command
    if any(item in {"-u", "-I"} for item in clean_command[1:3]):
        return clean_command
    return [clean_command[0], "-u", *clean_command[1:]]


def _runner_environment() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run_command_in_subprocess(
    *,
    command: list[str],
    stdout_handle: TextIO,
    stderr_handle: TextIO,
    console_stdout: TextIO | None,
    console_stderr: TextIO | None,
    job_paths: Any | None = None,
    runner_warnings: list[str] | None = None,
    timeout_seconds: int | float | None = None,
) -> int:
    if len(command) < 2:
        raise ValueError("command must contain python executable and script path")
    clean_command = _python_unbuffered_command(command)
    timeout_value = float(timeout_seconds or 0)
    stdout_tee = _TeeTextIO(
        stdout_handle,
        console_stdout,
        heartbeat=(lambda line: mark_job_heartbeat(job_paths, stream_name="stdout", line=line)) if job_paths is not None else None,
        runner_warnings=runner_warnings,
    )
    stderr_tee = _TeeTextIO(
        stderr_handle,
        console_stderr,
        heartbeat=(lambda line: mark_job_heartbeat(job_paths, stream_name="stderr", line=line)) if job_paths is not None else None,
        runner_warnings=runner_warnings,
    )
    env = _runner_environment()
    if job_paths is not None:
        env["EXECUTION_APP_JOB_ID"] = str(getattr(job_paths, "job_id", "") or "")
        env["EXECUTION_APP_TASK_NAME"] = str(load_job_metadata(str(getattr(job_paths, "job_id", "") or "")).get("task_name", "") or "")
    process = subprocess.Popen(
        clean_command,
        cwd=str(WORKSPACE_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        **_start_new_process_group_kwargs(),
    )
    if job_paths is not None:
        _patch_runtime_worker_payload(job_id=str(getattr(job_paths, "job_id", "")), worker_pid=int(process.pid))
        _safe_update_job_metadata(
            job_paths,
            runner_warnings=runner_warnings,
            runner_mode="subprocess",
            server_pid=int(os.getpid()),
            worker_pid=int(process.pid),
            worker_started_at=now_iso(),
            runner_timeout_seconds=int(timeout_value) if timeout_value > 0 else 0,
        )
    deadline = time.monotonic() + timeout_value if timeout_value > 0 else None

    def pump_stream(stream: Any, tee: _TeeTextIO) -> None:
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                tee.write(line)
            tee.flush()
        except Exception as exc:
            _append_limited_warning(runner_warnings, f"output_reader_error: {exc}")
        finally:
            try:
                stream.close()
            except Exception:
                pass

    stdout_thread = threading.Thread(target=pump_stream, args=(process.stdout, stdout_tee), name=f"execution-runner-stdout-{process.pid}", daemon=True)
    stderr_thread = threading.Thread(target=pump_stream, args=(process.stderr, stderr_tee), name=f"execution-runner-stderr-{process.pid}", daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    timed_out = False
    while True:
        return_code = process.poll()
        if return_code is not None:
            break
        if deadline is not None and time.monotonic() >= deadline:
            timed_out = True
            _append_limited_warning(runner_warnings, f"runner_timeout: exceeded {int(timeout_value)} seconds")
            if job_paths is not None:
                _safe_update_job_metadata(
                    job_paths,
                    runner_warnings=runner_warnings,
                    runner_mode="subprocess",
                    server_pid=int(os.getpid()),
                    worker_pid=int(process.pid),
                    runner_timeout_seconds=int(timeout_value),
                    runner_timed_out=True,
                    runner_timeout_at=now_iso(),
                )
            _terminate_process_tree(process)
            break
        time.sleep(0.2)
    try:
        process.wait(timeout=5)
    except Exception:
        _terminate_process_tree(process)
    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)
    stdout_tee.flush()
    stderr_tee.flush()
    if timed_out:
        return RUNNER_TIMEOUT_EXIT_CODE
    return int(process.returncode or 0)


def run_task_sync(
    *,
    task_name: str,
    python_executable: str = "",
    passthrough_args: list[str] | None = None,
    job_label: str = "",
    force_unlock: bool = False,
    echo_output: bool = True,
) -> dict[str, Any]:
    freeze_guard.assert_task_allowed(task_name)
    resolved_python = resolve_project_python_executable(python_executable or sys.executable)
    clean_passthrough = sanitize_passthrough_args(passthrough_args)
    command = build_task_command(task_name=task_name, python_executable=resolved_python, passthrough_args=clean_passthrough)
    job_paths = create_job_record(
        task_name=task_name,
        command=command,
        cwd=WORKSPACE_ROOT,
        python_executable=resolved_python,
        passthrough_args=clean_passthrough,
        job_label=job_label,
    )
    return _run_existing_job(
        job_paths=job_paths,
        task_name=task_name,
        command=command,
        python_executable=resolved_python,
        passthrough_args=clean_passthrough,
        job_label=job_label,
        force_unlock=force_unlock,
        echo_output=echo_output,
    )


def run_recorded_job_sync(
    *,
    job_id: str,
    force_unlock: bool = False,
    echo_output: bool = False,
) -> dict[str, Any]:
    metadata = load_job_metadata(str(job_id))
    if not metadata:
        raise FileNotFoundError(f"未找到作业元数据：{job_id}")
    command = [str(item) for item in metadata.get("command_argv", []) if str(item).strip()]
    if not command:
        raise ValueError(f"作业 {job_id} 不包含 command_argv。")
    task_name = str(metadata.get("task_name", "") or "")
    python_executable = resolve_project_python_executable(str(metadata.get("python_executable", "") or command[0]))
    command[0] = python_executable
    passthrough_args = sanitize_passthrough_args([str(item) for item in metadata.get("passthrough_args", []) if str(item).strip()])
    return _run_existing_job(
        job_paths=build_job_paths(str(job_id)),
        task_name=task_name,
        command=command,
        python_executable=python_executable,
        passthrough_args=passthrough_args,
        job_label=str(metadata.get("job_label", "") or ""),
        resumed_from_job_id=str(metadata.get("resumed_from_job_id", "") or ""),
        force_unlock=force_unlock,
        echo_output=echo_output,
    )


def _spawn_detached_job_worker(
    *,
    job_id: str,
    python_executable: str,
    force_unlock: bool = False,
) -> dict[str, Any]:
    clean_job_id = str(job_id or "").strip()
    if not clean_job_id:
        raise ValueError("job_id is required")
    job_paths = build_job_paths(clean_job_id)
    worker_stdout_log = job_paths.job_root / "worker_stdout.log"
    worker_stderr_log = job_paths.job_root / "worker_stderr.log"
    command = [
        str(python_executable),
        "-m",
        "daily_research.execution.job_worker",
        "--job-id",
        clean_job_id,
    ]
    if force_unlock:
        command.append("--force-unlock")
    stdout_handle = worker_stdout_log.open("a", encoding="utf-8")
    stderr_handle = worker_stderr_log.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=str(WORKSPACE_ROOT),
            stdout=stdout_handle,
            stderr=stderr_handle,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            env=_runner_environment(),
            **_start_new_process_group_kwargs(),
        )
    finally:
        stdout_handle.close()
        stderr_handle.close()
    payload = {
        "async_runner_mode": "detached_worker",
        "worker_launcher_pid": int(process.pid),
        "worker_command_argv": command,
        "worker_stdout_log": str(worker_stdout_log.resolve()),
        "worker_stderr_log": str(worker_stderr_log.resolve()),
        "worker_launch_requested_at": now_iso(),
    }
    update_job_metadata(job_paths, **payload)
    return payload


def _mark_job_launch_pending(job_paths: Any, *, task_name: str) -> None:
    pending_at = now_iso()
    payload = {
        "job_id": str(getattr(job_paths, "job_id", "") or ""),
        "task_name": str(task_name or ""),
        "pid": 0,
        "acquired_at": pending_at,
        "status": "launch_pending",
    }
    write_json_file(LOCK_PATH, payload)
    state = _read_json(STATE_PATH) or load_runtime_state()
    state["lock"] = dict(payload)
    state["current_job"] = {
        "job_id": str(getattr(job_paths, "job_id", "") or ""),
        "task_name": str(task_name or ""),
        "status": "launch_pending",
        "started_at": pending_at,
    }
    write_json_file(STATE_PATH, state)


def launch_task_async(
    *,
    task_name: str,
    python_executable: str = "",
    passthrough_args: list[str] | None = None,
    job_label: str = "",
    force_unlock: bool = False,
) -> dict[str, Any]:
    freeze_guard.assert_task_allowed(task_name)
    if not force_unlock and _has_active_runtime_job():
        raise ExecutionAppLockError("Execution app lock is already held by an active job.")
    resolved_python = resolve_project_python_executable(python_executable or sys.executable)
    clean_passthrough = sanitize_passthrough_args(passthrough_args)
    command = build_task_command(task_name=task_name, python_executable=resolved_python, passthrough_args=clean_passthrough)
    job_paths = create_job_record(
        task_name=task_name,
        command=command,
        cwd=WORKSPACE_ROOT,
        python_executable=resolved_python,
        passthrough_args=clean_passthrough,
        job_label=job_label,
    )
    _mark_job_launch_pending(job_paths, task_name=task_name)
    worker_payload = _spawn_detached_job_worker(
        job_id=job_paths.job_id,
        python_executable=resolved_python,
        force_unlock=force_unlock,
    )
    append_event("job_launch_requested", job_id=job_paths.job_id, task_name=task_name)
    return {
        "job_id": job_paths.job_id,
        "task_name": task_name,
        "status": "queued",
        "python_executable": resolved_python,
        "command": command,
        "stdout_log": str(job_paths.stdout_path.resolve()),
        "stderr_log": str(job_paths.stderr_path.resolve()),
        **worker_payload,
    }


def resume_task_sync(
    *,
    job_id: str = "",
    job_label: str = "",
    force_unlock: bool = False,
    echo_output: bool = True,
) -> dict[str, Any]:
    metadata = resolve_resume_metadata(job_id)
    command = [str(item) for item in metadata.get("command_argv", []) if str(item).strip()]
    if not command:
        raise ValueError(f"作业 {metadata.get('job_id', '')} 不包含 command_argv。")
    task_name = str(metadata.get("task_name", "") or "resumed_job")
    passthrough_args = strip_execution_timeout_args([str(item) for item in metadata.get("passthrough_args", []) if str(item).strip()])
    python_executable = resolve_project_python_executable(str(metadata.get("python_executable", "") or command[0]))
    command[0] = python_executable
    command = [command[0], command[1], *strip_execution_timeout_args(command[2:])] if len(command) > 2 else command
    effective_job_label = str(job_label or f"resume:{metadata.get('job_id', '')}")
    job_paths = create_job_record(
        task_name=task_name,
        command=command,
        cwd=WORKSPACE_ROOT,
        python_executable=python_executable,
        passthrough_args=passthrough_args,
        job_label=effective_job_label,
        resumed_from_job_id=str(metadata.get("job_id", "")),
    )
    return _run_existing_job(
        job_paths=job_paths,
        task_name=task_name,
        command=command,
        python_executable=python_executable,
        passthrough_args=passthrough_args,
        job_label=effective_job_label,
        resumed_from_job_id=str(metadata.get("job_id", "")),
        force_unlock=force_unlock,
        echo_output=echo_output,
    )


def resume_task_async(
    *,
    job_id: str = "",
    job_label: str = "",
    force_unlock: bool = False,
) -> dict[str, Any]:
    if not force_unlock and _has_active_runtime_job():
        raise ExecutionAppLockError("Execution app lock is already held by an active job.")
    metadata = resolve_resume_metadata(job_id)
    command = [str(item) for item in metadata.get("command_argv", []) if str(item).strip()]
    if not command:
        raise ValueError(f"作业 {metadata.get('job_id', '')} 不包含 command_argv。")
    task_name = str(metadata.get("task_name", "") or "resumed_job")
    passthrough_args = strip_execution_timeout_args([str(item) for item in metadata.get("passthrough_args", []) if str(item).strip()])
    python_executable = resolve_project_python_executable(str(metadata.get("python_executable", "") or command[0]))
    command[0] = python_executable
    command = [command[0], command[1], *strip_execution_timeout_args(command[2:])] if len(command) > 2 else command
    effective_job_label = str(job_label or f"resume:{metadata.get('job_id', '')}")
    job_paths = create_job_record(
        task_name=task_name,
        command=command,
        cwd=WORKSPACE_ROOT,
        python_executable=python_executable,
        passthrough_args=passthrough_args,
        job_label=effective_job_label,
        resumed_from_job_id=str(metadata.get("job_id", "")),
    )
    _mark_job_launch_pending(job_paths, task_name=task_name)
    worker_payload = _spawn_detached_job_worker(
        job_id=job_paths.job_id,
        python_executable=python_executable,
        force_unlock=force_unlock,
    )
    append_event("job_resume_requested", job_id=job_paths.job_id, task_name=task_name, resumed_from_job_id=str(metadata.get("job_id", "")))
    return {
        "job_id": job_paths.job_id,
        "task_name": task_name,
        "status": "queued",
        "python_executable": python_executable,
        "command": command,
        "stdout_log": str(job_paths.stdout_path.resolve()),
        "stderr_log": str(job_paths.stderr_path.resolve()),
        "resumed_from_job_id": str(metadata.get("job_id", "")),
        **worker_payload,
    }


def unlock_runtime(*, force: bool = False) -> dict[str, Any]:
    if not LOCK_PATH.exists():
        return {"status": "ok", "detail": "execution app 锁当前已清空"}
    if not force:
        raise ExecutionAppLockError("锁仍然存在。只有确认该作业已经消失后，才允许使用 force 重试。")
    clear_lock_file()
    return {"status": "ok", "detail": "execution app 锁已清理"}


def parse_raw_args_text(raw_args_text: str) -> list[str]:
    text = str(raw_args_text or "").strip()
    if not text:
        return []
    return [str(item) for item in shlex.split(text, posix=False)]


def strip_execution_timeout_args(values: list[str] | tuple[str, ...] | None) -> list[str]:
    items = [str(item) for item in (values or [])]
    stripped: list[str] = []
    skip_next = False
    for item in items:
        if skip_next:
            skip_next = False
            continue
        normalized = str(item or "").strip().lower()
        if normalized in {"--timeout", "--timeout-seconds", "--timeout_seconds"}:
            skip_next = True
            continue
        if normalized.startswith(("--timeout=", "--timeout-seconds=", "--timeout_seconds=")):
            continue
        stripped.append(item)
    return stripped


def build_passthrough_args_from_form(
    *,
    task_name: str,
    form_payload: dict[str, Any] | None = None,
    raw_args_text: str = "",
) -> list[str]:
    spec = get_task_spec(task_name)
    payload = form_payload if isinstance(form_payload, dict) else {}
    args: list[str] = []
    for field in spec.form_fields:
        raw_value = payload.get(field.name, field.default_value)
        if field.field_type == "boolean":
            if bool(raw_value):
                args.append(field.arg_flag)
            elif field.false_arg_flag:
                args.append(field.false_arg_flag)
            continue
        value = str(raw_value or "").strip()
        if not value:
            continue
        args.extend([field.arg_flag, value])
    args.extend(parse_raw_args_text(raw_args_text))
    return args


def list_active_thread_job_ids() -> list[str]:
    with _ACTIVE_JOB_THREADS_LOCK:
        return [job_id for job_id, thread in _ACTIVE_JOB_THREADS.items() if thread.is_alive()]


def task_field_sections(task_name: str) -> list[str]:
    sections: list[str] = []
    for field in get_task_spec(task_name).form_fields:
        if field.section not in sections:
            sections.append(field.section)
    return sections
