from __future__ import annotations

import csv
import importlib.util
import json
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from daily_research.baseline.data_provider import get_latest_completed_trading_date
from daily_research.deep_alpha.experiment_guardrails import resolve_project_python_executable
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
    clear_lock_file,
    create_job_record,
    ensure_runtime_layout,
    list_recent_job_metadata,
    load_job_metadata,
    load_runtime_state,
    mark_job_finished,
    mark_job_heartbeat,
    mark_job_started,
    read_json_file,
    tail_file,
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
ENVIRONMENT_PATH = PROJECT_ROOT / "environment.yml"
LATEST_TRADE_PLAN_PATH = EXECUTION_DIR / "output" / "latest_trade_plan.txt"
ACCOUNT_FILE_HEADERS = ("record_type", "stock", "shares", "cost_price", "available_cash")
FORMAL_DATA_PLATFORM_DOMAINS: tuple[str, ...] = (
    "market_daily",
    "trading_calendar",
    "universe_snapshot",
    "security_status",
    "limit_status",
    "industry_concept",
    "valuation",
)
_ACTIVE_JOB_THREADS: dict[str, threading.Thread] = {}
_ACTIVE_JOB_THREADS_LOCK = threading.Lock()


def sanitize_passthrough_args(values: list[str] | None) -> list[str]:
    items = [str(item) for item in (values or [])]
    if items[:1] == ["--"]:
        items = items[1:]
    return items


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


def load_account_snapshot(path: Path = POSITIONS_PATH) -> dict[str, Any]:
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


def positions_summary(path: Path = POSITIONS_PATH) -> dict[str, Any]:
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
            "execution_policy_label": "",
            "effective_live_target_weight_mode": "",
            "effective_live_execution_profile": "",
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
        "execution_policy_label": str(payload.get("execution_policy_label", "") or payload.get("execution_alignment_profile", "") or ""),
        "effective_live_target_weight_mode": str(payload.get("effective_live_target_weight_mode", "") or ""),
        "effective_live_execution_profile": str(payload.get("effective_live_execution_profile", "") or ""),
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


def latest_trade_plan_summary(
    *,
    max_lines: int = 80,
    output_dir: Path | str | None = None,
    latest_txt_path: Path | str | None = None,
) -> dict[str, Any]:
    resolved_output_dir = Path(output_dir) if output_dir is not None else EXECUTION_DIR / "output"
    resolved_latest_txt = Path(latest_txt_path) if latest_txt_path is not None else resolved_output_dir / "latest_trade_plan.txt"
    run_dir = _latest_trade_plan_run_dir(resolved_output_dir)
    summary_path = run_dir / "plan_summary.json" if run_dir is not None else None
    actions_path = run_dir / "actions_today.csv" if run_dir is not None else None
    holdings_path = run_dir / "holdings_snapshot.csv" if run_dir is not None else None
    watchlist_path = run_dir / "watchlist.csv" if run_dir is not None else None
    txt_path = resolved_latest_txt if resolved_latest_txt.exists() else (run_dir / "daily_trade_plan.txt" if run_dir is not None else resolved_latest_txt)
    lines = tail_file(txt_path, lines=max_lines) if txt_path.exists() else []
    summary = _read_json(summary_path) if summary_path is not None and summary_path.exists() else {}
    model_info = summary.get("model_freshness") if isinstance(summary.get("model_freshness"), dict) else {}
    return {
        "path": str(txt_path.resolve()),
        "exists": txt_path.exists(),
        "status": "ok" if txt_path.exists() or run_dir is not None else "missing",
        "summary": summary,
        "actions": _read_csv_preview(actions_path, limit=200) if actions_path and actions_path.exists() else [],
        "holdings": _read_csv_preview(holdings_path, limit=300) if holdings_path and holdings_path.exists() else [],
        "watchlist": _read_csv_preview(watchlist_path, limit=200) if watchlist_path and watchlist_path.exists() else [],
        "model_info": model_info,
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
    try:
        latest_completed = str(get_latest_completed_trading_date())
    except Exception:
        latest_completed = ""
    recommended_domains = list(FORMAL_DATA_PLATFORM_DOMAINS)
    return {
        "status": "ok",
        "lake_root": str((PROJECT_ROOT / "output" / "research_data_lake").resolve()),
        "catalog_status": catalog_status,
        "datasets": datasets,
        "data_platform": {
            "runs_root": str(platform_runs.resolve()),
            "latest_refresh_run": latest_refresh,
            "provider_plan": "default_free",
            "latest_completed_trading_date": latest_completed,
            "recommended_domains": recommended_domains,
            "default_refresh": {
                "as_of_date": latest_completed,
                "universe": "all_a",
                "domains": recommended_domains,
                "provider_plan": "default_free",
            },
        },
    }


def save_account_snapshot(
    *,
    available_cash: float | int | str | None,
    positions: list[dict[str, Any]] | None,
    path: Path = POSITIONS_PATH,
) -> dict[str, Any]:
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
    _write_csv_atomic(path, rows=rows)
    append_event("account_snapshot_saved", path=str(path.resolve()), position_count=len(normalized_positions), available_cash=cash_value)
    return load_account_snapshot(path)


def reset_account_snapshot_from_example(
    *,
    target_path: Path = POSITIONS_PATH,
    example_path: Path = POSITIONS_EXAMPLE_PATH,
) -> dict[str, Any]:
    if not example_path.exists():
        raise FileNotFoundError(f"未找到示例账户文件：{example_path}")
    example_snapshot = load_account_snapshot(example_path)
    restored = save_account_snapshot(
        available_cash=example_snapshot["available_cash"],
        positions=example_snapshot["positions"],
        path=target_path,
    )
    append_event("account_snapshot_reset_from_example", target_path=str(target_path.resolve()), example_path=str(example_path.resolve()))
    return restored


def build_status_payload(*, history_limit: int = 8) -> dict[str, Any]:
    ensure_runtime_layout()
    state = load_runtime_state()
    lock_payload = _read_json(LOCK_PATH)
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
        "latest_trade_plan": latest_trade_plan_summary(max_lines=24),
        "warnings": warnings,
        "updated_at": state.get("updated_at", ""),
    }


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
    add_check("runtime_lock", not bool(lock_payload), "未锁定" if not lock_payload else json.dumps(lock_payload, ensure_ascii=False))
    add_check("web_dependency_fastapi", importlib.util.find_spec("fastapi") is not None, "fastapi")
    add_check("web_dependency_uvicorn", importlib.util.find_spec("uvicorn") is not None, "uvicorn")
    add_check("web_dependency_jinja2", importlib.util.find_spec("jinja2") is not None, "jinja2")
    add_check("latest_trade_plan_artifact", LATEST_TRADE_PLAN_PATH.exists(), str(LATEST_TRADE_PLAN_PATH.resolve()))
    overall_ok = all(bool(item["ok"]) for item in checks)
    return {"status": "ok" if overall_ok else "degraded", "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": checks}


def list_tasks_payload(*, core_only: bool = False) -> list[dict[str, Any]]:
    specs = list_core_frontend_task_specs() if core_only else list_task_specs()
    return [serialize_task_spec(spec) for spec in specs]


def list_jobs_payload(*, limit: int = 20) -> list[dict[str, Any]]:
    return list_recent_job_metadata(limit=limit)


def build_job_detail_payload(job_id: str, *, lines: int = 80) -> dict[str, Any]:
    metadata = load_job_metadata(str(job_id))
    if not metadata:
        raise FileNotFoundError(f"未找到作业元数据：{job_id}")
    stdout_lines = tail_file(Path(str(metadata.get("stdout_log", ""))), lines=lines)
    stderr_lines = tail_file(Path(str(metadata.get("stderr_log", ""))), lines=lines)
    status = str(metadata.get("status", "") or "")
    return {
        "job_id": str(metadata.get("job_id", "")),
        "task_name": str(metadata.get("task_name", "")),
        "status": status,
        "metadata": metadata,
        "stdout_tail": stdout_lines,
        "stderr_tail": stderr_lines,
        "can_resume": status in {"failed", "blocked", "succeeded"},
    }


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


def _pump_stream(
    *,
    process_stream,
    target_handle,
    console_stream: TextIO | None,
    stream_name: str,
    job_paths,
    heartbeat_interval_seconds: float = 2.0,
) -> None:
    last_heartbeat = 0.0
    try:
        for raw_line in iter(process_stream.readline, ""):
            target_handle.write(raw_line)
            target_handle.flush()
            if console_stream is not None:
                console_stream.write(raw_line)
                console_stream.flush()
            now = time.monotonic()
            if now - last_heartbeat >= heartbeat_interval_seconds:
                mark_job_heartbeat(job_paths, stream_name=stream_name, line=raw_line)
                last_heartbeat = now
    finally:
        process_stream.close()


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
    try:
        with ExecutionAppLock(job_id=job_paths.job_id, task_name=task_name, force=force_unlock) as lock:
            mark_job_started(job_paths, lock_payload=lock.payload)
            with job_paths.stdout_path.open("w", encoding="utf-8") as stdout_handle:
                with job_paths.stderr_path.open("w", encoding="utf-8") as stderr_handle:
                    process = subprocess.Popen(
                        command,
                        cwd=str(WORKSPACE_ROOT),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        bufsize=1,
                    )
                    threads = [
                        threading.Thread(
                            target=_pump_stream,
                            kwargs={
                                "process_stream": process.stdout,
                                "target_handle": stdout_handle,
                                "console_stream": sys.stdout if echo_output else None,
                                "stream_name": "stdout",
                                "job_paths": job_paths,
                            },
                            daemon=True,
                        ),
                        threading.Thread(
                            target=_pump_stream,
                            kwargs={
                                "process_stream": process.stderr,
                                "target_handle": stderr_handle,
                                "console_stream": sys.stderr if echo_output else None,
                                "stream_name": "stderr",
                                "job_paths": job_paths,
                            },
                            daemon=True,
                        ),
                    ]
                    for thread in threads:
                        thread.start()
                    exit_code = int(process.wait())
                    for thread in threads:
                        thread.join()
            status = "succeeded" if exit_code == 0 else "failed"
            summary_note = spec.description
    except ExecutionAppLockError as exc:
        summary_note = str(exc)
        exit_code = 2
        status = "blocked"
    except Exception as exc:
        summary_note = str(exc)
        exit_code = 1
        status = "failed"
    finally:
        metadata = mark_job_finished(job_paths, status=status, exit_code=exit_code, summary_note=summary_note)
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


def run_task_sync(
    *,
    task_name: str,
    python_executable: str = "",
    passthrough_args: list[str] | None = None,
    job_label: str = "",
    force_unlock: bool = False,
    echo_output: bool = True,
) -> dict[str, Any]:
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


def launch_task_async(
    *,
    task_name: str,
    python_executable: str = "",
    passthrough_args: list[str] | None = None,
    job_label: str = "",
    force_unlock: bool = False,
) -> dict[str, Any]:
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
    thread = threading.Thread(
        target=_run_existing_job,
        kwargs={
            "job_paths": job_paths,
            "task_name": task_name,
            "command": command,
            "python_executable": resolved_python,
            "passthrough_args": clean_passthrough,
            "job_label": job_label,
            "force_unlock": force_unlock,
            "echo_output": False,
        },
        name=f"execution-app-{job_paths.job_id}",
        daemon=True,
    )
    with _ACTIVE_JOB_THREADS_LOCK:
        _ACTIVE_JOB_THREADS[job_paths.job_id] = thread
    thread.start()
    append_event("job_launch_requested", job_id=job_paths.job_id, task_name=task_name)
    return {
        "job_id": job_paths.job_id,
        "task_name": task_name,
        "status": "queued",
        "python_executable": resolved_python,
        "command": command,
        "stdout_log": str(job_paths.stdout_path.resolve()),
        "stderr_log": str(job_paths.stderr_path.resolve()),
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
    passthrough_args = [str(item) for item in metadata.get("passthrough_args", []) if str(item).strip()]
    python_executable = resolve_project_python_executable(str(metadata.get("python_executable", "") or command[0]))
    command[0] = python_executable
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
    metadata = resolve_resume_metadata(job_id)
    command = [str(item) for item in metadata.get("command_argv", []) if str(item).strip()]
    if not command:
        raise ValueError(f"作业 {metadata.get('job_id', '')} 不包含 command_argv。")
    task_name = str(metadata.get("task_name", "") or "resumed_job")
    passthrough_args = [str(item) for item in metadata.get("passthrough_args", []) if str(item).strip()]
    python_executable = resolve_project_python_executable(str(metadata.get("python_executable", "") or command[0]))
    command[0] = python_executable
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
    thread = threading.Thread(
        target=_run_existing_job,
        kwargs={
            "job_paths": job_paths,
            "task_name": task_name,
            "command": command,
            "python_executable": python_executable,
            "passthrough_args": passthrough_args,
            "job_label": effective_job_label,
            "resumed_from_job_id": str(metadata.get("job_id", "")),
            "force_unlock": force_unlock,
            "echo_output": False,
        },
        name=f"execution-app-{job_paths.job_id}",
        daemon=True,
    )
    with _ACTIVE_JOB_THREADS_LOCK:
        _ACTIVE_JOB_THREADS[job_paths.job_id] = thread
    thread.start()
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
