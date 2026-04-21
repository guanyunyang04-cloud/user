from __future__ import annotations

from pathlib import Path
from typing import Any

from daily_research.execution import app_service
from daily_research.execution.app_tasks import get_task_spec, serialize_task_spec


APP_TITLE = "Daily Research 执行控制台"
APP_SUBTITLE = "本地执行、监控、恢复与产物查看控制面板"
NAV_ITEMS = (
    {"path": "/", "label": "总览"},
    {"path": "/tasks", "label": "任务"},
    {"path": "/jobs", "label": "作业"},
    {"path": "/doctor", "label": "体检"},
    {"path": "/artifacts/trade-plan", "label": "交易计划"},
    {"path": "/continuous-policy", "label": "连续策略"},
    {"path": "/account", "label": "模拟账户"},
    {"path": "/guide", "label": "使用教程"},
    {"path": "/settings/runtime", "label": "运行时"},
)

STATUS_LABELS = {
    "ok": "正常",
    "queued": "排队中",
    "running": "运行中",
    "succeeded": "成功",
    "failed": "失败",
    "blocked": "阻塞",
    "degraded": "降级",
}

SECTION_LABELS = {
    "selection": "选择参数",
    "source": "来源参数",
    "window": "时间窗口",
    "mode": "运行模式",
    "manifest": "Manifest 参数",
    "runtime": "运行参数",
    "common": "通用参数",
    "launch": "启动参数",
}


def status_label(value: str) -> str:
    return STATUS_LABELS.get(str(value or "").strip().lower(), str(value or "未知"))


def section_label(value: str) -> str:
    return SECTION_LABELS.get(str(value or "").strip().lower(), str(value or "其他"))


def base_context(*, active_path: str) -> dict[str, Any]:
    paths = ui_paths()
    css_path = paths["static"] / "execution_console.css"
    js_path = paths["static"] / "execution_console.js"
    return {
        "app_title": APP_TITLE,
        "app_subtitle": APP_SUBTITLE,
        "nav_items": NAV_ITEMS,
        "active_path": active_path,
        "ui_asset_version": max(
            int(css_path.stat().st_mtime) if css_path.exists() else 0,
            int(js_path.stat().st_mtime) if js_path.exists() else 0,
        ),
    }


def dashboard_context() -> dict[str, Any]:
    payload = app_service.build_status_payload(history_limit=10)
    payload["doctor"] = app_service.build_doctor_payload()
    return payload


def tasks_context(*, selected_task: str = "") -> dict[str, Any]:
    tasks = app_service.list_tasks_payload(core_only=True)
    visible_names = {task["name"] for task in tasks}
    resolved = selected_task if selected_task in visible_names else (tasks[0]["name"] if tasks else "")
    selected = serialize_task_spec(get_task_spec(resolved)) if resolved else {}
    return {
        "tasks": tasks,
        "selected_task_name": resolved,
        "selected_task": selected,
    }


def jobs_context(*, limit: int = 30) -> dict[str, Any]:
    return {"jobs": app_service.list_jobs_payload(limit=limit)}


def job_detail_context(job_id: str, *, lines: int = 120) -> dict[str, Any]:
    return {"job": app_service.build_job_detail_payload(job_id, lines=lines)}


def doctor_context() -> dict[str, Any]:
    return {"doctor": app_service.build_doctor_payload()}


def trade_plan_context() -> dict[str, Any]:
    return {"artifact": app_service.latest_trade_plan_summary(max_lines=200)}


def continuous_policy_context() -> dict[str, Any]:
    return {"continuous_policy": app_service.continuous_policy_summary(action_rows_limit=16)}


def account_context() -> dict[str, Any]:
    return {"account": app_service.load_account_snapshot()}


def runtime_context() -> dict[str, Any]:
    return {"runtime": app_service.build_status_payload(history_limit=20)}


def guide_context() -> dict[str, Any]:
    return {
        "tutorial_markdown_path": str((Path(__file__).resolve().parent / "使用教程.md").resolve()),
    }


def ui_paths() -> dict[str, Path]:
    ui_root = Path(__file__).resolve().parent / "web"
    return {
        "ui_root": ui_root,
        "templates": ui_root / "templates",
        "static": ui_root / "static",
    }
