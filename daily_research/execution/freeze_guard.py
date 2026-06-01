from __future__ import annotations

EXECUTION_FREEZE_MODE = "frozen_skeleton_only"
EXECUTION_FREEZE_REASON = (
    "daily_research execution is frozen pending research-side rebuild and same-protocol model comparison"
)

ALLOWED_FROZEN_TASKS = {
    "execution-smoke",
    "provider-health-check",
    "candidate-backtest",
    "candidate-trade-plan",
}

FROZEN_WRITE_ACTIONS = {
    "active_manifest_write",
    "data_platform_refresh",
    "paper_account_apply_latest_plan",
    "paper_account_cash_flow",
    "paper_account_manual_adjustment",
    "production_full_fit",
    "save_account_snapshot",
    "trade_plan_generate",
}


class ExecutionFreezeError(RuntimeError):
    def __init__(self, action: str, detail: str = "") -> None:
        self.action = str(action or "").strip()
        self.detail = str(detail or "").strip()
        super().__init__(self.__str__())

    def __str__(self) -> str:
        suffix = f" {self.detail}" if self.detail else ""
        return f"{EXECUTION_FREEZE_MODE}: {self.action or 'unknown_action'} is disabled. {EXECUTION_FREEZE_REASON}.{suffix}"


def is_execution_frozen() -> bool:
    return True


def assert_task_allowed(task_name: str) -> None:
    normalized = str(task_name or "").strip()
    if is_execution_frozen() and normalized not in ALLOWED_FROZEN_TASKS:
        allowed = ", ".join(sorted(ALLOWED_FROZEN_TASKS))
        raise ExecutionFreezeError(normalized or "unknown_task", f"Allowed frozen tasks: {allowed}.")


def assert_write_action_allowed(action: str) -> None:
    normalized = str(action or "").strip()
    if is_execution_frozen() and normalized in FROZEN_WRITE_ACTIONS:
        raise ExecutionFreezeError(normalized)


def status_payload() -> dict[str, object]:
    return {
        "mode": EXECUTION_FREEZE_MODE,
        "frozen": is_execution_frozen(),
        "reason": EXECUTION_FREEZE_REASON,
        "allowed_tasks": sorted(ALLOWED_FROZEN_TASKS),
        "frozen_write_actions": sorted(FROZEN_WRITE_ACTIONS),
    }
