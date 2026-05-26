from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TaskFieldSpec:
    name: str
    label: str
    arg_flag: str
    false_arg_flag: str = ""
    field_type: str = "text"
    placeholder: str = ""
    help_text: str = ""
    default_value: str = ""
    default_checked: bool = False
    choices: tuple[str, ...] = ()
    section: str = "common"


@dataclass(frozen=True)
class ExecutionTaskSpec:
    name: str
    script_relative_path: str
    description: str
    category: str = "execution"
    default_args: tuple[str, ...] = ()
    form_fields: tuple[TaskFieldSpec, ...] = ()
    allow_raw_args: bool = True
    launcher_notes: tuple[str, ...] = ()
    safety_level: str = "safe"
    safety_summary: str = ""
    timeout_seconds: int = 0

    @property
    def script_path(self) -> Path:
        return (WORKSPACE_ROOT / self.script_relative_path).resolve()


COMMON_CANDIDATE_PROFILE_FIELD = TaskFieldSpec(
    name="candidate_profile",
    label="候选配置",
    arg_flag="--candidate-profile",
    placeholder="active_execution_strategy",
    help_text="留空则跟随当前默认执行候选。",
    section="selection",
)


TASK_SPECS: tuple[ExecutionTaskSpec, ...] = (
    ExecutionTaskSpec(
        name="execution-smoke",
        script_relative_path="daily_research/execution/run_execution_smoke.py",
        description="执行控制台端到端 smoke：验证作业提交、日志流、metadata 和锁释放；不碰 active/promotion。",
        category="diagnostic",
        timeout_seconds=60,
        safety_level="safe",
        safety_summary="只写 execution_app smoke 输出，用于验证执行端自身功能。",
    ),
    ExecutionTaskSpec(
        name="trade-plan",
        script_relative_path="daily_research/execution/run_trade_plan.py",
        description="生成默认次日交易计划。",
        timeout_seconds=1800,
        form_fields=(
            COMMON_CANDIDATE_PROFILE_FIELD,
            TaskFieldSpec(
                name="legacy_ml",
                label="旧版 ML 模式",
                arg_flag="--legacy-ml",
                field_type="boolean",
                help_text="仅用于兼容旧产物链路；默认不要开启。",
                section="mode",
            ),
        ),
        launcher_notes=("默认路径应保持跟随 active_execution_strategy。",),
    ),
    ExecutionTaskSpec(
        name="refresh-production-default",
        script_relative_path="daily_research/execution/update_default_candidate_production.py",
        description="刷新默认 production full-fit，并可选择更新 active manifest。",
        timeout_seconds=0,
        form_fields=(
            TaskFieldSpec(
                name="source_run_dir",
                label="来源运行目录",
                arg_flag="--source-run-dir",
                placeholder="留空则从当前 active manifest 自动解析",
                help_text="通常留空即可；只有需要指定 formal 来源 run 时才填写。",
                section="source",
            ),
            TaskFieldSpec(
                name="end_date",
                label="结束日期",
                arg_flag="--end-date",
                placeholder="20260421",
                help_text="可选的上线截止日期覆盖值，留空使用最新完成交易日。",
                section="window",
            ),
            TaskFieldSpec(
                name="strategy_name",
                label="策略名称",
                arg_flag="--strategy-name",
                placeholder="short_expert_policy_v5b_deployable_anchor_active",
                help_text="可选的 active manifest 策略名称覆盖值。",
                section="manifest",
            ),
        ),
        launcher_notes=(
            "该任务可能重建 production_root，并可能改写当前默认执行。",
            "默认执行当前应保持 10 交易日重训、每日刷新股票池、收敛后才 promotion。",
        ),
        safety_level="danger",
        safety_summary="会重建 production_root，并可能改写当前默认执行。",
    ),
    ExecutionTaskSpec(
        name="candidate-trade-plan",
        script_relative_path="daily_research/execution/run_research_candidate_trade_plan.py",
        description="基于候选配置生成研究候选交易计划。",
        timeout_seconds=1800,
        form_fields=(COMMON_CANDIDATE_PROFILE_FIELD,),
    ),
    ExecutionTaskSpec(
        name="candidate-backtest",
        script_relative_path="daily_research/execution/run_research_candidate_backtest.py",
        description="基于候选配置回放研究候选执行路径。",
        timeout_seconds=7200,
        form_fields=(COMMON_CANDIDATE_PROFILE_FIELD,),
    ),
    ExecutionTaskSpec(
        name="update-liquid-pool",
        script_relative_path="daily_research/execution/update_liquid_pool.py",
        description="刷新执行侧使用的 liquid300/500/800 股票池。",
        timeout_seconds=1800,
    ),
    ExecutionTaskSpec(
        name="data-platform-refresh",
        script_relative_path="daily_research/data_platform/refresh_daily.py",
        description="显式刷新 TDX-free data platform，注册 lake dataset，并把当前执行 manifest 接到最新 policy_input_bundle。",
        category="data_platform",
        timeout_seconds=7200,
        safety_level="caution",
        safety_summary="刷新研究数据湖；成功后只更新 active execution manifest 的 lake dataset 指针，不触发重训或 promotion。",
    ),
    ExecutionTaskSpec(
        name="provider-health-check",
        script_relative_path="daily_research/data_platform/provider_health.py",
        description="检查 data platform provider/domain 健康，并写入可流式展示的结构化进度。",
        category="data_platform",
        timeout_seconds=1800,
        safety_level="safe",
        safety_summary="只读 provider/domain 健康检查，不刷新数据湖、不改 active manifest。",
    ),
    ExecutionTaskSpec(
        name="refresh-production-live-panels",
        script_relative_path="daily_research/execution/refresh_production_live_panels.py",
        description="基于当前 active lake dataset 刷新 production live score/target panels；不训练、不 promotion。",
        category="data_platform",
        timeout_seconds=7200,
        safety_level="safe",
        safety_summary="只做 production signal inference/export，并校验 panels 覆盖最新完成交易日。",
    ),
    ExecutionTaskSpec(
        name="refresh-production-static-fallback",
        script_relative_path="daily_research/execution/refresh_production_static_fallback.py",
        description="刷新 production_root 的静态 fallback 产物与 manifest 元数据。",
        timeout_seconds=1800,
    ),
    ExecutionTaskSpec(
        name="global-strategy-leaderboard",
        script_relative_path="daily_research/execution/run_global_deployable_strategy_leaderboard.py",
        description="刷新可部署执行策略排行榜；默认只读。",
        timeout_seconds=7200,
        safety_level="caution",
        safety_summary="默认只读；只有显式传入危险开关后才会改写当前默认执行。",
    ),
    ExecutionTaskSpec(
        name="single-mapping-pipeline",
        script_relative_path="daily_research/execution/run_short_alpha_execution_single_mapping_candidate_pipeline.py",
        description="刷新 single-mapping 执行候选 pipeline 和 live monitor。",
        timeout_seconds=7200,
    ),
    ExecutionTaskSpec(
        name="activate-single-mapping",
        script_relative_path="daily_research/execution/activate_execution_single_mapping_candidate.py",
        description="把 single-mapping 候选提升为当前 active execution manifest。",
        timeout_seconds=1800,
        safety_level="danger",
        safety_summary="会直接改写当前 active execution manifest。",
    ),
    ExecutionTaskSpec(
        name="update-model",
        script_relative_path="daily_research/execution/update_model.py",
        description="旧版 ML 模型刷新入口。",
        category="legacy",
        timeout_seconds=7200,
    ),
    ExecutionTaskSpec(
        name="update-model-legacy-ml",
        script_relative_path="daily_research/execution/update_model_legacy_ml.py",
        description="旧版 ML 模型刷新兼容入口。",
        category="legacy",
        timeout_seconds=7200,
    ),
    ExecutionTaskSpec(
        name="trade-plan-legacy-ml",
        script_relative_path="daily_research/execution/run_trade_plan_legacy_ml.py",
        description="旧版 ML 交易计划兼容入口。",
        category="legacy",
        timeout_seconds=1800,
    ),
)


TASK_REGISTRY: dict[str, ExecutionTaskSpec] = {spec.name: spec for spec in TASK_SPECS}

CORE_FRONTEND_TASK_NAMES: tuple[str, ...] = (
    "trade-plan",
)


def get_task_spec(task_name: str) -> ExecutionTaskSpec:
    resolved = TASK_REGISTRY.get(str(task_name or "").strip())
    if resolved is None:
        available = ", ".join(sorted(TASK_REGISTRY))
        raise KeyError(f"未知执行任务：{task_name}。可用任务：{available}")
    return resolved


def list_task_specs() -> list[ExecutionTaskSpec]:
    return list(TASK_SPECS)


def list_core_frontend_task_specs() -> list[ExecutionTaskSpec]:
    return [TASK_REGISTRY[name] for name in CORE_FRONTEND_TASK_NAMES if name in TASK_REGISTRY]


def list_task_lines() -> list[str]:
    lines = ["执行应用任务列表："]
    for spec in TASK_SPECS:
        lines.append(f"- {spec.name}: {spec.description} [{spec.category}]")
    return lines


def build_task_command(
    *,
    task_name: str,
    python_executable: str,
    passthrough_args: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    spec = get_task_spec(task_name)
    command = [str(python_executable), str(spec.script_path), *spec.default_args]
    if passthrough_args:
        command.extend(str(item) for item in passthrough_args)
    return command


def serialize_task_field(field: TaskFieldSpec) -> dict[str, Any]:
    return {
        "name": field.name,
        "label": field.label,
        "arg_flag": field.arg_flag,
        "false_arg_flag": field.false_arg_flag,
        "field_type": field.field_type,
        "placeholder": field.placeholder,
        "help_text": field.help_text,
        "default_value": field.default_value,
        "default_checked": field.default_checked,
        "choices": list(field.choices),
        "section": field.section,
    }


def serialize_task_spec(spec: ExecutionTaskSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "description": spec.description,
        "category": spec.category,
        "script_path": str(spec.script_path),
        "default_args": list(spec.default_args),
        "form_fields": [serialize_task_field(field) for field in spec.form_fields],
        "allow_raw_args": bool(spec.allow_raw_args),
        "launcher_notes": list(spec.launcher_notes),
        "safety_level": spec.safety_level,
        "safety_summary": spec.safety_summary,
        "timeout_seconds": int(spec.timeout_seconds or 0),
    }
