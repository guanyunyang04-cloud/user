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
    field_type: str = "text"
    placeholder: str = ""
    help_text: str = ""
    default_value: str = ""
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
        name="trade-plan",
        script_relative_path="daily_research/execution/run_trade_plan.py",
        description="生成默认次日交易计划。",
        form_fields=(
            COMMON_CANDIDATE_PROFILE_FIELD,
            TaskFieldSpec(
                name="legacy_ml",
                label="旧版 ML 模式",
                arg_flag="--legacy-ml",
                field_type="boolean",
                help_text="改用旧版模型产物流，而不是当前执行候选链路。",
                section="mode",
            ),
        ),
        launcher_notes=("默认操作路径应保持在 active_execution_strategy。",),
    ),
    ExecutionTaskSpec(
        name="candidate-trade-plan",
        script_relative_path="daily_research/execution/run_research_candidate_trade_plan.py",
        description="基于候选配置生成研究候选交易计划。",
        form_fields=(COMMON_CANDIDATE_PROFILE_FIELD,),
    ),
    ExecutionTaskSpec(
        name="candidate-backtest",
        script_relative_path="daily_research/execution/run_research_candidate_backtest.py",
        description="基于候选配置回放研究候选执行路径。",
        form_fields=(COMMON_CANDIDATE_PROFILE_FIELD,),
    ),
    ExecutionTaskSpec(
        name="update-liquid-pool",
        script_relative_path="daily_research/execution/update_liquid_pool.py",
        description="刷新执行侧使用的 liquid300/500/800 股票池。",
        form_fields=(
            TaskFieldSpec(
                name="pool_sizes",
                label="股票池规模",
                arg_flag="--pool-sizes",
                default_value="300,500,800",
                help_text="用逗号分隔的流动性股票池规模。",
                section="window",
            ),
            TaskFieldSpec(
                name="start_date",
                label="起始日期",
                arg_flag="--start-date",
                default_value="20240101",
                placeholder="20240101",
                help_text="用于流动性排序的历史起始日期。",
                section="window",
            ),
            TaskFieldSpec(
                name="signal_date",
                label="信号日期",
                arg_flag="--signal-date",
                placeholder="20260413",
                help_text="可选的已完成交易日覆盖值。",
                section="window",
            ),
            TaskFieldSpec(
                name="refresh_cache",
                label="刷新缓存",
                arg_flag="--refresh-cache",
                field_type="boolean",
                help_text="在重建股票池前先刷新缓存的市场数据。",
                section="runtime",
            ),
        ),
    ),
    ExecutionTaskSpec(
        name="refresh-production-default",
        script_relative_path="daily_research/execution/update_default_candidate_production.py",
        description="从 formal winner 构建 production full-fit，并可选更新 active manifest。",
        form_fields=(
            TaskFieldSpec(
                name="source_run_dir",
                label="来源运行目录",
                arg_flag="--source-run-dir",
                placeholder=str(
                    (
                        WORKSPACE_ROOT
                        / "daily_research"
                        / "output"
                        / "short_alpha_formal_head2head_20260404_monthly_budgetnorm_r1"
                        / "runs"
                        / "state_liquidity_listwise_v1_20250318_20260331"
                    ).resolve()
                ),
                help_text="可选的显式 formal 来源运行目录。",
                section="source",
            ),
            TaskFieldSpec(
                name="end_date",
                label="结束日期",
                arg_flag="--end-date",
                placeholder="20260413",
                help_text="可选的启动截止日期覆盖值。",
                section="window",
            ),
            TaskFieldSpec(
                name="strategy_name",
                label="策略名称",
                arg_flag="--strategy-name",
                placeholder="deep_alpha_short_alpha_execalign_production_default",
                help_text="可选的策略 manifest 名称覆盖值。",
                section="manifest",
            ),
        ),
        launcher_notes=("此任务可能会更新 production_root 和当前 active execution manifest。",),
    ),
    ExecutionTaskSpec(
        name="refresh-production-static-fallback",
        script_relative_path="daily_research/execution/refresh_production_static_fallback.py",
        description="刷新 production_root 的静态 fallback 产物与 manifest 元数据。",
        form_fields=(
            TaskFieldSpec(
                name="production_root",
                label="Production 根目录",
                arg_flag="--production-root",
                placeholder=str(
                    (
                        WORKSPACE_ROOT
                        / "daily_research"
                        / "output"
                        / "deep_alpha_short_alpha_execalign_production_default"
                    ).resolve()
                ),
                help_text="当需要检查非默认根目录时，可用该参数覆盖 production root。",
                section="source",
            ),
        ),
    ),
    ExecutionTaskSpec(
        name="global-strategy-leaderboard",
        script_relative_path="daily_research/execution/run_global_deployable_strategy_leaderboard.py",
        description="刷新可部署执行策略排行榜。",
        form_fields=(
            TaskFieldSpec(
                name="activate_best",
                label="激活最佳结果",
                arg_flag="--activate-best",
                field_type="boolean",
                help_text="把最佳可部署策略写入 active manifest。",
                section="manifest",
            ),
        ),
    ),
    ExecutionTaskSpec(
        name="single-mapping-pipeline",
        script_relative_path="daily_research/execution/run_short_alpha_execution_single_mapping_candidate_pipeline.py",
        description="刷新 single-mapping 执行候选 pipeline 和 live monitor。",
        form_fields=(
            TaskFieldSpec(
                name="review_root",
                label="评审根目录",
                arg_flag="--review-root",
                placeholder="daily_research/output/short_alpha_targeted_weak_month_repair_review_...",
                help_text="由 targeted weak-month repair 工作流产出的评审根目录。",
                section="source",
            ),
            TaskFieldSpec(
                name="recent_audit_root",
                label="Recent 审计根目录",
                arg_flag="--recent-audit-root",
                placeholder="daily_research/output/short_alpha_recent_execution_policy_audit_...",
                help_text="用于构建 live monitor 的 recent 审计根目录。",
                section="source",
            ),
            TaskFieldSpec(
                name="static_profile",
                label="静态配置",
                arg_flag="--static-profile",
                default_value="regoff_k2_5d_ensemble_native_anchor",
                help_text="当候选当前不活跃时使用的 fallback 执行配置。",
                section="mode",
            ),
            TaskFieldSpec(
                name="live_only",
                label="仅更新 Live",
                arg_flag="--live-only",
                field_type="boolean",
                help_text="跳过重生成，只刷新 live monitor 输出。",
                section="mode",
            ),
        ),
    ),
    ExecutionTaskSpec(
        name="activate-single-mapping",
        script_relative_path="daily_research/execution/activate_execution_single_mapping_candidate.py",
        description="把 single-mapping 候选提升为当前 active execution manifest。",
        form_fields=(
            TaskFieldSpec(
                name="pipeline_root",
                label="Pipeline 根目录",
                arg_flag="--pipeline-root",
                placeholder="daily_research/output/short_alpha_execution_single_mapping_candidate_pipeline_...",
                help_text="由 single-mapping pipeline 任务生成的 pipeline 根目录。",
                section="source",
            ),
        ),
    ),
    ExecutionTaskSpec(
        name="update-model",
        script_relative_path="daily_research/execution/update_model.py",
        description="旧版 ML 模型刷新入口。",
        category="legacy",
    ),
    ExecutionTaskSpec(
        name="update-model-legacy-ml",
        script_relative_path="daily_research/execution/update_model_legacy_ml.py",
        description="旧版 ML 模型刷新兼容入口。",
        category="legacy",
    ),
    ExecutionTaskSpec(
        name="trade-plan-legacy-ml",
        script_relative_path="daily_research/execution/run_trade_plan_legacy_ml.py",
        description="旧版 ML 交易计划兼容入口。",
        category="legacy",
    ),
)


TASK_REGISTRY: dict[str, ExecutionTaskSpec] = {spec.name: spec for spec in TASK_SPECS}


def get_task_spec(task_name: str) -> ExecutionTaskSpec:
    resolved = TASK_REGISTRY.get(str(task_name or "").strip())
    if resolved is None:
        available = ", ".join(sorted(TASK_REGISTRY))
        raise KeyError(f"未知执行任务：{task_name}。可用任务：{available}")
    return resolved


def list_task_specs() -> list[ExecutionTaskSpec]:
    return list(TASK_SPECS)


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
        "field_type": field.field_type,
        "placeholder": field.placeholder,
        "help_text": field.help_text,
        "default_value": field.default_value,
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
    }
