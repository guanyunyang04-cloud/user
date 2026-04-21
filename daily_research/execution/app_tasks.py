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
        name="continuous-policy-protocol",
        script_relative_path="daily_research/continuous_policy/run_continuous_policy_protocol.py",
        description="按正式顺序执行 continuous policy 的 train -> evaluate -> shadow continuity -> export。",
        category="continuous_policy",
        form_fields=(
            TaskFieldSpec(
                name="pool_name",
                label="股票池",
                arg_flag="--pool-name",
                default_value="liquid500",
                help_text="协议统一使用的股票池；也可填 all_a 或 learned_all_a。",
                section="selection",
            ),
            TaskFieldSpec(
                name="label_preset",
                label="标签预设",
                arg_flag="--label-preset",
                field_type="select",
                default_value="balanced_v2",
                choices=("balanced_v2", "swing_v2", "defensive_v2", "holdcash_v3", "holdcash_v4"),
                help_text="控制生命周期 teacher 标签与行为风格。",
                section="selection",
            ),
            TaskFieldSpec(
                name="trainer_backend",
                label="训练后端",
                arg_flag="--trainer-backend",
                field_type="select",
                default_value="formal_torch_v2",
                choices=("formal_torch_v2", "formal_torch_seq_v3", "formal_torch_hier_v4", "prototype_gbdt_v1"),
                help_text="正式候选优先使用 torch 后端；prototype_gbdt_v1 只用于 shadow 原型。",
                section="selection",
            ),
            TaskFieldSpec(
                name="decoder_profile",
                label="解码预设",
                arg_flag="--decoder-profile",
                field_type="select",
                default_value="default_v2",
                choices=("default_v2", "budget_v3", "holdcash_v3", "reduceexit_v4", "cash_v4", "reduceexit_cash_v4"),
                help_text="控制组合预算、持有优先级与防御性收缩的解码器预设。",
                section="selection",
            ),
            TaskFieldSpec(
                name="train_start_date",
                label="训练起始",
                arg_flag="--train-start-date",
                default_value="20240102",
                placeholder="20240102",
                help_text="正式训练窗口起点。",
                section="window",
            ),
            TaskFieldSpec(
                name="train_end_date",
                label="训练结束",
                arg_flag="--train-end-date",
                default_value="20251231",
                placeholder="20251231",
                help_text="正式训练窗口终点。",
                section="window",
            ),
            TaskFieldSpec(
                name="eval_start_date",
                label="评估起始",
                arg_flag="--eval-start-date",
                default_value="20260102",
                placeholder="20260102",
                help_text="正式评估窗口起点。",
                section="window",
            ),
            TaskFieldSpec(
                name="eval_end_date",
                label="评估结束",
                arg_flag="--eval-end-date",
                placeholder="20260421",
                help_text="正式评估窗口终点，留空则使用最新完成交易日。",
                section="window",
            ),
            TaskFieldSpec(
                name="shadow_start_date",
                label="Shadow 起始",
                arg_flag="--shadow-start-date",
                placeholder="20260401",
                help_text="连续 shadow continuity 检查起点。",
                section="window",
            ),
            TaskFieldSpec(
                name="shadow_end_date",
                label="Shadow 结束",
                arg_flag="--shadow-end-date",
                placeholder="20260421",
                help_text="连续 shadow continuity 检查终点，也是最新导出信号日。",
                section="window",
            ),
            TaskFieldSpec(
                name="max_universe_size",
                label="最大股票数",
                arg_flag="--max-universe-size",
                placeholder="500",
                help_text="调试时可限制股票池规模；正式运行建议留空。",
                section="runtime",
            ),
            TaskFieldSpec(
                name="epochs",
                label="训练 Epoch",
                arg_flag="--epochs",
                default_value="32",
                help_text="正式 torch 后端初始预算至少 32 epoch。",
                section="runtime",
            ),
            TaskFieldSpec(
                name="resume_mode",
                label="续训模式",
                arg_flag="--resume-mode",
                field_type="select",
                default_value="strict",
                choices=("strict", "fresh"),
                help_text="正式训练默认 strict resume；fresh 会从头开始。",
                section="runtime",
            ),
            TaskFieldSpec(
                name="force_bootstrap_from_account",
                label="从账户重建",
                arg_flag="--force-bootstrap-from-account",
                field_type="boolean",
                help_text="导出时忽略已有 runtime state，直接从当前模拟账户重建。",
                section="runtime",
            ),
            TaskFieldSpec(
                name="tag",
                label="协议标签",
                arg_flag="--tag",
                placeholder="continuous_policy_protocol_r1",
                help_text="可选；留空则自动生成协议标签。",
                section="runtime",
            ),
        ),
        launcher_notes=("这是 continuous_policy 当前推荐的高层入口，会自动串联 train/evaluate/shadow/export。",),
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
    ),
    ExecutionTaskSpec(
        name="refresh-production-static-fallback",
        script_relative_path="daily_research/execution/refresh_production_static_fallback.py",
        description="刷新 production_root 的静态 fallback 产物与 manifest 元数据。",
    ),
    ExecutionTaskSpec(
        name="global-strategy-leaderboard",
        script_relative_path="daily_research/execution/run_global_deployable_strategy_leaderboard.py",
        description="刷新可部署执行策略排行榜；默认只读。",
        safety_level="caution",
        safety_summary="默认只读；只有显式传入危险开关后才会改写当前默认执行。",
    ),
    ExecutionTaskSpec(
        name="single-mapping-pipeline",
        script_relative_path="daily_research/execution/run_short_alpha_execution_single_mapping_candidate_pipeline.py",
        description="刷新 single-mapping 执行候选 pipeline 和 live monitor。",
    ),
    ExecutionTaskSpec(
        name="activate-single-mapping",
        script_relative_path="daily_research/execution/activate_execution_single_mapping_candidate.py",
        description="把 single-mapping 候选提升为当前 active execution manifest。",
        safety_level="danger",
        safety_summary="会直接改写当前 active execution manifest。",
    ),
    ExecutionTaskSpec(
        name="continuous-policy-train",
        script_relative_path="daily_research/continuous_policy/train_policy.py",
        description="训练连续型组合策略代理，产出 shadow model artifact。",
        category="continuous_policy",
    ),
    ExecutionTaskSpec(
        name="continuous-policy-evaluate",
        script_relative_path="daily_research/continuous_policy/evaluate_policy.py",
        description="评估连续型组合策略代理，并对照 teacher 与 active manifest 参考链路。",
        category="continuous_policy",
    ),
    ExecutionTaskSpec(
        name="continuous-policy-export",
        script_relative_path="daily_research/continuous_policy/export_action_panel.py",
        description="导出连续型组合策略的最新 shadow 动作面板。",
        category="continuous_policy",
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

CORE_FRONTEND_TASK_NAMES: tuple[str, ...] = (
    "trade-plan",
    "refresh-production-default",
    "continuous-policy-protocol",
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
    }
