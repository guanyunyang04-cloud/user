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
        safety_level="danger",
        safety_summary="会重建 production_root，并可能改写当前默认执行。",
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
        description="刷新可部署执行策略排行榜（默认只读）。",
        form_fields=(
            TaskFieldSpec(
                name="activate_winner",
                label="同步改写当前默认执行（危险）",
                arg_flag="--activate-winner",
                false_arg_flag="--no-activate-winner",
                field_type="boolean",
                help_text="默认关闭。关闭时只刷新排行榜产物；开启后才会把冠军写入 active manifest。",
                default_checked=False,
                section="manifest",
            ),
        ),
        launcher_notes=(
            "默认行为已改为只读刷新，不会再静默切走 active_execution_strategy。",
            "只有显式勾选“同步改写当前默认执行（危险）”后，才会覆盖当前默认执行。",
        ),
        safety_level="caution",
        safety_summary="默认只读；只有勾选危险开关后才会改写当前默认执行。",
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
        safety_level="danger",
        safety_summary="会直接改写当前 active execution manifest。",
    ),
    ExecutionTaskSpec(
        name="continuous-policy-protocol",
        script_relative_path="daily_research/continuous_policy/run_continuous_policy_protocol.py",
        description="按正式顺序执行 continuous policy train -> evaluate -> shadow continuity -> export。",
        category="continuous_policy",
        form_fields=(
            TaskFieldSpec(
                name="pool_name",
                label="股票池",
                arg_flag="--pool-name",
                default_value="liquid500",
                help_text="协议统一使用的股票池；也可以填 `all_a` 或 `learned_all_a`，让模型在全A候选域内学习选股。",
                section="selection",
            ),
            TaskFieldSpec(
                name="label_preset",
                label="标签预设",
                arg_flag="--label-preset",
                field_type="select",
                default_value="balanced_v2",
                choices=("balanced_v2", "swing_v2", "defensive_v2", "holdcash_v3", "holdcash_v4"),
                help_text="控制生命周期 teacher 标签与行为风格的正式预设。",
                section="selection",
            ),
            TaskFieldSpec(
                name="trainer_backend",
                label="训练后端",
                arg_flag="--trainer-backend",
                field_type="select",
                default_value="formal_torch_v2",
                choices=("formal_torch_v2", "formal_torch_seq_v3", "formal_torch_hier_v4", "prototype_gbdt_v1"),
                help_text="formal_torch_v2 / formal_torch_seq_v3 / formal_torch_hier_v4 支持 GPU/32epoch/strict resume；prototype_gbdt_v1 仅限 shadow 原型。",
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
                placeholder="20260413",
                help_text="正式评估窗口终点。",
                section="window",
            ),
            TaskFieldSpec(
                name="shadow_start_date",
                label="Shadow 起始",
                arg_flag="--shadow-start-date",
                placeholder="20260401",
                help_text="连续 shadow continuity 检查起点；留空则自动取评估窗口末段。",
                section="window",
            ),
            TaskFieldSpec(
                name="shadow_end_date",
                label="Shadow 结束",
                arg_flag="--shadow-end-date",
                placeholder="20260413",
                help_text="连续 shadow continuity 检查终点，同时也是最新导出信号日。",
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
                help_text="formal_torch_v2 / formal_torch_seq_v3 / formal_torch_hier_v4 的初始正式预算至少 32 epoch。",
                section="runtime",
            ),
            TaskFieldSpec(
                name="resume_mode",
                label="续训模式",
                arg_flag="--resume-mode",
                field_type="select",
                default_value="strict",
                choices=("strict", "fresh"),
                help_text="正式训练默认 strict resume；扩预算时沿同一 run_dir 续训。",
                section="runtime",
            ),
            TaskFieldSpec(
                name="force_bootstrap_from_account",
                label="从账户重建",
                arg_flag="--force-bootstrap-from-account",
                field_type="boolean",
                help_text="最新导出时忽略已有 runtime state，直接从当前模拟账户重建。",
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
        name="continuous-policy-train",
        script_relative_path="daily_research/continuous_policy/train_policy.py",
        description="训练连续型组合策略代理，产出 shadow model artifact 与 teacher 上限摘要。",
        category="continuous_policy",
        form_fields=(
            TaskFieldSpec(
                name="pool_name",
                label="股票池",
                arg_flag="--pool-name",
                default_value="liquid500",
                help_text="默认跟随当前 active manifest 的流动性股票池；也可以填 `all_a` 或 `learned_all_a` 启用全A学习选股。",
                section="selection",
            ),
            TaskFieldSpec(
                name="label_preset",
                label="标签预设",
                arg_flag="--label-preset",
                field_type="select",
                default_value="balanced_v2",
                choices=("balanced_v2", "swing_v2", "defensive_v2", "holdcash_v3", "holdcash_v4"),
                help_text="控制生命周期 teacher 标签与训练目标的预设。",
                section="selection",
            ),
            TaskFieldSpec(
                name="trainer_backend",
                label="训练后端",
                arg_flag="--trainer-backend",
                field_type="select",
                default_value="formal_torch_v2",
                choices=("formal_torch_v2", "formal_torch_seq_v3", "formal_torch_hier_v4", "prototype_gbdt_v1"),
                help_text="formal_torch_v2 / formal_torch_seq_v3 / formal_torch_hier_v4 是可 promotion 的正式候选；prototype_gbdt_v1 只做 shadow/teacher。",
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
                name="start_date",
                label="起始日期",
                arg_flag="--start-date",
                default_value="20250318",
                placeholder="20250318",
                help_text="训练区间起点，状态构建仍会自动补足 warmup 历史。",
                section="window",
            ),
            TaskFieldSpec(
                name="end_date",
                label="结束日期",
                arg_flag="--end-date",
                placeholder="20260413",
                help_text="留空则用最新已完成交易日。",
                section="window",
            ),
            TaskFieldSpec(
                name="max_universe_size",
                label="最大股票数",
                arg_flag="--max-universe-size",
                placeholder="300",
                help_text="调试或 smoke test 时可限制股票数。",
                section="runtime",
            ),
            TaskFieldSpec(
                name="epochs",
                label="训练 Epoch",
                arg_flag="--epochs",
                default_value="32",
                help_text="formal_torch_v2 / formal_torch_seq_v3 / formal_torch_hier_v4 的正式训练预算；首次正式训练至少 32 epoch。",
                section="runtime",
            ),
            TaskFieldSpec(
                name="resume_mode",
                label="续训模式",
                arg_flag="--resume-mode",
                field_type="select",
                default_value="strict",
                choices=("strict", "fresh"),
                help_text="正式训练默认 strict；若继续加预算，沿同一 run_dir 续训。",
                section="runtime",
            ),
            TaskFieldSpec(
                name="tag",
                label="运行标签",
                arg_flag="--tag",
                placeholder="continuous_policy_train_r1",
                help_text="可选；留空则自动生成时间戳标签。",
                section="runtime",
            ),
            TaskFieldSpec(
                name="refresh_cache",
                label="刷新缓存",
                arg_flag="--refresh-cache",
                field_type="boolean",
                help_text="强制重建原始数据与 prepared bundle 缓存。",
                section="runtime",
            ),
        ),
        launcher_notes=("默认保持 shadow 研究模式，不直接替换当前 live 执行桥。",),
    ),
    ExecutionTaskSpec(
        name="continuous-policy-evaluate",
        script_relative_path="daily_research/continuous_policy/evaluate_policy.py",
        description="评估连续型组合策略代理，并对照 teacher 上限与当前 active manifest 参考链路。",
        category="continuous_policy",
        form_fields=(
            TaskFieldSpec(
                name="model_path",
                label="模型路径",
                arg_flag="--model-path",
                placeholder="留空则自动使用 latest continuous-policy artifact",
                help_text="可选；通常直接留空使用最新训练结果。",
                section="source",
            ),
            TaskFieldSpec(
                name="pool_name",
                label="股票池",
                arg_flag="--pool-name",
                default_value="liquid500",
                help_text="评估时使用的股票池；也可以填 `all_a` 或 `learned_all_a` 做全A评估。",
                section="selection",
            ),
            TaskFieldSpec(
                name="label_preset",
                label="标签预设",
                arg_flag="--label-preset",
                field_type="select",
                default_value="",
                choices=("balanced_v2", "swing_v2", "defensive_v2", "holdcash_v3", "holdcash_v4"),
                help_text="留空则沿用模型 artifact 里记录的训练预设。",
                section="selection",
            ),
            TaskFieldSpec(
                name="start_date",
                label="起始日期",
                arg_flag="--start-date",
                default_value="20250318",
                placeholder="20250318",
                help_text="评估起点。",
                section="window",
            ),
            TaskFieldSpec(
                name="end_date",
                label="结束日期",
                arg_flag="--end-date",
                placeholder="20260413",
                help_text="留空则用最新已完成交易日。",
                section="window",
            ),
            TaskFieldSpec(
                name="max_universe_size",
                label="最大股票数",
                arg_flag="--max-universe-size",
                placeholder="300",
                help_text="调试时可限制评估宇宙规模。",
                section="runtime",
            ),
            TaskFieldSpec(
                name="tag",
                label="运行标签",
                arg_flag="--tag",
                placeholder="continuous_policy_eval_r1",
                help_text="可选；留空则自动生成时间戳标签。",
                section="runtime",
            ),
        ),
    ),
    ExecutionTaskSpec(
        name="continuous-policy-export",
        script_relative_path="daily_research/continuous_policy/export_action_panel.py",
        description="导出连续型组合策略的最新 shadow 动作面板，并写回组合连续状态。",
        category="continuous_policy",
        form_fields=(
            TaskFieldSpec(
                name="model_path",
                label="模型路径",
                arg_flag="--model-path",
                placeholder="留空则自动使用 latest continuous-policy artifact",
                help_text="可选；通常直接留空。",
                section="source",
            ),
            TaskFieldSpec(
                name="signal_date",
                label="信号日期",
                arg_flag="--signal-date",
                placeholder="20260413",
                help_text="留空则用最新已完成交易日。",
                section="window",
            ),
            TaskFieldSpec(
                name="pool_name",
                label="股票池",
                arg_flag="--pool-name",
                default_value="liquid500",
                help_text="导出动作面板时使用的股票池；也可以填 `all_a` 或 `learned_all_a` 导出全A learned selection 结果。",
                section="selection",
            ),
            TaskFieldSpec(
                name="tag",
                label="导出标签",
                arg_flag="--tag",
                placeholder="20260413",
                help_text="可选；留空则默认用 signal_date。",
                section="runtime",
            ),
            TaskFieldSpec(
                name="force_bootstrap_from_account",
                label="重置连续状态",
                arg_flag="--force-bootstrap-from-account",
                field_type="boolean",
                help_text="忽略已有 runtime state，直接从当前模拟账户重建连续状态。",
                section="runtime",
            ),
        ),
        launcher_notes=("导出只生成 shadow 动作与解释，不自动改写 active manifest。",),
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
