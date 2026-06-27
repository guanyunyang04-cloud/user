# 主脑状态程序
快照日期：`2026-06-27`

本文件保存 `H:\quant_project` 主脑的当前运行时对象。项目细节写入对应分脑；主脑只保存跨项目拓扑、共享对象和受保护对象类型。

## Module Interface
`exports`: `workspace_root = H:\quant_project`；`primary_project = daily_research`；`data_substrate = quant_data_platform`；`registered_child_brains = [daily_research, quant_data_platform, t0_project, daily_stock_analysis-main, traditional_quant_research]`。
`sensors`: route、capsule、bootstrap、health、doc_guard、integrity_check；它们提供诊断，不是准入仪式。

## Object Instances
### object `workspace_memory`
`type`: root_brain
`state`: 主脑维护共享对象、项目关系和受保护对象类型；项目事实、实验指标、命令细节写入分脑。
`methods`: `identify_project(path)`；`select_child_brain(task)`；`route_writeback(result)`；`run_brain_guard(scope)`。

### object `workspace_git_surface`
`type`: repo_surface
`state`: 默认工作面是 `main`；当前 worktree 审计只有 `H:\quant_project` 一个物理 worktree。
`activation`: repo-tracked mutation、提交、清理、迁移、分支或 worktree 操作。
`methods`: `inspect_status()`；`stage_or_commit_when_requested()`；`protect_unrelated_dirty_paths()`。

### object `daily_research_child`
`type`: production_research_child_brain
`path`: `daily_research/brain/`
`state`: 正式生产研究与执行主线；当前研究指针和执行冻结状态以该分脑 `state_center.md` 为准。
`protected_object`: `daily_research/output/active_execution_strategy.json`

### object `quant_data_platform_child`
`type`: shared_data_substrate_child_brain
`path`: `quant_data_platform/brain/`
`state`: 共享量化数据平台、provider ingest、canonical lake、registry、coverage audit、policy bundle、memmap 和 training pack owner。

### object `t0_project_child`
`type`: intraday_experiment_child_brain
`path`: `t0_project/brain/`
`state`: 盘中实验与 RL 原型；不替代 `daily_research` 正式执行默认。

### object `daily_stock_analysis_child`
`type`: independent_product_child_brain
`path`: `daily_stock_analysis-main/brain/`
`state`: 独立产品分脑；不改写 `daily_research` active artifact 或 promotion gate。

### object `traditional_quant_research_child`
`type`: traditional_quant_child_brain
`path`: `traditional_quant_research/brain/`
`state`: 传统量化方法研究分脑；项目事实、研究记录和局部命令以该分脑和项目产物为准。

### object `legacy_tdx_plugin_path`
`type`: deprecated_external_path
`path`: `H:\new_tdx64\PYPlugins\user`
`state`: 不承载当前项目；只可能出现在历史 reference、回滚说明或通达信原生用户插件场景。

## Pure Functions
- `select_child_brain(task)`: 根据路径、项目名、数据资产或用户目标返回相关分脑。
- `select_protected_objects(task)`: 只返回任务实际触碰的 canonical data、PIT/label、active execution、secret/external state 或 cross-project dirty objects。
- `resolve_data_owner(asset)`: canonical 数据集、policy bundle、registry、sharded memmap 和数据清理默认返回 `quant_data_platform_child`。
- `resolve_research_owner(task)`: 研究、模型、回测、执行候选和 active artifact 默认返回 `daily_research_child`。
- `classify_polling_state(handle)`: 观察窗口耗尽、仍有进展、明确失败、资源危险、用户停止分别返回不同状态。

## Procedures
### procedure `workspace_handoff`
`input`: user task
`steps`: 理解目标；选择相关对象和分脑；读取最小必要 state/reference/artifact；执行任务；验证；按对象写回。
`side_effects`: selected child brain or workspace docs only.

### procedure `brain_structure_change`
`input`: changed brain files, skill, manifest, workflow or registry
`steps`: 保持热路径 compact；长历史进 references；运行 doc guard、integrity check、structure audit 和 multi-paradigm lint；必要时同步 skill。
`validation`: `python -m tools.brain.doc_guard check --scope changed`；`python -m tools.brain.integrity_check --json`；`python brain/skills/workspace-brain/scripts/brain_runtime.py brain-structure-audit --cwd . --mode compact`；`python brain/skills/workspace-brain/scripts/brain_runtime.py multi-paradigm-lint --cwd . --scope attached`。

## Current Invariants
- 主脑不展开分脑 trial 指标、长 tag 或局部实验命令。
- 任何中文乱码疑似问题先用 UTF-8 读取复核，再判断是否是文件损坏。
- 需要等待外部状态或跨轮观察的任务应绑定 PID / job id / run id、日志、progress、summary、artifact、端口或 API status 等可观察 handle。
- 观察窗口耗尽不是失败证据；只有明确错误、资源危险、失败产物或用户停止才中断或写失败。
