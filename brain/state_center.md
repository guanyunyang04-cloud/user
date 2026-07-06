# 主脑状态程序
快照日期：`2026-07-01`

本文件只保存 `H:\quant_project` 当前跨项目对象。项目细节写入对应分脑；历史过程进入各自 `references/`。

## Module Interface
`exports`: `workspace_root = H:\quant_project`；`primary_project = daily_research`；`data_base_owner = quant_data_platform`；`registered_child_brains = [daily_research, quant_data_platform, t0_project, daily_stock_analysis-main, traditional_quant_research]`。
`sensors`: route、capsule、bootstrap、health、brain_sync_audit、doc_guard、integrity_check；传感器提供证据，不替代用户目标和文件事实。

## Object Instances
### object `workspace_memory`
`type`: root_brain
`state`: 主脑维护共享对象、项目关系、写回规则和受保护对象类型；项目事实、命令细节、研究指标写入分脑。
`methods`: `orchestrate_task(task)`；`route_writeback(result)`；`run_brain_sync_audit()`。

### object `workspace_git_surface`
`type`: repo_surface
`state`: 默认工作面是 `main`；当前工作区是 `H:\quant_project`。
`activation`: repo-tracked mutation、提交、清理、迁移、分支或 worktree 操作。
`methods`: `inspect_status()`；`protect_unrelated_dirty_paths()`；`stage_or_commit_when_requested()`。

### object `quant_data_platform_child`
`type`: shared_data_base_child_brain
`path`: `quant_data_platform/brain/`
`state`: QDP 当前事实是 v2 manifest-first 本地数据基底：`parquet + dataset.json + active.json`。`active.json` 只记录当前表指针和数据范围；memmap/training pack 是下游研究产物，不属于 active 数据基底。
`public_commands`: `qdp status/list/describe/check/rebuild/gc/update`。

### object `daily_research_child`
`type`: production_research_child_brain
`path`: `daily_research/brain/`
`state`: 正式生产研究与执行主线；消费 QDP v2 数据基底或显式下游研究产物。执行面以该分脑 `state_center.md` 为准。
`protected_object`: `daily_research/output/active_execution_strategy.json`

### object `cross_project_orchestration`
`type`: task_orchestration_policy
`state`: route/capsule now expose `primary_brain_id`、`supporting_brain_ids`、`object_routes` and `writeback_targets`; selected child is the primary owner, not the only project an agent may inspect.
`rule`: QDP active data base objects are owned by `quant_data_platform`; research artifacts, sequence packs, models, losses, evaluations, stock profiles and backtests are owned by `daily_research`.
`example`: QDP v2 data used for GRU/path-value training returns primary `daily_research` with supporting read-only `quant_data_platform`.
`artifact_default`: new model-ready training datasets should enter `daily_research/data/research_store/<artifact_id>/`; old `quant_data_platform/data/qdp_v2/research/sequence_pack/` artifacts remain compatibility research artifacts, not QDP active data base.

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

## Pure Functions
- `orchestrate_task(task)`: 根据路径、项目名、数据资产和用户目标返回 primary brain、supporting brains、object routes、读写模式和写回目标。
- `select_protected_objects(task)`: 只返回任务实际触碰的 active data base、PIT/label、active execution、secret/external state 或 cross-project dirty objects。
- `resolve_data_owner(asset)`: QDP v2 active data base、dataset manifests、raw parquet、provider ingest 和数据清理返回 `quant_data_platform_child`；研究 pack/memmap 返回其生成项目或显式 owner。
- `resolve_research_owner(task)`: 研究、模型、回测、执行候选和 active artifact 默认返回 `daily_research_child`。
- `requires_brain_writeback(change)`: 架构、CLI、数据指针、数据语义、质量结论、执行边界或稳定项目事实变化时返回 true。

## Procedures
### procedure `workspace_handoff`
`input`: user task
`steps`: 理解目标；选择相关对象和分脑；读取最小必要 state/reference/artifact；执行任务；验证；按对象写回。
`side_effects`: selected child brain or workspace docs only.

### procedure `brain_sync_closure`
`input`: changed files and task result
`steps`: 判断是否改变 durable project fact；若改变，更新对应分脑 hot path 或写 reference；运行 `brain_sync_audit`、doc guard 和必要结构检查；final 明确同步状态。
`side_effects`: brain docs or references.

## Current Invariants
- 主脑不展开分脑 trial 指标、长 tag 或局部实验命令。
- 当前事实写 hot path；历史证据写 `references/`。
- 脑区治理项目发生架构、CLI、active pointer、数据语义、质量结论或执行边界变化时，必须同步对应分脑，或在 final 明确说明未同步原因。
- 任何中文乱码疑似问题先用 UTF-8 读取复核；Windows/conda GBK 输出错误不等于文件损坏。
