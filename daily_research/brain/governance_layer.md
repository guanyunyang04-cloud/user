# Daily Research 治理对象
快照日期：`2026-07-10`

本文件描述对象级不变量、纯判断函数和治理过程。它不要求每次任务复述全部边界；agent 先选择相关对象，再激活对应治理。

## Governed Objects
### object `research_surface`
`scope`: scorer、feature、model、backtest、candidate review、research document。
`invariants`: 结论带证据等级；数据来自 QDP explicit id / manifest / pack；低预算结果只产生候选或诊断；forward-path fold 必须满足 `train.label_end_trade_date < oos_start_trade_date`，仅按 signal year 切分不构成 PIT 隔离；`fixed_oos` 不得参与 checkpoint selection 或 early stopping。
`inactive_boundaries`: live/default、broker、paper/live、active artifact。

### object `execution_surface`
`scope`: active/default、paper/live、broker、trade plan、execution restore、promotion。
`state`: `frozen_skeleton_only / awaiting_research_rebuild`
`invariants`: active artifact 只在显式授权和 promotion-grade 证据同时存在时进入变更过程；非执行任务只读或不激活。

### object `data_surface`
`scope`: QDP lake、canonical、provider ingest、memmap、training pack。
`invariants`: `daily_research` 消费 QDP，不直连在线 provider；provider 需求路由到 QDP 上游对象。

### object `evidence_surface`
`scope`: run tag、study/protocol summary、reference、registry、manifest。
`invariants`: loose latest 是候选线索；timeout、interrupted、smoke、dry-run 和 realtime tail label 不自动升级为 completed model-quality evidence。

## Pure Functions
- `select_relevant_objects(task)`: 根据任务文本、路径和产物选择 research / execution / data / evidence 对象。
- `classify_task(task)`: 返回 `research_work`、`data_substrate_work`、`execution_work`、`documentation_work` 或 `brain_maintenance`。
- `classify_evidence(run)`: 返回 `smoke_only`、`scout_only`、`evidence_grade`、`promotion_grade`、`diagnostic_only` 或 `insufficient`。
- `activate_execution_surface(objects, method)`: 只有对象和方法都触碰执行面时返回 true。
- `select_validation(changed_paths, task_class)`: 从 changed-surface、doc guard、integrity、QDP validate、project consistency 中选最小充分验证。
- `writeback_route(result)`: 当前状态写 `state_center`；长期事实写 `knowledge_center`；过程入口写 `operations_center`；长证据写 `references/`。

## Procedures
### procedure `research_work`
`input`: task、selected research object、QDP evidence、expected artifact
`steps`: 读取当前对象；执行研究或分析；标注证据等级；运行最小充分验证；把摘要写回 state 或 reference。
`side_effects`: research artifacts and brain notes；不触碰 active execution。

### procedure `data_requirement`
`input`: data need from research task
`steps`: 转成 QDP provider/canonical requirement；在 QDP 内 ingest / audit / canonicalize；返回 explicit manifest 或 pack 给 `daily_research`。
`side_effects`: QDP artifacts；`daily_research` 只记录消费关系。

### procedure `execution_change`
`input`: explicit user authorization、promotion-grade evidence、active artifact diff plan
`steps`: 激活 `execution_surface`；inspect active artifact；对齐 promotion evidence；执行对象级验证；再进入 restore / activate / trade plan。
`side_effects`: protected execution artifacts；过程需要单独说明。

### procedure `brain_maintenance`
`input`: brain docs or skill changes
`steps`: 保持热路径 compact；把长历史下沉到 `references/`；运行 `doc_guard`、`integrity_check` 和 `brain-structure-audit`；必要时更新 skill。
`side_effects`: brain docs, registry, skill files。

## Guard Entrypoints
- Brain docs: `python -m tools.brain.doc_guard check --scope changed`
- Brain integrity: `python -m tools.brain.integrity_check --json`
- Structure audit: `python brain/skills/workspace-brain/scripts/brain_runtime.py brain-structure-audit --cwd . --mode compact`
- Research consistency: `python daily_research/tools/project_consistency_check.py --mode research`
- Execution consistency: only when `execution_surface` is active, use `--mode execution` or `--mode full`
