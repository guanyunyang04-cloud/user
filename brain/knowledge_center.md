# 主脑知识中枢
快照日期：`2026-07-01`

## Object Classes
### class `workspace_memory`
`definition`: `H:\quant_project` 的个人研究者长期记忆和跨项目路由层。
`method`: 选择对象、选择分脑、保护唯一资产、要求必要写回。
`boundary`: 主脑保存跨项目事实；分脑保存项目事实；reference 保存历史证据。

### class `workspace_brain_skill`
`definition`: 脑区接管入口提示器。
`method`: 识别项目对象、受保护对象和写回需求。
`invariant`: 对 brain-governed 项目的 repo-tracked mutation，至少要检查相关分脑是否需要读入或写回。

### class `qdp_manifest_first_data_base`
`owner`: `quant_data_platform`
`definition`: 工作区共享本地数据基底；事实源是 `parquet + dataset.json + active.json`。
`not_definition`: archived v1 workflow artifacts and downstream training artifacts are not the active data base itself.
`consumers`: `daily_research`、`traditional_quant_research` 和后续研究项目。
`protected_methods`: active pointer 切换、raw parquet 删除、dataset manifest 改写、PIT 状态/范围改变。

### class `daily_research_active_artifact`
`definition`: `daily_research` 执行状态和 active artifact 对象。
`activation`: 执行、paper/live、active/default、broker、交易计划或 promotion。
`boundary`: 数据基底维护、provider 评估、普通研究计划不自动激活执行面。

### class `evidence_grade`
`definition`: 研究证据可信等级。
`levels`: smoke、scout、evidence-grade、promotion-grade。
`invariant`: 不把低预算、单 seed、短窗口、timeout 或 incomplete run 写成正式结论。

### class `brain_sync`
`definition`: 代码/数据/产物现实和脑区 hot path 的一致性。
`trigger`: 架构、CLI、数据基底、active 指针、质量结论、研究主线、执行边界或稳定命令变化。
`method`: 更新对应分脑 hot path；长细节进 reference；运行 `tools.brain.brain_sync_audit`。

## Procedures
- `route_project_task(task)`: select main brain object and child brain before touching repo state.
- `sync_brain_after_change(change)`: update hot-path state, knowledge, operations or governance when durable facts changed.
- `audit_brain_sync()`: run sync, integrity and structure checks before closing brain-maintenance work.

## Long-Term Lessons
- 如果主脑和分脑维护两套平行对象定义，后续 agent 很快会漂移。
- 如果当前状态只写聊天或终端，不写 brain，接管可靠性会明显下降。
- 如果 hot path 保留旧架构词，agent 会被旧叙事牵引；历史词应进入 references。
- 如果工具输出在 Windows/conda 下触发 GBK 错误，应优先用 yolos python 直跑或强制 UTF-8。
- 简单当前架构优先于兼容仪式；旧 wrapper、fallback 和入口在无活跃调用时应归档或删除。
