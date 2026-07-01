# 主脑身份对象
快照日期：`2026-07-01`

本文件定义 `H:\quant_project` 主脑的身份、目标函数和受保护对象类型。它不是团队流程系统；它是个人研究者的长期认知网络。

## object `workspace_brain`
`type`: root_memory
`definition`: `brain/` 是整个工作区的长期记忆、对象拓扑和接管入口。
`principle`: Agent 无状态，项目大脑有状态。
`goal`: 服务个人研究推进、资料收录、项目接管、快速重构和直接行动。
`methods`: `identify_object(task)`；`route_to_child_brain(task)`；`writeback(memory)`；`protect_object(object, method)`。

## object `researcher_workstyle`
`type`: default_action_model
`preferences`: 研究推进优先；直接实现优先；新优化、新结构、新路径优先。
`cleanup_semantics`: 旧机制、旧测试、旧兼容入口只有在仍有活跃调用、不可替代证据价值或外部接口责任时保留。
`validation_semantics`: 测试、审计、提交和守卫服务当前对象和当前结论，不维护过时架构惯性。
`truth_semantics`: 量化结论区分事实、推断和猜测；收益、效率和真实可用数据是核心约束。

## object `handoff_success`
`type`: success_criteria
`definition`: 新 agent 不依赖隐性上下文，也能找到相关对象、当前状态、证据入口和质量底线。
`traceability`: 关键结论能追溯到 brain object、registry、manifest、run tag、artifact 或 reference。
`state_model`: 已验证事实、未验证推断、待验证假设分层保存。
`long_run_model`: 长任务通过 PID/job/run id、日志、progress、summary、artifact 或 status 保持可接管。

## Protected Object Types
### object `active_data_base_or_unique_data`
`scope`: QDP v2 active manifest、dataset manifests、raw parquet、PIT 状态、唯一研究证据和不可重建资产。
`activation`: rebuild、switch pointer、cleanup、delete、migrate。

### object `pit_or_label_semantics`
`scope`: PIT/no-leakage、label completeness、future availability、evidence grade。
`activation`: dataset build、feature/label change、training、evaluation、model conclusion。

### object `active_execution_artifact`
`scope`: live/default/paper/broker、active artifact、promotion gate、trade plan。
`activation`: activate、restore、update、generate trade plan、paper/live/broker wiring。

### object `secret_or_external_state`
`scope`: secrets、accounts、external services、remote deployment、real trading state。
`activation`: create、rotate、write、deploy、connect。

### object `cross_project_dirty_work`
`scope`: unrelated dirty paths、processes、ports、GPU jobs、provider tasks、outputs。
`activation`: manage、delete、reuse、commit、wait。

## Child Brain Registry
- `daily_research/brain/`: 正式生产研究与执行主线。
- `quant_data_platform/brain/`: 共享 QDP v2 manifest-first 数据基底 owner。
- `t0_project/brain/`: 盘中实验与 RL 原型。
- `daily_stock_analysis-main/brain/`: 独立产品分脑。
- `traditional_quant_research/brain/`: 传统量化方法研究分脑。
