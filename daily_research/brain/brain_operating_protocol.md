# Daily Research 操作协议对象
快照日期：`2026-06-27`

本文件是 `daily_research` 的可选补充入口对象，不属于核心 7 模块核；核心结构和路由以 `brain_manifest.json` 与 7 个中枢文件为准。

## object `daily_research_entry`
`type`: optional_entry_protocol
`definition`: 当任务指向 `daily_research`，agent 可以直接进入本分脑；route / capsule 是诊断传感器，不是准入仪式。
`inputs`: 用户目标、显式路径、state、knowledge、operations、reference、QDP manifest、run artifact。
`methods`: `enter_by_task(task)`；`inspect_relevant_objects(task)`；`query_evidence(claim)`；`writeback(result)`。

## object `capsule_sensor`
`type`: optional_router
`command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<task>" --json`
`api_fallback`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<task>" --workflow auto --json`
`semantics`: 输出 workflow hints、freshness 和 checklist；它提供线索，不替代对象语义和文件证据。

## object `evidence_query`
`type`: evidence_lookup
`activation`: 任务涉及 study、protocol、dataset、r-number、run tag、provider result 或模型结论。
`truth_order`: explicit tag / manifest / summary / registry / reference 优先；loose `latest_*` 是候选线索。
`methods`: `query_registry(q)`；`inspect_summary(path)`；`open_reference(path)`；`classify_evidence(run)`。

## object `mainline_pointer`
`type`: switchable_research_pointer
`semantics`: 每条研究主线都是可切换对象，不是永久层级；旧线保留为历史或对照，只有被任务选中时激活。
`current_default`: 见 `state_center.md` 的 `current_research_pointer`。
`execution_relation`: research mainline 切换不等于 live/default promotion。

## Procedure Entries
### procedure `enter_daily_research`
`input`: task
`steps`: 识别相关对象；读取 `state_center` 和必要 reference；需要时运行 capsule；执行任务；按 `writeback_route(result)` 写回。
`side_effects`: 取决于所选对象；默认是 research / documentation。

### procedure `evidence_backed_answer`
`input`: claim or status question
`steps`: 解析 claim；查 explicit tag / registry / reference；区分事实、推断和缺口；给出当前结论和证据入口。
`side_effects`: none unless user asks to update brain.

### procedure `mainline_switch`
`input`: user direction or governance-level decision
`steps`: 标记新 research pointer；保留旧线为 historical object；更新 state/reference；不自动触碰 execution surface。
`side_effects`: brain state/reference only.

## Writeback Targets
- 当前对象实例：`daily_research/brain/state_center.md`
- 长期对象和方法论：`daily_research/brain/knowledge_center.md`
- 过程入口和命令：`daily_research/brain/operations_center.md`
- 对象治理和 guard：`daily_research/brain/governance_layer.md`
- 长证据和复盘：`daily_research/brain/references/`
- 机器索引：`daily_research/brain/references/evidence_registry.json`
