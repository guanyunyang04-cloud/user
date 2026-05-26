# Frontier Latest / Protocol Mismatch 记录

## 事实

- `daily_research` frontier 检查仍可能报告 loose latest 与 protocol latest 不一致。
- 当前风险类型包括 `loose_latest_stale_requires_explicit_tag` 与 `unregistered_latest_run_tags`。
- `latest_*` 文件和最新 output study 只能作为候选线索，不能自动提升为当前事实。

## 影响

- 新 agent 如果只读取 loose latest，可能把未登记、未复核或未写回的输出误认为正式研究结论。
- 当前处理策略是先生成机器可读 frontier 明细，再由人工确认是否修 registry、补 reference，或明确废弃该输出。

## 处理策略

- 回答当前主线、模型、数据集、promotion 或执行端状态时，优先使用 explicit study tag、protocol tag、dataset id 和 evidence registry。
- 对 `unregistered_latest_run_tags` 只生成 reconciliation proposal，不伪造完成证据。
- 如果已有正式 reference 覆盖对应 tag，则修 registry 或 scanner 匹配逻辑；如果没有，则先补短 reference 或 proposal。
