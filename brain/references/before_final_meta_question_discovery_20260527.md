# Before-final 元问题发现协议

日期：`2026-05-27`

## 结论
- `before_final` 的核心不是固定反思清单，而是任务闭合边界上的元问题发现。
- 当对象层任务准备交付但 final answer 尚未发出时，agent 低噪声判断是否暴露了问题框架、完成标准、方法选择、权威排序、评价机制或学习显著性缺口。
- 若工具 / audit 返回 clear，但人类反馈或任务事实显示闭合判断不成立，clear 只作为传感器信号；agent 应优先说明元问题，并征求是否沉淀为 agent/brain 学习。

## Supporting Example
- `20260527_150314_resolve_handoff-user_plan_conflicts_explicitly` 已降级为 supporting example。
- 它不再驱动 handoff 特例规则；它说明的是 `authority_order_mismatch` 与 `missed_learning_salience`：agent 把旧 handoff/保守边界置于用户明确计划之上，并在任务结束时没有主动识别可泛化的元认知缺陷。

## 当前边界
- 权限仍为 `propose_only`。
- 发现元问题后应先询问用户是否演化，不自动创建 proposal，不自动修改核心协议。
- 普通任务保持低噪声，不强制追加自检尾巴。
