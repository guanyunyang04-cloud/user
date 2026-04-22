# 研究日志

## 当前结论

- 本文件现在是 `daily_research` 分脑的轻量 episodic 入口，不再承载全部历史原文。
- 当前三层结构固定为：
  - 当前结论：本文件保留最新可执行结论、当前证据判断和后续写回纪律。
  - 证据索引：`daily_research/brain/references/episodic_memory_evidence_index_20260422.md`。
  - 历史原文：`daily_research/brain/references/episodic_memory_history_raw_20260317_20260422.md`。
- 当前接管默认仍然先读 `identity_layer.md -> state_center.md -> knowledge_center.md -> operations_center.md`；只有需要完整过程证据时才进入本文件和历史原文。
- 当前生产 / 执行结论仍以 `state_center.md` 为准；本文件只记录过程证据、归档索引和动作后复盘。
- 当前默认执行链仍是 `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`，默认 production root 仍是 `daily_research/output/short_expert_policy_v5b_execalign_production_default`。
- 当前默认执行分数语义已修为 selected composite decision score，对外字段使用 `model_decision_score` / `模型综合决策分`；`learned_score` 只作为 sub-head / debug 信号。
- 当前 continuous_policy 研究主矛盾仍是 `deploy intent not executable`：r9 已说明 `clip reduction != deploy executability`，后续不得把减少 budget clip 误当作执行意图闭环。
- 当前 r7 / r8 / r9 的证据关系：
  - r7：hierarchy 方向正确，但信用分配没有闭环。
  - r8：constraint-only portfolio defense 方向正确，但 budget clipping 与 deploy alignment 仍未闭环。
  - r9：`cash_constraint_intent_guard_v5` 能压低 clip，但主要冲突暴露为 `add -> hold`。
- 当前默认纪律：不启动训练、不切换 live 默认执行、不改写 promotion 结论，除非用户明确要求或 `state_center.md` 已更新为新的正式决策。
- 当前剩余维护风险：`daily_research/output` 与 `daily_research/cache` 仍是大体量热产物区；没有明确保留策略前继续不裁剪。

## 证据索引

- 详细标题索引：`daily_research/brain/references/episodic_memory_evidence_index_20260422.md`。
- 历史原文：`daily_research/brain/references/episodic_memory_history_raw_20260317_20260422.md`。
- 归档前原文件规模：`861319` UTF-8 bytes，约 `16573` 行，`302` 个二级章节。

| 证据层 | 入口 | 使用场景 |
| --- | --- | --- |
| 当前结论 | `daily_research/brain/episodic_memory.md` | 接管时快速确认最新过程结论和索引位置 |
| 证据索引 | `daily_research/brain/references/episodic_memory_evidence_index_20260422.md` | 需要按日期、阶段或标题定位旧证据 |
| 历史原文 | `daily_research/brain/references/episodic_memory_history_raw_20260317_20260422.md` | 需要完整复盘、核查指标、追溯原始实验叙述 |

## 历史原文

- 历史原文已原样归档到 `daily_research/brain/references/episodic_memory_history_raw_20260317_20260422.md`。
- 该归档保存 2026-03-17 至 2026-04-22 的长过程记录，包括 baseline、advanced_ml、deep_alpha、execution、short_expert、continuous_policy、默认执行维护和主分脑治理。
- 历史原文只作为证据，不自动覆盖当前状态；若与 `state_center.md`、`knowledge_center.md` 或最新正式验证冲突，以当前中心文档和最新验证为准。
- 后续新增 episodic 记录默认先写入本文件；当记录再次膨胀时，再按同样方式生成新的证据索引和历史原文归档。

## 2026-04-22 episodic_memory 历史归档瘦身

- 动作前自检：
  - 用户明确要求专门做一次 `episodic_memory.md` 历史归档瘦身，并分层为“当前结论 / 证据索引 / 历史原文”。
  - 本轮只处理脑文档结构，不启动训练、不停止训练、不切换 live 默认执行、不改写 promotion 结论。
  - 归档原则是保全优先：旧正文先原样归档，再生成索引，最后重写轻量入口。
- 已完成实现：
  - 将归档前完整 `episodic_memory.md` 原样保存为 `daily_research/brain/references/episodic_memory_history_raw_20260317_20260422.md`。
  - 生成 `daily_research/brain/references/episodic_memory_evidence_index_20260422.md`，按阶段和二级标题行号索引历史原文。
  - 将当前 `episodic_memory.md` 精简为三层入口：当前结论、证据索引、历史原文。
- 动作后复盘：
  - 事实：历史证据没有删除，只从默认接管入口下沉到 `brain/references/`。
  - 推断：后续接管成本会明显下降，因为 agent 不再需要默认加载 1.6 万行长日志。
  - 决策：未来只有需要完整证据链时才打开历史原文；日常接管以当前结论和索引为入口。