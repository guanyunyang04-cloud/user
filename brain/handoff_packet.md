# 主脑交接包

快照日期：`2026-04-12`

## 1. 当前状态摘要
- 当前工作区正式生产主线仍是 `daily_research`。
- 当前主脑升级重点是：把脑网络从“能接管”进一步升级成“像人脑一样分层、可替换 agent、状态外显、带治理闭环”的结构。
- 当前三个 child_brain 都已经升级到统一的接管快路；`daily_research` 仍是其中最复杂、最重生产责任的样板分脑。

## 2. 当前正在处理的问题
- 让接管不再依赖单个 agent 的隐性上下文。
- 让项目状态可以按 `过去 / 当下 / 未来` 统一表达。
- 让规则、教训、交接包、自检和漂移修复都有固定落点。

## 3. 已完成基础
- 现有主脑/分脑已经有 manifest、working/procedural/environment/action/episodic 等基础骨架。
- `daily_research` 已经完成 formal / recent / live 分层治理，并具备较强的实验与执行闭环。
- front-only、strict resume、默认 `10` 小时、优先最有效而非最小改动等硬纪律已经落进分脑。
- 现在三个 child_brain 都已补齐：
  - `identity_layer.md`
  - `handoff_packet.md`
  - `rule_memory.md`
  - `lesson_memory.md`
  - `temporal_state.md`
  - `governance_layer.md`

## 4. 当前风险
- 脑文件已经很多，若没有更强的角色分层，新 agent 容易读到大量信息但抓不住当前最重要状态。
- 若没有标准交接包和 lesson memory，经验仍可能埋在 episodic 里，接管成本高。
- 若没有治理层，自检和反偏移仍然主要依靠执行 agent 的自觉。

## 5. 推荐下一动作
- 优先接入 `daily_research/brain/identity_layer.md`
- 再看 `daily_research/brain/handoff_packet.md`
- 再看 `daily_research/brain/temporal_state.md`
- 然后才进入 `semantic / working / action / episodic`

若目标不是 `daily_research`，同样默认先看对应 child 的：

- `identity_layer.md`
- `handoff_packet.md`
- `temporal_state.md`

## 6. 不要重复的无效路径
- 不要再把“读完整个 episodic_memory”当成默认接管起点。
- 不要再把聊天记录当脑真源。
- 不要设计成“某个 agent 会做，所以系统能跑”。

## 7. 置信度与不确定性
- 高置信度：
  - 现在整个工作区脑网络已经统一到同一接管协议。
- 中置信度：
  - 后续真正的挑战不再是“有没有骨架”，而是“是否持续按骨架写回并维护一致性”。
