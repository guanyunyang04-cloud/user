# Daily Research 治理层

快照日期：`2026-04-12`

## 1. 目标
- 防止项目在长周期研究、多 agent 接管、长实验接力中发生目标偏移、上下文偏移和证据偏移。
- 强制任何执行都走“读取状态 -> 通过检查 -> 执行 -> 复盘 -> 写回 -> 重规划”的闭环。

## 2. 标准认知循环
1. `Wake`
   - 先读身份层和交接包。
2. `Locate`
   - 判断当前问题属于 formal、recent、live、promotion 的哪一层。
3. `Choose`
   - 用“目标贡献 × 紧急性 × 依赖满足度 × 可执行性 ÷ 风险成本”选任务。
4. `Preflight`
   - 过四检后才执行。
5. `Act`
   - 产出动作结果、证据、新事实、新问题、风险变化。
6. `Reflect`
   - 判断哪些信息要进入 semantic、lesson、rule、handoff。
7. `Replan`
   - 如结果改写路径，立刻更新 temporal state 与下一步建议。

## 3. 执行前四检
- 目标一致性检查：
  - 这件事是否服务当前北极星和当前主问题。
- 规则冲突检查：
  - 是否违反 formal / recent / live 分层、strict resume、前台执行等硬规则。
- 经验教训检查：
  - 是否踩中 lesson memory 里已知失败模式。
- 依赖完整性检查：
  - 数据、脚本、解释器、预算、上下文是否足够。

## 4. 偏移检测
- 目标偏移：
  - 当前任务已经不再服务真实主问题。
- 上下文偏移：
  - 使用旧 packet、旧结论或旧默认值。
- 行为偏移：
  - 执行后不写回，或只写日志不沉淀规则。
- 认知偏移：
  - 把 recent 当 promotion，把 selected profile 当 deployable truth。

## 5. 修复机制
- 轻度偏移：
  - 刷新 handoff packet、temporal state、working memory。
- 中度偏移：
  - 回到最近稳定 verdict，重做任务拆分和证据链。
- 重度偏移：
  - 冻结相关晋升判断，只允许继续做证据补齐与风险隔离。

## 6. 角色拆分
- 执行者：
  - 负责做事，不负责偷偷改规则。
- 审查者：
  - 负责检查证据、语义和约束是否一致。
- 复盘者：
  - 负责把这次结果变成长期资产。
- 即使只有一个 agent，也要按这三个角色依次思考和写回。

## 7. 当前治理守卫
- 正式实验收尾必须跑：
  - `python daily_research/tools/project_consistency_check.py`
  - `python daily_research/tools/doc_guard.py check`
- 新的默认接管顺序必须先看：
  - `identity_layer.md`
  - `handoff_packet.md`
  - `temporal_state.md`
- 旧的 `episodic_memory.md` 仍保留证据，但不再作为第一入口。
