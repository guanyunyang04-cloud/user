# Daily Research 教训记忆

快照日期：`2026-04-12`

## L001 - recent 胜利不能直接当 promotion 结论
- 事件：
  - 多次出现 corrected recent 很强的 challenger，但 formal / constrained formal 不足。
- 场景：
  - learned-control family、execution bridge 切换、同窗 recent 对照。
- 根因：
  - 把 recent frontier 误当成 deployable frontier。
- 信号：
  - recent 数值很亮眼，但 constrained replay 或 selected formal profile 明显掉队。
- 影响：
  - 容易误判默认升级时机。
- 修复：
  - 强制把 `fresh formal / constrained formal / recent / live` 分层叙述。
- 预防：
  - 新 frontier 必须同时回答 deployable constrained formal 表现。
- 升级动作：
  - 写入规则记忆与交接包。

## L002 - family 扩宽后，runner 也必须同步扩宽
- 事件：
  - `policy_v2 family` 首轮 pipeline 名义上跑了 constrained step，但实际只覆盖老 `policy_v2`。
- 场景：
  - family 扩展后沿用旧单模型 constrained runner。
- 根因：
  - pipeline 与 family 研究范围不同步。
- 信号：
  - recent frontier 已换人，但 constrained summary 仍只有旧模型。
- 影响：
  - 主线误以为“最大问题还是缺证据”，而不是“谁是 deployable winner 已经变了”。
- 修复：
  - 补齐 family constrained review。
- 预防：
  - family runner、family constrained、family recent 三者必须一起维护。
- 升级动作：
  - 写入脚本、brain、交接包。

## L003 - detached 长实验会降低接管可靠性
- 事件：
  - 后台 runner、分离终端或超时后无人接棒，容易让实验状态断裂。
- 场景：
  - 长 formal 训练、pipeline follow-through、跨脚本接力。
- 根因：
  - 运行状态只存在于后台进程，不在 brain 和当前前台链路里。
- 信号：
  - 需要额外追 PID、status 文件或后台日志才能知道实验是不是还活着。
- 影响：
  - 接管困难，失败后难恢复。
- 修复：
  - 长实验统一改成前台执行，并强制 `strict resume`。
- 预防：
  - 正式实验默认前台，默认 `10` 小时预算。
- 升级动作：
  - 写入 handoff rules 与 rule memory。

## L004 - hand-crafted repair 会出现明显天花板
- 事件：
  - current-default 多轮 repair 与 gross-control sweep 有改善，但整体进入边际收益递减。
- 场景：
  - current default gap 修复、market-state guard、gross map 调参。
- 根因：
  - 控制层修补有效，但手工规则无法持续扩大优势。
- 信号：
  - gross-only 还能涨，但 overlay 或更宽扫不再稳定增益。
- 影响：
  - 若继续广扫手工规则，会消耗算力和注意力。
- 修复：
  - 明确把优先级切回 learned-control。
- 预防：
  - 当 manual repair 进入冻结阶段，主线转回 learned-control。
- 升级动作：
  - 写入 procedural / project_map / handoff packet。

## L005 - selected profile 与 constrained best 分裂必须显式建模
- 事件：
  - `policy_v2c`、`policy_v4b`、`policy_v5b` 都展示了不同程度的 profile 敏感性。
- 场景：
  - execution profile 对模型输出非常敏感时。
- 根因：
  - 模型学习到的控制链对 bridge、候选数和集中度过于敏感。
- 信号：
  - fresh formal 看起来不错，但 constrained variant 与 selected profile 表现不同步。
- 影响：
  - 若只看 selected profile，容易高估 deployable 质量。
- 修复：
  - 显式引入 `fresh formal / constrained formal` 双视角。
- 预防：
  - 下一轮新模型默认同时看两层，不再只看 fresh formal。
- 升级动作：
  - 写入 temporal state 与 governance layer。

## L006 - requested recent cutoff 不等于 effective validation end
- 事件：
  - recent protocol 旧根把 `20260410` 当成 validation end，但 `calendar_months` 实际有效验证终点是 `2026-03-31`。
- 场景：
  - independent recent protocol、calendar-month split、旧 recent root 复用。
- 根因：
  - requested cutoff 和 effective validation end 没有被显式拆开。
- 信号：
  - 协议校验时 `valid_end` 与预期 recent cutoff 对不上。
- 影响：
  - 旧 recent root 会在严格协议下失效，且容易误讲 recent 时间边界。
- 修复：
  - recent protocol 现在显式保存 `requested_recent_end_date` 和 `recent_validation_end_date`。
- 预防：
  - 以后所有 recent summary、brain 叙述和 handoff 都必须同时写这两者。
- 升级动作：
  - 写入 `recent_model_protocol.py`、`action_system.md`、`working_memory.md`、`handoff_packet.md`。

## L007 - strongest-model recent 口径修正后，主问题会整体重排
- 事件：
  - strongest-model recent root 从旧口径切到 `short_alpha_recent_model_protocol_20260412_r1` 后，recent winner 切回 `short_expert_monthly_v1`。
- 场景：
  - strongest-model refresh、recent root 修正、brain 未同步时。
- 根因：
  - 旧 replay/旧 recent 叙事残留在工作记忆里。
- 信号：
  - 文档里还在把 `baseline_current` 写成 strongest-model recent winner。
- 影响：
  - 主问题会被错误表述成“current default 如何追 `baseline_current`”。
- 修复：
  - strongest-model 三层重新写回 brain，并把 learned-control 问题单独拆出。
- 预防：
  - 每次 recent 协议重刷后，必须同步刷新 handoff / temporal / working / action。
- 升级动作：
  - 写入 brain 高优先入口文件。
