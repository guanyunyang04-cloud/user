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
  - 强制把 `direct formal / constrained formal / recent / live` 分层叙述。
- 预防：
  - 新 frontier 必须同时回答 deployable constrained formal 表现。
- 升级动作：
  - 写入规则记忆与交接包。

## L002 - 缺 constrained formal 会让主线判断失真
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
  - 补齐 `policy_v2_family_constrained_execution_review.py`。
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
  - 后续正式实验默认前台，默认 `10` 小时预算。
- 升级动作：
  - 写入 handoff rules 与 rule memory。

## L004 - hand-crafted repair 会出现明显天花板
- 事件：
  - current-default 多轮 repair 与 gross-control sweep 有改善，但始终追不上真正 recent frontier。
- 场景：
  - current default gap 修复、market-state guard、gross map 调参。
- 根因：
  - 控制层修补有效，但边际收益快速递减。
- 信号：
  - gross-only 还能涨，但 overlay 或更宽扫已经不再稳定增益。
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
  - `policy_v2c` 与 `policy_v4b` 都出现 selected formal profile 和 constrained best profile 分裂。
- 场景：
  - execution profile 对模型输出非常敏感时。
- 根因：
  - 模型学习到的控制链对 bridge、候选数和集中度过于敏感。
- 信号：
  - fresh formal 看起来不错，但 constrained variant 明显塌陷。
- 影响：
  - 若只看 selected profile，容易高估 deployable 质量。
- 修复：
  - 显式引入 `direct formal / constrained formal` 双视角。
- 预防：
  - 下一轮新模型默认同时看两层，不再只看 fresh formal。
- 升级动作：
  - 写入 temporal state 与 governance layer。
