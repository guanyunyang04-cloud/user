# Daily Research 规则记忆

快照日期：`2026-04-12`

## 1. 宪法级规则
- 必须 `brain-first`，接管先读身份层和交接包，再进入正文。
- `Agent 无状态，项目大脑有状态`；关键状态必须外显，不得依赖单个 agent 的隐性上下文。
- formal、recent、live、promotion 四层语义必须分开，禁止混写。
- strongest-model 与 learned-control 都必须按各自层级分开表述。
- 事实、推断、假设必须分离；未经验证的推断不得升级为长期记忆。
- live 默认执行不得在证据不足时被静默改写。
- 正式实验必须支持 `strict resume`，长实验必须前台执行。
- 默认优先追求最有效的方案，不把“最小改动”当默认正义。

## 2. 策略级规则
- 当前默认读取顺序：
  - `identity -> handoff packet -> semantic -> rule memory -> lesson memory -> temporal state -> working -> action`
- 当前主线优先级：
  - 先保住 live 默认链稳定
  - 再推进 learned-control 的可部署升级
- 默认只做窄实验，不再广扫新 backbone、广扫手工规则或无边界试错。
- 每次执行前都要过四检：
  - 目标一致性检查
  - 规则冲突检查
  - 历史教训检查
  - 依赖完整性检查
- 每次执行后必须写回：
  - 新事实
  - 新问题
  - 风险变化
  - 推荐下一动作

## 3. 协议级规则
- recent 协议必须同时保存：
  - `requested_recent_end_date`
  - `recent_start_date`
  - `recent_end_date`
  - `recent_validation_end_date`
- 不得再把 requested cutoff 当成 effective validation end。
- 正式训练、formal 回放、recent 回放与最终 summary 默认统一使用 `yolos` 环境。
- 项目默认解释器必须走 project python，不得盲信当前 shell 的 `sys.executable`。

## 4. 经验级规则
- recent frontier 很强，不等于 deployable best 很强。
- constrained best 很强，也不等于 fresh-formal 已经过关。
- 若 selected formal profile 与 constrained best profile 分裂，先拆清 drift，再讨论 promotion。
- hand-crafted 控制层一旦进入收益递减区，就应把优先级切回 learned-control。
- 不要默认从 `episodic_memory.md` 开始接管；那是证据库，不是第一入口。
- 交接不是聊天续写，而是标准状态包接力。

## 5. 当前高优先级规则
- 当前默认执行继续冻结在 `short_expert_monthly_v1 + regoff_k2_5d_ensemble_native_anchor`。
- 当前 strongest-model 三层标准答案是：
  - `formal winner = short_expert_monthly_v1`
  - `recent winner = short_expert_monthly_v1`
  - `promotable winner = short_expert_monthly_v1`
- 当前 learned-control 要分四层讲：
  - current deployable candidate = `short_expert_policy_v5b__k1_20d = 0.1177`
  - current recent winner = `short_expert_policy_v5b = 0.1200`
  - current active v5 family fresh formal best = `short_expert_policy_v5b = 0.0839`
  - historical cross-family fresh formal best = `short_expert_policy_v4b = 0.1008`
- 当前下一轮主问题是：
  - 如何保住 `policy_v5b` 的 recent / constrained 优势，同时缩小 fresh-formal gap。
