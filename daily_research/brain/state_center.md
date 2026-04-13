# Daily Research 状态中枢

快照日期：`2026-04-13`

## 1. 当前接管摘要
- `daily_research` 是当前工作区的正式生产研究与执行主线
- 当前接管不再从 `episodic_memory.md` 起步
- 当前最小完备接管入口是：
  - `identity_layer.md`
  - `state_center.md`
  - `knowledge_center.md`
  - `operations_center.md`
  - `governance_layer.md`
- `2026-04-13` 接管复核已完成：
  - `project_consistency_check.py` 与 `doc_guard.py check` 已通过
  - `active_execution_strategy.json` 仍对齐 `short_expert_monthly_v1 + regoff_k2_5d_ensemble_native_anchor`
  - `operations_center.md` 的 recent / verdict 入口已纠偏到真实脚本
- `2026-04-13` 依赖环境真源已补齐：
  - `daily_research/environment.yml` 已创建并接入守卫
  - 当前 shell 若不在该环境内，运行报错应先判定为环境偏差而不是直接判定代码回归
- `2026-04-13` 统一运行口径已升级：
  - `daily_research` 任何程序都必须在 `yolos` 环境下运行
  - 脚本默认解释器与脚本内部转调不得再回退到 `quant` 或当前 shell Python

## 2. 当前状态
- 当前统一权重语义：
  - `research_raw_target_weight`
- strongest-model 当前三层答案已经重新对齐：
  - `formal winner = short_expert_monthly_v1`
  - `recent winner = short_expert_monthly_v1`
  - `promotable winner = short_expert_monthly_v1`
- strongest-model 当前 recent protocol batch tag：
  - `short_alpha_recent_model_protocol_20260412_r1`
- strongest-model 当前 recent winner root：
  - `daily_research/output/short_alpha_recent_model_protocol_20260412_r1__short_expert_monthly_v1`
- 当前 live 默认执行：
  - `short_expert_monthly_v1 + regoff_k2_5d_ensemble_native_anchor`
- 当前 learned-control 主锚点：
  - `deployable = short_expert_policy_v5b__k1_20d = 0.1177`
  - `recent = short_expert_policy_v5b = 0.1200`
  - `active v5 fresh formal best = short_expert_policy_v5b = 0.0839`
  - `historical cross-family fresh formal best = short_expert_policy_v4b = 0.1008`

## 3. 当前主问题
- strongest-model 主线已经收口，不再是当前阻塞点
- 当前真正的主问题是：
  - `policy_v5b` 的快桥脆弱性已确认
  - `policy_v5d / policy_v5e` 首轮 successor 未能同时保住 formal / constrained / recent
  - 下一轮需要换假设，而不是继续在温和平滑路径上盲调

## 4. 当前优先级
- 保持 `policy_v5b` 作为 learned-control 主研究锚点
- 冻结 live 默认执行，不做静默切换
- 把 `policy_v5d / policy_v5e` 明确记为已证伪首轮 successor
- 下一轮只做 `1-2` 个新 successor，且必须显式区别于本轮平滑思路
- 每轮实验后立即写回中枢与证据库

## 5. 当前边界
- formal、recent、promotion、live 不能混写
- recent 验证是 strongest-model verdict 的必备伴随证据
- `requested_recent_end_date` 与 `effective validation end` 必须分开叙述
- 默认执行写入前仍必须走 latest-data `production full-fit`
- 当前 recent 窗口按最近一年 `12` 个月定义
- broad hand-crafted repair、broad backbone、Mamba、TSFM、RL 不进入当前正式主线

## 6. 当前时态
- `Past`
  - strongest-model 当前叙事已从旧 `baseline_current` 口径切回真实 winner 口径
  - `policy_v5b bridge sensitivity audit` 已完成
- `Present`
  - learned-control 当前锚点是 `policy_v5b`
  - successor 首轮 `v5d / v5e` 已证明不足
- `Future`
  - 下一轮优先排查 bridge-speed sensitivity / slow-fast consistency
  - 每个新分支都必须同时跑 `formal + constrained formal + recent`

## 7. 当前风险
- `policy_v5b` 容易被误读成已经应当替换 live 默认
- 如果只看 constrained formal，会低估 fresh-formal gap
- 如果只看 fresh formal，又会错过 deployable 优势
- 如果不持续写回中枢，新 agent 仍可能沿用旧叙事
- 如果后续环境升级只改本机、不改 `daily_research/environment.yml`，依赖真源还会再次漂移
- 如果程序入口继续跟随当前 shell Python 而不是 `yolos`，环境漂移会被误判成代码问题

## 8. 推荐下一步
- 继续以 `policy_v5b` 作为 learned-control 主锚点
- 冻结 `policy_v5d / policy_v5e`，不进入 promotion 讨论
- 下一轮 keep gate 继续使用：
  - `constrained formal >= 0.110`
  - `recent >= 0.110`
  - `fresh formal > 0.0839`
