# Daily Research 当前判断

快照日期：`2026-04-12`

## 1. 角色定位
- `daily_research` 是当前工作区的正式生产研究与执行主线。
- 它负责 strongest-model、learned-control、production full-fit 和 live 默认执行的统一闭环。

## 2. 当前状态
- strongest-model 当前三层答案已经重新对齐：
  - `formal winner = short_expert_monthly_v1`
  - `recent winner = short_expert_monthly_v1`
  - `promotable winner = short_expert_monthly_v1`
- 当前 strongest-model 叙事是从 `2026-04-09` 之后持续重刷并收口到现口径的。
- strongest-model 当前 recent root：
  - `daily_research/output/short_alpha_recent_model_protocol_20260412_r1`
- requested recent cutoff：
  - `20260410`
- effective validation window：
  - `2025-04-11 -> 2026-03-31`
- 当前统一权重语义：
  - `research_raw_target_weight`
- 当前统一上限语义：
  - `follow_research_raw_no_global_cap`
- 当前 live 默认执行：
  - `short_expert_monthly_v1 + regoff_k2_5d_ensemble_native_anchor`
- 当前 learned-control 关键状态：
  - deployable winner = `short_expert_policy_v5b__k1_20d = 0.1177`
  - recent winner = `short_expert_policy_v5b = 0.1200`
  - active v5 family fresh formal best = `short_expert_policy_v5b = 0.0839`
  - historical cross-family fresh formal best = `short_expert_policy_v4b = 0.1008`

## 3. 当前优先级
- 当前最高优先级：
  - 以 `policy_v5b` 为主线缩小 fresh-formal gap，而不是重开 broad sweep。
- 冻结当前 live 默认执行，不做静默切换。
- 以 `policy_v5b` 为主线做窄迭代，重点缩小 fresh-formal gap。
- 保留 `policy_v5a / policy_v5c` 作为执行稳定性与 gross-teacher 对照。
- 继续把 `contract / project-python / strict-resume` 纪律压到剩余活跃脚本。
- 每轮实验后立即写回 handoff、temporal、lesson、working。

## 4. 当前边界
- formal 验证采用滚动窗口协议。
- recent 验证现在是 strongest-model research verdict 的必备伴随证据。
- 当前 recent 窗口按最近一年 `12` 个月定义。
- `requested_recent_end_date` 与 `effective recent validation end` 必须分开叙述。
- 默认执行写入前仍必须走 latest-data `production full-fit + highest family budget`。
- 正式训练与预训练默认起始 epoch 预算继续遵守 `32` start policy。
- broad hand-crafted repair sweep 继续冻结。
- broad backbone / Mamba / TSFM / RL 不进入当前主线。

## 5. 当前主问题
- strongest-model 层的主问题已经收口，不再是“current default 如何追 `baseline_current`”。
- 当前真正的主问题已经改写成：
  - 如何保住 `policy_v5b` 的 recent / constrained 优势，
  - 同时把它的 fresh-formal 表现从 `0.0839` 拉向当前 overall formal mainline `0.1012`。

## 6. 当前风险
- `policy_v5b` 容易被误读成已经应当替换 live 默认。
- 如果只看 constrained formal，会低估 `policy_v5b` 的 fresh-formal gap。
- 如果只看 fresh formal，又会错过 `policy_v5b` 已经建立的 deployable 优势。
- 如果不持续写回 brain，接管者仍可能沿用 `baseline_current / policy_v2b / policy_v4b` 的旧叙事。

## 7. 推荐下一步
- 下一轮只开窄版 `policy_v5` 后继分支：
  - execution-stability regularization
  - candidate-count / concentration regularization
  - 必要时保留 gross-teacher distillation
- 每个新分支都必须同时跑：
  - formal
  - constrained formal
  - recent
- 收尾固定跑：
  - `python daily_research/tools/project_consistency_check.py`
  - `python daily_research/tools/doc_guard.py check`
