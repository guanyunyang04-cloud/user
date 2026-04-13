# Daily Research 当前判断

快照日期：`2026-04-13`

## 1. 角色定位
- `daily_research` 是当前工作区的正式生产研究与执行主线。
- 它负责 strongest-model、learned-control、production full-fit 和 live 默认执行的统一闭环。

## 2. 当前状态
- strongest-model 当前三层答案已经重新对齐：
  - `formal winner = short_expert_monthly_v1`
  - `recent winner = short_expert_monthly_v1`
  - `promotable winner = short_expert_monthly_v1`
- 当前 strongest-model 叙事仍延续 `2026-04-09` 之后的默认执行接管收口。
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
- `policy_v5b bridge sensitivity audit` 已完成：
  - constrained deployable anchor 仍是 `policy_v5b__k1_20d = 0.1177`
  - `k1_3d` 在 constrained replay 下只有 `0.0248`
  - `k1_5d = 0.0834`
  - `cap6_k1_5d` 与 `cap4_g092_098_k1_5d` 对 `k1_5d` 为 no-op
- `policy_v5 successor` 首轮已完整跑完：
  - `policy_v5d = formal 0.0676 / constrained 0.0925 / recent 0.1071`
  - `policy_v5e = formal 0.0812 / constrained 0.0867 / recent 0.0934`
  - 二者都没有超过 `policy_v5b`

## 3. 当前优先级
- 当前最高优先级：
  - 保持 `policy_v5b` 作为 learned-control 主研究锚点，不误升 live。
- 冻结当前 live 默认执行，不做静默切换。
- 明确把 `policy_v5d / policy_v5e` 记为已证伪的首轮 successor，而不是继续默认复跑。
- 下一轮仍只做窄迭代，但必须换假设，不重复“温和 stability / concentration smoothing”。
- 继续把 `contract / project-python / strict-resume / 前台 10h` 纪律压到所有正式实验。
- 每轮实验后立即写回 handoff、temporal、lesson、working、action。

## 4. 当前边界
- formal 验证采用滚动窗口协议。
- recent 验证现在是 strongest-model research verdict 的必备伴随证据。
- 当前 recent 窗口按最近一年 `12` 个月定义。
- `requested_recent_end_date` 与 `effective recent validation end` 必须分开叙述。
- 默认执行写入前仍必须走 latest-data `production full-fit + highest family budget`。
- 正式训练与预训练默认起始 epoch 预算继续遵守 `32` start policy。
- broad hand-crafted repair sweep 继续冻结。
- broad backbone / Mamba / TSFM / RL 不进入当前主线。
- 不再把 external cap wrapper 当作当前主修复方向。

## 5. 当前主问题
- strongest-model 层的主问题已经收口，不再是“current default 如何追 `baseline_current`”。
- 当前真正的主问题已经改写成：
  - `policy_v5b` 的 fast-bridge fragility 已被审计确认，
  - `policy_v5d / policy_v5e` 这轮 successor 又未能同时保住 formal / constrained / recent，
  - 所以下一步要找的是不同于本轮平滑思路的新窄假设，而不是继续在同一路径上盲调。

## 6. 当前风险
- `policy_v5b` 容易被误读成已经应当替换 live 默认。
- 如果只看 constrained formal，会低估 `policy_v5b` 的 fresh-formal gap。
- 如果只看 fresh formal，又会错过 `policy_v5b` 已经建立的 deployable 优势。
- 如果继续沿 `v5d / v5e` 这类温和平滑路径追加试错，最可能重复无效路径。
- 如果把 external cap/gross wrapper 当成主修复手段，会再次落入 no-op 修补。
- 如果不持续写回 brain，接管者仍可能沿用 `baseline_current / policy_v2b / policy_v4b` 的旧叙事。

## 7. 推荐下一步
- 以 `policy_v5b` 继续作为当前 learned-control 主锚点。
- 冻结 `policy_v5d / policy_v5e`，不进入 promotion 讨论。
- 下一轮只开 `1-2` 个新 successor，但必须显式区别于本轮：
  - 优先针对 bridge-speed sensitivity / slow-fast consistency
  - 优先内生化 execution semantics，而不是再叠外部 cap
- 每个新分支都必须同时跑：
  - formal
  - constrained formal
  - recent
- 下一轮 keep gate 继续使用：
  - `constrained formal >= 0.110`
  - `recent >= 0.110`
  - `fresh formal > 0.0839`
- 收尾固定跑：
  - `python daily_research/tools/project_consistency_check.py`
  - `python daily_research/tools/doc_guard.py check`
