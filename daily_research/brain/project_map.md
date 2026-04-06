# Daily Research 项目地图

快照日期：`2026-04-06`

## 1. 一句话概括
- `daily_research` 是一条面向主板 A 股、以执行后净收益为唯一主目标的研究-执行统一链路。

## 2. 项目结构
- `baseline`
  - 数据、桥接、回测、日常执行底座
- `execution`
  - 默认执行入口、production promotion、active strategy、股票池维护
- `deep_alpha`
  - 长周期模型研究、formal runner、frontier 校准、execution policy audit
- `tools`
  - 文档校验、工作区维护、辅助分析脚本
- `brain`
  - 项目认知层与读写路由
- `cache / output / archive`
  - 缓存、产物、归档热区

## 3. 决策闭环
1. formal holdout
   - 负责研究 winner 判决
2. recent realistic replay
   - 负责执行 gate
3. production full-fit
   - 负责默认执行候选
4. global deployable leaderboard
   - 负责跨 universe 的统一最高口径
5. active strategy manifest
   - 负责日常执行真源
6. run_trade_plan
   - 负责生成次日开盘计划

## 4. 当前两条主线与全局口径
- liquid500 默认执行主线 / 当前全局 deployable winner
  - 模型线：`state_liquidity_listwise_v1`
  - execution policy：`regoff_k1_5d_ensemble_native_anchor`
  - panel mode：`raw`
  - active production run：`short_alpha_production_e40`
- liquid800 / mainboard 独立研究主线 / 当前全局 rank 2
  - 当前研究 winner：`dynamic_graph_no_priors`
  - 当前状态：已进入全局 deployable leaderboard，但当前仍低于 liquid500 short-alpha 主线

## 5. 已收敛事项
- 研究与执行目标已经统一到 `execution_first`。
- 默认执行已经 manifest-driven，不再靠脚本默认参数隐式决定。
- execution policy 已进入正式研究对象，不再是固定后置壳。
- family epoch budget 已进入正式协议，不再统一手填 `--epochs`。
- 月度分析已上升为主判读顺序，不再只看 headline 年化。
- liquid500 当前 active default 已完成：
  - formal
  - recent gate
  - production promotion
  - strict-resume epoch extension
  - profit-max execution policy promotion

## 6. 当前瓶颈
- liquid500 当前主线已经稳定，但线上月度样本还少，仍需继续积累 live 统计。
- liquid500 当前主线的主要问题已收敛到 weak-month repair，而不是再做一轮通用 execution policy 改写。
- liquid500 当前主线的月度 checkpoint objective 尚未证明优于 annual objective。
- 简单的 regime-conditioned execution policy 已正式验证不优于静态 `regoff_k1_5d_ensemble_native_anchor`。
- 显式按 `regoff_k1_5d_ensemble_native_anchor` 做的 fresh production retrain review 未能打赢当前 production root。
- liquid800 / mainboard 的 `dynamic_graph_no_priors` 虽强，但已补完 liquid500 同宇宙 formal challenger，当前仍未超过 short-alpha 主线。
- 架构复杂度 / 深度 / 结构在当前协议下已正式重做：
  - `baseline_current` 仍是 formal 月度优先 rank 1
  - 真正剩余的问题是弱月、兑现链与执行映射，不是继续盲目加深/加大
- 工作区缓存与历史产物体量很大，维护成本高，需持续按归档策略治理。

## 7. 当前不再是主线的方向
- `structure_context_only`
  - 保留为 raw 架构 challenger
  - 不再占用执行升级主优先级
- “只因 recent 爆发就直接把新架构推到默认执行”
  - 已被 current-protocol formal + 月度优先判读取代
- “所有家族统一 epoch”
  - 已被 family budget manifest 取代
- “raw winner 直接上执行”
  - 已被 formal -> gate -> production -> active manifest 四层协议取代

## 8. 当前判断边界
- 旧的 liquid500 / liquid800 formal 数值不能直接混排。
- 当前允许跨 universe 比较，但必须先进入：
  - `global_deployable_non_capacity_adjusted_v1`
- 当前全局 deployable winner 仍是：
  - `state_liquidity_listwise_v1 + regoff_k1_5d_ensemble_native_anchor`
- 所有“最高”表述都必须先说明：
  - 是否已经过 global deployable leaderboard
  - 成本口径
  - 是否已包含 execution policy promotion

## 9. 当前下一阶段
1. 继续累计 liquid500 short-alpha 上线后的月度净收益样本。
2. 以 `trend_down_low_vol` 与 `trend_up_low_vol` 为主，做 short-alpha 的 targeted weak-month repair。
3. 对新的 liquid500 challenger 默认先做 budget-normalized formal、recent realistic gate 与 execution policy audit。
4. 保持 `dynamic_graph_no_priors` 在 liquid800 / mainboard 独立推进；liquid500 侧除非再补 recent gate / production 证据，否则不进入默认执行升级链。
