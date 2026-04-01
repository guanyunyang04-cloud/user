# Daily Research 当前判断

快照日期：`2026-04-01`

## 1. 当前默认结论
- 每日默认策略：
  - `regoff_k2_10d_ensemble_native_anchor`
- 每日默认入口：
  - `daily_research/execution/run_trade_plan.py`
- 旧机器学习主线已不再是每日默认。
- 直接原因：
  - 旧机器学习主线在同窗 `2025-03-18 -> 2026-03-31` 上是 `24.34% / 10.75% / 0.580 / -14.42%`
  - `regoff_k2_realistic` 是 `44.12% / 25.75% / 1.722 / -8.55%`
- 因此：
  - 日常使用默认 `regoff_k2`
  - 旧机器学习主线仅保留为显式回退

## 2. 当前候选列表
### 2.1 执行侧
- 默认：
  - `regoff_k2_10d_ensemble_native_anchor`
- 更激进收益对照：
  - `regon_k1_10d_ensemble_native_anchor`
- 高换手无成本对照：
  - `execalign_auto_r4_topk2_1d_regoff`
  - 已在现实成本复核后降级。
- 显式回退：
  - `advanced_ml_current_code_live_anchor`

### 2.2 研究侧
- 当前研究优胜项：
  - `dynamic_graph_v1`
- 结构确认对照：
  - `dynamic_graph_no_priors`
- 风险收益比对照：
  - `dynamic_graph_topk4`
- 首轮失败挑战者：
  - `original_mamba_plain`

## 3. 当前稳定判断
- `target_weight` 直连桥是当前正确的默认桥接表达。
- 旧 `score -> weight` 不再是默认研究方向。
- `all-offset ensemble` 只有在固定 `rebalance_anchor_date` 时才有效。
- 周化收益只保留为辅助读数。
- 默认执行候选必须使用 `daily_live_*` 面板，而不是验证期 `daily_*_panel.csv`。
- 候选新鲜度必须按真实 `source_signal_date` 判断，不能用桥接后的执行日冒充“最新”。
- 当默认候选关闭市场过滤时，计划头部只展示市场状态，不再把它误写成“禁止开仓”。
- 主判据继续使用：
  - 年化收益
  - 超额年化收益
  - Sharpe
  - 最大回撤
- 自然年拆分、弱窗口拆分、同窗 H2H 现在是必备伴随视角。

## 4. 当前研究优先级
1. 做低换手、显式成本下仍能赢过 `regoff_k2_realistic` 的 execution-alignment 候选。
2. 继续把 `dynamic_graph_v1` 向 execution objective 对齐，不回头拧旧翻译器。
3. 把 `no_priors` 与 `topk4` 保留为研究对照，不升格为默认执行候选。
4. 只有在图路线边际增益放缓后，才打开 `state-conditioned MoE`。
5. RL 继续留在 `t0_project` 的执行层范围内，不进入当前默认日频 alpha 主线。

## 5. 升级门槛
- 必须有同窗正式比较。
- 必须有显式成本外部回放。
- 必须有多窗口 H2H。
- 必须使用主板范围股票池。
- 必须通过候选信号新鲜度检查。

## 6. 停止规则
- 不升级单个幸运调仓相位。
- 不升级只在无成本口径下成立的优胜项。
- 不升级“修了弱窗口却破坏强窗口”的方案。
- 除非重新正式取胜，否则不回头恢复 `score -> weight`。
- 长过程和长历史不写这里，统一写入 `episodic_memory.md`。
