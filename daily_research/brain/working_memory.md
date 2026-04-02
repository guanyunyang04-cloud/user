# Daily Research 当前判断

快照日期：`2026-04-02`

## 1. 当前默认结论
- 每日默认策略：
  - `regoff_k2_10d_ensemble_native_anchor`
  - 日常执行现已切到 `production full-fit` 版本，不再继续直接使用 `2025-03-17` 截止的 formal 冻结模型。
- 每日默认入口：
  - `daily_research/execution/run_trade_plan.py`
- 旧机器学习主线已不再是每日默认。
- 直接原因：
  - 旧机器学习主线在同窗 `2025-03-18 -> 2026-03-31` 上是 `24.34% / 10.75% / 0.580 / -14.42%`
  - `regoff_k2_realistic` 是 `44.12% / 25.75% / 1.722 / -8.55%`
- 因此：
  - 日常使用默认 `regoff_k2`
  - 旧机器学习主线仅保留为显式回退
  - formal winner 判决仍看 `deep_alpha_liquid500_dynamic_graph_bridge_20260401_formal_r1`
  - 日常 production 候选使用 `deep_alpha_liquid500_dynamic_graph_bridge_production_default`

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
- `short_alpha` 首轮容量优胜项：
  - `state_liquidity_listwise_v1`
- 结构确认对照：
  - `dynamic_graph_no_priors`
- 风险收益比对照：
  - `dynamic_graph_topk4`
- 首轮降级短线分支：
  - `short_target_v1`
  - `short_input_v1`
  - `short_combo_v1`
- 首轮失败挑战者：
  - `original_mamba_plain`

## 3. 当前稳定判断
- `target_weight` 直连桥是当前正确的默认桥接表达。
- 旧 `score -> weight` 不再是默认研究方向。
- `all-offset ensemble` 只有在固定 `rebalance_anchor_date` 时才有效。
- 周化收益只保留为辅助读数。
- 默认执行候选必须使用 `daily_live_*` 面板，而不是验证期 `daily_*_panel.csv`。
- production full-fit 的训练边界应解释为“截至上线前的全部可标注数据”，不是机械地把最后一个 live 日期直接并入监督训练。
- 当前默认 production full-fit 训练区间：
  - `2021-01-01 -> 2026-03-03`
- 当前默认 production live 刷新截止：
  - `2026-04-01`
- 候选新鲜度必须按真实 `source_signal_date` 判断，不能用桥接后的执行日冒充“最新”。
- 当默认候选关闭市场过滤时，计划头部只展示市场状态，不再把它误写成“禁止开仓”。
- 主判据继续使用：
  - 年化收益
  - 超额年化收益
  - Sharpe
  - 最大回撤
- 自然年拆分、弱窗口拆分、同窗 H2H 现在是必备伴随视角。
- `2025-03-18 -> 2026-04-01` 的 `short_alpha` 首轮 recent-formal 矩阵已经给出明确方向：
  - `state_context` 单开无效，反而明显退化
  - `state + liquidity + light ranking/listwise` 明显抬升了研究端收益与 Sharpe
  - 仅靠第一版短线目标重写与额外日线短线特征，暂时没有带来增益
- `state_liquidity_listwise_v1` 的三窗 formal head-to-head 已完成：
  - 均值超额年化 `26.16%`，高于基线 `14.61%`
  - 均值超额 Sharpe `1.199`，高于基线 `0.448`
  - 按超额年化与 Sharpe 都是 `2/3` 窗取胜
  - 但第一窗 `20230216_20240229` 明显退化，说明它不是 lucky run，也还不是无条件新前沿
- 因此当前瓶颈更像“上下文与收益排序耦合不足”，不是“先随手加更多短线标签和更多日线输入”。

## 4. 当前研究优先级
1. 把 `state_liquidity_listwise_v1` 接入 execution objective，做显式成本下的 head-to-head。
2. 继续把 `dynamic_graph_v1` 向 execution objective 对齐，不回头拧旧翻译器。
3. 把 `no_priors` 与 `topk4` 保留为研究对照，不升格为默认执行候选。
4. `short_target_v1`、`short_input_v1`、`short_combo_v1` 先降级，不作为当前默认研发主线；除非后续引入更直接的执行目标或更强的盘中/竞价信息。
5. 继续回看 `state_liquidity_listwise_v1` 的第一弱窗，必要时只做弱窗修复型小消融，而不是重开大矩阵。
6. 只有在图路线边际增益放缓后，才打开 `state-conditioned MoE`。
7. RL 继续留在 `t0_project` 的执行层范围内，不进入当前默认日频 alpha 主线。
8. 任何后续默认候选升格，都必须同时给出：
  - formal holdout winner 证据
  - production full-fit 重训版
  - 上线后的独立 live / paper 新样本

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
