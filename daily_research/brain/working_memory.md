# Daily Research 当前判断

快照日期：`2026-04-03`

## 1. 当前默认结论
- 每日默认策略：
  - `active_execution_strategy -> baseline_current_execfirst_winner`
  - formal source 已切到 `deep_alpha_architecture_execalign_formal_20260403_r2/runs/baseline_current_20250318_20260331`
  - 日常执行现已切到 `execution_aligned + production full-fit` 版本，不再继续直接使用旧 raw formal 冻结模型。
- 每日默认入口：
  - `daily_research/execution/run_trade_plan.py`
- 旧机器学习主线已不再是每日默认。
- 直接原因：
  - 旧机器学习主线在同窗 `2025-03-18 -> 2026-03-31` 上是 `24.34% / 10.75% / 0.580 / -14.42%`
  - `regoff_k2_realistic` 是 `44.12% / 25.75% / 1.722 / -8.55%`
- 因此：
  - 日常默认已从 legacy raw `regoff_k2` 升格为 execution-first winner
  - 旧机器学习主线仅保留为显式回退
  - formal winner 判决现看 `deep_alpha_architecture_execalign_formal_20260403_r2/runs/baseline_current_20250318_20260331`
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
- 当前多窗稳健结构挑战者：
  - `structure_context_only`
  - 仅限 raw holdout 研究侧；尚未通过 execution objective + 显式成本升级门槛
- 当前 execution-objective recent 升级候选：
  - `baseline_current + regoff_k2_10d_ensemble_native_anchor`
  - 已完成 execution-first formal 重跑、production full-fit promotion 与默认执行切换
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
- `deep_alpha` 架构复杂度 / 深度 / 结构 recent-formal 矩阵已完成：
  - 最近窗口 `2025-03-18 -> 2026-03-31` 的 excess annual winner 仍是 `baseline_current = 66.49% / 2.582`
  - 单纯加大容量到 `hidden_dim=160`、加深到 `4` 层、或切到 vanilla `transformer` / `mamba` 都没有超过当前基线
  - `structure_context_only` 是唯一接近基线且风险收益比更优的结构改动：`54.58% / 3.000 / -6.03%`
- `deep_alpha` 架构三窗 formal head-to-head 已完成：
  - `structure_context_only = 27.66% / 1.448`，相对基线 `14.61% / 0.448`，超额年化 `2/3` 窗取胜，Sharpe `3/3` 窗取胜
  - `graph_off_plain` 与 `depth_shallow_l1` 也提高了多窗均值，但最近窗口收益仍落后于基线
  - `capacity_large_h160` 多窗均值几乎只与基线打平，`depth_deep_l4` 仍不成立
- `deep_alpha` 架构 execution-objective 三窗 formal head-to-head 已完成：
  - 协议固定为 `train_eval_auto + robust_composite + realistic cost (3 / 7 / 10 bps)`
  - `baseline_current` 与 `structure_context_only` 三窗都选到了同一 execution bridge：`regoff_k2_10d_ensemble_native_anchor`
  - 因此这轮输赢不能再解释成“structure 只是桥没选对”
  - 三窗均值上，`baseline_current` 明显强于 `structure_context_only`
  - aligned holdout：`21.43% / 1.330` 对 `2.81% / 0.235`
  - realistic external replay：`15.15% / 0.872` 对 `2.09% / 0.205`
  - `structure_context_only` 只在 `20240301_20250317` 这一窗短暂取胜，其余窗口都落后
- `structure_context_only` 在 recent 窗口的 raw 研究优势没有穿过 execution objective：
  - raw recent：`54.58% / 3.000`
  - realistic replay recent：`12.86% / 0.921`
  - 对照 `baseline_current` realistic replay recent：`53.61% / 3.134`
- recent named-window H2H 先前给出的执行侧信号现已完成闭环验证：
  - `baseline_execalign_realistic` 相对当前默认 `regoff_k2_realistic` 在 `full_available + year + bridge + weak_window` 五个窗口全部取胜
  - 现在这条线已经补齐 execution-first formal rerun 与 production full-fit 证据，并正式替换默认执行
- 因此当前瓶颈更像“上下文与收益排序耦合不足”，不是“先随手加更多短线标签和更多日线输入”。
- 因此架构主结论也同步明确为：默认方向不是“继续堆复杂度 / 堆深度”。
 - raw 架构层面，`structure_context_only` 仍是最值得保留的低增参结构挑战者。
 - 但 execution-objective gate 已经说明：`structure_context_only` 的 raw 结构优势目前没有自然传导到执行侧。
 - 当前真正已经落地的统一默认 winner，是 `baseline_current + regoff_k2 execalign`。
 - 下一优先研究升级候选回到 `state_liquidity_listwise_v1` 的 execution-objective 对齐。

## 4. 当前研究优先级
1. 把 `baseline_current + regoff_k2 execalign` 推到 production-style full-fit / candidate 升级比较，因为它已在 recent realistic named-window H2H 中明显强于当前默认 `regoff_k2_realistic`。
2. `structure_context_only` 暂不继续升格执行候选；如需继续，只允许围绕“为什么 raw 优势经 `regoff_k2` bridge 后消失”做小诊断，不重开大矩阵。
3. 把 `state_liquidity_listwise_v1` 接入 execution objective，做显式成本下的 head-to-head。
4. 继续把 `dynamic_graph_v1` 向 execution objective 对齐，不回头拧旧翻译器。
5. 把 `graph_off_plain` 与 `depth_shallow_l1` 保留为稳健结构对照，不直接升格为默认执行候选。
6. 不再把“继续加大 `hidden_dim` / 继续加深层数 / 直接切 vanilla `transformer` 或 `mamba`”当作当前默认研发主方向。
7. 把 `no_priors` 与 `topk4` 保留为研究对照，不升格为默认执行候选。
8. `short_target_v1`、`short_input_v1`、`short_combo_v1` 先降级，不作为当前默认研发主线；除非后续引入更直接的执行目标或更强的盘中/竞价信息。
9. 继续回看 `state_liquidity_listwise_v1` 的第一弱窗，必要时只做弱窗修复型小消融，而不是重开大矩阵。
10. 只有在图路线边际增益放缓后，才打开 `state-conditioned MoE`。
11. RL 继续留在 `t0_project` 的执行层范围内，不进入当前默认日频 alpha 主线。
12. 任何后续默认候选升格，都必须同时给出：
  - formal holdout winner 证据
  - production full-fit 重训版
  - 上线后的独立 live / paper 新样本

## 5. `deep_alpha` 重训频率当前判决
- 当前正式输出目录：
  - `daily_research/output/deep_alpha_retrain_frequency_formal_20260402_r1`
- 排行榜口径：
  - 统一使用 `frequency_summary_common_window.csv`
  - 共同比较窗口为 `2025-03-18 -> 2026-03-27`
- 当前同协议结论：
  - `Retrain Monthly`：`36.71% / 1.484 / -16.75%`
  - `Retrain 63D`：`21.47% / 0.991 / -13.35%`
  - `Freeze 1Y`：`6.35% / 0.262 / -19.35%`
  - `Retrain 21D`：`3.68% / 0.208 / -13.47%`
- 因此在当前 `dynamic_graph_v1 + liquid500 + next_open` formal 口径下：
  - 模型不应再按“一次训练直接冻一年”来理解；
  - 月度重训明显优于一年冻结；
  - 但更高频的 `21D` 并没有继续变好，说明不是越频繁越优。
- 这条结论现在已经进入默认执行侧。
- 日常默认流程当前固定为：
  - 使用 `production full-fit`
  - 当最近一次 `launch_cutoff_date` 已跨入新的自然月时，按 `Retrain Monthly` 自动重训
  - 未跨月时只刷新 `daily_live_*` 面板
- 执行侧现已同步：
  - `run_trade_plan.py` 会先读取 `production_retrain_manifest.json`；
  - 若 `launch_cutoff_date` 已跨入新的自然月，则自动调用 `update_default_candidate_production.py`；
  - 完成后再把 `daily_live_*` 面板刷新到最新；
  - 同时仍保留 `21` 个交易日提醒与 `63` 个交易日拦截，除非显式 `--allow-stale-model` 放行。

## 6. 升级门槛
- 必须有同窗正式比较。
- 必须有显式成本外部回放。
- 必须有多窗口 H2H。
- 必须使用主板范围股票池。
- 必须通过候选信号新鲜度检查。

## 7. 停止规则
- 不升级单个幸运调仓相位。
- 不升级只在无成本口径下成立的优胜项。
- 不升级“修了弱窗口却破坏强窗口”的方案。
- 除非重新正式取胜，否则不回头恢复 `score -> weight`。
- 长过程和长历史不写这里，统一写入 `episodic_memory.md`。

## Execution-First Unification
- `deep_alpha` 的默认学习目标已不再是“raw holdout 看起来更强”，而是“真实执行后净收益更高”。
- 训练侧现在支持直接按 primary research backtest 选 checkpoint：
  - 默认研究目标：`execution_first`
  - 默认 checkpoint objective：`primary_annual_return`
  - 可选：`primary_excess_annual_return` / `primary_excess_sharpe`
- production promotion 已完整继承研究赢家的执行配置：
  - `update_default_candidate_production.py` 会透传 `research_objective_mode`
  - 会透传 `checkpoint_selection_objective`
  - 会透传 `execution_alignment_mode / objective / candidate_profiles / realistic cost`
- 默认执行端已经切到 manifest-driven：
  - 当前真源文件：`daily_research/output/active_execution_strategy.json`
  - `run_trade_plan.py` 默认先读 active strategy，再决定默认 profile
- 当前 active strategy 已正式上位：
  - `strategy_name = baseline_current_execfirst_winner`
  - `panel_mode = execution_aligned`
  - `execution_alignment_profile = regoff_k2_10d_ensemble_native_anchor`
  - `run_trade_plan.py` 默认直接读取这条 manifest，不再回落到 legacy raw default
- 当前 production promotion 语义也已固定：
  - 先按 `execution_first + train_eval_auto + robust_composite + 3/7/10bps` 判定 formal winner
  - 再冻结 formal winner 的 selected execution profile 做 production full-fit 重训
  - 不再允许 production full-fit 在 3 天内部监控窗上静默改写 execution profile
## Monthly Research Protocol
- `deep_alpha` 主研究协议现在已经支持并默认采用 `calendar_months` 作为时间单位。
- 当前月度默认窗口为：
  - `valid_months = 12`
  - `train_eval_window_months = 6`
  - `adaptive_task_window_months = 6`
- 月度协议不是把底层行情改成月线；底层仍然使用日频数据与 `next_open` 回测，但训练/验证切窗、近期训练评估窗口、自适应任务加权窗口，以及正式评估汇总都改为按自然月组织。
- 研究产物现在必须同时落盘月度证据：
  - `validation_rankic_monthly_summary.csv`
  - `monthly_score_panel.csv`
  - `monthly_target_weight_panel.csv`
  - `monthly_backtest_summary.csv`
- 若存在 execution-aligned 分支，还必须同时输出：
  - `execution_aligned_monthly_score_panel.csv`
  - `execution_aligned_monthly_target_weight_panel.csv`
  - `execution_aligned_monthly_backtest_summary.csv`
- 显式短窗仍保留日优先级：当命令已经明确给出 `train_end_date + valid_start_date + valid_days` 时，验证窗按显式交易日执行，而不是被月度默认窗口放大。
- 这条优先级是为了保证 production internal monitor、blockwise retrain 与历史 formal runner 不会被月度默认值误伤。
# 2026-04-04 月度 execution-first rich experiment 重跑结论

- 已完成的正式重跑：
  - `daily_research/output/deep_alpha_architecture_execalign_formal_20260403_monthly_r1`
  - `daily_research/output/short_alpha_formal_head2head_20260403_monthly_r1`
  - `daily_research/output/dynamic_graph_ablation_formal_20260403_monthly_r1`
- 当前最重要结论没有被推翻：
  - `baseline_current` 仍是统一后的 execution-first formal winner
  - 三窗 realistic replay 均值：`4.72% / 0.258`
  - `structure_context_only` 仍未通过 execution upgrade gate：`1.17% / 0.128`
- `short_alpha` 线出现了新的可跟进候选：
  - `state_liquidity_listwise_v1` 三窗 mean excess annual = `23.25%`
  - 基线 `baseline_current` = `22.71%`
  - 但 mean excess Sharpe 仍低于基线：`1.444 < 1.488`
  - 当前判定是“值得推进到 execution-objective 对齐”，不是“直接升格默认执行”
- `dynamic_graph` 线的旧结论被改写：
  - `dynamic_graph_no_priors` 三窗 mean excess annual / Sharpe = `20.98% / 1.083`
  - 明显强于 `dynamic_graph_v1` 的 `14.67% / 0.786`
  - 当前解释：在月度 execution-first 主板 formal 协议下，industry/style priors 现在更像噪声源而不是稳定增益
- 本轮还顺手修了两条研究基础设施：
  - `run_dynamic_graph_formal_ablation_matrix.py` 现在显式传 `--end-date 20260401`
  - `run_deep_alpha_research.py` / `pipeline_utils.py` 新增 `--force-raw-cache-path`，用于 TQ 波动时直接复用已落盘 raw cache
  - `baseline/data_provider.py` 对 TQ 初始化增加了独立 session 文件与 close/retry

# 2026-04-04 finetune epoch 预算充分性 formal 结论

- 输出目录：
  - `daily_research/output/deep_alpha_epoch_budget_formal_20260404_r1`
- 当前统一判决：
  - `baseline_current` 在 monthly execution-first 三窗下，当前 `8` epoch 不足
  - `16` epoch 的 mean replay excess annual / Sharpe = `8.50% / 0.494`
  - `8` epoch = `4.72% / 0.258`
  - `4` epoch = `4.95% / 0.266`
  - `12` epoch 与 `8` epoch 基本相同
- 关键暴露：
  - 真正的提升集中在窗口 `20240301_20250317`
  - selected checkpoint 从 `epoch 7` 延后到 `epoch 13`
  - execution profile 保持不变，说明增益来自训练预算而不是桥接 profile 偶然变化
  - 四档 `undertrained_count = 0/3` 且 `selected_epoch_hits_cap = 0/3`，说明当前基于 `valid_loss` 的 undertrained 诊断会漏掉 execution-first 的晚出现收益 checkpoint
- 当前建议：
  - 后续凡是重跑 `baseline_current` monthly execution-first formal，对照预算应以 `16` epoch 作为主参考
## 训练协议主线更新

- `deep_alpha` 训练主链现已支持两种继续训练协议：
  - `strict resume`
  - `warm_start continue`
- `strict resume` 会恢复：
  - `last_model_state_dict`
  - `optimizer / scheduler / scaler`
  - sampler epoch
  - 既有 `train_history`
- `warm_start` 只恢复选中模型参数，不继承旧 optimizer 路径。
- execution-first 预算压力诊断现已与 `valid_loss` 脱钩，主看：
  - `selected_epoch_ratio`
  - `selected_in_tail`
  - `selected_at_right_boundary`
  - `objective_aligned_budget_pressure`
- 家族级 epoch budget 现在必须先在 calibration windows 上冻结，再去跑正式 rich experiment。
- 当前冻结预算 manifest 真源：
  - `daily_research/output/deep_alpha_family_epoch_budget_latest.json`
- `2026-04-04` 家族级 monthly execution-first frontier 已正式落盘到：
  - `daily_research/output/deep_alpha_family_epoch_frontier_20260404_r2`
- 当前冻结预算判决更新为：
  - `baseline_current -> 12`
  - `structure_context_only -> 12`
  - `state_liquidity_listwise_v1 -> 32`
  - `dynamic_graph_no_priors -> 16`
- 当前应对这些预算的解释是：
  - `baseline` 在 `32` 右边界仍有 budget pressure，因此 `12` 只是当前 tested frontier 最优，不是“已证实训够”的终点
  - `structure` 在 `12/16` 打平且 `16` 无 budget pressure，当前可视为已基本训够
  - `short_alpha` 在 `32` 右边界仍是最优，说明该家族当前最吃预算，后续仍值得继续扩 frontier
  - `dynamic_graph` 在 `16/24` 打平且 `24` 无 budget pressure，当前可视为 `16` 已够用
