# Daily Research 稳定语义

快照日期：`2026-04-03`

## 1. 项目身份
- `daily_research` 维护一条可运行、可复核、可回退的 A 股日频研究与执行链路。
- 历史证据写入 `episodic_memory.md`。
- 当前判断写入 `working_memory.md`。
- 日常操作入口写入 `action_system.md`。

## 2. 固定边界
- 市场范围：仅上证 A 股与深证 A 股。
- 固定剔除：创业板、科创板、ST。
- 执行方式：盘后生成计划，次日开盘人工执行。
- 成交假设：`next_open`。
- 默认 Python 环境：`yolos`。
- 日常生成计划时不允许无条件静默重训模型；仅允许默认 production 候选按固定 `Retrain Monthly` 规则自动重训。

## 3. 当前稳定默认
- 每日默认执行策略：
  - `active_execution_strategy -> baseline_current_execfirst_winner`
  - execution profile 固定为 `regoff_k2_10d_ensemble_native_anchor`
  - 日常执行使用 `execution_aligned + production full-fit` 根目录，而不是继续直接使用 formal holdout 冻结模型。
- 每日默认入口：
  - `daily_research/execution/run_trade_plan.py`
- 每日默认输出：
  - `daily_research/execution/output/latest_trade_plan.txt`
- 默认候选源文件：
  - `deep_alpha_liquid500_dynamic_graph_bridge_production_default/execution_aligned_daily_live_score_panel.csv`
  - `deep_alpha_liquid500_dynamic_graph_bridge_production_default/execution_aligned_daily_live_target_weight_panel.csv`
- 默认入口会先按 `Retrain Monthly` 规则检查默认 production 候选是否已跨入新的自然月；若已跨月则先自动重训 production full-fit，否则只按已训练模型刷新默认候选 live 面板。
- formal 研究证据单独保留在：
  - `deep_alpha_architecture_execalign_formal_20260403_r2/runs/baseline_current_20250318_20260331`
- production full-fit 证据边界：
  - 只用于日常执行与上线前重训，不得回填为 formal holdout 证据。

## 4. 明确回退
- 仍保留显式旧主线回退：
  - `advanced_ml_current_code_live_anchor (ma50 baseline, lgbm520 v250) + liquid500 + next_open`
- 回退入口：
  - `daily_research/execution/run_trade_plan_legacy_ml.py`
- 回退维护入口：
  - `daily_research/execution/update_model_legacy_ml.py`

## 5. 当前研究格局
- 当前研究主前沿：
  - `deep_alpha dynamic_graph_v1`
- 当前多窗最强结构挑战者：
  - `structure_context_only`
- 当前最值得继续推进的 execution-upgrade 候选：
  - `baseline_current + regoff_k2 execalign`
- 当前默认执行候选：
  - `regoff_k2_10d_ensemble_native_anchor`
  - 日常计划标签：`dynamic_graph_regoff_k2_10d_ensemble_native_anchor_production_fullfit`
- 当前更激进收益对照：
  - `regon_k1_10d_ensemble_native_anchor`
- 当前高换手无成本对照：
  - `execalign_auto_r4_topk2_1d_regoff`
  - 在现实成本下不是每日默认候选。

## 6. 长期风险边界
- 旧快照时代的极高年化只保留为审计产物，不再作为当前升级目标。
- 旧 `next_open` 链路存在训练边界上的标签泄漏风险。
- 后续任何升级都不得绕过：
  - 显式成本回放
  - 同窗比较
  - 多窗口复核

## 7. `deep_alpha` 架构复杂度 / 深度 / 结构语义
- `deep_alpha` 架构复杂度 / 深度 / 结构已经有正式 recent-formal 矩阵与三窗 formal H2H，不再只靠直觉猜。
- 当前稳定结论只认：
  - `daily_research/output/deep_alpha_architecture_matrix_20260402_r1/summary.md`
  - `daily_research/output/deep_alpha_architecture_formal_head2head_20260402_r1/summary.md`
- 在当前 `liquid500 + top_bottom_bce + manual score head + next_open` formal 口径下：
  - 默认前沿仍是 `dynamic_graph_v1 / baseline_current`
  - “继续堆参数量 / 堆层数 / 切 vanilla transformer 或 mamba”不是默认升级方向
  - `structure_context_only` 是当前最可信的 raw holdout 多窗稳健结构挑战者
  - `graph_off_plain` 与 `depth_shallow_l1` 是有效对照，但还不是默认升格答案
- 因此：
  - 这一步 `structure_context_only` 的 execution-objective / 显式成本比较已经完成
  - raw 结构升级结论不能直接外推成执行升级结论

## 8. `deep_alpha` 架构 execution-objective 语义
- `structure_context_only` 虽然是 raw holdout 里的最强多窗结构挑战者，但它已经在 execution-objective + realistic cost gate 里失败。
- 当前稳定结论只认：
  - `daily_research/output/deep_alpha_architecture_execalign_formal_20260403_r2/summary.md`
- 在当前 `train_eval_auto + robust_composite + realistic cost` formal 口径下：
  - `baseline_current` 与 `structure_context_only` 都会选到 `regoff_k2_10d_ensemble_native_anchor`
  - 因此 `structure_context_only` 的落后不能再归因于“桥接 profile 选错”
  - `structure_context_only` 没有通过 execution-upgrade 门槛，不是当前默认执行升级答案
  - `baseline_current + regoff_k2 execalign` 已完成 formal 重跑、production full-fit promotion 与默认执行切换
  - 当前默认执行与 formal winner 已经统一
## 9. `deep_alpha` 重训频率语义
- `deep_alpha` 的重训频率结论已经有正式矩阵，不再只靠口头猜测。
- 当前稳定结论只认：
  - `daily_research/output/deep_alpha_retrain_frequency_formal_20260402_r1/frequency_summary_common_window.csv`
- 在当前 `dynamic_graph_v1 + liquid500 + next_open` formal 口径下：
  - `Retrain Monthly > Retrain 63D > Freeze 1Y > Retrain 21D`
- 因此：
  - 研究侧不能再把“训练一次直接用一年”当作默认优先答案；
  - 也不能把“重训越频繁越好”当成默认规律。
- 这条语义现在同时服务于 formal 研究、上线前重训节奏判断与默认执行自动重训规则。
- 生产边界当前固定为：
  - 默认执行仍使用 `production full-fit`；
  - 当最近一次 `launch_cutoff_date` 已跨入新的自然月时，`run_trade_plan.py` 会按 `Retrain Monthly` 自动重训；
  - 未跨月时只刷新 `daily_live_*` 面板；
  - 底层模型仍保留 `21` 个交易日提醒与 `63` 个交易日拦截护栏。

## 10. 建议阅读顺序
1. `semantic_memory.md`
2. `working_memory.md`
3. `action_system.md`
4. `project_map.md`
5. `episodic_memory.md`

## 11. Execution-First Stable Semantics
- `deep_alpha` 的稳定主目标定义为：
  - maximize after-cost executable net profit
  - not maximize raw holdout score in isolation
- 研究默认协议现在固定为：
  - `research_objective_mode = execution_first`
  - `execution_alignment_mode = train_eval_auto`
  - `execution_alignment_objective = robust_composite`
  - realistic cost = `3 / 7 / 10 bps`
- production promotion 的稳定语义现在固定为：
  - 必须继承研究赢家的 execution-alignment 配置，而不是只复制模型结构参数
  - promotion 完成后必须同时刷新 `active_execution_strategy.json`
- 默认执行的稳定语义现在固定为：
  - 不再依赖硬编码默认 profile
  - 先读 `daily_research/output/active_execution_strategy.json`
  - active manifest 才是默认执行候选的单一真源
- 当前 active manifest 已是正式真源：
  - `strategy_name = baseline_current_execfirst_winner`
  - `source_formal_run_dir = deep_alpha_architecture_execalign_formal_20260403_r2/runs/baseline_current_20250318_20260331`
  - `panel_mode = execution_aligned`
  - 默认执行与 formal winner 现在共用同一 execution-first 语义
## Monthly Time Semantics
- `deep_alpha` 的稳定时间协议新增 `calendar_months`，并作为当前主研究默认单位。
- 该协议的稳定含义是：
  - 训练/验证切分按自然月组织；
  - `train_eval` 近期窗口按自然月组织；
  - adaptive task weighting 的近期窗口按自然月组织；
  - 正式研究评估必须提供月度回测与月度 RankIC 汇总。
- 月度协议仍然建立在日频行情、日频样本与 `next_open` 回测之上；变化的是时间切窗与评估聚合口径，而不是把研究降采样成月线模型。
- 当调用方显式给出 `valid_start_date` 且同时给出 `valid_days` 时，系统稳定遵循“显式短窗优先”，优先按交易日截断验证窗。
- 这条显式短窗优先级是稳定语义，不是临时 workaround；它用于保护 production internal monitor、blockwise retrain 与旧 formal runner 的可复现性。
# 2026-04-04 稳定语义补充

- `deep_alpha` 当前稳定默认 winner 仍然是：
  - `baseline_current_execfirst_winner`
  - 证据目录：`daily_research/output/deep_alpha_architecture_execalign_formal_20260403_monthly_r1`
- `structure_context_only` 的稳定语义更新为：
  - raw 结构挑战者仍成立
  - 但在 monthly execution-first + realistic replay gate 下仍不成立
  - 不能直接作为默认执行升级答案
- `state_liquidity_listwise_v1` 的稳定语义更新为：
  - 已通过 monthly execution-first formal raw head-to-head
  - 当前是 `short_alpha` 线最值得继续推进到 execution-objective 对齐的候选
  - 还不是默认执行 winner
- `dynamic_graph_v1` 的稳定语义需要降级为：
  - 不再默认视为 dynamic graph 线最优结构
  - 在 `daily_research/output/dynamic_graph_ablation_formal_20260403_monthly_r1/summary.md` 下，`dynamic_graph_no_priors` 已成为新的 formal winner
- 长矩阵复跑的稳定韧性口径新增：
  - 遇到 TQ 抖动时，正式研究允许通过 `--force-raw-cache-path` 直接复用同协议 raw cache
  - 这属于研究韧性增强，不改变 formal 指标定义

## Finetune Epoch 预算稳定语义

- `baseline_current` 在 monthly execution-first 三窗下，`8` epoch 不是当前稳定充分预算。
- 当前已验证预算集合 `4 / 8 / 12 / 16` 中，`16` epoch 是最优 replay 预算：
  - mean replay excess annual / Sharpe = `8.50% / 0.494`
  - 对照 `8` epoch = `4.72% / 0.258`
- `12` epoch 与 `8` epoch 基本一致，说明这条线不是“线性多训一点就持续变好”，而是存在晚出现的有效 checkpoint。
- 当前 `training_diagnostics.status=stable` 只代表 `valid_loss` 未显示继续改善，不等于 execution-first 目标已经训够。
- 因此，对 `baseline_current` 的 monthly execution-first formal 复跑，应把 `16` epoch 视为当前主参考预算，直到新的正式矩阵推翻它。
## Finetune Resume 与 Budget Frontier 稳定语义

- `deep_alpha` 当前稳定支持：
  - `resume_mode = strict`
  - `resume_mode = warm_start`
- `strict` 的稳定语义是：
  - 继续同一条训练链
  - 恢复 last epoch 权重与 optimizer/scheduler/scaler 状态
  - 恢复 sampler epoch 与历史训练轨迹
- `warm_start` 的稳定语义是：
  - 只加载选中模型权重
  - 把继续训练当作新的优化路径重新开始
- execution-first 预算是否充足，当前稳定判据不再以 `valid_loss` 单独决定，而是以 objective-aligned budget pressure 为准。
- objective-aligned budget pressure 当前稳定包含：
  - `selected_in_tail`
  - `selected_at_right_boundary`
  - `still_improving` under current checkpoint objective
- rich experiment 的稳定预算来源不再是脚本内固定 `8`，而是：
  - 先运行 `run_family_epoch_frontier_calibration.py`
  - 冻结到 `daily_research/output/deep_alpha_family_epoch_budget_latest.json`
  - 再由 formal runner 默认读取这份 manifest
- 当前稳定冻结预算语义为：
  - `baseline -> 12`
  - `structure -> 12`
  - `short_alpha -> 32`
  - `dynamic_graph -> 16`
- `dynamic_graph` 的预算校准窗口允许晚于其他家族，这是稳定规则，不是临时例外；原因是该家族当前固定使用：
  - `start-date = 20220101`
  - `lookback_window = 120`
  - `rolling_liquidity_pool = liquid800`
  - 因而最早有效 sample date 天然晚于 `baseline / structure / short_alpha`
