# Daily Research 稳定语义

快照日期：`2026-04-02`

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
  - `deep_alpha dynamic_graph_v1 -> target_weight 直连桥 -> regoff_k2_10d_ensemble_native_anchor + liquid500 + next_open`
  - 日常执行使用 `production full-fit` 根目录，而不是继续直接使用 formal holdout 冻结模型。
- 每日默认入口：
  - `daily_research/execution/run_trade_plan.py`
- 每日默认输出：
  - `daily_research/execution/output/latest_trade_plan.txt`
- 默认候选源文件：
  - `deep_alpha_liquid500_dynamic_graph_bridge_production_default/daily_live_score_panel.csv`
  - `deep_alpha_liquid500_dynamic_graph_bridge_production_default/daily_live_target_weight_panel.csv`
- 默认入口会先按 `Retrain Monthly` 规则检查默认 production 候选是否已跨入新的自然月；若已跨月则先自动重训 production full-fit，否则只按已训练模型刷新默认候选 live 面板。
- formal 研究证据单独保留在：
  - `deep_alpha_liquid500_dynamic_graph_bridge_20260401_formal_r1`
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
  - `daily_research/output/deep_alpha_architecture_execalign_formal_20260403_r1/summary.md`
- 在当前 `train_eval_auto + robust_composite + realistic cost` formal 口径下：
  - `baseline_current` 与 `structure_context_only` 都会选到 `regoff_k2_10d_ensemble_native_anchor`
  - 因此 `structure_context_only` 的落后不能再归因于“桥接 profile 选错”
  - `structure_context_only` 没有通过 execution-upgrade 门槛，不是当前默认执行升级答案
  - `baseline_current + regoff_k2 execalign` 成为当前最值得继续推进的 execution-upgrade 候选
- 但在 production full-fit 与独立 live / paper 证据补齐前，这条新候选也不能静默替换默认执行
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
