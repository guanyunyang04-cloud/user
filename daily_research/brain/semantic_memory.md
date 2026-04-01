# Daily Research 稳定语义

快照日期：`2026-04-01`

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
- 日常生成计划时不允许静默重训模型。

## 3. 当前稳定默认
- 每日默认执行策略：
  - `deep_alpha dynamic_graph_v1 -> target_weight 直连桥 -> regoff_k2_10d_ensemble_native_anchor + liquid500 + next_open`
- 每日默认入口：
  - `daily_research/execution/run_trade_plan.py`
- 每日默认输出：
  - `daily_research/execution/output/latest_trade_plan.txt`
- 默认候选源文件：
  - `deep_alpha_liquid500_dynamic_graph_bridge_20260401_formal_r1/daily_live_score_panel.csv`
  - `deep_alpha_liquid500_dynamic_graph_bridge_20260401_formal_r1/daily_live_target_weight_panel.csv`
- 默认入口会在不重训的前提下，按已训练模型自动刷新默认候选 live 面板。

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
- 当前默认执行候选：
  - `regoff_k2_10d_ensemble_native_anchor`
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

## 7. 建议阅读顺序
1. `semantic_memory.md`
2. `working_memory.md`
3. `action_system.md`
4. `project_map.md`
5. `episodic_memory.md`
