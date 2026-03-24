# Daily Research 研究总纲

## 1. 当前默认决策
当前默认执行主线已经明确并冻结为：

- `advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`

当前默认口径同时明确为：

- 股票池：
  - 执行端日常使用 `liquid500_latest.txt`
  - 正式研究使用历史滚动 `liquid500 / liquid800`
- 成交假设：
  - `next_open`
- 调仓语义：
  - 日频目标更新，主入口默认 `rebalance_freq=1d`
- 状态门控：
  - `regime_ma_window=50`
  - `regime_max_annual_vol=0.32`
  - `trend_up_low_vol,trend_up_high_vol`
- 模型族：
  - `lgbm`
- 默认训练窗口：
  - `ml_train_window_days=504`

## 2. 当前项目判断
- `advanced_ml` 继续承担执行职责；
- `deep_alpha` 继续承担主研究职责，但尚未晋级执行端；
- `none / v2` 保留为规则层先验，不扩成新的执行主线；
- 连续状态分数保留为诊断层，不进入默认执行软调节逻辑；
- `ma47/48` 左侧边界带、`v21_volume_contraction_015` 等高收益旁支保留在研究附录，不再作为当前执行修复候选。

## 3. 当前优先级
### 优先级 A：完成 `deep_alpha` 当前主线的正式判决
目标：

- 不再扩新分支；
- 把“预训练 + 排序 fine-tune”路线做出可回溯的最终判断。

具体动作：

1. 固定正式框架：
   - 历史滚动高流动性股票池；
   - `next_open`；
   - 多窗口 walk-forward。
2. 优先处理仍然 `undertrained` 的窗口，先解决训练预算问题，再讨论结构升级。
3. 只有在多窗口持续改善且不破坏原本强窗口时，才允许讨论接近执行端。

### 优先级 B：维持执行主线稳态
目标：

- 保持当前默认执行链路可用、可复核、可回滚。

具体动作：

1. 继续按正式研究口径复核 `advanced_ml (ma50 baseline, lgbm)`；
2. 继续使用 `latest_ml_model.json` 的验证摘要与新鲜度保护；
3. 不让研究侧的局部高收益候选静默替换默认值；
4. 若后续再做更长历史复验，优先解决“更早数据边界 + 长训练窗预热”，而不是机械把 `start-date` 往前推。

### 优先级 C：把治理规则变成日常流程
目标：

- 避免 README、计划、缓存与输出再次失控堆积。

具体动作：

1. `README.md` 只写当前状态与入口；
2. `daily_research_plan.md` 只写当前决策、优先级和停止规则；
3. `research_log.md` 作为唯一时间顺序实验记录；
4. 定期执行：
   - `python daily_research/tools/doc_guard.py check`
   - `python daily_research/tools/workspace_maintenance.py report`
   - `python daily_research/tools/workspace_maintenance.py archive`

## 4. 当前不推进或已降级的方向
- `ma47/48` 左侧边界带：
  - 当前判断为“季度特定槽位命中”，不再晋级执行端。
- `trend_up_low_vol` 状态专属 ensemble 权重微调：
  - 当前判断为季度集中驱动，不继续扫更大权重网格。
- `v21_volume_contraction_015`：
  - 保留为规则层研究候选，但不正式晋级为默认 `v2.1`。
- 连续状态软调节：
  - 保留为诊断层，不进入当前执行层默认逻辑。
- 状态专属模型研究：
  - 在默认 `lgbm` 已切换且两个真实开仓状态都改善的前提下，暂不重启。

## 5. 已明确的边界
- 早期 `score + 5d` 的结论只严格对应 stage1 基线路径；
- 当前 `advanced_ml` 主线已经正式标准化为“日频目标更新”；
- 当前默认执行主线不是：
  - `advanced_ml + liquid500 + next_open`
- 而是更精确的：
  - `advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`
- 任何准备接近执行端的新方案，都必须先过：
  - 历史滚动股票池；
  - `next_open`；
  - 多窗口 walk-forward；
  - 强窗口不被破坏。

## 6. 停止规则
以下情况一律不晋级执行端：

1. 只修复最弱窗口，却破坏原本更强窗口；
2. 只在静态最新股票池上好看，未通过历史滚动股票池验证；
3. 只在单次样本或烟测中有效，未通过多窗口 walk-forward；
4. 关键窗口仍显示 `undertrained`；
5. 收益改善主要来自季度集中或少数日期放大；
6. 结论无法在 `research_log.md` 中回溯到对应实验与产物。

## 7. 文档维护规则
- `daily_research/README.md`
  - 只保留当前状态、入口和维护命令。
- `daily_research/daily_research_plan.md`
  - 只保留当前默认决策、优先级和停止规则。
- `daily_research/research_log.md`
  - 只保留时间顺序实验记录与阶段结论。
- 根目录 `README.md`
  - 只负责工作区级治理说明。
- 研究产物归档规则：
  - 以 `daily_research/archive_policy.json` 为准。
