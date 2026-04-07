# Daily Research 行动系统

快照日期：`2026-04-06`

## 1. 使用原则
- 正式研究、训练、回测、执行统一使用：
  - `C:\Users\ASUS\miniconda3\envs\yolos\python.exe`
- 默认执行的真源不是脚本默认参数，而是：
  - `daily_research/output/active_execution_strategy.json`
- 本文件默认讨论 liquid500 当前 active default。
- liquid800 / mainboard 命令只作为独立研究入口，不与 liquid500 默认执行混写。

## 2. 执行端详细使用指南

### 2.1 当前默认执行是怎么接线的
- 日常默认执行入口：
  - `daily_research/execution/run_trade_plan.py`
- 默认执行先读取：
  - `daily_research/output/active_execution_strategy.json`
- active strategy 会决定：
  - 当前默认策略名
  - 当前默认股票池
  - 当前使用哪份 live score / target-weight panel
  - 当前 panel mode
  - 当前 production root
  - 当前 execution policy label / spec

### 2.2 当前默认执行快照
- active strategy：
  - `state_liquidity_listwise_v1_execfirst_profitmax_global_winner`
- liquidity pool：
  - `liquid500`
- panel mode：
  - `raw`
- execution policy：
  - `regoff_k1_5d_ensemble_native_anchor`
- production root：
  - `daily_research/output/deep_alpha_short_alpha_execalign_production_default`
- active production run：
  - `daily_research/output/short_alpha_production_epoch_extension_20260405_r1/runs/short_alpha_production_e40`

### 2.3 每日默认执行标准流程
1. 先列出候选，确认 active default 没有意外切换
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_trade_plan.py --list-candidate-profiles
```

2. 如需确认真源，直接查看：
  - `daily_research/output/active_execution_strategy.json`

3. 生成默认次日开盘计划
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_trade_plan.py
```

4. 重点检查：
  - `daily_research/execution/output/latest_trade_plan.txt`
  - `daily_research/execution/output/<YYYYMMDD>/plan_summary.json`

### 2.4 preflight 现在会自动做什么
- 读取 active winner 对应的 `liquidity_pool_name`
- 检查对应股票池文件是否过期
- 如果股票池落后于最新完成交易日，自动刷新
- 只有自动刷新失败时才会中止执行

### 2.5 常用执行命令
#### 只列候选，不生成计划
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_trade_plan.py --list-candidate-profiles
```

#### 用默认 active strategy 生成计划
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_trade_plan.py
```

#### 显式指定现金
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_trade_plan.py --cash 200000
```

#### 显式指定持仓文件
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_trade_plan.py --positions-file daily_research\execution\current_positions.csv
```

#### 临时放宽 stale model 拦截
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_trade_plan.py --allow-stale-model
```

#### 临时关闭市场过滤
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_trade_plan.py --no-market-regime-filter
```

### 2.6 研究候选试跑，但不切默认执行
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_research_candidate_trade_plan.py --external-score-csv "<score_panel.csv>" --external-target-weight-csv "<target_weight_panel.csv>" --candidate-label "<candidate_label>"
```

### 2.7 研究候选回测复核
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_research_candidate_backtest.py --score-panel-csv "<score_panel.csv>" --target-weight-panel-csv "<target_weight_panel.csv>" --candidate-label "<candidate_label>" --output-dir "<output_dir>" --experiment-tag "<tag>"
```

## 3. 默认执行升级

### 3.1 用当前 formal winner 生成 production full-fit
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\update_default_candidate_production.py
```

### 3.2 promotion 后把新 production 写入 active strategy
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\update_default_candidate_production.py --activate-strategy --strategy-panel-mode auto
```

### 3.3 指定 formal run 做 promotion 并直接激活
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\update_default_candidate_production.py --source-run-dir "<formal_run_dir>" --strategy-name "<strategy_name>" --activate-strategy --strategy-panel-mode auto
```

### 3.4 promotion 后必须检查
- `daily_research/output/active_execution_strategy.json`
- `daily_research/output/deep_alpha_short_alpha_execalign_production_default/production_retrain_manifest.json`
- `daily_research/execution/output/latest_trade_plan.txt`

## 4. 当前高频研究入口

### 4.1 全局 deployable leaderboard
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_global_deployable_strategy_leaderboard.py
```

### 4.2 月度总判复盘
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_monthly_landscape_review.py
```

### 4.3 family epoch frontier calibration
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_family_epoch_frontier_calibration.py --list-families
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_family_epoch_frontier_calibration.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe"
```

### 4.4 short-alpha formal
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_formal_head2head.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --root-tag short_alpha_formal_head2head_20260404_monthly_budgetnorm_r1

# monthly-first multi-window formal for short_expert
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_formal_head2head.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --root-tag short_alpha_short_expert_formal_head2head_20260407_r2_monthlycheckpoint --profiles baseline_current,state_liquidity_listwise_v1,short_expert_monthly_v1 --checkpoint-selection-objective primary_monthly_robust_score

# extend short_alpha family budget to 32 when historical windows still show budget pressure
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_formal_head2head.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --root-tag short_alpha_short_expert_formal_head2head_20260407_r3_shortalpha32 --profiles baseline_current,state_liquidity_listwise_v1,short_expert_monthly_v1 --checkpoint-selection-objective primary_monthly_robust_score --family-epoch-budget-manifest daily_research/output/deep_alpha_family_epoch_budget_short_alpha32_20260407.json

# resolve remaining oldest-window budget pressure at 48 and use this as stabilized view
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_formal_head2head.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --root-tag short_alpha_short_expert_formal_head2head_20260407_r4_shortalpha48 --profiles baseline_current,state_liquidity_listwise_v1,short_expert_monthly_v1 --checkpoint-selection-objective primary_monthly_robust_score --family-epoch-budget-manifest daily_research/output/deep_alpha_family_epoch_budget_short_alpha48_20260407.json
```

### 4.5 dynamic-graph formal
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_dynamic_graph_formal_ablation_matrix.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --root-tag dynamic_graph_ablation_formal_20260404_monthly_budgetnorm_r1 --force-raw-cache-path daily_research\cache\deep_alpha\raw\6e5203c8cdec3a61.pkl
```

### 4.6 dynamic-graph liquid500 challenger
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_dynamic_graph_liquid500_challenger_head2head.py
```

### 4.7 architecture protocol refresh
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_architecture_protocol_refresh.py
```

### 4.8 graph_off_plain 预算补齐复核
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_graph_off_plain_budget_review.py
```

### 4.9 encoder_transformer_v1 稳定性复核
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_encoder_transformer_stability_review.py --epoch-budgets 4,8,12,16,24
```

### 4.10 short-alpha weak-month review
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_weak_month_review.py
```

### 4.11 short-alpha score-to-weight repair review
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_score_weight_repair_review.py
```

### 4.12 short-alpha conditional execution policy review
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_conditional_execution_policy_review.py
```

### 4.13 short-alpha targeted weak-month repair review
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_targeted_weak_month_repair_review.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --trigger-mode regime_market_state

# first validated first-week / multi-day repair branch
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_targeted_weak_month_repair_review.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --trigger-mode regime_firstweek_combo --root-tag short_alpha_targeted_weak_month_repair_regime_firstweek_combo_review_20260407_r1
```
当前推荐口径：
- 保持默认 `min-regime-support = 2`
- 不要为了让最新窗也触发而降到 `1`
- 已验证有效的 first-week trigger：
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_targeted_weak_month_repair_review.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --trigger-mode regime_firstweek_combo
```
- 可继续下钻但暂未转正的 trigger：
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_targeted_weak_month_repair_review.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --trigger-mode regime_weight_count
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_targeted_weak_month_repair_review.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --trigger-mode regime_signal_shape
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_targeted_weak_month_repair_review.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --trigger-mode regime_firstweek_weight_drift
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_targeted_weak_month_repair_review.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --trigger-mode regime_firstweek_score_followthrough
```

### 4.14 short-alpha profit-max production refresh
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_profitmax_production_refresh.py
```

### 4.15 short-alpha production epoch extension
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_production_epoch_extension.py
```

### 4.16 short-alpha checkpoint objective compare
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_checkpoint_objective_comparison.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe"
```

### 4.17 short-alpha short-horizon expert review
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_short_horizon_expert_review.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe"

# latest controlled-budget probe
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_short_horizon_expert_review.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --root-tag short_alpha_short_horizon_expert_review_20260406_r1_e4 --family-epoch-budget-manifest daily_research/output/deep_alpha_family_epoch_budget_short_alpha_e4_20260406.json

# output-head controlled-budget probe
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_short_horizon_expert_review.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --root-tag short_alpha_short_horizon_expert_scorehead_review_20260406_r1_e4 --family-epoch-budget-manifest daily_research/output/deep_alpha_family_epoch_budget_short_alpha_e4_20260406.json --profiles baseline_current,state_liquidity_listwise_v1,short_expert_scorehead_monthly_v1

# native family full-budget rerun (`short_alpha -> 24`)
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_short_horizon_expert_review.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --root-tag short_alpha_short_horizon_expert_review_20260406_r2_fullbudget --profiles baseline_current,state_liquidity_listwise_v1,short_expert_monthly_v1

# output-head native family full-budget rerun (`short_alpha -> 24`)
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_short_horizon_expert_review.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --root-tag short_alpha_short_horizon_expert_scorehead_review_20260406_r2_fullbudget --profiles baseline_current,state_liquidity_listwise_v1,short_expert_scorehead_monthly_v1

# richer feature+loss bundle review (`short_alpha -> 24/32/48`)
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_short_horizon_expert_review.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --root-tag short_alpha_short_horizon_expert_v2_review_20260407_r1_fullbudget --profiles baseline_current,state_liquidity_listwise_v1,short_expert_monthly_v1,short_expert_monthly_v2
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_short_horizon_expert_review.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --root-tag short_alpha_short_horizon_expert_v2_review_20260407_r2_shortalpha32 --profiles baseline_current,state_liquidity_listwise_v1,short_expert_monthly_v1,short_expert_monthly_v2 --family-epoch-budget-manifest daily_research/output/deep_alpha_family_epoch_budget_short_alpha32_20260407.json
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_short_horizon_expert_review.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --root-tag short_alpha_short_horizon_expert_v2_review_20260407_r3_shortalpha48 --profiles baseline_current,state_liquidity_listwise_v1,short_expert_monthly_v1,short_expert_monthly_v2 --family-epoch-budget-manifest daily_research/output/deep_alpha_family_epoch_budget_short_alpha48_20260407.json
```

## 5. 当前命令对应的最新判决
- `run_short_alpha_score_weight_repair_review.py`
  - 扩大的静态 bridge/profile 集合仍未翻掉 `regoff_k1_5d_ensemble_native_anchor`
- `run_short_alpha_targeted_weak_month_repair_review.py`
  - `regime_firstweek_combo` 已转正：`61.67% / 2.575` vs 静态 `53.29% / 2.170`
  - `delta_excess_annual = +8.38%`，`wins = 2/3`
  - 当前最有效映射：
    - `trend_up_low_vol|expand|stable -> topk3_1d_regoff`
    - `trend_down_low_vol|fade|tighten -> regon_k1_10d_ensemble_native_anchor`
  - 结论：弱月修复主线已进入 first-week / multi-day score-weight trigger 阶段，不再回到 broad execution-policy 扩搜
- `run_graph_off_plain_budget_review.py`
  - `graph_off_plain` 预算补齐后仍未通过 liquid500 challenger gate
- `run_encoder_transformer_stability_review.py`
  - `encoder_transformer_v1` 上行潜力真实存在，但当前稳定性仍不足以过 gate
- `run_short_alpha_conditional_execution_policy_review.py`
  - simple regime-conditioned policy 当前不成立
- `run_short_alpha_targeted_weak_month_repair_review.py`
  - 粗 `regime` trigger 仍不成立，`trend_vol` trigger 退回 `static_only`
  - `regime_market_state` 找到窄触发正结果：
    - `not_ready|unknown -> topk1_1d_regoff`
  - `market_state` 单独使用没有形成有效触发
  - `regime_market_state` 如降到 `min-regime-support = 1` 会重新转负，不应用来硬推最新窗
  - `regime_signal_shape` 明显失败：`45.71% / 1.895`，`0/3` 全败
  - `regime_weight_count` 更接近可用线，但仍失败：`51.83% / 2.106`，`0/3` 全败
  - 这说明 month-start 权重签名本身有信息，但还不足以单独完成修复
  - 当前仍只算 monitored repair candidate，不算 active default 升级
  - 从当前节点起，不再横向追加新的 broad execution policy review；后续只继续 `month-start / first-week trigger / score -> weight -> execution` 修复
- `run_short_alpha_short_horizon_expert_review.py`
  - latest ridge winner = `short_expert_monthly_v1`
  - `e4` probe 与 native `24` full-budget 同结论：`91.76% / 4.852`，月度正收益占比 `83.33%`
  - 相对当前主线 `state_liquidity_listwise_v1`：坏月更浅、胜率更高、monthly robust score 更高
  - 但月度中位数超额仍略低：`4.61% < 5.26%`
  - full-budget 训练诊断：`selected_epoch = 1`，`selected_in_tail = false`，`selected_at_right_boundary = false`，`still_improving = false`，`objective_aligned_budget_pressure = false`
  - 作为 latest-window 结果，它已经完成后续 multi-window formal 追证
  - `short_expert_scorehead_monthly_v1` output-head branch 在 `e4` 与 native `24` full-budget 下同样复现 `81.41% / 4.871`
  - 它只在坏月更浅 `-3.10% > -3.73%` 与 Sharpe 略高 `4.871 > 4.852` 上占优
  - 但月度正收益占比 `75.00% < 83.33%`、月度中位数超额 `3.69% < 4.61%`、monthly robust score `0.0857 < 0.1012`
  - full-budget 训练诊断同样稳定：`selected_epoch = 1`，`selected_in_tail = false`，`selected_at_right_boundary = false`，`still_improving = false`，`objective_aligned_budget_pressure = false`
  - 当前只保留为 output-head experimental branch，不替代 `short_expert_monthly_v1`，且不能再把它的落后归因于“训练次数不够”
  - `short_expert_monthly_v2` 已补齐 `24 -> 32 -> 48`：`24 = 47.58% / 3.121` 且 in-tail，`32 = 52.41% / 2.983` 仍 in-tail，`48 = 57.55% / 3.015` 且已 stable
  - fully-stabilized latest-window 视图下：月度正收益占比 `83.33%`，但月度中位数超额 `3.19% < 4.61%`，excess annual / Sharpe `57.55% / 3.015 < 91.76% / 4.852`
  - 结论：`short_expert_monthly_v2` 只保留为 monitored negative branch，不再沿同一大包 recipe 继续堆料；若要继续 model-side，改做 ablation
- `run_short_alpha_formal_head2head.py`
  - `short_expert_monthly_v1` 的 monthly-first formal 已补齐：
    - `24 epoch`：前两窗 `undertrained`
    - `32 epoch`：中间窗稳定，整体最亮眼，但最老窗仍 `undertrained`
    - `48 epoch`：最老窗稳定，但该窗明显回撤
  - fully-stabilized formal 视图（`r4_shortalpha48`）：
    - `short_expert_monthly_v1 = 41.45% / 2.411`
    - `state_liquidity_listwise_v1 = 33.78% / 1.985`
    - 月度正收益占比：`72.22% > 69.44%`
    - 最差月：`-4.06% > -6.68%`
    - 月度中位数超额仍略低：`2.742% < 2.779%`
  - 结论：它已经是 formal strong challenger，但还不是 clean promotion answer；而 `short_expert_v2` 的 latest-window budget-stable 结果也未打赢它
- `run_short_alpha_profitmax_production_refresh.py`
  - fresh profit-max production refresh 当前不成立

## 6. 执行端常见误区
- formal run 赢了，不等于默认执行已经切换。
- 研究候选试跑不应直接修改 `active_execution_strategy.json`。
- `--list-candidate-profiles` 只查看，不会改默认执行。
- 候选试跑与默认执行升级必须分开。

## 7. 执行端排错
- 看到 stale model warning：
  - 先查 active strategy 和 production manifest 的训练截止日、上线截止日
  - 再决定是临时执行还是先做 production retrain
- 计划里的信号日不新鲜：
  - 先看 active strategy 指向的 panel 是否更新
  - 再看 production root 的 live panel 是否刷新成功
- 默认池文件过期：
  - 现在 preflight 会自动刷新
  - 如仍失败，再单独运行 `daily_research/execution/update_liquid_pool.py`
- 想确认今天到底读了哪份 panel：
  - 直接看 `plan_summary.json`
- TQ 临时抖动：
  - 执行侧优先重试 `run_trade_plan.py`
  - 研究侧需要续跑时，再用 `--force-raw-cache-path`

## 8. 当前关键输出目录
- `daily_research/output/active_execution_strategy.json`
- `daily_research/output/global_deployable_strategy_leaderboard_20260406_r1`
- `daily_research/output/deep_alpha_family_epoch_budget_latest.json`
- `daily_research/output/deep_alpha_short_alpha_execalign_production_default`
- `daily_research/output/short_alpha_execution_policy_formal_review_20260405_r1`
- `daily_research/output/short_alpha_score_weight_repair_review_20260406_r1`
- `daily_research/output/short_alpha_weak_month_review_20260405_r1`
- `daily_research/output/short_alpha_conditional_execution_policy_review_20260405_r1`
- `daily_research/output/short_alpha_targeted_weak_month_repair_review_20260406_r1`
- `daily_research/output/short_alpha_targeted_weak_month_repair_regime_market_state_review_20260406_r1`
- `daily_research/output/short_alpha_targeted_weak_month_repair_regime_market_state_support1_review_20260406_r1`
- `daily_research/output/short_alpha_targeted_weak_month_repair_market_state_support1_review_20260406_r1`
- `daily_research/output/short_alpha_targeted_weak_month_repair_regime_signal_shape_review_20260406_r1`
- `daily_research/output/short_alpha_targeted_weak_month_repair_regime_weight_count_review_20260406_r1`
- `daily_research/output/short_alpha_targeted_weak_month_repair_regime_firstweek_combo_review_20260407_r1`
- `daily_research/output/short_alpha_profitmax_production_refresh_20260405_r1`
- `daily_research/output/short_alpha_production_epoch_extension_20260405_r1`
- `daily_research/output/short_alpha_checkpoint_objective_comparison_20260405_r1`
- `daily_research/output/short_alpha_short_horizon_expert_review_20260406_r1_e4`
- `daily_research/output/short_alpha_short_horizon_expert_scorehead_review_20260406_r1_e4`
- `daily_research/output/short_alpha_short_horizon_expert_review_20260406_r2_fullbudget`
- `daily_research/output/short_alpha_short_horizon_expert_scorehead_review_20260406_r2_fullbudget`
- `daily_research/output/short_alpha_short_horizon_expert_v2_review_20260407_r1_fullbudget`
- `daily_research/output/short_alpha_short_horizon_expert_v2_review_20260407_r2_shortalpha32`
- `daily_research/output/short_alpha_short_horizon_expert_v2_review_20260407_r3_shortalpha48`
- `daily_research/output/short_alpha_short_expert_formal_head2head_20260407_r2_monthlycheckpoint`
- `daily_research/output/short_alpha_short_expert_formal_head2head_20260407_r3_shortalpha32`
- `daily_research/output/short_alpha_short_expert_formal_head2head_20260407_r4_shortalpha48`
- `daily_research/output/deep_alpha_family_epoch_budget_short_alpha32_20260407.json`
- `daily_research/output/deep_alpha_family_epoch_budget_short_alpha48_20260407.json`
- `daily_research/output/deep_alpha_monthly_landscape_review_20260405_r1`
- `daily_research/output/deep_alpha_architecture_protocol_refresh_20260406_r1`
- `daily_research/output/graph_off_plain_budget_review_20260406_r1`
- `daily_research/output/encoder_transformer_stability_review_20260406_r1`
- `daily_research/output/dynamic_graph_ablation_formal_20260404_monthly_budgetnorm_r1`
- `daily_research/output/dynamic_graph_liquid500_challenger_20260405_r1`
- `daily_research/output/dynamic_graph_no_priors_execution_policy_formal_review_20260406_r1`
