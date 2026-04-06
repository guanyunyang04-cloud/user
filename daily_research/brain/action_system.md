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

### 4.13 short-alpha profit-max production refresh
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_profitmax_production_refresh.py
```

### 4.14 short-alpha production epoch extension
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_production_epoch_extension.py
```

### 4.15 short-alpha checkpoint objective compare
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_checkpoint_objective_comparison.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe"
```

## 5. 当前命令对应的最新判决
- `run_short_alpha_score_weight_repair_review.py`
  - 扩大的静态 bridge/profile 集合仍未翻掉 `regoff_k1_5d_ensemble_native_anchor`
- `run_graph_off_plain_budget_review.py`
  - `graph_off_plain` 预算补齐后仍未通过 liquid500 challenger gate
- `run_encoder_transformer_stability_review.py`
  - `encoder_transformer_v1` 上行潜力真实存在，但当前稳定性仍不足以过 gate
- `run_short_alpha_conditional_execution_policy_review.py`
  - simple regime-conditioned policy 当前不成立
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
- `daily_research/output/short_alpha_profitmax_production_refresh_20260405_r1`
- `daily_research/output/short_alpha_production_epoch_extension_20260405_r1`
- `daily_research/output/short_alpha_checkpoint_objective_comparison_20260405_r1`
- `daily_research/output/deep_alpha_monthly_landscape_review_20260405_r1`
- `daily_research/output/deep_alpha_architecture_protocol_refresh_20260406_r1`
- `daily_research/output/graph_off_plain_budget_review_20260406_r1`
- `daily_research/output/encoder_transformer_stability_review_20260406_r1`
- `daily_research/output/dynamic_graph_ablation_formal_20260404_monthly_budgetnorm_r1`
- `daily_research/output/dynamic_graph_liquid500_challenger_20260405_r1`
- `daily_research/output/dynamic_graph_no_priors_execution_policy_formal_review_20260406_r1`
