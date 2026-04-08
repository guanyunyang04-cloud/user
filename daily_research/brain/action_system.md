# Daily Research 行动系统

快照日期：`2026-04-08`

## 1. 默认执行真源
- 日常执行入口：
  - `daily_research/execution/run_trade_plan.py`
- 默认读取：
  - `daily_research/output/active_execution_strategy.json`
- 当前 active default：
  - strategy: `state_liquidity_listwise_v1_execfirst_single_mapping_candidate_active`
  - universe: `liquid500`
  - panel mode: `raw`
  - execution policy: `trend_up_low_vol|expand|stable -> topk3_1d_regoff`
  - production root: `daily_research/output/short_alpha_execution_single_mapping_candidate_pipeline_20260408_r2`
  - target weight semantics: `research_raw_target_weight`
  - cap mode: `follow_research_raw_no_global_cap`
  - 解释：trade plan 按研究 raw target weight 直达执行，不再额外套通用 `25%` 上限

## 2. 每日高频命令
### 2.1 列候选，不生成计划
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_trade_plan.py --list-candidate-profiles
```

### 2.2 生成默认次日交易计划
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_trade_plan.py
```

### 2.3 显式指定现金
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_trade_plan.py --cash 200000
```

### 2.4 显式指定持仓文件
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_trade_plan.py --positions-file daily_research\execution\current_positions.csv
```

## 3. 研究高频入口
### 3.1 全局 deployable leaderboard
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_global_deployable_strategy_leaderboard.py
```

### 3.2 liquid500 short-alpha formal
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_formal_head2head.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe"
```

### 3.3 targeted weak-month repair
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_targeted_weak_month_repair_review.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe"
```

### 3.4 short-horizon expert / penalty-only review
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_short_horizon_expert_review.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe"
```

### 3.5 execution policy audit
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_execution_policy_audit.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --run-dir "<run_dir>"
```

### 3.6 promotion 到 production
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\update_default_candidate_production.py --activate-strategy --strategy-panel-mode auto
```

production-only refresh while preserving current active execution candidate:
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\update_default_candidate_production.py --no-activate-strategy
```

### 3.7 single-mapping candidate pipeline
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_short_alpha_execution_single_mapping_candidate_pipeline.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe"
```

daily live-only refresh:
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_short_alpha_execution_single_mapping_candidate_pipeline.py --live-only --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe"
```

### 3.8 activate single-mapping default
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\activate_execution_single_mapping_candidate.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe"
```

### 3.9 consistency check
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\project_consistency_check.py
```

## 4. 当前正式口径
- execution-side
  - current best repair candidate: `trend_up_low_vol|expand|stable -> topk3_1d_regoff`
  - candidate pipeline root: `daily_research/output/short_alpha_execution_single_mapping_candidate_pipeline_20260408_r2`
  - 当前状态：已经升为 active default；最新 live 月份未触发，所以当前 live panel 实际等价于旧静态默认
- model-side
  - unified final root: `daily_research/output/short_alpha_penalty_only_narrow_ablation_final_20260408_r1`
  - current answer: no narrow split beats `state_liquidity_listwise_v1` cleanly
  - best restart point: `short_expert_penalty_only_monthly_v1`
  - strongest split challenger: `short_expert_penalty_only_light_monthly_v1`
  - heavy strict-resume root: `daily_research/output/short_alpha_penalty_only_heavy_review_20260408_r2_shortalpha64_resume`

## 5. 执行 / 月更移交清单
1. 先训练
- latest model
- highest family budget
- GPU only
- default start budget `32`

2. 如果只是同模型扩预算
- 必须 strict resume continuation

3. 然后才允许做
- execution audit
- recent realistic gate
- trade-plan replay
- production promotion

## 6. 最新要求优先
- 用户最新提出的规则默认覆盖旧的本地假设、旧默认值和旧说明文案。
- 如果新要求改变了默认链路，必须先更新：
  - active manifest
  - 默认入口脚本
  - trade plan 展示字段
  - 相关 brain 文档
- 在这四处没有统一前，不视为任务完成。

## 7. 当前明确停止项
- 不再重开 broad execution-policy sweep。
- 不再把 simple regime-conditioned policy 当作默认修复路线。
- 不再继续 `short_expert_monthly_v2`、`heavy penalty` 这类已证伪分支。
- 不再允许 execution-bound 结论来自 stale / low-budget / CPU 训练。

## 8. 当前最重要的文件
- active strategy
  - `daily_research/output/active_execution_strategy.json`
- latest trade plan
  - `daily_research/execution/output/latest_trade_plan.txt`
- execution candidate pipeline
  - `daily_research/output/short_alpha_execution_single_mapping_candidate_pipeline_20260408_r2`
- model-side final ablation summary
  - `daily_research/output/short_alpha_penalty_only_narrow_ablation_final_20260408_r1`
