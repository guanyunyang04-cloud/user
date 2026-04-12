# Daily Research 行动系统

快照日期：`2026-04-12`

## 1. 接管前先判型
- 当前默认接管入口先看：
  - `daily_research/brain/identity_layer.md`
  - `daily_research/brain/handoff_packet.md`
  - `daily_research/brain/temporal_state.md`
- `formal` 问题：先看 formal 根，不先看 production full-fit。
- `recent` 问题：先看 `short_alpha_recent_model_protocol_20260412_r1` 与对应 summary。
- `learned-control` 问题：先看 `policy_v5 family` 四个根。
- `live` 问题：先看 `active_execution_strategy.json`、production root、`latest_trade_plan.txt`。
- formal 每个窗口都必须使用该窗口起点前最新可标注数据训练当时最新模型。
- strongest-model 讨论必须同步并报 recent 验证。

## 2. 默认执行真源
- 默认执行入口：`daily_research/execution/run_trade_plan.py`
- 默认读取：`daily_research/output/active_execution_strategy.json`
- 当前 active strategy：`deep_alpha_short_alpha_execalign_production_default`
- 当前 active candidate label：`short_expert_monthly_v1__regoff_k2_5d_ensemble_native_anchor__active`
- 当前 active production root：`daily_research/output/deep_alpha_short_alpha_execalign_production_default`
- 当前权重语义：`research_raw_target_weight`
- 当前上限语义：`follow_research_raw_no_global_cap`
- 当前默认成本：`3 / 7 / 10` bps
- trade plan 与 active manifest 需要保留 `weight_generation_note`。

## 3. 高频命令
### 3.1 主脑优先接管顺序
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\brain_bootstrap.py --child daily_research --json
```

### 3.2 生成默认次日交易计划
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_trade_plan.py
```

### 3.3 strongest-model verdict 刷新
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\refresh_strongest_model_verdict.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --root-tag short_alpha_strongest_model_verdict_20260412_r1 --recent-model-root-tag short_alpha_recent_model_protocol_20260412_r1
```

### 3.4 current recent protocol 刷新
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\recent_protocol_completion_monitor.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --recent-model-root-tag short_alpha_recent_model_protocol_20260412_r1
```

### 3.5 v5 family 全流程
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\run_policy_v5_family_pipeline.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe"
```

### 3.6 v5 family formal review
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\policy_v5_family_formal_review.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe"
```

### 3.7 v5 family recent eval
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\policy_v5_family_recent_eval.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --recent-model-root-tag short_alpha_recent_model_protocol_20260412_r1
```

### 3.8 family constrained formal review
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\policy_v2_family_constrained_execution_review.py --family-formal-summary daily_research\output\short_alpha_policy_v5_family_formal_review_20260412_r1\summary.json --family-profiles short_expert_policy_v5a,short_expert_policy_v5b,short_expert_policy_v5c --output-root daily_research\output --root-tag short_alpha_policy_v5_family_constrained_execution_review_20260412_r1 --current-formal-run-dir daily_research\output\short_alpha_short_horizon_expert_review_20260406_r2_fullbudget\runs\short_expert_monthly_v1
```

### 3.9 family formal loss breakdown
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\policy_family_formal_loss_breakdown.py --formal-summary daily_research\output\short_alpha_policy_v5_family_formal_review_20260412_r1\summary.json --recent-summary daily_research\output\short_alpha_policy_v5_family_recent_eval_20260412_r1\summary.json --constrained-summary daily_research\output\short_alpha_policy_v5_family_constrained_execution_review_20260412_r1\summary.json --focus-profiles short_expert_policy_v5a,short_expert_policy_v5b,short_expert_policy_v5c --output-root daily_research\output --root-tag short_alpha_policy_v5_family_formal_loss_breakdown_20260412_r1
```

### 3.10 strongest winner 默认执行物化
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\update_default_candidate_production.py --source-run-dir daily_research\output\short_alpha_short_horizon_expert_review_20260406_r2_fullbudget\runs\short_expert_monthly_v1
```
说明：真正写入默认执行前，仍必须使用当前 highest family budget。

### 3.11 一致性检查
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\project_consistency_check.py
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\doc_guard.py check
```

## 4. 当前正式口径
- strongest-model 当前正式根：
  - `short_alpha_strongest_model_verdict_20260412_r1`
- strongest-model 当前 recent root：
  - `short_alpha_recent_model_protocol_20260412_r1`
- strongest-model 当前三层答案：
  - `formal winner = short_expert_monthly_v1`
  - `recent winner = short_expert_monthly_v1`
  - `promotable winner = short_expert_monthly_v1`
- learned-control 当前正式根：
  - `short_alpha_policy_v5_family_formal_review_20260412_r1`
  - `short_alpha_policy_v5_family_constrained_execution_review_20260412_r1`
  - `short_alpha_policy_v5_family_recent_eval_20260412_r1`
  - `short_alpha_policy_v5_family_formal_loss_breakdown_20260412_r1`

## 5. 当前禁止事项
- 不再用旧 `20260410_r1` recent root 充当 current recent 口径。
- 不再把 `baseline_current` 写成 strongest-model current recent winner。
- 不再把 `policy_v2b` 写成 current learned-control deployable 答案。
- 不再把 `policy_v4b` 写成 current learned-control recent 答案。
- 不再把 `production full-fit` 的最新训练数据混报成 formal 证据。

## 6. 当前结论
- 当前 strongest research model：`short_expert_monthly_v1`
- 当前 strongest stable base model：`state_liquidity_listwise_v1`
- 当前默认执行：`short_expert_monthly_v1__regoff_k2_5d_ensemble_native_anchor__active`
- 当前 learned-control deployable front-runner：`policy_v5b__k1_20d = 0.1177`
- 当前 learned-control recent winner：`policy_v5b = 0.1200`
- 当前 learned-control active v5 family fresh formal best：`policy_v5b = 0.0839`
- 当前操作含义：
  - live 默认执行不变
  - learned-control 主问题改成缩小 `policy_v5b` 的 fresh-formal gap
