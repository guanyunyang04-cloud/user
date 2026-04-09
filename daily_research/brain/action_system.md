# Daily Research 行动系统

快照日期：`2026-04-09`

## 1. 接管前先判型
- `formal` 问题：先看 formal 根，不先看 production full-fit。
- `recent` 问题：先看 recent/live 审计与 recent 窗口回放，不先把它说成 formal 证据。
- `live` 问题：先看 `active_execution_strategy.json`、production root、pipeline sidecar 与 `latest_trade_plan.txt`。
- 研究模型只允许使用 formal 评估开始前一天及更早的可标注数据。
- 只有执行模型才允许使用最新可标注数据做 `production full-fit`。

## 2. 默认执行真源
- 默认执行入口：`daily_research/execution/run_trade_plan.py`
- 默认读取：`daily_research/output/active_execution_strategy.json`
- 当前 active strategy：`state_liquidity_listwise_v1_execfirst_single_mapping_candidate_active`
- 当前 active pipeline root：`daily_research/output/short_alpha_execution_single_mapping_candidate_pipeline_20260409_r2`
- 当前权重语义：`research_raw_target_weight`
- 当前上限语义：`follow_research_raw_no_global_cap`
- 当前默认成本：`3 / 7 / 10` bps

## 3. 高频命令
### 3.1 生成默认次日交易计划
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_trade_plan.py
```

### 3.2 重刷 active candidate pipeline
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_short_alpha_execution_single_mapping_candidate_pipeline.py --live-only --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --root-tag short_alpha_execution_single_mapping_candidate_pipeline_20260409_r2
```

### 3.3 重新激活 single-mapping active default
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\activate_execution_single_mapping_candidate.py --pipeline-root daily_research\output\short_alpha_execution_single_mapping_candidate_pipeline_20260409_r2 --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe"
```

### 3.4 execution-only: production full-fit fresh refresh
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\update_default_candidate_production.py --source-run-dir daily_research\output\short_alpha_formal_head2head_20260409_recheck_r1\runs\state_liquidity_listwise_v1_20250318_20260331 --no-activate-strategy
```

### 3.5 同模型扩预算 strict resume
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_production_epoch_extension.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --epoch-budgets 32,64 --root-tag short_alpha_production_epoch_extension_20260409_r1 --no-activate-best
```

### 3.6 active candidate recent backtest
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_research_candidate_backtest.py --candidate-profile active_execution_strategy --start-date 20250318 --end-date 20260408
```

### 3.7 一致性检查
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\project_consistency_check.py
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\doc_guard.py check
```

### 3.8 30% 强月 signal-to-weight verdict
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\monthly_attack_signal_weight_verdict.py
```

## 4. 当前正式口径
- 当前 live 月未触发 `trend_up_low_vol|expand|stable -> topk3_1d_regoff`。
- 因此当前 live panel 来自 `daily_research/output/deep_alpha_short_alpha_execalign_production_default/static_fallback_daily_live_target_weight_panel.csv`。
- 当前 effective live mode 是 `static_fallback`，当前 effective profile 是 `regoff_k2_5d_ensemble_native_anchor`。
- 这个 fallback 已经与 active execution 使用同一套 raw-weight 语义，不再回退到旧 capped fallback。
- latest trade plan 会显式展示 effective profile、bridge 元信息和 `weight_generation_note`。

## 5. 当前引用顺序
- 被问到“formal 最强模型是什么”时，优先引用 `daily_research/output/short_alpha_formal_head2head_20260409_recheck_r1`。
- 被问到“当前 execution mainline 是什么”时，优先引用：
  - `daily_research/output/short_alpha_execution_semantic_concentration_verdict_20260409_r1`
  - `daily_research/output/short_alpha_production_execution_policy_audit_20260409_r2`
  - `daily_research/output/short_alpha_targeted_weak_month_repair_regime_firstweek_combo_expand_stable_topk3_monthly_k2_review_20260409_r3`
- 被问到“30% 强月方向谁更猛”时，优先引用 `daily_research/output/short_alpha_monthly_attack_signal_weight_verdict_20260409_r1`。

## 6. 当前禁止事项
- 不再用旧 capped fallback 解释 active execution。
- 不再让 active candidate backtest 默默回到 `0` 成本。
- 不再 fresh rerun 代替同模型扩预算。
- 不再把 `production full-fit` 的最新训练数据混报成 formal 证据。
- 不再让最新规则停留在口头层，不同步到 manifest、入口和 brain。

## 7. 当前默认升级快照
- 当前已验证默认 static fallback profile：`regoff_k2_5d_ensemble_native_anchor`
- 当前语义裁决根：`daily_research/output/short_alpha_execution_semantic_concentration_verdict_20260409_r1`
- 如只需刷新 fallback：
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\refresh_production_static_fallback.py --static-fallback-profile regoff_k2_5d_ensemble_native_anchor
```

## 8. 当前结论
- 当前执行侧短线主线答案：`regoff_k2_5d_ensemble_native_anchor`
- 在没有新的同窗月度优先证据之前，不允许再把 `trend_up_low_vol|expand|stable -> topk3_1d_regoff` 提升回默认 repair 叙事。
- 当前没有任何 challenger 达到稳定 `30%+` 月收益门槛。
- formal attack winner：`formal_current_equal_top5_k1_bridge`
- recent/live gate winner：`recent_current_live_k2_static`
- 当前操作含义：保留 `k2` 作为 live 默认主线，把 `formal_current_equal_top5_k1_bridge` 作为 research attack branch 继续微调，不直接上线。
