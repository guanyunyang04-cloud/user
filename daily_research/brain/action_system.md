# Daily Research 行动系统

快照日期：`2026-04-09`

## 1. 默认执行真源
- 默认执行入口：`daily_research/execution/run_trade_plan.py`
- 默认读取：`daily_research/output/active_execution_strategy.json`
- 当前 active strategy：`state_liquidity_listwise_v1_execfirst_single_mapping_candidate_active`
- 当前 active pipeline root：`daily_research/output/short_alpha_execution_single_mapping_candidate_pipeline_20260409_r2`
- 当前权重语义：`research_raw_target_weight`
- 当前上限语义：`follow_research_raw_no_global_cap`
- 当前默认成本：`3 / 7 / 10` bps

## 2. 高频命令
### 2.1 生成默认次日交易计划
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_trade_plan.py
```

### 2.2 重刷 active candidate pipeline
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_short_alpha_execution_single_mapping_candidate_pipeline.py --live-only --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --root-tag short_alpha_execution_single_mapping_candidate_pipeline_20260409_r2
```

### 2.3 重新激活 single-mapping active default
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\activate_execution_single_mapping_candidate.py --pipeline-root daily_research\output\short_alpha_execution_single_mapping_candidate_pipeline_20260409_r2 --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe"
```

### 2.4 production full-fit fresh refresh
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\update_default_candidate_production.py --source-run-dir daily_research\output\short_alpha_formal_head2head_20260409_recheck_r1\runs\state_liquidity_listwise_v1_20250318_20260331 --no-activate-strategy
```

### 2.5 同模型扩预算 strict resume
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_production_epoch_extension.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --epoch-budgets 32,64 --root-tag short_alpha_production_epoch_extension_20260409_r1 --no-activate-best
```

### 2.6 active candidate recent backtest
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_research_candidate_backtest.py --candidate-profile active_execution_strategy --start-date 20250318 --end-date 20260408
```

### 2.7 一致性检查
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\project_consistency_check.py
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\doc_guard.py check
```

## 3. 当前正式口径
- 当前 live 月未触发 `trend_up_low_vol|expand|stable -> topk3_1d_regoff`。
- 因此当前 live panel 来自 `daily_research/output/deep_alpha_short_alpha_execalign_production_default/static_fallback_daily_live_target_weight_panel.csv`。
- 当前 effective live mode 是 `static_fallback`，当前 effective profile 是 `regoff_k2_5d_ensemble_native_anchor`。
- 这个 fallback 已经与 active execution 使用同一套 raw-weight 语义，不再回退到旧 capped fallback。
- latest trade plan 会显式展示 effective profile、bridge 元信息和 `weight_generation_note`。

## 4. 当前禁止事项
- 不再用旧 capped fallback 解释 active execution。
- 不再让 active candidate backtest 默默回到 `0` 成本。
- 不再 fresh rerun 代替同模型扩预算。
- 不再让最新规则停留在口头层，不同步到 manifest、入口和 brain。
- `2026-04-09` 执行说明补充：
- 当前默认命令 `daily_research/execution/run_trade_plan.py` 生成的计划，会直接展示当前有效执行态；若 live 月未触发 `topk3_1d_regoff`，则会显式写出 `static_fallback`、当前 effective profile、bridge 元信息，以及为什么 `转权重前分数` 不要求与最终权重单调一致。
- 如需手动核对当前真实执行态，优先看三处：
- `daily_research/output/active_execution_strategy.json`
- `daily_research/output/short_alpha_execution_single_mapping_candidate_pipeline_20260409_r2/live_trigger_monitor.json`
- `daily_research/output/short_alpha_execution_single_mapping_candidate_pipeline_20260409_r2/daily_live_score_reference.json`

## 5. 2026-04-09 默认升级快照
- 当前已验证默认 static fallback profile：`regoff_k2_5d_ensemble_native_anchor`
- 当前语义裁决根：`daily_research/output/short_alpha_execution_semantic_concentration_verdict_20260409_r1`
- 如只需刷新 fallback：
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\refresh_production_static_fallback.py --static-fallback-profile regoff_k2_5d_ensemble_native_anchor
```

## 6. 月度优先默认
- 默认汇报顺序：先月度收益质量，再年化汇总。
- 默认关注最近月和弱月，不再只看跨年均值。
- 如果用户没有额外说明，后续 short_alpha 相关 verdict、repair、promotion 都应先回答“这个月度分布好不好、像不像短线模型”，再回答年化是否更高。

## 7. 2026-04-09 月度优先执行结论
- 当前执行侧短线主线答案：`regoff_k2_5d_ensemble_native_anchor`
- 在没有新的同窗月度优先证据之前，不允许再把 `trend_up_low_vol|expand|stable -> topk3_1d_regoff` 提升回默认 repair 叙事。
- 当被问到“当前 execution mainline 是什么”时，优先引用三处：
- `daily_research/output/short_alpha_execution_semantic_concentration_verdict_20260409_r1`
- `daily_research/output/short_alpha_production_execution_policy_audit_20260409_r2`
- `daily_research/output/short_alpha_targeted_weak_month_repair_regime_firstweek_combo_expand_stable_topk3_monthly_k2_review_20260409_r3`
- 下一包执行侧工作：
- 保持 `k2` static fallback live
- 停止 broad repair sweep
- 转向 `signal-to-weight` 或 `month-trigger` 设计
