# Daily Research 行动系统

快照日期：`2026-04-09`

## 1. 接管前先判型
- `formal` 问题：先看 formal 根，不先看 production full-fit。
- `recent` 问题：先看最近一年 `12` 个月窗口的审计与回放，不先把它说成 formal 证据。
- `live` 问题：先看 `active_execution_strategy.json`、production root、pipeline sidecar 与 `latest_trade_plan.txt`。
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

## 3. 高频命令
### 3.1 生成默认次日交易计划
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_trade_plan.py
```

### 3.2 strongest winner 默认执行物化
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\update_default_candidate_production.py --source-run-dir daily_research\output\short_alpha_short_horizon_expert_review_20260406_r2_fullbudget\runs\short_expert_monthly_v1
```
说明：最终默认执行必须先走 latest-data `production full-fit`，并默认使用当前最高 family budget，不允许省算力。

### 3.3 strongest-model verdict
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\refresh_strongest_model_verdict.py
```

### 3.4 active default recent 一年回放
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_research_candidate_backtest.py --candidate-profile active_execution_strategy --start-date 20250410 --end-date 20260409
```

### 3.5 同模型扩预算 strict resume
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_production_epoch_extension.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --epoch-budgets 32,64 --root-tag short_alpha_production_epoch_extension_20260409_r1 --no-activate-best
```

### 3.6 30% 强月 signal-to-weight verdict
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\monthly_attack_signal_weight_verdict.py
```

### 3.7 重刷 single-mapping 观察分支
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_short_alpha_execution_single_mapping_candidate_pipeline.py --live-only --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --root-tag short_alpha_execution_single_mapping_candidate_pipeline_20260409_r2
```

### 3.8 重新激活 single-mapping 观察分支
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\activate_execution_single_mapping_candidate.py --pipeline-root daily_research\output\short_alpha_execution_single_mapping_candidate_pipeline_20260409_r2 --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe"
```

### 3.9 一致性检查
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\project_consistency_check.py
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\doc_guard.py check
```

## 4. 当前正式口径
- 当前默认交易计划来自 `daily_research/output/deep_alpha_short_alpha_execalign_production_default/execution_aligned_daily_live_target_weight_panel.csv` 与同根 score panel。
- 当前 effective live mode 是 `execution_aligned_live`，当前 effective profile 是 `regoff_k2_5d_ensemble_native_anchor`。
- 当前默认执行已经按 strongest winner 完成 latest-data `production full-fit + highest family budget` 物化。
- `static_fallback_daily_live_target_weight_panel.csv` 仍保留在 production root 里作为安全基线，但不再是当前默认 target-weight 来源。
- latest trade plan 会显式展示 effective profile、bridge 元信息和 `weight_generation_note`。

## 5. 当前引用顺序
- 被问到“当前 strongest research model / formal 最强研究模型是什么”时，优先引用 `daily_research/output/short_alpha_strongest_model_verdict_20260409_r1`。
- 被问到 strongest-model 是否完整成立时，除 strongest-model verdict 外，必须补报对应 recent 验证，不允许只报 formal。
- 被问到“当前稳定 base model 是什么”时，优先引用 `daily_research/output/short_alpha_formal_head2head_20260409_recheck_r1`。
- 被问到“当前 execution mainline 是什么”时，优先引用：
  - `daily_research/output/active_execution_strategy.json`
  - `daily_research/output/deep_alpha_short_alpha_execalign_production_default/production_retrain_manifest.json`
  - `daily_research/execution/output/latest_trade_plan.txt`
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
- 当前 strongest research model：`short_expert_monthly_v1`
- 当前 strongest stable base model：`state_liquidity_listwise_v1`
- 当前默认执行：`short_expert_monthly_v1__regoff_k2_5d_ensemble_native_anchor__active`
- 当前执行侧短线主线答案：`regoff_k2_5d_ensemble_native_anchor`
- 当前协议已经改成：研究最强模型可以直接作为执行默认；recent 验证必须同步保留，但不再额外卡一层 promotion 哲学流程。
- 但真正写入默认执行前，winner 仍必须先做 latest-data `production full-fit + highest family budget`。
- 当前 `recent` 默认不是 `2026-03-05 -> 2026-04-08` 这种短监控切片，而是截至当前评估时点的最近一年 `12` 个月窗口。
- 当前 recent 一年 companion winner：`state_liquidity_listwise_v1`
- 在没有新的同窗月度优先证据之前，不允许再把 `trend_up_low_vol|expand|stable -> topk3_1d_regoff` 提升回默认 repair 叙事。
- 当前没有任何 challenger 达到稳定 `30%+` 月收益门槛。
- formal attack winner：`formal_current_equal_top5_k1_bridge`
- 当前操作含义：`short_expert_monthly_v1` 已经完成 production full-fit 并接管默认执行；`formal_current_equal_top5_k1_bridge` 继续作为 research attack branch 微调，不再抢当前默认位。
