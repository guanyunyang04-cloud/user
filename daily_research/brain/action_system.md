# Daily Research 行动系统

快照日期：`2026-04-11`

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

### 3.7 recent 一年根因拆解
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\recent_root_cause_breakdown.py
```

### 3.8 current default signal/cash repair verdict
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\current_default_signal_cash_repair_verdict.py
```

### 3.9 current default follow-up repair verdict
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\current_default_followup_repair_verdict.py
```

### 3.10 重刷 single-mapping 观察分支
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_short_alpha_execution_single_mapping_candidate_pipeline.py --live-only --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --root-tag short_alpha_execution_single_mapping_candidate_pipeline_20260409_r2
```

### 3.11 重新激活 single-mapping 观察分支
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\activate_execution_single_mapping_candidate.py --pipeline-root daily_research\output\short_alpha_execution_single_mapping_candidate_pipeline_20260409_r2 --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe"
```

### 3.12 一致性检查
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
- 被问到“模型能不能自己学选股和持仓转换”时，优先引用 `daily_research/output/short_alpha_policy_model_v1_design_20260409.md`，并明确当前研究分支名为 `short_expert_policy_v1`。
- 被问到“`short_expert_policy_v1` 现在到底强不强”时，优先引用 `daily_research/output/short_alpha_policy_v1_review_20260409_r1`，并同步对照 `short_alpha_short_horizon_expert_review_20260406_r2_fullbudget` 下的 `short_expert_monthly_v1` 同窗结果。
- 被问到“加深网络 / 更深 backbone 这轮有没有用”时，优先引用 `daily_research/output/short_alpha_deep_capacity_review_20260409_r1`，并明确 completed formal latest-window winner 仍是 `short_expert_monthly_v1`。
- 被问到“当前 strongest winner 在 recent 一年到底表现怎样”时，优先引用 `daily_research/output/short_alpha_strongest_model_verdict_20260409_r1`，不再引用旧 replay-based strongest recent readout。
- 被问到“为什么 current default 的月度分布不稳、问题到底出在哪”时，先引用 `daily_research/archive/output/replay_based_reference_index.md` 中的 `recent_root_cause_breakdown` 条目，再按需下钻原始根。
- 被问到“current default 下一包 signal-to-weight / cash sizing 修补谁最强”时，先引用同一索引中的 `signal_cash_repair`、`followup_repair` 与 `gross_control_sweep` 条目。
- 被问到“旧 execution audit / recent attack recheck / topk3 历史 repair 还要不要看”时，也先从同一索引进入，不再在主脑正文散落直引。
- 被问到“当前 execution mainline 是什么”时，优先引用：
  - `daily_research/output/active_execution_strategy.json`
  - `daily_research/output/deep_alpha_short_alpha_execalign_production_default/production_retrain_manifest.json`
  - `daily_research/execution/output/latest_trade_plan.txt`
  - `daily_research/output/short_alpha_execution_semantic_concentration_verdict_20260409_r1`
  - 如需补查旧 replay-based execution audit / historical repair，再转 `daily_research/archive/output/replay_based_reference_index.md`
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
- 当前 corrected recent 对照核心是 `baseline_current`；旧 replay-based companion 与 repair 入口统一见 `daily_research/archive/output/replay_based_reference_index.md`。
- 在没有新的同窗月度优先证据之前，不允许再把 `trend_up_low_vol|expand|stable -> topk3_1d_regoff` 提升回默认 repair 叙事。
- 当前没有任何 challenger 达到稳定 `30%+` 月收益门槛。
- formal attack winner：`formal_current_equal_top5_k1_bridge`
- 当前操作含义：`short_expert_monthly_v1` 已经完成 production full-fit 并接管默认执行；`formal_current_equal_top5_k1_bridge` 继续作为 research attack branch 微调，不再抢当前默认位。
- 当前 deep capacity 结论：纯加深 `patch_transformer` 没有带来 formal uplift，`short_expert_policy_v1_deep` 接近但仍未胜出；`short_expert_mamba_policy_v1` 因当前 GPU wall-clock 过慢未完成同协议评估。
- 当前 strongest winner 的 corrected recent 一年 readout：`short_expert_monthly_v1` monthly_robust `0.0778`、recent excess annual `55.11%`、positive month ratio `91.67%`、worst month `-2.94%`；同时 strongest-model 的 corrected recent winner 是 `baseline_current`，其 robust 为 `0.0982`。
- 当前 recent 根因拆解结论：月度分布不稳的第一主因更像是“顺风状态兑现不足 + cash sizing 不够状态化 + score-to-weight 转换偏弱”；股票池是上限约束，但不是 winner 与 companion 差距的第一主因。
- 当前 current-default repair 结论：第一轮 same-protocol 小修里，winner-side repair winner 是 `winner_current_target_market_state_guard_v1`；这一步先证明了“先修 cash sizing 比先修 signal-to-weight 更对”。
- 当前 current-default follow-up 结论只保留为旧 replay-based 机制参考：第二轮 winner-side repair winner 已进一步进到 `winner_current_target_market_state_guard_v2_balance`；它把 monthly_robust_score 从 `0.046` 抬到 `0.060`，但这条 repair 线不再直接代表 corrected strongest-model recent 层。
- 当前控制层迁移结论：winner 借用 companion 的 gross 会变好、companion 借用 winner 的 gross 会变差，说明 gross-control 已经是 current default gap 的有证据主因之一。
- current-default follow-up repair winner 的 current live preview 入口也已统一收口到 `daily_research/archive/output/replay_based_reference_index.md`。
- 当前 `policy_v1` 状态：`short_expert_policy_v1` 已接入训练/推理主链，并通过 validation-panel smoke；但 formal / recent 证据尚未补齐，当前仍是 research branch，不得冒充默认执行。
- 当前 `policy_v1` latest-formal 状态：单窗 review 已补齐，但仍落后于 `short_expert_monthly_v1`，所以当前只能继续作为 research branch。
- 当前 `policy_v1` deep 状态：`short_expert_policy_v1_deep` 已完成 latest-window formal 对照，但仍未超过当前 mainline；因此“可学习选股/持仓转换”方向保留，单纯加深不单独晋升。
- 2026-04-10 最新补充：
  - 被问到“current default 的 gross-control 这条线还能不能继续压”时，先引用 `daily_research/archive/output/replay_based_reference_index.md` 中的 `gross_control_sweep` 条目。
  - 被问到“`policy_v2` 到底有没有比 `policy_v1` 更强”时，先同时引用：
    - `daily_research/output/short_alpha_policy_v2_review_20260410_r1`
    - `daily_research/output/short_alpha_policy_v2_recent_eval_20260410_r1`
  - 当前 learned-control 正式口径：
    - `policy_v2` formal 还不是 winner
    - `policy_v2` corrected recent 一年明显优于 current default 与 `policy_v1`
    - `policy_v2` corrected recent robust `0.0871`，同时高于 current default `0.0778`
  - 当前 learned-control 默认动作不是 promotion，而是继续围绕 `short_expert_policy_v2` 排查 execution-alignment profile 漂移与收益弹性损失。
  - 被问到“`policy_v2` 的 formal gap 是不是只是桥太慢”时，优先引用：
    - `daily_research/output/short_alpha_policy_v2_constrained_execution_review_20260410_r1`
    - `daily_research/output/short_alpha_policy_v2_formal_loss_breakdown_20260410_r1`
  - 当前对此问题的正式答案是：
    - `policy_v2` 的 constrained formal best 仍是 `regoff_k2_20d_ensemble_native_anchor`
    - formal gap 不能再被简化成“桥太慢”
    - 下一步应优先改 learned score-to-weight 本体，而不是继续堆手工 candidate cap / gross band
  - 被问到“`policy_v3` 有没有资格接默认”时，优先引用：
    - `daily_research/output/short_alpha_policy_v3_review_20260410_r1`
    - `daily_research/output/short_alpha_policy_v3_recent_eval_20260410_r1`
  - 当前 `policy_v3` 正式口径：
    - formal profile 仍是 `regoff_k2_20d_ensemble_native_anchor`
    - formal `monthly_robust_score = 0.0786`，低于 current mainline `0.1012`、companion `0.0897`、`policy_v2 = 0.0830`
    - corrected recent `monthly_robust_score = 0.0390`，低于 current default `0.0778`，也明显低于 `policy_v2 = 0.0871`
    - 因此 `policy_v3` 当前不是默认候选，当前默认执行保持不变
  - 本轮正式训练、formal 回放、recent 回放与结论生成已统一锁定 `yolos` 环境。

### 3.13 current default gross-control sweep verdict
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\current_default_gross_control_sweep_verdict.py
```

### 3.14 policy_v2 recent eval
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\policy_v2_recent_eval.py
```

### 3.15 policy_v2 constrained formal review
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\policy_v2_constrained_execution_review.py
```

### 3.16 policy_v2 formal loss breakdown
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\policy_v2_formal_loss_breakdown.py
```

### 3.17 policy_v3 formal review
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\policy_v3_formal_review.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --execution-alignment-candidate-profiles "raw_1d,topk2_1d_regoff,regoff_k2_3d_ensemble_native_anchor,regoff_k2_5d_ensemble_native_anchor,regoff_k2_10d_ensemble_native_anchor,regoff_k2_20d_ensemble_native_anchor"
```

### 3.18 policy_v3 recent eval
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\policy_v3_recent_eval.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe"
```

### 3.19 corrected recent protocol refresh
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\refresh_strongest_model_verdict.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --recent-model-root-tag short_alpha_recent_model_protocol_20260410_r1
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\policy_v2_recent_eval.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --recent-model-root-tag short_alpha_recent_model_protocol_20260410_r1
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\policy_v3_recent_eval.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --recent-model-root-tag short_alpha_recent_model_protocol_20260410_r1
```

## 4. 当前接管口径覆盖
- 以下条目覆盖更早的 replay-recent 叙事。
- 旧 replay-based 机制参考统一入口：`daily_research/archive/output/replay_based_reference_index.md`
- strongest-model 当前标准答案：
  - `formal winner = short_expert_monthly_v1`
  - `recent winner = baseline_current`
  - `promotable winner = short_expert_monthly_v1`
- 当前 `short_expert_monthly_v1` 的 corrected recent 读数应使用 `short_alpha_strongest_model_verdict_20260409_r1/summary.json`：
  - `recent_monthly_robust_score = 0.0778`
  - `recent_excess_annual_return = 55.11%`
  - `execution_alignment_profile = regoff_k2_5d_ensemble_native_anchor`
- 当前 learned-control 的 corrected recent 标准答案：
  - `policy_v2` 是 recent winner，`monthly_robust_score = 0.0871`
  - `policy_v3` 只有 `monthly_robust_score = 0.0390`，当前不能接默认
- 本机正式训练纪律：
  - 固定 `yolos`
  - 前台执行
  - `num_workers = 0`
  - `pin_memory = false`
