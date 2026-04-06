# Daily Research 行动系统

快照日期：`2026-04-06`

## 1. 使用原则
- 正式研究、训练、回测、执行统一使用：
  - `C:\Users\ASUS\miniconda3\envs\yolos\python.exe`
- 本文件保留“当前仍高频使用、可直接执行、且有明确边界”的入口。
- 历史过程、旧结论、废弃主线统一写入 `episodic_memory.md`，不在本文件堆积。
- 默认执行的真源不是某个脚本参数，而是：
  - `daily_research/output/active_execution_strategy.json`
- 本文件默认讨论 liquid500 当前 active default。
- liquid800 / mainboard 研究线命令只作为独立研究入口，不与 liquid500 默认执行混写。

## 2. 执行端详细使用指南

### 2.1 执行端当前是怎么接线的
- 日常默认执行入口：
  - `daily_research/execution/run_trade_plan.py`
- 默认执行先读取：
  - `daily_research/output/active_execution_strategy.json`
- active strategy 会告诉执行端：
  - 当前默认策略名
  - 当前默认股票池（`liquidity_pool_name`）
  - 当前默认读取哪份 live score / target-weight panel
  - 当前是否使用 `execution_aligned` 面板
  - 当前 production root 在哪里
  - 当前底层模型的 execution profile / retrain 语义，以及被研究 winner 选中的精确 execution policy label / spec
- 结论：
  - “默认执行到底跑谁”，以 `active_execution_strategy.json` 为准
  - 不是以某个 formal run 名字、也不是以脑内口述为准
### 2.2 当前默认执行快照
- 当前 active strategy：
  - `state_liquidity_listwise_v1_execfirst_profitmax_global_winner`
- 当前 liquidity pool：
  - `liquid500`
- 当前 panel mode：
  - `raw`
- 当前 execution profile：
  - `regoff_k1_5d_ensemble_native_anchor`
  - 现在应理解为“当前 active winner 的执行法”；后续 fresh formal 默认扫描 `profit_max_v1`，不是只扫旧四档 profile
- 当前 production root：
  - `daily_research/output/deep_alpha_short_alpha_execalign_production_default`
- 当前 active production run：
  - `daily_research/output/short_alpha_production_epoch_extension_20260405_r1/runs/short_alpha_production_e40`
- 当前 production recipe 最新状态：
  - `selected_epoch = 25 / 40`
  - `objective_aligned_budget_pressure = false`
  - 当前默认执行已不再处于 recipe budget pressure 状态

### 2.3 每日默认执行标准流程
1. 先确认默认候选和 active strategy 没有意外切换：
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_trade_plan.py --list-candidate-profiles
```

2. 如需确认默认真源，直接查看：
  - `daily_research/output/active_execution_strategy.json`

3. 确认持仓输入文件：
  - 默认路径：`daily_research/execution/current_positions.csv`
  - `run_trade_plan.py` 默认会把这个文件注入 `--positions-file`
  - 如果账户现金不写在文件里，可以单独传 `--cash`

4. 生成默认次日开盘执行计划：
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_trade_plan.py
```

  - preflight 现在会先读 active manifest 里的 `liquidity_pool_name`
  - 再检查对应的 `liquid500_latest.txt` / `liquid800_latest.txt` 是否过期
  - 若默认池滞后于最新完成交易日，会先自动刷新
  - 若自动刷新失败，才会明确报错并中止执行

### 2.3.1 如需重做全局最高口径
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_global_deployable_strategy_leaderboard.py
```

  - 该命令会读取各主线的 formal execution-policy review
  - 按 `global_deployable_non_capacity_adjusted_v1` 口径重排当前全局 deployable winner
  - 并把 `active_execution_strategy.json` 同步成当前第一名

5. 执行后优先检查两个输出：
  - 文本计划：`daily_research/execution/output/latest_trade_plan.txt`
  - 结构化摘要：`daily_research/execution/output/<YYYYMMDD>/plan_summary.json`

6. 重点核对 `plan_summary.json` 里的这些字段：
  - `candidate_label`
  - `signal_date`
  - `candidate_freshness`
  - `model_retrain_freshness`
  - `trade_plan_target_weight_panel_csv`
  - `source_run_dir` 或 active manifest 对应的 production root

### 2.4 常用执行命令

#### 只列出当前可用候选，不生成计划
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

### 2.5 什么时候用默认执行，什么时候不用
- 用默认执行：
  - 你要生成“当前默认生产候选”的次日开盘计划
  - 你要检查当前 active strategy 是否还能正常落地
- 不要用默认执行：
  - 你只是想试跑某个研究候选，但不想影响默认生产
  - 你只是想做候选回测，不想出次日交易单
  - 你准备切换默认策略，但还没做 production promotion

### 2.6 研究候选试跑，但不切换默认执行
当你只是想把某个研究 run 的 score panel / target-weight panel 过一遍执行端，使用研究候选入口，不要直接改 active strategy。

#### 用候选 target-weight / score 生成试跑版交易计划
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_research_candidate_trade_plan.py --external-score-csv "<score_panel.csv>" --external-target-weight-csv "<target_weight_panel.csv>" --candidate-label "<candidate_label>"
```

适用场景：
- 不改默认执行，只看研究候选如果今天上，会生成什么计划
- 检查候选在执行层是否因为桥接、持仓、过滤、软状态而被明显改写

关键可调参数：
- `--target-weight-top-k`
- `--target-weight-min-weight`
- `--target-weight-power`
- `--target-weight-full-invest`
- `--soft-state-profile`
- `--no-market-regime-filter`

### 2.7 研究候选回测复核
当你要比较某个研究候选在执行约束下的历史净收益，而不是只看今天计划，使用：

```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_research_candidate_backtest.py --score-panel-csv "<score_panel.csv>" --target-weight-panel-csv "<target_weight_panel.csv>" --candidate-label "<candidate_label>" --output-dir "<output_dir>" --experiment-tag "<tag>"
```

最重要的可控参数：
- 交易成本：
  - `--transaction-cost-bps`
  - `--slippage-bps`
  - `--sell-tax-bps`
- 调仓口径：
  - `--rebalance-freq`
  - `--rebalance-offset-mode`
  - `--rebalance-anchor-date`
- 桥接限制：
  - `--target-weight-top-k`
  - `--target-weight-power`
  - `--target-weight-full-invest`
- 风控覆盖：
  - `--soft-state-profile`
  - `--no-market-regime-filter`

适用场景：
- recent realistic replay
- 与当前默认执行做 head-to-head
- 验证某个研究赢家是否真的穿过 execution gate

### 2.8 默认执行升级：promotion 标准流程
默认执行不能直接指向 formal run 目录。正式切换必须经过 production promotion。

#### 用当前 formal winner 生成 production full-fit
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\update_default_candidate_production.py
```

  - 当前无参默认值已指向 liquid500 现役主线：
    - source formal run = `state_liquidity_listwise_v1`
    - production root = `daily_research/output/deep_alpha_short_alpha_execalign_production_default`

#### promotion 后，把新 production 策略写入 active strategy
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\update_default_candidate_production.py --activate-strategy --strategy-panel-mode auto
```

#### 指定研究 run 做 promotion 并直接激活
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\update_default_candidate_production.py --source-run-dir "<formal_run_dir>" --strategy-name "<strategy_name>" --activate-strategy --strategy-panel-mode auto
```

promotion 后必须检查：
- active manifest：
  - `daily_research/output/active_execution_strategy.json`
- production manifest：
  - `daily_research/output/deep_alpha_short_alpha_execalign_production_default/production_retrain_manifest.json`
- 默认执行真实落盘结果：
  - `daily_research/execution/output/latest_trade_plan.txt`

### 2.8.1 short-alpha production recipe epoch extension
当默认执行已切到 short-alpha，且你要继续沿同一 production recipe 扩训练预算，使用：

```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_production_epoch_extension.py
```

这条入口会从当前 production manifest 读取 active production run，按 `24 -> 32 -> 40` 做 strict-resume continuation，生成 recent replay，按月度优先 ranking 选择 best recipe，并在更优时自动同步 production root、刷新 active strategy、重跑默认 `run_trade_plan.py`。
当前 latest output：`daily_research/output/short_alpha_production_epoch_extension_20260405_r1`
当前 latest 结果：`best recipe = e40`，`selected_epoch / budget = 25 / 40`，`objective_aligned_budget_pressure = false`
### 2.8.2 short-alpha checkpoint objective formal compare
当你要正式比较 liquid500 short-alpha 主线到底该按“年化”还是“月度稳健度”选 checkpoint，使用：

```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_checkpoint_objective_comparison.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe"
```

这条入口会复用 `short_alpha_formal_head2head_20260404_monthly_budgetnorm_r1`，新跑或续跑 `short_alpha_formal_head2head_20260405_monthly_checkpoint_r1`，并输出 `daily_research/output/short_alpha_checkpoint_objective_comparison_20260405_r1`。
当前 latest 判决：更优 objective 仍是 `primary_annual_return`；`monthly_robust` 会把 candidate 和 baseline 都抬高，但 baseline 追得更多，所以当前不把它前推为 liquid500 short-alpha 主线默认值。

### 2.8.3 execution policy profit-max audit
当你怀疑“当前执行法并不是这条研究线最赚钱的执行法”时，使用：

```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_execution_policy_audit.py --run-dir "<run_dir>" --panel-scope formal --selection-objective excess_annual_return --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe"
```

当前 latest liquid500 结论：三窗 formal profit-max execution policy review 已确认 `regoff_k1_5d_ensemble_native_anchor` 优于旧 `regoff_k2_10d_ensemble_native_anchor`，因此 active strategy 已切到 `raw panel + regoff_k1_5d_ensemble_native_anchor`。

### 2.8.4 short-alpha weak-month review
当你要先搞清楚“当前主线到底是在哪些月份、哪些 regime 下掉收益”，而不是直接盲改模型或 execution policy，使用：

```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_weak_month_review.py
```

当前 latest output：`daily_research/output/short_alpha_weak_month_review_20260405_r1`
当前 latest 结论：
- `36` 个月里有 `16` 个 weak months
- 弱月主要集中在 `trend_down_low_vol` 与 `trend_up_low_vol`
- 弱月里最优 policy 相对当前 `regoff_k1_5d_ensemble_native_anchor` 仍有平均 `8.04%` lift

### 2.8.5 short-alpha conditional execution policy review
当你怀疑“与其固定一个 execution policy，不如按 regime 条件切换”时，先做 leave-window-out review：

```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_conditional_execution_policy_review.py
```

当前 latest output：`daily_research/output/short_alpha_conditional_execution_policy_review_20260405_r1`
当前 latest 结论：
- 简单的 month-start-regime conditioned policy 对静态 `regoff_k1_5d_ensemble_native_anchor` 为 `0/3` 全败
- 当前不把这种简单 conditional policy 推到 active default

### 2.8.6 short-alpha profit-max production refresh
当你已经找到更赚钱的 liquid500 execution policy，但想确认“fresh production full-fit 按这套执行语义重训后，到底值不值得替换当前 production root”时，使用：

```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_profitmax_production_refresh.py
```

这条入口会：
- 先把 fresh review retrain 落到独立 review root
- 再用同一条 execution policy 做 recent live replay
- 只有 review root 在同一 policy 下打赢当前 production root，才回写默认执行

当前 latest output：`daily_research/output/short_alpha_profitmax_production_refresh_20260405_r1`
当前 latest 结论：
- fresh review 在同一 policy 下 recent replay `0/2` named-window 全败
- 默认 production root 保持不变

### 2.9 执行端常见误区
- 误区 1：formal run 赢了，就等于默认执行已经切换
  - 错。formal 只是研究证据；默认执行真正读取的是 active strategy manifest。
- 误区 2：直接把某个 run 的 panel 路径塞进日常执行，就等于完成升级
  - 错。这样会绕过 production full-fit、live panel 刷新和 retrain 语义。
- 误区 3：`--list-candidate-profiles` 会改默认执行
  - 错。它只是查看，不会写 manifest。
- 误区 4：研究候选试跑应该改 `active_execution_strategy.json`
  - 错。候选试跑走 `run_research_candidate_trade_plan.py` / `run_research_candidate_backtest.py`。

### 2.10 执行端排错
- 看到 stale model warning：
  - 先查 active strategy 和 production manifest 的训练截止日、上线截止日
  - 再决定是允许临时执行，还是先做 production retrain
- 当前 active default 为 `short_alpha` 且内部监控仍有 budget pressure：
  - 优先考虑做同 profile、同 objective 的 epoch extension
  - 不要因为监控提示就直接回退到旧 baseline，除非 replay 复核也一起转弱
- 当前 active default 已完成 `e40` epoch extension 且 budget pressure = false：
  - 默认不需要继续机械加练；只有当后续 fresh retrain / 新 production recipe 再次出现 pressure，或你要切到月度 checkpoint objective fresh run 时，才重开 extension
- 当前 short-alpha objective compare 已完成且 annual objective 仍胜出：
  - 默认不需要把 `update_default_candidate_production.py` 改成 monthly checkpoint objective；若之后再做 monthly objective 研究，优先放到 challenger 或非主线家族
- 计划里信号日不新鲜：
  - 优先查 active strategy 指向的 panel 文件是否更新
  - 再查 production root 的 live panel 是否刷新成功
- 默认池文件过期：
  - 现在默认 preflight 会自动刷新 `daily_research/execution/universe/liquid500_latest.txt`
  - 若仍失败，再单独运行 `daily_research/execution/update_liquid_pool.py`
- 想确认今天到底读了哪份 panel：
  - 看 `plan_summary.json`
  - 不要靠终端印象判断
- TQ 临时抖动时：
  - 执行侧优先重试 `run_trade_plan.py`
  - 研究侧需要 raw cache 续跑时，再用 `--force-raw-cache-path`

## 3. 家族级训练预算校准

### 查看可校准家族
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_family_epoch_frontier_calibration.py --list-families
```

### family-default frontier
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_family_epoch_frontier_calibration.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe"
```

### late-preformal second-stage frontier
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_family_epoch_frontier_calibration.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --families baseline --calibration-window-preset late_preformal --initial-budgets 4,8,12,16 --extension-budgets 24,32,40,48,64 --root-tag deep_alpha_family_epoch_frontier_baseline_stage2_20260404_r1
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_family_epoch_frontier_calibration.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --families short_alpha --calibration-window-preset late_preformal --initial-budgets 4,8,12,16 --extension-budgets 24,32,40,48,64 --root-tag deep_alpha_family_epoch_frontier_short_alpha_stage2_20260404_r1
```

- 当前 latest manifest：
  - `daily_research/output/deep_alpha_family_epoch_budget_latest.json`
- 当前冻结预算：
  - `baseline -> 4`
  - `structure -> 12`
  - `short_alpha -> 24`
  - `dynamic_graph -> 16`

## 4. 当前正式 rich experiment

先读各目录下的：
- `summary.md` 里的 `Monthly Priority Summary`
- 单个 run 里的 `primary_research_monthly_diagnostics.json`
- candidate replay 里的 `monthly_backtest_summary.csv` / `monthly_backtest_diagnostics.json`

### 月度总判复盘入口
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_monthly_landscape_review.py
```

- 当前 latest output：
  - `daily_research/output/deep_alpha_monthly_landscape_review_20260405_r1`
- 先读：
  - `report.md`
  - `track_profile_summary.csv`
  - `pairwise_compare_summary.csv`

### fresh run 使用月度 checkpoint objective
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_deep_alpha_research.py --checkpoint-selection-objective primary_monthly_robust_score
```

- 适用场景：
  - 你要正式比较“按年化选 checkpoint”和“按月度稳健度选 checkpoint”
  - 你准备重做 fresh formal / fresh production retrain，而不是沿旧 strict-resume 链继续
- 注意：
  - 不要在已经开始的 strict-resume training chain 里中途切 `--checkpoint-selection-objective`
  - 当前 production epoch extension 之所以仍保持 `primary_annual_return`，一方面是为了保证同一 resume 链历史可比，另一方面也是因为 liquid500 short-alpha fresh formal compare 当前仍支持 annual objective 留在默认位

### architecture protocol refresh
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_architecture_protocol_refresh.py
```

- 当前 latest output：
  - `daily_research/output/deep_alpha_architecture_protocol_refresh_20260406_r1`
- 这条入口会一次性完成：
  - recent complexity/depth/encoder/graph/context/structure 全矩阵
  - recent category winner 选择
  - multi-window formal H2H
  - 月度诊断 + RankIC + budget pressure 汇总
- 当前 latest 结论：
  - `baseline_current` 仍是 formal 月度优先 rank 1
  - `structure_context_only` 是当前最强 raw 架构 challenger，但不进入默认执行升级链
  - `encoder_transformer_v1` 与 `graph_off_plain` 保留为需要继续做稳定性 / 预算复核的候选

### short-alpha formal
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_formal_head2head.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --root-tag short_alpha_formal_head2head_20260404_monthly_budgetnorm_r1
```

### dynamic-graph formal
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_dynamic_graph_formal_ablation_matrix.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --root-tag dynamic_graph_ablation_formal_20260404_monthly_budgetnorm_r1 --force-raw-cache-path daily_research\cache\deep_alpha\raw\6e5203c8cdec3a61.pkl
```

### dynamic-graph liquid500 同宇宙 challenger H2H
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_dynamic_graph_liquid500_challenger_head2head.py
```

- 当前 latest output：
  - `daily_research/output/dynamic_graph_liquid500_challenger_20260405_r1`
- 当前 latest 结论：
  - `dynamic_graph_no_priors` 已补完 liquid500 同宇宙 formal challenger
  - 但 mean excess annual / Sharpe 仍低于 `state_liquidity_listwise_v1`
  - 当前不进入 liquid500 默认执行升级链

## 5. short-alpha 最近窗升级 gate

### 生成 short-alpha recent realistic replay
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\baseline\backtest_external_score_panel.py --data-source tq --benchmark 000300.SH --start-date 20250318 --end-date 20260401 --score-panel-csv daily_research\output\short_alpha_formal_head2head_20260404_monthly_budgetnorm_r1\runs\state_liquidity_listwise_v1_20250318_20260331\execution_aligned_daily_score_panel.csv --target-weight-panel-csv daily_research\output\short_alpha_formal_head2head_20260404_monthly_budgetnorm_r1\runs\state_liquidity_listwise_v1_20250318_20260331\execution_aligned_daily_target_weight_panel.csv --candidate-label short_alpha_execalign_realistic --rebalance-freq 1d --rebalance-offset-mode single --rebalance-anchor-date 20250318 --transaction-cost-bps 3 --slippage-bps 7 --sell-tax-bps 10 --no-market-regime-filter --output-dir daily_research\output\short_alpha_formal_head2head_20260404_monthly_budgetnorm_r1 --experiment-tag recent_replays/state_liquidity_listwise_v1_20250318_20260331
```

### 与当前默认执行做 multi-window H2H
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\execution_candidate_multiwindow_h2h.py --run-a daily_research\output\short_alpha_formal_head2head_20260404_monthly_budgetnorm_r1\recent_replays\state_liquidity_listwise_v1_20250318_20260331 --label-a short_alpha_execalign_realistic --run-b daily_research\output\execution_costreview_regoff_k2_realistic_20260401_r1 --label-b regoff_k2_realistic --output-dir daily_research\output\short_alpha_formal_head2head_20260404_monthly_budgetnorm_r1\recent_h2h_short_alpha_vs_current_default
```

## 6. TQ 抖动兜底

### 直接强制复用 raw cache
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_deep_alpha_research.py --force-raw-cache-path daily_research\cache\deep_alpha\raw\6e5203c8cdec3a61.pkl
```

- 当前已验证可用的 dynamic-graph raw cache：
  - `daily_research/cache/deep_alpha/raw/6e5203c8cdec3a61.pkl`

## 7. 当前关键输出目录
- `daily_research/output/active_execution_strategy.json`
- `daily_research/output/deep_alpha_family_epoch_budget_latest.json`
- `daily_research/output/deep_alpha_short_alpha_execalign_production_default`
- `daily_research/output/short_alpha_production_promotion_eval_20260405_r1`
- `daily_research/output/short_alpha_production_epoch_extension_20260405_r1`
- `daily_research/output/short_alpha_checkpoint_objective_comparison_20260405_r1`
- `daily_research/output/short_alpha_weak_month_review_20260405_r1`
- `daily_research/output/short_alpha_conditional_execution_policy_review_20260405_r1`
- `daily_research/output/short_alpha_profitmax_production_refresh_20260405_r1`
- `daily_research/output/deep_alpha_monthly_landscape_review_20260405_r1`
- `daily_research/output/deep_alpha_monthly_focus_smoke_20260405_r1`
- `daily_research/output/deep_alpha_architecture_execalign_formal_20260404_monthly_budgetnorm_r1`
- `daily_research/output/short_alpha_formal_head2head_20260404_monthly_budgetnorm_r1`
- `daily_research/output/dynamic_graph_ablation_formal_20260404_monthly_budgetnorm_r1`
- `daily_research/output/dynamic_graph_liquid500_challenger_20260405_r1`
