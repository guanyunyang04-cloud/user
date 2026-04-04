# Daily Research 行动系统

快照日期：`2026-04-03`

## 1. 用途
- 本文件只保留规范化的日常操作入口。
- 稳定事实写入 `semantic_memory.md`。
- 当前判断写入 `working_memory.md`。
- 长过程与实验细节写入 `episodic_memory.md`。

## 2. 固定操作边界
- 正式研究与执行脚本统一使用 `yolos`。
- 市场范围固定为主板范围：
  - 上证 A 股 + 深证 A 股
  - 剔除创业板、科创板、ST
- 执行方式固定为：
  - 盘后生成计划
  - 次日开盘人工执行
- 成交假设固定为：
  - `next_open`
- 日常默认流程不允许无条件静默重训模型；仅允许默认 production 候选按固定 `Retrain Monthly` 规则自动重训。
- 外部候选信号新鲜度保护必须保持开启。

## 3. 每日默认流程
### 第 1 步：刷新高流动性股票池
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\update_liquid_pool.py --start-date 20240101
```

### 第 2 步：更新账户快照
- 维护文件：
  - `daily_research/execution/current_positions.csv`

### 第 3 步：生成每日默认交易计划
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_trade_plan.py
```

- 默认候选配置：
  - `active_execution_strategy -> baseline_current_execfirst_winner`
- 该入口会先检查默认候选底层 `production full-fit` 模型是否已跨入新的自然月；若已跨月，会先按 `Retrain Monthly` 自动运行 `update_default_candidate_production.py`，随后再检查 `daily_live_*` 面板是否落后于最新完成交易日并自动刷新。
- 默认日常计划读取：
  - `deep_alpha_liquid500_dynamic_graph_bridge_production_default/execution_aligned_daily_live_score_panel.csv`
  - `deep_alpha_liquid500_dynamic_graph_bridge_production_default/execution_aligned_daily_live_target_weight_panel.csv`
- formal 研究证据保留在：
  - `deep_alpha_architecture_execalign_formal_20260403_r2/runs/baseline_current_20250318_20260331`
- 默认输出：
  - `daily_research/execution/output/latest_trade_plan.txt`

## 4. 模型重训规则
- 每日默认流程不是“无条件每天重训”，而是“按固定月度规则自动重训”。
- `run_trade_plan.py` 当前会先读取 `production_retrain_manifest.json`：
  - 若最近一次 `launch_cutoff_date` 已跨入新的自然月，则先自动运行 `daily_research/execution/update_default_candidate_production.py`
  - 若尚未跨月，则只基于现有已训练模型刷新 `daily_live_*` 面板
- 这样做的目的：
  - 让默认执行直接遵循 formal 已验证的 `Retrain Monthly` 结论
  - 避免继续把“冻结很久但面板每天刷新”的旧模型误当作最新 production
  - 继续把研究 winner 判决与 production 证据边界分开管理
- 默认 production 候选还会保留两层护栏：
  - 月度自动重训之外，仍显示 `21` 个交易日提醒阈值
  - 达到 `63` 个交易日时默认拦截，除非显式 `--allow-stale-model` 放行

### 需要重训的情况
- 股票池边界、交易约束、执行口径发生实质变化。
- 默认候选正式表现明显退化，或现实成本复核失效。
- 研究侧出现新的正式 winner，并准备进入执行升级比较。
- 上游研究 run 需要重新生成正式产物，而不仅仅是刷新 live 面板。
- 默认 production 候选最近一次 `launch_cutoff_date` 已跨入新的自然月。

### 当前默认候选的重训入口
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\update_default_candidate_production.py
```

- 该入口会：
  - 保留 formal holdout 研究证据不动
  - 自动计算“最新可标注训练日”
  - 用截至上线前的全部可标注数据重训一次 production full-fit
  - 冻结 formal winner 的 selected execution profile，避免 production full-fit 静默改写 execution strategy
  - 把稳定日常执行产物同步到 `deep_alpha_liquid500_dynamic_graph_bridge_production_default`

### 只刷新 live 面板，不重训
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\export_live_panels_from_run.py --run-dir daily_research\output\deep_alpha_liquid500_dynamic_graph_bridge_production_default
```

- 注意：
  - 这一步只能更新 `daily_live_*` 面板的新鲜度；
  - 不能替代默认入口里的月度自动重训逻辑；
  - 也不能消除底层 production 模型的重训时效提醒/拦截状态。

## 5. 显式回退流程
### 第 1 步：仅在需要时刷新旧机器学习产物
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\update_model_legacy_ml.py --data-source tq --start-date 20210101 --benchmark 000300.SH --regime-max-annual-vol 0.32 --regime-quadrants trend_up_low_vol,trend_up_high_vol --ml-target-horizons 5,10,20 --ml-horizon-weights 5:0.2,10:0.3,20:0.5 --ml-train-window-days 504
```

### 第 2 步：生成显式旧主线回退计划
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_trade_plan_legacy_ml.py
```

## 6. 候选工具
### 查看候选配置列表
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_trade_plan.py --list-candidate-profiles
```

### 生成候选交易计划
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_research_candidate_trade_plan.py --candidate-profile default
```

常用别名：
- `default`
- `aggressive`
- `robust_auto`
- `soft_guard`

### 生成候选回测
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_research_candidate_backtest.py --candidate-profile default
```

## 7. 研究入口
### 查看 dynamic-graph 配置列表
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_dynamic_graph_ablation.py --list-profiles
```

### 运行 dynamic-graph 正式消融矩阵
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_dynamic_graph_formal_ablation_matrix.py
```

### 查看 short-alpha 实验配置列表
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_experiment_matrix.py --list-profiles
```

### 运行 short-alpha recent-formal 矩阵
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_experiment_matrix.py
```

- 当前首轮 winner：
  - `state_liquidity_listwise_v1`
- 当前首轮降级分支：
  - `short_target_v1`
  - `short_input_v1`
  - `short_combo_v1`

### 运行 short-alpha 三窗 formal head-to-head
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_formal_head2head.py
```

- 当前正式结论：
  - `state_liquidity_listwise_v1` 不是 recent-window lucky run
  - 但第一窗仍退化，下一步先做 execution objective 对齐，不直接升格为默认执行候选

### 查看 architecture 配置列表
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_architecture_experiment_matrix.py --list-profiles
```

### 运行 architecture recent-formal 矩阵
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_architecture_experiment_matrix.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe"
```

- 当前正式输出：
  - `daily_research/output/deep_alpha_architecture_matrix_20260402_r1`
- 当前 recent-formal 结论：
  - `baseline_current` 仍是最近窗口收益 winner
  - `structure_context_only` 是当前最强的风险收益比结构挑战者
  - 单纯增加容量、增加深度或切换 vanilla `transformer` / `mamba` 都没有超过基线

### 运行 architecture 三窗 formal head-to-head
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_architecture_formal_head2head.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe"
```

- 当前正式输出：
  - `daily_research/output/deep_alpha_architecture_formal_head2head_20260402_r1`
- 当前多窗结论：
  - `structure_context_only` 是当前最可信的 raw holdout 多窗稳健升级方向
  - `graph_off_plain` 与 `depth_shallow_l1` 也有增益，但仍先保留为研究对照

### 运行 architecture execution-objective 三窗 formal head-to-head
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_architecture_execution_objective_head2head.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe"
```

- 当前正式输出：
  - `daily_research/output/deep_alpha_architecture_execalign_formal_20260403_r2`
- 当前 execution-objective 结论：
  - `structure_context_only` 的 raw 优势没有穿过 `train_eval_auto + robust_composite + realistic cost` gate
  - `baseline_current + regoff_k2 execalign` 已完成 formal 重跑、production full-fit promotion 与默认执行切换
  - 后续如再升级，必须以这条 execution-first 默认链路为对照

### 运行 `deep_alpha` 重训频率 formal 矩阵
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_retrain_frequency_formal_matrix.py --frequencies annual_freeze,quarterly_63d,monthly_calendar,every_21d --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe"
```

- 当前正式输出：
  - `daily_research/output/deep_alpha_retrain_frequency_formal_20260402_r1`
- 排行榜读取口径：
  - `frequency_summary_common_window.csv`
- 如需补跑中断实验：
  - 继续复用同一个 `root-tag`
  - runner 会自动复用已完成 block，不要另起新 tag 从头重训

## 8. 诊断与维护
### 性能分化报告
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\performance_dispersion_report.py --help
```

### 脑文档守卫
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\doc_guard.py check
```

### 工作区维护
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\workspace_maintenance.py report
```

## 9. 升级纪律
- 不允许静默替换默认入口。
- 新候选若未通过以下项目，不得升级：
  - 同窗正式比较
  - 显式成本外部回放
  - 多窗口 H2H
- 不允许升级单个幸运调仓相位。
- 本文件不记录长实验叙事。

## 10. Execution-First Unified Ops
### 查看当前 active execution strategy
```powershell
Get-Content daily_research\output\active_execution_strategy.json
```

### 让 production promotion 同时激活默认执行策略
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\update_default_candidate_production.py --activate-strategy --strategy-panel-mode auto
```

- 该入口现在会同时做三件事：
  - 继承研究赢家的 `research_objective_mode`
  - 继承 `checkpoint_selection_objective`
  - 继承 `execution_alignment_mode / objective / candidate_profiles / realistic cost`
- promotion 完成后还会刷新：
  - `daily_research/output/active_execution_strategy.json`

### 查看默认执行目前读到的 profile
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_trade_plan.py --list-candidate-profiles
```

- 如果接线正常，`default=` 应该显示：
  - `active_execution_strategy`
## Monthly Research Commands
### 月度正式研究入口
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_deep_alpha_research.py --research-time-unit calendar_months --valid-months 12 --train-eval-window-months 6 --adaptive-task-window-months 6
```

- 若只想缩短正式验证窗，可直接改：
  - `--valid-months`
- 若只想缩短 train-side score head / risk gate / adaptive 权重窗口，可直接改：
  - `--train-eval-window-months`
  - `--adaptive-task-window-months`

### 月度预训练入口
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\pretrain_deep_alpha_encoder.py --research-time-unit calendar_months --valid-months 12 --pretrain-valid-months 3
```

### 显式短窗优先级
- 若命令已经显式给出：
  - `--train-end-date`
  - `--valid-start-date`
  - `--valid-days`
- 则验证窗按显式交易日执行。
- 只有在未显式给出日窗，或主动把 `--valid-days 0` 交给月度协议时，`valid_months` 才主导验证窗长度。
# 2026-04-04 rich experiment 重跑入口

## 重新生成 monthly execution-first architecture execution H2H
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_architecture_execution_objective_head2head.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --root-tag deep_alpha_architecture_execalign_formal_20260403_monthly_r1
```

## 重新生成 monthly execution-first short-alpha formal H2H
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_short_alpha_formal_head2head.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --root-tag short_alpha_formal_head2head_20260403_monthly_r1
```

## 重新生成 monthly execution-first dynamic-graph formal ablation
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_dynamic_graph_formal_ablation_matrix.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --root-tag dynamic_graph_ablation_formal_20260403_monthly_r1
```

## TQ 波动时的 raw cache 强制复用入口
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_deep_alpha_research.py --force-raw-cache-path daily_research\cache\deep_alpha\raw\fc86ee4ed74716c2.pkl
```

## 当前已确认的新结果目录
- `daily_research/output/deep_alpha_architecture_execalign_formal_20260403_monthly_r1`
- `daily_research/output/short_alpha_formal_head2head_20260403_monthly_r1`
- `daily_research/output/dynamic_graph_ablation_formal_20260403_monthly_r1`

## Finetune Epoch 预算充分性 formal
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_epoch_budget_formal_matrix.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" --epoch-budgets 4,8,12,16
```

- 输出目录：
  - `daily_research/output/deep_alpha_epoch_budget_formal_20260404_r1`
- 当前判决：
  - `baseline_current` 在 monthly execution-first 三窗下，`16` epoch 的 mean replay excess annual / Sharpe = `8.50% / 0.494`
  - 当前参考 `8` epoch = `4.72% / 0.258`
  - `12` epoch 与 `8` epoch 基本相同；真正的提升出现在 `16` epoch
  - 提升集中在窗口 `20240301_20250317`，selected checkpoint 从 `epoch 7` 延后到 `epoch 13`
  - 四档 `undertrained_count` 都是 `0/3`，说明当前基于 `valid_loss` 的 undertrained 诊断不足以替代 execution-first 预算验证
## 训练续训与家族级 frontier 协议

### strict resume
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_deep_alpha_research.py --resume-run-dir daily_research\output\<old_run> --resume-mode strict --epochs 16
```

- `strict` 现在会恢复：
  - `last_model_state_dict`
  - `optimizer_state_dict`
  - `scheduler_state_dict`
  - `scaler_state_dict`
  - sampler epoch 与历史 `train_history`

### warm-start continue
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_deep_alpha_research.py --resume-run-dir daily_research\output\<old_run> --resume-mode warm_start --epochs 16
```

- `warm_start` 只加载选中模型参数，不继承 optimizer / scheduler / scaler 状态。

### 家族级 epoch frontier 校准
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_family_epoch_frontier_calibration.py --python-executable "C:\Users\ASUS\miniconda3\envs\yolos\python.exe"
```

- 默认家族：
  - `baseline`
  - `structure`
  - `short_alpha`
  - `dynamic_graph`
- 默认协议：
  - 先跑 `4/8/12/16`
  - 如果最右边界仍是最优，或仍有 objective-aligned budget pressure，再扩到 `24/32`
- 最新冻结预算 manifest：
  - `daily_research/output/deep_alpha_family_epoch_budget_latest.json`
- 当前已冻结的家族预算：
  - `baseline -> 12`
  - `structure -> 12`
  - `short_alpha -> 32`
  - `dynamic_graph -> 16`
- 当前 frontier 结果目录：
  - `daily_research/output/deep_alpha_family_epoch_frontier_20260404_r2`

### rich experiment 读取冻结预算
- `run_architecture_execution_objective_head2head.py`
- `run_short_alpha_formal_head2head.py`
- `run_dynamic_graph_formal_ablation_matrix.py`

- 这三条 formal runner 现在默认都会读取：
  - `daily_research/output/deep_alpha_family_epoch_budget_latest.json`
- 因此后续 rich experiment 默认不再手填统一 `--epochs`；除非显式做 budget stress test，否则应让 runner 直接读取家族预算 manifest
