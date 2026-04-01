# Daily Research Action System

## 1. 作用
本文件是 `daily_research` 分脑的行动系统，负责保存盘后执行链路和日常操作流程。

上游脑模块分工如下：

- 稳定主线与项目身份：
  - `semantic_memory.md`
- 当前默认决策与 upgrade gate：
  - `working_memory.md`
- 环境与命令口径：
  - `environment_model.md`
- 方法学与协同技能：
  - `procedural_memory.md`

## 2. 当前执行边界
`daily_research/execution/` 只负责三件事：

1. 盘后刷新高流动性股票池
2. 盘后更新离线模型产物
3. 盘后生成“次日开盘手工执行”的交易建议

这不是自动下单系统。当前执行端固定为：

- 盘后生成建议，次日开盘人工执行
- 默认读取离线模型，不在生成计划时临时重训

## 3. 当前默认执行口径
- 主线：
  - `advanced_ml_current_code_live_anchor (ma50 baseline, lgbm520 v250) + liquid500 + next_open`
- `2026-03-31` mainboard-only formal revalidation outputs：
  - `daily_research/output/advanced_ml_current_code_live_anchor_20260331_mainboard_formal_r1`
  - `daily_research/output/advanced_ml_attack_defense_controller_20260331_mainboard_formal_r1`
  - `daily_research/output/market_feature_profile_compare_20260331_mainboard_formal_r1`
  - `daily_research/output/advanced_ml_cross_profile_attack_defense_20260331_mainboard_formal_r1`
- 当前结论：
  - live-anchor bridge `v250 @ 504 / 21 / 520` on `20210101 -> 20260327`：`18.09% / 20.83% / 1.034`，但 weak Sharpe 已转负
  - `20190101 -> 20260327` same-protocol shortlist 已退化为：static defense `3.44% / 0.279`，best same-profile dynamic `3.59% / 0.286`，best cross-profile dynamic `2.23% / 0.222`
  - 因此 execution 侧不再继续深挖当前 wrapper 参数空间；默认下一步改为把研究侧更强机会集迁到 execution 候选
- 当前执行后端：
  - 当前仓 `baseline/train_trade_model.py`
  - 当前仓 `baseline/generate_daily_trade_plan.py`
  - 模型产物与计划文件仍写回当前 `daily_research/execution/`
- 默认股票池：
  - `universe/liquid500_latest.txt`
- 成交假设：
  - `next_open`
- 调仓语义：
  - `rebalance_freq=1d`
- 市场状态边界：
  - `regime_ma_window=50`
  - `regime_max_annual_vol=0.32`
  - `trend_up_low_vol,trend_up_high_vol`
- 默认模型族：
  - `lgbm`
- 默认训练窗口：
  - `ml_train_window_days=504`
- `lgbm_n_estimators=520`
- 默认状态集成：
  - 当前 wrapper 默认注入 `trend_up_low_vol=ml:0.25,none:0.25,v2:0.50`
  - 当前 wrapper 默认注入 `enhanced_profile=up_low_breakout_v2`

### 3.1 执行真实性红线
- `2026-03-29` 的隔离 worktree ablation 已确认：
  - 只要在当前代码里临时关闭 `ml_alpha.py::_label_lookahead_bars()` 的 label-safe gap，同口径 `legacy_v7 + no_auto_trim_history + liquid500 + next_open + lgbm` 就会从 `151.70% / 0.882` 回跳到 `958.89% / 2.447`
- 这说明旧快照高收益的主因不是更好的因子，而是 `next_open` 训练边界上的 `label leakage / look-ahead bias`
- 因此旧快照收益只保留为审计 artifact；当前执行 wrapper 已稳定在当前仓当前代码 live-anchor 后端，后续若再沿用、回滚或切换，必须先经过用户确认与桥接验证

## 4. 每日标准流程
### 第 1 步：更新高流动性股票池
```bash
python daily_research/execution/update_liquid_pool.py --start-date 20240101
```

### 第 2 步：更新离线模型产物
```bash
python daily_research/execution/update_model.py --data-source tq --start-date 20210101 --benchmark 000300.SH --regime-max-annual-vol 0.32 --regime-quadrants trend_up_low_vol,trend_up_high_vol --ml-target-horizons 5,10,20 --ml-horizon-weights 5:0.2,10:0.3,20:0.5 --ml-train-window-days 504
```

包装脚本会自动补：

- `--artifact-path=models/latest_ml_model.joblib`
- `--artifact-meta-path=models/latest_ml_model.json`
- `--stocks-file=universe/liquid500_latest.txt`
- `--ml-model-family=lgbm`
- `--lgbm-n-estimators=520`
- `--regime-ma-window=50`
- 并由当前仓 `baseline/train_trade_model.py` 在无泄漏口径下实际训练

### 第 3 步：更新账号快照
把真实持仓和可用现金写进：

- `daily_research/execution/current_positions.csv`

### 第 4 步：生成次日开盘计划
```bash
python daily_research/execution/run_trade_plan.py --data-source tq --start-date 20210101 --benchmark 000300.SH --holding-count 5 --rebalance-freq 1d --regime-ma-window 50 --regime-max-annual-vol 0.32 --regime-quadrants trend_up_low_vol,trend_up_high_vol --max-style-weight 0.50
```

包装脚本会自动补：

- `--positions-file=current_positions.csv`
- `--output-dir=output/`
- `--model-artifact=models/latest_ml_model.joblib`
- `--stocks-file=universe/liquid500_latest.txt`
- `--regime-ma-window=50`
- 并由当前仓 `baseline/generate_daily_trade_plan.py` 生成计划

### 第 4.5 步：把研究侧分数迁到 execution candidate
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_research_candidate_trade_plan.py `
  --data-source tq `
  --start-date 20210101 `
  --benchmark 000300.SH `
  --external-score-csv daily_research/output/deep_alpha_liquid800_dynamic_graph_20260331_smoke_v2/latest_scores.csv `
  --candidate-label dynamic_graph_v1_candidate `
  --experiment-tag dynamic_graph_candidate_bridge_20260331_smoke
```

- 这个入口专门承接研究侧 `latest_scores.csv`，当前已验证可直接消费 `deep_alpha` 产物
- 默认仍沿用 `liquid500_latest.txt + next_open + current_positions.csv`
- 输出写到 `daily_research/execution/output/research_candidates/`，不会覆盖默认 `execution/output/latest_trade_plan.txt`
- `2026-04-01` 深夜已把这条入口升级成双模式：除 `--external-score-csv` 之外，也支持 `--external-target-weight-csv`，这样候选计划可以直接承接研究端 `daily_target_weight_panel.csv`
- 当前可直接运行的 `target_weight` 直连候选计划入口：
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_research_candidate_trade_plan.py `
  --data-source tq `
  --start-date 20210101 `
  --benchmark 000300.SH `
  --external-target-weight-csv daily_research/output/deep_alpha_liquid500_dynamic_graph_bridge_20260401_formal_r1/daily_target_weight_panel.csv `
  --external-score-csv daily_research/output/deep_alpha_liquid500_dynamic_graph_bridge_20260401_formal_r1/daily_score_panel.csv `
  --candidate-label dynamic_graph_v1_target_weight_candidate `
  --experiment-tag dynamic_graph_target_weight_candidate_20260401_smoke `
  --no-market-regime-filter
```
- 这轮 smoke 已通过；由于 `2026-03-31` 最后一天研究权重本身就是全零，所以当前候选计划输出为“无明确调仓动作”，这代表桥接成功，不代表策略失效。

### 第 4.6 步：把整段研究分数 formal 化成 execution backtest
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_research_candidate_backtest.py `
  --data-source tq `
  --start-date 20250101 `
  --benchmark 000300.SH `
  --score-panel-csv daily_research/output/deep_alpha_liquid500_dynamic_graph_bridge_20260331_smoke/daily_score_panel.csv `
  --candidate-label deep_alpha_liquid500_dynamic_graph_bridge_smoke `
  --experiment-tag deep_alpha_liquid500_dynamic_graph_execbridge_20260331_smoke
```

- 这个入口承接 `deep_alpha/run_deep_alpha_research.py` 导出的 `daily_score_panel.csv`
- 第一条 smoke 已证明链路可用，但也暴露了当前主瓶颈：研究端短窗高收益并不会自动等价成 execution 高收益
- 若目标是尽快冲击 `100%` 年化，下一阶段应优先围绕“降低 research -> execution 翻译损耗”做实验，而不是继续泛扫旧 wrapper
- 当前最值得先 formal 化的翻译层方向是：`holding_count=3 + max_weight=0.35 + no_market_regime_filter`。在 `2026-03-31` 的 smoke 快扫里，这组仅靠 execution 翻译层就把同一候选从 `24.82%` 年化抬到了 `34.54%`
- `2026-04-01` 的正式口径结果已经出来：同一路线在 `deep_alpha_liquid500_dynamic_graph_bridge_20260401_formal_r1 -> deep_alpha_liquid500_dynamic_graph_execbridge_20260401_formal_r1` 上没有守住，`holding_count=3 + max_weight=0.35 + no_market_regime_filter` 只跑到 `12.54%` 年化、`-1.80%` 超额年化。
- 同日晚间对正式分数面板做了 12 格 quickscan，当前 formal best 改成 `holding_count=8 + max_weight=0.25 + no_market_regime_filter`，但也仅有 `16.02%` 年化、`1.24%` 超额年化；因此默认下一步不再是继续死拧这组翻译参数，而是要升级“研究 -> execution”的桥接表达本身。
- 当前入口已升级成双模式：`run_research_candidate_backtest.py` 既支持 `--score-panel-csv`，也支持 `--target-weight-panel-csv`；bridge metrics 现在会额外落盘 `weeklyized_return / excess_weeklyized_return`，但它们只作为辅指标，不替代年化收益主判据。
- `2026-04-01` 晚间已完成第一轮 `target_weight` 直连 formal：  
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_research_candidate_backtest.py `
  --data-source tq `
  --start-date 20250101 `
  --benchmark 000300.SH `
  --target-weight-panel-csv daily_research/output/deep_alpha_liquid500_dynamic_graph_bridge_20260401_formal_r1/daily_target_weight_panel.csv `
  --score-panel-csv daily_research/output/deep_alpha_liquid500_dynamic_graph_bridge_20260401_formal_r1/daily_score_panel.csv `
  --candidate-label dynamic_graph_v1_execbridge_weightpanel_formal_r1 `
  --experiment-tag deep_alpha_liquid500_dynamic_graph_execbridge_weightpanel_20260401_formal_r1 `
  --no-market-regime-filter
```
- 这轮结果是 `15.41%` 年化、`0.70%` 超额年化、`0.031` 超额 Sharpe、`-19.88%` 回撤：已经优于默认 aggressive score 重建桥，但仍未超过当日晚间 `holding_count=8 + max_weight=0.25 + no_market_regime_filter` 的 quickscan best。因此默认下一步从“继续调 score->weight 翻译参数”更新为：“以 `target_weight` 直连桥为优先表达，再去研究状态条件仓位控制 / execution objective 对齐”。
- `2026-04-01` 深夜已把这条线扫到更深处：`target_weight` 直连桥现在支持 `target_weight_top_k / min_weight / power / full_invest / rebalance_offset`，可以直接测试“集中化 + 周期执行 + 相位敏感性”。
- 当前关键结论：
  - `1d` 下的小修小补基本无效，桥接收益几乎不动；
  - `1d + top_k=2` 已能抬到 `21.30%` 年化、`5.84%` 超额年化；
  - `5d / 10d` 的单 offset 点估值可以冲到 `97%~117%` 年化，但相位极敏感，不能直接当生产方案；
  - 全 offset 等权 sleeve ensemble 仍能保留很强收益，因此“翻译损耗”方向已经被证实有应用价值。
- 当前这条方向的默认可复核产物在：
  - `daily_research/output/dynagraph_target_weight_bridge_scan_20260401_formal_r1/regoff_scan_summary.csv`
  - `daily_research/output/dynagraph_target_weight_bridge_scan_20260401_formal_r1/regon_scan_summary.csv`
  - `daily_research/output/dynagraph_target_weight_bridge_scan_20260401_formal_r1/rebalance_focus_scan_summary.csv`
  - `daily_research/output/dynagraph_target_weight_bridge_scan_20260401_formal_r1/offset_scan_aggregate.csv`
  - `daily_research/output/dynagraph_target_weight_bridge_scan_20260401_formal_r1/ensemble_summary.csv`
- 当前默认 verdict：继续沿 `target_weight` 直连桥推进，但主形式从“单 offset N-day 执行”升级为“offset-ensemble / phase-robust execution bridge”；不再把单点高收益 lucky run 当作默认候选。
- `2026-04-01` 深夜已把这条线原生化成 anchored 入口：`run_research_candidate_backtest.py` / `run_research_candidate_trade_plan.py` 现在都支持 `--rebalance-offset-mode all` 和 `--rebalance-anchor-date`。经验结论是：如果不固定 `rebalance_anchor_date`，all-offset ensemble 的结果会随着历史起点漂移，不能直接当生产口径。
- 当前默认 anchored formal backtest 入口：
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_research_candidate_backtest.py `
  --data-source tq `
  --start-date 20250101 `
  --benchmark 000300.SH `
  --target-weight-panel-csv daily_research/output/deep_alpha_liquid500_dynamic_graph_bridge_20260401_formal_r1/daily_target_weight_panel.csv `
  --score-panel-csv daily_research/output/deep_alpha_liquid500_dynamic_graph_bridge_20260401_formal_r1/daily_score_panel.csv `
  --rebalance-freq 10d `
  --rebalance-offset-mode all `
  --rebalance-anchor-date 2025-01-02 `
  --target-weight-top-k 2 `
  --candidate-label dynamic_graph_regoff_k2_10d_ensemble_native_anchor `
  --experiment-tag deep_alpha_liquid500_dynamic_graph_execbridge_regoff_k2_10d_ensemble_native_anchor_20260401_formal_r3 `
  --no-market-regime-filter
```
- 当前默认 anchored candidate trade-plan 入口：
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_research_candidate_trade_plan.py `
  --data-source tq `
  --start-date 20250101 `
  --benchmark 000300.SH `
  --external-target-weight-csv daily_research/output/deep_alpha_liquid500_dynamic_graph_bridge_20260401_formal_r1/daily_target_weight_panel.csv `
  --external-score-csv daily_research/output/deep_alpha_liquid500_dynamic_graph_bridge_20260401_formal_r1/daily_score_panel.csv `
  --rebalance-freq 10d `
  --rebalance-offset-mode all `
  --rebalance-anchor-date 2025-01-02 `
  --target-weight-top-k 2 `
  --candidate-label dynamic_graph_regoff_k2_10d_ensemble_native_anchor `
  --experiment-tag dynamic_graph_regoff_k2_10d_ensemble_native_anchor_20260401_candidate_r1 `
  --no-market-regime-filter
```
- 当前 anchored two-finalist formal verdict：
  - `regon_k1_10d_ensemble_native_anchor = 52.32% / 32.91% / 1.800 / -9.36%`
  - `regoff_k2_10d_ensemble_native_anchor = 52.14% / 32.75% / 2.188 / -7.72%`
  - 默认生产候选优先 `regoff_k2_10d_ensemble_native_anchor`，更偏收益上沿的对照保留 `regon_k1_10d_ensemble_native_anchor`
- `2026-04-01` 晚间已把这条线产品化成 profile 入口：
  - `run_research_candidate_backtest.py` / `run_research_candidate_trade_plan.py` 现在支持 `--candidate-profile`
  - 若不显式传面板路径或 profile，这两个入口都会默认落到当前默认候选 `regoff_k2_10d_ensemble_native_anchor`
  - 可用 `--list-candidate-profiles` 查看当前内置 profile、alias 与说明
- 当前一键默认候选入口：
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_research_candidate_backtest.py
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_research_candidate_trade_plan.py
```
- 当前激进收益对照入口：
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_research_candidate_backtest.py --candidate-profile aggressive
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_research_candidate_trade_plan.py --candidate-profile aggressive
```
- 当前高收益 upgrade shortlist 入口：
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_research_candidate_backtest.py --candidate-profile robust_auto
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_research_candidate_trade_plan.py --candidate-profile robust_auto
```
- 当前 soft sizing comparator 入口：
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_research_candidate_backtest.py --candidate-profile soft_guard
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_research_candidate_trade_plan.py --candidate-profile soft_guard
```

- `2026-04-01` 上午已补齐 exact same-window comparator：`advanced_ml_live_anchor_samewindow_20260401_formal_r1` 里的 `trend_up_low_vol_ml25_none25_v250` 在 `bridge_full (2025-03-18 -> 2026-03-31)` 上是 `24.34% / 10.75% / 0.580 / -14.42%`，因此 anchored `regoff_k2 / regon_k1` 对当前 live-anchor 的领先已经是同窗 formal 结论。
- `2026-04-01` 上午也把第一版 soft state-conditioned sizing formal 跑完，summary 在 `execution_target_weight_state_conditioned_verdict_20260401_r1`：
  - `regoff_k2_stateoff` 仍是默认 winner；`quadrant_guard_v1 / trend_guard_v1 / market_state_guard_v1` 都会明显压低年化，只带来有限回撤改善。
  - `regon_k1_stateoff` 仍是更激进的收益对照；`quadrant_guard_v1` 在当前 hard regime filter 下基本是 no-op，其余 soft profile 也没有拿到升级资格。
  - 默认执行候选因此不变：继续保留 `regoff_k2_10d_ensemble_native_anchor`。
- `2026-04-01` 深夜已补完 execution candidate multi-window H2H：
  - `regoff_k2 vs regon_k1`：`regoff_k2` 赢 `4/5` 个窗口的 excess Sharpe，`regon_k1` 赢 `4/5` 个窗口的 excess annual return
  - `regoff_k2 vs execalign_auto_r4`：`execalign_auto_r4` 赢 `5/5` 个窗口的 excess annual return，赢 `4/5` 个窗口的 excess Sharpe
  - 当前结论因此写死为：`regoff_k2` 继续保留默认生产候选，`execalign_auto_r4_topk2_1d_regoff` 升格到高收益 upgrade shortlist，不能静默替换默认值

### 4.7 第一步 soft state-conditioned sizing formal 入口
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_research_candidate_backtest.py `
  --data-source tq `
  --start-date 20250101 `
  --end-date 20260331 `
  --benchmark 000300.SH `
  --target-weight-panel-csv daily_research/output/deep_alpha_liquid500_dynamic_graph_bridge_20260401_formal_r1/daily_target_weight_panel.csv `
  --score-panel-csv daily_research/output/deep_alpha_liquid500_dynamic_graph_bridge_20260401_formal_r1/daily_score_panel.csv `
  --rebalance-freq 10d `
  --rebalance-offset-mode all `
  --rebalance-anchor-date 2025-01-02 `
  --target-weight-top-k 2 `
  --candidate-label dynamic_graph_regoff_k2_10d_market_state_guard_v1 `
  --experiment-tag deep_alpha_liquid500_dynamic_graph_execbridge_regoff_k2_10d_market_state_guard_v1_20260401_formal_r1 `
  --no-market-regime-filter `
  --soft-state-profile market_state_guard_v1
```
- 当前支持的 profile：`quadrant_guard_v1 / trend_guard_v1 / market_state_guard_v1`
- 这组能力已经同时接到 `run_research_candidate_trade_plan.py`；但截至 `2026-04-01` 上午，soft sizing 仍只是 formal comparator，不是默认生产候选。

## 5. 关键执行文件
- `daily_research/execution/update_liquid_pool.py`
- `daily_research/execution/update_model.py`
- `daily_research/execution/run_trade_plan.py`
- `daily_research/execution/run_research_candidate_trade_plan.py`
- `daily_research/execution/run_research_candidate_backtest.py`
- `daily_research/execution/current_positions.csv`
- `daily_research/execution/models/latest_ml_model.joblib`
- `daily_research/execution/models/latest_ml_model.json`
- `daily_research/execution/output/latest_trade_plan.txt`
- `daily_research/execution/output/research_candidates/latest_trade_plan.txt`

### 5.1 当前研发路线入口
- 研发路线固定入口仍是 `deep_alpha`，而不是执行端 wrapper。
- 统一解释器优先使用：
  - `C:\Users\ASUS\miniconda3\envs\yolos\python.exe`
- 当前股票研究范围固定为：上证 A + 深证 A，剔除创业板、科创板与 ST；`liquid300 / liquid500 / liquid800` 默认池已按该约束重建。
- 训练任务一旦启动默认不手动打断；如需提速，只允许做不改变结果口径的运行时优化。

#### A. 当前 strict anchor 复核入口（可直接运行）
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_minimal_matrix.py `
  --phase backbone `
  --root-tag deep_alpha_strict_liquid800_anchor_20260331_r1 `
  --rolling-liquidity-pool liquid800 `
  --window-count 3 `
  --valid-days 252 `
  --start-date 20220101 `
  --benchmark 000300.SH `
  --skip-existing `
  --safe-runtime-profile
```

#### B. `Mamba / SSM` 已完成首轮判决（新 universe 下暂不继续优化）
```powershell
# `deep_alpha_mamba_patch_head2head_20260331_mainboard_r2` 已在新 universe 下完成：
# patch = 3.66% / 3.71% / 0.137 / -23.84%
# original_mamba = -4.39% / -4.41% / -0.233 / -25.60%
# 结论：`patch` 3/3 窗口全胜，原始 `mamba` 记为“首轮失败”，先不直接追加优化。
```

#### C. `dynamic graph` 默认下一条研究分支
`dynamic graph` 现在是 `deep_alpha` 的默认后续入口；若动态图仍未把 strict frontier 抬高，再切到 `MoE`。
- 当前已落地的是 `v1 = daily-updated top-k peer graph feature layer`，技术 smoke 产物为 `daily_research/output/deep_alpha_liquid800_dynamic_graph_20260331_smoke_v2`；这只是链路验收，不是正式前沿判决。
- `2026-03-31` 的 strict formal 旧基线对比也已补完：`deep_alpha_relgraph_h2h_20260331_mainboard_r1` 中，`dynamic_graph_v1` 相对 `relation_baseline` 实现 `2/3` 窗口 Sharpe 胜、`3/3` 窗口总收益胜，因此下一条默认 gate 已更新为 `plain vs dynamic_graph_v1`，暂不直接切去 `MoE`。
- 同日晚间已补完 `plain vs dynamic_graph_v1` 的 strict head-to-head，汇总在 `daily_research/output/deep_alpha_plain_vs_dynagraph_h2h_20260331_mainboard_r1`；`dynamic_graph_v1` 已在均值上跑赢 `plain`，因此当前默认下一步改为：围绕 `dynamic_graph_v1` 做稳健性确认和图参数 / 先验消融，而不是直接切 `MoE`。
- `2026-04-01` 晚间已把这条后续研究产品化为 wrapper：`daily_research/deep_alpha/run_dynamic_graph_ablation.py`
  - 默认 profile 为 `dynamic_graph_v1`
  - 支持 `--list-profiles`
  - 支持 `plain_baseline / dynamic_graph_v1 / dynamic_graph_topk4 / dynamic_graph_topk12 / dynamic_graph_no_industry_boost / dynamic_graph_no_style_boost / dynamic_graph_no_priors`
  - wrapper 会自动补齐 strict `liquid800` 研究口径、`patch_transformer + manual`、`next_open` 对应研究配置，并自动生成唯一 `experiment-tag`
- `2026-04-01` 下午又补上了正式 ablation matrix：`daily_research/deep_alpha/run_dynamic_graph_formal_ablation_matrix.py`
  - 这条脚本会复用已存在的 `plain_baseline / dynamic_graph_v1` formal runs
  - 只补跑缺失的 `dynamic_graph_no_priors / dynamic_graph_topk4 / dynamic_graph_topk12`
  - 自动产出 `ablation_summary.csv / ablation_window_details.csv / ablation_vs_plain_and_winner.csv / summary.md`
- 当前 formal ablation verdict 已落盘在 `daily_research/output/dynamic_graph_ablation_formal_20260401_r1`：
  - `dynamic_graph_v1 = 15.64% / 0.534 / -25.71%`，仍是均值超额年化 winner
  - `dynamic_graph_no_priors = 15.05% / 0.515 / -20.17%`，说明图结构本身确有价值，prior 只提供小幅增益
  - `dynamic_graph_topk4 = 12.76% / 0.768 / -21.73%`，更像风险收益比对照，不是新的总收益默认 winner
  - `dynamic_graph_topk12 = 7.21% / 0.324 / -19.56%`，可视为当前 no-go 宽图区间
- `2026-04-01` 晚间又把 `dynamic_graph_v1 / dynamic_graph_no_priors / dynamic_graph_topk4` 接到了同一条 `liquid500 -> regoff_k2_10d anchored all-offset` execution bridge，汇总在 `daily_research/output/dynamic_graph_execution_bridge_h2h_20260401_r1`：
  - 研究端 `liquid500` formal holdout：`dynamic_graph_v1 = 18.56% / 5.61% / 0.232`，`dynamic_graph_no_priors = 37.35% / 22.34% / 0.901`，`dynamic_graph_topk4 = 73.81% / 54.82% / 2.667`
  - execution bridge formal：`dynamic_graph_v1 = 52.14% / 32.75% / 2.188 / -7.72%`
  - execution bridge formal：`dynamic_graph_no_priors = 36.81% / 19.38% / 1.193 / -12.05%`
  - execution bridge formal：`dynamic_graph_topk4 = 44.61% / 26.18% / 1.954 / -12.58%`
  - 当前结论：默认 execution winner 仍是 `dynamic_graph_v1`；`no_priors / topk4` 继续保留为研究对照，不升格为默认候选。
- `2026-04-01` 深夜已把 `dynamic_graph_v1 -> execution objective` 也产品化到 `run_deep_alpha_research.py`：
  - 新参数：`--execution-alignment-mode off/profile/train_eval_auto`
  - `train_eval_auto` 当前扫描：`raw_1d / topk2_1d_regoff / regoff_k2_10d_ensemble_native_anchor / regon_k1_10d_ensemble_native_anchor`
  - research output 现在会额外落盘：`execution_aligned_daily_score_panel.csv / execution_aligned_daily_target_weight_panel.csv / execution_alignment_objective_rows.csv`
- current formal verdict 已更新成两阶段：
  - 第一轮 `dynamic_graph_execution_objective_alignment_20260401_r1`：按 `excess_annual_return` 选中了 `regon_k1_10d_ensemble_native_anchor`，但 aligned export replay 只有 `37.26% / 19.76% / 1.145 / -11.56%`
  - 第二轮 `deep_alpha_liquid500_dynamic_graph_v1_execalign_auto_20260401_formal_r4`：按 `robust_composite + train_eval_window_days=252` 选中了 `topk2_1d_regoff`，external export replay 达到 `108.05% / 85.32% / 3.317 / -11.01%`
  - 汇总 verdict 在 `daily_research/output/execution_alignment_robust_upgrade_20260401_r1`
  - 当前结论：execution-objective alignment 已经产生新的 formal high-upside winner，但它的换手和回撤压力更高，所以当前只升格到 upgrade shortlist，不静默替换默认 `regoff_k2`
- 当前动态图研究入口：
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_dynamic_graph_ablation.py --list-profiles
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_dynamic_graph_ablation.py --profile default
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_dynamic_graph_ablation.py --profile dynamic_graph_no_priors
```
- 当前 formal ablation 汇总入口：
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_dynamic_graph_formal_ablation_matrix.py
```
- 当前 `liquid500` execution bridge H2H 复跑入口：
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_deep_alpha_research.py `
  --data-source tq `
  --liquidity-pool liquid500 `
  --start-date 20210101 `
  --benchmark 000300.SH `
  --train-end-date 20250317 `
  --valid-start-date 20250318 `
  --valid-days 252 `
  --encoder-family patch_transformer `
  --score-head-method manual `
  --holding-count 5 `
  --max-weight 0.25 `
  --rebalance-freq 1d `
  --dynamic-graph-layer `
  --dynamic-graph-top-k 8 `
  --dynamic-graph-temperature 0.35 `
  --dynamic-graph-industry-boost 0.0 `
  --dynamic-graph-style-boost 0.0 `
  --no-safe-runtime-profile `
  --experiment-tag deep_alpha_liquid500_dynamic_graph_no_priors_bridge_20260401_formal_r1

& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_deep_alpha_research.py `
  --data-source tq `
  --liquidity-pool liquid500 `
  --start-date 20210101 `
  --benchmark 000300.SH `
  --train-end-date 20250317 `
  --valid-start-date 20250318 `
  --valid-days 252 `
  --encoder-family patch_transformer `
  --score-head-method manual `
  --holding-count 5 `
  --max-weight 0.25 `
  --rebalance-freq 1d `
  --dynamic-graph-layer `
  --dynamic-graph-top-k 4 `
  --dynamic-graph-temperature 0.35 `
  --dynamic-graph-industry-boost 0.15 `
  --dynamic-graph-style-boost 0.05 `
  --no-safe-runtime-profile `
  --experiment-tag deep_alpha_liquid500_dynamic_graph_topk4_bridge_20260401_formal_r1

& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_research_candidate_backtest.py `
  --data-source tq `
  --start-date 20250101 `
  --benchmark 000300.SH `
  --target-weight-panel-csv daily_research/output/deep_alpha_liquid500_dynamic_graph_no_priors_bridge_20260401_formal_r1/daily_target_weight_panel.csv `
  --score-panel-csv daily_research/output/deep_alpha_liquid500_dynamic_graph_no_priors_bridge_20260401_formal_r1/daily_score_panel.csv `
  --rebalance-freq 10d `
  --rebalance-offset-mode all `
  --rebalance-anchor-date 2025-01-02 `
  --target-weight-top-k 2 `
  --candidate-label dynamic_graph_no_priors_regoff_k2_10d_ensemble_native_anchor `
  --experiment-tag deep_alpha_liquid500_dynamic_graph_no_priors_execbridge_regoff_k2_10d_ensemble_native_anchor_20260401_formal_r1 `
  --no-market-regime-filter

& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_research_candidate_backtest.py `
  --data-source tq `
  --start-date 20250101 `
  --benchmark 000300.SH `
  --target-weight-panel-csv daily_research/output/deep_alpha_liquid500_dynamic_graph_topk4_bridge_20260401_formal_r1/daily_target_weight_panel.csv `
  --score-panel-csv daily_research/output/deep_alpha_liquid500_dynamic_graph_topk4_bridge_20260401_formal_r1/daily_score_panel.csv `
  --rebalance-freq 10d `
  --rebalance-offset-mode all `
  --rebalance-anchor-date 2025-01-02 `
  --target-weight-top-k 2 `
  --candidate-label dynamic_graph_topk4_regoff_k2_10d_ensemble_native_anchor `
  --experiment-tag deep_alpha_liquid500_dynamic_graph_topk4_execbridge_regoff_k2_10d_ensemble_native_anchor_20260401_formal_r1 `
  --no-market-regime-filter
```
- 当前 execution-objective alignment 入口：
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_deep_alpha_research.py --list-execution-alignment-profiles

& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_deep_alpha_research.py `
  --data-source tq `
  --liquidity-pool liquid500 `
  --start-date 20210101 `
  --benchmark 000300.SH `
  --train-end-date 20250317 `
  --valid-start-date 20250318 `
  --valid-days 252 `
  --encoder-family patch_transformer `
  --score-head-method manual `
  --holding-count 5 `
  --max-weight 0.25 `
  --rebalance-freq 1d `
  --dynamic-graph-layer `
  --dynamic-graph-top-k 8 `
  --dynamic-graph-temperature 0.35 `
  --dynamic-graph-industry-boost 0.15 `
  --dynamic-graph-style-boost 0.05 `
  --no-safe-runtime-profile `
  --execution-alignment-mode train_eval_auto `
  --execution-alignment-objective robust_composite `
  --train-eval-window-days 252 `
  --experiment-tag deep_alpha_liquid500_dynamic_graph_v1_execalign_auto_20260401_formal_r4
```
- 当前固定 `regoff_k2` 对照入口：
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_deep_alpha_research.py `
  --data-source tq `
  --liquidity-pool liquid500 `
  --start-date 20210101 `
  --benchmark 000300.SH `
  --train-end-date 20250317 `
  --valid-start-date 20250318 `
  --valid-days 252 `
  --encoder-family patch_transformer `
  --score-head-method manual `
  --holding-count 5 `
  --max-weight 0.25 `
  --rebalance-freq 1d `
  --dynamic-graph-layer `
  --dynamic-graph-top-k 8 `
  --dynamic-graph-temperature 0.35 `
  --dynamic-graph-industry-boost 0.15 `
  --dynamic-graph-style-boost 0.05 `
  --no-safe-runtime-profile `
  --execution-alignment-mode profile `
  --execution-alignment-profile regoff_k2_10d_ensemble_native_anchor `
  --experiment-tag deep_alpha_liquid500_dynamic_graph_v1_execalign_regoff_profile_20260401_formal_r1
```
- 当前 execution-aligned export replay 入口：
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\baseline\backtest_external_score_panel.py `
  --data-source tq `
  --start-date 20250318 `
  --end-date 20260331 `
  --benchmark 000300.SH `
  --target-weight-panel-csv daily_research/output/deep_alpha_liquid500_dynamic_graph_v1_execalign_auto_20260401_formal_r4/execution_aligned_daily_target_weight_panel.csv `
  --score-panel-csv daily_research/output/deep_alpha_liquid500_dynamic_graph_v1_execalign_auto_20260401_formal_r4/execution_aligned_daily_score_panel.csv `
  --candidate-label execalign_auto_r4_topk2_1d_regoff `
  --experiment-tag deep_alpha_liquid500_dynamic_graph_v1_execalign_auto_export_replay_20260401_r4 `
  --rebalance-freq 1d `
  --no-market-regime-filter
```

#### D. `dynamic graph` 旧基线入口（可直接运行，用来定义 no-go baseline）
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_deep_alpha_research.py `
  --data-source tq `
  --rolling-liquidity-pool liquid800 `
  --start-date 20220101 `
  --benchmark 000300.SH `
  --encoder-family patch_transformer `
  --score-head-method manual `
  --relation-layer `
  --valid-days 252 `
  --safe-runtime-profile `
  --experiment-tag deep_alpha_liquid800_relation_baseline_20260331_smoke
```

#### E. `dynamic graph` 目标入口（功能分支落地后沿用同一入口）
```powershell
# 目标是不再发明新脚本，而是在 run_deep_alpha_research.py 上追加动态图配置后直接 head-to-head：
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_deep_alpha_research.py `
  --data-source tq `
  --rolling-liquidity-pool liquid800 `
  --start-date 20220101 `
  --benchmark 000300.SH `
  --encoder-family patch_transformer `
  --score-head-method manual `
  --dynamic-graph-layer `
  --dynamic-graph-top-k 8 `
  --dynamic-graph-temperature 0.35 `
  --dynamic-graph-industry-boost 0.15 `
  --dynamic-graph-style-boost 0.05 `
  --valid-days 252 `
  --safe-runtime-profile `
  --experiment-tag deep_alpha_liquid800_dynamic_graph_20260331_smoke
```

#### F. `state-conditioned MoE` 低成本 warm-start（排在 `dynamic graph` 之后）
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_deep_alpha_research.py `
  --data-source tq `
  --rolling-liquidity-pool liquid800 `
  --start-date 20220101 `
  --benchmark 000300.SH `
  --encoder-family patch_transformer `
  --return-head-mode liquidity_switch `
  --state-context `
  --liquidity-context `
  --structure-context `
  --aux-structure-task `
  --liquidity-layer `
  --valid-days 252 `
  --safe-runtime-profile `
  --experiment-tag deep_alpha_liquid800_moe_warmstart_20260331_smoke
```

#### G. `state-conditioned MoE` 目标入口（功能分支落地后仍沿用同一入口）
```powershell
# 若 warm-start 有效，再把同一脚本升级成真正的 sparse expert routing；不要新开并行评测框架。
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_deep_alpha_research.py `
  --data-source tq `
  --rolling-liquidity-pool liquid800 `
  --start-date 20220101 `
  --benchmark 000300.SH `
  --encoder-family patch_transformer `
  --valid-days 252 `
  --safe-runtime-profile `
  --experiment-tag deep_alpha_liquid800_sparse_moe_20260331_smoke
```

#### H. RL 执行层入口
- RL 不从 `daily_research` 进入，统一转到：
  - `t0_project/brain/action_system.md`

#### I. Performance Dispersion Diagnostic
- Use this after any formal run to judge whether results are carried by a few strong phases.
- Main outputs:
  - `calendar_year_summary.csv`
  - `rolling_window_summary.csv`
  - `named_window_summary.csv`
  - `summary.md`
- This is a phase-dispersion diagnostic only; it does not replace full-period CAGR.
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\performance_dispersion_report.py `
  --run-dir daily_research/output/deep_alpha_liquid500_dynamic_graph_execbridge_regoff_k2_10d_ensemble_native_anchor_20260401_formal_r3

& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\performance_dispersion_report.py `
  --run-dir daily_research/output/advanced_ml_current_code_live_anchor_20260331_mainboard_formal_r1 `
  --strict-label trend_up_low_vol_ml25_none25_v250
```

#### J. Execution Candidate Multi-Window H2H
- Use this after any candidate formal replay to compare phase robustness on the same overlap windows.
- Main outputs:
  - `window_metrics_<label>.csv`
  - `head2head_summary.csv`
  - `summary.md`
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\execution_candidate_multiwindow_h2h.py `
  --left-run-dir daily_research/output/deep_alpha_liquid500_dynamic_graph_execbridge_regoff_k2_10d_ensemble_native_anchor_20260401_formal_r3 `
  --left-label regoff_k2 `
  --right-run-dir daily_research/output/deep_alpha_liquid500_dynamic_graph_execbridge_regon_k1_10d_ensemble_native_anchor_20260401_formal_r3 `
  --right-label regon_k1

& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\execution_candidate_multiwindow_h2h.py `
  --left-run-dir daily_research/output/deep_alpha_liquid500_dynamic_graph_execbridge_regoff_k2_10d_ensemble_native_anchor_20260401_formal_r3 `
  --left-label regoff_k2 `
  --right-run-dir daily_research/output/deep_alpha_liquid500_dynamic_graph_v1_execalign_auto_export_replay_20260401_r4 `
  --right-label execalign_auto_r4
```

## 6. Gemini Final Closeout
- This step is temporarily disabled.
- From `2026-03-29`, Gemini is no longer part of the default final-answer workflow.
- Do not block delivery on Gemini closeout; current workspace evidence remains sufficient.

## 7. 执行安全边界
- 日常不启用实时训练
- 默认启用模型新鲜度保护
- 研究侧局部高收益候选不得静默替换 live 默认值
