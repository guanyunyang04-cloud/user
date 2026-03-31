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

## 5. 关键执行文件
- `daily_research/execution/update_liquid_pool.py`
- `daily_research/execution/update_model.py`
- `daily_research/execution/run_trade_plan.py`
- `daily_research/execution/current_positions.csv`
- `daily_research/execution/models/latest_ml_model.joblib`
- `daily_research/execution/models/latest_ml_model.json`
- `daily_research/execution/output/latest_trade_plan.txt`

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
## 6. Gemini Final Closeout
- This step is temporarily disabled.
- From `2026-03-29`, Gemini is no longer part of the default final-answer workflow.
- Do not block delivery on Gemini closeout; current workspace evidence remains sufficient.

## 7. 执行安全边界
- 日常不启用实时训练
- 默认启用模型新鲜度保护
- 研究侧局部高收益候选不得静默替换 live 默认值
