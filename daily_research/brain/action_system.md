# Daily Research 行动系统

快照日期：`2026-04-01`

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
- 日常默认流程不允许静默重训模型。
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
  - `regoff_k2_10d_ensemble_native_anchor`
- 该入口会先检查默认候选的 `daily_live_*` 面板是否落后于最新完成交易日；若落后，会在不重训的前提下自动刷新。
- 默认输出：
  - `daily_research/execution/output/latest_trade_plan.txt`

## 4. 模型重训规则
- 每日默认流程不重训。
- `run_trade_plan.py` 只会基于现有已训练模型刷新 `daily_live_*` 面板，不会静默重训。
- 这样做的目的：
  - 保持日常执行稳定
  - 避免把研究阶段的漂移直接带进每日计划
  - 让“默认执行”和“研究升级”分开管理

### 需要重训的情况
- 股票池边界、交易约束、执行口径发生实质变化。
- 默认候选正式表现明显退化，或现实成本复核失效。
- 研究侧出现新的正式 winner，并准备进入执行升级比较。
- 上游研究 run 需要重新生成正式产物，而不仅仅是刷新 live 面板。

### 当前默认候选的重训入口
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\run_deep_alpha_research.py --data-source tq --start-date 20210101 --end-date 20260401 --benchmark 000300.SH --liquidity-pool liquid500 --rolling-liquidity-pool liquid500 --pool-rebalance-days 21 --rebalance-freq 10d --rebalance-offset-mode all --rebalance-anchor-date 2025-01-02 --encoder-family patch_transformer --score-head-method manual --score-risk-mode plain --dynamic-graph-layer --dynamic-graph-top-k 8 --dynamic-graph-temperature 0.35 --dynamic-graph-industry-boost 0.15 --dynamic-graph-style-boost 0.10 --experiment-tag deep_alpha_liquid500_dynamic_graph_bridge_20260401_formal_r1
```

### 只刷新 live 面板，不重训
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\deep_alpha\export_live_panels_from_run.py --run-dir daily_research\output\deep_alpha_liquid500_dynamic_graph_bridge_20260401_formal_r1
```

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
