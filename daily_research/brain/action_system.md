# Daily Research 行动系统

快照日期：`2026-04-02`

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
  - `regoff_k2_10d_ensemble_native_anchor`
- 该入口会先检查默认候选底层 `production full-fit` 模型是否已跨入新的自然月；若已跨月，会先按 `Retrain Monthly` 自动运行 `update_default_candidate_production.py`，随后再检查 `daily_live_*` 面板是否落后于最新完成交易日并自动刷新。
- 默认日常计划读取：
  - `deep_alpha_liquid500_dynamic_graph_bridge_production_default/daily_live_score_panel.csv`
  - `deep_alpha_liquid500_dynamic_graph_bridge_production_default/daily_live_target_weight_panel.csv`
- formal 研究证据保留在：
  - `deep_alpha_liquid500_dynamic_graph_bridge_20260401_formal_r1`
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
