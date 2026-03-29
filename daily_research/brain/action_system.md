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
- 因此旧快照收益只保留为审计 artifact；当前执行 wrapper 已切回当前仓安全桥接后端，后续若再沿用、回滚或切换，必须先经过用户确认与桥接验证

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
## 7. Gemini Final Closeout
- Before every final user-facing reply in an active Gemini frontend collaboration session, run:
  - `daily_research\tools\gemini_frontend.cmd closeout -WorkSummary "..." -NextStep "..."`
- The closeout is used to confirm completed work and align on the next move.
- If the frontend window has been closed, treat that collaboration mode as ended.

## 6. 执行安全边界
- 日常不启用实时训练
- 默认启用模型新鲜度保护
- 研究侧局部高收益候选不得静默替换 live 默认值
