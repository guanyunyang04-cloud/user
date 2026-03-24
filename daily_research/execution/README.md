# 盘后执行端说明

## 1. 定位
`daily_research/execution/` 是当前正式执行链路的入口目录，只负责三件事：
1. 盘后刷新高流动性股票池；
2. 盘后更新离线模型产物；
3. 盘后生成“次日开盘手工执行”的交易建议。

这不是自动下单系统。当前执行端固定为：
- 盘后生成建议，次日开盘人工执行；
- 默认读取离线模型，不在生成计划时临时重训。

## 2. 当前默认执行口径
- 主线：`advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`
- 默认股票池：`universe/liquid500_latest.txt`
- 成交假设：`next_open`
- 调仓语义：日频目标更新，默认 `rebalance_freq=1d`
- 市场状态边界：`regime_ma_window=50`、`regime_max_annual_vol=0.32`、`trend_up_low_vol,trend_up_high_vol`
- 默认模型族：`lgbm`
- 默认训练窗口：`ml_train_window_days=504`

## 3. 每日标准流程
### 第 1 步：更新高流动性股票池
```bash
python daily_research/execution/update_liquid_pool.py --start-date 20240101
```
这一步会刷新：
- `universe/liquid300_latest.txt`
- `universe/liquid500_latest.txt`
- `universe/liquid800_latest.txt`
- `universe/liquidity_rank_latest.csv`
- `universe/liquidity_pool_summary_latest.csv`

常用补充参数：
- `--signal-date`
- `--pool-sizes`
- `--lookback-days`
- `--no-cache`
- `--refresh-cache`

### 第 2 步：更新离线模型产物
```bash
python daily_research/execution/update_model.py --data-source tq --start-date 20210101 --benchmark 000300.SH --regime-max-annual-vol 0.32 --regime-quadrants trend_up_low_vol,trend_up_high_vol --ml-target-horizons 5,10,20 --ml-horizon-weights 5:0.2,10:0.3,20:0.5 --ml-train-window-days 504
```
这个包装脚本会自动补：
- `--artifact-path=models/latest_ml_model.joblib`
- `--artifact-meta-path=models/latest_ml_model.json`
- `--stocks-file=universe/liquid500_latest.txt`
- `--ml-model-family=lgbm`
- 当前执行主线共享默认参数

`latest_ml_model.json` 里优先看：
- `trained_at`
- `latest_data_date`
- `model_family`
- `validation_summary`
- 当前 regime 参数与训练窗口信息

### 第 3 步：更新账号快照
把真实持仓和可用现金写进：
- `daily_research/execution/current_positions.csv`

推荐格式：
```csv
record_type,stock,shares,cost_price,available_cash
account,,,,200000
position,600000.SH,1000,10.52,
position,600036.SH,800,42.10,
position,000001.SZ,1200,12.38,
```

说明：
- `account` 行写账户级可用现金；
- `position` 行写单只持仓；
- 若文件不存在，`run_trade_plan.py` 会优先复制 `current_positions.example.csv`；
- 老格式 `stock,shares,cost_price` 仍兼容；
- 若既没有 `account` 行，也没有命令行传 `--cash`，系统会按 `0` 现金生成计划。

### 第 4 步：生成次日开盘计划
```bash
python daily_research/execution/run_trade_plan.py --data-source tq --start-date 20210101 --benchmark 000300.SH --holding-count 5 --rebalance-freq 1d --regime-ma-window 50 --regime-max-annual-vol 0.32 --regime-quadrants trend_up_low_vol,trend_up_high_vol --max-style-weight 0.50
```
这个包装脚本会自动补：
- `--positions-file=current_positions.csv`
- `--output-dir=output/`
- `--model-artifact=models/latest_ml_model.joblib`
- `--stocks-file=universe/liquid500_latest.txt`
- 当前执行主线共享默认参数

额外行为：
- 若 `current_positions.csv` 中存在 `account` 行，会自动读取 `available_cash`；
- `--cash` 是可选覆盖参数，传了就以命令行为准；
- 默认启用模型新鲜度保护：
  - 滞后 `1` 个交易日开始提醒；
  - 滞后 `3` 个交易日开始拦截；
  - 如确需继续，可显式传 `--allow-stale-model`；
- 只有隐藏参数 `--train-on-the-fly` 才会改为实时训练，日常不建议使用。

## 4. 目录与职责
- `update_liquid_pool.py`：刷新执行端高流动性股票池。
- `update_model.py`：包装 `baseline/train_trade_model.py`，训练并导出默认离线模型产物。
- `run_trade_plan.py`：包装 `baseline/generate_daily_trade_plan.py`，基于离线模型生成次日开盘建议。
- `current_positions.example.csv`：账号快照样例。
- `current_positions.csv`：当前真实账号快照。
- `models/`：默认模型产物目录。
- `universe/`：默认执行股票池目录。
- `output/`：每次运行的计划输出目录。

当前最重要的默认文件：
- `models/latest_ml_model.joblib`
- `models/latest_ml_model.json`
- `universe/liquid500_latest.txt`
- `output/latest_trade_plan.txt`

## 5. 输出文件怎么读
每次运行后会写出：
- `output/<信号日期>/`
- 或 `output/<experiment_tag>/`

运行目录里当前会有：
- `daily_trade_plan.txt`
- `actions_today.csv`
- `holdings_snapshot.csv`
- `watchlist.csv`
- `training_log.csv`
- `plan_summary.json`

同时会刷新：
- `output/latest_trade_plan.txt`

`latest_trade_plan.txt` 是每天优先看的文件。它会写清楚：
- 信号日期与计划执行日期；
- 当前市场状态；
- 是否允许开仓；
- 当前使用的模型文件与模型族；
- 模型训练时间、样本截止日、最新数据日；
- 模型新鲜度与验证摘要；
- 卖出、减仓、买入、加仓建议；
- 当前持仓概览与候选观察名单；
- 输入现金与计划后剩余现金估算。

## 6. 常用覆盖参数
### `update_model.py`
- `--ml-model-family histgb|etr|lgbm`
- `--ml-state-horizon-profiles`
- `--refresh-cache`
- `--no-cache`
- `--no-auto-trim-history`
- `--skip-validation-summary`
- `--validation-retrain-every-days`
- `--validation-min-observations`

### `run_trade_plan.py`
- `--cash`：临时覆盖账号快照中的现金。
- `--stocks`：临时改成小股票池测试。
- `--experiment-tag`：自定义输出目录名。
- `--refresh-cache`
- `--no-cache`
- `--no-auto-trim-history`
- `--stale-model-warn-trading-days`
- `--stale-model-max-trading-days`
- `--allow-stale-model`

## 7. 执行边界
- 这是盘后生成、次日开盘执行的人工建议，不是自动报单程序。
- 如果模型文件不存在，先运行 `update_model.py`。
- 如果当天市场状态不允许开仓，计划里可能只有卖出或减仓动作。
- 如果次日开盘明显跳空，优先按目标权重调整，不要机械照搬估算股数。
- 如果实际可用现金与计划输入现金不一致，以真实资金为准重新估算买入数量。
- 如果只是做小范围测试，优先显式传 `--stocks`，不要直接改默认股票池文件。

## 8. 执行端与研究端的分工
- 执行端默认冻结为：`advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`
- 执行端关心的是：明天开盘实际怎么交易。
- 正式研究端关心的是：历史滚动股票池下，策略升级是否可信。
- 因此执行端默认读取最新 `liquid500`，而正式研究继续使用历史滚动 `liquid500 / liquid800`。
