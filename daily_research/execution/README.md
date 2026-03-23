# 盘后执行端说明

## 1. 目录作用
`daily_research/execution/` 是当前正式执行链路的入口目录，负责三件事：

1. 盘后刷新高流动性股票池；
2. 盘后更新离线模型产物；
3. 盘后生成“次日开盘执行”的人工操作建议。

这套执行端的定位很明确：

- 盘后准备；
- 次日开盘执行；
- 人工下单，不做自动报单。

当前执行默认主线：

- `advanced_ml (ma50 baseline) + liquid500 + next_open`
- 已于 `2026-03-23` 从原 `ma60` 口径切换到 `ma50 baseline`

## 2. 目录与真实职责
- `update_liquid_pool.py`
  - 刷新执行端使用的高流动性股票池；
- `update_model.py`
  - 包装 `baseline/train_trade_model.py`；
  - 负责训练并导出最新模型产物；
- `run_trade_plan.py`
  - 包装 `baseline/generate_daily_trade_plan.py`；
  - 默认读取离线模型产物，生成次日开盘建议；
- `current_positions.example.csv`
  - 持仓样例；
- `current_positions.csv`
  - 当前真实持仓；
  - 如果文件不存在，`run_trade_plan.py` 会先按样例自动生成；
- `models/`
  - 模型产物目录；
  - 当前默认产物：
    - `latest_ml_model.joblib`
    - `latest_ml_model.json`
  - `latest_ml_model.json` 还会记录：
    - `train_summary`
    - `validation_summary`
    - 当前执行口径的 regime 参数
- `universe/`
  - 执行端股票池目录；
  - 当前默认维护：
    - `liquid300_latest.txt`
    - `liquid500_latest.txt`
    - `liquid800_latest.txt`
    - `liquidity_rank_latest.csv`
    - `liquidity_pool_summary_latest.csv`
- `output/`
  - 交易计划输出目录；
  - 当前最重要文件：
    - `latest_trade_plan.txt`

## 3. 包装脚本的默认行为
执行端两个主入口不是重新实现一套逻辑，而是对底层主程序做“默认参数注入”。

### 3.1 `update_model.py`
默认会自动补上：

- `--artifact-path`
  - `daily_research/execution/models/latest_ml_model.joblib`
- `--artifact-meta-path`
  - `daily_research/execution/models/latest_ml_model.json`
- `--stocks-file`
  - 如果你没有手动传 `--stocks` 或 `--stocks-file`，会默认读取：
    - `daily_research/execution/universe/liquid500_latest.txt`
- `--regime-ma-window`
  - 如果你没有手动传，会默认注入为 `50`

### 3.2 `run_trade_plan.py`
默认会自动补上：

- `--positions-file`
  - `daily_research/execution/current_positions.csv`
- `--output-dir`
  - `daily_research/execution/output`
- `--model-artifact`
  - `daily_research/execution/models/latest_ml_model.joblib`
- `--stocks-file`
  - 如果你没有手动传 `--stocks` 或 `--stocks-file`，会默认读取：
    - `daily_research/execution/universe/liquid500_latest.txt`
- `--regime-ma-window`
  - 如果你没有手动传，会默认注入为 `50`

额外行为：

- 如果 `current_positions.csv` 不存在，脚本会优先复制 `current_positions.example.csv`；
- 如果样例也不存在，才会创建一个最小表头文件。

## 4. 每日标准流程

### 第 0 步：更新高流动性股票池
推荐命令：

```bash
python daily_research/execution/update_liquid_pool.py --start-date 20240101
```

这一步会刷新：

- `liquid300_latest.txt`
- `liquid500_latest.txt`
- `liquid800_latest.txt`
- `liquidity_rank_latest.csv`
- `liquidity_pool_summary_latest.csv`

常用补充参数：

- `--signal-date`
  - 手动指定已完成交易日；
- `--pool-sizes`
  - 自定义导出的池规模，默认 `300,500,800`；
- `--lookback-days`
  - 流动性回看窗口，默认 `80`；
- `--no-cache`
  - 不读缓存；
- `--refresh-cache`
  - 强制刷新缓存。

### 第 1 步：更新模型产物
推荐命令：

```bash
python daily_research/execution/update_model.py --data-source tq --start-date 20210101 --benchmark 000300.SH --regime-ma-window 50 --regime-max-annual-vol 0.32 --regime-quadrants trend_up_low_vol,trend_up_high_vol --ml-target-horizons 5,10,20 --ml-horizon-weights 5:0.2,10:0.3,20:0.5 --ml-train-window-days 504
```

真实默认值与行为：

- 默认模型文件：
  - `models/latest_ml_model.joblib`
- 默认元数据文件：
  - `models/latest_ml_model.json`
- 默认股票池：
  - `universe/liquid500_latest.txt`
- 默认模型族：
  - `histgb`
- 默认多周期目标：
  - `5,10,20`
- 默认多周期权重：
  - `5:0.2,10:0.3,20:0.5`
- 默认训练窗口：
  - `504` 个交易日
- 当前默认执行状态边界：
  - `regime_ma_window=50`
- 默认验证摘要：
  - 写入 `latest_ml_model.json -> validation_summary`
  - 当前采用滚动 RankIC 摘要，默认 `21` 个交易日一个历史重训块

常用补充参数：

- `--ml-model-family histgb|etr|lgbm`
- `--ml-state-horizon-profiles`
  - 按市场状态指定周期权重
- `--refresh-cache`
- `--no-cache`
- `--no-auto-trim-history`
- `--skip-validation-summary`
- `--validation-retrain-every-days`
- `--validation-min-observations`

说明：

- 模型训练与交易计划生成已经拆开；
- 日常执行默认读取离线模型产物，不在计划生成时实时训练；
- 如果你没有先更新模型，`run_trade_plan.py` 会因为找不到模型产物而报错。

### 第 2 步：更新持仓文件
把你的真实持仓写入：

- `daily_research/execution/current_positions.csv`

格式：

```csv
stock,shares,cost_price
600000.SH,1000,10.52
600036.SH,800,42.10
000001.SZ,1200,12.38
```

字段说明：

- `stock`
  - 必须是 `600000.SH` 这种格式；
- `shares`
  - 当前持股数量；
- `cost_price`
  - 当前持仓成本价。

### 第 3 步：准备次日开盘可用现金
运行交易计划脚本时，需要传入预计到次日开盘可用的现金金额。

### 第 4 步：生成次日开盘计划
推荐命令：

```bash
python daily_research/execution/run_trade_plan.py --data-source tq --start-date 20210101 --benchmark 000300.SH --holding-count 5 --rebalance-freq 5d --cash 200000 --regime-ma-window 50 --regime-max-annual-vol 0.32 --regime-quadrants trend_up_low_vol,trend_up_high_vol --max-style-weight 0.50
```

真实默认值与行为：

- 默认持仓文件：
  - `current_positions.csv`
- 默认模型产物：
  - `models/latest_ml_model.joblib`
- 默认输出目录：
  - `output/`
- 默认股票池：
  - `universe/liquid500_latest.txt`
- 默认持仓数：
  - `5`
- 默认调仓频率：
  - `5d`
- 当前默认执行状态边界：
  - `regime_ma_window=50`
- 默认模型新鲜度保护：
  - 相对当前信号日滞后 `1` 个交易日开始提醒
  - 滞后 `3` 个交易日开始拦截
  - 如确需继续，可显式传入 `--allow-stale-model`
- 默认 lot size：
  - `100`
- 默认会读取离线模型产物；
- 只有隐藏参数 `--train-on-the-fly` 才会改为实时训练，日常不建议使用。

常用补充参数：

- `--refresh-cache`
- `--no-cache`
- `--no-auto-trim-history`
- `--stale-model-warn-trading-days`
- `--stale-model-max-trading-days`
- `--allow-stale-model`
- `--stocks`
  - 临时改成小股票池测试；
- `--experiment-tag`
  - 自定义输出目录名。

## 5. 输出文件
每次运行后，脚本会创建一个运行目录：

- `daily_research/execution/output/<信号日期>/`
- 或 `daily_research/execution/output/<experiment_tag>/`

并同步刷新：

- `daily_research/execution/output/latest_trade_plan.txt`

运行目录中当前真实会写出：

- `daily_trade_plan.txt`
- `actions_today.csv`
- `holdings_snapshot.csv`
- `watchlist.csv`
- `training_log.csv`
- `plan_summary.json`

`latest_trade_plan.txt` 是每天最值得看的文件。它会写清楚：

- 信号日期；
- 计划执行日期；
- 当前市场状态；
- 是否允许开仓；
- 当前使用的模型文件；
- 模型训练时间与训练样本截止日期；
- 模型最新数据日与模型新鲜度；
- 模型验证摘要；
- 卖出、减仓、买入、加仓建议；
- 当前持仓概览；
- 候选观察名单；
- 输入现金与计划后剩余现金估算。

## 6. 如何执行建议
次日开盘建议按下面顺序手工执行：

1. 先卖出；
2. 再减仓；
3. 再买入；
4. 最后加仓。

原因：

- 先释放现金；
- 更贴近组合调仓逻辑；
- 能减少“先买后卖导致现金不够”的问题。

## 7. 关键术语

### 7.1 市场状态
- `trend_up_low_vol`
  - 上涨低波；
- `trend_up_high_vol`
  - 上涨高波；
- `trend_down_low_vol`
  - 下跌低波；
- `trend_down_high_vol`
  - 下跌高波。

### 7.2 是否允许开仓
- `是`
  - 可以执行新增仓位动作；
- `否`
  - 更偏向风险控制，只建议卖出或减仓。

### 7.3 分数字段
- `综合分`
  - 最终排序与建仓决策使用的分数；
- `ML`
  - 机器学习横截面分数；
- `none`
  - 稳健基线分数；
- `v2`
  - `up_low_breakout_v2` 增强分数。

## 8. 执行端与研究端口径约定
- 执行端默认冻结为：
  - `advanced_ml (ma50 baseline) + liquid500 + next_open`
- 执行端股票池：
  - 每日盘后更新一次 `liquid500_latest.txt`
- 正式研究端股票池：
  - 历史滚动 `liquid500 / liquid800`
  - 默认每 `21` 个交易日重建一次

这样设计的原因是：

- 执行端关心的是“明天开盘实际交易什么”；
- 研究端关心的是“历史验证是否可信”；
- 因此二者可以同口径于 `next_open`，但不必同频于股票池刷新节奏。

## 9. 日常注意事项
- 这是盘后生成、次日开盘执行的人工建议，不是自动下单程序。
- 如果模型文件不存在，请先运行 `update_model.py`。
- 如果当天市场状态不允许开仓，计划里可能只有卖出或减仓动作。
- 如果你的实际可用现金与输入现金不一致，请以实际资金为准。
- 如果你只是做小范围测试，优先显式传 `--stocks`，不要直接改默认股票池文件。
