# 盘后执行端说明书

## 1. 这个文件夹是干什么的
`daily_research/execution/` 是日线研究主线的“具体执行端”。

你以后做尾盘操作，主要只需要看和维护这个目录里的内容，不需要再到 `baseline/`、`portfolio/`、`output/` 里来回切换。

这里的目标很明确：
- 每日盘后运行一次程序
- 自动生成“次日开盘执行”的操作建议
- 你根据文本建议在次日开盘手工下单

## 2. 目录结构
- `update_liquid_pool.py`
  - 盘后刷新高流动性基础池
- `update_model.py`
  - 离线更新模型产物
- `run_trade_plan.py`
  - 执行端统一入口，只负责读取模型产物并生成次日开盘建议
- `current_positions.csv`
  - 你当前真实持仓
- `current_positions.example.csv`
  - 持仓样例
- `models/`
  - 模型产物目录
  - 其中最重要的是：
    - `models/latest_ml_model.joblib`
  - 默认会学习多个周期，例如 `5/10/20` 日超额收益
- `universe/`
  - 每日更新的高流动性股票池目录
  - 默认会维护：
    - `liquid300_latest.txt`
    - `liquid500_latest.txt`
    - `liquid800_latest.txt`
- `output/`
  - 每次运行生成的结果
  - 其中最重要的是：
    - `output/latest_trade_plan.txt`

## 3. 每日使用流程

### 第零步：更新 liquid500 基础池
执行端默认股票池已经正式切到 `liquid500`。

先在盘后刷新当天的高流动性股票池：

```bash
python daily_research/execution/update_liquid_pool.py --start-date 20240101
```

这一步会在 `daily_research/execution/universe/` 下刷新：
- `liquid300_latest.txt`
- `liquid500_latest.txt`
- `liquid800_latest.txt`
- `liquidity_rank_latest.csv`
- `liquidity_pool_summary_latest.csv`

如果没有显式传入 `--stocks` 或 `--stocks-file`，执行端默认会读取：
- `daily_research/execution/universe/liquid500_latest.txt`

### 第一步：单独更新模型
模型训练和尾盘执行现在已经分开。

也就是说：
- `update_model.py` 负责训练并刷新模型产物
- `run_trade_plan.py` 只负责读取现成模型，生成次日开盘建议

推荐模型更新命令：

```bash
python daily_research/execution/update_model.py --data-source tq --stocks-file daily_research/execution/universe/liquid500_latest.txt --start-date 20210101 --benchmark 000300.SH --regime-max-annual-vol 0.32 --regime-quadrants trend_up_low_vol,trend_up_high_vol --ml-target-horizons 5,10,20 --ml-horizon-weights 5:0.2,10:0.3,20:0.5 --ml-train-window-days 504
```

训练端现在默认会：
- 自动把历史窗口裁剪到“当前训练真正需要的最近区间”
- 缓存原始日线数据和特征结果，重复训练会明显更快

常用补充参数：
- `--refresh-cache`
  - 强制刷新缓存，重新从 TQ 拉数并重算特征
- `--no-cache`
  - 临时关闭缓存
- `--no-auto-trim-history`
  - 临时关闭自动历史窗口裁剪

模型产物默认会写到：
- `daily_research/execution/models/latest_ml_model.joblib`

建议：
- 至少定期更新模型
- 更稳妥的做法是每天收盘后更新一次，或每周固定更新一次
- 如果当天没有更新模型，尾盘执行端也可以继续使用上一次的最新模型产物
- 当前推荐的多周期权重：
  - `5:0.2,10:0.3,20:0.5`
- 也支持按市场状态指定周期权重：
  - `--ml-state-horizon-profiles "trend_up_low_vol=5:0.15,10:0.25,20:0.60;trend_up_high_vol=5:0.30,10:0.35,20:0.35"`
- 但目前研究结论是：
  - 状态内周期权重的两轮实验还没有跑赢固定全局权重
  - 执行端日常仍建议先使用固定全局权重 `5:0.2,10:0.3,20:0.5`

### 第二步：更新持仓
先把你的当前持仓更新到：
- `daily_research/execution/current_positions.csv`

格式如下：

```csv
stock,shares,cost_price
600000.SH,1000,10.52
600036.SH,800,42.10
000001.SZ,1200,12.38
```

字段说明：
- `stock`
  - 股票代码，必须是 `600000.SH` 这种格式
- `shares`
  - 当前持股数量
- `cost_price`
  - 你的持仓成本价

如果某只股票已经卖完，就直接从文件里删掉。

### 第三步：确定次日开盘可用现金
运行脚本时，输入你预计到次日开盘可用的现金金额。

### 第四步：盘后运行程序
推荐命令：

```bash
python daily_research/execution/run_trade_plan.py --data-source tq --stocks-file daily_research/execution/universe/liquid500_latest.txt --start-date 20210101 --benchmark 000300.SH --holding-count 5 --rebalance-freq 5d --cash 200000 --regime-max-annual-vol 0.32 --regime-quadrants trend_up_low_vol,trend_up_high_vol --max-style-weight 0.50
```

推理端现在默认会：
- 自动裁剪到生成最新已收盘信号所需的最近历史窗口
- 复用训练/推理链路缓存，减少重复拉数和重复算因子
- 盘中不会使用当日未收盘的日线

如果你怀疑缓存不是最新的，可以加：

```bash
python daily_research/execution/run_trade_plan.py --data-source tq --stocks-file daily_research/execution/universe/liquid500_latest.txt --start-date 20210101 --benchmark 000300.SH --holding-count 5 --rebalance-freq 5d --cash 200000 --regime-max-annual-vol 0.32 --regime-quadrants trend_up_low_vol,trend_up_high_vol --max-style-weight 0.50 --refresh-cache
```

如果你只想先在小股票池测试：

```bash
python daily_research/execution/run_trade_plan.py --data-source tq --stocks 600000.SH,600036.SH,601318.SH,000001.SZ,000333.SZ,002415.SZ --start-date 20240101 --benchmark 000300.SH --holding-count 3 --rebalance-freq 5d --cash 200000
```

## 4. 最重要的输出文件
每次运行后，都会刷新：

- `daily_research/execution/output/latest_trade_plan.txt`

这就是你每天盘后最该看的文件。

它会写清楚：
- 信号日期
- 计划执行日期
- 当前市场状态
- 是否允许开仓
- 当前使用的是哪一个模型产物
- 模型训练时间和训练样本截止日期
- 次日建议卖出什么
- 次日建议减仓什么
- 次日建议买入什么
- 次日建议加仓什么
- 每条建议背后的原因和分数
- 当前持仓概览
- 候选观察名单

也会同步保存一份带日期或标签的目录，例如：

- `daily_research/execution/output/20260318/`
- `daily_research/execution/output/today_trade_plan/`

里面会有：
- `daily_trade_plan.txt`
- `actions_today.csv`
- `holdings_snapshot.csv`
- `watchlist.csv`
- `plan_summary.json`
- `training_log.csv`

## 5. 如何执行文本建议
建议在次日开盘按下面顺序手工执行：

1. 先卖出
2. 再减仓
3. 再买入
4. 最后加仓

原因：
- 这样可以先释放现金
- 也更接近组合调整逻辑

## 6. 文本里常见术语

### 市场状态
- `trend_up_low_vol`
  - 上涨低波
- `trend_up_high_vol`
  - 上涨高波
- `trend_down_low_vol`
  - 下跌低波
- `trend_down_high_vol`
  - 下跌高波

### 是否允许开仓
- `是`
  - 可以根据建议执行买入/加仓
- `否`
  - 程序会更偏向控制风险，不建议新增仓位

### 分数字段
- `综合分`
  - 最终用于排序和决策的总分
- `ML`
  - 机器学习横截面预测分数
- `none`
  - 稳健基线分数
- `v2`
  - `up_low_breakout_v2` 增强分数

## 7. 这套执行端的逻辑来源
当前执行端基于：
- 市场状态过滤
- `none` 稳健基线
- `up_low_breakout_v2` 增强版本
- 机器学习横截面排序
- 高集中组合构建

也就是说，这个执行端不是单纯拍脑袋给建议，而是：
- 先在盘后离线训练模型产物
- 再在盘后读取模型产物
- 用统一主线生成“次日开盘执行”建议

这样做的好处是：
- 训练和使用职责分开
- 尾盘执行更快、更稳定
- 你能明确知道今天建议基于哪一次训练结果

## 8. 你每天真正要做的事
你每天只需要做三件事：

1. 先视情况运行 `update_model.py`
2. 更新 `current_positions.csv`
3. 盘后运行 `run_trade_plan.py`
4. 打开 `output/latest_trade_plan.txt`，按建议在尾盘手工执行

## 9. 注意事项
- 这份建议是“尾盘人工执行建议”，不是自动下单。
- 如果你的实际可用现金和文件中输入的不一致，请以实际资金为准。
- 如果尾盘流动性不足，优先处理卖出和核心买入。
- 如果当天市场状态不允许开仓，程序可能只给出卖出/减仓建议。
- 如果你不确定当天是否该严格执行全部建议，优先执行：
  - 风险控制动作
  - 卖出动作
  - 排名最靠前的买入动作

## 执行与研究口径约定（2026-03-21）
- 执行端默认冻结为：`advanced_ml + liquid500 + next_open`。
- 执行端的流动池更新频率是：`每日盘后更新一次`。
  也就是先运行 `daily_research/execution/update_liquid_pool.py`，刷新 `daily_research/execution/universe/liquid500_latest.txt`，再据此训练或生成次日计划。
- 这样设计的原因是：执行端面对的是“明天要交易什么”，所以应优先使用最新已完成交易日的流动性排名结果。
- 模型训练频率和股票池刷新频率是两件事：模型可以按计划定期更新，但执行股票池仍然每日刷新。
- 任何研究策略如果没有在 `next_open`、高流动性股票池、样本外窗口下完成正式验证，不直接进入执行端。

## Runtime Notes (2026-03-22)
- Execution remains frozen at `advanced_ml + liquid500 + next_open`.
- The execution side refreshes `liquid500_latest.txt` every trading day after close.
- Deep-alpha research now has separate training diagnostics and safe runtime caps for this local machine.
