# Daily Research

## 1. 项目定位
`daily_research` 用于日线研究主线，目标是把“选股—组合—回测—评估—研究迭代”串成一套稳定流程。

当前阶段聚焦：
- 全A股票池
- 当日收盘调仓成交
- 高集中组合
- 相对基准超额评估
- 市场状态过滤
- 状态内动态权重

当前阶段暂不包含：
- 机器学习
- 强化学习
- 分钟级执行
- 行业中性化
- 基本面因子

说明：
- 上面这组“不包含”针对原 `run_daily_research.py` 的第一阶段基线。
- 现在已经新增一条更激进的交付线：
  - `baseline/run_advanced_daily_research.py`
  - 它会在现有因子框架上叠加机器学习横截面排序。

## 2. 文档入口
- `daily_research_plan.md`
  - 研究总纲、当前主线、下一步规划
- `research_log.md`
  - 已做实验、参数、结果、结论
- `execution/`
  - 盘后执行端
  - 包含持仓文件、统一入口脚本、输出目录和单独说明书
- `baseline/`
  - 第一阶段研究代码
- `output/`
  - 每次实验的输出目录

## 3. 当前推荐主线
当前建议拆成两条：

### 3.1 稳健基线
- 股票池：全A
- 基准：`000300.SH`
- 权重：`score`
- 调仓频率：`5d`
- 市场状态过滤：开启
- 状态白名单：
  - `trend_up_low_vol`
  - `trend_up_high_vol`
- 风格约束：开启
  - `max_style_weight=0.50`
- 状态内动态权重：
  - `none`

### 3.2 增强候选
- 在稳健基线之上，仅增强 `trend_up_low_vol`
- 当前最有希望的增强版本：
  - `up_low_breakout_v2`

## 4. 运行方式

### 4.1 TQ 全A研究
稳健基线：
```bash
python daily_research/baseline/run_daily_research.py --data-source tq --universe-scope all_a --start-date 20220101 --benchmark 000300.SH --weighting-method score --rebalance-freq 5d --market-regime-filter --regime-max-annual-vol 0.32 --regime-quadrants trend_up_low_vol,trend_up_high_vol --style-cap --max-style-weight 0.50 --state-alpha-profile none --experiment-tag current_baseline
```

增强候选：
```bash
python daily_research/baseline/run_daily_research.py --data-source tq --universe-scope all_a --start-date 20220101 --benchmark 000300.SH --weighting-method score --rebalance-freq 5d --market-regime-filter --regime-max-annual-vol 0.32 --regime-quadrants trend_up_low_vol,trend_up_high_vol --style-cap --max-style-weight 0.50 --state-alpha-profile up_low_breakout_v2 --experiment-tag current_candidate
```

### 4.2 TQ 指定股票池研究
```bash
python daily_research/baseline/run_daily_research.py --data-source tq --stocks 600000.SH,600036.SH,000001.SZ --start-date 20220101 --benchmark 000300.SH --weighting-method score --rebalance-freq 5d --experiment-tag custom_universe_test
```

### 4.3 CSV 离线研究
```bash
python daily_research/baseline/run_daily_research.py --data-source csv --csv-folder daily_research/data --benchmark 000300.SH --weighting-method score --rebalance-freq 5d --experiment-tag csv_stage1
```

### 4.4 先进版主程序
```bash
python daily_research/baseline/run_advanced_daily_research.py --data-source tq --universe-scope all_a --start-date 20210101 --benchmark 000300.SH --holding-count 5 --rebalance-freq 5d --regime-max-annual-vol 0.32 --regime-quadrants trend_up_low_vol,trend_up_high_vol --max-style-weight 0.50 --ml-target-horizons 5,10,20 --ml-horizon-weights 5:0.2,10:0.3,20:0.5 --ml-train-window-days 504 --ml-retrain-every-days 21 --experiment-tag advanced_ml_mainline
```

这条程序会做：
- 市场状态过滤
- `none / up_low_breakout_v2` 两套基线分数
- `sklearn` 梯度提升树滚动训练
- 机器学习分数与基线分数集成
- 高集中组合回测与完整输出

### 4.5 盘后策略与次日开盘执行
模型训练与执行现在分开，执行端默认高流动性基础池已经切到 `liquid500`：

- 训练模型：`daily_research/execution/update_model.py`
- 生成次日开盘建议：`daily_research/execution/run_trade_plan.py`

先在盘后刷新高流动性基础池：

```bash
python daily_research/execution/update_liquid_pool.py --start-date 20240101
```

默认会生成并刷新：
- `daily_research/execution/universe/liquid300_latest.txt`
- `daily_research/execution/universe/liquid500_latest.txt`
- `daily_research/execution/universe/liquid800_latest.txt`

先更新模型产物：

```bash
python daily_research/execution/update_model.py --data-source tq --stocks-file daily_research/execution/universe/liquid500_latest.txt --start-date 20210101 --benchmark 000300.SH --regime-max-annual-vol 0.32 --regime-quadrants trend_up_low_vol,trend_up_high_vol --ml-target-horizons 5,10,20 --ml-horizon-weights 5:0.2,10:0.3,20:0.5 --ml-train-window-days 504
```

如需研究状态内周期权重，也支持：

```bash
python daily_research/execution/update_model.py --data-source tq --stocks-file daily_research/execution/universe/liquid500_latest.txt --start-date 20210101 --benchmark 000300.SH --regime-max-annual-vol 0.32 --regime-quadrants trend_up_low_vol,trend_up_high_vol --ml-target-horizons 5,10,20 --ml-horizon-weights 5:0.2,10:0.3,20:0.5 --ml-state-horizon-profiles "trend_up_low_vol=5:0.15,10:0.25,20:0.60;trend_up_high_vol=5:0.30,10:0.35,20:0.35" --ml-train-window-days 504
```

当前研究结论：
- 状态内周期权重功能已经可用。
- 但截至目前，两轮状态内权重实验都没有稳定跑赢固定全局权重。
- 因此 advanced ML 主线仍推荐固定全局权重 `5:0.2,10:0.3,20:0.5`。

先准备持仓文件，例如：
- `daily_research/execution/current_positions.csv`
- 可参考：`daily_research/execution/current_positions.example.csv`

然后在盘后运行：
```bash
python daily_research/execution/run_trade_plan.py --data-source tq --stocks-file daily_research/execution/universe/liquid500_latest.txt --start-date 20210101 --benchmark 000300.SH --holding-count 5 --rebalance-freq 5d --positions-file daily_research/execution/current_positions.csv --cash 200000 --regime-max-annual-vol 0.32 --regime-quadrants trend_up_low_vol,trend_up_high_vol --max-style-weight 0.50 --experiment-tag today_trade_plan
```

输出：
- `daily_research/execution/output/latest_trade_plan.txt`
- `daily_research/execution/output/<日期或标签>/actions_today.csv`
- `daily_research/execution/output/<日期或标签>/watchlist.csv`
- `daily_research/execution/output/<日期或标签>/plan_summary.json`

说明：
- 这个脚本会每日刷新建议文本
- 你只需要根据 `latest_trade_plan.txt` 在次日开盘手工执行

## 5. 常用参数

### 基础参数
- `--data-source`
  - `tq` 或 `csv`
- `--stocks`
  - 可选，覆盖默认股票池
- `--start-date`
- `--end-date`
- `--benchmark`
- `--experiment-tag`

### 组合参数
- `--holding-count`
- `--weighting-method`
  - `equal` 或 `score`
- `--rebalance-freq`
- `--score-threshold`

### 市场状态过滤
- `--market-regime-filter`
- `--regime-ma-window`
- `--regime-vol-window`
- `--regime-max-annual-vol`
- `--regime-quadrants`

当前推荐：
- `--regime-max-annual-vol 0.32`
- `--regime-quadrants trend_up_low_vol,trend_up_high_vol`

### 风格约束
- `--style-cap`
- `--max-style-weight`

当前推荐：
- `--style-cap --max-style-weight 0.50`

### 状态内动态权重
- `--state-alpha-profile none`
- `--state-alpha-profile up_low_breakout_v1`
- `--state-alpha-profile up_low_breakout_v2`
- `--state-alpha-profile up_dual_v1`
- `--state-alpha-profile up_dual_v2`

当前推荐：
- 稳健基线：`--state-alpha-profile none`
- 增强候选：`--state-alpha-profile up_low_breakout_v2`

### 机器学习增强参数
- `--ml-target-horizon`
- `--ml-train-window-days`
- `--ml-retrain-every-days`
- `--ml-min-train-dates`
- `--ml-max-samples-per-day`
- `--ml-max-train-rows`
- `--ensemble-ml-weight`
- `--ensemble-none-weight`
- `--ensemble-v2-weight`

### 盘后建议参数
- `--positions-file`
- `--cash`
- `--lot-size`
- `--output-dir`

当前推荐：
- `--ml-target-horizon 20`
- `--ml-train-window-days 504`
- `--ml-retrain-every-days 21`
- `--ensemble-ml-weight 0.70`
- `--ensemble-none-weight 0.20`
- `--ensemble-v2-weight 0.10`

## 6. 市场状态说明
当前框架把市场分成四个象限：
- `trend_up_low_vol`
  - 基准高于中期均线，且波动率较低
- `trend_up_high_vol`
  - 基准高于中期均线，但波动率较高
- `trend_down_low_vol`
  - 基准低于中期均线，且波动率较低
- `trend_down_high_vol`
  - 基准低于中期均线，且波动率较高

当前主线只允许两个上涨象限开仓：
- `trend_up_low_vol`
- `trend_up_high_vol`

## 7. 状态内动态权重说明

### `up_low_breakout_v1`
- 仅在 `trend_up_low_vol` 中切换为“低波 + 结构突破增强”风格
- 主要强化：
  - `breakout_20`
  - `range_position_20`
  - `drawdown_20`
  - `up_day_ratio_10`

### `up_low_breakout_v2`
- 仅在 `trend_up_low_vol` 中切换为“更克制的低波 + 结构增强”
- 相比 `v1`：
  - 保留更多基线的低波/量价骨架
  - 降低过强的进攻化改动
- 当前是最有希望通过冻结验证的增强版本

### `up_dual_v1`
- `trend_up_low_vol`
  - 使用 `up_low_breakout_v1`
- `trend_up_high_vol`
  - 切换为“趋势延续 / 中期动量增强”
  - 主要强化：
    - `mom_60`
    - `ma_gap_20_60`
    - `trend_slope_20`
    - `up_day_ratio_10`

### `up_dual_v2`
- `trend_up_low_vol`
  - 使用 `up_low_breakout_v2`
- `trend_up_high_vol`
  - 延续 `up_dual_v1` 的高波动量增强
- 当前仍属于实验分支，且冻结验证结果明显弱于 `up_low_breakout_v2`

## 8. 输出文件
每次运行会在 `daily_research/output/<experiment_tag_or_timestamp>/` 下生成：
- `equity_curve.csv`
  - 组合、基准、超额净值与每日收益
- `actions.csv`
  - 建仓、加仓、减仓、清仓动作日志
- `metrics.json`
  - 核心回测指标
- `factor_ic_summary.csv`
  - 单因子/组合分数的 IC、RankIC、ICIR 汇总
- `factor_quantile_returns.csv`
  - 分层收益汇总
- `latest_scores.csv`
  - 最新交易日候选排序、目标权重和分组分数
- `regime_state.csv`
  - 若启用市场状态过滤，则输出每日状态

特定分析脚本还会生成：
- `quadrant_summary.csv`
- `year_quadrant_summary.csv`
- `walkforward_metrics.json`
- `train_profile_metrics.csv`
- `test_window_metrics.csv`
- `training_log.csv`
  - 先进版主程序的滚动训练日志

## 9. CSV 输入要求
CSV 模式要求每个股票一个文件，至少包含这些列：
- `Date` 或 `Datetime`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`
- `Amount`

如果使用 CSV 模式并指定基准，基准文件也需要在同一目录中，例如 `000300.SH.csv`。

## 10. 当前研究结论摘要
- `score + 5d` 明显优于 `equal + 1d`
- 市场状态过滤有效
- 上涨双象限白名单优于只保留 `trend_up_low_vol`
- `2025` 的问题不是没有 alpha，而是上涨低波环境中过于防守
- `up_low_breakout_v2` 是当前最有希望的增强版本
- `up_dual_v1 / up_dual_v2` 虽然能改善部分样本，但目前仍缺少足够稳健的样本外证据
- `trend_up_high_vol` 目前继续固定使用 `none` 更稳妥
- 在已测试的事前激活机制里，`20d RankIC + 按季度刷新 + 仅在 trend_up_low_vol 内切换 none / up_low_breakout_v2` 是当前最有前景的方案
- 训练窗口稳健性验证显示：
  - `18` 个月与 `24` 个月效果接近
  - `36` 个月明显变弱
  - 当前更推荐 `24` 个月训练窗口
- 为了更快交付一套更强程序，已新增先进版主程序：
  - `run_advanced_daily_research.py`
  - 使用 `sklearn` 梯度提升树做滚动横截面预测
  - 再与 `none / up_low_breakout_v2` 基线分数做集成
- 已经提炼出一个可解释的规则候选：
  - 当 `breakout_20 RankIC >= 0`
  - 且 `drawdown_20 RankIC >= 0`
  - 且 `range_position_20 RankIC >= -0.06`
  - 则在下一季度的 `trend_up_low_vol` 中启用 `up_low_breakout_v2`

下一步主线：
- 用连续回测口径重做一次 RankIC 激活对照
- 对规则版阈值做稳健性验证
## Deep Alpha 研究分支

除了 `baseline/` 与 `execution/` 这两条线，当前还新增了一条更偏“表示学习”的研究分支：

- `daily_research/deep_alpha/`

这条分支的目标不是继续微调少量因子权重，而是直接研究：

1. 市场状态学习
2. 股票时序表示学习
3. 多任务横截面排序

当前入口：

```bash
python daily_research/deep_alpha/run_deep_alpha_research.py --data-source tq --stocks 600000.SH,600036.SH,601318.SH,600519.SH,000333.SZ,000858.SZ,601166.SH,601939.SH --start-date 20220101 --benchmark 000300.SH --lookback-window 60 --valid-days 120 --epochs 2 --batch-size 128 --hidden-dim 64 --encoder-family transformer --experiment-tag deep_alpha_smoke
```

提速建议：

- 首次运行会自动缓存：
  - TQ 原始日线数据
  - `deep_alpha` 特征与目标
- 默认缓存目录：
  - `daily_research/cache/deep_alpha/`
- 常用提速参数：
  - `--num-workers 4`
  - `--pin-memory`
  - `--use-amp`
  - `--max-rank-pairs-per-group 2048`
- 如需强制重建缓存：
  - `--refresh-cache`
- 如需关闭缓存：
  - `--no-cache`

当前支持两类时序编码器：

- `--encoder-family gru`
- `--encoder-family transformer`

当前还支持排序增强训练：

- `--ranking-loss-weight 0.5`

这会把 `deep_alpha` 从单纯逐样本回归，升级为“回归 + 按日期横截面排序”的混合训练。

当前还支持第一版轻量关系层：

- `--relation-layer`

它会接入：
- 行业内排名
- 行业强度
- 风格强度

目前这层仍处于实验阶段，默认建议先关闭，等更大样本验证后再决定是否纳入主线。

当前还支持学习式二层打分头：

- `--score-head-method manual`
- `--score-head-method ridge --adaptive-task-weights`
- `--score-head-method lgbm --adaptive-task-weights --adaptive-task-window-days 126`

这一步的作用是：
- 不再手工规定 `5d / 10d / 20d / downside` 如何合成为持仓分数
- 让一个轻量二层模型直接学习“什么样的预测组合更值得进组合”

主要输出：

- `train_history.csv`
- `validation_predictions.csv`
- `validation_embeddings.csv`
- `validation_rankic_summary.csv`
- `market_state_frame.csv`
- `latest_scores.csv`
- `equity_curve.csv`
- `actions.csv`
- `deep_alpha_model.pt`
- `metrics.json`

这条分支目前还处在研究早期，目标是先建立“能真正学习时序结构和市场状态”的框架，再决定是否把成熟结果回灌到执行端。
## 文档维护约定
为避免 `research_log.md`、`daily_research_plan.md`、`README.md` 再次出现中文错码，后续统一按下面的方式维护：

- 所有文档默认按 `UTF-8` 保存。
- 优先使用正常编辑器保存，或使用 `apply_patch` 修改文档。
- 不再用容易引入编码问题的 shell 重定向方式直接追加中文内容。
- 如需安全追加日志段落，使用：

```bash
python daily_research/tools/doc_guard.py append --file daily_research/research_log.md --header "## 2026-03-20 新实验" --body-file your_note.md
```

- 更新文档后，执行一次检查：

```bash
python daily_research/tools/doc_guard.py check
```

- 如果检查结果里：
  - `replacement_char_count=0`
  - `suspicious_question_lines_in_tail=0`

  说明文档尾部没有明显编码污染。

- 如需查看当前工作区体积、缓存占用和可安全清理目标，执行：

```bash
python daily_research/tools/workspace_maintenance.py report
```

- 如需只清理可再生的 Python 缓存，执行：

```bash
python daily_research/tools/workspace_maintenance.py clean --targets pycache --apply
```

- 当前默认维护原则：
  - 优先清理 `__pycache__`、`*.pyc`、日志等可再生产物。
  - 不直接删除 `daily_research/output/`、`daily_research/cache/` 里的研究结果，除非已经确认可归档或可重建。
  - 归档规则统一维护在 `daily_research/archive_policy.json`。
  - 归档 payload 放到 `daily_research/archive/`，其中 manifest 与说明进 Git，payload 本体继续留在 Git 外。

- 如需预演当前归档候选，执行：

```bash
python daily_research/tools/workspace_maintenance.py archive
```

- 如需正式归档超出热区的研究产物，执行：

```bash
python daily_research/tools/workspace_maintenance.py archive --apply
```

## Deep Alpha 固定研究池
`deep_alpha` 现在支持直接复用执行端每日更新的高流动性股票池：

```bash
python daily_research/deep_alpha/run_deep_alpha_research.py --data-source tq --liquidity-pool liquid500 --start-date 20240101 --benchmark 000300.SH --encoder-family transformer --ranking-loss-weight 0.5 --listwise-loss-weight 0.25 --return-loss-mode regression --return-target-transform raw --score-risk-mode state_gate --epochs 4 --valid-days 252 --experiment-tag deep_alpha_liquid500_research
```

可选池：
- `--liquidity-pool liquid300`
- `--liquidity-pool liquid500`
- `--liquidity-pool liquid800`

当前建议：
- `liquid500`：高流动性主研究池
- `liquid800`：更广的高流动性验证池

当前最新复验说明：
- 每日更新固定池模式下，`deep_alpha` 的共享基线（`shared`）比当前这版 `targeted_liquidity_rank` 更稳。
- 因此后续高流动性专用研究，先以 `shared` 为固定参考基线，再继续做更轻、更定向的结构强化。

## 历史滚动高流动性研究框架
为了避免把“今天的 liquid500/liquid800”静态拿去回看更早历史，现在研究端已经支持**历史滚动高流动性股票池**。

这套框架的原则是：
- 信号口径：`盘后信号 -> 次日开盘执行`
- 股票池口径：用**当时**的 `ADV20` 排名重建，而不是用今天的幸存者名单回看历史
- 默认重建频率：每 `21` 个交易日

`deep_alpha` 正式研究命令示例：

```bash
python daily_research/deep_alpha/run_deep_alpha_research.py --data-source tq --rolling-liquidity-pool liquid500 --pool-rebalance-days 21 --start-date 20220101 --benchmark 000300.SH --encoder-family transformer --ranking-loss-weight 0.5 --listwise-loss-weight 0.25 --return-loss-mode regression --return-target-transform raw --score-risk-mode state_gate --epochs 4 --valid-days 252 --experiment-tag deep_alpha_liquid500_rolling
```

`advanced_ml` 同口径重验命令示例：

```bash
python daily_research/baseline/run_advanced_daily_research.py --data-source tq --rolling-liquidity-pool liquid500 --pool-rebalance-days 21 --start-date 20220101 --benchmark 000300.SH --holding-count 5 --rebalance-freq 5d --enhanced-profile up_low_breakout_v2 --experiment-tag advanced_ml_liquid500_rolling
```

运行后会额外输出：
- `rolling_liquidity_schedule.csv`
- `rolling_liquidity_summary.csv`

这两份文件用来检查：
- 每次重建日期
- 当期池子覆盖区间
- 每期实际池大小
- 历史滚动池的成员数量与换手变化

当前研究纪律：
- 执行端继续冻结为 `advanced_ml + liquid500 + next_open`
- `deep_alpha` 继续做高流动性专用研究
- 只有在**历史滚动高流动性股票池**下也能稳定跑赢，研究结果才允许进入执行候选

## 执行与研究口径分层（2026-03-21）
- 执行端使用：`advanced_ml + liquid500 + next_open`。
- 执行端股票池：`每日盘后更新一次 liquid500_latest.txt`，用于次日开盘执行。
- 正式研究端股票池：使用**历史滚动** `liquid500/liquid800`，默认每 `21` 个交易日重建一次。
- 这两套频率是故意不同的：
  - 执行端要尽量贴近明日真实可交易股票池，所以用每日最新结果。
  - 正式研究端要控制历史回看偏差与换池噪声，所以用历史滚动、低频重建。
- 另外保留一种“快速研究模式”：直接复用最新 `liquid500_latest.txt / liquid800_latest.txt` 做诊断与快筛；这类结果只能用于研究，不直接指导执行端。
- 后续任何候选策略要接近执行端，至少要依次通过：
  1. 历史滚动高流动性股票池正式验证；
  2. `next_open` 同口径样本外验证；
  3. 执行口径下的额外验证（必要时再做 shadow mode）。

## Deep Alpha Representation-Learning Upgrade
`deep_alpha` is now moving from local structural tweaks to a two-stage representation-learning workflow.

New workflow:
1. `patch-based masked self-supervised pretraining`
2. `return ranking fine-tune`

### Pretraining Example
```bash
python daily_research/deep_alpha/pretrain_deep_alpha_encoder.py --data-source tq --rolling-liquidity-pool liquid500 --pool-rebalance-days 21 --start-date 20220101 --benchmark 000300.SH --lookback-window 120 --patch-len 5 --hidden-dim 96 --transformer-heads 4 --transformer-layers 2 --mask-ratio 0.40 --epochs 12 --min-epochs 8 --auto-extend-undertrained --epoch-extend-step 4 --max-total-epochs 20 --valid-days 252 --pretrain-valid-days 63 --experiment-tag deep_alpha_pretrain_liq500
```

Adaptive-budget note:
- pretraining can now auto-extend when diagnostics still report `undertrained`
- the run stays inside one process and keeps optimizer / scheduler continuity
- use `--no-auto-extend-undertrained` if you want a hard fixed cap for a strict experiment

### Fine-Tuning Example
```bash
python daily_research/deep_alpha/run_deep_alpha_research.py --data-source tq --rolling-liquidity-pool liquid500 --pool-rebalance-days 21 --start-date 20220101 --benchmark 000300.SH --encoder-family patch_transformer --patch-len 5 --pretrained-encoder-path daily_research/output/deep_alpha_pretrain_liq500/pretrained_encoder.pt --return-loss-mode regression --return-target-transform raw --ranking-loss-weight 0.5 --listwise-loss-weight 0.25 --score-risk-mode state_gate --epochs 8 --min-epochs 4 --train-eval-window-days 126 --valid-days 252 --experiment-tag deep_alpha_pretrained_liq500
```

Current discipline:
- Execution stays frozen at `advanced_ml + liquid500 + next_open`
- `deep_alpha` is judged only under the formal framework: rolling `liquid500`, `next_open`, multi-window walk-forward
- We only revisit `shadow mode` if the pretrained version is more stable than the current `shared` baseline

## Training Safety And Runtime
- `deep_alpha` now records training diagnostics in `metrics.json` for both pretraining and fine-tuning:
  - `best_epoch`
  - `best_valid_loss`
  - `stopped_early`
  - `still_improving`
  - `status`
- This is used to avoid another “undertrained recipe gets killed too early” mistake.
- On the current local machine (`Ryzen 7 4800H + RTX 2060 6GB`, Windows), the safe runtime profile now auto-tunes:
  - `patch_transformer` batch size cap: `192`
  - `num_workers` cap: `2`
  - `prefetch_factor`: `1`
  - `AMP`: on when CUDA is available
- If you need to bypass these caps for a controlled test, add `--no-safe-runtime-profile`.
