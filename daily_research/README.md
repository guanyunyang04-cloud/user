# Daily Research

## 1. 项目定位
`daily_research` 是当前工作区的日线研究主线，负责三类任务：

1. 可解释基线与 `advanced_ml` 研究；
2. 盘后更新模型、生成次日开盘执行建议；
3. `deep_alpha` 表示学习研究。

当前项目已经从“连续堆实验”整理成三层分工：

- `baseline/`
  - 可解释基线与 `advanced_ml` 主线；
- `execution/`
  - 盘后更新、次日开盘执行建议；
- `deep_alpha/`
  - 表示学习研究，不直接进入执行端。

当前冻结结论：

- 执行端正式主线：`advanced_ml + liquid500 + next_open`
- `deep_alpha` 仍是研究主线，但尚未通过正式晋级门槛，不能进入执行端或 `shadow mode`

## 2. 文档分工
- `README.md`
  - 面向当前状态与使用入口；
- `daily_research_plan.md`
  - 面向当前主线、未解问题与下一步优先级；
- `research_log.md`
  - 面向按时间沉淀的实验记录、结果与结论。

推荐阅读顺序：

1. 先看 `README.md`，确认现在项目在做什么；
2. 再看 `daily_research_plan.md`，确认接下来要做什么；
3. 最后查 `research_log.md`，回看具体实验过程与证据。

## 3. 当前项目状态

### 3.1 可执行主线
- 执行口径固定为：
  - `advanced_ml + liquid500 + next_open`
- 执行端股票池：
  - 每日盘后更新 `liquid500_latest.txt`
- 执行方式：
  - 盘后出计划，次日开盘手工执行

### 3.2 正式研究口径
- 股票池：
  - 历史滚动 `liquid500 / liquid800`
- 换池频率：
  - 默认每 `21` 个交易日重建一次
- 成交假设：
  - `next_open`
- 验证方式：
  - 多窗口 walk-forward

### 3.3 `deep_alpha` 当前判断
- 已从概念验证阶段进入正式研究主线；
- 当前最重要的方向是：
  - `patch-based masked pretraining + return ranking fine-tune`
- 但截至目前仍未形成可晋级执行端的稳定结论。

## 4. 目录结构
- `daily_research/baseline/`
  - 可解释基线、`advanced_ml`、回测与分析脚本
- `daily_research/execution/`
  - 流动性股票池更新、模型更新、交易计划生成
- `daily_research/deep_alpha/`
  - 序列样本、模型、训练器、预训练与正式研究入口
- `daily_research/tools/`
  - 文档检查、工作区体检、清理与归档工具
- `daily_research/output/`
  - 近期研究输出热区
- `daily_research/cache/`
  - 近期研究缓存热区
- `daily_research/archive/`
  - 冷归档区

## 5. 当前推荐工作流

### 5.1 执行端日常流程
先更新高流动性股票池：

```bash
python daily_research/execution/update_liquid_pool.py --start-date 20240101
```

再更新执行模型：

```bash
python daily_research/execution/update_model.py --data-source tq --stocks-file daily_research/execution/universe/liquid500_latest.txt --start-date 20210101 --benchmark 000300.SH --regime-max-annual-vol 0.32 --regime-quadrants trend_up_low_vol,trend_up_high_vol --ml-target-horizons 5,10,20 --ml-horizon-weights 5:0.2,10:0.3,20:0.5 --ml-train-window-days 504
```

最后生成次日开盘执行建议：

```bash
python daily_research/execution/run_trade_plan.py --data-source tq --stocks-file daily_research/execution/universe/liquid500_latest.txt --start-date 20210101 --benchmark 000300.SH --holding-count 5 --rebalance-freq 5d --positions-file daily_research/execution/current_positions.csv --cash 200000 --regime-max-annual-vol 0.32 --regime-quadrants trend_up_low_vol,trend_up_high_vol --max-style-weight 0.50 --experiment-tag today_trade_plan
```

主要输出：

- `daily_research/execution/output/latest_trade_plan.txt`
- `daily_research/execution/output/<tag>/actions_today.csv`
- `daily_research/execution/output/<tag>/watchlist.csv`
- `daily_research/execution/output/<tag>/plan_summary.json`

### 5.2 `advanced_ml` 正式研究
```bash
python daily_research/baseline/run_advanced_daily_research.py --data-source tq --rolling-liquidity-pool liquid500 --pool-rebalance-days 21 --start-date 20220101 --benchmark 000300.SH --holding-count 5 --rebalance-freq 5d --enhanced-profile up_low_breakout_v2 --experiment-tag advanced_ml_liquid500_rolling
```

用途：

- 回看 `advanced_ml` 在正式研究口径下的表现；
- 与 `deep_alpha` 做同口径比较；
- 检查执行主线是否仍然稳健。

### 5.3 `deep_alpha` 预训练与正式研究
预训练：

```bash
python daily_research/deep_alpha/pretrain_deep_alpha_encoder.py --data-source tq --rolling-liquidity-pool liquid500 --pool-rebalance-days 21 --start-date 20220101 --benchmark 000300.SH --lookback-window 120 --patch-len 5 --hidden-dim 96 --transformer-heads 4 --transformer-layers 2 --mask-ratio 0.40 --epochs 12 --min-epochs 8 --auto-extend-undertrained --epoch-extend-step 4 --max-total-epochs 20 --valid-days 252 --pretrain-valid-days 63 --experiment-tag deep_alpha_pretrain_liq500
```

正式 fine-tune：

```bash
python daily_research/deep_alpha/run_deep_alpha_research.py --data-source tq --rolling-liquidity-pool liquid500 --pool-rebalance-days 21 --start-date 20220101 --benchmark 000300.SH --encoder-family patch_transformer --patch-len 5 --pretrained-encoder-path daily_research/output/deep_alpha_pretrain_liq500/pretrained_encoder.pt --return-loss-mode regression --return-target-transform raw --ranking-loss-weight 0.5 --listwise-loss-weight 0.25 --score-risk-mode state_gate --epochs 8 --min-epochs 4 --train-eval-window-days 126 --valid-days 252 --experiment-tag deep_alpha_pretrained_liq500
```

说明：

- `deep_alpha` 只能在正式研究口径下评价；
- 任何“只修复弱窗口、但破坏强窗口”的方案都不晋级。

## 6. 当前关键结论

### 6.1 基线与 `advanced_ml`
- `score + 5d` 明显优于 `equal + 1d`
- 市场状态过滤有效
- 当前上涨白名单固定为：
  - `trend_up_low_vol`
  - `trend_up_high_vol`
- `up_low_breakout_v2` 是当前唯一接近可用的增强候选
- `trend_up_high_vol` 的增强版本仍不稳健
- 执行端继续冻结为：
  - `advanced_ml + liquid500 + next_open`
- 但截至 `2026-03-19` 的最新正式子窗口 `2025-09-05 -> 2026-03-19`，执行主线已经出现明显转弱：
  - 超额收益 `-28.78%`
  - 超额 Sharpe `-1.382`
  - 因此执行方向研究已经重启，但当前执行默认值暂不切换
- 第一轮正式修复扫描已经完成：
  - 工具：`daily_research/baseline/scan_execution_repair_candidates.py`
  - 当前最稳的修复候选是仅在 `trend_up_low_vol` 下把集成权重调为 `ml=0.55 / none=0.25 / v2=0.20`
  - 在同口径正式扫描里，最新弱窗口超额收益从约 `-28.95%` 改善到 `-10.76%`，超额 Sharpe 从约 `-1.381` 改善到 `-0.583`
  - 同时全样本超额 Sharpe 仍约 `1.035`，超额最大回撤从约 `-41.53%` 收敛到约 `-31.22%`
  - 但该候选尚未把弱窗口修回正收益，因此暂不改执行默认值，只作为下一轮正式修复候选
- 第二轮正式扫描继续聚焦两类变量：
  - 在原 `regime_ma_window=60` 框架里，`0.55 / 0.25 / 0.20` 仍然是 `0.55 ~ 0.60` 区间内最优权重点；
  - 但更强的新信号来自状态启停：把 `regime_ma_window` 从 `60` 收到 `50` 后，`baseline` 本身就在三个正式窗口里同时优于当前执行主线
- `regime_ma_window=50` 的 `baseline` 结果：
  - 全样本超额收益约 `580.31%`
  - 全样本超额 Sharpe 约 `1.841`
  - 最新弱窗口超额收益约 `-10.87%`
  - 最新弱窗口超额 Sharpe 约 `-0.559`
- 当前判断：
  - 执行端修复的第一优先级已经转成“优先复验 `regime_ma_window=50`”
  - `trend_up_low_vol` 的分段权重微调继续保留，但已降为次优先级

### 6.2 `deep_alpha`
- `Transformer` 优于 `GRU`
- 表示学习主线已经收敛到：
  - 预训练编码器
  - 排序 fine-tune
  - 正式框架 walk-forward 验证
- 当前 `12` 轮预训练版本比早期版本更强，但仍未稳定胜过现有 `shared` 基线
- 预训练预算不足会扭曲结论，因此必须结合训练诊断一起判断

## 7. 策略晋级门槛
任何新方案想接近执行端，至少要同时满足：

1. 使用历史滚动高流动性股票池，而不是今天的静态股票池回看历史；
2. 使用 `next_open` 交易口径；
3. 通过多窗口 walk-forward；
4. 训练诊断不再显示关键窗口 `undertrained`；
5. 不能只在单个弱窗口改善、却破坏原本更强的窗口。

未满足以上条件时：

- 不进入执行端；
- 不进入 `shadow mode`；
- 只保留在研究分支。

## 8. 产物、归档与维护

### 8.1 热区与冷区
- 热区：
  - `daily_research/output/`
  - `daily_research/cache/`
- 冷区：
  - `daily_research/archive/output/`
  - `daily_research/archive/cache/`

当前规则：

- `output/` 保留近期实验目录与近期根目录汇总文件；
- 大体积哈希缓存按桶归档；
- 小体积高复用控制缓存默认常驻热区。

归档策略文件：

- `daily_research/archive_policy.json`

归档说明：

- `daily_research/archive/README.md`

### 8.2 常用维护命令
文档编码检查：

```bash
python daily_research/tools/doc_guard.py check
```

工作区体检：

```bash
python daily_research/tools/workspace_maintenance.py report
```

执行端收益体检：

```bash
python daily_research/tools/execution_health_check.py
```

执行端正式修复扫描：

```bash
python daily_research/baseline/scan_execution_repair_candidates.py --data-source tq --start-date 20220101 --benchmark 000300.SH --rolling-liquidity-pool liquid500 --pool-rebalance-days 21 --pool-adv-window 20 --holding-count 5 --rebalance-freq 5d --regime-max-annual-vol 0.32 --regime-quadrants trend_up_low_vol,trend_up_high_vol --ml-target-horizons 5,10,20 --ml-horizon-weights 5:0.2,10:0.3,20:0.5 --ml-train-window-days 504 --ml-model-family histgb
```

执行端第二轮扫描：

```bash
python daily_research/baseline/scan_execution_repair_candidates.py --candidate-set round2_weights --data-source tq --start-date 20220101 --benchmark 000300.SH --rolling-liquidity-pool liquid500 --pool-rebalance-days 21 --pool-adv-window 20 --holding-count 5 --rebalance-freq 5d --regime-max-annual-vol 0.32 --regime-quadrants trend_up_low_vol,trend_up_high_vol --ml-target-horizons 5,10,20 --ml-horizon-weights 5:0.2,10:0.3,20:0.5 --ml-train-window-days 504 --ml-model-family histgb
```

只清理可再生缓存：

```bash
python daily_research/tools/workspace_maintenance.py clean --targets pycache --apply
```

预演归档候选：

```bash
python daily_research/tools/workspace_maintenance.py archive
```

正式执行归档：

```bash
python daily_research/tools/workspace_maintenance.py archive --apply
```

### 8.3 文档维护规则
- 文档统一使用 UTF-8；
- `README.md` 只保留“当前状态 + 使用入口”；
- `daily_research_plan.md` 只保留“当前主线 + 下一步计划”；
- `research_log.md` 只保留“时间顺序的研究记录”；
- 不再把同一批内容同时写进三份文档；
- 中文内容不再用容易引入编码问题的 shell 重定向直接追加。
