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
  - 执行端默认值暂不切换，继续冻结为：`advanced_ml + liquid500 + next_open`
  - 新的“最强正式修复候选”已经从第一轮的 `ma60 + up_low_ml55_none25_v220`，更新为第二轮的 `ma50 baseline`
  - `ma60 + up_low_ml55_none25_v220` 仍保留为次一级备选
- 第三轮 `ma50 baseline` 专项复验已经完成：
  - `ma50 baseline` 在同一轮新产物中同时优于当前 `ma60 baseline` 与次一级备选 `ma60 + up_low_ml55_none25_v220`
  - 相对这两个对照，`ma50 baseline` 都是在 `17` 个季度里赢下 `13` 个季度，且最强季度贡献占正向季度总优势约 `20%`，不是单季度异常放大
  - 在 `ma50` 框架里再叠加 `up_low_ml55_none25_v220` 只把最新弱窗口从约 `-10.87% / -0.559` 轻微改善到约 `-9.31% / -0.509`，但会把全样本超额 Sharpe 从约 `1.841` 拉回约 `1.392`
  - 因此下一步不急着进入第四轮权重微调，而是先做 `ma50` 边界稳定性与轻量风险控制复验
- 第五轮 `ma50` 边界稳定性与轻量风险控制复验已经完成：
  - 在 `ma48 / ma49 / ma50 / ma51 / ma52` 中，聚合指标最强的候选跑到了 `ma48 + take20`
  - `ma48 + take20` 的全样本超额 Sharpe 约 `2.239`，最近完整窗口约 `116.70% / 2.930`，最新弱窗口约 `+3.24% / 0.157`
  - 但它相对 `ma50 baseline` 只在 `17` 个季度里的 `5` 个季度更强，且 `2025Q3` 一季就占了约 `54.92%` 的正向季度总优势；相对 `ma48 baseline` 的 `take20` 增益也高度集中
  - 这说明：`ma48` 的确跑出了更高上限，但当前还像“局部高收益候选”，不够稳定到直接替代 `ma50 baseline`
  - `ma50` 框架内的轻量风控只能带来很小的弱窗口改善，不足以改变当前主判断
  - 因此当前不更新执行默认值，也不把头号正式修复候选从 `ma50 baseline` 改掉；下一步改为对 `ma48` 做专门稳定性复验
- 第六轮 `ma48` 专门稳定性复验与季度集中度诊断已经完成：
  - `ma48 baseline` 的全样本超额 Sharpe 约 `2.202`，最近完整窗口约 `112.46% / 2.790`，最新弱窗口约 `+3.24% / 0.157`；左邻 `ma47 baseline` 也保持了约 `1.955` 的全样本超额 Sharpe 和约 `+1.79% / 0.106` 的最新弱窗口
  - 但 `ma48 baseline` 相对 `ma50 baseline` 只在 `17` 个季度里的 `5` 个季度更强，最佳季度 `2025Q3` 占正向季度总优势约 `53.86%`，Top3 季度占比约 `94.20%`
  - `ma48 baseline` 相对 `ma47 baseline` 虽然仍有总优势，但最佳季度 `2025Q3` 占正向季度总优势约 `71.02%`，说明 `47 -> 48` 的新增优势本身也高度集中
  - `ma48 + take20` 相对 `ma48 baseline` 只在 `17` 个季度里的 `2` 个季度更强，总增益很小且高度集中，不足以单独晋级
  - 这说明：`ma48` 不再像纯随机孤点，左侧 `ma47/48` 带值得继续复验；但当前还不足以把头号正式修复候选从 `ma50 baseline` 改掉
- 第七轮 `ma47/48` 左侧边界带稳定性复验已经完成：
  - `ma48 baseline` 的全样本超额 Sharpe 约 `2.202`，高于 `ma47 baseline` 的约 `1.955` 和 `ma50 baseline` 的约 `1.841`
  - 但 `2025Q3` 全季 `66` 个交易日都处于 `trend_up_low_vol`，说明这轮集中来源不是状态切换，而是同一状态内的选股与换仓差异
  - `ma48` 相对 `ma47` 的 `2025Q3` compound 超额边际约 `+62.43%`，相对 `ma50` 约 `+55.31%`；其中 `2025-09` 的月度边际最强，分别约 `+16.83%` 和 `+18.84%`
  - `ma48` 相对 `ma47` 的 Top5 正向日占 `2025Q3` 正向日总优势约 `41.12%`，相对 `ma50` 的 Top5 正向日占比约 `51.25%`
  - `ma48` 相对 `ma47 / ma50` 的平均持仓重叠 Jaccard 都在约 `0.69 ~ 0.70`，且约 `73% ~ 76%` 的日期至少重合 `4` 个名字，说明优势主要来自少数持仓槽位替换，而不是整套组合重写
- 第八轮 `trend_up_low_vol` 关键槽位复现诊断已经完成：
  - `ma48` 相对 `ma47` 的 `2025Q3` top5 槽位签名是 `301389.SZ / 301488.SZ / 603716.SH / 300436.SZ / 300486.SZ`
  - `ma48` 相对 `ma50` 的 `2025Q3` top5 槽位签名是 `301357.SZ / 300436.SZ / 301488.SZ / 300476.SZ / 601728.SH`
  - 这两组签名在其他 `trend_up_low_vol` 季度里的完整复现次数都是 `0`，放宽到“任意重叠”后也仍是 `0`
  - 相对 `ma47`，这 5 个 Q3 槽位名字在其他季度里连 `0.1%` 以上的平均正权重差都没有再次出现；相对 `ma50` 只有 `301488.SZ / 300476.SZ / 601728.SH` 各自零星出现 `1` 次，且只有 `301488.SZ` 落在正边际季度
  - 这说明：`2025Q3` 的优势更像“季度特定槽位命中”，还不是已经能跨季度稳定复放的固定签名
- 当前执行决策同步为：
  - 执行端默认值继续冻结为：`advanced_ml + liquid500 + next_open`
  - `ma50 baseline` 继续作为当前头号正式修复候选
  - `ma60 + up_low_ml55_none25_v220` 继续保留为次一级备选
  - `ma47/48` 左侧边界带继续保留为高收益研究分支，但当前按“季度特定槽位命中”看待；其中 `ma48 baseline` 仍是当前数值最强点，`ma48 + take20` 作为附加轻量风控版本保留
  - 下一步不直接切执行默认值，也不继续盲扫边界，而是先把 `2025Q3` 的槽位替换抽象成更稳定的 `trend_up_low_vol` 信号逻辑；如果抽象不出来，就停止该分支晋级
- 第九轮 `trend_up_low_vol` 信号逻辑抽象诊断已经完成：
  - 从 `2025Q3` 槽位替换里自动抽出的宽口径 `slot_logic` 因子为：`volatility_contraction / ma_gap_20_60 / volatility_20 / price_volume_divergence / mom_20 / long_regime_flag`，但它的季度 RankIC 均值约 `-0.009`，相对 `v2` 仅在 `3` 个非 `2025Q3` 季度更强，且这些正向改善约 `95.71%` 集中在单一季度
  - 更克制的 `slot_logic_shared` 最终只剩单因子 `volatility_contraction`；它的季度 RankIC 均值约 `0.044`，相对 `v2` 虽在 `5` 个非 `2025Q3` 季度更强，但正向改善约 `82.42%` 仍集中在 `2024Q3`，且在焦点季度 `2025Q3` 反而落后 `v2` 约 `-0.060`
  - 对 `ma48_vs_ma47 / ma48_vs_ma50` 的季度槽位复放，`slot_logic_shared` 在非焦点季度里分别只出现 `2 / 3` 个正向槽位边际，且没有在 `ma48` 其他正边际季度上形成稳定对齐；`slot_logic` 也没有同时满足“可解释槽位”与“可解释未来超额”
  - 这说明：`2025Q3` 的槽位替换目前还抽象不成可跨季度复放的稳定 `trend_up_low_vol` 信号逻辑，因此 `ma47/48` 左侧边界带停止晋级执行端，降级为纯研究旁支；下一步研究重心回到 `ma50 baseline` 内部升级，优先做状态专属 horizon 权重
- 第十轮 `ma50` 状态专属 horizon 权重首轮扫描已经完成：
  - 只调整 `trend_up_high_vol` 的 `5/10/20` 权重，只会改动全样本高波段表现，最新弱窗口 `2025-09-05 -> 2026-03-19` 保持在约 `-10.87% / -0.559`，没有任何净改善
  - 一旦同时动到 `trend_up_low_vol` 的 horizon 配比，最新弱窗口会明显恶化，最差一档约退到 `-18.96% / -0.927`
  - 这说明：在 `ma50 baseline` 下，状态专属 horizon 权重不是当前弱窗口修复的主增量来源，这条线暂时降级
- 第十一轮 `ma50` 状态专属 ensemble 权重首轮扫描已经完成：
  - 纯 `trend_up_high_vol` 的权重改法，对最新弱窗口几乎没有影响
  - 首个真正动到弱窗口的候选是 `trend_up_low_vol=ml0.60/none0.25/v20.15` 并叠加 `trend_up_high_vol=ml0.80/none0.15/v20.05`，它把最新弱窗口小幅改善到约 `-10.04% / -0.550`
  - 但这条候选的全样本超额 Sharpe 从 `1.841` 回落到约 `1.753`，回撤也更差，因此还不是可晋级的干净升级
- 第十三轮 `ma50` 状态专属 ensemble 第二轮精扫已经完成：
  - 本轮先隔离 `trend_up_low_vol`，固定 `trend_up_high_vol` 回到 baseline，只在 `ml=0.57 ~ 0.62` 一带细扫
  - `up_low_ml62_none23_v215` 是当前弱窗口修复最强点：最新弱窗口约修到 `-7.67% / -0.413`，最近完整窗口约 `91.04% / 2.279`
  - `up_low_ml61_none24_v215` 是相对更平衡的候选：最新弱窗口约 `-8.05% / -0.451`，最近完整窗口约 `91.54% / 2.337`
  - 但两者的全样本超额 Sharpe 仍都低于 baseline，分别约 `1.775 / 1.792` 对 `1.869`，且最强季度都集中在 `2026Q1`
  - 这说明：隔离 `trend_up_low_vol` 之后，ensemble 权重线确实还有信息量，但当前仍是“修弱窗口要付出全样本代价”，还不是可直接替换默认执行的干净升级
- 第十四轮 `ma50` ensemble 候选季度集中度与 `2026Q1` 归因诊断已经完成：
  - `up_low_ml62_none23_v215` 相对 baseline 的最佳季度 `2026Q1` 占正向季度总优势约 `45.27%`，Top3 季度占比约 `81.16%`
  - `up_low_ml61_none24_v215` 更集中，`2026Q1` 占正向季度总优势约 `52.20%`，Top3 季度占比约 `86.74%`
  - 两条候选在 `2026Q1` 的 Top5 正向日占比都约 `79%`，平均持仓重叠 Jaccard 仍约 `0.70`，说明增益主要来自少数日期放大和少数槽位替换
  - 月度上，两条候选的增量都主要集中在 `2026-01`；`2026-02` 反而回吐，`2026-03` 只有小修复
  - 这说明：这条 `trend_up_low_vol` ensemble 权重线已经确认仍属季度集中驱动，不再继续晋级执行端
- 第十二轮 `ma50` 波动阈值微调已经完成：
  - `regime_max_annual_vol=0.30 ~ 0.34` 五个点的正式收益指标完全一致
  - 复核 `regime_state.csv` 与 `actions.csv` 后确认：这些阈值只改动了少数 `trend_down_low_vol / trend_down_high_vol` 标签，`regime_on` 完全不变，实际持仓与交易动作也完全一致
  - 这说明：在当前 `ma50 + liquid500 + next_open` 框架里，简单波动阈值微调没有有效敏感度，这条线不再列为第一优先级
- 当前执行决策现更新为：
  - 执行端默认值已切换为：`advanced_ml (ma50 baseline) + liquid500 + next_open`
  - `ma50 baseline` 已从“头号正式修复候选”晋级为当前执行默认口径
  - `ma60 + up_low_ml55_none25_v220` 继续保留为次一级回滚备选
  - `ma47/48` 左侧边界带停止晋级执行端，降级为纯研究旁支；已有结论和正式产物保留，但不再作为当前执行修复候选
  - `up_low_ml62_none23_v215` 与 `up_low_ml61_none24_v215` 的正式诊断已经完成，并确认仍属季度集中驱动；两者作为研究附录保留，但不再继续晋级执行端
- 下一步若继续做执行端 ML 增量优化，优先切到 `ma50` 口径下的模型族对照或状态专属模型研究，而不是继续扫这条 ensemble 权重线
- `2026-03-24` 已完成当前执行口径 `ma50 + rolling liquid500 + next_open` 的正式模型族对照，并补跑完成了原本耗时最长的 `etr`：
  - `lgbm`：全样本超额收益约 `887.75%`，全样本超额 Sharpe 约 `2.300`；最近完整窗口约 `127.99% / 2.816`；最新弱窗口约 `9.79% / 0.475`
  - `histgb`：全样本超额收益约 `388.95%`，全样本超额 Sharpe 约 `1.487`；最近完整窗口约 `61.33% / 1.402`；最新弱窗口约 `-8.46% / -0.423`
  - `etr`：全样本超额收益约 `274.63%`，全样本超额 Sharpe 约 `1.443`；最近完整窗口约 `41.74% / 1.491`；最新弱窗口约 `6.46% / 0.433`
- `lgbm` 相对当前默认 `histgb` 在 `17` 个季度里有 `9` 个季度更强、`8` 个季度持平、`0` 个季度更弱；最佳季度 `2025Q3` 仅占正向季度总优势约 `29.67%`，不是单季度孤点。
- 因此当前执行默认模型仍保持 `histgb`，但 `lgbm` 已正式晋级为当前 `ma50` 执行口径下的头号模型族升级候选；`etr` 正式补跑后确认不构成头号 challenger，只保留为研究附录。
- `2026-03-24` 已完成 `ma50 + lgbm` 的切换前复核与第一轮执行端烟测：
  - 独立候选产物已生成在 `daily_research/output/ma50_lgbm_switch_review/models/ma50_lgbm_candidate.joblib`
  - 候选 `lgbm` 的滚动验证摘要高于当前默认 `histgb`：`full IC 0.121 > 0.107`，`recent126d IC 0.152 > 0.146`，`recent63d IC 0.219 > 0.212`
  - 独立烟测已跑通 `update_model.py -> run_trade_plan.py`，模型新鲜度为 `fresh`，信号日 `2026-03-23` 与当前默认计划给出同一笔卖出动作
  - 当时唯一未覆盖的是 `regime_on` 下的真实买入路径
- `2026-03-24` 已补完 `regime_on` 日期 `2026-03-11` 的点时烟测，并顺手修复了执行端一个真实买入 bug：
  - 原异常不是模型问题，而是 `generate_daily_trade_plan.py` 在买入腿里把 `target_weight` 当成了 `target_value`，导致空账户场景下可能出现“有目标仓位但无买单”
  - 修复后，`histgb` 在同一口径下给出 `4` 笔买入：`002470.SZ / 688800.SH / 300617.SZ / 002843.SZ`
  - 修复后，`lgbm` 在同一口径下给出 `4` 笔买入：`002470.SZ / 000510.SZ / 688800.SH / 300739.SZ`
  - 两边点时产物都保持 `fresh`，买入路径与之前 `2026-03-23` 的 `regime_off` 卖出路径一起，已经把切换前执行链路补全
- 因此当前默认模型从研究和执行两侧都已具备正式切换到 `lgbm` 的条件；若继续推进执行端 ML 增量优化，下一步应先做默认模型切换，再决定是否还有必要进入状态专属模型研究。
- 本轮产物汇总在：
  - `daily_research/output/advanced_ml_model_family_compare_20260323_formal_ma50_execution`
  - `model_family_compare_summary.csv`
  - `model_family_compare_report.json`
  - `model_family_compare_report.md`
  - 默认模型产物 `latest_ml_model.json` 已补充 `validation_summary`；执行端也已补上模型新鲜度保护，默认 `1` 个交易日滞后提醒、`3` 个交易日滞后拦截

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

执行端第三轮 `ma50` 专项复验汇总：

```bash
python daily_research/tools/ma50_revalidation_report.py --current-baseline-run daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round3_ma60_pair/baseline --backup-run daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round3_ma60_pair/up_low_ml55_none25_v220 --candidate-run daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round3_ma50_pair/baseline
```

执行端第五轮边界 / 轻量风控汇总（含 `ma47` 左邻补跑）：

```bash
python daily_research/tools/ma_boundary_risk_report.py --scan-dirs daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round5_ma47_boundary_risk daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round5_ma48_boundary_risk daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round5_ma49_boundary_risk daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round5_ma50_boundary_risk daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round5_ma51_boundary_risk daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round5_ma52_boundary_risk
```

执行端第六轮 `ma48` 稳定性 / 季度集中度诊断：

```bash
python daily_research/tools/ma48_stability_report.py --anchor-run daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round5_ma50_boundary_risk/baseline --candidate-run daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round5_ma48_boundary_risk/baseline --variant-run daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round5_ma48_boundary_risk/take20 --left-neighbor-run daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round5_ma47_boundary_risk/baseline --right-neighbor-run daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round5_ma49_boundary_risk/baseline --output-dir daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round6_ma48_stability
```

执行端第七轮 `ma47/48` 左侧边界带与 `2025Q3` 集中来源诊断：

```bash
python daily_research/tools/ma4748_band_report.py --ma47-run daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round5_ma47_boundary_risk/baseline --ma48-run daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round5_ma48_boundary_risk/baseline --ma50-run daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round5_ma50_boundary_risk/baseline --ma48-variant-run daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round5_ma48_boundary_risk/take20 --focus-quarter 2025Q3 --output-dir daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round7_ma4748_band
```

执行端第八轮 `trend_up_low_vol` 槽位复现诊断：

```bash
python daily_research/tools/trend_up_low_vol_slot_replay_report.py --ma47-run daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round5_ma47_boundary_risk/baseline --ma48-run daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round5_ma48_boundary_risk/baseline --ma50-run daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round5_ma50_boundary_risk/baseline --focus-quarter 2025Q3 --focus-quadrant trend_up_low_vol --top-n 5 --min-positive-delta 0.001 --output-dir daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round8_slot_replay
```

执行端第九轮 `trend_up_low_vol` 信号逻辑抽象诊断：

```bash
python daily_research/tools/trend_up_low_vol_signal_logic_report.py --ma47-run daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round5_ma47_boundary_risk/baseline --ma48-run daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round5_ma48_boundary_risk/baseline --ma50-run daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round5_ma50_boundary_risk/baseline --focus-quarter 2025Q3 --focus-quadrant trend_up_low_vol --top-days 10 --slot-top-n 5 --factor-top-k 6 --output-dir daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round9_signal_logic
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
