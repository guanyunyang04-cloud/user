# Daily Research

## 1. 项目定位
`daily_research` 是当前正式维护的日线研究与执行工作区，负责三类任务：

1. `baseline / advanced_ml` 可解释研究与正式回测；
2. 盘后更新执行模型，生成次日开盘手工执行建议；
3. `deep_alpha` 表示学习主线研究。

当前默认执行口径已经收敛为：

- `advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`
- 调仓语义：日频目标更新，主入口默认 `rebalance_freq=1d`
- 执行方式：盘后生成计划，次日开盘手工执行

当前项目判断：

- `deep_alpha` 是最重要的研究主线，但还没有通过正式晋级门槛；
- `none / v2` 继续保留为规则层先验，不扩成新的执行主线；
- 连续状态分数继续保留为诊断层与风险附录，不进入当前默认执行逻辑。

## 2. 阶段回顾
自立项以来，项目大致经历了五个阶段：

1. `2026-03-14 ~ 2026-03-18`
   - 建起 `baseline`，完成 `score + 5d`、市场状态过滤和 `up_low_breakout_v2` 等第一阶段收敛。
2. `2026-03-18 ~ 2026-03-20`
   - 引入 `advanced_ml`，并把执行口径统一到“盘后信号 -> 次日开盘执行”。
3. `2026-03-20 ~ 2026-03-22`
   - `deep_alpha` 进入正式研究框架：滚动高流动性股票池、`next_open`、多窗口 walk-forward。
4. `2026-03-22`
   - 开始做项目治理：入口去重、归档规则、工作区体检、文档分工。
5. `2026-03-23 ~ 2026-03-24`
   - 执行默认口径切到 `ma50 baseline + lgbm`，并把当前主线正式标准化为“日频目标更新”。

## 3. 文档分工
- `daily_research/README.md`
  - 只解决“现在项目是什么、怎么进主入口”。
- `daily_research/daily_research_plan.md`
  - 只解决“当前默认决策是什么、下一步优先级是什么”。
- `daily_research/research_log.md`
  - 只保留按时间推进的实验记录、证据与结论。
- `daily_research/execution/README.md`
  - 只讲执行端目录职责、默认参数与日常操作细节。

推荐阅读顺序：

1. 先看本文件，确认当前主线与入口；
2. 再看 `daily_research_plan.md`，确认当前优先级与停止规则；
3. 需要追溯实验过程时，再查 `research_log.md`。

## 4. 当前结构
- `daily_research/baseline/`
  - 可解释基线、`advanced_ml`、正式回测与诊断脚本。
- `daily_research/execution/`
  - 股票池刷新、模型更新、交易计划生成。
- `daily_research/deep_alpha/`
  - 预训练、fine-tune 与正式研究入口。
- `daily_research/tools/`
  - 文档守卫、工作区体检、清理与归档工具。
- `daily_research/cache/`
  - 研究缓存热区。
- `daily_research/output/`
  - 研究产物热区。
- `daily_research/archive/`
  - 冷归档区。

## 5. 当前日常入口
执行端默认链路：

```bash
python daily_research/execution/update_liquid_pool.py --start-date 20240101
python daily_research/execution/update_model.py --data-source tq --start-date 20210101 --benchmark 000300.SH --regime-max-annual-vol 0.32 --regime-quadrants trend_up_low_vol,trend_up_high_vol
python daily_research/execution/run_trade_plan.py --data-source tq --start-date 20210101 --benchmark 000300.SH --holding-count 5 --max-style-weight 0.50
```

说明：

- `update_model.py` 默认注入：
  - `liquid500_latest.txt`
  - `regime_ma_window=50`
  - `ml-model-family=lgbm`
- `run_trade_plan.py` 默认注入：
  - `current_positions.csv`
  - `latest_ml_model.joblib`
  - `liquid500_latest.txt`
  - `regime_ma_window=50`

执行端完整说明见 `daily_research/execution/README.md`。

正式研究入口：

```bash
python daily_research/baseline/run_advanced_daily_research.py --data-source tq --rolling-liquidity-pool liquid500 --pool-rebalance-days 21 --start-date 20210101 --benchmark 000300.SH --holding-count 5 --rebalance-freq 1d --regime-ma-window 50 --ml-model-family lgbm --experiment-tag advanced_ml_formal
```

`deep_alpha` 当前主线入口：

```bash
python daily_research/deep_alpha/pretrain_deep_alpha_encoder.py --data-source tq --rolling-liquidity-pool liquid500 --pool-rebalance-days 21 --start-date 20220101 --benchmark 000300.SH --lookback-window 120 --patch-len 5 --hidden-dim 96 --transformer-heads 4 --transformer-layers 2 --mask-ratio 0.40 --epochs 12 --min-epochs 8 --auto-extend-undertrained --epoch-extend-step 4 --max-total-epochs 20 --valid-days 252 --pretrain-valid-days 63 --experiment-tag deep_alpha_pretrain_liq500
python daily_research/deep_alpha/run_deep_alpha_research.py --data-source tq --rolling-liquidity-pool liquid500 --pool-rebalance-days 21 --start-date 20220101 --benchmark 000300.SH --encoder-family patch_transformer --patch-len 5 --pretrained-encoder-path daily_research/output/deep_alpha_pretrain_liq500/pretrained_encoder.pt --return-loss-mode regression --return-target-transform raw --ranking-loss-weight 0.5 --listwise-loss-weight 0.25 --score-risk-mode state_gate --rebalance-freq 1d --epochs 8 --min-epochs 4 --train-eval-window-days 126 --valid-days 252 --experiment-tag deep_alpha_pretrained_liq500
```

## 6. 当前已收敛结论
- 当前默认执行主线是：
  - `advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`
- 当前默认训练窗口是：
  - `ml_train_window_days=504`
- 当前默认市场门控是：
  - `regime_ma_window=50`
  - `regime_max_annual_vol=0.32`
  - `trend_up_low_vol,trend_up_high_vol`
- 当前默认模型已经从 `histgb` 切换到 `lgbm`；
- 当前主线不再按旧的“真 `score + 5d`”理解，而是按“日频目标更新”理解；
- `ma47/48` 左侧边界带、`v21_volume_contraction_015`、连续状态软调节都保留为研究附录，不进入默认执行层；
- `deep_alpha` 仍未完成最终正式判决。

## 7. 维护命令
- 文档结构检查：
  - `python daily_research/tools/doc_guard.py check`
- 工作区体检：
  - `python daily_research/tools/workspace_maintenance.py report`
- 归档预演：
  - `python daily_research/tools/workspace_maintenance.py archive`
- 安全清理 `__pycache__`：
  - `python daily_research/tools/workspace_maintenance.py clean --targets pycache --apply`

## 8. 维护规则
- 文档统一使用 UTF-8；
- `README.md` 只保留当前状态、入口和维护命令；
- `daily_research_plan.md` 只保留当前默认决策、优先级与停止规则；
- `research_log.md` 只保留时间顺序的实验记录；
- 不再把同一批信息同时追加到三份文档里；
- 新方案只有通过正式研究框架，才允许讨论接近执行端；
- 默认只清理可再生生成物，不直接删除模型产物、持仓快照和研究结论文件。
