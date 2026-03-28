# Daily Research

## 1. 项目定位
`daily_research` 是当前正式维护的日线研究与执行工作区，负责三类任务：

1. `baseline / advanced_ml` 的可解释研究、正式回测与执行复核；
2. 盘后更新离线模型，生成次日开盘手工执行建议；
3. `deep_alpha` 表示学习主线研究。

当前默认执行口径已经收敛为：

- `advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`
- 调仓语义：日频目标更新，主入口默认 `rebalance_freq=1d`
- 执行方式：盘后生成计划，次日开盘人工执行

## 2. 文档分工
- `daily_research/brain/README.md`
  - 只保留当前状态、主入口与文档边界。
- `daily_research/brain/project_map.md`
  - 只保留项目背景、已解决问题、当前瓶颈与未来方向，供协作快速建立上下文。
- `daily_research/brain/daily_research_plan.md`
  - 只保留当前默认决策、升级 shortlist、优先级与停止规则。
- `daily_research/brain/runtime_environment.md`
  - 只保留解释器、关键依赖与推荐调用方式。
- `daily_research/brain/research_log.md`
  - 只保留按时间顺序推进的实验记录、证据与结论。
- `daily_research/execution/README.md`
  - 只保留执行端目录职责、默认参数与日常操作细节。

以上五份正本文档现在统一集中在：

- `daily_research/brain/`
  - 作为 AI 工作流脑区目录使用

推荐阅读顺序：

1. 先看本文件，确认当前项目状态与主入口；
2. 再看 `daily_research/brain/project_map.md`，确认项目是如何演化到当前状态、真正瓶颈是什么；
3. 再看 `daily_research/brain/daily_research_plan.md`，确认当前待决策事项与优先级；
4. 需要落命令时，再看 `daily_research/brain/runtime_environment.md`；
5. 需要回溯证据时，再查 `daily_research/brain/research_log.md`。

## 3. 当前结构
- `daily_research/brain/`
  - AI 工作流脑区目录，集中承载当前状态、项目地图、计划、运行基线与研究日志。
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

## 4. 当前日常入口
执行端默认链路：

```bash
python daily_research/execution/update_liquid_pool.py --start-date 20240101
python daily_research/execution/update_model.py --data-source tq --start-date 20210101 --benchmark 000300.SH --regime-max-annual-vol 0.32 --regime-quadrants trend_up_low_vol,trend_up_high_vol
python daily_research/execution/run_trade_plan.py --data-source tq --start-date 20210101 --benchmark 000300.SH --holding-count 5 --max-style-weight 0.50
```

执行端完整说明见 `daily_research/execution/README.md`。

正式研究入口：

```bash
python daily_research/baseline/run_advanced_daily_research.py --data-source tq --rolling-liquidity-pool liquid500 --pool-rebalance-days 21 --start-date 20210101 --benchmark 000300.SH --holding-count 5 --rebalance-freq 1d --regime-ma-window 50 --ml-model-family lgbm --experiment-tag advanced_ml_formal
```

`deep_alpha` 当前主入口：

```bash
python daily_research/deep_alpha/pretrain_deep_alpha_encoder.py --data-source tq --rolling-liquidity-pool liquid500 --pool-rebalance-days 21 --start-date 20220101 --benchmark 000300.SH --lookback-window 120 --patch-len 5 --hidden-dim 96 --transformer-heads 4 --transformer-layers 2 --mask-ratio 0.40 --epochs 12 --min-epochs 8 --auto-extend-undertrained --epoch-extend-step 4 --max-total-epochs 20 --valid-days 252 --pretrain-valid-days 63 --experiment-tag deep_alpha_pretrain_liq500
python daily_research/deep_alpha/run_deep_alpha_research.py --data-source tq --rolling-liquidity-pool liquid500 --pool-rebalance-days 21 --start-date 20220101 --benchmark 000300.SH --encoder-family patch_transformer --patch-len 5 --pretrained-encoder-path daily_research/output/deep_alpha_pretrain_liq500/pretrained_encoder.pt --return-loss-mode regression --return-target-transform raw --ranking-loss-weight 0.5 --listwise-loss-weight 0.25 --score-risk-mode state_gate --rebalance-freq 1d --epochs 8 --min-epochs 4 --train-eval-window-days 126 --valid-days 252 --experiment-tag deep_alpha_pretrained_liq500
python daily_research/deep_alpha/run_minimal_matrix.py --phase backbone --root-tag deep_alpha_minimal_matrix_round1
```

## 5. 当前稳定结论
- 当前默认执行主线保持为：
  - `advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`
- 当前默认训练窗口保持为：
  - `ml_train_window_days=504`
- 当前默认状态门控保持为：
  - `regime_ma_window=50`
  - `regime_max_annual_vol=0.32`
  - `trend_up_low_vol,trend_up_high_vol`
- 当前默认 focus-state 集成权重保持为：
  - `trend_up_low_vol=ml:0.25,none:0.25,v2:0.50`
- 底层市场状态架构已升级为：
  - 兼容层继续保留 legacy `quadrant`
  - 底层同时输出连续 `trend/vol gap`、`trend_bucket`、`vol_bucket` 与 `market_state`
  - 上层入口已支持通过 `regime_state_selector` 切换状态标签来源
  - 当前 `state_alpha_profile` 这一层仍只支持 legacy `quadrant`，非 quadrant selector 会被显式拦截，避免静默失效
- 执行端当前默认值已于 2026-03-28 升级为：
  - `advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`
  - `trend_up_low_vol` 状态专属默认权重：`ml:0.25, none:0.25, v2:0.50`
- 这次切换依据当前代码口径的 formal R3 对照：
  - 静态 `v250`：`full_excess_sharpe = 0.759`，`weak_window_20250905_20260319_excess_sharpe = 1.073`，`trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe = 1.083`
  - 静态 `v255`：`0.675 / 0.281 / 0.094`
  - 当前没有动态控制器能同时压过新的 live 默认值与进攻对照
- 2026-03-28 之前基于旧 `market_features` 集合得出的 `v255` 偏强结论已失效；当前所有正式升级讨论统一以 live `v250` 默认值为基准。
- `none / v2` 保留为规则层先验，不扩成新的执行主线。
- 连续状态分数保留为诊断层，不进入当前默认执行软调节逻辑。
- `ma47/48` 左侧边界带、`v21_volume_contraction_015` 等高收益旁支保留在研究附录，不进入默认执行口径。
- `deep_alpha` 仍是正式研究主线之一，但尚未通过执行端晋级门槛。

## 6. 文档与产物边界
当前仍需主动维护的文档只有：

- `daily_research/brain/README.md`
  - 当前状态、主入口与文档边界
- `daily_research/brain/project_map.md`
  - 项目背景、当前瓶颈、未来方向与协作导航
- `daily_research/brain/daily_research_plan.md`
  - 当前默认决策、shortlist、优先级与停止规则
- `daily_research/brain/runtime_environment.md`
  - 解释器、依赖与运行口径
- `daily_research/brain/research_log.md`
  - 时间顺序实验记录
- `daily_research/execution/README.md`
  - 执行端日常操作

主要生成物边界：

- `cache/`
  - 可再生产物热区
- `output/`
  - 研究结果热区
- `archive/`
  - 冷归档区
- `execution/output/`
  - 每日计划输出

## 7. 维护命令
- 文档结构检查：
  - `python daily_research/tools/doc_guard.py check`
- 工作区体检：
  - `python daily_research/tools/workspace_maintenance.py report`
- 归档预演：
  - `python daily_research/tools/workspace_maintenance.py archive`
- 执行归档：
  - `python daily_research/tools/workspace_maintenance.py archive --apply`
- 安全清理 `__pycache__`：
  - `python daily_research/tools/workspace_maintenance.py clean --targets pycache --apply`

## 8. 维护规则
- 文档统一使用 UTF-8。
- 不再把同一批“当前状态 / 当前决策 / 环境基线”同时写进多份文档。
- 新方案只有通过正式研究框架，才允许讨论接近执行端。
- 默认只清理可再生产物，不直接删除模型产物、持仓快照和研究结论文件。

## 9. 历史入口
- 完整时间线、阶段实验记录与历史结论：`daily_research/brain/research_log.md`
