# 执行端盘后信号刷新闭环修复记录 2026-05-24

## 状态

- Status: `execution_signal_refresh_closure / data_refresh_signal_panel_closed / no retrain / no promotion`.
- Scope: `daily_research.execution` 数据刷新、production signal panel refresh、交易计划生成前 stale panel guard 和执行端页面展示。
- Active artifact impact: `daily_research/output/active_execution_strategy.json` 在 git diff 中保持为空；本轮只允许运行时同步 production root / panel path / dataset 指针等执行必要元数据。
- 本记录不是 production retrain、promotion、live/default 切换、真实 broker 下单或模型有效性提升证据。

## 事实

- active dataset 已同步到 `policy_input_bundle__be00cfadef3d77eb9e546f7d`，覆盖最新完成交易日 `2026-05-22`。
- production 锚点使用现有 full-fit run `short_expert_policy_v5b_execalign_production_fullfit_20260421_r1`，未执行 production retrain、promotion 或 broker 下单。
- production signal refresh 已产出 `daily_research/output/short_expert_policy_v5b_execalign_production_default/live_panel_refresh_manifest.json`，状态 `ok`，`target_panel_latest_date=2026-05-22`，`score_panel_latest_date=2026-05-22`，`target_position_count=3`。
- 最新真实交易计划产物在 `daily_research/execution/output/20260522/`，`actions_today.csv` 有 3 个买入动作：`002866.SZ`、`000065.SZ`、`600864.SH`；`watchlist.csv` 有 10 条；`source_signal_date=execution_signal_date=2026-05-22`；`signal_panel_status=ok`。
- `/api/data-sources` 显示 dataset synced、`signal_panel_status=ok`、`next_signal_action=skip`；交易计划页显示 3 个动作、市场状态 `trend_up_low_vol` 和 Signal Panel `ok`。
- `daily_research/output/active_execution_strategy.json` 在 git diff 中为空，本轮没有产生 tracked active artifact diff。

## 根因

- 行情数据刷新与 production live signal panel 刷新此前断链，导致 active dataset 已到 `2026-05-22`，但交易计划仍可能消费旧信号面板。
- lake 模式 signal refresh 起初未正确接入 rolling liquidity `policy_pool_view`，生产面板先落到全 A 口径。
- 即使接入 rolling liquid500 后，交易计划 wrapper 仍用旧的 `execution/universe/liquid500_latest.txt` 作为 stocks-file，和 production target panel 的最新信号 universe 不一致，导致 3 个正权重目标被当作池外候选丢弃，形成假空计划。
- `/api/data-sources` 曾用裁剪过的 active summary 判断 signal panel，缺少 `trade_plan_score_panel_csv` 等字段，造成真实 panel ok 但 API stale 的误报。

## 修复

- 增加 production anchor audit/sync 和 `refresh-production-live-panels` 安全任务，数据刷新成功或数据已最新但 signal stale 时自动刷新 production signal panels。
- `export_live_panels_from_run.py` 支持 lake dataset、sector board view、rolling pool view、style map augmentation，并写出 `live_panel_refresh_manifest.json`。
- `load_lake_market_data_with_pool_view()` 在 `pool_name=liquid*` 且未显式给 pool view 时自动构建/复用 rolling liquidity pool view。
- 交易计划 wrapper 对 external target-weight path 优先从 production target panel 最新信号日生成 `external_target_weight_latest.txt`，避免旧静态 pool 文件吞掉正权重目标。
- 交易计划默认启用 `--require-fresh-signal-panel`，若 production target/score panel 未覆盖最新完成交易日则阻断生成。
- 模型训练距交易日多久保留为信息展示，移除 retrain threshold warning 语义。

## 验收

- Python: `pytest daily_research/execution/tests/test_execution_console_v2_api.py daily_research/baseline/tests/test_generate_daily_trade_plan_external_panels.py daily_research/deep_alpha/test_lake_pool_view_bridge.py daily_research/execution/tests/test_lake_execution_data_source.py -q` 通过，57 passed。
- Frontend: `npm test -- --run` 通过，13 passed；`npm run build` 通过，仅 Vite chunk-size warning。
- Real signal refresh: `python daily_research/execution/refresh_production_live_panels.py --as-of-date 2026-05-22` 成功，panel latest date 为 `2026-05-22`。
- Real trade plan: `python daily_research/execution/run_trade_plan.py --candidate-profile active_execution_strategy` 成功，输出 3 个买入动作。
- 8765 smoke: `/healthz`、`/api/status`、`/api/data-sources`、`/api/trade-plan`、React 首页均返回 200；数据页和交易计划页显示上述结构化结果。

## 边界

- 本轮不代表 production retrain、promotion、live/default 切换或真实下单。
- 本轮允许更新 output 产物和 runtime/cache，但 tracked active manifest 无 diff。
- continuous 仍不进入执行端默认 UI/API。
