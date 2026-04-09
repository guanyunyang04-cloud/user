# Daily Research 方法记忆

快照日期：`2026-04-09`

## 1. 总原则
- 用户最新提出的要求拥有最高优先级。
- 新要求一旦改变默认链路，必须先统一代码、脚本入口、manifest、trade plan 展示和 brain 文档，再继续后续工作。
- 默认追求最有效，不追求最小改动。
- 不允许保留会误导判断的前后矛盾和过期冗余。

## 2. 训练纪律
- 训练默认从 `32` 起步。
- 如果 `32` 不够，只允许同模型 `strict resume` 续训，不允许 fresh rerun 代替。
- 训练一律使用 `GPU`；没有 CUDA 就视为阻塞。
- execution-bound、monthly refresh、production candidate 一律先补成 latest model + highest family budget。
- production full-fit 如果 `32` 被判 undertrained，必须继续 strict resume 到更高预算后再冻结。

## 3. 研究与执行统一
- 当前统一权重语义是 `research_raw_target_weight`。
- 当前统一上限语义是 `follow_research_raw_no_global_cap`。
- 研究怎么生成 target weight，执行就怎么执行。
- 不允许在 external target-weight 链上再偷偷套一层通用 `max_weight=0.25`。
- current static fallback 也必须从 raw panel 重新桥接，而不是回退到旧 capped panel。

## 4. 成本口径统一
- active candidate backtest 默认必须跟随 profile / manifest 注入真实成本。
- 默认成本口径是 `transaction_cost_bps=3`、`slippage_bps=7`、`sell_tax_bps=10`。
- 不允许再出现“同一条 active execution 链，一次带成本、一次不带成本”的比较。

## 5. execution-side 规则
- broad execution-policy sweep 已停止。
- 当前 execution-side 只沿 `month-start / first-week / signal-to-weight / month-trigger / execution` 下钻。
- 当前经过验证的 execution-side 静态基线与主线是 `regoff_k2_5d_ensemble_native_anchor`。
- `trend_up_low_vol|expand|stable -> topk3_1d_regoff` 当前只保留为 observation-only 的历史 targeted repair 分支，不再作为默认 repair 故事。
- daily trade plan 默认只走 `live-only` refresh。
- `live-only` 新 root 必须能够自动继承上一版 formal reference，不允许因为空 root 丢 formal summary。

## 6. model-side 规则
- 当前模型侧如重开，只从 `penalty-only` 窄修复继续。
- 已被证伪的大 bundle、`v2`、`heavy penalty` 不再回到默认主线。

## 7. 一致性自检
- 只要改动训练默认值、active manifest、candidate pipeline 或 brain 文档，收尾必须跑：
- `python daily_research/tools/project_consistency_check.py`
- `python daily_research/tools/doc_guard.py check`
- 如果 active execution 语义变了，还必须重刷：
- production root
- single-mapping candidate pipeline
- active manifest
- latest trade plan
- `2026-04-09` 一致性补充：
- 以后只要 active execution 仍是 external target-weight 链，就必须同时同步四层：`live target-weight panel`、`daily_live_score_reference.json`、`active_execution_strategy.json` 的 `effective_live_*` 字段、以及 `latest_trade_plan.txt` 的解释文本。
- 如果当前 live 权重来自 bridge 而不是同日 raw score 直达，trade plan 必须显式写明 `weight_generation_note`，不允许再出现“今天分数为什么对不上今天权重”的解释断层。
- 训练纪律不仅要写在入口脚本，也要沉到底层 dataclass 默认值；`DeepAlphaConfig`、主 runner、matrix fallback 三层必须同口径保持 `32` 起训，并清掉 stale execution profile default。

## 8. 2026-04-09 执行升级流程
- 如果默认 fallback bridge 在不重训模型的情况下发生变化，必须先重刷 production fallback 产物：
- `python daily_research/execution/refresh_production_static_fallback.py --static-fallback-profile <profile>`
- 然后再重跑 single-mapping pipeline 的 `--live-only`、重新激活 active manifest，并重跑 `run_trade_plan.py`。
- 当前已验证的默认 static fallback profile 是 `regoff_k2_5d_ensemble_native_anchor`。
- 当前已验证的 active pipeline root 是 `daily_research/output/short_alpha_execution_single_mapping_candidate_pipeline_20260409_r2`。
- pipeline 必须从 `static_fallback_daily_live_target_weight_meta.json` 读取当前 effective fallback profile，而不是相信过期硬编码默认值。

## 9. 月度优先纪律
- 用户最新要求具有最高优先级：以后默认按“月度收益优先、模型偏短线”推进。
- 研究评估默认先输出月度指标：positive month ratio、monthly median return、worst month、最长连续弱月、最近 1-3 个月表现。
- 如果年化更高但月度分布更差，不默认视为更优答案。
- 新实验优先级向短线倾斜：更重视最近窗口、弱月修补、月初到首周触发、短执行 bridge，而不是长期平均最优。
- 后续如果重开模型侧试验，优先短周期目标和短反馈验证，不优先扩大长周期 horizon。

## 10. 月度优先执行裁决纪律
- 在任何 promotion 之前，monthly-first execution verdict 都必须先写入独立的 scoreboard root。
- recent execution-policy audit 必须先把当前 live 静态基线纳入比较；截至 `2026-04-09` 这条基线是 `regoff_k2_5d_ensemble_native_anchor`。
- targeted repair review 必须让 targeted 与 static 在完全相同的 clipped window 上比较，不允许拿 targeted replay 去对照多出一天的 audit 指标。
- 如果 targeted mapping 在 recent 月份里只是中性，且在 leave-window-out 窗口里输掉或打平，就必须降级为 observation-only，而不是继续保留成默认 repair 故事。
- 当前月度优先裁决是：保留 `k2` static fallback 作为 execution-side mainline，`topk3` targeted repair 保持 observation-only。
