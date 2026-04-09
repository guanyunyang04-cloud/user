# Daily Research Procedural Memory

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
- 当前 execution-side 只沿 `month-start / first-week / score -> weight -> execution` 下钻。
- 当前唯一正式保留的 repair candidate 是 `trend_up_low_vol|expand|stable -> topk3_1d_regoff`。
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
