# Daily Research 方法记忆

快照日期：`2026-04-09`

## 1. 总原则
- 用户最新提出的要求拥有最高优先级。
- 新要求一旦改变默认链路，必须先统一代码、脚本入口、manifest、trade plan 展示和 brain 文档，再继续后续工作。
- 默认追求最有效，不追求最小改动。
- 不允许保留会误导判断的前后矛盾和过期冗余。

## 2. 接管前判型
- 任何问题先判定它属于 `formal`、`recent` 还是 `live`。
- 任何问题再判定它属于研究环、执行环，还是 promotion 边界。
- 研究问题默认先看 formal 根，不先看 production full-fit。
- recent 问题默认指最近一年 `12` 个月窗口，不是一个月短监控切片。
- 执行问题默认先看 active manifest、production root 和 recent/live 审计，不先拿 formal 结论替代 live 现实。

## 3. 研究模型协议
- formal 验证采用滚动窗口协议。
- 每个 formal 窗口都必须使用该窗口起点前最新可标注数据训练当时最新模型。
- 每个 formal 窗口的评估段保持独立 holdout，不允许把该窗评估期并回该窗训练集。
- 任何研究结论都必须显式写出 `train_end / valid_start / valid_end`。
- recent 验证是研究闭环必备伴随证据，不允许 strongest-model 只报 formal 不报 recent。
- formal 证据只能来自 formal 协议，不得用 production full-fit 结果回填。
- 重新寻找“最强模型”时，默认先锁同一 strongest-model gate：同股票池、同 checkpoint objective、同 window_count、同 formal 协议；不允许把 cross-family reference 或 annual-checkpoint reference 直接混进同一 winner 判定。
- strongest research model、strongest stable base model、live mainline 必须分开命名，不允许再压成一句“当前最强模型”。
- strongest research model 默认可以直接作为执行默认，不再额外增加独立 promotion 哲学阻塞层。
- 但最终默认执行产物必须先做 latest-data `production full-fit`，并默认使用该 family 当前最高预算；不允许省算力。

## 4. 执行模型协议
- 执行默认物化同样必须使用当前可标注最新数据做 `production full-fit`。
- `production full-fit` 的职责是物化最新 live panel、默认 fallback 和真实执行默认值。
- recent/live 监控负责回答“最近一年有没有兑现”与“当前 live 实际在跑什么”，不负责改写 formal winner。
- 如果 recent/live 变弱，默认先查市场状态、execution bridge、集中度和 live 约束，不允许先把原因偷换成“研究训练数据不够新”。

## 5. 训练纪律
- 训练默认从 `32` 起步。
- 如果 `32` 不够，只允许同模型 `strict resume` 续训，不允许 fresh rerun 代替。
- 训练一律使用 `GPU`；没有 CUDA 就视为阻塞。
- execution-bound、monthly refresh、production candidate 一律先补成 latest model + highest family budget。
- strongest winner 一旦要写入默认执行，也遵守同一条：latest model + highest family budget。
- production full-fit 如果 `32` 被判 undertrained，必须继续 strict resume 到更高预算后再冻结。

## 6. 研究与执行统一
- 当前统一权重语义是 `research_raw_target_weight`。
- 当前统一上限语义是 `follow_research_raw_no_global_cap`。
- 研究怎么生成 target weight，执行就怎么执行。
- 不允许在 external target-weight 链上再偷偷套一层通用 `max_weight=0.25`。
- current static fallback 也必须从 raw panel 重新桥接，而不是回退到旧 capped panel。

## 7. 成本口径统一
- active candidate backtest 默认必须跟随 profile / manifest 注入真实成本。
- 默认成本口径是 `transaction_cost_bps=3`、`slippage_bps=7`、`sell_tax_bps=10`。
- 不允许再出现“同一条 active execution 链，一次带成本、一次不带成本”的比较。

## 8. execution-side 规则
- broad execution-policy sweep 已停止。
- 当前 execution-side 只沿 `month-start / first-week / signal-to-weight / month-trigger / execution` 下钻。
- 当前经过验证的 execution-side 静态基线与主线是 `regoff_k2_5d_ensemble_native_anchor`。
- `trend_up_low_vol|expand|stable -> topk3_1d_regoff` 当前只保留为 observation-only 的历史 targeted repair 分支，不再作为默认 repair 故事。
- daily trade plan 默认只走 `live-only` refresh。
- `live-only` 新 root 必须能够自动继承上一版 formal reference，不允许因为空 root 丢 formal summary。

## 9. model-side 规则
- 当前模型侧如重开，只从 `penalty-only` 窄修复继续。
- 已被证伪的大 bundle、`v2`、`heavy penalty` 不再回到默认主线。

## 10. 月度优先与强月纪律
- 以后默认按“月度收益优先、模型偏短线”推进。
- 研究评估默认先输出月度指标：positive month ratio、monthly median return、worst month、最长连续弱月、最近 `1-3` 个月表现。
- 如果年化更高但月度分布更差，不默认视为更优答案。
- 在任何 promotion 之前，monthly-first execution verdict 都必须先写入独立的 scoreboard root。
- 如果目标改成“追求强攻击型月份”，必须单独落一份 `30% 强月 scoreboard`，不能继续拿普通 monthly-first scoreboard 代替。
- 强月研究要显式分开 `formal attack winner` 和 `recent/live gate winner`；只有两边都不退化，才允许讨论 live promotion。
- 自定义 challenger 若涉及 bridge 变换，必须遵守和官方回放一致的顺序：先过可交易过滤，再做 bridge；不能先 bridge 再过滤。
- 如果 formal attack winner 和 recent/live gate winner 不一致，默认把前者记为 research branch，保留后者为 live mainline。

## 11. 一致性自检
- 只要改动训练默认值、active manifest、candidate pipeline 或 brain 文档，收尾必须跑：
  - `python daily_research/tools/project_consistency_check.py`
  - `python daily_research/tools/doc_guard.py check`
- 如果 active execution 语义变了，还必须重刷：
  - production root
  - single-mapping candidate pipeline
  - active manifest
  - latest trade plan
- 只要 active execution 仍是 external target-weight 链，就必须同时同步四层：`live target-weight panel`、`daily_live_score_reference.json`、`active_execution_strategy.json` 的 `effective_live_*` 字段、以及 `latest_trade_plan.txt` 的解释文本。
- 如果当前 live 权重来自 bridge 而不是同日 raw score 直达，trade plan 必须显式写明 `weight_generation_note`，不允许再出现“今天分数为什么对不上今天权重”的解释断层。
- 训练纪律不仅要写在入口脚本，也要沉到底层 dataclass 默认值；`DeepAlphaConfig`、主 runner、matrix fallback 三层必须同口径保持 `32` 起训，并清掉 stale execution profile default。
