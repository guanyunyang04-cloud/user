# Daily Research 方法记忆

快照日期：`2026-04-12`

## 1. 总原则
- 用户最新提出的要求拥有最高优先级。
- 新要求一旦改变默认链路，必须先统一代码、脚本入口、manifest、trade plan 展示和 brain 文档，再继续后续工作。
- 默认追求最有效，不追求最小改动。
- 不允许保留会误导判断的前后矛盾和过期冗余。

## 2. 接管前判型
- 当前接管默认不从 `episodic_memory.md` 起步。
- 当前最小完备接管顺序是：
  - `identity_layer.md`
  - `handoff_packet.md`
  - `semantic_memory.md`
  - `rule_memory.md`
  - `lesson_memory.md`
  - `temporal_state.md`
  - `working_memory.md`
- 任何问题先判定它属于 `formal`、`recent`、`learned-control` 还是 `live`。
- 任何问题再判定它属于研究环、执行环，还是 promotion 边界。

## 3. 研究模型协议
- formal 验证采用滚动窗口协议。
- 每个 formal 窗口都必须使用该窗口起点前最新可标注数据训练当时最新模型。
- 每个 formal 窗口的评估段保持独立 holdout。
- 任何研究结论都必须显式写出 `train_end / valid_start / valid_end`。
- recent 验证是研究闭环必备伴随证据，不允许 strongest-model 只报 formal 不报 recent。
- formal 证据只能来自 formal 协议，不得用 production full-fit 结果回填。
- strongest research model、stable base model、learned-control frontier、live mainline 必须分开命名。

## 4. recent 协议方法
- current recent protocol 现在必须同时报告：
  - `requested_recent_end_date`
  - `recent_start_date`
  - `recent_end_date`
  - `recent_validation_end_date`
- 如使用 `calendar_months`，不得再把 requested cutoff 误写成 effective validation end。
- strongest-model recent 现在只承认 `independent_recent_model_as_of_recent_start`。

## 5. 执行模型协议
- 执行默认物化必须使用当前可标注最新数据做 `production full-fit`。
- strongest research model 默认可以直接作为执行默认。
- 但真正写入默认执行时，仍必须先走 latest-data `production full-fit`。
- `production full-fit` 的职责是物化最新 live panel、fallback 和真实执行默认值。
- recent/live 监控负责回答“最近一年有没有兑现”与“当前 live 实际在跑什么”，不负责改写 formal winner。
- 如果 recent/live 变弱，默认先查市场状态、execution bridge、集中度和 live 约束。

## 6. 训练纪律
- 训练默认从 `32` 起步。
- 如果 `32` 不够，只允许同模型 `strict resume` 续训，不允许 fresh rerun 代替。
- 训练一律使用 GPU；没有 CUDA 就视为阻塞。
- 正式训练、formal 回放、recent 回放与最终 summary 默认统一使用 `yolos` 环境。
- 长实验只允许前台执行，默认超时预算 `10` 小时。
- 当前统一权重语义是 `research_raw_target_weight`。
- 当前统一上限语义是 `follow_research_raw_no_global_cap`。

## 7. 统一标准答案
- 当前 strongest-model 三层标准答案是：
  - `formal winner = short_expert_monthly_v1`
  - `recent winner = short_expert_monthly_v1`
  - `promotable winner = short_expert_monthly_v1`
- 当前 learned-control 标准答案是：
  - `deployable constrained front-runner = short_expert_policy_v5b__k1_20d = 0.1177`
  - `current recent winner = short_expert_policy_v5b = 0.1200`
  - `active v5 family fresh formal best = short_expert_policy_v5b = 0.0839`
  - `historical cross-family fresh formal best = short_expert_policy_v4b = 0.1008`
- `baseline_current` 当前不再是 strongest-model recent winner，只保留为某些 family recent 对照参考。

## 8. model-side 方法
- hand-crafted repair 已进入冻结阶段，不再回到默认主线。
- learned-control 当前默认主研究分支已经前移到 `policy_v5`，锚点是 `policy_v5b`。
- 下一轮 learned-control 默认优先排查：
  - execution-stability regularization
  - candidate-count / concentration regularization
  - 在必要时保留 gross-teacher distillation
- 新的 learned-control 分支至少先过：
  - formal
  - constrained formal
  - recent
  - 再讨论 live promotion

## 9. 一致性自检
- 只要改动训练默认值、active manifest、candidate pipeline 或 brain 文档，收尾必须跑：
  - `python daily_research/tools/project_consistency_check.py`
  - `python daily_research/tools/doc_guard.py check`
- 如果当前 live 权重来自 bridge 而不是同日 raw score 直达，trade plan 必须显式写明 `weight_generation_note`。
