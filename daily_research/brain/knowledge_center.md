# Daily Research 知识中枢

快照日期：`2026-05-11`

## 1. 稳定事实
- `daily_research` 同时负责研究、formal 验证、recent 验证、production full-fit、live 执行和接管治理。
- strongest-model research winner、deployable learned-control、live mainline 必须显式区分。
- 当前统一权重语义是 `research_raw_target_weight`；当前统一上限语义是 `follow_research_raw_no_global_cap`。
- `daily_research/environment.yml` 是依赖环境真源；任何程序都必须在 `yolos` 环境下运行。
- 当前 `yolos` OpenMP runtime 冲突已原地修复；无 `KMP_DUPLICATE_LIB_OK` 的 `openmp_runtime_check.py --strict` 是验收标准。
- 当前 live 默认执行仍由 `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active` 承担。
- `daily_research/execution/run_execution_app.py` 是执行侧统一应用入口；execution app 运行态落到 `daily_research/output/execution_app`。
- execution Web 控制台使用 FastAPI，本地只监听 `127.0.0.1`，用户界面与使用教程统一使用简体中文，并包含 `/continuous-policy` 页面。
- continuous_policy 的终局目标是以日为单位进行连续决策的交易执行模型，不是固定调仓频率、固定持有周期或人工执行桥。
- continuous_policy 训练入口包括 `prototype_gbdt_v1`、`formal_torch_v2`、`formal_torch_seq_v3`、`formal_torch_hier_v4`；正式入口以 `run_continuous_policy_protocol.py` 为准。
- continuous_policy label / decoder 历史仍保留 `holdcash_v3`、`holdcash_v5` 等关键对照。
- latest 行为与结论真源为 `latest_behavior_audit_summary.json` 与 `latest_conclusion_ledger.json`。
- r39 是当前 continuous_policy 有效证据基线；r40-r52d 是 research / shadow 升级链。

## 2. 硬规则
- 必须 `brain-first`；默认接管顺序为 `identity -> state -> knowledge -> operations`。
- formal 验证采用滚动窗口协议；formal、recent、promotion、live 不得混写。
- recent 验证现在是 strongest-model 研究闭环必备伴随证据；recent 胜利不能直接当 promotion 结论。
- 默认做最高效、最合理的实验，允许扩实验但必须写清假设、成本边界和停止条件，不做无目的广扫。
- `epoch formal candidate` 至少从 `32` epoch 起步；不够就沿同一 run_dir 做 `strict resume`。
- continuous_policy formal protocol 必须显式产出 `training_evidence`；当前下限包括 `train_day_count >= 180`、`teacher_action_rows >= 10000`，且 `best_epoch` 不能贴着 `completed_epochs` 边缘。
- smoke、dry-run、short-window check、repaired confirm、insufficient evidence 不得升级为正式 verdict。
- continuous_policy 进入 promotion 讨论前，必须满足训练证据、v2 gate、stable confirm、receiver/source/cash 与 drawdown / monthly quality 同时达标。
- 所有训练、评估、bounded study 与 confirmatory rerun 默认前台运行；长任务必须有持久日志和进度文件。
- 代码合同、单测和 dry-run 只能证明接线正确，不能证明策略有效。
- 当前文档不得把 `KMP_DUPLICATE_LIB_OK` 写成默认解决方案；历史记录中出现该变量只作为历史事实。
- `Start-Job` 只属于局部联调经验，不应写成用户侧公开启动默认。
- 中文主分脑文档必须保持 UTF-8；PowerShell here-string 不是默认中文写入方案，批量整理中文正文时需用 `apply_patch` 或显式 UTF-8 写入并复核。
- `latest_*` 文件不得被直接视为无条件真源；必须先经 artifact freshness 判断 study / protocol / audit / ledger 是否同源、是否 stale。
- 当 loose `latest_*` 不同源时，必须用 explicit study tag capsule 读取：`brain_workflow status --workflow continuous_policy --study-tag <tag> --json`；指定 study 证据优先于混杂 latest 指针。
- workflow JSON 是运行态证据胶囊，不是 brain 权威事实；稳定结论仍必须写回对应中枢或 references archive。

## 3. 已验证教训
- 个股动作和组合决策不是同一个问题；真正目标是当前组合状态下最优仓位调整集合。
- 卖出比买入更难，因为卖出要判断继续持有的机会成本、现金价值和资金来源责任。
- simulator guard 只能做最后安全裁剪，不能承担主策略逻辑。
- 只追求单项 gate 清零会制造假进展；必须同时看收益、月度质量、drawdown、source count、cash timing 与 exposure utilization。
- r38 证明单边 source hard-negative 会压住强势误卖，但也会退化为过度保守和收益弱。
- r39 证明统一 allocation objective 能恢复收益和正 receiver-source spread，但 source breadth、cash timing 与 drawdown 仍未闭合。
- r48 证明 full-universe convex OPE 运行通道可用，但 source dead / exposure 低 / evidence edge 仍会失败。
- r52 证明 day-set native allocation vector 结构更接近目标，但 source release 不会自动闭合；native target validity 和 source threshold 仍需修复。
- r52b 的核心不是加长训练，而是让训练 projection、预测导出和 simulator validity 使用同一约束口径；若 `native_target_valid` 仍低，继续加 epoch 只会放大无效目标。
- r52b safe screening 已验证：validity-first projection 还没有把 native target 有效消费率拉出 r52 低位区间；当 invalid reason 集中在 `native_target_invalid_unsupported_receiver_count` 时，首要问题是 receiver 可执行域和导出 mask 同口径，而不是训练资源不足。
- r52c safe screening 已验证：receiver executable closure 可以把 simulator 边界的 `native_target_valid` 拉到 `0.975~1.0`、fallback 压到 `0~0.025`，但这只关闭 unsupported receiver validity；deployment / cash timing、exposure utilization、source depth 与 training evidence 仍未闭合，因此不能进入 confirmatory 或 22 epoch resume。
- r52d 当前只证明代码合同、dry-run 与 safe screening-only 路径成立：validation closure、train/sim alignment 与 deadband 常量共享已有测试，dry-run 与 safe screening 均确认 v40 / support flags / true-solver-disabled；explicit capsule 已进一步确认其不具备 confirmatory eligibility，不能解释成策略进展。
- `cp_v3_seq_holdcash_r1`、`cp_v3_seq_holdcash_r2`、`cp_v3_seq_holdcash_v5_formal_r1` 共同证明 hold/cash 改善必须经 formal evidence 复核。
- `cp_hier_v4_holdcash_r5` 证明 hierarchical branch 可改善 reversal，但没有学出 hold 前仍只是 research branch。

## 4. r10-r52d 知识索引
| 范围 | 结论 |
| --- | --- |
| r10-r18 | 旧 action/head + translation guard 能改善语义，但无法替代组合资金分配本体。 |
| r19-r23 | receiver/source/cash ranking 是正确转向，但执行合同干净不等于策略有效。 |
| r24-r30 | listwise / teacher / release / relief 暴露 source 放宽与 source 休眠的跷跷板。 |
| r31-r34 | receiver executable 与 source clean-pass 合同质量提高，但 confirm 仍不稳。 |
| r35-r39 | unified allocation / decision-focused / final objective 是当前最有效证据基线。 |
| r40-r48 | end-to-end allocation layer 与 convex/OPE 方向正确，但策略证据未过 stable confirm。 |
| r49-r50 | capital-flow closure 与 true solver 入口保留；r50 因本机负荷过高不作为默认长训路径。 |
| r51-r52 | native allocation vector 与 day-set batch 是当前轻量主线，但仍需修 target validity / source threshold。 |
| r52b | safe screening 未通过 validity 目标：native target valid 仍仅 1.25%-6.25%，fallback 仍 93.75%-98.75%，下一步应修 receiver executable mask / native target export / simulator validity 同口径。 |
| r52c | receiver executable closure 已通过 simulator 边界验证，但最新瓶颈转为 cash timing 为负、exposure utilization 约 0.33、training evidence insufficient 与 source depth 不稳；下一步不应重复 receiver mask 修补。 |
| r52d | validation closure 代码合同、测试、dry-run 与 safe screening-only 证据已存在；explicit capsule 判定 3/3 trials 均 evidence insufficient、composite<0、cash timing<0、exposure~0.33，因此不进入 confirmatory / resume / promotion / live。 |
| r52e | safe screening failed after 1/3 trials: resource gate stopped on source release dead, weak economics, low exposure and high actual cash; recomputed closure audit shows high deployable idle cash and `cash_semantics_mismatch`, so r52e cannot enter resume / confirmatory / promotion. |
| r53 | cash-funded allocation core v2 is a new research line. It materially improved cash/exposure closure in safe screening, but still has insufficient training evidence, negative cash timing, and strongly negative composite score; it is not eligible for confirmatory, strict resume, promotion, live/default, or active artifact changes. |

## 5. 文档边界
- `identity_layer.md`：使命、北极星、硬约束与禁区。
- `state_center.md`：当前状态、当前问题、优先级、边界和 handoff 摘要。
- `knowledge_center.md`：稳定事实、硬规则、长期教训和知识索引。
- `operations_center.md`：地图、环境、命令、流程和写回入口。
- `governance_layer.md`：治理闭环与接管纪律。
- `episodic_memory.md`：最新动作后复盘和 archive 入口；长历史进入 `references/`。

## 6. 归档入口
- 本文件归档前完整快照：`daily_research/brain/references/knowledge_center_archive_20260510.md`。
- 早期 continuous_policy 合同历史：`daily_research/brain/references/continuous_policy_design_contract_history_raw_20260424.md`。
- 早期证据索引：`daily_research/brain/references/continuous_policy_design_contract_evidence_index_20260424.md`。
