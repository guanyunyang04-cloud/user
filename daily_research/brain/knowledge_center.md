# Daily Research 知识中枢

快照日期：`2026-05-14`

## 1. 稳定事实
- `daily_research` 同时负责研究、formal 验证、recent 验证、production full-fit、live 执行和接管治理。
- strongest-model research winner、deployable learned-control、live mainline 必须显式区分。
- 当前 live 默认执行仍由 `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active` 承担。
- `daily_research/environment.yml` 是依赖环境真源；任何程序都必须在 `yolos` 环境下运行。
- 当前统一权重语义是 `research_raw_target_weight`；当前统一上限语义是 `follow_research_raw_no_global_cap`。
- continuous_policy 的终局目标是日级连续交易执行模型，不是固定调仓或人工执行桥。
- r39 仍是 continuous_policy 有效证据基线；r40-r70 是 research / shadow 升级链或基础设施证据。
- r64 full-window strict Gold 是当前 reusable training-safe Gold 数据集；realtime Gold 仍不能作为 completed training evidence。

## 2. 硬规则
- 必须 brain-first；默认接管顺序为 `identity -> state -> knowledge -> design contract -> operations -> governance`。
- formal、recent、promotion、live 不得混写。
- `latest_*` 文件不得被直接视为无条件真源；必须先判断 freshness，同源性和 explicit tag。
- code contract、unit tests、smoke、dry-run 只能证明接线正确，不能证明策略有效。
- failed / timeout / interrupted trial 只能写诊断，不得写 completed evidence。
- realtime tail label 必须显式标记 unobserved，不得计入 completed training evidence。
- active artifact diff 是硬失败。
- 主脑文档只保留控制面；长历史、完整 rXX 证据、长命令和复盘进入 `references/`。

## 3. 长期教训
- 个股动作和组合资金分配不是同一问题；真正目标是当前组合状态下最优仓位调整集合。
- 卖出比买入更难，因为卖出同时涉及继续持有机会成本、现金价值、资金来源责任和风险状态。
- 只追求单项 gate 清零会制造假进展；必须同时看收益、月度质量、drawdown、source count、cash timing、exposure 和 intent conflict。
- Clean target-sum closure 可以与 release/source/receiver flow disconnected 同时存在；closure 不是行为闭合的充分条件。
- 更强模型不是自动解决方案；若 target construction、receiver/source semantics 或 evidence route 错，放大模型只会更快放大错误。
- 工程复杂度会制造循环；runner、profile、loss、diagnostics 必须减少活动面，服务明确阻塞点。
- 数据集必须可复用、可审计、可查询；pickle/cache 可兼容，但新训练集应进入 DuckDB + Parquet data lake。
- 脑区是项目事实真源，skills 只是流程入口，不复制长历史。

## 4. 研究主线索引
- r10-r18：action/head + translation guard 改善语义，但不能替代组合资金分配本体。
- r19-r30：receiver/source/cash ranking、listwise、teacher、release/relief 暴露 source 放宽与休眠问题。
- r31-r39：receiver executable、source clean-pass、unified allocation、decision-focused objective，形成当前有效证据基线。
- r40-r48：end-to-end allocation layer、convex/OPE/solver 方向正确，但未过 stable confirm。
- r49-r52：capital-flow closure、true solver 入口、native allocation vector；仍受 source/exposure/evidence 阻塞。
- r53-r55：cash/exposure closure 有改善，cash timing 与 source/reduce/exit 仍死。
- r56-r61：release-first / core-v4 接线与诊断改善，但 release/source/receiver/target translation 未闭合。
- r62-r64：通用 data lake 与 full-universe strict Gold 完成，realtime full-window pending。
- r65：portfolio-set v5 替代 MLP 主线成为下一代 research backend，但首轮行为仍 negative。
- r67：论文驱动 DFL-PG v1 替换 portfolio-set v5 默认目标/loss/oracle/profile，机制 smoke 通过；评估/影子 source funding 与 translation 仍未闭合。
- r68：新增 `portfolio_cashflow_decision_v1` 合同并闭合 v5 source/receiver/cash translation；tiny strict-Gold shadow smoke 中 source target、receiver target、cashflow valid 与 intent conflict 指标已闭合，但仍不是 promotion 或 strategy-success 证据。
- r69：新增显式 value-arbitration 研究线，开始把 source 选择、cash timing、defense/deploy 竞争写入 v5 target/oracle；traincheck 有机制进展，但 receiver 覆盖和 constraint violation 未闭合，完整 smoke 因 TDX empty batch 未完成，不能视为 behavior acceptance。
- r70：修复 v5 base/r69 版本边界、r69 oracle constraint violation 与 TDX empty-batch evaluate blocker；tiny strict-Gold smoke 完整通过，但 behavior quality 与 training evidence 仍未达标，仍只能作为 research / shadow mechanism evidence。

## 5. 当前方法论
- 先直接 protocol smoke，再 study dry-run，再 safe screening；不得跳到 confirmatory。
- 每个重大研究结论必须写成 facts / inferences / assumptions / boundary。
- explicit dataset id、study tag、protocol tag 优先于 loose latest。
- 若 source/reduce/exit 仍为 0，结论必须写成行为闭环未打通，不能包装成“更多 epoch/loss”。
- 若 full Gold 或 realtime Gold 状态变化，必须同时记录 catalog entry、audit、row counts 和 label completeness。

## 6. 文档边界
- `state_center.md`：当前状态、阻塞、优先级、边界和 evidence 索引。
- `knowledge_center.md`：稳定事实、硬规则、长期教训和研究主线索引。
- `operations_center.md`：命令、流程、运行纪律和写回路线。
- `continuous_policy_design_contract.md`：仍有效设计边界和当前 active research contract。
- `references/`：完整 rXX 证据、历史失败路径、长命令、完整复盘。
- `evidence_registry.json`：机器可查 evidence 索引。

## 7. 归档入口
- 本文件归档前完整快照：`daily_research/brain/references/knowledge_center_archive_20260510.md`。
- 早期 continuous_policy 合同历史：`daily_research/brain/references/continuous_policy_design_contract_history_raw_20260424.md`。
- 早期证据索引：`daily_research/brain/references/continuous_policy_design_contract_evidence_index_20260424.md`。
