# Continuous Policy 设计合同

快照日期：`2026-05-14`

## 北极星
- 构建一个以日为单位进行连续决策的交易执行模型。
- 模型应从市场全局状态、个股时序路径、持仓上下文与组合约束中学习 `source / receiver / cash allocation`。
- 最终目标是组合层资金分配质量，不是固定调仓频率、固定持有周期或单股动作分类。
- 执行层只翻译策略语义为权重与订单，不得静默改写策略本体。

## 非目标
- 不把更像 teacher 当作最终目标；teacher 只是 warm start / auxiliary prior / heuristic scaffold。
- 不用 screening 高收益、短窗口 smoke、单项 replay 改善或单层指标替代正式 verdict。
- 不把 simulator guard 当主策略逻辑；guard 只能做最后安全裁剪和诊断。
- 不把 realtime tail label、failed trial、interrupted outer study 或 loose latest 写成 completed evidence。

## 当前绑定原则
- 个股动作语义与组合预算语义必须分层：个股层表达生命周期动作，组合层表达资金接收、资金释放、现金保留、gross / turnover / cost。
- 卖出、现金与资金来源是同一条 credit assignment 链；source 必须解释为“当前组合状态下更适合释放资金”，不是简单看跌。
- 个股未来上涨不等于今天该加仓；持仓仍有正 forward 也不一定永远不能卖，关键是相对 receiver、现金和风险的机会成本。
- 训练、预测、simulator 和 reporting 必须使用一致的 target weight / target delta 语义。
- full-universe strict Gold 可作为训练数据真源；realtime Gold 只能作为 research/audit，除非 tail label 完全 observed 且 catalog 标记 training-safe。

## 当前 Active Research Contract
- Active production anchor 仍是 `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`；continuous_policy 不得替代 live/default。
- Active new-study profiles 当前限制为：
  - `focused_seq_v1`
  - `split_heads_portfolio_daily_release_first_constrained_decoder_r56`
  - `split_heads_portfolio_daily_release_first_portfolio_set_v5_r65`
- r65/r67/r68 entry：`split_heads_portfolio_daily_release_first_portfolio_set_v5_r65` / `portfolio_set_v5_dfl_pg_v1`。
- r69 explicit research entry：`split_heads_portfolio_daily_value_arbitration_portfolio_set_v5_r69` / `portfolio_set_v5_dfl_pg_v1_r69_value_arbitration`；registered for explicit protocol/traincheck use but not part of active/default search profiles.
- r71 explicit research entry：`split_heads_portfolio_daily_multistage_regret_portfolio_set_v5_r71` / `portfolio_set_v5_dfl_pg_v1_r71_multistage_regret`；registered for explicit protocol/smoke use but not part of active/default search profiles.
- r65 backend：`formal_torch_portfolio_set_v5`，artifact type `continuous_policy_torch_portfolio_set_v5`，`promotable=False`。
- r65 默认 dataset：`continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1`。
- r67 internal version：`portfolio_set_v5_dfl_pg_v1`；旧 v5 artifacts 与 `alpha_result_value_budget_split_v48` alias 可兼容读取，但新训练默认只产出 DFL-PG v1 metadata。
- r70 version boundary：`portfolio_set_v5_dfl_pg_v1` 是 base v5 默认 internal version；`alpha_result_value_budget_split_v48` 与 `portfolio_set_release_first_decision_v1` 只作为兼容 alias 解析到 base DFL-PG；`portfolio_set_v5_dfl_pg_v1_r69_value_arbitration` 只能由显式 profile/loss 或 artifact metadata 启用。
- r65/r67 architecture：per-symbol temporal encoder、portfolio state token、latent set attention、DFL-PG decision oracle、release-first decoder；默认禁止 full-universe `O(N^2)` self-attention。
- r67/r68 prediction 必须只从 oracle-compatible output 派生 source/receiver/cash/target fields，并设置 `portfolio_cashflow_decision_v1_mode=1.0`；`release_first_allocation_v3_mode=1.0` 只保留兼容字段，不得在 v5 cashflow mode 下二次重算目标。
- r68 cashflow contract：`portfolio_cashflow_decision_v1` 是 v5 prediction、simulator、release trace 与 continuity metrics 的共享资金流语义；缺字段、方向冲突、oracle infeasible 或 turnover violation 必须 fail closed 并显式诊断。
- r69 value arbitration：source/release/defense/cash/reversal 只能作为训练目标与诊断改进，不得绕过 r68 cashflow contract，不得用 guard 静默修正成 promotion-looking behavior。
- r70 after repair：oracle feasibility 与 cashflow translation 不再是 r69/r70 主 blocker；后续不能再通过放宽版本边界、静默启用 r69 mode 或二次 solver 重算来制造改善。
- r71 multistage regret：source rebound、receiver deploy regret、cash defense regret、rotation spread regret、reversal action regret 与 crowding 只能作为显式 r71 训练目标和诊断；不得绕过 r68 cashflow contract，不得把 failed smoke 写成 behavior acceptance。
- r61 core-v4 保留为 baseline/ablation；不再作为下一代主模型承载新主线。

## 成功判定
- 训练证据必须 sufficient：足够 train days、teacher/action evidence 或等价有效证据、best epoch 不贴边，且使用 training-safe dataset。
- 行为证据必须闭合：source intent/source target/receiver target 非零且可解释，target gap、cash、intent translation conflict、reduce/exit、drawdown 不触发硬门槛。
- Study 证据必须完整：completed protocol summary + completed study summary；若外层 study 未汇总，只能写 protocol-level evidence。
- Promotion 讨论前必须同时满足 formal evidence、v2 gate、stable confirm、drawdown/monthly quality、source/receiver/cash closure 与 active artifact guard。

## 当前已知断点
- r53-r55：cash/exposure closure 改善，但 cash timing、source/reduce/exit 未闭合。
- r56：release-first allocator 代码合同存在，但安全筛选缺 completed evidence。
- r60-r61：profile binding、diagnostics、core-v4 接线有效，但 release/source/receiver/target translation 仍断。
- r64：full-window strict Gold 完成；realtime full-window Gold pending。
- r65：portfolio-set v5 接线成功，但 behavior negative：source intent/target、receiver target 仍为 0，intent conflict 为 1.0。
- r67：DFL-PG v1 机制与 tiny strict-Gold protocol smoke 通过，训练 target source/receiver 非零且 target conflict 为 0；但 evaluation source target 仍为 0、shadow intent conflict 仍高，不能视为行为闭合。
- r68：v5 cashflow translation closure smoke 通过，shadow source target=97、receiver target=139、intent conflict=0、cashflow valid=21/21；下一 blocker 是 evidence sufficiency、source quality、cash timing、drawdown/reversal，不再是 source/receiver translation dead。
- r69：value-arbitration traincheck 有机制进展，source target=1185、receiver target=160、wrong-side sell share=0；但 receiver 覆盖低于 r68、constraint violation 偏高、完整 evaluate/shadow smoke 因 TDX empty batch 未完成。
- r70：version boundary 与 oracle feasibility 修复完成，tiny strict-Gold smoke `protocol_r70_v5_version_boundary_oracle_repair_smoke_20260514_01` 完整通过，oracle violation 近零、cashflow valid=1、intent conflict=0；但 training evidence 仍 insufficient，cash timing、source quality 与 reversal 仍失败。
- r71：multi-stage regret 代码/测试闭合，base/r69/r71 版本边界保持隔离；但 `protocol_r71_multistage_regret_v5_behavior_smoke_20260514_01/_02/_03` evaluate 均被 TDX singleton empty data 阻断，`_04` train 被 CUDA busy 阻断，尚无 completed tiny behavior smoke。

## 禁止事项
- 禁止从 smoke、dry-run、interrupted wrapper、failed trial、runtime timeout 或 realtime tail label 推 promotion。
- 禁止默认启用 true solver、恢复父子进程 watchdog，或把 active profile 扩成历史菜单。
- 禁止把 source/reduce/exit dead 包装成“只需更多 epoch/loss”。
- 禁止在 `model_seq_v3.py` 或 core-v4 MLP 上继续堆下一代主逻辑，除非是 baseline/compatibility。
- 禁止修改 `daily_research/output/active_execution_strategy.json`。

## 下一步方向
- r66 当前任务是 brain/workflow maintenance，不推进策略训练。
- r71 后续策略研究应先解决 provider/GPU blocker 并完成 post-calibration tiny behavior smoke；只有 translation 不退化、oracle violation 近零且至少两个 behavior 指标优于 r70，才能考虑 strict resume。
- 后台运行只作为 OS 级 launcher/轮询能力，不能改变 study/protocol 单进程研究本体。
- 数据层下一步是 full-window realtime Gold build/audit；仍不得作为 completed training evidence。

## 历史索引
- r31-r39：receiver/source/cash contract 与 allocation objective baseline。
- r40-r48：end-to-end allocation、convex/OPE/solver 方向，均未过 stable confirm。
- r49-r52：capital-flow closure、native allocation vector、validation closure 研究链。
- r53-r55：cash-funded allocator、semantic budget、cash timing release controller。
- r56-r61：release-first allocator、core-v4、profile binding、decision-focused wiring。
- r62-r64：DuckDB + Parquet data lake 与 full-universe strict Gold。
- r65：portfolio-set v5 architecture upgrade。
- r67：paper-driven DFL-PG v1 replacement for portfolio-set v5。
- r68：cashflow decision v1 translation closure for portfolio-set v5。
- r69：value arbitration behavior-quality mechanism for portfolio-set v5。
- r70：version-boundary and oracle-feasibility repair for portfolio-set v5 / r69 value arbitration。
- r71：multi-stage regret behavior-quality mechanism for portfolio-set v5。
- 完整证据入口：`daily_research/brain/references/evidence_registry.json` 与 `daily_research/brain/references/r*_*.md`。

## 归档入口
- 早期合同历史：`daily_research/brain/references/continuous_policy_design_contract_history_raw_20260424.md`。
- 早期合同索引：`daily_research/brain/references/continuous_policy_design_contract_evidence_index_20260424.md`。
- 过程复盘入口：`daily_research/brain/episodic_memory.md`。
