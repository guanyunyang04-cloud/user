# Daily Research 状态中枢

快照日期：`2026-05-27`

## 当前结论
- `daily_research` 是当前正式生产研究与执行主线。
- active 执行物化真源：`daily_research/output/active_execution_strategy.json`。
- 当前 live 默认执行 label：`short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`。
- 当前 effective live execution profile：`regoff_k1_20d_ensemble_native_anchor`。
- 当前 production root：`daily_research/output/short_expert_policy_v5b_execalign_production_default`。
- 当前执行权重语义：`research_raw_target_weight`；权重上限语义：`follow_research_raw_no_global_cap`。
- 所有主线都是可显式切换的当前工作指针；切换后按切换后的主线继续，但不自动代表 live/default、promotion 或 active artifact 变更。
- 多 Horizon 交易效用排序当前研究主线指针：`alpha_multi_horizon_utility_policy_v1`（2026-05-23 命名迁移）；旧 `alpha_path20_neural_policy_v1` / `Path20` 保留为历史证据代号和代码 / study namespace，不再代表当前目标定义；`alpha_path20_sequence_policy_v1` 暂作 shadow comparison / secondary research route；三者均不代表 live/default。
- 最新多 Horizon 交易效用候选：`path20_horizon_discovery_no_alpha_gru_liquid500_h1_2_3_5_8_10_15_20_30_du_cost20_hit10_dd010_20260523_01` 完成并 `forecast_test_confirmed`，`trade_utility_score` 通过 validation/test gate；但 predicted best horizon 明显塌缩到 `30d`，仍是 research / shadow-only，不授权 liquid800、multi-seed、allocator、replay、live/default 或 active promotion。
- continuous_policy 当前仍是 `research / shadow_only`；未过 formal evidence、v2 gate、stable confirm 与 promotion gate 前，不得替代 active 执行链。
- 当前有效 continuous_policy 研究基线仍是 r39 allocation objective consolidation；r40-r74 均为 research / shadow 升级链或基础设施证据。
- r64 已产出 full-window strict Gold：`continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1`，`2010-01-04 -> 2026-04-10` observed strict window，`is_training_safe=true`，audit `ok`。
- r65 已新增 portfolio-set v5 后端，但 safe protocol 行为仍 source/receiver dead；不是策略有效性证据。
- r67 已用论文驱动 DFL-PG v1 替换 portfolio-set v5 内核默认目标、loss、oracle 与 profile default；tiny strict-Gold smoke `protocol_r67_paper_dfl_replace_v5_smoke_20260514_04` 完整跑通，但 training evidence 仍 `insufficient`，promotion gate 仍 `shadow_only`。
- r68 已新增 `portfolio_cashflow_decision_v1` 合同并打通 v5 prediction -> simulator -> release trace -> continuity metrics；tiny strict-Gold smoke `protocol_r68_cashflow_decision_v1_close_v5_smoke_20260514_03` 的 shadow source target=97、receiver target=139、intent conflict=0、cashflow valid=21/21，但仍只是 research / shadow translation-closure evidence。
- r69 已新增显式 `portfolio_set_v5_dfl_pg_v1_r69_value_arbitration` 研究线；traincheck `protocol_r69_value_arbitration_v5_behavior_traincheck_20260514_03` 显示 source target=1185、receiver target=160、wrong-side sell share=0，但 constraint violation 仍高且完整 tiny smoke `_01/_02` 在 evaluate 阶段被 TDX empty batch 中断；r69 尚未通过 behavior acceptance。
- r70 已修复 v5 版本边界与 r69 oracle violation：base 默认回到 `portfolio_set_v5_dfl_pg_v1`，r69 value arbitration 只在显式 profile/loss 或 artifact metadata 下启用；tiny strict-Gold smoke `protocol_r70_v5_version_boundary_oracle_repair_smoke_20260514_01` 完整通过 train/evaluate/shadow/export，oracle violation 近零、cashflow valid=1、intent conflict=0，但 training evidence 仍 `insufficient`，cash timing/reversal/source quality 仍失败，promotion gate 仍 `shadow_only`。
- r71 已新增显式 `portfolio_set_v5_dfl_pg_v1_r71_multistage_regret` 研究线，用 multi-stage regret / ordered goal-programming 改写行为质量目标；unit/contract/regression 已通过，但 `_01/_02/_03` evaluate 被 TDX singleton empty data 阻断、`_04` train 被 CUDA busy 阻断，尚无 completed tiny behavior smoke，不是 behavior acceptance。
- r72 已把 evaluate/shadow/export 接到通用 data lake evaluator；`protocol_r71_multistage_regret_v5_behavior_lake_smoke_20260515_02` 完整跑通且无 TDX empty blocker，但 source/receiver target 仍为 0，仍不是 r71 behavior acceptance。
- r73 已把 data lake 接入决策特征利用审计与 r71 collapse 诊断；`protocol_r73_lake_native_r71_collapse_repair_smoke_20260515_02` 完整跑通，eval/shadow 均保持 cashflow valid=1、intent conflict=0 且 source/receiver target 非零，但 training evidence 仍 `insufficient`、promotion gate 仍 `shadow_only`，行为质量仍有 cash timing、source quality、receiver-source spread 阻塞。
- r74 已新增显式 `portfolio_set_v5_dfl_pg_v1_r74_lake_behavior_quality` 研究线，并完成 tiny lake smoke `protocol_r74_lake_behavior_quality_v5_smoke_20260515_03`；cashflow valid=1、intent conflict=0、source/receiver 非零，source wrong-side sell、reversal、cash timing 与 shadow receiver-source spread 相对 r73 有改善，但 source/receiver 覆盖收缩、feature contract degraded rate=1.0、training evidence 仍 `insufficient`，仍不是 promotion 或 behavior-success verdict。
- 2026-05-23 已将 TDX-free data platform 升级到 V2：正式研究入口 lake-first，`tqcenter.py` / `pytdx` / `mootdx` 不再是 daily_research 主链路依赖；`refresh_daily` 支持 `--universe all_a|liquid500|file:<path>|symbols:<csv>`、真实交易日历、多 domain sidecar、Bronze/Silver 仲裁和显式 lake dataset 注册。
- 2026-05-27 执行端已改为手动-only：帮助页提供“刷新数据/信号 -> 生成交易计划 -> 模拟账户过账 -> 复核状态”的显式按钮流程；调度、轮询触发和一键每日流水线不属于当前产品面。
- 当前 fresh daily verdict：`daily_research/output/execution_app/daily_runs/20260526/verdict.json`，状态 `blocked:data_not_ready`，目标交易日 `2026-05-26`；完整证据见 `daily_research/brain/references/execution_daily_plan_state_machine_refactor_20260526.md`。

## 当前接管入口
- 默认读取顺序：`identity_layer.md -> state_center.md -> knowledge_center.md -> continuous_policy_design_contract.md -> operations_center.md -> governance_layer.md`。
- 首选工具入口：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<task>" --json`。
- 所有 `daily_research` 程序必须显式使用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`。
- 每日执行接管先看帮助页手动流程、作业证据和 daily verdict；Web 能打开、job succeeded 或 latest trade plan 存在都不能单独证明每日任务完成。
- PowerShell 中文显示异常时，先用显式 UTF-8 复读；不得直接判定文档损坏。
- `latest_*` 不得直接当真源；若 latest study/protocol/audit/ledger 不同源，必须使用 explicit run tag / protocol tag / dataset id。

## 当前主问题
- production 执行侧不是当前阻塞点；默认 active 继续由 `short_expert_policy_v5b` 承担。
- daily execution 当前阻塞点是 2026-05-26 数据源 readiness：候选交易日 formal refresh `market_daily` 为空，因此严格阻断并不生成新交易计划。
- `alpha_multi_horizon_utility_policy_v1` 的当前 blocker 是 target / loss / horizon-head stability：Stage 2.7 三个 target-normalization / horizon-head 候选 9/9 completed，但均未通过 stability gate；target normalization 能降低 30d concentration 且保留 rank/spread/hit 正信号，但 max negative months 仍为 4，Stage 3 architecture review 仍未解锁。
- continuous_policy 的核心瓶颈是组合日级资金分配：谁是 receiver、谁是 source、留多少 cash、承受多少 turnover / cost / drawdown。
- r53-r55 解决了部分 cash/exposure closure，但 source/reduce/exit 和 cash timing 没闭合。
- r56-r61 推进 release-first / core-v4 接线，证明诊断与部分接线有效，但行为仍未闭合。
- r62-r64 完成通用 data lake 与 full-window strict Gold；数据基础设施已不再是 strict 训练集的主阻塞。
- r65 用 portfolio-set v5 取代 MLP 主线，接线通过但行为失败：`release_first_source_intent_count=0`、`portfolio_daily_source_target_count=0`、`portfolio_daily_receiver_target_count=0`、`intent_translation_conflict_rate=1.0`。
- r67 修复第一层机制闭环：训练 target `source/receiver/cash` 非零且 `target_intent_translation_conflict_count=0`；但评估 source target 与 shadow translation 仍未闭合。
- r68 修复 v5 资金流翻译闭环：cashflow contract 在 `_03` shadow 21/21 天 valid，source/receiver target 均非零，release trace primary blocker 为 `none`。下一 blocker 已转为训练证据不足、source 选择质量、cash timing、drawdown/reversal，而不是 source/receiver translation dead。
- r69 将下一 blocker 进一步定位为：value arbitration 目标覆盖有所改善，但 receiver 覆盖不足、oracle constraint violation 偏高、完整 evaluate/shadow smoke 受 TDX 数据读取阻塞。
- r70 已解除“oracle feasibility / TDX empty batch”这两个机制阻塞；当前 blocker 继续收敛到 receiver 覆盖、cash timing、reversal、source positive-forward sell 与 sufficient training evidence。
- r71 已把上述 blocker 显式写入 multi-stage regret target/oracle/head calibration；r72 已解除 provider evaluate 阻塞；r73 已修复 lake smoke source/receiver target collapse；r74 已改善若干行为指标但暴露 feature-contract degradation 与 source/receiver coverage 收缩，下一步必须围绕特征合同健康、覆盖率、cash timing/source quality/receiver-source spread 与 sufficient training evidence 继续修正，再判断是否进入 strict resume。

## 当前优先级
- P0：冻结 live/default/promotion/active artifact，所有新线先保持 research / shadow-only。
- P1：每日执行端以手动帮助页流程、作业证据和 daily verdict 共同构成事实层；数据缺口严格阻断，不能回退旧交易日伪装“今日计划”。
- P2：保持脑区控制面简洁；长历史、完整复盘、长命令进入 `references/`。
- P3：`alpha_multi_horizon_utility_policy_v1` 下一步应收窄或重设 `15/20/30d` long-horizon utility family，并诊断 monthly regime / seed-specific negative months；只有 fullgrid seeds `7,11,19` 同时满足 rank/spread/hit 全正、mean monthly positive rate `>=0.75`、max negative months `<=2` 且 horizon concentration 不恶化，才允许 Stage 3 architecture review。
- P4：围绕 r71/r74 multi-stage regret 与 lake-native decision features 继续验证 receiver/deploy 平衡、cash timing、drawdown/reversal、source quality、feature contract health 与 sufficient training evidence；translation closure、oracle feasibility 和 lake source/receiver collapse 不再是当前主 blocker。
- P5：继续用 strict Gold dataset id 作为训练数据真源；realtime tail label 只可用于 research/audit。
- P6：补齐 TDX-free data platform 后续域：全 A universe discovery、交易日历、ST/退市/停牌、涨跌停、行业/概念、估值、资金/热点；这些进入 Bronze/Silver 后才能用于研究。
- P7：保持 study/protocol 单进程研究框架；长任务可用外部后台启动 + 前台轮询，但研究本体仍应可诊断、可恢复；默认轮询采用 `Wait-Process -Id <pid> -Timeout 7200`，以 PID 绑定等待支持提前完成即返回；`7200` 秒只是单轮前台等待窗口，耗尽后若 PID / 日志 / 产物仍推进且无代码错误证据，应继续下一轮轮询。

## 当前边界
- formal、recent、promotion、live 不得混写。
- smoke、dry-run、short-window check、interrupted wrapper、insufficient evidence、failed trial、realtime tail label 都不能升级为正式 verdict。
- safe screening 或代码合同结果不得改 `daily_research/output/active_execution_strategy.json`。
- failed / timeout trial 可写诊断，不得写 completed evidence；外层等待窗口耗尽本身也不得写成 failed evidence。
- outer study 未写 `study_summary.json` 时，只能写 protocol-level evidence，不能写 completed study verdict。

## 当前风险
- 文档继续堆 dated log 会削弱接管效率；只看 target-sum closure 会掩盖 release/source/receiver dead 与 intent translation conflict。
- 只增加 epoch、loss 或模型宽度可能掩盖 target construction 与 allocation semantics 断点；realtime Gold 若被误作 training-safe，会污染 completed training evidence。
- 后台 OS 进程轮询能降低交互超时风险，但不能替代 checkpoint、progress、artifact diagnostics。

## 最新证据索引
- r61 core-v4 decision wiring：`daily_research/brain/references/r61_release_first_decision_core_v4_status_20260514.md`。
- r62 DuckDB + Parquet research data lake：`daily_research/brain/references/r62_research_data_lake_status_20260514.md`。
- r63 brain-skill operating system：`daily_research/brain/references/r63_brain_skill_operating_system_status_20260514.md`。
- r64 full-universe strict Gold：`daily_research/brain/references/r64_full_universe_gold_data_lake_status_20260514.md`。
- r65 portfolio-set v5：`daily_research/brain/references/r65_portfolio_set_v5_status_20260514.md`。
- r66 brain maintenance：`daily_research/brain/references/r66_brain_maintenance_status_20260514.md`。
- r67 paper-DFL replace v5：`daily_research/brain/references/r67_paper_dfl_replace_v5_status_20260514.md`。
- r68 cashflow decision v1 close v5：`daily_research/brain/references/r68_cashflow_decision_v1_close_v5_status_20260514.md`。
- r69 value arbitration v5：`daily_research/brain/references/r69_value_arbitration_v5_status_20260514.md`。
- r70 v5 version boundary / oracle repair：`daily_research/brain/references/r70_v5_version_boundary_oracle_repair_status_20260514.md`。
- r71 multistage regret v5：`daily_research/brain/references/r71_multistage_regret_v5_status_20260514.md`。
- r72 data lake evaluator：`daily_research/brain/references/r72_data_lake_evaluator_status_20260515.md`。
- r73 lake-native r71 utilization：`daily_research/brain/references/r73_lake_native_r71_utilization_status_20260515.md`。
- r74 lake behavior quality：`daily_research/brain/references/r74_lake_behavior_quality_status_20260515.md`。
- 多 Horizon 交易效用命名迁移：`daily_research/brain/references/alpha_multi_horizon_utility_policy_mainline_rename_20260523.md`。
- 多 Horizon 交易效用首轮 / Stage 2.6 / Stage 2.7：`daily_research/brain/references/alpha_path20_horizon_discovery_result_20260523.md`、`daily_research/brain/references/alpha_multi_horizon_stage26_stability_root_cause_20260527.md`、`daily_research/brain/references/alpha_multi_horizon_stage27_target_head_stability_20260527.md`。
- PathPolicy 执行异常学习：`daily_research/brain/references/path_policy_execution_issue_learning_20260523.md`。
- TDX-free 数据平台决策：`daily_research/brain/references/tdx_free_data_platform_decision_20260523.md`。
- TDX-free 数据平台 V2：`daily_research/brain/references/tdx_free_data_platform_v2_20260523.md`。
- 机器索引：`daily_research/brain/references/evidence_registry.json`。

## 历史归档入口
- 完整快照 / 早期原文：`daily_research/brain/references/state_center_archive_20260510.md`、`daily_research/brain/references/state_center_history_raw_20260424.md`。
- 早期状态索引：`daily_research/brain/references/state_center_evidence_index_20260424.md`。
