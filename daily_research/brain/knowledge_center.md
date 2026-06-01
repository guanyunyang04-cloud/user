# Daily Research 知识中枢

快照日期：`2026-05-26`

## 1. 稳定事实
- `daily_research` 同时负责研究、formal 验证、recent 验证、production full-fit、live 执行和接管治理。
- 当前工作区根目录是 `H:\quant_project`；旧 `H:\new_tdx64\PYPlugins\user` 不是项目真源。
- strongest-model research winner、deployable learned-control、live mainline 必须显式区分。
- 当前 live 默认执行仍由 `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active` 承担。
- `daily_research/environment.yml` 是依赖环境真源；任何程序都必须在 `yolos` 环境下运行。
- 当前统一权重语义是 `research_raw_target_weight`；当前统一上限语义是 `follow_research_raw_no_global_cap`。
- 所有主线都是可显式切换的当前工作指针；切换后按切换后的主线继续，旧主线保留为历史或对照证据。
- 多 Horizon 交易效用排序当前研究主线指针是 `alpha_multi_horizon_utility_policy_v1`；这是 research mainline，不是 live/default 或 active execution mainline。
- evidence registry v3 使用三层证据索引：`research_programs` 表示稳定研究主线，`study_families` 表示阶段/实验族，`run_tags` 表示物理 run 实例；registry 只输出和读取这三层，不再存在旧 `study_tags` 字段。
- `Path20` / `alpha_path20_neural_policy_v1` 是历史证据代号和代码 namespace，不再代表当前目标定义；历史 run 仍按原字符串引用，新研究应写成多 horizon utility / ranking / calibration，而不是固定 20 日路径预测。
- continuous_policy 的终局目标是日级连续交易执行模型，不是固定调仓或人工执行桥。
- daily execution 的权威事实层是 Web 帮助页的手动步骤、作业证据和只读 daily verdict；Web 可运行、job succeeded、latest trade plan 或旧 runtime state 都不能单独证明每日任务完成。
- r39 仍是 continuous_policy 有效证据基线；r40-r74 是 research / shadow 升级链或基础设施证据。
- r64 full-window strict Gold 是当前 reusable training-safe Gold 数据集；realtime Gold 仍不能作为 completed training evidence。

## 2. 硬规则
- 必须 brain-first；默认接管顺序为 `identity -> state -> knowledge -> design contract -> operations -> governance`。
- formal、recent、promotion、live 不得混写。
- `latest_*` 文件不得被直接视为无条件真源；必须先判断 freshness，同源性和 explicit tag。
- code contract、unit tests、smoke、dry-run 只能证明接线正确，不能证明策略有效。
- failed / timeout / interrupted trial 只能写诊断，不得写 completed evidence；timeout 还必须先区分外层等待窗口耗尽、时间没给足、真实卡死和代码失败。
- realtime tail label 必须显式标记 unobserved，不得计入 completed training evidence。
- active artifact diff 是硬失败。
- 每日执行数据缺口必须严格阻断：候选交易日 `market_daily` 为空或 required domain blocked 时，只能写 `blocked:data_not_ready`，不得自动回退到上一完整交易日生成“今日计划”。
- 执行端只允许手动触发：Web 帮助页按钮或明确 CLI 单项任务才可提交作业；调度、轮询触发和一键每日流水线不属于当前产品面。
- 主脑文档只保留控制面；长历史、完整 rXX 证据、长命令和复盘进入 `references/`。
- 实验证据必须按预算可信度分级；不得把低预算 run 包装成模型质量结论：
  - `smoke_only`：只验证代码、数据、shape、loss 接线、artifact 落盘和入口可运行；不得解读模型优劣。
  - `scout_only`：只用于粗筛方向和生成下一步候选；不得淘汰主线、触发 stage gate、架构扩张、promotion 或 live/default 讨论。
  - `evidence_grade`：才允许支撑模型质量比较、阶段 gate 或方向选择；必须有预先声明的充分训练预算、有效 early stopping 空间、多 seed 覆盖、validation 收敛/稳定性检查和 test confirm。
  - `promotion_grade`：在 `evidence_grade` 之上，还必须满足正式 gate、跨时期/月度质量、成本、drawdown、replay/allocator 或执行约束，以及 active artifact 边界。
- 对 epoch 模型，固定 `2` epoch、`max_epochs_reached`、best epoch 贴近最后一轮、单 seed 或缺 validation 收敛证据的 run，只能标为 `smoke_only` 或 `scout_only`；即使 run status 是 completed，也不是 completed model-quality evidence。

## 3. 长期教训
- 个股动作和组合资金分配不是同一问题；真正目标是当前组合状态下最优仓位调整集合。
- 卖出比买入更难，因为卖出同时涉及继续持有机会成本、现金价值、资金来源责任和风险状态。
- 只追求单项 gate 清零会制造假进展；必须同时看收益、月度质量、drawdown、source count、cash timing、exposure 和 intent conflict。
- Clean target-sum closure 可以与 release/source/receiver flow disconnected 同时存在；closure 不是行为闭合的充分条件。
- 更强模型不是自动解决方案；若 target construction、receiver/source semantics 或 evidence route 错，放大模型只会更快放大错误。
- 固定 horizon 不是目标本体；当前 path_policy 目标已从“20 日路径预测”迁移为“多 horizon 交易效用排序”，后续判断以赚钱相关排序、spread、hit lift、月稳和 calibration 为主。
- 工程复杂度会制造循环；runner、profile、loss、diagnostics 必须减少活动面，服务明确阻塞点。
- 执行异常不是研究结论：pytest timeout、脚本入口失败、残留进程或资源挤占必须先做根因定位；若只是时间没给足，应移除或绕开该限制并继续受监管轮询；可复现且可修的问题应同时写入 reference、修工程入口或验证选择，并增加防复发测试。
- PathPolicy forecast dataset 全文件慢测源于完整 synthetic feature/label/horizon risk 构造；默认轻量验证应使用快速合同测试，完整慢测保留为 deferred long verification。
- `daily_research.path_policy.run_alpha_path20_protocol` 的标准入口是 `python -m ...`；直接脚本入口允许作为容错 smoke，但新命令记录和 reference 默认写包级入口。
- 数据集必须可复用、可审计、可查询；pickle/cache 可兼容，但新训练集应进入 DuckDB + Parquet data lake。
- `lake` 是研究存储真源，不是在线数据源；每日更新源是 `daily_research.data_platform` 的非 TDX online providers，写入 Bronze/Silver 后才能注册为研究 lake dataset。V2 默认使用 `--universe all_a` 和真实交易日历，CSV 只能作为入湖导入/补洞通道，不能被正式研究直接读取。
- 每日任务是手动 runbook：先确认 readiness，再手动刷新数据/信号、生成交易计划、模拟账户过账、复核状态；任一步数据缺口或 blocker 都必须停下并保留证据。
- 旧 runtime 只可归档为事故证据；测试和诊断必须使用隔离 runtime root，不能污染真实 `daily_research/output/execution_app`。
- TDX-family 已退出正式研究主链路：`tqcenter.py`、`pytdx`、`mootdx` 不得作为 `daily_research` 默认或正式 provider；若旧脚本保留这些名字，只能视为 legacy/historical path。
- 工作区迁移后，旧通达信插件 `user` 路径只能出现在历史 reference 或回滚说明中；新接管、新数据、新命令必须以 `H:\quant_project` 为根。
- 脑区是项目事实真源，skills 只是流程入口，不复制长历史。

## 4. 研究主线索引
- 滚动总览入口：`daily_research/brain/references/mainline_review_current.md`；用于检阅立项以来所有 durable 主线，但不替代 `state_center.md` 或 active artifact。
- 完整代码库检阅 successor：`daily_research/brain/references/brain_system_full_codebase_review_20260523.md`；用于当前模块地图、全库风险和 2026-05-22 旧路径/旧主线纠偏。
- `alpha_multi_horizon_utility_policy_v1`：当前 path_policy research pointer；目标是多 horizon 交易效用排序，首轮 horizon grid 为 `1,2,3,5,8,10,15,20,30`，当前 blocker 是 predicted best horizon 向 `30d` 塌缩。
- 近期 multi-horizon 阶段 family：`stage1_output_aux_grid`、`stage2_horizon_grid_calibration`、`stage25_stability_calibration`；新增 run tag 不应通过扩 evidence tag 前缀来代表新主线，必须归入既有或新声明的 `study_family`。
- `alpha_path20_neural_policy_v1`：2026-05-17 到 2026-05-23 的 Path20 neural-policy 历史主线；其 evidence 仍有效，但新结论必须按 `alpha_multi_horizon_utility_policy_v1` 解释。
- `alpha_path20_sequence_policy_v1`：shadow comparison / secondary research route；除非未来显式切换，不代表当前主线。
- `data_platform_v2` / TDX-free lake-first ingestion：当前数据入口主线；provider refresh/import 才能在线取数，正式训练、评估和 diagnostics 必须读 explicit lake dataset id。
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
- r71：新增显式 multi-stage regret 研究线，把 source rebound、receiver deploy regret、cash defense regret、rotation spread regret、reversal action regret 与 crowding diagnostics 写入 v5 target/oracle；代码与测试闭合，但 tiny behavior smoke 仍被 TDX singleton / CUDA blocker 中断，不能视为 behavior acceptance。
- r72：通用 data lake evaluator 已接入 evaluate / shadow / export，绕开 TDX singleton empty；`protocol_r71_multistage_regret_v5_behavior_lake_smoke_20260515_02` 完整跑通但 source/receiver target 仍为 0，因此是 infrastructure evidence，不是 r71 behavior acceptance。
- r73：新增 lake-native decision feature bundle 与 utilization report，并修复 r71 lake smoke source/receiver target collapse；`protocol_r73_lake_native_r71_collapse_repair_smoke_20260515_02` 完整跑通且 source/receiver 非零，但 cash timing、source quality、receiver-source spread 与 training evidence 仍未达标，仍不是 promotion 或 behavior-success verdict。
- r74：新增显式 lake behavior-quality 研究线，把高价值 lake 特征、feature contract severity、cash timing/source quality/receiver-source spread 诊断接入 v5；`protocol_r74_lake_behavior_quality_v5_smoke_20260515_03` 完整跑通且保持 cashflow/intent 闭合，多个行为指标相对 r73 改善，但 source/receiver 覆盖收缩、feature contract degraded rate=1.0、training evidence 仍 insufficient，仍不是 promotion 或 behavior-success verdict。

## 5. 当前方法论
- 先直接 protocol smoke，再 study dry-run，再 safe screening；不得跳到 confirmatory。
- 每个重大研究结论必须写成 facts / inferences / assumptions / boundary。
- explicit dataset id、protocol tag、run tag 和 reference doc 优先于 loose latest。
- 新多 horizon utility 实验必须归入明确的 `research_programs` 和 `study_families`；run tag 只表示物理实例，应显式写 pool、feature、model、seed、年份、output/loss、成本参数和 horizon grid，不得让命名把目标拉回固定 20 日路径误差。
- 旧 multi-horizon / short_v5b 等价 universe 是沪深主板口径，必须剔除创业板和科创板；新 lake / pool view 若用于旧线复刻或同口径比较，必须显式排除 `300,301,688,689` 前缀，并在 manifest 中保留 `exclude_symbol_prefixes`。
- corrected mainboard-only rebuild 通过 pool/memmap hard validation 后，仍必须把 `near_pass` 与 full pass 分开：rank IC、spread、monthly stability 和 concentration 过线不能抵消 hit lift min 为负；在旧 `156` feature schema 或 short_v5b payload 不可 replay 时，不得宣布新模型优于旧执行模型。
- TQ/BaoStock lineage audit 只能回答数据与标签是否等价；TQ read-only adapter 必须先用旧脚本路径调用 `tq.initialize(...)` 再读 `get_market_data`。若 TQ runtime 初始化失败，结论必须停在 `blocked_tq_unavailable`，不能把缺失 diff 解释为 BaoStock 差、TQ 优或旧结论失效；若 label 等价但 OHLCV 有 amount/missing 差异，也不能直接宣布旧 Stage 2.8 已 file-backed replay。
- 模型输入、架构、输出、loss、horizon grid 或训练配置的结论必须声明证据等级；低预算探索可以提出 follow-up，但不能作为“更好/更差/已失败/已通过”的稳定判断。
- 正式训练、评估和 diagnostics 必须读取显式 `policy_input_bundle__...` / Gold dataset id；不得在训练或诊断过程中临时在线抓取行情。
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
