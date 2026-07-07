# Daily Research 知识对象
快照日期：`2026-07-07`

本文件保存稳定对象类、长期事实和方法论。它不承载当前状态长卷，也不复刻历史证据；完整 rXX、长命令和 dated review 在 `references/`。

## Object Classes
### class `project_brain`
`instance`: `daily_research`
`definition`: 生产研究与执行分脑，连接研究、验证、候选执行、证据、接管记忆。
`root`: `H:\quant_project`
`environment`: `daily_research/environment.yml`；标准解释器是 `yolos` 环境。
`deprecated_root`: `H:\new_tdx64\PYPlugins\user` 只保留为历史证据和回滚说明。

### class `qdp_v2_data_base`
`owner`: `quant_data_platform`
`products`: active tables, dataset manifests, parquet shards, rebuildable caches/features, provider ingest outputs and quality proofs.
`consumer_contract`: `daily_research` 读取 explicit table/domain、dataset manifest 或 downstream pack。
`provider_contract`: `mootdx_online`、`BaoStock`、`CNInfo` 等在线源作为 QDP 上游 ingest / raw archive / normalization 对象。
`invariant`: 研究、训练和 diagnostics 不在 `daily_research` 中临时直连在线 provider。

### class `research_artifact_store`
`owner`: `daily_research`
`products`: sequence packs、memmaps、normalization、sample_index、labels、model outputs、prediction CSVs、study summaries and evaluation reports.
`preferred_root`: `daily_research/data/research_store`
`compatibility_roots`: none active; old `quant_data_platform/data/qdp_v2/research/` artifacts were physically migrated or deleted on 2026-07-07.
`invariant`: artifact manifests should declare owner project, source data base, artifact type, lifecycle and retention policy.

### class `research_program`
`properties`: objective、data contract、pool/PIT status、sample/label/target、features、model input、architecture、output semantics、loss、evaluation gate、execution-candidate bridge、evidence grade。
`methods`: `design_experiment()`；`run_scout()`；`promote_to_evidence_grade()`；`write_reference()`。
`quality_axes`: cost-aware return、rank/spread/hit lift、monthly stability、drawdown、turnover/cost drag、sample coverage、PIT/readiness。

### class `evidence_grade`
`levels`: `smoke_only`、`scout_only`、`evidence_grade`、`promotion_grade`
`semantics`: run completion, model-quality evidence and execution authority are separate dimensions.
`truth_sources`: explicit dataset id、manifest、run tag、protocol/study summary、reference、machine registry。
`weak_sources`: loose latest、single recent、unobserved realtime tail label、isolated terminal memory。

### class `execution_surface`
`state`: frozen skeleton / awaiting research rebuild.
`material_object`: `daily_research/output/active_execution_strategy.json`
`read_paths`: Web help page, job evidence, read-only daily verdict, active artifact inspection.
`change_method`: `governance_layer.procedure.execution_change`

## Long-Term Facts
- `daily_research` 当前研究消费 QDP v2 数据基底；旧 daily_research lake / copied manifest / single memmap 不再是数据 owner。
- QDP active data base and downstream pack status are recorded in `state_center.md` under `qdp_consumption`.
- 当前主要研究方向是 `seq100_path_value_research`：用过去 100 日路径和状态序列预测未来路径，路径摘要和 path value 从预测路径派生；收盘后短线选股经验保留为可复用研究线。
- Today-close anchor improves path-only test rank IC versus next-open anchor, but topK concentration is not yet decisive; treat as promising research evidence, not promotion evidence.
- `Path20` / `alpha_path20_neural_policy_v1` 是历史证据代号和代码 namespace，不再代表当前目标定义。
- continuous_policy 的长期思想是日级连续交易执行模型；当前不是 active/default 或执行解冻依据。
- daily execution 的事实层是手动流程、作业证据和只读 daily verdict；Web 可运行或旧 runtime state 只提供辅助线索。
- evidence registry v3 使用 `research_programs`、`study_families`、`run_tags` 三层索引。

## Long-Term Lessons
- 个股动作和组合资金分配是不同问题；真正目标是当前组合状态下的最优仓位调整集合。
- 卖出同时涉及继续持有机会成本、现金价值、资金来源责任和风险状态，比买入更难。
- 单项 gate 清零容易制造假进展；收益、月度质量、drawdown、source count、cash timing、exposure 和 intent conflict 要一起看。
- 更强模型不是自动解决方案；错误的 target、receiver/source semantics 或 evidence route 会被模型放大。
- 固定 horizon 不是目标本体；判断重点是赚钱相关排序、spread、hit lift、月稳和 calibration。
- 工程复杂度会制造循环；runner、profile、loss、diagnostics 应服务明确阻塞点。
- 执行异常不是研究结论；timeout、脚本入口失败、残留进程或资源挤占先归因，再决定证据等级。
- 数据资产要可复用、可审计、可查询；事实数据进入 QDP active data base，model-ready 训练 artifacts 进入 `daily_research/data/research_store`。
- 训练用数据集不是事实源；新 model-ready training artifacts 优先进入 `daily_research/data/research_store`，QDP active data base 只保存共享事实和质量证明。

## Research Line Index
### object `shortline_after_close_research`
`status`: current active research pointer.
`purpose`: 收盘后短线选股，围绕 D 收盘后更新、D+1 入场、T+1 约束下 D+2+ 退出的研究链。
`evidence`: see `state_center.md` and shortline references.

### object `alpha_v2_and_v2_reset_history`
`status`: historical / reusable infrastructure.
`usage`: 可作为模型、特征、数据基底和候选审查经验来源；不作为当前 active/default。

### object `multi_horizon_utility_history`
`status`: historical method library.
`usage`: 多 horizon utility、ranking、calibration、candidate bridge 和 local-state 思路可复用；旧 fixed 20d 语义不作为当前目标。

### object `continuous_policy_history`
`status`: research / shadow lineage.
`usage`: 组合资金流、source/receiver/cash、release-first、DFL-PG、value arbitration 等经验可作为长期方法论；当前不承担执行解冻。

### object `qdp_v2_manifest_first_ingestion`
`status`: current data substrate route.
`usage`: provider update/import、table rebuild、coverage audit and downstream pack production all route through QDP.

## Pure Functions
- `classify_run(run)`: output status and evidence grade separately.
- `compare_models(a, b)`: compare only under aligned dataset, pool, feature, label, cost, validation and evidence budget.
- `route_data_need(requirement)`: returns QDP provider/canonical task, not direct daily provider call.
- `activate_history_line(task)`: historical line activates only when task names it or evidence lookup requires it.

## Document Boundaries
- Current object instances: `state_center.md`
- Stable object classes and lessons: `knowledge_center.md`
- Procedure entries and commands: `operations_center.md`
- Object invariants and guard selection: `governance_layer.md`
- Full historical evidence: `references/`
- Machine index: `references/evidence_registry.json`
