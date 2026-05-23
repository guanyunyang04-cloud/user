# Daily Research 完整代码库检阅 2026-05-23

本记录是 2026-05-23 对 `daily_research` 完整代码库的 successor 检阅。
它更新 `brain_system_full_codebase_review_20260522.md` 中已经过时的工作区路径、模块地图和当前主线解释。

本记录不是 promotion authority，不是 live/default authority，也不是 completed strategy evidence。

## 范围

- 状态：`codebase_review / handoff_reference / no promotion`。
- 工作流：`brain`。
- 当前工作区根目录：`H:\quant_project`。
- active artifact 影响：`daily_research/output/active_execution_strategy.json` 保持不变。
- 检阅对象：`daily_research` tracked source、brain、tests 与显式 output summary；`output/cache/archive` 下的大型生成产物不做宽泛递归真源推断。
- 目的：固化当前代码结构、durable 主线、风险边界和后续安全动作。

## 使用证据

- `tools.brain.workflow capsule --task "详细检阅 daily_research 完整代码库，深入分析并做出更新" --json`。
- `daily_research/brain/state_center.md`。
- `daily_research/brain/knowledge_center.md`。
- `daily_research/brain/operations_center.md`。
- `daily_research/brain/brain_manifest.json`。
- `daily_research/brain/references/mainline_review_current.md`。
- `daily_research/brain/references/brain_system_full_codebase_review_20260522.md`。
- `daily_research/brain/references/full_codebase_review_20260515.md`。
- `daily_research/brain/references/tdx_free_data_platform_v2_20260523.md`。
- `daily_research/output/path_policy/studies/path20_horizon_discovery_no_alpha_gru_liquid500_h1_2_3_5_8_10_15_20_30_du_cost20_hit10_dd010_20260523_01/study_summary.json`。
- 排除 `output/cache/archive` 后，对 `daily_research/{baseline,continuous_policy,data_lake,data_platform,deep_alpha,execution,path_policy,tools}` 的 source scan。

## 事实

- 检阅时当前分支是 `main`，领先 `origin/main` 2 个提交。
- 修改前工作树干净，active artifact diff 为空。
- 主脑 capsule 将本任务路由到 `daily_research`。
- capsule guard 状态为 `ok`，但有两个 warning：loose latest stale 与 unregistered latest output tags。
- `daily_research` 仍是 production research and execution cortex。
- active execution material truth 仍是 `daily_research/output/active_execution_strategy.json`。
- 当前 live/default label 仍是 `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`。
- 当前 effective live execution profile 仍是 `regoff_k1_20d_ensemble_native_anchor`。
- `continuous_policy` 仍是 `research / shadow_only`。
- `path_policy` 仍是 `research / shadow-only`。
- 当前 path_policy research pointer 是 `alpha_multi_horizon_utility_policy_v1`。
- `Path20` / `alpha_path20_neural_policy_v1` 保留为历史证据代号和代码 / tag namespace，不再代表当前目标定义。
- reusable strict training-safe Gold dataset 仍是 `continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1`。
- 当前 repaired policy input bundle 仍是 `policy_input_bundle__7c8f58d851bce8179e1e9e2d`。
- TDX-free data platform V2 已成为一等 source module：`daily_research/data_platform`。
- 排除 `output/cache/archive` 后，当前 Python source count 为 318。
- 当前 test source count 为 61。
- 模块 Python 文件数：
  - `continuous_policy`: 74。
  - `deep_alpha`: 53。
  - `tools`: 53。
  - `baseline`: 45。
  - `path_policy`: 33。
  - `execution`: 32。
  - `data_lake`: 15。
  - `data_platform`: 11。
  - root files: 2。
- 测试文件数：
  - `continuous_policy`: 28。
  - `path_policy`: 16。
  - `data_platform`: 5。
  - `baseline`: 4。
  - `execution`: 3。
  - `data_lake`: 2。
  - `tools`: 2。
  - `deep_alpha`: 1。
- 最大 Python 风险面：
  - `daily_research/continuous_policy/model_seq_v3.py`: 12357 lines。
  - `daily_research/continuous_policy/portfolio_simulator.py`: 7011 lines。
  - `daily_research/continuous_policy/pipeline_utils.py`: 5879 lines。
  - `daily_research/continuous_policy/run_self_optimizing_study.py`: 5007 lines。
  - `daily_research/continuous_policy/tests/test_portfolio_daily_strategy_contracts.py`: 4812 lines。
  - `daily_research/path_policy/run_alpha_path20_protocol.py`: 3789 lines。
  - `daily_research/continuous_policy/model_portfolio_set_v5.py`: 3356 lines。
  - `daily_research/continuous_policy/analyze_behavior_gap.py`: 3319 lines。
  - `daily_research/continuous_policy/research_profile_registry.py`: 3104 lines。
  - `daily_research/path_policy/forecast_training.py`: 3058 lines。
  - `daily_research/baseline/generate_daily_trade_plan.py`: 2896 lines。
  - `daily_research/deep_alpha/run_deep_alpha_research.py`: 2730 lines。
  - `daily_research/execution/update_default_candidate_production.py`: 1628 lines。
- `daily_research/data_platform/contracts.py` 默认拒绝 TDX-family provider name。
- `daily_research/continuous_policy/state_builder.prepare_policy_inputs()` 有测试覆盖，正式入口拒绝 `tq/tdx/pytdx/mootdx`。
- `daily_research/path_policy/run_alpha_path20_protocol.py` 对 RL 和 forecast stage 拒绝 loose `latest/default/latest_*` lake dataset id。
- `daily_research/execution/active_manifest_guard.py` 在缺少显式确认时阻止 active manifest write。
- 当前 active manifest 仍携带旧 `H:\new_tdx64\PYPlugins\user` 绝对路径和 `data_source: tq`。
- 当前 brain 与 README 已声明 `H:\quant_project` 是工作区根目录，旧 `H:\new_tdx64\PYPlugins\user` 不是项目 source root。

## 推断

- `daily_research` 不是单模型 repo，而是研究、数据入口、data lake、验证、执行与治理合一的系统。
- 当前最大错误风险仍是边界混写：把 short-alpha live materialization、多 horizon ranking evidence 和 continuous allocation ambition 说成一个已经集成且可 promotion 的模型。
- 2026-05-22 检阅在三个方面过时：旧工作区根目录、缺少 `data_platform_v2`、仍把 `alpha_path20_neural_policy_v1` 当作当前 path_policy pointer。
- active manifest 的旧路径字段是 live artifact migration debt，不是把当前开发迁回旧 TDX 插件目录的依据。
- TDX-family 代码仍存在于 legacy baseline/deep_alpha/tools 路径中；准确边界不是“代码库无 TDX 字符串”，而是“正式新研究和数据平台边界默认 TDX-free，legacy 路径隔离或显式 gated”。
- `data_platform_v2` 改变了证据模型：online fetch health、provider coverage、CSV import、Bronze/Silver registration 都是 data admissibility evidence，不是策略有效性证据。
- `path_policy` 已成为当前最活跃的 opportunity-ranking research surface；代码 namespace 仍带 `path20_...`，新结论必须用 multi-horizon utility 语言，避免目标定义回退。
- `continuous_policy` 仍是组合 allocation 语义最丰富的研究面，但 blocker 仍是行为质量与证据充分性，不只是训练基础设施。
- `brain`、`tools` 和 tests 属于 control plane，不是模型旁边的可选文档；它们负责阻止历史错误重复。

## 假设

- “完整代码库”指 tracked source、brain、tests、manifest 与显式最新 evidence summary，不是对 `output/cache/archive` 下每个生成物做全量审计。
- 本次检阅允许更新 brain references 与接管地图，但不授权更新 `daily_research/output/active_execution_strategy.json`。
- 历史 reference 和 active manifest 中的旧路径字符串保留，除非未来有显式 migration / promotion 任务授权 artifact rewrite。
- 本次不启动 training、study、production refresh、active activation、broker automation 或 live/default change。

## 主线地图

- live/default execution line：`deep_alpha` / `short_alpha` / `short_expert_policy_v5b` 与 execution-aligned production bridge。
- execution alignment line：`execution` 与 `baseline/generate_daily_trade_plan.py` 将 active manifest 物化为日常 trade plan 行为。
- baseline factor/rule line：最早可执行研究线，仍是兼容与交易计划 plumbing。
- advanced ML/state/attack-defense line：历史 model-family comparison 与 validation foundation。
- continuous policy line：日级组合资金移动、source/receiver/cash、simulator semantics、training contracts 与 shadow protocols。
- multi-horizon utility line：当前 path_policy research pointer，目标是多 horizon 交易效用排序。
- historical Path20 neural line：2026-05-17 到 2026-05-23 的 path forecasting 证据与代码 namespace。
- Path20 sequence/RL line：带 leakage guard 与 no-oracle input 的 secondary shadow comparison route。
- data lake line：DuckDB/Parquet catalog、strict Gold、policy input bundles、pool views 与 sector board views。
- data platform line：非 TDX provider refresh/import boundary、Bronze/Silver normalization、domain sidecars 与 lake registration。
- brain/tools governance line：capsule、evidence registry、doc guard、integrity check、frontier scanner、active artifact guards 与 selective verification。

## 模块检阅

- `daily_research/brain`：项目事实层，保存当前状态、稳定规则、写回路线和 durable reference docs。
- `daily_research/tools`：维护与 guard 层；`project_consistency_check.py` 仍检查 active execution semantics，主脑平台工具位于 workspace `tools/brain`。
- `daily_research/data_platform`：TDX-free ingestion boundary，负责 provider contracts、provider manager、refresh、CSV import、domain normalization、coverage、conflict reports 与 lake registration。
- `daily_research/data_lake`：数据集身份与存储层，负责 catalog、strict/realtime label completeness、policy input loading、pool views 与 sector board views；它不是在线数据源。
- `daily_research/path_policy`：opportunity-ranking research layer，强制显式 lake dataset id、拒绝 loose latest id、写 shadow-only summary，并把 forecast/RL evidence 与 allocator/live evidence 分开。
- `daily_research/continuous_policy`：daily allocation research layer；最高风险面是 simulator semantics、action/head/loss/profile contracts、training contracts、source/receiver/cash behavior 与 promotion gate classification。
- `daily_research/deep_alpha`：live-anchor lineage 与更长周期 alpha 研究；当前 active default 来源仍在这一线。
- `daily_research/baseline`：legacy factor/ML 与 trade-plan compatibility layer；仍有大量 `tq` 路径，除非命令显式通过当前 guard，否则应视作 historical / compat。
- `daily_research/execution`：production boundary；`refresh-production-default`、`activate-single-mapping` 与 active manifest writes 都是 danger surfaces。
- `daily_research/output`、`daily_research/cache`、`daily_research/archive`：artifact stores；必须用 explicit tags、dataset ids 和 registries，不从 timestamp 或 loose latest pointer 推断真相。

## 当前前沿解释

- 最新 multi-horizon evidence `path20_horizon_discovery_no_alpha_gru_liquid500_h1_2_3_5_8_10_15_20_30_du_cost20_hit10_dd010_20260523_01` 已完成，verdict 为 `forecast_test_confirmed`。
- 该 run 使用 `policy_input_bundle__7c8f58d851bce8179e1e9e2d`、liquid500 pool view `policy_pool_view__c11400fa72ad263f3d1eecfa`、sector board view `policy_sector_board_view__ed15b2873f544e9e9b24aae5`、GRU static-context family、seed `7`、horizon grid `1,2,3,5,8,10,15,20,30`。
- validation/test trade-utility ranking 为正，但 predicted best horizon 明显偏向 `30d`。
- 这支持 calibration research，不支持 liquid800 expansion、allocator、replay、promotion 或 live/default change。
- capsule frontier 报告仍有 unregistered recent output tags，因此回答 “latest” 状态时必须使用 explicit study tag，直到 registry/reference 完成 reconciliation。

## 关键风险

- 未获 explicit promotion authority 就修改 active manifest。
- 未做 explicit tag 与 freshness check 就把 `latest_*`、`latest`、`default` 或 timestamp-newest artifact 当真源。
- 把 smoke、dry-run、failed、interrupted、unit、realtime-tail 或 protocol-only output 写成 completed strategy evidence。
- 通过 legacy baseline/deep_alpha default 把 TDX-family fallback 带回 formal research。
- 把 active manifest 的旧路径字段当成当前工作区根目录。
- 把 `data_platform_v2` provider health 或 CSV import 当成模型 performance evidence。
- 把 multi-horizon ranking evidence 当成 allocator、replay 或 live evidence。
- 因 v5/v6 backend 能跑 GPU/epoch/resume 就把它们视为 promotable；`formal_torch_portfolio_set_v5` 与 `formal_torch_decision_core_v6` 仍是 `promotable=false`。
- 修改 `portfolio_simulator.py`、`research_profile_registry.py`、`run_continuous_policy_protocol.py`、`run_alpha_path20_protocol.py` 或 active manifest scripts 时缺少 targeted tests。
- 让 slow path_policy integration tests 因短 timeout 或残留 pytest 进程制造 false negative。

## 本次更新

- 新增本 successor reference。
- 更新 `daily_research/brain/brain_manifest.json` body map，使一等 current modules 包含 `continuous_policy`、`path_policy`、`data_lake` 和 `data_platform`。
- 更新 `daily_research/README.md` 模块地图，补入 `path_policy` 与 `data_lake`。
- 更新 `daily_research/environment.yml` 注释，明确 `tqcenter` 是 legacy-only，不是当前 formal research dependency。
- 在 `daily_research/brain/knowledge_center.md` 增加本 full-codebase review successor 索引。
- 重建 `daily_research/brain/references/evidence_registry.json`。

## 后续允许动作

- 除非未来 explicit promotion task 授权，否则继续冻结 live/default 与 active artifact。
- multi-horizon utility 下一步只跑 constrained horizon-score / calibration variants；在 multi-seed 前必须保住 validation/test spread、hit lift 与 monthly stability。
- continuous_policy 下一步聚焦 feature contract health、source/receiver coverage、cash timing、source quality、receiver-source spread 与 sufficient training evidence，再讨论 strict resume 或 promotion。
- data platform 后续 provider health / refresh smoke 只记录为 data admissibility work，不记录为 strategy evidence。
- governance 下一步先通过 evidence registry / reference updates reconcile unregistered latest output tags，再回答 loose latest 类问题。
- active manifest 旧路径迁移必须单独做 explicit migration plan，含 rollback 与 guard validation；不能作为本次 review 的副作用。

## 验证

本次 review 应运行以下验证：

```powershell
git diff -- daily_research/output/active_execution_strategy.json
git diff --check
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow evidence-index --rebuild --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.doc_guard check
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json
```

本次不运行 full pytest suite、training、protocol、study、production refresh、broker action 或 active manifest write。
