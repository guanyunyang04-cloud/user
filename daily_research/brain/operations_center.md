# Daily Research 操作中枢

快照日期：`2026-05-31`

## 默认操作纪律
- 默认工作分支：`main`。
- 当前工作区根目录：`H:\quant_project`。
- Python 入口：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe`。
- 进入本分脑前，必须先由主脑 `tools.brain.workflow capsule` 路由到 `daily_research`。
- `H:\new_tdx64\PYPlugins\user` 已退出本项目主链路；历史 reference 中旧路径只作历史证据，不作为当前入口。
- 不使用 `KMP_DUPLICATE_LIB_OK` 作为默认方案。
- 不触碰 `daily_research/output/active_execution_strategy.json`，除非有明确 promotion 决策。
- 2026-06-01 起执行端进入冻结/保留骨架/等待重建状态：允许只读状态、数据 readiness、候选 backtest wrapper、候选 trade-plan wrapper 的研究评估用途和 payload inventory；禁止 live/default、paper/broker、正式 trade plan 生产用途、active manifest promotion、production root 重建、自动化每日执行和无授权删除执行合同。
- 2026-05-31 恢复边界：`daily_research/output/` 与 `daily_research/cache/` 曾被误删，当前空目录骨架只保证 manifest/integrity 入口存在；不得把缺失 payload 当成近期结论失效，也不得凭 state 文本手工重造 active artifact。
- 不把 smoke、dry-run、failed trial、interrupted outer study、realtime tail label 写成 completed evidence。
- 长训练或 study 需要 progress JSONL、latest progress JSON、stdout/stderr log 和明确 tag。

## 项目地图
- workspace root：`H:\quant_project`；brain 真源：`daily_research/brain/`。
- research data lake：`daily_research/output/research_data_lake/`。
- studies：`daily_research/output/path_policy/studies/`、`daily_research/output/continuous_policy/studies/`；protocols：`daily_research/output/continuous_policy/protocols/`。
- active artifact：`daily_research/output/active_execution_strategy.json`；execution app：`daily_research/execution/run_execution_app.py`；daily verdict root：`daily_research/output/execution_app/daily_runs/`。
- 以上 output/cache 路径在真实 payload 恢复前只代表目标位置；文件级 evidence lookup、replay、active artifact inspection 和 `project_consistency_check.py` 可能失败，结论回答需回到 brain references 与用户确认边界。

## 高频 Brain 命令
- 主脑 task capsule：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<task>" --json`
- current frontier freshness：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow current-frontier --json`
- explicit evidence status：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow status --workflow continuous_policy --run-tag <run_tag> --json`
- evidence registry rebuild：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow evidence-index --rebuild --json`
- evidence query：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow query --q <r_id|tag|dataset_id> --json`
- selective verification plan：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow verify-plan --json`
- brain guards：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.doc_guard check`
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json`

凡涉及“接管、当前状态、下一步、是否训练、主线推进到哪”的问题，必须先运行主脑 task capsule 或 current frontier；若 `brain_may_be_stale=true`，先读 output explicit tags 与 summary，再回答当前状态。

## Continuous Policy 运行口径
- Direct protocol smoke 先于 dry-run study，dry-run study 先于 safe screening。
- 默认 protocol runner 为 `in_process`；不恢复 r57 父/子进程 watchdog。
- Study runner 是逻辑编排层：选 trial、写 plan、调用 protocol、汇总 `study_summary.json`。
- Protocol 是单次训练/评估/shadow/export 层，写 `protocol_summary.json`。
- 若 `protocol_summary.json` 存在但 `study_summary.json` 缺失，只能记为 protocol-level evidence。
- 后台运行建议只把整个 study 作为一个 OS 后台进程启动；study 内部仍保持 `protocol_runner_mode=in_process`，前台只轮询 progress / PID / logs / summaries。
- 长训练或 study 的默认轮询实现为 `Start-Process -PassThru` 记录 PID，并用 `Wait-Process -Id <pid> -Timeout 7200` 等待；`7200` 秒只是单轮前台等待窗口，进程提前结束时立即返回，随后解析 progress、日志、summary、checkpoint 与评估产物。
- 单轮等待窗口耗尽后，先查 PID、exit code、日志、progress、summary / checkpoint / artifact 时间戳；若进程仍在推进且没有明确代码错误、资源危险或用户停止指令，继续下一轮 `Wait-Process -Id <pid> -Timeout 7200`，不得停止任务或写成 failed evidence。

## Multi Horizon Utility 运行口径
- 当前 path_policy 研究主线名：`alpha_multi_horizon_utility_policy_v1`，中文名为“多 Horizon 交易效用排序主线”。
- 旧 `Path20` / `alpha_path20_neural_policy_v1` / `path20_...` 字符串保留为历史 run 证据和代码 namespace；不得批量改写历史 tag，也不得把旧名解释成当前仍以固定 20 日路径预测为目标。
- 新实验必须显式写入 research program、study family 和 run tag：`research_programs` 查询稳定主线，`study_families` 查询阶段/实验族，`run_tags` 查询物理实例；run tag 需包含 pool、feature、model、seed、年份、output/loss、成本参数和 horizon grid，不得靠扩标签前缀表达新主线。
- 当前下一步只允许 constrained horizon-score / calibration 研究；不跑 liquid800、allocator、replay、live/default、promotion，除非新的 liquid500 seed-7 calibration 结果先通过 gate。
- 标准代码入口继续使用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.run_alpha_path20_protocol`，因为这是包级入口；直接脚本 `daily_research/path_policy/run_alpha_path20_protocol.py` 只作为容错 smoke，不能替代文档推荐入口。
- rebuild 差异审计入口：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.rebuild_lineage_diff_audit`
- 审计默认读取 `mh_rebuild_infra_v2_fullpool_anchor_20260531_01` 与 seed7/11/19 rebuild artifacts，输出 `rebuild_lineage_diff_audit.json`、`rebuild_lineage_diff_audit.md`、`rebuild_seed_monthly_spread.csv` 和 `rebuild_feature_columns.csv`；该命令只读研究 artifacts，不训练、不 promotion、不写 active/default。

## 实验预算可信度纪律
- 启动任何会影响模型输入、架构、输出、loss、horizon grid、stage gate 或后续方向选择的实验前，必须在计划或 tag 说明中声明证据等级：`smoke_only`、`scout_only`、`evidence_grade` 或 `promotion_grade`。
- `smoke_only` 只验证入口、shape、loss、数据和 artifact；允许 `1-2` epoch / 单 seed，但结论只能写“可运行/不可运行”。
- `scout_only` 只用于粗筛下一步候选；允许较小预算，但必须标明不能作为模型优劣、stage gate、架构扩张、promotion 或 live/default 的依据。
- `evidence_grade` 才能支撑模型质量比较或阶段决策；默认至少 `3` seeds，关键候选优先 `5` seeds，epoch 预算必须让 early stopping 有真实工作空间，并记录 `epochs_ran`、`best_epoch`、`stopped_reason`、train/validation loss 曲线和核心 validation metrics。
- 若 `stopped_reason=max_epochs_reached` 且 `best_epoch` 贴近最后一轮，必须判为训练预算不足或 scout evidence，除非另有充分收敛证据；test metrics 只能作为 confirm，不得用反复查看 test 来调参。
- `promotion_grade` 必须在 `evidence_grade` 之外再满足正式 gate、跨期/月度稳定、成本、drawdown、replay/allocator 或执行约束，并显式确认 `daily_research/output/active_execution_strategy.json` 边界。
- 复盘近期或历史实验时，如果发现 `epochs=2`、单 seed、best epoch 贴边、缺 learning curve 或缺多 seed 聚合，必须降级为 `smoke_only` / `scout_only`，即使目录里 `status=completed`。
- 写入 reference、state 或回答用户时，必须把“运行完成状态”和“证据可信等级”分开写；completed run 不自动等于 completed model-quality evidence。

## TDX-Free Data Platform 运行口径
- `lake` 是研究存储真源，不是在线数据源；`csv` 是导入/补洞通道；正式研究入口只使用 `--data-source lake --lake-dataset-id <explicit_id>`，不得传 `tq/tdx/pytdx/mootdx`。
- V2 每日更新入口：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.data_platform.refresh_daily --as-of-date YYYY-MM-DD --provider-plan default_free --universe all_a --domains market_daily,trading_calendar,universe_snapshot,security_status,limit_status,industry_concept,valuation --json`
- 小样本调试可传 `--symbols 000001.SZ,600000.SH`；正式每日更新默认 `--universe all_a`；`--universe` 支持 `all_a|liquid500|file:<path>|symbols:<csv>`，数据不足时应 blocked。
- `--domains` 表示本次尝试刷新的数据域；`--required-domains` 表示阻断条件。缺失 optional domain 不阻断 market daily 入湖，缺失 required domain 必须 blocked。
- V2 domain 包括 `market_daily`、`trading_calendar`、`universe_snapshot`、`security_status`、`limit_status`、`industry_concept`、`valuation`、`money_flow_hotspot`。
- refresh 输出必须包含 `refresh_run_id`、provider chain、domains、universe key、calendar source、Bronze provider parquet、Silver canonical parquet、`source_conflict_report`、coverage report、provider error report、blockers、registered dataset ids 和 manifest。
- coverage 不足、缺 benchmark、空 canonical market、required domain 缺失或严重 source conflict 时，refresh 必须 `blocked`，不得注册 research lake dataset。
- CSV 入湖入口：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.data_platform.import_csv --input <csv_or_folder> --domain market_daily --as-of-date YYYY-MM-DD --source-name manual_csv --json`
- CSV 必须先进 Bronze，再经 Silver normalize / quality gate 后注册 lake dataset；训练、评估、diagnostics 不得直接读散落 CSV。
- 训练、评估、diagnostics 只读已注册 lake dataset；禁止在研究流程中临时在线抓取外部行情。
- 本轮不实现实盘自动交易；后续 paper/live 需要单独设计 QMT/PTrade broker adapter、风控、合规报备和 kill switch。

## Daily Execution 运行口径
- 冻结期口径：Daily Execution 只保留骨架、只读诊断、手动流程说明和候选评估入口；不得作为生产交易计划、paper/live 或 broker 接线入口。
- 每日任务只走手动步骤；权威状态来自 Web 帮助页当前流程、作业证据路径和只读 daily verdict，不是 Web 是否能打开、单个 job 是否 succeeded、旧 runtime state 或 latest trade plan。
- 启动 Web：
  `conda run -n yolos python daily_research/execution/run_execution_web.py --port 8765`
- 统一应用入口：
  `conda run -n yolos python daily_research/execution/run_execution_app.py web --port 8765`
- 帮助页手动顺序：刷新数据/信号 -> 生成交易计划 -> 模拟账户过账 -> 复核状态。
- 所有任务都必须由按钮或明确 CLI 单项命令显式触发；调度、轮询触发和一键每日流水线不属于当前产品面。
- 数据 readiness 是硬门禁：候选日 `market_daily` 必须非空且覆盖率达标后才允许刷新、信号刷新和交易计划；不得用上一完整交易日伪装今日 completed。
- Web API 热路径只读 compact daily state：`/api/daily-run/status`、`/api/daily-run/latest`、`/api/data-readiness`、`/api/system/doctor`。
- 任务提交入口只保留单项显式动作：`/api/data-sources/refresh`、`/api/trade-plan/generate`、`/api/paper-account/apply-latest-plan` 和必要诊断/恢复入口。
- 当前 2026-05-26 fresh verdict 为 `blocked:data_not_ready`；后续操作以手动帮助页流程为准，完整旧状态机记录见 `daily_research/brain/references/execution_daily_plan_state_machine_refactor_20260526.md`。
- 冻结解除前置：解释 new lineage 研究结果与旧 Stage 2.8 / Stage 3G / short_v5b 证据差异；完成同口径候选 backtest 方案；明确旧 production payload 是恢复、归档为不可用，还是由新研究主线显式替代。`rebuild_lineage_diff_audit` 已完成第一项中的谱系差异解释，但尚未完成同口径 short_v5b backtest 或旧 production payload 路径决策。

## PathPolicy 执行异常处理口径
- `test_forecast_dataset.py` 是慢集成测试；修改 forecast 默认先跑 selective verification 推荐的快速合同测试，完整 forecast dataset/training 测试只在长验证、发布前或风险升高时跑。
- pytest timeout 后先查残留 pytest 子进程和单项复现，区分时间不足、资源挤占、真实死锁、fixture 慢和代码失败；只允许按本轮 pytest 命令行匹配后清理，不做泛化 Python 终止。

## 必跑守卫
- `git diff -- daily_research/output/active_execution_strategy.json`
- `git diff --check`
- brain guards：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.doc_guard check`；`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json`
- 修改 brain platform / workflow / registry / rules 后，跑：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest tools/brain/tests -q`

## 写回路由
- 当前状态和允许动作：`daily_research/brain/state_center.md`。
- 稳定规则和长期教训：`daily_research/brain/knowledge_center.md`。
- 操作命令和流程纪律：`daily_research/brain/operations_center.md`。
- 设计边界和 active research contract：`daily_research/brain/continuous_policy_design_contract.md`。
- 完整 rXX 证据、长命令、复盘：`daily_research/brain/references/`。
- 机器索引：`daily_research/brain/references/evidence_registry.json`。

## 历史归档入口
- 本文件归档前完整快照：`daily_research/brain/references/operations_center_archive_20260510.md`。
- 早期操作原文：`daily_research/brain/references/operations_center_history_raw_20260424.md`。
- 早期操作索引：`daily_research/brain/references/operations_center_evidence_index_20260424.md`。
