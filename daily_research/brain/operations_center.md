# Daily Research 操作中枢

快照日期：`2026-05-23`

## 默认操作纪律
- 默认工作分支：`main`。
- 当前工作区根目录：`H:\quant_project`。
- Python 入口：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe`。
- 进入本分脑前，必须先由主脑 `tools.brain.workflow capsule` 路由到 `daily_research`。
- `H:\new_tdx64\PYPlugins\user` 已退出本项目主链路；历史 reference 中旧路径只作历史证据，不作为当前入口。
- 不使用 `KMP_DUPLICATE_LIB_OK` 作为默认方案。
- 不触碰 `daily_research/output/active_execution_strategy.json`，除非有明确 promotion 决策。
- 不把 smoke、dry-run、failed trial、interrupted outer study、realtime tail label 写成 completed evidence。
- 长训练或 study 需要 progress JSONL、latest progress JSON、stdout/stderr log 和明确 tag。

## 项目地图
- workspace root：`H:\quant_project`。
- brain 真源：`daily_research/brain/`。
- research data lake：`daily_research/output/research_data_lake/`。
- path_policy studies：`daily_research/output/path_policy/studies/`。
- continuous_policy studies：`daily_research/output/continuous_policy/studies/`。
- continuous_policy protocols：`daily_research/output/continuous_policy/protocols/`。
- production active artifact：`daily_research/output/active_execution_strategy.json`。
- execution app：`daily_research/execution/run_execution_app.py`。

## 高频 Brain 命令
- 主脑 task capsule：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<task>" --json`
- API/no-plugin auto workflow capsule：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<task>" --workflow auto --json`
- current frontier freshness：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow current-frontier --json`
- workflow guide：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow workflow-guide --workflow <workflow_id> --json`
- explicit evidence status：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow status --workflow continuous_policy --study-tag <study_tag> --json`
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
- 旧 `Path20` / `alpha_path20_neural_policy_v1` / `path20_...` study tag 保留为历史证据和代码 namespace；不得批量改写历史 tag，也不得把旧名解释成当前仍以固定 20 日路径预测为目标。
- 新实验 tag 默认使用 `mh_utility_...` 前缀，并显式写入 pool、feature profile、model、seed、train/validation/test 年份、output/loss、cost/hit/drawdown 参数和 horizon grid。
- 当前下一步只允许 constrained horizon-score / calibration 研究；不跑 liquid800、allocator、replay、live/default、promotion，除非新的 liquid500 seed-7 calibration 结果先通过 gate。
- 标准代码入口继续使用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.run_alpha_path20_protocol`，因为这是包级入口；直接脚本 `daily_research/path_policy/run_alpha_path20_protocol.py` 只作为容错 smoke，不能替代文档推荐入口。

## TDX-Free Data Platform 运行口径
- `lake` 是研究存储真源，不是在线数据源；`csv` 是导入/补洞通道，不是每日自动更新方案。
- 正式研究入口只使用 `--data-source lake --lake-dataset-id <explicit_id>`；不得传 `tq/tdx/pytdx/mootdx`。
- V2 每日更新入口：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.data_platform.refresh_daily --as-of-date YYYY-MM-DD --provider-plan default_free --universe all_a --domains market_daily,trading_calendar,universe_snapshot,security_status,limit_status,industry_concept,valuation --json`
- 小样本调试可传 `--symbols 000001.SZ,600000.SH`；正式每日更新默认使用 `--universe all_a`，不再要求显式 symbols。
- `--universe` 支持 `all_a|liquid500|file:<path>|symbols:<csv>`；`liquid500` 必须从已有 lake/pool view 或 market amount 生成，数据不足时应 blocked。
- `--domains` 表示本次尝试刷新的数据域；`--required-domains` 表示阻断条件。缺失 optional domain 不阻断 market daily 入湖，缺失 required domain 必须 blocked。
- V2 domain 包括 `market_daily`、`trading_calendar`、`universe_snapshot`、`security_status`、`limit_status`、`industry_concept`、`valuation`、`money_flow_hotspot`。
- refresh 输出必须包含 `refresh_run_id`、provider chain、domains、universe key、calendar source、Bronze provider parquet、Silver canonical parquet、`source_conflict_report`、coverage report、provider error report、blockers、registered dataset ids 和 manifest。
- coverage 不足、缺 benchmark、空 canonical market、required domain 缺失或严重 source conflict 时，refresh 必须 `blocked`，不得注册 research lake dataset。
- CSV 入湖入口：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.data_platform.import_csv --input <csv_or_folder> --domain market_daily --as-of-date YYYY-MM-DD --source-name manual_csv --json`
- CSV 必须先进 Bronze，再经 Silver normalize / quality gate 后注册 lake dataset；训练、评估、diagnostics 不得直接读散落 CSV。
- 训练、评估、diagnostics 只读已注册 lake dataset；禁止在研究流程中临时在线抓取外部行情。
- 本轮不实现实盘自动交易；后续 paper/live 需要单独设计 QMT/PTrade broker adapter、风控、合规报备和 kill switch。

## PathPolicy 执行异常处理口径
- `test_forecast_dataset.py` 是慢集成测试，不是默认轻量合同测试；它的 synthetic fixture 会用 700/820/900 个交易日并走完整 feature、label、cumulative horizon 和 horizon risk 构造，单项可到分钟级。
- 修改 forecast 相关代码时，默认先跑 selective verification 推荐的快速合同测试；完整 `test_forecast_dataset.py` 与 `test_forecast_training.py` 放入 `deferred_long_commands`，需要长验证、发布前检查或风险升高时再跑。
- pytest timeout 后不得直接下失败结论；先查是否有本轮残留 pytest 进程，再单项复现慢测试，区分时间没给足、资源挤占、真实死锁、fixture 慢和代码失败。
- 手动清理残留进程只允许匹配本轮 pytest 命令行，只清理 pytest 子进程，不碰其他 Python 任务；示例：
  `Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*pytest*daily_research/path_policy/tests/test_forecast_dataset.py*' }`
- 如果确认要终止本轮残留 pytest，再对上述匹配结果执行 `Stop-Process -Id <ProcessId> -Force`；不要用泛化的 `Stop-Process python`。

## 必跑守卫
- `git diff -- daily_research/output/active_execution_strategy.json`
- `git diff --check`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.doc_guard check`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json`
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
