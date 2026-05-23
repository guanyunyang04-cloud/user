# Daily Research 操作中枢

快照日期：`2026-05-23`

## 默认操作纪律
- 默认工作分支：`main`。
- Python 入口：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe`。
- 进入本分脑前，必须先由主脑 `tools.brain.workflow capsule` 路由到 `daily_research`。
- 不使用 `KMP_DUPLICATE_LIB_OK` 作为默认方案。
- 不触碰 `daily_research/output/active_execution_strategy.json`，除非有明确 promotion 决策。
- 不把 smoke、dry-run、failed trial、interrupted outer study、realtime tail label 写成 completed evidence。
- 长训练或 study 需要 progress JSONL、latest progress JSON、stdout/stderr log 和明确 tag。

## 项目地图
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
- 长训练或 study 的默认轮询实现为 `Start-Process -PassThru` 记录 PID，并用 `Wait-Process -Id <pid> -Timeout 7200` 等待；进程提前结束时立即返回，随后解析 progress、日志、summary、checkpoint 与评估产物。

## Multi Horizon Utility 运行口径
- 当前 path_policy 研究主线名：`alpha_multi_horizon_utility_policy_v1`，中文名为“多 Horizon 交易效用排序主线”。
- 旧 `Path20` / `alpha_path20_neural_policy_v1` / `path20_...` study tag 保留为历史证据和代码 namespace；不得批量改写历史 tag，也不得把旧名解释成当前仍以固定 20 日路径预测为目标。
- 新实验 tag 默认使用 `mh_utility_...` 前缀，并显式写入 pool、feature profile、model、seed、train/validation/test 年份、output/loss、cost/hit/drawdown 参数和 horizon grid。
- 当前下一步只允许 constrained horizon-score / calibration 研究；不跑 liquid800、allocator、replay、live/default、promotion，除非新的 liquid500 seed-7 calibration 结果先通过 gate。
- 代码入口可继续使用现有 `daily_research.path_policy.run_alpha_path20_protocol`，因为这是兼容代码 namespace；报告、reference 和 brain 当前状态必须使用新主线名。

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
