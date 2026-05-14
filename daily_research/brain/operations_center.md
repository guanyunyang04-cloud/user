# Daily Research 操作中枢

快照日期：`2026-05-14`

## 默认操作纪律
- 默认工作分支：`main`。
- Python 入口：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe`。
- 不使用 `KMP_DUPLICATE_LIB_OK` 作为默认方案。
- 不触碰 `daily_research/output/active_execution_strategy.json`，除非有明确 promotion 决策。
- 不把 smoke、dry-run、failed trial、interrupted outer study、realtime tail label 写成 completed evidence。
- 长训练或 study 需要 progress JSONL、latest progress JSON、stdout/stderr log 和明确 tag。
- PowerShell 中文写入不作为默认文档编辑方式；中文正文优先用 `apply_patch` 或显式 UTF-8 工具链。

## 项目地图
- brain 真源：`daily_research/brain/`。
- research data lake：`daily_research/output/research_data_lake/`。
- continuous_policy studies：`daily_research/output/continuous_policy/studies/`。
- continuous_policy protocols：`daily_research/output/continuous_policy/protocols/`。
- production active artifact：`daily_research/output/active_execution_strategy.json`。
- execution app：`daily_research/execution/run_execution_app.py`。

## 高频 Brain 命令
- task capsule：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.tools.brain_workflow capsule --child daily_research --task "<task>" --json`
- explicit evidence status：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.tools.brain_workflow status --workflow continuous_policy --study-tag <study_tag> --json`
- evidence registry rebuild：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.tools.brain_workflow evidence-index --rebuild --json`
- evidence query：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.tools.brain_workflow query --q <r_id|tag|dataset_id> --json`
- brain guards：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/doc_guard.py check`
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/brain_integrity_check.py --json`

## Continuous Policy 运行口径
- Direct protocol smoke 先于 dry-run study，dry-run study 先于 safe screening。
- 默认 protocol runner 为 `in_process`；不恢复 r57 父/子进程 watchdog。
- Study runner 是逻辑编排层：选 trial、写 plan、调用 protocol、汇总 `study_summary.json`。
- Protocol 是单次训练/评估/shadow/export 层，写 `protocol_summary.json`。
- 若 `protocol_summary.json` 存在但 `study_summary.json` 缺失，只能记为 protocol-level evidence。
- 后台运行建议只把整个 study 作为一个 OS 后台进程启动；study 内部仍保持 `protocol_runner_mode=in_process`，前台只轮询 progress / PID / logs / summaries。

## 当前 r65 口径
- Active research profile：`split_heads_portfolio_daily_release_first_portfolio_set_v5_r65`。
- Backend/loss：`formal_torch_portfolio_set_v5` / `alpha_result_value_budget_split_v48`。
- 默认 strict Gold dataset：`continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1`。
- r65 是 `research / shadow-only / architecture upgrade`；不得 confirmatory、promotion、live/default 或 active artifact change。
- r65 下一步应修 held source creation、receiver target realization、target/action translation 与 day-set sampling。

## Data Lake 口径
- Full-window strict Gold 已完成并 audit-clean：`continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1`。
- Full-window realtime Gold 仍 pending；realtime tail labels 只能用于 research/audit。
- 构建或读取数据集必须使用 explicit dataset id，不用 loose latest 判断完成。
- Gold 构建采用 sharded/resumable builder，不再通过训练入口强行构建全量 Gold。

## 必跑守卫
- `git diff -- daily_research/output/active_execution_strategy.json`
- `git diff --check`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/doc_guard.py check`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/brain_integrity_check.py --json`
- 修改 brain workflow / registry / rules 后，跑：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/tools/tests/test_brain_capsule.py daily_research/tools/tests/test_brain_evidence_registry.py daily_research/tools/tests/test_brain_rules.py daily_research/tools/tests/test_brain_workflow_cli.py daily_research/tools/tests/test_daily_research_brain_skill.py -q`

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
