# Daily Research 操作中枢

快照日期：`2026-06-16`

## 默认操作纪律
- 默认工作分支：`main`；当前工作区根目录：`H:\quant_project`。
- Python 入口固定为 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`；接管本分脑时优先按用户目标、显式文件、state/reference 和当前输出证据行动，主脑 `capsule` / `current-frontier` 是可选路由与新鲜度诊断，不是硬前置。
- 继承主脑个人研究者直接行动风格：直接实现、直接重构、直接清理旧路径；验证只需足够支撑当前结论。
- `H:\new_tdx64\PYPlugins\user` 已退出本项目主链路；历史 reference 中旧路径只作历史证据，不作为当前入口。
- 不使用 `KMP_DUPLICATE_LIB_OK` 作为默认方案。
- 继承主脑项目任务命名空间：普通读写、短脚本、测试、临时产物、提交和轮询 / 异步任务默认限制在 `daily_research` profile；其它项目 dirty/output/process 只作摘要报告，不下钻、不复用、不写成本任务证据，除非用户扩展范围或声明 lease。
- 2026-06-01 起执行端冻结/保留骨架/等待重建：只允许只读状态、数据 readiness、候选 backtest wrapper、候选 trade-plan wrapper 的研究评估用途和 payload inventory；禁止 live/default、paper/broker、正式 trade plan、active manifest promotion、production root 重建、自动化每日执行和无授权删除执行合同。
- 2026-05-31 恢复边界：`daily_research/output/` 与 `daily_research/cache/` 曾被误删；空目录骨架只保证 manifest/integrity 入口存在，不代表旧 payload 恢复，不得凭 brain 文本手工重造 active artifact。

## 项目地图
- brain 真源：`daily_research/brain/`；research data lake：`daily_research/output/research_data_lake/`。
- studies：`daily_research/output/path_policy/studies/`、`daily_research/output/continuous_policy/studies/`；protocols：`daily_research/output/continuous_policy/protocols/`。
- active artifact：`daily_research/output/active_execution_strategy.json`；execution app：`daily_research/execution/run_execution_app.py`；daily verdict root：`daily_research/output/execution_app/daily_runs/`。
- 以上 output/cache 路径在真实 payload 恢复前只代表目标位置；文件级 evidence lookup、replay、active artifact inspection 和 project consistency 结论需回到 brain references 与用户确认边界。

## 高频 Brain 命令
- 可选主脑 task capsule：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<task>" --json`
- current frontier freshness：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow current-frontier --json`
- evidence registry rebuild：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow evidence-index --rebuild --json`
- evidence query：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow query --q <r_id|tag|dataset_id> --json`
- selective verification plan：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow verify-plan --json`
- brain guards：日常 changed-surface 用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.doc_guard check --scope changed`；全量维护才用裸 `doc_guard check`；结构定位用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json`
- 若 `brain_may_be_stale=true`，先读 explicit output tags 与 summary，再回答当前状态或下一步。

## Continuous Policy 运行口径
- Direct protocol smoke 先于 dry-run study，dry-run study 先于 safe screening。
- 默认 protocol runner 为 `in_process`；不恢复 r57 父/子进程 watchdog。
- Study runner 是逻辑编排层：选 trial、写 plan、调用 protocol、汇总 `study_summary.json`。
- Protocol 是单次训练/评估/shadow/export 层，写 `protocol_summary.json`。
- 若 `protocol_summary.json` 存在但 `study_summary.json` 缺失，只能记为 protocol-level evidence。
- 后台运行只作为 OS 级 launcher / 轮询能力；study 内部仍保持 `protocol_runner_mode=in_process`，agent 根据 progress / PID / logs / summaries / artifacts 自适应观察。
- 训练或 study 需要跨回合观察、后台驻留或保留执行证据时，用 `tools.brain.agent_run paths/register/launch/status --project-id daily_research --run-id <run>` 绑定 PID、stdout/stderr、progress、summary 和明确 tag；不强制所有等待任务套同一 launcher 或固定 monitor。
- 单轮观察窗口耗尽后，若 PID、日志、progress、summary / checkpoint / artifact 仍推进且无明确代码错误、资源危险或用户停止指令，继续按信号密度自适应轮询，不得停止任务或写成 failed evidence。

## Multi Horizon Utility 运行口径
- 当前 path_policy 研究主线名：`alpha_multi_horizon_utility_policy_v1`；当前 v2 reset 主线为 `daily_research_v2_research_reset`。
- 旧 `Path20` / `alpha_path20_neural_policy_v1` / `path20_...` 字符串保留为历史 run 证据和代码 namespace；不得批量改写历史 tag，也不得把旧名解释成当前仍以固定 20 日路径预测为目标。
- 新实验必须显式写入 research program、study family 和 run tag；run tag 需包含 pool、feature、model、seed、年份、output/loss、成本参数和 horizon grid。
- 标准包入口仍是 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.run_alpha_path20_protocol`；直接脚本只作为容错 smoke。
- V2 数据集、score bridge、candidate review、bad-month attribution、risk overlay、selective throttle、state sizing、local-state input/loss 与 horizon repair 的长命令族已下沉到 `daily_research/brain/references/daily_research_current_frontier_compaction_20260603.md` 及对应 dated references。
- 当前下一步只允许 research-only score-backtest bridge 与候选对照；不跑 liquid800、allocator、replay、live/default、promotion，除非后续 gate 与授权明确放行。
- 新训练默认消费 QDP canonical full sharded memmap / registry；旧每实验单体 `forecast_*.dat` builder、旧 copied manifest 兼容和旧 path_policy memmap 回归只作 legacy 参考，不进入常规研究门禁。
- 回答“当前数据集构建怎么样”时，直接读取 QDP status / registry / sharded manifest，再说明 daily_research 对该数据基底的消费与研究证据；不得把 daily_research 旧 lake、旧 single memmap、旧 copied manifest 当作 canonical 数据集 owner。

## 实验预算可信度纪律
- 启动任何会影响模型输入、架构、输出、loss、horizon grid、stage gate 或后续方向选择的实验前，必须声明证据等级：`smoke_only`、`scout_only`、`evidence_grade` 或 `promotion_grade`。
- `smoke_only` 只验证入口、shape、loss、数据和 artifact；`scout_only` 只用于粗筛下一步候选。
- `evidence_grade` 才能支撑模型质量比较或阶段决策；默认至少 3 seeds，关键候选优先 5 seeds，并记录 `epochs_ran`、`best_epoch`、`stopped_reason`、loss 曲线和核心 validation metrics。
- `promotion_grade` 必须在 `evidence_grade` 外再满足正式 gate、跨期/月度稳定、成本、drawdown、replay/allocator 或执行约束，并显式确认 active artifact 边界。
- 写入 reference、state 或回答用户时，必须把“运行完成状态”和“证据可信等级”分开写；completed run 不自动等于 completed model-quality evidence。

## TDX-Free Data Platform 运行口径
- `lake` 是研究存储真源，不是在线数据源；`csv` 是导入/补洞通道。
- 正式研究入口只使用 `--data-source lake --lake-dataset-id <explicit_id>`，不得传 `tq/tdx/pytdx/mootdx`。
- V2 refresh / import 命令和 domain 明细见 `daily_research/brain/references/tdx_free_data_platform_v2_20260523.md` 与 `daily_research/brain/references/daily_research_v2_dataset_contract_upgrade_20260602.md`。
- coverage 不足、缺 benchmark、空 canonical market、required domain 缺失或严重 source conflict 时，refresh 必须 `blocked`，不得注册 research lake dataset。
- 训练、评估、diagnostics 只读已注册 lake dataset；禁止在研究流程中临时在线抓取外部行情。

## Daily Execution 运行口径
- 冻结期 Daily Execution 只保留骨架、只读诊断、手动流程说明和候选评估入口；不得作为生产交易计划、paper/live 或 broker 接线入口。
- 每日任务只走手动步骤；权威状态来自 Web 帮助页当前流程、作业证据路径和只读 daily verdict。
- 统一应用入口：`conda run -n yolos python daily_research/execution/run_execution_app.py web --port 8765`
- 帮助页手动顺序：刷新数据/信号 -> 生成交易计划 -> 模拟账户过账 -> 复核状态。
- 数据 readiness 是硬门禁：候选日 `market_daily` 必须非空且覆盖率达标后才允许刷新、信号刷新和交易计划；不得用上一完整交易日伪装今日 completed。
- 当前 2026-05-26 fresh verdict 为 `blocked:data_not_ready`；完整旧状态机记录见 `daily_research/brain/references/execution_daily_plan_state_machine_refactor_20260526.md`。

## PathPolicy 执行异常处理口径
- `test_forecast_dataset.py` 是慢集成测试；修改 forecast 默认先跑 selective verification 推荐的快速合同测试。
- pytest timeout 后先查残留 pytest 子进程和单项复现，区分时间不足、资源挤占、真实死锁、fixture 慢和代码失败；只允许按本轮 pytest 命令行匹配后清理。

## 验证分层
- 默认开发验证走 changed-surface：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.selective_verification --paths <changed_paths> --json`，本次 blocking 只取 `blocking_commands`。
- 轻量 smoke lane：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research -m "smoke and not slow and not research and not data_heavy and not external and not benchmark" -q`。该 lane 只收集少数低成本合同测试，适合 brain / 配置 / 测试治理闭环。
- 项目普通车道只跑非慢速测试：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research -m "not slow and not research and not data_heavy and not external and not benchmark" -q`。`forecast_training`、`forecast_memmap_dataset`、`rl_protocol` 等训练 / memmap / 研究协议测试属于 `research` 或 `data_heavy` 车道，按 explicit nodeid 或阶段收口运行。
- `continuous_policy`、`data_lake`、`data_platform`、`execution` 的测试由根级 `daily_research/conftest.py` 自动归入 `research`、`data_heavy`、`external`、`guard`、`integration` 等 lane；新增重测试先放显式 lane，不进入默认开发闭环。
- 旧 `path_policy/tests/test_forecast_memmap_dataset.py` 属于 legacy 单体 memmap 保护网；QDP sharded memmap 成为默认后，不作为常规 blocking 测试。
- 高频硬边界：`git diff -- daily_research/output/active_execution_strategy.json` 和 `git diff --check`；active artifact 有 diff 时停止并回到 promotion authority。
- `current-frontier` 只在回答当前研究阶段、更新 frontier 判断或写入 evidence/state 前运行。
- `doc_guard` / `integrity_check` / `brain-burden-audit` 属于 brain 文档、workflow、registry、skill 或收尾维护守卫，不作为普通代码小改默认测试包。
- 修改 brain platform / workflow / registry / rules 后，优先跑 selective verification 推荐的工具测试；整包 `tools/brain/tests` 只作为维护/收尾扩展。

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
