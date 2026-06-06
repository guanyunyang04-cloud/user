# 主脑状态中枢

快照日期：`2026-05-23`

## 当前接管摘要
- 当前工作区根目录固定为 `H:\quant_project`。
- 旧通达信插件目录 `H:\new_tdx64\PYPlugins\user` 不再承载本项目；它应保持为空或只保留通达信原生用户插件文件。
- 工作区正式生产研究与执行主线仍是 `daily_research`。
- 已接入主脑的一级分脑固定为：`daily_research`、`t0_project`、`daily_stock_analysis-main`、`traditional_quant_research`。
- 默认接管顺序为：主脑 `identity -> state -> knowledge -> topology -> operations -> governance`，再进入目标分脑。
- 主脑只维护跨项目边界、共享规则和路由；项目事实、实验指标、命令细节写入对应分脑。

## 当前分支纠偏规则
- 期望分支：所有后续代码、文档与实验工作默认在 `main` 分支展开。
- 当前执行前置：如果 `git branch --show-current` 不是 `main`，任何会修改 repo-tracked 文件的任务都必须先纠偏到 `main`，或由用户显式撤销 `main-branch-only` 规则。
- 分支异常属于 preflight blocker，不属于研究证据、promotion 证据或分脑状态结论。

## 当前分脑状态
- `daily_research`：生产研究与执行主线；active 真源为 `daily_research/output/active_execution_strategy.json`；项目事实、rXX 证据、Path20 历史线 / multi_horizon_utility 当前主线 / continuous_policy / deep_alpha 当前结论以 `daily_research/brain/` 为准，主脑不展开 trial 指标、长 tag 或局部命令。
- `t0_project`：盘中实验与 RL 原型分脑；不得替代 `daily_research` 正式执行默认。
- `daily_stock_analysis-main`：独立产品分脑；不改写 `daily_research` active artifact 或 promotion gate。
- `traditional_quant_research`：传统量化方法研究分脑；项目事实、研究记录、实验证据与局部命令以 `traditional_quant_research/brain/` 和对应项目产物为准。

## 当前重点
- 保持 `daily_research` 的正式生产主线地位，同时冻结 live 默认执行的静默切换。
- 维护主脑作为共享脑核，不让主脑重新长成分脑实验日志。
- 分脑入口必须精炼；长过程、长命令和历史证据进入 `episodic_memory.md` 或 `brain/references/`。
- 当前所有 `daily_research` 任务必须显式使用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`。
- 后续 agent 不得从 `H:\new_tdx64\PYPlugins\user` 接管本项目；旧路径只可能出现在历史 reference 或回滚说明中。
- 所有任务默认先绑定项目任务命名空间：普通读写、短脚本、诊断、测试、临时产物、提交和进程管理都只属于当前 `project_profile` 允许范围；其它项目 dirty paths、输出和进程是外部并行工作，可摘要报告但不下钻、不复用、不写成本任务证据，除非用户明确扩展范围或声明 lease。
- 训练、评估、审计、bounded study、confirmatory rerun 与执行任务必须受监管运行：默认用 `tools.brain.agent_run paths/register/launch/status --project-id <project> --run-id <run>` 绑定 PID、持久 stdout/stderr、summary、run tag 和产物路径；GPU 进程确认工作后必须用带 project/run 身份的 `long_task_monitor wait-once` 执行 `Wait-Process -Id <pid> -Timeout 7200` 前台轮询，除非再次出现系统崩溃或宿主不可用。
- 长任务禁止无 PID、无日志、无产物定位的脱管后台化；禁止用固定 sleep 替代 PID 绑定等待；每轮轮询必须报告已用时间和预计剩余时间，进程提前结束时立即解析产物；窗口耗尽但 PID / 日志 / 产物仍推进时继续下一轮轮询，不中断任务。

## 当前边界
- 主脑不得记录具体 trial 指标、训练 tag 长列表或局部实验命令；这些属于分脑。
- 分脑不得改写跨项目读取顺序、主分脑拓扑或统一治理纪律；这些属于主脑。
- `daily_research` 的研究证据、rXX references、Path20 历史线 / multi_horizon_utility 当前主线 / continuous_policy / deep_alpha 状态和验证矩阵只读 `daily_research/brain/` 与对应实验产物；主脑只保留 promotion / live / active artifact 边界。
- 任何疑似中文乱码，先用 UTF-8 工具复核真实文件内容，不把终端编码显示问题当作文件损坏。

## 当前风险
- 如果主脑继续追加日期日志，接管会重新退化为长文扫描。
- 如果只改分脑、不改主脑，跨项目规则会再次漂移。
- 如果兼容入口、README 或教程保留 brain 未收录的规则，后续 agent 会绕过中枢。
- 如果只把长任务放进命名空间，而让普通短命令、临时产物、测试或报告散落到 workspace 根和其它项目，项目并行隔离仍然不成立。
- 如果长任务没有 PID、日志、progress、summary 或环境诊断可追溯，就不能写成正式证据；如果只有前台等待窗口耗尽而没有代码错误证据，也不能写成失败证据。
- 如果 agent 或脚本默认落到旧通达信插件 `user` 目录，先纠偏到 `H:\quant_project`，再继续操作。

## 推荐下一步
- 结构变更先跑 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json`；普通 brain 文档小改用 `tools.brain.doc_guard check --files <paths>` 或 `--scope changed`，全量维护再跑裸 `doc_guard check`；`daily_research/tools/project_consistency_check.py --mode research` 是研究态轻量守卫，`--mode execution/full` 只在执行或完整维护时跑。
- 新状态只写当前结论；过程复盘写到目标分脑 `episodic_memory.md`。
- 需要旧证据时从分脑 `brain/references/` 或实验产物读取，不把旧结论自动提升为当前状态。
