# 主脑状态中枢

快照日期：`2026-05-23`

## 当前接管摘要
- 当前工作区根目录固定为 `H:\quant_project`。
- 旧通达信插件目录 `H:\new_tdx64\PYPlugins\user` 不再承载本项目；它应保持为空或只保留通达信原生用户插件文件。
- 工作区正式生产研究与执行主线仍是 `daily_research`。
- 已接入主脑的一级分脑固定为：`daily_research`、`quant_data_platform`、`t0_project`、`daily_stock_analysis-main`、`traditional_quant_research`。
- 默认接管方式为：agent 先理解当前目标，再按需要读取主脑、分脑、代码或产物；不再把 route / capsule 当作必经流程。
- 主脑只维护少数共享事实和硬边界；项目事实、实验指标、命令细节写入对应分脑。

## 当前分支纠偏规则
- 期望分支：所有后续代码、文档与实验工作默认在 `main` 分支展开。
- 当前执行前置：如果 `git branch --show-current` 不是 `main`，任何会修改 repo-tracked 文件的任务都必须先纠偏到 `main`，或由用户显式撤销 `main-branch-only` 规则。
- 分支异常属于 preflight blocker，不属于研究证据、promotion 证据或分脑状态结论。

## 当前分脑状态
- `daily_research`：生产研究与执行主线；active 真源为 `daily_research/output/active_execution_strategy.json`；项目事实、rXX 证据、Path20 历史线 / multi_horizon_utility 当前主线 / continuous_policy / deep_alpha 当前结论以 `daily_research/brain/` 为准，主脑不展开 trial 指标、长 tag 或局部命令。
- `quant_data_platform`：共享量化数据平台与 canonical 数据基底分脑；负责 registry、coverage audit、policy bundle、memmap 治理和跨项目可复用数据契约。
- `t0_project`：盘中实验与 RL 原型分脑；不得替代 `daily_research` 正式执行默认。
- `daily_stock_analysis-main`：独立产品分脑；不改写 `daily_research` active artifact 或 promotion gate。
- `traditional_quant_research`：传统量化方法研究分脑；项目事实、研究记录、实验证据与局部命令以 `traditional_quant_research/brain/` 和对应项目产物为准。

## 当前重点
- 保持 `daily_research` 的正式生产主线地位，同时冻结 live 默认执行的静默切换。
- 维护主脑作为共享脑核，不让主脑重新长成分脑实验日志。
- 分脑入口必须精炼；长过程、长命令和历史证据进入 `episodic_memory.md` 或 `brain/references/`。
- 当前所有 `daily_research` 任务必须显式使用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`。
- 根目录项目身份以主脑 manifest 为准：已注册分脑是项目；`brain/`、`tools/`、`docs/` 是主脑基础设施；`canonical_data/` 是过渡数据资产入口；`a_stock_daily_selection/` 当前是待整理旧目录。
- 当前个人工作约定：canonical 数据集、`canonical_data_v1`、policy bundle、registry、sharded memmap 和数据清理默认看 `quant_data_platform`；`daily_research` 只消费数据并维护研究、模型、回测、执行候选和 active artifact 边界。
- 后续 agent 不得从 `H:\new_tdx64\PYPlugins\user` 接管本项目；旧路径只可能出现在历史 reference 或回滚说明中。
- 所有任务默认由 agent 自主判断范围：可读取相关项目事实，不混用无关 dirty/output/process；写入、清理、进程管理和提交按真实风险收敛到相关路径。
- 任何需要轮询、等待外部状态或跨多轮观察的任务都要有可观察 handle（PID / job id / run id、日志、progress、artifact、端口 / API status 等）解释进展；轮询间隔由 agent 根据任务信号自适应调整，不把固定 sleep、固定窗口或历史固定模板当作通用规则。
- 等待窗口耗尽不是失败证据；只有明确错误、资源危险、失败产物或用户停止才中断或写失败，仍有进展则继续自适应轮询。

## 当前边界
- 主脑不得记录具体 trial 指标、训练 tag 长列表或局部实验命令；这些属于分脑。
- 分脑不得改写主脑少数共享事实、拓扑和硬边界；这些属于主脑。
- `daily_research` 的研究证据、rXX references、Path20 历史线 / multi_horizon_utility 当前主线 / continuous_policy / deep_alpha 状态和验证矩阵只读 `daily_research/brain/` 与对应实验产物；主脑只保留 promotion / live / active artifact 边界。
- 任何疑似中文乱码，先用 UTF-8 工具复核真实文件内容，不把终端编码显示问题当作文件损坏。

## 当前风险
- 如果主脑继续追加日期日志，接管会重新退化为长文扫描。
- 如果只改分脑、不改主脑，跨项目规则会再次漂移。
- 如果兼容入口、README 或教程保留 brain 未收录的规则，后续 agent 会绕过中枢。
- 如果普通短命令、临时产物、测试或报告散落到 workspace 根和无关项目，接管仍会变慢；agent 应主动保持路径整洁。
- 如果需要轮询的任务没有可观察 handle、日志、progress、summary、artifact 或状态信号可追溯，就不能写成正式证据；如果只有观察窗口耗尽而没有明确失败信号，也不能写成失败证据。
- 如果 agent 或脚本默认落到旧通达信插件 `user` 目录，先纠偏到 `H:\quant_project`，再继续操作。

## 推荐下一步
- 结构变更先跑 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json`；普通 brain 文档小改用 `tools.brain.doc_guard check --files <paths>` 或 `--scope changed`，全量维护再跑裸 `doc_guard check`；`daily_research/tools/project_consistency_check.py --mode research` 是研究态轻量守卫，`--mode execution/full` 只在执行或完整维护时跑。
- 新状态只写当前结论；过程复盘写到目标分脑 `episodic_memory.md`。
- 需要旧证据时从分脑 `brain/references/` 或实验产物读取，不把旧结论自动提升为当前状态。
