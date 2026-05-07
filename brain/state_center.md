# 主脑状态中枢

快照日期：`2026-05-08`

## 当前接管摘要
- 工作区正式生产研究与执行主线仍是 `daily_research`。
- 已接入主脑的一级分脑固定为：`daily_research`、`t0_project`、`daily_stock_analysis-main`。
- 默认接管顺序为：主脑 `identity -> state -> knowledge -> topology -> operations -> governance`，再进入目标分脑。
- 主脑只维护跨项目边界、共享规则和路由；项目事实、实验指标、命令细节写入对应分脑。

## 当前分脑状态
- `daily_research`：生产研究与执行主线；active 真源为 `daily_research/output/active_execution_strategy.json`；continuous_policy 最新有效证据基线仍为 r39，最新代码入口为 r48 full-universe convex OPE allocation（基于真实 `cvxpy` / `cvxpylayers` solver layer，扩展候选覆盖、现实成本风险与 OPE 诊断），二者均属 `research / shadow_only`。
- `t0_project`：盘中实验与 RL 原型分脑；不得替代 `daily_research` 正式执行默认。
- `daily_stock_analysis-main`：独立产品分脑；不改写 `daily_research` active artifact 或 promotion gate。

## 当前重点
- 保持 `daily_research` 的正式生产主线地位，同时冻结 live 默认执行的静默切换。
- 维护主脑作为共享脑核，不让主脑重新长成分脑实验日志。
- 分脑入口必须精炼；长过程、长命令和历史证据进入 `episodic_memory.md` 或 `brain/references/`。
- 当前所有 `daily_research` 任务必须显式使用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`。
- 训练、评估、审计、bounded study、confirmatory rerun 与执行任务默认前台运行，窗口时限按 `10` 小时处理，不得中途人为中断；长任务必须保留持久 stdout/stderr 日志，并按 `2` 小时轮询。

## 当前边界
- 主脑不得记录具体 trial 指标、训练 tag 长列表或局部实验命令；这些属于分脑。
- 分脑不得改写跨项目读取顺序、主分脑拓扑或统一治理纪律；这些属于主脑。
- `daily_research` 的 continuous_policy 最新有效证据基线仍是 r39 `research / shadow_only`；r40 clean rerun 已证明运行通道可用但 stable confirm 为空；r48 只是最新代码合同与 dry-run 入口，即使已扩展到 full-universe aware solver / OPE 诊断，也仍尚无正式 screening / confirm verdict，不得 promotion、不得 live、不得改 active artifact。
- 任何疑似中文乱码，先用 UTF-8 工具复核真实文件内容，不把终端编码显示问题当作文件损坏。

## 当前风险
- 如果主脑继续追加日期日志，接管会重新退化为长文扫描。
- 如果只改分脑、不改主脑，跨项目规则会再次漂移。
- 如果兼容入口、README 或教程保留 brain 未收录的规则，后续 agent 会绕过中枢。
- 如果训练结果未核验 `device = cuda`、`cuda_available = true`、yolos 解释器与持久日志完整性，就不能写成正式证据。

## 推荐下一步
- 结构变更先跑 `brain_integrity_check.py --json`，再跑 `doc_guard.py check` 与 `project_consistency_check.py`。
- 新状态只写当前结论；过程复盘写到目标分脑 `episodic_memory.md`。
- 需要旧证据时从分脑 `brain/references/` 或实验产物读取，不把旧结论自动提升为当前状态。
