# Daily Research 脑区操作协议

快照日期：`2026-05-23`

## 目的
本协议定义主脑路由到 `daily_research` 后的项目操作方式。

`daily_research/brain/` 保存项目事实、证据、状态和治理规则。主脑平台负责接管、路由、全局规则和守卫入口；本分脑只负责 `daily_research` 的项目事实层。

本文件是 `daily_research` 可选补充协议，不属于主脑共享 7 模块核；核心结构、读取顺序和 attach 契约以 manifest 为准。

## 进入顺序
- 重大 `daily_research` 任务开始前，先运行主脑 task capsule：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<task>" --json`
- 只有当 `routing.selected_brain_id == "daily_research"` 且 `routing.status == "selected"` 时，才进入本分脑。
- 在修改 tracked files、启动训练、运行 study 或写结论前，先读 capsule 的 `child_context`、`guards` 和 `workflow_guide`。
- 若任务涉及 study、protocol、dataset 或 r-number，必须使用 explicit tag 或 reference，不直接相信 loose `latest_*`。

## API / 无插件 fallback
- API-only 或无插件场景，使用：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<task>" --workflow auto --json`
- 把 `workflow_selection`、`workflow_guide`、`required_checklist` 和 `stop_conditions` 作为本轮任务的操作指南。
- 用 `workflow-guide --workflow <workflow_id> --json` 只读查看 workflow。
- 该 fallback 只近似插件纪律；它不替代脑区真源，也不授予 active execution 权限。

## 行动纪律
- 重大动作前，区分事实、推断、假设和边界。
- 工作中优先使用主脑平台命令，避免临时解释。
- 重大动作后运行对应 guard，并判断是否需要 brain writeback。
- 若不需要写回，说明原因。

## 主线切换
- 每条研究主线都是可切换的当前工作指针，不是永久层级。
- 用户或治理规则显式切换主线后，后续工作按新主线继续。
- 旧主线证据保留为历史或对照，除非之后再次被选中。
- research mainline 切换不等于 live/default promotion，也不允许修改 `daily_research/output/active_execution_strategy.json`。

## 证据规则
- failed、interrupted、timeout、smoke、dry-run、diagnostic-only run 都不是 completed evidence。
- timeout 首先表示外层等待或观察窗口耗尽；若 PID 仍存活、日志或产物仍在推进，且没有明确代码错误、资源危险或用户停止指令，必须继续轮询，不得把它写成 failed evidence。
- forward outcome 未观测的 realtime tail label 不能写成 completed training evidence。
- Full Gold 训练集声明必须引用已注册的 Gold data-lake dataset，并给出 row count、date range 和 label completeness。
- active execution 变更需要未来明确 promotion authority；普通 research 必须保持 `daily_research/output/active_execution_strategy.json` 不变。

## Brain 与 Skill 分工
- 主脑平台保存接管、路由、全局规则、守卫入口和 workspace-level workflows。
- 本分脑保存 `daily_research` 项目真相：当前状态、规则、证据、设计合同和历史 verdict。
- `brain/skills/workspace-brain` 是 agent 入口 skill；不再发布 `daily-research-brain` skill。
- 文档语言遵循 `brain/language_policy.md`：中文语义 + 英文工程标识。

## 写回规则
- 主中枢保持 compact/current。
- 长 dated evidence、命令 transcript 和详细 r-number status 写入 `daily_research/brain/references/`。
- 机器 evidence index 写入 `daily_research/brain/references/evidence_registry.json`。
- `daily_research/brain/workflows/` 只保留 daily_research 项目 workflow；workspace governance workflow 位于 `brain/workflows/`。
- full codebase review 后，更新 `daily_research/brain/references/full_codebase_review_20260515.md` 或明确 successor。
- mainline review、reroute、deprecation 或 promotion analysis 后，更新 `daily_research/brain/references/mainline_review_current.md`。
- 新增或 materially change reference review docs 后，重建 evidence registry，并运行 doc/brain guards 后再声明写回完成。
