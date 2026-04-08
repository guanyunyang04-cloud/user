# Daily Research Procedural Memory

快照日期：`2026-04-08`

## 1. 文档分工
- `semantic_memory.md`
  - 只写长期稳定事实。
- `project_map.md`
  - 只写项目结构、主线演进和当前地图。
- `working_memory.md`
  - 只写当前判断、瓶颈和下一步。
- `action_system.md`
  - 只写高频入口、操作清单和当前可执行口径。
- `episodic_memory.md`
  - 只写带日期的实验过程和变更记录。

## 2. 总原则
- 默认目标函数是 `execution_first`，不是只看 raw holdout。
- 默认优先“最有效”，不是“最小改动”。
- 一次实验的亮眼结果，不等于正式主线升级结论。
- 所有跨 universe 的“最高”表述，必须先回到统一 deployable leaderboard 口径。
- 用户最新提出的要求，默认拥有最高优先级。
- 新要求一旦改变原有口径，必须先统一代码、脚本入口、manifest、trade plan 展示和 brain 文档，再继续后续工作。
- 不允许保留会误导后续判断的前后矛盾、过期冗余或半退役入口；如确需保留 legacy 包装，必须显式标注 deprecated / intentional legacy。

## 3. 判决顺序
1. formal holdout
- 判研究 winner。

2. recent realistic gate
- 判执行可落地性。

3. production full-fit
- 判默认执行候选是否可物化。

4. active strategy manifest
- 决定日常执行真源。

## 4. 月度优先规则
- 当前默认判读顺序是：
  - `positive_month_ratio`
  - `median_monthly_return`
  - `worst_monthly_return`
  - `top3_positive_month_share`
  - 然后才看 annual / Sharpe
- 北极星“月度正收益 > 30%”目前只作为长期方向，不直接改 formal gate。

## 5. 训练纪律
- 训练一律使用 GPU。
- 如果没有 CUDA，必须停下并报告阻塞，不能静默回落到 CPU。
- 长训练默认耐心等待，非硬错误或用户改优先级，不主动中断。
- 准备移交到 execution / monthly refresh / production candidate 的候选，必须先补成：
  - latest model
  - highest family budget
- 同一模型、同一配方、同一窗口下的扩预算，一律 strict resume continuation。
- 只有 resume chain 校验失败，才允许 fresh rerun，并明确记成 fallback。
- 默认训练预算从 `32` epoch 起步；如果 family manifest 给出更高起步预算，则直接服从更高预算。
- 如果 `32` 或当前 family 起步预算还不够，后续只允许沿同模型 strict resume continuation 扩预算，不回到 fresh rerun。
- 预训练与正式训练使用同一条默认纪律：
  - 起训 `32`
  - `min_epochs` 随默认预算同步抬升
  - GPU only

## 6. execution-side 研究规则
- broad execution-policy sweep 已冻结。
- 当前 execution-side 只沿这条链继续：
  - `month-start / first-week / multi-day score -> weight -> execution`
- external target-weight 路径的当前唯一语义是：
  - `research_raw_target_weight`
  - `follow_research_raw_no_global_cap`
  - 不允许再把通用 `max_weight=0.25` 误套到这条链上
- 如果某条修复在 formal 为正、recent delta 为正，但最近 live 月份未触发：
  - 记为 monitored repair candidate
  - 默认不直接升 active default
- 例外：
  - 如果它已经被物化成 scripted candidate pipeline
  - 且 live 未触发时会自动退回当前静态默认
  - 那么可以升成 active manifest，而不会改变当下未触发月份的实际执行结果
- `single-mapping candidate pipeline` 分成两层：
  - heavy research refresh：重算 formal replay / H2H / trigger tradeoff
  - live-only refresh：只更新 live monitor、`daily_live_target_weight_panel.csv`、`daily_live_score_panel.csv`
- 日常 trade plan 默认只走 `live-only refresh`，不再为每日执行重跑整条 formal candidate pipeline。

## 7. model-side 研究规则
- 大 bundle 不自动更优；先看 budget-stable 结果，再决定是否继续。
- 如果 bundle 补齐预算后仍落后：
  - 记为负证据
  - 后续默认拆成 ablation，而不是继续堆同类改动
- 当前模型侧如重开，默认从 `penalty-only` 窄修复继续，不回到 `v2 / heavy`

## 8. 产物与编码规则
- brain 文档默认中文。
- 终端日志、进度条、运行输出默认英文。
- PowerShell 读中文 markdown 如出现乱码，优先用 Python UTF-8 读取，不把它误判为文件损坏。

## 9. 一致性自检
- 只要改动了训练默认值、execution manifest 语义、candidate pipeline 入口或脑文档口径，收尾必须跑：
  - `python daily_research/tools/project_consistency_check.py`
  - `python daily_research/tools/doc_guard.py check`
- 只要用户提出了新的高优先级规则，也必须在收尾前确认：
  - 当前 active manifest 与 trade plan 已反映该规则
  - 相关 brain 文档已同步
  - 不存在仍按旧规则运行的默认入口
