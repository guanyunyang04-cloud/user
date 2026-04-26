# 主脑知识中枢

## 1. 固定规则
- `brain-first`
  - 先接主脑，再接分脑，再进 body
- `common-in-main`
  - 共享结构、共享顺序、共享治理只在主脑定义一次
- `local-in-child`
  - 分脑只维护项目事实、当前状态和 body 入口
- `brain-as-doc-hub`
  - 权威治理文档、接管文档和长文参考默认只留在 `brain/` 或 `brain/references/`
- `docs-into-brain`
  - README、教程、审计、迁移说明等文档内容必须先整合进对应主脑或分脑；body 顶层文档只保留简体中文索引、公开指南或兼容入口，不能成为平行真源
- `simplified-chinese-docs`
  - 工作区内面向人读的项目治理与接管文档默认使用简体中文；英文只保留在代码标识、命令、第三方专名、链接或产品必须的多语言公开文档中

## 2. 已验证教训
- 如果主脑和分脑维护两套平行接管顺序，后续 agent 很快会漂移
- 如果当前状态只写聊天或终端，不写 brain，接管可靠性会明显下降
- 如果把 `episodic_memory` 当默认入口，接管速度和质量都会恶化
- 如果长文继续散落在 body 顶层，brain 的中枢地位会被稀释
- 如果 README 里保留了 brain 未收录的接管规则、命令入口或稳定结论，后续接管会重新绕过大脑
- 如果接管入口、命令入口或写回路由已经漂移，先纠偏再重开实验，通常比直接推进更能降低误操作风险
- 如果控制台显示疑似中文乱码，先用 UTF-8 读取工具确认真实文件内容，不能把终端编码错觉当作文件损坏来修
- 脑内文档铁律：当前层标题、正文、规则、状态和复盘写回必须使用简体中文；命令、路径、指标名、tag、模型名等技术标识保留原文
- 主分脑结构变更后必须跑 `brain_integrity_check.py --json`，确认父子附着、读序、写回路由、body 映射和编码合同仍一致
- 项目任务运行纪律已改为前台优先：训练、评估、审计、bounded study、confirmatory rerun 与执行任务不得默认后台化，不得中途人为中断，单次窗口时限统一按 `10` 小时处理
- 当前 `daily_research` 任务必须显式使用 `yolos` 环境；GPU 训练任务完成后必须核验 `training_diagnostics.json` 中 `device = cuda` 与 `cuda_available = true`

## 3. 当前长期边界
- 主脑不是分脑事实库
- 分脑不是跨项目规则库
- `daily_research` 负责正式生产研究与执行主线
- `t0_project` 负责盘中实验与 RL 原型，不直接替代正式主线
- `daily_stock_analysis-main` 是独立产品分脑，不改写 `daily_research` 默认执行
## 2026-04-25 daily_research r19 路由说明
- 主脑事实：`split_heads_portfolio_daily_ranking_r19` 只属于 `daily_research`，是 `research / shadow` 的组合级 receiver/source/cash 路径；没有改变 live artifact、生产默认或 promotion gate。

## 2026-04-26 daily_research r20 路由说明
- 主脑事实：`portfolio_daily_ranking_v2_gated`、`cash_constraint_portfolio_daily_ranking_cash_aware_guard_v13`、`split_heads_portfolio_daily_ranking_stability_r20`、`portfolio_daily_effective_model_action`、`source_realized_sell_floor` 和 stable confirm gate 都是 `daily_research` 分脑事实；主脑只保留它们仍为 `research / shadow_only` 的全局边界。

## 2026-04-26 daily_research r21 路由说明
- 主脑事实：`cash_constraint_portfolio_daily_ranking_source_exec_guard_v14`、`split_heads_portfolio_daily_ranking_source_exec_r21`、`source_not_sold_ceiling` 与 `portfolio_daily_effective_capital_transfer_count` 都是 `daily_research` 分脑事实；主脑只保留它们用于修复 source target 真实释放资金，且仍为 `research / shadow_only`。
- 全局教训：source execution 修复必须同时继承 cash-aware 分支；否则会从“不会释放 source”转成“会释放 source 但现金分支再死”。r21 retry2 已把下一瓶颈推到 order translation 与 add-to-hold 冲突。

## 2026-04-26 daily_research r22 路由说明
- 主脑事实：`cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15`、`split_heads_portfolio_daily_ranking_receiver_exec_r22`、`portfolio_daily_receiver_exec_guarded`、`portfolio_daily_receiver_add_headroom` 与 `portfolio_daily_receiver_realized_deploy_rate` 都是 `daily_research` 分脑事实；主脑只保留它们用于修复 receiver target 的真实可买性，且仍为 `research / shadow_only`。
- 全局教训：组合级 receiver 不能只按信号排序，还必须在进入 core deploy 前满足执行 headroom；否则 source 已释放资金也会被 add-to-hold 冲突吞掉。r22 smoke 证明该 guard 有效，但 smoke 过 gate 不等于 promotion。
