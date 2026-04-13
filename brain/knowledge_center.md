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

## 2. 已验证教训
- 如果主脑和分脑维护两套平行接管顺序，后续 agent 很快会漂移
- 如果当前状态只写聊天或终端，不写 brain，接管可靠性会明显下降
- 如果把 `episodic_memory` 当默认入口，接管速度和质量都会恶化
- 如果长文继续散落在 body 顶层，brain 的中枢地位会被稀释
- 如果接管入口、命令入口或写回路由已经漂移，先纠偏再重开实验，通常比直接推进更能降低误操作风险

## 3. 当前长期边界
- 主脑不是分脑事实库
- 分脑不是跨项目规则库
- `daily_research` 负责正式生产研究与执行主线
- `t0_project` 负责盘中实验与 RL 原型，不直接替代正式主线
- `daily_stock_analysis-main` 是独立产品分脑，不改写 `daily_research` 默认执行
