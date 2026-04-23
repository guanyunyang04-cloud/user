# 主脑状态中枢

快照日期：`2026-04-23`

## 1. 当前接管摘要
- 工作区正式生产主线仍是 `daily_research`
- 已接入主脑的分脑固定为：
  - `daily_research`
  - `t0_project`
  - `daily_stock_analysis-main`
- 当前接管默认先走主脑，再进入目标分脑
- 当前高频接管快路稳定为：
  - `identity -> state -> knowledge -> topology -> operations -> governance`

## 2. 当前重点
- 维持主脑作为共享脑核，不再让分脑各自复制一套合同
- 维持“主脑管共性、分脑管区域差异、权威文档只留在 brain”
- 维持 `daily_research` 的正式生产主线地位
- 维持 `t0_project` 和 `daily_stock_analysis-main` 的边界不越权
- 当前若发现接管入口漂移，不直接重开实验
- 先修正接管入口，作为更高优先级的风险控制

## 3. 当前时态
- `Past`
  - 已完成主脑与三个分脑的统一附着
  - 已完成权威文档向各级 `brain/` 收口
- `Present`
  - 共享脑核合同已上收至主脑
  - 分脑读取顺序已改为继承派生，不再各自手写整份顺序
  - 主分脑完整性守卫已补齐为 `daily_research/tools/brain_integrity_check.py`
- `Future`
  - 继续压缩脑内重复文档
  - 继续让 agent 优先读取中枢，而不是盲扫旧说明

## 4. 当前风险
- 如果后续只改分脑、不改主脑，共性结构会重新漂移
- 如果后续仍把旧别名文件当权威正文，内部融合会失效
- 如果长期不写回中枢，接管会退化回依赖隐性会话上下文
- 如果结构变更只跑文本守卫、不跑主分脑完整性守卫，manifest 与实际接管路径可能静默分叉

## 5. 推荐下一步
- 若主脑或目标分脑存在入口、命令、写回路由漂移，先纠偏再执行重动作
- 所有结构变更继续先改主脑合同，再改区域分脑
- 后续新增文档默认先判断能否并入现有中枢
- 只有确实需要独立中枢时，才新增 brain 内新文件

## 2026-04-22 文档收口状态
- 当前新增全局文档治理要求：
  - README、教程、审计、迁移说明等文档内容必须先整合进对应 brain
  - body 顶层文档只保留简体中文索引、公开指南或兼容入口
  - 面向接管和治理的文档默认使用简体中文
- 已处理：
  - `daily_research/README.md` 已改为简体中文快速索引，并指向 `daily_research/brain/`
  - `daily_stock_analysis-main/README.md` 已标注 AI 接管真源为 `daily_stock_analysis-main/brain/`
  - `daily_stock_analysis-main` README 的稳定产品内容已整合到其 `knowledge_center.md` 与 `operations_center.md`
- 当前要求：
  - 后续 README / docs 改动必须同步评估 brain 写回
  - 文档结构变更后继续运行 `brain_integrity_check.py --json` 与 `doc_guard.py check`

## 2026-04-22 主分脑兼容入口收口状态
- 当前事实：
  - `daily_stock_analysis-main` 的 AI 兼容入口已统一指向分脑真源：`AGENTS.md`、`CLAUDE.md`、`.github/copilot-instructions.md`、`.github/instructions/governance.instructions.md`、`SKILL.md` 与 `strategies/README.md` 均不再作为平行权威正文。
  - `daily_research/execution/使用教程.md` 已标注权威操作真源为 `daily_research/brain/operations_center.md`。
  - `daily_research/tools/doc_guard.py check` 已增加上述入口必须回指 brain 的片段守卫。
- 当前决策：
  - AI 兼容文档可以保留为外部工具入口，但稳定规则、目录边界、验证矩阵和写回要求必须沉淀进对应 brain。
  - 若兼容入口与 brain 冲突，先按 brain 纠偏，再同步兼容入口。
- 当前验证：
  - `daily_research/tools/brain_integrity_check.py --json` 通过。
  - `daily_research/tools/doc_guard.py check` 通过。
  - `daily_stock_analysis-main/scripts/check_ai_assets.py` 通过。

## 2026-04-22 daily_research r10 代码侧修正状态
- 当前事实：
  - `daily_research` 已在其分脑中记录一轮新的 r10 code-side simulator 修正，目标是缓解预算/动作翻译漂移，而不是直接改写研究结论。
  - 该修正基于现有冠军模型 `cp_v3_deploy_executability_r10__study_r1__confirm_02` 做了 patched evaluation / audit，结果显著改善了 deploy realization、收益、Sharpe 与 cash timing。
  - 但 sell-side 相关瓶颈仍在，`daily_research` 分脑已明确保持 `shadow_only`，没有把这次 patched eval 误写成 promotion。
- 主脑层决策：
  - 主脑继续只记录“这轮代码侧修正已发生且已写回分脑”，不在主脑重复存放具体策略指标细节。
  - 若后续还有类似 code-side 修正，仍然必须先写回目标分脑，再在主脑留一句状态摘要，避免主脑变成平行实验日志。

## 2026-04-22 daily_research r10 卖出来源归因状态
- 当前事实：
  - `daily_research` 已补做一轮 r10 source attribution 复跑，用来拆清“真实卖出来自模型还是预算层”。
  - 新证据表明，当前 sell-side 的更深层问题是 `budget_origin_sell_share` 很高，而不是简单的 sell intent suppression。
  - `daily_research` 分脑已记录这轮复跑只用于来源归因，不改写正式 protocol verdict。
- 主脑层决策：
  - 主脑继续只保留一句全局判断：`daily_research` 当前已把 sell-side 问题从“是否被压掉”推进到“是谁在实际制造卖出”的责任分解阶段。
  - sell-source attribution 的详细指标、命令与策略含义，继续只写在 `daily_research/brain/`，不在主脑重复展开。

## 2026-04-23 daily_research r10 卖出来源解耦状态
- 当前事实：
  - `daily_research` 已完成一轮 r10 sell-source 解耦代码侧评估，当前分脑结论落在 `cash_constraint_sell_source_guard_v7` / v7c。
  - 这轮工作把真实卖出从隐藏 budget-origin 副作用拆成模型显式卖出、模型释放信号与显式 deploy funding rebalance。
  - 分脑已记录 v6、v7、v7b、v7c 的证据链；v7c 不是最高收益分支，但当前语义最稳健。
  - 本轮未启动训练、未停止训练、未切换 live 默认执行、未改写 promotion gate。
- 主脑层决策：
  - 主脑只记录状态摘要，不复制具体指标；详细证据、命令、合同与遗留问题继续以 `daily_research/brain/` 为真源。
  - 后续若继续推进，应优先在分脑中验证训练级 value/release/cash timing 闭环，而不是让主脑承载实验细节。
