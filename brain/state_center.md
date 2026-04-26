# 主脑状态中枢

快照日期：`2026-04-25`

## 2026-04-26 daily_research r19 bounded study 与运行纪律状态
- 当前事实：`daily_research` 已完成 `cp_v3_portfolio_daily_ranking_r19__study_r1`，共 `4` 个 screening trial、`2` 个 confirmatory trial，训练诊断均为 `formal_torch_seq_v3 / device=cuda / cuda_available=true`，任务按 `yolos` 口径执行。
- 主脑层结论：r19 仍是 `research / shadow_only` 证据；没有切换 live、没有改写 active artifact、没有改变 promotion gate。
- 关键判断：r19 证明组合级 receiver/source/cash ranking 指标已经可训练、可评估、可审计，但当前 `portfolio_daily_ranking_v1` 的 scoring 与真实收益/回撤仍不完全一致，具体实验细节继续由 `daily_research/brain/` 承载。
- 运行纪律已更新：项目任务默认前台运行、不中断，窗口时限统一为 `10` 小时；`daily_research` 任务必须使用 `yolos`，GPU 训练完成后必须核验 `device=cuda` 与 `cuda_available=true`。

## 2026-04-25 daily_research r18 根因与组合级方向状态
- 当前事实：`daily_research` 已落地 `split_heads_direct_action_pair_cost_guard_r18`、`cash_constraint_direct_action_pair_cost_guard_v11` 与 `direct_action_pair_cost_guard_v1`；r18 smoke2 已让 `direct_action_core_minus_pair_forward_excess_5d` 转正并降低换手，但仍是 research / `shadow_only`。
- 已固定五个卡点：个股动作不等于组合决策、局部动作目标会互相打架、卖出/现金/source credit assignment 最难、日频数据有盲区、有效 regime 样本小。
- 主脑层决策：具体 r18 指标、命令、审计与合同继续由 `daily_research/brain/` 承载；根因排序为目标函数与组合级决策不一致优先，其次是卖出责任分配，最后才是数据细度与 regime 样本。
- 后续方向：不再继续堆动作 loss，转向组合级日决策模型，学习谁获得资金、谁释放资金、释放多少以及是否保留现金。

## 2026-04-24 主分脑维护状态
- 当前事实：主脑与 `daily_research` 分脑接管链路已复核，身份层可变 live 默认旧口径已纠偏；active 执行物化真源仍是 `daily_research/output/active_execution_strategy.json`。
- 已完成 `daily_research` 的 `state_center.md`、`operations_center.md` 与 `continuous_policy_design_contract.md` 历史归档压缩；长原文与标题索引已下沉到 `daily_research/brain/references/`。
- 当前决策：主脑只记录维护摘要；`daily_research` 的具体 live、research、continuous_policy 状态继续由分脑承载。
- 若身份层再次出现可变 live 默认或具体实验指标，应视为文档职责漂移；若入口再次膨胀，应先追加 references / episodic 索引。
- r12-r18 只改变 research / shadow 搜索与审计口径；不改变 active artifact、live 默认执行或 promotion 结论，具体指标继续由分脑承载。

## 2026-04-23 daily_research r11 卖出来源契约训练侧接通状态
- 当前事实：
  - `daily_research` 已把 sell-source 问题从代码侧 calibration 继续推进到训练侧契约，新增了 `result_value_v10`、`alpha_result_value_budget_split_v10`、`split_heads_sell_source_contract_r11` 与 `sell_source_contract_v1`。
  - 已完成 `r11 dry-run`、短窗 `v9 vs v10` 教师回放比较，以及 `sell_source_contract_v1` 对真实 `protocol_summary.json` 的评分 smoke。
  - 新证据表明：训练侧契约已经接通，但当前短窗效果偏谨慎收缩，不宜把“契约已接通”直接等同于“正式收益已验证”。
  - 本轮仍未启动新的正式训练、未停止训练、未切换 live 默认执行、未改写 promotion 结论。
- 主脑层决策：
  - 主脑只记录推进阶段变化：`daily_research` 当前已经从“拆清卖出来源”进入“把卖出来源合同接入训练目标、损失和 study 打分”阶段。
  - 是否投入新的 10h bounded study，由分脑基于现有短窗证据继续执行；主脑不展开具体实验细节。

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
- 若主脑或目标分脑存在入口、命令、写回路由漂移，先纠偏；结构变更继续先改主脑合同，再改区域分脑
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

## 2026-04-23 daily_research r11-r11b 卖出来源与 held-side 状态摘要
- 当前事实：r11/r11b 已把 sell-source、funding release、held-side detail 和补充 confirm 接入正式研究链路；最佳稳定方向仍围绕 `result_value_v9` 与更强 funding discipline，而不是把 `result_value_v10` 或单独 `v11` 扶正。
- 主脑层决策：主脑只保留“主矛盾已收敛到 held-side release 学习与 deploy/order translation 耦合”的全局判断；具体命令、指标、逐仓证据、失败分支和导出修复继续只由 `daily_research/brain/` 承载。

## 2026-04-24 daily_research r13 动作价值统一入口状态
- 当前事实：`daily_research` 已落地 `split_heads_action_value_unification_r13`、`alpha_result_value_budget_split_v13` 与 `action_value_unification_v1`，并完成 `cp_v3_action_value_unification_r13__study_r1`；champion 为 `confirm_01 = alpha_result_value_budget_split_v13 + result_value_v10`，但仍是 `shadow_only`。
- 主脑层决策：主脑只记录“动作价值统一已完成正式 shadow 验证但仍不可上线，月度收益评价已进入分脑通用曲线指标”；不切 live、不改 promotion gate，细节继续由 `daily_research/brain/` 承载。

## 2026-04-24 daily_research r14 直接日级动作仲裁入口状态
- 当前事实：`daily_research` 已完成 `cp_v3_direct_action_value_r14__study_r1` screening 与 repaired confirm；`confirm_01 = alpha_result_value_budget_split_v14 + result_value_v9` 收益、Sharpe 与月度收益质量显著改善，但仍为 `shadow_only`，failure mode 仍是 `order_translation_drift`。
- 主脑层决策：r14 证明直接动作值仲裁有价值，但 release/funding 与订单翻译未闭合；不切 live、不改 active artifact、不改 promotion gate，细节继续由 `daily_research/brain/` 承载。

## 2026-04-26 daily_research r20 v2/v13 状态摘要
- 当前事实：`daily_research` 已把 r19 组合日频排序推进到 `portfolio_daily_ranking_v2_gated`、`cash_constraint_portfolio_daily_ranking_cash_aware_guard_v13` 与 `split_heads_portfolio_daily_ranking_stability_r20`；现金分支已转活，但 source realized sell 与 fresh confirm 回撤边界仍未闭合。
- 主脑层决策：主脑只记录全局摘要；r20 当前没有合格 v2 champion，仍为 `research / shadow_only`，不能进入 live 或 promotion，详细指标、命令、gate 报告和失败 confirm 证据继续以 `daily_research/brain/` 为真源。

## 2026-04-26 daily_research r21 source-exec 状态摘要
- 当前事实：`daily_research` 已新增 `cash_constraint_portfolio_daily_ranking_source_exec_guard_v14` 与 `split_heads_portfolio_daily_ranking_source_exec_r21`，把 source target 的真实 reduce/exit 从事后 gate 推进到模拟器执行链路，并新增 source not sold 与 effective capital transfer 审计。
- 当前事实：r21 retry2 smoke 已让 source execution 和 cash branch 同时转活，但仍因 order translation 与 add-to-hold 冲突没有合格 v2 champion。
- 主脑层决策：主脑只记录全局摘要；r21 仍为 `research / shadow_only`，不改变 active execution artifact，细节继续以 `daily_research/brain/` 为真源。
