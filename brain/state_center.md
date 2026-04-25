# 主脑状态中枢

快照日期：`2026-04-25`

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

## 2026-04-23 daily_research r11 bounded study 收口状态
- 当前事实：
  - `daily_research` 已完成正式 bounded self-opt：`cp_v3_sell_source_contract_r11__study_r1`，共 `4` 个 screening trial 与 `2` 个 confirmatory trial，`objective_profile = sell_source_contract_v1`。
  - 当前可复现最佳分支不是 `result_value_v10`，而是 `loss_profile = alpha_result_value_budget_split_v10` 搭配 `budget_objective = result_value_v9`。
  - `result_value_v10 + alpha_result_value_budget_split_v10` 在 screening 可行，但 fresh confirmatory `confirm_02` 明显失稳，不能据此把 `v10 objective` 扶正。
  - 已手动补做同口径 runner-up confirm：`cp_v3_sell_source_contract_r11__study_r1__confirm_03_runnerup_alla`；其结果未超过 `confirm_01`。
  - 本轮还在导出链路中发现并修复了空 `share_actions` 时的 export 崩溃问题。
- 主脑层决策：
  - 主脑只保留全局结论：当前训练侧最有价值的新增信号在 `v10 loss / funding-release discipline`，不在当前 `v10 objective`。
  - sell-source 主矛盾已从 `budget_origin_sell_share` 收敛到 `deploy_funding_rebalance` 过度依赖与 `reduce/exit` 学习不足；详细指标、命令与产物继续只写 `daily_research/brain/`。
## 2026-04-23 daily_research r11 held-side release/funding 合同收紧状态
- 当前事实：
  - `daily_research` 已把 held-side 诊断继续接到正式研究链路：`analyze_behavior_gap.py` 新增 `protected_hold / funding_release` 支持度、`deploy_funding_release_consistent_share`、`model_release_signal_forward_excess_5d` 等字段；`run_self_optimizing_study.py` 新增 `sell_source_contract_v2` 与 `split_heads_sell_source_contract_r11b`。
  - 已对 `r11` 的 `confirm_01 / confirm_02 / confirm_03_runnerup_alla` 串行重跑新审计，并生成对比产物：`daily_research/output/continuous_policy/analysis/protocol_contract_comparisons/r11_sell_source_contract_v2_compare_20260423.json`。
  - 新证据表明，当前 held-side 主问题不是“大量卖到强保护旧仓”，而是 `deploy_funding_release_consistent_share` 在现有 confirm 分支中仍接近 `0`，说明 release/funding 判据还没有真正学成。
  - 在 `sell_source_contract_v2` 下，当前排序仍由 `confirm_01` 领先，`confirm_03_runnerup_alla` 与 `confirm_02` 被进一步拉开；`r11b` dry-run 已可直接进入后续正式 study。
- 主脑层决策：
  - 主脑继续维持“分层方向正确、held-side 学习闭环尚未完成”的全局判断，不回退到“整体原理错误”叙事。
  - 后续如继续投入正式算力，应优先围绕 `result_value_v9 + alpha_result_value_budget_split_v10 + cash_constraint_sell_source_guard_v7` 的 held-side release/funding 学习加强版推进，而不是让主脑承载实验细节。

## 2026-04-23 daily_research r11b v11 正式 study 状态
- 当前事实：
  - `daily_research` 已完成正式 bounded self-opt：`cp_v3_sell_source_contract_r11b__study_r1`，`objective_profile = sell_source_contract_v2`，共 `4` 个 screening trial 与 `2` 个自动 confirmatory trial，并补做了 `confirm_03_semantic_v11v9`。
  - 训练侧已新增 `alpha_result_value_budget_split_v11`，并让 `funding_release_discipline_loss` 支持 `v10 / v11` 变体；held-side 审计现已可导出逐事件明细。
  - `v11 + result_value_v10` 不再像旧 `v10/v10` 那样彻底失稳，但自动 confirm 仍为 `shadow_only`，且 `deploy_funding_rebalance_sell_share = 0.9699`、`deploy_funding_release_consistent_share = 0.0`，说明更强 loss 并没有自动换来 release 学成。
  - `v11 + result_value_v9` 证明 funding 语义可以继续被清理，`deploy_funding_rebalance_sell_share = 0.6957`、`deploy_funding_rebalance_forward_excess_5d = -0.0121`，但 `deploy_intent_realized_rate = 0.2627`，说明 held-side 改善尚未和 deploy executability 同步成立。
  - held-side 逐事件明细显示：自动 confirm 共有 `129` 次 funding trim，集中在 `002371.SZ / 001309.SZ / 002049.SZ`；语义线只有 `16` 次 funding trim，集中在 `002049.SZ / 002157.SZ`，但两条线的 `deploy_funding_release_consistent_share` 都仍为 `0.0`。
- 主脑层决策：
  - 主脑维持“整体分层方向正确、真正瓶颈已收敛到 held-side release 学习与 deploy/order translation 耦合”的判断。
  - 当前不允许把 `result_value_v10` 升为默认 objective，也不允许把 `v11` 单独视为 promotion 证据；实验细节、命令与逐仓证据继续只写 `daily_research/brain/`。

## 2026-04-24 daily_research r13 动作价值统一入口状态
- 当前事实：`daily_research` 已落地 `split_heads_action_value_unification_r13`、`alpha_result_value_budget_split_v13` 与 `action_value_unification_v1`，并完成 `cp_v3_action_value_unification_r13__study_r1`；champion 为 `confirm_01 = alpha_result_value_budget_split_v13 + result_value_v10`，但仍是 `shadow_only`。
- 主脑层决策：主脑只记录“动作价值统一已完成正式 shadow 验证但仍不可上线，月度收益评价已进入分脑通用曲线指标”；不切 live、不改 promotion gate，细节继续由 `daily_research/brain/` 承载。

## 2026-04-24 daily_research r14 直接日级动作仲裁入口状态
- 当前事实：`daily_research` 已完成 `cp_v3_direct_action_value_r14__study_r1` screening 与 repaired confirm；`confirm_01 = alpha_result_value_budget_split_v14 + result_value_v9` 收益、Sharpe 与月度收益质量显著改善，但仍为 `shadow_only`，failure mode 仍是 `order_translation_drift`。
- 主脑层决策：r14 证明直接动作值仲裁有价值，但 release/funding 与订单翻译未闭合；不切 live、不改 active artifact、不改 promotion gate，细节继续由 `daily_research/brain/` 承载。
