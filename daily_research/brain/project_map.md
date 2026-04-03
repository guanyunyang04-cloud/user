# Daily Research 项目地图

快照日期：`2026-04-03`

## 1. 用途
- 本文件解释项目为什么会演化成现在这条主线。
- 它回答三件事：
  - 已经解决了什么
  - 真实瓶颈是什么
  - 还值得继续投入什么

## 2. 一句话概括
- `daily_research` 已经从松散研究沙盒，收成了一条单一的日常执行链路。
- 当前问题不再是“系统能不能跑起来”。
- 当前问题是“新的 alpha 家族能不能在现实成本和多窗口下，真正赢过当前默认执行策略”。

## 3. 主线演化
1. 因子与打分阶段：
   - 确认了排序和调仓节奏的重要性。
2. 市场状态阶段：
   - 确认了 A 股必须做条件化处理。
3. 研究执行对齐阶段：
   - 围绕 `next_open`、滚动股票池、盘后计划收口。
4. 双主线阶段：
   - `advanced_ml` 承担执行
   - `deep_alpha` 打开新 alpha 探索
5. 治理与收敛阶段：
   - 主板范围股票池
   - 正式 walk-forward
   - `target_weight` 直连桥
   - 固定锚点的全相位 ensemble
   - 明确旧主线回退

## 4. 已解决的问题
- 现在已有单一的每日默认策略。
- 现在已有单一的显式回退策略。
- 研究与执行的核心假设已经统一。
- “翻译损耗”方向已经被收口：
  - 有价值的是 `target_weight` 直连桥加执行节奏处理
  - 不是旧的 `score -> weight` 翻译器
- `dynamic_graph_v1` 已被证明是真实研究优胜项，不是 prior 假象。
- `short_alpha` 首轮矩阵已经说明：
  - 当前最有效的增益来自上下文容量与轻量排序损失
  - 不是第一版短线目标改写或额外日线短线输入
- `state_liquidity_listwise_v1` 已通过第一轮三窗 formal：
  - 不是单个 recent-window lucky run
  - 但第一窗仍明显偏弱
- 旧时代极高收益已因泄漏风险被降级为审计产物。
- 研究判决与日常执行现在已经正式拆成两层：
  - formal holdout 继续负责研究 winner 判决
  - production full-fit 负责默认日常执行

## 5. 当前瓶颈
- 研究端 alpha 迁到执行端时仍会损耗。
- 高收益 execution-alignment 候选往往在现实成本下变得过于高换手。
- `dynamic_graph_v1` 虽然已在研究端赢过 `plain`，但还没有全面统治执行端。
- 多窗口重训仍然是当前最大的研究时间成本。
- 新的 `state_liquidity_listwise_v1` 虽然已通过第一轮 formal，但还没有回答“加上执行翻译与现实成本后是否仍成立”。
- 想要在日线主板范围里抓“单票起爆前”信息，当前这第一版短线目标与短线特征仍然不够强。

## 6. 当前方向
- 执行侧：
  - 每日默认入口改为读取 `active_execution_strategy.json`
  - 当前 active strategy 已正式解析到 `baseline_current_execfirst_winner`
  - 每日默认已切到 `deep_alpha_liquid500_dynamic_graph_bridge_production_default`
  - 继续以 `regon_k1_10d_ensemble_native_anchor` 为收益上沿对照
- 桥接侧：
  - 继续沿 `target_weight` 直连桥推进
  - 继续做显式成本对齐
  - 继续做 cadence 与 ensemble 稳定化
- 研究侧：
  - 继续以 `dynamic_graph_v1` 为主前沿
  - `baseline_current + regoff_k2 execalign` 已完成 execution-first 正式上位
  - `state_liquidity_listwise_v1` 保留为次一级 execution objective 对齐候选
  - 保留 `dynamic_graph_no_priors` 与 `dynamic_graph_topk4` 作为对照
  - 暂不继续扩张第一版 `short_target / short_input` 路线
- 下一代家族只在图路线边际增益放缓后再开：
  - `state-conditioned MoE`
- RL 不进入当前默认主线：
  - 继续留在 `t0_project`

## 7. 协作规则
- 稳定事实写入 `semantic_memory.md`。
- 当前判断与升级门槛写入 `working_memory.md`。
- 日常命令写入 `action_system.md`。
- 长历史和长过程统一留在 `episodic_memory.md`。
- 上线前重训统一通过：
  - `daily_research/execution/update_default_candidate_production.py`
