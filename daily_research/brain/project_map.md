# Daily Research 项目地图

快照日期：`2026-04-05`

## 1. 一句话概括
- `daily_research` 已经从分散研究沙盒，收成了一条以执行后净收益为目标的研究-执行统一链路。

## 2. 主线演化
1. 因子与排序阶段：
   - 确认排序和调仓节奏是主矛盾。
2. 市场状态阶段：
   - 确认 A 股需要条件化处理。
3. 研究执行对齐阶段：
   - 围绕 `next_open`、滚动股票池、盘后计划收口。
4. 双主线阶段：
   - `advanced_ml` 承担旧执行主线
   - `deep_alpha` 打开新 alpha 探索
5. 统一目标阶段：
   - 研究与执行统一到 execution-first
   - default execution 改为 manifest-driven
   - family epoch budget 进入正式协议

## 3. 已解决的问题
- 已有单一的默认执行入口。
- 已有单一的显式回退入口。
- 研究与执行目标函数已经统一。
- production promotion 已能继承 research winner 的执行配置。
- training budget 不再是隐含变量，而是 family-based manifest。

## 4. 当前两条主线
- liquid500 默认执行升级线：
  - 当前 active default 已是 `state_liquidity_listwise_v1 + regoff_k2_10d_ensemble_native_anchor`
- liquid800 / mainboard 研究升级线：
  - 当前最强候选是 `dynamic_graph_no_priors`

## 5. 当前不再是主线的方向
- `structure_context_only`：
  - 仍可保留为 raw 架构挑战者
  - 但不再是 execution-upgrade 主线
- 旧的“统一手填 epoch”：
  - 已被 family budget manifest 取代
- 旧的“raw winner 直接上执行”：
  - 已被 formal -> recent gate -> production promotion 三层协议取代

## 6. 当前瓶颈
- `short_alpha` 虽然已经完成 production full-fit promotion 并切成 active default，但当前 production recipe 仍显示 budget pressure，后续还要补 epoch extension。
- `dynamic_graph_no_priors` 虽然是 liquid800 研究 winner，但还未证明自己适合作为 liquid500 默认执行替代。
- 多窗口、大矩阵与 production promotion 仍然是主要时间成本。

## 7. 当前下一阶段
1. 先把 `state_liquidity_listwise_v1` 的 production recipe 做 second-stage epoch frontier。
2. 用 production replay 复核扩预算后的 short-alpha 是否继续优于当前 active default。
3. 继续把 `dynamic_graph_no_priors` 保持为独立研究线，而不是直接混入 liquid500 默认执行升级判断。
