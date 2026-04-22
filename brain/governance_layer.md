# 主脑治理层

快照日期：`2026-04-13`

## 1. 治理目标
- 保证主脑与分脑始终遵守同一套精炼脑核
- 保证关键状态、规则、入口和风险始终外显
- 在结构漂移、状态过期、接管失真时强制修复

## 2. 标准闭环
1. `Wake`
   - 先读 `identity_layer.md` 和 `state_center.md`
2. `Locate`
   - 判断任务属于哪个分脑、哪个阶段、哪类问题
3. `Choose`
   - 选当前最该做的一件事
4. `Preflight`
   - 过目标、规则、教训、依赖四检
5. `Act`
   - 执行动作并保留证据
6. `Reflect`
   - 把结果沉淀回状态、知识、操作或治理
7. `Replan`
   - 如结果改写路径，立刻更新 state 与 next step

## 3. 守卫
- `python daily_research/tools/brain_bootstrap.py --child <brain_id> --json`
- `python daily_research/tools/brain_integrity_check.py --json`
- `python daily_research/tools/doc_guard.py check`
- 结构升级后必须保证：
  - manifest 可解析
  - 读取顺序可执行
  - 写回路由可追踪
  - 主脑 child 引用与分脑 manifest 一致
  - brain 中不存在平行真源

## 4. 写回纪律
- 工作区级当前状态写回 `brain/state_center.md`
- 工作区级长期知识写回 `brain/knowledge_center.md`
- 工作区级拓扑写回 `brain/master_brain.md`
- 工作区级操作与环境写回 `brain/operations_center.md`
- 工作区级治理写回 `brain/governance_layer.md`
