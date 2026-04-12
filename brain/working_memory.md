# 主脑当前判断

快照日期：`2026-04-12`

## 1. 角色定位
- 主脑负责工作区级拓扑、统一合同、跨项目边界和治理顺序。
- 当前正式生产主线仍是 `daily_research`。
- `t0_project` 与 `daily_stock_analysis-main` 都是已接入主脑的独立分脑，不直接改写生产主线。

## 2. 当前状态
- 主脑和三个分脑已经统一到同一套附着合同与接管快路。
- 当前工作区的高频接管入口已经稳定为：
  - `identity -> handoff -> rule -> lesson -> temporal -> working -> procedural -> governance`
- `daily_research` 仍是当前最复杂、最重生产责任的样板分脑。

## 3. 当前优先级
- 维持主脑与三个分脑的结构一致性、接管顺序和 manifest contract。
- 维持 `daily_research` 作为当前正式生产主线。
- 保持 `t0_project` 与正式执行主线隔离。
- 保持 `daily_stock_analysis-main` 作为独立产品分脑，不与执行主线混写。
- 持续把工作区级规则、教训和交接包写回主脑，而不是留在会话里。

## 4. 当前边界
- `daily_research` 的 live 默认值与 formal 结论仍是工作区最高优先级生产判断。
- `t0_project` 的实验结果不得直接替代 `daily_research` 默认值。
- `daily_stock_analysis-main` 是独立产品线，不接管 `daily_research` 执行默认值。
- 工作区级 AI 接入面是各级 brain，不是 `README` 或聊天记录。

## 5. 当前主问题
- 主脑层当前最重要的问题已经不是“有没有脑结构”，而是：
  - 如何长期维持“共性在主脑、差异在分脑、状态持续写回”这套治理闭环。

## 6. 当前风险
- 如果后续改动只写分脑、不写主脑，共性结构会再次漂移。
- 如果后续接管回到“先读长日志”，标准交接包会失去意义。
- 如果分脑长期不按模板写回，结构统一但状态会再次分裂。

## 7. 推荐下一步
- 继续让主脑只维护共性合同，不复制分脑内部事实。
- 继续让分脑只维护项目差异，不重复定义通用模块合同。
- 每次结构变更后都跑 `brain_bootstrap.py`、`doc_guard.py check` 和 `project_consistency_check.py`。
