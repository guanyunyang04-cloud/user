# 主脑当前判断

## 1. 当前全局优先级
- 一级优先级：
  - 巩固主脑与三个分脑的结构一致性、接管顺序和 manifest contract
- 一级优先级：
  - 维持 `daily_research` 作为当前正式生产主线
- 一级优先级：
  - 控制 `daily_research` 热工作集体积，优先归档陈旧 `cache/output`，避免大体量产物继续无界堆积
- 二级优先级：
  - 保持 `t0_project` 与正式执行主线隔离
- 二级优先级：
  - 保持 `daily_stock_analysis-main` 作为独立产品分脑，不与执行主线混写

## 2. 当前全局目标函数
- 对生产型分脑，默认目标函数为：
  - 收益优先、非降级
- 稳定性、坏市场收益、弱窗口修复：
  - 只能作为利润增益项或阶段控制器约束
- 若某分脑要接受“更稳但更低收益”的升级：
  - 必须由用户显式确认
  - 并写入该分脑 `working_memory.md`

## 3. 当前全局边界
- `daily_research` 的 live 默认值与 formal 结论仍是工作区最高优先级生产判断
- `t0_project` 的实验结果不得直接替代 `daily_research` 默认值
- `daily_stock_analysis-main` 是独立产品线，不接管 `daily_research` 执行默认值
- 后续 agent 的统一接入面是各级 brain，而不是 `README`
- `daily_research` 当前存在大体量 `cache/output` 热区，默认先做归档预演，不对最新活跃产物做盲删

## 4. 当前治理动作
- 工作区只保留 brain 体系作为长期认知载体
- 主脑持续维护 child_brains、body_root、attach_status 与 handoff contract
- 新增分项目时，必须先建立完整分脑，再允许接入主脑
- 当 `workspace_maintenance.py report` 出现热区告警时，优先走 `archive` dry-run 收敛 `daily_research/cache` 与 `daily_research/output`
