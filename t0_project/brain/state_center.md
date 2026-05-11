# T0 Project 状态中枢

快照日期：`2026-05-11`

## 1. 当前定位
- `t0_project` 是工作区内的盘中执行、执行抽象与 RL 实验分脑
- 它不直接替代 `daily_research` 的正式生产主线

## 2. 当前状态
- 当前仍应被视为实验分脑，而不是生产默认切换源
- 当前接管重点是：
  - 保持盘中执行与 RL 研究可接管、可验证、可复盘
  - 保持与 `daily_research` 的边界清晰
- 2026-05-11 维护结论：
  - 本轮只做离线静态验收，未连接真实通达信终端、券商接口或 live broker
  - `t0_project/**/*.py` 全量 `py_compile` 通过
  - 未产生可上行到 `daily_research` 生产主线的实验结论

## 3. 当前优先级
- 维持实验隔离
- 维持真实 body 入口与文档归宿一致
- 维持研究结论只在证据充分时再上行到主脑或 `daily_research`
- 后续任何执行侧变更必须先声明 dry-run / mock / real broker 边界

## 4. 当前风险
- 如果实验结论被口头外推到正式生产主线，会发生越权
- 如果长文和入口散落在 body，接管会继续分裂
- 如果静态验证被误写成实盘验证，会污染主分脑决策

## 5. 推荐下一步
- 接手前先看 `identity_layer.md`、`state_center.md`、`knowledge_center.md`、`operations_center.md`
- 真正影响正式主线的结论，必须回主脑和 `daily_research` 显式落盘
