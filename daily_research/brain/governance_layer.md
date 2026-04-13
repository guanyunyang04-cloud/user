# Daily Research 治理层

快照日期：`2026-04-13`

## 1. 治理目标
- 防止 strongest-model、learned-control 和 live 主线发生目标偏移、上下文偏移和证据偏移
- 强制所有正式实验都走“读取状态 -> 检查 -> 执行 -> 写回 -> 重规划”闭环

## 2. 执行前四检
- 目标一致性检查
  - 这件事是否仍服务当前主问题和当前生产主线
- 规则冲突检查
  - 是否违反 formal / recent / live / promotion 分层、strict resume、前台执行等硬规则
- 经验教训检查
  - 是否踩中 `knowledge_center.md` 中已知失败模式
- 依赖完整性检查
  - 数据、脚本、解释器、预算和上下文是否足够

## 3. 默认接管纪律
- 新 agent 默认先读：
  - `identity_layer.md`
  - `state_center.md`
  - `knowledge_center.md`
  - `operations_center.md`
- `episodic_memory.md` 只在需要完整证据时再读

## 4. 写回纪律
- 当前状态、当前优先级、当前 handoff、当前时态：
  - `state_center.md`
- 稳定事实、规则、教训：
  - `knowledge_center.md`
- 新命令口径、入口变化、环境和流程：
  - `operations_center.md`
- 时间顺序过程和原始证据：
  - `episodic_memory.md`

## 5. 守卫
- `python daily_research/tools/project_consistency_check.py`
- `python daily_research/tools/doc_guard.py check`
