# T0 Project Procedural Memory

## 1. 作用
本文档保存 `t0_project` 已验证的研究与实现规则。

## 2. 当前技能
### 2.1 隔离原则
- `t0_project` 的实验性结论不能直接外推为 `daily_research` 的正式默认值

### 2.2 执行接入顺序
- 先跑 `paper`
- 再做人工确认下单
- 最后才讨论真实自动交易接口

### 2.3 写入路由
- 稳定边界写 `semantic_memory.md`
- 当前优先级写 `working_memory.md`
- 环境与命令写 `environment_model.md`
- 执行抽象与流程写 `action_system.md`
- 单轮实验写 `episodic_memory.md`

### 2.4 主脑协同协议
- 接手前先读主脑 manifest，再读本分脑 manifest
- 若结构变更，先改 `brain_architecture.md`，再改 `brain_manifest.json`
- 若结论会影响正式生产主线，必须在主脑和 `daily_research` 分脑显式落盘，不能只在本分脑口头外推
