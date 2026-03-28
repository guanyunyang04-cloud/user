# T0 Project Semantic Memory

## 1. 项目身份
`t0_project/` 是通达信盘中 T+0、监控与强化学习的独立实验分项目，用来验证：

- 盘中信号
- 执行抽象
- RL 原型

它和 `daily_research/` 的正式执行主线严格隔离。

## 2. 当前判断
- 当前最主要的策略入口是 `integrated_tq_strategy.py`
- `execution/` 是执行抽象层与真实交易适配器骨架，不是已打通的自动下单系统
- `rl_agent/` 是强化学习研究分支，结论仍应按实验代码理解

## 3. 分脑模块
- `t0_project/brain/semantic_memory.md`
- `t0_project/brain/brain_architecture.md`
- `t0_project/brain/working_memory.md`
- `t0_project/brain/procedural_memory.md`
- `t0_project/brain/environment_model.md`
- `t0_project/brain/action_system.md`
- `t0_project/brain/episodic_memory.md`
- `t0_project/brain/brain_manifest.json`

## 4. 身子与脑子的映射
- 主策略 body：
  - `t0_project/integrated_tq_strategy.py`
  - `t0_project/backtest_integrated_strategy.py`
- 执行抽象 body：
  - `t0_project/execution/`
- 强化学习 body：
  - `t0_project/rl_agent/`
- 底层交互 body：
  - `t0_project/tqcenter.py`

## 5. 当前结构
- `integrated_tq_strategy.py`
  - 盘中策略主入口
- `my_t0_monitor.py`
  - 监控与辅助观察脚本
- `backtest_integrated_strategy.py`
  - 集成策略回测入口
- `select_stocks_only.py`
  - 轻量选股辅助脚本
- `execution/`
  - 订单、风控、Broker 抽象与 `paper/live` 适配层
- `rl_agent/`
  - 环境、训练、推理与强化学习实验代码
- `tqcenter.py`
  - 通达信 TQCenter 交互底层
