# T0 Project 操作中枢

## 1. 项目地图
- 策略入口：
  - `t0_project/integrated_tq_strategy.py`
  - `t0_project/backtest_integrated_strategy.py`
  - `t0_project/select_stocks_only.py`
- 监控入口：
  - `t0_project/my_t0_monitor.py`
- 执行入口：
  - `t0_project/execution`
- RL 入口：
  - `t0_project/rl_agent`
- 网关入口：
  - `t0_project/tqcenter.py`

## 2. 默认操作纪律
- 先确认实验边界和隔离规则
- 先看文档归宿和 body 入口，再进代码
- 长篇研究文档与历史说明统一收口到 `brain/references/`

## 3. 写回路由
- 当前状态与下一步：
  - `state_center.md`
- 稳定事实、规则、教训：
  - `knowledge_center.md`
- 入口、环境、流程与命令：
  - `operations_center.md`
- 治理和接管纪律：
  - `governance_layer.md`
- 单轮实验过程：
  - `episodic_memory.md`
