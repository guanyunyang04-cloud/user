# T0 Project Semantic Memory

## 1. 项目身份
`t0_project/` 是通达信盘中 T+0、监控与强化学习的独立实验分项目，负责验证：

- 盘中信号
- 执行抽象
- RL 原型

上级主脑位于：

- `brain/master_brain.md`
- `brain/brain_manifest.json`

当前接入状态：

- 已接入主脑
- 不承担 `daily_research` 的正式默认值职责

## 2. 当前稳定认知
- 主策略入口：
  - `integrated_tq_strategy.py`
- 回测入口：
  - `backtest_integrated_strategy.py`
- 执行抽象：
  - `execution/`
- 强化学习分支：
  - `rl_agent/`
- 底层交互：
  - `tqcenter.py`

## 3. 当前边界
- `t0_project` 是实验分脑，不是正式生产执行主线
- `execution/` 仍是抽象层和适配骨架，不应被视为已打通自动下单
- `rl_agent/` 的结论只按实验代码与实验记录解释

## 4. 身子与脑子的映射
- 策略 body：
  - `t0_project/integrated_tq_strategy.py`
  - `t0_project/backtest_integrated_strategy.py`
  - `t0_project/select_stocks_only.py`
- 监控 body：
  - `t0_project/my_t0_monitor.py`
- 执行抽象 body：
  - `t0_project/execution/`
- RL body：
  - `t0_project/rl_agent/`
- 交互底层 body：
  - `t0_project/tqcenter.py`

## 5. 默认进入顺序
1. 先读本文件
2. 再读 `brain_architecture.md`
3. 再读 `working_memory.md`
4. 再读 `procedural_memory.md`
5. 需要命令时读 `environment_model.md`
6. 需要执行边界时读 `action_system.md`
7. 需要实验时间证据时读 `episodic_memory.md`
