# T0 Project

## 1. 定位
`t0_project/` 是通达信盘中 T+0、监控与强化学习的独立实验区，用来验证盘中信号、执行抽象和 RL 原型。

它和 `daily_research/` 的正式执行主线严格分开：

- `daily_research/` 负责盘后研究、次日开盘执行建议与正式治理流程；
- `t0_project/` 负责盘中策略实验，不直接作为 `daily_research` 的默认执行入口。

## 2. 当前判断
- 当前最主要的策略入口是 `integrated_tq_strategy.py`。
- `execution/` 只是执行抽象层与真实交易适配器骨架，不是已打通的自动下单系统。
- `rl_agent/` 是强化学习研究分支，结论仍应按实验代码理解，不应默认视为可实盘部署模块。
- 目录下两份长篇中文文档更接近背景研究与架构论证，不是日常操作手册。

## 3. 推荐阅读顺序
1. 先看本文件，确认边界与入口。
2. 再看 `execution/README.md`，确认执行层抽象、`paper/live` 模式和安全边界。
3. 需要了解大背景或研究设想时，再看：
   - `先进技术通达信TQ策略研发.md`
   - `rl_agent/结合AI的通达信做T策略.md`

## 4. 当前结构
- `integrated_tq_strategy.py`
  - 盘中策略主入口。
- `my_t0_monitor.py`
  - 监控与辅助观察脚本。
- `backtest_integrated_strategy.py`
  - 集成策略回测入口。
- `select_stocks_only.py`
  - 轻量选股辅助脚本。
- `execution/`
  - 订单、风控、Broker 抽象与 `paper/live` 适配层。
- `rl_agent/`
  - 环境、训练、推理与强化学习实验代码。
- `tqcenter.py`
  - 通达信 TQCenter 交互底层。

## 5. 运行与安全边界
- `live` 模式当前仍是保护性骨架；未打通账户、委托、成交、撤单、持仓与异常恢复前，不应视为可实盘自动交易。
- 若只是验证执行流，应优先使用 `paper` 模式。
- 若需要和真实券商接口对接，应先把订单状态持久化、重复下单保护、超时撤单和断线恢复补齐。

## 6. 代码与产物边界
- 需要长期维护：
  - `t0_project/*.py`
  - `t0_project/execution/`
  - `t0_project/README.md`
  - `t0_project/*.md`
- 主要是生成物、默认不应手工维护：
  - `t0_project/models/`
  - `t0_project/logs/`
  - `t0_project/backtest_output/`
  - `t0_project/ppo_tdx_tensorboard/`
  - `t0_project/__pycache__/`

## 7. 常用维护命令
语法编译检查：

```bash
python -m compileall t0_project
```

只清理 `t0_project` 相关生成物：

```bash
python daily_research/tools/workspace_maintenance.py clean --targets pycache,tensorboard,t0_backtest_output,t0_logs
```

查看整个工作区体检：

```bash
python daily_research/tools/workspace_maintenance.py report
```

## 8. 维护规则
- `t0_project` 的实验结果不要静默替换 `daily_research` 的正式执行默认值。
- 背景研究长文继续保留，但日常入口、边界和维护规则以本文件与 `execution/README.md` 为准。
- 新增盘中策略若涉及真实执行能力，必须先在 `execution/` 抽象层补齐可验证的边界，再讨论接近自动化下单。
