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
- 默认写代码采用主脑 personal researcher direct-change：实验脚本、旧 helper、旧测试若无真实调用证据或证据价值，直接改到当前合约或删除
- 默认只做离线、mock 或静态验证；连接真实通达信、券商或 live broker 前必须显式确认
- 文档语言遵循 `brain/language_policy.md`：中文语义 + 英文工程标识；命令、路径、JSON key、workflow id 不翻译。

## 3. 验证入口
- 默认 changed-surface 验证：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.selective_verification --paths <changed_paths> --json`
  - 按 `blocking_commands` 执行本次最小验证；未映射 Python 改动先 manual review 或补同面测试
- Python 静态编译：
  - `python -m py_compile <all t0_project/**/*.py>`
  - 仅作为共享入口、执行边界或维护/收尾扩展验证，不作为普通单文件改动默认验证
- 实验入口仅在 dry-run / mock 边界明确时运行；不得把实验结果自动上行到生产主线

## 4. 写回路由
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
