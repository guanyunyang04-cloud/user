# Daily Stock Analysis Brain Architecture

## 1. 结构目标
`daily_stock_analysis-main/brain/` 是该项目的 AI 控制面，用来把复杂的产品 body 映射成可交接的脑结构。

## 2. 分脑模块
- `semantic_memory.md`
  - 项目身份、核心入口、body_map
- `working_memory.md`
  - 当前优先级、改动边界、近期治理目标
- `procedural_memory.md`
  - 仓库约束、验证矩阵、AI 协作规则
- `environment_model.md`
  - Python / Web / API / 测试命令口径
- `action_system.md`
  - 系统入口、模块分层、执行链路
- `episodic_memory.md`
  - 后续本地按时间记录的重要改动与证据
- `brain_manifest.json`
  - 机器可读读写路由、body_map、handoff_contract

## 3. 写入路由
- 稳定认知：
  - `semantic_memory.md`
- 当前优先级：
  - `working_memory.md`
- 可复用仓库规则：
  - `procedural_memory.md`
- 环境与命令：
  - `environment_model.md`
- 系统入口与模块路由：
  - `action_system.md`
- 时间顺序证据：
  - `episodic_memory.md`

## 4. Body 映射原则
- brain 不重复保存源码细节
- brain 只负责说明源码 body 应该如何被理解与进入
- agent 先接 brain，再进入 `src/ api/ apps/ bot/ data_provider/ tests/`

## 5. 去冗余规则
- 现有 `README.md`、`docs/`、`AGENTS.md`、`CLAUDE.md` 等可保留为 body 历史资产或上游说明
- 但当前 AI 接管入口以本分脑为准
