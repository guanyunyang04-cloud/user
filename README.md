# Workspace Overview

这个工作区目前保留两条彼此隔离的主线：

- `daily_research/`
  - 日线研究、正式执行口径、执行端包装脚本、`deep_alpha` 主研究线。
- `t0_project/`
  - 通达信盘中 T+0、执行抽象与强化学习实验区，不直接作为正式执行入口。

## 当前入口

- `daily_research/brain/`
  - `daily_research` 的 AI 工作流脑区目录，集中承载当前状态、项目地图、计划、运行基线与研究日志。
- `daily_research/brain/README.md`
  - `daily_research` 的当前状态、主入口与文档边界。
- `daily_research/brain/project_map.md`
  - `daily_research` 的项目背景、当前瓶颈、未来方向与协作导航。
- `daily_research/brain/daily_research_plan.md`
  - `daily_research` 的当前默认决策、升级 shortlist、优先级与停止规则。
- `daily_research/brain/runtime_environment.md`
  - `daily_research` 的解释器、依赖与推荐调用口径。
- `daily_research/brain/research_log.md`
  - 按时间顺序记录实验、证据与结论。
- `daily_research/execution/README.md`
  - 执行端日常操作手册。
- `t0_project/README.md`
  - `t0_project` 的边界、入口与安全约束。

## 代码与产物边界

- 需要长期维护的源码与文档：
  - `daily_research/baseline/`
  - `daily_research/brain/*.md`
  - `daily_research/deep_alpha/`
  - `daily_research/execution/`
  - `daily_research/tools/`
  - `t0_project/*.py`
  - `t0_project/execution/`
  - `t0_project/*.md`
- 主要是生成物、默认不应手工维护：
  - `daily_research/cache/`
  - `daily_research/output/`
  - `daily_research/archive/`
  - `daily_research/execution/output/`
  - `t0_project/backtest_output/`
  - `t0_project/logs/`
  - `t0_project/models/`
  - `t0_project/ppo_tdx_tensorboard/`
  - 全局 `__pycache__/`

## 常用维护命令

文档检查：

```bash
python daily_research/tools/doc_guard.py check
```

工作区体检：

```bash
python daily_research/tools/workspace_maintenance.py report
```

研究产物归档预演：

```bash
python daily_research/tools/workspace_maintenance.py archive
```

确认后执行归档：

```bash
python daily_research/tools/workspace_maintenance.py archive --apply
```

安全清理 `__pycache__`：

```bash
python daily_research/tools/workspace_maintenance.py clean --targets pycache --apply
```

Git 当前状态：

```bash
git status --short
```

## 当前维护规则

- `daily_research/brain/README.md` 只写当前状态、主入口与文档边界。
- `daily_research/brain/project_map.md` 只写项目背景、当前瓶颈、未来方向与协作导航。
- `daily_research/brain/daily_research_plan.md` 只写当前默认决策、shortlist、优先级与停止规则。
- `daily_research/brain/runtime_environment.md` 只写解释器、依赖与运行口径。
- `daily_research/brain/research_log.md` 只写时间顺序实验记录。
- 文档统一使用 UTF-8，不再用 shell 重定向直接追加中文内容。
- 默认只清理可再生产物，不直接删除模型产物、持仓快照和研究结论文件。
