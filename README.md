# 工作区入口

本 README 只作为 agent 和维护者的第一眼索引。权威接管真源是主脑 `brain/`，项目事实由主脑路由后的分脑维护。

当前工作区根目录：`H:\quant_project`。旧通达信插件目录 `H:\new_tdx64\PYPlugins\user` 不再承载本项目代码、脑区或研究产物。

## Agent 接管入口

接管、路由或修改 tracked 文件前，先运行主脑 capsule：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<task>" --json
```

只需要判断任务归属时运行：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow route --task "<task>" --json
```

需要显式启动主脑或分脑上下文时运行：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow bootstrap --brain <brain_id|workspace> --json
```

## 主分脑边界

- `brain/`：工作区主脑，管理接管顺序、任务路由、分支纪律、全局规则和禁区。
- `daily_research/brain/`：正式生产研究与执行主线的项目事实层。
- `t0_project/brain/`：盘中实验与 RL 原型事实层。
- `daily_stock_analysis-main/brain/`：独立产品项目事实层。

如果 README、AGENTS、CLAUDE、SKILL 与主脑冲突，以主脑 capsule 和 `brain/` 当前文档为准。

## 常用守卫

```powershell
git diff -- daily_research/output/active_execution_strategy.json
git diff --check
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.doc_guard check
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json
```

默认规则：在 `main` 分支继续；不经明确授权不修改 `daily_research/output/active_execution_strategy.json`；不把 loose `latest`、smoke、dry-run 或失败产物当成正式证据。
