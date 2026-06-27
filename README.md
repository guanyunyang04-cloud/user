# 工作区入口

Canonical brain source: `brain/master_brain.md`.


本 README 只作为 agent 和维护者的第一眼索引。权威接管真源是主脑 `brain/`，项目事实由主脑路由后的分脑维护。

当前工作区根目录：`H:\quant_project`。旧通达信插件目录 `H:\new_tdx64\PYPlugins\user` 不再承载本项目代码、脑区或研究产物。

## Agent 接管入口

默认先理解用户目标、当前 git 状态和最相关的 brain / body 文件；capsule、route、bootstrap 是可选诊断入口，不是修改 tracked 文件前的固定门禁。

需要机器可读接管摘要、守卫提示或跨 agent 交接时，可运行：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<task>" --json
```

只需要辅助判断任务归属时，可运行：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow route --task "<task>" --json
```

需要显式展开主脑或分脑上下文时，可运行：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow bootstrap --brain <brain_id|workspace> --json
```

## 主分脑边界

- `brain/`：工作区主脑，管理接管顺序、任务路由、项目拓扑、共享对象和少数保护语义。
- `daily_research/brain/`：正式生产研究与执行主线的项目事实层。
- `quant_data_platform/brain/`：共享量化数据平台、canonical 数据基底、registry 与 memmap 治理。
- `t0_project/brain/`：盘中实验与 RL 原型事实层。
- `daily_stock_analysis-main/brain/`：独立产品项目事实层。
- `traditional_quant_research/brain/`：传统量化方法研究事实层。

## 根目录身份

- 分脑项目：`daily_research/`、`quant_data_platform/`、`t0_project/`、`daily_stock_analysis-main/`、`traditional_quant_research/`。
- 主脑基础设施：`brain/`、`tools/`。
- 共享数据资产：`canonical_data/`，仅保留 registry/memmap/data 资产，治理入口归 `quant_data_platform/brain/`。
- 待整理旧资产：`a_stock_daily_selection/`，仅保留历史输出，不承载阅读文档或项目事实。
- 缓存/依赖：`.pytest_cache/`、`.playwright-cli/`、`node_modules/`。

如果 README、AGENTS、CLAUDE、SKILL 与主脑冲突，以 `brain/` 当前文档和实际工具守卫为准；capsule / route 输出只是诊断信号。

## 常用守卫

```powershell
git diff -- daily_research/output/active_execution_strategy.json
git diff --check
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.doc_guard check
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json
```

当前对象不变量：repo-tracked mutation 默认先确认 git surface；`daily_research/output/active_execution_strategy.json` 属于 active artifact；loose `latest`、smoke、dry-run 或失败产物只保留其原始证据等级。
