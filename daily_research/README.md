# daily_research

`daily_research` 是当前工作区的正式生产研究与执行主线，负责长期研究、连续策略、默认执行、Web 控制台、维护工具和项目分脑状态。

接管必须先从工作区主脑进入：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<task>" --json
```

主脑路由到 `daily_research` 后，`daily_research/brain/` 才是本项目事实层。本 README 只作为简体中文快速索引，不替代主脑或分脑中的当前状态、规则、证据和治理判断。

## 模块地图

- `baseline/`：历史研究链路与仍受支持的交易计划管线
- `continuous_policy/`：当前连续决策策略的训练、评估、导出与协议编排
- `deep_alpha/`：更长周期的模型架构、alpha 与执行策略研究
- `execution/`：执行应用、任务运行器、Web 控制台与 production 更新入口
- `tools/`：守卫、报告、维护工具与一致性检查
- `brain/`：当前状态、长期知识、治理规则与过程记忆
- `output/`、`cache/`、`archive/`：生成产物、热缓存与冷归档

## 环境

标准环境是 `yolos`，依赖真源为 [environment.yml](/H:/new_tdx64/PYPlugins/user/daily_research/environment.yml:1)。

本地前置依赖：
- `t0_project/tqcenter.py` 是工作区内的数据依赖，不由 Conda 安装；使用 `tq` 数据源时必须存在。

## 常用入口

先由主脑 capsule 确认本任务属于 `daily_research`，再使用以下项目入口：

- 连续策略正式协议：
  `python daily_research/continuous_policy/run_continuous_policy_protocol.py ...`
- 执行应用：
  `python daily_research/execution/run_execution_app.py run --task <task-name> -- ...`
- 执行 Web 控制台：
  `python daily_research/execution/run_execution_web.py`
- 工作区维护报告：
  `python daily_research/tools/workspace_maintenance.py report`

真实运行时优先使用显式 `yolos` Python，例如：

```powershell
C:\Users\ASUS\miniconda3\envs\yolos\python.exe -X utf8 daily_research\tools\project_consistency_check.py
```

## 验证

较大改动前后建议运行：

```powershell
python -m compileall -q daily_research
C:\Users\ASUS\miniconda3\envs\yolos\python.exe -m tools.brain.integrity_check --json
C:\Users\ASUS\miniconda3\envs\yolos\python.exe -X utf8 daily_research\tools\project_consistency_check.py
C:\Users\ASUS\miniconda3\envs\yolos\python.exe -m tools.brain.doc_guard check
```

## 治理

- 先运行主脑 capsule，再按 `routing.selected_brain_id` 进入 `daily_research/brain/`。
- 工作区根 `brain/brain_manifest.json` 定义所有分脑共享的主脑合同。
- 新的文档内容必须先整合进对应 brain；README 只保留简体中文索引和公开入口。
- 生成实验产物应留在 `daily_research/output/`，需要复核或裁剪时使用 `workspace_maintenance.py`。
