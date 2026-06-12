# daily_research

Canonical brain source: `daily_research/brain/identity_layer.md`.


`daily_research` 是当前工作区的正式生产研究与执行主线，负责长期研究、连续策略、默认执行、Web 控制台、维护工具和项目分脑状态。

当前工作区根目录：`H:\quant_project`。`daily_research` 不再运行在 `H:\new_tdx64\PYPlugins\user` 下，旧通达信插件用户目录不应保存本项目代码或产物。

接管必须先从工作区主脑进入：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<task>" --json
```

主脑路由到 `daily_research` 后，`daily_research/brain/` 才是本项目事实层。本 README 只作为简体中文快速索引，不替代主脑或分脑中的当前状态、规则、证据和治理判断。

## 模块地图

- `baseline/`：历史研究链路与仍受支持的交易计划管线
- `continuous_policy/`：当前连续决策策略的训练、评估、导出与协议编排
- `path_policy/`：多 Horizon 交易效用排序、历史 Path20 预测/RL 与 shadow 对照研究
- `data_lake/`：DuckDB/Parquet 研究数据集、显式 dataset id、pool view 与 sector/board view
- `data_platform/`：TDX-free 在线 provider、每日 refresh、Bronze/Silver 仲裁与 lake 注册入口
- `deep_alpha/`：更长周期的模型架构、alpha 与执行策略研究
- `execution/`：执行应用、任务运行器、Web 控制台与 production 更新入口
- `tools/`：守卫、报告、维护工具与一致性检查
- `brain/`：当前状态、长期知识、治理规则与过程记忆
- `output/`、`cache/`、`archive/`：生成产物、热缓存与冷归档

## 环境

标准环境是 `yolos`，依赖真源为 [environment.yml](daily_research/environment.yml)。

本地数据边界：
- `daily_research` 正式研究链路不再依赖 `t0_project/tqcenter.py`、`pytdx` 或 `mootdx`。
- `lake` 是研究存储真源；在线 provider 每日更新必须先进入 Bronze/Silver，再注册为显式 lake dataset id。
- `csv` 只作为入湖导入或补洞通道，不允许被正式训练/评估直接读取。
- V2 数据平台支持 `--universe all_a|liquid500|file:<path>|symbols:<csv>`；`--symbols` 只保留为小样本/显式调试入口。

## 常用入口

先由主脑 capsule 确认本任务属于 `daily_research`，再使用以下项目入口：

- 连续策略正式协议：
  `python daily_research/continuous_policy/run_continuous_policy_protocol.py ...`
- TDX-free V2 每日数据刷新：
  `python -m daily_research.data_platform.refresh_daily --as-of-date YYYY-MM-DD --provider-plan default_free --universe all_a --domains market_daily,trading_calendar,universe_snapshot,security_status,limit_status,industry_concept,valuation --json`
- CSV 入湖导入：
  `python -m daily_research.data_platform.import_csv --input <csv_or_folder> --domain market_daily --as-of-date YYYY-MM-DD --source-name manual_csv --json`
- 执行应用：
  `python daily_research/execution/run_execution_app.py run --task <task-name> -- ...`
- 执行 Web 控制台：
  `python daily_research/execution/run_execution_web.py`
- 工作区维护报告：
  `python daily_research/tools/workspace_maintenance.py report`

真实运行时优先使用显式 `yolos` Python，例如：

```powershell
C:\Users\ASUS\miniconda3\envs\yolos\python.exe -X utf8 daily_research\tools\project_consistency_check.py --mode research
```

## 验证

较大改动前后建议运行：

```powershell
python -m compileall -q daily_research
C:\Users\ASUS\miniconda3\envs\yolos\python.exe -m tools.brain.integrity_check --json
C:\Users\ASUS\miniconda3\envs\yolos\python.exe -X utf8 daily_research\tools\project_consistency_check.py --mode research
C:\Users\ASUS\miniconda3\envs\yolos\python.exe -m tools.brain.doc_guard check --scope changed
```

执行端解冻、active artifact 或 live-facing 合同变更时，再显式运行 `project_consistency_check.py --mode execution` 或 `--mode full` 与裸 `doc_guard check`。

## 治理

- 先运行主脑 capsule，再按 `routing.selected_brain_id` 进入 `daily_research/brain/`。
- 工作区根 `brain/brain_manifest.json` 定义所有分脑共享的主脑合同。
- 新的文档内容必须先整合进对应 brain；README 只保留简体中文索引和公开入口。
- 生成实验产物应留在 `daily_research/output/`，需要复核或裁剪时使用 `workspace_maintenance.py`。
