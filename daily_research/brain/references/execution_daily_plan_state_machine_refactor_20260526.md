# Daily Execution 每日计划状态机重构记录 2026-05-26

## 状态

- Status: `execution_daily_plan_state_machine_refactor / completed_control_plane_refactor / strict_data_block / no promotion`.
- Scope: `daily_research.execution` daily runner、data readiness、Windows Task Scheduler adapter、runtime 隔离、Web/API 清理和执行端文档。
- Active artifact impact: `daily_research/output/active_execution_strategy.json` 在 git diff 中保持为空。
- 本记录不是 production retrain、promotion、live/default 切换、真实 broker 下单或模型质量证据。

## 结论

- 每日盘后计划的权威状态改为 daily verdict：`daily_research/output/execution_app/daily_runs/YYYYMMDD/verdict.json`。
- 每日最终状态只允许 `completed` 或 `blocked`；目标日关键行情不可用时写 `blocked:data_not_ready`，不得自动回退到上一完整交易日生成“今日计划”。
- 自动盘后执行主体改为 Windows Task Scheduler 触发 CLI；Web 进程只展示状态和提供手动触发入口。

## 根因

- 旧执行端把“控制台能运行”和“每日计划能可靠完成”混在一起，缺少强状态机。
- 自动更新曾被 runtime scheduler 配置持久设为 `enabled=false`，导致盘后窗口持续 `skipped_disabled`。
- 2026-05-26 手动刷新遇到 formal refresh `market_daily` 空数据，manifest 阻断为 `empty_canonical_market` / `coverage_below_threshold` / `required_domain_blocked:market_daily` / `missing_benchmark`。
- 真实 runtime 被测试/诊断作业污染，导致每日事实层无法直接信任。

## 修复

- 新增 `daily_research.execution.daily_plan_runner`，阶段固定为 `preflight -> data_readiness -> data_refresh -> signal_refresh -> trade_plan -> paper_reconcile -> final_verdict`。
- 新增 `daily_research.execution.data_readiness`，正式 refresh manifest 的同日阻断优先于小样本 provider 探测，避免同日 0 行 full refresh 被误判 ready。
- 新增 `daily_research.execution.scheduler_cli`，提供 `install/status/uninstall`，系统任务名为 `DailyResearchDailyPlan`。
- Web 删除 `/api/status` 热路径、Jinja fallback、`web/static` / `web/templates` 和 Web 内 scheduler loop；React build 缺失时页面返回 404。
- 新增 compact API：`/api/daily-run/status`、`/api/daily-run/latest`、`/api/daily-run/{date}`、`/api/daily-run/run`、`/api/data-readiness`、`/api/system/doctor`。
- 旧真实 runtime 已归档到 `daily_research/output/execution_app_legacy_archive_20260526_221153/`，新状态不读取旧 runtime schema。
- 测试 runtime 可通过 `configure_runtime_root()` 或 `DAILY_RESEARCH_EXECUTION_RUNTIME_ROOT` 隔离；pytest 默认不写真实 `daily_research/output/execution_app`。

## 当前 Verdict

- Verdict path: `daily_research/output/execution_app/daily_runs/20260526/verdict.json`。
- Status: `blocked:data_not_ready`。
- Target trading date: `2026-05-26`。
- Evidence manifest: `daily_research/output/research_data_lake/data_platform/runs/refresh_daily_formal_free_v3_20260526_160417/refresh_manifest.json`。
- Coverage: `row_count=0`，`expected_rows=5208`，`coverage_ratio=0.0`。
- Scheduler status: `DailyResearchDailyPlan` 当前未安装，`scheduler_cli status --json` 返回 `status=missing`。

## 验收

- Python: `pytest daily_research/execution/tests -q` 通过，106 passed。
- Python: `pytest daily_research/data_platform/tests -q` 通过，46 passed，保留 1 个 pandas FutureWarning。
- Frontend: `npm test -- --run` 通过，27 passed。
- Frontend build: `npm run build` 通过，仅 Vite chunk-size warning。
- CLI: `daily_plan_runner --mode dry-run --json` 返回 `blocked:data_not_ready` 并写 verdict。
- CLI: `scheduler_cli status --json` 返回系统任务未安装。
- CLI: `run_execution_app.py doctor --json` 返回 `status=ok`。
- Guards: `doc_guard check` 与 `integrity_check --json` 通过；integrity 仅保留既有 noncanonical brain warnings。
- Active artifact: `git diff -- daily_research/output/active_execution_strategy.json` 为空。

## 边界

- 不修改 production strategy、promotion、live/default 或 active artifact。
- 不把旧 runtime 迁移为新事实层；旧目录只作事故证据。
- 不保留旧 Web scheduler public contract；系统任务是唯一自动盘后调度主体。
- 不自动创建 stale plan；旧计划只能人工参考。
