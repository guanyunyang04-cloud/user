# 全仓主分脑与代码维护审阅报告

日期：`2026-05-11`

## 1. 本轮目标
- 维护主脑与三个分脑的边界：主脑只保留跨项目规则和路由，分脑只保留项目事实与执行入口。
- 保留“任务分配不依赖 Codex 线程 URI”的规则，并把它落实为 brain 文档守卫。
- 纠偏 `daily_research` r52d 状态：代码合同、测试、dry-run 与 safe screening-only 已存在，但没有 confirmatory / stable verdict。
- 在不改 live/default/promotion、不改 `daily_research/output/active_execution_strategy.json` 的前提下，审阅并验证 brain tooling、研究代码、产品代码和实验代码。

## 2. 已完成修改
- `doc_guard.py` 新增 brain 文档运行时线程 URI 禁止规则；同时允许 `.pytest_cache/README.md` 不被误判为项目文档。
- `brain_workflow health` 改为并行运行 `brain_integrity`、`doc_guard`、`project_consistency`、`openmp_strict`，并输出每项耗时与总耗时。
- 新增/更新单测覆盖 health 并行聚合、每项耗时输出、brain 文档线程 URI 禁止规则与 pytest cache 例外。
- `daily_research` 分脑已写回 r52d 当前状态：profile、v40 loss、合同测试、dry-run 与 safe screening-only 证据存在，但尚无 confirmatory / stable 策略结论。
- `daily_research` 已补充 r52d explicit evidence capsule，并裁决 `r52d screening-only failed confirmatory eligibility`；本轮不启动 confirmatory / resume / promotion。
- `daily_stock_analysis-main` 修正 Windows 后端构建脚本、后端验证配置、Web ESLint flat config、Web smoke 认证边界和 Windows UTF-8 源码测试。
- `t0_project` 只做离线静态验收并写回实验隔离边界。

## 3. 确认修复的问题
- health 串行执行四项检查，容易在外层短窗口中误超时；已改为并行聚合。
- brain 文档缺少运行时线程 URI 的机器守卫；已加入 `doc_guard.py check`。
- pytest 生成的 `.pytest_cache/README.md` 会被文档布局守卫误判为外部项目文档；已加入明确例外。
- 分脑把 r52d 主要写成待设计状态，与代码中已有 r52d profile/tests 的事实不一致；已纠偏。
- `brain_workflow health` 的 OpenMP strict lane 曾继承父进程 `KMP_DUPLICATE_LIB_OK`，在连续策略测试导入后会误失败；现已对 strict lane 单独清理环境。
- DSA Windows 后端构建曾因 GBK 解码 UTF-8 requirements、以及 PyInstaller 扫入无关 torch hook 失败；已修复并复验通过。
- DSA 本地验证曾因 flake8 / ESLint 扫描生成目录、以及源码静态测试按 GBK 读 UTF-8 文件而失败；已修正。
- Web smoke 曾在 `ADMIN_AUTH_ENABLED=false` 时错误期待登录页；已改为先读认证状态。

## 4. 未修复或需决策问题
- r52d 已完成 safe screening-only 裁决：3/3 trials 均 evidence insufficient、composite<0、cash timing<0、exposure~0.33，因此不启动 confirmatory；仍缺的是下一轮 deployment / cash timing / exposure utilization / training evidence 修复方案。
- `bash scripts/ci_gate.sh all` 在当前 Windows 环境会进入 WSL bash，WSL 内没有 `python`，因此该入口未通过；PowerShell 等价分项已通过。
- DSA 桌面构建仍受 Windows Developer Mode / electron-builder symlink 与网络下载影响，未形成通过结论。
- Web `npm audit` 仍提示依赖漏洞，Vite build 仍提示主 chunk 偏大；需要产品/依赖升级决策。
- 超大文件拆分未作为本轮默认目标；除非发现可保持行为的低风险切分点，否则只记录遗留问题。

## 5. 验证记录
- Brain 守卫通过：
  - `brain_integrity_check.py --json`
  - `doc_guard.py check`
  - `project_consistency_check.py`
  - `openmp_runtime_check.py --strict`
  - `brain_workflow health --json`
  - 全仓线程 URI 搜索无命中
- `daily_research` 通过：
  - `pytest daily_research/tools/tests daily_research/continuous_policy/tests -q`：`126 passed`
  - r52d dry-run：`self_opt_study_r52d_validation_closure_dryrun_20260511_01` 通过
  - r52d safe screening-only：`self_opt_study_r52d_native_validation_closure_screening_safe_20260511_01` 完成 3/3 trials、0 failed、confirmatory disabled、true solver disabled
  - r52d explicit evidence verdict：`r52d screening-only failed confirmatory eligibility`
  - r52d evidence capsule：`daily_research/brain/references/r52d_native_validation_closure_status_20260511.md`
- `daily_stock_analysis-main` 通过：
  - `python scripts/check_ai_assets.py`
  - backend `py_compile`
  - `python -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics`
  - `./test.sh code`、`./test.sh yfinance`
  - `python -m pytest -m "not network"`：`1150 passed`
  - Web `npm ci`、`npm run lint`、`npm run test`、`npm run build`、`npm run test:smoke`；smoke 结果 `5 passed / 5 skipped`
  - `powershell -ExecutionPolicy Bypass -File scripts/build-backend.ps1`
- `t0_project` 通过：
  - `python -m py_compile` 覆盖全部 `t0_project/**/*.py`
- 全仓收尾：
  - 全仓运行时线程 URI 搜索：无匹配
  - `git diff --check`：通过
  - `git status --short --branch`：当前仅保留本轮 `daily_research` brain/reference 与维护报告写回改动；未提交、未推送

## 6. 回滚建议
- 若 health 并行聚合出现平台兼容问题，回滚 `daily_research/tools/brain_platform.py` 的 `ThreadPoolExecutor` 调度改动，并保留 `_run_check` 耗时字段。
- 若线程 URI 守卫误伤，应优先缩小扫描范围到 `BRAIN_DOC_PREFIXES` 下的 `.md/.json`，不要删除规则本身。
- 若 r52d 后续 confirmatory 或 stability 失败，只回写分脑状态与报告，不回滚 r52d 代码合同，除非测试证明合同本身错误。
- 若 DSA smoke helper 在认证开启环境异常，优先复核 `/api/v1/auth/status` 与 `DSA_WEB_SMOKE_PASSWORD`，不要退回固定登录页假设。
- 若桌面构建仍失败，优先在启用 Developer Mode 或 CI Windows runner 上复验 electron-builder 缓存解压，再判断是否需要脚本级修正。
