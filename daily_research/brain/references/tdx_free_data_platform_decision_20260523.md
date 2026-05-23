# TDX-Free Data Platform Decision - 2026-05-23

## 状态
- Status: `data_platform / TDX-free migration / lake-first research / no strategy change`。
- Scope: `daily_research` 数据入口、每日更新流水线、正式研究数据边界。
- Active artifact impact: `daily_research/output/active_execution_strategy.json` 必须保持不变。
- 本记录不是模型有效性证据、promotion 证据、live/default 变更或自动交易实现。
- 2026-05-23 V2 successor: `daily_research/brain/references/tdx_free_data_platform_v2_20260523.md`。本文件保留为 V1 决策历史，不再代表当前 refresh 完整能力。

## 事实
- `lake` 是研究存储真源，不是在线数据源；`csv` 是导入和补洞通道，不是长期自动更新方案。
- 旧研究链路中 `prepare_policy_inputs()`、`build_research_database` 和部分 continuous/path policy CLI 曾默认或允许 `tq` / `csv`。
- `tqcenter.py` / `pytdx` / `mootdx` 属于 TDX-family，本轮已从正式 `daily_research` 研究入口中移除。
- 新增 `daily_research.data_platform`，包含 provider contract、provider manager、TDX-family guard、provider adapters、daily refresh CLI、Bronze/Silver 写入和 lake bundle 注册。
- 新入口：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.data_platform.refresh_daily --as-of-date YYYY-MM-DD --provider-plan default_free --symbols 000001.SZ,600000.SH,000300.SH --json`。
- 第一版 provider 计划以非 TDX 源为主：`eastmoney_efinance`、`akshare_eastmoney`、`baostock`、`tushare_http_optional`、`sina_tencent_realtime`；其中 `tushare_http_optional` 仅在 token 配置后启用，`sina_tencent_realtime` 当前只作为后续实时/盘后补充边界。
- 正式研究入口现在应使用 `--data-source lake --lake-dataset-id <explicit_id>`；不得依赖 loose latest 或在线抓取。

## 推断
- 继续修 TDX 主链路会把平台固定在通达信插件目录和非标准本地环境上，无法形成独立量化研究平台。
- 正确边界是 `online providers -> Bronze raw -> Silver canonical -> Gold/research lake`，研究和训练只读已落地、可审计、可复现的数据集。
- 多源冲突不能静默平均；应写 `source_conflict_report`，严重冲突阻断研究 lake 注册。
- 免费源可以作为第一版每日更新入口，但严肃实盘前仍需要单独规划券商 broker adapter、风控、合规报备、paper/live 隔离和 kill switch。

## 假设
- 第一版不启动真实训练、study、allocator、replay、live/default、promotion 或实盘下单。
- 第一版不使用 `tqcenter.py`、`pytdx`、`mootdx`。
- 第一版允许 provider adapters 懒加载第三方依赖；依赖缺失或接口失败必须进入 provider `error_report`，不能污染 Silver。
- 第一版 refresh 需要显式 `--symbols`；全 A universe provider discovery 和交易日历细化属于后续扩展。

## 实现边界
- 新 package: `daily_research/data_platform/`。
- 标准 market schema: `symbol, trade_date, open, high, low, close, volume, amount, source, adjusted_flag`。
- `ProviderManager` 汇总 provider coverage 和 error report；TDX-family provider 名默认报错。
- `refresh_daily` 生成 `refresh_run_id`、Bronze provider parquet、Silver canonical market parquet、`source_conflict_report` 和 `refresh_manifest.json`。
- coverage 不足、空 canonical market、缺 benchmark 或严重冲突时，refresh 结果为 `blocked`，不注册 research lake bundle。
- 增量 refresh 只复用同一 provider plan、benchmark、复权口径和 symbol universe 的历史 data_platform lake bundle，避免换股票池后误跳过历史日期。
- `eastmoney_efinance` adapter 会保留请求中的交易所后缀，避免指数/benchmark 代码被六位数字规则误归一到错误市场。
- 默认 `default_free` provider plan 只包含已实现的日线 provider；`sina_tencent_realtime` 暂放到显式 `default_free_with_realtime` 计划，避免默认 refresh 产生已知无效错误。
- 通过 gate 后注册 `policy_input_bundle__...`，供 `load_policy_inputs_from_lake()`、PathPolicy 和 continuous_policy 使用。
- `prepare_policy_inputs()` 默认改为 `lake`，并对 `tq/tdx/pytdx/mootdx` 报错，错误信息指向 `refresh_daily`。
- 当前正式 CLI 的 `--data-source` 收紧为 `lake`。
- Legacy `baseline.data_provider._try_import_tq()` 默认不再导入 `tqcenter.py`；只有未来显式设置 `DAILY_RESEARCH_ALLOW_TDX_FAMILY=1` 才可能进入旧 TDX-family 路径。

## 后续允许动作
- 将 `refresh_daily` 扩展为全 A universe discovery，而不是要求手动 `--symbols`。
- 增加交易日历 provider，替代第一版的 business-day 近似。
- 将 CSV import 明确接入 Bronze/Silver 补洞通道，不允许训练直接读散落 CSV。
- 为行业、概念、估值、ST/退市/停牌、涨跌停、资金/热点分域增加 provider contract 和 quality gate。
- 单独设计 paper execution 与后续 QMT/PTrade broker adapter；数据平台不承担下单职责。

## 验证
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/data_platform/tests -q`
  (`16 passed`, including provider timeout returning without waiting for a hung worker, legacy TQ import disabled by default, efinance benchmark suffix preservation, universe-aware incremental refresh, and default provider plan excluding the unfinished realtime adapter)
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/data_lake/tests/test_research_data_lake.py::ResearchDataLakeTest::test_prepare_policy_inputs_lake_uses_data_lake daily_research/data_lake/tests/test_research_data_lake.py::ResearchDataLakeTest::test_build_research_database_cli_can_run_market_only_or_skip_market -q`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/path_policy/tests/test_run_alpha_path20_protocol_entrypoint.py -q`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest tools/brain/tests/test_evidence_registry.py::BrainEvidenceRegistryTest::test_adapter_indexes_tdx_free_data_platform_decision -q`
