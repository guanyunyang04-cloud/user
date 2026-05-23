# TDX-Free Data Platform V2 - 2026-05-23

## 状态
- Status: `data_platform_v2 / TDX-free daily ingestion / lake-first research / no strategy change`.
- Scope: `daily_research.data_platform` 多 domain refresh、CSV 入湖、research lake sidecar metadata 与正式研究数据边界。
- Active artifact impact: `daily_research/output/active_execution_strategy.json` 必须保持不变。
- 本记录不是模型有效性证据、promotion 证据、自动交易实现或 live/default 变更。

## 事实
- V1 已建立非 TDX provider contract、provider manager、`refresh_daily`、Bronze/Silver、manifest 和 lake bundle 注册，但仍偏向显式 `--symbols` 与 market daily 单域。
- V2 新增通用 domain contract：`market_daily`、`trading_calendar`、`universe_snapshot`、`security_status`、`limit_status`、`industry_concept`、`valuation`、`money_flow_hotspot`。
- V2 `refresh_daily` 支持 `--universe all_a|liquid500|file:<path>|symbols:<csv>`；`--symbols` 仍保留为显式小样本/调试入口，但不再是每日更新必需条件。
- V2 增量刷新使用 provider 交易日历；`all_a` 通过非 TDX universe snapshot 生成 symbol set，不依赖 `tqcenter.py`、`pytdx` 或 `mootdx`。
- 每个 domain 都写 Bronze provider raw、Silver canonical parquet、coverage/error summary 与 source conflict report；required domain blocked 时整体 refresh 不注册 Gold/research bundle。
- `policy_input_bundle` metadata 写入 sidecar domain dataset ids、calendar dataset id、universe snapshot dataset id 与 `data_platform_refresh_run_id`。
- 新增 `daily_research.data_platform.import_csv`：CSV 必须先进入 Bronze，再经 Silver normalize / quality gate 后才能注册 lake dataset；正式训练、评估、diagnostics 不得直接读取散落 CSV。
- 第一批 provider path 仍是非 TDX：`eastmoney_efinance`、`akshare_eastmoney`、`baostock`、`tushare_http_optional`、`sina_tencent_realtime`。其中 Tushare 仅 token 存在时启用；实时 provider 不参与默认历史真源。

## 推断
- 真正摆脱通达信插件目录的关键不是把 TDX 降级为备用源，而是让 daily refresh、universe、calendar、sidecar 和 lake 注册全链路都不需要 TDX-family。
- `lake` 和散落 `csv` 不能承担在线数据源职责；它们分别是研究存储真源和导入/补洞通道。
- 研究、训练和 diagnostics 必须只读 explicit lake dataset id；在线抓取只允许发生在 data platform refresh/import 边界内。
- 自动交易的后续路线应单独接 QMT/PTrade broker adapter、风控、合规报备与 kill switch；数据平台只负责可审计数据，不承担下单职责。

## 假设
- 本轮不启动真实训练、study、allocator、replay、live/default、promotion 或实盘下单。
- 本轮不新增 TDX-family fallback，不使用 `tqcenter.py`、`pytdx`、`mootdx`、`mootdx` alias 或其他 TDX-family 在线 provider。
- optional domain 缺失不阻断 market daily 入湖，除非该 domain 被显式放入 `--required-domains`。
- 免费源优先；付费/Token 数据源只能作为 optional enhancement，不成为 V2 的硬依赖。

## 标准命令
- 全 A V2 refresh:
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.data_platform.refresh_daily --as-of-date YYYY-MM-DD --provider-plan default_free --universe all_a --domains market_daily,trading_calendar,universe_snapshot,security_status,limit_status,industry_concept,valuation --json`
- 显式 required domain:
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.data_platform.refresh_daily --as-of-date YYYY-MM-DD --provider-plan default_free --universe all_a --domains market_daily,trading_calendar,universe_snapshot,valuation --required-domains market_daily,trading_calendar,universe_snapshot --json`
- CSV 入湖:
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.data_platform.import_csv --input <csv_or_folder> --domain market_daily --as-of-date YYYY-MM-DD --source-name manual_csv --json`

## 后续允许动作
- 以真实 provider smoke 或 dry-run health check 校准各免费源的可用性、速率限制和字段稳定性。
- 扩展 `money_flow_hotspot`、行业/概念和估值字段的质量门，但仍保持 optional domain 默认不阻断 market daily。
- 让 PathPolicy / Multi Horizon Utility / continuous_policy 逐步显式读取 sidecar metadata，禁止 loose latest。
- 单独规划 `paper_execution -> broker_adapter -> QMT/PTrade` 自动交易主线；不要把数据平台结果直接解释为可实盘交易能力。

## 禁止动作
- 不允许正式研究 CLI 传 `tq/tdx/pytdx/mootdx`。
- 不允许训练、评估、diagnostics 临时在线抓取行情。
- 不允许用散落 CSV 直接训练；必须先 import 到 Bronze/Silver/lake。
- 不允许把 provider health、refresh smoke 或 CSV import 记录为模型有效性证据。

## 验证
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/data_platform/tests -q`
  - Result: `24 passed`.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/data_lake/tests/test_research_data_lake.py -q`
  - Result: `25 passed`.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/path_policy/tests/test_run_alpha_path20_protocol_entrypoint.py -q`
  - Result: `2 passed`.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests/test_lake_evaluator_cli_contract.py -q`
  - Result: `3 passed`.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest tools/brain/tests/test_evidence_registry.py -q`
  - Result: `15 passed`.
- `git diff -- daily_research/output/active_execution_strategy.json`
  - Result: empty diff.
- `git diff --check`
  - Result: pass; PowerShell emitted a CRLF normalization warning for `tools/brain/tests/test_evidence_registry.py`, but no whitespace error.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.doc_guard check`
  - Result: `brain-integrity errors=0 warnings=3` from existing noncanonical brain/catalog warnings.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json`
  - Result: `status=ok`, `error_count=0`, `warning_count=3`.
