# Workspace Hot / Cold Index 2026-05-16

## 当前目的

本索引用于降低后续接管复杂度。它只说明热区、冷区和证据入口，不替代 `state_center.md`、`knowledge_center.md` 或 explicit evidence tag。

## 热区

- `daily_research/output/active_execution_strategy.json`：当前 live/default 物化真源；不得归档、删除或静默修改。
- `daily_research/output/short_expert_policy_v5b_execalign_production_default/`：当前 production root。
- `daily_research/output/continuous_policy/`：continuous_policy research / shadow evidence 热区；`latest_*` 只作便利入口，正式结论必须使用 explicit study tag、protocol tag 或 dataset id。
- `daily_research/output/research_data_lake/`：DuckDB + Parquet data lake 热区；strict Gold 与 lake bundle 证据从这里追溯。
- `daily_research/cache/industry_map_tq.csv`、`style_map_tq.csv`、`stock_name_map_tq.csv`、`universe_all_a_tq.csv`：小型共享映射缓存，默认保留。

## 冷区

- `daily_research/archive/output/`：过期 output run 和入口级归档。
- `daily_research/archive/cache/`：可重建或较旧的缓存 payload。
- `daily_research/archive/manifests/`：归档账本；manifest 是追溯移动来源的首选入口。

## 最新归档批次

- 批次：`handoff_simplification_20260516_01`。
- Manifest：`daily_research/archive/manifests/archive_handoff_simplification_20260516_01.json`。
- 移动规模：`109` 项，约 `21.07GB`。
- 规则分布：`advanced_ml_prepared_cache=47`，`advanced_ml_raw_cache=51`，`deep_alpha_features_cache=4`，`deep_alpha_raw_cache=4`，`output_runs=3`。

## 禁止事项

- 不从 cold archive 直接推出策略结论。
- 不把 `latest_*` 当作无条件真源。
- 不归档或删除当前 active manifest、production root、continuous_policy 热区、research data lake 热区。
- 不运行 `clean --apply` 或 `prune-archive --apply`，除非另起明确维护任务并先 dry-run。

## 后续维护节奏

- 常规接管先跑 `brain_workflow capsule`，再看本索引。
- 需要释放空间时先跑 `workspace_maintenance.py report` 与 `archive` dry-run。
- 只有 dry-run 候选不包含热区和 active artifact 时，才允许执行 `archive --apply`。
