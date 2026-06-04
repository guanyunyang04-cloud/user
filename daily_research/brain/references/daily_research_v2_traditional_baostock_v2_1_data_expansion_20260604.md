# Daily Research Traditional Baostock V2.1 Data Expansion

Date: `2026-06-04`

## Summary

- Status: `traditional_baostock_v2_1_data_expansion / import_complete / pool_contract_ok / smoke_memmap_ok / research-only / execution-frozen`.
- Research program: `daily_research_v2_research_reset`.
- Study family: `v2_traditional_baostock_v2_1_data_expansion`.
- Source project: `traditional_quant_research`.
- Source snapshot: `baostock_v2_pit_20160101_20260601_industry_metrics_month_start_20260603`.
- Imported daily_research dataset id: `policy_input_bundle__2082fee5bb1760972d8c9012`.
- New feature profile: `raw_kline_context_v2_tradeable_local_state_industry_metrics_v1`.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` unchanged.
- This is not model-quality evidence. It completes the data import, pool/view contracts, audit repair, and smoke memmap validation needed before formal same-period and long-history forecast A/B.

## Implementation

- Added research-only importer CLI:
  - `daily_research.data_lake.import_traditional_baostock_v2_snapshot`.
- Imported Traditional Baostock v2.1 tables into the daily_research lake instead of training directly from the Traditional project path:
  - `daily_universe.parquet`;
  - `daily_bars.parquet`;
  - `daily_status.parquet`;
  - `stock_industry.parquet`;
  - `daily_metrics.parquet`.
- Standardized key contract from `date/code` to `trade_date/symbol`; code format remains `000001.SZ` / `600000.SH`.
- Added bundle-bound sidecar loading in `daily_research.data_lake.policy_input_loader`.
- Hardened `v2_dataset_contract_audit` to prefer explicit bundle-bound `v2_status_sidecar` over old global `security_status`, avoiding false ST/status degradation for imported v2.1 snapshots.
- Added feature profile `raw_kline_context_v2_tradeable_local_state_industry_metrics_v1`, extending the local-state profile with:
  - `sector_context`;
  - `sector_relative_context`;
  - `turnover_context`;
  - `valuation_context`.
- Fixed feature-column cap behavior so augmented profiles preserve `sector_context` under the 192-column cap.

## Import Evidence

- Run tag: `daily_research_traditional_baostock_v2_1_import_20260604_01`.
- Import manifest: `daily_research/output/data_lake/imports/daily_research_traditional_baostock_v2_1_import_20260604_01/import_manifest.json`.
- Imported dataset id: `policy_input_bundle__2082fee5bb1760972d8c9012`.
- Source manifest fingerprint: `4582f2f68014651524b7068fb84fa57c3f309479c877da8583a905bd55bf7744`.
- Source quality fingerprint: `e96f5587a3e0a548532f6caf3f77122c4e976ac8c41039c56d3410cfbc3e45bd`.
- Date range: `2016-01-04` to `2026-06-01`.
- Symbol count: `3393`.
- Trade date count: `2526`.
- Market rows: `7,451,610`.
- Tradeable rows: `7,031,085`.
- Sidecar ids:
  - universe snapshot: `data_platform_universe_snapshot__9042cbf5bc4e75750e187da9`;
  - v2 status sidecar: `data_platform_v2_status_sidecar__362fde0f9394e00c1ff9d62b`;
  - industry sidecar: `data_platform_industry_concept__b4efea14aea248d56fbb2e76`;
  - valuation sidecar: `data_platform_valuation__0834d9bf877718c7c4975c18`.
- Industry frequency: `month-start-ffill`.
- Industry PIT semantics: `approximate_pit_month_start_forward_fill_not_exact_daily_industry_change`.
- Metrics coverage on tradeable rows: `turn/pctChg/peTTM/pbMRQ/psTTM/pcfNcfTTM = 1.0`.
- Industry coverage on tradeable rows: `0.9972726826656199`.
- Valuation lag policy: `peTTM/pbMRQ/psTTM/pcfNcfTTM` are feature-profile lagged by at least one trading day before model use.
- Benchmark caveat: benchmark is inherited from the current v2 baseline and covers `2017-01-03` to `2024-12-31`; formal long-history forecast should therefore start train evidence from `2017`, not `2016`.

## Pool/View Contracts

Same-period formal A/B pool:

- Pool id: `policy_pool_view__d7a56d5164470b590e4f5a40`.
- View kind: `rolling_liquidity_tradeable_mainboard`.
- Date range: `2018-01-02` to `2024-12-31`.
- Membership rows: `1699`.
- True cells: `847,281`.
- Universe size: `2693`.
- Member count min/median/max: `484 / 500 / 500`.
- Zero-member days: `0`.
- Audit: `v2_dataset_contract_ok`.
- Audit artifact: `daily_research/output/data_lake/audits/v2_dataset_contract_audit_traditional_baostock_v2_1_same_period_2018_2024_20260604_01/v2_dataset_contract_audit.json`.

Long-history formal A/B pool:

- Pool id: `policy_pool_view__eb690dd0becc330f029c21bd`.
- View kind: `rolling_liquidity_tradeable_mainboard`.
- Date range: `2016-01-04` to `2024-12-31`.
- Membership rows: `2187`.
- True cells: `1,087,460`.
- Universe size: `3004`.
- Member count min/median/max: `474 / 499 / 500`.
- Zero-member days: `0`.
- Audit: `v2_dataset_contract_ok`.
- Audit artifact: `daily_research/output/data_lake/audits/v2_dataset_contract_audit_traditional_baostock_v2_1_long_history_2016_2024_20260604_01/v2_dataset_contract_audit.json`.

Smoke-only static pool:

- Pool id: `policy_pool_view__04f11f51d7c0c7ae32038147`.
- View kind: `static_symbols`.
- Purpose: pipeline smoke only, not model evidence.
- Date range: `2018-01-02` to `2024-12-31`.
- Universe size: `32`.
- True cells: `54,368`.

## Feature And Memmap Smoke

- Smoke feature audit artifact: `daily_research/output/path_policy/data_expansion/traditional_baostock_v2_1_augmented_feature_smoke32_20260604_01/feature_profile_audit_summary.json`.
- Smoke feature audit status: `ok`.
- Smoke memmap artifact: `daily_research/output/path_policy/data_expansion/traditional_baostock_v2_1_augmented_memmap_smoke32_20260604_01/forecast_dataset_manifest.json`.
- Smoke memmap validation: `daily_research/output/path_policy/data_expansion/traditional_baostock_v2_1_augmented_memmap_smoke32_20260604_01/memmap_validation.json`.
- Smoke memmap status: `completed`; validation status: `ok`.
- Source pool: `policy_pool_view__04f11f51d7c0c7ae32038147`.
- Sample count: `288`, with train/validation/test each `96`.
- Feature store shape: `[1699, 32, 190]`.
- Feature group counts after the cap fix:
  - `sector_context=2`;
  - `sector_relative_context=10`;
  - `turnover_context=7`;
  - `valuation_context=28`;
  - `local_state_context=20`;
  - `raw_kline=15`;
  - `market_context=26`;
  - `peer_context=12`.
- Static context is enabled; vocab sizes are symbol `33`, exchange `3`, industry `19`, liquidity `6`, and price `6`.
- Role years for same-period smoke: train `2019-2022`, validation `2023`, test `2024`, purge `31` trading days.

## Verification

- Importer / audit / feature tests:
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/data_lake/tests/test_import_traditional_baostock_v2_snapshot.py daily_research/data_lake/tests/test_v2_dataset_contract_audit.py daily_research/path_policy/tests/test_v2_feature_profile_amount_checked.py -q`: `21 passed`.
- Research data lake tests:
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/data_lake/tests/test_research_data_lake.py -q`: `32 passed`.
- Targeted memmap contract tests:
  - `test_forecast_memmap_dataset_builds_lazy_store_and_batches`;
  - `test_validate_forecast_memmap_manifest_reports_ok_and_source_binding`;
  - `test_static_context_vocab_is_stable_and_memmap_samples_are_aligned`;
  - result: `3 passed`.
- Targeted forecast training compatibility tests:
  - `test_train_forecast_models_static_context_checkpoint_contract`;
  - `test_stock_mixer_uses_memmap_date_level_batching`;
  - `test_train_forecast_models_accepts_memmap_dataset_view`;
  - result: `3 passed`.
- Traditional source compatibility tests:
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest traditional_quant_research/tests/test_dataset_v2.py traditional_quant_research/tests/test_v2_daily_metrics_audit.py -q`: `4 passed`.
- Active artifact guard:
  - `git diff -- daily_research/output/active_execution_strategy.json`: clean.

## Non-Evidence And Caveats

- A full formal-pool same-period feature audit exceeded the local timeout and the residual Python process was stopped. It is a resource/throughput diagnostic only, not failed model-quality evidence.
- A combined large pytest command exceeded the local timeout and the residual process was stopped; targeted subsets above passed.
- The full `daily_research/path_policy/tests/test_forecast_memmap_dataset.py` command exceeded a 10-minute timeout; targeted memmap contract checks passed.
- The full `daily_research/path_policy/tests/test_forecast_training.py` command was not rerun after the earlier combined timeout; targeted compatibility checks passed.
- Smoke memmap used the 32-symbol static pool and must not be promoted to completed forecast evidence.
- No forecast model, score-backtest bridge, candidate matrix, or bad-month attribution has been completed on this augmented dataset yet.
- No completed model-quality evidence should be registered from failed, timeout, smoke-only, single-seed, or shadow-only outputs.

## Verdict

- The Traditional Baostock v2.1 data should be used, but only through explicit daily_research lake lineage. That is now implemented.
- Same-period and long-history A/B datasets are contract-ready at the pool/audit level.
- The augmented feature profile and sidecar contracts are pipeline-valid on a smoke pool, including industry, sector-relative, turnover, and lagged valuation fields.
- The project has not yet proven that data width or longer history improves forecast quality or forecast-to-portfolio transfer.
- Current blocker is `formal_augmented_memmap_and_three_seed_forecast_ab_not_completed`.

## Next Allowed Actions

- Build the full same-period augmented memmap with pool `policy_pool_view__d7a56d5164470b590e4f5a40` and feature profile `raw_kline_context_v2_tradeable_local_state_industry_metrics_v1`.
- Run a smoke seed first and label it explicitly `smoke_only`.
- If the full same-period pipeline is healthy, run three formal GRU-static seeds `7,11,19` against the current baseline years: train `2019-2022`, validation `2023`, test `2024`.
- Build the long-history augmented memmap with pool `policy_pool_view__eb690dd0becc330f029c21bd`; formal long-history forecast should use train `2017-2022`, validation `2023`, test `2024`.
- Only if three-seed augmented forecast is not worse than the baseline should it enter `v2_score_backtest_bridge`, `v2_candidate_review_matrix`, and `v2_bad_month_attribution`.
- Only after GRU-static proves data benefit should larger architecture runs be repeated on the augmented data.
- Do not modify active execution strategy, production root, paper/live, broker, or promotion artifacts.
