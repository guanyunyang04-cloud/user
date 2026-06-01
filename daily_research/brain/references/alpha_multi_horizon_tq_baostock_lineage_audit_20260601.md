# Alpha Multi Horizon TQ vs BaoStock Lineage Audit - 2026-06-01

## Facts

- Run tag: `tq_baostock_lineage_audit_20260601_01`.
- Research program: `alpha_multi_horizon_utility_policy_v1`.
- Dataset: `policy_input_bundle__45e3d8c059ba718426a9f887`.
- Corrected pool: `policy_pool_view__74f45f4f83263bccd64a8027`.
- Legacy source candidate: `H:\new_tdx64\PYPlugins\user\t0_project\tqcenter.py`.
- Output root: `daily_research/output/path_policy/studies/tq_baostock_lineage_audit_20260601_01`.
- Scope: research-only / shadow-only lineage audit; no model training; no execution promotion.
- Active artifact impact: none. `daily_research/output/active_execution_strategy.json` remains untouched.

## Artifacts

- `legacy156_feature_recovery_report.json`
- `legacy156_feature_recovery_report.md`
- `tq_baostock_ohlcv_diff.csv`
- `tq_baostock_ohlcv_diff_summary.json`
- `tq_baostock_next_open_label_diff.csv`
- `tq_baostock_next_open_label_diff_summary.json`
- `tq_baostock_lineage_audit.json`
- `tq_baostock_lineage_audit.md`

## Legacy 156 Feature Recovery

- Status: `legacy156_not_recovered`.
- Current manifest: `daily_research/output/path_policy/studies/mh_rebuild_mainboard_target_norm_head_constraint_raw_seed7_20260601_01/forecast_dataset_manifest.json`.
- Current feature store shape: `[1699,2596,116]`.
- Search roots:
  - `daily_research/output`
  - `daily_research/cache`
  - `daily_research/archive`
  - `daily_research/brain/references`
  - `H:\new_tdx64\PYPlugins\user\t0_project`
  - git manifest history for `forecast_dataset_manifest.json`
- Candidate count: `0`.
- Decision: `legacy156_profile_allowed=false`; keep `feature_schema_shift=unresolved_root_cause`.
- Boundary: loose prose mentioning `156` features is insufficient; do not fabricate a `legacy156` profile.

## TQ vs BaoStock Diff

- TQ status: `blocked_tq_unavailable`.
- TQ error: `TQ数据接口初始化失败`.
- Sample symbol count prepared from corrected mainboard pool: `120`.
- OHLCV diff status: `blocked_tq_unavailable`.
- Next-open label diff status: `blocked_tq_unavailable`.
- Replay verdict: `old_stage28_still_text_only`.

## Interpretation

- The audit harness now exists and can compare TQ `none/front` daily bars against the BaoStock-first lake once TQ initialization is available.
- This run did not prove BaoStock data is equivalent or non-equivalent to TQ, because the legacy TQ interface did not initialize.
- The old `156` feature schema remains unrecovered after file-backed search, so the corrected new `[1699,2596,116]` baseline still cannot be treated as an old Stage 2.8 payload replay.
- Old Stage 2.8 remains brain-confirmed historical evidence, not current file-backed replay evidence.

## Next Allowed Actions

- Fix or document the TQ runtime initialization path for read-only data extraction only.
- Rerun `tq_baostock_lineage_audit` after TQ can return `get_market_data` for daily bars.
- Continue searching external backups for a traceable `forecast_dataset_manifest.json` or complete `156` feature column list.
- Keep execution frozen; do not promote, rebuild production root, or write active execution strategy from this audit.
