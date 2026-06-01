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

- TQ status: `ok`.
- TQ initialization fix: `TQCenterAdapter` now calls `tq.initialize(...)` before `get_market_data`.
- TQ connection path: `H:\new_tdx64\PYPlugins\user\t0_project\tqcenter.py`.
- Sample symbol count prepared from corrected mainboard pool: `120`.
- OHLCV diff status: `material_data_source_shift`.
- Best price-label comparison dividend type: `none`.
- `none` price p95 relative diff mean: `0.0`; price mismatches exist mostly outside p95 and missing mismatch is about `3.53%`.
- `front` price p95 relative diff mean: `0.35637313122440645`; this is much worse than `none`, so the old/new close/open price lineage is likely unadjusted rather than front-adjusted.
- `Amount` differs materially: TQ/BaoStock amount relative diff p95 is about `0.9999000001`, consistent with a unit-scale mismatch rather than a next-open label break by itself.
- Next-open label diff status: `label_equivalent`.
- Max hit label flip rate: `0.0014824531454960379`.
- Future decision score correlation: `0.999750308636608`.
- Top20 hit label flip rate: `0.0`.
- Replay verdict: `old_stage28_still_text_only`.

## Interpretation

- TQ read-only access is available when initialized with the legacy script path before market reads.
- On the 120-symbol corrected-mainboard sample, BaoStock and TQ `none` OHLC prices are close enough for next-open label semantics by p95 price diff and hit-label flip criteria.
- The audit still marks OHLCV as `material_data_source_shift` because missing-rate and amount-unit differences are material, and amount-derived features may differ even when open/close labels are mostly equivalent.
- The old `156` feature schema remains unrecovered after file-backed search, so the corrected new `[1699,2596,116]` baseline still cannot be treated as an old Stage 2.8 payload replay.
- Old Stage 2.8 remains brain-confirmed historical evidence, not current file-backed replay evidence.

## Next Allowed Actions

- Investigate the `Amount` unit mismatch and whether any of the missing-rate rows correspond to suspensions, listing gaps, or fill policy differences.
- If extending the bridge, compare amount-derived feature columns separately and decide whether unit normalization is needed for a TQ-compatible feature profile.
- Continue searching external backups for a traceable `forecast_dataset_manifest.json` or complete `156` feature column list.
- Keep execution frozen; do not promote, rebuild production root, or write active execution strategy from this audit.
