# Daily Research V2 Traditional PIT Bridge - 2026-06-02

## Summary

- Status: `traditional_pit_bridge_absorbed / evidence_grade / research-only / execution-frozen`.
- Research program: `daily_research_v2_research_reset`.
- Study family: `v2_traditional_pit_tradeable_mainboard_baseline`.
- Source market dataset: `policy_input_bundle__45e3d8c059ba718426a9f887`.
- Traditional PIT source snapshot: `baostock_v2_pit_20160101_20260601_stockbasic_fixed`.
- Imported status sidecar: `data_platform_v2_status_sidecar__37dba59cdced261cddfedf11`.
- Traditional-PIT strict pool: `policy_pool_view__0af1d96b413fb8927270b680`.
- Anchor: `mh_v2_traditional_pit_tradeable_mainboard_anchor_20260602_01`.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` unchanged.

## What Changed

- Added a daily_research bridge CLI that imports the already-built `traditional_quant_research` PIT daily universe as a v2 status sidecar:
  - `python -m daily_research.data_lake.import_traditional_pit_status_sidecar --json`
- Added a traditional-PIT-backed v2 baseline wrapper:
  - `python -m daily_research.path_policy.v2_traditional_pit_baseline --write-task-list --validate-pool --json`
  - `python -m daily_research.path_policy.v2_traditional_pit_baseline --run-training --json`
  - `python -m daily_research.path_policy.v2_traditional_pit_baseline --run-comparison --json`
- The bridge is read-only with respect to `traditional_quant_research`; it maps `daily_universe.parquet` into daily_research's `v2_status_sidecar` contract and binds the result to the explicit daily_research market dataset id.

## Imported Status Sidecar

- Dataset id: `data_platform_v2_status_sidecar__37dba59cdced261cddfedf11`.
- Rows: `4,970,903`.
- Trade dates: `1699`.
- Symbols: `3175`.
- Tradeable rows: `4,780,834`.
- ST rows: `159,053`.
- Suspended rows: `35,638`.
- Reject reason counts:
  - accepted: `4,780,834`;
  - `st_on_date`: `159,053`;
  - `suspended_on_date`: `31,016`.
- Source snapshot quality:
  - failure count `0`;
  - source error count `0`;
  - original PIT snapshot covers `2016-01-04 -> 2026-06-01`.

## Strict Pool Validation

- Pool id: `policy_pool_view__0af1d96b413fb8927270b680`.
- Validation status: `ok`.
- Active universe size: `2602`.
- Membership symbols: `3175`.
- Daily member count min/median/max: `487 / 500 / 500`.
- Shortfall days: `781`, diagnostic only.
- Active excluded prefixes `300/301/688/689`: all `0`.
- Active ST/suspended/delisted/not-listed/missing-bar/not-tradeable rows: all `0`.
- Overlap vs current v2 strict pool `policy_pool_view__925e8604a91a9c07a5387fb1`:
  - baseline true cells `847,789`;
  - traditional-PIT true cells `847,580`;
  - overlap true cells `845,890`;
  - removed from baseline `1,899`;
  - added by traditional PIT `1,690`;
  - daily overlap ratio mean `0.9957800975067354`.

## Three-Seed Baseline Result

- Training status: `completed`.
- Failed tags: `[]`.
- Seed tags:
  - `mh_v2_traditional_pit_tradeable_mainboard_seed7_20260602_01`;
  - `mh_v2_traditional_pit_tradeable_mainboard_seed11_20260602_01`;
  - `mh_v2_traditional_pit_tradeable_mainboard_seed19_20260602_01`.
- Seed7 manifest validation:
  - status `ok`;
  - feature count `124`;
  - alpha-like feature count `0`;
  - amount unit policy `as_is`;
  - amount unit factor `1.0`.
- Aggregate gate: `near_pass`, not full pass.
- Aggregate test row:
  - seed count `3`;
  - rank IC mean/min `0.098312 / 0.070197`;
  - spread mean/min `0.036523 / 0.022745`;
  - hit lift mean/min `0.014643 / 0.005537`;
  - monthly positive rate mean/min `0.818182 / 0.727273`;
  - negative month count max `3`;
  - 30d concentration mean `0.584592`.
- Failed gate check:
  - `negative_month_count_max_le_2 = false`.

## Interpretation

- The traditional PIT dataset has been successfully absorbed into daily_research as a stricter status/pool contract.
- This improves data-contract confidence: historical ST and suspension flags are materially more complete than the prior v2 sidecar.
- The resulting strict pool is very close to the existing v2 strict pool but not identical; differences average around one name per day.
- The model result is healthy but does not beat the existing v2 strict baseline gate because it is `near_pass` rather than `pass`.
- Current v2 anchor remains `mh_v2_reset_tradeable_mainboard_anchor_20260601_01`; the traditional-PIT anchor is a high-quality bridge/comparison baseline, not the new default.

## Boundaries

- Do not promote to live/default from this result.
- Do not write or rebuild `daily_research/output/active_execution_strategy.json`.
- Do not mutate `traditional_quant_research` from the bridge; read its PIT snapshot only.
- Keep execution side `frozen_skeleton_only / awaiting_research_rebuild`.
- Future v2 research should prefer the traditional-PIT sidecar for stricter data-contract experiments, but current model-quality baseline comparisons must explicitly state whether they use the original v2 sidecar or the traditional-PIT sidecar.

## Verification

- `python -m pytest daily_research/data_lake/tests/test_import_traditional_pit_status_sidecar.py daily_research/path_policy/tests/test_v2_traditional_pit_baseline.py -q`: `7 passed`.
- `python -m daily_research.data_lake.import_traditional_pit_status_sidecar --json`: completed.
- `python -m daily_research.path_policy.v2_traditional_pit_baseline --write-task-list --validate-pool --json`: completed, pool validation `ok`.
- `python -m daily_research.path_policy.v2_traditional_pit_baseline --run-training --json`: completed, failed tags `[]`.
- `python -m daily_research.path_policy.v2_traditional_pit_baseline --run-comparison --json`: completed, gate `near_pass`.

## Next Allowed Actions

- Keep `mh_v2_reset_tradeable_mainboard_anchor_20260601_01` as the current pass-grade v2 baseline.
- Use `mh_v2_traditional_pit_tradeable_mainboard_anchor_20260602_01` as the stricter data-contract comparison baseline.
- Investigate the one failed gate dimension, `negative_month_count_max=3`, before replacing the current v2 anchor.
- Continue feature/model improvements with explicit pool/sidecar IDs and compare against both v2 strict and traditional-PIT strict baselines.
