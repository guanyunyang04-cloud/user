# Daily Research V2 Dataset Contract Upgrade - 2026-06-02

## Summary

- Status: `daily_research_v2_dataset_contract_upgrade / evidence_grade / research-only / execution-frozen`.
- Research program: `daily_research_v2_research_reset`.
- Study family: `v2_tradeable_mainboard_baseline`.
- Dataset id: `policy_input_bundle__45e3d8c059ba718426a9f887`.
- Status sidecar id: `data_platform_v2_status_sidecar__b896a110cf7802cc1653c22a`.
- Strict pool id: `policy_pool_view__925e8604a91a9c07a5387fb1`.
- Feature profile: `raw_kline_context_v2_tradeable_amount_checked`.
- Model/loss: `gru_sequence_static_context` + `target_norm_head_constraint_v1`.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` unchanged.

## Implemented Contract

- Added v2 dataset contract audit:
  - `python -m daily_research.data_lake.v2_dataset_contract_audit --json`
  - Output: `daily_research/output/data_lake/audits/v2_dataset_contract_audit_20260601_01/`.
- Added PIT/status sidecar assembly:
  - `python -m daily_research.data_lake.build_v2_status_sidecar --json`
  - Standard fields include listed/mainboard/common A/ST/suspended/delisted/tradeable/has_bar/reject_reason/source.
- Added strict pool kind: `rolling_liquidity_tradeable_mainboard`.
- Added v2 feature profile: `raw_kline_context_v2_tradeable_amount_checked`.
- Added v2 orchestration:
  - `python -m daily_research.path_policy.v2_research_reset_baseline --run-training --json`
  - `python -m daily_research.path_policy.v2_research_reset_baseline --run-comparison --json`

## Dataset Audit

- Latest audit output: `daily_research/output/data_lake/audits/v2_dataset_contract_audit_20260601_01/v2_dataset_contract_audit.json`.
- Audit verdict: `v2_dataset_contract_degraded`, not `blocked`.
- Market daily coverage: `ok`, rows `8,695,482`, trade dates `1699`, symbols `5118`.
- Benchmark coverage: `ok`, open rows `1699`, close rows `1699`.
- Amount unit policy: `as_is`.
  - `amount_to_close_volume_median=1.0001711135587465`.
  - `amount_unit_factor=1.0`.
- Degraded findings:
  - corrected old pool has active non-tradeable rows: suspended `2146`, ST `13`;
  - PIT status source contract is incomplete: universe/status sidecars are single-date and list/delist dates are blank;
  - missing/fill risk remains visible in raw market rows and must stay auditable.
- Interpretation: BaoStock-first lake is usable for v2 research, but the status/missing contract still needs phase-2 hardening.

## Strict Pool Evidence

- Strict pool validation output: `daily_research/output/path_policy/studies/mh_v2_reset_tradeable_mainboard_anchor_20260601_01/v2_strict_pool_validation.json`.
- Validation status: `ok`.
- Active universe size: `2602`.
- Daily member count min/median/max: `487 / 500 / 500`.
- Shortfall dates: `718`; diagnostic only, no blocker.
- Excluded active prefixes `300/301/688/689`: all `0`.
- Active ST/suspended/delisted/not-listed/missing-bar/not-tradeable rows: all `0`.
- Overlap vs corrected mainboard pool:
  - overlap true cells: `843990`;
  - removed from corrected: `5510`;
  - added by v2: `3799`;
  - daily overlap ratio mean: `0.9935138316656857`.

## V2 Memmap Contract

- Seed7 manifest: `daily_research/output/path_policy/studies/mh_v2_reset_tradeable_mainboard_seed7_20260601_01/forecast_dataset_manifest.json`.
- Manifest guard: `ok`.
- Feature store shape: `[1699,2602,124]`.
- Sample rows: train `449076`, validation `103413`, test `104471`.
- Feature groups:
  - state `59`;
  - raw kline `15`;
  - market context `26`;
  - peer context `12`;
  - regime context `8`;
  - history quality `4`;
  - alpha prior `0`.
- Alpha/score-like feature count: `0`.
- Amount audit in manifest:
  - policy `as_is`;
  - factor `1.0`;
  - median `1.000075266090736`;
  - p95 abs log error `0.010067455738293223`.

## Three-Seed Baseline Result

- Anchor: `mh_v2_reset_tradeable_mainboard_anchor_20260601_01`.
- Seed tags:
  - `mh_v2_reset_tradeable_mainboard_seed7_20260601_01`;
  - `mh_v2_reset_tradeable_mainboard_seed11_20260601_01`;
  - `mh_v2_reset_tradeable_mainboard_seed19_20260601_01`.
- Training summary status: `completed`; failed tags: `[]`.
- Aggregate gate: `pass`.
- Aggregate test row:
  - seed count `3`;
  - rank IC mean/min `0.1097597160364623 / 0.0833707008987291`;
  - spread mean/min `0.0395815603562681 / 0.0320546211112902`;
  - hit lift mean/min `0.0154853717382623 / 0.0093135606924032`;
  - monthly positive rate mean/min `0.8787878787878789 / 0.8181818181818182`;
  - negative month count max `2`;
  - thirty-day concentration mean `0.6153860880052838`.
- Gate checks all passed:
  - `seed_count_ge_3`;
  - `rank_ic_min_positive`;
  - `spread_min_positive`;
  - `hit_lift_min_positive`;
  - `monthly_positive_rate_mean_ge_075`;
  - `negative_month_count_max_le_2`;
  - `thirty_d_concentration_not_worse_than_stage28`.

## Per-Seed Test Snapshot

| seed | rank_ic_min | spread_min | hit_lift | decision_score_rank_ic | rank_ic_20d | rank_ic_30d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.041541219714843446 | 0.002554445658980474 | 0.015345790984592038 | 0.10683241779029147 | 0.13020747328470156 | 0.10515419728509086 |
| 11 | 0.02026214153975375 | 0.0022942287760995906 | 0.02179676353779186 | 0.13907602942036637 | 0.11390535326209038 | 0.06489737634365789 |
| 19 | 0.027206420717170256 | 0.0028388613592764604 | 0.009313560692403204 | 0.08337070089872911 | 0.0911677095412676 | 0.04725791027104983 |

## Interpretation

- The v2 strict tradeable mainboard input contract repairs the previous corrected mainboard `near_pass` failure shape.
- This result supports `daily_research_v2_research_reset` as the current path_policy / multi-horizon research baseline.
- This does not prove the model is promotion-ready. It is evidence-grade research, not promotion-grade execution evidence.
- Remaining data-contract blockers are now phase-2 research items, not blockers to starting v2 research:
  - true PIT listing/delist history;
  - fuller historical ST/suspension source rather than single-date sidecars plus market fallback;
  - missing/fill semantics;
  - optional limit status, industry, valuation, and market-cap domains.

## Boundaries

- Do not restore old `156` feature profile.
- Do not require old Stage 2.8 / short_v5b replay to continue v2 research.
- Do not promote to live/default from this baseline.
- Do not write `daily_research/output/active_execution_strategy.json`.
- Execution remains `frozen_skeleton_only / awaiting_research_rebuild`.

## Verification

- `python -m pytest daily_research/data_lake/tests/test_research_data_lake.py -q`: `32 passed`.
- `python -m pytest daily_research/data_platform/tests/test_domain_contract.py daily_research/data_platform/tests/test_refresh_daily.py -q`: `21 passed`, one pandas FutureWarning.
- `python -m pytest daily_research/path_policy/tests/test_mainboard_rebuild_baseline.py -q`: `5 passed`.
- v2 targeted tests: `25 passed`.
- `git diff -- daily_research/output/active_execution_strategy.json`: empty.
- `git diff --check`: clean.
- `python -m tools.brain.doc_guard check`: passed, structural warnings only.
- `python -m tools.brain.integrity_check --json`: `status=ok`, warnings only for acknowledged noncanonical brain paths.

## Next Allowed Actions

- Treat `mh_v2_reset_tradeable_mainboard_anchor_20260601_01` as the v2 baseline for new research comparisons.
- Continue phase-2 data contract hardening: true PIT listing/delist, stronger status sidecar sources, missing/fill semantics, limit/industry/valuation optional domains.
- Run v2 feature and model improvements against this strict baseline, with explicit dataset id, pool id, feature profile, seeds, costs, and gate.
- Keep execution frozen until a separately approved execution rebuild plan and candidate backtest bridge exist.
