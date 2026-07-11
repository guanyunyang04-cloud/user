# Path Policy Seq100 Purged Walk-Forward 2022-2025 Result

Date: `2026-07-10`

## Verdict
- Status: `completed / purged_expanding_walk_forward / no_default_change / no_active_execution_change`.
- Study family: `seq100_purged_walkforward_2022_2025`.
- The formal seq100 evaluation policy is now 2022-2025 purged expanding OOS. A train row is valid only when `label_end_trade_date < oos_start_trade_date`.
- The old `2024 validation + 2025 train-through-2024 forward test` evidence is pre-purge historical evidence. It must not be mixed with or used to override this study.
- `summary_v2_all_channels` does not replace the current default `daily_only_summary_v2_ohlcva_aux_low`.
- Both profiles had positive Top3 path-value-alpha point estimates in all four folds. All-channel had a slightly higher fold-equal Top3 mean (`10.03%` versus `9.38%`), but its paired advantage was only `+0.65 percentage points` (`0.0065`) and was not statistically resolved by HAC or 60-day moving-block bootstrap.
- Active artifact impact: unchanged. The study did not create or change `daily_research/output/active_execution_strategy.json`, QDP active pointers, broker state, trade plans, or live/default execution state.

## Leakage Correction
The source index assigns splits from signal years, while every label uses the next 60 trading days. Signal-year splitting alone therefore lets late-year training rows observe labels inside the next OOS year.

The strict rule is:

```text
label_end_date_idx = signal_date_idx + 60
train iff label_end_date_idx < oos_start_date_idx
```

The legacy 2025 roll-forward view trained through `2024-12-31`. Its last 60 signal dates, `2024-10-09..2024-12-31`, have labels ending in 2025, contaminating the nominal 2025 OOS. This affects `171,290` train rows. The same issue means the old 2024/2025 results cannot be appended to newly built 2022/2023 folds; all four years had to be rerun.

## Frozen Contract
- Source view: `daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva.json`.
- Folds: 2022, 2023, 2024, 2025 expanding train windows with one calendar-year OOS split.
- Splits: `train/oos`; there is no formal validation/test duplication.
- Normalization: feature dates from `2012-01-04` through the trading day before each OOS start. Public feature dates inside the 60-day purge gap are allowed; their incomplete future labels are not.
- Seed: `7`.
- Epochs: `1`, fixed before OOS.
- Checkpoint: final epoch; `early_stopping_patience=0`.
- Training-time OOS access: forbidden. OOS is evaluated once after the final checkpoint is saved.
- Prediction storage: `none` for large per-sample predictions; daily IC, daily TopK, compact Top100 candidates, aggregates, reports, and checkpoints are retained.
- Registered primary comparison: paired daily Top3 `alpha_path_trade_value_v2_60d`, all-channel minus daily-only default.
- Guardrails: Top1, Top10, IC, hit/loss rates, worst fold, and complete-case coverage.
- Primary dependence correction: Bartlett HAC with max lag 59 plus a 10,000-replication, non-circular 60-trading-day moving-block bootstrap over the continuously ordered 969-day OOS sequence (`oos_year`, then `trade_date`; base seed 7 with deterministic per-metric streams). Fold-grouped and fold-equal bootstraps are sensitivity checks only.

## Fold Audit

| OOS | OOS start | safe train signal end | max train label end | train rows | purged rows / dates | OOS rows / dates | label overlap | complete-case excluded |
|---|---|---|---|---:|---:|---:|---:|---:|
| 2022 | 2022-01-04 | 2021-10-08 | 2021-12-31 | 3,534,268 | 158,468 / 60 | 647,646 / 242 | 0 | 12,015 / 1.82% |
| 2023 | 2023-01-03 | 2022-09-30 | 2022-12-30 | 4,176,533 | 163,849 / 60 | 677,855 / 242 | 0 | 7,831 / 1.14% |
| 2024 | 2024-01-02 | 2023-09-28 | 2023-12-29 | 4,846,724 | 171,513 / 60 | 690,738 / 242 | 0 | 8,667 / 1.24% |
| 2025 | 2025-01-02 | 2024-10-08 | 2024-12-31 | 5,537,685 | 171,290 / 60 | 689,446 / 243 | 0 | 12,278 / 1.75% |

Each fold was also loaded through the real dataset reader. All-channel input was `[1,100,84]`, daily-only input was `[1,100,32]`, the OHLC target was `[1,60,4]`, the OHLCVA auxiliary target was `[1,60,6]`, and the price anchor was `today_close`.

## Fold Results
TopK values are mean daily path-value alpha against the same-day eligible universe.

| OOS | profile | IC | Top1 | Top3 | Top10 |
|---|---|---:|---:|---:|---:|
| 2022 | summary_v2_all_channels | 0.0885 | 7.86% | 7.71% | 6.05% |
| 2022 | daily_only_summary_v2_ohlcva_aux_low | 0.1010 | 6.61% | 5.68% | 4.95% |
| 2023 | summary_v2_all_channels | 0.0527 | 0.18% | 0.13% | 0.49% |
| 2023 | daily_only_summary_v2_ohlcva_aux_low | 0.0592 | 2.92% | 4.73% | 4.33% |
| 2024 | summary_v2_all_channels | 0.1619 | 22.65% | 22.32% | 17.44% |
| 2024 | daily_only_summary_v2_ohlcva_aux_low | 0.1558 | 20.79% | 18.73% | 15.23% |
| 2025 | summary_v2_all_channels | 0.1253 | 22.65% | 9.97% | 3.65% |
| 2025 | daily_only_summary_v2_ohlcva_aux_low | 0.1162 | 19.87% | 8.39% | 6.67% |

Fold-equal Top3 summary:

| profile | mean | worst fold | positive folds |
|---|---:|---:|---:|
| summary_v2_all_channels | 10.03% | 0.13% | 4/4 |
| daily_only_summary_v2_ohlcva_aux_low | 9.38% | 4.73% | 4/4 |

## Paired Results
All differences are `summary_v2_all_channels - daily_only_summary_v2_ohlcva_aux_low` over exactly matched dates and universes.

| metric | pooled difference | fold-equal difference | positive folds | HAC 95% CI | continuous ordered 60d MBB 95% CI |
|---|---:|---:|---:|---:|---:|
| Top1 path-value alpha | +0.79 pp | +0.79 pp | 3/4 | [-3.06 pp, +4.64 pp] | [-3.06 pp, +4.83 pp] |
| Top3 path-value alpha | +0.65 pp | +0.65 pp | 3/4 | [-2.41 pp, +3.72 pp] | [-2.76 pp, +3.49 pp] |
| Top10 path-value alpha | -0.89 pp | -0.89 pp | 2/4 | [-2.74 pp, +0.96 pp] | [-2.72 pp, +1.05 pp] |
| daily rank IC | -0.00095 | -0.00096 | 2/4 | [-0.01025, +0.00835] | [-0.00957, +0.00820] |

The all-channel Top3 yearly differences were `+2.03% / -4.60% / +3.59% / +1.58%` for 2022-2025. The 2023 reversal is large enough that the positive pooled mean cannot be treated as a stable dominance result. All-channel also did not broaden to Top10.

As a fold-preserving sensitivity check, grouped pooled 60-day MBB intervals were Top1 `[-2.82 pp, +5.07 pp]`, Top3 `[-2.65 pp, +3.07 pp]`, Top10 `[-2.44 pp, +0.48 pp]`, and IC `[-0.00987, +0.00755]`; fold-equal intervals were nearly identical. These grouped intervals are not the primary dependence correction because every replication keeps all four observed year weights fixed.

Top1 `selected_loss_5pct_rate` was lower for all-channel by `4.85 percentage points` on average, but the primary continuous ordered MBB interval was `[-14.55 pp, +3.51 pp]` and crossed zero. The grouped sensitivity interval `[-11.66 pp, -0.41 pp]` excluded zero because every replication retained the large 2025 effect; this remains an unresolved, 2025-driven risk diagnostic rather than a promotion result.

## Model Design Assessment
The core `summary_v2` design is coherent:

- Pointwise future-path loss supplies local trajectory supervision.
- Multi-horizon 5/10/20/40/60 OHLC-derived summary losses supply direct gradients for global path shape and timing.
- Ranking value remains derived from the predicted path, keeping the score consistent with the path explanation instead of training an unrelated score head.
- The same target/value semantics are used for both profiles, but they are profile bundles rather than a one-factor ablation: all-channel uses `gru_path_value` without VA auxiliary, while the default uses daily-only input plus `gru_ohlcva_aux_path_value` and VA weights `0.02/0.01`.

The all-channel profile bundle produced higher narrow TopK in three folds but nearly collapsed in 2023 and did not improve pooled IC or Top10. This study cannot attribute that pattern separately to the extra intraday/limit inputs, the absence of VA auxiliary supervision, the different output head, or their interaction. It shows bundle-level regime dependence rather than proving any individual channel family is noisy. The daily-only default has the stronger worst-fold Top3 result and simpler input surface.

Therefore:

- `summary_v2` is a reasonable model principle.
- This study does not compare summary_v2 against a base-summary model, so it supports feasibility and temporal behavior, not the incremental causal value of summary_v2 itself.
- `summary_v2_all_channels` is a valid narrow TopK comparison candidate.
- It is not established as the best overall profile and does not replace the daily-only low-VA default.

## Evaluation Policy Assessment
Purged multi-year rolling OOS provides higher confidence about temporal/regime robustness than one fixed recent-year test because it exposes four distinct markets and 969 paired OOS days. It does not create 969 independent observations: 60-day labels overlap heavily, folds share expanding training history, and the profiles were developed using earlier 2024/2025 evidence. HAC and the primary continuous ordered block bootstrap address serial dependence, but not profile-selection bias or single-seed training variance.

The primary moving-block bootstrap resamples non-circular 60-day blocks from the single chronologically ordered 969-day sequence, so source blocks can preserve dependence across observed fold boundaries and bootstrap samples can vary their effective year composition. The grouped bootstrap is a fold-preserving sensitivity check: it resamples inside each observed year while keeping the four year weights fixed, so it does not resample the market-regime population or preserve cross-year label overlap. A leave-one-year-out Top3 mean turns slightly negative (`-0.33%`) when 2024 is removed, so the positive pooled estimate is not a cross-regime dominance claim.

The practical policy is:

- Retire the duplicated fixed validation/test design as the formal verdict path.
- Use purged expanding `train/oos` folds for model comparison.
- Freeze profile, seed, epoch budget, primary metric, and checkpoint policy before evaluating outer OOS.
- Do not repeatedly tune against all historical OOS and then call it a clean test. Material profile redesign requires nested selection or a newly arriving future OOS period.

## Residual Limitations
- The source sample index requires complete future-60d labels. Relative to `input_valid & entry_buyable`, `1.14%-1.82%` of signal-day candidates are excluded by future `label_valid`; this complete-case filter may be optimistic for suspensions, delistings, or missing outcomes.
- Only seed 7 and one fixed epoch were run. This answers the frozen-config temporal question, not multi-seed training variance.
- Path-value alpha is a research ranking diagnostic, not realized portfolio PnL. It does not include a full execution simulation, capacity, turnover, slippage, or partial-label exit policy.
- Prior 2024/2025 runs influenced profile selection. This study improves evidence but is not a pristine prospective trial.

## Artifacts
- Builder/trainer/aggregator: `daily_research/path_policy/seq100_walkforward.py` and `daily_research/path_policy/qdp_v2_sequence_path_training.py`.
- Fold views: `daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva_purged_oos{2022,2023,2024,2025}.json`.
- Study root: `daily_research/output/path_policy/studies/seq100_purged_walkforward_2022_2025/`.
- Study summary: `daily_research/output/path_policy/studies/seq100_purged_walkforward_2022_2025/study_summary.json`.
- Paired table: `daily_research/output/path_policy/studies/seq100_purged_walkforward_2022_2025/paired_profile_comparison.csv`.
- Fold tables: `fold_split_metrics.csv` and `fold_topk_metrics.csv` under the study root.
- Daily evidence: `daily_topk_all_folds.csv` and `daily_rank_ic_all_folds.csv` under the study root.
- Run ledger: `daily_research/output/path_policy/studies/seq100_purged_walkforward_2022_2025/study_manifest.json`.
- Fold metadata reconciliation audit: `daily_research/output/path_policy/studies/seq100_purged_walkforward_2022_2025/fold_metadata_reconciliation.json`.

### Post-Run Fold Metadata Reconciliation

- After all eight model runs completed, the four fold view manifests were found to retain cloned legacy top-level `train_years` / `validation_years` / `test_years` fields even though their sample indexes and runtime contract already used only `train/oos`.
- The explicit `bind-legacy-run-contracts` migration first validated all eight embedded `final_model.pt` configs, checkpoint policies, final epochs, pack paths, run tags, feature metadata, output paths, and required evidence files. It then bound the fold manifests, training summaries, and study ledger to stable material contracts.
- Bound contract SHA256 values are 2022 `8896fcdbd7641244d68224ce9502792f63b9a5df3ffbc4c8ab84633ffc236525`, 2023 `9731f0bb572e291f09e814e362de483bf4ba08d8d9b7c170a6cdf1a0580a818d`, 2024 `7c716be57998651b29f05ee1e3311e4dc555667ca5e48f561d125dbdc62153bf`, and 2025 `9171a1eaac318a742cd07bed40218d704df29d7827c6371cacaa4fb27a6aeaa3`.
- `reconcile-bound-fold-metadata` rebuilt all four candidates in an isolated staging store and required exact sample-frame, parquet SHA256, normalization, and material-contract equality before updating any live view. It then changed only the fold JSON metadata to declare the observed train years, `oos_years=[fold_year]`, empty validation/test years, `split_roles={fit: train, evaluation: oos}`, and the OOS-specific end date.
- Live sample SHA256 values remained 2022 `8db4baadbe44873d0097075d29dc9a03f65e92fc83bd9ce774ed79e21ceb80f2`, 2023 `17b1e8aa7fc3680ddbf4f2ff95d268cdfa3f8d936f10b1a2c45aa44ca97cb6a4`, 2024 `8f0a7d70e71601d3ccef0804655e07d5af3d304feb0e7aedb33075e8a19fa612`, and 2025 `9a3ac1ee03cc86cfe55059dfee91d3c0282d7176c15028500af4bee5b6b9dec1`.
- This was a metadata-only correction: selected train/OOS sample rows, label-end dates, sample counts, normalization cutoffs/statistics, shared feature/label arrays, model checkpoints, daily evidence, aggregate metrics, and the study verdict did not change. No model was retrained or re-evaluated because the data consumed by the completed runs was unchanged. The 115.6 MB staging copies were removed after the completed audit and idempotent recheck.

### Run Tags

- `seq100_purged_walkforward_2022_2025`
- `summary_v2_all_channels_purged_oos2022_seed7_20260710_131412`
- `daily_only_summary_v2_ohlcva_aux_low_purged_oos2022_seed7_20260710_133647`
- `summary_v2_all_channels_purged_oos2023_seed7_20260710_135055`
- `daily_only_summary_v2_ohlcva_aux_low_purged_oos2023_seed7_20260710_141510`
- `summary_v2_all_channels_purged_oos2024_seed7_20260710_143051`
- `daily_only_summary_v2_ohlcva_aux_low_purged_oos2024_seed7_20260710_145831`
- `summary_v2_all_channels_purged_oos2025_seed7_20260710_151612`
- `daily_only_summary_v2_ohlcva_aux_low_purged_oos2025_seed7_20260710_154643`

## Next Allowed Actions
- Keep `daily_only_summary_v2_ohlcva_aux_low` as the default research profile.
- Keep `summary_v2_all_channels` as a narrow Top1/Top3 comparison branch; do not promote it from this study.
- If the source of the bundle difference matters, run a purged one-factor ablation that aligns model head and VA losses while changing only the input channel set.
- Before changing the default, test training variance with preregistered additional seeds or wait for a genuinely new future OOS period.
- Treat the complete-case universe issue as the next data/evaluation correction before execution-layer promotion.
- Execution-layer backtesting remains a separate task and requires explicit execution evidence; this study does not unfreeze execution.
