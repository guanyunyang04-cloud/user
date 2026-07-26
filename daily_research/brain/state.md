# Daily Research state

Updated: `2026-07-27`

- The registered frozen baseline remains `structured_joint_turnover_180x35_v2`
  (`L35V2`), but it is no longer valid selection evidence for deployment. The
  complete-PIT frozen audit found severe survivorship bias and a legacy
  next-open denominator distortion.
- The formal audit is
  `daily_research/research_records/seq100/pit_l35v2_survivorship_frozen_audit_2020_2025/`.
  For the legacy-selected Top1/1 + D14 strategy, 2023-2025 liquidated ending
  equity fell from CNY 52.18m to CNY 13.82m on the complete PIT universe; the
  2020-2025 replay ended at CNY 21.48 after a zero-recovery delisting.
- The complete PIT 180x35 pack is
  `daily_research/data/research_store/seq100_pit_l35v2_v1/`. It contains
  8,531,565 samples over 3,419 securities and six development fold views. The
  training purge is 60 trading days; execution-tail observations are not model
  targets.
- The deterministic signal-close 2x2 study is complete. `V2C-P0` won with
  `Top1 / 1 slot / fixed D44`; CNY 1m became CNY 9.10m after liquidation,
  annualized log growth was 0.76563, maximum drawdown was -48.53%, and all three
  annual log-growth values were positive. Three of four arms qualified.
- All six P1 folds passed the per-step Rank-gradient budget evidence. All 12
  successful checkpoints, predictions, account jobs, reports, and runners are
  retained under the ignored study output for reproduction.
- Probability variants, strict OOF tree reranking, and 2026 confirmation have
  not started and require separate contracts. The true-batch-1024 study remains
  paused and resumes only on explicit request.
- CLOSED. `seq100_pit_signal_quality_v1` terminated as `research_design_insufficient`
  on 2026-07-26 through the new audited design-invalidation channel. The active
  contract `daily_research/studies/seq100_pit_signal_quality_v1.json` is deleted;
  the byte-identical archive is
  `daily_research/research_records/seq100/seq100_pit_signal_quality_v1/contract.json`
  and the compact record is `artifact.json` beside it
  (`artifact_sha256=7a0ff473...4636e7`). No remaining formal cell may be started
  and no winner exists. Its frozen process artifacts are architecture evidence
  only.
- The terminal record is
  `design_invalidation/attempt_001/design_invalidation.json`
  (`design_invalidation_sha256=26a92e69...b461`),
  `defect.class=objective_cannot_answer_stated_question`, six bound evidence
  files, ten recorded inspected alternatives, `burned_fold_years=[2023,2024,2025]`,
  four trained cells with three completed, `retained_as=architecture_exploration_evidence`,
  and `successor.must_be_new_study_contract=true`. Every frozen artifact hash is
  preserved unchanged inside it: research freeze `c7853c5a...`, target manifest
  `00baa1d4...`, feature freeze `4cb21c30...`, formal matrix `189928c9...`.
  All four closeout gates passed and protected-object hashes were identical
  before and after.
- The interrupted TabM attempt and its durable checkpoint remain preserved, but
  the closed-study contract forbids resuming it. No training process is alive;
  there was no `evaluate`, winner, or 2026 confirmation.
- The closed-study target diagnosis over the three completed LightGBM
  folds (2,178,290 prediction rows) found that the frozen design ranks path
  robustness, not growth. `U = min(log1p(d5)/5, log1p(d10)/10, log1p(d20)/20)` is
  positive for only 21.7%/25.7%/30.9% of filled candidates, so six negative-`U`
  families reflect the metric, not absent alpha. Model IC against `d20` is only
  0.1256/0.1016/0.0675 while IC against `mdd20` is -0.5387/-0.5168/-0.6301.
  Perfect-foresight daily Top-1% annualized log growth is 1.03/1.72/1.28 for
  `pareto_ordinal_v1` versus 4.53/5.13/5.99 for `g20`, so the frozen target caps
  achievable growth near 23% of the return-only ceiling. Evidence:
  `daily_research/brain/references/seq100_signal_quality_target_diagnosis_20260726.md`.
- An independent Oracle review accepted that diagnosis as design evidence and
  rejected it as sufficient grounds to unfreeze a target inside a live study.
  Only the `g20` ceiling row is a true upper bound; the
  `IC(objective, model score)` column is circular because the score is predicted
  pareto times predicted fill; and the ceiling script filtered on `entry_filled`,
  which is future information. That review is why the study was closed rather
  than retargeted in place: the contract (`:229`), `research_freeze.json:267`,
  and `formal_matrix.json:736` all forbid formal replacement.
- A follow-up read-only IC-to-growth transfer estimate used formal-evaluator
  eligibility instead, ranking the full daily universe with unfilled entries
  realizing cash. Blending the true `g20` rank with independent noise to hit a
  target per-day rank IC gives daily Top-1% annualized log growth of 0.37/0.59/0.80
  at IC 0.02, 0.77/1.07/1.25 at IC 0.05, and 1.06/1.44/1.67 at IC 0.10 across
  2023/2024/2025. The trained model reaches IC 0.125/0.101/0.068 against `g20`
  but returns only 0.014/0.172/0.037, because its extreme tail is selected on low
  amplitude rather than on growth. Cash share inside the selected Top-1% is
  0.03-0.7%, so unfilled and limit-up candidates are not what caps the result.
  Evidence script: `tmp/seq100_ic_growth_transfer.py` (ignored output).
- SUCCESSOR NOT YET DESIGNED. The owner has accepted growth as the objective and
  the recorded successor requirement is a new study contract; no successor
  contract, target, or metric has been written yet. Any successor must treat
  2023-2025 as burned discovery years.
- PRE-2023 FUTURE-PATH ATLAS COMPLETE. The read-only discovery cohort contains
  6,096,195 full-universe candidates from `2010-09-29` through `2022-09-30`, with
  every D60 outcome ending by `2022-12-30`. Of these, 5,929,931 (97.27%) have a
  legal next-open entry and complete entry-relative D1-D60 path. The output keeps
  all rows, 52 path/execution descriptors, explicit missingness status, and the
  full 60-value close path. 2023-2026 supplied no row to the atlas.
- The original macro atlas searched K=3 through K=10. Within that declared
  range, expanding 2018/2020/2021/2022 fits all select eight PCA dimensions and
  K=3: declining 2,270,431 (38.29%, median D60 -12.33%), strong advancing
  777,402 (13.11%, +30.88%), and mild advance/oscillation 2,882,098 (48.60%,
  +3.22%). Pairwise expanding-window assignment ARI is 0.734652 minimum and
  0.845183 mean; matched-centroid correlation is at least 0.977011.
- PRE-2023 HORIZON SENSITIVITY COMPLETE. A full-universe follow-up independently
  clusters D1-D5, D1-D10, D1-D20, D1-D40, and D1-D60 with K=2 through K=10.
  Selected K over expanding 2018/2020/2021/2022 is D5 `3/3/2/3`, D10
  `2/3/2/2`, and D20/D40/D60 `2/2/2/2`. The statistical criterion therefore
  favors a stable coarse low/high split from D20 onward. The source K3 remains a
  useful middle-state analysis layer, not evidence of exactly three natural
  outcome classes.
- Fixed-K3 agreement with the source D60 macro rises sharply with observed
  prefix: ARI is 0.071/0.118/0.271/0.622/0.812 and purity is
  57.1%/60.6%/69.9%/86.8%/93.9% at D5/D10/D20/D40/D60. Only 33.13% of D5-high
  rows and 37.44% of D10-high rows become source M1 strong advance, versus
  54.60% at D20 and 80.19% at D40. Short-term strength is not long-term
  strength; outcome identity becomes materially locked only around D40.
- Cluster membership is taxonomy, not within-cluster quality. Centroid distance
  measures typicality, while quality remains the conditional distribution of
  amplitude, later-state transition, adversity, persistence, and execution.
  The D5-high D60 q10/median/q90 is -16.77%/+5.17%/+42.91%, so no cluster ID may
  be treated as a complete ordinal target. Authoritative evidence:
  `daily_research/brain/references/seq100_future_horizon_atlas_20260726.md`.
- Endpoint-detrended clustering selects two shapes inside each macro state. The
  strong-advance split is stable: progressively accelerating advance 355,284
  (D20 +11.64%, D60 +43.20%, median peak D55) versus fast advance then fade
  422,118 (D20 +21.23%, D60 +20.18%, peak D35). Deep-decline/continued-decline
  and early-rise-failure/late-start splits are tentative because temporal or
  prefix ARI falls below 0.60.
- These atlases prove only that one terminal scalar or one horizon-cluster ID
  collapses distinct realized path shapes. They do not prove that a vector label
  is more predictable or should be the successor output. No signal-day
  predictability test, training, target, horizon, slot count, or exit rule was
  selected.
- LEARNABILITY INPUT STEPS 1-2 COMPLETE. The full 8,674,588-row candidate index
  now has independently available D5/D10/D20/D40/D60 `g_H`, legal-close
  `mfe_H`, close-path `pre_peak_mae_H`, through-2022 frozen fixed-K3
  low/mid/high `state_H`, and next-open `entry_fill` candidates under ignored
  `tmp/seq100_learnability_inputs/attempt_001/`. Horizon-valid `g_H`/state counts
  are 8,326,420/8,311,178/8,280,652/8,219,582/8,158,478. Missing later horizons
  do not remove available shorter-horizon labels.
- The uniform base input is a read-only row-aligned reference to 296 continuous
  and five categorical causal F1-F5 features. It consumes neither the closed
  study target nor its D60-purged folds; future training must build new
  horizon-aware folds. No new technical-indicator family was added in this step.
- LEARNABILITY STEP 3 ACTIVE. The pre-result contract
  `daily_research/studies/seq100_path_label_learnability_v1.json` freezes one
  CPU LightGBM per label-horizon-fold, horizon-specific dependency purges,
  date-equal weights, separate 2023/2024/2025 reporting, overlap-robust HAC
  statistics, and a mechanical step-4 decision gate. The three years are
  burned discovery/evaluation folds, not pristine holdouts; 2026 remains
  forbidden. No prediction or target decision existed when this contract was
  registered.
- The 2025-12-31 outcome firewall passed: all 156,376 post-2025 rows are fully
  missing, 4,112 source recomputations had zero error, and an independent
  five-horizon cross-check over 5,929,931 pre-2023 paths had zero K3 or endpoint
  mismatch. 2023/2024/2025 remain burned future predictability folds and 2026
  remains untouched. No model was trained and no candidate label or output
  structure was selected. Evidence:
  `daily_research/brain/references/seq100_learnability_inputs_20260727.md`.
- The path-map audit now distinguishes exchange-level entry from reference-order
  affordability. Exactly 766 market-buyable rows, all `600519.SH`, cannot fund
  one 100-share lot with the CNY 100,000 diagnostic order; they remain clustered
  and carry `reference_order_unaffordable`, with zero unexplained missing return
  cells. Authoritative evidence:
  `daily_research/brain/references/seq100_future_path_atlas_20260726.md`.
- The full-history structure probe has now been independently audited. Its
  preliminary claims that one slot is structurally impossible, that slot count
  has a measured +0.29 annual-log first-step effect, and that
  `turn_low30 & ret20_mid` is a +0.032 baseline are withdrawn as decisions.
  The authoritative verdict is
  `daily_research/brain/references/seq100_full_history_structure_20260726.md`.
- The strict pre-2026 terminal audit covers 3,678 signal dates through
  `2025-09-02` and 8,236,774 fixed-20-day entries. Of 21,227 D80 zero marks,
  96.86% are long suspensions, none is a security entity that ended by D80 or by
  the observed endpoint, and 93.90% become legally sellable during D81-D324.
  Median recovery is D113. Two registered code changes account for 38 rows.
- Changing only the terminal convention moves annualized log growth from
  `-0.3084` to a conservative `-0.0835`; on the uncensored D324-complete cohort it
  moves from `-0.3732` to `-0.1260`. Variance drag remains real, but the prior
  3%-annual-wipeout narrative and structural-impossibility claim are rejected.
  Diversification, low-turnover/low-volatility shapes, and the mid-momentum
  basket remain hypotheses, not frozen requirements or baselines.
- Before successor target design, the evaluation path must support terminal
  holding and registered code-change continuity, then use
  `seq100_finite_capital_backtest.py` for a real cash-constrained slot test. No
  slot count, horizon, leverage, stop, or successor target is selected yet.
- EXIT-RULE QUESTION IS ANSWERED for label-definition purposes. A read-only
  comparison of 22 exit rules over the same daily Top-1% picks, using the
  persisted back-adjusted forward panel and the audited execution and cost path,
  found that no adaptive rule beats the best fixed horizon on the worst fold.
  Worst-fold annualized log growth: `fixed_60d` 0.0391, `barrier_+10/-10_v60`
  0.0371, `trail_15pct` 0.0364, `trail_8pct` 0.0175, `fixed_20d` -0.0194,
  `ma10_break` -0.0341, `ma5_break` -0.1049, `fixed_5d` -0.1282. Tighter stops
  are monotonically worse and 20-day verticals are uniformly worse than 60-day
  ones. The MA-5 break, the owner's core discretionary rule, ranks last or next
  to last and is -0.6987 on a random-pick control, because a 6-day mean holding
  pays roughly ten times the per-day round-trip cost drag of a 60-day holding.
  Therefore the successor's entry label may be defined against a fixed terminal
  exit, which resolves the label-versus-exit circularity. Evidence:
  `daily_research/brain/references/seq100_exit_rule_comparison_20260726.md`.
- Exit timing headroom is nevertheless large and uncaptured. Perfect-foresight
  best-close exit reaches 0.4592 worst-fold on model picks and 0.9141 on random
  picks, so with no selection skill at all perfect timing would outcompound any
  measured selection-plus-fixed-exit combination. Adaptive exit stays an open
  research lane that must earn its own evidence.
- The exit comparison reconfirmed the defensive character of the retired model.
  Under `fixed_20d`, model picks return -0.0194/0.1474/0.0142 against random
  picks -0.1855/-0.1260/0.2092: value added in the two weak years, destroyed in
  the strong year. Absolute levels come from a model trained on the retired
  target, so only the ordering of exit rules under a common selection is claimed,
  and every rule parameter was inspected on burned years.
- Training now pauses instead of dying under memory exhaustion.
  `_TrainingMonitor.relieve_memory_pressure()` trims the working set on a fixed
  cadence and `memory_exhausted()` raises `TrainingPaused` after a durable
  checkpoint when available physical memory stays below
  `LOW_MEMORY_PAUSE_AVAILABLE_GB = 1.0`.
- Fold source provenance is now `schema_version: 2` in
  `seq100_fold_contract.py`: backing-file identity is workspace-relative path
  plus byte size plus full SHA-256, with `mtime_ns` and absolute paths removed,
  so relocating or restoring a pack no longer invalidates fold contracts. A
  narrow v1 reader is retained deliberately; the six recorded
  `seq100_pit_l35v2_v1/folds/views/l35v2_pit_20*.json` views are still
  `schema_version: 1` with 90 backing files each and all six revalidate under it,
  so no registered hash changed.
- Neural runtime v2 was implemented and qualified before the study closed. It
  adds exact effective-batch loss semantics, per-adapter execution versions,
  real-hardware autotune, 30-second progress/ETA/resource heartbeats, 10-minute
  and epoch checkpoints, and safe `pause.request` resume. PatchTST passed a
  100k real-pack gate at 2,188.86/s last and 2,348.90/s median throughput; all
  five neural adapters passed real preflight and tiny-fit gates. Old TabM is no
  longer reusable under v2, while LightGBM remains reusable. Evidence:
  `daily_research/brain/references/seq100_signal_quality_runtime_optimization_20260725.md`.
- The old V4 pilot is diagnostic evidence only. It fixed the fictitious low-open
  path but selected pre-signal overheated names whose next-open execution often
  preceded mean reversion; it must not be resumed under its old contract.
- 2026 remains frozen confirmation only. No 2026 result may select a model,
  ranking rule, exit rule, Top-K, slot count, or account behavior. The audited
  full-history terminal result hard-cuts every price observation at 2025-12-31.
- Protected packs and the 15 registered model bundles remain unchanged. The
  retained study checkpoints are research evidence and are not registered or
  active for execution.

<!-- seq100-signal-quality:start -->
- `seq100_pit_signal_quality_v1` 已以 `research_design_insufficient` 收口；selector winner 为 `null`。
- 权威 compact record：`daily_research/research_records/seq100/seq100_pit_signal_quality_v1/artifact.json`。
- 正式模型 registry、active execution、QDP 与完整 PIT pack 均未修改；2026 仍只允许新合同确认。
<!-- seq100-signal-quality:end -->
