# Daily Research state

Updated: `2026-07-26`

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
- `seq100_pit_signal_quality_v1` is active. Its externally frozen review contains
  38 primary sources and six differentiated model families; target screening
  froze `pareto_ordinal_v1`, and the 2010-2020/2021/2022 feature screen froze F1.
  The frozen decision rule and feature profile live in `seq100_signal_quality.py`
  constants (`PATH_TARGET_PROFILE`, `FORMAL_FOLD_YEARS`) and in the per-attempt
  freeze JSON, not in the study contract; the contract JSON is a data-binding
  spec.
- Model screening is COMPLETE, not paused. `model_screen/attempt_001/formal_matrix.json`
  is `formal_matrix_frozen` at `2026-07-25T23:25:02+08:00` with
  `replacement_allowed: false`, `selected_feature_profile: F1`, `formal_seed: 7`,
  and `formal_matrix_sha256: 189928c9...e26f03` bound to the current
  `contract_sha256: f15d1af2...561b2`. All six families carry frozen prescreen
  metrics. Verified `adapter.execution_semantics_version` per family: TabM
  `prescreen_retry_001`, PatchTST `prescreen_retry_002`, Deep Sets, DeepHit, and
  NeuralNDCG all completed under their v2 semantics; only the superseded
  `tabm_multioutput/prescreen` carries no version and is retained as history.
  Prescreen `daily_ndcg_at_1pct`: TabM 0.2271, Deep Sets 0.2256, DeepHit 0.2232,
  NeuralNDCG 0.2184, PatchTST 0.2174, LightGBM 0.2026.
- Formal fold training is IN PROGRESS. The matrix is 6 families x 3 folds
  (2023/2024/2025) x first seed 7 = 18 cells; robustness seeds are `[17, 29]`.
  LightGBM is complete for all three folds at seed 7: 2023
  `ndcg@1%=0.2145 / IC=0.3735` in 668.83s over 5,374,206 train / 545,843
  development / 742,473 test candidates, 2024 `0.2439 / 0.3475`, 2025
  `0.1732 / 0.4336`. The 2025 head precision is the weakest while its IC is the
  strongest; treat that split as an `evaluate`-stage bootstrap question, not a
  conclusion. LightGBM's null `execution_semantics_version` is correct because it
  is not in `NEURAL_MODEL_IDS` and takes the non-accumulating branch.
- CLOSED. `seq100_pit_signal_quality_v1` terminated as `research_design_insufficient`
  on 2026-07-26 through the new audited design-invalidation channel. The active
  contract `daily_research/studies/seq100_pit_signal_quality_v1.json` is deleted;
  the byte-identical archive is
  `daily_research/research_records/seq100/seq100_pit_signal_quality_v1/contract.json`
  and the compact record is `artifact.json` beside it
  (`artifact_sha256=7a0ff473...4636e7`). No remaining formal cell may be started
  and no winner exists. Everything below this line describes the closed study.
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
- `training/tabm_multioutput/fold_2023/seed_7` has an interrupted `attempt_001`.
  Its `active.json` says `status=training` and `progress.json` says
  `phase=validation, epoch=1/10`, last written `2026-07-26T09:45:36+08:00`, but
  no training process is alive; it died when system memory reached 96.3% with
  RSS 8.79 GB. It holds `last_checkpoint.pt` (2.13 MB) and
  `resume_key_sha256=abf71ce4c3b33c264c5c14675b815a180bab57343a3d9d7937cd904ef7ef8354`
  under `execution_semantics_version=exact_effective_batch_global_loss_v2`, so it
  is resumable. Do not read `status=training` as a live run. Nothing else in the
  matrix has started: no `evaluate`, winner, or 2026 confirmation exists.
- `model_screen/training_budget_amendment.json` (`max_epochs` 3 to 10, `patience`
  1 to 2, `amendment_sha256=4dd61a1d...13c67`) is consumed by exactly one cell:
  that TabM `fold_2023/seed_7` resolved config. All three LightGBM folds carry
  `training_budget_amendment_sha256: null` and are unaffected.
- TARGET UNDER REVIEW. A read-only diagnosis over the three completed LightGBM
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
- Neural runtime v2 is implemented and qualified before screening resumes. It
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
  ranking rule, exit rule, Top-K, slot count, or account behavior.
- Protected packs and the 15 registered model bundles remain unchanged. The
  retained study checkpoints are research evidence and are not registered or
  active for execution.

<!-- seq100-signal-quality:start -->
- `seq100_pit_signal_quality_v1` 已以 `research_design_insufficient` 收口；selector winner 为 `null`。
- 权威 compact record：`daily_research/research_records/seq100/seq100_pit_signal_quality_v1/artifact.json`。
- 正式模型 registry、active execution、QDP 与完整 PIT pack 均未修改；2026 仍只允许新合同确认。
<!-- seq100-signal-quality:end -->
