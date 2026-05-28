# Alpha Multi-Horizon Stage 3A-3C Architecture/Input Scout - 2026-05-28

## Verdict

- Status: `stage30_completed_with_branch_blockers`, `stage31_completed`, `stage32_completed / no architecture upgrade / shadow-only`.
- Mainline: `alpha_multi_horizon_utility_policy_v1`.
- Study root: `daily_research/output/path_policy/studies/mh_stage30_arch_input_scout_20260528_01/`.
- Fixed target/loss baseline: `decision_utility_v1` output with `target_norm_head_constraint_v1`.
- Final decision: keep `gru_sequence_static_context + raw_kline_context_no_alpha_prior_v1 + target_norm_head_constraint_v1` as the Stage 3 baseline. `patch_transformer_static_context` reached Stage 3C but failed final 3-seed upgrade gate.
- Active artifact impact: `daily_research/output/active_execution_strategy.json remains unchanged`.
- Boundary: all runs are research / shadow-only; no allocator, replay, paper, live/default, broker, production root, active artifact, or promotion changes.

## Scope And Matrix

- Dataset: `policy_input_bundle__7c8f58d851bce8179e1e9e2d`.
- Universe scope: `full_rolling_liquid500`.
- Reused raw memmap manifest: `daily_research/output/path_policy/studies/path20_horizon_discovery_no_alpha_gru_liquid500_h1_2_3_5_8_10_15_20_30_du_cost20_hit10_dd010_20260523_01/forecast_dataset_manifest.json`.
- Raw manifest properties: `universe_size=2430`, `train_rows=452034`, `feature_store_shape=[1699,2430,156]`.
- Fixed horizons: `1,2,3,5,8,10,15,20,30`.
- Stage 3A scout seed: `7`; Stage 3B confirmation seed: `11`; Stage 3C final seed: `19`.
- Stage 3A raw architecture tasks completed: `patch_transformer_static_context`, `stock_mixer_sequence`, `sector_slot_mixer_sequence`.
- Sector input branch blocked: the sector anchor had full-pool size but `sector_context_feature_count=0`; sector-dependent architecture tasks were excluded rather than silently falling back.

## Stage 3A Scout Results

Validation-ranked scout vs Stage 2.8 seed7 GRU baseline:

| Candidate | Composite | Validation gate | Test veto | Stage 3B |
|---|---:|---|---|---|
| `patch_transformer_static_context + raw_kline_context_no_alpha_prior_v1` | `1.283495` | pass | no | yes |
| `sector_slot_mixer_sequence + raw_kline_context_no_alpha_prior_v1` | `0.957194` | pass | no | no after corrected strong-candidate selection |
| `stock_mixer_sequence + raw_kline_context_no_alpha_prior_v1` | `0.697762` | pass | no | no |

An implementation bug was fixed during the run: `negative_month_count_max=0` was being treated as missing because `_int()` used `value or default`. After the fix, `patch_transformer_static_context` correctly passed the Stage 3A hard gate and became the canonical Stage 3B candidate.

## Stage 3B Confirmation

- Canonical Stage 3B candidate: `patch_transformer_static_context + raw_kline_context_no_alpha_prior_v1`.
- Seed11 training tag: `mh31_confirm_raw_patch-transformer-static-context_seed11_20260528_01`.
- Stage 3B comparison status: `completed`.
- Stage 3B result: passed and advanced to Stage 3C.
- Two-seed test summary for patch: rank/spread/hit all positive, mean monthly positive rate `0.818182`, max negative months `2`.

Note: `mh31_confirm_raw_sector-slot-mixer-sequence_seed11_20260528_01` was run before the zero-value gate fix but is not part of the canonical Stage 3B/3C chain.

## Stage 3C Final Confirmation

- Stage 3C candidate: `patch_transformer_static_context + raw_kline_context_no_alpha_prior_v1`.
- Seed19 training tag: `mh32_final_raw_patch-transformer-static-context_seed19_20260528_01`.
- Final comparison summary: `stage32_final_arch_confirmation_summary.json`.
- Final status: `architecture_upgrade_allowed=false`, `evidence_grade_architecture_pass=false`.

3-seed test comparison, `pred_decision_score`:

| Metric | Stage 2.8 GRU baseline | Patch candidate | Interpretation |
|---|---:|---:|---|
| rank IC mean | `0.097677` | `0.099253` | slight gain |
| top-bottom spread mean | `0.037694` | `0.049589` | gain |
| hit lift mean | `0.016526` | `0.002137` | worse |
| hit lift min | positive | `-0.000133` | fails all-seed positivity |
| mean monthly positive rate | `0.878788` in Stage 2.8 reference | `0.787879` | still >= 0.75 |
| max negative months | `2` in Stage 2.8 reference | `3` | fails final gate |
| long horizon share mean | `0.815462` | `0.916702` | worse concentration |
| 30d concentration mean | `0.776934` | `0.916702` | worse concentration |
| pred/future horizon gap mean | `15.618355` | `16.871921` | worse gap |

## Interpretation

- The low-seed search policy was useful: it found `patch_transformer_static_context` as a plausible scout winner without spending 3 seeds on every candidate.
- The final 3-seed check was still necessary: patch's validation and two-seed profile looked promising, but seed19 exposed hit-lift fragility and renewed long-horizon concentration.
- The sector input branch did not actually test sector information; the built `raw_kline_context_sector_v1` full-pool artifact had zero sector context features. This is an input-construction blocker, not evidence that sector information is useless.
- The current result does not justify upgrading the architecture. Stronger architecture can amplify the repaired target/loss surface, but it can also reintroduce horizon concentration and monthly instability.

## Next Action

- Keep active execution unchanged.
- Keep Stage 3 architecture baseline as `gru_sequence_static_context + raw_kline_context_no_alpha_prior_v1 + target_norm_head_constraint_v1`.
- Do not promote PatchTST/patch transformer from this run.
- Before more architecture search, either fix the sector feature construction path or run narrowly targeted input/architecture scouts with single seed followed by confirmation only for clear winners.
- If pursuing patch further, treat it as a diagnostics candidate: investigate why seed19 drives hit lift negative and 30d concentration back up before allocating more full-pool training.
