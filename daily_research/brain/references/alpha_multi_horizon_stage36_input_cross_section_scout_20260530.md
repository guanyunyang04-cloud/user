# Alpha Multi-Horizon Stage 3G Input Cross-Section Scout - 2026-05-30

## Verdict

- Status: `stage36_completed`, `stage37_completed`, `stage38_completed / no Stage 39 candidate / shadow-only`.
- Mainline: `alpha_multi_horizon_utility_policy_v1`.
- Study root: `daily_research/output/path_policy/studies/mh_stage36_input_cross_section_scout_20260529_01/`.
- Fixed baseline contract: `decision_utility_v1` output with `target_norm_head_constraint_v1`, horizons `1,2,3,5,8,10,15,20,30`.
- Final decision: do not upgrade input or architecture. Keep the evidence-grade baseline as `gru_sequence_static_context + raw_kline_context_no_alpha_prior_v1 + target_norm_head_constraint_v1`.
- Active artifact impact: `daily_research/output/active_execution_strategy.json remains unchanged`.
- Boundary: all runs are research / shadow-only; no allocator, replay, paper, live/default, broker, production root, active artifact, or promotion changes.

## Scope And Feature Audit

- Dataset: `policy_input_bundle__7c8f58d851bce8179e1e9e2d`.
- Universe scope: `full_rolling_liquid500`.
- Pool view: `rolling_liquidity / rolling_liquid500`.
- Sector board view: `policy_sector_board_view__ed15b2873f544e9e9b24aae5`.
- Stage 36 scout seed: `7`; Stage 38 confirmation seed: `11`.
- `stage36_feature_profile_audit.json` confirms the new profiles were real feature additions:
  - `raw_kline_context_sector_relative_v1`: `sector_relative_context_feature_count=10`, `regime_context_feature_count=0`.
  - `raw_kline_context_regime_v1`: `sector_relative_context_feature_count=0`, `regime_context_feature_count=8`.
  - `raw_kline_context_sector_relative_regime_v1`: `sector_relative_context_feature_count=10`, `regime_context_feature_count=8`.
- Future-leakage smoke passed for all three profiles with `features_use_current_and_past_windows_only`.

## Stage 36 Scout Results

Single-seed full-pool GRU scout, ranked mainly by validation composite:

| Candidate | Composite | Validation rank / spread / hit | Test rank / spread / hit | Decision |
|---|---:|---:|---:|---|
| `sector_relative_regime + GRU` | `1.400195` | `0.158353 / 0.046670 / 0.028816` | `0.082617 / 0.031469 / 0.015882` | best scout, advanced |
| `sector_relative + GRU` | `1.297280` | `0.150341 / 0.051286 / 0.020304` | `0.089968 / 0.038562 / 0.011333` | diagnostic; test veto by rule |
| `regime + GRU` | `1.177999` | `0.145068 / 0.038714 / 0.021355` | `0.118610 / 0.041766 / 0.019839` | positive scout, not final confirmation path |

Interpretation: richer input can improve single-seed validation metrics. This is scout-only evidence, not an evidence-grade input upgrade.

## Stage 37 Architecture Retest

Using the best Stage 36 input, cross-sectional architectures did not beat same-input GRU:

| Candidate | Composite | Stage 37 pass |
|---|---:|---|
| `stock_mixer_sequence + sector_relative_regime` | `0.839823` | `false` |
| `sector_slot_mixer_sequence + sector_relative_regime` | `0.773037` | `false` |

Interpretation: the bottleneck is not solved by swapping in StockMixer or SectorSlot on this input. No architecture upgrade is allowed from Stage 37.

## Stage 38 Confirmation

Stage 38 trained one seed11 confirmation task:

- `mh38_confirm_gru_sequence_static_context_raw_kline_context_sector_relative_regime_v1_seed11_20260529_01`
- Training status: completed, return code `0`, failed tags `[]`.
- Stage 38 score: `0.483993`.
- Stage 39 candidates: `[]`.

Two-seed test comparison, `pred_decision_score`:

| Metric | Stage 2.8 raw GRU baseline seed7+11 | Sector-relative-regime GRU seed7+11 | Interpretation |
|---|---:|---:|---|
| rank IC mean | `0.104912` | `0.096316` | worse |
| rank IC min | `0.098630` | `0.082617` | positive but lower |
| top-bottom spread mean | `0.042493` | `0.034233` | worse |
| spread min | `0.042173` | `0.031469` | positive but lower |
| hit lift mean | `0.013845` | `0.020547` | better |
| hit lift min | `0.010771` | `0.015882` | better |
| mean monthly positive rate | `0.909091` | `0.818182` | passes threshold but lower |
| max negative months | `2` | `3` | fails Stage 38 gate |
| long horizon share mean | `0.913234` | `0.787515` | improved |
| 30d concentration mean | `0.912053` | `0.786105` | improved |
| pred/future horizon gap mean | `16.514089` | `15.391900` | improved, but not enough to offset gate failure |

Validation stayed strong for the candidate: validation rank/spread/hit `0.157617 / 0.045798 / 0.027136`, monthly positive rate `1.000000`, max negative months `0`, 30d concentration `0.768089`, pred/future gap `14.816830`.

## Interpretation

- Stage 3G confirms that sector-relative plus regime features are real, non-empty, full-pool inputs and can produce strong validation/scout signals.
- The signal did not survive the confirmation gate cleanly. On test, hit lift improved, and horizon concentration/gap improved, but rank IC and spread deteriorated, and max negative months rose to `3`.
- Therefore `raw_kline_context_sector_relative_regime_v1` is a useful diagnostic and feature direction, not a new evidence-grade baseline.
- Stage 37 shows StockMixer/SectorSlot still do not provide architecture-upgrade evidence in the current target/loss/output setup.

## Next Action

- Keep active execution unchanged.
- Keep Stage 3 baseline as `gru_sequence_static_context + raw_kline_context_no_alpha_prior_v1 + target_norm_head_constraint_v1`.
- Do not run Stage 39 for this plan.
- If continuing this line, focus on why sector-relative/regime improves validation and hit lift but hurts test rank/spread/month stability: regime split diagnostics, feature scaling/interaction ablations, and hit-lift-preserving rank regularization are better next targets than expanding architectures.
