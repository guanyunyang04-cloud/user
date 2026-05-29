# Alpha Multi-Horizon Stage 3D-3F Sector Input Repair Scout - 2026-05-29

## Verdict

- Status: `stage33_completed`, `stage34_completed`, `stage35_completed / no input-or-architecture upgrade / shadow-only`.
- Mainline: `alpha_multi_horizon_utility_policy_v1`.
- Study root: `daily_research/output/path_policy/studies/mh_stage33_sector_input_repair_scout_20260529_01/`.
- Fixed target/loss baseline: `decision_utility_v1` output with `target_norm_head_constraint_v1`.
- Final decision: sector input construction is repaired, but `raw_kline_context_sector_v1 + gru_sequence_static_context` failed the final 3-seed upgrade gate. Keep the Stage 3 baseline as `gru_sequence_static_context + raw_kline_context_no_alpha_prior_v1 + target_norm_head_constraint_v1`.
- Active artifact impact: `daily_research/output/active_execution_strategy.json remains unchanged`.
- Boundary: all runs are research / shadow-only; no allocator, replay, paper, live/default, broker, production root, active artifact, or promotion changes.

## Scope And Repair

- Dataset: `policy_input_bundle__7c8f58d851bce8179e1e9e2d`.
- Universe scope: `full_rolling_liquid500`.
- Sector board view: `policy_sector_board_view__ed15b2873f544e9e9b24aae5`.
- Fixed horizons: `1,2,3,5,8,10,15,20,30`.
- Stage 3D scout seed: `7`; Stage 3E confirmation seed: `11`; Stage 3F final seed: `19`.
- Stage 33 driver now explicitly passes `--sector-board-view-id policy_sector_board_view__ed15b2873f544e9e9b24aae5` and `--sector-board-view-kind latest_static_snapshot`.
- The sector anchor manifest is now valid full-pool sector input: `feature_profile=raw_kline_context_sector_v1`, `sector_context_feature_count=4`, `source_sector_board_view_id=policy_sector_board_view__ed15b2873f544e9e9b24aae5`, `source_pool_view_kind=rolling_liquidity`, `source_pool_view_name=rolling_liquid500`, `sample_count_by_role.train=452034`, `feature_store_shape=[1699,2430,192]`.

## Stage 3D Scout Results

Validation-ranked scout vs Stage 2.8 seed7 GRU raw baseline:

| Candidate | Composite | Validation gate | Test veto | Test rank / spread / hit | Stage 3E |
|---|---:|---|---|---:|---|
| `sector_gru_sequence_static_context` | `1.157877` | pass | no | `0.081391 / 0.029579 / 0.014760` | yes |
| `sector_stock_mixer_sequence` | `0.995857` | pass | no | `0.058370 / 0.022876 / 0.006774` | no |
| `sector_sector_slot_mixer_sequence` | `0.777465` | pass | no | `0.024494 / 0.004384 / 0.005189` | no |

Interpretation: sector metadata is now genuinely entering the model. In single-seed scout, sector GRU was the only strong candidate. Stock/slot improved concentration behavior but did not beat the strong-candidate threshold.

## Stage 3E Confirmation

- Candidate: `sector_gru_sequence_static_context`.
- Seed11 training tag: `mh34_confirm_sector_gru-sequence-static-context_seed11_20260529_01`.
- Stage 3E comparison status: `completed`.
- Stage 3E result: passed and advanced to Stage 3F.
- Two-seed test summary: rank IC `0.076729`, spread `0.030117`, hit lift `0.010341`, monthly positive rate `0.818182`, max negative months `2`.
- Advancement was driven by concentration/gap improvement, not by higher rank/spread/hit versus raw baseline.

## Stage 3F Final Confirmation

- Candidate: `sector_gru_sequence_static_context`.
- Seed19 training tag: `mh35_final_sector_gru-sequence-static-context_seed19_20260529_01`.
- Final comparison summary: `stage35_final_sector_confirmation_summary.json`.
- Final status: `input_or_architecture_upgrade_allowed=false`, `evidence_grade_input_or_architecture_pass=false`.

3-seed test comparison, `pred_decision_score`:

| Metric | Stage 2.8 GRU raw baseline | Sector GRU candidate | Interpretation |
|---|---:|---:|---|
| rank IC mean | `0.097677` | `0.070342` | worse |
| rank IC min | `0.083206` | `0.057570` | still positive |
| top-bottom spread mean | `0.037694` | `0.028569` | worse |
| spread min | `0.028098` | `0.025475` | still positive |
| hit lift mean | `0.016526` | `0.005616` | worse |
| hit lift min | `0.010771` | `-0.003834` | fails all-seed positivity |
| mean monthly positive rate | `0.878788` | `0.818182` | passes threshold |
| max negative months | `2` | `2` | passes threshold |
| long horizon share mean | `0.815462` | `0.243881` | much lower |
| 30d concentration mean | `0.776934` | `0.216956` | much lower |
| pred/future horizon gap mean | `15.618355` | `12.474114` | improved |

Validation comparison was strong for sector GRU: validation rank/spread/hit `0.145520 / 0.044618 / 0.027411`, monthly positive rate `1.000000`, negative months `0`, 30d concentration `0.578907`, gap `13.923030`.

## Interpretation

- The previous Stage 3A-3C sector blocker was real and is now repaired. `raw_kline_context_sector_v1` no longer silently produces zero sector context when the driver requests the registered sector-board view.
- Sector input helps the horizon chooser side: it materially reduces 30d concentration and pred/future horizon gap.
- Sector input did not improve the ranking/trading utility side on final 3-seed test. The candidate failed the full upgrade gate because test hit lift min became negative and rank/spread/hit means were lower than the Stage 2.8 raw baseline.
- Current evidence points to "sector context is useful diagnostic signal, but the naive sector numeric context is not yet a better production research baseline." It is not evidence that sector information is useless.

## Next Action

- Keep active execution unchanged.
- Keep Stage 3 baseline as `gru_sequence_static_context + raw_kline_context_no_alpha_prior_v1 + target_norm_head_constraint_v1`.
- Do not upgrade to sector GRU, sector stock mixer, or sector slot mixer from this run.
- Use sector features as diagnostics for horizon concentration and gap. If pursuing sector input further, redesign sector feature scaling/interaction rather than simply appending four sector context columns.
- Potential next targeted research: sector context ablation with rank-preserving regularization, or cross-sectional sector-relative features that target hit lift directly, still using single-seed scout followed by seed confirmation only for clear winners.
