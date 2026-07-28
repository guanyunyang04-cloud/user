# Current state

Updated: 2026-07-28

## Objective

Build a pre-entry selector that distinguishes upside opportunity from path
quality and risk, followed later by a separate post-entry state-update model.
Repository workflow simplification is complete; scientific scope, data,
checkpoints, predictions, and generated features were retained.

## Completed entry-role synthesis

`seq100_entry_role_synthesis_v1` reused 15 MFE and 9 state/risk OOS
predictions over the 2023-2025 folds. It trained no booster or meta model and
read no 2026 row. The compact result is
`daily_research/research_records/seq100/seq100_entry_role_synthesis_v1/result.json`.

- Use base plus `turnover_cost_proxy` for the `mfe_10` strong-candidate head.
- Use base plus `breakout_retest_levels` for the `mfe_20` strong-candidate head.
- `traditional_indicators` is a broad-ranking diagnostic, not a third MFE head.
- Keep `state_10` as the full low/mid/high probability vector. `P(high)` favors
  volatile opportunity, while expected state and low-state avoidance improve
  endpoint/path quality at an MFE cost; no single monotone state filter is selected.
- Keep `pre_peak_mae_10` and `pre_peak_mae_20` as separate adversity coordinates.
  Both reduce deep adverse paths, but neither is an opportunity-preserving hard filter.
- The current pre-entry contract has five output blocks: two MFE scalars, one
  state probability vector, and two horizon-specific adversity scalars.
- No fused score, retention threshold, holding model, exit rule, holding period,
  slot count, leverage, stop loss, or account policy has been selected.

## Completed post-entry incremental-information audit

`seq100_post_entry_incremental_information_v1` reused the frozen five-block
entry contract and constructed D1/D3/D5 hypothetical holding cohorts without
an interim survivor filter. It trained no booster or meta model, consumed only
2023-2025 outcomes, and read no 2026 row. The compact result is
`daily_research/research_records/seq100/seq100_post_entry_incremental_information_v1/result.json`.

- Continue recomputing the existing `mfe_10` and `mfe_20` heads each day. Within
  strong entry candidates, realized path and entry memory added no stable
  conditional information for either remaining-MFE target.
- Realized path added stable conditional information mainly for
  `pre_peak_mae_10` at D1, D3, and D5. `state_10` has much narrower supporting
  evidence and remains a secondary challenger. Regular-coverage evidence did
  not support a post-entry `pre_peak_mae_20` target.
- Entry-memory rank changes also add state/risk information beyond the current
  contract, so the challenger input should preserve both original-entry memory
  and the realized path.
- An existing position can legally sell at the first close after the new
  next-open anchor. This materially differs from the D2-first entry-label
  convention, so post-entry targets must be generated separately rather than
  reusing entry labels.
- Low-frequency tradability events are retained as sparse state alerts, not as
  evidence for a general path updater.
- The audit selects no continuation score, holding model, exit, switching rule,
  holding period, cost buffer, slot count, leverage, stop loss, or account policy.

## Next action

Run matched-capacity nested OOS challengers separately at D1, D3, and D5:

- A uses only the currently recomputed frozen entry contract.
- B uses A plus the supported original-entry memory and realized path/activity
  blocks.
- Train `pre_peak_mae_10` as the primary post-entry target and `state_10` as a
  secondary challenger. Do not train a new post-entry MFE head or a
  `pre_peak_mae_20` head at this stage.
- Use holding-specific next-open labels, identical learner capacity and
  evaluation support for A and B, and compare incremental OOS prediction only.
  Do not turn the comparison into an exit, switching, or account policy.
