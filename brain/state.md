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

## Next action

Freeze the five pre-entry output blocks. Design the post-entry state-update
target from original entry information plus realized D1/D3/D5 path, without
first introducing an entry fusion score or account policy.
