# Current state

Updated: 2026-07-28

## Objective

Determine which target and training objective best express pre-entry upside
opportunity. Repository workflow simplification is complete; scientific scope,
data, checkpoints, predictions, and generated features were retained.

## Completed feature audit

`seq100_mfe_feature_family_audit_v1` completed all 60 LightGBM tasks for
`mfe_10` and `mfe_20` over the 2023-2025 folds. Results are under
`daily_research/output/path_policy/studies/seq100_mfe_feature_family_audit_v1/`.

- No family improved global rank quality and the strongest 5% tail together
  strongly enough to pass every original gate; the negative control was exactly neutral.
- `traditional_indicators` improved D20 Rank IC in all three years and passed
  Rank-IC FDR, but reduced Top-5% opportunity quality in 2023 and 2025.
- `breakout_retest_levels` improved D20 Top-5% mean and tail hit rate in all
  three years, while its global Rank IC was not significant.
- `turnover_cost_proxy` similarly improved the D10 Top-5% mean and tail hit
  rate in all three years without significant global Rank-IC improvement.
- The folds are deliberately reused recent decision folds, not a pristine holdout.

## Current scientific position

- Pre-entry opportunity is best represented first by separate `mfe_10` and `mfe_20` scalar heads.
- `state_10` remains the core entry-state challenger; `state_20` is secondary.
- D3/D5 path probabilities remain useful for a later post-entry state-update model, not as independent entry rankings.
- Opportunity, path state, and pre-peak adverse movement should remain separate outputs until evidence supports a joint loss.
- No holding period, exit rule, slot count, leverage, stop loss, or final entry score has been selected.

## Next action

Do not add all feature families or start a broad state-label audit. Design a
focused objective-alignment comparison: D10 plus the turnover-cost proxy, D20
plus breakout/retest levels, and D20 traditional indicators as the broad-rank
comparator. Evaluate global ranking and strongest-candidate enrichment as
separate roles rather than requiring one feature family to optimize both.
