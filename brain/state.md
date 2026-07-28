# Current state

Updated: 2026-07-28

## Objective

Build a pre-entry selector that distinguishes upside opportunity from path
quality and risk, followed later by a separate post-entry state-update model.
Repository workflow simplification is complete; scientific scope, data,
checkpoints, predictions, and generated features were retained.

## Completed objective alignment

`seq100_mfe_objective_alignment_v1` compared source Huber, smooth upper-tail
weighted Huber, and daily LambdaRank for `mfe_10` and `mfe_20`. All 12 new
LightGBM tasks completed over the 2023-2025 folds. The compact result is
`daily_research/research_records/seq100/seq100_mfe_objective_alignment_v1/result.json`.

- Keep the original date-equal Huber as the primary continuous MFE objective.
- Tail weighting found slightly faster opportunities but also deeper pre-peak
  adversity, worse endpoints, and biased magnitude estimates; it is not the default.
- LambdaRank improved broad Rank IC in every horizon-year comparison, but
  consistently weakened the strongest realized MFE tail. Its smoother paths
  support a separate path-quality output rather than replacing the MFE head.

## Current scientific position

- Pre-entry opportunity is best represented first by separate `mfe_10` and `mfe_20` scalar heads.
- `state_10` remains the core entry-state challenger; `state_20` is secondary.
- D3/D5 path probabilities remain useful for a later post-entry state-update model, not as independent entry rankings.
- Opportunity, path state, and pre-peak adverse movement should remain separate outputs until evidence supports a joint loss.
- No holding period, exit rule, slot count, leverage, stop loss, or final entry score has been selected.
- The completed feature-family audit remains reusable: D10 turnover-cost helped
  the strongest tail, D20 breakout/retest helped the strongest tail, and D20
  traditional indicators helped broad ranking but hurt the tail in two years.

## Next action

Do not run more MFE objective variants or retrain feature models that already
exist. Reuse existing Huber predictions to make a no-retraining role synthesis
of D10 turnover-cost, D20 breakout/retest, and D20 traditional indicators.
Then test whether `state_10` and an independent adverse-path output improve
selection conditionally inside high predicted MFE candidates.
