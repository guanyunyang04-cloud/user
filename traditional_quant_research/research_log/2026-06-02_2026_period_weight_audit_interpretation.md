# 2026-06-02 2026 Period And Rolling Weight Audit Interpretation

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-02_2026_period_weight_audit_interpretation.md`.


## Question

Why did the strongest phase-2/phase-3 candidate input fail the 2026 half-year stability gate?

This audit adds two missing diagnostics:

- Horizon-aligned monthly and quarterly slices by exit period.
- Rolling IC weight audit by month and quarter.

## Evidence

- Snapshot: `baostock_v2_pit_20160101_20260601_stockbasic_fixed`.
- Evaluation window: `2026-01-01` to `2026-06-01`.
- History warm-up window: loaded from `2025-01-01` so rolling IC weights have prior history before 2026.
- Experiment output: `traditional_quant_research/output/experiments/multifactor_baseline/multifactor_baseline_20260602_125706`.
- Auto research log: `traditional_quant_research/research_log/2026-06-02_multifactor_2026_warm_history_period_weight_audit.md`.

The earlier 2026-only run at `multifactor_baseline_20260602_125258` is not a reliable rolling-weight audit because it loaded only 2026 data. Its rolling fallback date rate was `0.6667`. With `--history-start-date 2025-01-01`, fallback rate dropped to `0.1888`, and 2026 months had `0.0` fallback.

## Findings

Under strict weekly Top-100, 5-day non-overlapping horizon, raw returns, 0 bps:

- `multifactor_rolling_ic_weighted_score` was positive in January, February and April, but negative in March and May.
- Quarterly view: rolling IC was roughly flat in `2026Q1` and negative in `2026Q2`.
- `multifactor_low_corr_rank_score` behaved similarly, but was stronger than rolling IC in this 2026-only evaluation.
- Buffer reduced turnover but did not fix alpha decay. For rolling IC, gross return was already negative before cost in 2026, so buffer only softens cost drag.

Key monthly rolling IC 0 bps, buffer `1.0`:

| Period | Mean Net Return | Interpretation |
|---|---:|---|
| 2026-01 | 0.011740 | Strong positive |
| 2026-02 | 0.003505 | Mild positive |
| 2026-03 | -0.009546 | Main Q1 damage |
| 2026-04 | 0.014858 | Strong rebound |
| 2026-05 | -0.013817 | Main Q2 damage |

Rolling IC weight drift around the weak months:

- March increased weights on `neg_volatility_20d_z` and `neg_amplitude_20d_z`, while `reversal_5d_z` fell.
- May kept elevated `neg_volatility_20d_z`, lowered `momentum_20d_z`, and lowered `log_amount_mean_20d_z`.
- This suggests the 2026 failure is not just transaction cost. It is a regime and exposure problem: the rolling weight model shifted toward volatility/amplitude defensiveness, but the selected Top-100 baskets still lost money in March and May.

## Decision

Current status remains `candidate_input_watchlist`, not a strategy candidate.

Strategy candidate count remains `0`.

The next research step should not be traditional ML yet. First add execution constraints and exposure diagnostics:

- Limit-up buy restriction and limit-down sell restriction approximation.
- Suspension holding behavior for existing positions.
- Monthly exposure audit for selected baskets: liquidity proxy, volatility, amplitude, reversal, and momentum.
- Re-run 2026 period slices after those constraints to see whether March/May failure is alpha decay, execution friction, or unintended exposure.
