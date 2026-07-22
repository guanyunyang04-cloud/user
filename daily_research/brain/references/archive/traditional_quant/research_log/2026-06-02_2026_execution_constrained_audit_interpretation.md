# 2026-06-02 2026 Execution-Constrained Audit Interpretation

## Question

Does a first-pass execution constraint make the 2026 candidate-input failure better, worse, or merely noisier?

## Evidence

- Snapshot: `baostock_v2_pit_20160101_20260601_stockbasic_fixed`.
- History warm-up: `2025-01-01`.
- Evaluation window: `2026-01-01` to `2026-06-01`.
- Signals: `multifactor_rolling_ic_weighted_score`, `multifactor_low_corr_rank_score`.
- Portfolio: weekly Top-100, 5-day horizon, non-overlapping, buffer `1.0`.
- Execution approximation: `limit_threshold=0.095`; next-session limit-up or non-tradeable entries are not filled, unfilled capital remains cash, target exits blocked by limit-down/non-tradeable status are delayed to the next sellable close.
- Experiment output: `traditional_quant_research/output/experiments/multifactor_baseline/multifactor_baseline_20260602_133345`.
- Auto research log: `traditional_quant_research/brain/references/research_log/2026-06-02_multifactor_2026_execution_constrained_audit.md`.

## Result

Execution constraints made both candidate inputs worse.

| Signal | 0 bps Annualized | 0 bps Sharpe | 30 bps Annualized | 30 bps Sharpe | Exit Delays |
|---|---:|---:|---:|---:|---:|
| `multifactor_rolling_ic_weighted_score` | -0.137811 | -0.853836 | -0.300674 | -2.184832 | 2 |
| `multifactor_low_corr_rank_score` | -0.181404 | -1.013576 | -0.353211 | -2.317064 | 3 |

Important detail: `blocked_entry_count=0` and `entry_limit_up_count=0` for both tested signals. The constrained degradation is not caused by missed limit-up buys. It is caused by delayed exits from limit-down/non-sellable target exits.

## Monthly Diagnosis

Compared with the unconstrained basket audit:

- January positive returns were materially reduced because constrained non-overlap and delayed exits changed the trade schedule.
- March remained weak for rolling IC and became much worse for low-corr.
- April stayed strong.
- May remained weak but slightly less negative on the constrained slice.
- A new June exit period appeared because delayed exits pushed some holdings beyond the original May/June boundary, and those delayed exits were strongly negative.

This confirms the prior caution: the 2026 problem is not just a ranking diagnostic issue. Once execution constraints are included, the current candidate inputs fail more clearly.

## Decision

Current status remains `candidate_input_watchlist`.

Strategy candidate count remains `0`.

Do not promote either candidate to strategy status.

Next best research steps:

- Add a constrained-vs-unconstrained comparison table to the standard report when `--execution-constraints` is enabled.
- Test whether buffer `1.5/2.0` reduces constrained high-cost damage.
- Add exposure caps or a simple regime filter before considering traditional ML.
- Keep execution-constrained horizon backtest as a required gate for future candidates.
