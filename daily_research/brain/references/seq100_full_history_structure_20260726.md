# Seq100 full-history structure audit - 2026-07-26

## Status

This record supersedes the preliminary interpretation of the full-history
structure probe. The measurements remain useful as discovery evidence, but the
three proposed headline decisions do not survive audit:

1. Reject the claim that one slot is structurally impossible or carries a 3%
   annual total-loss probability.
2. Retain diversification only as a directional hypothesis; reject the measured
   slot-count effect size because it is materially driven by the terminal mark.
3. Reject `turn_low30 & ret20_mid` as a successor baseline. It is an
   unconfirmed, post-selected hypothesis.

No successor study, target, horizon, slot count, leverage, or stop rule is
selected by this evidence.

## Scope and frozen-year boundary

- Read-only against the protected
  `daily_research/data/research_store/seq100_pit_l35v2_v1/` pack.
- The terminal re-audit hard-cuts all outcome observations at `2025-12-31`.
  It therefore uses 3,678 signal dates from `2010-07-15` through `2025-09-02`,
  the last date whose complete D80 execution window ends in 2025. No 2026 price,
  signal, or outcome row enters the recorded result.
- Daily membership is signal-day `candidate_eligible`; entry fill is signal-day
  `entry_buyable`, which is the same file as the persisted `entry_filled` mask.
- The fixed-20-day audit contains 8,236,774 executed entries across the full PIT
  main-board universe. The fully observed D324 comparison contains 7,488,320
  entries and ends with signal date `2024-08-29`.
- Entry, legal-exit deferral, lots, commissions, stamp duty, transfer fee, and
  slippage reuse `seq100_exit_policy_audit` and the pack-bound
  `a_share_round_trip_cashflow_v1` contract.
- Group and basket features are signal-day values from `daily_state` and
  `turnover`; no future mask was used to form a group.

The original exploratory scripts admitted signal dates through `2025-12-31`
because the pack contains an 80-day tail into 2026. Those late rows are excluded
from every audited number below. Their prior headline values may be quoted only
to explain what was invalidated.

## Terminal-convention audit

The preliminary probe wrote any position with no legal sale from planned D20
through D80 to zero at D80. It then interpreted all such marks as delistings.
That interpretation is false.

Under the strict pre-2026 sample there are 21,227 D80 zero marks:

| diagnostic | count / value |
|---|---:|
| long suspension during D20-D80 | 20,560 (96.86%) |
| any suspension during D20-D80 | 97.67% |
| `is_delisted` during D20-D80 | 0 |
| security entity ended by D80 in QDP `symbol_history` | 0 |
| security entity ended by observed endpoint | 0 |
| recovered at a legal close during D81-D324 | 93.90% |
| recovery day, median / p90 / max | D113 / D163 / D324 |
| full D324 window observed but still unresolved | 5.77% |
| right-censored at `2025-12-31` | 0.33% |
| registered code change by D80 / observed endpoint | 19 / 38 |

The code-change rows are only `000043.SZ` and `000022.SZ`; their security
entities continue under registered successor codes. Treating the old ticker as
a zero-value security is also incorrect, although its 38 rows are too few to
explain the aggregate distortion.

The corrected diagnostic searches only D81 onward. The earlier process log that
reported a median recovery day of D1 was defective because it restarted the
search before the planned D20 exit. It is superseded by the strict run.

## Effect on the growth headline

The alternative convention holds a D80-unresolved position to its first legal
sellable close through D324. If the observation ends first, it keeps a zero mark
at the observed endpoint, so the all-date result is a conservative lower bound.
The full-D324 cohort has no such censoring.

| sample | zero at D80 annual log | extended annual log | zero mean net | extended mean net |
|---|---:|---:|---:|---:|
| all strict pre-2026 rows | -0.3084 | -0.0835 lower bound | +0.0676% | +0.3041% |
| full D324 comparable cohort | -0.3732 | -0.1260 | not retained | not retained |

The original all-date value was `-0.2939`; changing only the terminal convention
recovers most of that deficit. The remaining negative log growth is consistent
with variance drag, costs, and unresolved capital, but it does not prove that a
single-slot selected strategy is impossible. The prior conversion of a 0.25%
per-trade artificial zero rate into an approximately 3% annual probability of
losing the whole account is withdrawn.

The statistic is
`sum(log(1 + net_return)) / sum(holding_days) * 244`.
Writing it as `mean(log(1 + net_return)) / mean(holding_days) * 244` is exactly
the same estimator. Splitting signal dates into non-overlapping start offsets
does not turn it into an account simulation and supplies no independent check
of cash occupancy.

## Slot-count verdict

The earlier same-cohort bootstrap reported roughly +0.29 annual log growth from
one to two names, versus only about +0.05 from two to fifty. Averaging two names
necessarily dilutes the artificial D80 zero mark, so the terminal convention
drives a material and unmeasured share of that effect. The bootstrap also has no
cash release, order queue, overlapping cohorts, or correlated account path.

Diversification remains economically plausible, but neither a minimum slot
count nor its benefit magnitude is established. The next valid test is a
finite-capital replay with corrected terminal holding, code-change continuity,
real cash constraints, and explicit slot allocation. The existing implementation
surface is `daily_research/path_policy/seq100_finite_capital_backtest.py`.

## Feature-shape verdict

The original zero-at-D80 probe observed lower turnover and lower volatility as
monotone survivors, and an inverted-U relation for 5-day and 20-day past return.
Those shapes span 16 years and remain useful hypotheses, but their reported
annual-log levels share the contaminated terminal estimator. They must be
recomputed after execution semantics are corrected before entering a target,
feature freeze, or baseline.

One data-quality result is independent of the return convention:
`distance_to_20d_high` and `drawdown_from_20d_high` are bit-identical in the
pack. They represent one current signal, not two independent features.

## Basket verdict

`turn_low30 & ret20_mid` produced the best inspected basket headline:
annual log `+0.032`, 11 positive years out of 16, and mean cohort size about 360.
It is not a valid baseline because:

- 12 filters were inspected on the same full history before this row was named;
- the result did not clear a statistical-significance gate after inspection;
- the 20 start-offset values are transformations of the same overlapping
  cohort estimator, not independent account paths;
- the basket still uses the artificial terminal convention; and
- all concrete thresholds consume burned discovery data.

The correct status is `hypothesis_pending_account_validation`, not benchmark or
successor requirement.

## Consequence for successor design

The evidence does not yet answer what the model should learn. It narrows the
next work to three ordered questions:

1. Correct terminal holding and registered code-change continuity in the
   evaluation path without consuming 2026.
2. Re-estimate multi-horizon raw return distributions and signal-day group
   structure under that execution contract, without selecting a horizon from
   the frozen year.
3. Use a finite-capital account replay to measure how selection skill, exits,
   and slots interact. Only then can a new study freeze one target and metric.

The separate exit-rule comparison still supports using a simple fixed terminal
exit as a controlled label measurement: no inspected adaptive rule beat the best
fixed horizon on its worst burned fold. It does not select which fixed horizon,
and it does not settle account-level exits.

2023-2025 remain burned discovery years. 2026 remains untouched confirmation
material. A successor must be a new study contract.

## Evidence bindings

- Pack manifest SHA-256:
  `4f8417c2c382c12b9a4055596c572d39988a5d47e91e30736e313da84634fafd`
- QDP active manifest SHA-256:
  `ff39ae379d6edb09695430e7b0093c19d8ee2471b6b9bf1b5dd8492e5a23c8ff`
- Strict audit script: `tmp/seq100_terminal_convention.py`, SHA-256
  `034412426e7cb99794c504792b7f4ae5580ded8650717546fc50b1c4348795cd`
- Strict audit log: `tmp/seq100_structure/terminal_convention_run_v4_strict_pre2026.log`,
  SHA-256 `c535ebdf148fc69d459150d41b1a33f5d0cdcb825d5c2841a9c6017334559715`
- Terminal diagnostics: `tmp/seq100_structure/zero_recovery_diagnostics.parquet`,
  SHA-256 `8e4aed07c282d15d97cf869a00f653b4b9622045d3811eee06dbf48931f8f721`
- Preliminary moments / slot / basket artifacts, retained only as discovery
  process material: `853fb030...79319`, `7ab518ae...ca5b`,
  `d779a5a4...fe870`.
