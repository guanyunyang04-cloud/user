# Path Policy Seq100 Exit-Policy And Price-Basis Audit Result

Date: `2026-07-13`

Study family: `seq100_candidate_complete_exit_policy_audit`.

Status: `completed / winner_null / qdp_adjust_factor_blocker / no_execution_change`.

Active artifact impact: unchanged.

## Verdict

- The zero-training audit completed on the frozen candidate-complete v3
  rankings, candidate universe, entry/exit masks, and cost contract.
- Replacing the predicted exit with a fixed exit does not rescue executable
  Top3 ranking. The best diagnostic fixed rule is day 2, but its equal-year
  base alpha remains negative for all four years: `-1.44%` for baseline and
  `-1.36%` for hard-ST.
- A per-candidate, hindsight executable oracle also has negative Top3 alpha:
  `-1.64%` for baseline and `-2.01%` for hard-ST. Exit timing is therefore not
  the only defect.
- Executable-oracle alpha turns positive at Top5/Top10. At a common CNY
  `100,000` per-name allocation, Rank4-5 oracle alpha is `+4.60%/+5.89%` and
  Rank6-10 is `+6.42%/+6.45%` for baseline/hard-ST, while Rank1 is
  `-3.98%/-4.75%`. The score's extreme upper tail is inverted in executable
  space.
- The inversion is primarily a price-basis/data-quality failure. Baseline
  Rank1 has `+85.55%` mean back-adjusted best-exit return but only `+18.40%`
  raw-price unconstrained maximum return; adding raw sellability leaves it at
  `+18.41%`. The gap exists before exit-policy or transaction-cost effects.
- QDP active and the PIT research view contain invalid daily
  `back_adjust_factor` variation unrelated to corporate actions. `600076.SH`
  is a direct counterexample: it has no recorded corporate action after 2019,
  while its 2024 factor changes almost every trading day and creates implied
  adjustment ratios as high as `8.02x`.
- Do not train the planned soft-exit candidate, resume Q-only, promote Top10,
  freeze a champion, or change active execution. First repair and validate the
  QDP adjustment-factor substrate, rebuild an additive source pack, and rerun
  the zero-training bridge.

## Audit Contract And Evidence Identity

The audit entrypoint is
`daily_research.path_policy.seq100_exit_policy_audit`. It performs no model
training or checkpoint selection.

- Source study:
  `daily_research/output/path_policy/studies/seq100_candidate_complete_development_walkforward_20260712_v3`
- Source selection SHA-256:
  `3242fb6817374f5e133bd09cb751b42bb42a40df6b5a80f67d5bf709d8585083`
- Source pack:
  `daily_research/data/research_store/sequence_pack/qdp_v2_seq100_path60_todayclose_candidate_complete_2012_2025_v7/manifest.json`
- Source pack SHA-256:
  `710b0421e554c31912ef249ca0a3df8fc1b0b3ba8b595a06784cae8fa34b7bda`
- Cost-contract SHA-256:
  `f682af7908a6b95b389475171973539548170e732727e3617a84c872f6179289`
- Audit root:
  `daily_research/output/path_policy/studies/seq100_candidate_complete_exit_policy_audit_2022_2025_v1`
- Audit contract SHA-256:
  `f3e56321a1cbdab9c28e43d6b9842f6fd44a791bc91adfcce71fe944c75b7910`
- Reproducible price-basis diagnostic entrypoint:
  `daily_research.path_policy.seq100_exit_policy_price_basis_diagnostic`
- Price-basis diagnostic summary SHA-256:
  `078847f965012c66b407fab5e39d2f67ff12ffe19e2862016164cc9e53bd1309`

The run covered baseline and hard-ST, development years `2022..2025`, Top
`1/3/5/10`, base and double-slippage stress costs, and the following policies:

- Primary fixed exits: day `5/10/20/40/60`.
- Diagnostic fixed exit: day `2`.
- Frozen current predicted exit.
- Per-candidate cost-after executable oracle over planned days `2..60`.

All policies use the v3 execution contract: D+1 raw-open entry, no unfilled
replacement, T+1 minimum exit, raw-close sell, blocked-exit retry through
absolute D+80, 100-share lots, minimum commission, dated stamp tax, transfer
fee, and base/stress slippage. The terminal recovery fraction is zero.

Two evidence layers were produced:

1. Stateless daily Top1/3/5/10 cohorts against the policy-specific full
   candidate universe.
2. Stateful frozen daily Top3, maximum three positions, one-third target
   allocation, cash allowed, and no unfilled replacement.

The completed artifacts contain `124,032` daily rows, `512` yearly rows,
`128` equal-year rows, `128` stateful portfolio rows, and `19,380` frozen
Top10 price-basis bridge rows. Every audit key is unique, every policy group
has complete date coverage, and the executable oracle dominates every fixed
policy candidate by candidate. Runtime was `522.655` seconds behind the 1 GiB
memory guard; lowest available physical memory was `6.662 GiB`.

The protected QDP active SHA-256 remained
`E56F72A6CBA8BCF86055817F6A0EC5E7391271FB3C27B4D628C3ABC62944051E`.
`daily_research/output/active_execution_strategy.json` was absent before and
after the audit.

## Stateless Top3 Exit Decomposition

Values are four-year equal-weight means under base costs.

| Profile | Policy | Selected return | Universe return | Alpha | Selected terminal rate | Resolved exit day |
|---|---|---:|---:|---:|---:|---:|
| baseline | fixed day 2, diagnostic | -1.59% | -0.15% | -1.44% | 1.58% | 3.55 |
| baseline | fixed day 5 | -4.34% | -0.04% | -4.30% | 4.82% | 9.21 |
| baseline | predicted | -1.03% | +0.80% | -1.83% | 3.30% | 22.67 |
| baseline | executable oracle | +15.56% | +17.20% | -1.64% | 1.58% | 29.07 |
| hard-ST | fixed day 2, diagnostic | -1.51% | -0.15% | -1.36% | 1.38% | 3.38 |
| hard-ST | fixed day 5 | -4.52% | -0.04% | -4.48% | 4.71% | 9.09 |
| hard-ST | predicted | -6.92% | +1.07% | -7.99% | 9.87% | 45.41 |
| hard-ST | executable oracle | +15.19% | +17.20% | -2.01% | 1.38% | 28.86 |

Day 2 materially improves hard-ST relative to its predicted policy, so the
hard-ST exit head is genuinely harmful. It does not create positive alpha.
All five primary fixed horizons are worse than day 2 at Top3. Long holding
also concentrates terminal loss: selected terminal recovery rises to
`8.77%` for baseline and `10.35%` for hard-ST at the longer horizons.

Stress costs do not change the verdict. Baseline Top3 predicted and oracle
stress alpha are `-1.82%` and `-1.63%`; hard-ST values are `-7.97%` and
`-2.00%`.

## TopK And Rank-Tail Diagnosis

Equal-year base alpha by cumulative TopK:

| Profile | Policy | Top1 | Top3 | Top5 | Top10 |
|---|---|---:|---:|---:|---:|
| baseline | predicted | -2.89% | -1.83% | -1.24% | -0.06% |
| baseline | executable oracle | -4.08% | -1.64% | +0.86% | +3.66% |
| hard-ST | predicted | -10.36% | -7.99% | -5.26% | -2.28% |
| hard-ST | executable oracle | -4.85% | -2.01% | +1.15% | +3.82% |

To remove TopK allocation differences, the frozen Top10 names were replayed
at a common CNY `100,000` per-name allocation. Exact marginal rank-bucket
alpha is:

| Profile | Policy | Rank1 | Rank2-3 | Rank4-5 | Rank6-10 |
|---|---|---:|---:|---:|---:|
| baseline | predicted | -2.89% | -1.31% | -0.36% | +1.12% |
| baseline | executable oracle | -3.98% | -0.38% | +4.60% | +6.42% |
| hard-ST | predicted | -10.36% | -6.80% | -1.18% | +0.70% |
| hard-ST | executable oracle | -4.75% | -0.54% | +5.89% | +6.45% |

The rank tail also has a clear execution-quality gradient. Baseline entry fill
rises from `91.84%` at Rank1 to `98.64%` at Rank6-10; predicted terminal loss
falls from `3.92%` to `1.05%`. For hard-ST, predicted terminal loss falls from
`11.66%` at Rank1 to `2.93%` at Rank6-10.

This is not evidence for deploying Rank6-10. It shows that the score's most
extreme values are contaminated and that cumulative Top10 can hide the
failure of the intended Top1/Top3 action surface.

## Price-Basis Bridge And QDP Defect

The source opportunity and path labels use back-adjusted OHLC, whereas the v3
execution replay uses raw entry/exit prices. The following bridge uses the
same frozen selected names and compares the source back-adjusted best-exit
label with a more generous raw-price maximum over days 2..60.

| Profile | Rank bucket | Back-adjusted best-exit label | Raw unconstrained max | Raw sellable max | Implied factor ratio >1.1 |
|---|---|---:|---:|---:|---:|
| baseline | Rank1 | +85.55% | +18.40% | +18.41% | 63.27% |
| baseline | Rank2-3 | +41.83% | +19.26% | +19.31% | 38.10% |
| baseline | Rank4-5 | +41.84% | +23.11% | +23.10% | 27.59% |
| baseline | Rank6-10 | +34.03% | +24.75% | +24.72% | 15.79% |
| hard-ST | Rank1 | +81.87% | +17.45% | +17.43% | 63.06% |
| hard-ST | Rank6-10 | +33.99% | +24.72% | +24.70% | 17.72% |

Raw sellability barely changes the raw maximum. The dominant drop is between
the back-adjusted label and raw price itself, and it is strongest exactly
where the model score is highest.

`600076.SH` proves that this is not merely an execution engine omitting a
legitimate 2024 share entitlement:

- QDP `corporate_actions__a415e50cd459b3d93cc76e5a` records only 2018 and
  2019 cash dividends for this symbol, with no 2024 action.
- Raw 2024 prices are continuous and sourced from
  `traditional_baostock_pit_snapshot`.
- Active factor dataset `adjust_factor__4e0e31d3c1fd34bfcfa7a7dd` reports
  `back_adjust_factor` values `0.229630` on 2024-06-20, `0.530612` on
  2024-06-28, `1.123596` on 2024-07-02, and `1.387755` on 2024-07-03.
  These rows are sourced from the external Tonghuashun factor sidecar.
- The PIT factor dataset `adjust_factor__9d6feb2a5ba7f15a7635b42f` copies
  those exact rows as `active_qdp_dense`.
- Frozen high-rank samples for this symbol show back-adjusted best-exit
  returns above `+1,100%` while the same raw exit is about `+53%`, producing
  an implied adjustment ratio up to `8.0248x`.
- In the 120-calendar-day counterexample window, the active factor changes on
  `73/80` observed rows while the corporate-action count is zero.

The previous Gate-0 checks proved factor presence, positivity, provenance, and
key coverage. They did not prove temporal or event consistency. Therefore the
candidate-complete v7 pack is structurally complete but not price-semantics
clean, and its large opportunity alpha cannot be treated as economic alpha.

## Stateful Portfolio Evidence

The stateful layer uses frozen Top3, maximum three positions, and one-third
target allocations. These values reproduce the current raw-price execution
contract; because the ranking labels are factor-contaminated, they are a
blocking diagnostic rather than a trustworthy estimate of economic live PnL.

| Profile | Policy | Mean net return | Mean max drawdown | Positive years | Worst year |
|---|---|---:|---:|---:|---:|
| baseline | predicted | -54.80% | -64.93% | 0/4 | -99.18% |
| baseline | best non-oracle fixed, day 40 | -47.65% | -69.77% | 1/4 | -100.00% |
| baseline | executable oracle, diagnostic | +34.53% | -47.73% | 2/4 | -59.89% |
| hard-ST | predicted | -54.76% | -75.41% | 1/4 | -99.99% |
| hard-ST | best non-oracle fixed, day 60 | -55.37% | -75.43% | 1/4 | -100.00% |
| hard-ST | executable oracle, diagnostic | +8.74% | -51.69% | 2/4 | -83.05% |

No non-oracle policy is remotely eligible for promotion. Even the hindsight
oracle has only two positive years and severe worst-year loss.

## Verification

- Focused audit and authoritative candidate-execution tests: `12 passed`.
- Synthetic coverage includes planned exit, blocked/deferred exit, terminal
  recovery, unfilled cash, lot rounding, minimum fees, dated stamp tax, and
  base/stress slippage.
- On a real 2,939-name candidate date, vectorized fixed-policy replay exactly
  matched the authoritative scalar implementation for resolved exit day,
  shares, net return, and execution cost under both tested allocations and
  cost scenarios; maximum numeric error was zero.
- On the frozen Top100 source evidence, predicted-policy base and stress
  resolved day, shares, net return, and cost also matched exactly.
- QDP active hash and active-execution absence were guarded before and after
  the full run.

## Blockers

- QDP adjustment-factor temporal/event consistency is invalid: active and PIT factors contain daily changes that are not explained by corporate actions.
- Candidate-complete v7 is blocked: its back-adjusted opportunity labels and trained models cannot support a new champion or economic opportunity claim.
- Corrected Top3 evidence is absent: no repaired pack currently shows stable positive executable-oracle alpha.

## Next Allowed Actions

- Add a QDP adjustment-factor quality gate requiring piecewise stability between corporate actions, event-day adjusted-return continuity, and no arbitrary daily raw-price tracking; quarantine offending symbols/providers.
- Trace and repair the external Tonghuashun factor sidecar normalization, then build a new immutable QDP dataset and independent quality proof without overwriting evidence or changing active merely to make this study pass.
- Build an additive PIT view and new sequence pack with factor temporal/event validation in addition to key coverage and positivity.
- Define one economically coherent execution price basis: raw prices for limits/fills plus validated share/cash entitlement handling, or an equivalent validated total-return representation.
- Re-evaluate the frozen v3 score on corrected labels and rerun fixed/predicted/oracle auditing before retraining; retire the ranking target if corrected executable Top3 oracle alpha remains non-positive.
- Register one soft-exit additive study only if the corrected zero-training audit shows stable positive executable Top3 oracle alpha.

Until those steps complete: `winner=null`, baseline is a historical control,
hard-ST remains rejected, Q-only remains terminated, global-tail remains
paused, and execution remains frozen.
