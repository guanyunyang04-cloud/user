# Path Policy Seq100 Candidate-Complete Development v3 Result

Date: `2026-07-12`

Study family: `seq100_candidate_complete_development_walkforward`.

Status: `completed / winner_null / no_champion_freeze / no_execution_change`.

Active artifact impact: unchanged.

Audience: maintainers of the seq100 path-policy research line. This record
closes the v3 baseline-versus-hard-ST generation and defines the next
experiment boundary.

## Verdict

- All `8/8` registered jobs completed and passed evidence-integrity checks.
- Both profiles failed the pre-registered eligibility gates.
- `development_selection.json` records `winner = null`.
- Do not run `freeze-development`, promote either profile, or change the active execution artifact.
- Do not launch `global_tail_512` on this loss design; the dominant observed failure is predicted-exit realization, not lack of ex-post opportunity ranking.
- The next generation must first separate fixed, predicted, and oracle exit behavior, then test an execution-aligned soft exit objective as one core change.

## Blockers

- No profile passed the Top3, Top10, stress, and positive-year eligibility gates, so there is no champion that may be frozen.
- Predicted exit execution is negative despite strong ex-post opportunity ranking; exit-policy and loss alignment must be resolved first.

## Contract And Evidence Identity

The study consumes the approved contract
`seq100_candidate_complete_development_walkforward_contract_20260711_v1`.
The four years `2022..2025` are development/model-selection years; there is no
historical test set or historical outer audit. Each fold uses expanding
training data from 2012, an 80-trading-day maximum label-dependency purge, and
normalization fitted strictly before the development-year start.

- Source pack:
  `daily_research/data/research_store/sequence_pack/qdp_v2_seq100_path60_todayclose_candidate_complete_2012_2025_v7/manifest.json`
- Source pack SHA-256:
  `710b0421e554c31912ef249ca0a3df8fc1b0b3ba8b595a06784cae8fa34b7bda`
- Source rows: `8,204,961` supervised samples and `8,208,431`
  signal-day candidates across `3,509` dates and `3,404` PIT symbols.
- Study root:
  `daily_research/output/path_policy/studies/seq100_candidate_complete_development_walkforward_20260712_v3`
- Registry declared SHA-256:
  `58d91f2a5e937a238cad4b2bff96e97352f8d08016a8c6fe0191d246567f3d69`
- Registry file SHA-256:
  `d8d46621d79a6d3583987c719560872db7edbe3e9411f9ab50ab435a2b4d06c8`
- Result-ledger SHA-256:
  `2b547a5a427faa08cd809ecb6bdc84fce29f640a21f612d42c2ed0e29d714de2`
- Selection SHA-256:
  `3242fb6817374f5e133bd09cb751b42bb42a40df6b5a80f67d5bf709d8585083`

The registry body digest, all eight result digests, all 64 registered evidence
file digests, and 55 direct selection aggregates were independently replayed.
Every selection aggregate exactly equals the arithmetic mean of its four
ledger values. Candidate and executable-return coverage is `100%`, so the
negative verdict is not a missing-data artifact.

The predecessor v2 study remains retired under retirement SHA-256
`3c6c59f6a99339d715bb3a46a4d1998ad291fbcb005970fb852ce175426e5b8d`.
Its failure was FP16 overflow of the finite-count denominator
(`512 * 60 * 4 = 122,880`), not evidence about either model profile. No v2
training result was imported into v3.

## Training And Early Stopping

Both profiles used seed `7`, every eligible training row, batch size `512`, a
maximum of 10 epochs, and validation-loss early stopping on
`development_total_loss` with patience `2`, `min_delta = 0`, at least one
complete epoch, and best-checkpoint restoration. TopK metrics never selected a
within-run checkpoint.

| Profile | Development year | Best epoch | Completed epochs | Best optimizer step | Development loss | Stopped early |
|---|---:|---:|---:|---:|---:|---|
| baseline | 2022 | 1 | 3 | 11,064 | 0.119323 | yes |
| baseline | 2023 | 1 | 3 | 12,516 | 0.116117 | yes |
| baseline | 2024 | 1 | 3 | 13,968 | 0.125921 | yes |
| baseline | 2025 | 1 | 3 | 15,546 | 0.119493 | yes |
| hard_st | 2022 | 1 | 3 | 11,064 | 0.118691 | yes |
| hard_st | 2023 | 1 | 3 | 12,516 | 0.116395 | yes |
| hard_st | 2024 | 1 | 3 | 13,968 | 0.125380 | yes |
| hard_st | 2025 | 1 | 3 | 15,546 | 0.119855 | yes |

All eight folds independently selected epoch 1. This confirms that at least
one pass over all fold data is necessary and sufficient under the current
objective; it does not support selecting a partial-epoch checkpoint. The two
patience epochs added no validation-loss improvement. Total work was `318,564`
optimizer steps and `146,867,538` sample exposures.

All eight jobs ran behind a `1.0 GiB` available-physical-memory guard. The
lowest observed availability was `1.857..1.946 GiB`; no job crossed the guard.
The baseline/2022 wrapper recorded exit `120` only because its foreground
stdout transport closed after a 10-second shell timeout. All eight required
artifacts had already been written and were recovered without retraining. The
recovery record SHA-256 is
`842839d8493ea84ffc1ba6b0da87543922ea0590f04df1971f4bd9fabaf0f6b5`.

## Executable Development Results

Values below are equal-year, cost-after
`net_realized_plan_return_base_alpha`. They use the predicted exit plan,
manifest-bound fees and slippage, entry fill checks, T+1, blocked-exit retry,
and the conservative terminal rule.

| Profile | Year | Top1 | Top3 | Top5 | Top10 |
|---|---:|---:|---:|---:|---:|
| baseline | 2022 | -3.30% | -2.33% | -1.66% | -1.23% |
| baseline | 2023 | +0.20% | -1.44% | -2.66% | -2.25% |
| baseline | 2024 | -1.48% | -2.36% | -2.70% | -1.54% |
| baseline | 2025 | -6.98% | -1.20% | +2.05% | +4.77% |
| hard_st | 2022 | -18.33% | -17.04% | -14.35% | -10.36% |
| hard_st | 2023 | -9.95% | -9.63% | -8.36% | -4.73% |
| hard_st | 2024 | +0.54% | +4.81% | +8.64% | +8.00% |
| hard_st | 2025 | -13.70% | -10.09% | -6.98% | -2.04% |

Four-year equal-weight summary:

| Profile | Top1 | Top3 | Top5 | Top10 | Positive Top3 years |
|---|---:|---:|---:|---:|---:|
| baseline | -2.89% | -1.83% | -1.24% | -0.06% | 0/4 |
| hard_st | -10.36% | -7.99% | -5.26% | -2.28% | 1/4 |

The stress-cost Top3 means are also negative: `-1.82%` for baseline and
`-7.97%` for hard-ST. Baseline failed the positive mean Top3, mean Top10,
stress Top3, and three-positive-year gates. Hard-ST failed the same four gates.
Both passed the required execution-return and plan-coverage gate.

## Opportunity Versus Realized Exit

The model can rank names with strong ex-post path opportunity, but its
predicted exit plan does not realize that opportunity.

| Profile | TopK | Opportunity alpha | Executable alpha | Gap | Oracle regret |
|---|---:|---:|---:|---:|---:|
| baseline | 1 | +65.64% | -2.89% | 68.53 pp | 0.8348 |
| baseline | 3 | +39.87% | -1.83% | 41.70 pp | 0.5651 |
| baseline | 5 | +35.00% | -1.24% | 36.24 pp | 0.5124 |
| baseline | 10 | +25.07% | -0.06% | 25.13 pp | 0.4065 |
| hard_st | 1 | +62.29% | -10.36% | 72.65 pp | 0.8763 |
| hard_st | 3 | +38.75% | -7.99% | 46.74 pp | 0.6107 |
| hard_st | 5 | +34.67% | -5.26% | 39.93 pp | 0.5428 |
| hard_st | 10 | +25.52% | -2.28% | 27.81 pp | 0.4408 |

Baseline's mean development loss, rank IC, and path MAE are `0.120214`,
`0.118051`, and `0.097644`. Hard-ST slightly lowers mean loss to `0.120080`
and raises rank IC to `0.121047`, while executable Top3 alpha worsens by more
than six percentage points. Lower current loss is therefore not sufficient for
the action the model must ultimately take.

The hard straight-through path-value gradient makes the target's maximum/timing
choice sharper, but it does not model the discontinuous execution process:
predicted exit day, raw-price fillability, T+1, down-limit retry, costs, and
terminal loss. Its one positive year and three large negative years show that
it amplifies regime-sensitive timing error. Adding a global daily ranking tail
now would spend substantially more compute optimizing a value/exit interface
that has already failed.

## Next Allowed Actions

- Audit fixed, current predicted, and oracle executable exits on the frozen v3 candidates and cost contract.
- Train one execution-aligned soft exit candidate only if the audit confirms that exit timing is the dominant correctable gap.
- Test `global_tail_512` only after an execution-aligned candidate passes the pre-registered eligibility gates.

First run a frozen-prediction exit-policy audit on the same candidates and cost
contract. For each profile and Top1/3/5/10, compare:

1. Executable fixed close exits at predeclared horizons such as 5, 10, 20, 40,
   and 60 path days.
2. The current predicted-path exit day.
3. The best executable exit day as an oracle diagnostic only.

This decomposition answers whether the remaining signal is usable with a
simple holding rule, whether exit timing is the main defect, and how much of
the gap is irreducible ranking error. It must reuse the same signal-day
candidate universe, no rank replacement, T+1, blocked-exit retry, costs, and
terminal handling.

If fixed exits materially beat predicted exits, implement one new candidate
whose primary trainable action is a soft distribution over permissible exit
days. Optimize expected executable net return/value through that distribution,
using future execution availability only as a training label/mask and never as
an inference input. Keep path and summary reconstruction as auxiliary losses,
retain the same full-data four-year folds and loss-based early stopping, and
report Top1/3/5/10 with the same eligibility gates.

Only after an execution-aligned candidate passes should `global_tail_512` be
tested as a separate one-change candidate. Portfolio Sharpe, Sortino, drawdown,
turnover, concurrent positions, utilization, and uncertainty review remain a
mandatory pre-freeze stage, but were not triggered here because no profile was
eligible.

## Protected Boundaries

- QDP active data and manifests: unchanged.
- Active/default execution: unchanged.
- Champion freeze: not created.
- Historical test or outer audit: not created.
- True forward lockbox: not started; it begins only after a future eligible
  champion is retrained through 2025 and frozen.
