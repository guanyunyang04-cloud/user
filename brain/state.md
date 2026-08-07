# Current state

Updated: 2026-08-07

## Active objective

Build an autonomous daily A-share research system that observes the complete
causally available market path and account state, forms a latent market state,
estimates possible future paths, and compares the value of cash, buy, hold,
sell, and switch actions. The optimization object is risk-constrained account
wealth after legal execution and costs, not a fixed-D return label, a binary
"good stock" label, or imitation of hindsight tops and bottoms.

The current scientific object is continuous dynamic action advantage and
regret. Turning points, activity, pullbacks, breakouts, support, and resistance
are candidate coordinates or diagnostic descriptions; they are not assumed to
be natural market states. A model is introduced only when transparent analysis
cannot adequately express a verified conditional dependency.

No production model, portfolio, stop, or trading policy has been selected.
There is no active training or execution process.

## Formal research contract

- Formal history is 2012-01-01 through 2025-12-31. Years 2010-2011 are burn-in
  only. Do not read or use 2026 data.
- The primary universe is the point-in-time `quality_liquidity_pit` pool.
  Membership is applied at the probe date; path geometry may use each member's
  complete valid adjusted-price history available at that time.
- Information at a signal close may use daily, same-day five-minute, 557-field
  point-in-time, market, industry, cross-sectional, and account observations.
- Execution must model next-open entry, A-share T+1, limits, suspension,
  sellability, 100-share lots, commissions, stamp tax, slippage, capacity,
  concurrent candidates, cash, and terminal liquidation. Unknown execution or
  source state remains unknown and is never silently converted to false or
  used as a future candidate filter.
- Historical discovery may use all outcomes through 2025, but every simulated
  historical prediction or learner update must be reconstructed from the data
  and resolved experience available at that historical cutoff.
- Direct multi-horizon laws, joint path laws, and iterated transition kernels
  remain competing hypotheses. Markov time consistency is an advantage to
  test, not a theorem selecting the model class.
- Explain results in the conversation. Keep detailed tables and audits in the
  existing research outputs rather than in Brain.

## Authoritative data state

The corrected input chain is certified through 2025-12-31 with all blocking
checks passing. Historical ST state is based on dated name-change evidence;
nullable status and availability semantics are preserved. The formal PIT pool
has 4,191,601 stock-days, and the complete corrected feature intersection has
4,191,476 rows by 557 fields with no infinite or all-missing row/field.

Model and execution outputs created before the status repair are historical
baselines only. They must not be treated as trained on the corrected input;
current evaluation entry points enforce input and task fingerprint alignment.

QDP is the authority for source data and data semantics. Daily Research is the
authority for derived research inputs, predictions, account simulations, and
study conclusions. Important root paths are recorded in `brain/README.md`.

## Evidence that still governs the research

### What appears learnable

- Observable multi-scale path geometry contains stable but modest information
  about phase and turning timing. In strict 2019-2023 daily evaluation it adds
  about `0.0515` bit/date-cell over a causal prior with daily AUC about `0.633`.
  Timing improves more than remaining return magnitude. A first raw-sequence
  prototype was worse than geometry, although a small diagnostic blend found a
  tiny residual increment; this justifies better sequence research, not a deep
  model selection.
- Conditional scale, volatility, liquidity/activity state, and negative-tail
  ordering are more predictable than directional return. Existing distribution
  studies support risk gating and loss avoidance, not a claim that only risk
  can ever be predicted.
- Historical tree models rank maximum-favorable-excursion opportunities above
  random, but capture only a minority of the strongest future tails. Hindsight
  substitutions show that profitable opportunities exist ex post; the binding
  problem is prospective winner identification, timing, realization, and cost.

### What has failed or remains insufficient

- Raw volume shock, participation shock, and 20-day breakout are not standalone
  buys. Their tested 2023-2025 net outcomes are negative; shallow retests are
  relatively better but do not have a cost-robust absolute edge. Five-minute
  OHLCV measures activity and path structure, not trader identity or hidden
  "main-player funds" directly.
- Phase classifiers, one-extra-session continuation estimates, relative
  hold-versus-switch estimates, fixed-horizon return heads, and the broad v4
  policy surfaces have not produced a stable positive legal account. Turnover,
  switching costs, T+1 overnight exposure, weak magnitude estimation, and
  unstable extreme-tail selection are material failure modes.
- A retrospective 8%-take-profit/D20-timeout neighborhood was positive in its
  bounded study, but had roughly 43% drawdown and was never promoted. It is a
  historical clue about realization, not an active fixed-D strategy.
- Observable "hot" state is an opportunity gate rather than winner identity.
  Similar-looking failed cases are abundant, so successful chart examples
  cannot be used without matched failures and chronological testing.

## Dynamic no-fixed-horizon oracle

The audited one-slot hindsight oracle covers all 4,191,601 PIT stock-days under
base and double-slippage costs. Independent backward Bellman and chronological
recursions agree within `2.3e-13`. Only about `4.84%` of legal buy states have
positive dynamic advantage, and the oracle selects about 1,500 mostly
two-session trades. This is a perfect-future-information ceiling, not a return
forecast or deployable strategy.

Oracle winners have high volatility, turnover, range, and five-minute activity,
but the strongest mean daily linear observable IC is only about `0.031`. The
five nearest same-date observable neighbors are future-positive only about
`12.3%` of the time and have negative mean action advantage. A close observable
failure exists on every analyzed oracle-buy date. The current 557 fields expose
where opportunity may occur much better than which candidate will realize it.

The branch-coalescence study derives an action's natural decision interval
without imposing D5 or D20. Conditional on the final-terminal oracle policy,
buy and wait branches first return to a common cash state after a median 5
sessions; 90% resolve within 10, 99% within 17, and 99.9% within 23. Positive
actions resolve within 25 sessions and selected actions within 24. Rare
multi-year delays are genuine suspension/restructuring cases, strongly
nonpositive, and never selected.

Truncating the market at that common cash state reproduces all 4,184 sampled
full-terminal action values, including all 3,288 base selected actions, with
100% sign agreement and maximum absolute difference `3.36e-8`. An independent
validator traced 1,792 branches and found no blocking issue.

This identity is conditional on the final oracle. The final-2025 oracle's
resolution date and resolved-sample mask are not causally observable at an
earlier cutoff. Using them in an expanding learner would leak future policy
information through sample inclusion even when the numeric reward prefix ends
before the recorded resolution date.

The annual causal-prefix ledger removes that leakage by solving 28 independent
prefix oracles: 2012-2025 year ends under base and double-slippage costs. It
contains 15,832,630 timestamped experience versions, 8,329,390 first-learnable
rows, and 7,313,322 following-year revisions. Per cost scenario, 4,164,695 of
4,174,511 final-valid actions become naturally learnable by 2025; 4,047,617 are
available at their signal-year end and 117,078 first become available at the
following year end. This establishes annual as-of availability only, not the
exact day on which a label first became available. The independent validator
traced 432 version rows, exactly matched the final-2025 prefix policies to the
original oracle, and found no blocking issue.

Annual revisions are concentrated at the cutoff boundary. Once a prefix has at
least 10 observed sessions after branch coalescence, none of roughly 3.52
million update rows per cost scenario changes by more than `0.0002`; the few
remaining sign crossings are numerically near zero rather than economically
meaningful. The latest available annual version satisfies this maturity rule
for about `99.53%` of final-valid actions. Freeze
`sessions_after_resolution >= 10` for the first transparent baseline study.
This is a label-maturity buffer, not a holding horizon, entry rule, exit rule,
or profitability result.

## Current decisions and prohibitions

- Do not train a fixed-D "good stock" classifier or imitate final-oracle action
  signs as if their resolution timing were known online.
- Do not interpret hindsight wealth, MFE, tops, bottoms, or perfect exits as
  evidence of predictability or live profitability.
- Do not keep tuning the completed LightGBM and 2023-2025 execution surfaces.
  They are frozen historical baselines; a new model requires a new objective
  and matched causal comparison.
- Do not select deep learning, reinforcement learning, a transition kernel, or
  any hand-written trading rule before simpler causal baselines establish
  incremental continuous action value after costs.
- Do not remove legal constraints, failed neighbors, delisted/suspended paths,
  unavailable data, cash, or opportunity cost to make a result look stronger.

## True pause point and next steps

The annual causal-prefix ledger, dual-cost full run, independent validator, and
focused regression and static checks are complete. No predictor or causal
trading policy has yet been trained or evaluated. The next work is the first
strictly causal transparent action-value study:

1. For each out-of-sample year, train only on experience versions available by
   the preceding year end and require `sessions_after_resolution >= 10`.
2. Predict continuous buy advantage versus cash, its uncertainty, and
   cross-sectional ranking. Do not convert it into a fixed-D or "good stock"
   label.
3. Compare a past-only unconditional/regime prior, matched historical
   neighbors, conditional-bin or smooth additive estimates, and a transparent
   continuous-value baseline on identical folds and inputs.
4. Audit matched failures, calibration, top-ranked realized advantage, regret,
   annual and regime stability, feature-family ablation, and both cost
   scenarios. Freeze choices inside each training prefix.
5. Only after a baseline shows stable incremental decision value should it be
   replayed in the full legal account. Deep sequence/world models remain later
   challengers rather than assumed solutions.

## Current authoritative paths

- Dynamic oracle:
  `daily_research/path_policy/seq100_dynamic_oracle.py`
- Observable audit:
  `daily_research/path_policy/seq100_dynamic_oracle_observable_audit.py`
- Natural-resolution study and validator:
  `daily_research/path_policy/seq100_dynamic_oracle_resolution.py` and
  `daily_research/path_policy/seq100_dynamic_oracle_resolution_validate.py`
- Annual causal-prefix ledger and validator:
  `daily_research/path_policy/seq100_dynamic_oracle_causal_prefix.py` and
  `daily_research/path_policy/seq100_dynamic_oracle_causal_prefix_validate.py`
- Study configurations:
  `daily_research/studies/seq100_dynamic_oracle_v1.json`,
  `daily_research/studies/seq100_dynamic_oracle_observable_audit_v1.json`,
  `daily_research/studies/seq100_dynamic_oracle_resolution_v1.json`,
  `daily_research/studies/seq100_dynamic_oracle_causal_prefix_v1.json`
- Validated outputs:
  `daily_research/output/path_policy/studies/seq100_dynamic_oracle_v1/`,
  `daily_research/output/path_policy/studies/seq100_dynamic_oracle_observable_audit_v1/`,
  `daily_research/output/path_policy/studies/seq100_dynamic_oracle_resolution_v1/`,
  `daily_research/output/path_policy/studies/seq100_dynamic_oracle_causal_prefix_v1/`
- Durable prior evidence:
  `daily_research/research_records/seq100/`
