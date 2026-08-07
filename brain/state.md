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

The dynamic-oracle resolution implementation, full dual-cost run, prefix
truncation check, independent branch validator, 42-test regression suite, and
static checks are complete. The next work is not another fixed-horizon model.

1. Recompute the oracle and branch coalescence using only each historical
   prefix. Start with chronological fold cutoffs for correctness; use a finer
   incremental schedule only after equivalence is established.
2. At every cutoff, expose only actions whose values have resolved under that
   prefix oracle. Record when an experience first becomes learnable and how its
   target changes as the prefix grows.
3. Compare strictly causal matched history, conditional bins or smooth
   additive baselines, and transparent sequence/value baselines on continuous
   action advantage and regret. Include matched failures, annual/regime signs,
   calibration, cost stress, and feature-family ablation.
4. Only if an observable estimator adds stable net decision value, replay it in
   the full legal account with cash and competing candidates. A multiscale
   encoder is a later challenger, not the starting assumption.

## Current authoritative paths

- Dynamic oracle:
  `daily_research/path_policy/seq100_dynamic_oracle.py`
- Observable audit:
  `daily_research/path_policy/seq100_dynamic_oracle_observable_audit.py`
- Natural-resolution study and validator:
  `daily_research/path_policy/seq100_dynamic_oracle_resolution.py` and
  `daily_research/path_policy/seq100_dynamic_oracle_resolution_validate.py`
- Study configurations:
  `daily_research/studies/seq100_dynamic_oracle_v1.json`,
  `daily_research/studies/seq100_dynamic_oracle_observable_audit_v1.json`, and
  `daily_research/studies/seq100_dynamic_oracle_resolution_v1.json`
- Validated outputs:
  `daily_research/output/path_policy/studies/seq100_dynamic_oracle_v1/`,
  `daily_research/output/path_policy/studies/seq100_dynamic_oracle_observable_audit_v1/`,
  and
  `daily_research/output/path_policy/studies/seq100_dynamic_oracle_resolution_v1/`
- Durable prior evidence:
  `daily_research/research_records/seq100/`
