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

No production model, portfolio, stop, or trading policy has been selected. Two
strictly causal transparent predictor studies are complete; neither passed the
account-replay gate. There is no active training or execution process.

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

## Causal action-value evidence

The first transparent continuous-value baseline evaluated 7,892,894 strict
out-of-sample predictions over 2013-2025 and both cost scenarios. Observable
coordinates have weak but stable ranking information: the additive mean daily
Spearman is about `0.0446`. All smoothed expected action values remain negative,
the zero-threshold policy never acts, and unsmoothed matched cells lose money.
Similar observable failures exist for roughly `99.8%` of daily selections.
This rules out a simple shrinkage or threshold-tuning explanation and prohibits
account replay.

The follow-up distribution study expands the transparent representation to 66
multi-scale daily and five-minute coordinates and separately estimates positive
opportunity probability, upside contribution, downside contribution, and their
reconstructed expected value. It evaluates 1,575,340 strict out-of-sample rows
inside a top-20% activity gate. The activity rank is first computed on the full
PIT quality-liquidity pool and only then mapped to mature experiences; ranking
inside the outcome-eligible subset would be a sample-selection leak.

The activity gate is real but not sufficient. Under base costs, the positive
action fraction rises from about `5.11%` in the full pool to `7.73%` in top-20%
activity and `8.67%` in top-10%, while mean action value worsens from about
`-9.69%` to `-9.99%` in top-10%. Transparent additive models rank the positive
component with daily AUC about `0.635-0.643`; selecting the highest predicted
positive probability raises the realized positive fraction to roughly
`15-17%`, but also selects much larger negative outcomes, so mean action value
remains around `-10%`. No distribution model predicts a positive expected
action, passes the promotion gate, or authorizes account replay.

Within the active pool, `98.23%` of positive cases have a same-date failure in
the same coarse volatility, downside-volatility, range, minute-volatility, and
relative-turnover cell. Matched diagnostics do reveal narrow regularities:
high five-day price-amount correlation improves action value by about `0.32`
percentage point versus its low half under both costs (`10/13` positive years,
BH-adjusted `q` about `0.045`), but does not materially improve positive-event
probability and remains far from an absolute edge. High five-day volatility is
a weaker near-significant value clue; large minute range raises opportunity
probability but its value effect is episodic. Strong last-30-minute return,
high close-versus-VWAP, high close location, and late-hour amount concentration
are stable adverse next-open action clues inside already-active matched states.
These are hypotheses about severity and timing, not executable rules.

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

The continuous-value and opportunity/upside/downside transparent studies, full
dual-cost runs, independent validators, focused tests, and static checks are
complete. They establish that activity and several path coordinates contain
real opportunity information, but current summaries mostly distinguish
high-opportunity/high-loss states from lower-risk states rather than identify a
positive expected buy. Account replay remains prohibited.

The next study is a causal multi-scale path-structure representation rather
than another threshold tweak:

1. Formalize provisional and confirmed pivots, strokes, segments,
   consolidation zones, divergence, breakout, pullback, and exhaustion using
   only bars available at each date. Chan-theory-style structures are candidate
   grammars to test, not accepted truths; inclusion handling, scale, and future
   confirmation must be explicit and non-repainting.
2. Preserve several volatility-normalized scales instead of declaring one
   pivot threshold to be the true top or bottom. Compare the formal grammar to
   simpler causal zigzag/change-point and raw multi-scale sequence baselines.
3. Inside the full-pool causal activity gate, compare every candidate structure
   with same-date risk-matched failures. Focus first on whether price-amount
   confirmation and avoidance of late-day chase explain downside severity, not
   on maximizing the number of named patterns.
4. Evaluate positive probability, conditional upside, conditional downside,
   continuous action value, annual stability, multiple-testing correction, and
   both cost scenarios. A deterministic-looking chart pattern is not promoted
   unless its absolute expected action value is positive after costs.
5. Only a representation that passes this gate may advance to the legal account
   replay; a raw/deep sequence model remains a later challenger if transparent
   path grammars leave incremental information unexplained.

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
- Transparent continuous-value baseline and validator:
  `daily_research/path_policy/seq100_dynamic_action_value_baselines.py` and
  `daily_research/path_policy/seq100_dynamic_action_value_baselines_validate.py`
- Opportunity/upside/downside distribution study and validator:
  `daily_research/path_policy/seq100_dynamic_action_distribution.py` and
  `daily_research/path_policy/seq100_dynamic_action_distribution_validate.py`
- Study configurations:
  `daily_research/studies/seq100_dynamic_oracle_v1.json`,
  `daily_research/studies/seq100_dynamic_oracle_observable_audit_v1.json`,
  `daily_research/studies/seq100_dynamic_oracle_resolution_v1.json`,
  `daily_research/studies/seq100_dynamic_oracle_causal_prefix_v1.json`,
  `daily_research/studies/seq100_dynamic_action_value_baselines_v1.json`,
  `daily_research/studies/seq100_dynamic_action_distribution_v1.json`
- Validated outputs:
  `daily_research/output/path_policy/studies/seq100_dynamic_oracle_v1/`,
  `daily_research/output/path_policy/studies/seq100_dynamic_oracle_observable_audit_v1/`,
  `daily_research/output/path_policy/studies/seq100_dynamic_oracle_resolution_v1/`,
  `daily_research/output/path_policy/studies/seq100_dynamic_oracle_causal_prefix_v1/`,
  `daily_research/output/path_policy/studies/seq100_dynamic_action_value_baselines_v1/`,
  `daily_research/output/path_policy/studies/seq100_dynamic_action_distribution_v1/`
- Durable prior evidence:
  `daily_research/research_records/seq100/`
