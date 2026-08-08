# Current state

Updated: 2026-08-08

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

The Chan primary-source reassessment, first strict-parser vertical slice, the
curated nine-path boundary audit, and the first outcome-blind stratified audit
are complete; the 14-chart visual review is now passed. A fast, fixed-horizon
outcome probe has been added only as a diagnostic gate while the resumable
all-market panel is being optimized. The immediate research question is
whether strict buy-point candidates retain incremental, cost-aware path
information on broader frozen cases; this is not yet an account or profit
claim.

No production model, portfolio, stop, or trading policy has been selected.
Multiple strictly causal transparent predictor, path-structure, and executable
trade probes are complete; none passed the account-replay gate. There is no
active training or execution process.

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

## Continuous causal retest-state evidence

The independent `seq100_causal_retest_state_v1` study expands an upward center
breakout into a locked, prefix-only state: breakout strength and scale, time
since breakout, boundary distance, peak and trough excursion, drawdown from the
peak, touch episodes, path efficiency, amount contraction, and current versus
prior directional-leg coordinates. It replays 10,138,279 valid dense path rows
and emits all 4,191,476 quality-pool rows. The panel contains 39,523 upward
breakout events, 75,333 causal retest rows, 13,197 reentries, and 1,285,988
active-breakout rows. No 2026 row, duplicate, or missing pool row exists.

Strict expanding-year training covers both base and double-slippage costs,
two predeclared scopes (retest event and price-amount-confirmed breakout), and
the inherited 66-coordinate active-state baseline. It produces 48,160
sample-out-of-sample rows in 52 partitions. The spline and historical-neighbor
estimators never produce a positive, stable absolute action-value policy. On
retest rows, the spline top-ranked forced-buy value is about `-9.25%` (base) and
`-9.34%` (double slippage), only about `+0.10` and `+0.08` percentage point
relative to the inherited top-20% additive ranking, with confidence intervals
crossing zero. The neighbor estimator is worse; breakout ranking is also worse
than the inherited baseline. The promotion gate and account replay remain
prohibited.

The most reproducible continuous diagnostic is the fraction of breakout
expansion subsequently given back by the current close. Its high-versus-low
same-date, risk-matched contrast is about `+0.52` percentage point in action
value under both costs, with BH `q` about `0.041`, positive in 11 of 13 years.
The change is chiefly about `-0.50` percentage point less downside and only
about `+0.02` percentage point more upside/positive probability. It is a
conditional severity clue, not evidence that the action has positive absolute
value; similar failures remain common. Very large pooled contrasts for medium
retracement occur on only seven dates and are not evidence. Repeated touches,
breakout strength, and most other coordinates are unstable or non-significant.

The independent validator replays 24 symbols at 192 prefix cutoffs, compares
9,370,176 state cells and 70,656 persisted cells, checks all 48,160 prediction
rows and scope constraints, and independently recomputes 12 model and 104
coordinate HAC summaries. It passes; the maximum stored float difference is
`9.47e-6` from float32 panel quantization.

## Exploratory financing-balance evidence

A causal point-in-time probe aligned margin source dates to their recorded
availability dates and joined 2,661,196 quality-pool stock-days (2012-2025,
3,349 dates, 1,955 symbols) to next-open path outcomes. It is exploratory,
not a frozen predictor study or promotion candidate. Five-observation log
growth in financing balance `rzye` is not bullish alone: the highest decile's
D5 gross mean is about `+0.236%` versus `+0.382%` in the lowest; proportional
base net means are about `-0.058%` versus `+0.087%`, and double-slippage means
are negative in both. The high-balance-growth/low-price-growth quadrant is
only about `+0.038` percentage point better than low-balance-growth/low-price-
growth (HAC interval crosses zero, positive in 9/14 years). Its much larger
advantage over high-balance-growth/high-price-growth is primarily short-term
price reversal. A discarded first query used the raw source date as the signal
date and would have leaked one day; it is not evidence. Details are in
`daily_research/research_records/seq100/seq100_margin_balance_probe_20260808/`.

## Causal path-structure evidence

The first strictly causal multi-scale path grammar is complete. It emits daily
provisional and confirmed directional changes plus explicit Chan-style
candidate fractals, strokes, segments, centers, breakouts, retests, reentries,
and exhaustion. The grammar is a hypothesis representation, not a claim that
orthodox Chan theory is true. Its 2012-2025 quality-pool panel has all
4,191,476 rows with no duplicate, missing, or 2026 row.

Prefix validation reran 24 stocks at 192 truncated cutoffs and compared about
22.25 million feature cells without a historical-state change. Persisted-panel
and diagnostic recomputation also passed; float differences are storage
quantization only. Thus the representation is usable for causal research and
does not annotate an earlier date with a later-confirmed pivot.

None of 11 predeclared candidate patterns has positive absolute dynamic action
advantage or passes the dual-cost promotion gate. These values are forced-buy
advantage versus the causal-prefix oracle cash branch, not literal trade
returns. A negative value means the pattern alone does not identify when buying
beats waiting under the action-value objective.

The one corrected narrow dependency is an upward center breakout followed by a
causally observed boundary retest that closes above it. Its same-date,
same-risk-cell action-value increment is about `+0.267` percentage point under
base cost and `+0.265` under double slippage, with BH-adjusted `q` about
`0.026-0.027`. The improvement comes from roughly `0.28` percentage point less
downside, not a higher positive-event probability. Absolute action advantage
remains about `-9.5%`, all 13 annual absolute means are negative, only 7/13
matched annual differences are positive, and about 70% of positive retest
cases still have a same-pattern matched failure.

Buying the initial center breakout, buying a small-scale upward
reacceleration, and their union as an up-continuation rule are worse than
matched controls. Late-chase breakouts are especially adverse: about `-0.33`
percentage point matched action value, primarily from larger downside. A
breakout with above-median five-day price-amount correlation and below-median
last-30-minute return is relatively better in 10/13 years, but uncertainty
still crosses zero after correction. Nested pullback and exhaustion states do
not provide a cost-robust absolute edge.

## Chan-theory source audit

The current parser is not an orthodox Chan implementation and its negative
experiments do not reject the complete theory. Primary-source cross-checking of
the 108 lessons shows that the intended hierarchy is inclusion-normalized K
lines to fractals, strokes, feature-sequence-confirmed segments, lowest-level
centers, recursively higher-level centers and trend types, divergence, and the
three buy/sell-point classes. Pending or "middle-yin" states and event-versus-
confirmation time are part of the theory and cannot be silently backfilled.

The formalizable part is a deterministic causal path parser. The profit claims
are separate and unproved. In particular, the author never supplies a closed
measure for exact trend strength/divergence, later permits different unique
base constructions, admits legitimate recombination views, and in lesson 107
explicitly distinguishes theoretical next-leg existence from enough executable
profit to cover fees. Exact self-isomorphism, 100%-safe buy points, and stable
profit after delay, gaps, T+1, and costs must therefore remain empirical
hypotheses rather than mathematical consequences.

The previously scraped FURLEADER corpus is not a clean 108-lesson body: many
pages append unrelated article text. Use the archived lesson-by-lesson source
and attributed replies in `stockServ/chzhshch-108-plus` as the primary public
cross-check, with FURLEADER only as secondary corroboration. Direct scholarly
search found no reliable peer-reviewed causal test of complete Chan theory.

## Strict Chan parser foundation

The frozen dependency contract is
`daily_research/studies/seq100_strict_chan_definition_v1.json`. It separates 23
executable definitions from engineering conventions, ambiguity variants,
quantitative divergence coordinates, and empirical profit hypotheses. The new
implementation is independent of the validated weak grammar and does not
replace or reinterpret its historical results.

`seq100_strict_chan_parser.py` now emits inclusion-normalized bars, confirmed
fractals, conservatively locked strokes, standard feature sequences, both
segment-break cases, explicit pending/middle states, segment events, recursive
centers, center extension/expansion/new birth, same-level movements, trend
snapshots, named divergence coordinates, and first/second/third-point
candidates. Every immutable event records both structural event time and first
causal confirmation time. Candidate points are annotations only; no future-path,
profit, execution, or policy test has been run.

The QDP adapter pins the current five-minute, adjustment-factor, and raw-daily
datasets. It requires canonical 48-bar days, multiplies raw intraday OHLC by the
same-day cumulative `adjust_factor`, and splits parser episodes when a
positive-volume daily bar lacks complete five-minute data or a valid causal
factor. It never fabricates suspension bars. Fifteen external-archive shards
have malformed date metadata such as `2024-01-`, although their Parquet
`trade_date` values are valid full dates. The adapter audits this anomaly and
filters on the actual column rather than trusting shard metadata; an immutable
QDP metadata repair remains separate follow-up work.

A bounded real smoke test on `600000.SH` from 2024-01-01 through 2025-12-31
uses 23,280 complete adjusted bars with no missing positive day. The primary
variant produces 9,905 locked normalized bars, 5,166 fractals, 1,117 locked
strokes, 79 segments, 12 level-1 and 3 level-2 centers, 5 divergence-coordinate
events, and 16 point candidates. These are coverage counts, not evidence of
market validity. Eight prefix cutoffs compare 61,854 event records and about
1.20 million fields exactly with no future-confirmed writeback.

Real prefix recomputation exposed and fixed three confirmation-order defects:
case-2 invalidation could precede candidate recognition, a later segment could
be confirmed before an unresolved predecessor, and divergence or point events
could depend on unconfirmed structures. Candidate decisions now inherit the
latest dependency time, and every parse ends with a complete confirmation-time
DAG check.

The frozen nine-path audit spans trends, oscillations, long gaps, an adjustment
boundary, short history, one-price limit clusters, a limit-down reversal, and
both segment-break cases. Across 54,000 five-minute bars, 1,125 usable days, and
seven definition profiles, all case assertions and visual checks pass. Exact
prefix reruns compare 554,740 event records and 10,747,435 fields with no future
confirmation writeback. The primary view emits 192 segments, 33 centers, and 24
point candidates. Its 127 pending segment candidates include 59 confirmations,
67 invalidations, and one terminal unresolved state; median duration is 67 bars
and the 90th percentile is about 225 bars, so pending state and confirmation lag
cannot be collapsed into event time.

Definition sensitivity remains material across the audit. Strict containment
has segment/stroke Jaccard about `0.0445/0.1492` versus the primary view;
disjoint-fractal strokes have about `0.1012/0.4985`. Stroke-based lowest centers
produce 416 centers versus 33 and no identical point candidate. Fluctuation-
interval center relations leave the lower structures unchanged but reduce trend
Jaccard to about `0.623`, point-candidate Jaccard to `0.75`, and divergence count
to zero. Equality tiebreaks are inactive here and initial-direction lookahead is
confined to startup. These selected paths are parser boundary coverage, not a
statistically sufficient all-market sample and not grounds for choosing a
profitable or theoretically privileged variant.

## Stratified strict-parser audit

The frozen selection contract is
`daily_research/studies/seq100_strict_chan_stratified_audit_v1.json`. It selects
one globally unique symbol/date row from each of 14 deterministic strata: seven
2012-2025 two-year periods crossed with `.SH` and `.SZ`. The rank is a SHA-256
ordering of the study id, seed, stratum, symbol, and focal date. Only
`quality_liquidity_pit` `symbol` and `trade_date` are read for selection;
future-return, label, parser-event, and trade-result columns are not accepted by
the selector. The current PIT pool contains only `.SH` and `.SZ` rows; `.BJ` is
not silently treated as an empty exchange and is excluded by the explicit
population suffix contract.

The runner is
`daily_research/path_policy/seq100_strict_chan_stratified_audit.py`. It freezes
the sample and source fingerprints before materializing a generic audit spec,
then reuses all seven parser profiles, causal prefix validation, pending-state
aggregation, and chart generation. A changed input, spec, or runner fingerprint
cannot replace an existing sample unless `--refresh-sample` is explicit.

The first real run completed 14/14 cases across 152,928 five-minute bars and
3,186 usable days, with no missing positive-volume day. Prefix recomputation
compared 1,903,654 event records and 36,802,802 fields with zero future
confirmation writeback; all case assertions passed. The primary profile emitted
789 segments, 142 centers, and 139 point candidates. Its 470 pending candidates
resolved as 226 confirmed, 239 invalidated, and 5 unresolved at episode end;
median pending duration is 47 bars and P90 about 192 bars. These are parser
coverage and causal-lag measurements, not market-performance results. The
contact sheet and all 14 chart records have `visual_review_status: passed`.
This stratified run is still a small deterministic audit sample, not
statistically sufficient full-market validation.

## Strict panel and fast outcome gate

`daily_research/path_policy/seq100_strict_chan_panel.py` and
`daily_research/studies/seq100_strict_chan_panel_v1.json` define the resumable
2010-2025 five-minute panel. It now reads a contiguous sorted-symbol bucket in
one DuckDB batch, caches the symbol universe, writes burn-in events with an
explicit `is_formal` flag, publishes only 2012-2025 daily snapshots, and tracks
the latest state of each pending candidate. A one-symbol real run completed in
about 30 seconds; a three-symbol long-history run completed in about 145
seconds. Full-market execution is therefore not being started blindly; the
remaining bottleneck is Parquet I/O for long-history symbols, not parser
correctness.

`daily_research/path_policy/seq100_strict_chan_outcome_probe.py` evaluates the
frozen 14-case parser events at the next trading-date open with T+1 and base /
stress proportional costs (including minimum commission). The run produced
343 deduplicated buy candidates and 1,567 event-horizon rows. In the primary
profile, type-3 candidates are negative over 1-10 sessions; type-2 candidates
are positive in the small 5/3/2-case denominators at 5/10/20 sessions, but this
does not persist as a broad or independent stability result. The probe is a
path diagnostic only; no account replay or production policy is authorized.

## Absolute causal K-line strategy evidence

The `seq100_causal_pattern_strategy_probe_v1` study separates literal trade
profit from the stricter oracle-relative action value. It evaluates retest,
retest-plus-exhaustion, nested reacceleration, and breakout entries with causal
state invalidation, small/medium reversal, or post-entry exhaustion exits. Each
signal enters at the next open and exits only at the first later legal close;
T+1, limits, suspension, 100-share lots, minimum commission, stamp tax, and
base/double slippage are applied to a CNY 100,000 trade. No fixed holding day is
used.

The full 2012-2025 replay produces 306,755 nonoverlapping single-stock trades;
all resolve before the terminal date. Independent validation exactly
recomputes every entry, exit, and cost with zero numerical error and finds no
2026 row or overlap. None of eight predeclared strategies passes. In the
2013-2025 primary period, the relatively best entry is retest plus exhaustion
followed by a small-scale reversal exit: date-equal gross return is about
`+0.12%` (trade-weighted `+0.19%`), but date-equal net return is about `-0.17%`
under base cost and `-0.31%` under double slippage; the HAC lower bounds are
negative and only 5/13 years are positive. Other rules are worse.

The important failure mode is realization rather than absence of path
movement. For example, breakout entries experience roughly `+6.86%` mean
maximum close-to-close excursion but finish near flat gross (date-equal about
`-0.09%`, trade-weighted about `+0.07%`) under the tested invalidation exit.
Median holding is only 3-5 sessions.
The named states therefore expose volatile opportunity but the hand-written
structural exits surrender gains or stop too early. A positive fixed-D view is
not evidence that a causal dynamic exit can realize it.

## Open-position episode and mechanical-exit evidence

The `seq100_causal_exit_baselines_v1` study removes the old exit-policy
selection from entry construction. It extracts every actually filled causal
retest, nested-reacceleration, and breakout signal directly from the PIT
quality pool, allows overlapping episodes only for independent trade research,
and builds 10,444,952 close-time position states for 255,029 entries. State
includes return since entry, peak expansion and giveback, MAE, entry-time
20-session volatility, amount, and multi-scale structure. Market-regime,
industry, and five-minute coordinates are not yet joined; this is the
mechanical stopping baseline, not the final stopping panel.

Ten predeclared exits cover H2/H5/H10/H20/H40 controls, two volatility-scaled
stop/take-profit rules, a 3% trailing stop, a half-giveback rule, and the prior
small-reversal structure exit. A close-time trigger is filled only at the next
legal close. Exact finite cash, T+1, limits, suspension, board lots, minimum
commission, stamp tax, and base/double slippage remain enforced. Independent
validation exactly matches all entry keys and checks 10.44 million panel rows,
2,550,290 policy results, 840 independently reconstructed request/exit paths,
and costs with zero observed request-date or cash-return error; no 2026 data is
used.

No mechanical exit passes. The least negative candidate is a retest followed
by an H2 exit: 2013-2025 date-equal net return is about `-0.14%` under base cost
and `-0.28%` under double slippage, with only 2/13 and 1/13 positive years.
Longer holding exposes large intermediate movement without realization: for
retest H20, trade-weighted mean MFE is about `+8.49%` while mean gross exit
return is about `-0.19%`; comparable breakout and nested results are also
negative. Volatility stops, trailing stops, giveback exits, and structure exits
do not fix this. Exit timing is a real problem, but the all-signal entries have
too little unconditional gross edge to assume that exit learning alone can
rescue them. Strong annual variation and date-versus-trade weighting differences
make market regime and signal breadth necessary stopping/entry covariates.

## Future-informed stopping-ceiling evidence

The separately bounded `seq100_exit_stopping_ceiling_v1` diagnostic enumerates
every close-time sale request through session 40 and resolves it at the first
later legal close with exact finite costs. It is explicitly future-informed and
cannot be a policy or an online label. Independent brute-force validation of
1,000 entries and both costs finds zero request-date, exit-date, or net-return
error.

The ceiling is large but must not be confused with predictability. Across
2013-2025, date-equal mean best legal net return is roughly `12.7%-13.5%` for
all three entry types under both costs, about `84%-85%` of entries can be made
positive by hindsight, and the median best sale occurs around holding session
16-17. The request-time interquartile range is roughly sessions 4-31, with the
90th percentile near session 38. This confirms substantial path headroom and
the absence of one natural fixed exit day. However, breakout, retest, and
nested-reacceleration ceilings are very similar; maximizing over 40 future
sessions mechanically creates a large extreme even for volatile non-special
paths. A same-date, same-risk, same-activity non-pattern control is required
before attributing the headroom to the K-line entries.

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
- Do not interpret a rising `rzye` or a financing/price divergence as proof of
  fund accumulation or a buy signal; distinguish balance, reported new flow,
  price scaling, eligibility, and publication timing.
- Do not promote the exploratory D20 retest-plus-exhaustion mean. Its frozen
  no-fixed-horizon follow-up is negative after exact costs.
- Do not keep tuning mechanical stop percentages or fixed holding days. The
  predeclared family failed under both costs; further parameter search requires
  a new causal hypothesis and correction for search multiplicity.

## True pause point and next steps

The structure grammar, retest-state study, absolute K-line strategy probe,
open-position panel, mechanical stops, and future-informed stopping ceiling are
complete and independently validated. Account replay remains prohibited. The
mechanical results reject simple fixed or hand-written exits. The ceiling shows
large ex-post headroom but does not establish that either the entry shape or
the exit day is prospectively identifiable.

The source audit, frozen definition dependency table, independent parser,
five-minute adjustment/missingness adapter, synthetic tests, one-symbol output,
curated nine-path audit, 14-case stratified prefix/variant audit, and first
outcome gate are complete. The sample is not statistically broad enough to
support a market claim. The full 2010-2025 market panel is implemented but not
yet run end-to-end because the I/O path needs further batching or a cached
partition. Account replay and model training remain prohibited until the next
outcome gate is broad enough. The next bounded work is:

1. Expand the outcome probe to a larger outcome-blind structure sample and
   require positive case/year coverage under both cost scenarios before any
   account replay.
2. Benchmark a cached or year-sharded Parquet partition for the complete
   2010-2025 five-minute store. Preserve per-symbol episode boundaries and emit
   daily close-time snapshots for later `quality_liquidity_pit` joins without
   treating missing intraday history as a universe filter.
3. Repair the malformed external-archive shard date metadata through an
   immutable QDP metadata version or explicit audit artifact; do not mutate the
   pinned dataset manifest in place.
4. Test whether the complete structures add future-path information beyond
   volatility, activity, market regime, and the existing weak grammar. Only
   then test executable first/second/third-point policies under both costs.
5. Build a same-date, same-risk, same-activity non-pattern control and compare
   its legal stopping ceiling with each K-line entry type. This determines
   whether the large headroom belongs to the named pattern or merely to active
   volatile stocks and the 40-session maximum operator.
6. Join the existing causal market, industry, cross-sectional, and five-minute
   coordinates to each open-position day, including same-day signal breadth and
   explicit missingness when a held stock leaves the current quality pool.
7. Fit a transparent expanding-year fitted-Q baseline that separately predicts
   the value of `request_sell` and `hold`, using only prior-year episodes. Each
   following year is simulated chronologically with daily re-evaluation and no
   fixed D target. Mechanical policies remain frozen controls.
8. Determine whether peak giveback, exhaustion, market regime, signal breadth,
   or intraday state predicts residual waiting value. If causal stopping still
   loses, reject exit-only rescue and return to regime-conditioned entry
   selection rather than escalating model complexity.
9. Use financing balance, net financing flow, and financing/price divergence
   only as residual covariates after exact publication alignment; do not use
   them as entry gates unless they improve the stopping/value model out of
   sample under both costs.
10. Replay a legal account only after a frozen policy has positive absolute net
   value under base and double-slippage costs, a positive uncertainty lower
   bound, and majority-year stability. A raw/deep sequence model remains a
   later challenger if transparent state leaves reproducible residual value.

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
- Causal path-structure study and validator:
  `daily_research/path_policy/seq100_causal_path_structure.py` and
  `daily_research/path_policy/seq100_causal_path_structure_validate.py`
- Strict Chan definition, parser, QDP adapter, validator, and audits:
  `daily_research/studies/seq100_strict_chan_definition_v1.json`,
  `daily_research/path_policy/seq100_strict_chan_parser.py`,
  `daily_research/path_policy/seq100_strict_chan_intraday.py`, and
  `daily_research/path_policy/seq100_strict_chan_validate.py`; frozen audit plan
  and runner are
  `daily_research/studies/seq100_strict_chan_audit_v1.json` and
  `daily_research/path_policy/seq100_strict_chan_audit.py`; the outcome-blind
  stratified contract and runner are
  `daily_research/studies/seq100_strict_chan_stratified_audit_v1.json` and
  `daily_research/path_policy/seq100_strict_chan_stratified_audit.py`
- Absolute causal K-line strategy probe and validator:
  `daily_research/path_policy/seq100_causal_pattern_strategy_probe.py` and
  `daily_research/path_policy/seq100_causal_pattern_strategy_probe_validate.py`
- Open-position episode/mechanical-exit study and validator:
  `daily_research/path_policy/seq100_causal_exit_baselines.py` and
  `daily_research/path_policy/seq100_causal_exit_baselines_validate.py`
- Future-informed stopping ceiling and validator:
  `daily_research/path_policy/seq100_exit_stopping_ceiling.py` and
  `daily_research/path_policy/seq100_exit_stopping_ceiling_validate.py`
- Study configurations:
  `daily_research/studies/seq100_dynamic_oracle_v1.json`,
  `daily_research/studies/seq100_dynamic_oracle_observable_audit_v1.json`,
  `daily_research/studies/seq100_dynamic_oracle_resolution_v1.json`,
  `daily_research/studies/seq100_dynamic_oracle_causal_prefix_v1.json`,
  `daily_research/studies/seq100_dynamic_action_value_baselines_v1.json`,
  `daily_research/studies/seq100_dynamic_action_distribution_v1.json`,
  `daily_research/studies/seq100_causal_path_structure_v1.json`,
  `daily_research/studies/seq100_strict_chan_definition_v1.json`,
  `daily_research/studies/seq100_strict_chan_audit_v1.json`,
  `daily_research/studies/seq100_causal_pattern_strategy_probe_v1.json`,
  `daily_research/studies/seq100_causal_exit_baselines_v1.json`,
  `daily_research/studies/seq100_exit_stopping_ceiling_v1.json`
- Validated outputs:
  `daily_research/output/path_policy/studies/seq100_dynamic_oracle_v1/`,
  `daily_research/output/path_policy/studies/seq100_dynamic_oracle_observable_audit_v1/`,
  `daily_research/output/path_policy/studies/seq100_dynamic_oracle_resolution_v1/`,
  `daily_research/output/path_policy/studies/seq100_dynamic_oracle_causal_prefix_v1/`,
  `daily_research/output/path_policy/studies/seq100_dynamic_action_value_baselines_v1/`,
  `daily_research/output/path_policy/studies/seq100_dynamic_action_distribution_v1/`,
  `daily_research/output/path_policy/studies/seq100_causal_path_structure_v1/`,
  `daily_research/output/path_policy/studies/seq100_strict_chan_definition_v1/`,
  `daily_research/output/path_policy/studies/seq100_strict_chan_audit_v1/`,
  `daily_research/output/path_policy/studies/seq100_strict_chan_stratified_audit_v1/`
  (sample manifest, frozen audit spec, case/event/pending/prefix aggregates, and
  contact sheet),
  `daily_research/output/path_policy/studies/seq100_causal_pattern_strategy_probe_v1/`,
  `daily_research/output/path_policy/studies/seq100_causal_exit_baselines_v1/`,
  `daily_research/output/path_policy/studies/seq100_exit_stopping_ceiling_v1/`
- Durable prior evidence:
  `daily_research/research_records/seq100/`
