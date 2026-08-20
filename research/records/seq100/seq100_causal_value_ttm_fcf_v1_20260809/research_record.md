# Causal value, analyst revision, and TTM-FCF policy audit

Status: a simple TTM free-cash-flow-yield factor survives as a meaningful
secondary historical candidate, but the exact primary account fails the frozen
drawdown gate. The composite value/quality/revision thesis and valuation-target
exits are not promoted. Nothing here is a production strategy or profit
guarantee.

Date: 2026-08-09

## Research question and causal contract

This branch translates the supplemental fundamental-investing hypothesis into
point-in-time A-share tests. It asks whether relative value, company quality,
actual fundamental improvement, analyst forecast revision, and price
confirmation add stable net value inside the signal-date
`quality_liquidity_pit` universe.

The formal history is 2012-2025. No 2026 row is read. Financial fields enter
only after their recorded availability date. Analyst consensus uses research
reports available by the signal close, takes the latest forecast per
institution, targets the next fiscal year, and compares the current 180-day
consensus with the consensus available 60 calendar days earlier. Current and
prior consensus both require at least three institutions.

The supplied hypothesis that proximity to a 52-week low should be removed was
accepted before the study. Neither the 52-week low nor distance from it is used
as a reward, filter, target, or tie-breaker. Price is used only as causal
confirmation that deterioration has stopped or relative strength has improved.

## What was implemented

The causal value study compares seven frozen monthly Top48 policies, with no
more than four names per point-in-time industry:

- industry-relative positive PE/PB/FCF value;
- value plus actual earnings improvement;
- value plus analyst forecast revision;
- value plus price confirmation;
- the complete value/actual/revision/price rule;
- a quality-only control;
- transparent component controls.

Signals enter at the next legal open. Fixed D20 and D60 outcomes include the
conservative cost proxy and legal-sale deferral. Valuation-derived exits were
also tested: quarterly gap closure, industry median value, and the estimated
industry right edge. These exits are compared directly with the same entries
held to D60.

A separate overlay applies the value signal to the previously frozen
margin-observed D20 residual policy without refitting its model. A later audit
repairs the free-cash-flow definition: the compact field was the latest
cumulative fiscal period and therefore mixed Q1, half-year, Q3, and annual
values. The corrected causal TTM value is:

- annual report: current annual cash flow;
- Q1/Q2/Q3: current YTD + prior annual - prior same-quarter YTD;
- every component: the latest consolidated statement version known by the
  signal close.

TTM coverage is 95.05%. Its Spearman correlation with the old latest-period
field is only 0.516, so the old field is not accepted as TTM evidence.

## Main empirical findings

The complete value/actual/revision/price policy did not pass across periods.
Analyst revisions helped in earlier history but hurt in 2023-2025. Quality alone
was the weakest family. Adding value as either a gate or an equal-rank blend to
the frozen margin-residual model reduced its mean return in both development
and confirmation, so the overlay was rejected.

Valuation right-edge exits were especially unhelpful. Relative to the same
entries held to D60, they reduced average net return because they clipped rare
large winners. For the primary composite, the right-edge paired delta versus
D60 was about -2.05 percentage points in development and -2.92 points in
confirmation. Median and quarterly-gap exits were worse. A calculated fair
value range can be useful for human valuation, but this evidence does not
support using its right boundary as an automatic short-horizon take-profit.

The simplest value policy was more durable than the complex thesis. The pure
value D60 Top48 cohort had a full-history mean stress-net return of 3.44% per
monthly cohort, with HAC and block lower bounds near 0.85%. Removing the
exceptional 2014 year left a 2.42% mean and small positive lower bounds. The
selected excess and industry-residual means were about 1.19% and 1.17%.

After the TTM repair, the strongest transparent component was broad TTM
free-cash-flow yield rather than a precise Top10 stock picker. Top48 diagnostics
were substantially more stable than Top10. This matters operationally: the
evidence supports a diversified factor exposure, not confidence that the first
few ranked stocks are individually predictable.

## Exact finite-account falsification

One natural, non-optimized execution was frozen: each chronological month takes
12 of the TTM-FCF Top48 according to the deterministic rank-residue rule
`(rank - 1) mod 4 == month_ordinal mod 4`. The natural phase is zero; it was not
selected from the best of four observed phases.

The primary account uses CNY 1,000,000, 48 slots, no pyramiding, 100-share lots,
minimum commission, transfer fee, historical stamp tax, double slippage,
next-open entry, a D60 sell request, legal-sale deferral, and a maximum 0.5% of
signal-day amount per order. Raw exchange prices determine fills and costs;
the pack-pinned adjustment-factor ratio supplies total-return-equivalent marks
and proceeds.

The original bounded D80 pack semantics wrote 15 still-suspended positions to
zero. This is retained as a deliberately severe pressure case, but it is not a
literal account action. The primary execution instead continues to hold a
blocked position until its first legal sellable close, with terminal recovery
only at the frozen 2025-12-31 boundary. All 15 positions later sold legally;
the longest occupied 262 sessions. This semantic repair changes only execution
of already selected positions and does not use later outcomes to select stocks.

Primary 2012-2025 results:

- CNY 1,000,000 to CNY 2,737,860 after final liquidation (`+173.79%`).
- Annualized log growth 7.68%; signal-period trading-day CAGR 8.15%.
- Maximum drawdown `-36.65%`; 11 of 14 calendar years were positive.
- 1,537 closed trades, 50.55% winners, and 60.98 mean occupied sessions.
- Mean utilization 56.86%, maximum 42 concurrent positions.
- Fees and slippage CNY 234,684; turnover 108.49 times starting capital.
- Maximum observed order participation 0.4836%; no order exceeded the 0.5%
  capacity limit.

The frozen full-history gate fails only because drawdown exceeds its 35% limit
by 1.65 percentage points. The main drawdown occurs during the 2015 market
crash, although calendar 2015 itself ends up +18.43%. Negative calendar years
are 2016 (-5.78%), 2018 (-16.19%), and 2022 (-2.04%).

A fresh CNY 1,000,000 restart in 2023 ends 2025 at CNY 1,265,406, with 9.42%
annualized log growth and -10.51% maximum drawdown. Calendar returns are about
+0.15%, +14.65%, and +10.21%; the recent restart gate passes. These years were
already inspected and are not an untouched future sample.

Pressure and implementation sensitivities:

- D80 write-to-zero pressure case: CNY 2,047,007, 5.46% annualized log growth,
  and -47.86% drawdown. It fails the drawdown gate.
- Allowing independent monthly lots of a repeated stock: CNY 3,721,377, 10.02%
  annualized log growth, but -40.06% drawdown and only 10/14 positive years. It
  also fails and is not the primary rule.
- Raw-price-only sensitivity: CNY 1,799,450 and -45.11% drawdown. This omits
  total-return adjustment and is not the economically preferred result.

## Decision

The supplemental framework was worth testing, but its components do not behave
as the narrative suggests. There is no evidence here that combining more
reasonable-sounding fundamental filters automatically improves returns. The
durable part is simpler: a diversified, point-in-time TTM-FCF-yield exposure
has historical value after costs and in relative-return diagnostics.

It still does not meet the project's definition of a stable executable policy:
the primary exact account misses the drawdown gate, three calendar years lose
money, all historical periods are now consumed, and the rule was discovered
after inspecting the same history. It therefore remains a secondary
interpretable candidate, below the already retained risk-budgeted
margin-residual policy.

Do not tune rank phase, breadth, holding day, or exposure on 2012-2025 and call
the result independent. The legitimate next use is either a pre-frozen
diversifying sleeve in a separately declared multi-strategy portfolio test or
forward-only shadow validation after the research boundary is explicitly
advanced.

## Authoritative artifacts

- Causal value contract and implementation:
  `daily_research/studies/seq100_causal_value_policy_v1.json` and
  `daily_research/path_policy/seq100_causal_value_policy.py`
- Value overlay:
  `daily_research/studies/seq100_margin_value_overlay_v1.json` and
  `daily_research/path_policy/seq100_margin_value_overlay.py`
- TTM repair and value-definition robustness:
  `daily_research/studies/seq100_ttm_value_account_v1.json`,
  `daily_research/path_policy/seq100_ttm_value_account.py`, and
  `daily_research/path_policy/seq100_value_definition_robustness.py`
- Exact rotation account:
  `daily_research/studies/seq100_ttm_fcf_rotation_account_v1.json` and
  `daily_research/path_policy/seq100_ttm_fcf_rotation_account.py`
- Exact output summary:
  `daily_research/output/path_policy/studies/seq100_ttm_fcf_rotation_account_v1/summary.json`
