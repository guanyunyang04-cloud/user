# Current state

Updated: 2026-08-09

## Active objective

Find a repeatable, causally observable, legally executable A-share policy that
can grow a finite account after costs with defensible uncertainty and risk.
The objective is risk-constrained account wealth, not fidelity to Chan theory,
financing-flow stories, a fixed classifier score, or a favorable hindsight
chart.

The strongest current candidate is now the exact quantifiable main-board
30/30/25/10/5 value-growth policy with a frozen development-only risk budget.
It passed its frozen historical information and account gates and is retained
for forward shadow validation. It is not a production strategy or a profit
guarantee because every 2012-2025 year is consumed, three account years lost
money, and no independent forward sample has completed.

There is no active training, real order, or production process. A separately
isolated 2026-08-07 public-data Top10 shadow cohort is frozen with pending
forward fills only.

## Formal research contract

- Formal history is 2012-01-01 through 2025-12-31. Years 2010-2011 are burn-in
  only. The user authorized an isolated append-only 2026 shadow boundary; no
  2026 observation may flow back into historical selection, fitting, or claims.
- The primary universe is signal-date `quality_liquidity_pit`.
- Signal-close decisions may use only information causally available by that
  close. Selection never uses future entry, sellability, return, or parser
  outcomes.
- Execution must model next-open fills, T+1, suspension, limits, legal-sale
  deferral, 100-share lots, minimum commission, transfer fee, stamp tax,
  slippage, capacity, concurrent holdings, cash, and terminal liquidation.
- Formal performance uses adjusted total-return economics while costs and
  executable prices remain raw exchange prices.
- Development, confirmation, and account layers have separate gates. A strong
  account curve cannot override a failed source-alpha confirmation gate.
- 2023-2025 have now been consumed for the current candidate. Do not retune it
  on those years and call the result independent confirmation.

## Authoritative data state

The corrected Seq100 chain is certified through 2025-12-31. The complete
feature intersection has 4,191,476 rows by 557 fields. PIT ST, suspension,
limit, publication timing, nullable state, and historical membership semantics
are enforced. QDP owns source truth; Daily Research owns derived features,
models, studies, and account outputs.

Current input fingerprint:
`a2bea9ed174b0a04ea2e97af475e9b4f1da79d739288a0cd92114998e31784ae`.
Current label fingerprint:
`c93cf15be7f58642b34e086c0c25e00b56a4f46de514d01a35b3132f2e0bc65d`.

## Conclusions that govern the next step

### Strict Chan and K-line structures

The strict causal Chan parser, prefix audits, chart review, literal Type-1,
Type-2, and Type-3 buy-point screens, five holding horizons, independent Type-1
confirmation, transparent filters, richer structural attributes, matched
controls, and exit probes are complete. The parser/audits are valid, but no
tested buy-point family or richer structure representation produced stable
out-of-time net value. This rejects the current formal definitions as direct
trading rules; it does not prove that every Chan interpretation or every K-line
dependency is false.

Fixed take-profits, hard stops, trailing stops, and armed trailing stops did
not repair the residual entry edge. They generally clipped rare large winners
while retaining losers. Do not reopen this branch with another small holding
day/stop grid.

### Financing information

Financing-balance growth, consecutive increases, and aggregate financing-flow
cash gates did not show robust incremental value. Financing cannot be treated
as observed smart-money intent.

One related state did matter: requiring
`margin_detail_coverage_state == observed`. It is an observable coverage and
eligibility constraint, not a bullish balance-growth signal. Its mechanism may
be liquidity, margin eligibility, institutional coverage, or data quality and
is not identified.

### Causal value and TTM free-cash-flow branch

The first causal value study tested only an approximate mechanical core; its
composite and analyst-revision variants did not improve consistently. A later
study then froze the exact quantifiable main-board 30/30/25/10/5 weights,
forward consensus earnings yield, causal TTM FCF, cycle-normalized bear/base/
bull earnings, price confirmation and observable governance proxies. It still
does not fabricate historical qualitative moat, management judgment or manual
Top5 review. Distance from the 52-week low is explicitly forbidden. Prior
valuation-median/right-edge exits underperformed D60 holding by clipping rare
large winners and remain rejected as automatic take-profits.

The compact free-cash-flow field was not TTM: it mixed the latest Q1, H1, Q3,
or annual cumulative period. A causal TTM bridge was implemented from statement
versions available at each signal close. Coverage is about 95.1%, and its
Spearman correlation with the old latest-period field is only about 0.516. The
old field is not accepted as TTM evidence.

The durable component is a broad industry-aware TTM-FCF-yield exposure, not
precise Top10 stock picking. A natural, non-optimized monthly 12-of-Top48 rank
rotation was replayed with raw prices, total-return factors, exact lots/minimum
fees/taxes, double slippage, 0.5% amount capacity, no pyramiding, and D60 legal
sale requests. Blocked positions now remain held until their first legal sale;
the former D80 write-to-zero behavior is retained only as a pressure case.

The exact 2012-2025 account grew CNY 1 million to about CNY 2.738 million after
liquidation, with 7.68% annualized log growth, 11/14 positive years, and 36.65%
maximum drawdown. A fresh 2023-2025 restart ended near CNY 1.265 million with
10.51% maximum drawdown and all three years positive. The primary frozen gate
still failed because drawdown exceeded its 35% limit. Pyramiding raised growth
but worsened drawdown to about 40%; it also failed. This is a secondary
interpretable historical candidate and does not displace the exact
30/30/25/10/5 policy.

### Exact main-board 30/30/25/10/5 policy

The exact quantifiable Top10 policy passed its frozen full-history information
gate. Across 164 evaluable monthly cohorts, mean D60 stress-net return was about
4.31%, its HAC/block lower bounds were about 1.55%/1.50%, and the industry-
residual lower bounds were about 0.21%/0.23%. Twelve of fourteen annual cohort
means were positive and excluding the best year left a positive mean. The
complete score beat forward-value-only and TTM-FCF-only controls in point mean
in all three periods, but paired lower bounds were negative; added-component
superiority is therefore not statistically decisive. Top20, no-cycle and
no-governance controls remained positive. The governance proxy did not add
historical return.

The full-exposure exact account reached about CNY 4.305 million, but 40.95%
maximum drawdown failed the 30% gate. A single 15% target-volatility budget,
derived only from 2012-2019 realized account volatility, froze gross exposure
at 68.61%. That account reached about CNY 2.863 million over 2012-2025, with
8.02% annualized log growth, 13.89% volatility, 29.22% maximum drawdown, and
11/14 positive years. A fresh 2023-2025 restart reached about CNY 1.271 million
with 10.76% maximum drawdown, but lost 5.18% in 2023. Exact lots, minimum fees,
taxes, double slippage, 0.5% amount capacity, legal-sale deferral and total-
return factors are modeled. This is the strongest historical candidate, not a
stable-profit guarantee.

### Prior margin-residual benchmark

The earlier strongest candidate remains a useful non-value benchmark. Its
frozen rule is:

1. Start from the full daily PIT quality/liquidity pool.
2. Veto the highest predicted D20 bad-tail decile.
3. Require observed margin-detail coverage.
4. Rank by the frozen expanding-window LightGBM D20 stock-residual mean.
5. Cap each PIT industry at four names and retain Top48.
6. Enter next legal open and request a D20 close exit with legal-sale deferral.

Expanded development (2014-2022) passed: mean stress-net D20 cohort return was
about 1.68%, HAC lower bound 0.45%, block lower bound 0.38%, and 8/9 years were
positive.

Frozen confirmation (2023-2025) had positive point estimates in all three
years and strong residual/excess lower bounds. Mean stress-net cohort return
was about 1.44%, but the absolute HAC and block lower bounds were about -0.21%
and -0.20%. The formal alpha confirmation gate therefore failed narrowly.

### Finite account and risk

The company-action correction uses raw prices for fills and costs and the
pack-pinned back-adjust-factor ratio for total-return-equivalent marks and sale
proceeds. The factor panel covers every required raw open/close from 2014-2025;
no 2026 row is read. This is not an exact dividend-payment-date/odd-lot ledger.

At full exposure, the unchanged policy turned CNY 1 million into about CNY
6.25 million over 2014-2025 after double slippage, but maximum drawdown was
about 50%. A fresh 2023-2025 account ended near CNY 1.54 million with about 31%
maximum drawdown. Mechanical feasibility is strong, but full exposure is too
volatile to call stable.

A single risk budget was frozen without a grid: 15% target volatility divided
by development-only realized volatility fixed gross exposure at 54.78%.
Development and retrospective overlay gates both passed. The fraction did not
use 2023-2025 data, but the underlying base confirmation path was already
known, so this is not a second untouched alpha holdout. The risk account:

- 2014-2025: CNY 1 million to about CNY 3.10 million; annualized log growth
  about 10.0%; volatility about 14.2%; maximum drawdown about 29.3%.
- Fresh 2023-2025: CNY 1 million to about CNY 1.270 million; annualized log
  growth about 9.3%; volatility about 12.1%; maximum drawdown about 16.9%; all
  three calendar years positive.

Four development years remained negative. The risk layer improves usability;
it does not turn historical evidence into guaranteed annual profit.

## True pause point and next step

The exact quantifiable value-growth policy now has a frozen selector, causal
information test, exact finite account, development-only risk budget and an
isolated 2026 shadow cohort. All 2012-2025 evidence is consumed. Further
historical weight, breadth, holding-day, exit or risk-budget tuning would spend
known outcomes and must not be presented as confirmation.

Next legitimate actions, in order:

1. Build the operational mechanical signal generator for future month ends and
   an append-only fill/mark/exit ledger. The supplied 2026 public-data Top10 is
   already frozen separately and must not be silently relabeled as mechanical
   output.
2. From 2026-08-10 onward, append only actually observable shadow fills, field
   availability, costs, legal exits, turnover, drawdown and policy drift. Never
   feed these observations back into the 2012-2025 artifacts.
3. Accumulate enough independent forward cohorts to test absolute net value and
   risk. One favorable trade or month is not confirmation.
4. If a new historical hypothesis is pursued, give it a new contract. Do not
   reopen Chan-point, right-edge take-profit, financing-growth, TTM-FCF phase,
   score-weight or breadth grids on the consumed history.
5. Current status is `strongest_candidate_forward_shadow_not_production`.

## Authoritative paths

- Durable result and interpretation:
  `daily_research/research_records/seq100/seq100_margin_residual_policy_v1_20260808/`
- Causal-value/TTM-FCF result and interpretation:
  `daily_research/research_records/seq100/seq100_causal_value_ttm_fcf_v1_20260809/`
- Exact value-growth result and interpretation:
  `daily_research/research_records/seq100/seq100_exact_value_growth_policy_v1_20260809/`
- Margin residual policy and outputs:
  `daily_research/studies/seq100_margin_residual_policy_v1.json` and
  `daily_research/output/path_policy/studies/seq100_margin_residual_policy_v1/`
- Account/risk contract and outputs:
  `daily_research/studies/seq100_margin_residual_account_feasibility_v1.json`
  and
  `daily_research/output/path_policy/studies/seq100_margin_residual_account_feasibility_v1/`
- Implementations:
  `daily_research/path_policy/seq100_margin_residual_policy.py`,
  `daily_research/path_policy/seq100_margin_residual_account_feasibility.py`,
  and `daily_research/path_policy/seq100_finite_capital_backtest.py`.
- Value/TTM contracts and implementations:
  `daily_research/studies/seq100_causal_value_policy_v1.json`,
  `daily_research/studies/seq100_ttm_value_account_v1.json`,
  `daily_research/studies/seq100_ttm_fcf_rotation_account_v1.json`, and their
  matching `daily_research/path_policy/seq100_*` modules.
- Exact value-growth historical, account and shadow contracts:
  `daily_research/studies/seq100_exact_value_growth_policy_v1.json`,
  `daily_research/studies/seq100_exact_value_growth_account_v1.json`, and
  `daily_research/studies/seq100_exact_value_growth_2026_shadow_v1.json`, with
  matching `daily_research/path_policy/seq100_exact_value_growth_*` modules.
- Strict-Chan implementation and evidence remain under
  `daily_research/path_policy/seq100_strict_chan_*`,
  `daily_research/studies/seq100_strict_chan_*`, and their study outputs.
