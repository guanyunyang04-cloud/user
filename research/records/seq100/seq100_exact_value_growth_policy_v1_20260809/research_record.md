# Exact main-board value-growth policy and finite account

Date: 2026-08-09

## Question

Can the quantifiable core of the supplied main-board stock-selection workflow
be translated into a causal, repeatable policy that adds information beyond
simple value screens and grows a finite A-share account after legal execution
and costs?

The supplied workflow assigns 30% to valuation, 30% to actual earnings and
estimate revisions, 25% to quality, 10% to price confirmation and 5% to
governance. It forbids rewarding distance from the 52-week low and asks for
cycle-normalized earnings rather than blindly capitalizing peak profits.

## Frozen historical translation

- Formal history: 2012-01-01 through 2025-12-31. No 2026 row is read.
- Universe: `quality_liquidity_pit`. Every historical member already has one
  of the allowed Shanghai or Shenzhen main-board prefixes.
- Signal: the last PIT pool date in every calendar month.
- Selection: Top10 by the exact 30/30/25/10/5 score, with at most two names per
  PIT industry. A Top20/cap-three arm was frozen as a diversification control.
- Valuation: positive next-fiscal-year consensus net-profit yield, lower
  positive current PE and PB, causal TTM FCF yield, and cycle-normalized
  earnings yield.
- Earnings/revision: actual net-profit/revenue growth, performance-forecast
  change and CFO/income, plus 30- and 90-calendar-day next-FY net-profit and
  EPS revisions. Consensus requires at least three institutions.
- Cycle normalization: for general companies, causal TTM revenue times the
  median of up to five known annual parent-net margins; for financial firms,
  current PIT parent equity times the median known annual parent ROE. The
  trailing 25th/50th/75th percentiles are retained as bear/base/bull states;
  only the base state enters the score.
- Quality: ROE, net margin, CFO/income, FCF/revenue, and lower leverage,
  receivables, inventory, goodwill and borrowing ratios.
- Confirmation: both 20-day industry-relative return and 20-day trend slope
  must be non-negative. No low-price or 52-week-low reward is used.
- Governance: only observable announcement and data-conflict proxies are used.
  A recent delisting-risk announcement is a veto. Historical qualitative moat
  and management scores are not fabricated.
- Primary outcome: next-open buy request and D60 legal close exit, with a 0.6%
  round-trip cohort cost proxy. Candidate selection is materialized before any
  return is read.

EV/EBITDA was audited but excluded from the historical score: three-institution
coverage among matched 2012-2019 candidates was only about 46.5%, so requiring
it would materially redefine the early universe.

## Information result

The primary Top10 arm had 164 calendar-evaluable monthly cohorts. Mean selected
count was nearly ten; the first 2012 month had only six eligible names and cash
was retained for the missing slots.

| Period | Monthly stress-net mean | HAC 95% lower bound | Industry-residual mean | Positive years |
|---|---:|---:|---:|---:|
| 2012-2019 | 3.86% | -0.25% | 1.40% | 6/8 |
| 2020-2022 | 5.49% | 1.91% | 1.30% | 3/3 |
| 2023-2025 | 4.32% | -1.40% | 3.26% | 3/3 |
| 2012-2025 | 4.31% | 1.55% | 1.74% | 12/14 |

For the full history, the block lower bound for stress-net return was 1.50%.
The industry-residual HAC and block lower bounds were 0.21% and 0.23%.
Removing the best year left a 3.73% mean monthly stress-net return. Removing
2012, whose cycle-history coverage was unavailable, left a 4.49% stress-net
mean with 1.62% HAC and 1.69% block lower bounds; the industry-residual lower
bounds also remained positive.

The complete arm beat the forward-earnings-yield-only and TTM-FCF-only controls
in point mean in all three periods. Those paired improvements were not
individually significant: their full-history lower bounds remained negative.
Thus the evidence supports the complete ranking as the current best historical
candidate, but does not prove that every added component has independent
value.

Full-history diagnostic means:

| Arm | Monthly stress-net mean | Positive years |
|---|---:|---:|
| Exact Top10 | 4.31% | 12/14 |
| Exact Top20 | 3.88% | 12/14 |
| Exact without cycle normalization | 3.97% | 12/14 |
| Exact without governance | 4.35% | 12/14 |
| Forward value only | 3.07% | 11/14 |
| TTM FCF only | 2.85% | 10/14 |

The quantitative governance proxy did not add return in this sample. It is
retained as a risk convention, not claimed as historical alpha.

## Exact finite account

Because the frozen information gate passed, the primary selection was replayed
in the shared exact account kernel:

- CNY 1,000,000 initial cash;
- monthly Top10 orders and a 30-name concurrent-position budget;
- next legal raw-price open entry;
- D60 close sale request, deferred until the first legal sale;
- no pyramiding;
- 100-share board lots, minimum commission, transfer fee, historical stamp tax;
- double slippage and a 0.5% signal-day amount capacity cap;
- raw prices for fills and costs, with the pack-pinned back-adjust-factor ratio
  for total-return-equivalent marks and proceeds.

The full-exposure account reached CNY 4.305 million after liquidation, with
11.13% annualized log growth, 20.45% annualized volatility and 40.95% maximum
drawdown. It had 11/14 positive calendar years. It failed the frozen 30%
drawdown gate.

A single risk budget was then derived without a grid. Development-only
2012-2019 realized account volatility was 21.86%; 15% divided by that value
froze gross exposure at 68.6089%.

The risk-budgeted account:

- reached CNY 2.863 million after liquidation;
- annualized log growth: 8.02%;
- annualized volatility: 13.89%;
- maximum drawdown: 29.22%;
- positive years: 11/14;
- closed trades: 1,353;
- winning-trade rate: 53.4%;
- median net trade return: 1.35%;
- fees and slippage: about CNY 232,723;
- maximum observed signal-day amount participation: 0.3205%, below the 0.5%
  cap;
- no missing required total-return factor.

Annual risk-account returns were:

`2012 +4.72%, 2013 +9.01%, 2014 +28.14%, 2015 +11.31%, 2016 -1.67%,
2017 +13.26%, 2018 -12.17%, 2019 +13.72%, 2020 +4.00%, 2021 +10.09%,
2022 +3.11%, 2023 -1.35%, 2024 +15.62%, 2025 +17.40%`.

A fresh CNY 1,000,000 restart at the first 2023 signal reached CNY 1.271
million, with 9.59% annualized log growth, 10.32% volatility and 10.76% maximum
drawdown. It lost 5.18% in 2023 and gained 15.65%/15.90% in 2024/2025.

Profit was not dominated by a single stock. The five largest positive symbol
contributors supplied about 9.0% of aggregate positive symbol P&L and the ten
largest supplied about 16.2%; 527 distinct symbols traded.

## Interpretation

This is the strongest historical result yet obtained from the value branch and
the first supplied 30/30/25/10/5 translation to pass both its information gate
and its predeclared risk-budgeted account diagnostics. It is a serious candidate
for forward shadow execution.

It is not a stable-profit guarantee or a production strategy:

- all 2012-2025 years are already consumed and no independent historical
  holdout remains;
- three calendar years lost money and the historical maximum drawdown was
  still about 29%;
- period-specific uncertainty remains large, especially in 2012-2019 and
  2023-2025;
- paired improvement over the simple value controls was positive but not
  statistically decisive;
- 2012 lacked three years of cycle history and used the preregistered
  available-component fallback;
- qualitative moat/governance review was not historically backfilled;
- total-return-equivalent corporate-action accounting is not an exact
  dividend-payment-date and odd-lot cash ledger.

## Forward boundary

The user-supplied public-data Top10 as of the 2026-08-07 close was frozen in
`seq100_exact_value_growth_2026_shadow_v1`. It is deliberately labeled as a
manual/public-data cohort, not as the mechanical parser's 2026 output. Its ten
orders are pending the planned 2026-08-10 next open; no fill, exit or realized
performance has been recorded. Historical inputs were not modified.

Do not tune this rule further on 2012-2025. The next legitimate work is an
operational mechanical signal generator and append-only 2026+ shadow ledger,
followed by enough forward observations to judge field availability, fill
behavior, turnover, drawdown and whether the historical edge persists.

## Artifacts

- Historical study: `daily_research/studies/seq100_exact_value_growth_policy_v1.json`
- Historical implementation: `daily_research/path_policy/seq100_exact_value_growth_policy.py`
- Historical output: `daily_research/output/path_policy/studies/seq100_exact_value_growth_policy_v1/`
- Account study: `daily_research/studies/seq100_exact_value_growth_account_v1.json`
- Account implementation: `daily_research/path_policy/seq100_exact_value_growth_account.py`
- Account output: `daily_research/output/path_policy/studies/seq100_exact_value_growth_account_v1/`
- Forward study: `daily_research/studies/seq100_exact_value_growth_2026_shadow_v1.json`
- Forward implementation: `daily_research/path_policy/seq100_exact_value_growth_shadow.py`
- Forward output: `daily_research/output/path_policy/studies/seq100_exact_value_growth_2026_shadow_v1/`
