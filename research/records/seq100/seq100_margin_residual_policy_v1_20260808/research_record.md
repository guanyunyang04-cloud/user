# Margin-observed residual policy and finite-account feasibility

Status: strongest historical candidate so far; risk-budget confirmation passed,
but the source alpha policy's strict absolute confirmation gate failed. This is
research evidence, not a production strategy or a profit guarantee.

Date: 2026-08-08

## Decision

The literal strict-Chan route is not being promoted. Causal first-, second-,
and third-buy points, five fixed holding horizons, an independent Type-1
confirmation sample, transparent coordinate filters, richer structure
attributes, and several exits were tested. None produced a stable positive
uncertainty lower bound after costs.

The first materially stronger non-Chan candidate is retained for forward
validation:

1. Start from the signal-date `quality_liquidity_pit` universe.
2. Veto the highest predicted D20 bad-tail decile over the full daily pool.
3. Require `margin_detail_coverage_state == observed` at the signal close.
4. Rank by the frozen expanding-window LightGBM D20 stock-residual forecast.
5. Admit at most four names per point-in-time industry and retain Top48.
6. Enter at the next legal open, hold to D20, and defer a blocked sale to the
   first legal close in the bounded execution tail.
7. Apply lots, minimum commission, transfer fee, stamp tax, double slippage,
   no pyramiding, 48 slots, and a 0.5% signal-day amount cap.

The margin condition means that trustworthy stock-level financing detail is
observed for the name. It does **not** require financing balance to increase.
Direct balance growth, consecutive increases, aggregate market financing-flow
gates, and the narrative that financing automatically represents smart money
did not add robust value in the completed screens. The retained condition may
proxy margin eligibility, liquidity, institutional coverage, or data quality;
its economic mechanism is not identified.

## Frozen cohort evidence

Development covers 2014-2022. The policy was frozen after the 2017-2022
exploration; 2014-2016 forecasts were newly reconstructed before confirmation.
The D20 cohort return uses a conservative 0.6% cost proxy.

- Mean stress-net cohort return: `+1.6821%`.
- HAC 95% lower bound: `+0.4456%`.
- Moving-block 95% lower bound: `+0.3819%`.
- Selected excess-return HAC lower bound: `+0.8950%`.
- Positive years: `8/9`; observed-outcome coverage: `99.08%`.
- The predeclared development gate passed every check.

Confirmation covers 2023-2025 without a policy change.

- Mean stress-net cohort return: `+1.4403%`.
- Annual point estimates were positive in all three years.
- Stock-residual and selected-excess lower bounds remained clearly positive.
- Absolute stress-net HAC lower bound was `-0.2066%`; moving-block lower bound
  was `-0.2021%`.
- Therefore the strict confirmation gate failed. Relative ranking information
  confirmed, but standalone absolute value remains uncertain at the declared
  confidence level.

The separately tested market/industry hierarchical cash gate failed in
development. Simple trend, breadth, index, financing-flow, volatility,
downside-risk, size, and industry variants did not turn its absolute lower
bound positive. Fixed take-profit, hard-stop, trailing-stop, and armed-trailing
screens also failed: the residual policy depends on infrequent large winners,
so clipping the right tail generally hurt more than it helped.

## Corporate-action-corrected account replay

The finite-account replay is a non-promotional mechanical feasibility test. It
cannot override the failed cohort confirmation gate. Entry and costs use raw
exchange prices; marks and sale proceeds multiply by the pack-pinned
back-adjust-factor ratio. This is a total-return equivalent and fixes the old
raw-price split/dividend distortion, but it is not an exact dividend payment
date and odd-lot share ledger.

At full exposure, 2014-2025 produced:

- CNY 1,000,000 to CNY 6,249,537 after liquidation (`+524.95%`).
- Annualized log growth `16.27%`, annualized volatility `26.72%`, zero-rate
  Sharpe `0.734`.
- Maximum drawdown `-50.22%`.
- 6,731 closed trades, `52.13%` winners, and mean holding time 20.28 sessions.
- Mean utilization `93.28%`; fees and slippage about CNY 1.66 million.
- One order was actively capacity-capped; no filled order exceeded 0.5% of
  signal-day amount.

A fresh CNY 1,000,000 restart on 2023-2025 produced CNY 1,538,666 after
liquidation (`+53.87%`), with `-30.63%` maximum drawdown. The three annual
returns were `+13.70%`, `+20.83%`, and `+12.00%`.

The raw-price-only sensitivity account ended at CNY 4,270,131. Its lower value
confirms that the old corporate-action omission was materially conservative
overall rather than a source of inflated performance.

## Frozen risk budget

No exposure grid was searched. The base account's daily volatility through
2022 was `27.38%`. A predeclared 15% target divided by that development-only
volatility froze gross exposure at `54.77997%`, bounded to `[25%, 100%]`.

Development risk-account results (2014-2022):

- CNY 1,000,000 to CNY 2,195,007 after liquidation.
- Annualized log growth `9.04%`, volatility `14.40%`, maximum drawdown
  `-29.30%`.
- Five of nine calendar years were positive; the declared development risk
  gate passed.

The retrospective overlay confirmation (same frozen exposure, fresh 2023
account) also passed:

- CNY 1,000,000 to CNY 1,269,504 after liquidation (`+26.95%`).
- Annualized log growth `9.29%`, volatility `12.10%`, maximum drawdown
  `-16.87%`, Sharpe `0.763`.
- Annual returns: 2023 `+8.03%`, 2024 `+10.14%`, 2025 `+6.70%`.

Across 2014-2025, the same risk account ended at CNY 3,096,843, with `10.03%`
annualized log growth and `-29.30%` maximum drawdown. Four development years
were still negative, so this is materially smoother, not a claim of profit in
every year.

## Interpretation and next boundary

This is the first candidate in the current corrected-data program whose frozen
risk layer passed both development and a 2023-2025 mechanical check. The risk
fraction used development data only, but the underlying 2023-2025 base account
path had already been inspected, so this is not a second untouched alpha
holdout. It
is much stronger evidence than the Chan buy-point results or financing-balance
ranking narrative. It still does not prove guaranteed or permanently stable
profit because:

- the source alpha confirmation failed its strict absolute HAC/block lower
  bound by about 0.2 percentage point per D20 cohort;
- 2023-2025 are now consumed and cannot be reused to tune the policy;
- only about 5% of daily Top48 candidates become trades because D20 positions
  occupy the 48 slots;
- the total-return factor is not a literal corporate-action cash ledger;
- maximum historical drawdown remains about 29% even after risk scaling;
- regime change and future data remain unknown.

Do not retune this candidate on 2014-2025. The next legitimate step is to
freeze an operational daily signal generator and shadow trade it forward, or
test a genuinely new hypothesis under its own development/confirmation split.
The current 2012-2025 contract forbids using 2026, so 2026 forward evaluation
requires an explicit research-boundary change.

## Authoritative artifacts

- Policy contract:
  `daily_research/studies/seq100_margin_residual_policy_v1.json`
- Cohort development and confirmation:
  `daily_research/output/path_policy/studies/seq100_margin_residual_policy_v1/`
- Account and risk contract:
  `daily_research/studies/seq100_margin_residual_account_feasibility_v1.json`
- Account outputs:
  `daily_research/output/path_policy/studies/seq100_margin_residual_account_feasibility_v1/`
- Implementations:
  `daily_research/path_policy/seq100_margin_residual_policy.py` and
  `daily_research/path_policy/seq100_margin_residual_account_feasibility.py`
