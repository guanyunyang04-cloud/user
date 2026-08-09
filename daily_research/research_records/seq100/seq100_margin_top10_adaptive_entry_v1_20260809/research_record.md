# Margin Top10 Adaptive/Down-Day Entry Study (corrected 2026-08-09)

Status: completed adaptive retrospective study. The requested union rule failed
its entry gate. No finite-account replay or production use is authorized.

## Timing correction

This record supersedes the earlier run stored under the same study id. The
earlier implementation joined a financing observation to its next-session
`feature_available_date`, combined it with that later day's K-line, and then
bought one more session later. That made every decision one trading day slower
than the user's intended operation.

The corrected causal timeline is:

1. trading day T closes and creates the T financing balance and T K-line;
2. the T financing observation is published before the T+1 open;
3. the selector combines the T financing change with the T K-line;
4. the order enters at the first legal T+1 raw open;
5. because the newly bought A-share position cannot be sold on T+1, the fixed
   one-day holding rule first requests the T+2 close.

The code now requires `margin_source_date_idx == signal_date_idx` and
`margin_available_date_idx == signal_date_idx + 1`. Both assertions passed for
every candidate. This convention matches the exchange publication rules:

- SSE publishes prior-trading-day single-stock margin data before each trading
  day's open: <https://www.sse.com.cn/lawandrules/sselawsrules2025/trade/specific/margin/c/c_20250616_10782015.shtml>
- SZSE's margin-trading guide states the same previous-day/before-open timing:
  <https://docs.static.szse.cn/www/lawrules/service/member/W020230901534859402978.pdf>

## Frozen rule and execution

The selector requires at least three consecutive point-in-time increases in
`rzye`, then ranks the absolute CNY balance increase Top10 inside
`quality_liquidity_pit`. It buys after either:

1. the prior valid close crossed above the smoothed adaptive line and the
   signal-day low remained above the signal-day line (`cross_support`); or
2. the signal-day close was below the prior valid close (`down_close`).

Close below open remains a sensitivity. The adaptive line is the supplied
ER10/fast2/slow30 recursive DMA followed by EMA2, computed on back-adjusted
valid stock bars. Raw prices determine lots, fees and minimum commission;
adjustment factors preserve corporate-action economics. Each diagnostic order
uses CNY 100,000 and the frozen double-slippage contract. Missing or blocked
fills remain cash. The formal outcome range is 2012-2025; 2010-2011 is only
indicator burn-in, and no 2026 outcome is read.

## Coverage

- 33,586 Top10 candidates on 3,387 dates and 1,615 symbols.
- 1,184 `cross_support`, 15,260 `down_close`, and 16,174 primary-union
  candidates; only 270 satisfy both.
- The source-date/signal-date and publication-date/entry-date audits are true.
- A negative financing increment becomes available by D60 in 99.92% of cases;
  the median availability offset from the signal date is three sessions.

## Corrected primary result

| Rule | Period | Candidate exact net mean | Median | Win rate | Equal-date cash mean | HAC 95% interval | Positive years |
|---|---:|---:|---:|---:|---:|---:|---:|
| All Top10 | 2012-2025 | -0.387% | -0.580% | 42.21% | -0.384% | [-0.504%, -0.263%] | 1/14 |
| Prior-cross support | 2012-2025 | -0.010% | -0.439% | 44.25% | -0.052% | [-0.340%, +0.236%] | 8/14 |
| Down-close | 2012-2025 | -0.276% | -0.438% | 43.49% | -0.279% | [-0.400%, -0.158%] | 2/14 |
| User union | 2012-2025 | -0.262% | -0.438% | 43.56% | -0.281% | [-0.403%, -0.160%] | 2/14 |
| User union | 2023-2025 | -0.300% | -0.523% | 40.95% | -0.308% | [-0.517%, -0.100%] | 0/3 |
| Intersection | 2012-2025 | +0.036% | -0.538% | 42.70% | +0.002% | [-0.460%, +0.464%] | 7/14 |
| Intersection | 2023-2025 | +0.113% | -0.693% | 41.27% | -0.129% | [-0.770%, +0.512%] | 2/3 |

The correction materially changes the estimate for `cross_support`: it is now
near zero rather than clearly negative. It still is not confirmed profitable.
Its late-period candidate mean is +0.134%, but the equal-date mean is only
+0.015%, the interval is [-0.578%, +0.608%], the median is -0.660%, and only
1/3 late years is positive. The small intersection is even less conclusive.

The user's union remains decisively negative after the timing correction. Its
gross executable T+1-open to T+2-close mean is +0.166%, but the exact net mean
is -0.262%; the typical trade is negative and the small gross movement cannot
cover the execution contract. In 2023-2025 its gross/net means are about
+0.095%/-0.300%.

## Financing-decrease and higher-price exits

The first causally available negative balance increment requests that same
session's close, but never before T+2, and otherwise times out at D60. It does
not rescue the union:

- 2012-2025: candidate -0.283%, equal-date -0.325%, 3/14 positive years;
- 2023-2025: candidate -0.360%, equal-date -0.461%, 0/3 positive years.

The legal D2 high exceeded the entry price in 72.9% of filled union cases and
reached +0.5%, +1%, +2%, +3%, and +5% in about 65.4%, 56.5%, 41.8%, 30.4%,
and 16.9%. This shows that a higher legal price often exists, not that it can
be captured ex ante. Resting D2 take-profit targets from 0.5% through 5% all
retain negative full-history and late-period net means. The 5% union target,
for example, is -0.277% per candidate over 2012-2025 and -0.312% in 2023-2025.
Targets clip the rare large winners while misses retain their loss tails.

## Corrected fixed-model diagnostic

Models are fixed on 2012-2019, calibrated on 2020-2022 and evaluated once on
2023-2025. They are exploratory filters, not untouched confirmation.

- The next-close-up model has test AUC 0.562. Top1 continues 52.34% of the
  time but loses -0.429% per equal-date cash selection; Top3 loses -0.371%.
- The exact-legal-net model has test AUC 0.533. Its Top1 point estimate is
  +0.028% per day and improves on all Top10 by +0.465 percentage point, but
  its absolute HAC interval is [-0.341%, +0.398%] and only 2/3 years are
  positive. Top3 remains -0.201% per day.

The legal-net Top1 is a weak follow-up candidate, not evidence of stable
profit. It was selected after the historical study existed and needs forward
or genuinely new data before any production claim.

## Decision

The corrected result withdraws the earlier one-day-late conclusion. Under the
user's intended timing, the broad union rule still fails. Prior-cross support
alone deserves the softer label `weak_near_zero_unconfirmed`, not `rejected as
clearly negative`. No fixed target, financing-decrease exit or fixed model has
yet converted the hypothesis into a stable net-positive rule, so account
replay remains gated off.

## Reproducibility

- Contract: `daily_research/studies/seq100_margin_top10_adaptive_entry_v1.json`
- Implementation: `daily_research/path_policy/seq100_margin_top10_adaptive_entry.py`
- Tests: `daily_research/path_policy/tests/test_seq100_margin_top10_adaptive_entry.py`
- Output: `daily_research/output/path_policy/studies/seq100_margin_top10_adaptive_entry_v1/`
- Corrected study SHA-256:
  `8f1677026dae158e08d056bd8a82aa828dbb2282eee8da3f00d6069a2769166e`
- Corrected implementation SHA-256:
  `883c8e8183c0b420085bd9862f2344dfb072798d9da7e4dd6c6d502b181cccec`

