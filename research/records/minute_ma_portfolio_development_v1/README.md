# Minute-MA finite-cash development replay

This record summarizes the first full-universe account replay of the causal
minute-MA rule catalog. It is development evidence only. It does not select a
strategy, establish stable profitability, read 2025, or authorize QMT trading.

## Scope

- Signal outputs: April 2022 and September 2023 full point-in-time main-board
  studies.
- Trading dates: 39 (19 plus 20).
- Signals before the account layer: 6,113,466 plus 8,864,129 outcome rows.
- Account variants: 22 executable non-matched-control strategies times seven
  exit policies, or 154 accounts.
- Starting cash: CNY 100,000.
- Maximum simultaneous symbols: five.
- No separate single-stock or daily-entry cap.
- Base execution costs: 3 bps commission with CNY 5 minimum, 0.1 bps transfer
  fee, 7 bps slippage, and the historical A-share stamp-tax schedule.

Signals from different MA periods are deduplicated to the earliest causal
signal for one symbol/hour. Earlier signal minutes have priority. Same-minute
candidates use a deterministic hash tie-break with seed 7, so input row and
stock-code order cannot choose the portfolio. Same-time fills divide the cash
reserved for the remaining slots equally, subject to 100-share lots. The
account does not pyramid one symbol. T+1 is enforced. Stops, targets and
trailing rules are observed on the current minute and execute at the next legal
minute open. Suspended, delisted, unknown, missing and one-price-down states
delay exits instead of becoming synthetic fills. A newly opened position is
still marked and tracked from its observed same-session bars.

## Result

Nine of 154 accounts had a positive marked-to-market combined return. Three of
those nine ended with an unresolved position. The six positive accounts with
all positions closed were:

| Strategy | Exit policy | Combined return | 2022-04 segment | 2023-09 segment | Max drawdown | Closed trades |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `s1_touch_reclaim` | `next_open_3d` | 8.57% | 13.96% | -4.73% | -9.36% | 70 |
| `s0_distance_only` | `next_open_3d` | 5.00% | 12.98% | -7.06% | -13.38% | 70 |
| `s1_near_reversal` | `next_open_2d` | 4.21% | 9.57% | -4.89% | -8.04% | 100 |
| `s1_near_reversal` | `next_open_3d` | 4.11% | 5.69% | -1.49% | -12.46% | 70 |
| `s1_near_reversal` | `next_open_1d` | 3.82% | 4.94% | -1.06% | -5.78% | 195 |
| `s4_reclaim_volume_proxy` | `next_open_2d` | 3.59% | -1.18% | 4.83% | -8.09% | 100 |

No fully closed account was positive in both segments. The only account that
was marked positive in both segments, `s4_reclaim_volume_normal` with the fast
failure exit, ended with one unresolved position: CNY 83,277 cash plus a CNY
20,571 stale/terminal mark. It is not realizable-profit evidence. The nominal
best account returned 16.82% but also ended unresolved and reversed from
+64.65% in April 2022 to -29.05% in September 2023. Across all strategies, the
median combined return for every exit policy remained negative, ranging from
-11.90% to -17.08%.

These results are also specific to tie-break seed 7. The large change from the
earlier code-order replay confirms that selecting five names from many
same-minute candidates is part of the strategy, not an implementation detail.
No strategy or exit policy is selected until the candidate ranking or tie-break
survives multiple seeds and more market regimes.

The account layer produced 15,699 filled entries, 15,667 closed trades and 32
terminal open positions across all comparison accounts. Delayed-exit attempts
are reported in each account result. Counts are not unique market trades: each
signal is replayed independently under multiple strategy/exit accounts.

## Reproduction

Detailed generated files are intentionally kept outside Git under
`runs/minute_strategy_portfolio_apr_sep/` (`run_manifest.json`,
`portfolio_summary.json`, `equity.parquet`, and `trades.parquet`). Reproduce
the account comparison from the retained monthly event outputs with:

```powershell
$env:PYTHONPATH='H:\quant_project\src'
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quantlab research minute-strategy-portfolio `
  --workspace-root H:\quant_project `
  --signal-root runs/minute_ma_month_2022_04 `
  --signal-root runs/minute_ma_month_2023_09 `
  --output-root runs/minute_strategy_portfolio_apr_sep `
  --starting-cash 100000 --max-positions 5 --selection-seed 7 `
  --memory-floor-gib 0.5 --soft-memory-floor-gib 1.0 `
  --duckdb-threads 2 --duckdb-memory-floor-gib 2.0
```

The final account run took about 949 seconds. The earlier telemetry-instrumented
replay used the same two-month workload and observed a 5.60 GiB minimum system
availability, safely above the 0.5 GiB hard floor. The final rerun also
completed without a memory-floor exception.

The deterministic-seed follow-up is complete in
`research/records/minute_ma_portfolio_seed_stability_v1/`. It rejects the
unranked same-minute selection: all 154 strategy/exit variants have negative
ten-seed mean and median returns. The next step is a causal cross-sectional
rank, then additional 2022-2024 months. 2025 remains unread.
