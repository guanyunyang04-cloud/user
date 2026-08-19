# Technical research mainline

This package is the small active path for daily technical research. It does not
contain live order generation.

## Data contract

The current repaired matrix contains 4,191,476 rows from 2012-01-04 through
2025-12-31. The first two years of the underlying pack are available only as
lookback/training warm-up. No 2026 outcome is allowed.

- `158`: 104 daily price-volume features and 54 market-state features.
- `183`: the same 158 fields plus 25 features calculated from the signal day's
  5-minute bars.
- `raw60`: normalized raw open, high, low, close, volume and amount over the
  latest 60 trading days, a valid-bar mask, and 14 whole-market state fields.

The 158-field set is a validated prefix of the 183-field matrix. There is no
separate 158 data copy and no v2/v3 data namespace.

## Research contract

- Five expanding forward folds evaluate 2020-2025 with a 30-trading-day purge.
- The common target is executable D10 base-cost net return; portfolio stress
  replay applies double slippage.
- Models rank the complete finite-score slate before future fill/outcome status
  is consulted. An unfilled or unaffordable selected order remains cash and is
  never replaced by a lower-ranked stock.
- Account replay uses next-open entry, T+1, legal delayed exits, 100-share lots,
  finite cash, commission, transfer fee, stamp tax and slippage.
- The tree/sequence ensemble is a fixed 50/50 average of daily percentile
  ranks. Its weight is not searched on OOF data.

## Commands

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.technical verify
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.technical train-tree --features 158
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.technical evaluate-tree --features 158
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.technical train-tree --features 183
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.technical evaluate-tree --features 183
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.technical compare-tree
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.technical train-sequence
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.technical evaluate-sequence
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.technical evaluate-ensemble
```

`verify` performs the current lightweight quality gate, including complete-file
hashing, unique row keys, date bounds, finite/range checks and exact equality of
the 183 projection with its 557-field source matrix.

## Current interpretation

The 183-field tree does not provide a statistically reliable paired improvement
over the 158-field tree. Same-day 5-minute data remains an optional challenger,
not a required source. The raw sequence model adds a genuinely different rank
signal. Their fixed ensemble has the best current OOF Rank IC and uncapped
no-overlap account point estimates. Its IC increment is statistically clear
against the tree, but not against the raw sequence model; direct TopK and account
increments generally include zero. Winner-capped results are still weak. It is
a development benchmark, not evidence of stable profitability.
