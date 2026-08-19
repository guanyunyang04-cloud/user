# Daily Research

The active short-horizon research path is `daily_research.technical`. It keeps
data access, model scores and legal finite-cash portfolio replay separate. See
`technical/README.md` for the current contract and commands.

## Current baseline

- Research rows: 4,191,476 point-in-time stock-days through 2025-12-31.
- Primary tabular input: 158 fields derived from daily price/volume and market
  state.
- Optional challenger: 25 same-day 5-minute fields, for 183 fields in total.
- Independent model: a 60-trading-day raw OHLCVA temporal model plus 14
  whole-market state fields.
- Current development benchmark: a frozen 50/50 within-date rank average of the
  158-field tree and raw sequence model. It is historical research, not a live
  trading system.

The older `path_policy` package and `research_records` remain available as
historical evidence and as the source of the repaired data pack. They are no
longer the active model-development entry point. In particular, old D3 live
bundles and all model results produced before the minute-feature repair must not
be treated as current.

## Commands

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.technical verify
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.technical train-tree --features 158
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.technical train-tree --features 183
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.technical compare-tree
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.technical train-sequence
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.technical evaluate-ensemble
```

Outputs are written under `daily_research/output/technical/`. The minimum
research protections remain mandatory: point-in-time availability, unique and
aligned row keys, no 2026 outcome reads, purged forward folds, next-open
execution, T+1, suspension and price-limit constraints, transaction costs,
finite cash, and no replacement of an unfilled higher-ranked order.
