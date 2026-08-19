# Current state

Updated: 2026-08-19

## Objective and boundary

Research a repeatable A-share ranking policy using causally observable data and
legal after-cost execution. Treat all results as retrospective probability
evidence, not guaranteed profit. No live trading is active and no 2026 outcome
may be read.

The active short-horizon code is `daily_research/technical/`. The large
`daily_research/path_policy/` tree is historical/reference code, not the active
development interface. Historical conclusions remain in Git and
`daily_research/research_records/`.

## Current data

- Formal scope: 4,191,476 unique stock-days, 2012-01-04 through 2025-12-31;
  2010-2011 exists only as warm-up history.
- Repaired 557-field input fingerprint:
  `ade1a7997302f8ed0943601a060b3fe3a5267da05dbf614c34fc64cd153d6e10`.
- Current 183-field view fingerprint:
  `b7899cd9edd307e1a5b6fa6196648495d5a4f07ec11c9202c5440d814a62cc8e`.
- Current exact-target fingerprint:
  `8caad9d311eee892ec52c1f7583fd2101bb7eeb2c9c05e2ce3500a9718bc1303`.
- Full 183 matrix SHA-256:
  `81650e0fe8d87b07df74befba67345c06b4bbdd657b612105d73ed497c9efd38`.
- Lightweight quality gate passes: no duplicate keys, no 2026 rows, no Inf or
  all-empty rows, exact 557-to-183 projection equality, minimum field coverage
  99.8323%, and repaired minute range fields stay inside `[0, 1]`.

Three minute-feature defects were corrected in historical and live formulas:
trend efficiency now includes the first bar's open-to-close movement, hot-money
normalization uses consistent log returns, and tied intraday extrema select the
earliest bar deterministically. About 90.6% of finite trend-efficiency values
changed, so all pre-repair models that consumed minute fields are historical and
must not be reused as current evidence.

## Active feature and model contract

- `158`: 104 stable daily price-volume features plus 54 market-state features.
- `183`: the same 158 fields plus 25 signal-day 5-minute features. It is a view
  of the same matrix, not a separately versioned dataset.
- `raw60`: 60 trading days of normalized raw OHLC, volume, amount and valid-bar
  mask, plus 14 whole-market state fields.
- Five expanding forward folds cover 2020-2025 with a 30-day purge.
- Common model target: executable D10 base-cost net return ranking.
- Account replay: next-open entry, T+1, suspension/limit restrictions, legal
  delayed exit, 100-share lots, finite cash, all fees and double-slippage stress.
  Rankings are formed before future outcome validity is checked; an unfilled or
  unaffordable selection remains cash and is never replaced.

## Corrected-input results

| Model | OOF Rank IC | Positive-IC dates | Top10 stress | No-overlap stress | No-overlap 10% winner cap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tree, 158 | 0.09000 | 70.80% | +127.49% / -30.76% DD | +42.68% / -11.72% DD / 6 of 6 years | +2.74% |
| Tree, 183 | 0.08563 | 70.03% | +132.43% / -27.56% DD | +47.57% / -8.74% DD / 5 of 6 years | +6.60% |
| Raw60 sequence | 0.09654 | 76.75% | +55.34% / -28.84% DD | +28.80% / -11.48% DD / 5 of 6 years | -7.90% |
| Fixed 158-tree + raw60 rank mean | **0.10298** | 73.98% | +75.68% / -27.20% DD | **+48.73% / -10.12% DD / 6 of 6 years** | +5.99% |

The 183-minus-158 paired daily Rank IC increment is -0.00437 with HAC interval
`[-0.01146, +0.00273]`. Paired Top1/3/5/10 exact-net increments all cross zero.
Minute fields receive only about 0.24%-3.46% of tree gain by fold. They change
which extreme winners are selected, but have not proved a stable incremental
edge. Use 158 as the default daily tabular input; keep 183 only as a challenger.

The raw60 and tree scores have fold-average daily rank correlations of roughly
0.35-0.59, so they contain materially different information. The frozen 50/50
ensemble raises daily Rank IC over the tree by 0.01299 with HAC interval
`[0.00568, 0.02029]`; its +0.00644 increment over raw60 has interval
`[-0.00181, 0.01468]`. Direct Top1/3/5/10 exact-return increments and most paired
account-return increments include zero. Its no-overlap account point estimate is
best, but it has not proved broad economic dominance over the 158 tree. Keep it
as the current development benchmark because it records useful complementarity,
not because it is deployable. After capping each winner at 10%, no-overlap return
is only +5.99%, only 3 of 6 years are positive and the HAC lower bound is
negative. Right-tail dependence remains unresolved.

## Cleanup and design decisions

- Removed about 24.6 GiB of stale/reconstructible artifacts: the obsolete model
  dataset cache, rejected 364-field view, retired D3 forward bundle, outdated
  certification and all repair-invalidated model/evaluation outputs under the old
  full-market study. The five required current data/target directories remain.
- Do not create 183 v2/v3 copies. Repair current data in place; Git and result
  records preserve history.
- Do not use the full 557 fields as a default flat input. Financial, revision and
  announcement fields are sparse/PIT-sensitive and must return only as a separate
  age-aware branch that proves residual value.
- Do not revive the old D3 live prototype or repair it as part of historical
  model research. Build a small current live feature path only after a model is
  approved for paper trading.
- Preferred future sources: BaoStock for primary daily/status data; mootdx only
  if minute data proves value; AKShare for specific gaps and cross-checking;
  CNINFO for announcements and actual publication times. Do not automatically
  blend duplicate Eastmoney-derived feeds.

## Authoritative paths

- Contract and commands: `daily_research/technical/README.md`.
- Quality result: `daily_research/output/technical/quality.json`.
- Minute comparison: `daily_research/output/technical/tree_158_vs_183.json`.
- Tree evaluations: `daily_research/output/technical/tree_158/evaluation/result.json`
  and `tree_183/evaluation/result.json`.
- Raw sequence evaluation:
  `daily_research/output/technical/sequence_raw60/evaluation/result.json`.
- Current ensemble evaluation:
  `daily_research/output/technical/ensemble_tree158_raw60/evaluation/result.json`.

## Verification

- Current data quality command completed with status `ok` and zero forbidden
  2026 reads.
- All four evaluations match the current input, view, target and five fold-task
  fingerprints; stale fold/model combinations are rejected at load time.
- Focused technical/minute regression suite: 32 passed. Ruff, `compileall` and
  `git diff --check` pass (Git reports only the normal Windows LF/CRLF warning).

## Next work

First finish code/test/static verification of the simplified path. Then test the
raw60 signal across a small predeclared seed set and report the ensemble's paired
increment, not merely its best seed. If complementarity survives, investigate a
single predeclared tail-risk/selection-calibration method while holding features,
TopK, horizon and execution fixed. Do not search minute subsets, blend weights,
TopK, exits or LightGBM parameter grids on the same folds. Build historical/live
feature parity only after a paper-trading candidate passes the winner-cap and
seed-stability checks.
