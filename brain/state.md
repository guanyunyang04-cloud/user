# Current state

Updated: 2026-08-10

## Active objective

Find a repeatable A-share policy from the 2012-2025 PIT data that is causally
observable, legally executable after costs, and historically useful under the
user's pragmatic convention: the fixed forward validation folds may be used
for development, early stopping and final historical evaluation. This is
historical evidence, never a guaranteed-profit claim. No real trading or
shadow scoring is active; Tushare-backed fields are assumed normally available.

## Data and execution contract

- Primary universe: signal-date `quality_liquidity_pit`.
- Complete causal model matrix: 4,191,476 stock-days x 557 fields, certified
  through 2025-12-31; 2010-2011 is burn-in only.
- Five expanding causal validation folds cover 2020-2025. Each fold trains on
  all earlier eligible history with the configured purge; no 2026 outcome is
  read.
- Decisions use signal-close information only. Replay enforces next-open fill,
  T+1, suspension/limit constraints, legal-sale deferral, 100-share lots,
  fees, double slippage, finite cash, concurrent holdings and no rank
  substitution after an unfilled order.
- Formal economics mark adjusted total return but calculate costs and fills at
  raw exchange prices.

## Durable results

### Rules and signals already rejected as direct policies

The strict causal Chan parser is valid, but literal Type-1/2/3 buy points,
richer structural attributes, K-line path rules, fixed take-profits, stops and
trailing exits did not yield stable net value. Do not reopen a small Chan or
exit-rule grid. Financing-balance growth, financing Top10, and KAMA/adaptive-
line cross rules also lack robust absolute legal net value. Margin-detail
`observed` coverage can remain an eligibility feature, not smart-money intent.

### Existing longer-horizon candidates

1. The exact main-board 30/30/25/10/5 value-growth policy is the strongest
   interpretable historical candidate. Its frozen 68.61% gross risk budget
   grew CNY1m to about CNY2.863m in 2012-2025, with 8.02% annualized log growth,
   29.22% maximum drawdown and 11/14 positive years. It is historically
   promising, not a stable-profit guarantee.
2. The literal three-name version with D60, 45-50% gross and no replacement
   below ranks 1-3 is the concentrated challenger. At 50% gross it grew CNY1m
   to about CNY3.427m, with about 9.99% annual compound return, 22.89% maximum
   drawdown and 10/14 positive years, but it had four losing years.
3. The earlier D5 tree/sequence dual-market-gate Top10/D3-exit policy remains a
   short-horizon baseline: about 11.0% annualized and 15.5% maximum drawdown on
   its adaptive 2020-2025 account. It remains right-tail dependent and is not
   the current research frontier.

Detailed historical evidence remains under
`daily_research/research_records/seq100/` and the prior study/output paths
listed in the Git history. Do not treat the old D3 wording as the active model
search result below.

## Current D10 direct-payoff challenger

`seq100_full_market_multitask_forecast_v1` now directly ranks the exact
next-open-to-D10 legal net return with the complete 557-field input. The
completed tree model is `exact_net_return_d10_rank`, LightGBM LambdaRank
profile `strong_127`, five `outer_early_stop` folds.

Tree OOF result:

- Mean daily Rank IC: 0.07776; all five folds positive.
- Top10 double-slippage mean: +0.6474% per equal date, 5/6 years positive;
  its event-level HAC lower bound is +0.0366% and return deciles are strongly
  monotonic (Spearman about 0.988).
- Top1 and Top3 fail; the learned edge needs breadth rather than concentrated
  conviction.
- Exact finite-account Top10 replay uses 10% equity per D10 cohort. It grew
  CNY1m to CNY2.027m from 2020-2025, with -24.37% maximum drawdown and five
  positive years. Annual returns were -9.43%, +16.19%, +1.94%, +4.31%, +49.73%
  and +21.00%.
- It is not yet a stable-profit policy: the account daily HAC lower bound is
  slightly negative (-0.00416 percentage points), 2020 lost money, Top1/Top3
  lose, and capping the largest 10% of winners turns the double-slippage account
  negative (about -13.2%). The account relies materially on the right tail.

Authoritative artifacts:

- `daily_research/output/path_policy/studies/seq100_full_market_multitask_forecast_v1/payoff_evaluation/exact_net_return_d10_rank__strong_127/manifest.json`
- `daily_research/output/path_policy/studies/seq100_full_market_multitask_forecast_v1/payoff_account_replay/exact_net_return_d10_rank__strong_127/manifest.json`

## D10 sequence challenger: incomplete

The lookback-16, horizon-10 sequence challenger uses five `outer_early_stop`
folds (`maximum_epochs=60`, `patience=10`). It uses raw path windows plus the
causal tabular surface, with horizon-specific outputs. Folds 1 and 2 completed
and are saved; folds 3-5 are not yet complete.

- Fold 1: best epoch 2; Rank IC 0.12502; Top10 stress mean +1.3169%; perfectly
  monotonic deciles. Peak process RSS 6.40 GB.
- Fold 2: best epoch 2; Rank IC 0.07368; Top10 stress mean +0.5712%; strongly
  monotonic deciles. Peak process RSS 6.94 GB.
- These are encouraging but only two folds. Do not infer a sequence-model win
  or build an ensemble until all five OOF folds, sequence replay and comparable
  tree/sequence fusion are complete.

Implementation changes already present:

- `daily_research/path_policy/seq100_full_market_multitask_forecast.py` gained
  generic exact-horizon payoff labels and finite account replay helpers.
- `daily_research/path_policy/seq100_full_market_sequence_challenger.py` gained
  `--horizon`, D10 namespacing, `--evaluate-sequence-payoff` and
  `--replay-sequence-payoff`.
- Focused tests, Ruff, `py_compile` and `git diff --check` passed before the
  current partial sequence run.

Sequence artifact root:
`daily_research/output/path_policy/studies/seq100_full_market_multitask_forecast_v1/sequence_challenger/horizon_10/outer_early_stop/lookback_16/`.

## Exact next step

1. Complete only folds 3, 4 and 5 for horizon 10/lookback 16; do not rerun the
   completed folds 1-2.
2. Run the D10 sequence OOF payoff evaluation and exact account replay.
3. Compare tree, sequence and a within-date rank fusion on Rank IC, Top10
   double-slippage account wealth, maximum drawdown, annual/fold consistency,
   daily HAC lower bound and 10% winner-cap stress. Implement D10 fusion only
   after all sequence OOF predictions exist; do not overwrite D5 artifacts.
4. If fusion does not reduce tail dependence or improve stability, continue the
   broader planned model competition (lookback 32, StockMixer/Master-like
   variants, multi-seed trees, multitask return/downside/right-tail heads),
   using the same full 2012-2025 causal data and five folds.

Current status: `historical_candidates_exist; D10_tree_promising_but_tail_dependent; D10_sequence_validation_incomplete`.
