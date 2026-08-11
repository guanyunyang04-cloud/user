# Current state

Updated: 2026-08-11

## Active objective

Find a repeatable A-share policy from the 2012-2025 point-in-time data that is
causally observable, legally executable after costs, and historically useful.
The five fixed forward folds may be reused for early stopping, model choice and
historical evaluation under the user's convention. Results are retrospective
evidence, never a guaranteed-profit claim. No real trading is active and no
2026 outcome may be read. Treat the market as a partially observable,
non-stationary probability system: optimize stable conditional rankings,
net-return distributions and action value rather than deterministic direction.

## Data and execution contract

- Signal universe is `quality_liquidity_pit`.
- The certified causal matrix has 4,191,476 stock-days and 557 fields through
  2025-12-31; 2010-2011 is burn-in only.
- Five expanding folds cover 2020-2025 with purge and no future-fold training.
- Decisions use signal-close information, next-open fills, T+1, suspension and
  limit constraints, 100-share lots, fees, stamp tax, transfer fee, double
  slippage, finite cash, overlapping cohorts, delayed legal sales and no rank
  substitution after an unfilled order.
- Formal economics mark adjusted total return but calculate fills and costs at
  raw exchange prices.

## Durable negative conclusions

- The strict causal Chan parser, literal buy points, small structural-rule
  grids, fixed exits, financing-balance rules and KAMA crosses did not produce
  stable legal net value. Do not reopen those small grids.
- A common-high-rank abstention gate was tested diagnostically on the current
  tree/sequence selections; removing low-agreement names reduced Top10 stress
  performance. Model disagreement is useful complementarity, not a reason to
  require unanimous ranks.
- Lookback-32 sequence Fold 1 was trained as a resource pilot. Rank IC was
  0.12585 versus 0.12601 for lookback-16 and Top10 stress mean was about
  1.189% versus about 1.317%; the roughly 17-minute fold showed no clear gain.
  Do not spend four more folds on lookback-32 without a new reason.

## Existing interpretable candidates

- The exact main-board 30/30/25/10/5 value-growth policy remains a strong
  interpretable historical candidate: about CNY1m to CNY2.863m in 2012-2025,
  8.02% annualized log growth, 29.22% maximum drawdown and 11/14 positive
  years.
- The literal D60 Top3 concentrated version at 50% gross grew about CNY1m to
  CNY3.427m, with about 9.99% compound growth, 22.89% drawdown and 10/14
  positive years. Both are historical candidates, not stable-profit claims.

## D10 payoff model evidence

All current D10 payoff evaluations use the exact next-open-to-D10 legal net
return target and the full 557-field matrix. Formal breadths are now Top1,
Top3, Top5 and Top10.

- `strong_127` LightGBM: OOF Rank IC 0.07776; Top10 stress mean 0.6474%;
  exact Top10 stress account CNY1m to CNY2.027m, -24.37% drawdown, 5/6
  positive years; 10% winner cap leaves -13.19%.
- Lookback-16 sequence: OOF Rank IC 0.09938; exact Top10 stress account to
  CNY2.084m, -36.44% drawdown, 5/6 positive years; 10% cap leaves -36.94%.
- Single `qlib_capacity_210` LightGBM: OOF Rank IC 0.08037; Top10 stress
  account to about CNY1.434m with weaker stability. It is useful as a diverse
  component, not as a standalone policy.
- The original two-model mean rank fusion (`strong_127` plus sequence) has
  OOF Rank IC 0.10341 and exact Top10 stress account CNY2.453m, -28.13%
  drawdown, 6/6 positive years, positive daily HAC lower bound, but 10% cap
  remains -10.22%.
- The two-model minimum rank fusion is slightly more conservative (CNY2.439m,
  -28.04% drawdown, 10% cap -7.15%), but still fails the 5% cap and does not
  remove tail dependence.

## Rejected D10 target variants

Three pre-specified challengers were completed against the same exact D10
executable payoff and finite-account contract. None improves on the current
three-model benchmark, so do not tune weights or thresholds around them.

- A `market_excess_endpoint_return_d10_rank` `strong_127` tree contains useful
  ordering information (OOF Rank IC 0.07876), but Top10 stress has a negative
  event-level HAC lower bound. Its exact account gains 54.01% with -23.09%
  drawdown and 5/6 positive years; the 10% winner cap loses 25.71% and the 5%
  cap loses 64.40%. Reject it as a standalone or fusion component.
- The `exact_net_return_d10_q10` lower-quantile head reaches OOF Rank IC
  0.08846, but Top10 stress gains only 5.29%, draws down 37.45% and has 3/6
  positive years. Its best uncapped breadth is Top1 at +19.01%, also with only
  3/6 positive years. The downside head is not independently tradeable.
- The frozen 50/50 average of within-date return-rank and q10-rank reaches OOF
  Rank IC 0.09133, but its Top10 stress account gains only 26.01%, draws down
  26.40% and has 3/6 positive years; 2020-2022 all lose. The 10% cap is -0.44%
  and the 5% cap is -32.07%. It neither improves economics nor removes right-
  tail dependence.

## Current best historical challenger

The strongest current candidate is the independent three-model equal rank
fusion:

`sequence lookback-16 + strong_127 tree + qlib_capacity_210 tree`, with the
within-date percentile rank of each model averaged and no market gate.

- Five OOF Rank IC folds: 0.10490, 0.08195, 0.10972, 0.09085, 0.12244;
  combined Rank IC 0.10183.
- OOF Top1/3/5/10 stress means are approximately 1.193%, 0.922%, 0.866% and
  0.811%; all four breadths have positive HAC lower bounds and 6/6 positive
  years at the event level.
- Exact finite-account Top10 stress: CNY1m to CNY2.516m, +151.65% total,
  -28.04% maximum drawdown, 6/6 positive years, daily HAC lower about
  +0.0066 percentage points. Annual returns are approximately 13.32%, 7.84%,
  8.61%, 8.96%, 51.56% and 14.82% for 2020-2025.
- Top1/3/5/10 account stress breadths all have positive years and positive
  HAC lower bounds; Top10 has the lowest drawdown among them.
- The 10% winner-cap stress account still ends at about CNY1.057m (+5.66%),
  but its HAC lower bound is negative and only 3/6 years are positive. The 5%
  cap ends at about CNY0.431m (-56.93%). This materially improves but does not
  eliminate right-tail dependence; `stable_profit_claim_allowed` remains
  false.
- The three-model minimum fusion was replayed as a pre-specified control and
  was dominated by the mean (Top10 +151.70% but -30.29% drawdown and 10% cap
  only +2.99%).

The component set was selected after reviewing the reusable OOF evidence, so
the result is adaptive retrospective research, not independent confirmation.

## Authoritative artifacts

- Three-model mean OOF:
  `daily_research/output/path_policy/studies/seq100_full_market_multitask_forecast_v1/payoff_ensemble_evaluation/horizon_10/lookback_16/sequence_outer_early_stop__tree_outer_early_stop__tree_profiles_strong_127_qlib_capacity_210/manifest.json`
- Three-model mean account replay:
  `daily_research/output/path_policy/studies/seq100_full_market_multitask_forecast_v1/payoff_ensemble_account_replay/horizon_10/lookback_16/sequence_outer_early_stop__tree_outer_early_stop__tree_profiles_strong_127_qlib_capacity_210/manifest.json`
- Three-model minimum control uses the same paths with
  `__fusion_minimum__tree_profiles_strong_127_qlib_capacity_210`.
- Single-tree and sequence manifests are under the corresponding
  `payoff_evaluation`, `payoff_account_replay` and
  `sequence_payoff_evaluation` directories in the same study root.
- Lookback-32 pilot:
  `daily_research/output/path_policy/studies/seq100_full_market_multitask_forecast_v1/sequence_challenger/horizon_10/outer_early_stop/lookback_32/fold_1/manifest.json`
- Rejected market-relative tree evaluation/account:
  `daily_research/output/path_policy/studies/seq100_full_market_multitask_forecast_v1/payoff_evaluation/market_excess_endpoint_return_d10_rank__strong_127/manifest.json`
  and the matching path under `payoff_account_replay`.
- Rejected q10 tree evaluation/account:
  `daily_research/output/path_policy/studies/seq100_full_market_multitask_forecast_v1/payoff_evaluation/exact_net_return_d10_q10__strong_127/manifest.json`
  and the matching path under `payoff_account_replay`.
- Rejected frozen return/q10 fusion evaluation/account:
  `daily_research/output/path_policy/studies/seq100_full_market_multitask_forecast_v1/payoff_score_fusion_evaluation/horizon_10/strong_127__outer_early_stop/manifest.json`
  and the matching path under `payoff_score_fusion_account_replay`.

## Code and verification

- `daily_research/path_policy/seq100_full_market_multitask_forecast.py` now
  shares `PAYOFF_TOP_KS = (1, 3, 5, 10)` across payoff OOF and account tasks.
- `daily_research/path_policy/seq100_full_market_sequence_challenger.py` now
  supports horizon-isolated payoff evaluation, multi-tree profile rank fusion,
  independent artifact namespaces and the matching account replay. Audited
  account caches reject legacy manifests without an explicit zero 2026-read
  count.
- Focused and full joint tests pass: 40 tests. Ruff, `py_compile` and
  `git diff --check` pass. All new target-variant manifests plus the refreshed
  sequence and three-model mean account manifests explicitly report zero
  forbidden 2026 reads.

## Next step

Do not label the current challenger as stable or deploy it. Preserve the
three-model mean as the current historical benchmark and, when research
continues, choose one pre-specified architecture competition (for example a
lightweight StockMixer/MASTER-style sequence model or a return/downside
multitask head) using the same five folds and exact account contract. Avoid
another ad hoc fusion or small rule grid until a new model supplies genuinely
independent OOF information.
