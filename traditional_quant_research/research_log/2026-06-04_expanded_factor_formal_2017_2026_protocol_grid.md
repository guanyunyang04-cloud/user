# Expanded Factor Formal 2017-2026 Protocol Grid

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-04_expanded_factor_formal_2017_2026_protocol_grid.md`.


## Summary

Run `frontier_personal_protocol_grid_formal_2017_2026_expanded_top100_200_20260604` is the first formal long-sample personal protocol grid for `factor_set=expanded`.

It used Baostock-only expanded factors, `2017-2026` walk-forward evaluation, `20d/monthly/buffer=3.0`, `top_n=100,200`, `30 bps`, `100m` capital stress, `10 bps per 1 pct` participation impact, and personal capital `1m`.

The run completed all ten evaluation years and produced formal personal gate evidence. It does not create a new `personal_backtest_candidate`, `personal_paper_candidate`, or `strategy_candidate`.

Artifacts:

- `traditional_quant_research/output/experiments/frontier_personal_protocol_grid/frontier_personal_protocol_grid_formal_2017_2026_expanded_top100_200_20260604`
- merged combined constraint: `combined_constraint_merged/frontier_personal_protocol_grid_formal_2017_2026_expanded_top100_200_20260604_combined_merged`
- personal gate: `personal_candidate_gate/frontier_personal_candidate_gate_20260604_072954`

## Gate Result

- `factor_set`: `expanded`
- `years`: `2017-2026`
- `top_n_values`: `[100, 200]`
- `evidence_scopes`: `formal_personal_backtest_candidate_gate`
- `personal_backtest_candidate_count`: `0`
- `personal_paper_candidate_count`: `0`
- `strategy_candidate_count`: `0`
- `decision`: `keep_personal_research_backtest_only`
- Best row: `top200_multifactor_rolling_ic_weighted_score_baseline_penalty_0_25`
- Best mean annualized return: `0.048710`
- Best weakest-year annualized return: `-0.486795`
- Best positive-year rate: `0.6`
- Best worst max drawdown: `-0.147552`
- Best total periods: `60`
- Failed gates for best row: `return_gate,weak_year_damage_gate`

The best expanded row narrowly misses the formal `return_gate` (`0.048710` vs required `0.05`) but fails the weak-year gate decisively because the weakest year is `-0.486795`, below the required `-0.35` floor.

## Top-N Comparison

| top_n | best_signal | best_mean_annualized_return | best_min_annualized_return | best_positive_year_rate | best_worst_max_drawdown | decision |
|---:|---|---:|---:|---:|---:|---|
| 100 | `multifactor_rolling_ic_weighted_score` | `-0.063530` | `-0.379457` | `0.3` | `-0.219515` | `continue_research` |
| 200 | `multifactor_rolling_ic_weighted_score` | `0.048710` | `-0.486795` | `0.6` | `-0.147552` | `continue_research` |

`top_n=200` is materially better than `top_n=100` for expanded factors. This reinforces the earlier lesson that smaller personal-capital baskets are not automatically better; concentration still amplifies weak-year damage under the current signal family.

## Year-Level Shape

For the best expanded row (`top_n=200`, rolling IC), year-level annualized returns were:

| eval_year | annualized_return | max_drawdown | periods |
|---:|---:|---:|---:|
| 2017 | `-0.168491` | `-0.082315` | 4 |
| 2018 | `-0.486795` | `-0.054073` | 1 |
| 2019 | `0.215003` | `-0.147552` | 7 |
| 2020 | `0.024683` | `-0.136819` | 7 |
| 2021 | `0.095557` | `-0.080932` | 7 |
| 2022 | `-0.062919` | `-0.091805` | 8 |
| 2023 | `-0.108623` | `-0.122936` | 9 |
| 2024 | `0.348210` | `-0.108031` | 8 |
| 2025 | `0.527506` | `-0.009278` | 7 |
| 2026 | `0.102971` | `-0.003573` | 2 |

The expanded factor set improves the recent strong window, especially `2024-2026`, but it does not repair the old weak-year cluster. The same structural years remain problematic: `2017`, `2018`, `2022`, and `2023`.

## Core Comparison

Relevant core evidence:

- Prior formal core `top_n=200` personal gate (`frontier_personal_candidate_gate_20260604_004311`) promoted `multifactor_rolling_ic_weighted_score` to `personal_backtest_candidate`: mean annualized return `0.096737`, weakest year `-0.301352`, positive-year rate `0.6`, worst drawdown `-0.138549`, total periods `60`.
- Formal core `top_n=100` protocol grid (`frontier_personal_protocol_grid_formal_2017_2026_top20_50_100_20260604`) failed: best mean annualized return `-0.057341`, weakest year `-0.372182`, positive-year rate `0.2`.
- Formal expanded `top_n=100` is slightly worse than core `top_n=100` on mean return (`-0.063530` vs `-0.057341`) and has higher proxy exposure (`1.289833`, failing proxy exposure sanity).
- Formal expanded `top_n=200` is positive but weaker than the prior core `top_n=200` candidate on both mean return and weakest-year damage.

Therefore, `expanded` is not a drop-in upgrade over `core`. The current expanded factor family should remain a research axis, not replace the historical frontier evidence.

## Interpretation

The result is useful because it answers a real model-strength question: simply adding more Baostock-only price, amount, turnover, and pctChg-derived factors does not automatically create a stronger personal strategy.

The main failure is not the data source and not only the portfolio size. It is still the signal/regime model: the model has not learned a robust behavior for weak market breadth / weak 20-day market-strength regimes, especially in the historical weak years.

## Decision

- Keep the existing core `top_n=200` rolling IC `personal_backtest_candidate` as the only paper-bootstrap-ready historical candidate.
- Do not promote any expanded row to paper tracking.
- Do not replace `core` with `expanded` as the default frontier evidence.
- Preserve `expanded` as an explicit experimental factor axis for future factor-family selection, pruning, and regime-conditioned blending.

## Next Step

The next useful work is not another broad `expanded` grid. It is a prior-fit model rebuild:

1. Use `core` and `expanded` as separate factor families.
2. Fit factor-family selection, pruning, or blend rules only on years prior to each eval year.
3. Target weak years `2017/2018/2022/2023` explicitly through market breadth / regime-conditioned rules, without selecting thresholds from eval-year returns.
4. Rerun `2017-2026 / 20d/monthly/top_n=100,200 / 30bps / 100m / 10bps impact` through the formal personal gate.
