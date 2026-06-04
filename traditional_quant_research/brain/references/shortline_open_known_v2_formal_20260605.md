# Shortline Open-Known Factor v2 Formal Evidence - 2026-06-05

## Scope

This reference records the stable-cache formal evidence for the short-line A-share limit-up strategy v2. The experiment only uses Baostock daily data and keeps the A-share T+1 rule: buy at T+1 open, earliest sell from T+2. T+1 limit-up, T+1 close/high/low, and later returns are labels or exit outcomes only, not buy features.

## Stable Cache

- Cache dir: `traditional_quant_research/output/experiments/short_open_known_factor_rebuild/_event_cache`
- Cache years available after this run: `2016-2026`
- Missing formal cache years were built one by one to avoid crashing the machine:
  - 2019: `short_open_known_factor_rebuild_preopen_submit_20260605_001220`
  - 2020: `short_open_known_factor_rebuild_preopen_submit_20260605_001411`
  - 2021: `short_open_known_factor_rebuild_preopen_submit_20260605_001615`
  - 2022: `short_open_known_factor_rebuild_preopen_submit_20260605_001819`
  - 2023: `short_open_known_factor_rebuild_preopen_submit_20260605_002017`
  - 2024: `short_open_known_factor_rebuild_preopen_submit_20260605_002210`

The stable workflow used `--stable-cache`, `--build-event-cache-only` for yearly cache construction, single-threaded BLAS/OpenMP environment variables, and reduced event panel writes to keep memory and CPU pressure controlled.

## Formal Runs

### Preopen Submit

- Run: `short_open_known_factor_rebuild_preopen_submit_20260605_002417`
- Profile: `preopen_submit`
- Years requested: `2017-2026`
- Event rows: `114457`
- OOS trades: `85990`
- Feature leakage audit: `passed`
- Mean net return per trade: `1.4604%`
- Median net return per trade: `-3.7202%`
- Win rate: `35.55%`
- Payoff: `2.7309`
- Positive year rate in produced OOS yearly rows: `1.0`
- Executable trade rate: `0.7879`
- Near-limit trade rate: `0.1692`
- One-word limit-like trade rate: `0.1287`
- Near-one-word limit-like trade rate: `0.1340`
- Decision: `shortline_diagnostic_only`
- Failed gates: `incomplete_oos_years,contains_unfilled_near_limit_entries`

Interpretation: the strict pre-open feature path shows strong payoff, but a material share of selected trades depends on near-limit or one-word-like entry paths. The path is useful as strength diagnostics, not as a formal executable candidate.

### Executable Only

- Run: `short_open_known_factor_rebuild_executable_only_20260605_003252`
- Profile: `executable_only`
- Years requested: `2017-2026`
- Event rows: `114457`
- OOS trades: `15250`
- Feature leakage audit: `passed`
- Mean net return per trade: `-0.1360%`
- Median net return per trade: `-0.6606%`
- Win rate: `44.45%`
- Payoff: `1.1728`
- Positive year rate: `0.5`
- Worst year mean net return: `-1.2902%`
- Executable trade rate: `1.0`
- Near-limit and one-word trade rates: `0.0`
- Decision: `open_known_diagnostic`
- Failed gates: `nonpositive_mean,positive_year_rate_lt_0.8,negative_worst_year,payoff_lt_2,open_known_diagnostic`

Interpretation: after removing one-word, near-one-word, near-limit, and low-liquidity entry paths, the current rule library does not yet produce a profitable or stable short-line candidate. This is the most important practical result for personal small-capital execution.

### Open Print Filter

- Run: `short_open_known_factor_rebuild_open_print_filter_20260605_003723`
- Profile: `open_print_filter`
- Years requested: `2017-2026`
- Event rows: `114457`
- OOS trades: `85990`
- Feature leakage audit: `passed`
- Mean net return per trade: `1.4604%`
- Payoff: `2.7309`
- Decision: `open_known_diagnostic`
- Failed gates: `incomplete_oos_years,contains_unfilled_near_limit_entries,open_known_diagnostic`

Interpretation: open-known filtering is useful for operational diagnostics, but it cannot be treated as a pre-open candidate. In this formal run it still contains unfilled or near-limit entry paths.

## Factor Evidence Snapshot

- Strongest diagnostic buckets are dominated by hard-to-buy strength paths: `one_word_limit_like`, `near_one_word_limit_like`, `entry_open_near_limit`, and high `next_open_gap_pct`.
- KAMA-related conditions remain useful as interaction diagnostics rather than standalone candidate rules. `kama_bias_0_3` and KAMA slope interactions appear in the factor diagnostics, but they do not rescue the executable-only formal gate.
- Important interactions include `ma_compression_tight + second_plus_board` and `kama_slope_turn_positive + second_plus_board`, but the best interaction evidence still overlaps with next-day strength and availability problems.
- Sell diagnostics show that simple `window_close` around 3 days is competitive among current rule-grid exits; trend-break exits such as KAMA/MA5/MA10 break do not yet create a strong executable candidate.

## Governance Conclusion

- No `shortline_backtest_candidate` is created.
- The short-line v2 infrastructure is implemented and verified, but formal evidence says the current daily-only rule library is not yet a tradable candidate once entry executability is enforced.
- The research line should continue, but the next work should target alpha that separates buyable strong stocks from unbuyable limit-up continuation paths.

## Recommended Next Research

- Add a stricter pre-open executable proxy instead of relying on T+1 open-known flags: exclude prior-day one-word-like patterns, extreme prior-day open/close lock patterns, and low-liquidity names before ranking.
- Re-rank events using prior-fit historical post-limit-up behavior per stock and industry, not just current event shape.
- Study second-board and third-board regimes separately, because the most profitable diagnostics cluster around continuation strength but executable quality differs by board stage.
- Keep KAMA as a structural interaction feature: test KAMA slope turn, KAMA/MA compression, and low KAMA bias before limit-up, but do not let KAMA alone define a candidate.
- If daily-only evidence remains unable to identify intraday acceptance after open, plan a separate minute-level open-board or re-seal study.

## Verification

- `git diff --check -- traditional_quant_research`: passed, with line-ending warnings only.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest traditional_quant_research/tests/test_short_open_known_factor_rebuild.py -q`: `16 passed`.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest traditional_quant_research/tests/test_short_limitup_strategy_search.py -q`: `4 passed`.
