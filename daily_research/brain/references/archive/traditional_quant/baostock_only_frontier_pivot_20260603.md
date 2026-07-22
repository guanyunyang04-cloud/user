# Baostock-Only Frontier Pivot - 2026-06-03

## Decision

The current research mainline should stop blocking on paid or credentialed market-cap / float-cap data. Frontier review now defaults to a Baostock-only mode that uses available Baostock fields and price/volume-derived proxies, while preserving strict true-size evidence as an optional future upgrade line.

## Rationale

- Baostock has full local coverage for the current v2.1 study base, including daily OHLCV, amount, volume, `turn`, `pctChg`, valuation fields, and the month-start industry snapshot.
- Baostock does not provide reliable PIT `total_market_cap`, `float_market_cap`, or share-base fields. Prior probes showed daily history rejects market-cap/share candidates and `query_stock_basic` only returns identity/status fields.
- Tushare `daily_basic`, JoinQuant, and RQData remain credible true-size paths, but they require token/auth/subscription and are not available in the current environment.
- Treating `amount`, `volume`, `turn`, `log_amount_mean_20d_z`, AkShare/CNInfo reconstruction, or current quote probes as true size evidence would weaken the research contract.

## Gate Semantics

- Default promotion gate mode: `baostock_only`.
- Baostock-only candidates may be labeled `candidate-frontier/baostock_only` when return, year, sample, drawdown, execution, and proxy-style exposure gates pass.
- `daily_size_ready_for_research=False` is disclosed as `true_size_gate=False` in Baostock-only mode, but it is not a Baostock-only failure.
- `candidate-frontier/baostock_only` is not a `strategy_candidate`, not `out_of_sample_supported`, and not production evidence.
- `strategy_candidate` requires `research_mode=true_size`, `daily_size_ready_for_research=True`, and all other promotion gates passing.

## 2026-06-04 North Star Update

The active North Star is now `Baostock-only personal quant strategy research`.

The current mainline target is no longer institutional promotion. It is to produce at least one `personal_backtest_candidate` that can enter paper tracking under:

- Baostock-only data;
- 2017-2026 or equivalent long-window evidence;
- prior-fit / walk-forward validation;
- conservative cost and participation impact;
- limit-up / limit-down / suspension execution constraints;
- personal small-capital tradability assumptions;
- understandable alpha, regime, model-fusion, and portfolio-construction logic.

True-size, market-cap neutrality, institutional capacity, and production-grade promotion remain future upgrades. They must not block alpha, regime, model-fusion, portfolio construction, or personal execution research now.

## 2026-06-04 Personal Candidate Closure

The North Star transition has reached its first closure condition for the current goal:

- Formal personal gate run: `frontier_personal_candidate_gate_20260604_004311`.
- Source evidence: `low_corr_frontier_combined_constraint_audit_20260603_122047`.
- Passing candidate: `multifactor_rolling_ic_weighted_score` with `constraint_variant=baseline` and `exposure_penalty_strength=0.25`.
- Candidate id for tracking: `multifactor_rolling_ic_weighted_score_baseline_penalty_0_25`.
- Evidence window and protocol: `2017-2026`, prior-fit / walk-forward metadata, `20d/monthly/top_n=200/buffer=3.0`, `30bps`, `100m` capital stress, `10bps per 1 pct participation` impact, Baostock-only snapshot.
- Personal gate metrics: mean annualized return about `0.096737`, weakest annualized year about `-0.301352`, positive year rate `0.6`, worst max drawdown about `-0.138549`, total periods `60`, max proxy monthly mean abs active exposure about `1.111381`.
- Formal paper bootstrap run: `frontier_personal_paper_tracking_bootstrap_20260604_004320`.
- Bootstrap output: `paper_tracking_candidates.csv`, `paper_tracking_protocol.csv`, `paper_tracking_log_template.csv`, and `paper_tracking_review_rules.csv`.

This closes the current objective of finding at least one Baostock-only personal `personal_backtest_candidate` that can enter paper tracking. It does not create a `personal_paper_candidate`, `strategy_candidate`, `out_of_sample_supported`, or production candidate. Future promotion requires real paper records, by default at least `6` completed rebalance periods and at least `120` calendar days, plus execution completeness, paper return sanity, drawdown control, and single-period damage review.

## Current Next Work

Continue after this closure with Baostock-only model strengthening and paper tracking:

- start logging the formal paper tracking template for `multifactor_rolling_ic_weighted_score_baseline_penalty_0_25`;
- test personal-sized variants such as `top_n=20/50/100` against the same long-window standard;
- keep fixed weak years `2017/2018/2022/2023` visible in diagnostics;
- use prior-fit rules only, with each eval year fit from earlier years;
- keep true-size / institutional promotion as a future enhancement line.
