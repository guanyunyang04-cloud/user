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

## Current Next Work

Continue with Baostock-only weak-year rebuild and portfolio constraint research:

- fixed weak years: `2017/2018/2022/2023`;
- prior-fit rules only, with each eval year fit from earlier years;
- no traditional ML yet;
- no 2024-2026 frontier parameter chasing;
- no proxy or free current quote source can make the true size gate pass.
