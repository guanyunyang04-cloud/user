# V2 External Size Source Scout

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-03_v2_external_size_source_scout.md`.


## Summary

- `run_id`: `v2_external_size_source_scout_20260603_115009`
- `best_candidate`: `tushare_daily_basic`
- `v2_2_ready`: `False`
- `usable_now_count`: `0`
- `primary_blocker`: No locally authenticated and live-probed PIT daily market-cap/float-cap source is currently validated.

## Local Packages

- `tushare`: installed=`True`, version=`1.4.29`, auth_hint=`False`
- `jqdatasdk`: installed=`False`, version=``, auth_hint=`False`
- `rqdatac`: installed=`False`, version=``, auth_hint=`False`
- `akshare`: installed=`True`, version=`1.18.63`, auth_hint=`False`
- `efinance`: installed=`True`, version=`0.5.8`, auth_hint=`False`
- `baostock`: installed=`True`, version=`0.9.1`, auth_hint=`False`
- `pandas`: installed=`True`, version=`2.3.2`, auth_hint=`False`
- `pyarrow`: installed=`True`, version=`20.0.0`, auth_hint=`False`

## Candidate Matrix

- `tushare_daily_basic`: installed=`True`, auth=`auth_missing_or_unknown`, usable_for_v2_2=`False`; blocker: Tushare package/token/live daily_basic probe has not been locally validated.
- `joinquant_get_fundamentals_valuation`: installed=`False`, auth=`package_missing`, usable_for_v2_2=`False`; blocker: JQData package/auth/live valuation probe has not been locally validated.
- `rqdata_fundamental_or_factor`: installed=`False`, auth=`package_missing`, usable_for_v2_2=`False`; blocker: No local rqdatac package/auth or table-level PIT field probe is available.
- `akshare_public_endpoints`: installed=`True`, auth=`not_required`, usable_for_v2_2=`False`; blocker: Public endpoint PIT timing, historical revision policy and full-market cap coverage are unproven.
- `efinance_public_endpoints`: installed=`True`, auth=`not_required`, usable_for_v2_2=`False`; blocker: Historical PIT total/float market-cap coverage is not proven.
- `baostock_v2_1`: installed=`True`, auth=`not_required`, usable_for_v2_2=`False`; blocker: Baostock v2.1 lacks total_mv/circ_mv/market_cap/float_market_cap/share-base fields.

## Decision

Use Tushare daily_basic as the first v2.2 validation target; if token/auth is unavailable, evaluate JoinQuant or RQData subscriptions; keep AkShare/efinance as auxiliary public checks only.

## Next Actions

- Configure TUSHARE_TOKEN or TS_TOKEN and run a small historical daily_basic live probe for 600000.SH/000001.SZ.
- Verify symbol mapping, trade_date coverage, units and missingness against the existing v2.1 universe.
- If Tushare is unavailable, request JoinQuant or RQData credentials and repeat the same two-symbol probe.
- Until then, keep size controls proxy-only and do not promote candidate-frontier strategies.

## Interpretation

This is a data-source gate, not a strategy promotion. The current frontier remains `candidate-frontier/backtest_only` until true size/float-size data or a stricter proxy-only policy is validated.
