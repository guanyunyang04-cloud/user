# v2 Daily Size Production Cache Contract - 2026-06-03

## Scope

This note records the v2.2 daily size infrastructure contract. It is a code/contract milestone, not proof that live PIT market cap or float cap data has been ingested.

## Implemented

- `traditional_quant_research/size_source.py` now supports Tushare `daily_basic` annual cache fetches through `fetch_tushare_daily_size_cache()`.
- The standardized `daily_size` schema remains vendor-neutral: `date, code, total_market_cap, float_market_cap, total_share, float_share, free_share, market_cap_unit, share_unit, source, source_trade_date`.
- `traditional_quant_research/dataset_builder_v2.py fetch-size` resolves trade dates from explicit `--trade-dates`, cached v2 trade dates, or cached PIT stock lists. Formal full-year runs write `cache/daily_size/year=YYYY.parquet`; `--symbols/--max-symbols` smoke runs write `cache/daily_size/samples/sample=<hash>/year=YYYY.parquet`.
- `assemble --include-size` reads only formal annual `daily_size` cache, filters it to PIT `date, code` stock-list keys, writes snapshot-level `daily_size.parquet` when non-empty, and records `daily_size_rows/daily_size_fields` in `manifest.json`.
- Research loaders remain read-only through `dataset_v2.load_pit_daily_size()` and `load_tradeable_panel(..., include_size=True)`.

## Verification

- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest traditional_quant_research/tests/test_size_source.py traditional_quant_research/tests/test_dataset_builder_v2.py traditional_quant_research/tests/test_dataset_v2.py`
  - Result: `29 passed`.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest traditional_quant_research/tests`
  - Result: `167 passed`.
- `git diff --check -- traditional_quant_research`
  - Result: no whitespace errors; one pre-existing CRLF warning remains for `traditional_quant_research/research_log/2026-06-02_low_corr_frontier_neutralization_audit.md`.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.dataset_builder_v2 fetch-size --help`
  - Result: CLI exposes `fetch-size`, `--trade-dates`, `--include-size`, and `--size-token`.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.dataset_builder_v2 fetch-size --year 2026 --start-date 2026-06-01 --end-date 2026-06-01 --trade-dates 20260601 --symbols 600000.SH,000001.SZ --output-root traditional_quant_research/output/tmp_fetch_size_cli_smoke`
  - Result: without token it returns `skipped/auth_missing`, and the reported cache path is under `cache/daily_size/samples/sample=<hash>/year=2026.parquet`.

## Current Limitation

The latest real Tushare probe has advanced from `skipped/package_missing` to `skipped/auth_missing` after installing `tushare=1.4.29`. No authenticated live field probe has passed yet. Therefore latest v2 snapshot still has no real `daily_size.parquet`, and strategy promotion must continue to fail the true size gate.

## Next Step

Configure `TUSHARE_TOKEN` or `TS_TOKEN`, or switch to JoinQuant/RQData, run a two-symbol live `fetch-size` smoke, then assemble a small `--include-size` snapshot and rerun `v2_daily_size_audit`.
