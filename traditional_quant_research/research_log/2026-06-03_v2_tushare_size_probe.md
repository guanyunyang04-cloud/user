# V2 Tushare Size Probe

## Summary

- `run_id`: `v2_tushare_size_probe_20260603_114552`
- `status`: `skipped`
- `skip_reason`: `auth_missing`
- `symbols`: `600000.SH,000001.SZ`
- `trade_dates`: `20260525,20260601`
- `expected_pairs`: `4`
- `returned_pairs`: `0`
- `missing_pairs`: `4`
- `failure_count`: `1`
- `required_fields_present`: `False`
- `v2_2_ready`: `False`

## Artifacts

- `tushare_daily_basic_probe.csv`: raw Tushare daily_basic-shaped probe rows.
- `daily_size_probe.csv` / `daily_size_probe.parquet`: vendor-neutral v2 daily_size schema rows.
- `failures.csv`: package, auth, empty-result or query failures.
- `summary.json` / `summary.md`: structured and readable gate summary.

## Field Non-Null Rates

- `ts_code`: `None`
- `trade_date`: `None`
- `total_mv`: `None`
- `circ_mv`: `None`
- `total_share`: `None`
- `float_share`: `None`
- `free_share`: `None`

## Unit Assumptions

- `total_mv/circ_mv`: Tushare daily_basic documents market value fields in 10k CNY units.
- `total_share/float_share/free_share`: Tushare daily_basic documents share-base fields in 10k share units.

## Decision

Configure TUSHARE_TOKEN or TS_TOKEN before validating daily_basic as the v2.2 size source.

## Interpretation

This probe is a v2.2 data gate. It does not promote a strategy candidate. The frontier remains `candidate-frontier/backtest_only` until a PIT size source is live-validated and integrated into the snapshot.
